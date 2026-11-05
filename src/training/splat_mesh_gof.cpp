/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "splat_mesh_gof.hpp"

#include "components/rolling_shutter.hpp"
#include "core/camera.hpp"
#include "core/checked_arithmetic.hpp"
#include "core/collision_volume.hpp"
#include "core/cuda/memory_arena.hpp"
#include "core/logger.hpp"
#include "core/memory_pressure.hpp"
#include "core/mesh_data.hpp"
#include "core/splat_data.hpp"
#include "core/tensor/internal/cub_workspace.hpp"
#include "core/tensor/internal/cuda_stream_context.hpp"
#include "rasterization/gsplat/GofFusion.h"
#include "rasterization/gsplat/TsdfFusion.h"
#include "rasterization/gsplat_rasterizer.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <format>
#include <vector>

namespace lfs::training {
    namespace {
        using core::collisionError;
        using core::Device;
        using core::Tensor;
        using Clock = std::chrono::steady_clock;

        // One fragment is a pixel, a depth and a log transmittance; the sort needs half a dozen
        // copies of that while it runs. The cap is a sanity bound; device_buffer_preflight does
        // the real memory check against the free VRAM.
        constexpr uint64_t kFragmentAllowance = 48;
        constexpr uint64_t kMaxFragments = 128'000'000;
        constexpr float kMinFragmentAlpha = 1.f / 255.f;

        template <class T>
        struct DeviceArray {
            core::tensor_ops::ScopedDeviceBuffer storage;
            size_t count = 0;
            DeviceArray() = default;
            DeviceArray(const size_t n, cudaStream_t stream, const bool zeroed = false)
                : storage(core::checked_product(std::max<size_t>(n, 1), sizeof(T), "GOF allocation"), stream, "splat_mesh_gof"),
                  count(n) {
                if (zeroed)
                    LFS_CUDA_CHECK(cudaMemsetAsync(data(), 0, count * sizeof(T), stream));
            }
            [[nodiscard]] T* data() const { return storage.template as<T>(); }
        };

        [[nodiscard]] double seconds_since(const Clock::time_point start) {
            return std::chrono::duration<double>(Clock::now() - start).count();
        }

        void validate(const GofMeshSettings& s) {
            const bool ok = std::isfinite(s.voxel_size) && s.voxel_size > 0 && std::isfinite(s.min_opacity) &&
                            s.min_opacity > 0 && s.min_opacity < 1 && s.view_stride >= 1 && s.color_stride >= 1 &&
                            s.min_component_faces >= 0 && s.max_width >= 16;
            if (!ok)
                throw collisionError("mesh.settings_invalid", "Invalid mesh settings", ErrorCode::InvalidArgument,
                                     std::format("{}; {}", s.voxel_size, s.min_opacity));
        }

        void device_buffer_preflight(const size_t bytes, const char* operation) {
            if (!core::MemoryPressureCoordinator::instance().preflight({.operation = operation, .temporary_device_bytes = bytes}).ok)
                throw collisionError("mesh.memory", "Not enough GPU memory for the mesh; free memory or use a larger voxel size",
                                     ErrorCode::ResourceExhausted, std::format("{:.1f} GB", static_cast<double>(bytes) / (1 << 30)));
        }

        template <size_t N>
        void copy_coefficients(const Tensor& source, float (&target)[N], const size_t limit = N) {
            if (!source.is_valid() || source.numel() == 0)
                return;
            auto host = source.cpu().to(core::DataType::Float32).contiguous();
            std::memcpy(target, host.ptr<float>(), std::min({source.numel(), limit, N}) * sizeof(float));
        }

        void copy_matrix(const std::array<float, 16>& source, float (&target)[16]) {
            std::copy(source.begin(), source.end(), target);
        }

        // Mirrors what gsplat_rasterize_forward uses, so voxels land on the pixels that rendered them.
        [[nodiscard]] gsplat_lfs::TsdfView make_view(const core::Camera& camera) {
            gsplat_lfs::TsdfView view;
            const bool undistorted = camera.is_undistort_prepared();
            view.model = undistorted ? CameraModelType::PINHOLE
                                     : static_cast<CameraModelType>(static_cast<int>(camera.camera_model_type()));
            view.width = static_cast<uint32_t>(camera.image_width());
            view.height = static_cast<uint32_t>(camera.image_height());
            std::tie(view.fx, view.fy, view.cx, view.cy) = camera.get_intrinsics();
            if (!undistorted) {
                copy_coefficients(camera.radial_distortion(), view.radial);
                if (view.model == CameraModelType::THIN_PRISM_FISHEYE) {
                    copy_coefficients(camera.tangential_distortion(), view.thin_prism);
                } else if (view.model == CameraModelType::PINHOLE) {
                    copy_coefficients(camera.tangential_distortion(), view.tangential);
                    view.has_distortion = camera.radial_distortion().is_valid() && camera.radial_distortion().numel() > 0;
                    view.has_distortion |= camera.tangential_distortion().is_valid() && camera.tangential_distortion().numel() > 0;
                }
            }
            auto matrix = camera.world_view_transform().cpu().contiguous();
            std::memcpy(view.viewmat, matrix.ptr<float>(), sizeof(view.viewmat));

            const auto& motion = camera.rolling_shutter();
            if (motion && !undistorted && view.model != CameraModelType::EQUIRECTANGULAR) {
                const bool rows = motion->readout_rows;
                const int extent = static_cast<int>(rows ? view.height : view.width);
                if (extent > 1) {
                    auto R = camera.R().cpu().contiguous();
                    auto T = camera.T().cpu().contiguous();
                    std::array<float, 9> rotation{};
                    std::array<float, 3> translation{};
                    std::memcpy(rotation.data(), R.ptr<float>(), sizeof(rotation));
                    std::memcpy(translation.data(), T.ptr<float>(), sizeof(translation));
                    const auto poses = rolling_shutter_window_poses(*motion, rotation, translation, extent, 0, extent);
                    copy_matrix(poses.first, view.readout0);
                    copy_matrix(poses.last, view.readout1);
                    view.shutter = rows ? ShutterType::ROLLING_TOP_TO_BOTTOM : ShutterType::ROLLING_LEFT_TO_RIGHT;
                }
            }
            return view;
        }

        [[nodiscard]] std::array<float, 9> intrinsics_matrix(const gsplat_lfs::TsdfView& view) {
            if (view.model == CameraModelType::EQUIRECTANGULAR) {
                // The equirectangular model reads the full image size and tile offsets from K.
                return {static_cast<float>(view.width), 0.f, 0.f, 0.f, static_cast<float>(view.height), 0.f,
                        0.f, 0.f, 1.f};
            }
            return {view.fx, 0.f, view.cx, 0.f, view.fy, view.cy, 0.f, 0.f, 1.f};
        }

        // Restores the cameras' image sizes however the extraction ends.
        struct CameraSizes {
            struct Saved {
                core::Camera* camera;
                int width, height;
                bool loaded;
            };
            std::vector<Saved> saved;
            ~CameraSizes() {
                for (const auto& s : saved)
                    s.camera->restore_image_dimensions(s.width, s.height, s.loaded);
            }
        };

        // Renders a view's median depth, masked. The result stays valid until the next render.
        [[nodiscard]] Tensor render_median_depth(core::Camera& camera, core::SplatData& model,
                                                 const int max_width, cudaStream_t stream) {
            const size_t width = camera.image_width(), height = camera.image_height();
            auto depth = Tensor::empty({height, width}, Device::CUDA);
            depth.set_stream(stream);
            gsplat_lfs::FreeSpaceRender request;
            request.median_depth = depth.ptr<float>();
            Tensor background;
            auto forward = gsplat_rasterize_forward(camera, model, background, 0, 0, 0, 0, 1.f, false,
                                                    GsplatRenderMode::D, true, {}, request);
            if (!forward)
                throw collisionError("mesh.render_failed", forward.error(), ErrorCode::Internal);
            const cudaStream_t render_stream = forward->second.stream;
            core::GlobalArenaManager::instance().get_arena().end_frame(forward->second.frame_id, render_stream);
            // A worker without a current stream renders on the model's.
            LFS_ENSURE_CUDA_SUCCESS(cudaStreamSynchronize(render_stream), "Mesh render");
            if (camera.has_mask()) {
                auto mask = camera.load_mask_file(camera.mask_path(), 1, max_width, false, .5f, true, false)
                                .to(Device::CUDA, stream);
                if (mask.numel() == width * height) {
                    depth = depth.mul(mask.reshape({int(height), int(width)}).to(core::DataType::Float32));
                } else {
                    LOG_WARN("GOF mesh: mask size of {} does not match the render; using the view unmasked", camera.image_name());
                }
            }
            return depth.contiguous();
        }

        [[nodiscard]] Tensor contiguous_splats(const Tensor& source) {
            auto out = source.contiguous();
            return out;
        }
    } // namespace

    SplatMesh extract_splat_mesh_gof(core::SplatData& model, const std::span<core::Camera* const> cameras,
                                     const GofMeshSettings& settings, const std::stop_token stop,
                                     const core::ProgressCallback& progress) {
        validate(settings);
        const auto stage = [](const std::string_view key, const char* diagnostic) { return std::string(core::collisionStage(key, diagnostic)); };
        std::vector<core::Camera*> views;
        for (size_t i = 0; i < cameras.size(); i += settings.view_stride)
            if (cameras[i])
                views.push_back(cameras[i]);
        if (views.empty())
            throw collisionError("mesh.no_cameras", "The mesh needs the cameras of the dataset; load the project or dataset the model was trained on");
        if (model.size() == 0)
            throw collisionError("collision.geometry_invalid", "The model has no splats", ErrorCode::InvalidArgument);

        const auto stream = core::getCurrentCUDAStream();
        SplatMesh mesh;
        mesh.views = views.size();

        CameraSizes sizes;
        for (auto* camera : views) {
            sizes.saved.push_back({camera, camera->image_width(), camera->image_height(), camera->image_size_loaded()});
            camera->load_image_size(1, settings.max_width);
        }

        auto step = Clock::now();
        auto means = contiguous_splats(model.get_means());
        auto quats = contiguous_splats(model.rotation_raw());
        auto scales = contiguous_splats(model.scaling_raw());
        auto opacities = model.opacity_raw().contiguous();
        if (opacities.ndim() == 2 && opacities.shape()[1] == 1)
            opacities = opacities.squeeze(-1).contiguous();
        const uint32_t splats = static_cast<uint32_t>(means.shape()[0]);

        // Splats no view sees would read as occupied in every view's "no vote" and grow a shell.
        DeviceArray<uint8_t> visible(splats, stream, true);
        for (size_t i = 0; i < views.size(); ++i) {
            core::reportProgress(stop, progress, .1f * float(i) / views.size(), stage("mesh.stage_render", "Culling splats"));
            gsplat_lfs::gof_mark_visible(means.ptr<float>(), splats, make_view(*views[i]), visible.data(), stream);
        }
        LFS_ENSURE_CUDA_SUCCESS(cudaStreamSynchronize(stream), "GOF visibility");

        // Candidate voxels: one per visible splat, deduplicated, then dilated by one voxel so the
        // surface can be found between them and the free space around them.
        DeviceArray<uint64_t> seeds(splats, stream), counter(1, stream);
        uint64_t seeded = gsplat_lfs::gof_seed_keys(means.ptr<float>(), opacities.ptr<float>(), visible.data(), splats,
                                                    settings.voxel_size, settings.min_opacity, seeds.data(), counter.data(), stream);
        if (seeded == 0)
            throw collisionError("mesh.empty", "No splat survived the opacity and view filter; lower the minimum opacity");
        DeviceArray<uint64_t> seed_scratch(seeded, stream);
        seeded = gsplat_lfs::tsdf_sort_unique(seeds.data(), seed_scratch.data(), seeded, counter.data(), stream);
        seed_scratch = {};

        const size_t candidates = core::checked_product(seeded, size_t{27}, "GOF candidate voxels");
        DeviceArray<uint64_t> voxels(candidates, stream), voxel_scratch(candidates, stream);
        gsplat_lfs::gof_dilate_keys(seeds.data(), static_cast<uint32_t>(seeded), voxels.data(), stream);
        seeds = {};
        const uint64_t point_count = gsplat_lfs::tsdf_sort_unique(voxels.data(), voxel_scratch.data(), candidates,
                                                                  counter.data(), stream);
        if (point_count == 0)
            throw collisionError("mesh.empty", "No candidate voxel survived; use a larger voxel size");
        voxel_scratch = {};
        LOG_INFO("GOF mesh: {} splats, {} seeded voxels, {} candidate voxels", splats, seeded, point_count);

        DeviceArray<float> occlusion(point_count, stream);
        gsplat_lfs::gof_reset_occlusion(occlusion.data(), static_cast<uint32_t>(point_count), stream);
        mesh.seconds_render = seconds_since(step);

        // Every view votes with the occlusion in front of each voxel; the minimum wins.
        step = Clock::now();
        double seconds_fragments = 0.;
        for (size_t i = 0; i < views.size(); ++i) {
            core::reportProgress(stop, progress, .1f + .6f * float(i) / views.size(), stage("mesh.stage_fuse", "Integrating opacity"));
            const auto view = make_view(*views[i]);
            const size_t pixels = core::checked_product(view.width, view.height, "GOF pixels");
            if (pixels == 0)
                continue;

            const auto begin = Clock::now();
            DeviceArray<int32_t> radii(size_t(splats) * 2, stream, true);
            DeviceArray<float> means2d(size_t(splats) * 2, stream, true);
            DeviceArray<float> depths(splats, stream, true);
            DeviceArray<float> conics(size_t(splats) * 3, stream, true);
            const auto K_host = intrinsics_matrix(view);
            auto K_device = Tensor::from_vector(std::vector<float>(K_host.begin(), K_host.end()), {9}, Device::CUDA);
            K_device.set_stream(stream);
            gsplat_lfs::gof_project_view(means.ptr<float>(), quats.ptr<float>(), scales.ptr<float>(), opacities.ptr<float>(),
                                         K_device.ptr<float>(), splats, view, radii.data(), means2d.data(),
                                         depths.data(), conics.data(), stream);

            DeviceArray<uint32_t> counts(pixels, stream);
            DeviceArray<uint64_t> total(1, stream);
            const uint64_t fragments = gsplat_lfs::gof_count_fragments(
                radii.data(), means2d.data(), conics.data(), opacities.ptr<float>(), splats, view,
                kMinFragmentAlpha, counts.data(), total.data(), stream);
            seconds_fragments += seconds_since(begin);
            if (fragments == 0)
                continue;
            if (fragments > kMaxFragments)
                throw collisionError("mesh.memory",
                                     "One view needs more splat fragments than the mesh budget allows; use a smaller image width",
                                     ErrorCode::ResourceExhausted, std::format("{} fragments", fragments));
            device_buffer_preflight(fragments * kFragmentAllowance, "GOF fragments");

            DeviceArray<uint32_t> starts(pixels, stream), cursor(pixels, stream);
            gsplat_lfs::gof_scan_counts(counts.data(), starts.data(), static_cast<uint32_t>(pixels), stream);
            LFS_CUDA_CHECK(cudaMemcpyAsync(cursor.data(), starts.data(), pixels * sizeof(uint32_t), cudaMemcpyDeviceToDevice, stream));

            DeviceArray<uint32_t> pixels_used(fragments, stream), values(fragments, stream), values_sorted(fragments, stream), pixels_sorted(fragments, stream);
            DeviceArray<uint64_t> keys(fragments, stream), keys_sorted(fragments, stream);
            DeviceArray<float> depth_used(fragments, stream), log_t_used(fragments, stream), depth_sorted(fragments, stream), log_t_sorted(fragments, stream);

            gsplat_lfs::gof_fill_fragments(radii.data(), means2d.data(), depths.data(), conics.data(), opacities.ptr<float>(),
                                           splats, view, kMinFragmentAlpha, cursor.data(), pixels_used.data(),
                                           depth_used.data(), log_t_used.data(), stream);
            gsplat_lfs::gof_sort_fragments(pixels_used.data(), log_t_used.data(), depth_used.data(), fragments,
                                           keys.data(), keys_sorted.data(), values.data(), values_sorted.data(),
                                           pixels_sorted.data(), depth_sorted.data(), log_t_sorted.data(), stream);
            gsplat_lfs::gof_prefix_fragments(starts.data(), counts.data(), log_t_sorted.data(),
                                             static_cast<uint32_t>(pixels), stream);
            gsplat_lfs::gof_query_points(voxels.data(), static_cast<uint32_t>(point_count), settings.voxel_size, view,
                                         starts.data(), counts.data(), depth_sorted.data(), log_t_sorted.data(),
                                         occlusion.data(), stream);
            LFS_ENSURE_CUDA_SUCCESS(cudaStreamSynchronize(stream), "GOF view");
        }
        mesh.seconds_fusion = seconds_since(step) - seconds_fragments;

        // surface_nets wants positive values in free space, while the occlusion is high inside.
        step = Clock::now();
        core::reportProgress(stop, progress, .75f, stage("mesh.stage_extract", "Extracting surface"));
        std::vector<uint64_t> keys_host(point_count);
        std::vector<float> occlusion_host(point_count);
        LFS_CUDA_CHECK(cudaMemcpyAsync(keys_host.data(), voxels.data(), point_count * sizeof(uint64_t), cudaMemcpyDeviceToHost, stream));
        LFS_CUDA_CHECK(cudaMemcpyAsync(occlusion_host.data(), occlusion.data(), point_count * sizeof(float), cudaMemcpyDeviceToHost, stream));
        LFS_ENSURE_CUDA_SUCCESS(cudaStreamSynchronize(stream), "GOF voxel download");
        std::vector<float> sdf(point_count);
        for (size_t i = 0; i < point_count; ++i)
            sdf[i] = .5f - occlusion_host[i];
        occlusion = {};
        occlusion_host = {};
        mesh.band_voxels = point_count;
        surface_nets(keys_host, sdf, settings.voxel_size, mesh);
        keys_host = {};
        sdf = {};
        remove_small_mesh_components(mesh, settings.min_component_faces);
        if (mesh.faces.empty())
            throw collisionError("mesh.empty", "No surface survived; lower the minimum opacity or use a larger voxel size");
        compute_splat_mesh_normals(mesh);
        mesh.seconds_extract = seconds_since(step);

        step = Clock::now();
        const auto vertex_count = static_cast<uint32_t>(mesh.vertices.size());
        DeviceArray<float> vertices_device(size_t(vertex_count) * 3, stream);
        LFS_CUDA_CHECK(cudaMemcpyAsync(vertices_device.data(), mesh.vertices.data(), size_t(vertex_count) * 3 * sizeof(float), cudaMemcpyHostToDevice, stream));
        DeviceArray<float> color_sum(size_t(vertex_count) * 3, stream, true);
        DeviceArray<float> color_weight(vertex_count, stream, true);
        const float tolerance = 2.5f * settings.voxel_size;
        for (size_t i = 0; i < views.size(); i += settings.color_stride) {
            core::reportProgress(stop, progress, .8f + .2f * float(i) / views.size(), stage("mesh.stage_color", "Coloring"));
            auto& camera = *views[i];
            if (!camera.has_image())
                continue;
            const auto view = make_view(camera);
            auto photo = camera.load_and_get_image(1, settings.max_width, false, false);
            const size_t pixels = size_t(view.width) * view.height;
            if (photo.numel() != pixels * 3) {
                LOG_WARN("GOF mesh: photo size of {} does not match the render; skipping its colors", camera.image_name());
                continue;
            }
            photo = photo.to(Device::CUDA, stream).to(core::DataType::Float32).contiguous();
            auto depth = render_median_depth(camera, model, settings.max_width, stream);
            gsplat_lfs::mesh_color_accumulate(vertices_device.data(), vertex_count, view, depth.ptr<float>(),
                                              photo.ptr<float>(), tolerance, color_sum.data(), color_weight.data(), stream);
            LFS_ENSURE_CUDA_SUCCESS(cudaStreamSynchronize(stream), "GOF mesh color");
        }
        std::vector<float> sums(size_t(vertex_count) * 3);
        std::vector<float> weights(vertex_count);
        LFS_CUDA_CHECK(cudaMemcpyAsync(sums.data(), color_sum.data(), sums.size() * sizeof(float), cudaMemcpyDeviceToHost, stream));
        LFS_CUDA_CHECK(cudaMemcpyAsync(weights.data(), color_weight.data(), weights.size() * sizeof(float), cudaMemcpyDeviceToHost, stream));
        LFS_ENSURE_CUDA_SUCCESS(cudaStreamSynchronize(stream), "GOF mesh color download");
        mesh.colors.resize(vertex_count);
        size_t colored = 0;
        for (size_t v = 0; v < vertex_count; ++v) {
            colored += weights[v] > 0.f;
            for (int c = 0; c < 3; ++c) {
                const float value = weights[v] > 0.f ? 255.f * sums[3 * v + c] / weights[v] : 128.f;
                mesh.colors[v][c] = static_cast<uint8_t>(std::clamp(std::lround(value), 0l, 255l));
            }
        }
        mesh.colored_vertex_share = float(colored) / float(vertex_count);
        mesh.seconds_color = seconds_since(step);
        core::reportProgress(stop, progress, 1.f, stage("mesh.stage_color", "Coloring"));
        return mesh;
    }

} // namespace lfs::training
