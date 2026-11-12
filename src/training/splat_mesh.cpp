/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#define NOMINMAX
#include "splat_mesh.hpp"
#include "splat_mesh_parallel.hpp"

#if defined(_WIN32)
#  include <windows.h>
#  include <psapi.h>
#endif

#include "components/ppisp.hpp"
#include "components/rolling_shutter.hpp"
#include "core/camera.hpp"
#include "core/checked_arithmetic.hpp"
#include "core/collision_volume.hpp"
#include "core/cuda/memory_arena.hpp"
#include "core/host_metrics.hpp"
#include "core/image_io.hpp"
#include "core/image_loader.hpp"
#include "core/logger.hpp"
#include "core/memory_pressure.hpp"
#include "core/mesh_data.hpp"
#include "core/native_fs.hpp"
#include "core/path_utils.hpp"
#include "core/splat_data.hpp"
#include "core/tensor/internal/cub_workspace.hpp"
#include "core/tensor/internal/cuda_stream_context.hpp"
#include "rasterization/gsplat/GeometryFeatures.h"
#include "rasterization/gsplat/TsdfFusion.h"
#include "rasterization/gsplat_rasterizer.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <format>
#include <fstream>
#include <meshoptimizer.h>
#include <mutex>
#include <new>
#include <numeric>
#include <queue>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <utility>

namespace lfs::training {
    namespace {
        using core::collisionError;
        using core::Device;
        using core::Tensor;
        using Clock = std::chrono::steady_clock;

        constexpr uint64_t kAxisStep[3] = {uint64_t{1}, uint64_t{1} << gsplat_lfs::kTsdfKeyBits,
                                           uint64_t{1} << (2 * gsplat_lfs::kTsdfKeyBits)};
        constexpr uint64_t kAxisMask = (uint64_t{1} << gsplat_lfs::kTsdfKeyBits) - 1;
        // Against whole-mesh reduction on a 2 cm indoor scene at 5 mm: 17% fewer faces, 0.02 dB more color
        // against the photos and a p99 deviation of 6.0 instead of 6.9 mm. At half this, 38% fewer faces and 0.02 dB less.
        constexpr float kSimplifyColorWeight = 0.9f;
        // The simplifier's single-precision quadrics lose a tolerance once a plane lies this many
        // tolerances from the corner of its piece: at 5 mm, 25 m beyond a 6 m crop keeps 39% of the
        // faces instead of 13%, and 100 m keeps 63%. Smaller pieces gain nothing and grow the seams.
        constexpr float kPieceExtentPerTolerance = 4000.f;
        constexpr double kMeshShareOfFreeMemory = 0.8;
        constexpr double kTsdfShareOfFreeMemory = 0.8;
        // The rest holds a depth map, a photo and its blurs at the bake width.
        constexpr double kTextureShareOfFreeMemory = 0.7;
        // A batch of pages holds 52 bytes per covered texel on the GPU, whatever the size of the scene, and each batch reads
        // the photos that see it once more. This takes an indoor scene in one batch and an outdoor one in 4, against 3 at
        // 128M: 203 s against 209 s of baking, at 9.9 GB of GPU memory against 12.0.
        constexpr size_t kBakeBatchTexels = 80'000'000;
        // Pages of a batch are composed this many covered texels at a time, 32 bytes each in RAM.
        constexpr size_t kComposeChunkTexels = 16'000'000;
        // Detail below this share of the image comes from the view chosen for the face, and only broader colors are leveled
        // across views. Smaller lets the mean of misregistered views blur lettering, which spans about 1/100 of the image on
        // an indoor scene; larger shows the seams between views on white walls at 1/50 and bands colored lights at 1/20.
        constexpr int kBlendWidthPerRadius = 100;
        // The surface of a reduced level is within about 1.4 times its allowed error, and depth maps have their own.
        constexpr float kFaceErrorToTolerance = 2.f;
        constexpr int kOsgbMaxLevels = 4;
        constexpr int kOsgbMaxPage = 4096;
        // Sharper weights rest each vertex on fewer views and lost color on both test scenes.
        constexpr int kSimplifyThreads = 8;
        // What RAM use rises by from the end of fusion to its peak, as measured on an outdoor scene at
        // 4 cm, 2 cm and 1 cm with a range limit and an indoor one at 2 cm (5.6M to 105M voxels): within
        // 0.2 GB of 0.3 GB + 65 bytes per voxel. Faces per voxel stay within 1.5 to 1.8.
        constexpr double kRamBaseBytes = 0.3 * (1ull << 30);
        constexpr double kRamBytesPerVoxel = 65.;
        // The auto voxel is this fraction of one median splat spacing: slightly finer than the
        // point density so thin structures survive, without paying for sub-spacing voxels.
        constexpr float kAutoVoxelDetail = .75f;
        // Bounds on the automatic size: below the minimum the voxel grid explodes in memory, above
        // the maximum the surface gets too coarse to be worth fusing.
        constexpr float kAutoVoxelSizeMin = .015f;
        constexpr float kAutoVoxelSizeMax = .2f;
        // Automatic band cap, as a multiple of the truncation (a zero trunc_cap asks for this).
        constexpr float kAutoTruncCapBands = 5.f;
        // Automatic simplify tolerance, as a fraction of the voxel: just under one voxel keeps the
        // voxel detail without paying for the faces that only differ within a voxel.
        constexpr float kAutoSimplifyVoxels = .9f;
        // Automatic hole perimeter, as a multiple of the voxel: a gap narrower than twenty voxels is
        // a sampling hole, while a wider one is surface no view ever fused.
        constexpr float kAutoHolePerimeterVoxels = 20.f;
        // A rim edge longer than this many voxels is not a sampling gap inside a fused patch: either
        // the topology pass deleted faces around it, or several boundary loops meet at one vertex
        // and the walk stitched them together. Filling such a loop throws a sliver across the
        // surface, so the fill skips it and leaves the rim open instead.
        constexpr float kFillMaxEdgeVoxels = 100.f;
        // A rim whose area is this small against its bounding-box diagonal squared is a slit, not a
        // hole: merging coincident vertices welds two sheets together, the faces that end up sharing
        // an edge are dropped, and what stays open is a long, thin seam. Filling it zips the seam up
        // with long triangles that lie across the surface, which is what a sliver looks like.
        constexpr double kRimMinFillRatio = .001;

        constexpr uint32_t kAutoKeyBits = 21;
        constexpr uint32_t kAutoKeyLimit = 1u << kAutoKeyBits;

        // One-line RAM + GPU snapshot tag, useful to spot the phase that owns the peak of a run.
        // Windows-only — falls back gracefully on other platforms.
        static void log_phase_mem(const char* tag) {
            float ram_mb = 0.f, priv_mb = 0.f, gpu_used_mb = 0.f, gpu_total_mb = 0.f;
#if defined(_WIN32)
            PROCESS_MEMORY_COUNTERS_EX pmc{};
            pmc.cb = sizeof(pmc);
            GetProcessMemoryInfo(GetCurrentProcess(), reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&pmc), sizeof(pmc));
            ram_mb = pmc.WorkingSetSize / (1024.f * 1024.f);
            priv_mb = pmc.PrivateUsage / (1024.f * 1024.f);
#endif
            size_t free_bytes = 0, total_bytes = 0;
            if (const auto err = cudaMemGetInfo(&free_bytes, &total_bytes); err == cudaSuccess) {
                gpu_used_mb = (total_bytes - free_bytes) / (1024.f * 1024.f);
                gpu_total_mb = total_bytes / (1024.f * 1024.f);
            }
            LOG_INFO("MEM [{}]: RAM {:.0f} MB (priv {:.0f} MB), GPU {:.0f}/{:.0f} MB",
                     tag, ram_mb, priv_mb, gpu_used_mb, gpu_total_mb);
        }

        template <class T>
        struct DeviceArray {
            core::tensor_ops::ScopedDeviceBuffer storage;
            size_t count = 0;
            DeviceArray() = default;
            DeviceArray(const size_t n, cudaStream_t stream, const bool zeroed = false)
                : storage(core::checked_product(std::max<size_t>(n, 1), sizeof(T), "Mesh allocation"), stream, "splat_mesh"), count(n) {
                if (zeroed)
                    LFS_CUDA_CHECK(cudaMemsetAsync(data(), 0, count * sizeof(T), stream));
            }
            [[nodiscard]] T* data() const { return storage.template as<T>(); }
        };

        [[nodiscard]] double seconds_since(const Clock::time_point start) {
            return std::chrono::duration<double>(Clock::now() - start).count();
        }

        void validate(const core::SplatMeshSettings& s) {
            const bool ok = std::isfinite(s.voxel_size) && s.voxel_size >= 0 && s.block_voxels >= 2 && s.block_voxels <= 32 &&
                            std::isfinite(s.truncation) && s.truncation >= 0 && std::isfinite(s.fill_hole_perimeter) &&
                            s.fill_hole_perimeter >= 0 && std::isfinite(s.full_weight_range) &&
                            s.full_weight_range >= 0 && s.view_stride >= 1 && s.color_stride >= 1 && s.min_views >= 1 &&
                            s.min_views <= 65535 && s.max_width >= 16 &&
                            std::isfinite(s.simplify_error) && s.simplify_error >= 0 && std::isfinite(s.max_range) && s.max_range >= 0 &&
                            std::isfinite(s.trunc_gradient) && s.trunc_gradient >= 0 &&
                            std::isfinite(s.trunc_cap) && s.trunc_cap >= 0 && s.ear_cut_holes >= 0 &&
                            std::isfinite(s.spurious_edge_factor) && s.spurious_edge_factor >= 0 &&
                            std::isfinite(s.spurious_component_factor) && s.spurious_component_factor >= 0 &&
                            s.remove_isolated_small_components >= 0 && s.isolated_component_max_faces >= 0 &&
                            std::isfinite(s.isolated_component_neighbor_radius) && s.isolated_component_neighbor_radius >= 0.f &&
                            std::isfinite(s.texel_size) && s.texel_size >= 0 && (s.from_mesh.empty() || s.texel_size > 0) && s.texture_size >= 64 && s.texture_size <= 16384 && s.texture_width >= 0 && (s.texture_width == 0 || s.texture_width >= 16);
            if (!ok)
                throw collisionError("mesh.settings_invalid", "Invalid mesh settings", ErrorCode::InvalidArgument,
                                     std::format("{}; {}", s.voxel_size, s.truncation));
        }

        // The voxel size used when the caller leaves it at 0: the median spacing between
        // neighbouring splats. On a cell grid, cell / sqrt(points per cell) is the local spacing of
        // a surface sampling, and its point-weighted median is insensitive to dense and sparse
        // regions alike. The grid is coarsened until cells hold several points, then divided while
        // they still do; the finest such scale is the spacing, and finer grids only measure their cell.
        [[nodiscard]] float estimate_voxel_size(const core::SplatData& model) {
            constexpr size_t kMinPoints = 256;
            constexpr int kRounds = 4;
            constexpr float kEnoughPoints = .35f; // about eight points per cell
            const float fallback = core::SplatMeshSettings{}.voxel_size;
            const size_t count = model.size();
            if (count < kMinPoints)
                return fallback;
            const auto host = model.get_means().cpu().to(core::DataType::Float32).contiguous();
            if (!host.is_valid() || host.numel() != count * 3)
                return fallback;
            const float* points = host.ptr<float>();
            std::array<float, 3> low{FLT_MAX, FLT_MAX, FLT_MAX}, high{-FLT_MAX, -FLT_MAX, -FLT_MAX};
            for (size_t i = 0; i < count; ++i) {
                const float* p = points + 3 * i;
                if (!std::isfinite(p[0]) || !std::isfinite(p[1]) || !std::isfinite(p[2]))
                    continue;
                for (int axis = 0; axis < 3; ++axis) {
                    low[axis] = std::min(low[axis], p[axis]);
                    high[axis] = std::max(high[axis], p[axis]);
                }
            }
            const double dx = double(high[0]) - double(low[0]);
            const double dy = double(high[1]) - double(low[1]);
            const double dz = double(high[2]) - double(low[2]);
            const double diagonal = std::sqrt(dx * dx + dy * dy + dz * dz);
            if (!(diagonal > 0.) || !std::isfinite(diagonal))
                return fallback;
            float cell = float(diagonal / 1024.);
            float spacing = fallback;
            // Measures the point-weighted median of cell / sqrt(points per cell). Returns false when
            // the cells hold too few points for that quotient to be a spacing rather than the cell.
            const auto measure = [&](const float cell_size, float& result) {
                std::vector<uint64_t> keys;
                keys.reserve(count);
                const auto axis_bits = [&](const float value, const int axis) {
                    const double index = double(value - low[axis]) / double(cell_size);
                    return uint64_t(std::min(std::max(index, 0.), double(kAutoKeyLimit - 1)));
                };
                for (size_t i = 0; i < count; ++i) {
                    const float* p = points + 3 * i;
                    if (!std::isfinite(p[0]) || !std::isfinite(p[1]) || !std::isfinite(p[2]))
                        continue;
                    keys.push_back(axis_bits(p[0], 0) | (axis_bits(p[1], 1) << kAutoKeyBits) |
                                   (axis_bits(p[2], 2) << (2 * kAutoKeyBits)));
                }
                if (keys.size() < kMinPoints)
                    return false;
                std::sort(keys.begin(), keys.end());
                std::vector<std::pair<float, uint32_t>> occupied;
                occupied.reserve(keys.size());
                for (size_t i = 0; i < keys.size();) {
                    const size_t first = i;
                    while (++i < keys.size() && keys[i] == keys[first]) {
                    }
                    const uint64_t per_cell = i - first;
                    occupied.emplace_back(cell_size / std::sqrt(float(per_cell)), static_cast<uint32_t>(per_cell));
                }
                std::sort(occupied.begin(), occupied.end());
                uint64_t seen = 0;
                for (const auto& [value, weight] : occupied) {
                    seen += weight;
                    if (2 * seen >= keys.size()) {
                        result = value;
                        return result <= kEnoughPoints * cell_size;
                    }
                }
                return false;
            };
            // Coarsen until the cells hold several points, then divide while they still do: the
            // finest such scale is the local spacing, and finer grids only measure their own cell.
            bool enough = false;
            for (int round = 0; round < kRounds && !enough; ++round) {
                enough = measure(cell, spacing);
                if (!enough)
                    cell *= 2.f;
            }
            if (!enough)
                return std::isfinite(spacing) && spacing > 0.f ? spacing : fallback;
            for (int round = 0; round < kRounds; ++round) {
                float finer = spacing;
                if (!measure(.5f * cell, finer))
                    break;
                cell *= .5f;
                spacing = finer;
            }
            return std::isfinite(spacing) && spacing > 0.f ? spacing : fallback;
        }

        // Fails now, rather than after minutes of extraction or in the page file.
        void ram_preflight(const uint64_t voxels) {
            const auto host = core::host_metrics::sample();
            if (!host.ram_valid)
                return;
            const double needed = kRamBaseBytes + kRamBytesPerVoxel * double(voxels);
            const double available = double(host.system_total_bytes - host.system_used_bytes);
            LOG_INFO("Mesh: {} voxels need about {:.1f} GB of RAM, {:.1f} GB available", voxels, needed / (1ull << 30), available / (1ull << 30));
            if (needed > available)
                throw collisionError("mesh.ram", "Not enough RAM for the mesh; use a larger voxel size or a maximum range", ErrorCode::ResourceExhausted,
                                     std::format("{:.1f} GB needed, {:.1f} GB available", needed / (1ull << 30), available / (1ull << 30)));
        }

        // Pages stay in RAM at 3 bytes per page texel, of which covered texels fill most. A batch of them holds 16 bytes per
        // page texel while rasterizing and 23 per covered texel to bake. Unwrapping, choosing views and leveling add
        // about 100 bytes per face.
        void texture_ram_preflight(const SplatMesh& mesh, const core::SplatMeshSettings& s) {
            const auto host = core::host_metrics::sample();
            if (!host.ram_valid)
                return;
            const double page_texels = double(s.texture_size) * s.texture_size;
            // Levels of detail add a quarter of the texels at each step, and about half the faces in all.
            const double covered = 4. / 3. * 1.3 * mesh.area / (double(s.texel_size) * s.texel_size);
            const double faces = 1.5 * double(mesh.faces.size());
            const double needed = kSimplifyThreads * 16. * page_texels + 23. * std::min(covered, double(kBakeBatchTexels)) + 5. * covered + 100. * faces;
            const double available = double(host.system_total_bytes - host.system_used_bytes);
            LOG_INFO("Mesh: {:.0f}M texels need about {:.1f} GB of RAM, {:.1f} GB available", covered / 1e6, needed / (1ull << 30), available / (1ull << 30));
            if (needed > available)
                throw collisionError("mesh.texture_ram", "Not enough RAM for the mesh texture; use a larger texel size", ErrorCode::ResourceExhausted,
                                     std::format("{:.1f} GB needed, {:.1f} GB available", needed / (1ull << 30), available / (1ull << 30)));
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
            view.model = undistorted ? CameraModelType::PINHOLE : static_cast<CameraModelType>(static_cast<int>(camera.camera_model_type()));
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

        void camera_rays(const gsplat_lfs::TsdfView& view, float* rays, cudaStream_t stream) {
            const float K[9] = {view.fx, 0.f, view.cx, 0.f, view.fy, view.cy, 0.f, 0.f, 1.f};
            auto K_device = Tensor::from_vector(std::vector<float>(K, K + 9), {9}, Device::CUDA);
            const bool radial = view.model == CameraModelType::FISHEYE || view.model == CameraModelType::THIN_PRISM_FISHEYE || view.has_distortion;
            const bool tangential = view.model == CameraModelType::PINHOLE && view.has_distortion;
            auto radial_device = Tensor::from_vector(std::vector<float>(view.radial, view.radial + 6), {6}, Device::CUDA);
            auto tangential_device = Tensor::from_vector(std::vector<float>(view.tangential, view.tangential + 2), {2}, Device::CUDA);
            auto prism_device = Tensor::from_vector(std::vector<float>(view.thin_prism, view.thin_prism + 4), {4}, Device::CUDA);
            const bool prism = view.model == CameraModelType::THIN_PRISM_FISHEYE;
            gsplat_lfs::geometry_camera_rays(rays, view.width, view.height, K_device.ptr<float>(), view.model,
                                             radial ? radial_device.ptr<float>() : nullptr,
                                             tangential ? tangential_device.ptr<float>() : nullptr,
                                             prism ? prism_device.ptr<float>() : nullptr, stream);
            LFS_ENSURE_CUDA_SUCCESS(cudaStreamSynchronize(stream), "Mesh camera rays");
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

        // Diagnostic only. With LFS_MESH_DUMP_DIR set, writes every depth map the fusion actually
        // read as a colored image, plus the reasons collect dropped voxels. LFS_MESH_DUMP_RAW adds
        // the 16-bit millimetre copies. Nothing here runs unless the variable is set, and the mesh
        // never reads these files.
        struct MeshDepthDump {
            bool active = false;
            bool raw = false;
            std::filesystem::path vis_dir, raw_dir;
            std::ofstream stats, rejects;
            std::array<uint64_t, 5> reject_totals{};

            void open(const std::filesystem::path& dir, const core::SplatMeshSettings& s, const size_t view_count,
                      const float truncation) {
                raw = std::getenv("LFS_MESH_DUMP_RAW") != nullptr;
                std::error_code error;
                vis_dir = dir / "vis";
                std::filesystem::create_directories(vis_dir, error);
                if (raw) {
                    raw_dir = dir / "raw";
                    std::filesystem::create_directories(raw_dir, error);
                }
                stats.open(core::native_fs::native_path(dir / "depth_stats.csv"), std::ios::trunc);
                rejects.open(core::native_fs::native_path(dir / "collect_reject.csv"), std::ios::trunc);
                active = stats.is_open() && rejects.is_open();
                if (!active) {
                    LOG_WARN("Mesh: could not write the depth dump under {}; continuing without it", core::path_to_utf8(dir));
                    return;
                }
                stats << std::format(
                    "# voxel={} block={} truncation={} full_weight_range={} max_range={} view_stride={} min_views={} max_width={} views={}\n"
                    "index,uid,image,width,height,valid_pixels,valid_share,invalid_pixels,min_m,p05_m,p50_m,p95_m,max_m\n",
                    s.voxel_size, s.block_voxels, truncation, s.full_weight_range, s.max_range, s.view_stride, s.min_views,
                    s.max_width, view_count);
                rejects << "considered,views_below_min,outside_band,no_sign_change,emitted,emitted_share\n";
                LOG_INFO("Mesh: dumping the depth maps of {} views to {}", view_count, core::path_to_utf8(dir));
            }

            // 0 in the depth map is the rasterizer's "transmittance never fell below 0.5"; magenta in
            // the image marks it, so those pixels stand out from the blue-to-red depth ramp.
            void write(const size_t index, const int uid, const std::string& name, const uint32_t width, const uint32_t height,
                       const std::vector<float>& depth) {
                if (!active)
                    return;
                std::vector<float> valid;
                valid.reserve(depth.size());
                float lo = FLT_MAX, hi = -FLT_MAX;
                for (const float d : depth)
                    if (d > 0.f && std::isfinite(d)) {
                        valid.push_back(d);
                        lo = std::min(lo, d);
                        hi = std::max(hi, d);
                    }
                if (valid.empty()) {
                    lo = 0.f;
                    hi = 0.f;
                }
                const auto order = [&](const double q) {
                    const auto k = std::ptrdiff_t(q * double(valid.size() - 1));
                    std::nth_element(valid.begin(), valid.begin() + k, valid.end());
                    return valid[size_t(k)];
                };
                const float p05 = valid.empty() ? 0.f : order(.05);
                const float p50 = valid.empty() ? 0.f : order(.50);
                const float p95 = valid.empty() ? 0.f : order(.95);
                std::string file = sanitize(name);
                const auto dot = file.find_last_of('.');
                if (dot != std::string::npos)
                    file.erase(dot);
                file = std::format("{:04d}_{}.png", index, file);
                std::vector<uint8_t> rgb(depth.size() * 3);
                const float span = std::max(p95 - p05, 1e-4f);
                for (size_t i = 0; i < depth.size(); ++i) {
                    uint8_t* pixel = rgb.data() + 3 * i;
                    if (!(depth[i] > 0.f)) {
                        pixel[0] = 255;
                        pixel[1] = 0;
                        pixel[2] = 255;
                        continue;
                    }
                    jet(std::clamp((depth[i] - p05) / span, 0.f, 1.f), pixel);
                }
                if (!core::save_png(vis_dir / file, rgb.data(), int(width), int(height), 3, 8, 1))
                    LOG_WARN("Mesh: could not write the depth image {}", core::path_to_utf8(vis_dir / file));
                if (raw) {
                    std::vector<uint16_t> millimetres(depth.size());
                    for (size_t i = 0; i < depth.size(); ++i)
                        millimetres[i] = static_cast<uint16_t>(std::clamp<long>(std::lround(double(depth[i]) * 1000.), 0, 65535));
                    core::save_png(raw_dir / file, millimetres.data(), int(width), int(height), 1, 16, 1);
                }
                stats << std::format("{},{},{},{},{},{},{:.4f},{},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f}\n", index, uid, file, width, height,
                                     valid.size(), double(valid.size()) / double(width * height), depth.size() - valid.size(),
                                     lo, p05, p50, p95, hi);
            }

            void add_reject(const std::array<uint64_t, 5>& counts) {
                for (size_t i = 0; i < reject_totals.size(); ++i)
                    reject_totals[i] += counts[i];
            }

            void finish() {
                if (!active)
                    return;
                const uint64_t considered = reject_totals[0];
                rejects << std::format("{},{},{},{},{},{:.6f}\n", considered, reject_totals[1], reject_totals[2], reject_totals[3],
                                       reject_totals[4], considered > 0 ? double(reject_totals[4]) / double(considered) : 0.);
                stats.close();
                rejects.close();
                active = false;
            }

        private:
            static std::string sanitize(std::string name) {
                for (char& c : name)
                    if (c == '/' || c == '\\' || c == ':' || c == '*' || c == '?' || c == '"' || c == '<' || c == '>' || c == '|')
                        c = '_';
                return name;
            }

            static void jet(const float t, uint8_t* rgb) {
                const auto channel = [](const float x, const float offset) {
                    return static_cast<uint8_t>(std::clamp(1.5f - std::abs(4.f * x - offset), 0.f, 1.f) * 255.f);
                };
                rgb[0] = channel(t, 3.f);
                rgb[1] = channel(t, 2.f);
                rgb[2] = channel(t, 1.f);
            }
        };

        // Masks as one bit per pixel, for renders that come back to a view: decoding one takes about as long as
        // rendering it, and a texture bake renders each view three times or more.
        struct MaskCache {
            struct Entry {
                bool usable = false;
                std::vector<uint32_t> bits;
            };
            std::unordered_map<const core::Camera*, Entry> entries;
            DeviceArray<uint32_t> device;
        };

        // Renders a view's median depth, masked. The result stays valid until the next render.
        [[nodiscard]] Tensor render_median_depth(core::Camera& camera, core::SplatData& model,
                                                 const core::SplatMeshSettings& settings, cudaStream_t stream, MaskCache* masks = nullptr) {
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
                const auto pixels = static_cast<uint32_t>(width * height);
                const size_t words = (size_t(pixels) + 31) / 32;
                if (masks) {
                    if (const auto found = masks->entries.find(&camera); found != masks->entries.end()) {
                        if (found->second.usable) {
                            if (masks->device.count < words)
                                masks->device = DeviceArray<uint32_t>(words, stream);
                            LFS_CUDA_CHECK(cudaMemcpyAsync(masks->device.data(), found->second.bits.data(), words * sizeof(uint32_t), cudaMemcpyHostToDevice, stream));
                            depth = depth.contiguous();
                            gsplat_lfs::mesh_apply_mask(depth.ptr<float>(), pixels, masks->device.data(), stream);
                            // The host bits must outlive the copy.
                            LFS_ENSURE_CUDA_SUCCESS(cudaStreamSynchronize(stream), "Mesh mask");
                        }
                        return depth.contiguous();
                    }
                }
                auto mask = camera.load_mask_file(camera.mask_path(), 1, settings.max_width, settings.invert_mask,
                                                  settings.mask_threshold, true, false, false)
                                .to(Device::CUDA, stream);
                const bool usable = mask.numel() == width * height;
                if (usable) {
                    mask = mask.reshape({int(height), int(width)}).to(core::DataType::Float32).contiguous();
                    depth = depth.mul(mask);
                } else {
                    LOG_WARN("Mesh: mask size of {} does not match the render; using the view unmasked", camera.image_name());
                }
                if (masks) {
                    auto& entry = masks->entries[&camera];
                    entry.usable = usable;
                    if (usable) {
                        if (masks->device.count < words)
                            masks->device = DeviceArray<uint32_t>(words, stream);
                        gsplat_lfs::mesh_pack_mask(mask.ptr<float>(), pixels, masks->device.data(), stream);
                        entry.bits.resize(words);
                        LFS_CUDA_CHECK(cudaMemcpyAsync(entry.bits.data(), masks->device.data(), words * sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
                        LFS_ENSURE_CUDA_SUCCESS(cudaStreamSynchronize(stream), "Mesh mask bits");
                    }
                }
            }
            return depth.contiguous();
        }

        // Not through the loader's cache of re-encoded JPEGs: a second load of a photo from it can return
        // another photo's content, and every texture bake loads each photo a second time.
        [[nodiscard]] Tensor load_photo(core::Camera& camera, const int max_width) {
            return camera.load_and_get_image(1, max_width, false, false, [](core::ImageLoadParams params) {
                params.skip_blob_cache = true;
                return core::load_image_cached(params);
            });
        }

        struct Tsdf {
            gsplat_lfs::TsdfGrid grid;
            DeviceArray<uint64_t> keys;
            DeviceArray<float> sum, weight, carve;
            DeviceArray<uint16_t> views;
            DeviceArray<uint32_t> near_blocks, near_count;
            DeviceArray<float> raw_low, raw_high, low, high;
            uint32_t block_count = 0;
            size_t tiles = 0;
        };

        // Adds keys to a sorted unique set. Returns the new set; count is its size.
        [[nodiscard]] DeviceArray<uint64_t> merge_unique_keys(const DeviceArray<uint64_t>& existing, const size_t existing_count,
                                                              const DeviceArray<uint64_t>& added, size_t& count, cudaStream_t stream) {
            const size_t total = existing_count + added.count;
            DeviceArray<uint64_t> merged(total, stream), scratch(total, stream), counter(1, stream);
            if (existing_count > 0)
                LFS_CUDA_CHECK(cudaMemcpyAsync(merged.data(), existing.data(), existing_count * sizeof(uint64_t), cudaMemcpyDeviceToDevice, stream));
            LFS_CUDA_CHECK(cudaMemcpyAsync(merged.data() + existing_count, added.data(), added.count * sizeof(uint64_t), cudaMemcpyDeviceToDevice, stream));
            count = gsplat_lfs::tsdf_sort_unique(merged.data(), scratch.data(), total, counter.data(), stream);
            return merged;
        }

        [[nodiscard]] size_t tsdf_block_bytes(const gsplat_lfs::TsdfGrid& grid) {
            return size_t(grid.block) * grid.block * grid.block * 10 + 12;
        }

        // A run of whole block layers along z, the slowest axis of the sorted keys, with the
        // layers next to it for the neighbors of its surface voxels.
        struct TsdfChunk {
            size_t first = 0, last = 0;           // blocks whose voxels this chunk reports
            size_t held_first = 0, held_last = 0; // blocks it holds
        };

        [[nodiscard]] std::vector<TsdfChunk> plan_chunks(const std::vector<uint64_t>& keys, const size_t max_blocks) {
            const auto layer_of = [](const uint64_t key) { return key >> (2 * gsplat_lfs::kTsdfKeyBits); };
            std::vector<size_t> starts;
            for (size_t b = 0; b < keys.size(); ++b)
                if (b == 0 || layer_of(keys[b]) != layer_of(keys[b - 1]))
                    starts.push_back(b);
            starts.push_back(keys.size());
            const size_t layers = starts.size() - 1;
            std::vector<TsdfChunk> chunks;
            for (size_t layer = 0; layer < layers;) {
                const size_t held_first = starts[layer > 0 ? layer - 1 : 0];
                size_t end = layer + 1;
                const auto held_last = [&](const size_t after) { return starts[std::min(after + 1, layers)]; };
                while (end < layers && held_last(end + 1) - held_first <= max_blocks)
                    ++end;
                if (held_last(end) - held_first > max_blocks)
                    throw collisionError("mesh.memory", "Not enough GPU memory for the mesh; free memory or use a larger voxel size",
                                         ErrorCode::ResourceExhausted, std::format("{} blocks in 3 layers, room for {}", held_last(end) - held_first, max_blocks));
                chunks.push_back({starts[layer], starts[end], held_first, held_last(end)});
                layer = end;
            }
            return chunks;
        }

        void allocate_tsdf(Tsdf& tsdf, DeviceArray<uint64_t> keys, const size_t count, cudaStream_t stream) {
            const size_t voxels_per_block = size_t(tsdf.grid.block) * tsdf.grid.block * tsdf.grid.block;
            const size_t voxels = core::checked_product(count, voxels_per_block, "TSDF voxels");
            device_buffer_preflight(core::checked_product(count, tsdf_block_bytes(tsdf.grid), "TSDF bytes"), "Mesh TSDF");
            tsdf.keys = std::move(keys);
            tsdf.block_count = static_cast<uint32_t>(count);
            tsdf.sum = DeviceArray<float>(voxels, stream, true);
            tsdf.weight = DeviceArray<float>(voxels, stream, true);
            tsdf.views = DeviceArray<uint16_t>(voxels, stream, true);
            tsdf.carve = DeviceArray<float>(count, stream, true);
            tsdf.near_blocks = DeviceArray<uint32_t>(count, stream);
            tsdf.near_count = DeviceArray<uint32_t>(1, stream);
        }

        void integrate_view(Tsdf& tsdf, const gsplat_lfs::TsdfView& view, const Tensor& depth, cudaStream_t stream) {
            const size_t tiles = size_t(gsplat_lfs::tsdf_tile_count(view.width)) * gsplat_lfs::tsdf_tile_count(view.height);
            if (tiles != tsdf.tiles) {
                tsdf.tiles = tiles;
                for (auto* bounds : {&tsdf.raw_low, &tsdf.raw_high, &tsdf.low, &tsdf.high})
                    *bounds = DeviceArray<float>(tiles, stream);
            }
            gsplat_lfs::tsdf_tile_bounds(depth.ptr<float>(), view.width, view.height, tsdf.raw_low.data(), tsdf.raw_high.data(),
                                         tsdf.low.data(), tsdf.high.data(), stream);
            const uint32_t near_count = gsplat_lfs::tsdf_classify_blocks(tsdf.keys.data(), tsdf.block_count, view, tsdf.grid,
                                                                         tsdf.low.data(), tsdf.high.data(), tsdf.carve.data(),
                                                                         tsdf.near_blocks.data(), tsdf.near_count.data(), stream);
            gsplat_lfs::tsdf_integrate(tsdf.keys.data(), tsdf.near_blocks.data(), near_count, view, tsdf.grid, depth.ptr<float>(),
                                       tsdf.sum.data(), tsdf.weight.data(), tsdf.views.data(), stream);
        }

        struct SparseVoxels {
            std::vector<uint64_t> keys;
            std::vector<float> tsdf;
        };

        // Appends a chunk's voxels, sorted. Chunks arrive in key order, so all of them stay sorted.
        void collect_voxels(const Tsdf& tsdf, const uint32_t first_block, const uint32_t last_block, const uint32_t min_views, SparseVoxels& all,
                            MeshDepthDump& dump, cudaStream_t stream) {
            DeviceArray<uint64_t> counter(1, stream);
            const bool tally = dump.active;
            DeviceArray<uint64_t> reasons(tally ? 5 : 0, stream, true);
            const auto collect = [&](uint64_t* keys, float* values, uint64_t* counts) {
                return gsplat_lfs::tsdf_collect_voxels(tsdf.keys.data(), tsdf.block_count, first_block, last_block, tsdf.grid, tsdf.sum.data(),
                                                       tsdf.weight.data(), tsdf.views.data(), tsdf.carve.data(),
                                                       min_views, keys, values, counter.data(), stream, counts);
            };
            const uint64_t count = collect(nullptr, nullptr, tally ? reasons.data() : nullptr);
            if (tally) {
                std::array<uint64_t, 5> host{};
                LFS_CUDA_CHECK(cudaMemcpyAsync(host.data(), reasons.data(), host.size() * sizeof(uint64_t), cudaMemcpyDeviceToHost, stream));
                LFS_ENSURE_CUDA_SUCCESS(cudaStreamSynchronize(stream), "Mesh collect reasons");
                dump.add_reject(host);
            }
            if (count == 0)
                return;
            device_buffer_preflight(count * 12, "Mesh voxels");
            DeviceArray<uint64_t> keys(count, stream);
            DeviceArray<float> values(count, stream);
            (void)collect(keys.data(), values.data(), nullptr);
            std::vector<uint64_t> host_keys(count);
            std::vector<float> host_values(count);
            LFS_CUDA_CHECK(cudaMemcpyAsync(host_keys.data(), keys.data(), count * sizeof(uint64_t), cudaMemcpyDeviceToHost, stream));
            LFS_CUDA_CHECK(cudaMemcpyAsync(host_values.data(), values.data(), count * sizeof(float), cudaMemcpyDeviceToHost, stream));
            LFS_ENSURE_CUDA_SUCCESS(cudaStreamSynchronize(stream), "Mesh voxel download");
            std::vector<uint32_t> order(count);
            std::iota(order.begin(), order.end(), 0u);
            std::sort(order.begin(), order.end(), [&](const uint32_t a, const uint32_t b) { return host_keys[a] < host_keys[b]; });
            const size_t base = all.keys.size();
            all.keys.resize(base + count);
            all.tsdf.resize(base + count);
            for (size_t i = 0; i < count; ++i) {
                all.keys[base + i] = host_keys[order[i]];
                all.tsdf[base + i] = host_values[order[i]];
            }
        }
    } // namespace

    void surface_nets(const std::span<const uint64_t> keys, const std::span<const float> tsdf, const float voxel, SplatMesh& mesh,
                      const float max_step) {
        // Calls found(a, b, axis) for each voxel pair along an axis where the surface crosses, in key order per axis.
        const auto for_each_crossing = [&](const int axis, auto&& found) {
            const uint64_t step = kAxisStep[axis];
            // key + step is sorted too, so one walk finds every neighbor.
            size_t other = 0;
            for (size_t a = 0; a < keys.size(); ++a) {
                const uint64_t target = keys[a] + step;
                while (other < keys.size() && keys[other] < target)
                    ++other;
                if (other == keys.size())
                    break;
                if (keys[other] != target)
                    continue;
                const float va = tsdf[a], vb = tsdf[other];
                // A jump from free space to the far end of the band is where two observations meet, not a surface.
                // Grazing views steepen real crossings well past one voxel's share of the band, so this stays permissive.
                if ((va < 0) == (vb < 0) || !(std::abs(va - vb) < max_step))
                    continue;
                found(a, va, vb);
            }
        };
        // The cells around a crossing, in the order that faces it outward.
        const auto cell_offsets = [](const int axis, uint64_t(&offsets)[4]) {
            const uint64_t step_j = kAxisStep[(axis + 1) % 3], step_k = kAxisStep[(axis + 2) % 3];
            offsets[0] = 0;
            offsets[1] = step_j;
            offsets[2] = step_j + step_k;
            offsets[3] = step_k;
        };

        // Every cell touched, sorted and unique. A vertex per cell.
        const auto crossing_start = Clock::now();
        std::vector<uint64_t> cells;
        size_t quad_count = 0;
        for (int axis = 0; axis < 3; ++axis) {
            const auto axis_start = Clock::now();
            uint64_t offsets[4];
            cell_offsets(axis, offsets);
            const size_t before = cells.size();
            for_each_crossing(axis, [&](const size_t a, float, float) {
                for (const auto offset : offsets)
                    cells.push_back(keys[a] - offset);
                ++quad_count;
            });
            if (cells.size() != before) {
                std::sort(cells.begin() + std::ptrdiff_t(before), cells.end());
                std::inplace_merge(cells.begin(), cells.begin() + std::ptrdiff_t(before), cells.end());
                cells.erase(std::unique(cells.begin(), cells.end()), cells.end());
            }
            LOG_INFO("Mesh: surface nets axis {} cells in {:.2f}s, {} cells, {} quads so far", axis,
                     seconds_since(axis_start), cells.size(), quad_count);
        }
        cells.shrink_to_fit();
        LOG_INFO("Mesh: surface nets crossings in {:.2f}s, {} cells from {} quads", seconds_since(crossing_start), cells.size(), quad_count);
        if (cells.empty())
            return;

        const auto corners_start = Clock::now();
        // Each vertex averages the crossings of its cell.
        std::vector<std::array<double, 3>> sums(cells.size(), {0., 0., 0.});
        std::vector<uint8_t> crossings(cells.size(), 0);
        mesh.faces.clear();
        mesh.faces.reserve(quad_count * 2);
        for (int axis = 0; axis < 3; ++axis) {
            const auto axis_start = Clock::now();
            uint64_t offsets[4];
            cell_offsets(axis, offsets);
            // The keys rise along the walk, so each corner's cell only moves forward.
            size_t cursor[4] = {0, 0, 0, 0};
            for_each_crossing(axis, [&](const size_t a, const float va, const float vb) {
                float p[3] = {float(int64_t(keys[a] & kAxisMask) - gsplat_lfs::kTsdfKeyOffset),
                              float(int64_t((keys[a] >> gsplat_lfs::kTsdfKeyBits) & kAxisMask) - gsplat_lfs::kTsdfKeyOffset),
                              float(int64_t((keys[a] >> (2 * gsplat_lfs::kTsdfKeyBits)) & kAxisMask) - gsplat_lfs::kTsdfKeyOffset)};
                p[axis] += va / (va - vb);
                const float x = p[0] * voxel, y = p[1] * voxel, z = p[2] * voxel;
                int32_t corner[4];
                for (int c = 0; c < 4; ++c) {
                    const uint64_t cell = keys[a] - offsets[c];
                    while (cells[cursor[c]] < cell)
                        ++cursor[c];
                    const size_t vertex = cursor[c];
                    corner[c] = static_cast<int32_t>(vertex);
                    sums[vertex][0] += x;
                    sums[vertex][1] += y;
                    sums[vertex][2] += z;
                    ++crossings[vertex];
                }
                if (va >= 0)
                    std::reverse(corner, corner + 4);
                mesh.faces.push_back({corner[0], corner[1], corner[2]});
                mesh.faces.push_back({corner[0], corner[2], corner[3]});
            });
            LOG_INFO("Mesh: surface nets axis {} corners in {:.2f}s, {} faces so far", axis,
                     seconds_since(axis_start), mesh.faces.size());
        }
        cells = {};
        mesh.vertices.resize(sums.size());
        for (size_t v = 0; v < sums.size(); ++v) {
            const double n = crossings[v];
            mesh.vertices[v] = {float(sums[v][0] / n), float(sums[v][1] / n), float(sums[v][2] / n)};
        }
        mesh.raw_faces = mesh.faces.size();
        LOG_INFO("Mesh: surface nets corners in {:.2f}s, {} vertices, {} faces", seconds_since(corners_start),
                 mesh.vertices.size(), mesh.faces.size());
        log_phase_mem("surface_nets_done");
    }

    void remove_small_mesh_components(SplatMesh& mesh, const int min_faces) {
        std::vector<uint32_t> parent(mesh.vertices.size());
        std::iota(parent.begin(), parent.end(), 0u);
        const auto find = [&](uint32_t v) {
            while (parent[v] != v) {
                parent[v] = parent[parent[v]];
                v = parent[v];
            }
            return v;
        };
        for (const auto& f : mesh.faces) {
            parent[find(f[1])] = find(f[0]);
            parent[find(f[2])] = find(f[0]);
        }
        std::vector<uint32_t> sizes(mesh.vertices.size(), 0);
        for (const auto& f : mesh.faces)
            ++sizes[find(f[0])];
        mesh.components_total = std::ranges::count_if(sizes, [](const uint32_t s) { return s > 0; });
        mesh.components_kept = std::ranges::count_if(sizes, [&](const uint32_t s) { return s > 0 && int64_t(s) >= min_faces; });
        std::vector<int32_t> remap(mesh.vertices.size(), -1);
        std::vector<std::array<float, 3>> vertices;
        std::vector<std::array<int32_t, 3>> faces;
        faces.reserve(mesh.faces.size());
        for (const auto& f : mesh.faces) {
            if (int64_t(sizes[find(f[0])]) < min_faces)
                continue;
            std::array<int32_t, 3> mapped{};
            for (int i = 0; i < 3; ++i) {
                auto& target = remap[f[i]];
                if (target < 0) {
                    target = static_cast<int32_t>(vertices.size());
                    vertices.push_back(mesh.vertices[f[i]]);
                }
                mapped[i] = target;
            }
            faces.push_back(mapped);
        }
        mesh.vertices = std::move(vertices);
        mesh.faces = std::move(faces);
    }

    namespace {
        // One rim loop filled as a fan around a new center vertex that carries the rim's average
        // attributes. The fill also runs after coloring when the mesh was too large to clean before
        // simplification, so the new centre has to take those attributes from the rim itself.
        void fill_rim_by_fan(SplatMesh& mesh, const std::vector<int32_t>& loop) {
            // The rim runs along existing triangles, so the new ones run against it.
            if (loop.size() == 3) {
                mesh.faces.push_back({loop[2], loop[1], loop[0]});
                return;
            }
            std::array<double, 3> center{0., 0., 0.};
            for (const auto vertex : loop)
                for (int i = 0; i < 3; ++i)
                    center[i] += mesh.vertices[vertex][i];
            const auto middle = static_cast<int32_t>(mesh.vertices.size());
            const bool keeps_normals = mesh.normals.size() == mesh.vertices.size();
            const bool keeps_colors = mesh.colors.size() == mesh.vertices.size();
            std::array<float, 3> normal{0.f, 0.f, 0.f};
            std::array<uint8_t, 3> color{128, 128, 128};
            std::array<uint32_t, 3> color_sum{0, 0, 0};
            if (keeps_normals)
                for (const auto vertex : loop)
                    for (int i = 0; i < 3; ++i)
                        normal[i] += mesh.normals[vertex][i];
            if (keeps_colors)
                for (const auto vertex : loop)
                    for (int i = 0; i < 3; ++i)
                        color_sum[i] += mesh.colors[vertex][i];
            const double count = static_cast<double>(loop.size());
            mesh.vertices.push_back({float(center[0] / count), float(center[1] / count), float(center[2] / count)});
            if (keeps_normals) {
                const double length = std::max(std::sqrt(double(normal[0]) * normal[0] + double(normal[1]) * normal[1] +
                                                          double(normal[2]) * normal[2]),
                                               1e-30);
                mesh.normals.push_back({float(normal[0] / length), float(normal[1] / length), float(normal[2] / length)});
            }
            if (keeps_colors) {
                for (int i = 0; i < 3; ++i)
                    color[i] = static_cast<uint8_t>(std::clamp<uint32_t>(color_sum[i] / uint32_t(loop.size()), 0u, 255u));
                mesh.colors.push_back(color);
            }
            for (size_t i = 0; i < loop.size(); ++i)
                mesh.faces.push_back({loop[(i + 1) % loop.size()], loop[i], middle});
        }

        // ---------------------------------------------------------------------------------------
        // Ear filling ported from VCG's vcglib (vcg/complex/algorithms/hole.h), which is what OpenMVS
        // runs through Hole::EarCuttingIntersectionFill<SelfIntersectionEar>. Half-edge positions
        // become rim indices plus an edge/face table, but the traversal order, the ear weight and
        // every rejection test are the library's.
        // ---------------------------------------------------------------------------------------
        struct P3 {
            double x = 0., y = 0., z = 0.;
            [[nodiscard]] double operator[](const int axis) const { return axis == 0 ? x : (axis == 1 ? y : z); }
        };
        [[nodiscard]] inline P3 point_of(const std::array<float, 3>& p) {
            return {double(p[0]), double(p[1]), double(p[2])};
        }
        [[nodiscard]] inline P3 operator+(const P3 a, const P3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
        [[nodiscard]] inline P3 operator-(const P3 a, const P3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
        [[nodiscard]] inline P3 operator*(const P3 a, const double s) { return {a.x * s, a.y * s, a.z * s}; }
        [[nodiscard]] inline double dot(const P3 a, const P3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
        [[nodiscard]] inline P3 cross(const P3 a, const P3 b) {
            return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
        }
        [[nodiscard]] inline double norm(const P3 a) { return std::sqrt(dot(a, a)); }
        [[nodiscard]] inline P3 normalized(const P3 a) {
            const double length = norm(a);
            return length > 0. ? a * (1. / length) : P3{};
        }
        // vcg::Angle
        [[nodiscard]] inline double angle(const P3 a, const P3 b) {
            const double length = norm(a) * norm(b);
            return length > 0. ? std::acos(std::clamp(dot(a, b) / length, -1., 1.)) : 0.;
        }
        // vcg::Quality: twice the area over the longest edge squared, 0.866 for an equilateral triangle.
        [[nodiscard]] inline double triangle_quality(const P3 p0, const P3 p1, const P3 p2) {
            const P3 d10 = p1 - p0, d20 = p2 - p0, d12 = p1 - p2;
            const double twice_area = norm(cross(d10, d20));
            if (!(twice_area > 0.))
                return 0.;
            const double longest = std::max({dot(d10, d10), dot(d20, d20), dot(d12, d12)});
            return longest > 0. ? twice_area / longest : 0.;
        }

        constexpr double kEpsil = 1e-6; // VCG's EPSIL in Clean::TestFaceFaceIntersection
        constexpr double kPi = 3.14159265358979323846;

        // vcg::coplanar_tri_tri
        [[nodiscard]] bool coplanar_triangles(const P3 n, const P3 v0, const P3 v1, const P3 v2, const P3 u0,
                                              const P3 u1, const P3 u2) {
            const double size[3] = {std::abs(n[0]), std::abs(n[1]), std::abs(n[2])};
            int i0 = 0, i1 = 2;
            if (size[0] > size[1]) {
                if (size[0] > size[2]) {
                    i0 = 1;
                    i1 = 2;
                } else {
                    i0 = 0;
                    i1 = 1;
                }
            } else if (size[2] > size[1]) {
                i0 = 0;
                i1 = 1;
            }
            const auto edge_edge = [&](const P3 e0, const P3 e1, const P3 a0, const P3 a1) {
                const double ax = e1[i0] - e0[i0], ay = e1[i1] - e0[i1];
                const double bx = a0[i0] - a1[i0], by = a0[i1] - a1[i1];
                const double cx = e0[i0] - a0[i0], cy = e0[i1] - a0[i1];
                const double f = ay * bx - ax * by;
                const double d = by * cx - bx * cy;
                if ((f > 0. && d >= 0. && d <= f) || (f < 0. && d <= 0. && d >= f)) {
                    const double e = ax * cy - ay * cx;
                    if (f > 0.) {
                        if (e >= 0. && e <= f)
                            return true;
                    } else if (e <= 0. && e >= f) {
                        return true;
                    }
                }
                return false;
            };
            const auto edge_against = [&](const P3 e0, const P3 e1) {
                return edge_edge(e0, e1, u0, u1) || edge_edge(e0, e1, u1, u2) || edge_edge(e0, e1, u2, u0);
            };
            if (edge_against(v0, v1) || edge_against(v1, v2) || edge_against(v2, v0))
                return true;
            const auto inside = [&](const P3 p, const P3 a, const P3 b, const P3 c) {
                double ea = b[i1] - a[i1], eb = -(b[i0] - a[i0]), ec = -ea * a[i0] - eb * a[i1];
                const double d0 = ea * p[i0] + eb * p[i1] + ec;
                ea = c[i1] - b[i1];
                eb = -(c[i0] - b[i0]);
                ec = -ea * b[i0] - eb * b[i1];
                const double d1 = ea * p[i0] + eb * p[i1] + ec;
                ea = a[i1] - c[i1];
                eb = -(a[i0] - c[i0]);
                ec = -ea * c[i0] - eb * c[i1];
                const double d2 = ea * p[i0] + eb * p[i1] + ec;
                return d0 * d1 > 0. && d0 * d2 > 0.;
            };
            return inside(v0, u0, u1, u2) || inside(u0, v0, v1, v2);
        }

        // vcg::NoDivTriTriIsect (Moller). Only the first plane's normal is normalized, as in VCG.
        [[nodiscard]] bool no_div_tri_tri_isect(const P3 v0, const P3 v1, const P3 v2, const P3 u0, const P3 u1,
                                                const P3 u2) {
            const P3 n1 = normalized(cross(v1 - v0, v2 - v0));
            const double d1 = -dot(n1, v0);
            const double du0 = dot(n1, u0) + d1, du1 = dot(n1, u1) + d1, du2 = dot(n1, u2) + d1;
            const double du0du1 = du0 * du1, du0du2 = du0 * du2;
            if (du0du1 > 0. && du0du2 > 0.)
                return false;
            const P3 n2 = cross(u1 - u0, u2 - u0);
            const double d2 = -dot(n2, u0);
            const double dv0 = dot(n2, v0) + d2, dv1 = dot(n2, v1) + d2, dv2 = dot(n2, v2) + d2;
            const double dv0dv1 = dv0 * dv1, dv0dv2 = dv0 * dv2;
            if (dv0dv1 > 0. && dv0dv2 > 0.)
                return false;
            const P3 direction = cross(n1, n2);
            int index = 0;
            double largest = std::abs(direction[0]);
            if (const double value = std::abs(direction[1]); value > largest) {
                largest = value;
                index = 1;
            }
            if (const double value = std::abs(direction[2]); value > largest)
                index = 2;
            const auto intervals = [](const double vv0, const double vv1, const double vv2, const double d0,
                                      const double d1, const double d2, const double d0d1, const double d0d2,
                                      double& a, double& b, double& c, double& x0, double& x1) {
                if (d0d1 > 0.) {
                    a = vv2;
                    b = (vv0 - vv2) * d2;
                    c = (vv1 - vv2) * d2;
                    x0 = d2 - d0;
                    x1 = d2 - d1;
                } else if (d0d2 > 0.) {
                    a = vv1;
                    b = (vv0 - vv1) * d1;
                    c = (vv2 - vv1) * d1;
                    x0 = d1 - d0;
                    x1 = d1 - d2;
                } else if (d1 * d2 > 0. || d0 != 0.) {
                    a = vv0;
                    b = (vv1 - vv0) * d0;
                    c = (vv2 - vv0) * d0;
                    x0 = d0 - d1;
                    x1 = d0 - d2;
                } else if (d1 != 0.) {
                    a = vv1;
                    b = (vv0 - vv1) * d1;
                    c = (vv2 - vv1) * d1;
                    x0 = d1 - d0;
                    x1 = d1 - d2;
                } else if (d2 != 0.) {
                    a = vv2;
                    b = (vv0 - vv2) * d2;
                    c = (vv1 - vv2) * d2;
                    x0 = d2 - d0;
                    x1 = d2 - d1;
                } else {
                    return false; // coplanar: the caller tests the projected triangles
                }
                return true;
            };
            double a, b, c, x0, x1;
            if (!intervals(v0[index], v1[index], v2[index], dv0, dv1, dv2, dv0dv1, dv0dv2, a, b, c, x0, x1))
                return coplanar_triangles(n1, v0, v1, v2, u0, u1, u2);
            double d, e, f, y0, y1;
            if (!intervals(u0[index], u1[index], u2[index], du0, du1, du2, du0du1, du0du2, d, e, f, y0, y1))
                return coplanar_triangles(n1, v0, v1, v2, u0, u1, u2);
            const double xx = x0 * x1, yy = y0 * y1, xxyy = xx * yy;
            double first[2] = {a * xxyy + b * x1 * yy, a * xxyy + c * x0 * yy};
            double second[2] = {d * xxyy + e * xx * y1, d * xxyy + f * xx * y0};
            std::ranges::sort(first);
            std::ranges::sort(second);
            return !(first[1] < second[0] || second[1] < first[0]);
        }

        // vcg::IntersectionLineTriangle (Moller-Trumbore).
        [[nodiscard]] bool line_triangle(const P3 origin, const P3 direction, const P3 vert0, const P3 vert1,
                                         const P3 vert2, double& t, double& u, double& v) {
            const P3 edge1 = vert1 - vert0, edge2 = vert2 - vert0;
            const P3 pvec = cross(direction, edge2);
            const double det = dot(edge1, pvec);
            if (!(std::abs(det) > kEpsil))
                return false; // parallel to the plane
            const double inverse = 1. / det;
            const P3 tvec = origin - vert0;
            const P3 qvec = cross(tvec, edge1);
            if (det > kEpsil) {
                u = dot(tvec, pvec);
                if (u < 0. || u > det)
                    return false;
                v = dot(direction, qvec);
                if (v < 0. || u + v > det)
                    return false;
            } else {
                u = dot(tvec, pvec);
                if (u > 0. || u < det)
                    return false;
                v = dot(direction, qvec);
                if (v > 0. || u + v < det)
                    return false;
            }
            t = dot(edge2, qvec) * inverse;
            u *= inverse;
            v *= inverse;
            return true;
        }

        // vcg::IntersectionSegmentTriangle, without the bounding box early-outs it only uses for speed.
        [[nodiscard]] bool segment_triangle(const P3 from, const P3 to, const P3 v0, const P3 v1, const P3 v2,
                                            double& a, double& b) {
            const P3 delta = to - from;
            const double length = norm(delta);
            if (!(length > 0.))
                return false;
            double t, u, v;
            if (!line_triangle(from, delta * (1. / length), v0, v1, v2, t, u, v))
                return false;
            a = u;
            b = v;
            return t >= 0. && t <= length;
        }

        // The adjacency the ear tests need: how many faces sit on an edge (one means border), the
        // normalized face normals and the angle-weighted vertex normals OpenMVS refreshes before it
        // closes holes. Edges the fill creates go in `added`.
        struct EarTopology {
            std::vector<std::pair<uint64_t, int32_t>> edges;           // undirected key and face, sorted
            std::unordered_map<uint64_t, std::array<int32_t, 2>> added; // edges the fill created
            std::vector<P3> face_normals;
            std::vector<P3> vertex_normals;

            [[nodiscard]] static uint64_t key(const int32_t a, const int32_t b) {
                const auto lo = static_cast<uint32_t>(std::min(a, b)), hi = static_cast<uint32_t>(std::max(a, b));
                return (uint64_t(lo) << 32) | hi;
            }
            [[nodiscard]] std::pair<std::vector<std::pair<uint64_t, int32_t>>::const_iterator,
                                    std::vector<std::pair<uint64_t, int32_t>>::const_iterator>
            range(const int32_t a, const int32_t b) const {
                const auto edge_key = key(a, b);
                const auto first = std::lower_bound(edges.begin(), edges.end(), edge_key,
                                                    [](const auto& edge, const uint64_t value) { return edge.first < value; });
                const auto last = std::upper_bound(first, edges.end(), edge_key,
                                                   [](const uint64_t value, const auto& edge) { return value < edge.first; });
                return {first, last};
            }
            [[nodiscard]] int faces_on(const int32_t a, const int32_t b) const {
                const auto [first, last] = range(a, b);
                int count = static_cast<int>(last - first);
                if (const auto it = added.find(key(a, b)); it != added.end())
                    for (const auto face : it->second)
                        count += face >= 0 ? 1 : 0;
                return count;
            }
            [[nodiscard]] bool exists(const int32_t a, const int32_t b) const { return faces_on(a, b) > 0; }
            [[nodiscard]] bool border(const int32_t a, const int32_t b) const { return faces_on(a, b) == 1; }
            [[nodiscard]] int32_t face_on(const int32_t a, const int32_t b) const {
                if (const auto it = added.find(key(a, b)); it != added.end() && it->second[0] >= 0)
                    return it->second[0];
                const auto [first, last] = range(a, b);
                return first == last ? -1 : first->second;
            }
            void add_edge(const int32_t a, const int32_t b, const int32_t face) {
                auto [it, inserted] = added.try_emplace(key(a, b), std::array<int32_t, 2>{-1, -1});
                if (it->second[0] < 0)
                    it->second[0] = face;
                else if (it->second[1] < 0)
                    it->second[1] = face;
            }
        };

        [[nodiscard]] EarTopology build_ear_topology(const SplatMesh& mesh) {
            EarTopology topo;
            topo.face_normals.resize(mesh.faces.size());
            topo.vertex_normals.assign(mesh.vertices.size(), P3{});
            topo.edges.reserve(mesh.faces.size() * 3);
            for (size_t f = 0; f < mesh.faces.size(); ++f) {
                const auto& face = mesh.faces[f];
                const P3 points[3] = {point_of(mesh.vertices[face[0]]), point_of(mesh.vertices[face[1]]),
                                      point_of(mesh.vertices[face[2]])};
                const P3 normal = normalized(cross(points[1] - points[0], points[2] - points[0]));
                topo.face_normals[f] = normal;
                for (int i = 0; i < 3; ++i) {
                    const auto next = (i + 1) % 3, last = (i + 2) % 3;
                    topo.vertex_normals[face[i]] = topo.vertex_normals[face[i]] +
                                                   normal * angle(points[next] - points[i], points[last] - points[i]);
                    topo.edges.emplace_back(EarTopology::key(face[i], face[next]), static_cast<int32_t>(f));
                }
            }
            for (auto& normal : topo.vertex_normals)
                normal = normalized(normal);
            std::ranges::sort(topo.edges);
            return topo;
        }

        // vcg::face::CountSharedVertex
        [[nodiscard]] int count_shared_vertices(const std::array<int32_t, 3>& first,
                                                const std::array<int32_t, 3>& second) {
            int shared = 0;
            for (const auto vertex : first)
                shared += vertex == second[0] || vertex == second[1] || vertex == second[2] ? 1 : 0;
            return shared;
        }

        // vcg::Clean::TestFaceFaceIntersection: 3 shared vertices coincide, 2 are left to the caller's
        // border test, 0 test the triangles and 1 tests the opposite edge shrunk toward the shared one.
        [[nodiscard]] bool test_face_face_intersection(const std::array<int32_t, 3>& first,
                                                       const std::array<int32_t, 3>& second,
                                                       const std::vector<std::array<float, 3>>& positions) {
            const auto point = [&](const std::array<int32_t, 3>& face, const int i) {
                return point_of(positions[face[i]]);
            };
            const int shared = count_shared_vertices(first, second);
            if (shared == 3)
                return true;
            if (shared == 0)
                return no_div_tri_tri_isect(point(first, 0), point(first, 1), point(first, 2), point(second, 0),
                                            point(second, 1), point(second, 2));
            if (shared != 1)
                return false;
            int first_index = -1, second_index = -1;
            for (int i = 0; i < 3 && first_index < 0; ++i)
                for (int j = 0; j < 3; ++j)
                    if (first[i] == second[j]) {
                        first_index = i;
                        second_index = j;
                        break;
                    }
            const P3 shared_half = point(first, first_index) * 0.5;
            const P3 from = point(first, (first_index + 1) % 3) * 0.5 + shared_half;
            const P3 to = point(first, (first_index + 2) % 3) * 0.5 + shared_half;
            double a, b;
            if (segment_triangle(from, to, point(second, 0), point(second, 1), point(second, 2), a, b))
                return !(a + b >= 1. || a <= kEpsil || b <= kEpsil);
            const P3 other_from = point(second, (second_index + 1) % 3) * 0.5 + shared_half;
            const P3 other_to = point(second, (second_index + 2) % 3) * 0.5 + shared_half;
            if (segment_triangle(other_from, other_to, point(first, 0), point(first, 1), point(first, 2), a, b))
                return !(a + b >= 1. || a <= kEpsil || b <= kEpsil);
            return false;
        }

        struct EarCandidate {
            int32_t node = -1, node_previous = -1, node_next = -1; // positions in the rim
            int32_t corner = -1, previous = -1, next = -1;         // the vertices they hold
            double weight = 0.;
            bool concave = false;
            // vcg::MinimumWeightEar: concave ears sort last, the rest by shape quality less a small
            // dihedral penalty. std::priority_queue pops the largest.
            [[nodiscard]] bool operator<(const EarCandidate& other) const {
                if (concave && !other.concave)
                    return true;
                if (!concave && other.concave)
                    return false;
                return weight < other.weight;
            }
        };

        // vcg::TrivialEar and MinimumWeightEar::ComputeQuality.
        [[nodiscard]] EarCandidate make_ear(const SplatMesh& mesh, const EarTopology& topo,
                                            const std::vector<int32_t>& loop, const int32_t previous_node,
                                            const int32_t node, const int32_t next_node) {
            const int32_t corner = loop[node], previous = loop[previous_node], next = loop[next_node];
            const P3 p0 = point_of(mesh.vertices[corner]);
            const P3 p1 = point_of(mesh.vertices[previous]);
            const P3 p2 = point_of(mesh.vertices[next]);
            const P3 normal = cross(p1 - p0, p2 - p0);
            double corner_angle = angle(p2 - p0, p1 - p0);
            if (dot(normal, topo.vertex_normals[corner]) < 0.)
                corner_angle = 2. * kPi - corner_angle; // a reversed ear, as TrivialEar::ComputeAngle
            double dihedral = 0.;
            if (const int32_t face = topo.face_on(corner, next); face >= 0)
                dihedral = std::max(dihedral, angle(normal, topo.face_normals[face]));
            if (const int32_t face = topo.face_on(previous, corner); face >= 0)
                dihedral = std::max(dihedral, angle(normal, topo.face_normals[face]));
            return {node, previous_node, next_node, corner, previous, next,
                    triangle_quality(p0, p1, p2) - (dihedral / kPi) * 0.1, corner_angle > kPi};
        }

        // vcg::SelfIntersectionEar::Close, then TrivialEar::Close.
        [[nodiscard]] bool close_ear(SplatMesh& mesh, EarTopology& topo, const EarCandidate& ear,
                                     const int32_t hole_size, std::vector<int32_t>& ring) {
            const int32_t corner = ear.corner, previous = ear.previous, next = ear.next;
            const int32_t before = topo.face_on(previous, corner), after = topo.face_on(corner, next);
            if (before >= 0 && before == after) // both ears sides on one face
                return false;
            // The new chord must not already exist; a three edge hole's chord is its third rim edge.
            if (hole_size > 3 && topo.exists(previous, next))
                return false;
            const std::array<int32_t, 3> candidate = {next, corner, previous};
            for (const auto face : ring) {
                const auto& other = mesh.faces[face];
                if (test_face_face_intersection(candidate, other, mesh.vertices))
                    return false;
                if (count_shared_vertices(candidate, other) == 2) {
                    // Only the two rim sides may be shared with a surrounding face, and both of them
                    // are border edges there.
                    int32_t shared[2] = {-1, -1};
                    int count = 0;
                    for (const auto vertex : candidate)
                        if (vertex == other[0] || vertex == other[1] || vertex == other[2])
                            if (count < 2)
                                shared[count++] = vertex;
                    if (count < 2 || !topo.border(shared[0], shared[1]))
                        return false;
                }
            }
            mesh.faces.push_back(candidate);
            topo.face_normals.push_back(normalized(cross(point_of(mesh.vertices[corner]) - point_of(mesh.vertices[next]),
                                                         point_of(mesh.vertices[previous]) - point_of(mesh.vertices[next]))));
            const auto face = static_cast<int32_t>(mesh.faces.size() - 1);
            topo.add_edge(previous, next, face);
            ring.push_back(face); // SelfIntersectionEar::AdjacencyRing grows with every closed ear
            return true;
        }

        // vcg::Hole::FillHoleEar with SelfIntersectionEar. Returns true when it closed any ear.
        bool fill_hole_ear_cut(SplatMesh& mesh, EarTopology& topo, const std::vector<int32_t>& loop,
                               std::vector<int32_t>& ring) {
            const auto count = static_cast<int32_t>(loop.size());
            if (count < 3)
                return false;
            // Vertices the rim visits more than once are non-manifold and get no ear.
            std::vector<char> non_manifold(count, 0);
            for (int32_t i = 0; i < count; ++i)
                for (int32_t j = i + 1; j < count; ++j)
                    if (loop[i] == loop[j])
                        non_manifold[i] = non_manifold[j] = 1;
            std::vector<int32_t> previous(count), next(count);
            for (int32_t i = 0; i < count; ++i) {
                previous[i] = (i + count - 1) % count;
                next[i] = (i + 1) % count;
            }
            std::vector<char> alive(count, 1);
            std::priority_queue<EarCandidate> heap;
            const auto push = [&](const int32_t node) {
                if (!non_manifold[node])
                    heap.push(make_ear(mesh, topo, loop, previous[node], node, next[node]));
            };
            for (int32_t node = 0; node < count; ++node)
                push(node);
            int32_t hole_size = count;
            bool closed_any = false;
            while (hole_size > 2 && !heap.empty()) {
                const EarCandidate ear = heap.top();
                heap.pop();
                if (!alive[ear.node] || previous[ear.node] != ear.node_previous || next[ear.node] != ear.node_next)
                    continue; // the rim moved since this ear was built
                if (non_manifold[ear.node_previous] && non_manifold[ear.node_next])
                    continue; // both ends non-manifold: a degenerate ear
                if (!close_ear(mesh, topo, ear, hole_size, ring))
                    continue;
                closed_any = true;
                const int32_t node = ear.node, before = ear.node_previous, after = ear.node_next;
                alive[node] = 0;
                next[before] = after;
                previous[after] = before;
                --hole_size;
                push(before);
                push(after);
            }
            return closed_any;
        }
    } // namespace

    namespace {
        struct RimLoop {
            std::vector<int32_t> vertices; // rim edges run loop[i] -> loop[i + 1] -> loop[0]
            float perimeter = 0.f;
            float longest_edge = 0.f;
            float diagonal = 0.f; // of the bounding box
            double area = 0.;     // of the polygon the rim spans, by Newell's method
        };

        // A hole spans an area; a seam left by welding two sheets or by deleting the faces around an
        // edge does not. The ratio is scale free, so it does not need the voxel size.
        [[nodiscard]] bool rim_is_sliver(const RimLoop& loop) {
            return loop.diagonal > 0.f && loop.area < kRimMinFillRatio * double(loop.diagonal) * double(loop.diagonal);
        }

        // Walks the edges that have no triangle on their other side into closed boundary loops.
        // A vertex where more than one rim edge starts, or more than one ends, is shared by several
        // boundaries: the walk cannot tell which edge continues which loop, and stitching them
        // yields a rim that jumps across the surface. Those loops are dropped rather than filled.
        [[nodiscard]] std::vector<RimLoop> find_rim_loops(const SplatMesh& mesh) {
            const auto key = [](const int32_t from, const int32_t to) { return (uint64_t(uint32_t(from)) << 32) | uint32_t(to); };
            std::vector<uint64_t> directed;
            directed.reserve(mesh.faces.size() * 3);
            for (const auto& f : mesh.faces)
                for (int i = 0; i < 3; ++i)
                    directed.push_back(key(f[i], f[(i + 1) % 3]));
            std::ranges::sort(directed);
            std::vector<uint64_t> rim;
            for (const auto edge : directed)
                if (!std::ranges::binary_search(directed, (edge << 32) | (edge >> 32)))
                    rim.push_back(edge);
            directed = {};
            const auto rim_start = [&](const int32_t vertex) { return std::ranges::lower_bound(rim, key(vertex, 0)) - rim.begin(); };
            // Vertices with an ambiguous rim, in either direction.
            std::vector<char> junction(mesh.vertices.size(), 0);
            const auto mark_junctions = [&](const std::vector<uint64_t>& edges) {
                for (size_t i = 0; i < edges.size();) {
                    size_t j = i + 1;
                    while (j < edges.size() && (edges[j] >> 32) == (edges[i] >> 32))
                        ++j;
                    if (j - i > 1)
                        junction[static_cast<size_t>(edges[i] >> 32)] = 1;
                    i = j;
                }
            };
            mark_junctions(rim);
            std::vector<uint64_t> incoming;
            incoming.reserve(rim.size());
            for (const auto edge : rim)
                incoming.push_back((edge << 32) | (edge >> 32));
            std::ranges::sort(incoming);
            mark_junctions(incoming);
            incoming = {};
            const auto edge_length = [&](const int32_t a, const int32_t b) {
                const auto& p = mesh.vertices[a];
                const auto& q = mesh.vertices[b];
                return std::hypot(p[0] - q[0], p[1] - q[1], p[2] - q[2]);
            };
            std::vector<char> used(rim.size(), 0);
            std::vector<RimLoop> loops;
            RimLoop loop;
            for (size_t first = 0; first < rim.size(); ++first) {
                if (used[first])
                    continue;
                used[first] = 1;
                const auto start = static_cast<int32_t>(rim[first] >> 32);
                auto current = static_cast<int32_t>(rim[first] & 0xffffffffu);
                loop.vertices.assign({start});
                loop.perimeter = edge_length(start, current);
                loop.longest_edge = loop.perimeter;
                loop.diagonal = 0.f;
                loop.area = 0.;
                bool closed = current == start;
                bool broken = junction[static_cast<size_t>(start)] != 0;
                while (!closed && !broken) {
                    if (junction[static_cast<size_t>(current)]) {
                        broken = true;
                        break;
                    }
                    loop.vertices.push_back(current);
                    size_t next = rim_start(current);
                    while (next < rim.size() && static_cast<int32_t>(rim[next] >> 32) == current && used[next])
                        ++next;
                    if (next == rim.size() || static_cast<int32_t>(rim[next] >> 32) != current) {
                        broken = true;
                        break;
                    }
                    used[next] = 1;
                    const auto following = static_cast<int32_t>(rim[next] & 0xffffffffu);
                    const auto length = edge_length(current, following);
                    loop.perimeter += length;
                    loop.longest_edge = std::max(loop.longest_edge, static_cast<float>(length));
                    current = following;
                    closed = current == start;
                }
                if (broken || !closed || loop.vertices.size() < 3)
                    continue;
                // The rim is closed, so its span is a fair measure of the hole it leaves.
                std::array<float, 3> low = mesh.vertices[loop.vertices.front()];
                std::array<float, 3> high = low;
                double nx = 0., ny = 0., nz = 0.;
                for (size_t i = 0; i < loop.vertices.size(); ++i) {
                    const auto& p = mesh.vertices[loop.vertices[i]];
                    const auto& q = mesh.vertices[loop.vertices[(i + 1) % loop.vertices.size()]];
                    for (int axis = 0; axis < 3; ++axis) {
                        low[axis] = std::min(low[axis], p[axis]);
                        high[axis] = std::max(high[axis], p[axis]);
                    }
                    nx += double(p[1]) * double(q[2]) - double(p[2]) * double(q[1]);
                    ny += double(p[2]) * double(q[0]) - double(p[0]) * double(q[2]);
                    nz += double(p[0]) * double(q[1]) - double(p[1]) * double(q[0]);
                }
                loop.diagonal = static_cast<float>(std::hypot(high[0] - low[0], high[1] - low[1], high[2] - low[2]));
                loop.area = .5 * std::sqrt(nx * nx + ny * ny + nz * nz);
                loops.push_back(loop);
            }
            return loops;
        }

        // OpenMVS closes a hole with ear clipping when its boundary has fewer edges than
        // nCloseHoles, so the cap is an edge count. Every rejection test is VCG's. Loops with an
        // edge longer than max_edge are not offered to the ear any more: they are not sampling
        // gaps, and clipping them would span the surface.
        size_t fill_rim_loops_by_ear(SplatMesh& mesh, const std::vector<RimLoop>& loops, const int ear_cut_edges,
                                     const float max_edge, size_t& refused) {
            std::vector<const RimLoop*> candidates;
            for (const auto& loop : loops)
                if (static_cast<int>(loop.vertices.size()) < ear_cut_edges) {
                    if ((max_edge > 0.f && loop.longest_edge > max_edge) || rim_is_sliver(loop))
                        ++refused;
                    else
                        candidates.push_back(&loop);
                }
            if (candidates.empty())
                return 0;
            auto topo = build_ear_topology(mesh);
            // AdjacencyRing: every face around the boundary, i.e. the union of the faces incident
            // to the rim vertices. It grows with the ears the fill creates, as it does in VCG.
            std::unordered_map<int32_t, std::vector<int32_t>> rim_faces;
            for (size_t f = 0; f < mesh.faces.size(); ++f)
                for (const auto vertex : mesh.faces[f])
                    rim_faces[vertex].push_back(static_cast<int32_t>(f));
            size_t filled = 0;
            for (const auto* loop : candidates) {
                std::vector<int32_t> ring;
                for (const auto vertex : loop->vertices) {
                    const auto it = rim_faces.find(vertex);
                    if (it != rim_faces.end())
                        ring.insert(ring.end(), it->second.begin(), it->second.end());
                }
                std::ranges::sort(ring);
                ring.erase(std::unique(ring.begin(), ring.end()), ring.end());
                const auto before = mesh.faces.size();
                filled += fill_hole_ear_cut(mesh, topo, loop->vertices, ring) ? 1 : 0;
                for (auto f = before; f < mesh.faces.size(); ++f)
                    for (const auto vertex : mesh.faces[f])
                        rim_faces[vertex].push_back(static_cast<int32_t>(f));
            }
            return filled;
        }

        size_t fill_rim_loops_by_fan(SplatMesh& mesh, const std::vector<RimLoop>& loops, const float max_perimeter,
                                     const float max_edge, size_t& refused) {
            size_t filled = 0;
            for (const auto& loop : loops)
                if (loop.perimeter <= max_perimeter) {
                    // A rim with one long edge, or a rim that spans no area, is the same sliver risk
                    // the ear pass guards against.
                    if ((max_edge > 0.f && loop.longest_edge > max_edge) || rim_is_sliver(loop)) {
                        ++refused;
                        continue;
                    }
                    fill_rim_by_fan(mesh, loop.vertices);
                    ++filled;
                }
            return filled;
        }
    } // namespace

    size_t fill_small_mesh_holes(SplatMesh& mesh, const float max_perimeter, const int ear_cut_edges,
                                 const float max_edge) {
        if (max_perimeter <= 0.f && ear_cut_edges <= 0)
            return 0;
        size_t filled = 0;
        size_t refused = 0;
        // Ear first: it keeps the rim and the concave shape. A hole it cannot close stays open,
        // and the fan pass re-reads the boundary so it never fills a rim the ear faces already
        // consumed; a hole the ear closed only in part leaves a smaller rim for that pass.
        if (ear_cut_edges > 0)
            filled += fill_rim_loops_by_ear(mesh, find_rim_loops(mesh), ear_cut_edges, max_edge, refused);
        if (max_perimeter > 0.f)
            filled += fill_rim_loops_by_fan(mesh, find_rim_loops(mesh), max_perimeter, max_edge, refused);
        if (refused > 0)
            LOG_INFO("Mesh: left {} rims open, a long edge (> {:.3f}) or no span means topology damage rather than a gap",
                     refused, max_edge);
        return filled;
    }

    namespace {
        // Drops the vertices the faces no longer use and renumbers the faces to match.
        void compact_mesh(SplatMesh& mesh) {
            const size_t count = mesh.vertices.size();
            const bool has_normals = mesh.normals.size() == count;
            const bool has_colors = mesh.colors.size() == count;
            std::vector<int32_t> remap(count, -1);
            std::vector<std::array<float, 3>> vertices;
            std::vector<std::array<float, 3>> normals;
            std::vector<std::array<uint8_t, 3>> colors;
            vertices.reserve(count);
            normals.reserve(has_normals ? count : 0);
            colors.reserve(has_colors ? count : 0);
            for (auto& face : mesh.faces)
                for (auto& vertex : face) {
                    auto& target = remap[vertex];
                    if (target < 0) {
                        target = static_cast<int32_t>(vertices.size());
                        vertices.push_back(mesh.vertices[vertex]);
                        if (has_normals)
                            normals.push_back(mesh.normals[vertex]);
                        if (has_colors)
                            colors.push_back(mesh.colors[vertex]);
                    }
                    vertex = target;
                }
            mesh.vertices = std::move(vertices);
            mesh.normals = std::move(normals);
            mesh.colors = std::move(colors);
        }

        // Diagonal of the bounding box, in scene units; 0 for an empty mesh.
        [[nodiscard]] double mesh_extent(const SplatMesh& mesh) {
            if (mesh.vertices.empty())
                return 0.;
            std::array<float, 3> lo = mesh.vertices.front(), hi = mesh.vertices.front();
            for (const auto& p : mesh.vertices)
                for (int i = 0; i < 3; ++i) {
                    lo[i] = std::min(lo[i], p[i]);
                    hi[i] = std::max(hi[i], p[i]);
                }
            const double d[3] = {double(hi[0]) - lo[0], double(hi[1]) - lo[1], double(hi[2]) - lo[2]};
            return std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
        }

        // Every corner of every face as an undirected edge key, sorted, paired with its face.
        [[nodiscard]] std::vector<std::pair<uint64_t, int32_t>> sorted_edges(const SplatMesh& mesh) {
            std::vector<std::pair<uint64_t, int32_t>> edges;
            edges.reserve(mesh.faces.size() * 3);
            for (int32_t f = 0; f < static_cast<int32_t>(mesh.faces.size()); ++f)
                for (int i = 0; i < 3; ++i) {
                    int32_t a = mesh.faces[f][i], b = mesh.faces[f][(i + 1) % 3];
                    if (a > b)
                        std::swap(a, b);
                    edges.emplace_back((uint64_t(uint32_t(a)) << 32) | uint32_t(b), f);
                }
            std::ranges::sort(edges);
            return edges;
        }

    } // namespace

    namespace {
        // The sort-based passes below need roughly 16 bytes per edge, which for a 200M face mesh is
        // more than the mesh itself and takes minutes. Meshes above this budget are cleaned after
        // simplification, when they are orders of magnitude smaller.
        constexpr size_t kCleanupFaceLimit = 20'000'000;

        // Degeneracy is relative to the mesh so that one threshold serves every scene scale. The
        // cross product is twice the triangle area, which makes 1e-9 of the extent squared about a
        // tenth of a millimetre on a ten metre scene: noise, not the smallest surface nets triangle.
        size_t drop_degenerate_faces(SplatMesh& mesh) {
            const double extent = mesh_extent(mesh);
            const double epsilon = 1e-9 * extent * extent;
            return std::erase_if(mesh.faces, [&](const std::array<int32_t, 3>& face) {
                if (face[0] == face[1] || face[1] == face[2] || face[0] == face[2])
                    return true;
                const auto& a = mesh.vertices[face[0]];
                const auto& b = mesh.vertices[face[1]];
                const auto& c = mesh.vertices[face[2]];
                const double u[3] = {double(b[0]) - a[0], double(b[1]) - a[1], double(b[2]) - a[2]};
                const double v[3] = {double(c[0]) - a[0], double(c[1]) - a[1], double(c[2]) - a[2]};
                const double n[3] = {u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]};
                return std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]) <= epsilon;
            });
        }
    } // namespace

    size_t clean_mesh_topology(SplatMesh& mesh) {
        if (mesh.faces.empty())
            return 0;
        size_t removed = drop_degenerate_faces(mesh);
        LOG_INFO("Mesh: dropped {} zero-area faces", removed);
        if (mesh.faces.empty())
            return removed;
        if (mesh.faces.size() > kCleanupFaceLimit) {
            LOG_INFO("Mesh: {} faces is above the cleanup budget, only degenerate faces were dropped",
                     mesh.faces.size());
            return removed;
        }

        // Faces with the same three corners, whichever order they list them in.
        if (!mesh.faces.empty()) {
            std::vector<std::array<int32_t, 3>> order(mesh.faces.size());
            for (size_t f = 0; f < mesh.faces.size(); ++f) {
                order[f] = mesh.faces[f];
                std::ranges::sort(order[f]);
            }
            std::vector<uint32_t> sorted(mesh.faces.size());
            std::iota(sorted.begin(), sorted.end(), 0u);
            std::ranges::sort(sorted, [&](const uint32_t a, const uint32_t b) { return order[a] < order[b]; });
            std::vector<char> keep(mesh.faces.size(), 1);
            for (size_t i = 1; i < sorted.size(); ++i)
                if (order[sorted[i]] == order[sorted[i - 1]])
                    keep[sorted[i]] = 0;
            std::vector<std::array<int32_t, 3>> faces;
            faces.reserve(mesh.faces.size());
            for (size_t f = 0; f < mesh.faces.size(); ++f)
                if (keep[f])
                    faces.push_back(mesh.faces[f]);
            removed += mesh.faces.size() - faces.size();
            LOG_INFO("Mesh: dropped {} repeated faces", mesh.faces.size() - faces.size());
            mesh.faces = std::move(faces);
        }

        // Vertices at the same position, merged onto the one the faces meet first. Merging can
        // collapse a face onto itself and welds pieces that only touched, which the passes below
        // then separate again.
        const double extent = mesh_extent(mesh);
        if (mesh.vertices.size() > 1 && extent > 0.) {
            const size_t before = mesh.vertices.size();
            const double quantum = std::max(1e-6 * extent, 1e-9);
            std::vector<std::array<int64_t, 3>> cell(before);
            for (size_t v = 0; v < before; ++v)
                for (int i = 0; i < 3; ++i)
                    cell[v][i] = static_cast<int64_t>(std::llround(double(mesh.vertices[v][i]) / quantum));
            std::vector<uint32_t> sorted(before);
            std::iota(sorted.begin(), sorted.end(), 0u);
            std::ranges::sort(sorted, [&](const uint32_t a, const uint32_t b) { return cell[a] < cell[b]; });
            const bool has_normals = mesh.normals.size() == before;
            const bool has_colors = mesh.colors.size() == before;
            std::vector<int32_t> representative;
            representative.reserve(before);
            std::vector<int32_t> remap(before, 0);
            for (size_t i = 0; i < before; ++i) {
                const auto v = sorted[i];
                if (i > 0 && cell[v] == cell[sorted[i - 1]]) {
                    remap[v] = remap[sorted[i - 1]];
                    continue;
                }
                remap[v] = static_cast<int32_t>(representative.size());
                representative.push_back(static_cast<int32_t>(v));
            }
            if (representative.size() < before) {
                const size_t folded_before = removed;
                // Replace only the geometry arrays: the report fields on the mesh (how many vertices
                // were split, how many holes the fills closed) belong to the whole extraction and
                // must survive this pass, which runs twice.
                std::vector<std::array<float, 3>> vertices;
                std::vector<std::array<float, 3>> normals;
                std::vector<std::array<uint8_t, 3>> colors;
                std::vector<std::array<int32_t, 3>> faces;
                vertices.reserve(representative.size());
                faces.reserve(mesh.faces.size());
                if (has_normals)
                    normals.reserve(representative.size());
                if (has_colors)
                    colors.reserve(representative.size());
                for (const auto source : representative) {
                    vertices.push_back(mesh.vertices[source]);
                    if (has_normals)
                        normals.push_back(mesh.normals[source]);
                    if (has_colors)
                        colors.push_back(mesh.colors[source]);
                }
                for (const auto& face : mesh.faces) {
                    std::array<int32_t, 3> mapped{};
                    for (int i = 0; i < 3; ++i)
                        mapped[i] = remap[face[i]];
                    if (mapped[0] == mapped[1] || mapped[1] == mapped[2] || mapped[0] == mapped[2])
                        ++removed;
                    else
                        faces.push_back(mapped);
                }
                mesh.vertices = std::move(vertices);
                mesh.normals = std::move(normals);
                mesh.colors = std::move(colors);
                mesh.faces = std::move(faces);
                LOG_INFO("Mesh: merged {} coincident vertices at {:.1e}, which folded {} faces",
                         before - representative.size(), quantum, removed - folded_before);
            }
        }

        // Faces that share an edge with more than one other face. Each edge keeps its first two, and
        // dropping the rest cannot give another edge a third face, so one pass is enough.
        if (mesh.faces.size() > 1) {
            const auto edges = sorted_edges(mesh);
            std::vector<char> drop(mesh.faces.size(), 0);
            for (size_t i = 0; i < edges.size();) {
                size_t j = i;
                while (j < edges.size() && edges[j].first == edges[i].first)
                    ++j;
                size_t kept = 0;
                for (size_t k = i; k < j; ++k) {
                    const auto face = edges[k].second;
                    if (drop[face])
                        continue;
                    if (++kept > 2)
                        drop[face] = 1;
                }
                i = j;
            }
            std::vector<std::array<int32_t, 3>> faces;
            faces.reserve(mesh.faces.size());
            for (size_t f = 0; f < mesh.faces.size(); ++f)
                if (!drop[f])
                    faces.push_back(mesh.faces[f]);
            removed += mesh.faces.size() - faces.size();
            LOG_INFO("Mesh: dropped {} faces on edges shared by more than two faces",
                     mesh.faces.size() - faces.size());
            mesh.faces = std::move(faces);
        }

        // Vertices whose faces hang together as more than one fan: each fan beyond the first gets its
        // own copy of the vertex. VCG runs this iteratively up to 10 times because splitting one
        // vertex can expose a fan that was previously split across two non-manifold vertices; five
        // rounds settle every mesh seen so far, and each further round only costs time.
        size_t split = 0;
        constexpr int kSplitIterations = 5;
        for (int iteration = 0; iteration < kSplitIterations; ++iteration) {
            const size_t vertex_count = mesh.vertices.size();
            const bool has_normals = mesh.normals.size() == vertex_count;
            const bool has_colors = mesh.colors.size() == vertex_count;
            std::vector<std::vector<int32_t>> incident(vertex_count);
            for (int32_t f = 0; f < static_cast<int32_t>(mesh.faces.size()); ++f)
                for (int i = 0; i < 3; ++i)
                    incident[mesh.faces[f][i]].push_back(f);
            std::vector<std::pair<int32_t, int32_t>> around;
            std::vector<int32_t> parent;
            std::vector<int32_t> group;
            std::vector<int32_t> copy;
            size_t pass_split = 0;
            for (size_t v = 0; v < vertex_count; ++v) {
                const auto& faces = incident[v];
                if (faces.size() < 2)
                    continue;
                around.clear();
                for (size_t slot = 0; slot < faces.size(); ++slot) {
                    const auto& face = mesh.faces[faces[slot]];
                    for (int i = 0; i < 3; ++i)
                        if (face[i] != static_cast<int32_t>(v))
                            around.emplace_back(face[i], static_cast<int32_t>(slot));
                }
                std::ranges::sort(around);
                parent.resize(faces.size());
                std::iota(parent.begin(), parent.end(), 0);
                const auto root = [&](int32_t x) {
                    while (parent[x] != x)
                        x = parent[x] = parent[parent[x]];
                    return x;
                };
                for (size_t i = 1; i < around.size(); ++i) {
                    if (around[i].first != around[i - 1].first)
                        continue;
                    const int32_t a = root(around[i].second), b = root(around[i - 1].second);
                    if (a != b)
                        parent[a] = b;
                }
                group.resize(faces.size());
                bool fan_split = false;
                for (size_t slot = 0; slot < faces.size(); ++slot) {
                    group[slot] = root(static_cast<int32_t>(slot));
                    fan_split |= group[slot] != group[0];
                }
                if (!fan_split)
                    continue;
                copy.assign(faces.size(), -1);
                for (size_t slot = 0; slot < faces.size(); ++slot) {
                    if (group[slot] == group[0])
                        continue;
                    int32_t target = -1;
                    for (size_t other = 0; other < slot; ++other)
                        if (group[other] == group[slot] && copy[other] >= 0) {
                            target = copy[other];
                            break;
                        }
                    if (target < 0) {
                        target = static_cast<int32_t>(mesh.vertices.size());
                        mesh.vertices.push_back(mesh.vertices[v]);
                        if (has_normals)
                            mesh.normals.push_back(mesh.normals[v]);
                        if (has_colors)
                            mesh.colors.push_back(mesh.colors[v]);
                        ++pass_split;
                    }
                    copy[slot] = target;
                    for (auto& corner : mesh.faces[faces[slot]])
                        if (corner == static_cast<int32_t>(v)) {
                            corner = target;
                            break;
                        }
                }
            }
            split += pass_split;
            if (pass_split == 0)
                break;
        }
        if (split > 0) {
            mesh.vertices_split_non_manifold += split;
            LOG_INFO("Mesh: split {} vertices that joined more than one fan of faces", split);
        }
        compact_mesh(mesh);
        return removed;
    }

    // OpenMVS runs this after hole filling (and smoothing, which we dropped) to strip any non-manifold
    // edges or vertices the earlier passes may have introduced. Unlike clean_mesh_topology it does
    // not merge coincident vertices, split non-manifold ones or dedupe faces — those are the
    // operations that reopened the freshly filled holes.
    size_t clean_non_manifold_only(SplatMesh& mesh) {
        if (mesh.faces.empty())
            return 0;
        size_t removed = 0;
        // Non-manifold edges are the ones shared by more than two faces; drop every face after the
        // first two on such an edge. Each face dropped is counted up to three times, but that is fine
        // because a face that was on three non-manifold edges still only gets dropped once.
        const auto edges = sorted_edges(mesh);
        std::vector<char> drop(mesh.faces.size(), 0);
        for (size_t i = 0; i < edges.size();) {
            size_t j = i;
            while (j < edges.size() && edges[j].first == edges[i].first)
                ++j;
            size_t kept = 0;
            for (size_t k = i; k < j; ++k) {
                const auto face = edges[k].second;
                if (drop[face])
                    continue;
                if (++kept > 2)
                    drop[face] = 1;
            }
            i = j;
        }
        std::vector<std::array<int32_t, 3>> faces;
        faces.reserve(mesh.faces.size());
        for (size_t f = 0; f < mesh.faces.size(); ++f)
            if (!drop[f])
                faces.push_back(mesh.faces[f]);
        removed += mesh.faces.size() - faces.size();
        mesh.faces = std::move(faces);
        // RemoveNonManifoldVertex falls out of compact_mesh: a vertex whose incident faces were
        // stripped down to zero is dropped, along with the per-vertex normals and colors.
        compact_mesh(mesh);
        return removed;
    }

    size_t remove_mesh_spikes(SplatMesh& mesh) {
        size_t removed = 0;
        std::vector<uint32_t> incident;
        std::vector<char> dead;
        while (mesh.faces.size() > 1) {
            incident.assign(mesh.vertices.size(), 0);
            for (const auto& face : mesh.faces)
                for (const auto vertex : face)
                    ++incident[vertex];
            dead.assign(mesh.vertices.size(), 0);
            bool any = false;
            for (size_t v = 0; v < incident.size(); ++v)
                if (incident[v] == 1) {
                    dead[v] = 1;
                    any = true;
                }
            if (!any)
                break;
            const size_t before = mesh.faces.size();
            std::erase_if(mesh.faces, [&](const std::array<int32_t, 3>& face) {
                return dead[face[0]] || dead[face[1]] || dead[face[2]];
            });
            if (mesh.faces.size() == before)
                break;
            removed += before - mesh.faces.size();
            compact_mesh(mesh);
        }
        return removed;
    }

    // OpenMVS Mesh::Clean 的 "remove spurious" pass, split into two independent knobs:
    //   edge_factor      - drop faces with any edge longer than the 95th-percentile edge x factor
    //   component_factor - drop connected components whose bounding-box diagonal is shorter than
    //                      the 55th-percentile edge x factor
    // Each phase is off when its factor is 0, so they can be tuned (and ablated) separately.
    size_t remove_spurious_components(SplatMesh& mesh, const float edge_factor, const float component_factor) {
        if ((edge_factor <= 0.f && component_factor <= 0.f) || mesh.faces.empty())
            return 0;
        // Gather every undirected edge's squared length; we need it twice but it is cheap.
        std::vector<float> squared;
        squared.reserve(mesh.faces.size() * 3);
        for (const auto& face : mesh.faces)
            for (int i = 0; i < 3; ++i) {
                const auto& a = mesh.vertices[face[i]];
                const auto& b = mesh.vertices[face[(i + 1) % 3]];
                const float dx = a[0] - b[0], dy = a[1] - b[1], dz = a[2] - b[2];
                squared.push_back(dx * dx + dy * dy + dz * dz);
            }
        if (squared.empty())
            return 0;
        const auto nth = [&](size_t pct) {
            const size_t idx = std::min(squared.size() - 1, (pct * squared.size()) / 100);
            std::nth_element(squared.begin(), squared.begin() + static_cast<std::ptrdiff_t>(idx), squared.end());
            return squared[idx];
        };
        const float sq95 = nth(95);
        const float sq55 = nth(55);
        const float long_edge = edge_factor > 0.f ? std::sqrt(std::max(sq95, 0.f)) * edge_factor : 0.f;
        const float long_diameter = component_factor > 0.f ? std::sqrt(std::max(sq55, 0.f)) * component_factor : 0.f;

        // Phase 1: drop faces whose edges exceed the long-edge threshold on any of the three.
        size_t removed = 0;
        if (long_edge > 0.f) {
            const float long_edge_squared = long_edge * long_edge;
            std::vector<std::array<int32_t, 3>> kept;
            kept.reserve(mesh.faces.size());
            for (const auto& face : mesh.faces) {
                bool floater = false;
                for (int i = 0; i < 3 && !floater; ++i) {
                    const auto& a = mesh.vertices[face[i]];
                    const auto& b = mesh.vertices[face[(i + 1) % 3]];
                    const float dx = a[0] - b[0], dy = a[1] - b[1], dz = a[2] - b[2];
                    if (dx * dx + dy * dy + dz * dz > long_edge_squared)
                        floater = true;
                }
                if (!floater)
                    kept.push_back(face);
                else
                    ++removed;
            }
            mesh.faces = std::move(kept);
            compact_mesh(mesh);
        }
        if (mesh.faces.empty() || long_diameter <= 0.f)
            return removed;

        // Phase 2: drop connected components whose axis-aligned bounding-box diagonal is below the
        // diameter threshold. OpenMVS does not cap the component's face count here.
        {
            const size_t face_count = mesh.faces.size();
            std::vector<std::vector<int32_t>> incident(mesh.vertices.size());
            for (size_t f = 0; f < face_count; ++f)
                for (const auto v : mesh.faces[f])
                    incident[v].push_back(static_cast<int32_t>(f));
            std::vector<char> drop(face_count, 0);
            std::vector<int32_t> visited(face_count, -1);
            std::vector<int32_t> queue;
            std::vector<int32_t> members;
            int32_t label = 0;
            for (size_t start = 0; start < face_count; ++start) {
                if (visited[start] >= 0)
                    continue;
                queue.clear();
                members.clear();
                queue.push_back(static_cast<int32_t>(start));
                visited[start] = label;
                std::array<float, 3> lo = mesh.vertices[mesh.faces[start][0]];
                std::array<float, 3> hi = lo;
                while (!queue.empty()) {
                    const auto face = queue.back();
                    queue.pop_back();
                    members.push_back(face);
                    for (const auto v : mesh.faces[face]) {
                        const auto& p = mesh.vertices[v];
                        for (int i = 0; i < 3; ++i) {
                            lo[i] = std::min(lo[i], p[i]);
                            hi[i] = std::max(hi[i], p[i]);
                        }
                        for (const auto next : incident[v]) {
                            if (visited[next] < 0) {
                                visited[next] = label;
                                queue.push_back(next);
                            }
                        }
                    }
                }
                const float diag = std::hypot(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]);
                // VCG's RemoveSmallConnectedComponentsDiameter drops components whose bounding-box
                // diagonal is below the threshold, so this keeps everything at or above it.
                if (diag < long_diameter)
                    for (const auto face : members)
                        drop[face] = 1;
                ++label;
            }
            std::vector<std::array<int32_t, 3>> kept;
            kept.reserve(mesh.faces.size());
            for (size_t f = 0; f < face_count; ++f)
                if (!drop[f])
                    kept.push_back(mesh.faces[f]);
                else
                    ++removed;
            mesh.faces = std::move(kept);
        }
        compact_mesh(mesh);
        return removed;
    }

    // Drops connected components whose face count is below max_faces and whose every vertex sits
    // further than neighbor_radius from any vertex of a different component. Small components that
    // stay close to the main mesh are left alone — they are almost always real patches the TSDF
    // fusion just did not bridge. Only fragments that are both small AND isolated get removed.
    size_t remove_isolated_small_mesh_components(SplatMesh& mesh, const int max_faces, const float neighbor_radius) {
        if (max_faces <= 0 || neighbor_radius <= 0.f || mesh.faces.empty())
            return 0;

        const size_t face_count = mesh.faces.size();
        const size_t vertex_count = mesh.vertices.size();

        // --- Phase 1: BFS label every face with its connected component ---
        std::vector<std::vector<int32_t>> incident(vertex_count);
        for (size_t f = 0; f < face_count; ++f)
            for (const auto v : mesh.faces[f])
                incident[v].push_back(static_cast<int32_t>(f));

        std::vector<int32_t> face_label(face_count, -1);
        std::vector<std::vector<uint32_t>> comp_faces;
        std::vector<std::vector<uint32_t>> comp_vertices;

        {
            std::vector<int32_t> queue;
            for (size_t start = 0; start < face_count; ++start) {
                if (face_label[start] >= 0)
                    continue;
                const int32_t label = static_cast<int32_t>(comp_faces.size());
                queue.clear();
                comp_faces.push_back({});
                comp_vertices.push_back({});
                queue.push_back(static_cast<int32_t>(start));
                face_label[start] = label;
                while (!queue.empty()) {
                    const auto face = queue.back();
                    queue.pop_back();
                    comp_faces.back().push_back(static_cast<uint32_t>(face));
                    for (const auto v : mesh.faces[face]) {
                        comp_vertices.back().push_back(static_cast<uint32_t>(v));
                        for (const auto next : incident[v]) {
                            if (face_label[next] < 0) {
                                face_label[next] = label;
                                queue.push_back(next);
                            }
                        }
                    }
                }
                std::ranges::sort(comp_vertices.back());
                comp_vertices.back().erase(std::ranges::unique(comp_vertices.back()).begin(), comp_vertices.back().end());
            }
        }

        const int32_t num_comps = static_cast<int32_t>(comp_faces.size());
        if (num_comps <= 1)
            return 0;

        // --- Phase 2: spatial hash, one bucket per neighbor_radius-wide cell ---
        auto cell_index = [&](const float coord) { return static_cast<int64_t>(std::floor(coord / neighbor_radius)); };
        auto cell_key = [](const int64_t cx, const int64_t cy, const int64_t cz) -> uint64_t {
            // Morton-style hash with large primes — well-distributed, no collisions to worry about.
            return static_cast<uint64_t>(cx) * 73856093ULL
                 ^ static_cast<uint64_t>(cy) * 19349663ULL
                 ^ static_cast<uint64_t>(cz) * 83492791ULL;
        };

        // bucket_key -> list of component labels that have a vertex in that bucket
        std::unordered_map<uint64_t, std::vector<int32_t>> cell_comps;
        cell_comps.reserve(comp_vertices.size());
        for (int32_t comp = 0; comp < num_comps; ++comp) {
            std::vector<uint64_t> seen_cells;
            seen_cells.reserve(comp_vertices[comp].size());
            for (const auto v : comp_vertices[comp]) {
                const auto& p = mesh.vertices[v];
                seen_cells.push_back(cell_key(cell_index(p[0]), cell_index(p[1]), cell_index(p[2])));
            }
            std::ranges::sort(seen_cells);
            seen_cells.erase(std::ranges::unique(seen_cells).begin(), seen_cells.end());
            for (const auto key : seen_cells)
                cell_comps[key].push_back(comp);
        }

        // --- Phase 3: a small component is isolated iff no vertex has a different-component neighbor ---
        std::vector<char> drop_comp(num_comps, 0);
        for (int32_t comp = 0; comp < num_comps; ++comp) {
            if (comp_faces[comp].size() >= static_cast<size_t>(max_faces))
                continue;
            bool found_neighbor = false;
            for (const auto v : comp_vertices[comp]) {
                if (found_neighbor)
                    break;
                const auto& p = mesh.vertices[v];
                const auto cx = cell_index(p[0]);
                const auto cy = cell_index(p[1]);
                const auto cz = cell_index(p[2]);
                for (int64_t dx = -1; dx <= 1 && !found_neighbor; ++dx)
                    for (int64_t dy = -1; dy <= 1 && !found_neighbor; ++dy)
                        for (int64_t dz = -1; dz <= 1 && !found_neighbor; ++dz) {
                            const auto it = cell_comps.find(cell_key(cx + dx, cy + dy, cz + dz));
                            if (it == cell_comps.end())
                                continue;
                            for (const auto other : it->second)
                                if (other != comp) { found_neighbor = true; break; }
                        }
            }
            if (!found_neighbor)
                drop_comp[comp] = 1;
        }

        // --- Phase 4: delete faces of dropped components ---
        size_t removed = 0;
        std::vector<std::array<int32_t, 3>> kept;
        kept.reserve(mesh.faces.size());
        for (size_t f = 0; f < face_count; ++f) {
            if (drop_comp[face_label[f]])
                ++removed;
            else
                kept.push_back(mesh.faces[f]);
        }
        mesh.faces = std::move(kept);
        if (removed > 0) {
            compact_mesh(mesh);
            const int dropped = static_cast<int>(std::ranges::count(drop_comp, 1));
            LOG_INFO("Mesh: dropped {} faces from {} small isolated components (< {} faces, isolated > {:.3f})",
                     removed, dropped, max_faces, neighbor_radius);
        }
        return removed;
    }

    void compute_splat_mesh_normals(SplatMesh& mesh) {
        std::vector<std::array<double, 3>> sum(mesh.vertices.size(), {0., 0., 0.});
        double area = 0.;
        for (const auto& f : mesh.faces) {
            const auto& a = mesh.vertices[f[0]];
            const auto& b = mesh.vertices[f[1]];
            const auto& c = mesh.vertices[f[2]];
            const double u[3] = {double(b[0]) - a[0], double(b[1]) - a[1], double(b[2]) - a[2]};
            const double v[3] = {double(c[0]) - a[0], double(c[1]) - a[1], double(c[2]) - a[2]};
            const double n[3] = {u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]};
            area += 0.5 * std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
            for (const auto vertex : f)
                for (int i = 0; i < 3; ++i)
                    sum[vertex][i] += n[i];
        }
        mesh.area = area;
        mesh.normals.resize(mesh.vertices.size());
        for (size_t i = 0; i < sum.size(); ++i) {
            const auto& n = sum[i];
            const double length = std::max(std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]), 1e-30);
            mesh.normals[i] = {float(n[0] / length), float(n[1] / length), float(n[2] / length)};
        }
    }

    namespace {
        // Orders mesh.faces so that each returned range of up to max_faces and max_extent is a compact piece of space.
        void split_faces(SplatMesh& mesh, const size_t first, const size_t last, const size_t max_faces, const float max_extent, const int threads_left,
                         std::vector<std::pair<size_t, size_t>>& ranges, std::mutex& ranges_mutex) {
            std::array<float, 3> low{FLT_MAX, FLT_MAX, FLT_MAX}, high{-FLT_MAX, -FLT_MAX, -FLT_MAX};
            for (size_t f = first; f < last; ++f) {
                const auto& p = mesh.vertices[mesh.faces[f][0]];
                for (int i = 0; i < 3; ++i) {
                    low[i] = std::min(low[i], p[i]);
                    high[i] = std::max(high[i], p[i]);
                }
            }
            int axis = 0;
            for (int i = 1; i < 3; ++i)
                if (high[i] - low[i] > high[axis] - low[axis])
                    axis = i;
            if (last - first <= 1 || (last - first <= max_faces && high[axis] - low[axis] <= max_extent)) {
                std::lock_guard lock(ranges_mutex);
                ranges.emplace_back(first, last);
                return;
            }
            const size_t middle = first + (last - first) / 2;
            std::nth_element(mesh.faces.begin() + std::ptrdiff_t(first), mesh.faces.begin() + std::ptrdiff_t(middle), mesh.faces.begin() + std::ptrdiff_t(last),
                             [&](const auto& a, const auto& b) { return mesh.vertices[a[0]][axis] < mesh.vertices[b[0]][axis]; });
            if (threads_left > 1) {
                std::exception_ptr failure;
                std::thread other([&] {
                    try {
                        split_faces(mesh, first, middle, max_faces, max_extent, threads_left / 2, ranges, ranges_mutex);
                    } catch (...) {
                        failure = std::current_exception();
                    }
                });
                try {
                    split_faces(mesh, middle, last, max_faces, max_extent, threads_left - threads_left / 2, ranges, ranges_mutex);
                } catch (...) {
                    other.join();
                    throw;
                }
                other.join();
                if (failure)
                    std::rethrow_exception(failure);
            } else {
                split_faces(mesh, first, middle, max_faces, max_extent, 1, ranges, ranges_mutex);
                split_faces(mesh, middle, last, max_faces, max_extent, 1, ranges, ranges_mutex);
            }
        }

        // Simplifies each piece on its own, so RAM follows the piece size. Vertices that pieces share stay put,
        // so no cracks open between them, and the seams keep their density. Leaves unused vertices for the caller.
        void simplify_in_pieces(SplatMesh& mesh, const float max_error, const float color_weight, const size_t piece_faces) {
            const int threads = std::clamp(int(std::thread::hardware_concurrency()), 1, kSimplifyThreads);
            std::vector<std::pair<size_t, size_t>> ranges;
            std::mutex ranges_mutex;
            split_faces(mesh, 0, mesh.faces.size(), piece_faces, kPieceExtentPerTolerance * max_error, threads, ranges, ranges_mutex);
            std::ranges::sort(ranges);

            std::vector<int32_t> owner(mesh.vertices.size(), -1);
            std::vector<uint8_t> shared(mesh.vertices.size(), 0);
            for (size_t r = 0; r < ranges.size(); ++r)
                for (size_t f = ranges[r].first; f < ranges[r].second; ++f)
                    for (const auto vertex : mesh.faces[f]) {
                        if (owner[vertex] < 0)
                            owner[vertex] = static_cast<int32_t>(r);
                        else if (owner[vertex] != static_cast<int32_t>(r))
                            shared[vertex] = 1;
                    }
            owner = {};

            const bool has_colors = mesh.colors.size() == mesh.vertices.size();
            std::vector<size_t> kept(ranges.size(), 0);
            std::vector<float> errors(ranges.size(), 0.f);
            std::atomic<size_t> next{0};
            std::vector<std::exception_ptr> failures(static_cast<size_t>(threads));
            const auto work = [&](const int thread) {
                try {
                    for (size_t r = next++; r < ranges.size(); r = next++) {
                        const auto [first, last] = ranges[r];
                        std::vector<int32_t> used;
                        used.reserve((last - first) * 3);
                        for (size_t f = first; f < last; ++f)
                            used.insert(used.end(), mesh.faces[f].begin(), mesh.faces[f].end());
                        std::ranges::sort(used);
                        used.erase(std::unique(used.begin(), used.end()), used.end());
                        std::vector<unsigned int> indices;
                        indices.reserve((last - first) * 3);
                        for (size_t f = first; f < last; ++f)
                            for (const auto vertex : mesh.faces[f])
                                indices.push_back(static_cast<unsigned int>(std::ranges::lower_bound(used, vertex) - used.begin()));
                        std::vector<float> positions(used.size() * 3), attributes(has_colors ? used.size() * 3 : 0);
                        std::vector<unsigned char> lock(used.size());
                        for (size_t v = 0; v < used.size(); ++v) {
                            std::memcpy(&positions[3 * v], mesh.vertices[used[v]].data(), 12);
                            lock[v] = shared[used[v]];
                            if (has_colors)
                                for (int c = 0; c < 3; ++c)
                                    attributes[3 * v + c] = float(mesh.colors[used[v]][c]) / 255.f;
                        }
                        const float weights[3] = {color_weight, color_weight, color_weight};
                        const size_t index_count = meshopt_simplifyWithUpdate(
                            indices.data(), indices.size(), positions.data(), used.size(), 12, has_colors ? attributes.data() : nullptr, 12, weights,
                            has_colors ? 3 : 0, lock.data(), 0, max_error, meshopt_SimplifyErrorAbsolute | meshopt_SimplifyPrune, &errors[r]);
                        // A vertex that moved belongs to this piece alone.
                        for (size_t v = 0; v < used.size(); ++v) {
                            if (lock[v])
                                continue;
                            std::memcpy(mesh.vertices[used[v]].data(), &positions[3 * v], 12);
                            if (has_colors)
                                for (int c = 0; c < 3; ++c)
                                    mesh.colors[used[v]][c] = static_cast<uint8_t>(std::clamp(std::lround(255.f * attributes[3 * v + c]), 0l, 255l));
                        }
                        for (size_t i = 0; i < index_count / 3; ++i)
                            for (int c = 0; c < 3; ++c)
                                mesh.faces[first + i][c] = used[indices[3 * i + c]];
                        kept[r] = index_count / 3;
                    }
                } catch (...) {
                    failures[size_t(thread)] = std::current_exception();
                    next = ranges.size();
                }
            };
            std::vector<std::thread> pool;
            for (int t = 1; t < threads; ++t)
                pool.emplace_back(work, t);
            work(0);
            for (auto& thread : pool)
                thread.join();
            for (const auto& failure : failures)
                if (failure)
                    std::rethrow_exception(failure);

            size_t out = 0;
            for (size_t r = 0; r < ranges.size(); ++r) {
                std::copy_n(mesh.faces.begin() + std::ptrdiff_t(ranges[r].first), kept[r], mesh.faces.begin() + std::ptrdiff_t(out));
                out += kept[r];
            }
            mesh.faces.resize(out);
            mesh.simplify_result_error = errors.empty() ? 0.f : *std::ranges::max_element(errors);
        }
    } // namespace

    std::vector<std::pair<size_t, size_t>> split_mesh_into_pieces(SplatMesh& mesh, const size_t max_faces, const float max_extent, const int threads) {
        std::vector<std::pair<size_t, size_t>> ranges;
        std::mutex ranges_mutex;
        split_faces(mesh, 0, mesh.faces.size(), max_faces, max_extent, std::clamp(int(std::thread::hardware_concurrency()), 1, threads), ranges, ranges_mutex);
        std::ranges::sort(ranges);
        return ranges;
    }

    void simplify_splat_mesh(SplatMesh& mesh, const float max_error, const float color_weight, const size_t piece_faces) {
        mesh.faces_before_simplify = mesh.faces.size();
        if (max_error <= 0.f || mesh.faces.empty())
            return;
        const bool has_colors = mesh.colors.size() == mesh.vertices.size();
        simplify_in_pieces(mesh, max_error, color_weight, piece_faces);

        std::vector<int32_t> remap(mesh.vertices.size(), -1);
        std::vector<std::array<float, 3>> vertices;
        std::vector<std::array<uint8_t, 3>> colors;
        for (auto& face : mesh.faces)
            for (auto& vertex : face) {
                auto& target = remap[vertex];
                if (target < 0) {
                    target = static_cast<int32_t>(vertices.size());
                    vertices.push_back(mesh.vertices[vertex]);
                    if (has_colors)
                        colors.push_back(mesh.colors[vertex]);
                }
                vertex = target;
            }
        mesh.vertices = std::move(vertices);
        mesh.colors = std::move(colors);
        compute_splat_mesh_normals(mesh);
    }

    namespace {
        // Takes each page's colors from the photos, through the same appearance inverse as the vertex colors. Uses every
        // photo: a surface's best view is one or two photos out of hundreds, and a stride leaves it out by chance, which
        // took lettering from 7 m instead of 2.5 m.
        // Views and the leveling between them are settled over the whole mesh first, so that nothing steps where one batch
        // meets the next. Then pages go through in batches of neighbors, each held in memory only while it bakes and
        // read only from the photos that can see it. page_groups are runs of pages that stay in one batch.
        void bake_texture(SplatMesh& mesh, core::SplatData& model, const std::span<core::Camera* const> views, const core::SplatMeshSettings& settings,
                          cudaStream_t stream, const auto& report, const PPISP* appearance, const std::span<const std::pair<size_t, size_t>> page_groups) {
            const size_t page_count = mesh.page_first_face.size() - 1;
            mesh.pages.assign(page_count, {});
            core::GlobalArenaManager::instance().get_arena().release_idle_for_pressure();
            core::Tensor::trim_memory_pool();
            constexpr size_t bytes_per_texel = 52;
            const auto room = core::MemoryPressureCoordinator::instance().preflight({.operation = "Mesh texture"});
            const size_t available = room.effective_free_bytes > room.required_peak_bytes ? room.effective_free_bytes - room.required_peak_bytes : 0;
            const size_t max_texels = std::min(size_t(double(available) * kTextureShareOfFreeMemory) / bytes_per_texel, kBakeBatchTexels);

            auto bake_settings = settings;
            bake_settings.max_width = settings.texture_width;
            // A coarse level lies as far from the surface as its faces were allowed to move.
            const float base_tolerance = 2.5f * settings.voxel_size;
            const auto tolerance_of = [&](const size_t face) { return mesh.face_error.empty() ? base_tolerance : std::max(base_tolerance, kFaceErrorToTolerance * mesh.face_error[face]); };

            // Batches of whole groups, counted in covered texels: about a fifth of the page texels on both test scenes.
            std::vector<size_t> covered(page_count);
            detail::parallel_for(page_count, kSimplifyThreads, [&](const size_t p) { covered[p] = rasterize_mesh_texels(mesh, p).pixels.size(); });
            struct Batch {
                size_t first_page = 0, last_page = 0;
                std::vector<uint8_t> uses_view;
            };
            std::vector<Batch> batches;
            size_t held = 0;
            for (const auto& [first, last] : page_groups) {
                size_t texels = 0;
                for (size_t p = first; p < last; ++p)
                    texels += covered[p];
                if (batches.empty() || held + texels > max_texels) {
                    batches.push_back({.first_page = first, .last_page = last, .uses_view = std::vector<uint8_t>(views.size(), 0)});
                    held = 0;
                } else {
                    batches.back().last_page = last;
                }
                held += texels;
            }
            // Choosing views takes a pass over the photos; the batches, with the seams, about one more in all.
            const float pass_share = 1.f / 2.f;

            // Which view sets the detail of each face.
            const auto face_count = static_cast<uint32_t>(mesh.faces.size());
            FaceViewCandidates candidates(face_count);
            MaskCache masks;
            {
                DeviceArray<float> vertices(mesh.vertices.size() * 3, stream), scores(face_count, stream);
                DeviceArray<int32_t> faces(size_t(face_count) * 3, stream);
                DeviceArray<float> tolerances(face_count, stream);
                {
                    std::vector<float> host_tolerances(face_count);
                    for (size_t f = 0; f < face_count; ++f)
                        host_tolerances[f] = tolerance_of(f);
                    LFS_CUDA_CHECK(cudaMemcpyAsync(tolerances.data(), host_tolerances.data(), face_count * sizeof(float), cudaMemcpyHostToDevice, stream));
                    LFS_ENSURE_CUDA_SUCCESS(cudaStreamSynchronize(stream), "Mesh tolerances");
                }
                LFS_CUDA_CHECK(cudaMemcpyAsync(vertices.data(), mesh.vertices.data(), mesh.vertices.size() * 3 * sizeof(float), cudaMemcpyHostToDevice, stream));
                LFS_CUDA_CHECK(cudaMemcpyAsync(faces.data(), mesh.faces.data(), size_t(face_count) * 3 * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
                std::vector<float> host(face_count);
                for (size_t i = 0; i < views.size(); ++i) {
                    report(pass_share * float(i) / float(views.size()));
                    auto& camera = *views[i];
                    if (!camera.has_image())
                        continue;
                    camera.load_image_size(1, settings.texture_width);
                    const auto view = make_view(camera);
                    auto depth = render_median_depth(camera, model, bake_settings, stream, &masks);
                    gsplat_lfs::mesh_face_view_score(vertices.data(), faces.data(), face_count, view, depth.ptr<float>(), tolerances.data(), scores.data(), stream);
                    LFS_CUDA_CHECK(cudaMemcpyAsync(host.data(), scores.data(), host.size() * sizeof(float), cudaMemcpyDeviceToHost, stream));
                    LFS_ENSURE_CUDA_SUCCESS(cudaStreamSynchronize(stream), "Mesh face views");
                    candidates.add(static_cast<int32_t>(i), host);
                }
            }
            LOG_INFO("Mesh: {} of {} views carry rolling shutter motion", std::ranges::count_if(views, [](const core::Camera* camera) { return camera->rolling_shutter().has_value(); }), views.size());
            const auto face_views = choose_face_views(mesh, candidates);
            {
                std::vector<int32_t> used;
                for (const auto label : face_views)
                    if (label >= 0)
                        used.push_back(label);
                const size_t chosen = used.size();
                std::ranges::sort(used);
                used.erase(std::unique(used.begin(), used.end()), used.end());
                LOG_INFO("Mesh: {} views set the detail of {} of {} faces", used.size(), chosen, face_views.size());
            }
            // A batch needs the views chosen for its faces, and those nearly as good, which fill in what a chosen view cannot
            // see. A fisheye sees most of a room, so leaving out the rest saves most of the reads.
            constexpr size_t kCandidates = FaceViewCandidates::kPerFace;
            constexpr float kFillsIn = .5f;
            size_t batch_views_total = 0;
            for (auto& batch : batches) {
                for (size_t f = mesh.page_first_face[batch.first_page]; f < mesh.page_first_face[batch.last_page]; ++f) {
                    if (face_views[f] >= 0)
                        batch.uses_view[size_t(face_views[f])] = 1;
                    for (size_t slot = 0; slot < kCandidates; ++slot)
                        if (const float score = candidates.scores[f * kCandidates + slot]; score > 0.f && score >= kFillsIn * candidates.scores[f * kCandidates])
                            batch.uses_view[size_t(candidates.views[f * kCandidates + slot])] = 1;
                }
                batch_views_total += size_t(std::ranges::count(batch.uses_view, uint8_t{1}));
            }
            candidates = FaceViewCandidates(0);
            LOG_INFO("Mesh: {} batches of at most {} M covered texels read {:.1f} times the photos in all", batches.size(), max_texels / 1000000, double(batch_views_total) / double(std::max<size_t>(views.size(), 1)));

            // A photo and its blur, through the appearance inverse; false if it has no use here.
            DeviceArray<float> scratch, weights, low;
            const auto read_view = [&](core::Camera& camera, const gsplat_lfs::TsdfView& view, Tensor& depth, Tensor& photo) {
                depth = render_median_depth(camera, model, bake_settings, stream, &masks);
                photo = load_photo(camera, settings.texture_width);
                const size_t pixels = size_t(view.width) * view.height;
                if (photo.numel() != pixels * 3) {
                    LOG_WARN("Mesh: photo size of {} does not match the render; skipping its colors", camera.image_name());
                    return false;
                }
                photo = photo.to(Device::CUDA, stream).to(core::DataType::Float32).contiguous();
                if (appearance && appearance->is_known_camera(camera.camera_id()) && appearance->is_known_frame(camera.uid()))
                    photo = appearance->invert(photo, camera.camera_id(), camera.uid(), stream);
                if (low.count != pixels * 3) {
                    scratch = DeviceArray<float>(pixels * 3, stream);
                    low = DeviceArray<float>(pixels * 3, stream);
                    weights = DeviceArray<float>(pixels * 2, stream);
                }
                gsplat_lfs::mesh_photo_low_pass(photo.ptr<float>(), depth.ptr<float>(), view.width, view.height, std::max(int(view.width) / kBlendWidthPerRadius, 1),
                                                scratch.data(), weights.data(), low.data(), stream);
                return true;
            };

            // Seam samples are read in the first batch's pass over the photos, which then takes every view, and leveled
            // before its pages are composed.
            auto leveling = plan_color_leveling(mesh, face_views);
            const auto sample_count = static_cast<uint32_t>(leveling.sample_labels.size());
            struct SeamArrays {
                DeviceArray<float> positions, sum, weight, high, best, tolerances;
                DeviceArray<int32_t> labels;
            } seams;
            if (sample_count > 0) {
                seams.positions = DeviceArray<float>(size_t(sample_count) * 3, stream);
                seams.sum = DeviceArray<float>(size_t(sample_count) * 3, stream, true);
                seams.weight = DeviceArray<float>(sample_count, stream, true);
                seams.high = DeviceArray<float>(size_t(sample_count) * 3, stream, true);
                seams.best = DeviceArray<float>(sample_count, stream, true);
                seams.tolerances = DeviceArray<float>(sample_count, stream);
                seams.labels = DeviceArray<int32_t>(sample_count, stream);
                std::vector<float> host_tolerances(sample_count);
                for (size_t s = 0; s < sample_count; ++s)
                    host_tolerances[s] = tolerance_of(leveling.sample_faces[s]);
                LFS_CUDA_CHECK(cudaMemcpyAsync(seams.positions.data(), leveling.sample_positions.data(), size_t(sample_count) * 3 * sizeof(float), cudaMemcpyHostToDevice, stream));
                LFS_CUDA_CHECK(cudaMemcpyAsync(seams.labels.data(), leveling.sample_labels.data(), size_t(sample_count) * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
                LFS_CUDA_CHECK(cudaMemcpyAsync(seams.tolerances.data(), host_tolerances.data(), size_t(sample_count) * sizeof(float), cudaMemcpyHostToDevice, stream));
                LFS_ENSURE_CUDA_SUCCESS(cudaStreamSynchronize(stream), "Mesh seam samples upload");
            }
            const auto finish_seams = [&] {
                std::vector<float> host_colors(size_t(sample_count) * 3, -1.f);
                if (sample_count > 0) {
                    std::vector<float> means(size_t(sample_count) * 3), mean_weights(sample_count), direct(size_t(sample_count) * 3), bests(sample_count);
                    LFS_CUDA_CHECK(cudaMemcpyAsync(means.data(), seams.sum.data(), means.size() * sizeof(float), cudaMemcpyDeviceToHost, stream));
                    LFS_CUDA_CHECK(cudaMemcpyAsync(mean_weights.data(), seams.weight.data(), mean_weights.size() * sizeof(float), cudaMemcpyDeviceToHost, stream));
                    LFS_CUDA_CHECK(cudaMemcpyAsync(direct.data(), seams.high.data(), direct.size() * sizeof(float), cudaMemcpyDeviceToHost, stream));
                    LFS_CUDA_CHECK(cudaMemcpyAsync(bests.data(), seams.best.data(), bests.size() * sizeof(float), cudaMemcpyDeviceToHost, stream));
                    LFS_ENSURE_CUDA_SUCCESS(cudaStreamSynchronize(stream), "Mesh seam samples download");
                    seams = {};
                    for (size_t s = 0; s < sample_count; ++s)
                        for (size_t c = 0; c < 3; ++c) {
                            if (std::isinf(bests[s]))
                                host_colors[3 * s + c] = direct[3 * s + c];
                            // Where no view sets the detail, the texels hold the mean of all views.
                            else if (leveling.sample_labels[s] < 0 && mean_weights[s] > 0.f)
                                host_colors[3 * s + c] = means[3 * s + c] / mean_weights[s];
                        }
                }
                solve_color_leveling(leveling, mesh, host_colors);
                leveling.release_samples();
            };
            if (batches.empty())
                finish_seams();
            if (!batches.empty() && sample_count > 0) {
                batch_views_total -= size_t(std::ranges::count(batches.front().uses_view, uint8_t{1}));
                batch_views_total += views.size();
            }

            size_t observed = 0, total = 0, batch_views_done = 0;
            for (size_t b = 0; b < batches.size(); ++b) {
                const auto& batch = batches[b];
                const bool with_seams = b == 0 && sample_count > 0;
                const size_t pages_here = batch.last_page - batch.first_page;
                std::vector<MeshTexels> texels(pages_here);
                detail::parallel_for(pages_here, kSimplifyThreads, [&](const size_t p) { texels[p] = rasterize_mesh_texels(mesh, batch.first_page + p); });
                size_t count = 0;
                for (const auto& page : texels)
                    count += page.pixels.size();
                DeviceArray<float> positions(count * 3, stream), low_sum(count * 3, stream, true), low_weight(count, stream, true);
                DeviceArray<float> high(count * 3, stream, true), best_weight(count, stream, true);
                DeviceArray<int32_t> labels(count, stream);
                DeviceArray<float> tolerances(count, stream);
                {
                    std::vector<int32_t> host_labels;
                    std::vector<float> host_tolerances;
                    host_labels.reserve(count);
                    host_tolerances.reserve(count);
                    for (const auto& page : texels)
                        for (const auto face : page.faces) {
                            host_labels.push_back(face_views[face]);
                            host_tolerances.push_back(tolerance_of(face));
                        }
                    LFS_CUDA_CHECK(cudaMemcpyAsync(labels.data(), host_labels.data(), count * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
                    LFS_CUDA_CHECK(cudaMemcpyAsync(tolerances.data(), host_tolerances.data(), count * sizeof(float), cudaMemcpyHostToDevice, stream));
                    size_t offset = 0;
                    for (const auto& page : texels) {
                        LFS_CUDA_CHECK(cudaMemcpyAsync(positions.data() + 3 * offset, page.positions.data(), page.positions.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
                        offset += page.pixels.size();
                    }
                    LFS_ENSURE_CUDA_SUCCESS(cudaStreamSynchronize(stream), "Mesh texture upload");
                }
                // Rasterized again to compose: holding them through the photos and the leveling took 23 bytes per texel.
                texels = std::vector<MeshTexels>(pages_here);
                for (size_t i = 0; i < views.size(); ++i) {
                    if (!batch.uses_view[i] && !with_seams)
                        continue;
                    report(pass_share * (1.f + float(batch_views_done++) / float(std::max<size_t>(batch_views_total, 1))));
                    auto& camera = *views[i];
                    if (!camera.has_image())
                        continue;
                    camera.load_image_size(1, settings.texture_width);
                    const auto view = make_view(camera);
                    Tensor depth, photo;
                    if (!read_view(camera, view, depth, photo))
                        continue;
                    // Blurred, so that misregistration and grain do not read as a step.
                    if (with_seams)
                        gsplat_lfs::mesh_texture_accumulate(seams.positions.data(), sample_count, view, static_cast<int32_t>(i), seams.labels.data(), depth.ptr<float>(), low.data(), low.data(), seams.tolerances.data(),
                                                            seams.sum.data(), seams.weight.data(), seams.high.data(), seams.best.data(), stream);
                    if (batch.uses_view[i])
                        gsplat_lfs::mesh_texture_accumulate(positions.data(), static_cast<uint32_t>(count), view, static_cast<int32_t>(i), labels.data(), depth.ptr<float>(), photo.ptr<float>(), low.data(), tolerances.data(),
                                                            low_sum.data(), low_weight.data(), high.data(), best_weight.data(), stream);
                    LFS_ENSURE_CUDA_SUCCESS(cudaStreamSynchronize(stream), "Mesh texture");
                }
                // The batch's sums stay on the GPU while the leveling is solved, and come down a few pages at a time.
                if (b == 0)
                    finish_seams();
                std::vector<size_t> page_offsets(pages_here + 1, 0);
                for (size_t p = 0; p < pages_here; ++p)
                    page_offsets[p + 1] = page_offsets[p] + covered[batch.first_page + p];
                std::vector<float> sums, sum_weights, highs, bests;
                std::atomic<size_t> seen_here{0};
                for (size_t first = 0; first < pages_here;) {
                    size_t last = first + 1;
                    while (last < pages_here && page_offsets[last + 1] - page_offsets[first] <= kComposeChunkTexels)
                        ++last;
                    const size_t from = page_offsets[first], n = page_offsets[last] - from;
                    sums.resize(n * 3);
                    sum_weights.resize(n);
                    highs.resize(n * 3);
                    bests.resize(n);
                    LFS_CUDA_CHECK(cudaMemcpyAsync(sums.data(), low_sum.data() + 3 * from, n * 3 * sizeof(float), cudaMemcpyDeviceToHost, stream));
                    LFS_CUDA_CHECK(cudaMemcpyAsync(sum_weights.data(), low_weight.data() + from, n * sizeof(float), cudaMemcpyDeviceToHost, stream));
                    LFS_CUDA_CHECK(cudaMemcpyAsync(highs.data(), high.data() + 3 * from, n * 3 * sizeof(float), cudaMemcpyDeviceToHost, stream));
                    LFS_CUDA_CHECK(cudaMemcpyAsync(bests.data(), best_weight.data() + from, n * sizeof(float), cudaMemcpyDeviceToHost, stream));
                    LFS_ENSURE_CUDA_SUCCESS(cudaStreamSynchronize(stream), "Mesh texture download");
                    detail::parallel_for(last - first, kSimplifyThreads, [&](const size_t k) {
                        const size_t p = first + k;
                        texels[p] = rasterize_mesh_texels(mesh, batch.first_page + p);
                        const size_t page_texels = texels[p].pixels.size(), offset = page_offsets[p] - from;
                        if (page_texels != covered[batch.first_page + p])
                            throw collisionError("selection.operation_failed_generic", "A page rasterized differently the second time", ErrorCode::Internal);
                        std::vector<float> colors(page_texels * 3);
                        size_t seen_page = 0;
                        for (size_t t = 0; t < page_texels; ++t) {
                            const size_t at = offset + t;
                            const bool seen = sum_weights[at] > 0.f;
                            seen_page += seen;
                            if (!seen) {
                                colors[3 * t] = colors[3 * t + 1] = colors[3 * t + 2] = -1.f;
                                continue;
                            }
                            float level[3];
                            leveling.offset_at(mesh, texels[p].faces[t], &texels[p].positions[3 * t], level);
                            const bool direct = std::isinf(bests[at]);
                            for (size_t c = 0; c < 3; ++c)
                                colors[3 * t + c] = std::max((direct ? 0.f : sums[3 * at + c] / sum_weights[at]) + highs[3 * at + c] + level[c], 0.f);
                        }
                        seen_here += seen_page;
                        mesh.pages[batch.first_page + p] = compose_mesh_page(texels[p], colors, mesh.page_side(batch.first_page + p));
                        texels[p] = {};
                    });
                    first = last;
                }
                observed += seen_here;
                total += count;
            }
            mesh.texels = total;
            mesh.observed_texel_share = total > 0 ? float(observed) / float(total) : 0.f;
            LOG_INFO("Mesh: {} pages, {} texels, {:.1f}% seen by a view", page_count, total, 100.f * mesh.observed_texel_share);
        }
        // The surface from the depth of the views: fused, cleaned up, colored at its vertices and reduced.
        void extract_surface(SplatMesh& mesh, core::SplatData& model, const std::span<core::Camera* const> views, const core::SplatMeshSettings& settings, cudaStream_t stream,
                             const std::stop_token stop, const core::ProgressCallback& progress, const PPISP* appearance, const float geometry_share) {
            const auto stage = [](const std::string_view key, const char* diagnostic) { return std::string(core::collisionStage(key, diagnostic)); };
            // Widen the band to the local depth spacing where neighbouring pixels disagree, which is
            // the one lever that reaches voxels no view ever touched. Defaults come from the settings;
            // LFS_MESH_TRUNC_GRAD / LFS_MESH_TRUNC_CAP can still override them from the environment.
            float trunc_gradient = settings.trunc_gradient, trunc_cap = settings.trunc_cap;
            if (const char* text = std::getenv("LFS_MESH_TRUNC_GRAD"); text != nullptr && *text != '\0')
                trunc_gradient = std::strtof(text, nullptr);
            if (const char* text = std::getenv("LFS_MESH_TRUNC_CAP"); text != nullptr && *text != '\0')
                trunc_cap = std::strtof(text, nullptr);
            // A zero cap asks for the one that scales with the voxel grid: four bands keeps the cap above
            // the truncation so the gradient still widens something, and keeps the block reach bounded.
            if (!(trunc_cap > 0.f))
                trunc_cap = kAutoTruncCapBands * settings.band();
            Tsdf tsdf;
            tsdf.grid = {.voxel = settings.voxel_size, .block = static_cast<uint32_t>(settings.block_voxels),
                         .trunc = settings.band(), .trunc_gradient = trunc_gradient, .trunc_cap = trunc_cap,
                         .full_weight_range = settings.full_weight_range, .max_range = settings.max_range};
            mesh.voxel_size = settings.voxel_size;
            mesh.truncation = tsdf.grid.trunc;
            if (trunc_gradient > 0.f)
                LOG_INFO("Mesh: band {} to {} by {} times the local depth step", tsdf.grid.trunc, tsdf.grid.trunc_cap, trunc_gradient);
            const float block_extent = settings.voxel_size * static_cast<float>(settings.block_voxels);
            const uint32_t touch_reach = static_cast<uint32_t>(std::max(1.f, std::ceil(gsplat_lfs::tsdf_widest_band(tsdf.grid) / block_extent)));

            MeshDepthDump depth_dump;
            if (const char* dir = std::getenv("LFS_MESH_DUMP_DIR"); dir != nullptr && *dir != '\0')
                depth_dump.open(dir, settings, views.size(), tsdf.grid.trunc);

            // Depth stays in RAM between the touch and the integrate passes, which need the block set first.
            auto step = Clock::now();
            std::vector<std::vector<float>> depths(views.size());
            DeviceArray<uint64_t> block_keys;
            Tensor rays;
            size_t block_count = 0;
            size_t touch_batches = 0, split_touch_views = 0;
            double seconds_touch = 0.;
            for (size_t i = 0; i < views.size(); ++i) {
                core::reportProgress(stop, progress, geometry_share * .5f * float(i) / views.size(), stage("mesh.stage_render", "Rendering depth"));
                if (i > 0 && (i % 50 == 0 || i + 1 == views.size()))
                    log_phase_mem(std::format("render_view_{}/{}", i, views.size()).c_str());
                auto& camera = *views[i];
                camera.load_image_size(1, settings.max_width);
                const auto view = make_view(camera);
                // Once touch has needed splitting, do not let idle render caches compete with the next
                // view and the growing key set. The normal-memory path keeps those caches for speed.
                if (split_touch_views > 0) {
                    core::GlobalArenaManager::instance().get_arena().release_idle_for_pressure();
                    core::Tensor::trim_memory_pool();
                }
                auto depth = render_median_depth(camera, model, settings, stream);
                const auto touch_start = Clock::now();
                const size_t pixels = size_t(view.width) * view.height;
                if (!rays.is_valid() || rays.numel() != pixels * 3 || i == 0 || views[i - 1]->camera_id() != camera.camera_id()) {
                    rays = Tensor::empty({pixels * 3}, Device::CUDA);
                    rays.set_stream(stream);
                    camera_rays(view, rays.ptr<float>(), stream);
                }
                const size_t samples = size_t((view.width + 1) / 2) * ((view.height + 1) / 2);
                const size_t touch_stride = 2 * size_t(touch_reach) + 1;
                size_t view_batches = 0;
                for (size_t first_sample = 0; first_sample < samples;) {
                    // Before this batch the existing keys, depth and rays are already resident. Budget
                    // everything that joins them: added, merged, scratch, the counter and CUB's workspace.
                    const auto room = core::MemoryPressureCoordinator::instance().preflight({.operation = "Mesh depth touch"});
                    const size_t available = room.effective_free_bytes > room.required_peak_bytes
                                                 ? room.effective_free_bytes - room.required_peak_bytes
                                                 : 0;
                    const size_t budget = size_t(double(available) * kMeshShareOfFreeMemory);
                    size_t batch_samples = samples - first_sample;
                    size_t added_count = 0, required = 0;
                    for (;;) {
                        added_count = core::checked_product(batch_samples, touch_stride, "Mesh touch keys");
                        const size_t total = block_count + added_count;
                        const size_t arrays = core::saturating_add(
                            core::saturating_multiply(added_count, sizeof(uint64_t)),
                            core::saturating_multiply(total, 2 * sizeof(uint64_t)));
                        required = core::saturating_add(
                            core::saturating_add(arrays, gsplat_lfs::tsdf_sort_unique_workspace_bytes(total, stream)),
                            sizeof(uint64_t));
                        if (required <= budget)
                            break;
                        if (batch_samples == 1)
                            throw collisionError("mesh.memory", "Not enough GPU memory to merge TSDF block keys; free memory or use a larger voxel size",
                                                 ErrorCode::ResourceExhausted,
                                                 std::format("{} existing blocks, {:.1f} MiB available", block_count,
                                                             static_cast<double>(available) / (1 << 20)));
                        batch_samples = std::max<size_t>(1, batch_samples / 2);
                    }
                    if (batch_samples < samples)
                        LOG_INFO("Mesh: touch view {}: {} samples + {} existing blocks, {:.2f} GiB temporary, {:.2f} GiB available",
                                 i, batch_samples, block_count, static_cast<double>(required) / (1 << 30),
                                 static_cast<double>(available) / (1 << 30));
                    DeviceArray<uint64_t> added(added_count, stream);
                    gsplat_lfs::tsdf_touch_keys_range(depth.ptr<float>(), rays.ptr<float>(), view, tsdf.grid, touch_reach,
                                                      first_sample, batch_samples, added.data(), stream);
                    block_keys = merge_unique_keys(block_keys, block_count, added, block_count, stream);
                    first_sample += batch_samples;
                    ++view_batches;
                }
                touch_batches += view_batches;
                split_touch_views += view_batches > 1;
                seconds_touch += seconds_since(touch_start);
                depths[i].resize(pixels);
                LFS_CUDA_CHECK(cudaMemcpyAsync(depths[i].data(), depth.ptr<float>(), pixels * sizeof(float), cudaMemcpyDeviceToHost, stream));
                LFS_ENSURE_CUDA_SUCCESS(cudaStreamSynchronize(stream), "Mesh depth download");
                depth_dump.write(i, camera.uid(), camera.image_name(), view.width, view.height, depths[i]);
            }
            rays = {};
            mesh.seconds_render = seconds_since(step) - seconds_touch;
            LOG_INFO("Mesh: rendering in {:.2f}s and depth touch in {:.2f}s over {} views, {} blocks",
                     mesh.seconds_render, seconds_touch, views.size(), block_count);
            if (split_touch_views > 0)
                LOG_INFO("Mesh: adaptive depth touch split {} views into {} batches", split_touch_views, touch_batches);
            log_phase_mem("after_render_and_touch");
            if (block_count == 0)
                throw collisionError("mesh.empty", "No surface was rendered from these cameras");
            if (block_count > (size_t{1} << 31))
                throw collisionError("mesh.memory", "Too many TSDF blocks; use a larger voxel size", ErrorCode::ResourceExhausted);

            step = Clock::now();
            mesh.blocks = block_count;
            std::vector<uint64_t> host_blocks(block_count);
            LFS_CUDA_CHECK(cudaMemcpyAsync(host_blocks.data(), block_keys.data(), block_count * sizeof(uint64_t), cudaMemcpyDeviceToHost, stream));
            LFS_ENSURE_CUDA_SUCCESS(cudaStreamSynchronize(stream), "Mesh block download");
            block_keys = {};
            // What the renders left behind is the largest thing in the way of the TSDF.
            core::GlobalArenaManager::instance().get_arena().release_idle_for_pressure();
            core::Tensor::trim_memory_pool();
            const auto room = core::MemoryPressureCoordinator::instance().preflight({.operation = "Mesh TSDF"});
            const size_t available = room.effective_free_bytes > room.required_peak_bytes ? room.effective_free_bytes - room.required_peak_bytes : 0;
            // The rest holds a depth map, the normals, the voxels on their way to RAM and whatever else runs on the GPU.
            const size_t max_blocks = std::min(size_t(double(available) * kTsdfShareOfFreeMemory) / tsdf_block_bytes(tsdf.grid), size_t{1} << 31);
            const auto chunks = plan_chunks(host_blocks, max_blocks);
            LOG_INFO("Mesh: {} blocks, {:.2f} GB TSDF in {} chunks, {:.2f} GB free", block_count,
                     static_cast<double>(block_count * tsdf_block_bytes(tsdf.grid)) / (1 << 30), chunks.size(), static_cast<double>(available) / (1 << 30));
            SparseVoxels voxels;
            const float view_steps = float(chunks.size()) * float(views.size());
            for (size_t c = 0; c < chunks.size(); ++c) {
                const auto& chunk = chunks[c];
                {
                    const auto grid = tsdf.grid;
                    tsdf = {};
                    tsdf.grid = grid;
                }
                core::Tensor::trim_memory_pool();
                const size_t held = chunk.held_last - chunk.held_first;
                DeviceArray<uint64_t> held_keys(held, stream);
                LFS_CUDA_CHECK(cudaMemcpyAsync(held_keys.data(), host_blocks.data() + chunk.held_first, held * sizeof(uint64_t), cudaMemcpyHostToDevice, stream));
                allocate_tsdf(tsdf, std::move(held_keys), held, stream);
                log_phase_mem(std::format("tsdf_chunk_{}/{}_allocated", c + 1, chunks.size()).c_str());
                for (size_t i = 0; i < views.size(); ++i) {
                    core::reportProgress(stop, progress, geometry_share * (.5f + .3f * (float(c) * float(views.size()) + float(i)) / view_steps), stage("mesh.stage_fuse", "Fusing depth"));
                    const auto view = make_view(*views[i]);
                    auto depth = Tensor::empty({size_t(view.height), size_t(view.width)}, Device::CUDA);
                    depth.set_stream(stream);
                    LFS_CUDA_CHECK(cudaMemcpyAsync(depth.ptr<float>(), depths[i].data(), depths[i].size() * sizeof(float), cudaMemcpyHostToDevice, stream));
                    integrate_view(tsdf, view, depth, stream);
                    LFS_ENSURE_CUDA_SUCCESS(cudaStreamSynchronize(stream), "Mesh fusion");
                }
                collect_voxels(tsdf, static_cast<uint32_t>(chunk.first - chunk.held_first), static_cast<uint32_t>(chunk.last - chunk.held_first),
                               static_cast<uint32_t>(settings.min_views), voxels, depth_dump, stream);
            }
            depth_dump.finish();
            host_blocks = {};
            tsdf = {};
            core::Tensor::trim_memory_pool();
            mesh.seconds_fusion = seconds_since(step);
            log_phase_mem("after_fusion");
            ram_preflight(voxels.keys.size());

            step = Clock::now();
            core::reportProgress(stop, progress, geometry_share * .8f, stage("mesh.stage_extract", "Extracting surface"));
            mesh.band_voxels = voxels.keys.size();
            LOG_INFO("Mesh: extracting surface from {} voxels at voxel size {:.4f}", mesh.band_voxels, settings.voxel_size);
            surface_nets(voxels.keys, voxels.tsdf, settings.voxel_size, mesh);
            voxels = {};
            LOG_INFO("Mesh: surface nets in {:.2f}s, {} vertices, {} faces", seconds_since(step), mesh.vertices.size(),
                     mesh.faces.size());
            if (mesh.faces.empty())
                throw collisionError("mesh.empty", "No surface survived; lower the view count or use a larger voxel size");
            // Every topology pass waits until simplification has spent the face budget. The raw surface
            // nets mesh is an order of magnitude denser, and deleting its faces first leaves rims the
            // simplifier has to keep at full resolution, which is what shows up as clusters of sub-voxel
            // triangles. OpenMVS decimates before its later Clean calls for the same reason.
            auto sub = Clock::now();
            compute_splat_mesh_normals(mesh);
            LOG_INFO("Mesh: normals in {:.2f}s", seconds_since(sub));
            mesh.seconds_extract = seconds_since(step);

            step = Clock::now();
            const auto vertex_count = static_cast<uint32_t>(mesh.vertices.size());
            DeviceArray<float> vertices_device(size_t(vertex_count) * 3, stream);
            LFS_CUDA_CHECK(cudaMemcpyAsync(vertices_device.data(), mesh.vertices.data(), size_t(vertex_count) * 3 * sizeof(float), cudaMemcpyHostToDevice, stream));
            DeviceArray<float> color_sum(size_t(vertex_count) * 3, stream, true);
            DeviceArray<float> color_weight(vertex_count, stream, true);
            const float tolerance = 2.5f * settings.voxel_size;
            for (size_t i = 0; i < views.size(); i += settings.color_stride) {
                core::reportProgress(stop, progress, geometry_share * (.85f + .1f * float(i) / views.size()), stage("mesh.stage_color", "Coloring"));
                auto& camera = *views[i];
                if (!camera.has_image())
                    continue;
                const auto view = make_view(camera);
                auto photo = load_photo(camera, settings.max_width);
                const size_t pixels = size_t(view.width) * view.height;
                if (photo.numel() != pixels * 3) {
                    LOG_WARN("Mesh: photo size of {} does not match the render; skipping its colors", camera.image_name());
                    continue;
                }
                photo = photo.to(Device::CUDA, stream).to(core::DataType::Float32).contiguous();
                if (appearance && appearance->is_known_camera(camera.camera_id()) && appearance->is_known_frame(camera.uid())) {
                    photo = appearance->invert(photo, camera.camera_id(), camera.uid(), stream);
                    ++mesh.views_color_corrected;
                }
                auto depth = Tensor::empty({size_t(view.height), size_t(view.width)}, Device::CUDA);
                depth.set_stream(stream);
                LFS_CUDA_CHECK(cudaMemcpyAsync(depth.ptr<float>(), depths[i].data(), depths[i].size() * sizeof(float), cudaMemcpyHostToDevice, stream));
                gsplat_lfs::mesh_color_accumulate(vertices_device.data(), vertex_count, view, depth.ptr<float>(), photo.ptr<float>(), tolerance, color_sum.data(), color_weight.data(), stream);
                LFS_ENSURE_CUDA_SUCCESS(cudaStreamSynchronize(stream), "Mesh color");
            }
            std::vector<float> sums(size_t(vertex_count) * 3);
            std::vector<float> weights(vertex_count);
            LFS_CUDA_CHECK(cudaMemcpyAsync(sums.data(), color_sum.data(), sums.size() * sizeof(float), cudaMemcpyDeviceToHost, stream));
            LFS_CUDA_CHECK(cudaMemcpyAsync(weights.data(), color_weight.data(), weights.size() * sizeof(float), cudaMemcpyDeviceToHost, stream));
            LFS_ENSURE_CUDA_SUCCESS(cudaStreamSynchronize(stream), "Mesh color download");
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
            LOG_INFO("Mesh: colored with the trained appearance taken out of {} views", mesh.views_color_corrected);
            // Reducing faces is the peak of RAM use; it needs none of these, and recomputes the normals.
            depths = {};
            sums = {};
            weights = {};
            if (settings.simplify_error > 0.f)
                mesh.normals = {};
            vertices_device = {};
            color_sum = {};
            color_weight = {};
            core::Tensor::trim_memory_pool();
            core::reportProgress(stop, progress, geometry_share * .95f, stage("mesh.stage_simplify", "Reducing faces"));
            log_phase_mem("before_simplify");
            step = Clock::now();
            simplify_splat_mesh(mesh, settings.simplify_error, kSimplifyColorWeight);
            mesh.seconds_simplify = seconds_since(step);
            LOG_INFO("Mesh: simplification in {:.2f}s, {} vertices, {} faces", mesh.seconds_simplify,
                     mesh.vertices.size(), mesh.faces.size());
            log_phase_mem("after_simplify");
            // Reducing faces moves the kept vertices and merges their neighbors, which can leave the
            // same corner several times. Cleaning after the decimation keeps that damage off the raw
            // mesh: deleting faces up front leaves rims the simplifier would keep at full resolution.
            sub = Clock::now();
            if (settings.clean_topology)
                mesh.faces_removed_degenerate += clean_mesh_topology(mesh);
            // Same OpenMVS ordering: spurious before spikes, so spikes do not waste time on junk
            // triangles the long-edge pass eliminates first.
            if (settings.spurious_edge_factor > 0.f || settings.spurious_component_factor > 0.f)
                mesh.faces_removed_spurious += remove_spurious_components(mesh, settings.spurious_edge_factor,
                                                                          settings.spurious_component_factor);
            if (settings.remove_spikes)
                mesh.faces_removed_spikes += remove_mesh_spikes(mesh);
            // Pass #1 hole fill; simplification leaves the boundary structure intact enough to run it
            // before another topology sweep.
            mesh.holes_filled += fill_small_mesh_holes(mesh, settings.fill_hole_perimeter, settings.ear_cut_holes,
                                                       kFillMaxEdgeVoxels * settings.voxel_size);
            size_t second_topology = 0;
            if (settings.clean_topology) {
                second_topology = clean_mesh_topology(mesh);
                mesh.faces_removed_degenerate += second_topology;
            }
            mesh.holes_filled += fill_small_mesh_holes(mesh, settings.fill_hole_perimeter, settings.ear_cut_holes,
                                                       kFillMaxEdgeVoxels * settings.voxel_size);
            size_t post_clean = 0;
            if (settings.clean_non_manifold) {
                post_clean = clean_non_manifold_only(mesh);
                mesh.faces_removed_degenerate += post_clean;
            }
            size_t isolated_dropped = 0;
            if (settings.remove_isolated_small_components && settings.isolated_component_max_faces > 0 && settings.voxel_size > 0.f) {
                isolated_dropped = remove_isolated_small_mesh_components(
                    mesh, settings.isolated_component_max_faces,
                    settings.isolated_component_neighbor_radius * settings.voxel_size);
                mesh.faces_removed_spurious += isolated_dropped;
            }
            LOG_INFO("Mesh: cleanup in {:.2f}s, {} holes filled ({}), second topology {} faces dropped, {} non-manifold faces dropped{}",
                     seconds_since(sub), mesh.holes_filled,
                     settings.ear_cut_holes > 0 ? std::format("ear cut under {} edges, fan for the rest", settings.ear_cut_holes) : "fan",
                     second_topology, post_clean,
                     isolated_dropped > 0 ? std::format(", {} from small isolated components", isolated_dropped) : std::string{});
            // No topology clean after the final hole fill. A full clean would collapse coincident
            // vertices and split non-manifold ones, which can reopen holes the fills just closed.
            // The non-manifold-only pass already happens before that final fill so its light touch
            // can expose small rims that fill then closes.
            LOG_INFO("Mesh: leaves {} vertices, {} faces, {} normals and {} colors", mesh.vertices.size(),
                     mesh.faces.size(), mesh.normals.size(), mesh.colors.size());
            LOG_INFO("Mesh: dropped {} faces as degenerate or non-manifold and {} as spikes",
                     mesh.faces_removed_degenerate, mesh.faces_removed_spikes);
            core::reportProgress(stop, progress, geometry_share, stage("mesh.stage_simplify", "Reducing faces"));
            LOG_INFO("Mesh: render {:.2f}s, fusion {:.2f}s, extract {:.2f}s, color {:.2f}s, simplify {:.2f}s",
                     mesh.seconds_render + seconds_touch, mesh.seconds_fusion, mesh.seconds_extract, mesh.seconds_color,
                     mesh.seconds_simplify);
            log_phase_mem("mesh_done");
        }
    } // namespace

    SplatMesh extract_splat_mesh(core::SplatData& model, const std::span<core::Camera* const> cameras,
                                 const core::SplatMeshSettings& settings, const std::stop_token stop, const core::ProgressCallback& progress,
                                 const PPISP* appearance) try {
        validate(settings);
        const auto stage = [](const std::string_view key, const char* diagnostic) { return std::string(core::collisionStage(key, diagnostic)); };
        // Baking takes about as long as all the rest.
        const float geometry_share = settings.texel_size > 0.f ? .5f : 1.f;
        std::vector<core::Camera*> views;
        for (size_t i = 0; i < cameras.size(); i += settings.view_stride)
            if (cameras[i])
                views.push_back(cameras[i]);
        if (views.empty())
            throw collisionError("mesh.no_cameras", "The mesh needs the cameras of the dataset; load the project or dataset the model was trained on");
        if (model.size() == 0)
            throw collisionError("collision.geometry_invalid", "The model has no splats", ErrorCode::InvalidArgument);
        // A voxel size of 0 asks for the one the model can resolve, so a scene needs no hand-tuned
        // size. Everything below uses the resolved copy; the caller's settings stay untouched.
        core::SplatMeshSettings resolved = settings;
        if (resolved.voxel_size <= 0.f) {
            // Clamped so that a very small or very large scene still lands on a size the fusion and
            // the cleanup stages can actually afford.
            const float automatic = kAutoVoxelDetail * estimate_voxel_size(model);
            resolved.voxel_size = std::clamp(automatic, kAutoVoxelSizeMin, kAutoVoxelSizeMax);
            LOG_INFO("Mesh: voxel size {:.4f} = {:.2f} x median splat spacing (auto, clamped to [{}, {}])",
                     resolved.voxel_size, kAutoVoxelDetail, kAutoVoxelSizeMin, kAutoVoxelSizeMax);
        }
        // A zero tolerance asks for a fraction of one voxel, so the face count follows the detail the
        // voxel chose instead of needing its own per-scene number.
        if (!(resolved.simplify_error > 0.f)) {
            resolved.simplify_error = kAutoSimplifyVoxels * resolved.voxel_size;
            LOG_INFO("Mesh: simplify tolerance {:.4f} = {:.2f} x voxel size (auto)", resolved.simplify_error,
                     kAutoSimplifyVoxels);
        }
        // A zero perimeter asks for the one the voxel grid implies, so a scene scaled in metres,
        // centimetres or arbitrary COLMAP units still closes the same kind of sampling holes.
        if (!(resolved.fill_hole_perimeter > 0.f)) {
            resolved.fill_hole_perimeter = kAutoHolePerimeterVoxels * resolved.voxel_size;
            LOG_INFO("Mesh: hole perimeter {:.4f} = {:.0f} x voxel size (auto)", resolved.fill_hole_perimeter,
                     kAutoHolePerimeterVoxels);
        }

        const auto stream = core::getCurrentCUDAStream();
        SplatMesh mesh;
        mesh.views = views.size();
        CameraSizes sizes;
        for (auto* camera : cameras)
            if (camera)
                sizes.saved.push_back({camera, camera->image_width(), camera->image_height(), camera->image_size_loaded()});

        if (settings.from_mesh.empty()) {
            extract_surface(mesh, model, views, resolved, stream, stop, progress, appearance, geometry_share);
        } else {
            core::reportProgress(stop, progress, 0.f, stage("mesh.stage_read", "Reading mesh"));
            auto read = read_splat_mesh_ply(core::utf8_to_path(settings.from_mesh));
            read.views = mesh.views;
            mesh = std::move(read);
            LOG_INFO("Mesh: {} vertices and {} faces from {}", mesh.vertices.size(), mesh.faces.size(), settings.from_mesh);
        }
        auto step = Clock::now();
        if (settings.texel_size > 0.f) {
            texture_ram_preflight(mesh, settings);
            step = Clock::now();
            core::reportProgress(stop, progress, .5f, stage("mesh.stage_unwrap", "Unwrapping"));
            mesh.tiled = std::make_shared<TiledMesh>(build_tiled_mesh(mesh, settings.texel_size, std::min(settings.texture_size, kOsgbMaxPage), kOsgbMaxLevels, settings.simplify_error, stop));
            mesh.seconds_unwrap = seconds_since(step);
            step = Clock::now();
            std::vector<core::Camera*> all_views;
            for (auto* camera : cameras)
                if (camera)
                    all_views.push_back(camera);
            // The levels of a tile stay together, and tiles come in the order of the splits: neighbors in the list are neighbors in space.
            std::vector<std::pair<size_t, size_t>> page_groups;
            for (const auto& level : mesh.tiled->levels) {
                const size_t last = level.first_page + level.page_count;
                if (level.level == 0 || page_groups.empty())
                    page_groups.push_back({level.first_page, last});
                else
                    page_groups.back().second = last;
            }
            bake_texture(mesh.tiled->combined, model, all_views, settings, stream, [&](const float value) { core::reportProgress(stop, progress, .6f + .4f * value, stage("mesh.stage_bake", "Baking texture")); }, appearance, page_groups);
            mesh.seconds_bake = seconds_since(step);
            mesh.charts = mesh.tiled->combined.charts;
            mesh.texels = mesh.tiled->combined.texels;
            mesh.observed_texel_share = mesh.tiled->combined.observed_texel_share;
        }
        return mesh;
    } catch (const std::bad_alloc&) {
        throw collisionError("mesh.ram", "Not enough RAM for the mesh; use a larger voxel size or a maximum range", ErrorCode::ResourceExhausted);
    }

    SplatMesh read_splat_mesh_ply(const std::filesystem::path& path) {
        const auto bad = [&](const std::string_view why) { return collisionError("mesh.read_failed", "Could not read the mesh; it must be a PLY of triangles or polygons", ErrorCode::InvalidArgument, std::format("{}: {}", core::path_to_utf8(path), why)); };
        std::ifstream in(core::native_fs::native_path(path), std::ios::binary);
        if (!in)
            throw collisionError("mesh.read_failed", "Could not open the mesh", ErrorCode::NotFound, core::path_to_utf8(path));
        struct Property {
            std::string name, type, count_type; // count_type is set for a list
        };
        struct Element {
            std::string name;
            size_t count = 0;
            std::vector<Property> properties;
        };
        std::vector<Element> elements;
        bool ascii = false, started = false, ended = false;
        for (std::string line; !ended && std::getline(in, line);) {
            if (!line.empty() && line.back() == '\r')
                line.pop_back();
            std::istringstream words(line);
            std::string word;
            words >> word;
            if (!started) {
                if (word != "ply")
                    throw bad("no ply header");
                started = true;
            } else if (word == "format") {
                words >> word;
                ascii = word == "ascii";
                if (!ascii && word != "binary_little_endian")
                    throw bad(word);
            } else if (word == "element") {
                Element element;
                words >> element.name >> element.count;
                elements.push_back(std::move(element));
            } else if (word == "property") {
                if (elements.empty())
                    throw bad("a property before any element");
                Property property;
                words >> property.type;
                if (property.type == "list")
                    words >> property.count_type >> property.type;
                words >> property.name;
                elements.back().properties.push_back(std::move(property));
            } else if (word == "end_header") {
                ended = true;
            }
        }
        if (!ended)
            throw bad("no end of header");

        const auto type_size = [&](const std::string_view type) -> size_t {
            if (type == "char" || type == "uchar" || type == "int8" || type == "uint8")
                return 1;
            if (type == "short" || type == "ushort" || type == "int16" || type == "uint16")
                return 2;
            if (type == "int" || type == "uint" || type == "int32" || type == "uint32" || type == "float" || type == "float32")
                return 4;
            if (type == "double" || type == "float64")
                return 8;
            throw bad(type);
        };
        const auto read_number = [&](const std::string_view type) -> double {
            if (ascii) {
                double value = 0.;
                if (!(in >> value))
                    throw bad("the file ends early");
                return value;
            }
            char raw[8];
            if (!in.read(raw, std::streamsize(type_size(type))))
                throw bad("the file ends early");
            const auto as = [&]<typename T>(T value) {
                std::memcpy(&value, raw, sizeof(T));
                return double(value);
            };
            if (type == "char" || type == "int8")
                return as(int8_t{});
            if (type == "uchar" || type == "uint8")
                return as(uint8_t{});
            if (type == "short" || type == "int16")
                return as(int16_t{});
            if (type == "ushort" || type == "uint16")
                return as(uint16_t{});
            if (type == "int" || type == "int32")
                return as(int32_t{});
            if (type == "uint" || type == "uint32")
                return as(uint32_t{});
            if (type == "float" || type == "float32")
                return as(float{});
            return as(double{});
        };

        SplatMesh mesh;
        for (const auto& element : elements) {
            if (element.name == "vertex") {
                const auto find = [&](const std::string_view name) {
                    for (size_t p = 0; p < element.properties.size(); ++p)
                        if (element.properties[p].name == name)
                            return int(p);
                    return -1;
                };
                const int position[3] = {find("x"), find("y"), find("z")};
                const int color[3] = {find("red"), find("green"), find("blue")};
                if (position[0] < 0 || position[1] < 0 || position[2] < 0)
                    throw bad("vertices without x, y, z");
                const bool has_color = color[0] >= 0 && color[1] >= 0 && color[2] >= 0;
                mesh.vertices.resize(element.count);
                mesh.colors.assign(element.count, {128, 128, 128});
                std::vector<double> values(element.properties.size());
                for (size_t v = 0; v < element.count; ++v) {
                    for (size_t p = 0; p < values.size(); ++p) {
                        if (!element.properties[p].count_type.empty())
                            throw bad("a list property on vertices");
                        values[p] = read_number(element.properties[p].type);
                    }
                    for (int c = 0; c < 3; ++c) {
                        mesh.vertices[v][c] = float(values[size_t(position[c])]);
                        if (has_color)
                            mesh.colors[v][c] = static_cast<uint8_t>(std::clamp(values[size_t(color[c])], 0., 255.));
                    }
                }
            } else {
                const bool faces = element.name == "face";
                if (faces)
                    mesh.faces.reserve(element.count);
                std::vector<int32_t> corners;
                for (size_t e = 0; e < element.count; ++e)
                    for (const auto& property : element.properties) {
                        if (property.count_type.empty()) {
                            read_number(property.type);
                            continue;
                        }
                        const auto count = size_t(read_number(property.count_type));
                        const bool indices = faces && (property.name == "vertex_indices" || property.name == "vertex_index");
                        corners.resize(count);
                        for (size_t c = 0; c < count; ++c) {
                            const double value = read_number(property.type);
                            corners[c] = int32_t(value);
                        }
                        if (!indices)
                            continue;
                        // A polygon as a fan.
                        for (size_t c = 2; c < count; ++c)
                            mesh.faces.push_back({corners[0], corners[c - 1], corners[c]});
                    }
            }
        }
        for (const auto& face : mesh.faces)
            for (const auto vertex : face)
                if (vertex < 0 || size_t(vertex) >= mesh.vertices.size())
                    throw bad("a face refers to a vertex that does not exist");
        if (mesh.faces.empty())
            throw bad("no faces");
        compute_splat_mesh_normals(mesh);
        return mesh;
    }

    void write_splat_mesh_ply(const SplatMesh& mesh, const std::filesystem::path& path) {
        if (!mesh.uv_source.empty()) {
            const auto welded = weld_chart_seams(mesh);
            SplatMesh surface;
            surface.vertices.reserve(welded.originals.size());
            surface.normals.reserve(welded.originals.size());
            surface.colors.reserve(welded.originals.size());
            for (const auto original : welded.originals) {
                surface.vertices.push_back(mesh.vertices[original]);
                surface.normals.push_back(mesh.normals[original]);
                surface.colors.push_back(mesh.colors[original]);
            }
            surface.faces.reserve(mesh.faces.size());
            for (const auto& face : mesh.faces)
                surface.faces.push_back({welded.index[face[0]], welded.index[face[1]], welded.index[face[2]]});
            write_splat_mesh_ply(surface, path);
            return;
        }
        const auto temporary = std::filesystem::path(path).concat(".tmp");
        {
            std::ofstream out(core::native_fs::native_path(temporary), std::ios::binary | std::ios::trunc);
            out << std::format("ply\nformat binary_little_endian 1.0\nelement vertex {}\n"
                               "property float x\nproperty float y\nproperty float z\n"
                               "property float nx\nproperty float ny\nproperty float nz\n"
                               "property uchar red\nproperty uchar green\nproperty uchar blue\n"
                               "element face {}\nproperty list uchar int vertex_indices\nend_header\n",
                               mesh.vertices.size(), mesh.faces.size());
            static_assert(std::endian::native == std::endian::little);
            std::vector<char> buffer;
            buffer.reserve(size_t{1} << 24);
            const auto flush = [&] {
                out.write(buffer.data(), std::streamsize(buffer.size()));
                buffer.clear();
            };
            const auto append = [&](const void* data, const size_t bytes) {
                const auto* begin = static_cast<const char*>(data);
                buffer.insert(buffer.end(), begin, begin + bytes);
                if (buffer.size() >= (size_t{1} << 24) - 64)
                    flush();
            };
            // A missing attribute means the arrays were dropped along the way, which must not turn
            // into an out of bounds read on the last step before the file is written.
            const std::array<float, 3> no_normal{0.f, 0.f, 0.f};
            const std::array<uint8_t, 3> no_color{128, 128, 128};
            const bool has_normals = mesh.normals.size() == mesh.vertices.size();
            const bool has_colors = mesh.colors.size() == mesh.vertices.size();
            for (size_t v = 0; v < mesh.vertices.size(); ++v) {
                append(mesh.vertices[v].data(), 12);
                append((has_normals ? mesh.normals[v] : no_normal).data(), 12);
                append((has_colors ? mesh.colors[v] : no_color).data(), 3);
            }
            const uint8_t corners = 3;
            for (const auto& f : mesh.faces) {
                append(&corners, 1);
                append(f.data(), 12);
            }
            flush();
            out.close();
            if (!out)
                throw collisionError("mesh.write_failed", "Could not write the mesh; check disk space and permissions", ErrorCode::Unavailable,
                                     core::path_to_utf8(path));
        }
        std::error_code error;
        core::native_fs::rename(temporary, path, error);
        if (error)
            throw collisionError("mesh.write_failed", error.message(), ErrorCode::Unavailable, core::path_to_utf8(path));
    }

    core::MeshData splat_mesh_to_mesh_data(const SplatMesh& mesh) {
        const size_t v = mesh.vertices.size(), f = mesh.faces.size();
        auto vertices = Tensor::empty({v, size_t{3}}, Device::CPU);
        auto normals = Tensor::empty({v, size_t{3}}, Device::CPU);
        auto colors = Tensor::empty({v, size_t{4}}, Device::CPU);
        auto indices = Tensor::empty({f, size_t{3}}, Device::CPU, core::DataType::Int32);
        const bool has_normals = mesh.normals.size() == v;
        const bool has_colors = mesh.colors.size() == v;
        std::memcpy(vertices.ptr<float>(), mesh.vertices.data(), v * 12);
        if (has_normals)
            std::memcpy(normals.ptr<float>(), mesh.normals.data(), v * 12);
        else
            std::fill_n(normals.ptr<float>(), v * 3, 0.f);
        std::memcpy(indices.ptr<int32_t>(), mesh.faces.data(), f * 12);
        auto* out = colors.ptr<float>();
        const auto linear = [](const uint8_t value) {
            const float s = float(value) / 255.f;
            return s <= .04045f ? s / 12.92f : std::pow((s + .055f) / 1.055f, 2.4f);
        };
        for (size_t i = 0; i < v; ++i) {
            for (int c = 0; c < 3; ++c)
                out[4 * i + c] = linear(has_colors ? mesh.colors[i][c] : uint8_t{128});
            out[4 * i + 3] = 1.f;
        }
        core::MeshData data(std::move(vertices), std::move(indices));
        data.normals = std::move(normals);
        if (mesh.pages.empty()) {
            data.colors = std::move(colors);
            return data;
        }
        auto texcoords = Tensor::empty({v, size_t{2}}, Device::CPU);
        std::memcpy(texcoords.ptr<float>(), mesh.uvs.data(), v * 8);
        data.texcoords = std::move(texcoords);
        // The viewport treats texture values as linear, as it does vertex colors.
        std::array<uint8_t, 256> to_linear{};
        for (int i = 0; i < 256; ++i)
            to_linear[size_t(i)] = static_cast<uint8_t>(std::lround(255.f * linear(uint8_t(i))));
        for (size_t p = 0; p < mesh.pages.size(); ++p) {
            core::TextureImage image;
            image.width = image.height = mesh.page_size;
            image.channels = 3;
            image.pixels.resize(mesh.pages[p].size());
            std::ranges::transform(mesh.pages[p], image.pixels.begin(), [&](const uint8_t value) { return to_linear[value]; });
            data.texture_images.push_back(std::move(image));
            core::Material material;
            material.name = std::format("page_{}", p);
            material.albedo_tex = static_cast<uint32_t>(p + 1);
            data.materials.push_back(std::move(material));
            data.submeshes.push_back({3 * mesh.page_first_face[p], 3 * (mesh.page_first_face[p + 1] - mesh.page_first_face[p]), p});
        }
        return data;
    }

} // namespace lfs::training
