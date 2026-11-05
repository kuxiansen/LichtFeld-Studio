/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "GofFusion.h"
#include "Cameras.cuh"
#include "Projection.h"
#include "core/cuda_error.hpp"
#include "core/tensor/internal/cub_workspace.hpp"
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_reduce.cuh>
#include <cub/device/device_scan.cuh>

namespace gsplat_lfs {
    namespace {
        constexpr uint32_t kThreads = 256;
        constexpr float kReadoutMargin = 0.1f;
        constexpr float kMinRange = 0.05f;
        constexpr float kMaxAlpha = 0.9999f;
        constexpr float kEps2d = 1e-4f; // the offline GOF tool's covariance floor

        __device__ __forceinline__ float sigmoid(const float raw) {
            return 1.f / (1.f + __expf(-raw));
        }

        template <class Body>
        __device__ __forceinline__ void with_model(const TsdfView& v, Body&& body) {
            if (v.model == CameraModelType::FISHEYE) {
                OpenCVFisheyeCameraModel<>::Parameters p = {};
                p.resolution = {v.width, v.height};
                p.shutter_type = v.shutter;
                p.principal_point = {v.cx, v.cy};
                p.focal_length = {v.fx, v.fy};
                p.radial_coeffs = {v.radial[0], v.radial[1], v.radial[2], v.radial[3]};
                const OpenCVFisheyeCameraModel<> model(p);
                body(model);
            } else if (v.model == CameraModelType::THIN_PRISM_FISHEYE) {
                ThinPrismFisheyeCameraModel<>::Parameters p = {};
                p.resolution = {v.width, v.height};
                p.shutter_type = v.shutter;
                p.principal_point = {v.cx, v.cy};
                p.focal_length = {v.fx, v.fy};
                p.radial_coeffs = {v.radial[0], v.radial[1], v.radial[2], v.radial[3]};
                p.thin_prism_coeffs = {v.thin_prism[0], v.thin_prism[1], v.thin_prism[2], v.thin_prism[3]};
                const ThinPrismFisheyeCameraModel<> model(p);
                body(model);
            } else if (v.model == CameraModelType::EQUIRECTANGULAR) {
                EquirectangularCameraModel::Parameters p = {};
                p.resolution = {v.width, v.height};
                p.shutter_type = ShutterType::GLOBAL;
                const EquirectangularCameraModel model(p);
                body(model);
            } else if (v.has_distortion) {
                OpenCVPinholeCameraModel<>::Parameters p = {};
                p.resolution = {v.width, v.height};
                p.shutter_type = v.shutter;
                p.principal_point = {v.cx, v.cy};
                p.focal_length = {v.fx, v.fy};
                p.radial_coeffs = {v.radial[0], v.radial[1], v.radial[2], v.radial[3], v.radial[4], v.radial[5]};
                p.tangential_coeffs = {v.tangential[0], v.tangential[1]};
                p.thin_prism_coeffs = {v.thin_prism[0], v.thin_prism[1], v.thin_prism[2], v.thin_prism[3]};
                const OpenCVPinholeCameraModel<> model(p);
                body(model);
            } else {
                PerfectPinholeCameraModel::Parameters p = {};
                p.resolution = {v.width, v.height};
                p.shutter_type = v.shutter;
                p.principal_point = {v.cx, v.cy};
                p.focal_length = {v.fx, v.fy};
                const PerfectPinholeCameraModel model(p);
                body(model);
            }
        }

        __device__ __forceinline__ RollingShutterParameters readout_of(const TsdfView& v) {
            return v.shutter == ShutterType::GLOBAL ? RollingShutterParameters(v.viewmat, nullptr)
                                                    : RollingShutterParameters(v.readout0, v.readout1);
        }

        // Continuous image coordinates and the range the depth channel holds.
        template <class Model>
        __device__ __forceinline__ bool project_f(const Model& model, const TsdfView& v,
                                                  const RollingShutterParameters& readout,
                                                  const glm::fvec3& world, glm::fvec2& uv, float& range) {
            const float* m = v.viewmat;
            const glm::fvec3 p(m[0] * world.x + m[1] * world.y + m[2] * world.z + m[3],
                               m[4] * world.x + m[5] * world.y + m[6] * world.z + m[7],
                               m[8] * world.x + m[9] * world.y + m[10] * world.z + m[11]);
            range = v.model == CameraModelType::PINHOLE ? p.z : glm::length(p);
            if (!(range > kMinRange))
                return false;
            const auto image = model.template world_point_to_image_point_shutter_pose<>(world, readout, kReadoutMargin);
            if (!image.valid_flag)
                return false;
            uv = image.imagePoint;
            return true;
        }

        // Ascending order for the radix sort, which compares unsigned keys.
        __device__ __forceinline__ uint32_t orderable(const float value) {
            const uint32_t bits = __float_as_uint(value);
            return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
        }

        __device__ __forceinline__ float unorderable(const uint32_t bits) {
            return __uint_as_float((bits & 0x80000000u) ? (bits & 0x7fffffffu) : ~bits);
        }

        __device__ __forceinline__ uint64_t encode_key(const int x, const int y, const int z) {
            return static_cast<uint64_t>(x + kTsdfKeyOffset) |
                   (static_cast<uint64_t>(y + kTsdfKeyOffset) << kTsdfKeyBits) |
                   (static_cast<uint64_t>(z + kTsdfKeyOffset) << (2 * kTsdfKeyBits));
        }

        __device__ __forceinline__ glm::ivec3 decode_key(const uint64_t key) {
            const uint64_t mask = (uint64_t{1} << kTsdfKeyBits) - 1;
            return {static_cast<int>(key & mask) - kTsdfKeyOffset,
                    static_cast<int>((key >> kTsdfKeyBits) & mask) - kTsdfKeyOffset,
                    static_cast<int>((key >> (2 * kTsdfKeyBits)) & mask) - kTsdfKeyOffset};
        }

        struct Bounds {
            int x0, y0, x1, y1;
            bool empty;
        };

        __device__ __forceinline__ Bounds fragment_bounds(const int32_t* radii, const float* means2d,
                                                          const uint32_t i, const TsdfView& view) {
            const int32_t rx = radii[2 * i], ry = radii[2 * i + 1];
            if (rx <= 0 || ry <= 0)
                return {0, 0, 0, 0, true};
            const float mx = means2d[2 * i], my = means2d[2 * i + 1];
            const int x0 = max(0, static_cast<int>(floorf(mx - static_cast<float>(rx))));
            const int x1 = min(static_cast<int>(view.width) - 1, static_cast<int>(floorf(mx + static_cast<float>(rx))));
            const int y0 = max(0, static_cast<int>(floorf(my - static_cast<float>(ry))));
            const int y1 = min(static_cast<int>(view.height) - 1, static_cast<int>(floorf(my + static_cast<float>(ry))));
            return {x0, y0, x1, y1, x0 > x1 || y0 > y1};
        }

        __device__ __forceinline__ float fragment_alpha(const float opacity, const float* conics,
                                                        const float mx, const float my, const int x, const int y) {
            const float du = static_cast<float>(x) + .5f - mx;
            const float dv = static_cast<float>(y) + .5f - my;
            const float mahalanobis = conics[0] * du * du + 2.f * conics[1] * du * dv + conics[2] * dv * dv;
            return opacity * __expf(-.5f * mahalanobis);
        }

        __global__ void count_kernel(const int32_t* radii, const float* means2d, const float* opacities,
                                     const float* conics, const uint32_t count, const TsdfView view,
                                     const float min_alpha, uint32_t* counts, unsigned long long* total) {
            for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += gridDim.x * blockDim.x) {
                const Bounds b = fragment_bounds(radii, means2d, i, view);
                if (b.empty)
                    continue;
                const float opacity = sigmoid(opacities[i]);
                if (!(opacity > min_alpha))
                    continue;
                const float mx = means2d[2 * i], my = means2d[2 * i + 1];
                const float* c = conics + 3 * i;
                for (int y = b.y0; y <= b.y1; ++y) {
                    for (int x = b.x0; x <= b.x1; ++x) {
                        if (fragment_alpha(opacity, c, mx, my, x, y) > min_alpha) {
                            atomicAdd(counts + static_cast<uint32_t>(y) * view.width + x, 1u);
                            atomicAdd(total, 1ull);
                        }
                    }
                }
            }
        }

        __global__ void fill_kernel(const int32_t* radii, const float* means2d, const float* depths,
                                    const float* opacities, const float* conics, const uint32_t count,
                                    const TsdfView view, const float min_alpha, uint32_t* cursor,
                                    uint32_t* pixels_out, float* depth_out, float* log_t_out) {
            for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += gridDim.x * blockDim.x) {
                const Bounds b = fragment_bounds(radii, means2d, i, view);
                if (b.empty)
                    continue;
                const float opacity = sigmoid(opacities[i]);
                if (!(opacity > min_alpha))
                    continue;
                const float mx = means2d[2 * i], my = means2d[2 * i + 1];
                const float* c = conics + 3 * i;
                for (int y = b.y0; y <= b.y1; ++y) {
                    for (int x = b.x0; x <= b.x1; ++x) {
                        const float alpha = fragment_alpha(opacity, c, mx, my, x, y);
                        if (!(alpha > min_alpha))
                            continue;
                        const uint32_t pixel = static_cast<uint32_t>(y) * view.width + x;
                        const uint32_t slot = atomicAdd(cursor + pixel, 1u);
                        pixels_out[slot] = pixel;
                        depth_out[slot] = depths[i];
                        log_t_out[slot] = logf(1.f - fminf(alpha, kMaxAlpha));
                    }
                }
            }
        }

        __global__ void seed_kernel(const float* means, const float* opacities, const uint8_t* visible,
                                    const uint32_t count, const float voxel, const float min_opacity,
                                    uint64_t* keys, unsigned long long* counter) {
            for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += gridDim.x * blockDim.x) {
                if (visible != nullptr && visible[i] == 0)
                    continue;
                if (!(sigmoid(opacities[i]) >= min_opacity))
                    continue;
                const float x = means[3 * i] / voxel, y = means[3 * i + 1] / voxel, z = means[3 * i + 2] / voxel;
                if (!isfinite(x) || !isfinite(y) || !isfinite(z))
                    continue;
                const uint64_t slot = atomicAdd(counter, 1ull);
                keys[slot] = encode_key(static_cast<int>(floorf(x)), static_cast<int>(floorf(y)),
                                        static_cast<int>(floorf(z)));
            }
        }

        __global__ void dilate_kernel(const uint64_t* keys, const uint32_t count, uint64_t* out) {
            for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += gridDim.x * blockDim.x) {
                const glm::ivec3 v = decode_key(keys[i]);
                uint32_t n = 0;
                for (int dx = -1; dx <= 1; ++dx)
                    for (int dy = -1; dy <= 1; ++dy)
                        for (int dz = -1; dz <= 1; ++dz)
                            out[i * 27 + n++] = encode_key(v.x + dx, v.y + dy, v.z + dz);
            }
        }

        __global__ void mark_visible_kernel(const float* means, const uint32_t count, const TsdfView view,
                                            uint8_t* visible) {
            for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += gridDim.x * blockDim.x) {
                if (visible[i] != 0)
                    continue;
                const glm::fvec3 world(means[3 * i], means[3 * i + 1], means[3 * i + 2]);
                const RollingShutterParameters readout = readout_of(view);
                bool seen = false;
                float range = 0.f;
                glm::fvec2 uv{};
                with_model(view, [&](const auto& model) { seen = project_f(model, view, readout, world, uv, range); });
                if (!seen)
                    continue;
                if (uv.x < 0.f || uv.x >= static_cast<float>(view.width) || uv.y < 0.f ||
                    uv.y >= static_cast<float>(view.height))
                    continue;
                visible[i] = 1;
            }
        }

        __global__ void pack_kernel(const uint32_t* pixels, const float* depth, const uint32_t count,
                                    uint64_t* keys, uint32_t* values) {
            for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += gridDim.x * blockDim.x) {
                keys[i] = (static_cast<uint64_t>(pixels[i]) << 32) | orderable(depth[i]);
                values[i] = i;
            }
        }

        __global__ void gather_kernel(const uint64_t* keys, const uint32_t* values, const float* log_t_in,
                                      const uint32_t count, uint32_t* pixels_out, float* depth_out,
                                      float* log_t_out) {
            for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += gridDim.x * blockDim.x) {
                const uint64_t key = keys[i];
                pixels_out[i] = static_cast<uint32_t>(key >> 32);
                depth_out[i] = unorderable(static_cast<uint32_t>(key & 0xffffffffu));
                log_t_out[i] = log_t_in[values[i]];
            }
        }

        __global__ void prefix_kernel(const uint32_t* starts, const uint32_t* counts, float* log_t,
                                      const uint32_t pixels) {
            for (uint32_t pixel = blockIdx.x * blockDim.x + threadIdx.x; pixel < pixels; pixel += gridDim.x * blockDim.x) {
                const uint32_t stop = starts[pixel] + counts[pixel];
                float running = 0.f;
                for (uint32_t i = starts[pixel]; i < stop; ++i) {
                    running += log_t[i];
                    log_t[i] = running;
                }
            }
        }

        // The offline tool's opacity_before: 1 - transmittance of the fragments strictly in front.
        __device__ __forceinline__ float opacity_before(const uint32_t pixel, const float d,
                                                        const uint32_t* starts, const uint32_t* counts,
                                                        const float* depth, const float* cum) {
            const uint32_t first = starts[pixel];
            uint32_t lo = first, hi = first + counts[pixel];
            while (lo < hi) {
                const uint32_t mid = (lo + hi) >> 1;
                if (depth[mid] < d)
                    lo = mid + 1;
                else
                    hi = mid;
            }
            if (lo == first)
                return 0.f;
            return 1.f - __expf(cum[lo - 1]);
        }

        __global__ void query_kernel(const uint64_t* keys, const uint32_t point_count, const float voxel,
                                     const TsdfView view, const uint32_t* starts, const uint32_t* counts,
                                     const float* depth, const float* cum, float* occlusion) {
            for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < point_count; i += gridDim.x * blockDim.x) {
                if (occlusion[i] <= 0.f)
                    continue;
                const glm::ivec3 v = decode_key(keys[i]);
                const glm::fvec3 world(v.x * voxel, v.y * voxel, v.z * voxel);
                const RollingShutterParameters readout = readout_of(view);
                float result = 1.f; // GOF's no-vote for a view that does not see the point
                with_model(view, [&](const auto& model) {
                    glm::fvec2 uv{};
                    float range = 0.f;
                    if (!project_f(model, view, readout, world, uv, range))
                        return;
                    if (uv.x < 0.f || uv.x >= static_cast<float>(view.width) || uv.y < 0.f ||
                        uv.y >= static_cast<float>(view.height))
                        return;
                    const int x0 = static_cast<int>(floorf(uv.x)), y0 = static_cast<int>(floorf(uv.y));
                    const float fx = uv.x - static_cast<float>(x0), fy = uv.y - static_cast<float>(y0);
                    float accumulated = 0.f, weight = 0.f;
                    for (int dy = 0; dy < 2; ++dy) {
                        for (int dx = 0; dx < 2; ++dx) {
                            const float w = (dx ? fx : 1.f - fx) * (dy ? fy : 1.f - fy);
                            const int x = min(max(x0 + dx, 0), static_cast<int>(view.width) - 1);
                            const int y = min(max(y0 + dy, 0), static_cast<int>(view.height) - 1);
                            accumulated += w * opacity_before(static_cast<uint32_t>(y) * view.width + x, range,
                                                              starts, counts, depth, cum);
                            weight += w;
                        }
                    }
                    result = weight > 0.f ? accumulated / fmaxf(weight, 1e-8f) : 1.f;
                });
                occlusion[i] = fminf(occlusion[i], result);
            }
        }

        __global__ void reset_kernel(float* data, const uint32_t count, const float value) {
            for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += gridDim.x * blockDim.x)
                data[i] = value;
        }

        [[nodiscard]] uint32_t launch_blocks(const uint64_t count) {
            return static_cast<uint32_t>((count + kThreads - 1) / kThreads);
        }
    } // namespace

    void gof_project_view(const float* means, const float* quats, const float* scales,
                          const float* opacities, const float* K, const uint32_t count,
                          const TsdfView& view, int32_t* radii, float* means2d, float* depths,
                          float* conics, cudaStream_t stream) {
        if (count == 0)
            return;
        const bool radial = view.model == CameraModelType::FISHEYE ||
                            view.model == CameraModelType::THIN_PRISM_FISHEYE || view.has_distortion;
        const bool tangential = view.model == CameraModelType::PINHOLE && view.has_distortion;
        const bool prism = view.model == CameraModelType::THIN_PRISM_FISHEYE;
        const UnscentedTransformParameters ut_params{};

        // The kernel reads these on the device; the view struct lives on the host. The async
        // frees are stream-ordered behind the launch, so the copies stay valid until it ends.
        auto upload = [&](const float* host, const size_t count) -> float* {
            float* device = nullptr;
            LFS_CUDA_CHECK(cudaMallocAsync(&device, count * sizeof(float), stream));
            LFS_CUDA_CHECK(cudaMemcpyAsync(device, host, count * sizeof(float), cudaMemcpyHostToDevice, stream));
            return device;
        };
        float* viewmat_device = upload(view.viewmat, 16);
        float* radial_device = radial ? upload(view.radial, 6) : nullptr;
        float* tangential_device = tangential ? upload(view.tangential, 2) : nullptr;
        float* prism_device = prism ? upload(view.thin_prism, 4) : nullptr;

        launch_projection_ut_3dgs_fused_kernel(
            means, quats, scales, opacities, viewmat_device, nullptr, K,
            count, 1, view.width, view.height, kEps2d, .01f, 10000.f, 0.f, view.model, ut_params,
            view.shutter, radial_device, tangential_device,
            prism_device, radii, means2d, depths, conics, nullptr, stream);
        LFS_CUDA_LAUNCH_CHECK(stream, "gof.project");

        LFS_CUDA_CHECK(cudaFreeAsync(viewmat_device, stream));
        if (radial_device)
            LFS_CUDA_CHECK(cudaFreeAsync(radial_device, stream));
        if (tangential_device)
            LFS_CUDA_CHECK(cudaFreeAsync(tangential_device, stream));
        if (prism_device)
            LFS_CUDA_CHECK(cudaFreeAsync(prism_device, stream));
    }

    uint64_t gof_count_fragments(const int32_t* radii, const float* means2d, const float* conics,
                                 const float* opacities, const uint32_t count, const TsdfView& view,
                                 const float min_alpha, uint32_t* counts, uint64_t* total,
                                 cudaStream_t stream) {
        LFS_CUDA_CHECK(cudaMemsetAsync(counts, 0, sizeof(uint32_t) * view.width * view.height, stream));
        LFS_CUDA_CHECK(cudaMemsetAsync(total, 0, sizeof(uint64_t), stream));
        if (count > 0)
            count_kernel<<<launch_blocks(count), kThreads, 0, stream>>>(
                radii, means2d, opacities, conics, count, view, min_alpha, counts,
                reinterpret_cast<unsigned long long*>(total));
        LFS_CUDA_LAUNCH_CHECK(stream, "gof.count");
        uint64_t fragments = 0;
        LFS_CUDA_CHECK(cudaMemcpyAsync(&fragments, total, sizeof(uint64_t), cudaMemcpyDeviceToHost, stream));
        LFS_CUDA_CHECK(cudaStreamSynchronize(stream));
        return fragments;
    }

    void gof_scan_counts(const uint32_t* counts, uint32_t* starts, const uint32_t pixels, cudaStream_t stream) {
        if (pixels == 0)
            return;
        lfs::core::tensor_ops::run_cub_operation("gof.scan", stream, [&](void* workspace, size_t& bytes) {
            return cub::DeviceScan::ExclusiveSum(workspace, bytes, counts, starts, pixels, stream);
        });
    }

    void gof_fill_fragments(const int32_t* radii, const float* means2d, const float* depths,
                            const float* conics, const float* opacities, const uint32_t count,
                            const TsdfView& view, const float min_alpha, uint32_t* cursor,
                            uint32_t* pixels_out, float* depth_out, float* log_t_out, cudaStream_t stream) {
        if (count == 0)
            return;
        fill_kernel<<<launch_blocks(count), kThreads, 0, stream>>>(radii, means2d, depths, opacities, conics, count,
                                                                view, min_alpha, cursor, pixels_out, depth_out,
                                                                log_t_out);
        LFS_CUDA_LAUNCH_CHECK(stream, "gof.fill");
    }

    void gof_sort_fragments(const uint32_t* pixels, const float* log_t, const float* depth,
                            const uint64_t fragments, uint64_t* keys, uint64_t* keys_sorted,
                            uint32_t* values, uint32_t* values_sorted, uint32_t* pixels_out,
                            float* depth_out, float* log_t_out, cudaStream_t stream) {
        if (fragments == 0)
            return;
        const uint32_t count = static_cast<uint32_t>(fragments);
        pack_kernel<<<launch_blocks(fragments), kThreads, 0, stream>>>(pixels, depth, count, keys, values);
        LFS_CUDA_LAUNCH_CHECK(stream, "gof.pack");
        lfs::core::tensor_ops::run_cub_operation("gof.sort", stream, [&](void* workspace, size_t& bytes) {
            return cub::DeviceRadixSort::SortPairs(workspace, bytes, keys, keys_sorted, values, values_sorted,
                                                   count, 0, 64, stream);
        });
        gather_kernel<<<launch_blocks(fragments), kThreads, 0, stream>>>(keys_sorted, values_sorted, log_t, count,
                                                                     pixels_out, depth_out, log_t_out);
        LFS_CUDA_LAUNCH_CHECK(stream, "gof.gather");
    }

    void gof_prefix_fragments(const uint32_t* starts, const uint32_t* counts, float* log_t,
                              const uint32_t pixels, cudaStream_t stream) {
        if (pixels == 0)
            return;
        prefix_kernel<<<launch_blocks(pixels), kThreads, 0, stream>>>(starts, counts, log_t, pixels);
        LFS_CUDA_LAUNCH_CHECK(stream, "gof.prefix");
    }

    void gof_query_points(const uint64_t* keys, const uint32_t point_count, const float voxel,
                          const TsdfView& view, const uint32_t* starts, const uint32_t* counts,
                          const float* depth, const float* cum, float* occlusion,
                          cudaStream_t stream) {
        if (point_count == 0)
            return;
        query_kernel<<<launch_blocks(point_count), kThreads, 0, stream>>>(keys, point_count, voxel, view, starts,
                                                                      counts, depth, cum, occlusion);
        LFS_CUDA_LAUNCH_CHECK(stream, "gof.query");
    }

    void gof_reset_occlusion(float* occlusion, const uint32_t count, cudaStream_t stream) {
        if (count == 0)
            return;
        reset_kernel<<<launch_blocks(count), kThreads, 0, stream>>>(occlusion, count, 1.f);
        LFS_CUDA_LAUNCH_CHECK(stream, "gof.reset");
    }

    void gof_mark_visible(const float* means, const uint32_t count, const TsdfView& view,
                          uint8_t* visible, cudaStream_t stream) {
        if (count == 0)
            return;
        mark_visible_kernel<<<launch_blocks(count), kThreads, 0, stream>>>(means, count, view, visible);
        LFS_CUDA_LAUNCH_CHECK(stream, "gof.visible");
    }

    uint64_t gof_seed_keys(const float* means, const float* opacities, const uint8_t* visible,
                           const uint32_t count, const float voxel, const float min_opacity,
                           uint64_t* keys, uint64_t* counter, cudaStream_t stream) {
        LFS_CUDA_CHECK(cudaMemsetAsync(counter, 0, sizeof(uint64_t), stream));
        if (count > 0)
            seed_kernel<<<launch_blocks(count), kThreads, 0, stream>>>(
                means, opacities, visible, count, voxel, min_opacity, keys,
                reinterpret_cast<unsigned long long*>(counter));
        LFS_CUDA_LAUNCH_CHECK(stream, "gof.seed");
        uint64_t seeded = 0;
        LFS_CUDA_CHECK(cudaMemcpyAsync(&seeded, counter, sizeof(uint64_t), cudaMemcpyDeviceToHost, stream));
        LFS_CUDA_CHECK(cudaStreamSynchronize(stream));
        return seeded;
    }

    void gof_dilate_keys(const uint64_t* keys, const uint32_t count, uint64_t* out, cudaStream_t stream) {
        if (count == 0)
            return;
        dilate_kernel<<<launch_blocks(count), kThreads, 0, stream>>>(keys, count, out);
        LFS_CUDA_LAUNCH_CHECK(stream, "gof.dilate");
    }

} // namespace gsplat_lfs
