/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "Common.h"
#include "TsdfFusion.h"
#include <cstdint>

namespace gsplat_lfs {

    // Projects the splats once for a view. radii, means2d, depths and conics must be zeroed by
    // the caller, because splats that are skipped leave them untouched. K is the 3x3 intrinsics
    // on the device, row-major.
    void gof_project_view(const float* means, const float* quats, const float* scales,
                          const float* opacities, const float* K, uint32_t count,
                          const TsdfView& view, int32_t* radii, float* means2d, float* depths,
                          float* conics, cudaStream_t stream);

    // Fragments per pixel; clears counts, accumulates total and returns the fragment count.
    [[nodiscard]] uint64_t gof_count_fragments(const int32_t* radii, const float* means2d,
                                               const float* conics, const float* opacities,
                                               uint32_t count, const TsdfView& view,
                                               float min_alpha, uint32_t* counts, uint64_t* total,
                                               cudaStream_t stream);

    // starts[pixel] is the first slot of that pixel; one entry per pixel.
    void gof_scan_counts(const uint32_t* counts, uint32_t* starts, uint32_t pixels,
                         cudaStream_t stream);

    // Writes the fragments into the slots cursor hands out; the caller seeds cursor from starts.
    void gof_fill_fragments(const int32_t* radii, const float* means2d, const float* depths,
                            const float* conics, const float* opacities, uint32_t count,
                            const TsdfView& view, float min_alpha, uint32_t* cursor,
                            uint32_t* pixels_out, float* depth_out, float* log_t_out,
                            cudaStream_t stream);

    // Sorts pixel-major by depth. keys, keys_sorted, values and values_sorted are scratch of
    // the fragment count; the three outputs hold the pixel-major stream.
    void gof_sort_fragments(const uint32_t* pixels, const float* log_t, const float* depth,
                            uint64_t fragments, uint64_t* keys, uint64_t* keys_sorted,
                            uint32_t* values, uint32_t* values_sorted, uint32_t* pixels_out,
                            float* depth_out, float* log_t_out, cudaStream_t stream);

    // Turns log_t into its running sum within each pixel segment, in place.
    void gof_prefix_fragments(const uint32_t* starts, const uint32_t* counts, float* log_t,
                              uint32_t pixels, cudaStream_t stream);

    // Multi-view minimum of the occlusion in front of each voxel center. Fills occlusion with
    // 1 before the first view, GOF's neutral value, with gof_reset_occlusion.
    void gof_query_points(const uint64_t* keys, uint32_t point_count, float voxel,
                          const TsdfView& view, const uint32_t* starts, const uint32_t* counts,
                          const float* depth, const float* cum, float* occlusion,
                          cudaStream_t stream);

    void gof_reset_occlusion(float* occlusion, uint32_t count, cudaStream_t stream);

    // Sets visible[i] for the splats whose center this view sees.
    void gof_mark_visible(const float* means, uint32_t count, const TsdfView& view,
                          uint8_t* visible, cudaStream_t stream);

    // One key per kept splat; keys has room for count and counter starts at zero.
    [[nodiscard]] uint64_t gof_seed_keys(const float* means, const float* opacities,
                                         const uint8_t* visible, uint32_t count, float voxel,
                                         float min_opacity, uint64_t* keys, uint64_t* counter,
                                         cudaStream_t stream);

    // The 27 voxels around each key; out holds 27 * count keys.
    void gof_dilate_keys(const uint64_t* keys, uint32_t count, uint64_t* out,
                         cudaStream_t stream);

} // namespace gsplat_lfs
