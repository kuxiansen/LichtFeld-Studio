/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "splat_mesh.hpp"
#include "splat_mesh_pack.hpp"
#include <cstdint>
#include <span>
#include <vector>

namespace lfs::training {
    // A view's photo and its median-depth render, on the CPU: the photo in float planes of 0..1 sRGB,
    // channel-major (c * width * height + y * width + x), and the depth as the range from the camera,
    // 0 where the render saw nothing.
    struct PatchBakePhoto {
        const float* photo = nullptr;
        const float* depth = nullptr;
        int width = 0, height = 0;
    };

    // One patch's rect on its page: the texels it covers, and which of them its view could vouch for.
    struct PatchBakeRect {
        int width = 0, height = 0;
        std::vector<uint8_t> rgb;  // 3 per texel, sRGB as in the photos
        std::vector<uint8_t> valid; // 1 where the view's own depth render says the photo's pixel is the surface
    };

    // Paints one patch's rect from its view's photo. A copied patch takes the photo's own pixels texel for
    // texel; a resampled one walks the rect's texels over the patch's faces, and takes each from where its
    // point of the surface truly projects in the photo, so that the fisheye's curve never reaches the page.
    // A texel the depth render cannot vouch for — an occluder's, or one the photo never saw — stays invalid
    // for the fill that follows. face_error, per face of the mesh, widens the depth band a face is accepted
    // within, as the atlas path's tolerance does; 0 when the mesh carries none.
    [[nodiscard]] PatchBakeRect bake_patch_texture(const SplatMesh& mesh, const MeshPatch& patch, const PatchPlacement& place,
                                                   const PatchView& view, const PatchBakePhoto& photo,
                                                   const float* face_error, float base_tolerance);

    // Fills a rect's invalid texels: outward from the valid ones, so that filtering reaches no neighbor, and
    // what the rings do not reach takes the patch's own vertex colors.
    void fill_patch_rect(PatchBakeRect& rect, const SplatMesh& mesh, const MeshPatch& patch);

    // Writes a filled rect into its page, at its placement.
    void apply_patch_texture(uint8_t* page, int side, const PatchPlacement& place, const PatchBakeRect& rect);

    // Reads the colors the views give at the seam samples planned for the mesh: the photo blurred around the pixel
    // each sample projects to, with the pixels whose depth lies on the same surface, so that grain and
    // misregistration do not read as an exposure step. sample_indices lists the samples whose label is label;
    // colors holds 3 per sample, 0..1 sRGB, and keeps -1 where the view does not vouch for the sample.
    void sample_patch_seam_colors(const ColorLeveling& plan, std::span<const uint32_t> sample_indices, int32_t label,
                                  const PatchView& view, const PatchBakePhoto& photo, const float* face_error, float base_tolerance,
                                  std::span<float> colors);

    // OpenMVS's seam leveling, the second of the two the setting switches between: one gain per view, from the
    // steps between views at the seam samples, and then an offset per corner solved only within band_pixels of a
    // seam and held at zero outside it. Each view keeps its own colors away from the seams, the two sides meet
    // across the band, and the band itself is what OpenMVS's local Poisson solves for. Applied to the baked pages
    // as apply_patch_gain then apply_patch_offsets; the atlas path's plan/solve keeps the other mode.
    struct PatchSeamLeveling {
        static constexpr uint32_t kNoOffset = ~uint32_t{0};
        std::vector<uint32_t> corner_unknowns; // 3 per face, kNoOffset away from the seams
        std::vector<float> offsets;            // 3 per unknown, 0..1
        std::vector<float> gains;              // 3 per view label
        size_t band_faces = 0;
        size_t iterations = 0;
    };
    [[nodiscard]] PatchSeamLeveling solve_patch_seam_leveling(const SplatMesh& mesh, std::span<const int32_t> face_labels,
                                                              std::span<const PatchView> views, const ColorLeveling& plan,
                                                              std::span<const float> colors, int band_pixels = 16);

    // Scales the texels of a patch's rect by its view's gain: the global half of OpenMVS's seam leveling.
    void apply_patch_gain(uint8_t* page, int side, const PatchPlacement& place, const float* gain);

    // Adds the leveling offsets to a patch's texels, interpolated over each face's corners as the atlas path does.
    // Corners with no unknown (at or beyond the band's edge) contribute nothing. Runs after the page is baked and
    // before the mesh is laid out around it.
    void apply_patch_offsets(uint8_t* page, int side, const SplatMesh& mesh, const MeshPatch& patch, const PatchPlacement& place,
                             const PatchView& view, std::span<const uint32_t> corner_unknowns, std::span<const float> offsets);

    // A page's patches leave gaps between their rects: filled outward from them for this many rings, so that
    // bilinear samples and reductions reach no background, and what the rings do not reach keeps the page's
    // own fill. The color pages need no rings: their cells are painted whole.
    void finish_patch_page(uint8_t* page, int side, const PatchPage& page_patches, std::span<const PatchPlacement> placements,
                           int rings);

    // Rebuilds the mesh around the patch atlas: faces in page order, a vertex split wherever a patch border
    // or a color page cuts one, each new vertex carrying the UV its patch gives it, and the faces no patch
    // covers on pages of their own, painted with the vertex colors. mesh.page_size and mesh.pages must be
    // set first: color_pages of them follow the atlas pages, and their cells are painted here.
    void apply_patch_layout(SplatMesh& mesh, const PatchSet& patches, const PatchAtlas& atlas, std::span<const PatchView> views,
                            int color_pages);
} // namespace lfs::training
