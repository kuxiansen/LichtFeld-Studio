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
