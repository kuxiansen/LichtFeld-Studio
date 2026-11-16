/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "splat_mesh_patch.hpp"
#include <cstdint>
#include <span>
#include <stop_token>
#include <vector>

namespace lfs::training {
    // Where a patch's texels go on a page, and what one of its texels covers on the surface.
    struct PatchPlacement {
        int page = -1;                          // into PatchAtlas.pages
        int x = 0, y = 0, width = 0, height = 0; // texels on the page
        int source_x = 0, source_y = 0;          // the photo pixel the copied rect starts at
        float texel = 0.f;                       // scene units per texel
        bool warp = false;                       // false: the photo's pixels are copied texel for texel
    };

    // Patches of one density, packed onto a page of their own.
    struct PatchPage {
        int size = 0;                // texels along each side
        std::vector<size_t> patches; // into PatchAtlas.placements
    };

    struct PatchAtlas {
        std::vector<PatchPlacement> placements; // per patch, in the order the patches were built
        std::vector<PatchPage> pages;
        size_t texels = 0;                 // texels the patches cover, padding not counted
        size_t copies = 0, resampled = 0;  // patches copied from the photo or rendered through a camera of their own
        size_t shrunk = 0;                 // patches whose rect was cut down to fit a page
        float texel_copied = 0.f;          // median, scene units
        float texel_resampled = 0.f;       // median, scene units
        double seconds = 0.;
    };

    struct PatchPackSettings {
        int page_size = 4096;   // texels along each side, at most 8192
        int padding = 4;        // texels kept clear around each patch
        float texel_size = 0.f; // the texel the surface asks for, scene units; 0 keeps the photos' own density
        float texel_min = 0.f;  // the finest texel allowed, scene units; 0 means 1 mm
    };

    // Gives every patch its rect on a page. A patch whose fit held on is copied texel for texel from the photo's pixels
    // at the density those pixels have; one that had to be resampled is laid out at the requested texel, within
    // [texel_min, texel_size], and a patch whose rect would outgrow a page is laid out coarser until it fits. Patches
    // whose densities are within a factor of two share pages, packed by MaxRects, so that a page can be filtered and
    // reduced as one. Faces no patch covers keep the vertex colors.
    [[nodiscard]] PatchAtlas pack_mesh_patches(const SplatMesh& mesh, const PatchSet& patches, std::span<const PatchView> views,
                                               const PatchPackSettings& settings, std::stop_token stop = {});
} // namespace lfs::training
