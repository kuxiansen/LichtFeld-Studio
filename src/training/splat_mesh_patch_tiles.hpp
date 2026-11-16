/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "splat_mesh.hpp"
#include "splat_mesh_bake.hpp"
#include "splat_mesh_pack.hpp"
#include "splat_mesh_patch.hpp"
#include <array>
#include <cstdint>
#include <functional>
#include <span>
#include <stop_token>
#include <vector>

namespace lfs::training {
    // One tile level to texture with affine patches: its geometry, and the patch view each of its faces reads.
    struct PatchTiledLevel {
        SplatMesh mesh;
        std::vector<int32_t> face_views; // per face, into views; -1 where the vertex colors stand in
        float texel = 0.f;               // the texel the level asks for, scene units
        size_t tile = 0, level = 0;
        std::array<float, 3> center{0.f, 0.f, 0.f};
        float radius = 0.f;
    };

    struct PatchTiledSettings {
        int page_size = 4096;
        int padding = 4;
        float texel_min = 0.f;
        float base_tolerance = 0.f;
        int seam_leveling_mode = 0;
    };

    // A view's photo and its depth render for the patches that read it. The buffers the photo points at stay
    // valid until the next call; false skips the view, as when its photo does not match the camera the patches
    // were built against.
    using PatchPhotoSource = std::function<bool(size_t view, PatchBakePhoto& photo)>;

    struct PatchTiledPages {
        size_t pages = 0;
        size_t charts = 0;
        size_t texels = 0;
        float observed_texel_share = 0.f;
        float texel = 0.f; // the level's texels averaged over its patches, for the LOD's full-detail distance
    };

    // Textures the levels a photo at a time, so that each is read once for every level that needs it, then levels
    // each level's seams and lays it out around its pages. The meshes are left in page order.
    [[nodiscard]] std::vector<PatchTiledPages> bake_patch_tiled_levels(std::vector<PatchTiledLevel>& levels,
                                                                      std::span<const PatchView> views, const PatchTiledSettings& settings,
                                                                      const PatchPhotoSource& photo_source,
                                                                      const std::function<void(float)>& report, std::stop_token stop);

    // The levels as one mesh, faces in the levels' order, so that one pass over the photos can choose every
    // level's views before they are textured, and the labels split back by face count.
    [[nodiscard]] SplatMesh combine_patch_tiled_levels(const std::vector<PatchTiledLevel>& levels);

    // Every level as one mesh, the finest levels first, so that the OSGB model and the viewer's base mesh share
    // one set of pages: the base mesh's page p is base.tiled->combined.pages[p]. Keeps the base mesh's other
    // fields (its stats), sets its tiled model, and leaves the levels empty to free their geometry.
    void assemble_patch_tiled_mesh(std::vector<PatchTiledLevel>& levels, std::span<const PatchTiledPages> textures,
                                   SplatMesh& base);

} // namespace lfs::training
