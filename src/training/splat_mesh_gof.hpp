/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/progress.hpp"
#include "splat_mesh.hpp"
#include <cstdint>
#include <span>
#include <stop_token>

namespace lfs::core {
    class Camera;
    class SplatData;
} // namespace lfs::core

namespace lfs::training {

    // The candidate points are the voxels of the splat centers, dilated by one voxel, and the field
    // is the multi-view minimum of the occlusion in front of each of them, as in Gaussian Opacity
    // Fields. Distances are in scene units.
    struct GofMeshSettings {
        float voxel_size = .02f; // candidate and mesh voxel
        float min_opacity = .1f; // splats below this never seed a voxel
        int view_stride = 2;     // use every Nth view for the field
        int color_stride = 2;    // color from every Nth fused view
        int min_component_faces = 100;
        int max_width = 640; // alpha field and photo width limit
    };

    // Extracts the surface of the opacity field with naive surface nets and colors it from the
    // photos. Mesh is in model space; camera image sizes are restored afterwards.
    [[nodiscard]] SplatMesh extract_splat_mesh_gof(core::SplatData& model,
                                                   std::span<core::Camera* const> cameras,
                                                   const GofMeshSettings& settings,
                                                   std::stop_token stop = {},
                                                   const core::ProgressCallback& progress = {});

} // namespace lfs::training
