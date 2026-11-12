/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include <algorithm>
#include <string>

namespace lfs::core {

    // Distances are in scene units. The defaults are tuned on metric indoor scans.
    struct SplatMeshSettings {
        float voxel_size = .0f; // 0 derives the voxel as 0.75x the model's median splat spacing
        int block_voxels = 8;
        float truncation = 0.f;        // band around the surface; narrower than the depth spread opens holes. 0 = automatic
        // Widens the band where neighbouring pixels of a depth map disagree: band = max(trunc,
        // trunc_gradient * local depth step), capped at trunc_cap. 0 keeps trunc everywhere.
        float trunc_gradient = 1.5f;
        float trunc_cap = 0.f;         // 0 derives the cap as 5x the truncation, so it scales with the voxel
        float full_weight_range = 1.f; // views beyond this range count less, by 1/r^2; 0 = equal weights
        int view_stride = 2;           // fuse every Nth view
        int color_stride = 2;          // vertex colors from every Nth fused view; a texture uses every photo
        int min_views = 2;             // views a voxel needs to count as observed
        // Closes holes whose rim is shorter than this, in scene units. 0 derives it as 20x the
        // voxel, so the threshold follows the scene's detail instead of one hand-tuned number.
        float fill_hole_perimeter = .0f;
        // OpenMVS's adaptive outlier pass, split in two. The first drops faces with any edge longer
        // than the mesh's 95th-percentile edge times the factor; the second drops connected pieces
        // whose bounding-box diagonal is shorter than the 55th-percentile edge times the factor.
        // 0 turns a phase off.
        float spurious_edge_factor = 30.f;
        float spurious_component_factor = 10.f;
        // OpenMVS fills a hole by ear clipping when its boundary has fewer edges than this, keeping
        // the rim vertices and concave rims; larger holes use the fan when they fall under
        // fill_hole_perimeter. 0 keeps the fan for every hole.
        int ear_cut_holes = 50;
        float simplify_error = .0f;    // surface deviation allowed when reducing faces; 0 = 0.9 x voxel (auto)
        // Drops zero-area and repeated faces, merges vertices at the same position, drops faces that
        // share an edge with more than one other face and splits vertices whose faces form more than
        // one fan. Cheap, and a no-op on a clean mesh. Runs after simplification, and again after the
        // first hole fill so any topology the fill introduced is repaired before the second pass.
        bool clean_topology = true;
        // After the second hole fill, runs a lightweight clean that only drops faces sharing an
        // edge with three others and isolated vertices. It never reopens closed holes.
        bool clean_non_manifold = true;
        bool remove_spikes = true;     // drops vertices that hang off a single face, repeatedly
        // After every topology pass and hole fill, drops connected components that are both small
        // (under isolated_component_max_faces faces) and isolated from every other component by
        // more than isolated_component_neighbor_radius voxel widths. Components that stay near the
        // main mesh are left alone — they are likely real patches that just failed to fuse.
        int remove_isolated_small_components = 1;
        int isolated_component_max_faces = 100;
        float isolated_component_neighbor_radius = 10.f;
        float max_range = 0.f;         // surfaces no view saw from closer are not meshed; 0 = no limit
        int max_width = 1600;          // render and photo width limit
        float texel_size = 0.f;        // size of a texel on the surface, for an OSGB model beside the PLY; 0 = vertex colors only
        int texture_size = 4096;       // side of a texture page
        int texture_width = 0;         // render and photo width limit when baking; 0 = the photos' own
        std::string from_mesh;         // a PLY to bake instead of extracting the surface, UTF-8
        bool invert_mask = false;
        float mask_threshold = .5f;

        // A band of under four voxels opens holes on surfaces seen at grazing angles.
        [[nodiscard]] float band() const { return truncation > 0.f ? truncation : std::max(.06f, 4.f * voxel_size); }
    };

} // namespace lfs::core
