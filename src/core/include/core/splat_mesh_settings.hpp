/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include <algorithm>
#include <string>

namespace lfs::core {

    // What the mesh is written as beside the OSGB model: a vertex-colored PLY, a GLB that embeds the texture
    // pages as JPEGs when the mesh has them, or both.
    enum class MeshFileFormat {
        Ply = 0,
        Glb = 1,
        PlyAndGlb = 2,
    };

    // Distances are in scene units. The defaults are tuned on metric indoor scans.
    struct SplatMeshSettings {
        float voxel_size = .0f; // 0 derives the voxel as 0.75x the model's median splat spacing
        int block_voxels = 8;
        float truncation = 0.f;        // band around the surface; narrower than the depth spread opens holes. 0 = automatic
        // Widens the band where neighbouring pixels of a depth map disagree: band = max(trunc,
        // trunc_gradient * local depth step), capped at trunc_cap. 0 keeps trunc everywhere.
        float trunc_gradient = 1.25f;
        float trunc_cap = 0.f;         // 0 derives the cap as 5x the truncation, so it scales with the voxel
        float full_weight_range = 1.f; // views beyond this range count less, by 1/r^2; 0 = equal weights
        int view_stride = 1;           // fuse every Nth view
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
        // Weight of the color term against the surface error when reducing faces. Colors keep faces where
        // they change; with a baked texture the vertex colors only fill texels no photo saw, so lowering
        // this trades their smoothness for fewer faces. 0 ignores color.
        float simplify_color_weight = .3f;
        // Collapses the vertices the simplify's pieces share, so that a cut through the surface no longer stays at the
        // density of the fused mesh next to faces that lost most of theirs. Off keeps the original seam untouched.
        bool simplify_seam_thinning = true;
        // Recolors the vertices from the photos of the views that see them once the faces are reduced, so that the
        // exported colors describe the simplified surface instead of the fused one whose colors the simplify, the
        // denoise and the seam thinning interpolated and averaged. Costs one more depth render over all views.
        bool reproject_vertex_colors = true;
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
        int isolated_component_max_faces = 150;
        float isolated_component_neighbor_radius = 8.f;
        float max_range = 0.f;         // surfaces no view saw from closer are not meshed; 0 = no limit
        int max_width = 1600;          // render and photo width limit
        // Bakes the texture pages and writes the OSGB model beside the PLY; off keeps the vertex colors only.
        // Until this switch existed, a texel size of 0 meant "no texture"; it now asks for the automatic size.
        bool bake_texture = true;
        float texel_size = 0.f;        // size of a texel on the surface; 0 derives it as a quarter of the voxel
        int texture_size = 4096;       // side of a texture page
        // Unsharp mask applied to every page before it is written, as OpenMVS does: out = in + amount * (in - blur).
        // It restores the edge contrast that resampling and JPEG give up; 0 writes the pages untouched.
        float texture_sharpen = .5f;
        // OpenMVS's local seam leveling, as a page-space option beside the global leveling: along every seam between
        // two views the border takes the mean of both sides, and the correction diffuses over a 16-texel band.
        bool texture_local_seam_leveling = false;
        // Takes the view chosen for each face its own pixel, without bilinear and without letting the four depth
        // corners veto it; the other views only reject what lies in front. Off reproduces the original sampler:
        // every view is bilinear and all four corners must sit on the surface. Kept to compare the two in one build.
        bool texture_own_pixel_sampling = true;
        // Composes every texel as the mean of the blurred photos over all views plus the detail of the view picked
        // for it, instead of taking the chosen view's photo whole. A photo's exposure, white balance and vignetting
        // then stop turning into a patch per face, which a few-metre triangle makes visible; the chosen view still
        // supplies the fine detail. Off is the original composition.
        bool texture_cross_view_color = true;
        int lod_levels = 4;            // levels of detail per OSGB tile; 1 writes the finest level only
        int texture_width = 0;         // render and photo width limit when baking; 0 = the photos' own
        std::string from_mesh;         // a PLY to bake instead of extracting the surface, UTF-8
        MeshFileFormat file_format = MeshFileFormat::Ply;
        bool invert_mask = false;
        float mask_threshold = .5f;

        // A band of under four voxels opens holes on surfaces seen at grazing angles.
        [[nodiscard]] float band() const { return truncation > 0.f ? truncation : std::max(.06f, 4.f * voxel_size); }
    };

} // namespace lfs::core
