/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/progress.hpp"
#include "core/splat_mesh_settings.hpp"
#include <array>
#include <cstdint>
#include <filesystem>
#include <span>
#include <stop_token>
#include <vector>

namespace lfs::core {
    class Camera;
    class SplatData;
    struct MeshData;
} // namespace lfs::core

namespace lfs::training {
    class PPISP;

    struct SplatMesh {
        std::vector<std::array<float, 3>> vertices;
        std::vector<std::array<float, 3>> normals;
        std::vector<std::array<uint8_t, 3>> colors; // sRGB, as in the photos
        std::vector<std::array<int32_t, 3>> faces;

        // What each stage found, for reports.
        float voxel_size = 0.f; // as resolved when the settings asked for automatic
        float truncation = 0.f;
        size_t views = 0;
        size_t blocks = 0;
        uint64_t band_voxels = 0;
        size_t raw_faces = 0;
        size_t holes_filled = 0;
        size_t faces_before_simplify = 0;
        size_t faces_removed_degenerate = 0; // zero-area, repeated or non-manifold faces and the repeats of merged vertices
        size_t faces_removed_spikes = 0;
        size_t faces_removed_spurious = 0;    // outlier long-edge faces and oversized disconnected components
        size_t vertices_split_non_manifold = 0;
        float simplify_result_error = 0.f;
        size_t components_kept = 0;
        size_t components_total = 0;
        double area = 0.;
        float colored_vertex_share = 0.f;
        size_t views_color_corrected = 0;
        double seconds_render = 0.;
        double seconds_fusion = 0.;
        double seconds_extract = 0.;
        double seconds_color = 0.;
        double seconds_simplify = 0.;
    };

    // Renders transmittance median depth from the cameras, fuses it into a sparse TSDF, extracts
    // the surface with naive surface nets and colors it from the photos, taken back through the
    // model's trained appearance when there is one. Mesh is in model space.
    // Camera image sizes are changed while it runs and restored afterwards.
    [[nodiscard]] SplatMesh extract_splat_mesh(core::SplatData& model,
                                               std::span<core::Camera* const> cameras,
                                               const core::SplatMeshSettings& settings,
                                               std::stop_token stop = {}, const core::ProgressCallback& progress = {},
                                               const PPISP* appearance = nullptr);

    // Sorted sparse voxel keys (21 bits per axis, gsplat_lfs::kTsdfKeyOffset) and their TSDF
    // values. Fills vertices and faces with outward-facing triangles.
    void surface_nets(std::span<const uint64_t> keys, std::span<const float> tsdf, float voxel, SplatMesh& mesh,
                      float max_step = 1.5f);
    // Removes connected components with fewer faces and the vertices they leave unused.
    void remove_small_mesh_components(SplatMesh& mesh, int min_faces);
    // Closes open boundaries. A rim shorter than max_perimeter scene units gets a fan around a new
    // centre vertex. When ear_cut_edges > 0 a rim with fewer boundary edges than that is instead
    // filled with the ear clipping OpenMVS uses (VCG's SelfIntersectionEar), which keeps the rim
    // vertices and leaves concave rims intact. Returns how many loops it closed.
    size_t fill_small_mesh_holes(SplatMesh& mesh, float max_perimeter, int ear_cut_edges = 0, float max_edge = 0.f);
    // Drops zero-area and repeated faces, merges vertices that share a position, drops faces that
    // share an edge with more than one other face and splits vertices whose faces form more than one
    // fan. Keeps vertices, normals and colors in step and drops the vertices it leaves unused.
    // Returns how many faces it dropped.
    size_t clean_mesh_topology(SplatMesh& mesh);
    // Drops vertices that hang off a single face, until none are left, with the faces they empty.
    // Returns how many faces it dropped.
    size_t remove_mesh_spikes(SplatMesh& mesh);
    // OpenMVS's adaptive outlier pass, split in two: drop faces with any edge longer than the 95th
    // percentile edge times edge_factor, then drop connected components whose bounding-box diagonal
    // is shorter than the 55th percentile edge times component_factor. A factor of 0 disables its
    // phase.
    size_t remove_spurious_components(SplatMesh& mesh, float edge_factor, float component_factor);
    // Area-weighted vertex normals and the mesh area.
    void compute_splat_mesh_normals(SplatMesh& mesh);
    // Reduces faces while the surface stays within about max_error scene units, moving the kept
    // vertices to fit and blending their colors. Larger color_weight keeps more faces where the
    // color changes; it is not a distance, and 0 ignores color and leaves kept vertices with their
    // own colors. Refreshes normals and area. Works in pieces of at most piece_faces and of a size
    // the simplifier's precision can resolve max_error in, keeping the vertices that pieces share.
    void simplify_splat_mesh(SplatMesh& mesh, float max_error, float color_weight, size_t piece_faces = 3'000'000);

    void write_splat_mesh_ply(const SplatMesh& mesh, const std::filesystem::path& path);
    [[nodiscard]] core::MeshData splat_mesh_to_mesh_data(const SplatMesh& mesh);

} // namespace lfs::training
