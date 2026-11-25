/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/progress.hpp"
#include "core/splat_mesh_settings.hpp"
#include <array>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <stop_token>
#include <utility>
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

        // With a texture: per vertex, with vertices repeated along chart seams. u runs right and v down the
        // page, from 0 to 1. Faces of page p are faces[page_first_face[p], page_first_face[p + 1]).
        std::vector<std::array<float, 2>> uvs;
        std::vector<int32_t> uv_source; // per vertex, one number per point of the surface, whichever charts repeat it
        std::vector<size_t> page_first_face;
        std::vector<int> page_sizes;             // per page, when pages differ; else every page is page_size
        std::vector<float> page_texels;          // per page, when the texels differ; else every page is the level's texel
        std::vector<float> face_error;           // how far each face may lie from the surface, when faces differ
        std::vector<std::vector<uint8_t>> pages; // JPEG 95 of the page's RGB, sRGB as in the photos, as the writers embed it
        int page_size = 0;
        [[nodiscard]] int page_side(const size_t page) const { return page_sizes.empty() ? page_size : page_sizes[page]; }
        std::shared_ptr<struct TiledMesh> tiled; // with a texel size: the OSGB model to write, baked

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
        size_t charts = 0;
        size_t texels = 0;
        float observed_texel_share = 0.f;
        double seconds_unwrap = 0.;
        double seconds_bake = 0.;
    };

    // Renders transmittance median depth from the cameras, fuses it into a sparse TSDF, extracts
    // the surface with naive surface nets and colors it from the photos, taken back through the
    // model's trained appearance when there is one. Mesh is in model space.
    // Camera image sizes are changed while it runs and restored afterwards.
    // scratch_dir, when given, holds the depth maps the fusion needs between the render and the
    // integrate passes: one temp file there instead of several GB in RAM. Empty keeps them in RAM.
    [[nodiscard]] SplatMesh extract_splat_mesh(core::SplatData& model,
                                               std::span<core::Camera* const> cameras,
                                               const core::SplatMeshSettings& settings,
                                               std::stop_token stop = {}, const core::ProgressCallback& progress = {},
                                               const PPISP* appearance = nullptr,
                                               const std::filesystem::path& scratch_dir = {});

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
    // own colors. Refreshes normals and area. Works in pieces of at most piece_faces and of a size the
    // simplifier's precision can resolve max_error in, keeping the vertices that pieces share.
    // seam_thinning then collapses the seam those shared vertices form, which would otherwise keep the
    // density of the fused mesh next to a surface that lost most of its faces.
    void simplify_splat_mesh(SplatMesh& mesh, float max_error, float color_weight, size_t piece_faces = 3'000'000,
                             bool seam_thinning = true);

    // Orders mesh.faces so that each returned range is a compact piece of space of at most max_faces faces
    // and max_extent scene units along each axis.
    [[nodiscard]] std::vector<std::pair<size_t, size_t>> split_mesh_into_pieces(SplatMesh& mesh, size_t max_faces, float max_extent, int threads);
    // The same, over a run of mesh.faces: the other faces stay where they are.
    [[nodiscard]] std::vector<std::pair<size_t, size_t>> split_mesh_range_into_pieces(SplatMesh& mesh, size_t first, size_t last, size_t max_faces, float max_extent, int threads);

    // Gives the mesh charts packed onto pages of texel_size scene units per texel, repeating the vertices on chart
    // seams and dropping faces without area. No page is larger than max_page texels along a side. Works on pieces no
    // chart of which can outgrow a page, of at most piece_faces. Neighboring pieces of up to group_faces in all share
    // pages, packed apart from the others: packing runs on one thread and slows with the charts and pages it holds, and
    // took over 50 minutes on an outdoor scene as one group. Each group's last page is partly empty.
    // vertex_gsd, when given, is per vertex of mesh what one photo pixel covers there. The faces then fall into half
    // octaves of that detail, and each octave unwraps at its own texel, rounded down it: a small sign keeps the photos'
    // pixels while the far ground around it pays for no more. level doubles that texel, so a reduced level carries a
    // coarser copy of the same detail. Sets uvs, page_first_face, page_size, page_sizes, page_texels and charts.
    void unwrap_splat_mesh(SplatMesh& mesh, float texel_size, int max_page, std::stop_token stop = {}, size_t piece_faces = 40'000,
                           size_t group_faces = 1'000'000, std::span<const float> vertex_gsd = {}, int level = 0);

    // The texels of a page that faces cover, and those within reach of a bilinear sample of them.
    struct MeshTexels {
        std::vector<uint32_t> pixels;  // row * page_size + column
        std::vector<float> positions;  // 3 per texel, on the surface
        std::vector<uint8_t> fallback; // 3 per texel, from the vertex colors, for texels no view saw
        std::vector<uint32_t> faces;   // per texel, the face it lies on
    };
    [[nodiscard]] MeshTexels rasterize_mesh_texels(const SplatMesh& mesh, size_t page);
    // The best views of each face, as mesh_face_view_score rates them.
    struct FaceViewCandidates {
        static constexpr size_t kPerFace = 8;
        std::vector<float> scores; // kPerFace per face, best first, 0 where unused
        std::vector<int32_t> views;
        explicit FaceViewCandidates(const size_t faces) : scores(faces * kPerFace, 0.f), views(faces * kPerFace, -1) {}
        void add(int32_t view, std::span<const float> face_scores);
    };
    // One view per face, or -1 where none sees it. A view that sets the detail of a whole object cannot break it apart,
    // however its views are misregistered, so views are given out in large regions: the view that can serve the largest
    // area takes all of it, then the next, where serving means seeing a face at no less than half the score of its
    // best view. Faces mostly surrounded by another view that sees them well enough then join it.
    [[nodiscard]] std::vector<int32_t> choose_face_views(const SplatMesh& mesh, const FaceViewCandidates& candidates);

    // Takes the steps between views out of a texture and leaves the photos otherwise as they are: a smooth offset per
    // view on the vertices of its region, as small as closes the seams (Waechter et al. 2014, global seam leveling).
    // Where a point of the surface borders n views, there is one sample per view to read from them.
    struct ColorLeveling {
        std::vector<uint32_t> corner_unknowns; // 3 per face
        std::vector<int32_t> sample_labels;    // per sample, the view to read, or -1 for the mean of all
        std::vector<float> sample_positions;   // 3 per sample
        std::vector<uint32_t> sample_faces;    // per sample, a face at that point
        std::vector<uint32_t> sample_unknowns; // per sample
        std::vector<uint32_t> sample_points;   // per sample, in rising order: samples of one point are neighbors
        std::vector<float> offsets;            // 3 per unknown, once solved
        size_t iterations = 0;
        void release_samples();
        // The offset at a point of a face.
        void offset_at(const SplatMesh& mesh, uint32_t face, const float* position, float* offset) const;
    };
    [[nodiscard]] ColorLeveling plan_color_leveling(const SplatMesh& mesh, std::span<const int32_t> face_views);
    // colors: 3 per sample, 0 to 1, negative where the sample's view does not see it.
    void solve_color_leveling(ColorLeveling& leveling, const SplatMesh& mesh, std::span<const float> colors);

    // The page image from colors (3 per texel, 0 to 1, negative where unobserved), with the rest
    // of the page filled outward from the charts, so that filtering and compression reach no background.
    [[nodiscard]] std::vector<uint8_t> compose_mesh_page(const MeshTexels& texels, const std::vector<float>& colors, int page_size);

    // Unsharp mask on a composed page: out = in + amount * (in - gaussian(in, sigma 1.5)). amount 0 leaves it untouched.
    void sharpen_mesh_page(std::vector<uint8_t>& rgb, int page_size, float amount);

    // OpenMVS's local seam leveling, in page space: along every seam between two view labels the border takes the mean
    // of both sides, and the correction is diffused over a band of 16 texels on each side. It runs after the global
    // leveling, so a page's colors already carry its offsets.
    // colors: 3 per texel, 0 to 1, negative where unobserved; adjusted in place.
    struct SeamLevelingStats {
        size_t seam_texels = 0; // border texels that took the mean
        size_t band_texels = 0; // texels the correction was diffused into
        int iterations = 0;
        float max_delta = 0.f;
    };
    [[nodiscard]] SeamLevelingStats level_mesh_page_seams(const SplatMesh& mesh, const MeshTexels& texels,
                                                          std::span<const int32_t> face_views, size_t page,
                                                          std::vector<float>& colors);

    // Undoes the repetition of vertices along chart seams: which merged vertex each vertex is, and one original of each
    // merged vertex. Geometry stays connected across seams, so a viewer that computes its own normals shades it whole.
    struct WeldedVertices {
        std::vector<int32_t> index;
        std::vector<int32_t> originals;
    };
    [[nodiscard]] WeldedVertices weld_chart_seams(const SplatMesh& mesh);

    // The surface alone, connected across chart seams: the same geometry with a texture as without.
    void write_splat_mesh_ply(const SplatMesh& mesh, const std::filesystem::path& path);
    // The surface as a self-contained glTF binary, y up. With pages, each becomes a primitive with its UVs and
    // an embedded JPEG; without, the vertex colors are written and converted to linear, as glTF expects.
    void write_splat_mesh_glb(const SplatMesh& mesh, const std::filesystem::path& path);

    // One level of detail of one tile: a span of the vertices and pages of TiledMesh::combined.
    struct TileLevel {
        size_t tile = 0;
        size_t level = 0; // 0 is the finest
        size_t first_page = 0, page_count = 0;
        size_t first_vertex = 0, vertex_count = 0;
        float texel = 0.f;
        std::array<float, 3> center{0.f, 0.f, 0.f}; // of the tile, the same on every level
        float radius = 0.f;
    };
    // Every level of every tile as one mesh, so that one pass over the photos bakes them all.
    struct TiledMesh {
        SplatMesh combined;
        std::vector<TileLevel> levels;
    };
    // Splits the mesh into tiles of at most tile_faces, whose charts fit a page of max_page, and reduces each to up to
    // max_levels of detail: each allows 4 times the surface deviation of the last and doubles the texel. Vertices that tiles
    // share stay put on every level, so no cracks open between tiles. Reorders base.faces. Unwraps every level; baking remains.
    // vertex_gsd, when given, is per vertex of base what one photo pixel covers there: a tile's texel then follows the
    // tenth of its vertices the photos hold the sharpest, clamped, instead of the size asked for.
    [[nodiscard]] TiledMesh build_tiled_mesh(SplatMesh& base, float texel_size, int max_page, int max_levels, float finest_error, std::stop_token stop = {}, size_t tile_faces = 150'000, std::span<const float> vertex_gsd = {});
    // Writes a model DasViewer opens: folder/metadata.xml and folder/Data/<tile>/<tile>.osgb, the coarsest level of a tile, with
    // the next finer one paged in as the tile fills the screen.
    // A mesh from a PLY file, binary or text; polygons become fans. To bake a mesh made earlier or edited elsewhere.
    [[nodiscard]] SplatMesh read_splat_mesh_ply(const std::filesystem::path& path);
    void write_splat_mesh_osgb(const TiledMesh& tiled, const std::filesystem::path& folder, const core::ProgressCallback& progress = {}, std::stop_token stop = {});

    [[nodiscard]] core::MeshData splat_mesh_to_mesh_data(const SplatMesh& mesh);

} // namespace lfs::training
