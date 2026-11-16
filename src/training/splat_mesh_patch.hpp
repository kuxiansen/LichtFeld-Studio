/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "splat_mesh.hpp"
#include <array>
#include <cstdint>
#include <span>
#include <stop_token>
#include <vector>

namespace lfs::core {
    class Camera;
} // namespace lfs::core

namespace lfs::training {
    // The parts of a camera that patch building reads: the pose, and how it puts a direction on the photo.
    struct PatchView {
        std::array<float, 16> world_to_camera{}; // row-major; camera x is row 0 of it
        std::array<float, 3> center{};           // where the camera stands, in the world
        float fx = 0.f, fy = 0.f, cx = 0.f, cy = 0.f;
        int width = 0, height = 0;
        int model = 0;            // 0 pinhole, 1 opencv pinhole with radial and tangential distortion, 2 opencv fisheye
        float radial[6] = {};     // opencv pinhole k1..k6, fisheye k1..k4
        float tangential[2] = {}; // opencv pinhole p1, p2
    };
    // The camera's pose and projection, or false for a model patches cannot read (an equirectangular one). An
    // undistorted camera reads as a pinhole, and a thin prism fisheye leaves out its prism terms.
    [[nodiscard]] bool make_patch_view(const core::Camera& camera, PatchView& view);
    // Where a world point lands in the view's photo, in pixels, u right and v down. False behind the camera's front
    // plane, where photos have no pixel.
    [[nodiscard]] bool project_patch_point(const PatchView& view, const float* world, float* pixel);

    // A patch is a connected set of faces one photo sees, taken as one piece: the surface mapped onto that photo's
    // pixels by an affine fit, and either copied from them texel for texel or resampled through a camera of its own.
    struct MeshPatch {
        int32_t view = -1;                            // into the views the patches were built against
        bool warp = false;                            // false: the photo's pixels are the texture's, texel for texel
        std::vector<uint32_t> faces;                  // into mesh.faces
        std::array<std::array<float, 4>, 2> affine{}; // pixel = affine * (x, y, z, 1)
        std::array<float, 2> low{}, high{};           // where the projection puts the patch, pixels; not clamped to the photo
        float residual = 0.f;                         // the fit's largest deviation from the projection over the patch, pixels
        std::array<float, 3> center{};                // the middle of the patch on the surface
        double area = 0.;                             // surface the faces cover
    };

    struct PatchSet {
        std::vector<MeshPatch> patches;
        std::vector<int32_t> face_patch; // per face, its patch, or -1 where the vertex colors stand in
        size_t faces_dropped = 0;        // in patches too small to be worth pages of their own
    };

    // Splits the faces of each view into connected patches. A patch is bisected until its affine fit holds within a
    // quarter pixel, so that its texels can be the photo's own pixels; a patch the fit never holds on, or one too
    // small to be worth pages, is resampled instead. Faces of no view, patches under 32 pixels across and faces the
    // camera sees from behind it keep the vertex colors. Tile borders are cut where the tiles' own meshes end.
    [[nodiscard]] PatchSet build_mesh_patches(const SplatMesh& mesh, std::span<const int32_t> face_views,
                                              std::span<const PatchView> views, std::stop_token stop = {});
} // namespace lfs::training
