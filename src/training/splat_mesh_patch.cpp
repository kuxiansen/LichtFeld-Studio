/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "splat_mesh_patch.hpp"
#include "core/camera.hpp"
#include "core/logger.hpp"
#include "splat_mesh_parallel.hpp"
#include <algorithm>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <tuple>
#include <utility>

namespace lfs::training {
    namespace {
        // A patch whose fit holds within this many pixels takes the photo's pixels as its texels, unresampled.
        constexpr float kFitTolerance = .25f;
        // The most a copied patch may be across: a page then holds several of them with room for margins.
        constexpr float kCopySpan = 512.f;
        // A patch the fit does not hold on is cut down to this; below it another seam costs more than resampling.
        constexpr float kSplitSpan = 64.f;
        // Below this a patch is not worth pages of its own; its faces keep the vertex colors.
        constexpr float kMinSpan = 32.f;
        constexpr int kMaxSplitDepth = 24;
        constexpr int kPatchThreads = 8;
        constexpr uint32_t kNoFace = ~uint32_t{0};
        // Three corners, the middle of each edge, and the face's own middle.
        constexpr int kFaceSamples = 7;

        struct Fit {
            std::array<std::array<float, 4>, 2> affine{};
            std::array<float, 2> low{}, high{};
            float residual = 0.f;
            std::array<float, 3> center{};
            std::array<float, 3> spread{}; // of the faces' centres along each axis, for cutting the patch
            double area = 0.;
            size_t samples = 0;
        };

        // Both pixel channels of the affine fit at once. False when the samples span too little of a plane to fix one.
        [[nodiscard]] bool solve_affine(const double normal[4][4], const double right[4][2], std::array<std::array<float, 4>, 2>& affine) {
            double matrix[4][6];
            for (int r = 0; r < 4; ++r) {
                for (int c = 0; c < 4; ++c)
                    matrix[r][c] = normal[r][c];
                for (int c = 0; c < 2; ++c)
                    matrix[r][4 + c] = right[r][c];
            }
            const double scale = (normal[0][0] + normal[1][1] + normal[2][2] + normal[3][3]) / 4.;
            if (!(scale > 0.))
                return false;
            // A patch flat across an axis leaves the system empty along it. A ridge settles that axis to zero and
            // leaves the fit elsewhere as it was.
            for (int c = 0; c < 4; ++c)
                matrix[c][c] += 1e-10 * scale;
            for (int column = 0; column < 4; ++column) {
                int pivot = column;
                for (int r = column + 1; r < 4; ++r)
                    if (std::abs(matrix[r][column]) > std::abs(matrix[pivot][column]))
                        pivot = r;
                if (!(std::abs(matrix[pivot][column]) > 1e-12 * scale))
                    return false;
                if (pivot != column)
                    for (int c = 0; c < 6; ++c)
                        std::swap(matrix[pivot][c], matrix[column][c]);
                const double lead = matrix[column][column];
                for (int c = column; c < 6; ++c)
                    matrix[column][c] /= lead;
                for (int r = 0; r < 4; ++r) {
                    if (r == column)
                        continue;
                    const double factor = matrix[r][column];
                    if (factor == 0.)
                        continue;
                    for (int c = column; c < 6; ++c)
                        matrix[r][c] -= factor * matrix[column][c];
                }
            }
            for (int c = 0; c < 4; ++c) {
                affine[0][size_t(c)] = float(matrix[c][4]);
                affine[1][size_t(c)] = float(matrix[c][5]);
            }
            return true;
        }

        // The face's sample points: its corners, the middle of each edge, and its own middle.
        void face_points(const SplatMesh& mesh, const uint32_t face, std::array<std::array<float, 3>, kFaceSamples>& points) {
            const auto& corner = mesh.faces[size_t(face)];
            const auto& a = mesh.vertices[size_t(corner[0])];
            const auto& b = mesh.vertices[size_t(corner[1])];
            const auto& c = mesh.vertices[size_t(corner[2])];
            points[0] = a;
            points[1] = b;
            points[2] = c;
            for (size_t k = 0; k < 3; ++k)
                for (size_t axis = 0; axis < 3; ++axis)
                    points[3 + k][axis] = .5f * (points[k][axis] + points[(k + 1) % 3][axis]);
            for (size_t axis = 0; axis < 3; ++axis)
                points[6][axis] = (a[axis] + b[axis] + c[axis]) / 3.f;
        }

        // Where the view puts the patch's sample points, the affine fit of the surface onto them, and the fit's largest
        // deviation from the projection. Bounds are the projection's own, before clamping to the photo.
        [[nodiscard]] Fit fit_faces(const std::span<const uint32_t> faces, const SplatMesh& mesh, const PatchView& view) {
            Fit fit;
            // The fit runs about the patch's own centre: scene coordinates far from the origin lose precision.
            std::array<double, 3> box_low{DBL_MAX, DBL_MAX, DBL_MAX}, box_high{-DBL_MAX, -DBL_MAX, -DBL_MAX};
            for (const auto face : faces)
                for (const auto corner : mesh.faces[size_t(face)])
                    for (size_t axis = 0; axis < 3; ++axis) {
                        box_low[axis] = std::min(box_low[axis], double(mesh.vertices[size_t(corner)][axis]));
                        box_high[axis] = std::max(box_high[axis], double(mesh.vertices[size_t(corner)][axis]));
                    }
            std::array<float, 3> origin{};
            for (size_t axis = 0; axis < 3; ++axis)
                origin[axis] = float(.5 * (box_low[axis] + box_high[axis]));

            double normal[4][4] = {}, right[4][2] = {};
            for (const auto face : faces) {
                const auto& corner = mesh.faces[size_t(face)];
                const auto& a = mesh.vertices[size_t(corner[0])];
                const auto& b = mesh.vertices[size_t(corner[1])];
                const auto& c = mesh.vertices[size_t(corner[2])];
                const double u[3] = {double(b[0]) - a[0], double(b[1]) - a[1], double(b[2]) - a[2]};
                const double v[3] = {double(c[0]) - a[0], double(c[1]) - a[1], double(c[2]) - a[2]};
                const double twice_area = std::hypot(u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]);
                fit.area += .5 * twice_area;
                // Each corner stands for a third of the face; faces without area still steer the fit a little.
                const double weight = std::max(twice_area, 2.) / 6.;
                for (const auto index : corner) {
                    const auto& position = mesh.vertices[size_t(index)];
                    float pixel[2];
                    if (!project_patch_point(view, position.data(), pixel))
                        continue;
                    const double q[4] = {double(position[0]) - origin[0], double(position[1]) - origin[1], double(position[2]) - origin[2], 1.};
                    for (int r = 0; r < 4; ++r) {
                        for (int c2 = 0; c2 < 4; ++c2)
                            normal[r][c2] += weight * q[r] * q[c2];
                        right[r][0] += weight * double(pixel[0]) * q[r];
                        right[r][1] += weight * double(pixel[1]) * q[r];
                    }
                    ++fit.samples;
                }
            }
            if (fit.samples == 0)
                return fit;

            const bool solved = solve_affine(normal, right, fit.affine);
            // Hand the fit back in the mesh's own coordinates.
            for (size_t channel = 0; channel < 2; ++channel) {
                auto& row = fit.affine[channel];
                row[3] -= row[0] * origin[0] + row[1] * origin[1] + row[2] * origin[2];
            }

            std::array<double, 2> low{DBL_MAX, DBL_MAX}, high{-DBL_MAX, -DBL_MAX};
            std::array<double, 3> centers{0., 0., 0.}, center_low{DBL_MAX, DBL_MAX, DBL_MAX}, center_high{-DBL_MAX, -DBL_MAX, -DBL_MAX};
            double worst = solved ? 0. : std::numeric_limits<double>::infinity();
            std::array<std::array<float, 3>, kFaceSamples> points;
            for (const auto face : faces) {
                face_points(mesh, face, points);
                for (const auto& point : points) {
                    float pixel[2];
                    if (!project_patch_point(view, point.data(), pixel))
                        continue;
                    low[0] = std::min(low[0], double(pixel[0]));
                    low[1] = std::min(low[1], double(pixel[1]));
                    high[0] = std::max(high[0], double(pixel[0]));
                    high[1] = std::max(high[1], double(pixel[1]));
                    if (solved) {
                        const double du = double(pixel[0]) - (double(fit.affine[0][0]) * point[0] + double(fit.affine[0][1]) * point[1] + double(fit.affine[0][2]) * point[2] + fit.affine[0][3]);
                        const double dv = double(pixel[1]) - (double(fit.affine[1][0]) * point[0] + double(fit.affine[1][1]) * point[1] + double(fit.affine[1][2]) * point[2] + fit.affine[1][3]);
                        worst = std::max(worst, std::hypot(du, dv));
                    }
                }
                for (size_t axis = 0; axis < 3; ++axis) {
                    centers[axis] += double(points[6][axis]);
                    center_low[axis] = std::min(center_low[axis], double(points[6][axis]));
                    center_high[axis] = std::max(center_high[axis], double(points[6][axis]));
                }
            }
            if (low[0] > high[0])
                return fit;
            for (size_t axis = 0; axis < 3; ++axis) {
                fit.center[axis] = float(centers[axis] / double(faces.size()));
                fit.spread[axis] = float(center_high[axis] - center_low[axis]);
            }
            fit.low = {float(low[0]), float(low[1])};
            fit.high = {float(high[0]), float(high[1])};
            fit.residual = float(worst);
            return fit;
        }

        // Halves the faces across the axis they spread most along, at the median of their centres. False when they all
        // sit at one place along every axis, where a cut would separate nothing.
        [[nodiscard]] bool split_faces(const std::span<uint32_t> faces, const SplatMesh& mesh, const std::array<float, 3>& spread,
                                       std::span<uint32_t>& first, std::span<uint32_t>& second) {
            size_t axis = 0;
            for (size_t c = 1; c < 3; ++c)
                if (spread[c] > spread[axis])
                    axis = c;
            if (!(spread[axis] > 0.f) || faces.size() < 2)
                return false;
            std::vector<std::pair<float, uint32_t>> order(faces.size());
            for (size_t i = 0; i < faces.size(); ++i) {
                float sum = 0.f;
                for (const auto index : mesh.faces[size_t(faces[i])])
                    sum += mesh.vertices[size_t(index)][axis];
                order[i] = {sum / 3.f, faces[i]};
            }
            const auto middle = order.begin() + std::ptrdiff_t(order.size() / 2);
            std::nth_element(order.begin(), middle, order.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
            if (!(middle->first > order.front().first))
                return false;
            for (size_t i = 0; i < faces.size(); ++i)
                faces[i] = order[i].second;
            first = faces.first(order.size() / 2);
            second = faces.subspan(order.size() / 2);
            return true;
        }

        // A patch the fit holds on is kept whole until it outgrows a page; one it does not hold on is cut until it
        // does, or until its pieces stop being worth pages, in which case they are resampled instead.
        void cut_patch(const std::span<uint32_t> faces, const SplatMesh& mesh, const PatchView& view, const int32_t view_index,
                       const int depth, std::vector<MeshPatch>& patches, size_t& dropped, const std::stop_token& stop) {
            if (faces.empty())
                return;
            if (stop.stop_requested())
                throw core::OperationCancelled();
            const auto fit = fit_faces(faces, mesh, view);
            if (fit.samples == 0) {
                dropped += faces.size();
                return;
            }
            const float span = std::max(fit.high[0] - fit.low[0], fit.high[1] - fit.low[1]);
            const bool fits = fit.residual <= kFitTolerance;
            if (faces.size() > 1 && depth < kMaxSplitDepth && (fits ? span > kCopySpan : span > kSplitSpan)) {
                std::span<uint32_t> first, second;
                if (split_faces(faces, mesh, fit.spread, first, second)) {
                    cut_patch(first, mesh, view, view_index, depth + 1, patches, dropped, stop);
                    cut_patch(second, mesh, view, view_index, depth + 1, patches, dropped, stop);
                    return;
                }
            }
            if (span < kMinSpan || !(fit.residual < std::numeric_limits<float>::infinity())) {
                dropped += faces.size();
                return;
            }
            MeshPatch patch;
            patch.view = view_index;
            patch.warp = !fits;
            patch.faces.assign(faces.begin(), faces.end());
            patch.affine = fit.affine;
            patch.low = fit.low;
            patch.high = fit.high;
            patch.residual = fit.residual;
            patch.center = fit.center;
            patch.area = fit.area;
            patches.push_back(std::move(patch));
        }

        // Faces of one view that meet edge to edge, as ranges of one flat list.
        struct FaceGroups {
            std::vector<uint32_t> faces;
            std::vector<std::pair<size_t, size_t>> ranges;
            std::vector<int32_t> views; // per range
        };

        // The faces across an edge, by the surface's own vertices: chart and tile seams repeat vertices, and uv_source
        // says which repeats are one point. Three slots a face, any of them kNoFace.
        [[nodiscard]] std::vector<std::array<uint32_t, 3>> face_neighbors(const SplatMesh& mesh) {
            struct Edge {
                uint64_t key;
                uint32_t face;
            };
            std::vector<Edge> edges(mesh.faces.size() * 3);
            const auto source = [&](const int32_t vertex) {
                return uint32_t(mesh.uv_source.empty() ? vertex : mesh.uv_source[size_t(vertex)]);
            };
            size_t at = 0;
            for (size_t f = 0; f < mesh.faces.size(); ++f)
                for (size_t c = 0; c < 3; ++c) {
                    const auto p = source(mesh.faces[f][c]), q = source(mesh.faces[f][(c + 1) % 3]);
                    edges[at++] = {(uint64_t(std::min(p, q)) << 32) | std::max(p, q), uint32_t(f)};
                }
            std::ranges::sort(edges, {}, &Edge::key);
            std::vector<std::array<uint32_t, 3>> neighbors(mesh.faces.size(), {kNoFace, kNoFace, kNoFace});
            const auto link = [&](const uint32_t a, const uint32_t b) {
                for (auto& slot : neighbors[a])
                    if (slot == kNoFace) {
                        slot = b;
                        return;
                    }
            };
            for (size_t e = 0; e + 1 < edges.size(); ++e)
                if (edges[e].key == edges[e + 1].key) {
                    link(edges[e].face, edges[e + 1].face);
                    link(edges[e + 1].face, edges[e].face);
                }
            return neighbors;
        }

        [[nodiscard]] FaceGroups group_faces(const SplatMesh& mesh, const std::span<const int32_t> face_views, const std::stop_token& stop) {
            const auto neighbors = face_neighbors(mesh);
            FaceGroups groups;
            groups.faces.reserve(mesh.faces.size());
            std::vector<uint8_t> seen(mesh.faces.size(), 0);
            std::vector<uint32_t> stack;
            for (size_t f = 0; f < mesh.faces.size(); ++f) {
                if (seen[f] || face_views[f] < 0)
                    continue;
                if (stop.stop_requested())
                    throw core::OperationCancelled();
                const auto view = face_views[f];
                const auto first = groups.faces.size();
                stack.assign(1, uint32_t(f));
                seen[f] = 1;
                while (!stack.empty()) {
                    const auto face = stack.back();
                    stack.pop_back();
                    groups.faces.push_back(face);
                    for (const auto other : neighbors[size_t(face)])
                        if (other != kNoFace && !seen[size_t(other)] && face_views[size_t(other)] == view) {
                            seen[size_t(other)] = 1;
                            stack.push_back(other);
                        }
                }
                groups.ranges.emplace_back(first, groups.faces.size());
                groups.views.push_back(view);
            }
            return groups;
        }

        void copy_coefficients(const core::Tensor& source, float* target, const size_t count) {
            if (!source.is_valid() || source.numel() == 0)
                return;
            auto host = source.cpu().to(core::DataType::Float32).contiguous();
            std::memcpy(target, host.ptr<float>(), std::min(source.numel(), count) * sizeof(float));
        }
    } // namespace

    bool make_patch_view(const core::Camera& camera, PatchView& view) {
        const bool undistorted = camera.is_undistort_prepared();
        const auto model = camera.camera_model_type();
        view = {};
        view.width = camera.image_width();
        view.height = camera.image_height();
        std::tie(view.fx, view.fy, view.cx, view.cy) = camera.get_intrinsics();
        auto pose = camera.world_view_transform().cpu().contiguous();
        std::memcpy(view.world_to_camera.data(), pose.ptr<float>(), sizeof(float) * view.world_to_camera.size());
        const float* m = view.world_to_camera.data();
        // Where the camera stands: -R^T t, with R the rows of the pose's rotation and t its translation.
        for (size_t axis = 0; axis < 3; ++axis)
            view.center[axis] = -(m[axis] * m[3] + m[4 + axis] * m[7] + m[8 + axis] * m[11]);
        if (undistorted) {
            view.model = 0;
            return true;
        }
        if (model == core::CameraModelType::FISHEYE) {
            view.model = 2;
            copy_coefficients(camera.radial_distortion(), view.radial, 4);
            return true;
        }
        if (model == core::CameraModelType::THIN_PRISM_FISHEYE) {
            LOG_WARN("Mesh: the thin prism fisheye is read as a plain fisheye; its prism terms are left out");
            view.model = 2;
            copy_coefficients(camera.radial_distortion(), view.radial, 4);
            return true;
        }
        if (model == core::CameraModelType::EQUIRECTANGULAR)
            return false;
        const bool distorted = (camera.radial_distortion().is_valid() && camera.radial_distortion().numel() > 0) ||
                               (camera.tangential_distortion().is_valid() && camera.tangential_distortion().numel() > 0);
        view.model = distorted ? 1 : 0;
        if (distorted) {
            copy_coefficients(camera.radial_distortion(), view.radial, 6);
            copy_coefficients(camera.tangential_distortion(), view.tangential, 2);
        }
        return true;
    }

    bool project_patch_point(const PatchView& view, const float* world, float* pixel) {
        const float* m = view.world_to_camera.data();
        const double x = double(m[0]) * world[0] + double(m[1]) * world[1] + double(m[2]) * world[2] + m[3];
        const double y = double(m[4]) * world[0] + double(m[5]) * world[1] + double(m[6]) * world[2] + m[7];
        const double z = double(m[8]) * world[0] + double(m[9]) * world[1] + double(m[10]) * world[2] + m[11];
        if (!(z > 0.))
            return false;
        double u = 0., v = 0.;
        if (view.model == 2) {
            // OpenCV fisheye: a ray at angle theta to the axis lands at f * theta, bent by k1..k4 of theta squared.
            const double radius = std::hypot(x, y);
            if (radius > 1e-9) {
                const double theta = std::atan2(radius, z), square = theta * theta;
                const double scaled = theta * (1. + square * (view.radial[0] + square * (view.radial[1] + square * (view.radial[2] + square * view.radial[3]))));
                u = double(view.fx) * scaled * x / radius + view.cx;
                v = double(view.fy) * scaled * y / radius + view.cy;
            } else {
                u = view.cx;
                v = view.cy;
            }
        } else if (view.model == 1) {
            const double nx = x / z, ny = y / z;
            const double r2 = nx * nx + ny * ny;
            const float* k = view.radial;
            const double radial = (1. + r2 * (k[0] + r2 * (k[1] + r2 * k[2]))) / (1. + r2 * (k[3] + r2 * (k[4] + r2 * k[5])));
            const double a1 = 2. * nx * ny, a2 = r2 + 2. * nx * nx, a3 = r2 + 2. * ny * ny;
            u = double(view.fx) * (radial * nx + double(view.tangential[0]) * a1 + double(view.tangential[1]) * a2) + view.cx;
            v = double(view.fy) * (radial * ny + double(view.tangential[0]) * a3 + double(view.tangential[1]) * a1) + view.cy;
        } else {
            u = double(view.fx) * x / z + view.cx;
            v = double(view.fy) * y / z + view.cy;
        }
        if (!std::isfinite(u) || !std::isfinite(v))
            return false;
        pixel[0] = float(u);
        pixel[1] = float(v);
        return true;
    }

    PatchSet build_mesh_patches(const SplatMesh& mesh, const std::span<const int32_t> face_views, const std::span<const PatchView> views,
                                const std::stop_token stop) {
        using Clock = std::chrono::steady_clock;
        const auto start = Clock::now();
        PatchSet set;
        set.face_patch.assign(mesh.faces.size(), -1);
        if (mesh.faces.size() != face_views.size() || views.empty())
            return set;
        auto groups = group_faces(mesh, face_views, stop);

        struct Cut {
            std::vector<MeshPatch> patches;
            size_t dropped = 0;
        };
        std::vector<Cut> cuts(groups.ranges.size());
        detail::parallel_for(cuts.size(), kPatchThreads, [&](const size_t g) {
            const auto [first, last] = groups.ranges[g];
            const auto view = size_t(groups.views[g]);
            auto faces = std::span<uint32_t>(groups.faces).subspan(first, last - first);
            if (view >= views.size()) {
                cuts[g].dropped = faces.size();
                return;
            }
            cut_patch(faces, mesh, views[view], groups.views[g], 0, cuts[g].patches, cuts[g].dropped, stop);
        });
        groups = {};

        size_t copies = 0, warps = 0, covered = 0, labeled = 0;
        for (const auto view : face_views)
            labeled += view >= 0 ? 1 : 0;
        for (auto& cut : cuts) {
            set.faces_dropped += cut.dropped;
            for (auto& patch : cut.patches) {
                const auto index = int32_t(set.patches.size());
                covered += patch.faces.size();
                for (const auto face : patch.faces)
                    set.face_patch[size_t(face)] = index;
                (patch.warp ? warps : copies)++;
                set.patches.push_back(std::move(patch));
            }
        }
        LOG_INFO("Mesh: {} patches cover {} of {} faces of {} views in {:.1f} s, {} copied and {} resampled; {} faces ({} %) too small to paint",
                 set.patches.size(), covered, labeled, views.size(), std::chrono::duration<double>(Clock::now() - start).count(), copies, warps,
                 set.faces_dropped, labeled > 0 ? std::lround(100. * double(set.faces_dropped) / double(labeled)) : 0l);
        return set;
    }
} // namespace lfs::training
