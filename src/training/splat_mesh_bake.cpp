/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "splat_mesh_bake.hpp"
#include "splat_mesh_patch.hpp"
#include "core/logger.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <numeric>
#include <unordered_map>

namespace lfs::training {
    namespace {
        // As the atlas path's tolerance: a coarse surface lies as far from the photos as its faces were
        // allowed to move, and the depth band a texel is accepted within grows with that.
        constexpr float kFaceErrorToTolerance = 2.f;
        // Invalid texels reach this far for a valid neighbor before the vertex colors stand in. The packer
        // keeps four texels clear around every rect, so a page's gaps are covered with room.
        constexpr int kRectFillRings = 8;
        // A leftover face's cell on a color page: 2x2 texels, one of gap, four to the next cell.
        constexpr int kColorCellStep = 4, kColorCellOrigin = 2;

        // Bilinear over the photo's float planes, border-clamped, pixel centers at integer coordinates.
        void sample_photo(const float* photo, const int width, const int height, const double u, const double v, uint8_t out[3]) {
            const double cu = std::clamp(u, 0., double(std::max(width - 1, 0)));
            const double cv = std::clamp(v, 0., double(std::max(height - 1, 0)));
            const int x0 = std::min(int(cu), std::max(width - 2, 0));
            const int y0 = std::min(int(cv), std::max(height - 2, 0));
            const double fx = cu - x0, fy = cv - y0, gx = 1. - fx, gy = 1. - fy;
            const size_t pixels = size_t(width) * height, at = size_t(y0) * width + x0;
            for (int c = 0; c < 3; ++c) {
                const float* plane = photo + c * pixels + at;
                const double value = gy * (gx * plane[0] + fx * plane[1]) + fy * (gx * plane[width] + fx * plane[width + 1]);
                out[c] = static_cast<uint8_t>(std::clamp(std::lround(255. * value), 0l, 255l));
            }
        }

        // The photo's own pixel, taken whole: what a copied patch's texel is.
        void take_photo_pixel(const float* photo, const int width, const int height, const int x, const int y, uint8_t out[3]) {
            const size_t pixels = size_t(width) * height, at = size_t(y) * width + x;
            for (int c = 0; c < 3; ++c)
                out[c] = static_cast<uint8_t>(std::clamp(std::lround(255. * photo[c * pixels + at]), 0l, 255l));
        }

        // A patch's mean vertex color, its faces' corners averaged: what texels no photo vouched for fall back to.
        void mean_patch_color(const SplatMesh& mesh, const MeshPatch& patch, uint8_t out[3]) {
            out[0] = out[1] = out[2] = 128;
            if (mesh.colors.empty())
                return;
            double sum[3] = {};
            size_t count = 0;
            for (const auto face : patch.faces)
                for (const auto corner : mesh.faces[size_t(face)]) {
                    const auto& color = mesh.colors[size_t(corner)];
                    for (int c = 0; c < 3; ++c)
                        sum[c] += color[c];
                    ++count;
                }
            if (count > 0)
                for (int c = 0; c < 3; ++c)
                    out[c] = static_cast<uint8_t>(std::clamp(std::lround(sum[c] / double(count)), 0l, 255l));
        }

        // The photo around a pixel, averaged with the pixels whose depth lies on the same surface: the broad color a
        // seam is leveled by, with grain and misregistration taken out and occluders left out.
        [[nodiscard]] bool sample_photo_blurred(const PatchBakePhoto& photo, const double u, const double v, const int radius,
                                                const double tolerance, float out[3]) {
            const int width = photo.width, height = photo.height;
            const int cx = std::clamp(int(std::lround(u)), 0, width - 1);
            const int cy = std::clamp(int(std::lround(v)), 0, height - 1);
            const double center = photo.depth[size_t(cy) * width + cx];
            if (!(center > 0.))
                return false;
            const size_t pixels = size_t(width) * height;
            double sum[3] = {};
            double count = 0.;
            for (int y = std::max(0, cy - radius); y <= std::min(height - 1, cy + radius); ++y)
                for (int x = std::max(0, cx - radius); x <= std::min(width - 1, cx + radius); ++x) {
                    const size_t at = size_t(y) * width + x;
                    const double measured = photo.depth[at];
                    if (!(measured > 0.) || std::abs(measured - center) > tolerance)
                        continue;
                    for (int c = 0; c < 3; ++c)
                        sum[c] += photo.photo[c * pixels + at];
                    count += 1.;
                }
            if (!(count > 0.))
                return false;
            for (int c = 0; c < 3; ++c)
                out[c] = float(sum[c] / count);
            return true;
        }

        // A sparse system of three-channel unknowns, solved by conjugate gradients with the diagonal as
        // preconditioner: the atlas path's leveling solver, over whatever graph a mode builds.
        struct ThreeChannelSystem {
            size_t unknowns = 0;
            std::vector<float> diagonal;
            std::vector<float> right; // 3 per unknown
            std::vector<std::pair<uint64_t, float>> entries;

            void couple(const uint32_t a, const uint32_t b, const float weight) {
                if (a == b)
                    return;
                diagonal[a] += weight;
                diagonal[b] += weight;
                entries.push_back({(uint64_t(a) << 32) | b, -weight});
                entries.push_back({(uint64_t(b) << 32) | a, -weight});
            }

            [[nodiscard]] std::vector<float> solve(size_t& iterations) const {
                constexpr int kMaxIterations = 2000;
                constexpr double kTolerance = 1e-3;
                std::vector<float> x(unknowns * 3, 0.f);
                if (unknowns == 0)
                    return x;
                std::vector<std::pair<uint64_t, float>> sorted = entries;
                std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
                std::vector<size_t> row_first(unknowns + 1, 0);
                std::vector<uint32_t> columns;
                std::vector<float> values;
                for (size_t e = 0; e < sorted.size(); ++e) {
                    if (e > 0 && sorted[e].first == sorted[e - 1].first) {
                        values.back() += sorted[e].second;
                        continue;
                    }
                    columns.push_back(uint32_t(sorted[e].first));
                    values.push_back(sorted[e].second);
                    ++row_first[size_t(sorted[e].first >> 32) + 1];
                }
                std::partial_sum(row_first.begin(), row_first.end(), row_first.begin());
                const auto multiply = [&](const std::vector<float>& in, std::vector<float>& out) {
                    for (size_t i = 0; i < unknowns; ++i) {
                        float sum[3];
                        for (size_t c = 0; c < 3; ++c)
                            sum[c] = diagonal[i] * in[3 * i + c];
                        for (size_t e = row_first[i]; e < row_first[i + 1]; ++e)
                            for (size_t c = 0; c < 3; ++c)
                                sum[c] += values[e] * in[3 * size_t(columns[e]) + c];
                        for (size_t c = 0; c < 3; ++c)
                            out[3 * i + c] = sum[c];
                    }
                };
                const auto dot = [&](const std::vector<float>& a, const std::vector<float>& b, const bool scaled) {
                    std::array<double, 3> total{0., 0., 0.};
                    for (size_t i = 0; i < unknowns; ++i) {
                        const double scale = scaled ? 1. / double(diagonal[i]) : 1.;
                        for (size_t c = 0; c < 3; ++c)
                            total[c] += scale * double(a[3 * i + c]) * double(b[3 * i + c]);
                    }
                    return total;
                };
                std::vector<float> residual = right, direction(unknowns * 3), product(unknowns * 3);
                for (size_t i = 0; i < unknowns; ++i)
                    for (size_t c = 0; c < 3; ++c)
                        direction[3 * i + c] = residual[3 * i + c] / diagonal[i];
                auto rho = dot(residual, residual, true);
                const auto start = dot(right, right, false);
                int iteration = 0;
                for (; iteration < kMaxIterations; ++iteration) {
                    const auto norm = dot(residual, residual, false);
                    bool done = true;
                    for (size_t c = 0; c < 3; ++c)
                        done = done && norm[c] <= kTolerance * kTolerance * start[c];
                    if (done)
                        break;
                    multiply(direction, product);
                    const auto curvature = dot(direction, product, false);
                    std::array<float, 3> alpha;
                    for (size_t c = 0; c < 3; ++c)
                        alpha[c] = curvature[c] > 0. ? float(rho[c] / curvature[c]) : 0.f;
                    for (size_t i = 0; i < unknowns * 3; ++i) {
                        // Three channels share the index only through the block layout: alpha repeats per channel.
                        const size_t c = i % 3;
                        x[i] += alpha[c] * direction[i];
                        residual[i] -= alpha[c] * product[i];
                    }
                    const auto next_rho = dot(residual, residual, true);
                    std::array<float, 3> beta;
                    for (size_t c = 0; c < 3; ++c)
                        beta[c] = rho[c] > 0. ? float(next_rho[c] / rho[c]) : 0.f;
                    for (size_t i = 0; i < unknowns * 3; ++i) {
                        const size_t c = i % 3;
                        direction[i] = residual[i] / diagonal[i / 3] + beta[c] * direction[i];
                    }
                    rho = next_rho;
                }
                iterations = size_t(iteration);
                return x;
            }
        };
    } // namespace

    PatchBakeRect bake_patch_texture(const SplatMesh& mesh, const MeshPatch& patch, const PatchPlacement& place, const PatchView& view,
                                     const PatchBakePhoto& photo, const float* face_error, const float base_tolerance) {
        PatchBakeRect rect;
        rect.width = std::max(place.width, 0);
        rect.height = std::max(place.height, 0);
        if (rect.width == 0 || rect.height == 0 || !photo.photo || !photo.depth)
            return rect;
        rect.rgb.assign(size_t(rect.width) * rect.height * 3, 0);
        rect.valid.assign(size_t(rect.width) * rect.height, 0);
        if (photo.width < 2 || photo.height < 2)
            return rect;
        // The rect's own grid is the photo's window under the placement's scale, and the mesh's UVs on the
        // page are the same map: a texel and its texture stay one point of the surface.
        const double scale = place.scale > 0.f ? double(place.scale) : 1.;
        const double source_x = place.source_x, source_y = place.source_y;
        const int width = photo.width, height = photo.height;
        for (const auto face : patch.faces) {
            const auto& corners = mesh.faces[size_t(face)];
            const float* a = mesh.vertices[size_t(corners[0])].data();
            const float* b = mesh.vertices[size_t(corners[1])].data();
            const float* c = mesh.vertices[size_t(corners[2])].data();
            float pa[2], pb[2], pc[2];
            if (!project_patch_point(view, a, pa) || !project_patch_point(view, b, pb) || !project_patch_point(view, c, pc))
                continue;
            const double ax = (pa[0] - source_x) * scale, ay = (pa[1] - source_y) * scale;
            const double bx = (pb[0] - source_x) * scale, by = (pb[1] - source_y) * scale;
            const double cx = (pc[0] - source_x) * scale, cy = (pc[1] - source_y) * scale;
            const double area = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
            if (!(std::abs(area) > 1e-12))
                continue;
            const double tolerance = face_error != nullptr ? std::max(double(base_tolerance), kFaceErrorToTolerance * double(face_error[face]))
                                                           : double(base_tolerance);
            const int x0 = std::max(0, int(std::floor(std::min({ax, bx, cx}) - .5)));
            const int x1 = std::min(rect.width - 1, int(std::ceil(std::max({ax, bx, cx}) - .5)));
            const int y0 = std::max(0, int(std::floor(std::min({ay, by, cy}) - .5)));
            const int y1 = std::min(rect.height - 1, int(std::ceil(std::max({ay, by, cy}) - .5)));
            for (int y = y0; y <= y1; ++y)
                for (int x = x0; x <= x1; ++x) {
                    const double px = x + .5, py = y + .5;
                    const double wa = ((bx - px) * (cy - py) - (cx - px) * (by - py)) / area;
                    const double wb = ((cx - px) * (ay - py) - (ax - px) * (cy - py)) / area;
                    const double wc = 1. - wa - wb;
                    if (wa < -1e-9 || wb < -1e-9 || wc < -1e-9)
                        continue;
                    const double point[3] = {wa * a[0] + wb * b[0] + wc * c[0], wa * a[1] + wb * b[1] + wc * c[1], wa * a[2] + wb * b[2] + wc * c[2]};
                    const float here[3] = {float(point[0]), float(point[1]), float(point[2])};
                    const double dx = point[0] - view.center[0], dy = point[1] - view.center[1], dz = point[2] - view.center[2];
                    const double range = std::sqrt(dx * dx + dy * dy + dz * dz);
                    uint8_t color[3];
                    if (place.warp) {
                        float pixel[2];
                        if (!project_patch_point(view, here, pixel) || !(pixel[0] >= 0.) || !(pixel[1] >= 0.) || !(pixel[0] <= width - 1) ||
                            !(pixel[1] <= height - 1))
                            continue;
                        const int dpx = std::clamp(int(pixel[0]), 0, width - 1), dpy = std::clamp(int(pixel[1]), 0, height - 1);
                        const double measured = photo.depth[size_t(dpy) * width + dpx];
                        if (!(measured > 0.) || std::abs(measured - range) >= tolerance)
                            continue;
                        sample_photo(photo.photo, width, height, pixel[0], pixel[1], color);
                    } else {
                        const int fx = int(std::lround(source_x + double(x) / scale));
                        const int fy = int(std::lround(source_y + double(y) / scale));
                        if (fx < 0 || fy < 0 || fx >= width || fy >= height)
                            continue;
                        const double measured = photo.depth[size_t(fy) * width + fx];
                        if (!(measured > 0.) || std::abs(measured - range) >= tolerance)
                            continue;
                        take_photo_pixel(photo.photo, width, height, fx, fy, color);
                    }
                    const size_t at = size_t(y) * rect.width + x;
                    rect.valid[at] = 1;
                    std::memcpy(&rect.rgb[3 * at], color, 3);
                }
        }
        return rect;
    }

    void fill_patch_rect(PatchBakeRect& rect, const SplatMesh& mesh, const MeshPatch& patch) {
        if (rect.width <= 0 || rect.height <= 0)
            return;
        const int w = rect.width, h = rect.height;
        for (int ring = 0; ring < kRectFillRings; ++ring) {
            bool any = false;
            for (int y = 0; y < h; ++y)
                for (int x = 0; x < w; ++x) {
                    const size_t at = size_t(y) * w + x;
                    if (rect.valid[at] != 0)
                        continue;
                    bool taken = false;
                    for (int dy = -1; dy <= 1 && !taken; ++dy)
                        for (int dx = -1; dx <= 1; ++dx) {
                            if ((dx == 0 && dy == 0) || x + dx < 0 || y + dy < 0 || x + dx >= w || y + dy >= h)
                                continue;
                            const size_t other = size_t(y + dy) * w + (x + dx);
                            if (rect.valid[other] == 1) {
                                std::memcpy(&rect.rgb[3 * at], &rect.rgb[3 * other], 3);
                                rect.valid[at] = 2;
                                any = true;
                                taken = true;
                                break;
                            }
                        }
                }
            if (!any)
                break;
            for (auto& value : rect.valid)
                if (value == 2)
                    value = 1;
        }
        uint8_t fallback[3];
        mean_patch_color(mesh, patch, fallback);
        for (size_t at = 0; at < rect.valid.size(); ++at)
            if (rect.valid[at] == 0)
                std::memcpy(&rect.rgb[3 * at], fallback, 3);
    }

    void apply_patch_texture(uint8_t* page, const int side, const PatchPlacement& place, const PatchBakeRect& rect) {
        if (!page || rect.width != place.width || rect.height != place.height)
            return;
        for (int y = 0; y < rect.height; ++y) {
            const int row = place.y + y;
            if (row < 0 || row >= side)
                continue;
            const int x0 = std::max(0, -place.x), x1 = std::min(rect.width, side - place.x);
            if (x1 <= x0)
                continue;
            std::memcpy(page + (size_t(row) * side + place.x + x0) * 3, rect.rgb.data() + (size_t(y) * rect.width + size_t(x0)) * 3,
                        size_t(x1 - x0) * 3);
        }
    }

    void sample_patch_seam_colors(const ColorLeveling& plan, const std::span<const uint32_t> sample_indices, const int32_t label,
                                  const PatchView& view, const PatchBakePhoto& photo, const float* face_error, const float base_tolerance,
                                  const std::span<float> colors) {
        if (!photo.photo || !photo.depth || photo.width < 2 || photo.height < 2 || colors.size() < plan.sample_labels.size() * 3)
            return;
        // About one percent of the image's width, as the atlas path blurs, bounded so the reads stay cheap.
        const int radius = std::clamp(photo.width / 100, 1, 8);
        const int width = photo.width;
        for (const auto s : sample_indices) {
            if (plan.sample_labels[s] != label)
                continue;
            const float* position = &plan.sample_positions[3 * size_t(s)];
            float pixel[2];
            if (!project_patch_point(view, position, pixel))
                continue;
            if (!(pixel[0] >= 0.f && pixel[1] >= 0.f && pixel[0] <= width - 1 && pixel[1] <= photo.height - 1))
                continue;
            const int px = std::clamp(int(std::lround(pixel[0])), 0, width - 1);
            const int py = std::clamp(int(std::lround(pixel[1])), 0, photo.height - 1);
            const double measured = photo.depth[size_t(py) * width + px];
            const double dx = position[0] - view.center[0], dy = position[1] - view.center[1], dz = position[2] - view.center[2];
            const double range = std::sqrt(dx * dx + dy * dy + dz * dz);
            const float tolerance = face_error != nullptr
                                        ? std::max(base_tolerance, kFaceErrorToTolerance * face_error[plan.sample_faces[s]])
                                        : base_tolerance;
            if (!(measured > 0.) || std::abs(measured - range) >= tolerance)
                continue;
            float color[3];
            if (!sample_photo_blurred(photo, pixel[0], pixel[1], radius, std::max(double(tolerance), 1e-3), color))
                continue;
            for (int c = 0; c < 3; ++c)
                colors[3 * size_t(s) + c] = color[c];
        }
    }

    void apply_patch_gain(uint8_t* page, const int side, const PatchPlacement& place, const float* gain) {
        if (!page || !gain || side <= 0)
            return;
        const int y0 = std::max(0, place.y), y1 = std::min(side, place.y + place.height);
        const int x0 = std::max(0, place.x), x1 = std::min(side, place.x + place.width);
        for (int y = y0; y < y1; ++y)
            for (int x = x0; x < x1; ++x) {
                uint8_t* pixel = page + (size_t(y) * side + x) * 3;
                for (int c = 0; c < 3; ++c)
                    pixel[c] = static_cast<uint8_t>(std::clamp(std::lround(double(pixel[c]) * double(gain[c])), 0l, 255l));
            }
    }

    void apply_patch_offsets(uint8_t* page, const int side, const SplatMesh& mesh, const MeshPatch& patch, const PatchPlacement& place,
                             const PatchView& view, const std::span<const uint32_t> corner_unknowns, const std::span<const float> offsets) {
        const size_t unknowns = offsets.size() / 3;
        if (!page || side <= 0 || place.width <= 0 || place.height <= 0 || unknowns == 0 ||
            corner_unknowns.size() < mesh.faces.size() * 3)
            return;
        // The rect's texels are walked the way the bake walks them, face by face, so each texel gets the offset of
        // the point of the surface it shows; a texel two faces both claim is offset once.
        const double scale = place.scale > 0.f ? double(place.scale) : 1.;
        const double source_x = place.source_x, source_y = place.source_y;
        std::vector<uint8_t> painted(size_t(place.width) * place.height, 0);
        for (const auto face : patch.faces) {
            const auto& corners = mesh.faces[size_t(face)];
            float pa[2], pb[2], pc[2];
            if (!project_patch_point(view, mesh.vertices[size_t(corners[0])].data(), pa) ||
                !project_patch_point(view, mesh.vertices[size_t(corners[1])].data(), pb) ||
                !project_patch_point(view, mesh.vertices[size_t(corners[2])].data(), pc))
                continue;
            const double ax = (pa[0] - source_x) * scale, ay = (pa[1] - source_y) * scale;
            const double bx = (pb[0] - source_x) * scale, by = (pb[1] - source_y) * scale;
            const double cx = (pc[0] - source_x) * scale, cy = (pc[1] - source_y) * scale;
            const double area = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
            if (!(std::abs(area) > 1e-12))
                continue;
            const uint32_t corner[3] = {corner_unknowns[3 * size_t(face)], corner_unknowns[3 * size_t(face) + 1],
                                        corner_unknowns[3 * size_t(face) + 2]};
            const int x0 = std::max(0, int(std::floor(std::min({ax, bx, cx}) - .5)));
            const int x1 = std::min(place.width - 1, int(std::ceil(std::max({ax, bx, cx}) - .5)));
            const int y0 = std::max(0, int(std::floor(std::min({ay, by, cy}) - .5)));
            const int y1 = std::min(place.height - 1, int(std::ceil(std::max({ay, by, cy}) - .5)));
            for (int y = y0; y <= y1; ++y)
                for (int x = x0; x <= x1; ++x) {
                    const size_t at = size_t(y) * place.width + x;
                    if (painted[at] != 0)
                        continue;
                    const double px = x + .5, py = y + .5;
                    const double wa = ((bx - px) * (cy - py) - (cx - px) * (by - py)) / area;
                    const double wb = ((cx - px) * (ay - py) - (ax - px) * (cy - py)) / area;
                    const double wc = 1. - wa - wb;
                    if (wa < -1e-9 || wb < -1e-9 || wc < -1e-9)
                        continue;
                    painted[at] = 1;
                    const int page_x = place.x + x, page_y = place.y + y;
                    if (page_x < 0 || page_y < 0 || page_x >= side || page_y >= side)
                        continue;
                    const double weight[3] = {std::max(wa, 0.), std::max(wb, 0.), std::max(wc, 0.)};
                    double offset[3] = {};
                    for (size_t k = 0; k < 3; ++k)
                        if (corner[k] < unknowns && weight[k] > 0.)
                            for (int c = 0; c < 3; ++c)
                                offset[c] += weight[k] * double(offsets[3 * size_t(corner[k]) + c]);
                    uint8_t* pixel = page + (size_t(page_y) * side + page_x) * 3;
                    for (int c = 0; c < 3; ++c)
                        pixel[c] = static_cast<uint8_t>(std::clamp(std::lround(double(pixel[c]) + 255. * offset[c]), 0l, 255l));
                }
        }
    }

    PatchSeamLeveling solve_patch_seam_leveling(const SplatMesh& mesh, const std::span<const int32_t> face_labels,
                                                const std::span<const PatchView> views, const ColorLeveling& plan,
                                                const std::span<const float> colors, const int band_pixels) {
        // As the atlas path weighs a seam step: ten times an edge along the surface.
        constexpr float kEdgeWeight = .1f;
        constexpr float kSeamWeight = 1.f;
        // Keeps a component that no band edge anchors from drifting.
        constexpr float kPull = 1e-6f;
        constexpr int kMaxRings = 8;
        const size_t labels = views.size();
        const size_t face_count = mesh.faces.size();
        PatchSeamLeveling out;
        out.gains.assign(labels * 3, 1.f);
        const size_t sample_count = plan.sample_labels.size();
        if (sample_count == 0 || face_count == 0 || labels == 0 || colors.size() < sample_count * 3)
            return out;
        // The views' gains: a view brighter than another by a factor takes that factor out of every color it gives.
        ThreeChannelSystem gains;
        gains.unknowns = labels;
        gains.diagonal.assign(labels, kPull);
        gains.right.assign(labels * 3, 0.f);
        size_t gain_steps = 0;
        for (size_t first = 0; first < sample_count;) {
            size_t last = first;
            while (last < sample_count && plan.sample_points[last] == plan.sample_points[first])
                ++last;
            for (size_t a = first; a < last; ++a)
                for (size_t b = a + 1; b < last; ++b) {
                    const int32_t la = plan.sample_labels[a], lb = plan.sample_labels[b];
                    if (la < 0 || lb < 0 || la == lb || size_t(la) >= labels || size_t(lb) >= labels)
                        continue;
                    std::array<float, 3> step{};
                    bool valid = true;
                    for (size_t c = 0; c < 3 && valid; ++c) {
                        const float ca = colors[3 * a + c], cb = colors[3 * b + c];
                        if (!(ca > 1.f / 255.f) || !(cb > 1.f / 255.f))
                            valid = false;
                        else
                            step[c] = std::log(cb) - std::log(ca);
                    }
                    if (!valid)
                        continue;
                    gains.couple(uint32_t(la), uint32_t(lb), 1.f);
                    for (size_t c = 0; c < 3; ++c) {
                        gains.right[3 * size_t(la) + c] += step[c];
                        gains.right[3 * size_t(lb) + c] -= step[c];
                    }
                    ++gain_steps;
                }
            first = last;
        }
        size_t gain_iterations = 0;
        const auto gain_log = gains.solve(gain_iterations);
        float lowest = 1.f, highest = 1.f;
        for (size_t l = 0; l < labels; ++l)
            for (size_t c = 0; c < 3; ++c) {
                const float value = std::clamp(std::exp(gain_log[3 * l + c]), .25f, 4.f);
                out.gains[3 * l + c] = value;
                lowest = std::min(lowest, value);
                highest = std::max(highest, value);
            }
        // The band: how many faces a seam's step is spread over, so that band_pixels of the photo take it up.
        std::vector<double> edge_pixels(labels, 0.);
        std::vector<size_t> edge_counts(labels, 0);
        for (size_t f = 0; f < face_count; ++f) {
            const int32_t label = face_labels[f];
            if (label < 0 || size_t(label) >= labels)
                continue;
            const auto& corners = mesh.faces[f];
            float pixel[3][2];
            bool ok = true;
            for (int c = 0; c < 3 && ok; ++c)
                ok = project_patch_point(views[size_t(label)], mesh.vertices[size_t(corners[c])].data(), pixel[c]);
            if (!ok)
                continue;
            double width = 0.;
            for (int c = 0; c < 3; ++c) {
                const int next = (c + 1) % 3;
                const double dx = double(pixel[next][0]) - pixel[c][0], dy = double(pixel[next][1]) - pixel[c][1];
                width += std::sqrt(dx * dx + dy * dy);
            }
            edge_pixels[size_t(label)] += width / 3.;
            ++edge_counts[size_t(label)];
        }
        // A band one face wide would hold no corner of its own: the face's corners all lie on its edge, which the
        // band's boundary fixes. Two faces is the narrowest band that can take a step up.
        std::vector<int> rings(labels, 2);
        for (size_t l = 0; l < labels; ++l)
            if (edge_counts[l] > 0)
                rings[l] = std::clamp(int(std::lround(double(band_pixels) / std::max(edge_pixels[l] / double(edge_counts[l]), 1e-3))),
                                      2, kMaxRings);
        // Faces are neighbors across a shared edge, as the atlas path finds them.
        constexpr uint32_t kNone = ~uint32_t{0};
        struct Edge {
            uint64_t key;
            uint32_t face;
        };
        std::vector<Edge> edges;
        edges.reserve(face_count * 3);
        for (size_t f = 0; f < face_count; ++f)
            for (int c = 0; c < 3; ++c) {
                const uint32_t p = mesh.faces[f][c], q = mesh.faces[f][(c + 1) % 3];
                edges.push_back({(uint64_t(std::min(p, q)) << 32) | std::max(p, q), static_cast<uint32_t>(f)});
            }
        std::sort(edges.begin(), edges.end(), [](const Edge& a, const Edge& b) { return a.key < b.key; });
        std::vector<std::array<uint32_t, 3>> neighbors(face_count, {kNone, kNone, kNone});
        for (size_t e = 0; e + 1 < edges.size(); ++e)
            if (edges[e].key == edges[e + 1].key) {
                for (auto& slot : neighbors[edges[e].face])
                    if (slot == kNone) {
                        slot = edges[e + 1].face;
                        break;
                    }
                for (auto& slot : neighbors[edges[e + 1].face])
                    if (slot == kNone) {
                        slot = edges[e].face;
                        break;
                    }
            }
        edges = {};
        // Ring distance from the seams, within one view: a seam only levels its own view's faces.
        std::vector<uint8_t> depth(face_count, 255);
        std::vector<uint32_t> queue;
        queue.reserve(face_count / 8 + 1);
        for (const auto face : plan.sample_faces)
            if (face < face_count && depth[face] != 0) {
                depth[face] = 0;
                queue.push_back(face);
            }
        for (size_t head = 0; head < queue.size(); ++head) {
            const uint32_t face = queue[head];
            const int32_t label = face_labels[face];
            // The seeds are one face per seam point, not every face the point lies on: the extra ring completes the
            // seam's own faces, so the band that is left inside the boundary is the one the pixels ask for.
            if (label < 0 || size_t(label) >= labels || int(depth[face]) >= rings[size_t(label)] + 1)
                continue;
            for (const auto other : neighbors[face]) {
                if (other == kNone || face_labels[other] != label || depth[other] != 255)
                    continue;
                depth[other] = uint8_t(depth[face] + 1);
                queue.push_back(other);
            }
        }
        // A corner is free only where every face of its own view is in the band: the band's outer corners keep
        // their photo colors, which is the boundary condition the local solve needs.
        const size_t unknowns_total = plan.offsets.size() / 3;
        std::vector<uint8_t> flags(unknowns_total, 0);
        for (size_t f = 0; f < face_count; ++f) {
            const uint8_t bit = depth[f] == 255 ? 2 : 1;
            for (int c = 0; c < 3; ++c) {
                const uint32_t unknown = plan.corner_unknowns[3 * f + c];
                if (unknown < unknowns_total)
                    flags[unknown] |= bit;
            }
        }
        out.corner_unknowns.assign(face_count * 3, PatchSeamLeveling::kNoOffset);
        std::unordered_map<uint32_t, uint32_t> local_of;
        for (size_t f = 0; f < face_count; ++f) {
            if (depth[f] == 255)
                continue;
            ++out.band_faces;
            for (int c = 0; c < 3; ++c) {
                const uint32_t unknown = plan.corner_unknowns[3 * f + c];
                if (unknown >= unknowns_total || flags[unknown] != 1)
                    continue;
                const auto [it, inserted] = local_of.emplace(unknown, uint32_t(local_of.size()));
                (void)inserted;
                out.corner_unknowns[3 * f + c] = it->second;
            }
        }
        ThreeChannelSystem band;
        band.unknowns = local_of.size();
        band.diagonal.assign(band.unknowns, kPull);
        band.right.assign(band.unknowns * 3, 0.f);
        constexpr float kHalfEdge = .5f * kEdgeWeight;
        for (size_t f = 0; f < face_count; ++f) {
            if (depth[f] == 255)
                continue;
            const uint32_t* corner = &out.corner_unknowns[3 * f];
            for (int c = 0; c < 3; ++c) {
                const uint32_t a = corner[c], b = corner[(c + 1) % 3];
                if (a != PatchSeamLeveling::kNoOffset)
                    band.diagonal[a] += kHalfEdge;
                if (b != PatchSeamLeveling::kNoOffset)
                    band.diagonal[b] += kHalfEdge;
                if (a != PatchSeamLeveling::kNoOffset && b != PatchSeamLeveling::kNoOffset && a != b) {
                    band.entries.push_back({(uint64_t(a) << 32) | b, -kHalfEdge});
                    band.entries.push_back({(uint64_t(b) << 32) | a, -kHalfEdge});
                }
            }
        }
        // The seam steps the band has left to take up, on the gained colors.
        size_t seam_steps = 0;
        for (size_t first = 0; first < sample_count;) {
            size_t last = first;
            while (last < sample_count && plan.sample_points[last] == plan.sample_points[first])
                ++last;
            for (size_t a = first; a < last; ++a)
                for (size_t b = a + 1; b < last; ++b) {
                    const int32_t la = plan.sample_labels[a], lb = plan.sample_labels[b];
                    if (la < 0 || lb < 0 || la == lb || size_t(la) >= labels || size_t(lb) >= labels)
                        continue;
                    const auto ia = local_of.find(plan.sample_unknowns[a]);
                    const auto ib = local_of.find(plan.sample_unknowns[b]);
                    const bool free_a = ia != local_of.end(), free_b = ib != local_of.end();
                    if (!free_a && !free_b)
                        continue;
                    std::array<float, 3> step{};
                    bool valid = true;
                    for (size_t c = 0; c < 3 && valid; ++c) {
                        const float ca = colors[3 * a + c];
                        const float cb = colors[3 * b + c];
                        if (!(ca >= 0.f) || !(cb >= 0.f))
                            valid = false;
                        else
                            step[c] = out.gains[3 * size_t(lb) + c] * cb - out.gains[3 * size_t(la) + c] * ca;
                    }
                    if (!valid)
                        continue;
                    ++seam_steps;
                    if (free_a) {
                        const uint32_t ua = ia->second;
                        band.diagonal[ua] += kSeamWeight;
                        for (size_t c = 0; c < 3; ++c)
                            band.right[3 * size_t(ua) + c] += kSeamWeight * step[c];
                        if (free_b) {
                            const uint32_t ub = ib->second;
                            band.entries.push_back({(uint64_t(ua) << 32) | ub, -kSeamWeight});
                            band.entries.push_back({(uint64_t(ub) << 32) | ua, -kSeamWeight});
                        }
                    }
                    if (free_b) {
                        const uint32_t ub = ib->second;
                        band.diagonal[ub] += kSeamWeight;
                        for (size_t c = 0; c < 3; ++c)
                            band.right[3 * size_t(ub) + c] -= kSeamWeight * step[c];
                    }
                }
            first = last;
        }
        out.offsets = band.solve(out.iterations);
        LOG_INFO("Mesh: OpenMVS seam leveling, {} gain steps over {} views (gains {:.2f} to {:.2f}, {} iterations), {} steps in a band of {} faces, "
                 "{} unknowns, {} iterations",
                 gain_steps, labels, lowest, highest, gain_iterations, seam_steps, out.band_faces, band.unknowns, out.iterations);
        return out;
    }

    void finish_patch_page(uint8_t* page, const int side, const PatchPage& page_patches, const std::span<const PatchPlacement> placements,
                           const int rings) {
        if (!page || side <= 0)
            return;
        // What the page's patches fill, and what their rings reach into: filled outward, so that bilinear
        // samples and reductions reach no background, exactly as a chart page is composed.
        std::vector<uint8_t> state(size_t(side) * side, 0);
        for (const auto index : page_patches.patches) {
            const auto& place = placements[index];
            for (int y = std::max(0, place.y); y < std::min(side, place.y + place.height); ++y) {
                const int x0 = std::max(0, place.x), x1 = std::min(side, place.x + place.width);
                if (x1 > x0)
                    std::memset(&state[size_t(y) * side + x0], 1, size_t(x1 - x0));
            }
        }
        const auto neighbors = [&](const size_t at, auto&& visit) {
            const size_t x = at % size_t(side), y = at / size_t(side);
            for (int dy = -1; dy <= 1; ++dy)
                for (int dx = -1; dx <= 1; ++dx) {
                    if ((dx == 0 && dy == 0) || (dx < 0 && x == 0) || (dy < 0 && y == 0) || (dx > 0 && x + 1 == size_t(side)) ||
                        (dy > 0 && y + 1 == size_t(side)))
                        continue;
                    visit(size_t(ptrdiff_t(y) + dy) * side + size_t(ptrdiff_t(x) + dx));
                }
        };
        std::vector<uint32_t> frontier, next;
        for (size_t at = 0; at < state.size(); ++at)
            if (state[at] == 1)
                neighbors(at, [&](const size_t other) {
                    if (state[other] == 0) {
                        state[other] = 2;
                        frontier.push_back(static_cast<uint32_t>(other));
                    }
                });
        for (int ring = 0; ring < rings && !frontier.empty(); ++ring) {
            next.clear();
            for (const auto entry : frontier) {
                const size_t at = entry;
                bool taken = false;
                neighbors(at, [&](const size_t other) {
                    if (!taken && state[other] == 1) {
                        std::memcpy(&page[3 * at], &page[3 * other], 3);
                        taken = true;
                    }
                });
                state[at] = 1;
                neighbors(at, [&](const size_t other) {
                    if (state[other] == 0) {
                        state[other] = 2;
                        next.push_back(static_cast<uint32_t>(other));
                    }
                });
            }
            frontier.swap(next);
        }
    }

    void apply_patch_layout(SplatMesh& mesh, const PatchSet& patches, const PatchAtlas& atlas, std::span<const PatchView> views,
                           const int color_pages) {
        const int side = std::max(mesh.page_size, 1);
        const size_t atlas_pages = atlas.pages.size();
        const size_t page_count = atlas_pages + size_t(std::max(color_pages, 0));
        const bool has_colors = !mesh.colors.empty();
        const bool has_error = !mesh.face_error.empty();
        // One vertex per surface point per patch that repeats it, each with the UV that patch gives it: a
        // patch border cuts a vertex in two, and its two sides carry their own texture. A leftover face has
        // its own slot, so a vertex two of them share splits as well.
        std::unordered_map<uint64_t, int32_t> seen;
        seen.reserve(mesh.vertices.size() * 2 + 1);
        std::vector<std::array<float, 3>> vertices;
        std::vector<std::array<uint8_t, 3>> colors;
        std::vector<std::array<float, 2>> uvs;
        std::vector<int32_t> uv_source;
        std::vector<std::array<int32_t, 3>> faces;
        std::vector<float> face_error;
        vertices.reserve(mesh.vertices.size());
        uvs.reserve(mesh.vertices.size());
        uv_source.reserve(mesh.vertices.size());
        faces.reserve(mesh.faces.size());
        if (has_colors)
            colors.reserve(mesh.vertices.size());
        if (has_error)
            face_error.reserve(mesh.faces.size());
        const auto corner_of = [&](const uint64_t key, const std::array<float, 2>& uv, const int32_t source) {
            const auto [it, inserted] = seen.emplace(key, static_cast<int32_t>(vertices.size()));
            if (inserted) {
                vertices.push_back(mesh.vertices[size_t(source)]);
                uvs.push_back(uv);
                uv_source.push_back(source);
                if (has_colors)
                    colors.push_back(mesh.colors[size_t(source)]);
            }
            return it->second;
        };
        std::vector<size_t> page_faces(page_count, 0);
        // The patch pages, in the packed order the placements give.
        for (size_t p = 0; p < atlas_pages; ++p) {
            for (const auto index : atlas.pages[p].patches) {
                const auto& patch = patches.patches[index];
                const auto& place = atlas.placements[index];
                const PatchView& view = views[size_t(patch.view)];
                const double scale = place.scale > 0.f ? double(place.scale) : 1.;
                for (const auto face : patch.faces) {
                    const auto& corners = mesh.faces[size_t(face)];
                    std::array<int32_t, 3> out{};
                    for (int c = 0; c < 3; ++c) {
                        float pixel[2];
                        std::array<float, 2> uv{float(place.x + place.width / 2) / side, float(place.y + place.height / 2) / side};
                        if (project_patch_point(view, mesh.vertices[size_t(corners[c])].data(), pixel)) {
                            uv[0] = float((double(place.x) + (double(pixel[0]) - place.source_x) * scale) / side);
                            uv[1] = float((double(place.y) + (double(pixel[1]) - place.source_y) * scale) / side);
                        }
                        out[c] = corner_of((uint64_t(uint32_t(corners[c])) << 32) | uint32_t(index + 1), uv, corners[c]);
                    }
                    faces.push_back(out);
                    if (has_error)
                        face_error.push_back(mesh.face_error[size_t(face)]);
                    ++page_faces[p];
                }
            }
        }
        // The color pages: each leftover face in a cell of its own, painted with its own vertex colors and
        // sampled at its cell's triangle, so the page shows the vertex colors the mesh had.
        const int cols = std::max(1, side / kColorCellStep);
        const size_t cells_per_page = size_t(cols) * cols;
        size_t cell = 0;
        if (color_pages > 0 && !mesh.pages.empty())
            for (size_t f = 0; f < patches.face_patch.size(); ++f) {
                if (patches.face_patch[f] >= 0)
                    continue;
                const size_t page = std::min(atlas_pages + cell / cells_per_page, mesh.pages.size() - 1);
                const size_t in_page = cell % cells_per_page;
                const int ox = kColorCellOrigin + kColorCellStep * int(in_page % size_t(cols));
                const int oy = kColorCellOrigin + kColorCellStep * int(in_page / size_t(cols));
                const auto& corners = mesh.faces[f];
                uint8_t paint[3] = {128, 128, 128};
                if (has_colors) {
                    double sum[3] = {};
                    for (const auto corner : corners) {
                        const auto& color = mesh.colors[size_t(corner)];
                        for (int c = 0; c < 3; ++c)
                            sum[c] += color[c];
                    }
                    for (int c = 0; c < 3; ++c)
                        paint[c] = static_cast<uint8_t>(std::clamp(std::lround(sum[c] / 3.), 0l, 255l));
                }
                auto* image = mesh.pages[page].data();
                for (int dy = 0; dy < 2; ++dy)
                    for (int dx = 0; dx < 2; ++dx)
                        std::memcpy(&image[3 * (size_t(oy + dy) * side + (ox + dx))], paint, 3);
                std::array<int32_t, 3> out{};
                for (int c = 0; c < 3; ++c) {
                    const std::array<float, 2> uv{float(ox + (c == 1 ? 1.5f : .5f)) / side, float(oy + (c == 2 ? 1.5f : .5f)) / side};
                    out[c] = corner_of((uint64_t(uint32_t(corners[c])) << 32) | (0x80000000u + uint32_t(f)), uv, corners[c]);
                }
                faces.push_back(out);
                if (has_error)
                    face_error.push_back(mesh.face_error[f]);
                ++page_faces[page];
                ++cell;
            }
        // The page runs: the faces are already in page order.
        mesh.page_first_face.assign(page_count + 1, 0);
        for (size_t p = 0; p < page_count; ++p)
            mesh.page_first_face[p + 1] = mesh.page_first_face[p] + page_faces[p];
        mesh.vertices = std::move(vertices);
        mesh.colors = std::move(colors);
        mesh.uvs = std::move(uvs);
        mesh.uv_source = std::move(uv_source);
        mesh.faces = std::move(faces);
        if (has_error)
            mesh.face_error = std::move(face_error);
        else
            mesh.face_error.clear();
        compute_splat_mesh_normals(mesh);
    }
} // namespace lfs::training
