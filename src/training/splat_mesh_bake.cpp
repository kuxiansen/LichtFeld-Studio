/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "splat_mesh_bake.hpp"
#include "splat_mesh_patch.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
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
