/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "splat_mesh_pack.hpp"
#include "core/collision_volume.hpp"
#include "core/logger.hpp"
#include "core/progress.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <format>
#include <limits>
#include <utility>

namespace lfs::training {
    namespace {
        constexpr int kMaxPage = 8192;
        constexpr float kDefaultTexelMin = .001f;
        // A patch this oblique to its view says nothing about the detail there.
        constexpr float kMinCosine = .2f;

        struct Rect {
            int x = 0, y = 0, w = 0, h = 0;
        };

        [[nodiscard]] bool contains(const Rect& outer, const Rect& inner) {
            return inner.x >= outer.x && inner.y >= outer.y && inner.x + inner.w <= outer.x + outer.w && inner.y + inner.h <= outer.y + outer.h;
        }

        [[nodiscard]] bool intersects(const Rect& a, const Rect& b) {
            return a.x < b.x + b.w && b.x < a.x + a.w && a.y < b.y + b.h && b.y < a.y + a.h;
        }

        // MaxRects, best short side fit: each rect goes where the free space it leaves is least lopsided.
        class Packer {
        public:
            explicit Packer(const int size) : size_(size), free_(1, Rect{0, 0, size, size}) {}

            [[nodiscard]] bool insert(const int w, const int h, Rect& placed) {
                int best = -1;
                long best_short = std::numeric_limits<long>::max(), best_long = std::numeric_limits<long>::max();
                for (size_t i = 0; i < free_.size(); ++i) {
                    const auto& space = free_[i];
                    if (space.w < w || space.h < h)
                        continue;
                    const long short_side = std::min(space.w - w, space.h - h), long_side = std::max(space.w - w, space.h - h);
                    if (short_side < best_short || (short_side == best_short && long_side < best_long)) {
                        best_short = short_side;
                        best_long = long_side;
                        best = int(i);
                    }
                }
                if (best < 0)
                    return false;
                placed = {free_[size_t(best)].x, free_[size_t(best)].y, w, h};
                split(placed);
                return true;
            }

            [[nodiscard]] int size() const { return size_; }

        private:
            void split(const Rect& used) {
                std::vector<Rect> next;
                next.reserve(free_.size() + 4);
                for (const auto& space : free_) {
                    if (!intersects(space, used)) {
                        next.push_back(space);
                        continue;
                    }
                    if (used.x > space.x)
                        next.push_back({space.x, space.y, used.x - space.x, space.h});
                    if (used.x + used.w < space.x + space.w)
                        next.push_back({used.x + used.w, space.y, space.x + space.w - (used.x + used.w), space.h});
                    if (used.y > space.y)
                        next.push_back({space.x, space.y, space.w, used.y - space.y});
                    if (used.y + used.h < space.y + space.h)
                        next.push_back({space.x, used.y + used.h, space.w, space.y + space.h - (used.y + used.h)});
                }
                free_.clear();
                for (size_t i = 0; i < next.size(); ++i) {
                    const auto& space = next[i];
                    if (space.w <= 0 || space.h <= 0)
                        continue;
                    bool redundant = false;
                    for (size_t j = 0; j < next.size() && !redundant; ++j) {
                        const auto& other = next[j];
                        if (i == j || other.w <= 0 || other.h <= 0 || !contains(other, space))
                            continue;
                        // Two equal rects survive once, the later of them.
                        redundant = !(other.x == space.x && other.y == space.y && other.w == space.w && other.h == space.h) || i > j;
                    }
                    if (!redundant)
                        free_.push_back(space);
                }
            }

            int size_ = 0;
            std::vector<Rect> free_;
        };

        // What one photo pixel covers on the face, in scene units: range / (pixels per radian * the cosine between the
        // surface and the ray). Pixels per radian is the fisheye polynomial's slope, or the pinhole's tangent.
        [[nodiscard]] float face_gsd(const SplatMesh& mesh, const size_t face, const PatchView& view) {
            const auto& a = mesh.vertices[size_t(mesh.faces[face][0])];
            const auto& b = mesh.vertices[size_t(mesh.faces[face][1])];
            const auto& c = mesh.vertices[size_t(mesh.faces[face][2])];
            const double u[3] = {double(b[0]) - a[0], double(b[1]) - a[1], double(b[2]) - a[2]};
            const double v[3] = {double(c[0]) - a[0], double(c[1]) - a[1], double(c[2]) - a[2]};
            const double n[3] = {u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]};
            const double twice = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
            if (!(twice > 0.))
                return 0.f;
            const double middle[3] = {(double(a[0]) + b[0] + c[0]) / 3., (double(a[1]) + b[1] + c[1]) / 3., (double(a[2]) + b[2] + c[2]) / 3.};
            const double ray[3] = {middle[0] - view.center[0], middle[1] - view.center[1], middle[2] - view.center[2]};
            const double range = std::sqrt(ray[0] * ray[0] + ray[1] * ray[1] + ray[2] * ray[2]);
            if (!(range > 1e-3))
                return 0.f;
            const double cosine = std::abs(n[0] * ray[0] + n[1] * ray[1] + n[2] * ray[2]) / (twice * range);
            if (cosine < kMinCosine)
                return 0.f;
            const float* m = view.world_to_camera.data();
            const double x = m[0] * middle[0] + m[1] * middle[1] + m[2] * middle[2] + m[3];
            const double y = m[4] * middle[0] + m[5] * middle[1] + m[6] * middle[2] + m[7];
            const double z = m[8] * middle[0] + m[9] * middle[1] + m[10] * middle[2] + m[11];
            if (!(z > 0.))
                return 0.f;
            const double angle = std::atan2(std::hypot(x, y), z), square = angle * angle;
            const double focal = std::min(view.fx, view.fy);
            const double per_radian = view.model == 2
                ? focal * (1. + square * (3. * view.radial[0] + square * (5. * view.radial[1] + square * (7. * view.radial[2] + 9. * square * view.radial[3]))))
                : focal / (std::cos(angle) * std::cos(angle));
            if (!(per_radian > 0.))
                return 0.f;
            return float(range / (per_radian * cosine));
        }

        // The patch's area weighted median ground sample distance, or what its covered area over its pixels says where no
        // face is squarely seen.
        [[nodiscard]] float patch_gsd(const SplatMesh& mesh, const MeshPatch& patch, const PatchView& view) {
            std::vector<std::pair<float, double>> samples;
            samples.reserve(patch.faces.size());
            double total = 0.;
            for (const auto face : patch.faces) {
                const auto& a = mesh.vertices[size_t(mesh.faces[face][0])];
                const auto& b = mesh.vertices[size_t(mesh.faces[face][1])];
                const auto& c = mesh.vertices[size_t(mesh.faces[face][2])];
                const double u[3] = {double(b[0]) - a[0], double(b[1]) - a[1], double(b[2]) - a[2]};
                const double v[3] = {double(c[0]) - a[0], double(c[1]) - a[1], double(c[2]) - a[2]};
                const double area = .5 * std::hypot(u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]);
                if (!(area > 0.))
                    continue;
                if (const float gsd = face_gsd(mesh, size_t(face), view); gsd > 0.f) {
                    samples.emplace_back(gsd, area);
                    total += area;
                }
            }
            if (samples.empty()) {
                const double pixels = std::max(1., double(patch.high[0] - patch.low[0] + 1.)) * std::max(1., double(patch.high[1] - patch.low[1] + 1.));
                return patch.area > 0. ? float(std::sqrt(patch.area / pixels)) : 0.f;
            }
            std::ranges::sort(samples, {}, &std::pair<float, double>::first);
            double seen = 0.;
            for (const auto& [gsd, area] : samples) {
                seen += area;
                if (seen >= .5 * total)
                    return gsd;
            }
            return samples.back().first;
        }

        // One patch's rect on a page, before packing.
        struct Planned {
            size_t patch = 0;
            int wide = 1, tall = 1;
            int source_x = 0, source_y = 0;
            float texel = 0.f;
            int bucket = 0;
        };

        [[nodiscard]] float median_of(std::vector<float>& values) {
            if (values.empty())
                return 0.f;
            const auto middle = values.begin() + std::ptrdiff_t(values.size() / 2);
            std::nth_element(values.begin(), middle, values.end());
            return *middle;
        }
    } // namespace

    PatchAtlas pack_mesh_patches(const SplatMesh& mesh, const PatchSet& patches, const std::span<const PatchView> views,
                                 const PatchPackSettings& settings, const std::stop_token stop) {
        using Clock = std::chrono::steady_clock;
        const auto start = Clock::now();
        PatchAtlas atlas;
        atlas.placements.resize(patches.patches.size());
        if (patches.patches.empty() || views.empty())
            return atlas;

        const int page_size = std::clamp(settings.page_size, 16, kMaxPage);
        const int padding = std::clamp(settings.padding, 0, page_size / 4);
        const int usable = page_size - 2 * padding;
        const float texel_min = settings.texel_min > 0.f ? settings.texel_min : kDefaultTexelMin;

        std::vector<Planned> planned(patches.patches.size());
        std::vector<float> copied_texels, resampled_texels;
        for (size_t i = 0; i < patches.patches.size(); ++i) {
            if (stop.stop_requested())
                throw core::OperationCancelled();
            auto& placement = atlas.placements[i];
            const auto& patch = patches.patches[i];
            placement.warp = patch.warp;
            placement.source_x = int(std::floor(patch.low[0]));
            placement.source_y = int(std::floor(patch.low[1]));
            const float span_x = std::max(.0f, patch.high[0] - patch.low[0]);
            const float span_y = std::max(.0f, patch.high[1] - patch.low[1]);
            const float gsd = patch_gsd(mesh, patch, views[size_t(patch.view)]);
            float texel = gsd;
            float scale = 1.f;
            if (patch.warp) {
                // A resampled patch can be drawn at the texel the surface asks for, as long as that is not finer than the
                // smallest allowed one; a photo coarser than that would only be enlarged.
                if (gsd > 0.f && settings.texel_size > 0.f)
                    texel = std::clamp(gsd, texel_min, std::max(settings.texel_size, texel_min));
                else if (gsd > 0.f)
                    texel = std::max(gsd, texel_min);
                if (!(texel > 0.f))
                    texel = texel_min;
                scale = texel > 0.f ? gsd / texel : 1.f;
                if (!(scale > 0.f))
                    scale = 1.f;
            }
            planned[i] = {.patch = i,
                          .wide = std::max(1, int(std::ceil(span_x * scale)) + 1),
                          .tall = std::max(1, int(std::ceil(span_y * scale)) + 1),
                          .source_x = placement.source_x,
                          .source_y = placement.source_y,
                          .texel = texel > 0.f ? texel : kDefaultTexelMin,
                          .bucket = int(std::floor(std::log2(std::max(texel, kDefaultTexelMin))))};
            // A rect that cannot fit a page is drawn coarser until it does.
            if (planned[i].wide > usable || planned[i].tall > usable) {
                const float shrink = float(usable) / float(std::max(planned[i].wide, planned[i].tall));
                planned[i].wide = std::max(1, int(float(planned[i].wide) * shrink));
                planned[i].tall = std::max(1, int(float(planned[i].tall) * shrink));
                planned[i].texel /= shrink;
                planned[i].bucket = int(std::floor(std::log2(std::max(planned[i].texel, kDefaultTexelMin))));
                ++atlas.shrunk;
            }
            placement.texel = planned[i].texel;
            // What one photo pixel takes on the rect: on a copied patch the photo's own pixels are the texture,
            // texel for texel; on a resampled one the rect's own grid is the photo's window under this scale,
            // which the shrink above cuts with the rect it fits. The bake and the mesh's UVs both read it here,
            // so that a texel and its texture stay the same point of the surface.
            if (patch.warp) {
                const float scale_x = planned[i].wide > 1 ? float(planned[i].wide - 1) / std::max(span_x, 1e-6f) : 1e9f;
                const float scale_y = planned[i].tall > 1 ? float(planned[i].tall - 1) / std::max(span_y, 1e-6f) : 1e9f;
                placement.scale = std::min(scale_x, scale_y);
                if (!(placement.scale > 0.f) || !std::isfinite(placement.scale))
                    placement.scale = 1.f;
            } else
                placement.scale = 1.f;
            (patch.warp ? resampled_texels : copied_texels).push_back(planned[i].texel);
        }

        // Patches of one density share pages, so that a page can be mip-mapped and reduced as one.
        std::vector<int> buckets;
        buckets.reserve(planned.size());
        for (const auto& plan : planned)
            buckets.push_back(plan.bucket);
        std::ranges::sort(buckets);
        buckets.erase(std::unique(buckets.begin(), buckets.end()), buckets.end());

        for (const auto bucket : buckets) {
            std::vector<size_t> order;
            for (size_t i = 0; i < planned.size(); ++i)
                if (planned[i].bucket == bucket)
                    order.push_back(i);
            std::ranges::sort(order, [&](const size_t a, const size_t b) {
                return std::max(planned[a].wide, planned[a].tall) > std::max(planned[b].wide, planned[b].tall);
            });
            Packer packer(page_size);
            PatchPage page{.size = page_size, .patches = {}};
            for (const auto index : order) {
                if (stop.stop_requested())
                    throw core::OperationCancelled();
                Rect placed;
                if (!packer.insert(planned[index].wide + 2 * padding, planned[index].tall + 2 * padding, placed)) {
                    atlas.pages.push_back(std::move(page));
                    page = {.size = page_size, .patches = {}};
                    packer = Packer(page_size);
                    if (!packer.insert(planned[index].wide + 2 * padding, planned[index].tall + 2 * padding, placed))
                        throw core::collisionError("mesh.pack_failed", "Could not pack a mesh patch onto a page", ErrorCode::Internal,
                                                   std::format("{}x{} texels on a page of {}", planned[index].wide, planned[index].tall, page_size));
                }
                auto& placement = atlas.placements[index];
                placement.page = int(atlas.pages.size());
                placement.x = placed.x + padding;
                placement.y = placed.y + padding;
                placement.width = planned[index].wide;
                placement.height = planned[index].tall;
                page.patches.push_back(index);
                atlas.texels += size_t(planned[index].wide) * size_t(planned[index].tall);
                (placement.warp ? atlas.resampled : atlas.copies)++;
            }
            atlas.pages.push_back(std::move(page));
        }

        atlas.texel_copied = median_of(copied_texels);
        atlas.texel_resampled = median_of(resampled_texels);
        atlas.seconds = std::chrono::duration<double>(Clock::now() - start).count();
        const double page_texels = double(atlas.pages.size()) * double(page_size) * double(page_size);
        LOG_INFO("Mesh: {} patches packed onto {} pages of {} in {:.1f} s: {:.2f} M texels ({:.0f} % of the pages), "
                 "{} copied at a {:.2f} mm texel, {} resampled at {:.2f} mm; {} cut down to fit a page",
                 atlas.placements.size(), atlas.pages.size(), page_size, atlas.seconds, double(atlas.texels) / 1e6,
                 page_texels > 0. ? 100. * double(atlas.texels) / page_texels : 0., atlas.copies,
                 1000. * atlas.texel_copied, atlas.resampled, 1000. * atlas.texel_resampled, atlas.shrunk);
        return atlas;
    }
} // namespace lfs::training
