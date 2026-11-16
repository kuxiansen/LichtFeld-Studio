/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "splat_mesh_patch_tiles.hpp"
#include "core/logger.hpp"
#include "splat_mesh_parallel.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <numeric>

namespace lfs::training {
    namespace {
        constexpr int kPatchThreads = 8;
        // A leftover face's cell on a color page: 2x2 texels, one of gap, four to the next cell.
        constexpr int kColorCellStep = 4;
        // Padding keeps four texels clear around every rect, so eight rings cover a page's gaps from their edges.
        constexpr int kRectFillRings = 8;

        // One level's atlas while it is baked, kept beside the levels so that a photo serves them all.
        struct LevelWork {
            SplatMesh* mesh = nullptr;
            const std::vector<int32_t>* labels = nullptr;
            PatchSet patches;
            PatchAtlas atlas;
            int side = 0;
            int color_pages = 0;
            size_t page_count = 0;
            size_t painted = 0;
            ColorLeveling leveling;
            std::vector<std::vector<uint32_t>> patches_of_view;
            std::vector<std::vector<uint32_t>> seams_of_view;
            std::vector<float> seam_colors;
            std::atomic<uint64_t> observed{0};
        };
    } // namespace

    std::vector<PatchTiledPages> bake_patch_tiled_levels(std::vector<PatchTiledLevel>& levels, const std::span<const PatchView> views,
                                                         const PatchTiledSettings& settings, const PatchPhotoSource& photo_source,
                                                         const std::function<void(float)>& report, const std::stop_token stop) {
        const auto step_start = std::chrono::steady_clock::now();
        const int side = std::clamp(settings.page_size, 16, 8192);
        std::vector<LevelWork> work(levels.size());
        size_t total_patches = 0;
        for (size_t i = 0; i < levels.size(); ++i) {
            if (stop.stop_requested())
                throw core::OperationCancelled();
            auto& level = levels[i];
            auto& mesh = level.mesh;
            if (mesh.faces.empty())
                continue;
            auto& w = work[i];
            w.mesh = &mesh;
            w.labels = &level.face_views;
            w.side = side;
            w.patches = build_mesh_patches(mesh, level.face_views, views, stop);
            w.atlas = pack_mesh_patches(mesh, w.patches, views,
                                        {.page_size = side,
                                         .padding = settings.padding,
                                         .texel_size = level.texel,
                                         .texel_min = settings.texel_min},
                                        stop);
            for (const auto patch : w.patches.face_patch)
                w.painted += patch >= 0 ? 1 : 0;
            const size_t cells_per_page = size_t(std::max(1, side / kColorCellStep)) * size_t(std::max(1, side / kColorCellStep));
            const size_t leftover = w.patches.face_patch.size() - w.painted;
            w.color_pages = leftover > 0 ? int((leftover + cells_per_page - 1) / cells_per_page) : 0;
            w.page_count = w.atlas.pages.size() + size_t(w.color_pages);
            mesh.page_size = side;
            mesh.pages.assign(w.page_count, std::vector<uint8_t>(size_t(side) * size_t(side) * 3, 128));
            w.patches_of_view.resize(views.size());
            for (size_t p = 0; p < w.patches.patches.size(); ++p) {
                const int32_t view = w.patches.patches[p].view;
                if (view >= 0 && size_t(view) < w.patches_of_view.size())
                    w.patches_of_view[size_t(view)].push_back(static_cast<uint32_t>(p));
            }
            // The points the views share, read once their pages are baked and leveled afterwards.
            w.leveling = plan_color_leveling(mesh, level.face_views);
            const size_t samples = w.leveling.sample_labels.size();
            w.seams_of_view.resize(views.size());
            for (size_t s = 0; s < samples; ++s) {
                const int32_t label = w.leveling.sample_labels[s];
                if (label >= 0 && size_t(label) < w.seams_of_view.size())
                    w.seams_of_view[size_t(label)].push_back(static_cast<uint32_t>(s));
            }
            w.seam_colors.assign(samples * 3, -1.f);
            for (const auto& list : w.patches_of_view)
                total_patches += list.size();
            LOG_INFO("Mesh: level {} of tile {}: {} patches paint {} of {} faces, {:.2f} M texels of {} pages",
                     level.level, level.tile, w.patches.patches.size(), w.painted, level.face_views.size(),
                     double(w.atlas.texels) / 1e6, w.page_count);
        }

        // One pass over the photos: each is read once for every level that has patches of it.
        size_t done = 0;
        for (size_t v = 0; v < views.size(); ++v) {
            if (stop.stop_requested())
                throw core::OperationCancelled();
            bool used = false;
            for (const auto& w : work)
                used = used || (!w.patches_of_view.empty() &&
                                (!w.patches_of_view[v].empty() || !w.seams_of_view[v].empty()));
            if (!used)
                continue;
            PatchBakePhoto photo;
            if (photo_source(v, photo)) {
                for (auto& w : work) {
                    if (w.mesh == nullptr)
                        continue;
                    const float* face_error = w.mesh->face_error.empty() ? nullptr : w.mesh->face_error.data();
                    if (!w.seams_of_view[v].empty())
                        sample_patch_seam_colors(w.leveling, w.seams_of_view[v], static_cast<int32_t>(v), views[v], photo,
                                                 face_error, settings.base_tolerance, w.seam_colors);
                    const auto& list = w.patches_of_view[v];
                    detail::parallel_for(list.size(), kPatchThreads, [&](const size_t k) {
                        const size_t p = list[k];
                        const auto& place = w.atlas.placements[p];
                        if (place.page < 0 || size_t(place.page) >= w.page_count)
                            return;
                        auto rect = bake_patch_texture(*w.mesh, w.patches.patches[p], place, views[v], photo, face_error,
                                                       settings.base_tolerance);
                        for (const auto value : rect.valid)
                            if (value != 0)
                                w.observed.fetch_add(1, std::memory_order_relaxed);
                        fill_patch_rect(rect, *w.mesh, w.patches.patches[p]);
                        apply_patch_texture(w.mesh->pages[size_t(place.page)].data(), w.side, place, rect);
                    });
                    done += list.size();
                }
            } else {
                for (const auto& w : work)
                    if (!w.patches_of_view.empty())
                        done += w.patches_of_view[v].size();
            }
            report(float(done) / float(std::max<size_t>(total_patches, 1)));
        }

        // The seams take up their steps on the pages before each level is laid out around them.
        std::vector<PatchTiledPages> textures(levels.size());
        for (size_t i = 0; i < levels.size(); ++i) {
            auto& w = work[i];
            if (w.mesh == nullptr) {
                textures[i].texel = levels[i].texel;
                continue;
            }
            auto& mesh = *w.mesh;
            const size_t samples = w.leveling.sample_labels.size();
            if (samples > 0) {
                std::vector<uint32_t> seam_unknowns;
                std::vector<float> seam_offsets, gains;
                if (settings.seam_leveling_mode == 0) {
                    solve_color_leveling(w.leveling, mesh, w.seam_colors);
                    w.leveling.release_samples();
                    seam_unknowns = std::move(w.leveling.corner_unknowns);
                    seam_offsets = std::move(w.leveling.offsets);
                } else {
                    auto seam = solve_patch_seam_leveling(mesh, *w.labels, views, w.leveling, w.seam_colors);
                    w.leveling.release_samples();
                    seam_unknowns = std::move(seam.corner_unknowns);
                    seam_offsets = std::move(seam.offsets);
                    gains = std::move(seam.gains);
                }
                detail::parallel_for(w.patches.patches.size(), kPatchThreads, [&](const size_t p) {
                    const auto& place = w.atlas.placements[p];
                    const int32_t label = w.patches.patches[p].view;
                    if (place.page < 0 || size_t(place.page) >= w.page_count || label < 0 ||
                        size_t(label) >= views.size())
                        return;
                    auto* page = mesh.pages[size_t(place.page)].data();
                    if (!gains.empty())
                        apply_patch_gain(page, w.side, place, &gains[3 * size_t(label)]);
                    apply_patch_offsets(page, w.side, mesh, w.patches.patches[p], place, views[size_t(label)],
                                        seam_unknowns, seam_offsets);
                });
            } else {
                w.leveling.release_samples();
            }
            apply_patch_layout(mesh, w.patches, w.atlas, views, w.color_pages);
            detail::parallel_for(w.atlas.pages.size(), kPatchThreads, [&](const size_t p) {
                finish_patch_page(mesh.pages[p].data(), w.side, w.atlas.pages[p], w.atlas.placements, kRectFillRings);
            });
            // The distance a full-detail page covers in pixels: the texels over the surface the patches cover.
            double texel_sum = 0., area_sum = 0.;
            for (size_t p = 0; p < w.patches.patches.size(); ++p) {
                const auto& patch = w.patches.patches[p];
                const auto& place = w.atlas.placements[p];
                if (place.page < 0 || !(patch.area > 0.))
                    continue;
                texel_sum += double(place.texel) * patch.area;
                area_sum += patch.area;
            }
            textures[i].pages = w.page_count;
            textures[i].charts = w.patches.patches.size();
            textures[i].texels = w.atlas.texels;
            textures[i].observed_texel_share = w.atlas.texels > 0 ? float(w.observed.load()) / float(w.atlas.texels) : 0.f;
            textures[i].texel = area_sum > 0. ? float(texel_sum / area_sum) : levels[i].texel;
            LOG_INFO("Mesh: level {} textured on {} pages, {} patches, {:.1f}% of their texels seen by the patch's own view",
                     levels[i].level, w.page_count, w.patches.patches.size(), 100. * textures[i].observed_texel_share);
        }
        LOG_INFO("Mesh: {} tile levels textured in {:.1f} s", levels.size(), std::chrono::duration<double>(std::chrono::steady_clock::now() - step_start).count());
        return textures;
    }

    SplatMesh combine_patch_tiled_levels(const std::vector<PatchTiledLevel>& levels) {
        SplatMesh combined;
        size_t vertices = 0, faces = 0;
        for (const auto& level : levels) {
            vertices += level.mesh.vertices.size();
            faces += level.mesh.faces.size();
        }
        combined.vertices.reserve(vertices);
        combined.normals.reserve(vertices);
        combined.colors.reserve(vertices);
        combined.faces.reserve(faces);
        combined.face_error.reserve(faces);
        for (const auto& level : levels) {
            const auto& mesh = level.mesh;
            const auto base_vertex = static_cast<int32_t>(combined.vertices.size());
            combined.vertices.insert(combined.vertices.end(), mesh.vertices.begin(), mesh.vertices.end());
            combined.normals.insert(combined.normals.end(), mesh.normals.begin(), mesh.normals.end());
            combined.colors.insert(combined.colors.end(), mesh.colors.begin(), mesh.colors.end());
            for (size_t f = 0; f < mesh.faces.size(); ++f) {
                const auto& face = mesh.faces[f];
                combined.faces.push_back({base_vertex + face[0], base_vertex + face[1], base_vertex + face[2]});
                combined.face_error.push_back(mesh.face_error.empty() ? 0.f : mesh.face_error[f]);
            }
        }
        return combined;
    }

    void assemble_patch_tiled_mesh(std::vector<PatchTiledLevel>& levels, const std::span<const PatchTiledPages> textures,
                                   SplatMesh& base) {
        // Levels first, so that the finest levels' pages, which the viewer's base mesh shows, come first.
        std::vector<size_t> order(levels.size());
        std::iota(order.begin(), order.end(), size_t{0});
        std::stable_sort(order.begin(), order.end(), [&](const size_t a, const size_t b) {
            if (levels[a].level != levels[b].level)
                return levels[a].level < levels[b].level;
            return levels[a].tile < levels[b].tile;
        });
        TiledMesh tiled;
        auto& all = tiled.combined;
        all = {};
        tiled.levels.clear();
        size_t vertices = 0, faces = 0, page_count = 0;
        for (const auto index : order) {
            const auto& mesh = levels[index].mesh;
            vertices += mesh.vertices.size();
            faces += mesh.faces.size();
            page_count += mesh.page_first_face.size() > 0 ? mesh.page_first_face.size() - 1 : 0;
        }
        all.vertices.reserve(vertices);
        all.normals.reserve(vertices);
        all.colors.reserve(vertices);
        all.uvs.reserve(vertices);
        all.uv_source.reserve(vertices);
        all.faces.reserve(faces);
        all.face_error.reserve(faces);
        all.page_first_face.reserve(page_count + 1);
        all.page_sizes.reserve(page_count);
        all.page_first_face.push_back(0);
        int32_t source_base = 0;
        int page_size = 0;
        size_t base_vertices = 0, base_faces = 0, base_pages = 0;
        for (const auto index : order) {
            auto& level = levels[index];
            auto& mesh = level.mesh;
            TileLevel entry;
            entry.tile = level.tile;
            entry.level = level.level;
            entry.first_page = all.page_first_face.size() - 1;
            entry.page_count = mesh.page_first_face.size() > 0 ? mesh.page_first_face.size() - 1 : 0;
            entry.texel = textures[index].texel > 0.f ? textures[index].texel : level.texel;
            entry.center = level.center;
            entry.radius = level.radius;
            entry.first_vertex = all.vertices.size();
            entry.vertex_count = mesh.vertices.size();
            tiled.levels.push_back(entry);
            const auto base_vertex = static_cast<int32_t>(all.vertices.size());
            all.vertices.insert(all.vertices.end(), mesh.vertices.begin(), mesh.vertices.end());
            all.normals.insert(all.normals.end(), mesh.normals.begin(), mesh.normals.end());
            all.colors.insert(all.colors.end(), mesh.colors.begin(), mesh.colors.end());
            all.uvs.insert(all.uvs.end(), mesh.uvs.begin(), mesh.uvs.end());
            int32_t top = 0;
            for (const auto source : mesh.uv_source) {
                all.uv_source.push_back(source_base + source);
                top = std::max(top, source);
            }
            source_base += top + 1;
            for (size_t p = 0; p + 1 < mesh.page_first_face.size(); ++p) {
                for (size_t f = mesh.page_first_face[p]; f < mesh.page_first_face[p + 1]; ++f) {
                    const auto& face = mesh.faces[f];
                    all.faces.push_back({base_vertex + face[0], base_vertex + face[1], base_vertex + face[2]});
                    all.face_error.push_back(mesh.face_error.empty() ? 0.f : mesh.face_error[f]);
                }
                all.page_first_face.push_back(all.faces.size());
                all.page_sizes.push_back(mesh.page_side(p));
                if (p < mesh.pages.size())
                    all.pages.push_back(std::move(mesh.pages[p]));
                else
                    all.pages.emplace_back();
            }
            page_size = std::max(page_size, mesh.page_size);
            all.charts += textures[index].charts;
            if (level.level == 0) {
                // Level-major, so the finest levels' faces and pages are a prefix of the combined mesh.
                base_vertices = all.vertices.size();
                base_faces = all.faces.size();
                base_pages = all.page_sizes.size();
            }
            mesh = {};
        }
        all.page_size = page_size;
        double observed = 0.;
        for (size_t i = 0; i < levels.size(); ++i) {
            all.texels += textures[i].texels;
            observed += double(textures[i].observed_texel_share) * double(textures[i].texels);
        }
        all.observed_texel_share = all.texels > 0 ? float(observed / double(all.texels)) : 0.f;

        // The base mesh: the finest levels, with their pages kept in the tiled model rather than repeated here.
        base.vertices.assign(all.vertices.begin(), all.vertices.begin() + std::ptrdiff_t(base_vertices));
        base.normals.assign(all.normals.begin(), all.normals.begin() + std::ptrdiff_t(base_vertices));
        base.colors.assign(all.colors.begin(), all.colors.begin() + std::ptrdiff_t(base_vertices));
        base.uvs.assign(all.uvs.begin(), all.uvs.begin() + std::ptrdiff_t(base_vertices));
        base.uv_source.assign(all.uv_source.begin(), all.uv_source.begin() + std::ptrdiff_t(base_vertices));
        base.faces.assign(all.faces.begin(), all.faces.begin() + std::ptrdiff_t(base_faces));
        base.face_error.assign(all.face_error.begin(), all.face_error.begin() + std::ptrdiff_t(base_faces));
        base.page_first_face.assign(all.page_first_face.begin(), all.page_first_face.begin() + std::ptrdiff_t(base_pages) + 1);
        base.page_sizes.assign(all.page_sizes.begin(), all.page_sizes.begin() + std::ptrdiff_t(base_pages));
        base.page_size = page_size;
        base.pages.clear();
        base.pages_in_tiled = true;

        size_t finest_faces = 0;
        for (const auto& level : tiled.levels)
            if (level.level == 0)
                finest_faces += all.page_first_face[level.first_page + level.page_count] - all.page_first_face[level.first_page];
        LOG_INFO("Mesh: {} tile levels, {} faces on the finest level and {} in all, on {} pages", tiled.levels.size(),
                 finest_faces, all.faces.size(), all.page_sizes.size());
        base.tiled = std::make_shared<TiledMesh>(std::move(tiled));
    }

} // namespace lfs::training
