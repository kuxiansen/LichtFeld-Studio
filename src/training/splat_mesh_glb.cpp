/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "splat_mesh.hpp"

#include "core/collision_volume.hpp"
#include "core/native_fs.hpp"
#include "core/path_utils.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <format>
#include <fstream>
#include <nlohmann/json.hpp>
#include <stb_image_write.h>
#include <string>
#include <vector>

// Writes the mesh as a self-contained glTF 2.0 binary. One buffer holds the vertex data, the indices and one JPEG
// per texture page; no glTF library is needed for this, the container is the same JSON chunk plus binary chunk
// layout the SPZ export writes.
namespace lfs::training {
    namespace {
        using core::collisionError;
        using json = nlohmann::json;

        constexpr uint32_t kGlbMagic = 0x46546C67; // "glTF"
        constexpr uint32_t kChunkJson = 0x4E4F534A;
        constexpr uint32_t kChunkBin = 0x004E4942;
        constexpr int kJpegQuality = 95;

        // The scene is right-handed with y down and z forward, as the PLY and SPZ exports treat it; glTF is y up
        // with z back. Flipping y and z is a rotation, so triangle winding stays as it is.
        [[nodiscard]] std::array<float, 3> to_gltf(const std::array<float, 3>& v) { return {v[0], -v[1], -v[2]}; }

        // glTF vertex colors are linear; the mesh's are sRGB, as in the photos.
        [[nodiscard]] uint8_t to_linear(const uint8_t value) {
            const float s = float(value) / 255.f;
            const float linear = s <= .04045f ? s / 12.92f : std::pow((s + .055f) / 1.055f, 2.4f);
            return static_cast<uint8_t>(std::lround(255.f * linear));
        }

        [[nodiscard]] std::vector<uint8_t> encode_page_jpeg(const std::vector<uint8_t>& rgb, const int side,
                                                            const std::string& name) {
            std::vector<uint8_t> jpeg;
            const auto append = [](void* context, void* data, const int size) {
                auto* out = static_cast<std::vector<uint8_t>*>(context);
                const auto* bytes = static_cast<const uint8_t*>(data);
                out->insert(out->end(), bytes, bytes + size);
            };
            if (side <= 0 || rgb.size() < size_t(side) * size_t(side) * 3 ||
                !stbi_write_jpg_to_func(append, &jpeg, side, side, 3, rgb.data(), kJpegQuality))
                throw collisionError("mesh.write_failed", "Could not write the mesh texture", ErrorCode::Unavailable, name);
            return jpeg;
        }

        struct Block {
            size_t offset = 0;
            size_t size = 0;
        };

        struct PrimitiveRange {
            size_t page = 0;
            size_t first_face = 0;
            size_t face_count = 0;
            Block indices;
        };
    } // namespace

    void write_splat_mesh_glb(const SplatMesh& mesh, const std::filesystem::path& path) {
        const auto path_text = core::path_to_utf8(path);
        if (mesh.vertices.empty() || mesh.faces.empty())
            throw collisionError("mesh.write_failed", "The mesh has no faces to write", ErrorCode::Unavailable, path_text);

        // The atlas texture keeps its UVs and pages on the tiled model, not on the base mesh the extraction
        // returns; the affine texture keeps the finest levels on the base mesh and only its pages in the tiled
        // model. Take whichever holds the textured surface, and from the tiled model only the finest levels: a
        // GLB has no LOD, the coarser levels are the same surface simplified.
        const SplatMesh* surface = &mesh;
        const std::vector<std::vector<uint8_t>>* pages = nullptr;
        std::vector<size_t> surface_pages;
        if (mesh.page_first_face.size() > 1 && mesh.uvs.size() == mesh.vertices.size()) {
            const auto& source = mesh.pages_in_tiled && mesh.tiled ? mesh.tiled->combined.pages : mesh.pages;
            if (source.size() >= mesh.page_first_face.size() - 1) {
                pages = &source;
                surface_pages.resize(mesh.page_first_face.size() - 1);
                for (size_t p = 0; p < surface_pages.size(); ++p)
                    surface_pages[p] = p;
            }
        }
        if (pages == nullptr && mesh.tiled) {
            const auto& combined = mesh.tiled->combined;
            if (combined.page_first_face.size() > 1 && combined.uvs.size() == combined.vertices.size() &&
                combined.pages.size() >= combined.page_first_face.size() - 1) {
                surface = &combined;
                pages = &combined.pages;
                for (const auto& level : mesh.tiled->levels)
                    if (level.level == 0)
                        for (size_t p = level.first_page; p < level.first_page + level.page_count; ++p)
                            surface_pages.push_back(p);
            }
        }
        const bool textured = pages != nullptr;

        std::vector<PrimitiveRange> ranges;
        for (const size_t p : surface_pages) {
            const auto first = size_t(surface->page_first_face[p]);
            const auto last = size_t(surface->page_first_face[p + 1]);
            if (last < first || last > surface->faces.size())
                throw collisionError("mesh.write_failed", "The mesh page exceeds its faces", ErrorCode::Unavailable, path_text);
            if (last > first)
                ranges.push_back({p, first, last - first, {}});
        }
        if (!textured) {
            ranges.push_back({0, 0, surface->faces.size(), {}});
        } else if (ranges.empty()) {
            throw collisionError("mesh.write_failed", "The mesh has no faces to write", ErrorCode::Unavailable, path_text);
        }

        std::vector<std::vector<uint8_t>> jpegs;
        if (textured) {
            jpegs.reserve(ranges.size());
            for (const auto& range : ranges) {
                const auto side = surface->page_side(range.page);
                if (side <= 0 || (*pages)[range.page].size() < size_t(side) * size_t(side) * 3)
                    throw collisionError("mesh.write_failed", "The mesh texture page is incomplete", ErrorCode::Unavailable,
                                         path_text);
                jpegs.push_back(encode_page_jpeg((*pages)[range.page], side, std::format("page_{}", range.page)));
            }
        }

        // Keep only the vertices the written faces use: the tiled model holds a copy per level.
        std::vector<int32_t> remap(surface->vertices.size(), -1);
        std::vector<int32_t> used;
        used.reserve(surface->vertices.size());
        for (const auto& range : ranges) {
            for (size_t f = range.first_face; f < range.first_face + range.face_count; ++f) {
                for (const auto vertex : surface->faces[f]) {
                    if (vertex < 0 || size_t(vertex) >= surface->vertices.size())
                        throw collisionError("mesh.write_failed", "The mesh refers to a missing vertex", ErrorCode::Unavailable,
                                             path_text);
                    auto& slot = remap[size_t(vertex)];
                    if (slot < 0) {
                        slot = static_cast<int32_t>(used.size());
                        used.push_back(vertex);
                    }
                }
            }
        }
        const size_t vertex_count = used.size();

        // Lay the binary chunk out up front, so the JSON that precedes it can carry the offsets.
        size_t cursor = 0;
        const auto reserve = [&cursor](const size_t size) {
            Block block{cursor, size};
            cursor += size;
            cursor = (cursor + 3) & ~size_t{3};
            return block;
        };
        const Block positions = reserve(vertex_count * 12);
        const Block normals = reserve(vertex_count * 12);
        Block texcoords{};
        Block colors{};
        if (textured)
            texcoords = reserve(vertex_count * 8);
        else
            colors = reserve(vertex_count * 4);
        for (auto& range : ranges)
            range.indices = reserve(range.face_count * 12);
        std::vector<Block> images;
        images.reserve(jpegs.size());
        for (const auto& jpeg : jpegs)
            images.push_back(reserve(jpeg.size()));
        const size_t bin_length = cursor;

        std::array<float, 3> lo{FLT_MAX, FLT_MAX, FLT_MAX};
        std::array<float, 3> hi{-FLT_MAX, -FLT_MAX, -FLT_MAX};
        for (const auto vertex : used) {
            const auto p = to_gltf(surface->vertices[size_t(vertex)]);
            for (int c = 0; c < 3; ++c) {
                lo[size_t(c)] = std::min(lo[size_t(c)], p[size_t(c)]);
                hi[size_t(c)] = std::max(hi[size_t(c)], p[size_t(c)]);
            }
        }

        json buffer_views = json::array();
        const auto view_of = [&buffer_views](const Block& block) {
            buffer_views.push_back({{"buffer", 0}, {"byteOffset", block.offset}, {"byteLength", block.size}});
            return buffer_views.size() - 1;
        };
        json accessors = json::array();
        json attributes = json::object();

        attributes["POSITION"] = accessors.size();
        accessors.push_back({{"bufferView", view_of(positions)}, {"componentType", 5126}, {"count", vertex_count},
                             {"type", "VEC3"}, {"min", {lo[0], lo[1], lo[2]}}, {"max", {hi[0], hi[1], hi[2]}}});
        attributes["NORMAL"] = accessors.size();
        accessors.push_back({{"bufferView", view_of(normals)}, {"componentType", 5126}, {"count", vertex_count},
                             {"type", "VEC3"}});
        if (textured) {
            attributes["TEXCOORD_0"] = accessors.size();
            accessors.push_back({{"bufferView", view_of(texcoords)}, {"componentType", 5126}, {"count", vertex_count},
                                 {"type", "VEC2"}});
        } else {
            attributes["COLOR_0"] = accessors.size();
            accessors.push_back({{"bufferView", view_of(colors)}, {"componentType", 5121}, {"count", vertex_count},
                                 {"type", "VEC4"}, {"normalized", true}});
        }

        json primitives = json::array();
        json materials = json::array();
        json images_json = json::array();
        json textures = json::array();
        for (const auto& range : ranges) {
            const auto indices = accessors.size();
            accessors.push_back({{"bufferView", view_of(range.indices)}, {"componentType", 5125},
                                 {"count", range.face_count * 3}, {"type", "SCALAR"}});
            primitives.push_back({{"attributes", attributes}, {"indices", indices}, {"mode", 4}});
        }
        if (textured) {
            for (size_t i = 0; i < ranges.size(); ++i) {
                primitives[i]["material"] = i;
                materials.push_back({{"name", std::format("page_{}", ranges[i].page)},
                                     {"doubleSided", true},
                                     {"pbrMetallicRoughness",
                                      {{"baseColorTexture", {{"index", i}}}, {"metallicFactor", 0.0}, {"roughnessFactor", 1.0}}}});
                images_json.push_back({{"name", std::format("page_{}", ranges[i].page)}, {"bufferView", view_of(images[i])},
                                       {"mimeType", "image/jpeg"}});
                textures.push_back({{"sampler", 0}, {"source", i}});
            }
        } else {
            primitives[0]["material"] = 0;
            materials.push_back({{"name", "vertex_colors"},
                                 {"doubleSided", true},
                                 {"pbrMetallicRoughness",
                                  {{"baseColorFactor", {1.0, 1.0, 1.0, 1.0}}, {"metallicFactor", 0.0}, {"roughnessFactor", 1.0}}}});
        }

        json doc;
        doc["asset"] = {{"version", "2.0"}, {"generator", "LichtFeld Studio"}};
        doc["scene"] = 0;
        doc["scenes"] = json::array({json{{"nodes", json::array({0})}}});
        doc["nodes"] = json::array({json{{"mesh", 0}, {"name", "mesh"}}});
        doc["meshes"] = json::array({json{{"name", "mesh"}, {"primitives", primitives}}});
        doc["buffers"] = json::array({json{{"byteLength", bin_length}}});
        doc["bufferViews"] = std::move(buffer_views);
        doc["accessors"] = std::move(accessors);
        doc["materials"] = std::move(materials);
        if (textured) {
            doc["images"] = std::move(images_json);
            doc["textures"] = std::move(textures);
            doc["samplers"] = json::array({json{{"magFilter", 9729}, {"minFilter", 9987}, {"wrapS", 33071}, {"wrapT", 33071}}});
        }
        std::string json_text = doc.dump();
        json_text.resize((json_text.size() + 3) & ~size_t{3}, ' ');

        const auto temporary = std::filesystem::path(path).concat(".tmp");
        {
            std::ofstream out(core::native_fs::native_path(temporary), std::ios::binary | std::ios::trunc);
            const auto write_u32 = [&out](const uint32_t value) {
                const std::array<char, 4> bytes{char(value & 0xFF), char((value >> 8) & 0xFF),
                                                char((value >> 16) & 0xFF), char((value >> 24) & 0xFF)};
                out.write(bytes.data(), 4);
            };
            const uint64_t total = 12 + 8 + json_text.size() + 8 + bin_length;
            if (total > 0xFFFFFFFFull)
                throw collisionError("mesh.write_failed", "The mesh is too large for a GLB file", ErrorCode::Unavailable, path_text);
            write_u32(kGlbMagic);
            write_u32(2);
            write_u32(static_cast<uint32_t>(total));
            write_u32(static_cast<uint32_t>(json_text.size()));
            write_u32(kChunkJson);
            out.write(json_text.data(), std::streamsize(json_text.size()));
            write_u32(static_cast<uint32_t>(bin_length));
            write_u32(kChunkBin);

            std::vector<char> buffer;
            buffer.reserve(size_t{1} << 24);
            const auto flush = [&] {
                out.write(buffer.data(), std::streamsize(buffer.size()));
                buffer.clear();
            };
            const auto append = [&](const void* data, const size_t bytes) {
                const auto* begin = static_cast<const char*>(data);
                buffer.insert(buffer.end(), begin, begin + bytes);
                if (buffer.size() >= (size_t{1} << 24) - 64)
                    flush();
            };

            static_assert(std::endian::native == std::endian::little);
            for (const auto vertex : used) {
                const auto p = to_gltf(surface->vertices[size_t(vertex)]);
                append(p.data(), 12);
            }
            if (surface->normals.size() == surface->vertices.size()) {
                for (const auto vertex : used) {
                    const auto n = to_gltf(surface->normals[size_t(vertex)]);
                    append(n.data(), 12);
                }
            } else {
                const std::array<float, 3> zero{0.f, 0.f, 0.f};
                for (size_t v = 0; v < vertex_count; ++v)
                    append(zero.data(), 12);
            }
            if (textured) {
                for (const auto vertex : used)
                    append(surface->uvs[size_t(vertex)].data(), 8);
            } else {
                std::array<uint8_t, 256> linear{};
                for (size_t i = 0; i < linear.size(); ++i)
                    linear[i] = to_linear(static_cast<uint8_t>(i));
                const std::array<uint8_t, 4> fallback{linear[128], linear[128], linear[128], 255};
                const bool has_colors = surface->colors.size() == surface->vertices.size();
                for (const auto vertex : used) {
                    std::array<uint8_t, 4> rgba = fallback;
                    if (has_colors) {
                        const auto& color = surface->colors[size_t(vertex)];
                        rgba[0] = linear[color[0]];
                        rgba[1] = linear[color[1]];
                        rgba[2] = linear[color[2]];
                    }
                    append(rgba.data(), 4);
                }
            }
            for (const auto& range : ranges) {
                for (size_t f = range.first_face; f < range.first_face + range.face_count; ++f) {
                    const std::array<int32_t, 3> face{remap[size_t(surface->faces[f][0])],
                                                      remap[size_t(surface->faces[f][1])],
                                                      remap[size_t(surface->faces[f][2])]};
                    append(face.data(), 12);
                }
            }
            size_t written = images.empty() ? bin_length : images.front().offset;
            for (size_t p = 0; p < images.size(); ++p) {
                const auto& jpeg = jpegs[p];
                append(jpeg.data(), jpeg.size());
                written = images[p].offset + jpeg.size();
                const size_t next = p + 1 < images.size() ? images[p + 1].offset : bin_length;
                const std::array<char, 4> zeros{};
                while (written < next) {
                    const auto bytes = std::min<size_t>(next - written, zeros.size());
                    append(zeros.data(), bytes);
                    written += bytes;
                }
            }
            flush();
            out.close();
            if (!out)
                throw collisionError("mesh.write_failed", "Could not write the mesh; check disk space and permissions",
                                     ErrorCode::Unavailable, path_text);
        }
        std::error_code error;
        core::native_fs::rename(temporary, path, error);
        if (error)
            throw collisionError("mesh.write_failed", error.message(), ErrorCode::Unavailable, path_text);
    }

} // namespace lfs::training
