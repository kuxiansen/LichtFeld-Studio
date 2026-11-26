/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "io/loaders/colmap_loader.hpp"
#include "core/camera.hpp"
#include "core/error_reporter.hpp"
#include "core/image_io.hpp"
#include "core/logger.hpp"
#include "core/native_fs.hpp"
#include "core/path_utils.hpp"
#include "core/point_cloud.hpp"
#include "formats/colmap.hpp"
#include "formats/ply.hpp"
#include "io/error.hpp"
#include "io/filesystem_utils.hpp"
#include "io/loaders/loader_utils.hpp"
#include <chrono>
#include <cstring>
#include <filesystem>
#include <format>
#include <system_error>
#include <unordered_set>
#include <vector>

namespace lfs::io {

    // Import types from lfs::core for convenience
    using lfs::core::DataType;
    using lfs::core::Device;
    using lfs::core::PointCloud;
    using lfs::core::Tensor;

    namespace {
        // Voxel coordinate packing: offsets negative coordinates and packs the three
        // indices into one int64 key, so an occupancy table is a single hash lookup.
        constexpr int64_t kVoxelHashOffset = 1 << 20;

        int64_t voxelHash(const int ix, const int iy, const int iz) {
            constexpr int64_t kSpan = kVoxelHashOffset * 2;
            return int64_t{ix} + kVoxelHashOffset +
                   (int64_t{iy} + kVoxelHashOffset) * kSpan +
                   (int64_t{iz} + kVoxelHashOffset) * kSpan * kSpan;
        }

        int coordToVoxel(const float x, const float voxel) {
            return x >= 0.0f ? static_cast<int>(x / voxel)
                             : static_cast<int>((x - voxel + 1e-6f) / voxel);
        }

        // Merge a dense point cloud (points3D.ply, typically laser-derived) into the
        // COLMAP sparse one (points3D.bin): every sparse point is kept, and one dense
        // point fills each voxel the sparse cloud does not occupy. Dense columns adopt
        // the sparse dtype so the row copy stays byte aligned.
        PointCloud mixDenseWithSparse(const PointCloud& sparse, const PointCloud& dense, const float voxel_size) {
            std::unordered_set<int64_t> occupied;
            occupied.reserve(static_cast<size_t>(sparse.size()));
            {
                auto means = sparse.means.cpu().contiguous();
                auto acc = means.accessor<float, 2>();
                for (int64_t i = 0; i < sparse.size(); ++i)
                    occupied.insert(voxelHash(coordToVoxel(acc(i, 0), voxel_size),
                                              coordToVoxel(acc(i, 1), voxel_size),
                                              coordToVoxel(acc(i, 2), voxel_size)));
            }

            std::vector<size_t> picked;
            picked.reserve(static_cast<size_t>(dense.size()));
            std::unordered_set<int64_t> picked_voxels;
            picked_voxels.reserve(occupied.size());
            {
                auto means = dense.means.cpu().contiguous();
                auto acc = means.accessor<float, 2>();
                for (int64_t i = 0; i < dense.size(); ++i) {
                    const int64_t key = voxelHash(coordToVoxel(acc(i, 0), voxel_size),
                                                  coordToVoxel(acc(i, 1), voxel_size),
                                                  coordToVoxel(acc(i, 2), voxel_size));
                    if (occupied.find(key) == occupied.end() && picked_voxels.insert(key).second)
                        picked.push_back(static_cast<size_t>(i));
                }
            }

            const size_t sparse_rows = static_cast<size_t>(sparse.size());
            auto concat_field = [&](const Tensor& sparse_t, const Tensor& dense_t) -> Tensor {
                const bool sparse_ok = sparse_t.is_valid() && sparse_t.numel() > 0 && sparse_t.ndim() == 2;
                const bool dense_ok = dense_t.is_valid() && dense_t.numel() > 0 && dense_t.ndim() == 2;
                if (!sparse_ok && !dense_ok)
                    return Tensor();
                const Tensor& ref = sparse_ok ? sparse_t : dense_t;
                auto dst = Tensor::zeros({sparse_ok ? sparse_rows + picked.size() : picked.size(),
                                          ref.shape()[1]},
                                         ref.device(), ref.dtype());
                const size_t row_bytes = ref.shape()[1] * lfs::core::dtype_size(ref.dtype());
                uint8_t* dst_ptr = static_cast<uint8_t*>(dst.data_ptr());
                if (sparse_ok) {
                    auto src = sparse_t.cpu().contiguous();
                    std::memcpy(dst_ptr, src.data_ptr(), sparse_rows * row_bytes);
                    dst_ptr += sparse_rows * row_bytes;
                }
                if (dense_ok) {
                    auto src = dense_t.to(ref.dtype()).cpu().contiguous();
                    const uint8_t* src_ptr = static_cast<const uint8_t*>(src.data_ptr());
                    for (const size_t idx : picked) {
                        std::memcpy(dst_ptr, src_ptr + idx * row_bytes, row_bytes);
                        dst_ptr += row_bytes;
                    }
                }
                return dst;
            };

            PointCloud result;
            result.means = concat_field(sparse.means, dense.means);
            result.colors = concat_field(sparse.colors, dense.colors);
            result.normals = concat_field(sparse.normals, dense.normals);
            result.sh0 = concat_field(sparse.sh0, dense.sh0);
            result.shN = concat_field(sparse.shN, dense.shN);
            result.opacity = concat_field(sparse.opacity, dense.opacity);
            result.scaling = concat_field(sparse.scaling, dense.scaling);
            result.rotation = concat_field(sparse.rotation, dense.rotation);
            result.emit_zero_normals = sparse.emit_zero_normals;
            result.attribute_names = sparse.attribute_names;

            LOG_INFO("Mix init: sparse={} pts in {} voxels; dense={} pts -> pick {} supplementary (voxel={} m)",
                     sparse.size(), occupied.size(), dense.size(), picked.size(), voxel_size);
            return result;
        }
    } // namespace

    Result<LoadResult> ColmapLoader::load(
        const std::filesystem::path& path,
        const LoadOptions& options) {

        LOG_TIMER("COLMAP Loading");
        auto start_time = std::chrono::high_resolution_clock::now();

        // Validate directory exists
        if (!lfs::core::native_fs::exists(path)) {
            return make_error(ErrorCode::PATH_NOT_FOUND,
                              "COLMAP dataset path does not exist", path);
        }

        if (!lfs::core::native_fs::is_directory(path)) {
            return make_error(ErrorCode::NOT_A_DIRECTORY,
                              "COLMAP dataset must be a directory", path);
        }

        if (is_load_cancel_requested(options)) {
            return make_error(ErrorCode::CANCELLED, "COLMAP dataset load cancelled", path);
        }

        // Report initial progress
        if (options.progress) {
            options.progress(0.0f, "Loading COLMAP dataset...");
        }

        // Get search paths for COLMAP files
        auto search_paths = get_colmap_search_paths(path);

        // Check for required COLMAP files in any of the search paths
        auto cameras_bin = find_file_in_paths(search_paths, "cameras.bin");
        auto images_bin = find_file_in_paths(search_paths, "images.bin");
        auto points_bin = find_file_in_paths(search_paths, "points3D.bin");

        auto cameras_txt = find_file_in_paths(search_paths, "cameras.txt");
        auto images_txt = find_file_in_paths(search_paths, "images.txt");
        auto points_txt = find_file_in_paths(search_paths, "points3D.txt");

        auto points_ply = find_file_in_paths(search_paths, "points3D.ply");

        bool has_cameras = !cameras_bin.empty();
        bool has_images = !images_bin.empty();
        bool has_points = !points_bin.empty();

        bool has_cameras_text = !cameras_txt.empty();
        bool has_images_text = !images_txt.empty();
        bool has_points_text = !points_txt.empty();

        bool has_points_ply = !points_ply.empty();

        if ((has_cameras || has_images || has_points) &&
            (has_cameras_text || has_images_text || has_points_text)) {
            LOG_WARN("Found both binary and text COLMAP files. Prioritizing binary files.");
        }

        bool trying_text = !(has_cameras && has_images) && (has_cameras_text && has_images_text);

        // Validate we have required files
        if ((!has_cameras || !has_images) && !trying_text) {
            return make_error(ErrorCode::MISSING_REQUIRED_FILES,
                              std::format("Missing required COLMAP files. cameras.bin: {}, images.bin: {}",
                                          has_cameras ? "found" : "missing",
                                          has_images ? "found" : "missing"),
                              path);
        }

        if ((!has_cameras_text || !has_images_text) && trying_text) {
            return make_error(ErrorCode::MISSING_REQUIRED_FILES,
                              std::format("Missing required COLMAP text files. cameras.txt: {}, images.txt: {}",
                                          has_cameras_text ? "found" : "missing",
                                          has_images_text ? "found" : "missing"),
                              path);
        }

        // Determine images folder
        std::string actual_images_folder = options.images_folder;
        std::filesystem::path image_dir = path / lfs::core::utf8_to_path(actual_images_folder);

        // A relative images folder that is not inside the dataset may sit next to it,
        // resolved from the working directory like the dataset path itself.
        if (const auto requested = lfs::core::utf8_to_path(options.images_folder);
            requested.is_relative() && !lfs::core::native_fs::exists(image_dir)) {
            std::error_code ec;
            if (lfs::core::native_fs::is_directory(requested, ec)) {
                image_dir = std::filesystem::absolute(requested, ec).lexically_normal();
                actual_images_folder = lfs::core::path_to_utf8(image_dir);
                LOG_INFO("Images folder '{}' is outside the dataset; using {}",
                         options.images_folder, actual_images_folder);
            }
        }

        auto is_dataset_root = [&](const std::filesystem::path& candidate) {
            if (candidate.empty()) {
                return false;
            }

            std::error_code ec;
            bool equivalent = lfs::core::native_fs::equivalent(candidate, path, ec);
            if (!ec) {
                return equivalent;
            }

            return candidate.lexically_normal() == path.lexically_normal();
        };

        // If specified folder doesn't exist, check for flat structure
        if (!lfs::core::native_fs::exists(image_dir)) {
            LOG_TIMER_DEBUG("COLMAP resolve image directory");
            const auto cameras_txt_parent =
                cameras_txt.empty() ? std::filesystem::path{} : cameras_txt.parent_path();
            const auto cameras_bin_parent =
                cameras_bin.empty() ? std::filesystem::path{} : cameras_bin.parent_path();

            bool is_flat_structure = is_dataset_root(cameras_txt_parent) ||
                                     is_dataset_root(cameras_bin_parent);

            if (is_flat_structure) {
                bool has_images_in_root = false;
                for (const auto& entry : lfs::core::native_fs::directory_iterator(path)) {
                    if (lfs::core::native_fs::entry_is_regular_file(path, entry) && is_image_file(entry.path())) {
                        has_images_in_root = true;
                        break;
                    }
                }

                if (has_images_in_root) {
                    actual_images_folder = ".";
                    image_dir = path;
                    LOG_INFO("Detected flat structure - using root directory for images");
                } else {
                    return make_error(ErrorCode::MISSING_REQUIRED_FILES,
                                      std::format("Images directory '{}' not found and no images in root",
                                                  options.images_folder),
                                      path);
                }
            } else {
                return make_error(ErrorCode::MISSING_REQUIRED_FILES,
                                  std::format("Images directory '{}' not found in the dataset or the working directory",
                                              options.images_folder),
                                  path);
            }
        }

        try {
            if (options.progress) {
                options.progress(10.0f, "Validating COLMAP dataset layout...");
            }

            throw_if_load_cancel_requested(options, "COLMAP dataset validation cancelled");

            // Validation only mode
            if (options.validate_only) {
                LOG_TIMER_DEBUG("COLMAP validate_only dataset layout");
                if (auto validation_result = validate_colmap_dataset_layout(path, actual_images_folder, options); !validation_result) {
                    return std::unexpected(validation_result.error());
                }

                if (options.progress) {
                    options.progress(100.0f, "COLMAP validation complete");
                }

                LOG_DEBUG("COLMAP validation successful");

                auto end_time = std::chrono::high_resolution_clock::now();
                return LoadResult{
                    .data = LoadedScene{
                        .cameras = {},
                        .point_cloud = nullptr},
                    .scene_center = Tensor::zeros({3}, Device::CPU),
                    .loader_used = name(),
                    .load_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time),
                    .warnings = (has_points || has_points_text || has_points_ply) ? std::vector<std::string>{} : std::vector<std::string>{"No sparse point cloud found (points3D.bin|txt|ply) - will use random initialization"},
                    .georeference = std::nullopt};
            }

            // Load cameras and images
            if (options.progress) {
                options.progress(20.0f, "Reading camera parameters...");
            }

            std::vector<std::shared_ptr<Camera>> cameras;
            Tensor scene_center;
            std::vector<std::string> warnings;
            ColmapPointCloudRecords binary_point_records;

            if (has_cameras && has_images) {
                LOG_DEBUG("Reading binary COLMAP data");
                LOG_TIMER_DEBUG("COLMAP read binary cameras and images");
                auto result = read_colmap_cameras_and_images(
                    path, actual_images_folder, options, &binary_point_records);
                if (!result) {
                    return std::unexpected(result.error());
                }
                std::tie(cameras, scene_center) = std::move(result->value);
                for (auto& diagnostic : result->warnings) {
                    warnings.push_back(diagnostic.message);
                }
            } else if (has_cameras_text && has_images_text) {
                LOG_DEBUG("Reading text COLMAP data");
                LOG_TIMER_DEBUG("COLMAP read text cameras and images");
                auto result = read_colmap_cameras_and_images_text(path, actual_images_folder, options,
                                                                  &binary_point_records);
                if (!result) {
                    return std::unexpected(result.error());
                }
                std::tie(cameras, scene_center) = std::move(result->value);
                for (auto& diagnostic : result->warnings) {
                    warnings.push_back(diagnostic.message);
                }
            } else {
                return make_error(ErrorCode::MISSING_REQUIRED_FILES,
                                  "No valid COLMAP camera and image data found", path);
            }

            if (options.progress) {
                options.progress(40.0f, std::format("Creating {} cameras...", cameras.size()));
            }

            LOG_DEBUG("Creating {} camera objects", cameras.size());

            bool images_have_alpha = false;
            {
                LOG_TIMER_DEBUG("COLMAP detect image alpha");
                images_have_alpha = detect_camera_alpha(cameras, options.cancel_requested);
            }

            if (options.progress) {
                options.progress(60.0f, "Loading point cloud...");
            }

            throw_if_load_cancel_requested(options, "COLMAP point cloud load cancelled");

            // Load point cloud: points3D.ply > points3D.bin > points3D.txt.
            // Track filtering requires COLMAP sparse records, so prefer .bin/.txt when enabled.
            std::shared_ptr<PointCloud> point_cloud;
            const bool use_colmap_track_filter = options.min_track_length > 0;
            bool point_cloud_mixed = false;

            // Mix mode: points3D.ply is the dense (e.g. LAS-derived) cloud and points3D.bin
            // is the COLMAP sparse cloud. It runs only when both exist and the sparse cloud
            // is read unfiltered, so every sparse point can still be kept.
            if (options.mix_ply_with_sparse && has_points_ply && has_points && !use_colmap_track_filter) {
                const float voxel_size = options.mix_voxel_size > 0.f ? options.mix_voxel_size : 0.2f;
                auto dense = load_ply_point_cloud(points_ply, options);
                if (dense) {
                    auto sparse = read_colmap_point_cloud(path, options, &binary_point_records);
                    if (sparse) {
                        point_cloud = std::make_shared<PointCloud>(
                            mixDenseWithSparse(sparse->value, *dense, voxel_size));
                        point_cloud_mixed = true;
                        for (auto& diagnostic : sparse->warnings)
                            warnings.push_back(diagnostic.message);
                        LOG_INFO("Mix init: {} points from points3D.ply (dense) + {} from points3D.bin (sparse) -> {} mixed",
                                 dense->size(), sparse->value.size(), point_cloud->size());
                    } else {
                        LOG_WARN("Mix init: failed to read points3D.bin ({}); using points3D.ply only",
                                 sparse.error().format());
                    }
                } else {
                    LOG_WARN("Mix init: failed to read points3D.ply ({}); falling back to the normal load", dense.error());
                }
            }

            if (!point_cloud && has_points_ply && !use_colmap_track_filter) {
                LOG_INFO("Loading custom point cloud from points3D.ply");
                LOG_TIMER_DEBUG("COLMAP load points3D.ply");
                auto pc_result = load_ply_point_cloud(points_ply, options);
                if (pc_result) {
                    point_cloud = std::make_shared<PointCloud>(std::move(*pc_result));
                    LOG_INFO("Loaded {} points from points3D.ply", point_cloud->size());
                } else {
                    if (is_load_cancel_requested(options)) {
                        return std::unexpected(make_error(ErrorCode::CANCELLED, pc_result.error(), points_ply));
                    }
                    LOG_WARN("Failed to load points3D.ply: {}, falling back", pc_result.error());
                }
            } else if (has_points_ply && use_colmap_track_filter) {
                if (has_points || has_points_text) {
                    LOG_INFO("Skipping points3D.ply because COLMAP min track length filter is enabled");
                } else {
                    LOG_WARN("COLMAP min track length filter requested, but only points3D.ply is available; loading unfiltered PLY");
                    LOG_TIMER_DEBUG("COLMAP load points3D.ply");
                    auto pc_result = load_ply_point_cloud(points_ply, options);
                    if (pc_result) {
                        point_cloud = std::make_shared<PointCloud>(std::move(*pc_result));
                        LOG_INFO("Loaded {} points from points3D.ply", point_cloud->size());
                    } else {
                        if (is_load_cancel_requested(options)) {
                            return std::unexpected(make_error(ErrorCode::CANCELLED, pc_result.error(), points_ply));
                        }
                        LOG_WARN("Failed to load points3D.ply: {}, falling back", pc_result.error());
                    }
                }
            }
            if (!point_cloud && has_points) {
                LOG_DEBUG("Loading binary point cloud");
                LOG_TIMER_DEBUG("COLMAP load binary point cloud");
                if (use_colmap_track_filter) {
                    auto pc_result = read_colmap_point_cloud_with_stats(
                        path, options, &binary_point_records);
                    if (!pc_result) {
                        return std::unexpected(pc_result.error());
                    }
                    auto loaded_pc = std::move(pc_result->value);
                    point_cloud = std::make_shared<PointCloud>(std::move(loaded_pc.point_cloud));
                    const std::string message = std::format(
                        "COLMAP min track length filter: min track length {}, total points {}, after filtering {}",
                        options.min_track_length,
                        loaded_pc.total_points,
                        loaded_pc.points_after_filtering);
                    LOG_INFO("{}", message);
                    warnings.push_back(message);
                    for (auto& diagnostic : pc_result->warnings) {
                        warnings.push_back(diagnostic.message);
                    }
                } else {
                    auto pc_result = read_colmap_point_cloud(path, options, &binary_point_records);
                    if (!pc_result) {
                        return std::unexpected(pc_result.error());
                    }
                    point_cloud = std::make_shared<PointCloud>(std::move(pc_result->value));
                    for (auto& diagnostic : pc_result->warnings) {
                        warnings.push_back(diagnostic.message);
                    }
                }
                LOG_INFO("Loaded {} points from COLMAP", point_cloud->size());
            } else if (!point_cloud && has_points_text) {
                LOG_DEBUG("Loading text point cloud");
                LOG_TIMER_DEBUG("COLMAP load text point cloud");
                if (use_colmap_track_filter) {
                    auto pc_result = read_colmap_point_cloud_text_with_stats(path, options, &binary_point_records);
                    if (!pc_result) {
                        return std::unexpected(pc_result.error());
                    }
                    auto loaded_pc = std::move(pc_result->value);
                    point_cloud = std::make_shared<PointCloud>(std::move(loaded_pc.point_cloud));
                    const std::string message = std::format(
                        "COLMAP min track length filter: min track length {}, total points {}, after filtering {}",
                        options.min_track_length,
                        loaded_pc.total_points,
                        loaded_pc.points_after_filtering);
                    LOG_INFO("{}", message);
                    warnings.push_back(message);
                    for (auto& diagnostic : pc_result->warnings) {
                        warnings.push_back(diagnostic.message);
                    }
                } else {
                    auto pc_result = read_colmap_point_cloud_text(path, options, &binary_point_records);
                    if (!pc_result) {
                        return std::unexpected(pc_result.error());
                    }
                    point_cloud = std::make_shared<PointCloud>(std::move(pc_result->value));
                    for (auto& diagnostic : pc_result->warnings) {
                        warnings.push_back(diagnostic.message);
                    }
                }
                LOG_INFO("Loaded {} points from COLMAP text file", point_cloud->size());
            } else if (!point_cloud) {
                LOG_WARN("No point cloud found - will use random initialization");
                point_cloud = std::make_shared<PointCloud>();
            }

            if (point_cloud->size() == 0) {
                warnings.push_back("No sparse points available - using random initialization");
            }

            // Centralize scene
            std::optional<ImportGeoreference> import_georeference;
            {
                LOG_TIMER_DEBUG("COLMAP centralize scene");
                auto centralization =
                    centralize_scene(cameras, point_cloud, options.centralize, scene_center);
                scene_center = std::move(centralization.scene_center);
                import_georeference = std::move(centralization.georeference);
            }

            if (options.progress) {
                options.progress(100.0f, "COLMAP loading complete");
            }

            auto end_time = std::chrono::high_resolution_clock::now();
            auto load_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                end_time - start_time);

            auto scene_center_cpu = scene_center.cpu();
            const float* sc_ptr = scene_center_cpu.ptr<float>();

            LoadResult result{
                .data = LoadedScene{
                    .cameras = std::move(cameras),
                    .point_cloud = std::move(point_cloud),
                    .point_cloud_mixed = point_cloud_mixed},
                .scene_center = scene_center,
                .images_have_alpha = images_have_alpha,
                .loader_used = name(),
                .load_time = load_time,
                .warnings = std::move(warnings),
                .georeference = std::move(import_georeference)};

            LOG_DEBUG("  - Scene center: [{:.3f}, {:.3f}, {:.3f}]",
                      sc_ptr[0], sc_ptr[1], sc_ptr[2]);

            return result;

        } catch (const LoadCancelledError& e) {
            return make_error(ErrorCode::CANCELLED, e.what(), path);
        } catch (const lfs::Exception& e) {
            lfs::Error error = lfs::Error(e.error())
                                   .with_context("load COLMAP dataset", LFS_SOURCE_SITE_CURRENT(),
                                                 lfs::SmallFields{}.add("path", lfs::core::path_to_utf8(path)));
            lfs::core::ErrorReporter::get().report(error, lfs::core::ReportChannel::OwnerLog);
            return std::unexpected(from_lfs_error(error));
        } catch (const std::exception& e) {
            return make_error(ErrorCode::CORRUPTED_DATA,
                              std::format("Failed to load COLMAP dataset: {}", e.what()), path);
        }
    }

    bool ColmapLoader::canLoad(const std::filesystem::path& path) const {
        if (!safe_exists(path) || !safe_is_directory(path)) {
            return false;
        }

        auto search_paths = get_colmap_search_paths(path);

        // Check for COLMAP files in any location
        const std::vector<std::string> colmap_files = {
            "cameras.bin", "cameras.txt",
            "images.bin", "images.txt"};

        for (const auto& filename : colmap_files) {
            if (!find_file_in_paths(search_paths, filename).empty()) {
                return true;
            }
        }

        return false;
    }

    std::string ColmapLoader::name() const {
        return "COLMAP";
    }

    std::vector<std::string> ColmapLoader::supportedExtensions() const {
        return {}; // Directory-based, no file extensions
    }

    int ColmapLoader::priority() const {
        return 5; // Medium priority
    }

} // namespace lfs::io
