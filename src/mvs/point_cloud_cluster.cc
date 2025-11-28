//Copyright (c) 2019, SenseTime Group.
//All rights reserved.

#include "util/misc.h"
#include "mvs/utils.h"
#include "util/ply.h"
#include "base/common.h"

#include "point_cloud_cluster.h"

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/compute_average_spacing.h>
#include <CGAL/Search_traits_2.h>
// #include <CGAL/squared_distance_2.h>
#include <CGAL/Search_traits_3.h>
// #include <CGAL/squared_distance_3.h>
#include <CGAL/Orthogonal_k_neighbor_search.h>
#include <vector>
#include <fstream>
#include <boost/tuple/tuple.hpp>


namespace sensemap {
namespace mvs {

void PointCloudCluster::Options::Print() const {
  PrintHeading2("PointCloudCluster::Options");
  PrintOption(min_pts_per_cluster);
  PrintOption(max_pts_per_cluster);
  PrintOption(cell_size_factor);
  PrintOption(cell_size);
  PrintOption(valid_spacing_factor);
}

bool PointCloudCluster::Options::Check() const {
  CHECK_OPTION_GT(min_pts_per_cluster, 0);
//  CHECK_OPTION_GT(max_pts_per_cluster, 0);
  CHECK_OPTION_LE(min_pts_per_cluster, max_pts_per_cluster);
  CHECK_OPTION_GT(cell_size, 0.0f);
  CHECK_OPTION_GE(valid_spacing_factor, 0.0);
  return true;
}

PointCloudCluster::PointCloudCluster(const Options& options,
                                     const std::string& workspace_path)
    : num_reconstruction_(0),
      options_(options),
      workspace_path_(workspace_path) {
  CHECK(options_.Check());
}

void PointCloudCluster::ReadWorkspace() {
  num_reconstruction_ = 0;
  std::cout << "Reading workspace..." << std::endl;
    for (size_t reconstruction_idx = 0; ;reconstruction_idx++) {
        const auto& reconstruction_path =
            JoinPaths(workspace_path_, std::to_string(reconstruction_idx));
        const auto& dense_reconstruction_path =
            JoinPaths(reconstruction_path, DENSE_DIR);
        if (!ExistsDir(dense_reconstruction_path)) {
            break;
        }

        num_reconstruction_++;
    }
}

void PointCloudCluster::Run() {
  options_.Print();
  std::cout << std::endl;

  ReadWorkspace();

  for (size_t reconstruction_idx = 0; reconstruction_idx < num_reconstruction_;
       reconstruction_idx++) {

    PrintHeading1(StringPrintf("Clustering# %d", reconstruction_idx));

    if (IsStopped()) {
      GetTimer().PrintMinutes();
      return;
    }

    auto reconstruction_path =
      JoinPaths(workspace_path_, std::to_string(reconstruction_idx));
    auto dense_reconstruction_path = JoinPaths(reconstruction_path, DENSE_DIR);
    if (!ExistsDir(dense_reconstruction_path)) {
        continue;
    }

    auto sparse_reconstruction_path = JoinPaths(dense_reconstruction_path, SPARSE_DIR);
    if (!ExistsDir(sparse_reconstruction_path)) {
        continue;
    }

    auto cluster_reconstruction_path = JoinPaths(reconstruction_path, CLUSTER_DIR);
    if (ExistsDir(cluster_reconstruction_path)) {
        CHECK(boost::filesystem::remove_all(cluster_reconstruction_path));
    }
//    CreateDirIfNotExists(cluster_reconstruction_path);
    CHECK(boost::filesystem::create_directory(cluster_reconstruction_path));

    auto fused_path = JoinPaths(dense_reconstruction_path, FUSION_NAME);
    if (!ExistsFile(fused_path)) {
        continue;
    }
    std::vector<PlyPoint> ply_points = ReadPly(fused_path);
    std::size_t point_num = ply_points.size();

    std::vector<std::vector<uint32_t> > fused_points_visibility;
    auto fused_vis_path = fused_path + ".vis";
    if (!ExistsFile(fused_vis_path)) {
        continue;
    }
    ReadPointsVisibility(fused_vis_path, fused_points_visibility);
    if (fused_points_visibility.size() != point_num) {
        continue;
    }

    std::cout << point_num << " points to be clustered" << std::endl;

    std::vector<Eigen::Vector3f> points(point_num);
    for (std::size_t i = 0; i < point_num; ++i) {
        points[i] = Eigen::Map<Eigen::Vector3f>((float*)&ply_points[i]);
    }

    std::vector<int> point_cluster_map;
    std::vector<std::set<std::size_t> > cluster_images_map;
    int cluster_num = Cluster(point_cluster_map, cluster_images_map, points, fused_points_visibility);

    // std::shared_ptr<Reconstruction> reconstruction = std::make_shared<Reconstruction>();
    // reconstruction->ReadReconstruction(sparse_reconstruction_path);
    // std::cout << reconstruction->NumRegisterImages() << " source registered images" << std::endl;
    // std::cout << reconstruction->NumCameras() << " source cameras" << std::endl;
    // std::cout << reconstruction->NumMapPoints() << " source map points" << std::endl;
    // const auto &register_image_ids = reconstruction->RegisterImageIds();

    std::vector<std::vector<PlyPoint> > clustered_points(cluster_num);
    std::vector<std::vector<std::vector<uint32_t> > > clustered_points_visibility(cluster_num);
    for (std::size_t i = 0; i < point_num; ++i) {
        auto cluster_idx = point_cluster_map[i];
        if (cluster_idx < 0) {
            continue;
        }

        clustered_points[cluster_idx].emplace_back(ply_points[i]);
        clustered_points_visibility[cluster_idx].emplace_back(fused_points_visibility[i]);
    }

    std::cout << "Number of clusters: " << cluster_num
              << std::endl;
    GetTimer().PrintMinutes();

    // Save clustered points, visibility and SfM.
    for (std::size_t cluster_idx = 0; cluster_idx < cluster_num; cluster_idx++) {
        std::cout << "Cluster: " << cluster_idx << std::endl;
        auto cluster_path = JoinPaths(cluster_reconstruction_path, std::to_string(cluster_idx));
        CreateDirIfNotExists(cluster_path);

        auto dense_cluster_path = JoinPaths(cluster_path, DENSE_DIR);
        CreateDirIfNotExists(dense_cluster_path);

        auto sparse_cluster_path = JoinPaths(dense_cluster_path, SPARSE_DIR);
        CreateDirIfNotExists(sparse_cluster_path);

// //#define REINDEX_VISIBILITY
// #ifdef REINDEX_VISIBILITY
//         std::shared_ptr<Reconstruction> cluster_reconstruction = std::make_shared<Reconstruction>();

//         std::vector<Image> cluster_images;
//         std::unordered_set<image_t> cluster_image_ids;
//         EIGEN_STL_UMAP(camera_t, Camera) cluster_cameras;
//         EIGEN_STL_UMAP(mappoint_t, MapPoint) cluster_mappoints;
//         std::unordered_map<std::size_t, std::size_t> image_idx_map;
//         for (auto image_idx : cluster_images_map[cluster_idx]) {
//             image_idx_map[image_idx] = cluster_images.size();

//             auto image_id = register_image_ids[image_idx];
//             const auto &image = reconstruction->Image(image_id);
//             cluster_images.push_back(image);
//             cluster_image_ids.insert(image_id);

//             auto camera_id = image.CameraId();
//             cluster_cameras[camera_id] = reconstruction->Camera(camera_id);

//             for (const auto &point_2d : image.Points2D()) {
//                 if (!point_2d.HasMapPoint()) {
//                     continue;
//                 }

//                 auto mappoint_id = point_2d.MapPointId();
//                 cluster_mappoints[mappoint_id] = reconstruction->MapPoint(mappoint_id);
//             }
//         }

//         for (auto &image : cluster_images) {
//             Image cluster_image;

//             cluster_image.SetImageId(image.ImageId());
//             cluster_image.SetName(image.Name());
//             cluster_image.SetCameraId(image.CameraId());
//             cluster_image.SetLabelId(image.LabelId());
//             cluster_image.SetRegistered(false);

//             cluster_image.SetPoseFlag(image.HasPose());
//             cluster_image.SetQvec(image.Qvec());
//             cluster_image.SetTvec(image.Tvec());
//             cluster_image.SetQvecPrior(image.QvecPrior());
//             cluster_image.SetTvecPrior(image.TvecPrior());

//             auto points_2d = image.Points2D();
//             for (auto &point_2d : points_2d) {
//                 point_2d.SetMask(false);
//                 point_2d.SetMapPointId(kInvalidMapPointId);
//             }
//             cluster_image.SetPoints2D(points_2d);

//             cluster_reconstruction->AddImage(cluster_image);
//             cluster_reconstruction->RegisterImage(cluster_image.ImageId());
//         }
//         std::cout << cluster_reconstruction->NumRegisterImages() << " registered images clustered" << std::endl;

//         for (const auto &camera : cluster_cameras) {
//             cluster_reconstruction->AddCamera(camera.second);
//         }
//         std::cout << cluster_reconstruction->NumCameras() << " cameras clustered" << std::endl;

//         for (auto &mappoint : cluster_mappoints) {
//             const auto &elements = mappoint.second.Track().Elements();
//             Track track;
//             for (const auto &element : elements) {
//                 if (cluster_image_ids.find(element.image_id) != cluster_image_ids.end()) {
//                     track.AddElement(element);
//                 }
//             }
//             if (track.Length() == 0) {
//                 continue;
//             }

//             cluster_reconstruction->AddMapPoint(mappoint.second.XYZ(), track, mappoint.second.Color());
//         }
//         std::cout << cluster_reconstruction->NumMapPoints() << " map points clustered" << std::endl;

//         cluster_reconstruction->WriteReconstruction(sparse_cluster_path);

//         for (auto &point_visibility : clustered_points_visibility[cluster_idx]) {
//             for (auto &image_idx : point_visibility) {
//                 image_idx = image_idx_map[image_idx];
//             }
//         }
// //#else
// //        reconstruction->WriteReconstruction(sparse_cluster_path);
// #endif
        auto cluster_fused_path = JoinPaths(dense_cluster_path, FUSION_NAME);
        WriteBinaryPlyPoints(cluster_fused_path, clustered_points[cluster_idx], false, true);

        auto cluster_fused_vis_path = cluster_fused_path + ".vis";
        WritePointsVisibility(cluster_fused_vis_path, clustered_points_visibility[cluster_idx]);
        std::cout << clustered_points[cluster_idx].size() << " fused points clustered" << std::endl;
    }
  }
}

// Types
typedef CGAL::Exact_predicates_inexact_constructions_kernel kernel_t;
typedef kernel_t::Point_2 point_2_t;
typedef kernel_t::FT FT;
typedef CGAL::Search_traits_2<kernel_t> tree_traits_2_t;
typedef CGAL::Orthogonal_k_neighbor_search<tree_traits_2_t> neighbor_search_2_t;
typedef neighbor_search_2_t::iterator search_2_iterator_t;
typedef neighbor_search_2_t::Tree tree_2_t;
// Concurrency
#ifdef CGAL_LINKED_WITH_TBB
typedef CGAL::Parallel_tag concurrency_tag_t;
#else
typedef CGAL::Sequential_tag concurrency_tag_t;
#endif

std::size_t PointCloudCluster::GridCluster(std::vector<int> &cell_cluster_map,
                                           std::vector<std::set<std::size_t> > &cluster_images_map,
                                           const std::vector<std::size_t> &cell_point_count,
                                           const std::vector<std::set<std::size_t> > &cell_images_map,
                                           const std::vector<float> &cell_average_spacing,
                                           const std::size_t grid_size_x,
                                           const std::size_t grid_size_y,
                                           const float valid_spacing) {
    const std::size_t grid_side = grid_size_x;
    const std::size_t grid_slide = grid_side * grid_size_y;
//    const std::size_t grid_volume = grid_slide * grid_size_z;

//    static const int index_offset[6][3] = { {0, -1, 0}, {-1, 0, 0}, {0, 1, 0},
//                                            {1, 0,  0}, {0, 0, -1}, {0, 0, 1} };
    // static const int index_offset[4][2] = { {0, -1}, {-1, 0}, {0, 1}, {1, 0} };
    static const int index_offset[8][2] = { {0, -1}, {-1, 0}, {0, 1}, {1, 0},
                                            {-1, -1}, {1, -1}, {1, 1}, {-1, 1} };

    std::size_t cell_num = cell_point_count.size();
    cell_cluster_map.resize(cell_num);
    std::fill(cell_cluster_map.begin(), cell_cluster_map.end(), -1);

    std::vector<std::pair<std::size_t, std::size_t> > dense_cells;
    dense_cells.reserve(cell_num);
    std::vector<std::size_t> sparse_cells;
    sparse_cells.reserve(cell_num);
    std::vector<unsigned char> cell_type_map(cell_num, 0);
    for (std::size_t i = 0; i < cell_num; i++) {
        if (cell_point_count[i] == 0) {
            continue;
        }

        if (cell_average_spacing[i] > valid_spacing) {
            sparse_cells.push_back(i);
            cell_type_map[i] = 128;
            continue;
        }

        dense_cells.emplace_back(cell_point_count[i], i);
        cell_type_map[i] = 255;
    }
    dense_cells.shrink_to_fit();
    // std::sort(dense_cells.begin(), dense_cells.end(), std::greater<std::pair<std::size_t, std::size_t> >());
    sparse_cells.shrink_to_fit();

    std::vector<std::set<std::size_t> >().swap(cluster_images_map);
    if (dense_cells.empty()) {
        for (auto cell_idx : sparse_cells) {
            cell_cluster_map[cell_idx] = 0;
        }

        return sparse_cells.empty() ? 0 : 1;
    }

    std::size_t cluster_idx = 0;
    std::size_t dense_cell_num = dense_cells.size();
    std::vector<unsigned char> cell_visited_map(cell_num);
    std::vector<std::size_t> cluster_point_count;
    std::vector<std::shared_ptr<tree_2_t> > cluster_tree_map;
    std::vector<std::vector<std::size_t> > cluster_cells_map;

    for (std::size_t i = 0; i < dense_cell_num; i++) {
        int cell_idx = dense_cells[i].second;
        if (cell_cluster_map[cell_idx] != -1) {
            continue;
        }

        std::size_t num_visited_points = 0;
        memset(cell_visited_map.data(), 0, cell_num * sizeof(unsigned char));
        cluster_images_map.push_back(std::set<std::size_t>());
        auto &cluster_images = cluster_images_map.back();
        cluster_tree_map.push_back(std::make_shared<tree_2_t>());
        auto cluster_tree = cluster_tree_map.back();
        cluster_cells_map.push_back(std::vector<std::size_t>());
        auto &cluster_cells = cluster_cells_map.back();

        while (cell_idx >= 0 && cell_idx < cell_num
            && num_visited_points < options_.min_pts_per_cluster) {
            std::queue<int> Q;
            Q.push(cell_idx);
            cell_visited_map[cell_idx] = 255;

            while (!Q.empty() && num_visited_points < options_.max_pts_per_cluster) {
                auto cell_idx_front = Q.front();
                Q.pop();

                cell_cluster_map[cell_idx_front] = cluster_idx;
                cluster_cells.push_back(cell_idx_front);
                num_visited_points += cell_point_count[cell_idx_front];
                const auto &cell_images = cell_images_map[cell_idx_front];
                cluster_images.insert(cell_images.begin(), cell_images.end());

//                int z_cell = cell_idx_front / grid_slide;
//                int y_cell = cell_idx_front % grid_slide / grid_side;
//                int x_cell = cell_idx_front % grid_slide % grid_side;
                int y_cell = cell_idx_front / grid_side;
                int x_cell = cell_idx_front % grid_side;
                cluster_tree->insert(point_2_t(x_cell, y_cell));

//                for (int j = 0; j < 6; ++j) {
                for (int j = 0; j < 4; ++j) {
                // for (int j = 0; j < 8; ++j) {
//                    int z_cell_nbr = z_cell + index_offset[j][2];
                    int y_cell_nbr = y_cell + index_offset[j][1];
                    int x_cell_nbr = x_cell + index_offset[j][0];
//                    int cell_idx_nbr = z_cell_nbr * grid_slide + y_cell_nbr * grid_side + x_cell_nbr;
                    int cell_idx_nbr = y_cell_nbr * grid_side + x_cell_nbr;
                    if (x_cell_nbr < 0 || x_cell_nbr >= grid_size_x
                     || y_cell_nbr < 0 || y_cell_nbr >= grid_size_y
                     || cell_visited_map[cell_idx_nbr]
                     || cell_cluster_map[cell_idx_nbr] != -1
                     || cell_type_map[cell_idx_nbr] != 255) {
                        continue;
                    }

                    Q.push(cell_idx_nbr);
                    cell_visited_map[cell_idx_nbr] = 255;
                }
            }

            // Find a new seeding cell
            // cell_idx = -1;
            // int max_common_images = -1;
            // for (std::size_t j = i + 1; j < dense_cell_num; j++) {
            //     auto cell_idx_next = dense_cells[j].second;
            //     if (cell_cluster_map[cell_idx_next] != -1) {
            //         continue;
            //     }

            //     std::vector<std::size_t> common_images;
            //     const auto &cell_images = cell_images_map[cell_idx_next];
            //     std::set_intersection(cluster_images.begin(), cluster_images.end(),
            //                           cell_images.begin(), cell_images.end(),
            //                           std::insert_iterator<std::vector<std::size_t> >(common_images, common_images.begin()));
            //     if ((int)common_images.size() > max_common_images) {
            //         cell_idx = cell_idx_next;
            //         max_common_images = common_images.size();
            //     }
            // }
            cell_idx = -1;
            // float min_cell_dist = FLT_MAX;
            // for (std::size_t j = i + 1; j < dense_cell_num; j++) {
            //     auto cell_idx_next = dense_cells[j].second;
            //     if (cell_cluster_map[cell_idx_next] != -1) {
            //         continue;
            //     }

            //     if (cell_type_map[cell_idx_next] != 255) {
            //         continue;
            //     }

            //     int y_cell_next = cell_idx_next / grid_side;
            //     int x_cell_next = cell_idx_next % grid_side;
            //     point_2_t query(x_cell_next, y_cell_next);
            //     neighbor_search_2_t search(*cluster_tree, query, 1);
            //     search_2_iterator_t search_iterator = search.begin();
            //     if (search_iterator == search.end()) {
            //         continue;
            //     }

            //     // point_2_t p = search_iterator->first;
            //     float cell_dist = std::sqrt(search_iterator->second);
            //     if (cell_dist > 2.0f) {
            //         continue;
            //     }

            //     if (cell_idx == -1 || cell_dist < min_cell_dist) {
            //         cell_idx = cell_idx_next;
            //         min_cell_dist = cell_dist;
            //     }
            // }
        }

        cluster_point_count.push_back(num_visited_points);
        std::cout << num_visited_points << " points clustered" << std::endl;
        cluster_idx++;
    }

    std::size_t cluster_num = cluster_idx;
    std::vector<std::size_t> cluster_idx_map;
    cluster_idx_map.reserve(cluster_num);
    for (std::size_t i = 0; i < cluster_num; i++) {
        if (cluster_point_count[i] >= options_.min_pts_per_cluster) {
            cluster_idx_map.push_back(i);
        }
    }
    cluster_idx_map.shrink_to_fit();
    if (cluster_idx_map.empty()) {
        for (auto cell_idx : dense_cells) {
            cell_cluster_map[cell_idx.second] = 0;
        }
        for (auto cell_idx : sparse_cells) {
            cell_cluster_map[cell_idx] = 0;
        }

        return 1;
    }

    for (std::size_t i = 0; i < cluster_num; i++) {
        if (cluster_point_count[i] >= options_.min_pts_per_cluster) {
            continue;
        }

        std::cout << "Merging cluster " << i << std::endl;
        const auto &cur_cluster_images = cluster_images_map[i];
        auto cur_cluster_tree = cluster_tree_map[i];
        const auto &cur_cluster_cells = cluster_cells_map[i];

        int merge_cluster_idx = -1;
        float min_cluster_dist = FLT_MAX;
        for (std::size_t cluster_idx : cluster_idx_map) {
            float cluster_dist = 0.0f;
            auto cluster_tree = cluster_tree_map[cluster_idx];
            for (auto cell_idx : cur_cluster_cells) {
                int y_cell = cell_idx / grid_side;
                int x_cell = cell_idx % grid_side;
                point_2_t query(x_cell, y_cell);
                neighbor_search_2_t search(*cluster_tree, query, 1);
                search_2_iterator_t search_iterator = search.begin();
                if (search_iterator == search.end()) {
                    continue;
                }

                // point_2_t p = search_iterator->first;
                float cell_dist = std::sqrt(search_iterator->second);
                cluster_dist += cell_dist;
            }

            if (merge_cluster_idx == -1 || cluster_dist < min_cluster_dist) {
                merge_cluster_idx = cluster_idx;
                min_cluster_dist = cluster_dist;
            }
        }

        for (auto cell_idx : cur_cluster_cells) {
            cell_cluster_map[cell_idx] = merge_cluster_idx;
        }
        cluster_images_map[merge_cluster_idx].insert(cur_cluster_images.begin(), cur_cluster_images.end());
        cluster_point_count[merge_cluster_idx] += cluster_point_count[i];
        cluster_tree_map[merge_cluster_idx]->insert(cur_cluster_tree->begin(), cur_cluster_tree->end());
        cluster_cells_map[merge_cluster_idx].insert(cluster_cells_map[merge_cluster_idx].end(), cur_cluster_cells.begin(), cur_cluster_cells.end());
    }

    for (std::size_t i = 0; i < cluster_idx_map.size(); i++) {
        const auto &cluster_idx = cluster_idx_map[i];
        cluster_images_map[i] = cluster_images_map[cluster_idx];
        cluster_point_count[i] = cluster_point_count[cluster_idx];
        cluster_tree_map[i] = cluster_tree_map[cluster_idx];
        cluster_cells_map[i] = cluster_cells_map[cluster_idx];
        for (auto cell_idx : cluster_cells_map[i]) {
            cell_cluster_map[cell_idx] = i;
        }
    }
    cluster_num = cluster_idx_map.size();
    cluster_images_map.resize(cluster_num);
    cluster_point_count.resize(cluster_num);
    cluster_tree_map.resize(cluster_num);
    cluster_cells_map.resize(cluster_num);

    for (auto cell_idx : sparse_cells) {
        int y_cell = cell_idx / grid_side;
        int x_cell = cell_idx % grid_side;
        point_2_t query(x_cell, y_cell);

        int merge_cluster_idx = -1;
        float min_cluster_dist = FLT_MAX;
        for (std::size_t i = 0; i < cluster_num; i++) {
            auto cluster_tree = cluster_tree_map[i];
            neighbor_search_2_t search(*cluster_tree, query, 1);
            search_2_iterator_t search_iterator = search.begin();
            if (search_iterator == search.end()) {
                continue;
            }

            // point_2_t p = search_iterator->first;
            float cluster_dist = std::sqrt(search_iterator->second);
            if (merge_cluster_idx == -1 || cluster_dist < min_cluster_dist) {
                merge_cluster_idx = i;
                min_cluster_dist = cluster_dist;
            }
        }

        if (merge_cluster_idx == -1) {
            continue;
        }

        cell_cluster_map[cell_idx] = merge_cluster_idx;
        const auto &cell_images = cell_images_map[cell_idx];
        cluster_images_map[merge_cluster_idx].insert(cell_images.begin(), cell_images.end());
        cluster_point_count[merge_cluster_idx] += cell_point_count[cell_idx];
        // cluster_tree_map[merge_cluster_idx]->insert(query);
        cluster_cells_map[merge_cluster_idx].push_back(cell_idx);
    }
//    std::cout << cluster_num << " clusters" << std::endl;
//    for (std::size_t i = 0; i < cluster_num; i++) {
//        std::cout << cluster_point_count[i] << " points clustered as cluster " << i << std::endl;
//    }

    return cluster_num;
}

// Types
// typedef CGAL::Exact_predicates_inexact_constructions_kernel kernel_t;
typedef kernel_t::Point_3 point_3_t;
// typedef kernel_t::FT FT;
typedef CGAL::Search_traits_3<kernel_t> tree_traits_3_t;
typedef CGAL::Orthogonal_k_neighbor_search<tree_traits_3_t> neighbor_search_3_t;
typedef neighbor_search_3_t::iterator search_3_iterator_t;
typedef neighbor_search_3_t::Tree tree_3_t;
// Data type := index, followed by the point, followed by three integers that
// define the Red Green Blue color of the point.
// typedef boost::tuple<int, Point, int, int, int> IndexedPointWithColorTuple;
typedef boost::tuple<int, point_3_t> indexed_point_3_tuple_t;
// // Concurrency
// #ifdef CGAL_LINKED_WITH_TBB
// typedef CGAL::Parallel_tag concurrency_tag_t;
// #else
// typedef CGAL::Sequential_tag concurrency_tag_t;
// #endif

std::size_t PointCloudCluster::Cluster(std::vector<int> &point_cluster_map,
                                       std::vector<std::set<std::size_t> > &cluster_images_map,
                                       const std::vector<Eigen::Vector3f> &points,
                                       const std::vector<std::vector<uint32_t> > &points_visibility,
                                       const unsigned int nb_neighbors) {
    std::cout << "PointCloudCluster::Cluster" << std::endl;

    // Compute Pivot.
    Eigen::Matrix3f pivot;
    Eigen::Vector3f centroid(Eigen::Vector3f::Zero());
    for (const auto &point : points) {
        centroid += point;
    }
    std::size_t point_num = points.size();
    centroid /= point_num;

    Eigen::Matrix3f M = Eigen::Matrix3f::Zero();
    for (int i = 0; i < 3; ++i) {
        for (const auto &point : points) {
            M(i, 0) += (point[i] - centroid[i]) * (point[0] - centroid[i]);
            M(i, 1) += (point[i] - centroid[i]) * (point[1] - centroid[i]);
            M(i, 2) += (point[i] - centroid[i]) * (point[2] - centroid[i]);
        }
    }

    Eigen::JacobiSVD<Eigen::Matrix3f> svd(M, Eigen::ComputeFullU | Eigen::ComputeFullV);
    // std::cout << svd.singularValues() << std::endl;

    pivot = svd.matrixU().transpose();

    // {
    //     FILE *fp = fopen("pivot.obj", "w");

    //     Eigen::Vector3f xaxis = centroid + (Eigen::Vector3f)pivot.row(0) * 10;
    //     Eigen::Vector3f yaxis = centroid + (Eigen::Vector3f)pivot.row(1) * 10;
    //     Eigen::Vector3f zaxis = centroid + (Eigen::Vector3f)pivot.row(2) * 10;

    //     fprintf(fp, "v %f %f %f 0 0 0\n", centroid[0], centroid[1], centroid[2]);
    //     fprintf(fp, "v %f %f %f 255 0 0\n", xaxis[0], xaxis[1], xaxis[2]);
    //     fprintf(fp, "v %f %f %f 0 255 0\n", yaxis[0], yaxis[1], yaxis[2]);
    //     fprintf(fp, "v %f %f %f 0 0 255\n", zaxis[0], zaxis[1], zaxis[2]);

    //     fclose(fp);
    // }

    // Transform points & Calculate BoundingBox.
    Eigen::Vector3f box_min, box_max;
    box_min = box_max = pivot * points[0];
    std::vector<Eigen::Vector3f> transformed_points(point_num);
    for (int i = 0; i < point_num; ++i) {
        auto &point = transformed_points[i];
        point = pivot * points[i];
        
        box_min[0] = std::min(box_min[0], point[0]);
        box_min[1] = std::min(box_min[1], point[1]);
        box_min[2] = std::min(box_min[2], point[2]);
        box_max[0] = std::max(box_max[0], point[0]);
        box_max[1] = std::max(box_max[1], point[1]);
        box_max[2] = std::max(box_max[2], point[2]);
    }

    // {
    //     FILE *fp = fopen("transformed_points.obj", "w");
    //     for (const auto &point : transformed_points) {
    //         fprintf(fp, "v %f %f %f\n", point[0], point[1], point[2]);
    //     }
    //     fclose(fp);
    // }

    FT average_spacing = (FT)0.0;
    std::vector<FT> point_spacings(point_num);
    {
        std::vector<point_3_t> points_3(point_num);
        for (std::size_t i = 0; i < point_num; i++) {
            const auto &point = points[i];
            points_3[i] = point_3_t(point[0], point[1], point[2]);
        }

        // Instantiate a KD-tree search.
        std::cout << "Instantiating a search tree for point cloud spacing ..." << std::endl;
        tree_3_t tree(points_3.begin(), points_3.end());

        // iterate over input points, compute and output point spacings
        for (std::size_t i = 0; i < point_num; i++) {
            const auto &query = points_3[i];
            // performs k + 1 queries (if unique the query point is
            // output first). search may be aborted when k is greater
            // than number of input points
            neighbor_search_3_t search(tree, query, nb_neighbors + 1);
            auto &point_spacing = point_spacings[i];
            point_spacing = (FT)0.0;
            std::size_t k = 0;
            for (search_3_iterator_t search_iterator = search.begin(); search_iterator != search.end() && k <= nb_neighbors; search_iterator++, k++)
            {
                // point_3_t p = search_iterator->first;
                // point_spacing += std::sqrt(CGAL::squared_distance(query, p));
                point_spacing += std::sqrt(search_iterator->second);
            }
            // output point spacing
            if (k > 1) {
                point_spacing /= (FT)(k - 1);
            }

            average_spacing += point_spacing;
                std::cout << "\r";
                std::cout << "Point spacings computed [" << i + 1 << " / " << point_num << "]" << std::flush;
        }
        std::cout << std::endl;
        average_spacing /= (FT)point_num;
        std::cout << "Average spacing: " << average_spacing << std::endl;
    }

    const float cell_size = options_.cell_size_factor <= 0 ?
                                options_.cell_size
                              : (options_.cell_size_factor * average_spacing);
    const std::size_t grid_size_x = static_cast<std::size_t>((box_max.x() - box_min.x()) / cell_size) + 1;
    const std::size_t grid_size_y = static_cast<std::size_t>((box_max.y() - box_min.y()) / cell_size) + 1;
//    const std::size_t grid_size_z = static_cast<std::size_t>((box_max.z() - box_min.z()) / cell_size) + 1;
    const std::size_t grid_side = grid_size_x;
    const std::size_t grid_slide = grid_side * grid_size_y;
//    const std::size_t grid_volume = grid_slide * grid_size_z;

    std::vector<std::size_t> cell_point_count(grid_slide, 0);
    std::vector<std::set<std::size_t> > cell_images_map(grid_slide);
    std::vector<std::size_t> point_cell_map(point_num);
    std::vector<float> cell_average_spacing(grid_slide, 0.0f);

    for (std::size_t i = 0; i < point_num; ++i) {
        const auto &point = transformed_points[i];
        std::size_t x_cell = static_cast<std::size_t>((point.x() - box_min.x()) / cell_size);
        std::size_t y_cell = static_cast<std::size_t>((point.y() - box_min.y()) / cell_size);
//        std::size_t z_cell = static_cast<std::size_t>((point.z() - box_min.z()) / cell_size);

//        std::size_t cell_idx = z_cell * grid_slide + y_cell * grid_side + x_cell;
        std::size_t cell_idx = y_cell * grid_side + x_cell;
        cell_point_count[cell_idx]++;
        const auto &point_visibility = points_visibility[i];
        cell_images_map[cell_idx].insert(point_visibility.begin(), point_visibility.end());
        point_cell_map[i] = cell_idx;
        cell_average_spacing[cell_idx] += point_spacings[i];
    }
    for (std::size_t i = 0; i < grid_slide; ++i) {
        const auto &point_count = cell_point_count[i];
        if (point_count == 0) {
            continue;
        }

        cell_average_spacing[i] /= point_count;
    }

//    std::size_t num_points = 0;
//    for (const auto &point_count : cell_point_count) {
//        num_points += point_count;
//    }
//    std::cout << "Number of points: " << num_points << std::endl;

//    transformed_points.clear();

    std::vector<int> cell_cluster_map;
    const float valid_spacing = average_spacing * options_.valid_spacing_factor;
    std::size_t cluster_num
        = GridCluster(cell_cluster_map, cluster_images_map,
                      cell_point_count, cell_images_map, cell_average_spacing,
                      grid_size_x, grid_size_y, valid_spacing);


//    cell_point_count.clear();

    point_cluster_map.resize(point_num);
    memset(point_cluster_map.data(), -1, point_num * sizeof(int));
    for (std::size_t i = 0; i < point_num; ++i) {
        point_cluster_map[i] = cell_cluster_map[point_cell_map[i]];
    }

    return cluster_num;
}

}  // namespace mvs
}  // namespace sensemap
