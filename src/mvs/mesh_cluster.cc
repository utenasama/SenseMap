//Copyright (c) 2019, SenseTime Group.
//All rights reserved.

#include "util/misc.h"
#include "mvs/utils.h"
#include "util/obj.h"
#include "base/common.h"

#include "mesh_cluster.h"

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/compute_average_spacing.h>
#include <CGAL/Search_traits_2.h>
// #include <CGAL/squared_distance_2.h>
// #include <CGAL/Search_traits_3.h>
// #include <CGAL/squared_distance_3.h>
#include <CGAL/Orthogonal_k_neighbor_search.h>
#include <vector>
#include <fstream>
#include <boost/tuple/tuple.hpp>


namespace sensemap {
namespace mvs {

void MeshCluster::Options::Print() const {
  PrintHeading2("MeshCluster::Options");
  PrintOption(min_faces_per_cluster);
  PrintOption(max_faces_per_cluster);
  PrintOption(cell_size_factor);
  PrintOption(cell_size);
  PrintOption(valid_spacing_factor);
}

bool MeshCluster::Options::Check() const {
  CHECK_OPTION_GT(min_faces_per_cluster, 0);
//  CHECK_OPTION_GT(max_faces_per_cluster, 0);
  CHECK_OPTION_LE(min_faces_per_cluster, max_faces_per_cluster);
  CHECK_OPTION_GT(cell_size, 0.0f);
  CHECK_OPTION_GE(valid_spacing_factor, 0.0);
  return true;
}

MeshCluster::MeshCluster(const Options& options,
                                     const std::string& workspace_path)
    : num_reconstruction_(0),
      options_(options),
      workspace_path_(workspace_path) {
  CHECK(options_.Check());
}

void MeshCluster::ReadWorkspace() {
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

void MeshCluster::Run() {
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

    auto model_path = JoinPaths(dense_reconstruction_path, MODEL_NAME);
    if (!ExistsFile(model_path)) {
        continue;
    }
    TriangleMesh mesh;
    ReadTriangleMeshObj(model_path, mesh, true);
    std::size_t vertex_num = mesh.vertices_.size();
    std::size_t face_num = mesh.faces_.size();
    std::cout << face_num << " faces to be clustered" << std::endl;

    std::vector<int> face_cluster_map;
    int cluster_num = Cluster(face_cluster_map, mesh);

    std::vector<std::vector<std::size_t> > clustered_faces(cluster_num);
    for (std::size_t i = 0; i < face_num; ++i) {
        auto cluster_idx = face_cluster_map[i];
        if (cluster_idx < 0) {
            continue;
        }

        clustered_faces[cluster_idx].emplace_back(i);
    }

    std::cout << "Number of clusters: " << cluster_num
              << std::endl;
    GetTimer().PrintMinutes();

    // Save clustered meshes.
    for (std::size_t cluster_idx = 0; cluster_idx < cluster_num; cluster_idx++) {
        std::cout << "Cluster: " << cluster_idx << std::endl;
        auto cluster_path = JoinPaths(cluster_reconstruction_path, std::to_string(cluster_idx));
        CreateDirIfNotExists(cluster_path);

        auto dense_cluster_path = JoinPaths(cluster_path, DENSE_DIR);
        CreateDirIfNotExists(dense_cluster_path);

        auto sparse_cluster_path = JoinPaths(dense_cluster_path, SPARSE_DIR);
        CreateDirIfNotExists(sparse_cluster_path);

        TriangleMesh cluster_mesh;
        cluster_mesh.vertices_.reserve(mesh.vertices_.size());
        cluster_mesh.vertex_normals_.reserve(mesh.vertex_normals_.size());
        cluster_mesh.vertex_colors_.reserve(mesh.vertex_colors_.size());
        cluster_mesh.vertex_labels_.reserve(mesh.vertex_labels_.size());
        cluster_mesh.faces_.reserve(mesh.faces_.size());
        cluster_mesh.face_normals_.reserve(mesh.face_normals_.size());
        for (auto face_id : clustered_faces[cluster_idx]) {
            cluster_mesh.faces_.push_back(mesh.faces_[face_id]);
        }
        if (!mesh.face_normals_.empty()) {
            for (auto face_id : clustered_faces[cluster_idx]) {
                cluster_mesh.face_normals_.push_back(mesh.face_normals_[face_id]);
            }
        }
        std::vector<int> vertex_idx_map(vertex_num, -1);
        for (auto face_id : clustered_faces[cluster_idx]) {
            const auto &face = mesh.faces_[face_id];
            vertex_idx_map[face[0]] = 0;
            vertex_idx_map[face[1]] = 0;
            vertex_idx_map[face[2]] = 0;
        }
        for (std::size_t i = 0; i < vertex_num; i++) {
            if (vertex_idx_map[i] == -1) {
                continue;
            }

            vertex_idx_map[i] = cluster_mesh.vertices_.size();
            cluster_mesh.vertices_.push_back(mesh.vertices_[i]);
        }
        if (!mesh.vertex_normals_.empty()) {
            for (std::size_t i = 0; i < vertex_num; i++) {
                if (vertex_idx_map[i] == -1) {
                    continue;
                }

                cluster_mesh.vertex_normals_.push_back(mesh.vertex_normals_[i]);
            }
        }
        if (!mesh.vertex_colors_.empty()) {
            for (std::size_t i = 0; i < vertex_num; i++) {
                if (vertex_idx_map[i] == -1) {
                    continue;
                }

                cluster_mesh.vertex_colors_.push_back(mesh.vertex_colors_[i]);
            }
        }
        if (!mesh.vertex_labels_.empty()) {
            for (std::size_t i = 0; i < vertex_num; i++) {
                if (vertex_idx_map[i] == -1) {
                    continue;
                }

                cluster_mesh.vertex_labels_.push_back(mesh.vertex_labels_[i]);
            }
        }
        cluster_mesh.vertices_.shrink_to_fit();
        cluster_mesh.vertex_normals_.shrink_to_fit();
        cluster_mesh.vertex_colors_.shrink_to_fit();
        cluster_mesh.vertex_labels_.shrink_to_fit();
        cluster_mesh.faces_.shrink_to_fit();
        cluster_mesh.face_normals_.shrink_to_fit();
        for (auto &face : cluster_mesh.faces_) {
            face[0] = vertex_idx_map[face[0]];
            face[1] = vertex_idx_map[face[1]];
            face[2] = vertex_idx_map[face[2]];
        }

        auto cluster_model_path = JoinPaths(dense_cluster_path, MODEL_NAME);
        WriteTriangleMeshObj(cluster_model_path, cluster_mesh);
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

std::size_t MeshCluster::GridCluster(std::vector<int> &cell_cluster_map,
                                           const std::vector<std::size_t> &cell_face_count,
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

    std::size_t cell_num = cell_face_count.size();
    cell_cluster_map.resize(cell_num);
    std::fill(cell_cluster_map.begin(), cell_cluster_map.end(), -1);

    std::vector<std::pair<std::size_t, std::size_t> > dense_cells;
    dense_cells.reserve(cell_num);
    std::vector<std::size_t> sparse_cells;
    sparse_cells.reserve(cell_num);
    std::vector<unsigned char> cell_type_map(cell_num, 0);
    for (std::size_t i = 0; i < cell_num; i++) {
        if (cell_face_count[i] == 0) {
            continue;
        }

        if (cell_average_spacing[i] > valid_spacing) {
            sparse_cells.push_back(i);
            cell_type_map[i] = 128;
            continue;
        }

        dense_cells.emplace_back(cell_face_count[i], i);
        cell_type_map[i] = 255;
    }
    dense_cells.shrink_to_fit();
    // std::sort(dense_cells.begin(), dense_cells.end(), std::greater<std::pair<std::size_t, std::size_t> >());
    sparse_cells.shrink_to_fit();

    if (dense_cells.empty()) {
        for (auto cell_idx : sparse_cells) {
            cell_cluster_map[cell_idx] = 0;
        }

        return sparse_cells.empty() ? 0 : 1;
    }

    std::size_t cluster_idx = 0;
    std::size_t dense_cell_num = dense_cells.size();
    std::vector<unsigned char> cell_visited_map(cell_num);
    std::vector<std::size_t> cluster_face_count;
    std::vector<std::shared_ptr<tree_2_t> > cluster_tree_map;
    std::vector<std::vector<std::size_t> > cluster_cells_map;

    for (std::size_t i = 0; i < dense_cell_num; i++) {
        int cell_idx = dense_cells[i].second;
        if (cell_cluster_map[cell_idx] != -1) {
            continue;
        }

        std::size_t num_visited_faces = 0;
        memset(cell_visited_map.data(), 0, cell_num * sizeof(unsigned char));
        cluster_tree_map.push_back(std::make_shared<tree_2_t>());
        auto cluster_tree = cluster_tree_map.back();
        cluster_cells_map.push_back(std::vector<std::size_t>());
        auto &cluster_cells = cluster_cells_map.back();

        while (cell_idx >= 0 && cell_idx < cell_num
            && num_visited_faces < options_.min_faces_per_cluster) {
            std::queue<int> Q;
            Q.push(cell_idx);
            cell_visited_map[cell_idx] = 255;

            while (!Q.empty() && num_visited_faces < options_.max_faces_per_cluster) {
                auto cell_idx_front = Q.front();
                Q.pop();

                cell_cluster_map[cell_idx_front] = cluster_idx;
                cluster_cells.push_back(cell_idx_front);
                num_visited_faces += cell_face_count[cell_idx_front];

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

        cluster_face_count.push_back(num_visited_faces);
        std::cout << num_visited_faces << " faces clustered" << std::endl;
        cluster_idx++;
    }

    std::size_t cluster_num = cluster_idx;
    std::vector<std::size_t> cluster_idx_map;
    cluster_idx_map.reserve(cluster_num);
    for (std::size_t i = 0; i < cluster_num; i++) {
        if (cluster_face_count[i] >= options_.min_faces_per_cluster) {
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
        if (cluster_face_count[i] >= options_.min_faces_per_cluster) {
            continue;
        }

        std::cout << "Merging cluster " << i << std::endl;
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
        cluster_face_count[merge_cluster_idx] += cluster_face_count[i];
        cluster_tree_map[merge_cluster_idx]->insert(cur_cluster_tree->begin(), cur_cluster_tree->end());
        cluster_cells_map[merge_cluster_idx].insert(cluster_cells_map[merge_cluster_idx].end(), cur_cluster_cells.begin(), cur_cluster_cells.end());
    }

    for (std::size_t i = 0; i < cluster_idx_map.size(); i++) {
        const auto &cluster_idx = cluster_idx_map[i];
        cluster_face_count[i] = cluster_face_count[cluster_idx];
        cluster_tree_map[i] = cluster_tree_map[cluster_idx];
        cluster_cells_map[i] = cluster_cells_map[cluster_idx];
        for (auto cell_idx : cluster_cells_map[i]) {
            cell_cluster_map[cell_idx] = i;
        }
    }
    cluster_num = cluster_idx_map.size();
    cluster_face_count.resize(cluster_num);
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

        // if (merge_cluster_idx == -1) {
        //     continue;
        // }

        cell_cluster_map[cell_idx] = merge_cluster_idx;
        cluster_face_count[merge_cluster_idx] += cell_face_count[cell_idx];
        // cluster_tree_map[merge_cluster_idx]->insert(query);
        cluster_cells_map[merge_cluster_idx].push_back(cell_idx);
    }
//    std::cout << cluster_num << " clusters" << std::endl;
//    for (std::size_t i = 0; i < cluster_num; i++) {
//        std::cout << cluster_face_count[i] << " faces clustered as cluster " << i << std::endl;
//    }

    return cluster_num;
}

std::size_t MeshCluster::Cluster(std::vector<int> &face_cluster_map,
                                 const TriangleMesh &mesh) {
    std::cout << "MeshCluster::Cluster" << std::endl;
    if (mesh.vertices_.empty()|| mesh.faces_.empty()) {
        std::vector<int>().swap(face_cluster_map);
        return 0;
    }

    // Compute Pivot.
    Eigen::Matrix3f pivot;
    Eigen::Vector3d centroid(Eigen::Vector3d::Zero());
    for (const auto &vertex : mesh.vertices_) {
        centroid += vertex;
    }
    std::size_t vertex_num = mesh.vertices_.size();
    centroid /= vertex_num;

    Eigen::Matrix3f M = Eigen::Matrix3f::Zero();
    for (int i = 0; i < 3; ++i) {
        for (const auto &vertex : mesh.vertices_) {
            M(i, 0) += (vertex[i] - centroid[i]) * (vertex[0] - centroid[i]);
            M(i, 1) += (vertex[i] - centroid[i]) * (vertex[1] - centroid[i]);
            M(i, 2) += (vertex[i] - centroid[i]) * (vertex[2] - centroid[i]);
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
    std::size_t face_num = mesh.faces_.size();
    std::vector<Eigen::Vector3f> transformed_tri_centroids(face_num);
    Eigen::Vector3f box_min, box_max;
    {
        auto &transformed_tri_centroid = transformed_tri_centroids[0];
        const auto &face = mesh.faces_[0];
        transformed_tri_centroid = pivot * (mesh.vertices_[face[0]] + mesh.vertices_[face[1]] + mesh.vertices_[face[2]]).cast<float>() / 3.0f;
        box_min = box_max = pivot * transformed_tri_centroid;
    }
    for (int i = 1; i < face_num; ++i) {
        auto &transformed_tri_centroid = transformed_tri_centroids[i];
        const auto &face = mesh.faces_[i];
        transformed_tri_centroid = pivot * (mesh.vertices_[face[0]] + mesh.vertices_[face[1]] + mesh.vertices_[face[2]]).cast<float>() / 3.0f;
        
        box_min[0] = std::min(box_min[0], transformed_tri_centroid[0]);
        box_min[1] = std::min(box_min[1], transformed_tri_centroid[1]);
        box_min[2] = std::min(box_min[2], transformed_tri_centroid[2]);
        box_max[0] = std::max(box_max[0], transformed_tri_centroid[0]);
        box_max[1] = std::max(box_max[1], transformed_tri_centroid[1]);
        box_max[2] = std::max(box_max[2], transformed_tri_centroid[2]);
    }

    float average_spacing = 0.0f;
    std::vector<float> face_spacings(face_num);
    for (int i = 0; i < face_num; ++i) {
        const auto &face = mesh.faces_[i];
        Eigen::Vector3f v1 = mesh.vertices_[face[0]].cast<float>();
        Eigen::Vector3f v2 = mesh.vertices_[face[1]].cast<float>();
        Eigen::Vector3f v3 = mesh.vertices_[face[2]].cast<float>();
        face_spacings[i] = ((v1 - v2).norm() + (v2 - v3).norm() + (v1 - v3).norm()) / 3.0f;
        average_spacing += face_spacings[i];
    }
    average_spacing /= face_num;

    const float cell_size = options_.cell_size_factor <= 0 ?
                                options_.cell_size
                              : (options_.cell_size_factor * average_spacing);
    const std::size_t grid_size_x = static_cast<std::size_t>((box_max.x() - box_min.x()) / cell_size) + 1;
    const std::size_t grid_size_y = static_cast<std::size_t>((box_max.y() - box_min.y()) / cell_size) + 1;
//    const std::size_t grid_size_z = static_cast<std::size_t>((box_max.z() - box_min.z()) / cell_size) + 1;
    const std::size_t grid_side = grid_size_x;
    const std::size_t grid_slide = grid_side * grid_size_y;
//    const std::size_t grid_volume = grid_slide * grid_size_z;

    std::vector<std::size_t> cell_face_count(grid_slide, 0);
    std::vector<std::size_t> face_cell_map(face_num);
    std::vector<float> cell_average_spacing(grid_slide, 0.0f);

    for (std::size_t i = 0; i < face_num; ++i) {
        const auto &transformed_tri_centroid = transformed_tri_centroids[i];
        std::size_t x_cell = static_cast<std::size_t>((transformed_tri_centroid.x() - box_min.x()) / cell_size);
        std::size_t y_cell = static_cast<std::size_t>((transformed_tri_centroid.y() - box_min.y()) / cell_size);
//        std::size_t z_cell = static_cast<std::size_t>((transformed_tri_centroid.z() - box_min.z()) / cell_size);

//        std::size_t cell_idx = z_cell * grid_slide + y_cell * grid_side + x_cell;
        std::size_t cell_idx = y_cell * grid_side + x_cell;
        cell_face_count[cell_idx]++;
        face_cell_map[i] = cell_idx;
        cell_average_spacing[cell_idx] += face_spacings[i];
    }
    for (std::size_t i = 0; i < grid_slide; ++i) {
        const auto &face_count = cell_face_count[i];
        if (face_count == 0) {
            continue;
        }

        cell_average_spacing[i] /= face_count;
    }

//    transformed_tri_centroids.clear();

    std::vector<int> cell_cluster_map;
    const float valid_spacing = average_spacing * options_.valid_spacing_factor;
    std::size_t cluster_num
        = GridCluster(cell_cluster_map, cell_face_count, cell_average_spacing,
                      grid_size_x, grid_size_y, valid_spacing);

//    cell_face_count.clear();

    face_cluster_map.resize(face_num);
    memset(face_cluster_map.data(), -1, face_num * sizeof(int));
    for (std::size_t i = 0; i < face_num; ++i) {
        face_cluster_map[i] = cell_cluster_map[face_cell_map[i]];
    }

    return cluster_num;
}

}  // namespace mvs
}  // namespace sensemap
