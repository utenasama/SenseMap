#include "octree.h"

#include "util/logging.h"
#include "util/math.h"

#include <iostream>

namespace sensemap {
namespace lidar {

OctoTree::OctoTree() {}

OctoTree::OctoTree(const double voxel_size, const double g_grid_size, const size_t max_layer, const size_t layer, 
                   const std::vector<size_t> layer_point_size, const std::vector<size_t> min_layer_point_size, 
                   const std::vector<size_t> max_layer_point_size, const uint64_t create_time, const Voxel::Option & option)
    : voxel_size_(voxel_size),
      g_grid_size_(g_grid_size),
      max_layer_(max_layer),
      layer_(layer),
      layer_point_size_(layer_point_size),
      min_layer_point_size_(min_layer_point_size),
      max_layer_point_size_(max_layer_point_size),
      voxel_option_(option) {
    
    octo_state_ = 0;
    new_points_num_ = 0;
    update_points_num_ = 0;
    all_points_num_ = 0;
    init_octo_ = false;
    update_enable_ = true;
    max_feature_update_threshold_ = layer_point_size_[layer_];
    min_points_size_ = min_layer_point_size_[layer_];
    max_points_size_ = max_layer_point_size_[layer_];
    update_size_threshold_ = 5;

    g_grid_inv_size_ = 1.0 / g_grid_size_;

    grid_points_.clear();
    // new_grid_points_.clear();

    tree_dirty_ = false;
    lifetime_ = 0;
    lastest_time_ = 0;
    create_time_ = create_time;

    for (int i = 0; i < 8; i++) {
        leaves_[i] = nullptr;
    }

    voxel_ = new Voxel(voxel_option_);
}

void OctoTree::InsertPoint(const OctoTree::Point & point) {
    if (update_enable_) {
        double locate_point[3] = { point.x -  world_origin_[0],
                                point.y -  world_origin_[1],
                                point.z -  world_origin_[2] };
        uint32_t x = locate_point[0] / g_grid_size_;
        uint32_t y = locate_point[1] / g_grid_size_;
        uint32_t z = locate_point[2] / g_grid_size_;
        uint64_t locate_code = z * GRID_SLICE + y * GRID_LENGTH + x;
        grid_points_[locate_code]++;
        if (grid_point_times_.find(locate_code) != grid_point_times_.end()) {
            grid_point_times_[locate_code] = std::min((uint32_t)point.lifetime, grid_point_times_[locate_code]);
        } else {
            grid_point_times_[locate_code] = point.lifetime;
        }
        new_points_num_++;
        all_points_num_++;
    }
}

void OctoTree::InitOctoTree() {
    if (all_points_num_ > max_feature_update_threshold_) {
        InitVoxel();
        if (voxel_->IsFeature()) {
            octo_state_ = 0;
            if (all_points_num_ > max_points_size_) {
                update_enable_ = false;
            }
            if (lifetime_ == 0) {
                for (auto & grid_point : grid_point_times_) {
                    if (grid_point.second == 0) continue;
                    lifetime_ = std::max(lifetime_, (uint64_t)grid_point.second);
                }
            }
        } else {
            octo_state_ = 1;
            CutOctoTree();
        }
        init_octo_ = true;
        new_points_num_ = 0;
    }
}

void OctoTree::CutOctoTree() {
    if (layer_ >= max_layer_) {
        octo_state_ = 0;
        return;
    }

    const double leaf_voxel_size = 0.5 * voxel_size_;
    const double quater_length = 0.5 * leaf_voxel_size;
    for (auto & grid_point : grid_points_) {
        if (grid_point.second == 0) continue;
        uint32_t z = grid_point.first * GRID_INV_SLICE;
        uint32_t yx = grid_point.first - z * GRID_SLICE;
        uint32_t y = yx * GRID_INV_LENGTH;
        uint32_t x = yx - y * GRID_LENGTH;
        Eigen::Vector3d point(x * g_grid_size_ + world_origin_[0], 
                              y * g_grid_size_ + world_origin_[1], 
                              z * g_grid_size_ + world_origin_[2]);

        int xyz[3] = {0, 0, 0};
        if (point[0] > voxel_center_[0]) {
            xyz[0] = 1;
        }
        if (point[1] > voxel_center_[1]) {
            xyz[1] = 1;
        }
        if (point[2] > voxel_center_[2]) {
            xyz[2] = 1;
        }
        int leafnum = 4 * xyz[0] + 2 * xyz[1] + xyz[2];
        if (leaves_[leafnum] == nullptr) {
            leaves_[leafnum] = new OctoTree(
                leaf_voxel_size, g_grid_size_, max_layer_, layer_ + 1, layer_point_size_, min_layer_point_size_, max_layer_point_size_, create_time_, voxel_option_);
            leaves_[leafnum]->voxel_center_[0] = voxel_center_[0] + (2 * xyz[0] - 1) * quater_length;
            leaves_[leafnum]->voxel_center_[1] = voxel_center_[1] + (2 * xyz[1] - 1) * quater_length;
            leaves_[leafnum]->voxel_center_[2] = voxel_center_[2] + (2 * xyz[2] - 1) * quater_length;
            leaves_[leafnum]->world_origin_[0] = world_origin_[0];
            leaves_[leafnum]->world_origin_[1] = world_origin_[1];
            leaves_[leafnum]->world_origin_[2] = world_origin_[2];
        }
        leaves_[leafnum]->InsertPointInternal(grid_point.first, grid_point_times_.at(grid_point.first), grid_point.second);
    }
    for (uint i = 0; i < 8; i++) {
        if (leaves_[i] != nullptr) {
            if (leaves_[i]->all_points_num_ > leaves_[i]->max_feature_update_threshold_) {
                leaves_[i]->InitVoxel();
                if (leaves_[i]->voxel_->IsFeature()) {
                    leaves_[i]->octo_state_ = 0;
                    if (leaves_[i]->all_points_num_ > leaves_[i]->max_points_size_) {
                        leaves_[i]->update_enable_ = false;
                    }
                    if (leaves_[i]->lifetime_ == 0) {
                        for (auto & grid_point : leaves_[i]->grid_point_times_) {
                            if (grid_point.second == 0) continue;
                            leaves_[i]->lifetime_ = std::max(leaves_[i]->lifetime_, (uint64_t)grid_point.second);
                        }
                    }
                } else {
                    leaves_[i]->octo_state_ = 1;
                    leaves_[i]->CutOctoTree();
                }
                leaves_[i]->init_octo_ = true;
                leaves_[i]->new_points_num_ = 0;
            }
        }
    }
}

void OctoTree::RemoveFromOctoTree(const OctoTree::Point & point) {
    double locate_point[3] = { point.x -  world_origin_[0],
                               point.y -  world_origin_[1],
                               point.z -  world_origin_[2] };
    uint32_t x = locate_point[0] * g_grid_inv_size_;
    uint32_t y = locate_point[1] * g_grid_inv_size_;
    uint32_t z = locate_point[2] * g_grid_inv_size_;
    uint64_t locate_code = z * GRID_SLICE + y * GRID_LENGTH + x;
    RemoveFromOctoTreeInternal(locate_code);
}

void OctoTree::RemoveFromOctoTreeLazy(const Point & point) {
    // double locate_point[3] = { point.x -  world_origin_[0],
    //                            point.y -  world_origin_[1],
    //                            point.z -  world_origin_[2] };
    // uint32_t x = locate_point[0] * g_grid_inv_size_;
    // uint32_t y = locate_point[1] * g_grid_inv_size_;
    // uint32_t z = locate_point[2] * g_grid_inv_size_;
    // uint64_t locate_code = z * GRID_SLICE + y * GRID_LENGTH + x;
    // RemoveFromOctoTreeInternalLazy(locate_code);
    RemoveFromOctoTreeInternalLazy(point);
}

void OctoTree::RemoveFromKdTree(const Point & point) {
    double locate_point[3] = { point.x -  world_origin_[0],
                               point.y -  world_origin_[1],
                               point.z -  world_origin_[2] };
    uint32_t x = locate_point[0] / g_grid_size_;
    uint32_t y = locate_point[1] / g_grid_size_;
    uint32_t z = locate_point[2] / g_grid_size_;
    uint64_t locate_code = z * GRID_SLICE + y * GRID_LENGTH + x;
    auto iter = tree_points_.find(locate_code);
    if (iter != tree_points_.end() && iter->second > 0) {
        tree_points_[locate_code]--;
        if (tree_points_[locate_code] == 0) {
            tree_points_.erase(locate_code);
            tree_point_times_.erase(locate_code);
            tree_point_intensities_.erase(locate_code);
        }
        tree_dirty_ = true;
    }
}

void OctoTree::AppendToOctoTree(const OctoTree::Point & point, const double verbose) {
    // double locate_point[3] = { point.x -  world_origin_[0],
    //                            point.y -  world_origin_[1],
    //                            point.z -  world_origin_[2] };
    // uint32_t x = locate_point[0] / g_grid_size_;
    // uint32_t y = locate_point[1] / g_grid_size_;
    // uint32_t z = locate_point[2] / g_grid_size_;
    // uint64_t locate_code = z * GRID_SLICE + y * GRID_LENGTH + x;
    // AppendToOctoTreeInternal(locate_code, point.lifetime, 1);
    AppendToOctoTreeInternal(point, 1, true, verbose);
}

void OctoTree::AppendToOctoTreeLazy(const Point & point) {
    // double locate_point[3] = { point.x -  world_origin_[0],
    //                            point.y -  world_origin_[1],
    //                            point.z -  world_origin_[2] };
    // uint32_t x = locate_point[0] / g_grid_size_;
    // uint32_t y = locate_point[1] / g_grid_size_;
    // uint32_t z = locate_point[2] / g_grid_size_;
    // uint64_t locate_code = z * GRID_SLICE + y * GRID_LENGTH + x;
    // AppendToOctoTreeInternalLazy(locate_code, point.lifetime, 1);
    AppendToOctoTreeInternalLazy(point, 1);
}

void OctoTree::AppendToKdTree(const Point & point) {
    double locate_point[3] = { point.x -  world_origin_[0],
                               point.y -  world_origin_[1],
                               point.z -  world_origin_[2] };
    uint32_t x = locate_point[0] / g_grid_size_;
    uint32_t y = locate_point[1] / g_grid_size_;
    uint32_t z = locate_point[2] / g_grid_size_;
    uint64_t locate_code = z * GRID_SLICE + y * GRID_LENGTH + x;
    tree_points_[locate_code]++;
    tree_point_intensities_[locate_code] = point.intensity;
    if (tree_point_times_.find(locate_code) != tree_point_times_.end()) {
        tree_point_times_[locate_code] = std::min((uint32_t)point.lifetime, tree_point_times_[locate_code]);
    } else {
        tree_point_times_[locate_code] = point.lifetime;
    }
    lifetime_ = std::max(lifetime_, point.lifetime);
    tree_dirty_ = true;
}

void OctoTree::RebuildOctree() {
    if (voxel_dirty_) {
        for (int i = 0; i < 8; ++i) {
            if (leaves_[i]) {
                leaves_[i]->RebuildOctree();
            }
        }
        voxel_->ComputeFeature();
        if (voxel_->IsDetermined() &&
            voxel_->min_eval_ < voxel_option_.plane_min_eigen_eval &&
            voxel_->min_max_eigen_ratio_ < voxel_option_.plane_min_max_eigen_ratio &&
            voxel_->min_mid_eigen_ratio_ < voxel_option_.plane_min_mid_eigen_ratio &&
            voxel_->mid_max_eigen_ratio_ > voxel_option_.plane_mid_max_eigen_ratio &&
            all_points_num_ >= 20/*min_points_size_*/) {
            bool planarity = true;
            for (int i = 0; i < 8; ++i) {
                if (leaves_[i]) {
                    if (!leaves_[i]->voxel_->IsFeature() && leaves_[i]->all_points_num_ >= 20/*min_points_size_*/) {
                        planarity = false;
                        break;
                    }
                }
            }
            if (planarity) {
                const double cos_thres = std::cos(DEG2RAD(10.0));
                Eigen::Vector3d m_pivot = voxel_->GetPivot().normalized();
                Eigen::Vector3d m_var = voxel_->GetEx();
                for (int i = 0; i < 8; ++i) {
                    if (leaves_[i] && leaves_[i]->voxel_->IsFeature()) {
                        Eigen::Vector3d m_sub_pivot = leaves_[i]->voxel_->GetPivot().normalized();
                        Eigen::Vector3d m_sub_var = leaves_[i]->voxel_->GetEx();
                        double pivot_diff = std::abs(m_sub_pivot.dot(m_pivot));
                        double dist = std::abs(m_pivot.dot(m_sub_var - m_var));
                        if (pivot_diff < cos_thres || dist > 0.1) {
                            planarity = false;
                            break;
                        }
                    }
                }
            }
            voxel_->SetFeature(planarity);
        } else {
            voxel_->SetFeature(false);
        }
        if (voxel_->IsFeature()) {
            if (lifetime_ == 0) {
                lifetime_ = lastest_time_;
            }
            if (all_points_num_ >= max_points_size_) {
                update_enable_ = false;
            }
        }
        new_points_num_ = 0;
        voxel_dirty_ = false;
    }
}

void OctoTree::RebuildKdTree() {
    if (tree_dirty_) {
        tree_.clear();
        for(auto ptree_point : tree_points_) {
            if (ptree_point.second == 0) continue;
            uint64_t locate_code = ptree_point.first;
            uint32_t z = locate_code * GRID_INV_SLICE;
            uint32_t yx = locate_code - z * GRID_SLICE;
            uint32_t y = yx * GRID_INV_LENGTH;
            uint32_t x = yx - y * GRID_LENGTH;
            Point_3 point(x * g_grid_size_ + world_origin_[0], 
                        y * g_grid_size_ + world_origin_[1], 
                        z * g_grid_size_ + world_origin_[2]);
            tree_.insert(point);
        }

        if (tree_.size() > 0) {
        #ifdef CGAL_LINKED_WITH_TBB
            tree_.build<CGAL::Parallel_tag>();
        #else
            tree_.build();
        #endif
        }
        tree_dirty_ = false;
    }
}

bool OctoTree::FindKNeighbor(const Eigen::Vector3d & point, std::vector<Eigen::Vector3d> & nearest_point, const int K) {
    if (tree_dirty_) {
        RebuildKdTree();
    }

    if (tree_.empty()) return false;

    Point_3 query(point[0], point[1], point[2]);
    K_neighbor_search search(tree_, query, K);
    int num_nearest = 0;
    for(K_neighbor_search::iterator it = search.begin(); it != search.end(); it++) {
        Point_3 p = it->first;
        nearest_point[num_nearest++] = Eigen::Vector3d(p.x(), p.y(), p.z());
    }
    return true;
}

OctoTree* OctoTree::LocateOctree(const OctoTree::Point & point, const int terminal_layer, const bool verbose) {
    if (voxel_->IsFeature() || layer_ >= terminal_layer || layer_ >= max_layer_) {
        return this;
    }
    int xyz[3] = {0, 0, 0};
    if (point.x > voxel_center_[0]) {
        xyz[0] = 1;
    }
    if (point.y > voxel_center_[1]) {
        xyz[1] = 1;
    }
    if (point.z > voxel_center_[2]) {
        xyz[2] = 1;
    }
    int leafnum = 4 * xyz[0] + 2 * xyz[1] + xyz[2];
    if (verbose) {
        std::cout << point.x << " " << point.y << " " << point.z << std::endl;
        std::cout << "LOC[locate], " << "Layer: " << layer_ << "/" << max_layer_ << " " << leafnum << " " << leaves_[leafnum] << std::endl;
    }
    if (leaves_[leafnum] != nullptr) {
        return leaves_[leafnum]->LocateOctree(point, terminal_layer);
    } else {
        return nullptr;
    }
}

uint64_t OctoTree::LocateCode(const OctoTree::Point & point) {
    double locate_point[3] = { point.x -  world_origin_[0],
                               point.y -  world_origin_[1],
                               point.z -  world_origin_[2] };
    uint32_t x = locate_point[0] * g_grid_inv_size_;
    uint32_t y = locate_point[1] * g_grid_inv_size_;
    uint32_t z = locate_point[2] * g_grid_inv_size_;
    uint64_t locate_code = z * GRID_SLICE + y * GRID_LENGTH + x;
    return locate_code;
}

uint64_t OctoTree::CreateTime() { return create_time_; }

bool OctoTree::ExistPoint(const OctoTree::Point & point) {
    double locate_point[3] = { point.x -  world_origin_[0],
                               point.y -  world_origin_[1],
                               point.z -  world_origin_[2] };
    uint32_t x = locate_point[0] * g_grid_inv_size_;
    uint32_t y = locate_point[1] * g_grid_inv_size_;
    uint32_t z = locate_point[2] * g_grid_inv_size_;
    uint64_t locate_code = z * GRID_SLICE + y * GRID_LENGTH + x;
    auto loc = grid_points_.find(locate_code);
    if (loc != grid_points_.end() && loc->second > 0) {
        return true;
    }
    return false;
}

void OctoTree::GetAllLeafs(std::vector<OctoTree*> & leafs) {
    if (layer_ >= max_layer_ || voxel_->IsFeature()) {
        leafs.push_back(this);
        return;
    }
    for (size_t i = 0; i < 8; ++i) {
        if (leaves_[i]) {
            leaves_[i]->GetAllLeafs(leafs);
        }
    }
}

void OctoTree::GetLayerLeafs(std::vector<OctoTree*> & leafs) {
    leafs.clear();
    leafs.shrink_to_fit();
    for (size_t i = 0; i < 8; ++i) {
        if (leaves_[i]) {
            leafs.emplace_back(leaves_[i]);
        }
    }
}

Voxel* OctoTree::GetVoxel() {
    return voxel_;
}

void OctoTree::GetGridPoints(std::vector<Eigen::Vector3d> & points) {
    points.reserve(all_points_num_);
    for (auto & grid_point : grid_points_) {
        if (grid_point.second == 0) continue;
        uint32_t z = grid_point.first * GRID_INV_SLICE;
        uint32_t yx = grid_point.first - z * GRID_SLICE;
        uint32_t y = yx * GRID_INV_LENGTH;
        uint32_t x = yx - y * GRID_LENGTH;
        Eigen::Vector3d XYZ(x * g_grid_size_ + world_origin_[0], 
                            y * g_grid_size_ + world_origin_[1], 
                            z * g_grid_size_ + world_origin_[2]);
        points.push_back(XYZ);
    }
}

void OctoTree::GetLidarTimes(std::unordered_set<uint32_t> & lidar_times) { lidar_times = lidar_times_; }

void OctoTree::GetTreePoints(std::vector<Eigen::Vector3d> & points) {
    for (auto & point : tree_points_) {
        if (point.second == 0) continue;
        uint32_t z = point.first * GRID_INV_SLICE;
        uint32_t yx = point.first - z * GRID_SLICE;
        uint32_t y = yx * GRID_INV_LENGTH;
        uint32_t x = yx - y * GRID_LENGTH;
        Eigen::Vector3d XYZ(x * g_grid_size_ + world_origin_[0], 
                            y * g_grid_size_ + world_origin_[1], 
                            z * g_grid_size_ + world_origin_[2]);
        points.push_back(XYZ);
    }
}

void OctoTree::GetOctreeCorners(std::vector<Eigen::Vector3d> & points) {
    Eigen::Vector3d corner;
    corner[0] = voxel_center_[0] - 0.5 * voxel_size_;
    corner[1] = voxel_center_[1] - 0.5 * voxel_size_;
    corner[2] = voxel_center_[2] - 0.5 * voxel_size_;
    points.push_back(corner);

    corner[0] = voxel_center_[0] + 0.5 * voxel_size_;
    corner[1] = voxel_center_[1] - 0.5 * voxel_size_;
    corner[2] = voxel_center_[2] - 0.5 * voxel_size_;
    points.push_back(corner);

    corner[0] = voxel_center_[0] + 0.5 * voxel_size_;
    corner[1] = voxel_center_[1] + 0.5 * voxel_size_;
    corner[2] = voxel_center_[2] - 0.5 * voxel_size_;
    points.push_back(corner);

    corner[0] = voxel_center_[0] - 0.5 * voxel_size_;
    corner[1] = voxel_center_[1] + 0.5 * voxel_size_;
    corner[2] = voxel_center_[2] - 0.5 * voxel_size_;
    points.push_back(corner);

    corner[0] = voxel_center_[0] - 0.5 * voxel_size_;
    corner[1] = voxel_center_[1] - 0.5 * voxel_size_;
    corner[2] = voxel_center_[2] + 0.5 * voxel_size_;
    points.push_back(corner);

    corner[0] = voxel_center_[0] + 0.5 * voxel_size_;
    corner[1] = voxel_center_[1] - 0.5 * voxel_size_;
    corner[2] = voxel_center_[2] + 0.5 * voxel_size_;
    points.push_back(corner);

    corner[0] = voxel_center_[0] + 0.5 * voxel_size_;
    corner[1] = voxel_center_[1] + 0.5 * voxel_size_;
    corner[2] = voxel_center_[2] + 0.5 * voxel_size_;
    points.push_back(corner);

    corner[0] = voxel_center_[0] - 0.5 * voxel_size_;
    corner[1] = voxel_center_[1] + 0.5 * voxel_size_;
    corner[2] = voxel_center_[2] + 0.5 * voxel_size_;
    points.push_back(corner);
}

void OctoTree::InsertPointInternal(const uint64_t & locate_code, const uint32_t &createtime, const int count = 1) {
    if (update_enable_) {
        grid_points_[locate_code] += count;
        if (grid_point_times_.find(locate_code) != grid_point_times_.end()) {
            grid_point_times_[locate_code] = std::min(createtime, grid_point_times_[locate_code]);
        } else {
            grid_point_times_[locate_code] = createtime;
        }
        new_points_num_ += count;
        all_points_num_ += count;
    }
}

void OctoTree::InitVoxel() {
    std::vector<Eigen::Vector3d> points;
    points.reserve(grid_points_.size());
    for (auto & grid_point : grid_points_) {
        if (grid_point.second == 0) continue;
        uint32_t z = grid_point.first * GRID_INV_SLICE;
        uint32_t yx = grid_point.first - z * GRID_SLICE;
        uint32_t y = yx * GRID_INV_LENGTH;
        uint32_t x = yx - y * GRID_LENGTH;
        Eigen::Vector3d XYZ(x * g_grid_size_ + world_origin_[0], 
                            y * g_grid_size_ + world_origin_[1], 
                            z * g_grid_size_ + world_origin_[2]);
        points.insert(points.end(), grid_point.second, XYZ);
    }
    voxel_->Init(points);
}

void OctoTree::AppendToOctoTreeInternal(const Point & point, const int count, bool eval, const double verbose = false) {
    if (!update_enable_) {
        return;
    }
    new_points_num_ += count;
    all_points_num_ += count;

    double locate_point[3] = { point.x -  world_origin_[0],
                               point.y -  world_origin_[1],
                               point.z -  world_origin_[2] };
    uint64_t x = locate_point[0] * g_grid_inv_size_;
    uint64_t y = locate_point[1] * g_grid_inv_size_;
    uint64_t z = locate_point[2] * g_grid_inv_size_;
    uint64_t locate_code = z * GRID_SLICE + y * GRID_LENGTH + x;
    grid_points_[locate_code] += count;
    lidar_time_points_[point.lifetime]++;
    lidar_times_.insert(point.lifetime);

    lastest_time_ = std::max(point.lifetime, lastest_time_);
    voxel_->Add(Eigen::Vector3d(&point.x));
    if (layer_ < max_layer_) {
        int xyz[3] = {0, 0, 0};
        if (point.x > voxel_center_[0]) {
            xyz[0] = 1;
        }
        if (point.y > voxel_center_[1]) {
            xyz[1] = 1;
        }
        if (point.z > voxel_center_[2]) {
            xyz[2] = 1;
        }
        const double leaf_voxel_size = 0.5 * voxel_size_;
        const double quater_length = 0.5 * leaf_voxel_size;
        int leafnum = 4 * xyz[0] + 2 * xyz[1] + xyz[2];
        if (leaves_[leafnum] == nullptr) {
            leaves_[leafnum] = new OctoTree(
                leaf_voxel_size, g_grid_size_, max_layer_, layer_ + 1, layer_point_size_, min_layer_point_size_, max_layer_point_size_, point.lifetime, voxel_option_);
            leaves_[leafnum]->layer_point_size_ = layer_point_size_;
            leaves_[leafnum]->voxel_center_[0] = voxel_center_[0] + (2 * xyz[0] - 1) * quater_length;
            leaves_[leafnum]->voxel_center_[1] = voxel_center_[1] + (2 * xyz[1] - 1) * quater_length;
            leaves_[leafnum]->voxel_center_[2] = voxel_center_[2] + (2 * xyz[2] - 1) * quater_length;
            leaves_[leafnum]->world_origin_[0] = world_origin_[0];
            leaves_[leafnum]->world_origin_[1] = world_origin_[1];
            leaves_[leafnum]->world_origin_[2] = world_origin_[2];
        }
        leaves_[leafnum]->AppendToOctoTreeInternal(point, count, eval);
        if (verbose) {
            std::cout << point.x << " " << point.y << " " << point.z << std::endl;
            std::cout << "LOC, " << "Layer: " << layer_ << "/" << max_layer_ << " " << leafnum << " " << leaves_[leafnum] << std::endl;
        }
    }
    
    // if (new_points_num_ > update_size_threshold_) {
        voxel_->ComputeFeature();
        new_points_num_ = 0;
    // }

    if (voxel_->IsDetermined() &&
        voxel_->min_eval_ < voxel_option_.plane_min_eigen_eval &&
        voxel_->min_max_eigen_ratio_ < voxel_option_.plane_min_max_eigen_ratio &&
        voxel_->min_mid_eigen_ratio_ < voxel_option_.plane_min_mid_eigen_ratio &&
        voxel_->mid_max_eigen_ratio_ > voxel_option_.plane_mid_max_eigen_ratio &&
        all_points_num_ >= 20/*min_points_size_*/) {
        bool planarity = true;
        for (int i = 0; i < 8; ++i) {
            if (leaves_[i]) {
                if (!leaves_[i]->voxel_->IsFeature() && leaves_[i]->all_points_num_ >= 20/*min_points_size_*/) {
                    planarity = false;
                    break;
                }
            }
        }
        if (planarity) {
            const double cos_thres = std::cos(DEG2RAD(10.0));
            Eigen::Vector3d m_pivot = voxel_->GetPivot().normalized();
            Eigen::Vector3d m_var = voxel_->GetEx();
            for (int i = 0; i < 8; ++i) {
                if (leaves_[i] && leaves_[i]->voxel_->IsFeature()) {
                    Eigen::Vector3d m_sub_pivot = leaves_[i]->voxel_->GetPivot().normalized();
                    Eigen::Vector3d m_sub_var = leaves_[i]->voxel_->GetEx();
                    double pivot_diff = std::abs(m_sub_pivot.dot(m_pivot));
                    double dist = std::abs(m_pivot.dot(m_sub_var - m_var));
                    if (pivot_diff < cos_thres || dist > 0.1) {
                        planarity = false;
                        break;
                    }
                }
            }
        }
        voxel_->SetFeature(planarity);
    } else {
        voxel_->SetFeature(false);
    }
    if (voxel_->IsFeature()) {
        if (lifetime_ == 0) {
            lifetime_ = lastest_time_;
        }
        if (all_points_num_ >= max_points_size_) {
            update_enable_ = false;
        }
    }
}

void OctoTree::AppendToOctoTreeInternal(const uint64_t & locate_code, const uint32_t lifetime, const int count = 1) {
    if (!init_octo_) {
        new_points_num_ += count;
        all_points_num_ += count;
        grid_points_[locate_code] += count;
        if (grid_point_times_.find(locate_code) != grid_point_times_.end()) {
            grid_point_times_[locate_code] = std::min(lifetime, grid_point_times_[locate_code]);
        } else {
            grid_point_times_[locate_code] = lifetime;
        }
        if (all_points_num_ > max_feature_update_threshold_) {
            InitOctoTree();
        }
    } else if (update_enable_) {
        new_points_num_ += count;
        all_points_num_ += count;
        grid_points_[locate_code] += count;
        if (grid_point_times_.find(locate_code) != grid_point_times_.end()) {
            grid_point_times_[locate_code] = std::min(lifetime, grid_point_times_[locate_code]);
        } else {
            grid_point_times_[locate_code] = lifetime;
        }
        // new_grid_points_[locate_code] += count;
        uint32_t z = locate_code * GRID_INV_SLICE;
        uint32_t yx = locate_code- z * GRID_SLICE;
        uint32_t y = yx * GRID_INV_LENGTH;
        uint32_t x = yx - y * GRID_LENGTH;
        Eigen::Vector3d new_point(x * g_grid_size_ + world_origin_[0], 
                                  y * g_grid_size_ + world_origin_[1], 
                                  z * g_grid_size_ + world_origin_[2]);
        voxel_->Add(new_point);
        if (new_points_num_ > update_size_threshold_) {
            voxel_->ComputeFeature();
            new_points_num_ = 0;
        }
        if (voxel_->IsFeature()) {
            if (all_points_num_ >= max_points_size_) {
                octo_state_ = 0;
                update_enable_ = false;
            }
            if (lifetime_ == 0) {
                for (auto & grid_point : grid_point_times_) {
                    if (grid_point.second == 0) continue;
                    lifetime_ = std::max(lifetime_, (uint64_t)grid_point.second);
                }
            }
        } else if (layer_ >= max_layer_) {
            octo_state_ = 0;
            update_enable_ = true;
        } else {
            octo_state_ = 1;
            update_enable_ = true;

            bool split = false;
            for (int i = 0; i < 8; ++i) {
                if (leaves_[i]) {
                    split = true;
                    break;
                }
            }

            std::vector<std::pair<uint64_t, int> > grid_points;
            if (!split) {
                grid_points.reserve(all_points_num_);
                for (auto grid_point : grid_points_) {
                    grid_points.emplace_back(grid_point.first, grid_point.second);
                }
            } else {
                grid_points.emplace_back(locate_code, 1);
            }

            for (auto & grid_point : grid_points) {
                uint32_t z = grid_point.first * GRID_INV_SLICE;
                uint32_t yx = grid_point.first - z * GRID_SLICE;
                uint32_t y = yx * GRID_INV_LENGTH;
                uint32_t x = yx - y * GRID_LENGTH;
                Eigen::Vector3d new_point(x * g_grid_size_ + world_origin_[0], 
                                            y * g_grid_size_ + world_origin_[1], 
                                            z * g_grid_size_ + world_origin_[2]);
                int xyz[3] = {0, 0, 0};
                if (new_point[0] > voxel_center_[0]) {
                    xyz[0] = 1;
                }
                if (new_point[1] > voxel_center_[1]) {
                    xyz[1] = 1;
                }
                if (new_point[2] > voxel_center_[2]) {
                    xyz[2] = 1;
                }

                const double leaf_voxel_size = 0.5 * voxel_size_;
                const double quater_length = 0.5 * leaf_voxel_size;
                int leafnum = 4 * xyz[0] + 2 * xyz[1] + xyz[2];
                if (leaves_[leafnum] == nullptr) {
                    leaves_[leafnum] = new OctoTree(
                        leaf_voxel_size, g_grid_size_, max_layer_, layer_ + 1, layer_point_size_, min_layer_point_size_, max_layer_point_size_, lifetime, voxel_option_);
                    leaves_[leafnum]->layer_point_size_ = layer_point_size_;
                    leaves_[leafnum]->voxel_center_[0] = voxel_center_[0] + (2 * xyz[0] - 1) * quater_length;
                    leaves_[leafnum]->voxel_center_[1] = voxel_center_[1] + (2 * xyz[1] - 1) * quater_length;
                    leaves_[leafnum]->voxel_center_[2] = voxel_center_[2] + (2 * xyz[2] - 1) * quater_length;
                    leaves_[leafnum]->world_origin_[0] = world_origin_[0];
                    leaves_[leafnum]->world_origin_[1] = world_origin_[1];
                    leaves_[leafnum]->world_origin_[2] = world_origin_[2];
                }
                leaves_[leafnum]->AppendToOctoTreeInternal(grid_point.first, lifetime, grid_point.second);
            }
        }
    }
}

void OctoTree::AppendToOctoTreeInternalLazy(const uint64_t & locate_code, const uint64_t lifetime, const int count) {
    if (!update_enable_) { return ; }
    voxel_dirty_ = true;
    grid_points_[locate_code] += count;
    if (grid_point_times_.find(locate_code) != grid_point_times_.end()) {
        grid_point_times_[locate_code] = std::min(grid_point_times_[locate_code], (uint32_t)lifetime);
    } else {
        grid_point_times_[locate_code] = lifetime;
    }

    all_points_num_ += count;

    if (layer_ >= max_layer_) { return; }

    uint32_t z = locate_code * GRID_INV_SLICE;
    uint32_t yx = locate_code - z * GRID_SLICE;
    uint32_t y = yx * locate_code;
    uint32_t x = yx - y * GRID_LENGTH;
    Eigen::Vector3d new_point(x * g_grid_size_ + world_origin_[0], 
                            y * g_grid_size_ + world_origin_[1], 
                            z * g_grid_size_ + world_origin_[2]);
    int xyz[3] = {0, 0, 0};
    if (new_point[0] > voxel_center_[0]) {
        xyz[0] = 1;
    }
    if (new_point[1] > voxel_center_[1]) {
        xyz[1] = 1;
    }
    if (new_point[2] > voxel_center_[2]) {
        xyz[2] = 1;
    }
    int leafnum = 4 * xyz[0] + 2 * xyz[1] + xyz[2];
    if (leaves_[leafnum]) {
        leaves_[leafnum]->AppendToOctoTreeInternalLazy(locate_code, lifetime, count);
    }
}

void OctoTree::AppendToOctoTreeInternalLazy(const Point & point, const int count) {
    voxel_dirty_ = true;
    all_points_num_ += count;

    double locate_point[3] = { point.x -  world_origin_[0],
                               point.y -  world_origin_[1],
                               point.z -  world_origin_[2] };
    uint64_t x = locate_point[0] * g_grid_inv_size_;
    uint64_t y = locate_point[1] * g_grid_inv_size_;
    uint64_t z = locate_point[2] * g_grid_inv_size_;
    uint64_t locate_code = z * GRID_SLICE + y * GRID_LENGTH + x;
    grid_points_[locate_code] += count;
    lidar_time_points_[point.lifetime]++;
    lidar_times_.insert(point.lifetime);

    lastest_time_ = std::max(lastest_time_, point.lifetime);
    voxel_->Add(Eigen::Vector3d(&point.x));

    if (layer_ >= max_layer_) { return; }

    int xyz[3] = {0, 0, 0};
    if (point.x > voxel_center_[0]) {
        xyz[0] = 1;
    }
    if (point.y > voxel_center_[1]) {
        xyz[1] = 1;
    }
    if (point.z > voxel_center_[2]) {
        xyz[2] = 1;
    }
    const double leaf_voxel_size = 0.5 * voxel_size_;
    const double quater_length = 0.5 * leaf_voxel_size;
    int leafnum = 4 * xyz[0] + 2 * xyz[1] + xyz[2];
    if (leaves_[leafnum] == nullptr) {
        leaves_[leafnum] = new OctoTree(
            leaf_voxel_size, g_grid_size_, max_layer_, layer_ + 1, layer_point_size_, min_layer_point_size_, max_layer_point_size_, point.lifetime, voxel_option_);
        leaves_[leafnum]->layer_point_size_ = layer_point_size_;
        leaves_[leafnum]->voxel_center_[0] = voxel_center_[0] + (2 * xyz[0] - 1) * quater_length;
        leaves_[leafnum]->voxel_center_[1] = voxel_center_[1] + (2 * xyz[1] - 1) * quater_length;
        leaves_[leafnum]->voxel_center_[2] = voxel_center_[2] + (2 * xyz[2] - 1) * quater_length;
        leaves_[leafnum]->world_origin_[0] = world_origin_[0];
        leaves_[leafnum]->world_origin_[1] = world_origin_[1];
        leaves_[leafnum]->world_origin_[2] = world_origin_[2];
    }
    leaves_[leafnum]->AppendToOctoTreeInternalLazy(point, count);
}

void OctoTree::RemoveFromOctoTreeInternalLazy(const uint64_t & locate_code) {
    voxel_dirty_ = true;
    update_enable_ = true;
    auto old_loc = grid_points_.find(locate_code);
    if (old_loc != grid_points_.end()) {
        CHECK_GT(old_loc->second, 0);
        grid_points_[locate_code]--;
        if (grid_points_[locate_code] == 0) {
            grid_points_.erase(locate_code);
            grid_point_times_.erase(locate_code);
        }
        all_points_num_--;
        new_points_num_ = 0;
        CHECK(all_points_num_ >= 0);

        // TODO: reset status
        if (all_points_num_ <= max_feature_update_threshold_) {
            update_enable_ = true;
            new_points_num_ = 0;
            lifetime_ = 0;
            voxel_->Reset();
        }
    }

    if (layer_ >= max_layer_) { return; }

    uint32_t z = locate_code * GRID_INV_SLICE;
    uint32_t yx = locate_code - z * GRID_SLICE;
    uint32_t y = yx * locate_code;
    uint32_t x = yx - y * GRID_LENGTH;
    Eigen::Vector3d old_point(x * g_grid_size_ + world_origin_[0], 
                            y * g_grid_size_ + world_origin_[1], 
                            z * g_grid_size_ + world_origin_[2]);
    int xyz[3] = {0, 0, 0};
    if (old_point[0] > voxel_center_[0]) {
        xyz[0] = 1;
    }
    if (old_point[1] > voxel_center_[1]) {
        xyz[1] = 1;
    }
    if (old_point[2] > voxel_center_[2]) {
        xyz[2] = 1;
    }
    int leafnum = 4 * xyz[0] + 2 * xyz[1] + xyz[2];
    if (leaves_[leafnum]) {
        leaves_[leafnum]->RemoveFromOctoTreeInternalLazy(locate_code);
    }
}

void OctoTree::RemoveFromOctoTreeInternalLazy(const Point & point) {
    double locate_point[3] = { point.x -  world_origin_[0],
                               point.y -  world_origin_[1],
                               point.z -  world_origin_[2] };
    uint64_t x = locate_point[0] * g_grid_inv_size_;
    uint64_t y = locate_point[1] * g_grid_inv_size_;
    uint64_t z = locate_point[2] * g_grid_inv_size_;
    uint64_t locate_code = z * GRID_SLICE + y * GRID_LENGTH + x;
    auto iter = grid_points_.find(locate_code);
    if (iter != grid_points_.end()) {
        if (iter->second > 1) {
            grid_points_[locate_code]--;
        } else if (iter->second == 1) {
            grid_points_.erase(locate_code);
        }
        if (lidar_time_points_.find(point.lifetime) != lidar_time_points_.end()) {
            lidar_time_points_[point.lifetime]--;
            if (lidar_time_points_[point.lifetime] == 0) {
                lidar_time_points_.erase(point.lifetime);
                lidar_times_.erase(point.lifetime);
            }
        }
        voxel_dirty_ = true;
        all_points_num_--;
        new_points_num_ = 0;

        if (all_points_num_ < max_points_size_) {
            update_enable_ = true;
        }

        CHECK(all_points_num_ >= 0);

        voxel_->Sub(Eigen::Vector3d(&point.x));

        if (layer_ >= max_layer_) { return; }

        int xyz[3] = {0, 0, 0};
        if (point.x > voxel_center_[0]) {
            xyz[0] = 1;
        }
        if (point.y > voxel_center_[1]) {
            xyz[1] = 1;
        }
        if (point.z > voxel_center_[2]) {
            xyz[2] = 1;
        }
        int leafnum = 4 * xyz[0] + 2 * xyz[1] + xyz[2];
        // if (leaves_[leafnum]->all_points_num_ <= 0) {
        //     std::cout << "layer: " << layer_ << std::endl;
        //     std::cout << "all_points_num_: " << leaves_[leafnum]->all_points_num_ << std::endl;
        //     std::cout << "grid_points: " << leaves_[leafnum]->grid_points_.size() << std::endl;
        // }
        // assert(leaves_[leafnum]->all_points_num_ > 0);
        if (leaves_[leafnum] && leaves_[leafnum]->all_points_num_ > 0) {
            leaves_[leafnum]->RemoveFromOctoTreeInternalLazy(point);
        }
    }
}

void OctoTree::RemoveFromOctoTreeInternal(const uint64_t & locate_code) {
    // if (update_enable_) {
        bool exist = false;
        auto loc = grid_points_.find(locate_code);
        if (loc != grid_points_.end()) {
            CHECK_GT(loc->second, 0);
            grid_points_[locate_code]--;
            if (grid_points_[locate_code] == 0) {
                grid_points_.erase(locate_code);
                grid_point_times_.erase(locate_code);
            }
            exist = true;
            all_points_num_--;
        }
        if (exist) {
            update_points_num_++;

            uint32_t z = locate_code * GRID_INV_SLICE;
            uint32_t yx = locate_code - z * GRID_SLICE;
            uint32_t y = yx * GRID_INV_LENGTH;
            uint32_t x = yx - y * GRID_LENGTH;
            Eigen::Vector3d point(x * g_grid_size_ + world_origin_[0], 
                                y * g_grid_size_ + world_origin_[1], 
                                z * g_grid_size_ + world_origin_[2]);

            if (init_octo_) {
                voxel_->Sub(point);
                if (update_points_num_ > update_size_threshold_) {
                    voxel_->ComputeFeature();
                    if (voxel_->IsFeature()) {
                        octo_state_ = 0;
                        if (all_points_num_ >= max_points_size_) {
                            update_enable_ = false;
                        } else {
                            update_enable_ = true;
                        }
                    } else {
                        if (layer_ < max_layer_) {
                            octo_state_ = 1;
                        }
                        update_enable_ = true;
                    }
                    update_points_num_ = 0;
                }
            }

            int xyz[3] = {0, 0, 0};
            if (point[0] > voxel_center_[0]) {
                xyz[0] = 1;
            }
            if (point[1] > voxel_center_[1]) {
                xyz[1] = 1;
            }
            if (point[2] > voxel_center_[2]) {
                xyz[2] = 1;
            }
            int leafnum = 4 * xyz[0] + 2 * xyz[1] + xyz[2];
            if (leaves_[leafnum]) {
                leaves_[leafnum]->RemoveFromOctoTreeInternal(locate_code);
            }
        }
    // }
}

double OctoTree::GetVoxelSize() {
    return voxel_size_;
}

} // lidar namespace
} // sensemap namespace