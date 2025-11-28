#include "voxel_map.h"

#include "util/logging.h"

namespace sensemap {

Voxel::Option VoxelMap::Option::VoxelOption() {
    Voxel::Option option;
    option.verbose = verbose;
    option.line_min_max_eigen_ratio = line_min_max_eigen_ratio;
    option.line_min_mid_eigen_ratio = line_min_mid_eigen_ratio;
    option.plane_mid_max_eigen_ratio = plane_mid_max_eigen_ratio;
    option.plane_min_mid_eigen_ratio = plane_min_mid_eigen_ratio;
    option.plane_min_eigen_eval = plane_min_eigen_eval;
    return option;
}

VoxelMap::VoxelMap(const Option & option, const uint64_t lifetime) 
    : option_(option), lifetime_(lifetime) {}

void VoxelMap::BuildVoxelMap(const std::vector<lidar::OctoTree::Point> & points, const uint64_t lifetime) {
    const Voxel::Option & voxel_option = option_.VoxelOption();
    const uint32_t plsize = points.size();
    for (uint i = 0; i < plsize; i++) {
        Eigen::Map<const Eigen::Vector3d> p_v(&points[i].x);
        float loc_xyz[3];
        for (int j = 0; j < 3; j++) {
            loc_xyz[j] = p_v[j] / option_.voxel_size;
            if (loc_xyz[j] < 0) {
                loc_xyz[j] -= 1.0;
            }
        }
        VOXEL_LOC position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1], (int64_t)loc_xyz[2]);
        auto iter = feat_map_.find(position);
        if (iter == feat_map_.end()) {
            lidar::OctoTree *octo_tree = new lidar::OctoTree(option_.voxel_size, option_.voxel_size / GRID_LENGTH, 
                option_.max_layer, 0, option_.layer_point_size, option_.min_layer_point_size, option_.max_layer_point_size, lifetime, voxel_option);
            feat_map_[position] = octo_tree;
            double voxel_center[3];
            voxel_center[0] = (0.5 + position.x) * option_.voxel_size;
            voxel_center[1] = (0.5 + position.y) * option_.voxel_size;
            voxel_center[2] = (0.5 + position.z) * option_.voxel_size;
            feat_map_[position]->voxel_center_[0] = voxel_center[0];
            feat_map_[position]->voxel_center_[1] = voxel_center[1];
            feat_map_[position]->voxel_center_[2] = voxel_center[2];
            feat_map_[position]->world_origin_[0] = voxel_center[0] - 0.5 * option_.voxel_size;
            feat_map_[position]->world_origin_[1] = voxel_center[1] - 0.5 * option_.voxel_size;
            feat_map_[position]->world_origin_[2] = voxel_center[2] - 0.5 * option_.voxel_size;
        }
        feat_map_[position]->InsertPoint(points[i]);
    }
    std::cout << "Construct Done!" << std::endl << std::flush;
    for (auto iter = feat_map_.begin(); iter != feat_map_.end(); ++iter) {
        iter->second->InitOctoTree();
    }
}

void VoxelMap::AppendToVoxelMap(const std::vector<lidar::OctoTree::Point> & points) {
    const Voxel::Option & voxel_option = option_.VoxelOption();
    uint32_t plsize = points.size();
    for (uint32_t i = 0; i < plsize; i++) {
        Eigen::Map<const Eigen::Vector3d> p_v(&points[i].x);
        float loc_xyz[3];
        for (int j = 0; j < 3; j++) {
            loc_xyz[j] = p_v[j] / option_.voxel_size;
            if (loc_xyz[j] < 0) {
                loc_xyz[j] -= 1.0;
            }
        }
        VOXEL_LOC position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1], (int64_t)loc_xyz[2]);
        // std::cout << "LOC[APP]: " << position.x << " " << position.y << " " << position.z << " " << feat_map_.count(position) << std::endl;
        auto iter = feat_map_.find(position);
        if (iter == feat_map_.end()) {
            lidar::OctoTree *octo_tree = new lidar::OctoTree(option_.voxel_size, option_.voxel_size / GRID_LENGTH, 
                option_.max_layer, 0, option_.layer_point_size, option_.min_layer_point_size, option_.max_layer_point_size, points[i].lifetime, voxel_option);
            double voxel_center[3];
            voxel_center[0] = (0.5 + position.x) * option_.voxel_size;
            voxel_center[1] = (0.5 + position.y) * option_.voxel_size;
            voxel_center[2] = (0.5 + position.z) * option_.voxel_size;
            octo_tree->voxel_center_[0] = voxel_center[0];
            octo_tree->voxel_center_[1] = voxel_center[1];
            octo_tree->voxel_center_[2] = voxel_center[2];
            octo_tree->world_origin_[0] = voxel_center[0] - 0.5 * option_.voxel_size;
            octo_tree->world_origin_[1] = voxel_center[1] - 0.5 * option_.voxel_size;
            octo_tree->world_origin_[2] = voxel_center[2] - 0.5 * option_.voxel_size;
            feat_map_[position] = octo_tree;
        }
        feat_map_.at(position)->AppendToOctoTree(points[i]);
        // feat_map_.at(position)->LocateOctree(points[i]);
    }
}

void VoxelMap::UpdateVoxelMapLazy(const std::vector<lidar::OctoTree::Point> & old_points,
                                  const std::vector<lidar::OctoTree::Point> & new_points) {
    for (uint32_t i = 0; i < old_points.size(); i++) {
        Eigen::Map<const Eigen::Vector3d> old_point(&old_points[i].x);
        float old_loc_xyz[3];
        for (int j = 0; j < 3; j++) {
            old_loc_xyz[j] = old_point[j] / option_.voxel_size;
            if (old_loc_xyz[j] < 0) {
                old_loc_xyz[j] -= 1.0;
            }
        }
        VOXEL_LOC old_position((int64_t)old_loc_xyz[0], (int64_t)old_loc_xyz[1], (int64_t)old_loc_xyz[2]);
        if (feat_map_.find(old_position) == feat_map_.end() ||
            !feat_map_[old_position]->ExistPoint(old_points[i])) {
            continue;
        }

        // use_points[i] = 1;

        feat_map_.at(old_position)->RemoveFromOctoTreeLazy(old_points[i]);
        updated_nodes_.insert(old_position);
    }

    for (uint32_t i = 0; i < new_points.size(); i++) {
        const Eigen::Vector3d new_point(&new_points[i].x);
        float loc_xyz[3];
        for (int j = 0; j < 3; j++) {
            loc_xyz[j] = new_point[j] / option_.voxel_size;
            if (loc_xyz[j] < 0) {
                loc_xyz[j] -= 1.0;
            }
        }
        VOXEL_LOC position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1], (int64_t)loc_xyz[2]);
        auto iter = feat_map_.find(position);
        if (iter == feat_map_.end()) {
            lidar::OctoTree *octo_tree = new lidar::OctoTree(option_.voxel_size, option_.voxel_size / GRID_LENGTH, 
                option_.max_layer, 0, option_.layer_point_size, option_.min_layer_point_size, option_.max_layer_point_size, new_points[i].lifetime, option_.VoxelOption());
            double voxel_center[3];
            voxel_center[0] = (0.5 + position.x) * option_.voxel_size;
            voxel_center[1] = (0.5 + position.y) * option_.voxel_size;
            voxel_center[2] = (0.5 + position.z) * option_.voxel_size;
            octo_tree->voxel_center_[0] = voxel_center[0];
            octo_tree->voxel_center_[1] = voxel_center[1];
            octo_tree->voxel_center_[2] = voxel_center[2];
            octo_tree->world_origin_[0] = voxel_center[0] - 0.5 * option_.voxel_size;
            octo_tree->world_origin_[1] = voxel_center[1] - 0.5 * option_.voxel_size;
            octo_tree->world_origin_[2] = voxel_center[2] - 0.5 * option_.voxel_size;
            feat_map_[position] = octo_tree;
        }
        feat_map_.at(position)->AppendToOctoTreeLazy(new_points[i]);
        updated_nodes_.insert(position);
    }
}

void VoxelMap::RebuildOctree() {
    std::vector<VOXEL_LOC> locs;
    locs.reserve(updated_nodes_.size());
    for (auto loc : updated_nodes_) {
        locs.push_back(loc);
    }
#pragma omp parallel for
    for (int i = 0; i < locs.size(); ++i) {
        feat_map_.at(locs[i])->RebuildOctree();
    }
    updated_nodes_.clear();
}

void VoxelMap::AppendToKdTree(const std::vector<lidar::OctoTree::Point> & points) {
    const Voxel::Option & voxel_option = option_.VoxelOption();
    float loc_xyz[3];
    for (uint32_t i = 0; i < points.size(); i++) {
        Eigen::Map<const Eigen::Vector3d> p_v(&points[i].x);
        for (int j = 0; j < 3; j++) {
            loc_xyz[j] = p_v[j] / option_.voxel_size;
            if (loc_xyz[j] < 0) {
                loc_xyz[j] -= 1.0;
            }
        }
        VOXEL_LOC position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1], (int64_t)loc_xyz[2]);

        if (corner_feat_map_.find(position) == corner_feat_map_.end()) {
            corner_feat_map_[position] = new lidar::OctoTree(option_.voxel_size, option_.voxel_size / GRID_LENGTH, 
                option_.max_layer, 0, option_.layer_point_size, option_.min_layer_point_size, option_.max_layer_point_size, points[i].lifetime, voxel_option);
            double voxel_center[3];
            voxel_center[0] = (0.5 + position.x) * option_.voxel_size;
            voxel_center[1] = (0.5 + position.y) * option_.voxel_size;
            voxel_center[2] = (0.5 + position.z) * option_.voxel_size;
            corner_feat_map_[position]->voxel_center_[0] = voxel_center[0];
            corner_feat_map_[position]->voxel_center_[1] = voxel_center[1];
            corner_feat_map_[position]->voxel_center_[2] = voxel_center[2];
            corner_feat_map_[position]->world_origin_[0] = voxel_center[0] - 0.5 * option_.voxel_size;
            corner_feat_map_[position]->world_origin_[1] = voxel_center[1] - 0.5 * option_.voxel_size;
            corner_feat_map_[position]->world_origin_[2] = voxel_center[2] - 0.5 * option_.voxel_size;
        }
        corner_feat_map_[position]->AppendToKdTree(points[i]);
        updated_treenodes_.insert(position);
    }
}

void VoxelMap::UpdateTree(const std::vector<lidar::OctoTree::Point> & old_points,
                          const std::vector<lidar::OctoTree::Point> & new_points) {
    CHECK_EQ(old_points.size(), new_points.size());
    
    const Voxel::Option & voxel_option = option_.VoxelOption();
    float loc_xyz[3];

    for (uint32_t i = 0; i < old_points.size(); i++) {
        // Remove old point.
        Eigen::Map<const Eigen::Vector3d> old_point(&old_points[i].x);
        for (int j = 0; j < 3; j++) {
            loc_xyz[j] = old_point[j] / option_.voxel_size;
            if (loc_xyz[j] < 0) {
                loc_xyz[j] -= 1.0;
            }
        }
        VOXEL_LOC old_position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1], (int64_t)loc_xyz[2]);
        if (corner_feat_map_.find(old_position) == corner_feat_map_.end()) {
            continue;
        }
        uint64_t old_locate_code = corner_feat_map_.at(old_position)->LocateCode(old_points[i]);

        // Append new point.
        Eigen::Map<const Eigen::Vector3d> new_point(&new_points[i].x);
        for (int j = 0; j < 3; j++) {
            loc_xyz[j] = new_point[j] / option_.voxel_size;
            if (loc_xyz[j] < 0) {
                loc_xyz[j] -= 1.0;
            }
        }
        VOXEL_LOC new_position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1], (int64_t)loc_xyz[2]);
        if (corner_feat_map_.find(new_position) == corner_feat_map_.end()) {
            lidar::OctoTree *octo_tree = new lidar::OctoTree(option_.voxel_size, option_.voxel_size / GRID_LENGTH, 
                option_.max_layer, 0, option_.layer_point_size, option_.min_layer_point_size, option_.max_layer_point_size, new_points[i].lifetime, voxel_option);
            corner_feat_map_[new_position] = octo_tree;
            double voxel_center[3];
            voxel_center[0] = (0.5 + new_position.x) * option_.voxel_size;
            voxel_center[1] = (0.5 + new_position.y) * option_.voxel_size;
            voxel_center[2] = (0.5 + new_position.z) * option_.voxel_size;
            corner_feat_map_[new_position]->voxel_center_[0] = voxel_center[0];
            corner_feat_map_[new_position]->voxel_center_[1] = voxel_center[1];
            corner_feat_map_[new_position]->voxel_center_[2] = voxel_center[2];
            corner_feat_map_[new_position]->world_origin_[0] = voxel_center[0] - 0.5 * option_.voxel_size;
            corner_feat_map_[new_position]->world_origin_[1] = voxel_center[1] - 0.5 * option_.voxel_size;
            corner_feat_map_[new_position]->world_origin_[2] = voxel_center[2] - 0.5 * option_.voxel_size;
        }

        uint64_t new_locate_code = corner_feat_map_.at(new_position)->LocateCode(new_points[i]);
        if (old_position == new_position && old_locate_code == new_locate_code) {
            continue;
        }

        corner_feat_map_.at(old_position)->RemoveFromKdTree(old_points[i]);
        corner_feat_map_.at(new_position)->AppendToKdTree(new_points[i]);

        updated_treenodes_.insert(old_position);
        updated_treenodes_.insert(new_position);
    }
}

void VoxelMap::RebuildTree() {
    std::vector<VOXEL_LOC> locs;
    locs.reserve(updated_treenodes_.size());
    for (auto loc : updated_treenodes_) {
        locs.push_back(loc);
    }
#pragma omp parallel for
    for (int i = 0; i < locs.size(); ++i) {
        corner_feat_map_.at(locs[i])->RebuildKdTree();
    }
    updated_treenodes_.clear();
}

std::vector<lidar::OctoTree*> VoxelMap::AbstractTreeVoxels() {
    std::vector<lidar::OctoTree*> octree_list;
    for (auto voxel_map : corner_feat_map_) {
        octree_list.push_back(voxel_map.second);
    }
    return octree_list;
}

lidar::OctoTree* VoxelMap::LocateOctree(const lidar::OctoTree::Point & point, const int terminal_layer) {
    Eigen::Map<const Eigen::Vector3d> xyz(&point.x);
    float loc_xyz[3];
    for (int j = 0; j < 3; j++) {
        loc_xyz[j] = xyz[j] / option_.voxel_size;
        if (loc_xyz[j] < 0) {
            loc_xyz[j] -= 1.0;
        }
    }
    VOXEL_LOC position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1], (int64_t)loc_xyz[2]);
    // std::cout << "LOC[locate]: " << position.x << " " << position.y << " " << position.z << " " << feat_map_.count(position) << std::endl;
    if (feat_map_.find(position) != feat_map_.end()) {
        return feat_map_[position]->LocateOctree(point, terminal_layer);
    } else {
        return nullptr;
    }
}

lidar::OctoTree* VoxelMap::LocateRoot(const lidar::OctoTree::Point & point) {
    Eigen::Map<const Eigen::Vector3d> xyz(&point.x);
    float loc_xyz[3];
    for (int j = 0; j < 3; j++) {
        loc_xyz[j] = xyz[j] / option_.voxel_size;
        if (loc_xyz[j] < 0) {
            loc_xyz[j] -= 1.0;
        }
    }
    VOXEL_LOC position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1], (int64_t)loc_xyz[2]);
    if (feat_map_.find(position) != feat_map_.end()) {
        return feat_map_[position];
    } else if (corner_feat_map_.find(position) != corner_feat_map_.end()) {
        return corner_feat_map_[position];
    } else {
        return nullptr;
    }   
}

lidar::OctoTree* VoxelMap::LocateCornerPoint(const lidar::OctoTree::Point & point) {
    Eigen::Map<const Eigen::Vector3d> xyz(&point.x);
    float loc_xyz[3];
    for (int j = 0; j < 3; j++) {
        loc_xyz[j] = xyz[j] / option_.voxel_size;
        if (loc_xyz[j] < 0) {
            loc_xyz[j] -= 1.0;
        }
    }
    VOXEL_LOC position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1], (int64_t)loc_xyz[2]);
    if (corner_feat_map_.find(position) != corner_feat_map_.end()) {
        return corner_feat_map_[position];
    } else {
        return nullptr;
    }
}

bool VoxelMap::FindNearestNeighborPlane(const lidar::OctoTree::Point & point, Eigen::Vector3d & n_var, Eigen::Vector3d & n_pivot, Eigen::Matrix3d & n_inv_cov) {
    Eigen::Vector3d xyz(point.x, point.y, point.z);
    float loc_xyz[3];
    for (int j = 0; j < 3; j++) {
        loc_xyz[j] = xyz[j] / option_.voxel_size;
        if (loc_xyz[j] < 0) {
            loc_xyz[j] -= 1.0;
        }
    }
    std::vector<VOXEL_LOC> positions;
    positions.reserve(9);
    VOXEL_LOC position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1], (int64_t)loc_xyz[2]);
    VOXEL_LOC temp_position;
    for (int dz = -1; dz <= 1; ++dz) {
        for (int dx = -1; dx <= 1; ++dx) {
            for (int dy = -1; dy <= 1; ++dy) {
                temp_position.x = position.x + dx;
                temp_position.y = position.y + dy;
                temp_position.z = position.z + dz;
                positions.push_back(temp_position);
            }
        }
    }
    std::vector<lidar::OctoTree*> octree_list;
    for (VOXEL_LOC loc : positions) {
        if (feat_map_.find(loc) != feat_map_.end()) {
            std::vector<lidar::OctoTree*> sub_octree_list;
            feat_map_[loc]->GetAllLeafs(sub_octree_list);
            octree_list.insert(octree_list.end(), sub_octree_list.begin(), sub_octree_list.end());
        }
    }

    bool find = false;
    double min_dist = std::numeric_limits<double>::max();
    for (lidar::OctoTree* octree : octree_list) {
        Voxel * voxel = octree->GetVoxel();
        Eigen::Vector3d m_var = voxel->GetEx();
        Eigen::Vector3d m_pivot = voxel->GetPivot();
        Eigen::Matrix3d m_inv_cov = voxel->GetInvCov();

        double proj_len = (xyz - m_var).dot(m_pivot);
        if (min_dist > proj_len) {
            min_dist = proj_len;
            n_var = m_var;
            n_pivot = m_pivot;
            n_inv_cov = m_inv_cov;
            find = true;
        }
    }
    return find;
}

bool VoxelMap::FindKNeighborExact(const Eigen::Vector3d & point, std::vector<Eigen::Vector3d> & nearest_point, lidar::OctoTree* &nearest_octree, const int K) {
    float loc_xyz[3];
    for (int j = 0; j < 3; j++) {
        loc_xyz[j] = point[j] / option_.voxel_size;
        if (loc_xyz[j] < 0) {
            loc_xyz[j] -= 1.0;
        }
    }
    std::vector<VOXEL_LOC> positions;
    positions.reserve(9);
    VOXEL_LOC position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1], (int64_t)loc_xyz[2]);
    VOXEL_LOC temp_position;
    for (int dz = -1; dz <= 1; ++dz) {
        for (int dx = -1; dx <= 1; ++dx) {
            for (int dy = -1; dy <= 1; ++dy) {
                temp_position.x = position.x + dx;
                temp_position.y = position.y + dy;
                temp_position.z = position.z + dz;
                positions.push_back(temp_position);
            }
        }
    }
    std::vector<lidar::OctoTree*> octree_list;
    octree_list.reserve(27);
    for (VOXEL_LOC loc : positions) {
        if (corner_feat_map_.find(loc) != corner_feat_map_.end()) {
            octree_list.push_back(corner_feat_map_[loc]);
        }
    }
    
    bool find = false;
    double min_dist = std::numeric_limits<double>::max();
    std::vector<Eigen::Vector3d> near_points;
    near_points.resize(1);
    for (auto octree : octree_list) {
        if (octree->FindKNeighbor(point, near_points, 1)) {
            for (auto near_p : near_points) {
                double dist = (point - near_p).norm();
                if (min_dist > dist) {
                    min_dist = dist;
                    nearest_point = std::vector<Eigen::Vector3d>{near_p};
                    nearest_octree = octree;
                    find = true;
                }
            }
        }
    }

    return find;
}

std::vector<lidar::OctoTree*> VoxelMap::AbstractFeatureVoxels(bool force) {
    std::vector<lidar::OctoTree*> octree_list;
    for (auto voxel_map : feat_map_) {
        std::vector<lidar::OctoTree*> sub_octree_list;
        voxel_map.second->GetAllLeafs(sub_octree_list);
        // octree_list.insert(octree_list.end(), sub_octree_list.begin(), sub_octree_list.end());
        for (auto sub_octree : sub_octree_list) {
            if (force || sub_octree->GetVoxel()->IsFeature()) {
                octree_list.push_back(sub_octree);
            }
        }
    }
    return octree_list;
}

std::vector<lidar::OctoTree*> VoxelMap::AbstractSweepFeatureVoxels(const std::vector<lidar::OctoTree::Point> & points) {
    std::vector<lidar::OctoTree*> octree_list;
    for (auto voxel_map : feat_map_) {
        std::vector<lidar::OctoTree*> sub_octree_list;
        voxel_map.second->GetAllLeafs(sub_octree_list);
        octree_list.insert(octree_list.end(), sub_octree_list.begin(), sub_octree_list.end());
    }

    std::vector<Eigen::Vector6d> aabbs;
    aabbs.resize(octree_list.size());
    for(int idx = 0; idx < octree_list.size(); idx++){
        const auto& octree = octree_list.at(idx);

        auto& pnt = aabbs.at(idx);
        pnt[0] = octree->voxel_center_[0] - 0.5 * octree->GetVoxelSize();
        pnt[1] = octree->voxel_center_[1] - 0.5 * octree->GetVoxelSize();
        pnt[2] = octree->voxel_center_[2] - 0.5 * octree->GetVoxelSize();
        pnt[3] = octree->voxel_center_[0] + 0.5 * octree->GetVoxelSize();
        pnt[4] = octree->voxel_center_[1] + 0.5 * octree->GetVoxelSize();
        pnt[5] = octree->voxel_center_[2] + 0.5 * octree->GetVoxelSize();
    }

    std::set<int> select_octrees;
    for (uint32_t i = 0; i < points.size(); i++) {
        for (int idx = 0; idx < aabbs.size(); idx++){
            const auto& pnt = aabbs.at(idx);
            if (points[i].x > pnt[0] && points[i].y > pnt[1] && points[i].z > pnt[2] && 
                points[i].x < pnt[3] && points[i].y < pnt[4] && points[i].z < pnt[5]){
                select_octrees.insert(idx);
            }
        }
    }

    std::vector<lidar::OctoTree*> select_octree_list;
    select_octree_list.reserve(select_octrees.size());
    for (const auto& idx : select_octrees){
        select_octree_list.push_back(octree_list.at(idx));
    }
    
    return select_octree_list;
}

}