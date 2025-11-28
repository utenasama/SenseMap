//Copyright (c) 2023, SenseTime Group.
//All rights reserved.

#ifndef SENSEMAP_LIDAR_VOXEL_MAP_H_
#define SENSEMAP_LIDAR_VOXEL_MAP_H_

#define HASH_P 116101
#define MAX_N 10000000000

#include <unordered_map>
#include <unordered_set>
#include <string>

#include "lidar/octree.h"
#include "util/types.h"

class VOXEL_LOC {
public:
  int64_t x, y, z;

  VOXEL_LOC(int64_t vx = 0, int64_t vy = 0, int64_t vz = 0)
      : x(vx), y(vy), z(vz) {}

  bool operator==(const VOXEL_LOC &other) const {
    return (x == other.x && y == other.y && z == other.z);
  }
};

// Hash value
namespace std {
template <> struct hash<VOXEL_LOC> {
  int64_t operator()(const VOXEL_LOC &s) const {
    using std::hash;
    using std::size_t;
    return ((((s.z) * HASH_P) % MAX_N + (s.y)) * HASH_P) % MAX_N + (s.x);
  }
};
} // namespace std

namespace sensemap {

class VoxelMap {
public:
    struct Option {
        double voxel_size;
        size_t max_layer;
        std::vector<size_t> layer_point_size;
        std::vector<size_t> min_layer_point_size;
        std::vector<size_t> max_layer_point_size;

        bool verbose = false;
        double plane_min_eigen_eval = 0.03;
        double plane_min_max_eigen_ratio = 0.03;
        double plane_min_mid_eigen_ratio = 0.03;
        double plane_mid_max_eigen_ratio = 0.6;
        double line_min_max_eigen_ratio = 0.01;
        double line_min_mid_eigen_ratio = 0.9;

        Voxel::Option VoxelOption();
    };
public:
    VoxelMap(const Option & option, const uint64_t lifetime);

    void BuildVoxelMap(const std::vector<lidar::OctoTree::Point> & points, const uint64_t lifetime);

    void AppendToVoxelMap(const std::vector<lidar::OctoTree::Point> & points);

    // void UpdateVoxelMap(const std::vector<lidar::OctoTree::Point> & old_points,
    //                     const std::vector<lidar::OctoTree::Point> & new_points);

    void UpdateVoxelMapLazy(const std::vector<lidar::OctoTree::Point> & old_points,
                            const std::vector<lidar::OctoTree::Point> & new_points);

    void RebuildOctree();

    lidar::OctoTree* LocateOctree(const lidar::OctoTree::Point & point, const int terminal_layer);

    lidar::OctoTree* LocateRoot(const lidar::OctoTree::Point & point);

    lidar::OctoTree* LocateCornerPoint(const lidar::OctoTree::Point & point);

    bool FindNearestNeighborPlane(const lidar::OctoTree::Point & point, Eigen::Vector3d & n_var, Eigen::Vector3d & n_pivot, Eigen::Matrix3d & n_inv_cov);

    // DEBUG.
    std::vector<lidar::OctoTree*> AbstractFeatureVoxels(bool force = false);
    std::vector<lidar::OctoTree*> AbstractSweepFeatureVoxels(const std::vector<lidar::OctoTree::Point> & points);

    void AppendToKdTree(const std::vector<lidar::OctoTree::Point> & points);

    void UpdateTree(const std::vector<lidar::OctoTree::Point> & old_points,
                    const std::vector<lidar::OctoTree::Point> & new_points);

    bool FindKNeighborExact(const Eigen::Vector3d & point, std::vector<Eigen::Vector3d> & nearest_point, lidar::OctoTree* &nearest_octree, const int K);
    
    void RebuildTree();

    std::vector<lidar::OctoTree*> AbstractTreeVoxels();

private:
    Option option_;
    std::unordered_map<VOXEL_LOC, lidar::OctoTree*> feat_map_;
    std::unordered_set<VOXEL_LOC> updated_nodes_;

    std::unordered_map<VOXEL_LOC, lidar::OctoTree*> corner_feat_map_;
    std::unordered_set<VOXEL_LOC> updated_treenodes_;

    uint64_t lifetime_;
};

}

#endif