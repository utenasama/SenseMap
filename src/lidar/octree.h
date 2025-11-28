//Copyright (c) 2023, SenseTime Group.
//All rights reserved.

#ifndef SENSEMAP_LIDAR_OCTREE_H_
#define SENSEMAP_LIDAR_OCTREE_H_

#include <vector>
#include <unordered_map>
#include <Eigen/Core>
#include <Eigen/Dense>

#include "voxel.h"
#include "util/types.h"

#include <CGAL/Simple_cartesian.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Search_traits_3.h>
#include <CGAL/Search_traits_adapter.h>
#include <CGAL/Orthogonal_k_neighbor_search.h>

#ifdef CGAL_LINKED_WITH_TBB
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#endif

// #define GRID_LENGTH 1024
// #define GRID_SLICE (GRID_LENGTH * GRID_LENGTH)
const static int GRID_LENGTH = 1024;
const static int GRID_SLICE = GRID_LENGTH * GRID_LENGTH;
const static double GRID_INV_LENGTH = 1.0 / GRID_LENGTH;
const static double GRID_INV_SLICE = 1.0 / GRID_SLICE;

namespace sensemap {
namespace lidar {
typedef CGAL::Exact_predicates_inexact_constructions_kernel Kernel;
typedef Kernel::Point_3                                     Point_3;
typedef CGAL::Search_traits_3<Kernel>                       Traits_base;
typedef CGAL::Orthogonal_k_neighbor_search<Traits_base>     K_neighbor_search;
typedef K_neighbor_search::Tree                             Tree;

class OctoTree {
public:
    struct Point {
        double x, y, z;
        double curv;
        double intensity;
        uint64_t lifetime;
        uint8_t type; // 0: plane, 1: corner
    };
public:
    OctoTree();
    OctoTree(const double voxel_size, const double g_grid_size, const size_t max_layer, const size_t layer, 
             const std::vector<size_t> layer_point_size, const std::vector<size_t> min_layer_point_size, 
             const std::vector<size_t> max_layer_point_size, const uint64_t create_time, const Voxel::Option & option);

    void InsertPoint(const Point & point);

    void InitOctoTree();

    void CutOctoTree();
    
    // OctoTree for plane detection.
    void AppendToOctoTree(const Point & point, const double verbose = false);

    void AppendToOctoTreeLazy(const Point & point);

    // KdTree for querying nearest point of corner point 
    void AppendToKdTree(const Point & point);

    void RemoveFromOctoTree(const Point & point);

    void RemoveFromOctoTreeLazy(const Point & point);

    void RemoveFromKdTree(const Point & point);

    void RebuildOctree();

    void RebuildKdTree();

    bool FindKNeighbor(const Eigen::Vector3d & point, std::vector<Eigen::Vector3d> & nearest_point, const int K);

    OctoTree* LocateOctree(const Point & point, const int terminal_layer, const bool verbose = false);

    uint64_t LocateCode(const Point & point);

    uint64_t CreateTime();

    bool ExistPoint(const Point & point);

    Voxel* GetVoxel();

    void GetAllLeafs(std::vector<OctoTree*> & leafs);

    void GetLayerLeafs(std::vector<OctoTree*> & leafs);

    void GetGridPoints(std::vector<Eigen::Vector3d> & points);

    void GetLidarTimes(std::unordered_set<uint32_t> & lidar_times);

    void GetTreePoints(std::vector<Eigen::Vector3d> & points);

    // Debug
    void GetOctreeCorners(std::vector<Eigen::Vector3d> & points);
    double GetVoxelSize();

private:
    void InsertPointInternal(const uint64_t & locate_code, const uint32_t &createtime, const int count);
    void InitVoxel();

    void AppendToOctoTreeInternal(const uint64_t & locate_code, const uint32_t lifetime, const int count);
    void AppendToOctoTreeInternal(const Point & point, const int count, bool eval, const double verbose);

    void AppendToOctoTreeInternalLazy(const uint64_t & locate_code, const uint64_t lifetime, const int count);
    void AppendToOctoTreeInternalLazy(const Point & point, const int count);

    void RemoveFromOctoTreeInternalLazy(const uint64_t & locate_code);
    void RemoveFromOctoTreeInternalLazy(const Point & point);

    void RemoveFromOctoTreeInternal(const uint64_t & locate_code);

public:
    Voxel* voxel_;
    double voxel_center_[3];
    double world_origin_[3];

    Voxel::Option voxel_option_;

    uint64_t lifetime_ = 0;
    uint64_t lastest_time_ = 0;
    uint64_t create_time_ = 0;

    double voxel_size_;
    double g_grid_size_;
    double g_grid_inv_size_;
    size_t max_layer_;
    size_t layer_;
    int octo_state_; // 0 is end of tree, 1 is not
    std::vector<size_t> layer_point_size_;
    std::vector<size_t> min_layer_point_size_;
    std::vector<size_t> max_layer_point_size_;
    size_t min_points_size_;
    size_t max_points_size_;

    size_t max_feature_update_threshold_;
    size_t update_size_threshold_;

    size_t new_points_num_;
    size_t update_points_num_;
    size_t all_points_num_;
    bool update_enable_;
    bool init_octo_;
    
    bool voxel_dirty_;
    std::unordered_map<uint64_t, int> grid_points_;
    // std::unordered_map<uint64_t, int> new_grid_points_;
    std::unordered_map<uint64_t, uint32_t> grid_point_times_;
    std::unordered_map<uint32_t, uint32_t> lidar_time_points_;
    std::unordered_set<uint32_t> lidar_times_;

    bool tree_dirty_;
    std::unordered_map<uint64_t, int> tree_points_;
    std::unordered_map<uint64_t, float> tree_point_intensities_;
    std::unordered_map<uint64_t, uint32_t> tree_point_times_;
    Tree tree_;

    OctoTree *leaves_[8];
};
} // namespace lidar
} // namespace sensemap

#endif