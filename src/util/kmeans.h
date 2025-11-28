// Copyright (c) 2019, SenseTime Group.
// All rights reserved.

#ifndef SENSEMAP_UTIL_KMEANS_H_
#define SENSEMAP_UTIL_KMEANS_H_

#include <Eigen/Eigen>
#include <iostream>
#include <unordered_map>
#include <unordered_set>

#include <opencv2/opencv.hpp>
#include "util/ply.h"

namespace sensemap {

struct Tuple {
    Tuple() { name.clear(); }
    ~Tuple() { name.clear(); }
    Eigen::Vector3d location;
    int id;
    std::string name;
    Eigen::Vector3d gps;
    double dist = 0;
};

class KMeans {
public:
    int m_k;
    int srand_seed = -1;
    typedef std::vector<Tuple> VecPoint_t;

    // Cloud to cluster.
    VecPoint_t mv_pntcloud;
    // k kinds cloud.
    std::vector<VecPoint_t> m_grp_pntcloud;
    // Center of every cluster.
    std::vector<Tuple> mv_center;

    int max_iter = 60;  // 1653

    int max_point_size = -1;
    int fixed_size = -1;

    KMeans() { m_k = 0; }

    inline void SetK(int k_) {
        m_k = k_;
        m_grp_pntcloud.resize(m_k);
    }

    inline void SetMaxIter(int num) {
        max_iter = num;
    }

    inline void SetFixedSize(int size) {
        fixed_size = size;
    }

    void MovetoCenter();
    void SetRandSeed(int seed);

    //kmeans++
    double GetClosestDist(Tuple& point, std::vector<Tuple>& centers);
    bool PlusInit();
    bool PlusCluster();

    //kmeans
    bool InitKCenter();
    bool Cluster();
    bool UpdateGroupCenter();
    bool ComputeGroupCenter();
    double DistBetweenPoints(const Tuple& p1, const Tuple& p2);
    bool ExistCenterShift(std::vector<Tuple>& prev_center, std::vector<Tuple>& cur_center);

    //kmean same size
    bool SameSizeCluster();
    void FindNeighborsAndCommonPoints(
        std::vector<std::unordered_map<int, std::vector<Tuple>>>& neighbors_points,
        std::vector<std::vector<int>>& neighbors);
    void FindNeighborsAndCommonPoints_EdgeNearest(
        std::vector<std::unordered_map<int, std::vector<Tuple>>>& neighbors_points,
        std::vector<std::vector<int>>& neighbors);
    void FindNeighborsAndCommonPoints_AllPoints(
        std::vector<std::unordered_map<int, std::vector<Tuple>>>& neighbors_points,
        std::vector<std::vector<int>>& neighbors);

    //kmean same size with connection
    float ComputeConnectScore(const int point_id, 
        const VecPoint_t& tar_pntcloud,
        const std::vector<std::vector<int>>& connection,
        const std::unordered_set<int>& grp_ids = std::unordered_set<int>());
    
    bool FilterWeakConnect(std::unordered_set<int>& filter_point_id, const int grp_id,
        const std::vector<std::vector<int>>& connection, 
        const std::unordered_set<int>& const_ids, 
        const float score_thr = 0.25);
    bool SortConnectScore(std::vector<std::pair<int, float>>& points_score, 
        const std::vector<std::vector<int>>& connection,
        const std::unordered_set<int>& const_ids,         
        const VecPoint_t& tar_pntcloud, const VecPoint_t& pntcloud);
    bool SortDistScore(std::vector<std::pair<int, float>>& points_score, 
        const std::unordered_set<int>& const_ids, 
        const VecPoint_t& tar_pntcloud, const VecPoint_t& pntcloud);

    bool UpdateGrpIds(const VecPoint_t& pntclout, std::unordered_set<int>& const_ids);
    
    bool SameSizeClusterWithConnection(const std::vector<std::vector<int>> connection);
    
    void FindNeighborsAndCommonPointsWithConnection(
        std::vector<std::unordered_map<int, std::vector<Tuple>>>& neighbors_points,
        std::vector<std::vector<int>>& neighbors,
        const std::vector<std::vector<int>>& connection);
    void FindNeighborsAndCommonPointsWithConnection_AllPoints(
        std::vector<std::unordered_map<int, std::vector<Tuple>>>& neighbors_points,
        std::vector<std::vector<int>>& neighbors,
        const std::vector<std::vector<int>>& connection);

    // output cluster result with PLY
    bool WritePointCloud(std::string SavePlyPath);
    bool WritePointCloud(std::string SavePlyPath, 
                        std::vector<VecPoint_t> grp_pntcloud);

private:
    const float DIST_NEAR_ZERO = 0.001;  // 0.001
};

}  // namespace sensemap
#endif
