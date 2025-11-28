//Copyright (c) 2019, SenseTime Group.
//All rights reserved.

#ifndef SENSEMAP_BASE_LIDAR_CORRESPONDENCE_H_
#define SENSEMAP_BASE_LIDAR_CORRESPONDENCE_H_

#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Dense>

#include "base/camera.h"
#include "base/point2d.h"
#include "base/image.h"
#include "base/reconstruction.h"
#include "util/logging.h"
#include "util/math.h"
#include "util/types.h"

#include "mvs/depth_map.h"
#include "util/ply.h"
#include "util/misc.h"
#include "util/nanoflann.hpp"

#include "lidar_sweep.h"

#define DISTORTION 0

namespace sensemap {

typedef nanoflann::KDTreeSingleIndexAdaptor<
		nanoflann::L2_Simple_Adaptor<float, LidarPointCloud> ,
		LidarPointCloud,
		3 /* dim */
		> kd_tree_t;

constexpr double SCAN_PERIOD = 0.1;
// constexpr double DISTANCE_SQ_THRESHOLD = 25;
constexpr double DISTANCE_SQ_THRESHOLD = 10
;
constexpr double NEARBY_SCAN = 2.5;
// constexpr double DISTANCE_SQ_THRESHOLD = 2.5;
// constexpr double NEARBY_SCAN = 2.5;

struct LidarEdgeCorrespondence{
    Eigen::Vector3d ref_point_;
    Eigen::Vector3d src_point_a_;
	Eigen::Vector3d src_point_b_;
    int ref_idx = -1;
    int src_a_idx = -1;
    int src_b_idx = -1;
    double s;
    double dist = -1;
};

struct LidarPlaneCorrespondence{
    Eigen::Vector3d ref_point_;
    Eigen::Vector3d src_point_j_;
	Eigen::Vector3d src_point_l_;
    Eigen::Vector3d src_point_m_;
    int ref_idx = -1; 
    int src_j_idx = -1;
    int src_l_idx = -1;
    int src_m_idx = -1;
    double s;
    double dist = -1;
};

struct LidarPointCorrespondence{
    Eigen::Vector3d ref_point_;
    Eigen::Vector3d src_point_;
    int ref_idx = -1;
    int src_idx = -1;
    double s;
    double dist = -1;
};

void TransformSweepToSweep(Eigen::Matrix4d& Tr, PlyPoint const *const pi, 
                           PlyPoint *const po);

double LidarPointsEdgeResidual(const Eigen::Vector3d curr_point_t, 
                         const Eigen::Vector3d last_point_a_t,
					     const Eigen::Vector3d last_point_b_t, 
                         double q1[4], double t1[3], 
                         double q2[4], double t2[3],
                         Eigen::Matrix4d& Tr_ref2src, 
                         Eigen::Matrix4d& Tr_lidar2camera);


double LidarPointsPlaneResidual(const Eigen::Vector3d curr_point_t, 
                                const Eigen::Vector3d last_point_j_t,
                                const Eigen::Vector3d last_point_l_t, 
                                const Eigen::Vector3d last_point_m_t,
                                double q1[4], double t1[3], 
                                double q2[4], double t2[3],
                                Eigen::Matrix4d& Tr_ref2src, 
                                Eigen::Matrix4d& Tr_lidar2camera);

// find correspondence for corner&plane feature
bool LidarPointsCorrespondence(LidarSweep& sweep_ref, 
                LidarSweep& sweep_src,
                Eigen::Matrix4d& Tr_ref2src, 
                std::vector<struct LidarEdgeCorrespondence>& corner_corrs,
                std::vector<struct LidarPlaneCorrespondence>& surf_corrs,
                std::vector<struct LidarPointCorrespondence>& pnt_corrs,
                std::string save_path = "./");

// bool LidarPointsCorrespondence(LidarSweep& sweep_ref, 
//                 Eigen::Matrix4d& Tr_r2w,
//                 Reconstruction *reconstruction,
//                 kd_tree_t & corner_kdtree,
//                 kd_tree_t & surf_kdtree, 
//                 std::unordered_map<size_t, std::vector<size_t>> & corner_corrs,
//                 std::unordered_map<size_t, std::vector<size_t>> & surf_corrs);

// find correspondence for corner feature
int CornerCorrespondence(LidarSweep& sweep_ref, 
                    LidarSweep& sweep_src,
                    Eigen::Matrix4d& Tr_ref2src, 
                    std::vector<struct LidarEdgeCorrespondence>& corner_corrs);
                    
// int CornerCorrespondence(LidarSweep& sweep_ref, 
//                     Eigen::Matrix4d& Tr_ref2w,
//                     LidarPointCloud& global_points,
//                     kd_tree_t & corner_kdtree,
//                     std::unordered_map<size_t, std::vector<size_t>> & corner_corrs);

// find correspondence for plane features
int SurfCorrespondence(LidarSweep& sweep_ref, LidarSweep& sweep_src,
                       Eigen::Matrix4d& Tr_ref2src, 
                       std::vector<struct LidarPlaneCorrespondence>& surf_corrs);

// int SurfCorrespondence(LidarSweep& sweep_ref, 
//                        Eigen::Matrix4d& Tr_ref2w,
//                        LidarPointCloud& global_points, 
//                        kd_tree_t & surf_kdtree, 
//                        std::unordered_map<size_t, std::vector<size_t>> & surf_corrs);

int PntCorrespondence(LidarSweep& sweep_ref, 
                    LidarSweep& sweep_src,
                    Eigen::Matrix4d& Tr_ref2src, 
                    std::vector<struct LidarPointCorrespondence>& pnt_corrs);
                    
} // namespace sensemap

#endif // SENSEMAP_BASE_LIDAR_CORRESPONDENCE_H_