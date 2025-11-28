//Copyright (c) 2024, SenseTime Group.
//All rights reserved.

#ifndef SENSEMAP_LIDAR_ABSOLUTE_POSE_H_
#define SENSEMAP_LIDAR_ABSOLUTE_POSE_H_

#include <array>
#include <vector>

#include <Eigen/Dense>

#include "util/alignment.h"
#include "util/types.h"
#include "util/logging.h"
#include "util/threading.h"
#include "base/camera.h"
#include "optim/ransac/loransac.h"

#include "lidar_sweep.h"
#include "voxel_map.h"

namespace sensemap {

struct LidarAbsolutePoseRefinementOptions {

    int num_iteration_pose_estimation = 2;

    double error_tolerance = 1e-4;

    // Convergence criterion.
    double gradient_tolerance = 1.0;

    // Maximum number of solver iterations.
    int max_num_iterations = 100;

    // Scaling factor determines at which residual robustification takes place.
    double loss_function_scale = 1.0;

    // Whether to print final summary.
    bool print_summary = true;

    void Check() const {
        CHECK_GE(gradient_tolerance, 0.0);
        CHECK_GE(max_num_iterations, 0);
        CHECK_GE(loss_function_scale, 0.0);
    }
};

bool RefineAbsolutePose(const LidarAbsolutePoseRefinementOptions & options, 
                        Reconstruction *reconstruction,
                        const image_t image_id,
                        const std::vector<lidar::OctoTree::Point> & points, 
                        VoxelMap * voxel_map, 
                        Eigen::Vector4d * qvec, 
                        Eigen::Vector3d * tvec);
}

#endif