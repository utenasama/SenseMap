//Copyright (c) 2019, SenseTime Group.
//All rights reserved.

#ifndef SENSEMAP_ESTIMATORS_RELATIVE_POSE_H_
#define SENSEMAP_ESTIMATORS_RELATIVE_POSE_H_

#include <vector>

#include <Eigen/Core>

#include <ceres/ceres.h>

#include "base/camera_models.h"
#include "optim/ransac/loransac.h"

#include "util/logging.h"
#include "util/types.h"

namespace sensemap {

// Estimate relative from 2D-2D correspondences.
//
// Pose of first camera is assumed to be at the origin without rotation. Pose
// of second camera is given as world-to-image transformation,
// i.e. `x2 = [R | t] * X2`.
//
// @param ransac_options       RANSAC options.
// @param points1              Corresponding 2D points.
// @param points2              Corresponding 2D points.
// @param qvec                 Estimated rotation component as
//                             unit Quaternion coefficients (w, x, y, z).
// @param tvec                 Estimated translation component.
//
// @return                     Number of RANSAC inliers.
size_t EstimateRelativePose(const RANSACOptions& ransac_options,
                            const std::vector<Eigen::Vector2d>& points1,
                            const std::vector<Eigen::Vector2d>& points2,
                            Eigen::Vector4d* qvec, Eigen::Vector3d* tvec);

// Refine relative pose of two cameras.
//
// Minimizes the Sampson error between corresponding normalized points using
// a robust cost function, i.e. the corresponding points need not necessarily
// be inliers given a sufficient initial guess for the relative pose.
//
// Assumes that first camera pose has projection matrix P = [I | 0], and
// pose of second camera is given as transformation from world to camera system.
//
// Assumes that the given translation vector is normalized, and refines
// the translation up to an unknown scale (i.e. refined translation vector
// is a unit vector again).
//
// @param options          Solver options.
// @param points1          First set of corresponding points.
// @param points2          Second set of corresponding points.
// @param qvec             Unit Quaternion rotation coefficients (w, x, y, z).
// @param tvec             3x1 translation vector.
//
// @return                 Flag indicating if solution is usable.
bool RefineRelativePose(const ceres::Solver::Options& options,
                        const std::vector<Eigen::Vector2d>& points1,
                        const std::vector<Eigen::Vector2d>& points2,
                        Eigen::Vector4d* qvec, Eigen::Vector3d* tvec);

}  // namespace sensemap

#endif  // SENSEMAP_ESTIMATORS_RELATIVE_POSE_H_
