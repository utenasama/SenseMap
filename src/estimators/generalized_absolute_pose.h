// Copyright (c) 2019, SenseTime Group.
// All rights reserved.

#ifndef SENSEMAP_ESTIMATORS_GENERALIZED_ABSOLUTE_POSE_H_
#define SENSEMAP_ESTIMATORS_GENERALIZED_ABSOLUTE_POSE_H_

#include <vector>

#include <Eigen/Core>

#include "util/alignment.h"
#include "util/types.h"

namespace sensemap {

// Solver for the Generalized P3P problem (NP3P or GP3P), based on:
//
//      Lee, Gim Hee, et al. "Minimal solutions for pose estimation of a
//      multi-camera system." Robotics Research. Springer International
//      Publishing, 2016. 521-538.
//
// This class is based on an original implementation by Federico Camposeco.
class GP3PEstimator {
public:
    // The generalized image observations, which is composed of the relative pose
    // of the specific camera in the generalized camera and its image observation.
    struct X_t {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        // The relative transformation from the generalized camera to the camera
        // frame of the observation.
        Eigen::Matrix3x4d rel_tform;
        // The 2D image feature observation.
        Eigen::Vector2d xy;
    };

    // The observed 3D feature points in the world frame.
    typedef Eigen::Vector3d Y_t;
    // The transformation from the world to the generalized camera frame.
    typedef Eigen::Matrix3x4d M_t;

    // The minimum number of samples needed to estimate a model.
    static const int kMinNumSamples = 3;

    // Estimate the most probable solution of the GP3P problem from a set of
    // three 2D-3D point correspondences.
    static std::vector<M_t> Estimate(const std::vector<X_t>& points2D, const std::vector<Y_t>& points3D);

    // Calculate the squared cosine distance error between the rays given a set of
    // 2D-3D point correspondences and a projection matrix of the generalized
    // camera.
    static void Residuals(const std::vector<X_t>& points2D, const std::vector<Y_t>& points3D, const M_t& proj_matrix,
                          std::vector<double>* residuals);
};

}  // namespace sensemap

EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION_CUSTOM(sensemap::GP3PEstimator::X_t)

#endif  // SENSEMAP_ESTIMATORS_GENERALIZED_ABSOLUTE_POSE_H_
