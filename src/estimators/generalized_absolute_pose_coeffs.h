// Copyright (c) 2019, SenseTime Group.
// All rights reserved.

#ifndef SENSEMAP_ESTIMATORS_GENERALIZED_ABSOLUTE_POSE_COEFFS_H_
#define SENSEMAP_ESTIMATORS_GENERALIZED_ABSOLUTE_POSE_COEFFS_H_

#include <Eigen/Core>

namespace sensemap {

Eigen::Matrix<double, 9, 1> ComputeDepthsSylvesterCoeffs(
    const Eigen::Matrix<double, 3, 6>& K);

}  // namespace sensemap

#endif  // SENSEMAP_ESTIMATORS_GENERALIZED_ABSOLUTE_POSE_COEFFS_H_
