//Copyright (c) 2019, SenseTime Group.
//All rights reserved.

#ifndef SENSEMAP_SRC_ESTIMATORS_AFFINE_TRANSFORM_H_
#define SENSEMAP_SRC_ESTIMATORS_AFFINE_TRANSFORM_H_

#include <vector>

#include <Eigen/Core>

#include "util/alignment.h"
#include "util/types.h"

namespace sensemap {

class AffineTransformEstimator {
 public:
  typedef Eigen::Vector2d X_t;
  typedef Eigen::Vector2d Y_t;
  typedef Eigen::Matrix<double, 2, 3> M_t;

  // The minimum number of samples needed to estimate a model.
  static const int kMinNumSamples = 3;

  // Estimate the affine transformation from at least 3 correspondences.
  static std::vector<M_t> Estimate(const std::vector<X_t>& points1,
								   const std::vector<Y_t>& points2);

  // Compute the squared transformation error.
  static void Residuals(const std::vector<X_t>& points1,
						const std::vector<Y_t>& points2, const M_t& E,
						std::vector<double>* residuals);
};

}  // namespace sensemap

#endif  // SENSEMAP_SRC_ESTIMATORS_AFFINE_TRANSFORM_H_
