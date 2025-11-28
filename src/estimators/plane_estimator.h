//Copyright (c) 2019, SenseTime Group.
//All rights reserved.

#ifndef SENSEMAP_ESTIMATORS_PLANE_ESTIMATOR_H_
#define SENSEMAP_ESTIMATORS_PLANE_ESTIMATOR_H_

#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/SVD>

// #include "util/alignment.h"
#include "util/types.h"

namespace sensemap {

// Plane Estimator from points.
class PlaneEstimator {
 public:
  typedef Eigen::Vector3d X_t;
  typedef Eigen::Vector3d Y_t;
  typedef Eigen::Vector4d M_t;

  // The minimum number of samples needed to estimate a model.
  static const int kMinNumSamples = 3;

  // Estimate Plane Equation.
  //
  // The number of points must be at least 3.
  //  
  // @param points     Set of points.
  //
  // @return           Plane Equation parameters.
  static std::vector<M_t> Estimate(const std::vector<X_t>& points);

  // Calculate point to plane error for each point.
  //
  // @param points     Set of points.
  // @param H          Plane Equation vector.
  // @param residuals  Output vector of residuals.
  static void Residuals(const std::vector<X_t>& points, const M_t& H,
                        std::vector<double>* residuals);
};

class PlaneLocalEstimator {
public:
  typedef Eigen::Vector3d X_t;
  typedef Eigen::Vector3d Y_t;
  typedef Eigen::Vector4d M_t;

  // The minimum number of samples needed to estimate a model.
  static const int kMinNumSamples = 3;

  // Estimate Plane Equation.
  //
  // The number of points must be at least 3.
  //  
  // @param points     Set of points.
  //
  // @return           Plane Equation parameters.
  static std::vector<M_t> Estimate(const std::vector<X_t>& points);

  // Calculate point to plane error for each point.
  //
  // @param points     Set of points.
  // @param H          Plane Equation vector.
  // @param residuals  Output vector of residuals.
  static void Residuals(const std::vector<X_t>& points, const M_t& H,
                        std::vector<double>* residuals);
};

// Plane Estimator from points.
class WeightedPlaneEstimator {
 public:
  typedef Eigen::Vector6d X_t;
  typedef Eigen::Vector6d Y_t;
  typedef Eigen::Vector4d M_t;

  // The minimum number of samples needed to estimate a model.
  static const int kMinNumSamples = 3;

  // Estimate Plane Equation.
  //
  // The number of points must be at least 3.
  //  
  // @param points     Set of points.
  //
  // @return           Plane Equation parameters.
  static std::vector<M_t> Estimate(const std::vector<X_t>& points);

  // Calculate point to plane error for each point.
  //
  // @param points     Set of points.
  // @param H          Plane Equation vector.
  // @param residuals  Output vector of residuals.
  static void Residuals(const std::vector<X_t>& points, const M_t& H,
                        std::vector<double>* residuals);
};

class WeightedPlaneLocalEstimator {
public:
  typedef Eigen::Vector6d X_t;
  typedef Eigen::Vector6d Y_t;
  typedef Eigen::Vector4d M_t;

  // The minimum number of samples needed to estimate a model.
  static const int kMinNumSamples = 3;

  // Estimate Plane Equation.
  //
  // The number of points must be at least 3.
  //  
  // @param points     Set of points.
  //
  // @return           Plane Equation parameters.
  static std::vector<M_t> Estimate(const std::vector<X_t>& points);

  // Calculate point to plane error for each point.
  //
  // @param points     Set of points.
  // @param H          Plane Equation vector.
  // @param residuals  Output vector of residuals.
  static void Residuals(const std::vector<X_t>& points, const M_t& H,
                        std::vector<double>* residuals);
};

}  // namespace sensemap

#endif  // SENSEMAP_ESTIMATORS_PLANE_ESTIMATOR_H_
