//Copyright (c) 2019, SenseTime Group.
//All rights reserved.

#ifndef SENSEMAP_ESTIMATORS_TRANSLATION_TRANSFORM_H_
#define SENSEMAP_ESTIMATORS_TRANSLATION_TRANSFORM_H_

#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>


#include "util/logging.h"
#include "util/types.h"

namespace sensemap {

// Estimate a N-D translation transformation between point pairs.
template <int kDim>
class TranslationTransformEstimator {
 public:
  typedef Eigen::Matrix<double, kDim, 1> X_t;
  typedef Eigen::Matrix<double, kDim, 1> Y_t;
  typedef Eigen::Matrix<double, kDim, 1> M_t;

  // The minimum number of samples needed to estimate a model.
  static const int kMinNumSamples = 1;

  // Estimate the 2D translation transform.
  //
  // @param points1      Set of corresponding source 2D points.
  // @param points2      Set of corresponding destination 2D points.
  //
  // @return             Translation vector.
  static std::vector<M_t> Estimate(const std::vector<X_t>& points1,
                                   const std::vector<Y_t>& points2);

  // Calculate the squared translation error.
  //
  // @param points1      Set of corresponding source 2D points.
  // @param points2      Set of corresponding destination 2D points.
  // @param translation  Translation vector.
  // @param residuals    Output vector of residuals for each point pair.
  static void Residuals(const std::vector<X_t>& points1,
                        const std::vector<Y_t>& points2, const M_t& translation,
                        std::vector<double>* residuals);
};

////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////

template <int kDim>
std::vector<typename TranslationTransformEstimator<kDim>::M_t>
TranslationTransformEstimator<kDim>::Estimate(const std::vector<X_t>& points1,
                                              const std::vector<Y_t>& points2) {
  CHECK_EQ(points1.size(), points2.size());

  X_t mean_src = X_t::Zero();
  Y_t mean_dst = Y_t::Zero();

  for (size_t i = 0; i < points1.size(); ++i) {
    mean_src += points1[i];
    mean_dst += points2[i];
  }

  mean_src /= points1.size();
  mean_dst /= points2.size();

  std::vector<M_t> models(1);
  models[0] = mean_dst - mean_src;

  return models;
}

template <int kDim>
void TranslationTransformEstimator<kDim>::Residuals(
    const std::vector<X_t>& points1, const std::vector<Y_t>& points2,
    const M_t& translation, std::vector<double>* residuals) {
  CHECK_EQ(points1.size(), points2.size());

  residuals->resize(points1.size());

  for (size_t i = 0; i < points1.size(); ++i) {
    const M_t diff = points2[i] - points1[i] - translation;
    (*residuals)[i] = diff.squaredNorm();
  }
}

}  // namespace sensemap

#endif  // SENSEMAP_ESTIMATORS_TRANSLATION_TRANSFORM_H_
