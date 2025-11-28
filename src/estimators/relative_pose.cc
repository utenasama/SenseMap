//Copyright (c) 2019, SenseTime Group.
//All rights reserved.

#include "util/ceres_types.h"
#include "util/types.h"
#include "util/misc.h"
#include "base/pose.h"
#include "base/matrix.h"
#include "base/essential_matrix.h"
#include "base/cost_functions.h"
#include "estimators/essential_matrix.h"
#include "optim/bundle_adjustment.h"
#include "relative_pose.h"

namespace sensemap {

size_t EstimateRelativePose(const RANSACOptions& ransac_options,
                            const std::vector<Eigen::Vector2d>& points1,
                            const std::vector<Eigen::Vector2d>& points2,
                            Eigen::Vector4d* qvec, Eigen::Vector3d* tvec) {
  RANSAC<EssentialMatrixFivePointEstimator> ransac(ransac_options);
  const auto report = ransac.Estimate(points1, points2);

  if (!report.success) {
    return 0;
  }

  std::vector<Eigen::Vector2d> inliers1(report.support.num_inliers);
  std::vector<Eigen::Vector2d> inliers2(report.support.num_inliers);

  size_t j = 0;
  for (size_t i = 0; i < points1.size(); ++i) {
    if (report.inlier_mask[i]) {
      inliers1[j] = points1[i];
      inliers2[j] = points2[i];
      j += 1;
    }
  }

  Eigen::Matrix3d R;

  std::vector<Eigen::Vector3d> points3D;
  PoseFromEssentialMatrix(report.model, inliers1, inliers2, &R, tvec,
                          &points3D);

  *qvec = RotationMatrixToQuaternion(R);

  if (IsNaN(*qvec) || IsNaN(*tvec)) {
    return 0;
  }

  return points3D.size();
}

bool RefineRelativePose(const ceres::Solver::Options& options,
                        const std::vector<Eigen::Vector2d>& points1,
                        const std::vector<Eigen::Vector2d>& points2,
                        Eigen::Vector4d* qvec, Eigen::Vector3d* tvec) {
  CHECK_EQ(points1.size(), points2.size());

  // CostFunction assumes unit quaternions.
  *qvec = NormalizeQuaternion(*qvec);

  const double kMaxL2Error = 1.0;
  ceres::LossFunction* loss_function = new ceres::CauchyLoss(kMaxL2Error);

  ceres::Problem problem;

  for (size_t i = 0; i < points1.size(); ++i) {
    ceres::CostFunction* cost_function =
        RelativePoseCostFunction::Create(points1[i], points2[i]);
    problem.AddResidualBlock(cost_function, loss_function, qvec->data(),
                             tvec->data());
  }

  ceres::LocalParameterization* quaternion_parameterization =
      new ceres::QuaternionParameterization;
#if CERES_VERSION_MAJOR >= 2 && CERES_VERSION_MINOR >= 1
        problem.SetManifold(qvec->data(), quaternion_parameterization);
#else
        problem.SetParameterization(qvec->data(), quaternion_parameterization);
#endif

  ceres::SphereParameterization* homogeneous_parameterization = new ceres::SphereParameterization(3);
#if CERES_VERSION_MAJOR >= 2 && CERES_VERSION_MINOR >= 1
        problem.SetManifold(tvec->data(), homogeneous_parameterization);
#else
        problem.SetParameterization(tvec->data(), homogeneous_parameterization);
#endif

  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  return summary.IsSolutionUsable();
}

} // namespace sensemap