//Copyright (c) 2019, SenseTime Group.
//All rights reserved.

#include <Eigen/Geometry>
#include <unordered_set>
#include <string>

#include "base/projection.h"
#include "base/triangulation.h"
#include "estimators/essential_matrix.h"
#include "optim/ransac/combination_sampler.h"
#include "optim/ransac/loransac.h"
#include "util/logging.h"
#include "util/math.h"

#include "estimators/triangulation.h"

namespace sensemap {

void TriangulationEstimator::SetMinTriAngle(const double min_tri_angle) {
  CHECK_GE(min_tri_angle, 0);
  min_tri_angle_ = min_tri_angle;
}

void TriangulationEstimator::SetResidualType(const ResidualType residual_type) {
  residual_type_ = residual_type;
}

std::vector<TriangulationEstimator::M_t> TriangulationEstimator::Estimate(
	const std::vector<X_t> &point_data,
	const std::vector<Y_t> &pose_data) const{
	CHECK_GE(point_data.size(), 2);
	CHECK_EQ(point_data.size(), pose_data.size());

	if (point_data.size() == 2){
		// Two-view triangulation.
		const M_t xyz = TriangulatePoint(
			*pose_data[0].camera, *pose_data[1].camera, 
			pose_data[0].proj_matrix, pose_data[1].proj_matrix,
			point_data[0].point_normalized, point_data[1].point_normalized,
			point_data[0].point_bearing, point_data[1].point_bearing);

		// Check for cheirality constraint.
		for (const auto &pose : pose_data){
			if(pose.camera->ModelName().compare("SPHERICAL") == 0 ){
				continue;
			}
			if (!HasPointPositiveDepth(pose.proj_matrix, xyz)){
				return std::vector<M_t>();
			}
		}

		if (CalculateTriangulationAngle (pose_data[0].proj_center,
										pose_data[1].proj_center,
										xyz) >= min_tri_angle_){
			return std::vector<M_t>{xyz};
		}
		// if (pose_data[0].camera->ModelName().compare("SPHERICAL") == 0){
		// 	const M_t xyz = TriangulatePointHomogeneous(
		// 		pose_data[0].proj_matrix, pose_data[1].proj_matrix,
		// 		point_data[0].point_bearing, point_data[1].point_bearing);

		// 	if (CalculateTriangulationAngle (pose_data[0].proj_center,
		// 									pose_data[1].proj_center,
		// 									xyz) >= min_tri_angle_){
		// 		return std::vector<M_t>{xyz};
		// 	}
		// }
		// else{
		// 	const M_t xyz = TriangulatePoint(
		// 		pose_data[0].proj_matrix, pose_data[1].proj_matrix,
		// 		point_data[0].point_normalized, point_data[1].point_normalized);

		// 	if (HasPointPositiveDepth(pose_data[0].proj_matrix, xyz) &&
		// 		HasPointPositiveDepth(pose_data[1].proj_matrix, xyz) &&
		// 		CalculateTriangulationAngle(pose_data[0].proj_center,
		// 									pose_data[1].proj_center,
		// 									xyz) >= min_tri_angle_){
		// 		return std::vector<M_t>{xyz};
		// 	}
		// }
	}
	else{
		// Multi-view triangulation.
		std::vector<Camera> cameras;
		cameras.reserve(point_data.size());
		std::vector<Eigen::Matrix3x4d> proj_matrices;
		proj_matrices.reserve(point_data.size());
		std::vector<Eigen::Vector2d> points;
		points.reserve(point_data.size());
		std::vector<Eigen::Vector3d> hpoints;
		hpoints.reserve(point_data.size());
		for (size_t i = 0; i < point_data.size(); ++i){
			cameras.push_back(*pose_data[i].camera);
			proj_matrices.push_back(pose_data[i].proj_matrix);
			points.push_back(point_data[i].point_normalized);
			hpoints.push_back(point_data[i].point_bearing);
		}
		const M_t xyz = TriangulateMultiViewPoint(cameras, proj_matrices, points, hpoints);
		
		// Check for cheirality constraint.
		for (const auto &pose : pose_data){
			if(pose.camera->ModelName().compare("SPHERICAL") == 0 ){
				continue;
			}
			if (!HasPointPositiveDepth(pose.proj_matrix, xyz)){
				return std::vector<M_t>();
			}
		}
		

		// Check for sufficient triangulation angle.
		for (size_t i = 0; i < pose_data.size(); ++i){
			for (size_t j = 0; j < i; ++j){
				const double tri_angle = CalculateTriangulationAngle(
					pose_data[i].proj_center, pose_data[j].proj_center, xyz);
				if (tri_angle >= min_tri_angle_){
					return std::vector<M_t>{xyz};
				}
			}
		}
	}

	return std::vector<M_t>();
}

void TriangulationEstimator::Residuals(const std::vector<X_t>& point_data,
                                       const std::vector<Y_t>& pose_data,
                                       const M_t& xyz,
                                       std::vector<double>* residuals) const {
	CHECK_EQ(point_data.size(), pose_data.size());
	residuals->resize(point_data.size());

	for (size_t i = 0; i < point_data.size(); ++i){
		if (residual_type_ == ResidualType::REPROJECTION_ERROR){

			(*residuals)[i] = CalculateSquaredReprojectionError(
				point_data[i].point, xyz, pose_data[i].proj_matrix,
				*pose_data[i].camera);
		}
		else if (residual_type_ == ResidualType::ANGULAR_ERROR){

			if(pose_data[i].camera->ModelName().compare("SPHERICAL")==0){
				const double angular_error = 
					CalculateAngularErrorSphericalCamera(
									point_data[i].point_bearing, xyz, 
									pose_data[i].proj_matrix);
				(*residuals)[i] = angular_error * angular_error;
			}
			else{
				const double angular_error = CalculateNormalizedAngularError(
					point_data[i].point_normalized, xyz, 
					pose_data[i].proj_matrix);
				(*residuals)[i] = angular_error * angular_error;	
			}	
		}
	}
}

bool EstimateTriangulation(
    const EstimateTriangulationOptions& options,
    const std::vector<TriangulationEstimator::PointData>& point_data,
    const std::vector<TriangulationEstimator::PoseData>& pose_data,
    std::vector<char>* inlier_mask, Eigen::Vector3d* xyz) {
  CHECK_NOTNULL(inlier_mask);
  CHECK_NOTNULL(xyz);
  CHECK_GE(point_data.size(), 2);
  CHECK_EQ(point_data.size(), pose_data.size());
  options.Check();

  // Robustly estimate track using LORANSAC.
  LORANSAC<TriangulationEstimator, TriangulationEstimator,
           InlierSupportMeasurer, CombinationSampler>
      ransac(options.ransac_options);
  ransac.estimator.SetMinTriAngle(options.min_tri_angle);
  ransac.estimator.SetResidualType(options.residual_type);
  ransac.local_estimator.SetMinTriAngle(options.min_tri_angle);
  ransac.local_estimator.SetResidualType(options.residual_type);
  const auto report = ransac.Estimate(point_data, pose_data);
  if (!report.success) {
    return false;
  }

  *inlier_mask = report.inlier_mask;
  *xyz = report.model;

  return report.success;
}

}  // namespace sensemap
