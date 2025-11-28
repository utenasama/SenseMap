//Copyright (c) 2019, SenseTime Group.
//All rights reserved.

#include "util/math.h"
#include "robust_rotation.h"

#include <ceres/rotation.h>
#include <graph/utils.h>
#include "optim/solver/l1_solver.h"

namespace sensemap {

namespace {

Eigen::Vector3d MultiplyRotations(const Eigen::Vector3d& rotation1,
                                  const Eigen::Vector3d& rotation2) {
	Eigen::Matrix3d rotation1_mat, rotation2_mat;
	ceres::AngleAxisToRotationMatrix(rotation1.data(), rotation1_mat.data());
	ceres::AngleAxisToRotationMatrix(rotation2.data(), rotation2_mat.data());

	const Eigen::Matrix3d rotation = rotation1_mat * rotation2_mat;
	Eigen::Vector3d rotation_aa;
	ceres::RotationMatrixToAngleAxis(rotation.data(), rotation_aa.data());
	return rotation_aa;
}

} // unnamed namespace

bool RobustRotationEstimator::EstimateRotations(
		const EIGEN_STL_UMAP
				(image_pair_t, struct CorrespondenceGraph::ImagePair)& image_pairs,
		std::unordered_map<image_t , Eigen::Vector3d>* global_orientations) {
	for(const auto& pair : image_pairs){
		Eigen::Vector3d relative_rotation;
		ceres::QuaternionToAngleAxis(/*pair.second.qvec.data(),*/
									 pair.second.two_view_geometry.qvec.data(),
		                             relative_rotation.data());
		AddRelativeRotationConstraint(pair.first, relative_rotation, pair.second.two_view_geometry.confidence);
	}
	return EstimateRotations(global_orientations);
}

void RobustRotationEstimator::AddRelativeRotationConstraint(
		const image_pair_t &view_id_pair,
		const Eigen::Vector3d &relative_rotation,
		const double confidence) {
	relative_rotations_.emplace_back(view_id_pair, relative_rotation);
	relative_rotation_confs_[view_id_pair] = confidence;
}


bool RobustRotationEstimator::EstimateRotations(
		std::unordered_map<image_t, Eigen::Vector3d> *global_orientations) {
	CHECK_GT(relative_rotations_.size(), 0)
		<< "Relative rotation constraints must be added to the robust rotation "
				"solver before estimating global rotations.";
	global_orientations_ = CHECK_NOTNULL(global_orientations);
	// Compute a mapping of view ids to indices in the linear system. One rotation
	// will have an index of -1 and will not be added to the linear system. This
	// will remove the gauge freedom (effectively holding one camera as the
	// identity rotation).
	int index = -1;
	view_id_to_index_.reserve(global_orientations->size());
	for (const auto& orientation : *global_orientations) {
		view_id_to_index_[orientation.first] = index;
		++index;
	}

	Eigen::SparseMatrix<double> sparse_mat;
	SetupLinearSystem();

	if (!SolveL1Regression()) {
		LOG(ERROR) << "Could not solve the L1 regression step.";
		return false;
	}

	if (!SolveIRLS()) {
		LOG(ERROR) << "Could not solve the least squares error step.";
		return false;
	}


//
	return true;
}

void RobustRotationEstimator::SetupLinearSystem() {
	// The rotation change is one less than the number of global rotations because
	// we keep one rotation constant.
//	rotation_change_.resize((global_orientations_->size() - 1) * 3);
//	relative_rotation_error_.resize(relative_rotations_.size() * 3);
//	sparse_matrix_.resize(relative_rotations_.size() * 3,
//	                      (global_orientations_->size() - 1) * 3);

	// For each relative rotation constraint, add an entry to the sparse
	// matrix. We use the first order approximation of angle axis such that:
	// R_ij = R_j - R_i. This makes the sparse matrix just a bunch of identity
	// matrices.
	int rotation_error_index = 0;
	std::vector<Eigen::Triplet<double> > triplet_list;
	for (const auto& relative_rotation : relative_rotations_) {
		image_t image_id1;
		image_t image_id2;
		sensemap::utility::PairIdToImagePair(relative_rotation.first,
		                                     &image_id1, &image_id2);
        if(view_id_to_index_.count(image_id1) == 0 || view_id_to_index_.count(image_id2) == 0){
            continue;
        }

		double weight = 1.0;
#ifdef TWO_VIEW_CONFIDENCE
 		if (options_.has_proir) {
 			weight = options_.prior_weight * relative_rotation_confs_.at(relative_rotation.first);
 		} else {
 			weight = 1.0 + relative_rotation_confs_.at(relative_rotation.first);
 		}
#endif


        const int view1_index =
				FindOrDie(view_id_to_index_, image_id1);
		if (view1_index != kConstantRotationIndex) {
			triplet_list.emplace_back(3 * rotation_error_index,
			                          3 * view1_index,
			                          -weight);
			triplet_list.emplace_back(3 * rotation_error_index + 1,
			                          3 * view1_index + 1,
			                          -weight);
			triplet_list.emplace_back(3 * rotation_error_index + 2,
			                          3 * view1_index + 2,
			                          -weight);
		}

		const int view2_index =
				FindOrDie(view_id_to_index_, image_id2);
		if (view2_index != kConstantRotationIndex) {
			triplet_list.emplace_back(3 * rotation_error_index + 0,
			                          3 * view2_index + 0,
			                          weight);
			triplet_list.emplace_back(3 * rotation_error_index + 1,
			                          3 * view2_index + 1,
			                          weight);
			triplet_list.emplace_back(3 * rotation_error_index + 2,
			                          3 * view2_index + 2,
			                          weight);
		}

		++rotation_error_index;
	}
	if(options_.has_proir){
	    double prior_weight = options_.prior_weight;
	    for(const auto& prior : options_.orientation_proirs){
            const int view_index =
                    FindOrDie(view_id_to_index_, prior.first);
            if(view_index == kConstantRotationIndex)
                continue;
            triplet_list.emplace_back(3 * rotation_error_index + 0,
                                      3 * view_index + 0,
                                      prior_weight);
            triplet_list.emplace_back(3 * rotation_error_index + 1,
                                      3 * view_index + 1,
                                      prior_weight);
            triplet_list.emplace_back(3 * rotation_error_index + 2,
                                      3 * view_index + 2,
                                      prior_weight);
            ++rotation_error_index;
	    }
	}

    rotation_change_.resize((global_orientations_->size() - 1) * 3);
    relative_rotation_error_.resize(rotation_error_index * 3);
    sparse_matrix_.resize(rotation_error_index * 3,
                          (global_orientations_->size() - 1) * 3);

	sparse_matrix_.setFromTriplets(triplet_list.begin(), triplet_list.end());
}

// Computes the relative rotation error based on the current global
// orientation estimates.
void RobustRotationEstimator::ComputeRotationError() {
	int rotation_error_index = 0;
	for (const auto& relative_rotation : relative_rotations_) {
		image_t image_id1;
		image_t image_id2;
		sensemap::utility::PairIdToImagePair(relative_rotation.first,
		                                     &image_id1, &image_id2);

		if(global_orientations_->count(image_id1) == 0 || global_orientations_->count(image_id2) == 0){
			continue;
		}

		double weight = 1.0;
#ifdef TWO_VIEW_CONFIDENCE
		if (options_.has_proir) {
 			weight = options_.prior_weight * relative_rotation_confs_.at(relative_rotation.first);
 		} else {
 			weight = 1.0 + relative_rotation_confs_.at(relative_rotation.first);
 		}
#endif

		const Eigen::Vector3d& relative_rotation_aa = relative_rotation.second;
		const Eigen::Vector3d& rotation1 =
				FindOrDie(*global_orientations_, image_id1);
		const Eigen::Vector3d& rotation2 =
				FindOrDie(*global_orientations_, image_id2);

		// Compute the relative rotation error as:
		//   R_err = R2^t * R_12 * R1.
		relative_rotation_error_.segment<3>(3 * rotation_error_index) = weight * 
				MultiplyRotations(-rotation2,
				                  MultiplyRotations(relative_rotation_aa, rotation1));
		++rotation_error_index;
	}
    if(options_.has_proir) {
        double prior_weight = options_.prior_weight;
        for (const auto &prior : options_.orientation_proirs) {
            const int view_index =
                    FindOrDie(view_id_to_index_, prior.first);
            if (view_index == kConstantRotationIndex)
                continue;
            const Eigen::Vector3d& rotation =
				FindOrDie(*global_orientations_, prior.first);

            relative_rotation_error_.segment<3>(3 * rotation_error_index) =
                    prior_weight * MultiplyRotations( -rotation, prior.second);
            ++rotation_error_index;
        }
    }
}

bool RobustRotationEstimator::SolveL1Regression() {
	static const double kConvergenceThreshold = 1e-3;

	L1Solver<Eigen::SparseMatrix<double> >::Options options;
	options.max_num_iterations = 5;
	L1Solver<Eigen::SparseMatrix<double> > l1_solver(options, sparse_matrix_);

	rotation_change_.setZero();

	for (int i = 0; i < options_.max_num_l1_iterations; i++) {
		ComputeRotationError();
		l1_solver.Solve(relative_rotation_error_, &rotation_change_);
		UpdateGlobalRotations();

		if (relative_rotation_error_.norm() < kConvergenceThreshold) {
			break;
		}
		options.max_num_iterations *= 2;
		l1_solver.SetMaxIterations(options.max_num_iterations);
	}
	return true;
}

bool RobustRotationEstimator::SolveIRLS() {
	static const double kConvergenceThreshold = 1e-3;
	// This is the point where the Huber-like cost function switches from L1 to
	// L2.
	static const double kSigma = DegToRad(5.0);

	// Set up the linear solver and analyze the sparsity pattern of the
	// system. Since the sparsity pattern will not change with each linear solve
	// this can help speed up the solution time.
	SparseCholeskyLLt linear_solver;
	linear_solver.AnalyzePattern(sparse_matrix_.transpose() * sparse_matrix_);
	if (linear_solver.Info() != Eigen::Success) {
		LOG(ERROR) << "Cholesky decomposition failed.";
		return false;
	}

	VLOG(2) << "Iteration   Error           Delta";
	const std::string row_format = "  % 4d     % 4.4e     % 4.4e";

	Eigen::ArrayXd errors, weights;
	Eigen::SparseMatrix<double> at_weight;
	for (int i = 0; i < options_.max_num_irls_iterations; i++) {
		const Eigen::VectorXd prev_rotation_change = rotation_change_;
		ComputeRotationError();

		// Compute the weights for each error term.
		errors =
				(sparse_matrix_ * rotation_change_ - relative_rotation_error_).array();
		weights = kSigma / (errors.square() + kSigma * kSigma).square();

		// Update the factorization for the weighted values.
		at_weight =
				sparse_matrix_.transpose() * weights.matrix().asDiagonal();
		linear_solver.Factorize(at_weight * sparse_matrix_);
		if (linear_solver.Info() != Eigen::Success) {
			LOG(ERROR) << "Failed to factorize the least squares system.";
			return false;
		}

		// Solve the least squares problem..
		rotation_change_ =
				linear_solver.Solve(at_weight * relative_rotation_error_);
		if (linear_solver.Info() != Eigen::Success) {
			LOG(ERROR) << "Failed to solve the least squares system.";
			return false;
		}

		UpdateGlobalRotations();

		// Log some statistics for the output.
		const double rotation_change_sq_norm =
				(prev_rotation_change - rotation_change_).squaredNorm();
		VLOG(2) << StringPrintf(row_format.c_str(), i, errors.square().sum(),
		                        rotation_change_sq_norm);
		if (rotation_change_sq_norm < kConvergenceThreshold) {
			VLOG(1) << "IRLS Converged in " << i + 1 << " iterations.";
			break;
		}
	}
	return true;
}

void RobustRotationEstimator::UpdateGlobalRotations() {
	for (auto& rotation : *global_orientations_) {
		const int view_index = FindOrDie(view_id_to_index_, rotation.first);
		if (view_index == kConstantRotationIndex) {
			continue;
		}

		// Apply the rotation change to the global orientation.
		const Eigen::Vector3d& rotation_change =
				rotation_change_.segment<3>(3 * view_index);
		rotation.second = MultiplyRotations(rotation.second, rotation_change);
	}
}


} // namespace sensemap

