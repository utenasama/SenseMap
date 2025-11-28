//Copyright (c) 2019, SenseTime Group.
//All rights reserved.
#include "robust_rotation_optimizer.h"
#include "util/logging.h"
#include "utils.h"
#include "optim/solver/l1_solver.h"
#include "optim/solver/sparse_cholesky_llt.h"

namespace sensemap{

bool RobustRotationOptimizer::OptimizeRotations(
	const EIGEN_STL_MAP(ViewIdPair, Eigen::Vector3d) &relative_rotations,
	const view_t constant_view_id,
	EIGEN_STL_UMAP(view_t, Eigen::Vector3d) *global_rotations){
	
	constant_view_id_ = constant_view_id;
	for (const auto &relative_rotation : relative_rotations){
		AddRelativeRotationConstraint(relative_rotation.first, 
									  relative_rotation.second);
	}
	return OptimizeRotations(global_rotations);
}

void RobustRotationOptimizer::AddRelativeRotationConstraint(
	const ViewIdPair &view_id_pair, const Eigen::Vector3d &relative_rotation)
{
	// Store the relative orientation constraint.
	relative_rotations_.emplace_back(view_id_pair, relative_rotation);
}

bool RobustRotationOptimizer::OptimizeRotations(
	EIGEN_STL_UMAP(view_t, Eigen::Vector3d) *global_rotations){

	CHECK_GT(relative_rotations_.size(), 0)
		<< "Relative rotation constraints must be added to the robust rotation "
		   "solver before estimating global rotations.";
	global_rotations_ = CHECK_NOTNULL(global_rotations);

	// Compute a mapping of view ids to indices in the linear system. One rotation
	// will have an index of -1 and will not be added to the linear system. This
	// will remove the gauge freedom (effectively holding one camera as the
	// identity rotation).

	int index = -1;
	view_id_to_index_.reserve(global_rotations->size());

	CHECK(global_rotations->find(constant_view_id_) != global_rotations->end());
    CHECK(global_rotations->at(constant_view_id_) == Eigen::Vector3d::Zero());
    view_id_to_index_[constant_view_id_] = index++;

	for (const auto &orientation : *global_rotations){
		if(orientation.first == constant_view_id_){
			continue;
		}
		view_id_to_index_[orientation.first] = index;
		++index;
	}

	Eigen::SparseMatrix<double> sparse_mat;
	SetupLinearSystem();

	if (!SolveL1Regression()){
		LOG(ERROR) << "Could not solve the L1 regression step.";
		return false;
	}

	if (!SolveIRLS()){
		LOG(ERROR) << "Could not solve the least squares error step.";
		return false;
	}

	return true;
}

// Set up the sparse linear system.
void RobustRotationOptimizer::SetupLinearSystem(){

	// The rotation change is one less than the number of global rotations 
	// because we keep one rotation constant.
	rotation_change_.resize((global_rotations_->size() - 1) * 3);
	relative_rotation_error_.resize(relative_rotations_.size() * 3);
	sparse_matrix_.resize(relative_rotations_.size() * 3,
						  (global_rotations_->size() - 1) * 3);

	// For each relative rotation constraint, add an entry to the sparse
	// matrix. We use the first order approximation of angle axis such that:
	// R_ij = R_j - R_i. This makes the sparse matrix just a bunch of identity
	// matrices.
	int rotation_error_index = 0;
	std::vector<Eigen::Triplet<double>> triplet_list;
	for (const auto &relative_rotation : relative_rotations_){

		const int view1_index = globalmotion::FindOrDie(view_id_to_index_, 
												relative_rotation.first.first);

		if (view1_index != kConstantRotationIndex){
			triplet_list.emplace_back(3 * rotation_error_index,
									  3 * view1_index,
									  -1.0);
			triplet_list.emplace_back(3 * rotation_error_index + 1,
									  3 * view1_index + 1,
									  -1.0);
			triplet_list.emplace_back(3 * rotation_error_index + 2,
									  3 * view1_index + 2,
									  -1.0);
		}

		const int view2_index = globalmotion::FindOrDie(view_id_to_index_, 
												relative_rotation.first.second);
		if (view2_index != kConstantRotationIndex){
			triplet_list.emplace_back(3 * rotation_error_index + 0,
									  3 * view2_index + 0,
									  1.0);
			triplet_list.emplace_back(3 * rotation_error_index + 1,
									  3 * view2_index + 1,
									  1.0);
			triplet_list.emplace_back(3 * rotation_error_index + 2,
									  3 * view2_index + 2,
									  1.0);
		}

		++rotation_error_index;
	}
	sparse_matrix_.setFromTriplets(triplet_list.begin(), triplet_list.end());


	std::cout<<"[RobustRotationOptimizer::SetupLinearSystem] A:"<<std::endl;
	std::cout<<sparse_matrix_<<std::endl;

}

// Computes the relative rotation error based on the current global
// orientation estimates.
void RobustRotationOptimizer::ComputeRotationError(){
	std::cout<<"[RobustRotationOptimizer::ComputeRotationError] "
			 <<"relative_rotation_error construction: "<<std::endl;

	int rotation_error_index = 0;
	for (const auto &relative_rotation : relative_rotations_)
	{
		const Eigen::Vector3d &relative_rotation_aa = relative_rotation.second;
		const Eigen::Vector3d &rotation1 = 
				globalmotion::FindOrDie(*global_rotations_, 
										relative_rotation.first.first);
		const Eigen::Vector3d &rotation2 =
				globalmotion::FindOrDie(*global_rotations_, 
										relative_rotation.first.second);

		// Compute the relative rotation error as:
		//   R_err = R2^t * R_12 * R1.
		relative_rotation_error_.segment<3>(3 * rotation_error_index) =
			globalmotion::MultiplyRotations(-rotation2,
						globalmotion::MultiplyRotations(relative_rotation_aa, 
														rotation1));

		std::cout<<"R_"<<relative_rotation.first.first<<": "<<std::endl;
		std::cout<<rotation1<<std::endl;
		std::cout<<"R_"<<relative_rotation.first.second<<": "<<std::endl;
		std::cout<<rotation2<<std::endl;
		std::cout<<"relative_r: "<<std::endl;
		std::cout<<relative_rotation.second<<std::endl;	
		std::cout<<"R_err:"<<std::endl;
		std::cout<<relative_rotation_error_.segment<3>(3 * rotation_error_index)
				 <<std::endl;

		++rotation_error_index;
	}
	std::cout<<"[RobustRotationOptimizer::ComputeRotationError] "
			 <<"relative_rotation_error"<<std::endl;
	std::cout<<relative_rotation_error_<<std::endl;
}

bool RobustRotationOptimizer::SolveL1Regression(){
	static const double kConvergenceThreshold = 1e-3;

	L1Solver<Eigen::SparseMatrix<double>>::Options options;
	options.max_num_iterations = 5;
	L1Solver<Eigen::SparseMatrix<double>> l1_solver(options, sparse_matrix_);

	rotation_change_.setZero();
	for (int i = 0; i < options_.max_num_l1_iterations; i++){
		ComputeRotationError();
		l1_solver.Solve(relative_rotation_error_, &rotation_change_);
		UpdateGlobalRotations();

		if (relative_rotation_error_.norm() < kConvergenceThreshold){
			break;
		}
		options.max_num_iterations *= 2;
		l1_solver.SetMaxIterations(options.max_num_iterations);
	}
	return true; 
}

// Update the global orientations using the current value in the
// rotation_change.
void RobustRotationOptimizer::UpdateGlobalRotations(){
	for (auto &rotation : *global_rotations_){
		const int view_index = globalmotion::FindOrDie(view_id_to_index_, 
													   rotation.first);
		if (view_index == kConstantRotationIndex){
			continue;
		}

		// Apply the rotation change to the global orientation.
		const Eigen::Vector3d &rotation_change =
			rotation_change_.segment<3>(3 * view_index);
		rotation.second = globalmotion::MultiplyRotations(rotation.second, 
														  rotation_change);
	}
}

bool RobustRotationOptimizer::SolveIRLS(){

	static const double kConvergenceThreshold = 1e-3;
	// This is the point where the Huber-like cost function switches from L1 to
	// L2.
	static const double kSigma = 5.0*M_PI / 180.0;

	// Set up the linear solver and analyze the sparsity pattern of the
	// system. Since the sparsity pattern will not change with each linear solve
	// this can help speed up the solution time.
	SparseCholeskyLLt linear_solver;
	linear_solver.AnalyzePattern(sparse_matrix_.transpose() * sparse_matrix_);
	
	if (linear_solver.Info() != Eigen::Success){
		LOG(ERROR) << "Cholesky decomposition failed.";
		return false;
	}

	VLOG(2) << "Iteration   Error           Delta";
	const std::string row_format = "  % 4d     % 4.4e     % 4.4e";

	Eigen::ArrayXd errors, weights;
	Eigen::SparseMatrix<double> at_weight;
	for (int i = 0; i < options_.max_num_irls_iterations; i++){
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

		if (linear_solver.Info() != Eigen::Success){
			LOG(ERROR) << "Failed to factorize the least squares system.";
			return false;
		}

		// Solve the least squares problem..
		rotation_change_ =
			linear_solver.Solve(at_weight * relative_rotation_error_);
		if (linear_solver.Info() != Eigen::Success){
			LOG(ERROR) << "Failed to solve the least squares system.";
			return false;
		}

		UpdateGlobalRotations();

		// Log some statistics for the output.
		const double rotation_change_sq_norm =
			(prev_rotation_change - rotation_change_).squaredNorm();
		VLOG(2) << StringPrintf(row_format.c_str(), i, errors.square().sum(),
								rotation_change_sq_norm);

		if (rotation_change_sq_norm < kConvergenceThreshold){
			VLOG(1) << "IRLS Converged in " << i + 1 << " iterations.";
			break;
		}
	}
	return true;
	
}

}//namespace sensemap