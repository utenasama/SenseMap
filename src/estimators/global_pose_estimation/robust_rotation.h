//Copyright (c) 2019, SenseTime Group.
//All rights reserved.

#ifndef SENSEMAP_ESTIMATORS_GLOBAL_ESTIMATION_ROBUST_ROTATION_H_
#define SENSEMAP_ESTIMATORS_GLOBAL_ESTIMATION_ROBUST_ROTATION_H_

#include <unordered_map>
#include <Eigen/Eigen>
#include <vector>

#include "util/types.h"
#include "global_rotation.h"
#include "graph/maximum_spanning_tree_graph.h"

namespace sensemap {

// Computes the global rotations given relative rotations and an initial guess
// for the global orientations. The robust algorithm of "Efficient and Large
// Scale Rotation Averaging" by Chatterjee and Govindu (ICCV 2013) is used to
// obtain accurate solutions that are robust to outliers.
//
// The general strategy of this algorithm is to minimize the relative rotation
// error (using the difference between relative rotations and the corresponding
// global rotations) with L1 minimization first, then a reweighted least
// squares. The L1 minimization is relatively slow, but provides excellent
// robustness to outliers. Then the L2 minimization (which is much faster) can
// refine the solution to be very accurate.
class RobustRotationEstimator : public GlobalRotationEstimator {
public:
	struct Options {
		// Maximum number of times to run L1 minimization. L1 is very slow (compared
		// to L2), but is very robust to outliers. Typically only a few iterations
		// are needed in order for the solution to reside within the cone of
		// convergence for L2 solving.
		int max_num_l1_iterations = 5;

		// The number of iterative reweighted least squares iterations to perform.
		int max_num_irls_iterations = 100;

		bool has_proir = false;

		double prior_weight = 1.0f;
        // The global orientation priors for each camera.
        std::map<camera_t, Eigen::Vector3d> orientation_proirs;
	};

	explicit RobustRotationEstimator(const Options& options)
			: options_(options) {}

	// Estimates the global orientations of all views based on an initial
	// guess. Returns true on successful estimation and false otherwise.
	bool EstimateRotations(
			const EIGEN_STL_UMAP(image_pair_t, struct CorrespondenceGraph::ImagePair) &image_pairs,
			std::unordered_map<image_t , Eigen::Vector3d>* orientations) override;

	// An alternative interface is to instead add relative rotation constraints
	// one by one with AddRelativeRotationConstraint, then call the
	// EstimateRotations interface below. This allows the caller to add multiple
	// constraints for the same view id pair, which may lead to more accurate
	// rotation estimates. Please see the following reference for an example of
	// how to obtain multiple constraints for pairs of views:
	//
	//   "Parallel Structure from Motion from Local Increment to Global Averaging"
	//   by Zhu et al (Arxiv 2017). https://arxiv.org/abs/1702.08601
	void AddRelativeRotationConstraint(const image_pair_t& view_id_pair,
	                                   const Eigen::Vector3d& relative_rotation,
									   const double confidence);

	// Given the relative rotation constraints added with
	// AddRelativeRotationConstraint, this method returns the robust estimation of
	// global camera orientations. Like the method above, this requires an initial
	// estimate of the global orientations.
	bool EstimateRotations(
			std::unordered_map<image_t , Eigen::Vector3d>* global_orientations);

protected:
	// Sets up the sparse linear system such that dR_ij = dR_j - dR_i. This is the
	// first-order approximation of the angle-axis rotations. This should only be
	// called once.
	void SetupLinearSystem();

	// Computes the relative rotation error based on the current global
	// orientation estimates.
	void ComputeRotationError();

	// Performs the L1 robust loss minimization.
	bool SolveL1Regression();

	// Performs the iteratively reweighted least squares.
	bool SolveIRLS();

	// Updates the global rotations based on the current rotation change.
	void UpdateGlobalRotations();



	// We keep one of the rotations as constant to remove the ambiguity of the
	// linear system.
	static const int kConstantRotationIndex = -1;

	// Options for the solver.
	Options options_;

	// The pairwise relative rotations used to compute the global rotations.
	std::vector<std::pair<image_pair_t, Eigen::Vector3d> > relative_rotations_;

	// The confidence of pairwise relatives.
	std::unordered_map<image_pair_t, double> relative_rotation_confs_; 

	// The global orientation estimates for each camera.
	std::unordered_map<image_t, Eigen::Vector3d>* global_orientations_{};

	// The sparse matrix used to maintain the linear system. This is matrix A in
	// Ax = b.
	Eigen::SparseMatrix<double> sparse_matrix_;

	// Map of ViewIds to the corresponding positions of the view's orientation in
	// the linear system.
	std::unordered_map<camera_t, int> view_id_to_index_;

	// x in the linear system Ax = b.
	Eigen::VectorXd rotation_change_;

	// b in the linear system Ax = b.
	Eigen::VectorXd relative_rotation_error_;
};

}

#endif //SENSEMAP_ESTIMATORS_GLOBAL_ESTIMATION_ROBUST_ROTATION_H_
