// Copyright (c) 2019, SenseTime Group.
// All rights reserved.
#ifndef SENSEMAP_OPTIM_ROBUST_ROTATION_OPTIMIZER_H_
#define SENSEMAP_OPTIM_ROBUST_ROTATION_OPTIMIZER_H_

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <unordered_map>
#include "util/types.h"

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
class RobustRotationOptimizer {
public:
    struct Options {
        // Maximum number of times to run L1 minimization. L1 is very slow
        // (compared to L2), but is very robust to outliers. Typically only a
        // few iterations are needed in order for the solution to reside within
        // the cone of convergence for L2 solving.
        int max_num_l1_iterations = 5;

        // The number of iterative reweighted least squares iterations to
        // perform.
        int max_num_irls_iterations = 100;
    };

    RobustRotationOptimizer(const Options& options) : options_(options) {}

    // Estimates the global orientations of all views based on an initial
    // guess. Returns true on successful estimation and false otherwise.
    bool OptimizeRotations(const EIGEN_STL_MAP(ViewIdPair, Eigen::Vector3d) & relative_rotations,
                           const view_t constant_view_id, EIGEN_STL_UMAP(view_t, Eigen::Vector3d) * global_rotations);

    // An alternative interface is to instead add relative rotation constraints
    // one by one with AddRelativeRotationConstraint, then call the
    // EstimateRotations interface below. This allows the caller to add multiple
    // constraints for the same view id pair, which may lead to more accurate
    // rotation estimates. Please see the following reference for an example of
    // how to obtain multiple constraints for pairs of views:
    //
    // "Parallel Structure from Motion from Local Increment to Global Averaging"
    //   by Zhu et al (Arxiv 2017). https://arxiv.org/abs/1702.08601
    void AddRelativeRotationConstraint(const ViewIdPair& view_id_pair, const Eigen::Vector3d& relative_rotation);

    // Given the relative rotation constraints added with
    // AddRelativeRotationConstraint, this method returns the robust estimation
    // of global camera orientations. Like the method above, this requires an
    // initial estimate of the global orientations.
    bool OptimizeRotations(EIGEN_STL_UMAP(view_t, Eigen::Vector3d) * global_rotations);

protected:
    // Sets up the sparse linear system such that dR_ij = dR_j - dR_i. This is
    // the first-order approximation of the angle-axis rotations. This should
    // only be called once.
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

    // the id of the constant view
    view_t constant_view_id_;

    // Options for the solver.
    const Options options_;

    // The pairwise relative rotations used to compute the global rotations.
    std::vector<std::pair<ViewIdPair, Eigen::Vector3d>,
                Eigen::aligned_allocator<std::pair<ViewIdPair, Eigen::Vector3d> > >
        relative_rotations_;

    // The global orientation estimates for each camera.
    EIGEN_STL_UMAP(view_t, Eigen::Vector3d) * global_rotations_;

    // The sparse matrix used to maintain the linear system. This is matrix A in
    // Ax = b.
    Eigen::SparseMatrix<double> sparse_matrix_;

    // Map of ViewIds to the corresponding positions of the view's orientation
    // in the linear system.
    std::unordered_map<view_t, int> view_id_to_index_;

    // x in the linear system Ax = b.
    Eigen::VectorXd rotation_change_;

    // b in the linear system Ax = b.
    Eigen::VectorXd relative_rotation_error_;
};

}  // namespace sensemap

#endif  // SENSEMAP_OPTIM_L1_SCALE_OPTIMIZER_H_