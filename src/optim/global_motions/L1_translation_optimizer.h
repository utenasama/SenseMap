// Copyright (c) 2019, SenseTime Group.
// All rights reserved.
#ifndef SENSEMAP_OPTIM_L1_TRANSLATION_OPTIMIZER_H_
#define SENSEMAP_OPTIM_L1_TRANSLATION_OPTIMIZER_H_

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <unordered_map>
#include "util/types.h"

namespace sensemap {

// Optimize the global translations given relative translations and initial
// values. The algorithm simply performs L1 minimization of the linear
// system Ax=b, composed of T_j-T_i=T_ij

class L1TranslationOptimizer {
public:
    struct Options {
        int max_num_l1_iterations = 5;
    };

    L1TranslationOptimizer(const Options& options) : options_(options) {}

    void AddRelativeScaleConstraint(const ViewIdPair& view_id_pair, const Eigen::Vector3d& relative_translation);

    bool OptimizeTranslations(const EIGEN_STL_MAP(ViewIdPair, Eigen::Vector3d) & relative_translations,
                              const view_t constant_view_id,
                              EIGEN_STL_UMAP(view_t, Eigen::Vector3d) * global_translations);

private:
    // Esimate the global translations, given the relative translations has been
    // added with AddRelativeScaleConstraint.
    bool OptimizeTranslations(EIGEN_STL_UMAP(view_t, Eigen::Vector3d) * global_translations);

    // Set up the sparse linear systems
    void SetupLinearSystem();

    // the id of the constant view
    view_t constant_view_id_;

    // perform the L1 minimization
    bool SolveL1Regression();

    void GetGlobalTranslations();

    Options options_;

    // The Matrix A in the linear system Ax=b
    Eigen::SparseMatrix<double> sparse_matrix_;

    // x in the linear system Ax=b
    Eigen::VectorXd estimated_translations_;

    // b in the linear system Ax=b
    Eigen::VectorXd relative_translation_errors_;

    // the pairwise relative scales
    std::vector<std::pair<ViewIdPair, Eigen::Vector3d>,
                Eigen::aligned_allocator<std::pair<ViewIdPair, Eigen::Vector3d> > >
        relative_translations_;

    // map from view id to the index of the corresponding param in the linear
    // system
    std::unordered_map<view_t, int> view_id_to_index_;

    // the gloal scales
    EIGEN_STL_UMAP(view_t, Eigen::Vector3d) * global_translations_;
};

}  // namespace sensemap

#endif  // SENSEMAP_OPTIM_L1_TRANSLATION_OPTIMIZER_H_