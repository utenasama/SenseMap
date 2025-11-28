//Copyright (c) 2019, SenseTime Group.
//All rights reserved.
#ifndef SENSEMAP_OPTIM_L1_SCALE_OPTIMIZER_H_
#define SENSEMAP_OPTIM_L1_SCALE_OPTIMIZER_H_

#include "util/types.h"
#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <unordered_map>

namespace sensemap{

//Optimize the global scales given relative scales and initial values. 
//The algorithm simply performs L1 minimization of the linear
//system Ax=b, composed of log(s_j)-log(s_i)=log(s_ij)
 
class L1ScaleOptimizer{
public:
    
    struct Options{
        int max_num_l1_iterations = 5;
    };
    
    L1ScaleOptimizer(const Options& options) : options_(options){}

    void AddRelativeScaleConstraint(const ViewIdPair& view_id_pair,
                                    double relative_scale);

    bool OptimizeScales(
                const std::map<ViewIdPair,double>& relative_scales,
                const view_t constant_view_id,
                std::unordered_map<view_t, double>* global_scales);

private:
    
    //Esimate the global scales, given the relative scales has been added with
    //AddRelativeScaleConstraint.
    bool OptimizeScales(std::unordered_map<view_t,double>* global_scales);
    
    //Set up the sparse linear systems
    void SetupLinearSystem();
    
    //compute the scale error, i.e. log(s_ij)
    void ComputeScaleError();

    //perform the robust L1 minimization
    bool SolveL1Regression();

    void GetGlobalScales();

    Options options_;
    
    // the id of the constant view
    view_t constant_view_id_;

    //The Matrix A in the linear system Ax=b
    Eigen::SparseMatrix<double> sparse_matrix_;

    //x in the linear system Ax=b
    Eigen::VectorXd estimated_scales_;

    //b in the linear system Ax=b
    Eigen::VectorXd relative_scale_errors_;

    //the pairwise relative scales
    std::vector<std::pair<ViewIdPair,double> > relative_scales_;

    //map from view id to the index of the corresponding param in the linear 
    //system
    std::unordered_map<view_t, int> view_id_to_index_;

    //the gloal scales
    std::unordered_map<view_t,double>* global_scales_;

};
} //namespace sensemap
#endif //SENSEMAP_OPTIM_L1_SCALE_OPTIMIZER_H_