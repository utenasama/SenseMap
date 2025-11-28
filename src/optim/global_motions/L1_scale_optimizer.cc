//Copyright (c) 2019, SenseTime Group.
//All rights reserved.
#include "L1_scale_optimizer.h"
#include "util/logging.h"
#include "utils.h"
#include "optim/solver/l1_solver.h"

namespace sensemap{

bool L1ScaleOptimizer::OptimizeScales(
                const std::map<ViewIdPair,double>& relative_scales,
                const view_t constant_view_id,
                std::unordered_map<view_t, double>* global_scales){
    
    constant_view_id_=constant_view_id;
    for(const auto& relative_scale: relative_scales){
        AddRelativeScaleConstraint(relative_scale.first, relative_scale.second);
    }                
    return OptimizeScales(global_scales);
}


void L1ScaleOptimizer::AddRelativeScaleConstraint(
                                    const ViewIdPair& view_id_pair,
                                    double relative_scale){

    relative_scales_.emplace_back(view_id_pair,relative_scale);
}


bool L1ScaleOptimizer::OptimizeScales(
                            std::unordered_map<view_t,double>* global_scales){
    CHECK_GT(relative_scales_.size(),0);
    CHECK_NOTNULL(global_scales);
    global_scales_=global_scales;

    //map the view id to index of the params in in the linear equation system
    int index = -1;
    view_id_to_index_.reserve(global_scales->size());
    CHECK(global_scales->find(constant_view_id_) != global_scales->end());
    CHECK(global_scales->at(constant_view_id_) == 1.0);
    view_id_to_index_[constant_view_id_] = index++;

    for(const auto& scale: *global_scales){
        if(scale.first == constant_view_id_){
            continue;
        }
        view_id_to_index_[scale.first] = index;
        index++;
    }

    SetupLinearSystem();
    
    if(!SolveL1Regression()){
        LOG(ERROR) << "Could not solve the L1 regression step";
        return false;
    }
    
    GetGlobalScales();
    return true;
}

void L1ScaleOptimizer::SetupLinearSystem(){

    //the scale of the first view is set to one and kept constant
    estimated_scales_.resize(global_scales_->size()-1);
    relative_scale_errors_.resize(relative_scales_.size());
    sparse_matrix_.resize(relative_scales_.size(), global_scales_->size()-1);

    //set up A
    int scale_error_index = 0;
    std::vector<Eigen::Triplet<double> > triplet_list;
    for(const auto& relative_scale: relative_scales_) {
        const int view1_index=globalmotion::FindOrDie(view_id_to_index_, 
                                        relative_scale.first.first);
       
        if(view1_index !=-1){
            triplet_list.emplace_back(scale_error_index, view1_index, -1.0);
        }

        const int view2_index=globalmotion::FindOrDie(view_id_to_index_, 
                                        relative_scale.first.second);
        if(view2_index !=-1){
            triplet_list.emplace_back(scale_error_index, view2_index, 1.0);
        }    

        ++ scale_error_index;
    }
    sparse_matrix_.setFromTriplets(triplet_list.begin(), triplet_list.end());

    std::cout<<"[L1ScaleOptimizer::SetupLinearSystem] Matrix A: "<<std::endl;
    std::cout<<sparse_matrix_<<std::endl;

    //set up initial values of x
    for(const auto& global_scale: *global_scales_){
        const int view_index= 
                globalmotion::FindOrDie(view_id_to_index_,global_scale.first);        
        if(view_index != -1){
            estimated_scales_(view_index)=log(global_scale.second);
        }
    }

    std::cout<<"[L1ScaleOptimizer::SetupLinearSystem]: initial X: "<<std::endl;
    std::cout<<estimated_scales_<<std::endl;

    Eigen::VectorXd Ax=sparse_matrix_*estimated_scales_;
    std::cout<<"[L1ScaleOptimizer::SetupLinearSystem]  AX: "<<std::endl;
    std::cout<<Ax<<std::endl;

    
    //set up b
    ComputeScaleError();
    std::cout<<"[L1ScaleOptimizer::SetupLinearSystem]: initial b:"<<std::endl;
    std::cout<<relative_scale_errors_<<std::endl;

}

void L1ScaleOptimizer::ComputeScaleError(){
    
    int scale_error_index=0;
    for(const auto relative_scale: relative_scales_){
        relative_scale_errors_(scale_error_index) = log(relative_scale.second);    
        scale_error_index++;
    }
}


bool L1ScaleOptimizer::SolveL1Regression(){

    L1Solver<Eigen::SparseMatrix<double> >::Options options;
    options.max_num_iterations = 5;
    L1Solver<Eigen::SparseMatrix<double> > l1_solver(options, sparse_matrix_);

    l1_solver.Solve(relative_scale_errors_, &estimated_scales_);

    return true;
}

void L1ScaleOptimizer::GetGlobalScales(){
    for(auto &scale: *global_scales_){
        const int view_index = 
                globalmotion::FindOrDie(view_id_to_index_, scale.first);
        if(view_index == -1){
            scale.second=1.0;
        }
        else{
            scale.second=exp(estimated_scales_(view_index));
        }
    }
}



}//namespace sensemap

