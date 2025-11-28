//Copyright (c) 2019, SenseTime Group.
//All rights reserved.
#include "L1_translation_optimizer.h"
#include "util/logging.h"
#include "utils.h"
#include "optim/solver/l1_solver.h"

namespace sensemap{

bool L1TranslationOptimizer::OptimizeTranslations(
    const EIGEN_STL_MAP(ViewIdPair,Eigen::Vector3d)& relative_translations,
    const view_t constant_view_id,
    EIGEN_STL_UMAP(view_t, Eigen::Vector3d)* global_translations){

    constant_view_id_ = constant_view_id;
    for(const auto& relative_trans: relative_translations){
        AddRelativeScaleConstraint(relative_trans.first, relative_trans.second);
    }                
    return OptimizeTranslations(global_translations);
}


void L1TranslationOptimizer::AddRelativeScaleConstraint(
                                const ViewIdPair& view_id_pair,
                                const Eigen::Vector3d& relative_translation){

    relative_translations_.emplace_back(view_id_pair,relative_translation);
}


bool L1TranslationOptimizer::OptimizeTranslations(
            EIGEN_STL_UMAP(view_t,Eigen::Vector3d)* global_translations){
    CHECK_GT(relative_translations_.size(),0);
    CHECK_NOTNULL(global_translations);
    global_translations_=global_translations;

    //map the view id to index of the params in in the linear equation system
    int index = -1;
    view_id_to_index_.reserve(global_translations->size());
    CHECK(global_translations->find(constant_view_id_) != global_translations->end());
    CHECK(global_translations->at(constant_view_id_) == Eigen::Vector3d::Zero())
          <<global_translations->at(constant_view_id_);

    view_id_to_index_[constant_view_id_] = index++;

    for(const auto& translation: *global_translations){
        if(translation.first == constant_view_id_){
            continue;
        }        
        view_id_to_index_[translation.first] = index;
        index++;
    }

    SetupLinearSystem();
    
    if(!SolveL1Regression()){
        LOG(ERROR) << "Could not solve the L1 regression step";
        return false;
    }
    
    GetGlobalTranslations();
    return true;
}

void L1TranslationOptimizer::SetupLinearSystem(){

    //the scale of the first view is set to one and kept constant
    estimated_translations_.resize((global_translations_->size()-1)*3);
    relative_translation_errors_.resize(relative_translations_.size()*3);
    sparse_matrix_.resize(relative_translations_.size()*3, 
                          (global_translations_->size()-1)*3);

    //set up A
    int trans_error_index = 0;
    std::vector<Eigen::Triplet<double> > triplet_list;
    for(const auto& relative_trans: relative_translations_) {
        const int view1_index=globalmotion::FindOrDie(view_id_to_index_, 
                                        relative_trans.first.first);
       
        if(view1_index !=-1){
            triplet_list.emplace_back(trans_error_index*3+0, 
                                      view1_index*3+0, 
                                      -1.0);
            triplet_list.emplace_back(trans_error_index*3+1, 
                                      view1_index*3+1, 
                                      -1.0);
            triplet_list.emplace_back(trans_error_index*3+2, 
                                      view1_index*3+2, 
                                      -1.0); 
        }

        const int view2_index=globalmotion::FindOrDie(view_id_to_index_, 
                                        relative_trans.first.second);
        if(view2_index !=-1){
            triplet_list.emplace_back(trans_error_index*3+0, 
                                      view2_index*3+0, 
                                      1.0);
            triplet_list.emplace_back(trans_error_index*3+1, 
                                      view2_index*3+1, 
                                      1.0);
            triplet_list.emplace_back(trans_error_index*3+2, 
                                      view2_index*3+2, 
                                      1.0);
        }    
        ++ trans_error_index;
    }
    sparse_matrix_.setFromTriplets(triplet_list.begin(), triplet_list.end());

    //set up initial values of x
    for(const auto& global_trans: *global_translations_){
        const int view_index= 
                globalmotion::FindOrDie(view_id_to_index_,global_trans.first);        
        if(view_index != -1){
            estimated_translations_.segment<3>(view_index*3)=
                                                        global_trans.second;
        }
    }

    
    Eigen::VectorXd Ax=sparse_matrix_*estimated_translations_;
    std::cout<<"[L1TranslationOptimizer::SetupLinearSystem] Initial Ax: "
             <<std::endl;
    std::cout<<Ax<<std::endl;
    
    //set up b
    trans_error_index=0;
    for(const auto relative_trans: relative_translations_){
        relative_translation_errors_.segment<3>(trans_error_index * 3) =
                                        relative_trans.second;
        trans_error_index++;
    }

    std::cout<<"[L1TranslationOptimizer::SetupLinearSystem] b: "
             <<std::endl;
    std::cout<<relative_translation_errors_<<std::endl;


    Eigen::VectorXd errors=Ax-relative_translation_errors_;
    std::cout<<"[L1TranslationOptimizer::SetupLinearSystem] errors: "
             <<std::endl;
    std::cout<<errors<<std::endl;
    std::cout<<"[L1TranslationOptimizer::SetupLinearSystem] errors norm: "
             <<errors.norm()<<std::endl;


}

bool L1TranslationOptimizer::SolveL1Regression(){
    
    L1Solver<Eigen::SparseMatrix<double> >::Options options;
    options.max_num_iterations = 5;
    L1Solver<Eigen::SparseMatrix<double> > l1_solver(options, sparse_matrix_);

    l1_solver.Solve(relative_translation_errors_, &estimated_translations_);

    return true;


}

void L1TranslationOptimizer::GetGlobalTranslations(){
    for(auto &trans: *global_translations_){
        const int view_index = 
                globalmotion::FindOrDie(view_id_to_index_, trans.first);
        if(view_index == -1){
            trans.second=Eigen::Vector3d::Zero();
        }
        else{
            trans.second=estimated_translations_.segment<3>(view_index*3);
        }
    }
}



}//namespace sensemap

