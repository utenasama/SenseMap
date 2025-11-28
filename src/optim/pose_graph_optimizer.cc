//Copyright (c) 2019, SenseTime Group.
//All rights reserved.

#include "base/cost_functions.h"

#include "pose_graph_optimizer.h"

#include "optim/global_motions/utils.h"

namespace sensemap {

void PoseGraphOptimizer::AddConstraint(const image_t image_id1,
                                       const image_t image_id2,
                                       const Eigen::Quaterniond& relative_qvec,
                                       const Eigen::Vector3d& relative_tvec,double weight) {

    //Check the optimization method
    CHECK(options_.optimization_method == OPTIMIZATION_METHOD::SE3);

    class Image& image1 = reconstruction_->Image(image_id1);
    class Image& image2 = reconstruction_->Image(image_id2);

    ceres::CostFunction* cost_function =
            PoseGraphSE3ErrorTerm::Create(relative_qvec, relative_tvec, weight);

    problem_.AddResidualBlock(cost_function, loss_function_,
                              image1.Tvec().data(), image1.Qvec().data(),
                              image2.Tvec().data(), image2.Qvec().data());
    if (image_id_params_.count(image_id1) == 0) {
        image_id_params_.insert(image_id1);
        SetParameterization(&problem_, image1.Qvec().data(), local_parameterization_);
    }
    if (image_id_params_.count(image_id2) == 0) {
        image_id_params_.insert(image_id2);
        SetParameterization(&problem_, image2.Qvec().data(), local_parameterization_);
    }
}


void PoseGraphOptimizer::AddConstraint(const image_t image_id1,
                                       const image_t image_id2,
                                       Eigen::Vector7d& sim3_pose, double weight) {

    // Check the optimization method
    CHECK(options_.optimization_method == OPTIMIZATION_METHOD::SIM3);

    class Image& image1 = reconstruction_->Image(image_id1);
    class Image& image2 = reconstruction_->Image(image_id2);

    ceres::CostFunction* cost_function = PoseGraphSIM3ErrorTerm::Create(sim3_pose, weight);


    problem_.AddResidualBlock(cost_function, loss_function_, image1.Sim3pose().data(), image2.Sim3pose().data());

}


void PoseGraphOptimizer::SetParameterConstant(const image_t image_id) {
    class Image& image = reconstruction_->Image(image_id);

    //Check the optimization method
    if(options_.optimization_method == OPTIMIZATION_METHOD::SE3){
        problem_.SetParameterBlockConstant(image.Qvec().data());
        problem_.SetParameterBlockConstant(image.Tvec().data());
    }else{
        problem_.SetParameterBlockConstant(image.Sim3pose().data());
    }

}

void PoseGraphOptimizer::AddConstraint(const std::pair<cluster_t, image_t> image_id1,
                                       const std::pair<cluster_t, image_t> image_id2,
                                       const Eigen::Quaterniond& relative_qvec,
                                       const Eigen::Vector3d& relative_tvec) {

    //Check the optimization method
    CHECK(options_.optimization_method == OPTIMIZATION_METHOD::SE3);

    class Image& image1 = reconstruction_manager_->Get(image_id1.first)->Image(image_id1.second);
    class Image& image2 = reconstruction_manager_->Get(image_id2.first)->Image(image_id2.second);

    ceres::CostFunction* cost_function =
        PoseGraphSE3ErrorTerm::Create(relative_qvec, relative_tvec);

    problem_.AddResidualBlock(cost_function, loss_function_,
                             image1.Tvec().data(), image1.Qvec().data(),
                             image2.Tvec().data(), image2.Qvec().data());
    if (image_id_params_.count(image_id1.second) == 0) {
        image_id_params_.insert(image_id1.second);
        SetParameterization(&problem_, image1.Qvec().data(), local_parameterization_);
    }
    if (image_id_params_.count(image_id2.second) == 0) {
        image_id_params_.insert(image_id2.second);
        SetParameterization(&problem_, image2.Qvec().data(), local_parameterization_);
    }
}


void PoseGraphOptimizer::AddConstraint(const std::pair<cluster_t, image_t> image_id1,
                                       const std::pair<cluster_t, image_t> image_id2,
                                       Eigen::Vector7d& sim3_pose, double weight) {

    // Check the optimization method
    CHECK(options_.optimization_method == OPTIMIZATION_METHOD::SIM3);

    class Image& image1 = reconstruction_manager_->Get(image_id1.first)->Image(image_id1.second);
    class Image& image2 = reconstruction_manager_->Get(image_id2.first)->Image(image_id2.second);

    if(image1.Sim3pose() == image2.Sim3pose()){
        std::cout << "Error we get two same image" << std::endl; 
        return;
    }

    ceres::CostFunction* cost_function = PoseGraphSIM3ErrorTerm::Create(sim3_pose);

    if (weight > 1){
        ceres::ScaledLoss *cluster_loss_function = 
            new ceres::ScaledLoss(new ceres::TrivialLoss(), 
            static_cast<double>(weight), ceres::DO_NOT_TAKE_OWNERSHIP);
        problem_.AddResidualBlock(cost_function, cluster_loss_function, image1.Sim3pose().data(), image2.Sim3pose().data());
    } else {
        problem_.AddResidualBlock(cost_function, loss_function_, image1.Sim3pose().data(), image2.Sim3pose().data());
    }
}


void PoseGraphOptimizer::SetParameterConstant(const std::pair<cluster_t, image_t> image_id) {
    CHECK(reconstruction_manager_->Get(image_id.first)->ExistsImage(image_id.second))<<"reconstruction "<<image_id.first<<" not have "<<image_id.second<<std::endl;
    class Image& image = reconstruction_manager_->Get(image_id.first)->Image(image_id.second);

     //Check the optimization method
    if(options_.optimization_method == OPTIMIZATION_METHOD::SE3){
        problem_.SetParameterBlockConstant(image.Qvec().data());
        problem_.SetParameterBlockConstant(image.Tvec().data());
    }else{
        problem_.SetParameterBlockConstant(image.Sim3pose().data());
    }
}

void PoseGraphOptimizer::SetSIM3PoseParameterConstant(const std::pair<cluster_t, image_t> image_id) {
    class Image& image = reconstruction_manager_->Get(image_id.first)->Image(image_id.second);

      ceres::SubsetParameterization *sim3_pose_parameterization =
            new ceres::SubsetParameterization(7,{0,1,2,3,4,5});
    SetParameterization(&problem_, image.Sim3pose().data(), sim3_pose_parameterization);
}

void PoseGraphOptimizer::SetSIM3ScaleParameterConstant(const std::pair<cluster_t, image_t> image_id) {
    class Image& image = reconstruction_manager_->Get(image_id.first)->Image(image_id.second);

    ceres::SubsetParameterization *sim3_scale_parameterization =
            new ceres::SubsetParameterization(7,{6});
    SetParameterization(&problem_, image.Sim3pose().data(), sim3_scale_parameterization);
}

void PoseGraphOptimizer::Solve() {
    ceres::Solver::Options options;
    options.max_num_iterations = options_.max_num_iterations;
    options.linear_solver_type = ceres::SPARSE_SCHUR;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    options.minimizer_progress_to_stdout = options_.minimizer_progress_to_stdout;

    std::cout << "PoseGraph::Solve" << std::endl;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem_, &summary);

    std::cout << summary.FullReport() << std::endl;
}

void PoseGraphOptimizer::SetParameterization(ceres::Problem *problem, double* values, ceres::LocalParameterization *local_parameterization) {
#if CERES_VERSION_MAJOR >= 2 && CERES_VERSION_MINOR >= 1
    problem->SetManifold(values, local_parameterization);
#else
    problem->SetParameterization(values, local_parameterization);
#endif
}

} // namespace sensemap
