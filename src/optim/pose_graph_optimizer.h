#include <utility>

//Copyright (c) 2019, SenseTime Group.
//All rights reserved.

#ifndef SENSEMAP_OPTIM_POSE_GRAPH_OPTIMIZER_H_
#define SENSEMAP_OPTIM_POSE_GRAPH_OPTIMIZER_H_

#include <unordered_set>
#include <ceres/ceres.h>

#include "util/types.h"
#include "util/ceres_types.h"
#include "base/reconstruction.h"
#include "base/reconstruction_manager.h"

namespace sensemap {

class PoseGraphOptimizer {

public:
    enum OPTIMIZATION_METHOD{
        SE3 = 0,
        SIM3 = 1
    };

    struct Options {
        // Define optimization method
        OPTIMIZATION_METHOD optimization_method = OPTIMIZATION_METHOD::SE3;

        // Define loss_function
        bool lossfunction_enable = false;

        // Define max number of iteration
        size_t max_num_iterations = 200;

        // Display the log output
        bool minimizer_progress_to_stdout = true;

    };
    struct Vertex{
        Eigen::Quaterniond qvec;
        Eigen::Vector3d tvec;
        double scale;
    };

    struct Edge{
        image_t id_begin;
        image_t id_end;
        size_t label;
        size_t num_corrs;
        Vertex relative_pose;
    };

public:
    PoseGraphOptimizer(PoseGraphOptimizer::Options options, Reconstruction* reconstruction)
            :options_(options), reconstruction_(reconstruction) {
        loss_function_ = options_.lossfunction_enable ? new ceres::HuberLoss(1.0) : nullptr;

        local_parameterization_ = new ceres::QuaternionParameterization;
    }

    PoseGraphOptimizer(PoseGraphOptimizer::Options options, std::shared_ptr<ReconstructionManager> reconstruction_manager)
            :options_(options), reconstruction_manager_(reconstruction_manager) {
        loss_function_ = options_.lossfunction_enable ? new ceres::HuberLoss(1.0) : nullptr;

        local_parameterization_ = new ceres::QuaternionParameterization;
    }

    void AddConstraint(const image_t image_id1,
                       const image_t image_id2,
                       const Eigen::Quaterniond& relative_qvec,
                       const Eigen::Vector3d& relative_tvec,
                       double weight = 1.0);

    void AddConstraint(const image_t image_id1,
                       const image_t image_id2,
                       Eigen::Vector7d& sim3_pose, 
                       double weight = 1.0);

    void SetParameterConstant(const image_t image_id);


    void AddConstraint(const std::pair<cluster_t, image_t> image_id1,
                       const std::pair<cluster_t, image_t> image_id2,
                       const Eigen::Quaterniond& relative_qvec,
                       const Eigen::Vector3d& relative_tvec);

    void AddConstraint(const std::pair<cluster_t, image_t> image_id1,
                       const std::pair<cluster_t, image_t> image_id2,
                       Eigen::Vector7d& sim3_pose,
                       double weight = 1.0);
    
    void SetParameterConstant(const std::pair<cluster_t, image_t> image_id);
    void SetSIM3PoseParameterConstant(const std::pair<cluster_t, image_t> image_id);
    void SetSIM3ScaleParameterConstant(const std::pair<cluster_t, image_t> image_id);

    void Solve();

private:
    inline void SetParameterization(ceres::Problem *problem, double* values, ceres::LocalParameterization *local_parameterization);

private:
    ceres::Problem problem_;
    ceres::LossFunction* loss_function_;

    Options options_;
    std::shared_ptr<ReconstructionManager> reconstruction_manager_;
    Reconstruction* reconstruction_;
    ceres::LocalParameterization* local_parameterization_;
    std::unordered_set<image_t> image_id_params_;

};

typedef PoseGraphOptimizer::Vertex Vertex;
typedef PoseGraphOptimizer::Edge Edge;

} //namespace sensemap
#endif //SENSEMAP_OPTIMIZER_POSE_GRAPH_OPTIMIZER_H_
