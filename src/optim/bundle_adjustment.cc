// Copyright (c) 2019, SenseTime Group.
// All rights reserved.

#include "optim/bundle_adjustment.h"

#include <iomanip>

#ifdef OPENMP_ENABLED
#include <omp.h>
#endif

#include <iostream>

#include "base/camera_models.h"
#include "base/cost_functions.h"
#include "base/pose.h"
#include "base/projection.h"
#include "util/misc.h"
#include "util/threading.h"
#include "util/timer.h"
#include "util/kmeans.h"
#include <memory>
#ifdef USE_OPENBLAS
#include "openblas/cblas.h"
#endif

namespace sensemap {
namespace {
float GetDepthPointAndWeight(const Camera& camera, const Point2D& point2D, 
                             Eigen::Vector3d& point3D) {
    const float depth = point2D.Depth();
    if (depth > 0) {
        Eigen::Vector2d point2D_normalized = camera.ImageToWorld(point2D.XY());
        point3D[0] = point2D_normalized(0) * depth;
        point3D[1] = point2D_normalized(1) * depth; 
        point3D[2] = depth;
        return point2D.DepthWeight();
    } else {
        point3D = Eigen::Vector3d::Zero();
        return 0.0;
    }
}
}

////////////////////////////////////////////////////////////////////////////////
// BundleAdjustmentOptions
////////////////////////////////////////////////////////////////////////////////

ceres::LossFunction *BundleAdjustmentOptions::CreateLossFunction() const {
    ceres::LossFunction *loss_function = nullptr;
    switch (loss_function_type) {
        case LossFunctionType::TRIVIAL:
            loss_function = new ceres::TrivialLoss();
            break;
        case LossFunctionType::SOFT_L1:
            loss_function = new ceres::SoftLOneLoss(loss_function_scale);
            break;
        case LossFunctionType::Huber:
            loss_function = new ceres::HuberLoss(loss_function_scale);
            break;
        case LossFunctionType::CAUCHY:
            loss_function = new ceres::CauchyLoss(loss_function_scale);
            break;
    }
    CHECK_NOTNULL(loss_function);
    return loss_function;
}

bool BundleAdjustmentOptions::Check() const {
    CHECK_OPTION_GE(loss_function_scale, 0);
    return true;
}

////////////////////////////////////////////////////////////////////////////////
// BundleAdjustmentConfig
////////////////////////////////////////////////////////////////////////////////

BundleAdjustmentConfig::BundleAdjustmentConfig() {}

size_t BundleAdjustmentConfig::NumImages() const { return image_ids_.size(); }

size_t BundleAdjustmentConfig::NumPoints() const {
    return variable_mappoint_ids_.size() + constant_mappoint_ids_.size();
}

size_t BundleAdjustmentConfig::NumConstantCameras() const { return constant_camera_ids_.size(); }

size_t BundleAdjustmentConfig::NumConstantPoses() const { return constant_poses_.size(); }

size_t BundleAdjustmentConfig::NumConstantTvecs() const { return constant_tvecs_.size(); }

size_t BundleAdjustmentConfig::NumVariablePoints() const { return variable_mappoint_ids_.size(); }

size_t BundleAdjustmentConfig::NumConstantPoints() const { return constant_mappoint_ids_.size(); }

size_t BundleAdjustmentConfig::NumConstantGNSS() const { return constant_gnss_ids_.size(); }

size_t BundleAdjustmentConfig::NumSweeps() const { return sweep_ids_.size(); }

size_t BundleAdjustmentConfig::NumResiduals(const Reconstruction &reconstruction) const {
    // Count the number of observations for all added images.
    size_t num_observations = 0;
    for (const image_t image_id : image_ids_) {
        num_observations += reconstruction.Image(image_id).NumMapPoints();
    }

    // Count the number of observations for all added Map Points that are not
    // already added as part of the images above.

    auto NumObservationsForPoint = [this, &reconstruction](const mappoint_t mappoint_id) {
        size_t num_observations_for_point = 0;
        const auto &mappoint = reconstruction.MapPoint(mappoint_id);
        for (const auto &track_el : mappoint.Track().Elements()) {
            if (image_ids_.count(track_el.image_id) == 0) {
                num_observations_for_point += 1;
            }
        }
        return num_observations_for_point;
    };

    for (const auto mappoint_id : variable_mappoint_ids_) {
        num_observations += NumObservationsForPoint(mappoint_id);
    }
    for (const auto mappoint_id : constant_mappoint_ids_) {
        num_observations += NumObservationsForPoint(mappoint_id);
    }

    return 2 * num_observations;
}

void BundleAdjustmentConfig::AddImage(const image_t image_id) { image_ids_.insert(image_id); }

bool BundleAdjustmentConfig::HasImage(const image_t image_id) const {
    return image_ids_.find(image_id) != image_ids_.end();
}

void BundleAdjustmentConfig::RemoveImage(const image_t image_id) { image_ids_.erase(image_id); }

void BundleAdjustmentConfig::AddSweep(const sweep_t sweep_id) { sweep_ids_.insert(sweep_id); }

void BundleAdjustmentConfig::AddSweepImagePair(const sweep_t sweep_id, const image_t image_id) {
    sweep_image_pairs_[image_id] = sweep_id;
}

void BundleAdjustmentConfig::SetLidar2CamMatrix(const Eigen::Matrix3x4d lidar_2_cam) {
    lidar_to_cam_matrix_ = lidar_2_cam;
}

bool BundleAdjustmentConfig::HasSweep(const sweep_t sweep_id) const{
    return sweep_ids_.find(sweep_id) != sweep_ids_.end();
}
void BundleAdjustmentConfig::RemoveSweep(const sweep_t sweep_id) { sweep_ids_.erase(sweep_id); }

void BundleAdjustmentConfig::SetConstantCamera(const camera_t camera_id) { constant_camera_ids_.insert(camera_id); }

void BundleAdjustmentConfig::SetVariableCamera(const camera_t camera_id) { constant_camera_ids_.erase(camera_id); }

bool BundleAdjustmentConfig::IsConstantCamera(const camera_t camera_id) const {
    return constant_camera_ids_.find(camera_id) != constant_camera_ids_.end();
}

void BundleAdjustmentConfig::SetConstantPose(const image_t image_id) {
    CHECK(HasImage(image_id));
    CHECK(!HasConstantTvec(image_id));
    constant_poses_.insert(image_id);
}

void BundleAdjustmentConfig::SetVariablePose(const image_t image_id) { constant_poses_.erase(image_id); }

bool BundleAdjustmentConfig::HasConstantPose(const image_t image_id) const {
    return constant_poses_.find(image_id) != constant_poses_.end();
}

void BundleAdjustmentConfig::SetConstantSweep(const sweep_t sweep_id) {
    CHECK(HasSweep(sweep_id));
    const_sweeps_.insert(sweep_id);
}

void BundleAdjustmentConfig::SetVariableSweep(const sweep_t sweep_id) { const_sweeps_.erase(sweep_id); }

bool BundleAdjustmentConfig::HasConstantSweep(const sweep_t sweep_id) const {
    return const_sweeps_.find(sweep_id) != const_sweeps_.end();
}

void BundleAdjustmentConfig::SetConstantTvec(const image_t image_id, const std::vector<int> &idxs) {
    CHECK_GT(idxs.size(), 0);
    CHECK_LE(idxs.size(), 3);
    CHECK(HasImage(image_id));
    CHECK(!HasConstantPose(image_id));
    CHECK(!VectorContainsDuplicateValues(idxs)) << "Tvec indices must not contain duplicates";
    constant_tvecs_.emplace(image_id, idxs);
}

void BundleAdjustmentConfig::RemoveConstantTvec(const image_t image_id) { constant_tvecs_.erase(image_id); }

bool BundleAdjustmentConfig::HasConstantTvec(const image_t image_id) const {
    return constant_tvecs_.find(image_id) != constant_tvecs_.end();
}

const std::unordered_set<image_t> &BundleAdjustmentConfig::Images() const { return image_ids_; }

const std::unordered_set<sweep_t> &BundleAdjustmentConfig::Sweeps() const { return sweep_ids_; }

const std::unordered_map<sweep_t, image_t> BundleAdjustmentConfig::SweepImagePairs() const { return sweep_image_pairs_; }

const std::unordered_set<mappoint_t> &BundleAdjustmentConfig::VariablePoints() const { return variable_mappoint_ids_; }

const std::unordered_set<mappoint_t> &BundleAdjustmentConfig::ConstantPoints() const { return constant_mappoint_ids_; }

const std::vector<int> &BundleAdjustmentConfig::ConstantTvec(const image_t image_id) const {
    return constant_tvecs_.at(image_id);
}

const std::unordered_set<camera_t>& BundleAdjustmentConfig::ConstantCameraIds() const {
    return constant_camera_ids_;
}

const std::unordered_set<image_t>& BundleAdjustmentConfig::ConstantPoses() const {
    return constant_poses_;
}

const std::unordered_map<image_t, std::vector<int>>& BundleAdjustmentConfig::ConstantTVecs() const {
    return constant_tvecs_;
}

void BundleAdjustmentConfig::AddVariablePoint(const mappoint_t mappoint_id) {
    CHECK(!HasConstantPoint(mappoint_id));
    variable_mappoint_ids_.insert(mappoint_id);
}

void BundleAdjustmentConfig::AddConstantPoint(const mappoint_t mappoint_id) {
    CHECK(!HasVariablePoint(mappoint_id));
    constant_mappoint_ids_.insert(mappoint_id);
}

bool BundleAdjustmentConfig::HasPoint(const mappoint_t mappoint_id) const {
    return HasVariablePoint(mappoint_id) || HasConstantPoint(mappoint_id);
}

bool BundleAdjustmentConfig::HasVariablePoint(const mappoint_t mappoint_id) const {
    return variable_mappoint_ids_.find(mappoint_id) != variable_mappoint_ids_.end();
}

bool BundleAdjustmentConfig::HasConstantPoint(const mappoint_t mappoint_id) const {
    return constant_mappoint_ids_.find(mappoint_id) != constant_mappoint_ids_.end();
}

void BundleAdjustmentConfig::RemoveVariablePoint(const mappoint_t mappoint_id) {
    variable_mappoint_ids_.erase(mappoint_id);
}

void BundleAdjustmentConfig::RemoveConstantPoint(const mappoint_t mappoint_id) {
    constant_mappoint_ids_.erase(mappoint_id);
}

const std::unordered_set<image_t> &BundleAdjustmentConfig::GNSSImages() const { return constant_gnss_ids_; }

void BundleAdjustmentConfig::AddGNSS(const image_t image_id) { constant_gnss_ids_.insert(image_id); }

bool BundleAdjustmentConfig::HasGNSS(const image_t image_id) const {
    return constant_gnss_ids_.find(image_id) != constant_gnss_ids_.end();
}

void BundleAdjustmentConfig::RemoveGNSS(const image_t image_id) { constant_gnss_ids_.erase(image_id); }

////////////////////////////////////////////////////////////////////////////////
// BundleAdjuster
////////////////////////////////////////////////////////////////////////////////

BundleAdjuster::BundleAdjuster(const BundleAdjustmentOptions &options, const BundleAdjustmentConfig &config)
    : options_(options), config_(config) {
    CHECK(options_.Check());
}

void BundleAdjuster::SetParameterization(ceres::Problem *problem, double* values, ceres::LocalParameterization *local_parameterization) {
#if CERES_VERSION_MAJOR >= 2 && CERES_VERSION_MINOR >= 1
    problem->SetManifold(values, local_parameterization);
#else
    problem->SetParameterization(values, local_parameterization);
#endif
}

void IterativeTrackSelection(const Reconstruction *reconstruction,
                             const std::vector<mappoint_t> &all_mappoint_ids,
                             const std::vector<mappoint_t> &const_mappoint_ids,
                             std::vector<mappoint_t> &selected_mappoint_ids,
                             const size_t max_cover_per_view) {
    std::unordered_set<mappoint_t> unique_mappoint_ids;
    unique_mappoint_ids.reserve(const_mappoint_ids.size());
    for (auto mappoint_id : const_mappoint_ids) {
        unique_mappoint_ids.insert(mappoint_id);
    }
    std::unordered_map<image_t, int> cover_view;
    for (size_t i = 0; i < all_mappoint_ids.size(); ++i) {
        auto mappoint_id = all_mappoint_ids.at(i);
        auto mappoint = reconstruction->MapPoint(mappoint_id);
        auto track = mappoint.Track();
        if (unique_mappoint_ids.find(mappoint_id) != unique_mappoint_ids.end()) {
            std::unordered_set<image_t> track_images;
            for (auto track_elem : track.Elements()) {
                track_images.insert(track_elem.image_id);
            }
            for (auto image_id : track_images) {
                cover_view[image_id]++;
            }
            continue;
        }
        bool selected = false;
        for (auto track_elem : track.Elements()) {
            image_t image_id = track_elem.image_id;
            std::unordered_map<image_t, int>::iterator it = cover_view.find(image_id);
            if (it == cover_view.end() || it->second < max_cover_per_view) {
                cover_view[image_id]++;
                selected = true;
                break;
            }
        }
        if (selected) {
            std::unordered_set<image_t> track_images;
            for (auto track_elem : track.Elements()) {
                track_images.insert(track_elem.image_id);
            }
            for (auto image_id : track_images) {
                cover_view[image_id]++;
            }
            selected_mappoint_ids.push_back(mappoint_id);
        }
    }
}

size_t BundleAdjuster::PrecomputeResiduals(Reconstruction* reconstruction) {
    size_t num_residuals = 0;
    for (const image_t image_id : config_.Images()) {
        class Image &image = reconstruction->Image(image_id);
        for (const Point2D &point2D : image.Points2D()) {
            if (point2D.HasMapPoint()) {
                num_residuals++;
            }
        }
    }
    return num_residuals;
}

bool BundleAdjuster::PyramidSolve(Reconstruction* reconstruction) {
    PrintHeading2("Bundle Adjustment Pyramid Solver");
    std::vector<std::pair<mappoint_t, class MapPoint*> > mappoints;
    for (auto &mappoint : reconstruction->MapPoints()) {
        mappoints.emplace_back(mappoint.first, (class MapPoint *)&mappoint.second);
    }

    std::sort(mappoints.begin(), mappoints.end(), 
        [](const std::pair<mappoint_t, class MapPoint*> &mappoint1, 
           const std::pair<mappoint_t, class MapPoint*> &mappoint2) {
        const class Track *track1 = &mappoint1.second->Track();
        const class Track *track2 = &mappoint2.second->Track();
        if (track1->Elements().size() == track2->Elements().size()) {
            return mappoint1.second->Error() < mappoint2.second->Error();
        } else {
            return track1->Elements().size() > track2->Elements().size();
        }
    });
    std::vector<mappoint_t> sorted_mappoint_ids;
    sorted_mappoint_ids.reserve(mappoints.size());
    for (auto & mappoint : mappoints) {
        sorted_mappoint_ids.push_back(mappoint.first);
    }

    size_t max_cover_per_view = 400;
    int iter = 0;

    std::vector<mappoint_t> const_mappoint_ids;
    while(const_mappoint_ids.size() < sorted_mappoint_ids.size()) {
        PrintHeading2(StringPrintf("iteration%d", iter++));
        std::vector<mappoint_t> ba_mappoint_ids;
        IterativeTrackSelection(reconstruction, sorted_mappoint_ids,
                                const_mappoint_ids, ba_mappoint_ids,
                                max_cover_per_view);
        std::cout << StringPrintf("max cover per view: %d\n", max_cover_per_view);
        std::cout << StringPrintf("selected mappoints: %d ", ba_mappoint_ids.size());
        std::cout << StringPrintf("fix mappoints: %d ", const_mappoint_ids.size());
        std::cout << StringPrintf("all mappoints: %d\n", sorted_mappoint_ids.size());
        
        BundleAdjustmentOptions options = options_;
        // if (max_cover_per_view > 4000) {
        //     options.refine_extrinsics = false;
        // }

        FastSolve(reconstruction, const_mappoint_ids, ba_mappoint_ids, options);
        for (const auto &camera : reconstruction->Cameras()){
            std::cout << "Camera#" << camera.first << ", param: " << camera.second.ParamsToString() << std::endl;
        }
        
        const_mappoint_ids.insert(const_mappoint_ids.end(), ba_mappoint_ids.begin(), ba_mappoint_ids.end());
        max_cover_per_view = (max_cover_per_view << 1);
    }
    return true;
}

bool BundleAdjuster::FastSolve(Reconstruction *reconstruction, 
                               std::vector<mappoint_t> &const_mappoint_ids, 
                               std::vector<mappoint_t> &ba_mappoint_ids,
                               const BundleAdjustmentOptions& options) {
    CHECK_NOTNULL(reconstruction);
    // CHECK(!problem_) << "Cannot use the same BundleAdjuster multiple times";

    problem_.reset(new ceres::Problem());

    ceres::LossFunction *loss_function = options.CreateLossFunction();

    SetUp(reconstruction, const_mappoint_ids, ba_mappoint_ids, loss_function, options);

    if (problem_->NumResiduals() == 0) {
        return false;
    }

    ceres::Solver::Options solver_options = options.solver_options;

    // Empirical choice.
    const size_t kMaxNumImagesDirectDenseSolver = 50;
    const size_t kMaxNumImagesDirectSparseSolver = 25000;
    // const size_t kMaxNumImageExplicitSchurComplemtent = 10000;
    const size_t num_images = config_.NumImages();
    if (num_images <= kMaxNumImagesDirectDenseSolver) {
        solver_options.linear_solver_type = ceres::DENSE_SCHUR;
#if CERES_VERSION_MAJOR >= 2 && CERES_VERSION_MINOR >= 2 && !defined(CERES_NO_CUDA)
        solver_options.dense_linear_algebra_library_type = ceres::CUDA;
#endif
    } else if (num_images <= kMaxNumImagesDirectSparseSolver) {
        solver_options.linear_solver_type = ceres::SPARSE_SCHUR;
    } else {  // Indirect sparse (preconditioned CG) solver.
        solver_options.linear_solver_type = ceres::ITERATIVE_SCHUR;
        solver_options.preconditioner_type = ceres::SCHUR_JACOBI;
        // if (num_images <= kMaxNumImageExplicitSchurComplemtent) {
        //     solver_options.use_explicit_schur_complement = true;
        // }
    }

    solver_options.num_threads = GetEffectiveNumThreads(-1);
#if CERES_VERSION_MAJOR < 2
    solver_options.num_linear_solver_threads = GetEffectiveNumThreads(-1);
#endif  // CERES_VERSION_MAJOR

    std::string solver_error;
    CHECK(solver_options.IsValid(&solver_error)) << solver_error;

    ceres::Solve(solver_options, problem_.get(), &summary_);

    if (solver_options.minimizer_progress_to_stdout) {
        std::cout << std::endl;
    }

    if (options.print_summary) {
        PrintHeading2("Bundle adjustment report");
        PrintSolverSummary(summary_);
    }

    // TearDown(reconstruction);

    return true;
}

size_t GetThreadsNum(const size_t blocks_count, const size_t threads_thres){
    size_t system_threads_num = std::thread::hardware_concurrency();

    if (system_threads_num <= 0) return 1;

    if (threads_thres <= 0) {
        return std::min(blocks_count, system_threads_num);
    }

    return std::min(std::min(threads_thres, blocks_count), system_threads_num);
}

bool BundleAdjuster::Solve(Reconstruction *reconstruction) {
    CHECK_NOTNULL(reconstruction);
    CHECK(!problem_) << "Cannot use the same BundleAdjuster multiple times";

    if (options_.use_gravity) {
        gravity_.setZero();
        for (auto & image_id : config_.Images()) {
            auto & image = reconstruction->Image(image_id);
            Eigen::Vector3d gravity = image.gravity_;
            Eigen::Matrix3d R = image.RotationMatrix();
            gravity_ += R.transpose() * gravity;
        }

        if (!gravity_.isZero()) {
            gravity_.normalize();
        }
    }

    problem_.reset(new ceres::Problem());

    ceres::LossFunction *loss_function = options_.CreateLossFunction();
    
//     if(options_.refine_points_only){
//         // for (const image_t image_id : config_.Images()) {
//         //     AddImageToStructOnlyProblem(image_id, reconstruction, loss_function);
//         // }

//         mappoint_t point_count = reconstruction->MapPointIds().size();
//         int block_count =  GetEffectiveNumThreads(options_.solver_options.num_threads);
//         if(point_count < block_count ){
//             block_count = 1;
//         }
//         std::cout<<"block count: "<<block_count<<std::endl;
//         std::unordered_set<mappoint_t> mappoint_ids_set = reconstruction->MapPointIds();
//         std::vector<mappoint_t> mappoint_ids;
//         mappoint_ids.reserve(mappoint_ids_set.size());
//         for(const auto& mappoint_id: mappoint_ids_set){
//             mappoint_ids.push_back(mappoint_id);
//         }

//         std::vector<std::vector<mappoint_t>> mappoint_blocks(block_count);

//         int start_id = 0;
//         mappoint_t block_mappoint_count = point_count / block_count;
//         for(int block_id = 0; block_id < block_count - 1; ++block_id){
//             for(mappoint_t i = 0; i< block_mappoint_count; ++i){
//                 mappoint_blocks[block_id].push_back(mappoint_ids[start_id++]);
//             }
//         }
//         for(; start_id < point_count; start_id ++){
//             mappoint_blocks[block_count-1].push_back(mappoint_ids[start_id]) ;
//         }


//         int omp_thres_num = GetThreadsNum(block_count, -1);
//         std::cout<<"omp_thres_num: "<<omp_thres_num<<std::endl;

// #pragma omp parallel for schedule(dynamic) num_threads(omp_thres_num)
//         for (int block_id = 0; block_id < block_count; ++ block_id) {
//             std::unique_ptr<ceres::Problem> block_problem;
//             block_problem.reset(new ceres::Problem());          
//             ceres::LossFunction *block_loss_function = options_.CreateLossFunction();

//             for(const auto& mappoint_id: mappoint_blocks[block_id]){
//                 AddPointToStructOnlyProblem(block_problem,mappoint_id,reconstruction,block_loss_function);
//             }

//             ceres::Solver::Options solver_options = options_.solver_options;
//             solver_options.linear_solver_type = ceres::SPARSE_SCHUR;
//             solver_options.num_threads = GetEffectiveNumThreads(1);

//             std::string solver_error;
//             CHECK(solver_options.IsValid(&solver_error)) << solver_error;

//             ceres::Solver::Summary block_summary;
//             ceres::Solve(solver_options, block_problem.get(), &block_summary);

//             if (options_.print_summary) {
//                 PrintHeading2("Bundle adjustment report");
//                 PrintSolverSummary(block_summary);
//             }
//         }
//         return true;
//     }
//     else{
        SetUp(reconstruction, loss_function);
    // }

    if (problem_->NumResiduals() == 0) {
        return false;
    }

    ceres::Solver::Options solver_options = options_.solver_options;

    // Empirical choice.
    const size_t kMaxNumImagesDirectDenseSolver = 50;
    const size_t kMaxNumImagesDirectSparseSolver = 25000;
    // const size_t kMaxNumImageExplicitSchurComplemtent = 10000;
    const size_t num_images = config_.NumImages();
    if (num_images <= kMaxNumImagesDirectDenseSolver) {
        solver_options.linear_solver_type = ceres::DENSE_SCHUR;
#if CERES_VERSION_MAJOR >= 2 && CERES_VERSION_MINOR >= 2 && !defined(CERES_NO_CUDA)
        solver_options.dense_linear_algebra_library_type = ceres::CUDA;
#endif
    } else if (num_images <= kMaxNumImagesDirectSparseSolver) {
        solver_options.linear_solver_type = ceres::SPARSE_SCHUR;
    } else {  // Indirect sparse (preconditioned CG) solver.
        solver_options.linear_solver_type = ceres::ITERATIVE_SCHUR;
        solver_options.preconditioner_type = ceres::SCHUR_JACOBI;
        // if (num_images <= kMaxNumImageExplicitSchurComplemtent) {
        //     solver_options.use_explicit_schur_complement = true;
        // }
    }

    solver_options.num_threads = GetEffectiveNumThreads(-1);
#if CERES_VERSION_MAJOR < 2
    solver_options.num_linear_solver_threads = GetEffectiveNumThreads(-1);
#endif  // CERES_VERSION_MAJOR

    std::string solver_error;
    CHECK(solver_options.IsValid(&solver_error)) << solver_error;

    ceres::Solve(solver_options, problem_.get(), &summary_);

    if (solver_options.minimizer_progress_to_stdout) {
        std::cout << std::endl;
    }

    if (options_.print_summary) {
        PrintHeading2("Bundle adjustment report");
        PrintSolverSummary(summary_);
    }

    TearDown(reconstruction);

    return true;
}

const ceres::Solver::Summary &BundleAdjuster::Summary() const { return summary_; }

void BundleAdjuster::SetUp(Reconstruction *reconstruction,
                           std::vector<mappoint_t>& const_mappoint_ids,
                           std::vector<mappoint_t>& ba_mappoint_ids,
                           ceres::LossFunction *loss_function,
                           const BundleAdjustmentOptions& options) {

    std::cout << "gnss images , camera num: " << config_.NumConstantGNSS() << " " << gnss_camera_ids_.size() << std::endl;
    std::cout << "plane constrain: " << options.plane_constrain << std::endl;
    std::cout << "gba weighted options: " << options.gba_weighted << std::endl;
    std::cout << "use prior pose: " << options.use_prior_relative_pose << std::endl;
    std::cout << "use gps prior: " << options.use_prior_absolute_location << ", prior weight: "
              << options.prior_absolute_location_weight << ", is aligned: " << reconstruction->b_aligned <<std::endl;
    std::cout << "refine focal length, principal point, extra params, local extrinsics: "
              << options.refine_focal_length << " " << options.refine_principal_point << " "
              << options.refine_extra_params << " " << options.refine_local_extrinsics << std::endl;

    const double eps = 1e-3;

    std::vector<mappoint_t> mappoint_ids;
    mappoint_ids.reserve(ba_mappoint_ids.size() + const_mappoint_ids.size());
    mappoint_ids.insert(mappoint_ids.end(), ba_mappoint_ids.begin(), ba_mappoint_ids.end());
    mappoint_ids.insert(mappoint_ids.end(), const_mappoint_ids.begin(), const_mappoint_ids.end());

    std::unordered_set<image_t> image_ids;
    std::unordered_map<image_t, int> observations_per_image;
    for (size_t i = 0; i < mappoint_ids.size(); ++i) {
        mappoint_t mappoint_id = mappoint_ids.at(i);
        class MapPoint& mappoint = reconstruction->MapPoint(mappoint_id);
        const class Track &track = mappoint.Track();
        if (track.Length() <= 1) {
            continue;
        }

        bool fix_mappoint = (i >= ba_mappoint_ids.size());

        for (const auto &track_el : track.Elements()) {
            observations_per_image[track_el.image_id]++;

            class Image &image = reconstruction->Image(track_el.image_id);
            class Camera &camera = reconstruction->Camera(image.CameraId());
            const Point2D &point2D = image.Point2D(track_el.point2D_idx);

            if (image_ids.find(track_el.image_id) == image_ids.end()) {
                image_ids.insert(track_el.image_id);
                camera_ids_.insert(image.CameraId());

                // CostFunction assumes unit quaternions.
                image.NormalizeQvec();
            }

            double *qvec_data = image.Qvec().data();
            double *tvec_data = image.Tvec().data();
            double *camera_params_data = camera.ParamsData();

            const bool constant_pose = !options.refine_extrinsics || config_.HasConstantPose(track_el.image_id);

            ceres::CostFunction *cost_function = nullptr;

            if (constant_pose) {
                if (fix_mappoint) {
                    switch (camera.ModelId()) {
                    #define CAMERA_MODEL_CASE(CameraModel)\
                        case CameraModel::kModelId:\
                            cost_function = \
                                BundleAdjustmentConstantPoseAndMapPointCostFunction<CameraModel>::Create(image.Qvec(), image.Tvec(), point2D.XY(), mappoint.XYZ()); \
                            break;
                        CAMERA_MODEL_SWITCH_CASES
                    #undef CAMERA_MODEL_CASE
                    }
                    problem_->AddResidualBlock(cost_function, loss_function, camera_params_data);
                } else {
                    switch (camera.ModelId()) {
                    #define CAMERA_MODEL_CASE(CameraModel)\
                        case CameraModel::kModelId:\
                            cost_function = \
                                BundleAdjustmentConstantPoseCostFunction<CameraModel>::Create(image.Qvec(), image.Tvec(), point2D.XY()); \
                            break;
                        CAMERA_MODEL_SWITCH_CASES
                    #undef CAMERA_MODEL_CASE
                    }
                    problem_->AddResidualBlock(cost_function, loss_function, mappoint.XYZ().data(), camera_params_data);
                }
            } else {
                if (fix_mappoint) {
                    switch (camera.ModelId()) {
                    #define CAMERA_MODEL_CASE(CameraModel)\
                        case CameraModel::kModelId:\
                            cost_function = BundleAdjustmentConstantMapPointCostFunction<CameraModel>::Create(point2D.XY(), mappoint.XYZ()); \
                            break;
                        CAMERA_MODEL_SWITCH_CASES
                    #undef CAMERA_MODEL_CASE
                    }
                    problem_->AddResidualBlock(cost_function, loss_function, qvec_data, tvec_data, camera_params_data);
                } else {
                    switch (camera.ModelId()) {
                    #define CAMERA_MODEL_CASE(CameraModel)\
                        case CameraModel::kModelId:\
                            cost_function = BundleAdjustmentCostFunction<CameraModel>::Create(point2D.XY()); \
                            break;
                        CAMERA_MODEL_SWITCH_CASES
                    #undef CAMERA_MODEL_CASE
                    }
                    problem_->AddResidualBlock(cost_function, loss_function, qvec_data, tvec_data, mappoint.XYZ().data(), camera_params_data);
                }
            }
        }
    }

    for (const auto &image_id : image_ids) {
        class Image &image = reconstruction->Image(image_id);
        class Camera &camera = reconstruction->Camera(image.CameraId());

        size_t num_observations = std::max(observations_per_image.at(image_id), 1);

        double *qvec_data = image.Qvec().data();
        double *tvec_data = image.Tvec().data();
        double *camera_params_data = camera.ParamsData();

        const bool constant_pose = !options.refine_extrinsics || config_.HasConstantPose(image_id);

        if (!constant_pose && options.use_prior_absolute_location && reconstruction->b_aligned) {
            // add prior absolute location constrain from gps prior
            if (reconstruction->prior_locations_gps.find(image_id) != reconstruction->prior_locations_gps.end()) {
                
                CHECK(reconstruction->prior_locations_gps_inlier.find(image_id) !=
                    reconstruction->prior_locations_gps_inlier.end());
                CHECK(reconstruction->prior_horizontal_locations_gps_inlier.find(image_id) !=
                    reconstruction->prior_horizontal_locations_gps_inlier.end());
                
                if (reconstruction->prior_locations_gps_inlier.at(image_id)) {
                    Eigen::Vector3d prior_c = reconstruction->prior_locations_gps.at(image_id).first;
                    
                    ceres::CostFunction *cost_function = nullptr;
                    if(options.optimization_use_horizontal_gps_only){
                        cost_function = PriorAbsoluteLocationOnPlaneCostFunction::Create(
                            prior_c, reconstruction->projection_plane, options.prior_absolute_location_weight);
                    }
                    else{
                        cost_function =
                            PriorAbsoluteLocationCostFunction::Create(prior_c, options.prior_absolute_location_weight,
                                options.prior_absolute_location_weight, options.prior_absolute_location_weight);
                    }
                    problem_->AddResidualBlock(cost_function, loss_function, qvec_data, tvec_data);
                }
                else if (reconstruction->prior_horizontal_locations_gps_inlier.at(image_id)) {
                    Eigen::Vector3d prior_c = reconstruction->prior_locations_gps.at(image_id).first;
                    prior_c[2] = 0;

                    ceres::CostFunction *cost_function = nullptr;
                    cost_function = PriorAbsoluteLocationOnPlaneCostFunction::Create(
                        prior_c, reconstruction->projection_plane, options.prior_absolute_location_weight);

                    problem_->AddResidualBlock(cost_function, loss_function, qvec_data, tvec_data);
                }
            }

            // add prior absolute pose constrain from RTK prior
            if (image.HasQvecPrior() && image.HasTvecPrior() && image.RtkFlag() == 50) {

                Eigen::Vector4d prior_q = image.QvecPrior();
                // Eigen::Vector3d prior_c = -QuaternionToRotationMatrix(prior_q) * image.TvecPrior();
                Eigen::Vector3d prior_c = image.TvecPrior();

                ceres::ScaledLoss *rtk_loss_function = 
                        new ceres::ScaledLoss(new ceres::TrivialLoss(), 
                        static_cast<double>(num_observations), ceres::DO_NOT_TAKE_OWNERSHIP);
                
                // double weight_q = options.prior_absolute_location_weight * 0.1;

                double weight_q = options.prior_absolute_orientation_weight;
                if (image.HasOrientStd()){
                    weight_q = options.prior_absolute_orientation_weight / image.OrientStd();
                }
                double weight_x = image.HasRtkStd() ? options_.prior_absolute_location_weight / (image.RtkStdLat() + eps) : 
                                                        options_.prior_absolute_location_weight * 10.0;
                double weight_y = image.HasRtkStd() ? options_.prior_absolute_location_weight / (image.RtkStdLon() + eps) :
                                                        options_.prior_absolute_location_weight * 10.0;
                double weight_z = image.HasRtkStd() ? options_.prior_absolute_location_weight / (image.RtkStdHgt() + eps) :
                                                        options_.prior_absolute_location_weight * 10.0;

                ceres::CostFunction *cost_function = PriorAbsolutePoseCostFunction::Create(
                    prior_q, prior_c, weight_q, weight_x, weight_y, weight_z);
                problem_->AddResidualBlock(cost_function, rtk_loss_function, qvec_data, tvec_data);
            }
        }

        // Set pose parameterization.
        if (!constant_pose) {
            ceres::LocalParameterization *quaternion_parameterization = new ceres::QuaternionParameterization;
            SetParameterization(problem_.get(), qvec_data, quaternion_parameterization);
            if (config_.HasConstantTvec(image_id)) {
                const std::vector<int> &constant_tvec_idxs = config_.ConstantTvec(image_id);
                ceres::SubsetParameterization *tvec_parameterization =
                    new ceres::SubsetParameterization(3, constant_tvec_idxs);
                SetParameterization(problem_.get(), tvec_data, tvec_parameterization);
            }
        }
    }

    ParameterizeCameras(reconstruction);
}

void BundleAdjuster::SetUp(Reconstruction *reconstruction, ceres::LossFunction *loss_function) {
    // Warning: AddPointsToProblem assumes that AddImageToProblem is called first.
    // Do not change order of instructions!
    for (const image_t image_id : config_.Images()) {
        AddImageToProblem(image_id, reconstruction, loss_function);
    }
    for (const image_t image_id : config_.GNSSImages()) {
        AddNovatelToProblem(image_id, reconstruction, loss_function);
    }
    for (const auto mappoint_id : config_.VariablePoints()) {
        AddPointToProblem(mappoint_id, reconstruction, loss_function);
    }
    for (const auto mappoint_id : config_.ConstantPoints()) {
        AddPointToProblem(mappoint_id, reconstruction, loss_function);
    }

    if (config_.SweepImagePairs().size() >= 10) {
        AddSfMToLidarConstraint(reconstruction);
    }

    uint32_t m_num_visible_point = 0;
    const auto register_image_ids = reconstruction->RegisterImageIds();
    for (auto register_image_id : register_image_ids) {
        m_num_visible_point += reconstruction->Image(register_image_id).NumVisibleMapPoints();
    }
    m_num_visible_point /= register_image_ids.size();

    // lidar
    std::cout << "lidar_weight: " << options_.lidar_weight << std::endl;
    if (options_.lidarsweep_voxel_gnss){
        std::cout << "LidarVoxelToProblem added "  << config_.NumSweeps() << " sweeps"<< std::endl;
        // for (const auto sweep_id : config_.Sweeps()){
        //     AddLidarToProblem(sweep_id, reconstruction, loss_function);
        // }
        for (auto image_lidar : config_.SweepImagePairs()) {
            AddLidarToProblem(image_lidar.second, image_lidar.first, m_num_visible_point, reconstruction, loss_function);
        }
        // if (config_.NumSweeps() > 0) {
        //     AddMapPointToVoxelMap(reconstruction, loss_function);
        // }
    } else {
        std::cout << "LidarFrame2FrameToProblem added "  << config_.NumSweeps() << " sweeps"<< std::endl;
        for (const auto sweep_id : config_.Sweeps()){
            AddLidarFrame2FrameToProblem(sweep_id, reconstruction, loss_function, options_.lidar_weight);
        }
    }
    std::cout << "gnss images , camera num: " << config_.NumConstantGNSS() << " " << gnss_camera_ids_.size() << std::endl;
    std::cout << "plane constrain: " << options_.plane_constrain << std::endl;
    std::cout << "gba weighted options: " << options_.gba_weighted << std::endl;
    std::cout << "use prior pose: " << options_.use_prior_relative_pose << std::endl;
    std::cout << "use gps prior: " << options_.use_prior_absolute_location << ", prior weight: "
              << options_.prior_absolute_location_weight << ", is aligned: " << reconstruction->b_aligned <<std::endl;
    std::cout << "refine focal length, principal point, extra params, local extrinsics: "
              << options_.refine_focal_length << " " << options_.refine_principal_point << " "
              << options_.refine_extra_params << " " << options_.refine_local_extrinsics << std::endl;
    ParameterizeCameras(reconstruction);
    if (options_.parameterize_points_with_track) {
        ParameterizePoints(reconstruction);
    }
    else {
        ParameterizePointsWithoutTrack(reconstruction);
    }
    ParameterizePoses(reconstruction);
}

void BundleAdjuster::TearDown(Reconstruction *) {
    // Nothing to do
}

void BundleAdjuster::AddImageToProblem(const image_t image_id, Reconstruction *reconstruction,
                                       ceres::LossFunction *loss_function) {
    Image &image = reconstruction->Image(image_id);
    Camera &camera = reconstruction->Camera(image.CameraId());
    // CostFunction assumes unit quaternions.
    image.NormalizeQvec();

    double *qvec_data = image.Qvec().data();
    double *tvec_data = image.Tvec().data();
    double *camera_params_data = camera.ParamsData();

    Eigen::Matrix3d R = image.RotationMatrix();
    Eigen::Vector3d t = image.Tvec();

    double *local_qvec1_data;
    double *local_tvec1_data;
    double *local_qvec2_data;
    double *local_tvec2_data;

    const bool constant_pose = !options_.refine_extrinsics || config_.HasConstantPose(image_id);

    const bool constant_intrinsics =
        !options_.refine_focal_length && !options_.refine_extra_params && !options_.refine_principal_point;

    double weight = 1.0;

    const std::vector<uint32_t> &local_image_indices = image.LocalImageIndices();
    std::vector<double *> local_qvec_data;
    std::vector<double *> local_tvec_data;
    std::vector<double *> local_camera_params_data;
    // This is a camera-rig
    if (camera.NumLocalCameras() > 1) {
        int local_param_size = camera.LocalParams().size() / camera.NumLocalCameras();

        for (size_t i = 0; i < camera.NumLocalCameras(); ++i) {
            local_qvec_data.push_back(camera.LocalQvecsData() + 4 * i);
            local_tvec_data.push_back(camera.LocalTvecsData() + 3 * i);
            local_camera_params_data.push_back(camera.LocalIntrinsicParamsData() + i * local_param_size);
        }
    }

    std::unordered_set<int> local_camera_in_the_image;

    const double eps = 1e-3;

    // Add residuals to bundle adjustment problem.
    size_t num_observations = 0;
    int point2d_id = -1;
    for (const Point2D &point2D : image.Points2D()) {
        point2d_id++;
        if (!point2D.HasMapPoint()) {
            continue;
        }

        uint32_t local_camera_id = local_image_indices[point2d_id];

        num_observations += 1;
        mappoint_num_observations_[point2D.MapPointId()] += 1;

        MapPoint &mappoint = reconstruction->MapPoint(point2D.MapPointId());
        assert(mappoint.Track().Length() > 1);

        uint64_t mappoint_create_time = mappoint.CreateTime();
        double loop_weight = 1.0;
        if (mappoint_create_time != 0 && image.create_time_ > 0) {
            int time_diff = std::abs((int)image.create_time_ - (int)mappoint_create_time);
            if (time_diff > 200) loop_weight = 20.0;
            else if (time_diff > 100) {
                loop_weight = 0.19 * time_diff - 18;
            }
            // if (time_diff > 200) loop_weight = 40.0;
            // else {
            //     loop_weight = std::max(1.0, time_diff / 5.0);
            // }
        } else {
            std::cout << StringPrintf("Image %s has no timestamp!", image.Name().c_str()) << std::endl;
        }

        // if (config_.Sweeps().size() > 0) {
        //     loop_weight *= 5.0;
        // }

        Eigen::Vector3d point3D_mea;  
        const float info = GetDepthPointAndWeight(camera, point2D, point3D_mea);
        double observation_factor = 1.0;
        if (point2D.InOverlap() && camera.NumLocalCameras() > 1) {
            Eigen::Vector4d local_qvec(local_qvec_data[local_camera_id]);
            Eigen::Vector3d local_tvec(local_tvec_data[local_camera_id]);
            Eigen::Matrix3d local_R = QuaternionToRotationMatrix(local_qvec);
            Eigen::Vector3d Xc = local_R *(R * mappoint.XYZ() + t) + local_tvec;
            observation_factor = 1.0 + 9 * std::exp(-Xc[2] * Xc[2] / 10.0);
            // if (Xc[2] < 1.5) {
            //     observation_factor = 10.0f;
            // }
        }

        ceres::CostFunction *cost_function = nullptr;

        if (constant_pose) {
            if (camera.NumLocalCameras() > 1) {
                // for camera-rig
                if (camera.ModelName().compare("OPENCV_FISHEYE") == 0 && constant_intrinsics) {
                    local_camera_in_the_image.insert(local_camera_id);

                    double f = camera.LocalMeanFocalLength(local_camera_id);
                    Eigen::Vector3d bearing = camera.LocalImageToBearing(local_camera_id, point2D.XY());
                    cost_function =
                        LargeFovRigBundleAdjustmentConstantPoseCostFunction<OpenCVFisheyeCameraModel>::Create(
                            image.Qvec(), image.Tvec(), bearing, f * loop_weight);

                    problem_->AddResidualBlock(cost_function, loss_function, local_qvec_data[local_camera_id],
                                               local_tvec_data[local_camera_id], mappoint.XYZ().data());

                } else {
                    switch (camera.ModelId()) {
#define CAMERA_MODEL_CASE(CameraModel)                                                    \
    case CameraModel::kModelId:                                                           \
        cost_function = RigBundleAdjustmentConstantPoseCostFunction<CameraModel>::Create( \
            image.Qvec(), image.Tvec(), point2D.XY(), observation_factor * loop_weight);  \
        break;
                        CAMERA_MODEL_SWITCH_CASES
#undef CAMERA_MODEL_CASE
                    }
                    local_camera_in_the_image.insert(local_camera_id);

                    problem_->AddResidualBlock(cost_function, loss_function, local_qvec_data[local_camera_id],
                                               local_tvec_data[local_camera_id], mappoint.XYZ().data(),
                                               local_camera_params_data[local_camera_id]);
                }

            } else {
                // for monocular camera
                if (camera.ModelName().compare("SPHERICAL") == 0) {
                    Eigen::Vector3d bearing = camera.ImageToBearing(point2D.XY());
                    double f = camera.FocalLength();

                    cost_function = SphericalBundleAdjustmentConstantPoseCostFunction<SphericalCameraModel>::Create(
                        image.Qvec(), image.Tvec(), bearing, f, loop_weight);

                    local_qvec2_data = camera_params_data + 10;
                    local_tvec2_data = camera_params_data + 14;

                    problem_->AddResidualBlock(cost_function, loss_function, mappoint.XYZ().data(),
                                               local_qvec2_data, local_tvec2_data);
                } else if (camera.ModelName().compare("OPENCV_FISHEYE") == 0 && constant_intrinsics){
                    Eigen::Vector3d bearing = camera.ImageToBearing(point2D.XY());
                    double f = camera.MeanFocalLength();
                    cost_function = LargeFovBundleAdjustmentConstantPoseCostFunction<OpenCVFisheyeCameraModel>::Create(
                        image.Qvec(), image.Tvec(), bearing, f, weight * loop_weight);

                    problem_->AddResidualBlock(cost_function, loss_function, mappoint.XYZ().data());
                } else {
                    switch (camera.ModelId()) {
#define CAMERA_MODEL_CASE(CameraModel)                                                                            \
    case CameraModel::kModelId:                                                                                   \
        cost_function = BundleAdjustmentConstantPoseCostFunction<CameraModel>::Create(image.Qvec(), image.Tvec(), \
                                                                                      point2D.XY(), weight * loop_weight); \
        break;
                        CAMERA_MODEL_SWITCH_CASES
#undef CAMERA_MODEL_CASE
                    }
                    problem_->AddResidualBlock(cost_function, loss_function, mappoint.XYZ().data(), camera_params_data);
                    if (point3D_mea.z() > 0 && reconstruction->depth_enabled) {
                        cost_function =
                            BundleAdjustmentConstantPoseDepthCostFunction::Create(
                                image.Qvec(), image.Tvec(), point3D_mea, options_.rgbd_ba_depth_weight * info * loop_weight
                            );
                        problem_->AddResidualBlock(cost_function, loss_function, 
                                                mappoint.XYZ().data());
                    }
                }
            }
        } else {
            if (camera.NumLocalCameras() > 1) {
                // for camera-rig
                if (camera.ModelName().compare("OPENCV_FISHEYE") == 0 && constant_intrinsics) {
                    local_camera_in_the_image.insert(local_camera_id);

                    double f = camera.LocalMeanFocalLength(local_camera_id);
                    Eigen::Vector3d bearing = camera.LocalImageToBearing(local_camera_id, point2D.XY());
                    cost_function =
                        LargeFovRigBundleAdjustmentCostFunction<OpenCVFisheyeCameraModel>::Create(bearing, f, loop_weight);
                    problem_->AddResidualBlock(cost_function, loss_function, qvec_data, tvec_data,
                                               local_qvec_data[local_camera_id], local_tvec_data[local_camera_id],
                                               mappoint.XYZ().data());
                } else {
                    switch (camera.ModelId()) {
#define CAMERA_MODEL_CASE(CameraModel)                                                      \
    case CameraModel::kModelId:                                                             \
        cost_function = RigBundleAdjustmentCostFunction<CameraModel>::Create(point2D.XY(), observation_factor * loop_weight); \
        break;
                        CAMERA_MODEL_SWITCH_CASES
#undef CAMERA_MODEL_CASE
                    }
                    local_camera_in_the_image.insert(local_camera_id);
                    problem_->AddResidualBlock(cost_function, loss_function, qvec_data, tvec_data,
                                               local_qvec_data[local_camera_id], local_tvec_data[local_camera_id],
                                               mappoint.XYZ().data(), local_camera_params_data[local_camera_id]);
                }

            } else {
                if (camera.ModelName().compare("SPHERICAL") == 0) {
                    Eigen::Vector3d bearing = camera.ImageToBearing(point2D.XY());
                    double f = camera.FocalLength();
                    
                    cost_function =
                        SphericalBundleAdjustmentCostFunction<SphericalCameraModel>::Create(bearing, f, loop_weight);

                    local_qvec2_data = camera_params_data + 10;
                    local_tvec2_data = camera_params_data + 14;

                    problem_->AddResidualBlock(cost_function, loss_function, qvec_data, tvec_data,
                                               mappoint.XYZ().data(), local_qvec2_data, local_tvec2_data);
                } else if(camera.ModelName().compare("OPENCV_FISHEYE") == 0 && constant_intrinsics){ 
                    Eigen::Vector3d bearing = camera.ImageToBearing(point2D.XY());
                    double f = camera.MeanFocalLength();
                    cost_function =
                        LargeFovBundleAdjustmentCostFunction<OpenCVFisheyeCameraModel>::Create(bearing, f, weight * loop_weight);
                    problem_->AddResidualBlock(cost_function, loss_function, qvec_data, tvec_data,
                                               mappoint.XYZ().data());

                } else {
                    switch (camera.ModelId()) {
#define CAMERA_MODEL_CASE(CameraModel)                                                           \
    case CameraModel::kModelId:                                                                  \
        cost_function = BundleAdjustmentCostFunction<CameraModel>::Create(point2D.XY(), weight * loop_weight); \
        break;
                        CAMERA_MODEL_SWITCH_CASES
#undef CAMERA_MODEL_CASE
                    }
                    problem_->AddResidualBlock(cost_function, loss_function, qvec_data, tvec_data,
                                               mappoint.XYZ().data(), camera_params_data);
                    if (point3D_mea.z() > 0 && reconstruction->depth_enabled) {
                        cost_function = 
                            BundleAdjustmentDepthCostFunction::Create(point3D_mea, options_.rgbd_ba_depth_weight * info * loop_weight);
                        problem_->AddResidualBlock(cost_function, loss_function, 
                                                qvec_data, tvec_data, 
                                                mappoint.XYZ().data());
                    }
                }
            }
        }
    }

    if (options_.refine_local_extrinsics && options_.local_relative_translation_constraint && camera.NumLocalCameras() > 1) {
        Eigen::Vector3d weight(num_observations * 0.2f, num_observations * 0.2f, num_observations * 0.2f);
        for (size_t i = 1; i < camera.NumLocalCameras(); ++i) {
            ceres::CostFunction *cost_function = RigLocalRelativeBundleAdjustmentPoseFunction::Create(weight);
            problem_->AddResidualBlock(cost_function, loss_function, local_qvec_data[0], local_tvec_data[0], 
                                       local_qvec_data[i], local_tvec_data[i]);
        }
    }

    // add plane constrain
    bool on_plane = reconstruction->planes_for_images.find(image_id) != reconstruction->planes_for_images.end();
    if (!constant_pose && options_.plane_constrain && on_plane) {
        Eigen::Vector4d plane = reconstruction->planes_for_images.at(image_id);
        ceres::CostFunction *cost_function = PlaneConstrainCostFunction::Create(
            plane, reconstruction->baseline_distance, num_observations, options_.plane_weight);

        problem_->AddResidualBlock(cost_function, loss_function, qvec_data, tvec_data);
    }

    // add prior relative pose constrain
    if (!constant_pose && options_.use_prior_relative_pose) {
        image_t neighbor_image_id = image_id + 1;

        for (image_t i = 1; i < 10; ++i) {
            bool relative_pose_valid = false;

            relative_pose_valid =
                (config_.Images().count(neighbor_image_id) > 0) &&
                (reconstruction->prior_rotations.find(image_id) != reconstruction->prior_rotations.end()) &&
                (reconstruction->prior_rotations.find(neighbor_image_id) != reconstruction->prior_rotations.end());

            if (relative_pose_valid) {
                Eigen::Vector4d prior_qvec1 = reconstruction->prior_rotations.at(image_id);
                Eigen::Vector3d prior_tvec1 = reconstruction->prior_translations.at(image_id);
                Eigen::Matrix3d prior_R1 = QuaternionToRotationMatrix(prior_qvec1);

                Eigen::Vector4d prior_qvec2 = reconstruction->prior_rotations.at(neighbor_image_id);
                Eigen::Vector3d prior_tvec2 = reconstruction->prior_translations.at(neighbor_image_id);
                Eigen::Matrix3d prior_R2 = QuaternionToRotationMatrix(prior_qvec2);

                Eigen::Matrix3d prior_relative_R12 = prior_R2 * prior_R1.transpose();
                Eigen::Vector3d prior_relative_t12 = prior_tvec2 - prior_relative_R12 * prior_tvec1;
                Eigen::Vector4d prior_relative_q12 = RotationMatrixToQuaternion(prior_relative_R12);

                Image &neighbor_image = reconstruction->Image(neighbor_image_id);

                // CostFunction assumes unit quaternions.
                neighbor_image.NormalizeQvec();
                double *qvec_data2 = neighbor_image.Qvec().data();
                double *tvec_data2 = neighbor_image.Tvec().data();

                if (options_.use_prior_distance_only) {
                    ceres::CostFunction *cost_function = PriorRelativeDistanceCostFunction::Create(
                        prior_relative_q12, prior_relative_t12,
                        static_cast<double>(num_observations) * options_.prior_pose_weight);
                    problem_->AddResidualBlock(cost_function, loss_function, qvec_data, tvec_data, qvec_data2,
                                               tvec_data2);
                } else if (options_.use_prior_translation_only) {
                    ceres::CostFunction *cost_function = PriorRelativeTranslationCostFunction::Create(
                        prior_relative_q12, prior_relative_t12,
                        static_cast<double>(num_observations) * options_.prior_pose_weight);
                    problem_->AddResidualBlock(cost_function, loss_function, qvec_data, tvec_data, qvec_data2,
                                               tvec_data2);
                } else {
                    ceres::CostFunction *cost_function = PriorRelativePoseCostFunction::Create(
                        prior_relative_q12, prior_relative_t12,
                        static_cast<double>(num_observations) * options_.prior_pose_weight);
                    problem_->AddResidualBlock(cost_function, loss_function, qvec_data, tvec_data, qvec_data2,
                                               tvec_data2);
                }

                if (invovled_image_extrinsics_.count(image_id) == 0) {
                    invovled_image_extrinsics_.insert(image_id);
                }

                if (invovled_image_extrinsics_.count(neighbor_image_id) == 0) {
                    invovled_image_extrinsics_.insert(neighbor_image_id);
                }
            }
            neighbor_image_id++;
        }
    }

    // add prior relative pose constrain
    if (options_.use_prior_aggressively && !constant_pose && options_.use_prior_relative_pose && 
        (reconstruction->prior_rotations.find(image_id) != reconstruction->prior_rotations.end()) &&
        (reconstruction->prior_translations.find(image_id) != reconstruction->prior_translations.end())) {
        
        // find neighbors that have prior poses in [-100, +100], maximum 30
        size_t relative_count = 0; 
        for (size_t i = 2; i < 202 && relative_count < 30; ++i) {
            image_t neighbor_image_id;
            if (i % 2) {
                neighbor_image_id = image_id + i / 2;
            }
            else {
                neighbor_image_id = image_id - i / 2;
            }

            // farther frames have less weight
            double distance_weight = std::exp(-0.0002 * (i / 2) * (i / 2));

            // By default, we apply 1 prior constraint for every 1 valid ovservation, 
            // i.e., the basic weight is exactly `num_observations`.
            double ovservation_weight = num_observations;

            // However, images with suffecient/insuffecient ovservations should consider less/more on priors. 
            ovservation_weight *= 3.0 / (1.5 + std::log(num_observations * num_observations * 0.0003 + 1));
            
            // The final weight
            double weight = distance_weight * ovservation_weight * options_.prior_pose_weight;

            bool relative_pose_valid = false;
            relative_pose_valid =
                (config_.Images().count(neighbor_image_id) > 0) &&
                (reconstruction->prior_rotations.find(neighbor_image_id) != reconstruction->prior_rotations.end());

            if (relative_pose_valid) {
                relative_count++;

                Eigen::Vector4d prior_qvec1 = reconstruction->prior_rotations.at(image_id);
                Eigen::Vector3d prior_tvec1 = reconstruction->prior_translations.at(image_id);
                Eigen::Matrix3d prior_R1 = QuaternionToRotationMatrix(prior_qvec1);

                Eigen::Vector4d prior_qvec2 = reconstruction->prior_rotations.at(neighbor_image_id);
                Eigen::Vector3d prior_tvec2 = reconstruction->prior_translations.at(neighbor_image_id);
                Eigen::Matrix3d prior_R2 = QuaternionToRotationMatrix(prior_qvec2);

                Eigen::Matrix3d prior_relative_R12 = prior_R2 * prior_R1.transpose();
                Eigen::Vector3d prior_relative_t12 = prior_tvec2 - prior_relative_R12 * prior_tvec1;
                Eigen::Vector4d prior_relative_q12 = RotationMatrixToQuaternion(prior_relative_R12);

                Image &neighbor_image = reconstruction->Image(neighbor_image_id);

                // CostFunction assumes unit quaternions.
                neighbor_image.NormalizeQvec();
                double *qvec_data2 = neighbor_image.Qvec().data();
                double *tvec_data2 = neighbor_image.Tvec().data();

                if (options_.use_prior_distance_only) {
                    ceres::CostFunction *cost_function = PriorRelativeDistanceCostFunction::Create(
                        prior_relative_q12, prior_relative_t12, weight);
                    problem_->AddResidualBlock(cost_function, loss_function, qvec_data, tvec_data, qvec_data2,
                                               tvec_data2);
                } else if (options_.use_prior_translation_only) {
                    ceres::CostFunction *cost_function = PriorRelativeTranslationCostFunction::Create(
                        prior_relative_q12, prior_relative_t12, weight);
                    problem_->AddResidualBlock(cost_function, loss_function, qvec_data, tvec_data, qvec_data2,
                                               tvec_data2);
                } else {
                    ceres::CostFunction *cost_function = PriorRelativePoseCostFunction::Create(
                        prior_relative_q12, prior_relative_t12, weight);
                    problem_->AddResidualBlock(cost_function, loss_function, qvec_data, tvec_data, qvec_data2,
                                               tvec_data2);
                }

                if (invovled_image_extrinsics_.count(image_id) == 0) {
                    invovled_image_extrinsics_.insert(image_id);
                }

                if (invovled_image_extrinsics_.count(neighbor_image_id) == 0) {
                    invovled_image_extrinsics_.insert(neighbor_image_id);
                }
            }
        }
    }

    // add icp relative pose
    if (!constant_pose && options_.use_icp_relative_pose && reconstruction->depth_enabled) {
        Image &image = reconstruction->Image(image_id);
        auto &icp_links = image.icp_links_;
        int icp_cnt = 0;
        for(auto lk:icp_links){
            if(!config_.Images().count(lk.ref_id_)) continue;
            Image &lk_image = reconstruction->Image(lk.ref_id_);

            // CostFunction assumes unit quaternions.
            lk_image.NormalizeQvec();
            double *lk_qvec = lk_image.Qvec().data();
            double *lk_tvec = lk_image.Tvec().data();

//            double ovservation_weight = 0.1 + std::log(num_observations / 50.0 + 1.0);

            double weight =  lk.conf_ * options_.icp_base_weight;
            //  lk.infomation_ = Eigen::Matrix6d::Identity();
            ceres::CostFunction *cost_function = ICPRelativePoseCostFunction::Create(
                    lk.X_, lk.infomation_, weight);
            problem_->AddResidualBlock(cost_function, new ceres::HuberLoss(options_.icp_base_weight/5.0), qvec_data, tvec_data, lk_qvec,
                                       lk_tvec);

            std::vector<double*> paras={qvec_data, tvec_data, lk_qvec, lk_tvec};
            icp_cnt++;
            if (invovled_image_extrinsics_.count(lk.ref_id_) == 0) {
                invovled_image_extrinsics_.insert(lk.ref_id_);
            }
        }
        if (icp_cnt && invovled_image_extrinsics_.count(image_id) == 0) {
            invovled_image_extrinsics_.insert(image_id);
        }
    }

    // add gravity
    if (!constant_pose && options_.use_gravity) {
        Image &image = reconstruction->Image(image_id);
        auto &cur_g = image.gravity_;
        auto &world_g = gravity_;
        if(!cur_g.isZero() && !world_g.isZero()){
            auto loss_func =  new ceres::HuberLoss(0.01);
            ceres::CostFunction *cost_function = GravityCostFunction::Create(
                    world_g, cur_g, Eigen::Matrix3d::Identity(), -1.0, options_.gravity_base_weight);
            problem_->AddResidualBlock(cost_function, nullptr, qvec_data);
            std::vector<double*> paras={qvec_data};
        }
    }

    // add time domain smoothing
    if (!constant_pose && options_.use_time_domain_smoothing) {
        Image &image = reconstruction->Image(image_id);
        const long long timestamp = image.timestamp_;

        if (timestamp > 0) {
            image_t prev_image_id = -1;
            image_t next_image_id = -1;

            for (size_t i = 1; i <= 10 && prev_image_id == -1; ++i) {
                image_t neighbor_image_id = image_id - i;
                if (config_.Images().count(neighbor_image_id) > 0) {
                    prev_image_id = neighbor_image_id;
                }
            }

            for (size_t i = 1; i <= 10 && next_image_id == -1; ++i) {
                image_t neighbor_image_id = image_id + i;
                if (config_.Images().count(neighbor_image_id) > 0) {
                    next_image_id = neighbor_image_id;
                }
            }

            if (prev_image_id != -1 && next_image_id != -1) {
                Image &prev_image = reconstruction->Image(prev_image_id);
                Image &next_image = reconstruction->Image(next_image_id);
                const long long prev_timestamp = prev_image.timestamp_;
                const long long next_timestamp = next_image.timestamp_;

                if (prev_timestamp < timestamp && timestamp < next_timestamp) {
                    // a large time-range has smaller weight due to uncertainty
                    // +/-0.1s ~= 60% of the base weight
                    const long long prev_time = timestamp - prev_timestamp;
                    const long long next_time = next_timestamp - timestamp;
                    const double weight = options_.time_domain_smoothing_weight * 
                        std::exp(-5.0 * prev_time) *
                        std::exp(-5.0 * next_time);

                    ceres::CostFunction *cost_function = TimeDomainSmoothingCostFunction::Create(
                        prev_time, next_time, weight);
                    problem_->AddResidualBlock(cost_function, loss_function, 
                        qvec_data, tvec_data, 
                        prev_image.Qvec().data(), prev_image.Tvec().data(),
                        next_image.Qvec().data(), next_image.Tvec().data());

                    // const Eigen::Vector3d prev_position = -prev_image.RotationMatrix().transpose() * prev_image.Tvec();
                    // const Eigen::Vector3d curr_position = -image.RotationMatrix().transpose() * image.Tvec();
                    // const Eigen::Vector3d next_position = -next_image.RotationMatrix().transpose() * next_image.Tvec();
                    // const Eigen::Vector3d prev_velocity = (curr_position - prev_position) / prev_time;
                    // const Eigen::Vector3d next_velocity = (next_position - curr_position) / next_time;
                    // std::cout << prev_image_id << " " << image_id << " " << next_image_id << ": " << weight << std::endl;
                    // std::cout << "Velocity: " 
                    //           << prev_velocity.transpose() << " -> " 
                    //           << next_velocity.transpose() << std::endl;
                }
            }
        }
    }

    if (!constant_pose && options_.use_prior_absolute_location && reconstruction->b_aligned) {
        // add prior absolute location constrain from gps prior
        if (reconstruction->prior_locations_gps.find(image_id) != reconstruction->prior_locations_gps.end()) {
            
            CHECK(reconstruction->prior_locations_gps_inlier.find(image_id) !=
                  reconstruction->prior_locations_gps_inlier.end());
            CHECK(reconstruction->prior_horizontal_locations_gps_inlier.find(image_id) !=
                  reconstruction->prior_horizontal_locations_gps_inlier.end());
            
            if (reconstruction->prior_locations_gps_inlier.at(image_id)) {
                Eigen::Vector3d prior_c = reconstruction->prior_locations_gps.at(image_id).first;
                
                ceres::CostFunction *cost_function = nullptr;
                if(options_.optimization_use_horizontal_gps_only){
                    cost_function = PriorAbsoluteLocationOnPlaneCostFunction::Create(
                        prior_c, reconstruction->projection_plane, options_.prior_absolute_location_weight);
                }
                else{
                    cost_function =
                        PriorAbsoluteLocationCostFunction::Create(prior_c, options_.prior_absolute_location_weight,
                            options_.prior_absolute_location_weight, options_.prior_absolute_location_weight);
                }
                problem_->AddResidualBlock(cost_function, loss_function, qvec_data, tvec_data);
            }
            else if (reconstruction->prior_horizontal_locations_gps_inlier.at(image_id)) {
                Eigen::Vector3d prior_c = reconstruction->prior_locations_gps.at(image_id).first;
                prior_c[2] = 0;

                ceres::CostFunction *cost_function = nullptr;
                cost_function = PriorAbsoluteLocationOnPlaneCostFunction::Create(
                    prior_c, reconstruction->projection_plane, options_.prior_absolute_location_weight);

                problem_->AddResidualBlock(cost_function, loss_function, qvec_data, tvec_data);
            }
        }

        // add prior absolute pose constrain from RTK prior
        if (image.HasQvecPrior() && image.HasTvecPrior() && image.RtkFlag() == 50) {

            Eigen::Vector4d prior_q = image.QvecPrior();
            // Eigen::Vector3d prior_c = -QuaternionToRotationMatrix(prior_q) * image.TvecPrior();
            Eigen::Vector3d prior_c = image.TvecPrior();

            ceres::ScaledLoss *rtk_loss_function = 
                    new ceres::ScaledLoss(new ceres::TrivialLoss(), 
                    static_cast<double>(std::max(num_observations, (size_t)50)), ceres::DO_NOT_TAKE_OWNERSHIP);
            
            // double weight_q = options_.prior_absolute_location_weight * 0.1;
            double weight_q = 0.0;
            double weight_x = image.HasRtkStd() ? options_.prior_absolute_location_weight / (image.RtkStdLat() + eps) : 
                                                    options_.prior_absolute_location_weight * 10.0;
            double weight_y = image.HasRtkStd() ? options_.prior_absolute_location_weight / (image.RtkStdLon() + eps) :
                                                    options_.prior_absolute_location_weight * 10.0;
            double weight_z = image.HasRtkStd() ? options_.prior_absolute_location_weight / (image.RtkStdHgt() + eps) :
                                                    options_.prior_absolute_location_weight * 10.0;

            if (image.PriorQvecGood()) {
                ceres::CostFunction *cost_function = PriorAbsolutePoseCostFunction::Create(
                    prior_q, prior_c, weight_q, weight_x, weight_y, weight_z);
                problem_->AddResidualBlock(cost_function, rtk_loss_function, qvec_data, tvec_data);
            } else {
                ceres::CostFunction *cost_function = PriorAbsoluteLocationCostFunction::Create(
                    image.TvecPrior(), weight_x, weight_y, weight_z);
                problem_->AddResidualBlock(cost_function, rtk_loss_function, qvec_data, tvec_data);
            }
            if (num_observations == 0) {
                // Set pose parameterization.
                if (!constant_pose) {
                    ceres::LocalParameterization *quaternion_parameterization = new ceres::QuaternionParameterization;
                    SetParameterization(problem_.get(), qvec_data, quaternion_parameterization);
                    if (config_.HasConstantTvec(image_id)) {
                        const std::vector<int> &constant_tvec_idxs = config_.ConstantTvec(image_id);
                        ceres::SubsetParameterization *tvec_parameterization =
                            new ceres::SubsetParameterization(3, constant_tvec_idxs);
                        SetParameterization(problem_.get(), tvec_data, tvec_parameterization);
                    }
                }
            }
        } else if (image.HasTvecPrior() && image.PriorInlier() && image.RtkFlag() != 50) {
            ceres::ScaledLoss *rtk_loss_function = 
                    new ceres::ScaledLoss(new ceres::TrivialLoss(), 
                    static_cast<double>(std::max(num_observations, (size_t)50)), ceres::DO_NOT_TAKE_OWNERSHIP);

            ceres::CostFunction *cost_function = PriorAbsoluteDistanceCostFunction::Create(
                image.TvecPrior(), options_.prior_absolute_location_weight);
            problem_->AddResidualBlock(cost_function, rtk_loss_function, qvec_data, tvec_data);
            if (num_observations == 0) {
                // Set pose parameterization.
                if (!constant_pose) {
                    ceres::LocalParameterization *quaternion_parameterization = new ceres::QuaternionParameterization;
                    SetParameterization(problem_.get(), qvec_data, quaternion_parameterization);
                    if (config_.HasConstantTvec(image_id)) {
                        const std::vector<int> &constant_tvec_idxs = config_.ConstantTvec(image_id);
                        ceres::SubsetParameterization *tvec_parameterization =
                            new ceres::SubsetParameterization(3, constant_tvec_idxs);
                        SetParameterization(problem_.get(), tvec_data, tvec_parameterization);
                    }
                }
            }
        }
        // if (image.HasTvecPrior()) {
        //     if (image.RtkFlag() == 50) {
                
        //         double weight_x = image.HasRtkStd() ? options_.prior_absolute_location_weight / (image.RtkStdLat() + eps) : 
        //                                               options_.prior_absolute_location_weight * 10.0;
        //         double weight_y = image.HasRtkStd() ? options_.prior_absolute_location_weight / (image.RtkStdLon() + eps) :
        //                                               options_.prior_absolute_location_weight * 10.0;
        //         double weight_z = image.HasRtkStd() ? options_.prior_absolute_location_weight / (image.RtkStdHgt() + eps) :
        //                                               options_.prior_absolute_location_weight * 10.0;

        //         ceres::ScaledLoss *rtk_loss_function = 
        //                 new ceres::ScaledLoss(new ceres::TrivialLoss(), 
        //                 static_cast<double>(num_observations), ceres::DO_NOT_TAKE_OWNERSHIP);

        //         ceres::CostFunction *cost_function = PriorAbsoluteLocationCostFunction::Create(
        //             image.TvecPrior(), weight_x, weight_y, weight_z);
        //         problem_->AddResidualBlock(cost_function, rtk_loss_function, qvec_data, tvec_data);
        //     } else if (image.PriorInlier()) {
        //         ceres::ScaledLoss *rtk_loss_function = 
        //                 new ceres::ScaledLoss(new ceres::TrivialLoss(), 
        //                 static_cast<double>(num_observations), ceres::DO_NOT_TAKE_OWNERSHIP);

        //         ceres::CostFunction *cost_function = PriorAbsoluteDistanceCostFunction::Create(
        //             image.TvecPrior(), options_.prior_absolute_location_weight);
        //         problem_->AddResidualBlock(cost_function, rtk_loss_function, qvec_data, tvec_data);
        //     }
        // }
    }

    if (constant_pose && (constant_pose_images_.count(image_id) == 0)) {
        constant_pose_images_.insert(image_id);
    }

    if (num_observations > 0) {
        camera_ids_.insert(image.CameraId());

        if (camera.NumLocalCameras() > 1) {
            if (local_camera_ids_.find(image.CameraId()) != local_camera_ids_.end()) {
                for (auto local_camera_id : local_camera_in_the_image) {
                    local_camera_ids_.at(image.CameraId()).insert(local_camera_id);
                }
            } else {
                local_camera_ids_.emplace(image.CameraId(), local_camera_in_the_image);
            }
        }

        // Set pose parameterization.
        if (!constant_pose) {
            ceres::LocalParameterization *quaternion_parameterization = new ceres::QuaternionParameterization;
            SetParameterization(problem_.get(), qvec_data, quaternion_parameterization);
            if (config_.HasConstantTvec(image_id)) {
                const std::vector<int> &constant_tvec_idxs = config_.ConstantTvec(image_id);
                ceres::SubsetParameterization *tvec_parameterization =
                    new ceres::SubsetParameterization(3, constant_tvec_idxs);
                SetParameterization(problem_.get(), tvec_data, tvec_parameterization);
            }
        }
    }
}

void BundleAdjuster::AddImageToStructOnlyProblem(const image_t image_id, Reconstruction *reconstruction,
                                       ceres::LossFunction *loss_function) {
    Image &image = reconstruction->Image(image_id);
    Camera &camera = reconstruction->Camera(image.CameraId());
    // CostFunction assumes unit quaternions.
    image.NormalizeQvec();
    
    std::vector<double> camera_params = camera.Params(); 

    const std::vector<uint32_t> &local_image_indices = image.LocalImageIndices();
    std::vector<Eigen::Vector4d> local_qvecs;
    std::vector<Eigen::Vector3d> local_tvecs;
    std::vector<double *> local_camera_params_data;

    if(camera.NumLocalCameras() > 1){
        int local_param_size = camera.LocalParams().size() / camera.NumLocalCameras();
        local_qvecs.resize(camera.NumLocalCameras());
        local_tvecs.resize(camera.NumLocalCameras());
        for(size_t i = 0; i < camera.NumLocalCameras(); ++i){
            camera.GetLocalCameraExtrinsic(i,local_qvecs[i],local_tvecs[i]);   
            local_camera_params_data.push_back(camera.LocalIntrinsicParamsData() + i * local_param_size);     
        }
    }

    // Add residuals to bundle adjustment problem.
    
    int point2d_id = -1;
    for (const Point2D &point2D : image.Points2D()) {
        point2d_id++;
        if (!point2D.HasMapPoint()) {
            continue;
        }

        uint32_t local_camera_id = local_image_indices[point2d_id];


        MapPoint &mappoint = reconstruction->MapPoint(point2D.MapPointId());
        CHECK(mappoint.Track().Length() > 1);

        ceres::CostFunction *cost_function = nullptr;

        if (camera.NumLocalCameras() > 1) {
            // for camera-rig
            Eigen::Vector4d qvec = ConcatenateQuaternions(image.Qvec(), local_qvecs[local_camera_id]);
            Eigen::Vector3d tvec = QuaternionRotatePoint(local_qvecs[local_camera_id], image.Tvec()) + local_tvecs[local_camera_id];
            int local_param_size = camera.LocalParams().size() / camera.NumLocalCameras();
            if (camera.ModelName().compare("OPENCV_FISHEYE") == 0) {
        
                double f = camera.LocalMeanFocalLength(local_camera_id);
                Eigen::Vector3d bearing = camera.LocalImageToBearing(local_camera_id, point2D.XY());

                cost_function = LargeFovStructBundleAdjustmentCostFunction<OpenCVFisheyeCameraModel>::Create(
                    qvec, tvec, bearing, f);
                problem_->AddResidualBlock(cost_function, loss_function, mappoint.XYZ().data());

            } else {
                switch (camera.ModelId()) {
#define CAMERA_MODEL_CASE(CameraModel)                                                              \
    case CameraModel::kModelId:                                                                     \
        cost_function = StructBundleAdjustmentCostFunction<CameraModel>::Create(                    \
            qvec, tvec, local_camera_params_data[local_camera_id], local_param_size, point2D.XY()); \
        break;
                    CAMERA_MODEL_SWITCH_CASES
#undef CAMERA_MODEL_CASE
                }

                problem_->AddResidualBlock(cost_function, loss_function, mappoint.XYZ().data());
            }

        } else {
            // for monocular camera
            if (camera.ModelName().compare("SPHERICAL") == 0) {
                Eigen::Vector3d bearing = camera.ImageToBearing(point2D.XY());
                double f = camera.FocalLength();
                Eigen::Vector4d qvec;
                Eigen::Vector3d tvec;
                Eigen::Vector4d qvec2;
                Eigen::Vector3d tvec2;

                for (size_t i = 0; i < 4; ++i) {
                    qvec2[i] = camera_params[i + 10];
                }
                for (size_t i = 0; i < 3; ++i) {
                    tvec2[i] = camera_params[i + 14];
                }
                if (bearing[2] < 0) {
                    qvec = ConcatenateQuaternions(image.Qvec(), qvec2);
                    tvec = QuaternionRotatePoint(qvec2,image.Tvec()) + tvec2;
                } else {
                    qvec = image.Qvec();
                    tvec = image.Tvec();
                }
                cost_function =
                    SphericalStructBundleAdjustmentCostFunction<SphericalCameraModel>::Create(qvec, tvec, bearing, f);
                problem_->AddResidualBlock(cost_function, loss_function, mappoint.XYZ().data());

            } else if (camera.ModelName().compare("OPENCV_FISHEYE") == 0) {
                Eigen::Vector3d bearing = camera.ImageToBearing(point2D.XY());
                double f = camera.MeanFocalLength();
                cost_function = LargeFovStructBundleAdjustmentCostFunction<OpenCVFisheyeCameraModel>::Create(
                    image.Qvec(), image.Tvec(), bearing, f);

                problem_->AddResidualBlock(cost_function, loss_function, mappoint.XYZ().data());
            } else {
                switch (camera.ModelId()) {
#define CAMERA_MODEL_CASE(CameraModel)                                                             \
    case CameraModel::kModelId:                                                                    \
        cost_function = StructBundleAdjustmentCostFunction<CameraModel>::Create(                   \
            image.Qvec(), image.Tvec(), camera_params.data(), camera_params.size(), point2D.XY()); \
        break;
                    CAMERA_MODEL_SWITCH_CASES
#undef CAMERA_MODEL_CASE
                }
                problem_->AddResidualBlock(cost_function, loss_function, mappoint.XYZ().data());
            }
        }
    }  
}

void BundleAdjuster::AddPointToStructOnlyProblem(std::unique_ptr<ceres::Problem>& problem, const mappoint_t mappoint_id,
                                                 Reconstruction *reconstruction, ceres::LossFunction *loss_function) {
    MapPoint &mappoint = reconstruction->MapPoint(mappoint_id);

    for (const auto &track_el : mappoint.Track().Elements()) {
        Image &image = reconstruction->Image(track_el.image_id);
        Camera &camera = reconstruction->Camera(image.CameraId());
        const Point2D &point2D = image.Point2D(track_el.point2D_idx);

        std::vector<double> camera_params = camera.Params();

        const std::vector<uint32_t> &local_image_indices = image.LocalImageIndices();
        std::vector<Eigen::Vector4d> local_qvecs;
        std::vector<Eigen::Vector3d> local_tvecs;
        std::vector<double *> local_camera_params_data;

        if (camera.NumLocalCameras() > 1) {
            int local_param_size = camera.LocalParams().size() / camera.NumLocalCameras();
            local_qvecs.resize(camera.NumLocalCameras());
            local_tvecs.resize(camera.NumLocalCameras());
            for (size_t i = 0; i < camera.NumLocalCameras(); ++i) {
                camera.GetLocalCameraExtrinsic(i, local_qvecs[i], local_tvecs[i]);
                local_camera_params_data.push_back(camera.LocalIntrinsicParamsData() + i * local_param_size);
            }
        }

        local_camera_t local_camera_id = local_image_indices[track_el.point2D_idx];

        ceres::CostFunction *cost_function = nullptr;

        if (camera.NumLocalCameras() > 1) {
            // for camera-rig
            Eigen::Vector4d qvec = ConcatenateQuaternions(image.Qvec(), local_qvecs[local_camera_id]);
            Eigen::Vector3d tvec =
                QuaternionRotatePoint(local_qvecs[local_camera_id], image.Tvec()) + local_tvecs[local_camera_id];
            int local_param_size = camera.LocalParams().size() / camera.NumLocalCameras();
            if (camera.ModelName().compare("OPENCV_FISHEYE") == 0) {
                double f = camera.LocalMeanFocalLength(local_camera_id);
                Eigen::Vector3d bearing = camera.LocalImageToBearing(local_camera_id, point2D.XY());

                cost_function = LargeFovStructBundleAdjustmentCostFunction<OpenCVFisheyeCameraModel>::Create(
                    qvec, tvec, bearing, f);
                problem->AddResidualBlock(cost_function, loss_function, mappoint.XYZ().data());

            } else {
                switch (camera.ModelId()) {
#define CAMERA_MODEL_CASE(CameraModel)                                                              \
    case CameraModel::kModelId:                                                                     \
        cost_function = StructBundleAdjustmentCostFunction<CameraModel>::Create(                    \
            qvec, tvec, local_camera_params_data[local_camera_id], local_param_size, point2D.XY()); \
        break;
                    CAMERA_MODEL_SWITCH_CASES
#undef CAMERA_MODEL_CASE
                }

                problem->AddResidualBlock(cost_function, loss_function, mappoint.XYZ().data());
            }

        } else {
            // for monocular camera
            if (camera.ModelName().compare("SPHERICAL") == 0) {
                Eigen::Vector3d bearing = camera.ImageToBearing(point2D.XY());
                double f = camera.FocalLength();
                Eigen::Vector4d qvec;
                Eigen::Vector3d tvec;
                Eigen::Vector4d qvec2;
                Eigen::Vector3d tvec2;

                for (size_t i = 0; i < 4; ++i) {
                    qvec2[i] = camera_params[i + 10];
                }
                for (size_t i = 0; i < 3; ++i) {
                    tvec2[i] = camera_params[i + 14];
                }
                if (bearing[2] < 0) {
                    qvec = ConcatenateQuaternions(image.Qvec(), qvec2);
                    tvec = QuaternionRotatePoint(qvec2, image.Tvec()) + tvec2;
                } else {
                    qvec = image.Qvec();
                    tvec = image.Tvec();
                }
                cost_function =
                    SphericalStructBundleAdjustmentCostFunction<SphericalCameraModel>::Create(qvec, tvec, bearing, f);
                problem->AddResidualBlock(cost_function, loss_function, mappoint.XYZ().data());

            } else if (camera.ModelName().compare("OPENCV_FISHEYE") == 0) {
                Eigen::Vector3d bearing = camera.ImageToBearing(point2D.XY());
                double f = camera.MeanFocalLength();
                cost_function = LargeFovStructBundleAdjustmentCostFunction<OpenCVFisheyeCameraModel>::Create(
                    image.Qvec(), image.Tvec(), bearing, f);

                problem->AddResidualBlock(cost_function, loss_function, mappoint.XYZ().data());
            } else {
                switch (camera.ModelId()) {
#define CAMERA_MODEL_CASE(CameraModel)                                                             \
    case CameraModel::kModelId:                                                                    \
        cost_function = StructBundleAdjustmentCostFunction<CameraModel>::Create(                   \
            image.Qvec(), image.Tvec(), camera_params.data(), camera_params.size(), point2D.XY()); \
        break;
                    CAMERA_MODEL_SWITCH_CASES
#undef CAMERA_MODEL_CASE
                }
                problem->AddResidualBlock(cost_function, loss_function, mappoint.XYZ().data());
            }
        }
    }
}

void BundleAdjuster::AddPointToProblem(const mappoint_t mappoint_id, Reconstruction *reconstruction,
                                       ceres::LossFunction *loss_function) {
    MapPoint &mappoint = reconstruction->MapPoint(mappoint_id);
    uint64_t mappoint_create_time = mappoint.CreateTime();

    // Is Map Point already fully contained in the problem? I.e. its entire track
    // is contained in `variable_image_ids`, `constant_image_ids`,
    // `constant_x_image_ids`.

    const bool constant_intrinsics =
        !options_.refine_focal_length && !options_.refine_extra_params && !options_.refine_principal_point;

    if (mappoint_num_observations_[mappoint_id] == mappoint.Track().Length()) {
        return;
    }

    for (const auto &track_el : mappoint.Track().Elements()) {
        // Skip observations that were already added in `FillImages`.
        if (config_.HasImage(track_el.image_id)) {
            continue;
        }

        mappoint_num_observations_[mappoint_id] += 1;

        Image &image = reconstruction->Image(track_el.image_id);
        Camera &camera = reconstruction->Camera(image.CameraId());
        const Point2D &point2D = image.Point2D(track_el.point2D_idx);

        Eigen::Matrix3d R = image.RotationMatrix();
        Eigen::Vector3d t = image.Tvec();

        Eigen::Vector3d point3D_mea;
        const float info = GetDepthPointAndWeight(camera, point2D, point3D_mea);

        double loop_weight = 1.0;
        if (mappoint_create_time != 0 && image.create_time_ > 0) {
            int time_diff = std::abs((int)image.create_time_ - (int)mappoint_create_time);
            if (time_diff > 200) loop_weight = 20.0;
            else if (time_diff > 100) {
                loop_weight = 0.19 * time_diff - 18;
            }
            // if (time_diff > 200) loop_weight = 40.0;
            // else {
            //     loop_weight = std::max(1.0, time_diff / 5.0);
            // }
        }

        int local_image_id = image.LocalImageIndices()[track_el.point2D_idx];
        double *local_qvec_data;
        double *local_tvec_data;
        double *local_camera_params_data;

        double *camera_params_data = camera.ParamsData();

        // We do not want to refine the camera of images that are not
        // part of `constant_image_ids_`, `constant_image_ids_`,
        // `constant_x_image_ids_`.
        if (camera_ids_.count(image.CameraId()) == 0) {
            camera_ids_.insert(image.CameraId());
            config_.SetConstantCamera(image.CameraId());
        }

        ceres::CostFunction *cost_function = nullptr;

        if (camera.NumLocalCameras() > 1) {
            // for camera-rig
            double observation_factor = 1.0;
            if (local_camera_ids_.find(image.CameraId()) != local_camera_ids_.end()) {
                local_camera_ids_.at(image.CameraId()).insert(local_image_id);
            } else {
                std::unordered_set<int> local_image_id_set;
                local_image_id_set.insert(local_image_id);
                local_camera_ids_.emplace(image.CameraId(), local_image_id_set);
            }

            int local_param_size = camera.LocalParams().size() / camera.NumLocalCameras();
            local_qvec_data = camera.LocalQvecsData() + 4 * local_image_id;
            local_tvec_data = camera.LocalTvecsData() + 3 * local_image_id;
            local_camera_params_data = camera.LocalIntrinsicParamsData() + local_image_id * local_param_size;

            if (point2D.InOverlap()) {
                Eigen::Vector4d local_qvec(local_qvec_data);
                Eigen::Vector3d local_tvec(local_tvec_data);
                Eigen::Matrix3d local_R = QuaternionToRotationMatrix(local_qvec);
                Eigen::Vector3d Xc = local_R *(R * mappoint.XYZ() + t) + local_tvec;
                observation_factor = 1.0 + 9 * std::exp(-Xc[2] * Xc[2] / 10.0);
                // if (Xc[2] < 1.5) {
                //     observation_factor = 10.0f;
                // }
            }

            if (camera.ModelName().compare("OPENCV_FISHEYE") == 0 && constant_intrinsics) {
                double f = camera.LocalMeanFocalLength(local_image_id);
                Eigen::Vector3d bearing = camera.LocalImageToBearing(local_image_id,point2D.XY());
                cost_function = LargeFovRigBundleAdjustmentConstantPoseCostFunction<OpenCVFisheyeCameraModel>::Create(
                    image.Qvec(), image.Tvec(), bearing, f * loop_weight);

                problem_->AddResidualBlock(cost_function, loss_function, local_qvec_data, local_tvec_data,
                                           mappoint.XYZ().data());

            } else {
                switch (camera.ModelId()) {
#define CAMERA_MODEL_CASE(CameraModel)                                                    \
    case CameraModel::kModelId:                                                           \
        cost_function = RigBundleAdjustmentConstantPoseCostFunction<CameraModel>::Create( \
            image.Qvec(), image.Tvec(), point2D.XY(), observation_factor * loop_weight);                \
        break;
                    CAMERA_MODEL_SWITCH_CASES
#undef CAMERA_MODEL_CASE
                }

                problem_->AddResidualBlock(cost_function, loss_function, local_qvec_data, local_tvec_data,
                                           mappoint.XYZ().data(), local_camera_params_data);
            }

        } else {
            // for monocular camera
            if (camera.ModelName().compare("SPHERICAL") == 0) {
                Eigen::Vector3d bearing = camera.ImageToBearing(point2D.XY());
                double f = camera.FocalLength();

                cost_function = SphericalBundleAdjustmentConstantPoseCostFunction<SphericalCameraModel>::Create(
                    image.Qvec(), image.Tvec(), bearing, f, loop_weight);

                double *local_qvec2_data = camera_params_data + 10;
                double *local_tvec2_data = camera_params_data + 14;

                problem_->AddResidualBlock(cost_function, loss_function, mappoint.XYZ().data(),
                                           local_qvec2_data, local_tvec2_data);
            } else if (camera.ModelName().compare("OPENCV_FISHEYE") == 0 && constant_intrinsics){
                Eigen::Vector3d bearing = camera.ImageToBearing(point2D.XY());
                double f = camera.MeanFocalLength();
                cost_function = LargeFovBundleAdjustmentConstantPoseCostFunction<OpenCVFisheyeCameraModel>::Create(
                    image.Qvec(), image.Tvec(), bearing, f, loop_weight);

                problem_->AddResidualBlock(cost_function, loss_function, mappoint.XYZ().data());
            } else {
                switch (camera.ModelId()) {
#define CAMERA_MODEL_CASE(CameraModel)                                                                               \
    case CameraModel::kModelId:                                                                                      \
        cost_function =                                                                                              \
            BundleAdjustmentConstantPoseCostFunction<CameraModel>::Create(image.Qvec(), image.Tvec(), point2D.XY(), loop_weight); \
        break;
                    CAMERA_MODEL_SWITCH_CASES
#undef CAMERA_MODEL_CASE
                }
                problem_->AddResidualBlock(cost_function, loss_function, mappoint.XYZ().data(), camera.ParamsData());
                if (point3D_mea.z() > 0 && reconstruction->depth_enabled) {
                    cost_function =
                        BundleAdjustmentConstantPoseDepthCostFunction::Create(
                            image.Qvec(), image.Tvec(), point3D_mea, options_.rgbd_ba_depth_weight * info * loop_weight
                        );
                    problem_->AddResidualBlock(cost_function, loss_function, 
                                                mappoint.XYZ().data());
                }
            }
        }
    }
}

void BundleAdjuster::AddNovatelToProblem(const image_t image_id, Reconstruction *reconstruction,
                                       ceres::LossFunction *loss_function) {
    Image &image = reconstruction->Image(image_id);
    Camera &camera = reconstruction->Camera(image.CameraId());
    // CostFunction assumes unit quaternions.
    // image.NormalizeQvec();
    image.NormalizePriorQvec();

    Eigen::Vector4d prior_qvec = image.QvecPrior();
    Eigen::Vector3d prior_tvec = QuaternionToRotationMatrix(prior_qvec) * -image.TvecPrior();

    if (!camera.HasDisturb()){
        return;
    }
    // std::cout << "AddNovatelToProblem" << image.ImageId() << std::endl;
    double *qvec_data = camera.QvecDisturb().data();
    double *tvec_data = camera.TvecDisturb().data();
    double *camera_params_data = camera.ParamsData();
    // std::cout << "AddNovatelToProblem: " << camera.QvecDisturb() << "\n " << camera.TvecDisturb() << std::endl;

    const bool constant_pose = !options_.refine_extrinsics || config_.HasConstantPose(image_id);

    double weight = 1.0;

    // Add residuals to bundle adjustment problem.
    size_t num_observations = 0;
    int point2d_id = -1;
    for (const Point2D &point2D : image.Points2D()) {
        point2d_id++;
        if (!point2D.HasMapPoint()) {
            continue;
        }

        num_observations += 1;
        mappoint_num_observations_[point2D.MapPointId()] += 1;

        MapPoint &mappoint = reconstruction->MapPoint(point2D.MapPointId());
        assert(mappoint.Track().Length() > 1);

        ceres::CostFunction *cost_function = nullptr;
        switch (camera.ModelId()) {
#define CAMERA_MODEL_CASE(CameraModel)                                                           \
case CameraModel::kModelId:                                                                  \
cost_function = BundleAdjustmentNovatelCostFunction<CameraModel>::Create(prior_qvec, prior_tvec, point2D.XY(), weight); \
break;
            CAMERA_MODEL_SWITCH_CASES
#undef CAMERA_MODEL_CASE
        }
        problem_->AddResidualBlock(cost_function, loss_function, qvec_data, tvec_data,
                                    mappoint.XYZ().data(), camera_params_data);
    }
 
    if (num_observations > 0) {
        camera_ids_.insert(image.CameraId());
        gnss_camera_ids_.insert(image.CameraId());

        double weight_q = options_.prior_gnss2camera_extri_weight * 0.1;
        double weight_x = options_.prior_gnss2camera_extri_weight;
        double weight_y = options_.prior_gnss2camera_extri_weight * 5;
        double weight_z = options_.prior_gnss2camera_extri_weight ;

        double *qvec_data = camera.QvecDisturb().data();
        double *tvec_data = camera.TvecDisturb().data();

        Eigen::Vector4d prior_qvec = camera.QvecPriorDisturb();
        Eigen::Vector3d prior_tvec = camera.TvecPriorDisturb();
        Eigen::Vector3d prior_c = ProjectionCenterFromPose(prior_qvec, prior_tvec);
    
        ceres::ScaledLoss *gnss_loss_function = 
                new ceres::ScaledLoss(new ceres::TrivialLoss(), 
                static_cast<double>(num_observations / 3 + 1), ceres::DO_NOT_TAKE_OWNERSHIP);
            
        ceres::CostFunction *cost_function =
                PriorAbsolutePoseCostFunction::Create(prior_qvec, prior_c, weight_q,
                weight_x, weight_y, weight_z);
        problem_->AddResidualBlock(cost_function, gnss_loss_function, qvec_data, tvec_data);
    }
}

void BundleAdjuster::AddLidarToProblem(const sweep_t sweep_id, const image_t image_id, const uint32_t m_num_visible_point,
                                       Reconstruction* reconstruction, ceres::LossFunction* loss_function) {
    const uint32_t num_reg_images = reconstruction->NumRegisterImages();
    class LidarSweep & lidar_sweep = reconstruction->LidarSweep(sweep_id);
    Eigen::Matrix3x4d inv_proj_matrix = lidar_sweep.InverseProjectionMatrix();
    std::shared_ptr<class VoxelMap> & voxel_map = reconstruction->VoxelMap();
    const class Image & image = reconstruction->Image(image_id);
    if (!image.IsRegistered()) {
        return;
    }

    const bool has_constant_pose = config_.HasConstantSweep(sweep_id);

    double * qvec_data = lidar_sweep.Qvec().data();
    double * tvec_data = lidar_sweep.Tvec().data();

    LidarPointCloud pc;
    LidarPointCloud ref_less_surfs = lidar_sweep.GetSurfPointsLessFlat();
    LidarPointCloud ref_corners = lidar_sweep.GetCornerPointsLessSharp();
    const size_t num_surf = ref_less_surfs.points.size();
    const size_t num_corner = ref_corners.points.size();
    pc = std::move(ref_less_surfs);
    pc += std::move(ref_corners);

    const double factor = (double)image.NumVisibleMapPoints() / (double)m_num_visible_point;
    // const double factor = (double)image.NumVisibleMapPoints() / (double)pc.points.size();

    size_t num_surf_observation = 0, num_corner_observation = 0;
    for (size_t i = 0; i < pc.points.size(); ++i) {
        Eigen::Vector3d point(pc.points[i].x, pc.points[i].y, pc.points[i].z);
        Eigen::Vector3d query = inv_proj_matrix * point.homogeneous();
        
        lidar::OctoTree::Point p;
        p.x = query[0];
        p.y = query[1];
        p.z = query[2];

        lidar::OctoTree * loc = voxel_map->LocateOctree(p, 3);
        if (!loc) continue;

        Voxel * voxel = loc->voxel_;
        if (voxel->IsScatter()) continue;

        if (!voxel->IsDetermined()) {
            voxel->ComputeFeature();
        }

        if (!voxel->IsDetermined()) continue;

        double weight = 1.0;
        // if (voxel->IsFeature()) {
        //     if (loc->lifetime_ != 0) {
        //         // double time_diff = std::abs((long long)(pc.points[i].lifetime - loc->lifetime_)) * 1.0 / (double)1e9;
        //         // weight = std::clamp((std::sqrt(1 + time_diff * 0.2) - 1), 1.0, 40.0);
        //         int time_diff = std::abs((int)pc.points[i].lifetime - (int)loc->lifetime_);
        //         // if (time_diff > 200) weight = 40.0;
        //         // else {
        //         //     weight = std::max(1.0, time_diff / 5.0);
        //         // }
        //         if (time_diff > 200) weight = 20.0;
        //         else if (time_diff > 100) {
        //             weight = 0.19 * time_diff - 18;
        //         }
        //     }
        // }
        // int time_diff = std::abs((int)pc.points[i].lifetime - (int)loc->lastest_time_);
        // if (time_diff > 200) weight = 5.0;
        // else if (time_diff > 100) {
        //     weight = 0.04 * time_diff - 3;
        // }
        double time_diff = std::abs((double)pc.points[i].lifetime - (double)loc->create_time_) / (double)1e9;
        // weight = 20.0 - 10.0 / (1 + 19.0 * std::exp(-0.2 * time_diff));
        weight = 40.0 - 20.0 / (1 + std::exp(-0.1 * time_diff));

        double exp_weight = 1.0;//1.0 - std::exp(-pc.points[i].intensity * pc.points[i].intensity / 30);
        double regu_weight = factor * weight * exp_weight;

        Eigen::Vector3d m_var = voxel->GetEx();
        Eigen::Vector3d m_pivot = voxel->GetPivot();
        Eigen::Matrix3d m_inv_cov = voxel->GetInvCov();

        ceres::CostFunction* cost_function = LidarAbsolutePoseCostFunction::Create(point, m_var, m_pivot, m_inv_cov, regu_weight);
        problem_->AddResidualBlock(cost_function, loss_function, qvec_data, tvec_data);
        num_surf_observation++;
    }
    // std::cout << "Sweep#" << sweep_id << std::endl;
    // std::cout << "num_surf_observation: " << num_surf_observation << std::endl;
    // std::cout << "num_corner_observation: " << num_corner_observation << std::endl;

    // add plane constrain
    bool on_plane = reconstruction->planes_for_lidars.find(sweep_id) != reconstruction->planes_for_lidars.end();
    if (!has_constant_pose && options_.plane_constrain && on_plane) {
        Eigen::Vector4d plane = reconstruction->planes_for_lidars.at(sweep_id);
        ceres::CostFunction *cost_function = PlaneConstrainCostFunction::Create(
            plane, reconstruction->baseline_distance, num_surf_observation, options_.plane_weight);

        problem_->AddResidualBlock(cost_function, loss_function, qvec_data, tvec_data);
    }

    if (num_surf_observation > 0) {
        // Quaternion parameterization.
        ceres::LocalParameterization* quaternion_parameterization = new ceres::QuaternionParameterization;
        SetParameterization(problem_.get(), qvec_data, quaternion_parameterization);
        if (has_constant_pose) {
            problem_->SetParameterBlockConstant(qvec_data);
            problem_->SetParameterBlockConstant(tvec_data);
        }
    }
}

void BundleAdjuster::AddMapPointToVoxelMap(Reconstruction* reconstruction, ceres::LossFunction* loss_function) {
    // Collect all mappoints.
    std::unordered_set<mappoint_t> mappoint_ids;
    for (const image_t image_id : config_.Images()) {
        class Image & image = reconstruction->Image(image_id);
        for (point2D_t point2D_id = 0; point2D_id < image.NumPoints2D(); ++point2D_id) {
            class Point2D & point2D = image.Point2D(point2D_id);
            if (point2D.HasMapPoint()) {
                mappoint_ids.insert(point2D.MapPointId());
            }
        }
    }
    for (const auto mappoint_id : config_.VariablePoints()) {
        mappoint_ids.insert(mappoint_id);
    }

    std::shared_ptr<class VoxelMap> & voxel_map = reconstruction->VoxelMap();

    std::vector<mappoint_t> unique_mappoint_ids;
    unique_mappoint_ids.insert(unique_mappoint_ids.end(), mappoint_ids.begin(), mappoint_ids.end());

    std::vector<lidar::OctoTree*> locs(unique_mappoint_ids.size());
    for (size_t i = 0; i < unique_mappoint_ids.size(); ++i) {
        mappoint_t mappoint_id = unique_mappoint_ids[i];
        Eigen::Vector3d XYZ = reconstruction->MapPoint(mappoint_id).XYZ();
        lidar::OctoTree::Point point;
        point.x = XYZ[0];
        point.y = XYZ[1];
        point.z = XYZ[2];
        point.type = 0;
        locs[i] = voxel_map->LocateOctree(point, 3);
    }

    for (size_t i = 0; i < locs.size(); ++i) {
        if (!locs[i]) continue;
        Voxel * voxel = locs[i]->voxel_;
        if (!voxel->IsFeature()) continue;

        mappoint_t mappoint_id = unique_mappoint_ids[i];
        class MapPoint& mappoint = reconstruction->MapPoint(mappoint_id);

        Eigen::Vector3d m_var = voxel->GetEx();
        Eigen::Vector3d m_pivot = voxel->GetPivot();
        Eigen::Matrix3d m_inv_cov = voxel->GetInvCov();

        Eigen::Vector3d ray = mappoint.XYZ() - m_var;
        double proj_len = ray.dot(m_pivot);
        Eigen::Vector3d residual_vec = m_pivot * proj_len;
        double y = residual_vec.transpose() * m_inv_cov * residual_vec;
        if (y > 2) {
            // double error = voxel->Error(mappoint.XYZ());
            // std::cout << "filter: " << error << std::endl;
            continue;
        }
        // // double error = voxel->Error(mappoint.XYZ());
        // // std::cout << "error: " << error << std::endl;
        ceres::CostFunction* cost_function = LidarAbsoluteDistanceCostFunction::Create(m_var, m_pivot, m_inv_cov, options_.lidar_weight);
        // ceres::CostFunction* cost_function = LidarAbsoluteDistanceCostFunction::Create(m_var, m_pivot, options_.lidar_weight);
        problem_->AddResidualBlock(cost_function, loss_function, mappoint.XYZ().data());
    }
}

void BundleAdjuster::AddSfMToLidarConstraint(Reconstruction* reconstruction) {
    const auto & sweep_image_pairs = config_.SweepImagePairs();
    if (sweep_image_pairs.size() > 3) {
        std::unordered_set<mappoint_t> mappoint_ids;
        for (auto sweep_image : sweep_image_pairs) {
            image_t image_id = sweep_image.first;
            sweep_t sweep_id = sweep_image.second;
            class Image & image = reconstruction->Image(image_id);
            if (!image.IsRegistered()) {
                continue;
            }
            for (size_t point2d_idx = 0; point2d_idx < image.NumPoints2D(); ++point2d_idx) {
                class Point2D point2d = image.Point2D(point2d_idx);
                if (!point2d.HasMapPoint()) {
                    continue;
                }
                mappoint_t mappoint_id = point2d.MapPointId();
                mappoint_ids.insert(mappoint_id);
            }
        }
        std::unordered_map<image_t, std::unordered_map<image_t, int> > shared_points;
        for (auto mappoint_id : mappoint_ids) {
            class MapPoint & mappoint = reconstruction->MapPoint(mappoint_id);
            class Track & track = mappoint.Track();
            for (size_t i = 0; i < track.Length(); ++i) { 
                const int image_idx1 = track.Element(i).image_id;
                for (size_t j = 0; j < i; ++j) {
                    const int image_idx2 = track.Element(i).image_id;
                    if (image_idx1 != image_idx2) {
                        shared_points[image_idx1][image_idx2]++;
                        shared_points[image_idx2][image_idx1]++;
                    }
                }
            }
        }

        std::vector<std::pair<sweep_t, long long> > sweep_timestamps;
        sweep_timestamps.clear();
        const auto & sweep_ids = reconstruction->RegisterSweepIds();
        for (const auto & sweep_id :  sweep_ids) {
            sweep_timestamps.emplace_back(sweep_id, reconstruction->LidarSweep(sweep_id).timestamp_);
        }
        std::sort(sweep_timestamps.begin(), sweep_timestamps.end(), 
            [&](const std::pair<sweep_t, long long> & a, const std::pair<sweep_t, long long> & b){
                return a.second < b.second;
            });

        const double weight = 1.0;//options_.lidar_weight;

        Eigen::Matrix4d h_lidar_to_cam = Eigen::Matrix4d::Identity();
        h_lidar_to_cam.topRows(3) = config_.lidar_to_cam_matrix_;
        Eigen::Matrix4d h_cam_to_lidar = h_lidar_to_cam.inverse();
        Eigen::Vector4d cam_to_lidar_qvec = RotationMatrixToQuaternion(h_cam_to_lidar.block<3, 3>(0, 0));
        Eigen::Vector3d cam_to_lidar_tvec = h_cam_to_lidar.block<3, 1>(0, 3);

        Eigen::Vector4d l2c_qvec = RotationMatrixToQuaternion(config_.lidar_to_cam_matrix_.block<3, 3>(0, 0));
        Eigen::Vector3d l2c_tvec = config_.lidar_to_cam_matrix_.block<3, 1>(0, 3);

        point2D_t m_visible_map_per_image = 0;
        const std::vector<image_t> register_image_ids = reconstruction->RegisterImageIds();
        for (auto image_id : register_image_ids) {
            auto image = reconstruction->Image(image_id);
            m_visible_map_per_image += image.NumVisibleMapPoints();
        }
        m_visible_map_per_image /= (double)register_image_ids.size();
        std::cout << "m_visible_map_per_image: " << m_visible_map_per_image << std::endl;

        for (auto sweep_image : sweep_image_pairs) {
            image_t image_id = sweep_image.first;
            sweep_t sweep_id = sweep_image.second;

            const bool has_constant_lidar_pose = config_.HasConstantSweep(sweep_id);
            const bool has_constant_image_pose = config_.HasConstantPose(image_id);

            class Image & image = reconstruction->Image(image_id);
            if (!image.IsRegistered()) {
                continue;
            }
            double* qvec_data = image.Qvec().data();
            double* tvec_data = image.Tvec().data();

            Eigen::Vector3d C = image.ProjectionCenter();

            class LidarSweep & lidar_sweep = reconstruction->LidarSweep(sweep_id);
            if (!lidar_sweep.IsRegistered()) {
                continue;
            }

            // {
            //     // Eigen::Matrix4d lP = Eigen::Matrix4d::Identity();
            //     // lP.topRows(3) = image.ProjectionMatrix();
            //     // Eigen::Matrix3x4d lP_t = (h_cam_to_lidar * lP).topRows(3);

            //     // Eigen::Matrix3d R = lP_t.block<3, 3>(0, 0);
            //     // Eigen::Vector4d prior_q = RotationMatrixToQuaternion(R);
            //     // Eigen::Vector3d prior_c = -R.transpose() * lP_t.block<3, 1>(0, 3);

            //     ceres::ScaledLoss *loss_function = new ceres::ScaledLoss(new ceres::TrivialLoss(), image.NumMapPoints(), ceres::DO_NOT_TAKE_OWNERSHIP);
            //     ceres::CostFunction *cost_function = LidarCameraConstPoseCostFunction::Create(
            //         cam_to_lidar_qvec, cam_to_lidar_tvec, image.Qvec(), image.Tvec(), weight * 1.0, weight * 5.0, weight * 5.0, weight * 5.0);
            //     problem_->AddResidualBlock(cost_function, loss_function, lidar_sweep.Qvec().data(), lidar_sweep.Tvec().data());

            //     // ceres::ScaledLoss *loss_function = new ceres::ScaledLoss(new ceres::TrivialLoss(), image.NumMapPoints(), ceres::DO_NOT_TAKE_OWNERSHIP);
            //     // ceres::CostFunction *cost_function = PriorAbsolutePoseCostFunction::Create(prior_q, prior_c, weight * 1.0, weight * 5.0, weight * 5.0, weight * 5.0);
            //     // problem_->AddResidualBlock(cost_function, loss_function, lidar_sweep.Qvec().data(), lidar_sweep.Tvec().data());
                
            //     ceres::LocalParameterization *quaternion_parameterization = new ceres::QuaternionParameterization;
            //     SetParameterization(problem_.get(), lidar_sweep.Qvec().data(), quaternion_parameterization);
            //     if (has_constant_lidar_pose) {
            //         problem_->SetParameterBlockConstant(lidar_sweep.Qvec().data());
            //         problem_->SetParameterBlockConstant(lidar_sweep.Tvec().data());
            //     }
            // }

            uint64_t time_diff = lidar_sweep.timestamp_ - image.timestamp_;
            if (false && (double)time_diff / 1e9 < 0.1 && sweep_image_pairs.size() > 100) {
                // ceres::ScaledLoss *loss_function = new ceres::ScaledLoss(new ceres::TrivialLoss(), image.NumMapPoints(), ceres::DO_NOT_TAKE_OWNERSHIP);
                ceres::ScaledLoss *loss_function = new ceres::ScaledLoss(new ceres::TrivialLoss(), m_visible_map_per_image, ceres::DO_NOT_TAKE_OWNERSHIP);
                ceres::CostFunction *cost_function = LidarCameraPoseCostFunction::Create(l2c_qvec, l2c_tvec, 5.0, 100.0, 100.0, 100.0);
                problem_->AddResidualBlock(cost_function, loss_function, lidar_sweep.Qvec().data(), lidar_sweep.Tvec().data(),
                                            image.Qvec().data(), image.Tvec().data());
                
                // ceres::LocalParameterization *quaternion_parameterization = new ceres::QuaternionParameterization;
                // SetParameterization(problem_.get(), lidar_sweep.Qvec().data(), quaternion_parameterization);
                if (has_constant_lidar_pose) {
                    problem_->SetParameterBlockConstant(lidar_sweep.Qvec().data());
                    problem_->SetParameterBlockConstant(lidar_sweep.Tvec().data());
                }
                // SetParameterization(problem_.get(), image.Qvec().data(), quaternion_parameterization);
                if (has_constant_image_pose) {
                    problem_->SetParameterBlockConstant(image.Qvec().data());
                    problem_->SetParameterBlockConstant(image.Tvec().data());
                }
                // ceres::ScaledLoss *loss_function = new ceres::ScaledLoss(new ceres::TrivialLoss(), m_visible_map_per_image, ceres::DO_NOT_TAKE_OWNERSHIP);
                // ceres::CostFunction *cost_function = LidarCameraConstPoseCostFunction::Create(
                //     // cam_to_lidar_qvec, cam_to_lidar_tvec, image.Qvec(), image.Tvec(), weight * 1.0, weight * 5.0, weight * 5.0, weight * 5.0);
                //     cam_to_lidar_qvec, cam_to_lidar_tvec, image.Qvec(), image.Tvec(), weight * 5.0, weight * 100.0, weight * 100.0, weight * 100.0);
                // problem_->AddResidualBlock(cost_function, loss_function, lidar_sweep.Qvec().data(), lidar_sweep.Tvec().data());
                // ceres::LocalParameterization *quaternion_parameterization = new ceres::QuaternionParameterization;
                // SetParameterization(problem_.get(), lidar_sweep.Qvec().data(), quaternion_parameterization);
                // if (has_constant_lidar_pose) {
                //     problem_->SetParameterBlockConstant(lidar_sweep.Qvec().data());
                //     problem_->SetParameterBlockConstant(lidar_sweep.Tvec().data());
                // }
            } else {
                Eigen::Matrix4d lP = Eigen::Matrix4d::Identity();
                lP.topRows(3) = lidar_sweep.ProjectionMatrix();
                Eigen::Matrix3x4d lP_t = config_.lidar_to_cam_matrix_ * lP;

                Eigen::Matrix3d R = lP_t.block<3, 3>(0, 0);
                Eigen::Vector4d prior_q = RotationMatrixToQuaternion(R);
                Eigen::Vector3d prior_c = -R.transpose() * lP_t.block<3, 1>(0, 3);

                // // Interpolate Pose.
                // {
                //     long long image_timestamp = image.timestamp_;
                //     long long lidar_timestamp = lidar_sweep.timestamp_;
                //     if (lidar_timestamp > image_timestamp) { // forward interpolate
                //         class LidarSweep* before_lidar_sweep = NULL;
                //         for (int k = sweep_timestamps.size() - 1; k >= 0; --k) {
                //             auto & near_lidar_sweep = reconstruction->LidarSweep(sweep_timestamps[k].first);
                //             if (!near_lidar_sweep.IsRegistered() || near_lidar_sweep.timestamp_ > image_timestamp) {
                //                 continue;
                //             }
                //             if (near_lidar_sweep.timestamp_ <= image_timestamp) {
                //                 if (lidar_timestamp - near_lidar_sweep.timestamp_ < 5 * 1e9) { // 5s
                //                     before_lidar_sweep = &near_lidar_sweep;
                //                     break;
                //                 }
                //             }
                //         }
                //         if (before_lidar_sweep) {
                //             double t = (image_timestamp - before_lidar_sweep->timestamp_) * 1.0 / (lidar_timestamp - before_lidar_sweep->timestamp_);
                //             Eigen::Vector4d interpolate_qvec;
                //             Eigen::Vector3d interpolate_tvec;
                //             InterpolatePose(before_lidar_sweep->Qvec(), before_lidar_sweep->Tvec(), lidar_sweep.Qvec(), lidar_sweep.Tvec(), 
                //                             t, &interpolate_qvec, &interpolate_tvec);

                //             lP.block<3, 3>(0, 0) = QuaternionToRotationMatrix(interpolate_qvec);
                //             lP.block<3, 1>(0, 3) = interpolate_tvec;
                //             lP_t = config_.lidar_to_cam_matrix_ * lP;

                //             R = lP_t.block<3, 3>(0, 0);
                //             prior_q = RotationMatrixToQuaternion(R);
                //             prior_c = -R.transpose() * lP_t.block<3, 1>(0, 3);
                //         }
                //     } else if (lidar_timestamp < image_timestamp) {
                //         class LidarSweep* after_lidar_sweep = NULL;
                //         // std::cout << "sweep_timestamps.size()[backward]: " << sweep_timestamps.size() << std::endl;
                //         for (size_t k = 0; k < sweep_timestamps.size(); ++k) {
                //             auto & near_lidar_sweep = reconstruction->LidarSweep(sweep_timestamps[k].first);
                //             if (!near_lidar_sweep.IsRegistered() || near_lidar_sweep.timestamp_ < image_timestamp) {
                //                 continue;
                //             }
                //             if (near_lidar_sweep.timestamp_ >= image_timestamp) {
                //                 if (near_lidar_sweep.timestamp_ - lidar_timestamp < 5 * 1e9) { // 5s
                //                     after_lidar_sweep = &near_lidar_sweep;
                //                     break;
                //                 }
                //             }
                //         }
                //         if (after_lidar_sweep) {
                //             double t = (image_timestamp - lidar_timestamp) * 1.0 / (after_lidar_sweep->timestamp_ - lidar_timestamp);
                //             Eigen::Vector4d interpolate_qvec;
                //             Eigen::Vector3d interpolate_tvec;
                //             InterpolatePose(lidar_sweep.Qvec(), lidar_sweep.Tvec(), after_lidar_sweep->Qvec(), after_lidar_sweep->Tvec(), 
                //                             t, &interpolate_qvec, &interpolate_tvec);

                //             lP.block<3, 3>(0, 0) = QuaternionToRotationMatrix(interpolate_qvec);
                //             lP.block<3, 1>(0, 3) = interpolate_tvec;
                //             lP_t = config_.lidar_to_cam_matrix_ * lP;

                //             R = lP_t.block<3, 3>(0, 0);
                //             prior_q = RotationMatrixToQuaternion(R);
                //             prior_c = -R.transpose() * lP_t.block<3, 1>(0, 3);
                //         }
                //     }
                // }

                ceres::ScaledLoss *loss_function = new ceres::ScaledLoss(new ceres::TrivialLoss(), m_visible_map_per_image, ceres::DO_NOT_TAKE_OWNERSHIP);
                // ceres::ScaledLoss *loss_function = new ceres::ScaledLoss(new ceres::TrivialLoss(), 
                //     std::max(image.NumMapPoints(), m_visible_map_per_image), ceres::DO_NOT_TAKE_OWNERSHIP);
                // ceres::ScaledLoss *loss_function = new ceres::ScaledLoss(new ceres::TrivialLoss(), 1, ceres::DO_NOT_TAKE_OWNERSHIP);
                ceres::CostFunction *cost_function = PriorAbsolutePoseCostFunction::Create(prior_q, prior_c, weight * 5.0, weight * 100.0, weight * 100.0, weight * 100.0);
                // ceres::CostFunction *cost_function = PriorAbsolutePoseCostFunction::Create(prior_q, prior_c, weight * 1.0, weight * 5.0, weight * 5.0, weight * 5.0);
                problem_->AddResidualBlock(cost_function, loss_function, qvec_data, tvec_data);
                
                // ceres::LocalParameterization *quaternion_parameterization = new ceres::QuaternionParameterization;
                // SetParameterization(problem_.get(), qvec_data, quaternion_parameterization);
                if (has_constant_image_pose) {
                    problem_->SetParameterBlockConstant(qvec_data);
                    problem_->SetParameterBlockConstant(tvec_data);
                }
            }

            if (shared_points.find(image_id) != shared_points.end()) {
                std::vector<std::pair<image_t, int> > shared_image_points;
                shared_image_points.reserve(shared_points[image_id].size());
                for (auto shared_image : shared_points[image_id]) {
                    shared_image_points.emplace_back(shared_image.first, shared_image.second);
                }
                if (shared_image_points.size() > 0) {
                    size_t nth = std::min(shared_image_points.size() - 1, (size_t)2);
                    std::nth_element(shared_image_points.begin(), shared_image_points.begin() + nth, shared_image_points.end(),
                        [&](const std::pair<image_t, int> & a, const std::pair<image_t, int> & b) {
                            return a.second > b.second;
                        });
                }
                
                for (auto shared_image : shared_image_points) {
                    sweep_t sweep_id2 = sweep_image_pairs.at(shared_image.first);
                    class LidarSweep & lidar_sweep2 = reconstruction->LidarSweep(sweep_id2);
                    if (!lidar_sweep2.IsRegistered()) {
                        continue;
                    }
                    double prior_distance = (lidar_sweep.ProjectionCenter() - lidar_sweep2.ProjectionCenter()).norm();

                    ceres::ScaledLoss *loss_function = new ceres::ScaledLoss(new ceres::TrivialLoss(), m_visible_map_per_image, ceres::DO_NOT_TAKE_OWNERSHIP);
                    ceres::CostFunction *cost_function = LidarRelativeDistanceCostFunction::Create(prior_distance, 1.0);
                    problem_->AddResidualBlock(cost_function, loss_function, lidar_sweep.Qvec().data(), lidar_sweep.Tvec().data(), lidar_sweep2.Qvec().data(), lidar_sweep2.Tvec().data());

                    ceres::LocalParameterization *quaternion_parameterization = new ceres::QuaternionParameterization;
                    SetParameterization(problem_.get(), lidar_sweep.Qvec().data(), quaternion_parameterization);
                    SetParameterization(problem_.get(), lidar_sweep2.Qvec().data(), quaternion_parameterization);
                    if (has_constant_lidar_pose) {
                        problem_->SetParameterBlockConstant(lidar_sweep.Qvec().data());
                        problem_->SetParameterBlockConstant(lidar_sweep.Tvec().data());
                        problem_->SetParameterBlockConstant(lidar_sweep2.Qvec().data());
                        problem_->SetParameterBlockConstant(lidar_sweep2.Tvec().data());
                    }
                }
            }
        }
    }
}

void BundleAdjuster::AddLidarFrame2FrameToProblem(const sweep_t sweep_id,
                                            Reconstruction *reconstruction,
                                            ceres::LossFunction *loss_function,
                                            const double loss_weight){
    std::cout << "add Lidar " << sweep_id << " to problem ..." << std::endl;
    LidarSweep &lidarsweep = reconstruction->LidarSweep(sweep_id);
    Eigen::Matrix4d Tr_ref_inv_4d = Eigen::Matrix4d::Identity();
    Tr_ref_inv_4d.topRows(3) = lidarsweep.InverseProjectionMatrix();

    lidarsweep.NormalizeQvec();
    double *qvec_ref_data = lidarsweep.Qvec().data();
    double *tvec_ref_data = lidarsweep.Tvec().data();

    if (reconstruction->common_view_map.empty()){
        int num_observation = 0;
        // std::cout << "noSubmap add to sweep " << sweep_id << ": ";
        for (auto sweep_src : config_.Sweeps()){
            if( sweep_src == (sweep_t)sweep_id){
                continue;
            }

            LidarSweep &lidarsweep_src = reconstruction->LidarSweep(sweep_src);

            const bool constant_pose = !options_.refine_extrinsics || 
                                    (config_.HasConstantSweep(sweep_src) && 
                                    config_.HasConstantSweep(sweep_id)) ;
            if (constant_pose) {
                continue;
            }

            // std::cout << sweep_src << "(" << image_src.Name() << ")  " ;
            lidarsweep_src.NormalizeQvec();
            double *qvec_src_data = lidarsweep_src.Qvec().data();
            double *tvec_src_data = lidarsweep_src.Tvec().data();

            // transform ref to src
            Eigen::Matrix4d Tr_src_inv_4d = Eigen::Matrix4d::Identity();
            Tr_src_inv_4d.topRows(3) = lidarsweep_src.InverseProjectionMatrix();
            Eigen::Matrix4d Tr_ref2src_4d = Tr_src_inv_4d.inverse() * Tr_ref_inv_4d;
            std::vector<struct LidarEdgeCorrespondence> edge_point_corr;
            std::vector<struct LidarPlaneCorrespondence> plane_point_corrs;
            std::vector<struct LidarPointCorrespondence> point_point_corrs;

            // std::string save_name = std::to_string(sweep_id) + "-" + std::to_string(sweep_src);
            std::string save_name = GetPathBaseName(lidarsweep.Name()) + "-" + GetPathBaseName(lidarsweep_src.Name());
            // std::string save_dir = "F2F_" + std::to_string(reconstruction->RegisterSweepIds().size());
            std::string save_path = options_.save_path + "/" + save_name;
            if (!LidarPointsCorrespondence(lidarsweep, lidarsweep_src, Tr_ref2src_4d, 
                                    edge_point_corr, plane_point_corrs, point_point_corrs,
                                    save_path)){
                continue;
            };
            // std::cout << "AddLidarFrame2FrameToProblem: " << sweep_id << " - " << sweep_src << ", " 
            //     << edge_point_corr.size() << ", " << plane_point_corrs.size() << std::endl;

            // caculate edge residual 
            // double residual_edge = 0.0;
            // int debug_log_1 = 0; 
            // int debug_log_2 = 0; 
            num_observation = edge_point_corr.size() + plane_point_corrs.size() + point_point_corrs.size();

            for (auto edge_point : edge_point_corr){
                Eigen::Vector3d curr_point_t, last_point_a_t, last_point_b_t;
                curr_point_t = edge_point.ref_point_;
                last_point_a_t = edge_point.src_point_a_;
                last_point_b_t = edge_point.src_point_b_;
                // std::cout << "edge_point.s: " << edge_point.s << std::endl;

                // residual_edge += LidarPointsEdgeResidual(curr_point_t, 
                //                 last_point_a_t, last_point_b_t,
                //                 qvec_ref_data, tvec_ref_data, 
                //                 qvec_src_data, tvec_src_data,
                //                 Tr_ref2src_4d, Tr_lidar2camera_4d);
                if (1){
                    if (config_.HasConstantSweep(sweep_id) && 
                        !config_.HasConstantSweep(sweep_src)){
                        ceres::CostFunction *cost_function = 
                            BundleAdjustmentLidarEdgeConstantRefPoseCostFunction::Create(
                            lidarsweep.Qvec(), lidarsweep.Tvec(), curr_point_t, last_point_a_t, 
                            last_point_b_t, 1.0, loss_weight * 3 * edge_point.s);
                        problem_->AddResidualBlock(cost_function, loss_function, 
                                                   qvec_src_data, tvec_src_data);
                    } else if (!config_.HasConstantSweep(sweep_id) && 
                                config_.HasConstantSweep(sweep_src)){
                        ceres::CostFunction *cost_function = 
                            BundleAdjustmentLidarEdgeConstantSrcPoseCostFunction::Create(
                            lidarsweep_src.Qvec(), lidarsweep_src.Tvec(), curr_point_t, 
                            last_point_a_t, last_point_b_t, 1.0, loss_weight * 3 * edge_point.s);
                        problem_->AddResidualBlock(cost_function, loss_function, 
                                                   qvec_ref_data, tvec_ref_data);
                    } else {
                        ceres::CostFunction *cost_function = 
                            BundleAdjustmentLidarEdgeCostFunction::Create(
                            curr_point_t, last_point_a_t, last_point_b_t, 1.0, loss_weight * 3 * edge_point.s);
                        problem_->AddResidualBlock(cost_function, loss_function, 
                            qvec_ref_data, tvec_ref_data, qvec_src_data, tvec_src_data);
                    }
                }
            }

            // double residual_plane = 0.0;
            for (auto plane_point : plane_point_corrs) {
                Eigen::Vector3d curr_point_t, last_point_j_t, last_point_l_t, last_point_m_t;
                curr_point_t = plane_point.ref_point_;
                last_point_j_t = plane_point.src_point_j_;
                last_point_l_t = plane_point.src_point_l_;
                last_point_m_t = plane_point.src_point_m_;

                // std::cout << "plane_point.s: " << plane_point.s << std::endl;
                //  residual_plane += LidarPointsPlaneResidual(curr_point_t, 
                //                 last_point_j_t, last_point_l_t, last_point_m_t,
                //                 qvec_ref_data, tvec_ref_data, 
                //                 qvec_src_data, tvec_src_data,
                //                 Tr_ref2src_4d, Tr_lidar2camera_4d);


                if (1){
                    if (config_.HasConstantSweep(sweep_id) && 
                        !config_.HasConstantSweep(sweep_src)){
                        ceres::CostFunction *cost_function = 
                            BundleAdjustmentLidarPlaneConstantRefPoseCostFunction::Create(
                            lidarsweep.Qvec(), lidarsweep.Tvec(), curr_point_t, last_point_j_t, 
                            last_point_l_t, last_point_m_t, 1.0, loss_weight * plane_point.s);
                        problem_->AddResidualBlock(cost_function, loss_function, 
                                                qvec_src_data, tvec_src_data);
                    } else if (!config_.HasConstantSweep(sweep_id) && 
                                config_.HasConstantSweep(sweep_src)){
                        ceres::CostFunction *cost_function = 
                            BundleAdjustmentLidarPlaneConstantSrcPoseCostFunction::Create(
                            lidarsweep_src.Qvec(), lidarsweep_src.Tvec(), curr_point_t, 
                            last_point_j_t, last_point_l_t, last_point_m_t, 1.0, loss_weight * plane_point.s);
                        problem_->AddResidualBlock(cost_function, loss_function, 
                                                qvec_ref_data, tvec_ref_data);
                    } else {
                        ceres::CostFunction *cost_function = 
                            BundleAdjustmentLidarPlaneCostFunction::Create(curr_point_t, 
                            last_point_j_t, last_point_l_t, last_point_m_t, 1.0, loss_weight * plane_point.s);
                        problem_->AddResidualBlock(cost_function, loss_function, 
                            qvec_ref_data, tvec_ref_data, qvec_src_data, tvec_src_data);
                    }
                }            
            }
            for (auto point_point : point_point_corrs){
                Eigen::Vector3d curr_point_t, last_point_t;
                curr_point_t = point_point.ref_point_;
                last_point_t = point_point.src_point_;

                if (1){
                    if (config_.HasConstantSweep(sweep_id) && 
                        !config_.HasConstantSweep(sweep_src)){
                        ceres::CostFunction *cost_function = 
                            BundleAdjustmentLidarPointConstantRefPoseCostFunction::Create(
                            lidarsweep.Qvec(), lidarsweep.Tvec(), curr_point_t, last_point_t, 
                            1.0, loss_weight * point_point.s * 0.5);
                        problem_->AddResidualBlock(cost_function, loss_function, 
                                                   qvec_src_data, tvec_src_data);
                    } else if (!config_.HasConstantSweep(sweep_id) && 
                                config_.HasConstantSweep(sweep_src)){
                        ceres::CostFunction *cost_function = 
                            BundleAdjustmentPointEdgeConstantSrcPoseCostFunction::Create(
                            lidarsweep_src.Qvec(), lidarsweep_src.Tvec(), curr_point_t, 
                            last_point_t, 1.0, loss_weight * point_point.s * 0.5);
                        problem_->AddResidualBlock(cost_function, loss_function, 
                                                   qvec_ref_data, tvec_ref_data);
                    } else {
                        ceres::CostFunction *cost_function = 
                            BundleAdjustmentLidarPointCostFunction::Create(
                            curr_point_t, last_point_t, 1.0f, loss_weight * point_point.s * 0.5);
                        problem_->AddResidualBlock(cost_function, loss_function, 
                            qvec_ref_data, tvec_ref_data, qvec_src_data, tvec_src_data);
                    }
                }
            }
            // std::cout << "feature corr num: " << edge_point_corr.size() + plane_point_corrs.size()
            //     << "(" << edge_point_corr.size() << "/" <<  plane_point_corrs.size() << ")\t" 
            //     << "residual(" << image.Name() <<  "/" << image_src.Name() << ") : "
            //     << (residual_edge + residual_plane) / (edge_point_corr.size() + plane_point_corrs.size())
            //     << "(e:" << residual_edge/edge_point_corr.size() << " / p:" 
            //     << residual_plane/plane_point_corrs.size() << ")" << std::endl;
            // if (!config_.HasConstantSweep(sweep_src)){
            //     ceres::LocalParameterization *quaternion_parameterization = new ceres::QuaternionParameterization;
            //     SetParameterization(problem_.get(), qvec_src_data, quaternion_parameterization);
            // }

        }

        if (!config_.HasConstantSweep(sweep_id) && num_observation > 0){
            ceres::LocalParameterization *quaternion_parameterization = new ceres::QuaternionParameterization;
            SetParameterization(problem_.get(), qvec_ref_data, quaternion_parameterization);
        }
    } 
}

void BundleAdjuster::AddExtraLidar2CameraToProblem(const camera_t camera_id,
                                            Reconstruction *reconstruction,
                                            ceres::LossFunction *loss_function,
                                            const double loss_weight,
                                            const Eigen::Matrix4d absolute_pose){
    auto& camera = reconstruction->Camera(camera_id);
    const Eigen::Quaterniond absolute_qua(absolute_pose.block<3,3>(0,0));
    const Eigen::Vector3d absolute_t = absolute_pose.block<3,1>(0,3);

    double *qvec_data = camera.LidarQvecs().data();
    double *tvec_data = camera.LidarTvecs().data();

    // double loss_weight = reconstruction->NumLidarSweep() * loss_weight;
    const double camera_weight = reconstruction->NumLidarSweep() * 10; 

    ceres::CostFunction *cost_function = 
        BundleAdjustmentAbsolatePoseCostFunction::Create(absolute_qua, absolute_t, camera_weight);

    problem_->AddResidualBlock(cost_function, loss_function, qvec_data, tvec_data);
}

void BundleAdjuster::ParameterizeCameras(Reconstruction *reconstruction) {
    const bool constant_camera =
        !options_.refine_focal_length && !options_.refine_principal_point && !options_.refine_extra_params;
    

    for (const camera_t camera_id : camera_ids_) {
        Camera &camera = reconstruction->Camera(camera_id);

        if (camera.NumLocalCameras() > 1) {
            
            if (local_camera_ids_.find(camera_id) == local_camera_ids_.end()) {
                continue;
            }

            std::unordered_set<int> local_camera_set = local_camera_ids_.at(camera_id);

            int local_param_size = camera.LocalParams().size() / camera.NumLocalCameras();

            for (auto local_camera_id : local_camera_set) {
                double *local_qvec_data = camera.LocalQvecsData() + 4 * local_camera_id;
                double *local_tvec_data = camera.LocalTvecsData() + 3 * local_camera_id;
                double *local_camera_params_data =
                    camera.LocalIntrinsicParamsData() + local_camera_id * local_param_size;

                if(!(camera.ModelName().compare("OPENCV_FISHEYE") == 0 && constant_camera)){
                    if (constant_camera || config_.IsConstantCamera(camera_id)) {
                        problem_->SetParameterBlockConstant(local_camera_params_data);
                    } else {
                        std::vector<int> const_camera_params;
                        if (!options_.refine_focal_length) {
                            const std::vector<size_t> &params_idxs = camera.FocalLengthIdxs();
                            const_camera_params.insert(const_camera_params.end(), params_idxs.begin(),
                                                       params_idxs.end());
                        }
                        if (!options_.refine_principal_point) {
                            const std::vector<size_t> &params_idxs = camera.PrincipalPointIdxs();
                            const_camera_params.insert(const_camera_params.end(), params_idxs.begin(),
                                                       params_idxs.end());
                        }
                        if ((!options_.refine_extra_params) && (camera.ModelName().compare("SPHERICAL") != 0)) {
                            const std::vector<size_t> &params_idxs = camera.ExtraParamsIdxs();
                            const_camera_params.insert(const_camera_params.end(), params_idxs.begin(),
                                                       params_idxs.end());
                        }

                        if (const_camera_params.size() > 0) {
                            ceres::SubsetParameterization *camera_params_parameterization =
                                new ceres::SubsetParameterization(static_cast<int>(local_param_size),
                                                                  const_camera_params);
                            SetParameterization(problem_.get(), local_camera_params_data, camera_params_parameterization);
                        }
                    }
                }

                if (options_.refine_local_extrinsics && local_camera_id > 0) {
                    ceres::LocalParameterization *quaternion_parameterization = new ceres::QuaternionParameterization;
                    SetParameterization(problem_.get(), local_qvec_data, quaternion_parameterization);
                } else {
                    problem_->SetParameterBlockConstant(local_qvec_data);
                    problem_->SetParameterBlockConstant(local_tvec_data);
                }
            }
            continue;
        }

        if (constant_camera || config_.IsConstantCamera(camera_id)) {

            if(camera.ModelName().compare("SPHERICAL") != 0 && camera.ModelName().compare("OPENCV_FISHEYE") != 0){
                problem_->SetParameterBlockConstant(camera.ParamsData());
            }
            if (camera.ModelName().compare("SPHERICAL") == 0) {
                double *camera_params_data = camera.ParamsData();

                double *local_qvec2_data = camera_params_data + 10;
                double *local_tvec2_data = camera_params_data + 14;
                problem_->SetParameterBlockConstant(local_qvec2_data);
                problem_->SetParameterBlockConstant(local_tvec2_data);
            }
            continue;
        } else {
            if(!(camera.ModelName()=="SPHERICAL")){
                std::vector<int> const_camera_params;

                if (!options_.refine_focal_length) {
                    const std::vector<size_t> &params_idxs = camera.FocalLengthIdxs();
                    const_camera_params.insert(const_camera_params.end(), params_idxs.begin(), params_idxs.end());
                }
                if (!options_.refine_principal_point) {
                    const std::vector<size_t> &params_idxs = camera.PrincipalPointIdxs();
                    const_camera_params.insert(const_camera_params.end(), params_idxs.begin(), params_idxs.end());
                }
                if ((!options_.refine_extra_params) && (camera.ModelName().compare("SPHERICAL") != 0)) {
                    const std::vector<size_t> &params_idxs = camera.ExtraParamsIdxs();
                    const_camera_params.insert(const_camera_params.end(), params_idxs.begin(), params_idxs.end());
                }

                for (size_t idx : camera.FocalLengthIdxs()) {
                    double est_focal = camera.ParamsData()[idx];
                    problem_->SetParameterLowerBound(camera.ParamsData(), idx,
                                                     options_.lower_bound_focal_length_factor * est_focal);
                    problem_->SetParameterUpperBound(camera.ParamsData(), idx,
                                                     options_.upper_bound_focal_length_factor * est_focal);
                }
                for (size_t idx : camera.PrincipalPointIdxs()) {
                    problem_->SetParameterLowerBound(camera.ParamsData(), idx, 0);
                    problem_->SetParameterUpperBound(camera.ParamsData(), idx,
                                                     std::max(camera.Width(), camera.Height()));
                }

                for (size_t idx : camera.ExtraParamsIdxs()) {
                    problem_->SetParameterLowerBound(camera.ParamsData(), idx, -0.99);
                    problem_->SetParameterUpperBound(camera.ParamsData(), idx, 0.99);
                }

                if (const_camera_params.size() > 0) {
                    if (camera.ModelName().compare("SPHERICAL") != 0) {
                        ceres::SubsetParameterization *camera_params_parameterization =
                            new ceres::SubsetParameterization(static_cast<int>(camera.NumParams()),
                                                              const_camera_params);
                        SetParameterization(problem_.get(), camera.ParamsData(), camera_params_parameterization);
                    } else {
                        problem_->SetParameterBlockConstant(camera.ParamsData());
                    }
                }
            }

            if (camera.ModelName().compare("SPHERICAL") == 0) {
                double *camera_params_data = camera.ParamsData();

                double *local_qvec2_data = camera_params_data + 10;
                double *local_tvec2_data = camera_params_data + 14;

                if (options_.refine_extra_params) {
                    ceres::LocalParameterization *quaternion_parameterization2 = new ceres::QuaternionParameterization;
                    SetParameterization(problem_.get(), local_qvec2_data, quaternion_parameterization2);
                } else {
                    problem_->SetParameterBlockConstant(local_qvec2_data);
                    problem_->SetParameterBlockConstant(local_tvec2_data);
                }
            }
        }
    }

    for(const camera_t camera_id : gnss_camera_ids_){
        Camera &camera = reconstruction->Camera(camera_id);
        if (!camera.HasDisturb()) {
            continue;
        }
        double *qvec_data = camera.QvecDisturb().data();
        double *tvec_data = camera.TvecDisturb().data();
        ceres::LocalParameterization *quaternion_parameterization = new ceres::QuaternionParameterization;
        SetParameterization(problem_.get(), qvec_data, quaternion_parameterization);
#if 0
        // ceres::SubsetParameterization *tvec_parameterization =
        //     new ceres::SubsetParameterization(3, {1});
        // SetParameterization(problem_.get(), tvec_data, tvec_parameterization);
#else
        for(size_t idx = 0; idx < camera.TvecDisturb().size(); idx++){
            problem_->SetParameterLowerBound(tvec_data, idx, -1.0);
            problem_->SetParameterUpperBound(tvec_data, idx, 1.0);
        }
#endif
    }

    for (const camera_t camera_id : lidar_camera_ids_){
        std::cout << "add lidar_camera_ids_ " << camera_id << std::endl;
        Camera &camera = reconstruction->Camera(camera_id);
        double *qvec_l2c_data = camera.LidarQvecs().data();
        double *tvec_l2c_data = camera.LidarTvecs().data();
        if (0){
        // if (options_.refine_lidar2cam_params && config_.NumSweeps() > options_.refine_min_numsweeps
        //     && options_.ba_residual_type == BundleAdjustResidualType::FRAME2FRAME){
            ceres::LocalParameterization *quaternion_parameterization = new ceres::QuaternionParameterization;
            SetParameterization(problem_.get(), qvec_l2c_data, quaternion_parameterization);
        } else {
            problem_->SetParameterBlockConstant(qvec_l2c_data);
            problem_->SetParameterBlockConstant(tvec_l2c_data);
        }
    }
}

void BundleAdjuster::ParameterizePoints(Reconstruction *reconstruction) {
    for (const auto elem : mappoint_num_observations_) {
        MapPoint &mappoint = reconstruction->MapPoint(elem.first);
        if (mappoint.Track().Length() > elem.second) {
            problem_->SetParameterBlockConstant(mappoint.XYZ().data());
        }
    }

    for (const mappoint_t mappoint_id : config_.ConstantPoints()) {
        MapPoint &mappoint = reconstruction->MapPoint(mappoint_id);
        problem_->SetParameterBlockConstant(mappoint.XYZ().data());
    }
}

void BundleAdjuster::ParameterizePointsWithoutTrack(Reconstruction *reconstruction) {
    for (const mappoint_t mappoint_id : config_.ConstantPoints()) {
        MapPoint &mappoint = reconstruction->MapPoint(mappoint_id);
        problem_->SetParameterBlockConstant(mappoint.XYZ().data());
    }
}

void BundleAdjuster::ParameterizePoses(Reconstruction *reconstruction) {
    std::cout << "invovled image extrinsics num: " << invovled_image_extrinsics_.size() << std::endl;
    int const_camera_count = 0;
    for (const image_t image_id : invovled_image_extrinsics_) {
        if (constant_pose_images_.count(image_id) > 0) {
            Image &image = reconstruction->Image(image_id);

            double *qvec_data = image.Qvec().data();
            double *tvec_data = image.Tvec().data();
            problem_->SetParameterBlockConstant(qvec_data);
            problem_->SetParameterBlockConstant(tvec_data);
            const_camera_count++;
        }
    }
    std::cout << "const camera count: " << const_camera_count << std::endl;
}

void PrintSolverSummary(const ceres::Solver::Summary &summary) {
    std::cout << std::right << std::setw(16) << "Residuals : ";
    std::cout << std::left << summary.num_residuals_reduced << std::endl;

    std::cout << std::right << std::setw(16) << "Parameters : ";
    std::cout << std::left << summary.num_effective_parameters_reduced << std::endl;

    std::cout << std::right << std::setw(16) << "Iterations : ";
    std::cout << std::left << summary.num_successful_steps + summary.num_unsuccessful_steps << std::endl;

    std::cout << std::right << std::setw(16) << "Time : ";
    std::cout << std::left << summary.total_time_in_seconds << " [s]" << std::endl;

    std::cout << std::right << std::setw(16) << "Initial cost : ";
    std::cout << std::right << std::setprecision(6) << std::sqrt(summary.initial_cost / summary.num_residuals_reduced)
              << " [px]" << std::endl;

    std::cout << std::right << std::setw(16) << "Final cost : ";
    std::cout << std::right << std::setprecision(6) << std::sqrt(summary.final_cost / summary.num_residuals_reduced)
              << " [px]" << std::endl;

    std::cout << std::right << std::setw(16) << "Termination : ";

    std::string termination = "";

    switch (summary.termination_type) {
        case ceres::CONVERGENCE:
            termination = "Convergence";
            break;
        case ceres::NO_CONVERGENCE:
            termination = "No convergence";
            break;
        case ceres::FAILURE:
            termination = "Failure";
            break;
        case ceres::USER_SUCCESS:
            termination = "User success";
            break;
        case ceres::USER_FAILURE:
            termination = "User failure";
            break;
        default:
            termination = "Unknown";
            break;
    }

    std::cout << std::right << termination << std::endl;
    std::cout << std::endl;
}


// BlockBundleAdjuster
BlockBundleAdjuster::BlockBundleAdjuster(const BundleAdjustmentOptions& options,
                                         const BundleAdjustmentConfig& config)
    : options_(options),
      config_(config) {
    block_options_ = BlockBundleAdjustmentOptions(config.NumImages(), options.block_size, 
                                                  options.block_common_image_num,
                                                  options.min_connected_points_for_common_images);
}

void BlockBundleAdjuster::SetParameterization(ceres::Problem *problem, double* values, ceres::LocalParameterization *local_parameterization) {
#if CERES_VERSION_MAJOR >= 2 && CERES_VERSION_MINOR >= 1
    problem->SetManifold(values, local_parameterization);
#else
    problem->SetParameterization(values, local_parameterization);
#endif
}

bool BlockBundleAdjuster::Solve(Reconstruction* reconstruction) {
    if (options_.use_gravity) {
        gravity_.setZero();
        for (auto & image_id : config_.Images()) {
            auto & image = reconstruction->Image(image_id);
            Eigen::Vector3d gravity = image.gravity_;
            Eigen::Matrix3d R = image.RotationMatrix();
            gravity_ += R.transpose() * gravity;
        }

        if (!gravity_.isZero()) {
            gravity_.normalize();
        }
    }

    std::unordered_map<image_pair_t, int> corres_map;
    std::vector<Block> blocks;
    DivideBlocks(reconstruction, &blocks, &corres_map);

    if (blocks.size() <= 1) {
        BundleAdjuster bundle_adjuster(options_, config_);
        return bundle_adjuster.Solve(reconstruction);
    }

    std::cout << "Solve Blocks Sequentially." << std::endl;

    const auto & correspondence_graph = reconstruction->GetCorrespondenceGraph();

    // Solve Block sequentially.
    std::unordered_set<image_t> solved_image_ids;
    std::unordered_set<camera_t> solved_camera_ids;
    std::unordered_set<camera_t> refine_camera_ids;

    PrintHeading1(StringPrintf("Bundle Block Adjustment #%d", blocks[0].id));

    std::cout << "Block images: " << blocks[0].config.NumImages() << std::endl;

    const std::unordered_set<image_t> & block_image_ids = blocks[0].config.Images();
    std::unordered_set<mappoint_t> variable_mappoint_ids;
    for (auto image_id : block_image_ids) {
        class Image &image = reconstruction->Image(image_id);
        for (const auto point2D : image.Points2D()) {
            if (point2D.HasMapPoint()) {
                variable_mappoint_ids.insert(point2D.MapPointId());
            }
        }
        refine_camera_ids.insert(image.CameraId());
    }

    // Eliminate short track in the first block.
    for (auto mappoint_id : variable_mappoint_ids) {
        const auto & track = reconstruction->MapPoint(mappoint_id).Track();
        int track_degree_in_block = 0;
        for (auto track_elem : track.Elements()) {
            if (block_image_ids.find(track_elem.image_id) != block_image_ids.end()) {
                track_degree_in_block++;
            }
        }
        if (track_degree_in_block < 3) {
            blocks[0].mappoints_in_mask.insert(mappoint_id);
        }
    }

    std::cout << "Optimized Camera: ";
    for (auto camera_id : refine_camera_ids) {
        std::cout << camera_id << " ";
    }
    std::cout << std::endl;

    SolveBlock(reconstruction, &blocks[0]);
    for (auto image_id : block_image_ids) {
        solved_image_ids.insert(image_id);
        camera_t camera_id = reconstruction->Image(image_id).CameraId();
        solved_camera_ids.insert(camera_id);
    }

    for (const auto &camera : reconstruction->Cameras()){
        if (camera.second.NumLocalCameras() <= 1) {
            std::cout << "Camera#" << camera.first << ", param: " << camera.second.ParamsToString() << std::endl;
        } else {
            std::cout << "Camera#" << camera.first << ", param: " << VectorToCSV(camera.second.LocalParams()) << std::endl;
        }
    }

    for (size_t i = 1; i < blocks.size(); ++i) {
        Block & block = blocks.at(i);
        PrintHeading1(StringPrintf("Bundle Block Adjustment #%d", block.id));

        const auto & block_image_ids = block.config.Images();
        
        refine_camera_ids.clear();
        
        std::unordered_set<camera_t> block_camera_ids;
        for (auto image_id : block_image_ids) {
            camera_t camera_id = reconstruction->Image(image_id).CameraId();
            block_camera_ids.insert(camera_id);
            if (solved_camera_ids.find(camera_id) != solved_camera_ids.end()) {
                block.config.SetConstantCamera(camera_id);
            } else {
                refine_camera_ids.insert(camera_id);
            }
        }

        std::unordered_map<image_t, int> anchor_images;
        std::unordered_set<mappoint_t> variable_mappoint_ids;
        for (auto image_id : block_image_ids) {
            const auto & neighbor_images = correspondence_graph->ImageNeighbor(image_id);
            for (auto neighbor_image_id : neighbor_images) {
                if (solved_image_ids.find(neighbor_image_id) == solved_image_ids.end()) {
                    continue;
                }
                image_pair_t pair_id = utility::ImagePairToPairId(image_id, neighbor_image_id);
                auto iter = corres_map.find(pair_id);
                if (iter != corres_map.end()) {
                    // int num_corrs = std::max(anchor_images[neighbor_image_id], iter->second);
                    // anchor_images[neighbor_image_id] = num_corrs;
                    anchor_images[neighbor_image_id] += iter->second;
                }
            }

            class Image &image = reconstruction->Image(image_id);
            for (const auto point2D : image.Points2D()) {
                if (point2D.HasMapPoint()) {
                    variable_mappoint_ids.insert(point2D.MapPointId());
                }
            }
        }

        // Add anchor images and cameras.
        std::vector<std::pair<image_t, int> > sorted_anchor_images;
        sorted_anchor_images.insert(sorted_anchor_images.end(), anchor_images.begin(), anchor_images.end());
        if (sorted_anchor_images.size() > block_options_.common_images_num) {
            std::nth_element(sorted_anchor_images.begin(), 
                             sorted_anchor_images.begin() + block_options_.common_images_num,
                             sorted_anchor_images.end(), 
                [&](const std::pair<image_t, int> &a, const std::pair<image_t, int> &b) {
                    return a.second > b.second;
                });
        }
        size_t num_images_fixed = std::min(sorted_anchor_images.size(), block_options_.common_images_num);
        for (size_t j = 0; j < num_images_fixed; ++j) {
            image_t image_id = sorted_anchor_images[j].first;
        // for (auto anchor_image : anchor_images) {
        //     image_t image_id = anchor_image.first;
            block.config.AddImage(image_id);
            block.config.SetConstantPose(image_id);

            class Image image = reconstruction->Image(image_id);
            camera_t camera_id = image.CameraId();
            if (block_camera_ids.find(camera_id) == block_camera_ids.end()) {
                block.config.SetConstantCamera(camera_id);
            }
        }
        std::cout << StringPrintf("Add %d anchor images\n", num_images_fixed);
        std::cout << "Block images: " << blocks[1].config.NumImages() << std::endl;
        std::cout << "Optimized Camera: ";
        for (auto camera_id : refine_camera_ids) {
            std::cout << camera_id << " ";
        }
        std::cout << std::endl;

        for (auto mappoint_id : variable_mappoint_ids) {
            auto & track_elems = reconstruction->MapPoint(mappoint_id).Track().Elements();
            int num_observation_solved = 0; 
            for (auto track_elem : track_elems) {
                if (solved_image_ids.find(track_elem.image_id) != solved_image_ids.end()) {
                    num_observation_solved++;
                }
            }
            int num_observation_unsolved = track_elems.size() - num_observation_solved;
            if (num_observation_solved > 0) {
                if (num_observation_solved > num_observation_unsolved &&
                    num_observation_solved > 3) {
                    block.config.AddConstantPoint(mappoint_id);
                } else {
                    block.config.AddVariablePoint(mappoint_id);
                }
            }
        }
        std::cout << StringPrintf("Add %d mappoints\n", variable_mappoint_ids.size());

        // // Add anchor camera.
        // for (auto image_id : block_image_ids) {
        //     camera_t camera_id = reconstruction->Image(image_id).CameraId();
        //     if (solved_camera_ids.find(camera_id) != solved_camera_ids.end()) {
        //         block.config.SetConstantCamera(camera_id);
        //     } else if (block_camera_ids.find(camera_id) != block_camera_ids.end()) {
        //         refine_camera_ids.insert(camera_id);
        //     }
        // }

        // Solve Block.
        if (SolveBlock(reconstruction, &block)) {
            solved_image_ids.insert(block_image_ids.begin(), block_image_ids.end());
            for (auto image_id : block_image_ids) {
                camera_t camera_id = reconstruction->Image(image_id).CameraId();
                solved_camera_ids.insert(camera_id);
            }

            for (const auto &camera : reconstruction->Cameras()){
                if (camera.second.NumLocalCameras() <= 1) {
                    std::cout << "Camera#" << camera.first << ", param: " << camera.second.ParamsToString() << std::endl;
                } else {
                    std::cout << "Camera#" << camera.first << ", param: " << VectorToCSV(camera.second.LocalParams()) << std::endl;
                }
            }
        }
    }
    return true;
}

void BlockBundleAdjuster::DivideBlocks(Reconstruction* reconstruction, std::vector<Block>* blocks,
                                       std::unordered_map<image_pair_t, int>* corres_map) {
    Timer timer;
    timer.Start();
    auto images = config_.Images();
    auto constant_cameras = config_.ConstantCameraIds();
    auto constant_poses = config_.ConstantPoses();
    auto constant_tvecs = config_.ConstantTVecs();

    auto AddImages = [](
        Block* block, const image_t image_id,
        std::unordered_set<image_t> &constant_poses,
        std::unordered_map<image_t, std::vector<int>> &constant_tvecs) -> bool {
        block->config.AddImage(image_id);
        if (constant_poses.count(image_id)) block->config.SetConstantPose(image_id);
        if (constant_tvecs.count(image_id)) block->config.SetConstantTvec(image_id, constant_tvecs[image_id]);
        return true;
    };

    // image2mappoints
    const auto &all_mappoints = reconstruction->MapPointIds();
    typedef std::unordered_map<image_t, std::unordered_set<mappoint_t>> IMAGE2MAP;
    typedef std::unordered_map<image_pair_t, int> CORRESMAP;

    int threads_num = GetEffectiveNumThreads(-1);

    std::vector<mappoint_t> all_mappoints_vec;
    all_mappoints_vec.assign(all_mappoints.begin(), all_mappoints.end());

    std::cout<<"Covisibility computation thread: "<< threads_num << std::endl;
    std::vector<IMAGE2MAP> omp_image2mappoints(threads_num);
    std::vector<CORRESMAP> omp_correspondence_mapper(threads_num);
    std::vector<std::vector<mappoint_t>> mappoint_threads(threads_num);

    IMAGE2MAP image2mappoints;
    CORRESMAP correspondence_mapper;

    #pragma omp parallel for schedule(dynamic) num_threads(threads_num)
    for (int p = 0; p < all_mappoints_vec.size(); ++ p) {
        int tid = omp_get_thread_num();

        auto &tmp_image2mappoints = omp_image2mappoints[tid];
        auto &tmp_correspondence_mapper = omp_correspondence_mapper[tid];

        const auto map_id = all_mappoints_vec[p];
        class MapPoint& mappoint = reconstruction->MapPoint(map_id);
        class Track& track = mappoint.Track();
        int track_length = track.Length();
        const auto &track_eles = track.Elements();

        for (size_t i = 0; i < track_length; ++i) {
            const TrackElement& track_el = track_eles[i];
            tmp_image2mappoints[track_el.image_id].insert(map_id);
         
        }
        if(track_length > 200){
            continue;
        }
        for (size_t i = 0; i < track_length; ++ i) {
            for (size_t j = i + 1; j < track_length; ++ j) {
                image_t s = track_eles[i].image_id;
                image_t e = track_eles[j].image_id;

                const image_pair_t pair_id = sensemap::utility::ImagePairToPairId(s, e);
                tmp_correspondence_mapper[pair_id]++;
            }
        }
    }

    for (int i = 0; i < threads_num; ++ i) {
        const auto &tmp_image2mappoints = omp_image2mappoints[i];
        const auto &tmp_correspondence_mapper = omp_correspondence_mapper[i];

        for (const auto item : tmp_image2mappoints) {
            image2mappoints[item.first].insert(item.second.begin(), item.second.end());
        }

        for (const auto item : tmp_correspondence_mapper) {
            int cnt = 0;
            if (correspondence_mapper.count(item.first)) cnt = correspondence_mapper[item.first];
            correspondence_mapper[item.first] = cnt + item.second;
        }
    }

    KMeans kmeans;
    size_t cluster_num = (images.size() - 1) / block_options_.max_block_images + 1;
    kmeans.SetK(cluster_num);
    kmeans.SetFixedSize(block_options_.block_images_num);
    
    for (const auto &img : images) {
        const auto &image = reconstruction->Image(img);
        const auto &r = image.RotationMatrix();
        const auto &t = image.Tvec();
        const auto &location = -r.transpose() * t;

        Tuple tuple;
        tuple.id = img;
        tuple.location = location;
        kmeans.mv_pntcloud.emplace_back(tuple);
    }
    bool kmeans_res = kmeans.SameSizeCluster();
    if (!kmeans_res) {
        return ;
    }

    // divide blocks
    std::unordered_set<image_t> valid_images;
    for (int i = 0; i < kmeans.mv_center.size(); ++ i) {
        // create new block
        Block block;
        block.id = i;

        // find max groups
        const auto &tuples = kmeans.m_grp_pntcloud[i];
        DisjointSet jset(tuples.size());
        for (int ti = 0; ti < tuples.size(); ++ ti) {
            for (int tj = ti + 1; tj < tuples.size(); ++ tj) {
                const image_pair_t pair_id = sensemap::utility::ImagePairToPairId(tuples[ti].id, tuples[tj].id);
                if (correspondence_mapper.count(pair_id)) {
                    jset.Merge(ti, tj);
                }
            }
        }
        const auto &max_group_idxs = jset.MaxComponent();
        std::unordered_set<mappoint_t> visited_mappoints;
        for (const auto &idx : max_group_idxs) {
            image_t cur_image = tuples[idx].id;

            // 1. add image / pose / tvec
            AddImages(&block, cur_image, constant_poses, constant_tvecs);

            valid_images.insert(cur_image);
        }
        blocks->push_back(block);
    }

    std::swap(*corres_map, correspondence_mapper);

    const auto & correspondence_graph = reconstruction->GetCorrespondenceGraph();

    // remain images
    std::unordered_set<image_t> remain_images;
    for (const auto &img : images) {
        if (valid_images.count(img)) continue;
        remain_images.insert(img);
    }
    int prev_remain_imgsize = remain_images.size();
    while (!remain_images.empty()) {
        for (const auto img : remain_images) {
            block_t target_block_id = -1;
            size_t max_covisible = 0;
            // for (const auto &block : blocks) {
            for (size_t i = 0; i < blocks->size(); ++i) {
                auto & block = blocks->at(i);
                const auto &block_imgs = block.config.Images();

                size_t covisible_cnt = 0;
                for (const auto &blk_img : block_imgs) {
                    const image_pair_t pair_id = sensemap::utility::ImagePairToPairId(img, blk_img);

                    if (correspondence_mapper.count(pair_id) == 0) continue;

                    auto covis = correspondence_mapper[pair_id];
                    if (covisible_cnt < covis) covisible_cnt = covis;
                }

                if (covisible_cnt > max_covisible) {
                    max_covisible = covisible_cnt;
                    target_block_id = block.id;
                }
            }

            if (max_covisible == 0) {
                for (size_t i = 0; i < blocks->size(); ++i) {
                    auto & block = blocks->at(i);
                    const auto &block_imgs = block.config.Images();

                    size_t covisible_cnt = 0;
                    for (const auto &blk_img : block_imgs) {
                        const image_pair_t pair_id = sensemap::utility::ImagePairToPairId(img, blk_img);
                        size_t num_corrs = correspondence_graph->NumCorrespondencesBetweenImages(img, blk_img);
                        if (num_corrs == 0) continue;
                        if (covisible_cnt < num_corrs) covisible_cnt = num_corrs;
                    }
                    if (covisible_cnt > max_covisible) {
                        max_covisible = covisible_cnt;
                        target_block_id = block.id;
                    }
                }
            }
            
	        if (max_covisible == 0) continue;
            auto block = blocks->at(target_block_id);

            const auto &mappoints = image2mappoints[img];
            // add to block
            // 1. add image / pose / tvec
            AddImages(&block, img, constant_poses, constant_tvecs);
        
            valid_images.insert(img);
        }

        remain_images.clear();
        for (const auto &img : images) {
            if (valid_images.count(img)) continue;
            remain_images.insert(img);
        }

        if (remain_images.size() == prev_remain_imgsize) {
            std::cout << "BlockBA Err : can not find block for images: " << std::endl;
            for (auto img : remain_images) {
                std::cout << img << ' ';
            }
            std::cout << std::endl;
            return ;
        }

        prev_remain_imgsize = remain_images.size();
    }

    std::cout << StringPrintf("Blocks: %d\n", blocks->size());
    for (size_t i = 0; i < blocks->size(); ++i) {
        auto block = blocks->at(i);
        std::cout << block.id << " " << block.config.NumImages() << std::endl;
    }
    std::cout << StringPrintf("Divide Blocks Time Cost: %.3fsec\n", timer.ElapsedSeconds());
}

bool BlockBundleAdjuster::SolveBlock(Reconstruction* reconstruction, Block* block) {
    std::unique_ptr<ceres::Problem> problem;
    problem.reset(new ceres::Problem());
    ceres::LossFunction *loss_function = options_.CreateLossFunction();
    ceres::Solver::Summary summary;

    SetUp(reconstruction, problem.get(), loss_function, block);

    if (problem->NumResiduals() == 0) {
        return 0;
    }
    
    ceres::Solver::Options solver_options = options_.solver_options;

    // Empirical choice.
    const size_t kMaxNumImagesDirectSparseSolver = 25000;
    const size_t num_images = config_.NumImages();
    if (num_images <= kMaxNumImagesDirectSparseSolver) {
        solver_options.linear_solver_type = ceres::SPARSE_SCHUR;
    } else {  // Indirect sparse (preconditioned CG) solver.
        solver_options.linear_solver_type = ceres::ITERATIVE_SCHUR;
        solver_options.preconditioner_type = ceres::SCHUR_JACOBI;
    }
    
    solver_options.num_threads = GetEffectiveNumThreads(-1);
#if CERES_VERSION_MAJOR < 2
    solver_options.num_linear_solver_threads = GetEffectiveNumThreads(-1);
#endif  // CERES_VERSION_MAJOR

    std::string solver_error;
    CHECK(solver_options.IsValid(&solver_error)) << solver_error;

    ceres::Solve(solver_options, problem.get(), &summary);

    if (solver_options.minimizer_progress_to_stdout) {
        std::cout << std::endl;
    }

    if (options_.print_summary) {
        PrintHeading2("Bundle adjustment report");
        PrintSolverSummary(summary);
    }
    return true;
}

void BlockBundleAdjuster::SetUp(Reconstruction* reconstruction, 
                                ceres::Problem* problem, 
                                ceres::LossFunction *loss_function, 
                                Block* block) {
    // Warning: AddPointsToProblem assumes that AddImageToProblem is called first.
    // Do not change order of instructions!
    for (const image_t image_id : block->config.Images()) {
        AddImageToProblem(image_id, reconstruction, problem, loss_function, block);
    }
    for (const auto mappoint_id : block->config.VariablePoints()) {
        AddPointToProblem(mappoint_id, reconstruction, problem, loss_function, block);
    }
    for (const auto mappoint_id : block->config.ConstantPoints()) {
        AddPointToProblem(mappoint_id, reconstruction, problem, loss_function, block);
    }
    std::cout << "plane constrain: " << options_.plane_constrain << std::endl;
    std::cout << "gba weighted options: " << options_.gba_weighted << std::endl;
    std::cout << "use prior relative pose: " << options_.use_prior_relative_pose << std::endl;
    std::cout << "use gps prior: " << options_.use_prior_absolute_location << " prior weight: "
              << options_.prior_absolute_location_weight<<std::endl;
    std::cout << "refine focal length, principal point, extra params, local extrinsics: "
              << options_.refine_focal_length << " " << options_.refine_principal_point << " "
              << options_.refine_extra_params << " " << options_.refine_local_extrinsics << std::endl;
    ParameterizeCameras(reconstruction, problem, block);
    // if (options_.parameterize_points_with_track) {
    ParameterizePoints(reconstruction, problem, block);
    // }
    // else {
    //     ParameterizePointsWithoutTrack(reconstruction, problem, block);
    // }
    ParameterizePoses(reconstruction, problem, block);
}

void BlockBundleAdjuster::AddImageToProblem(const image_t image_id, 
                                            Reconstruction *reconstruction, 
                                            ceres::Problem* problem,
                                            ceres::LossFunction *loss_function, 
                                            Block* block) {
    BundleAdjustmentConfig config = block->config;

    Image &image = reconstruction->Image(image_id);
    Camera &camera = reconstruction->Camera(image.CameraId());
    // CostFunction assumes unit quaternions.
    image.NormalizeQvec();

    double *qvec_data = image.Qvec().data();
    double *tvec_data = image.Tvec().data();
    double *camera_params_data = camera.ParamsData();

    Eigen::Matrix3d R = image.RotationMatrix();
    Eigen::Vector3d t = image.Tvec();

    double *local_qvec1_data;
    double *local_tvec1_data;
    double *local_qvec2_data;
    double *local_tvec2_data;

    const bool constant_pose = !options_.refine_extrinsics || config.HasConstantPose(image_id);

    const bool constant_intrinsics =
        !options_.refine_focal_length && !options_.refine_extra_params && !options_.refine_principal_point;

    double weight = 1.0;

    const std::vector<uint32_t> &local_image_indices = image.LocalImageIndices();
    std::vector<double *> local_qvec_data;
    std::vector<double *> local_tvec_data;
    std::vector<double *> local_camera_params_data;
    // This is a camera-rig
    if (camera.NumLocalCameras() > 1) {
        int local_param_size = camera.LocalParams().size() / camera.NumLocalCameras();

        for (size_t i = 0; i < camera.NumLocalCameras(); ++i) {
            local_qvec_data.push_back(camera.LocalQvecsData() + 4 * i);
            local_tvec_data.push_back(camera.LocalTvecsData() + 3 * i);
            local_camera_params_data.push_back(camera.LocalIntrinsicParamsData() + i * local_param_size);
        }
    }

    std::unordered_set<int> local_camera_in_the_image;

    const double eps = 1e-3;

    // Add residuals to bundle adjustment problem.
    size_t num_observations = 0;
    int point2d_id = -1;
    for (const Point2D &point2D : image.Points2D()) {
        point2d_id++;
        if (!point2D.HasMapPoint() || 
            block->mappoints_in_mask.find(point2D.MapPointId()) != block->mappoints_in_mask.end()) {
            continue;
        }

        uint32_t local_camera_id = local_image_indices[point2d_id];

        num_observations += 1;
        block->mappoint_num_observations[point2D.MapPointId()] += 1;

        MapPoint &mappoint = reconstruction->MapPoint(point2D.MapPointId());
        assert(mappoint.Track().Length() > 1);

        Eigen::Vector3d point3D_mea;  
        const float info = GetDepthPointAndWeight(camera, point2D, point3D_mea);    
        double observation_factor = 1.0;
        if (point2D.InOverlap()) {
            Eigen::Vector4d local_qvec(local_qvec_data[local_camera_id]);
            Eigen::Vector3d local_tvec(local_tvec_data[local_camera_id]);
            Eigen::Matrix3d local_R = QuaternionToRotationMatrix(local_qvec);
            Eigen::Vector3d Xc = local_R *(R * mappoint.XYZ() + t) + local_tvec;
            observation_factor = 1.0 + 9 * std::exp(-Xc[2] * Xc[2] / 10.0);
            // if (Xc[2] < 1.5) {
            //     observation_factor = 10.0f;
            // }
        }

        ceres::CostFunction *cost_function = nullptr;

        if (constant_pose) {
            if (camera.NumLocalCameras() > 1) {
                // for camera-rig
                if (camera.ModelName().compare("OPENCV_FISHEYE") == 0 && constant_intrinsics) {
                    local_camera_in_the_image.insert(local_camera_id);

                    double f = camera.LocalMeanFocalLength(local_camera_id);
                    Eigen::Vector3d bearing = camera.LocalImageToBearing(local_camera_id, point2D.XY());
                    cost_function =
                        LargeFovRigBundleAdjustmentConstantPoseCostFunction<OpenCVFisheyeCameraModel>::Create(
                            image.Qvec(), image.Tvec(), bearing, f);

                    problem->AddResidualBlock(cost_function, loss_function, local_qvec_data[local_camera_id],
                                               local_tvec_data[local_camera_id], mappoint.XYZ().data());

                } else {
                    switch (camera.ModelId()) {
#define CAMERA_MODEL_CASE(CameraModel)                                                    \
    case CameraModel::kModelId:                                                           \
        cost_function = RigBundleAdjustmentConstantPoseCostFunction<CameraModel>::Create( \
            image.Qvec(), image.Tvec(), point2D.XY(), observation_factor);                \
        break;
                        CAMERA_MODEL_SWITCH_CASES
#undef CAMERA_MODEL_CASE
                    }
                    local_camera_in_the_image.insert(local_camera_id);

                    problem->AddResidualBlock(cost_function, loss_function, local_qvec_data[local_camera_id],
                                               local_tvec_data[local_camera_id], mappoint.XYZ().data(),
                                               local_camera_params_data[local_camera_id]);
                }

            } else {
                // for monocular camera
                if (camera.ModelName().compare("SPHERICAL") == 0) {
                    Eigen::Vector3d bearing = camera.ImageToBearing(point2D.XY());
                    double f = camera.FocalLength();

                    cost_function = SphericalBundleAdjustmentConstantPoseCostFunction<SphericalCameraModel>::Create(
                        image.Qvec(), image.Tvec(), bearing, f);

                    local_qvec2_data = camera_params_data + 10;
                    local_tvec2_data = camera_params_data + 14;

                    problem->AddResidualBlock(cost_function, loss_function, mappoint.XYZ().data(),
                                               local_qvec2_data, local_tvec2_data);
                } else if (camera.ModelName().compare("OPENCV_FISHEYE") == 0 && constant_intrinsics){
                    Eigen::Vector3d bearing = camera.ImageToBearing(point2D.XY());
                    double f = camera.MeanFocalLength();
                    cost_function = LargeFovBundleAdjustmentConstantPoseCostFunction<OpenCVFisheyeCameraModel>::Create(
                        image.Qvec(), image.Tvec(), bearing, f, weight);

                    problem->AddResidualBlock(cost_function, loss_function, mappoint.XYZ().data());
                } else {
                    switch (camera.ModelId()) {
#define CAMERA_MODEL_CASE(CameraModel)                                                                            \
    case CameraModel::kModelId:                                                                                   \
        cost_function = BundleAdjustmentConstantPoseCostFunction<CameraModel>::Create(image.Qvec(), image.Tvec(), \
                                                                                      point2D.XY(), weight);      \
        break;
                        CAMERA_MODEL_SWITCH_CASES
#undef CAMERA_MODEL_CASE
                    }
                    problem->AddResidualBlock(cost_function, loss_function, mappoint.XYZ().data(), camera_params_data);
                    if (point3D_mea.z() > 0 && reconstruction->depth_enabled) {
                        cost_function =
                            BundleAdjustmentConstantPoseDepthCostFunction::Create(
                                image.Qvec(), image.Tvec(), point3D_mea, options_.rgbd_ba_depth_weight * info
                            );
                        problem->AddResidualBlock(cost_function, loss_function, 
                                                mappoint.XYZ().data());
                    }
                }
            }
        } else {
            if (camera.NumLocalCameras() > 1) {
                // for camera-rig
                if (camera.ModelName().compare("OPENCV_FISHEYE") == 0 && constant_intrinsics) {
                    local_camera_in_the_image.insert(local_camera_id);

                    double f = camera.LocalMeanFocalLength(local_camera_id);
                    Eigen::Vector3d bearing = camera.LocalImageToBearing(local_camera_id, point2D.XY());
                    cost_function =
                        LargeFovRigBundleAdjustmentCostFunction<OpenCVFisheyeCameraModel>::Create(bearing, f);
                    problem->AddResidualBlock(cost_function, loss_function, qvec_data, tvec_data,
                                               local_qvec_data[local_camera_id], local_tvec_data[local_camera_id],
                                               mappoint.XYZ().data());
                } else {
                    switch (camera.ModelId()) {
#define CAMERA_MODEL_CASE(CameraModel)                                                      \
    case CameraModel::kModelId:                                                             \
        cost_function = RigBundleAdjustmentCostFunction<CameraModel>::Create(point2D.XY(), observation_factor); \
        break;
                        CAMERA_MODEL_SWITCH_CASES
#undef CAMERA_MODEL_CASE
                    }
                    local_camera_in_the_image.insert(local_camera_id);
                    problem->AddResidualBlock(cost_function, loss_function, qvec_data, tvec_data,
                                               local_qvec_data[local_camera_id], local_tvec_data[local_camera_id],
                                               mappoint.XYZ().data(), local_camera_params_data[local_camera_id]);
                }

            } else {
                if (camera.ModelName().compare("SPHERICAL") == 0) {
                    Eigen::Vector3d bearing = camera.ImageToBearing(point2D.XY());
                    double f = camera.FocalLength();
                    
                    cost_function =
                        SphericalBundleAdjustmentCostFunction<SphericalCameraModel>::Create(bearing, f);

                    local_qvec2_data = camera_params_data + 10;
                    local_tvec2_data = camera_params_data + 14;

                    problem->AddResidualBlock(cost_function, loss_function, qvec_data, tvec_data,
                                               mappoint.XYZ().data(), local_qvec2_data, local_tvec2_data);
                } else if(camera.ModelName().compare("OPENCV_FISHEYE") == 0 && constant_intrinsics){ 
                    Eigen::Vector3d bearing = camera.ImageToBearing(point2D.XY());
                    double f = camera.MeanFocalLength();
                    cost_function =
                        LargeFovBundleAdjustmentCostFunction<OpenCVFisheyeCameraModel>::Create(bearing, f, weight);
                    problem->AddResidualBlock(cost_function, loss_function, qvec_data, tvec_data,
                                               mappoint.XYZ().data());

                } else {
                    switch (camera.ModelId()) {
#define CAMERA_MODEL_CASE(CameraModel)                                                           \
    case CameraModel::kModelId:                                                                  \
        cost_function = BundleAdjustmentCostFunction<CameraModel>::Create(point2D.XY(), weight); \
        break;
                        CAMERA_MODEL_SWITCH_CASES
#undef CAMERA_MODEL_CASE
                    }
                    problem->AddResidualBlock(cost_function, loss_function, qvec_data, tvec_data,
                                               mappoint.XYZ().data(), camera_params_data);
                    if (point3D_mea.z() > 0 && reconstruction->depth_enabled) {
                        cost_function = 
                            BundleAdjustmentDepthCostFunction::Create(point3D_mea, options_.rgbd_ba_depth_weight * info);
                        problem->AddResidualBlock(cost_function, loss_function, 
                                                qvec_data, tvec_data, 
                                                mappoint.XYZ().data());
                    }
                }
            }
        }
    }

    if (options_.refine_local_extrinsics && options_.local_relative_translation_constraint && camera.NumLocalCameras() > 1) {
        Eigen::Vector3d weight(num_observations * 0.2f, num_observations * 0.2f, num_observations * 0.2f);
        for (size_t i = 1; i < camera.NumLocalCameras(); ++i) {
            ceres::CostFunction *cost_function = RigLocalRelativeBundleAdjustmentPoseFunction::Create(weight);
            problem->AddResidualBlock(cost_function, loss_function, local_qvec_data[0], local_tvec_data[0], 
                                       local_qvec_data[i], local_tvec_data[i]);
        }
    }

    // add plane constrain
    bool on_plane = reconstruction->planes_for_images.find(image_id) != reconstruction->planes_for_images.end();
    if (!constant_pose && options_.plane_constrain && on_plane) {
        Eigen::Vector4d plane = reconstruction->planes_for_images.at(image_id);
        ceres::CostFunction *cost_function = PlaneConstrainCostFunction::Create(
            plane, reconstruction->baseline_distance, num_observations, options_.plane_weight);

        problem->AddResidualBlock(cost_function, loss_function, qvec_data, tvec_data);
    }

    // add prior relative pose constrain
    if (!constant_pose && options_.use_prior_relative_pose) {
        image_t neighbor_image_id = image_id + 1;

        for (image_t i = 1; i < 10; ++i) {
            bool relative_pose_valid = false;

            relative_pose_valid =
                (config.Images().count(neighbor_image_id) > 0) &&
                (reconstruction->prior_rotations.find(image_id) != reconstruction->prior_rotations.end()) &&
                (reconstruction->prior_rotations.find(neighbor_image_id) != reconstruction->prior_rotations.end());

            if (relative_pose_valid) {
                Eigen::Vector4d prior_qvec1 = reconstruction->prior_rotations.at(image_id);
                Eigen::Vector3d prior_tvec1 = reconstruction->prior_translations.at(image_id);
                Eigen::Matrix3d prior_R1 = QuaternionToRotationMatrix(prior_qvec1);

                Eigen::Vector4d prior_qvec2 = reconstruction->prior_rotations.at(neighbor_image_id);
                Eigen::Vector3d prior_tvec2 = reconstruction->prior_translations.at(neighbor_image_id);
                Eigen::Matrix3d prior_R2 = QuaternionToRotationMatrix(prior_qvec2);

                Eigen::Matrix3d prior_relative_R12 = prior_R2 * prior_R1.transpose();
                Eigen::Vector3d prior_relative_t12 = prior_tvec2 - prior_relative_R12 * prior_tvec1;
                Eigen::Vector4d prior_relative_q12 = RotationMatrixToQuaternion(prior_relative_R12);

                Image &neighbor_image = reconstruction->Image(neighbor_image_id);

                // CostFunction assumes unit quaternions.
                neighbor_image.NormalizeQvec();
                double *qvec_data2 = neighbor_image.Qvec().data();
                double *tvec_data2 = neighbor_image.Tvec().data();

                if (options_.use_prior_distance_only) {
                    ceres::CostFunction *cost_function = PriorRelativeDistanceCostFunction::Create(
                        prior_relative_q12, prior_relative_t12,
                        static_cast<double>(num_observations) * options_.prior_pose_weight);
                    problem->AddResidualBlock(cost_function, loss_function, qvec_data, tvec_data, qvec_data2,
                                               tvec_data2);
                } else if (options_.use_prior_translation_only) {
                    ceres::CostFunction *cost_function = PriorRelativeTranslationCostFunction::Create(
                        prior_relative_q12, prior_relative_t12,
                        static_cast<double>(num_observations) * options_.prior_pose_weight);
                    problem->AddResidualBlock(cost_function, loss_function, qvec_data, tvec_data, qvec_data2,
                                               tvec_data2);
                } else {
                    ceres::CostFunction *cost_function = PriorRelativePoseCostFunction::Create(
                        prior_relative_q12, prior_relative_t12,
                        static_cast<double>(num_observations) * options_.prior_pose_weight);
                    problem->AddResidualBlock(cost_function, loss_function, qvec_data, tvec_data, qvec_data2,
                                               tvec_data2);
                }

                if (block->invovled_image_extrinsics.count(image_id) == 0) {
                    block->invovled_image_extrinsics.insert(image_id);
                }

                if (block->invovled_image_extrinsics.count(neighbor_image_id) == 0) {
                    block->invovled_image_extrinsics.insert(neighbor_image_id);
                }
            }
            neighbor_image_id++;
        }
    }

    // add prior relative pose constrain
    if (options_.use_prior_aggressively && !constant_pose && options_.use_prior_relative_pose && 
        (reconstruction->prior_rotations.find(image_id) != reconstruction->prior_rotations.end()) &&
        (reconstruction->prior_translations.find(image_id) != reconstruction->prior_translations.end())) {
        
        // find neighbors that have prior poses in [-100, +100], maximum 30
        size_t relative_count = 0; 
        for (size_t i = 2; i < 202 && relative_count < 30; ++i) {
            image_t neighbor_image_id;
            if (i % 2) {
                neighbor_image_id = image_id + i / 2;
            }
            else {
                neighbor_image_id = image_id - i / 2;
            }

            // farther frames have less weight
            double distance_weight = std::exp(-0.0002 * (i / 2) * (i / 2));

            // By default, we apply 1 prior constraint for every 1 valid ovservation, 
            // i.e., the basic weight is exactly `num_observations`.
            double ovservation_weight = num_observations;

            // However, images with suffecient/insuffecient ovservations should consider less/more on priors. 
            ovservation_weight *= 3.0 / (1.5 + std::log(num_observations * num_observations * 0.0003 + 1));
            
            // The final weight
            double weight = distance_weight * ovservation_weight * options_.prior_pose_weight;

            bool relative_pose_valid = false;
            relative_pose_valid =
                (config.Images().count(neighbor_image_id) > 0) &&
                (reconstruction->prior_rotations.find(neighbor_image_id) != reconstruction->prior_rotations.end());

            if (relative_pose_valid) {
                relative_count++;

                Eigen::Vector4d prior_qvec1 = reconstruction->prior_rotations.at(image_id);
                Eigen::Vector3d prior_tvec1 = reconstruction->prior_translations.at(image_id);
                Eigen::Matrix3d prior_R1 = QuaternionToRotationMatrix(prior_qvec1);

                Eigen::Vector4d prior_qvec2 = reconstruction->prior_rotations.at(neighbor_image_id);
                Eigen::Vector3d prior_tvec2 = reconstruction->prior_translations.at(neighbor_image_id);
                Eigen::Matrix3d prior_R2 = QuaternionToRotationMatrix(prior_qvec2);

                Eigen::Matrix3d prior_relative_R12 = prior_R2 * prior_R1.transpose();
                Eigen::Vector3d prior_relative_t12 = prior_tvec2 - prior_relative_R12 * prior_tvec1;
                Eigen::Vector4d prior_relative_q12 = RotationMatrixToQuaternion(prior_relative_R12);

                Image &neighbor_image = reconstruction->Image(neighbor_image_id);

                // CostFunction assumes unit quaternions.
                neighbor_image.NormalizeQvec();
                double *qvec_data2 = neighbor_image.Qvec().data();
                double *tvec_data2 = neighbor_image.Tvec().data();

                if (options_.use_prior_distance_only) {
                    ceres::CostFunction *cost_function = PriorRelativeDistanceCostFunction::Create(
                        prior_relative_q12, prior_relative_t12, weight);
                    problem->AddResidualBlock(cost_function, loss_function, qvec_data, tvec_data, qvec_data2,
                                               tvec_data2);
                } else if (options_.use_prior_translation_only) {
                    ceres::CostFunction *cost_function = PriorRelativeTranslationCostFunction::Create(
                        prior_relative_q12, prior_relative_t12, weight);
                    problem->AddResidualBlock(cost_function, loss_function, qvec_data, tvec_data, qvec_data2,
                                               tvec_data2);
                } else {
                    ceres::CostFunction *cost_function = PriorRelativePoseCostFunction::Create(
                        prior_relative_q12, prior_relative_t12, weight);
                    problem->AddResidualBlock(cost_function, loss_function, qvec_data, tvec_data, qvec_data2,
                                               tvec_data2);
                }

                if (block->invovled_image_extrinsics.count(image_id) == 0) {
                    block->invovled_image_extrinsics.insert(image_id);
                }

                if (block->invovled_image_extrinsics.count(neighbor_image_id) == 0) {
                    block->invovled_image_extrinsics.insert(neighbor_image_id);
                }
            }
        }
    }

    // add icp relative pose
    if (!constant_pose && options_.use_icp_relative_pose && reconstruction->depth_enabled) {
        Image &image = reconstruction->Image(image_id);
        auto &icp_links = image.icp_links_;
        int icp_cnt = 0;
        for(auto lk:icp_links){
            if(!config.Images().count(lk.ref_id_)) continue;
            Image &lk_image = reconstruction->Image(lk.ref_id_);

            // CostFunction assumes unit quaternions.
            lk_image.NormalizeQvec();
            double *lk_qvec = lk_image.Qvec().data();
            double *lk_tvec = lk_image.Tvec().data();

//            double ovservation_weight = 0.1 + std::log(num_observations / 50.0 + 1.0);

            double weight =  lk.conf_ * options_.icp_base_weight;
            //  lk.infomation_ = Eigen::Matrix6d::Identity();
            ceres::CostFunction *cost_function = ICPRelativePoseCostFunction::Create(
                    lk.X_, lk.infomation_, weight);
            problem->AddResidualBlock(cost_function, new ceres::HuberLoss(options_.icp_base_weight/5.0), qvec_data, tvec_data, lk_qvec,
                                       lk_tvec);

            std::vector<double*> paras={qvec_data, tvec_data, lk_qvec, lk_tvec};
            icp_cnt++;
            if (block->invovled_image_extrinsics.count(lk.ref_id_) == 0) {
                block->invovled_image_extrinsics.insert(lk.ref_id_);
            }
        }
        if (icp_cnt && block->invovled_image_extrinsics.count(image_id) == 0) {
            block->invovled_image_extrinsics.insert(image_id);
        }
    }

    // add gravity
    if (!constant_pose && options_.use_gravity) {
        Image &image = reconstruction->Image(image_id);
        auto &cur_g = image.gravity_;
        auto &world_g = gravity_;
        if(!cur_g.isZero() && !world_g.isZero()){
            auto loss_func =  new ceres::HuberLoss(0.01);
            ceres::CostFunction *cost_function = GravityCostFunction::Create(
                    world_g, cur_g, Eigen::Matrix3d::Identity(), -1.0, options_.gravity_base_weight);
            problem->AddResidualBlock(cost_function, nullptr, qvec_data);
            std::vector<double*> paras={qvec_data};
        }
    }

    // add time domain smoothing
    if (!constant_pose && options_.use_time_domain_smoothing) {
        Image &image = reconstruction->Image(image_id);
        const long long timestamp = image.timestamp_;

        if (timestamp > 0) {
            image_t prev_image_id = -1;
            image_t next_image_id = -1;

            for (size_t i = 1; i <= 10 && prev_image_id == -1; ++i) {
                image_t neighbor_image_id = image_id - i;
                if (config.Images().count(neighbor_image_id) > 0) {
                    prev_image_id = neighbor_image_id;
                }
            }

            for (size_t i = 1; i <= 10 && next_image_id == -1; ++i) {
                image_t neighbor_image_id = image_id + i;
                if (config.Images().count(neighbor_image_id) > 0) {
                    next_image_id = neighbor_image_id;
                }
            }

            if (prev_image_id != -1 && next_image_id != -1) {
                Image &prev_image = reconstruction->Image(prev_image_id);
                Image &next_image = reconstruction->Image(next_image_id);
                const long long prev_timestamp = prev_image.timestamp_;
                const long long next_timestamp = next_image.timestamp_;

                if (prev_timestamp < timestamp && timestamp < next_timestamp) {
                    // a large time-range has smaller weight due to uncertainty
                    // +/-0.1s ~= 60% of the base weight
                    const long long prev_time = timestamp - prev_timestamp;
                    const long long next_time = next_timestamp - timestamp;
                    const double weight = options_.time_domain_smoothing_weight * 
                        std::exp(-5.0 * prev_time) *
                        std::exp(-5.0 * next_time);

                    ceres::CostFunction *cost_function = TimeDomainSmoothingCostFunction::Create(
                        prev_time, next_time, weight);
                    problem->AddResidualBlock(cost_function, loss_function, 
                        qvec_data, tvec_data, 
                        prev_image.Qvec().data(), prev_image.Tvec().data(),
                        next_image.Qvec().data(), next_image.Tvec().data());
                }
            }
        }
    }

    if (!constant_pose && options_.use_prior_absolute_location && reconstruction->b_aligned) {
        // add prior absolute location constrain from gps prior
        if (reconstruction->prior_locations_gps.find(image_id) != reconstruction->prior_locations_gps.end()) {
            
            CHECK(reconstruction->prior_locations_gps_inlier.find(image_id) !=
                  reconstruction->prior_locations_gps_inlier.end());
            CHECK(reconstruction->prior_horizontal_locations_gps_inlier.find(image_id) !=
                  reconstruction->prior_horizontal_locations_gps_inlier.end());
            
            if (reconstruction->prior_locations_gps_inlier.at(image_id)) {
                Eigen::Vector3d prior_c = reconstruction->prior_locations_gps.at(image_id).first;
                
                ceres::CostFunction *cost_function = nullptr;
                if(options_.optimization_use_horizontal_gps_only){
                    cost_function = PriorAbsoluteLocationOnPlaneCostFunction::Create(
                        prior_c, reconstruction->projection_plane, options_.prior_absolute_location_weight);
                }
                else{
                    cost_function =
                        PriorAbsoluteLocationCostFunction::Create(prior_c, options_.prior_absolute_location_weight);
                }
                problem->AddResidualBlock(cost_function, loss_function, qvec_data, tvec_data);
            }
            else if (reconstruction->prior_horizontal_locations_gps_inlier.at(image_id)) {
                Eigen::Vector3d prior_c = reconstruction->prior_locations_gps.at(image_id).first;
                prior_c[2] = 0;

                ceres::CostFunction *cost_function = nullptr;
                cost_function = PriorAbsoluteLocationOnPlaneCostFunction::Create(
                    prior_c, reconstruction->projection_plane, options_.prior_absolute_location_weight);

                problem->AddResidualBlock(cost_function, loss_function, qvec_data, tvec_data);
            }
        }

        // add prior absolute pose constrain from RTK prior
        if (image.HasQvecPrior() && image.HasTvecPrior() && image.RtkFlag() == 50) {

            Eigen::Vector4d prior_q = image.QvecPrior();
            // Eigen::Vector3d prior_c = -QuaternionToRotationMatrix(prior_q) * image.TvecPrior();
            Eigen::Vector3d prior_c = image.TvecPrior();

            ceres::ScaledLoss *rtk_loss_function = 
                    new ceres::ScaledLoss(new ceres::TrivialLoss(), 
                    static_cast<double>(std::max(num_observations, (size_t)50)), ceres::DO_NOT_TAKE_OWNERSHIP);
            
            // double weight_q = options_.prior_absolute_location_weight * 0.1;
            double weight_q = 0.0;
            double weight_x = image.HasRtkStd() ? options_.prior_absolute_location_weight / (image.RtkStdLat() + eps) : 
                                                    options_.prior_absolute_location_weight * 10.0;
            double weight_y = image.HasRtkStd() ? options_.prior_absolute_location_weight / (image.RtkStdLon() + eps) :
                                                    options_.prior_absolute_location_weight * 10.0;
            double weight_z = image.HasRtkStd() ? options_.prior_absolute_location_weight / (image.RtkStdHgt() + eps) :
                                                    options_.prior_absolute_location_weight * 10.0;

            if (image.PriorQvecGood()) {
                ceres::CostFunction *cost_function = PriorAbsolutePoseCostFunction::Create(
                    prior_q, prior_c, weight_q, weight_x, weight_y, weight_z);
                problem->AddResidualBlock(cost_function, rtk_loss_function, qvec_data, tvec_data);
            } else {
                ceres::CostFunction *cost_function = PriorAbsoluteLocationCostFunction::Create(
                    image.TvecPrior(), weight_x, weight_y, weight_z);
                problem->AddResidualBlock(cost_function, rtk_loss_function, qvec_data, tvec_data);
            }
            if (num_observations == 0) {
                // Set pose parameterization.
                if (!constant_pose) {
                    ceres::LocalParameterization *quaternion_parameterization = new ceres::QuaternionParameterization;
                    SetParameterization(problem, qvec_data, quaternion_parameterization);
                    if (config.HasConstantTvec(image_id)) {
                        const std::vector<int> &constant_tvec_idxs = config.ConstantTvec(image_id);
                        ceres::SubsetParameterization *tvec_parameterization =
                            new ceres::SubsetParameterization(3, constant_tvec_idxs);
                        SetParameterization(problem, tvec_data, tvec_parameterization);
                    }
                }
            }
        } else if (image.HasTvecPrior() && image.PriorInlier() && image.RtkFlag() != 50) {
            ceres::ScaledLoss *rtk_loss_function = 
                    new ceres::ScaledLoss(new ceres::TrivialLoss(), 
                    static_cast<double>(std::max(num_observations, (size_t)50)), ceres::DO_NOT_TAKE_OWNERSHIP);

            ceres::CostFunction *cost_function = PriorAbsoluteDistanceCostFunction::Create(
                image.TvecPrior(), options_.prior_absolute_location_weight);
            problem->AddResidualBlock(cost_function, rtk_loss_function, qvec_data, tvec_data);
            if (num_observations == 0) {
                // Set pose parameterization.
                if (!constant_pose) {
                    ceres::LocalParameterization *quaternion_parameterization = new ceres::QuaternionParameterization;
                    SetParameterization(problem, qvec_data, quaternion_parameterization);
                    if (config.HasConstantTvec(image_id)) {
                        const std::vector<int> &constant_tvec_idxs = config.ConstantTvec(image_id);
                        ceres::SubsetParameterization *tvec_parameterization =
                            new ceres::SubsetParameterization(3, constant_tvec_idxs);
                        SetParameterization(problem, tvec_data, tvec_parameterization);
                    }
                }
            }
        }
        // if (image.HasTvecPrior()) {
        //     if (image.RtkFlag() == 50) {

        //         double weight_x = image.HasRtkStd() ? options_.prior_absolute_location_weight / (image.RtkStdLat() + eps) : 
        //                                               options_.prior_absolute_location_weight * 10.0;
        //         double weight_y = image.HasRtkStd() ? options_.prior_absolute_location_weight / (image.RtkStdLon() + eps) :
        //                                               options_.prior_absolute_location_weight * 10.0;
        //         double weight_z = image.HasRtkStd() ? options_.prior_absolute_location_weight / (image.RtkStdHgt() + eps) :
        //                                               options_.prior_absolute_location_weight * 10.0;

        //         ceres::ScaledLoss *rtk_loss_function = 
        //                 new ceres::ScaledLoss(new ceres::TrivialLoss(), 
        //                 static_cast<double>(num_observations), ceres::DO_NOT_TAKE_OWNERSHIP);

        //         ceres::CostFunction *cost_function = PriorAbsoluteLocationCostFunction::Create(
        //             image.TvecPrior(), weight_x, weight_y, weight_z);
        //         problem->AddResidualBlock(cost_function, rtk_loss_function, qvec_data, tvec_data);
        //     } else if (image.PriorInlier()) {
        //         ceres::ScaledLoss *rtk_loss_function = 
        //                 new ceres::ScaledLoss(new ceres::TrivialLoss(), 
        //                 static_cast<double>(num_observations), ceres::DO_NOT_TAKE_OWNERSHIP);

        //         ceres::CostFunction *cost_function = PriorAbsoluteDistanceCostFunction::Create(
        //             image.TvecPrior(), options_.prior_absolute_location_weight);
        //         problem->AddResidualBlock(cost_function, rtk_loss_function, qvec_data, tvec_data);
        //     }
        // }
    }

    if (constant_pose && (constant_pose_images_.count(image_id) == 0)) {
        constant_pose_images_.insert(image_id);
    }

    if (num_observations > 0) {
        block->camera_ids.insert(image.CameraId());

        if (camera.NumLocalCameras() > 1) {
            if (block->local_camera_ids.find(image.CameraId()) != block->local_camera_ids.end()) {
                for (auto local_camera_id : local_camera_in_the_image) {
                    block->local_camera_ids.at(image.CameraId()).insert(local_camera_id);
                }
            } else {
                block->local_camera_ids.emplace(image.CameraId(), local_camera_in_the_image);
            }
        }

        // Set pose parameterization.
        if (!constant_pose) {
            ceres::LocalParameterization *quaternion_parameterization = new ceres::QuaternionParameterization;
            SetParameterization(problem, qvec_data, quaternion_parameterization);
            if (config.HasConstantTvec(image_id)) {
                const std::vector<int> &constant_tvec_idxs = config.ConstantTvec(image_id);
                ceres::SubsetParameterization *tvec_parameterization =
                    new ceres::SubsetParameterization(3, constant_tvec_idxs);
                SetParameterization(problem, tvec_data, tvec_parameterization);
            }
        }
    }
}

void BlockBundleAdjuster::AddPointToProblem(const mappoint_t mappoint_id,
                                            Reconstruction* reconstruction,
                                            ceres::Problem* problem,
                                            ceres::LossFunction* loss_function, 
                                            Block* block) {
    MapPoint &mappoint = reconstruction->MapPoint(mappoint_id);

    // Is Map Point already fully contained in the problem? I.e. its entire track
    // is contained in `variable_image_ids`, `constant_image_ids`,
    // `constant_x_image_ids`.

    const bool constant_intrinsics =
        !options_.refine_focal_length && !options_.refine_extra_params && !options_.refine_principal_point;

    if (block->mappoint_num_observations[mappoint_id] == mappoint.Track().Length() || 
        block->mappoints_in_mask.find(mappoint_id) != block->mappoints_in_mask.end()) {
        return;
    }

    for (const auto &track_el : mappoint.Track().Elements()) {
        // Skip observations that were already added in `FillImages`.
        if (block->config.HasImage(track_el.image_id)) {
            continue;
        }

        block->mappoint_num_observations[mappoint_id] += 1;

        Image &image = reconstruction->Image(track_el.image_id);
        Camera &camera = reconstruction->Camera(image.CameraId());
        const Point2D &point2D = image.Point2D(track_el.point2D_idx);

        Eigen::Matrix3d R = image.RotationMatrix();
        Eigen::Vector3d t = image.Tvec();

        Eigen::Vector3d point3D_mea;
        const float info = GetDepthPointAndWeight(camera, point2D, point3D_mea);

        int local_image_id = image.LocalImageIndices()[track_el.point2D_idx];
        double *local_qvec_data;
        double *local_tvec_data;
        double *local_camera_params_data;

        double *camera_params_data = camera.ParamsData();

        // We do not want to refine the camera of images that are not
        // part of `constant_image_ids_`, `constant_image_ids_`,
        // `constant_x_image_ids_`.
        if (block->camera_ids.count(image.CameraId()) == 0) {
            block->camera_ids.insert(image.CameraId());
            block->config.SetConstantCamera(image.CameraId());
        }

        ceres::CostFunction *cost_function = nullptr;

        if (camera.NumLocalCameras() > 1) {
            // for camera-rig
            double observation_factor = 1.0;
            if (block->local_camera_ids.find(image.CameraId()) != block->local_camera_ids.end()) {
                block->local_camera_ids.at(image.CameraId()).insert(local_image_id);
            } else {
                std::unordered_set<int> local_image_id_set;
                local_image_id_set.insert(local_image_id);
                block->local_camera_ids.emplace(image.CameraId(), local_image_id_set);
            }

            int local_param_size = camera.LocalParams().size() / camera.NumLocalCameras();
            local_qvec_data = camera.LocalQvecsData() + 4 * local_image_id;
            local_tvec_data = camera.LocalTvecsData() + 3 * local_image_id;
            local_camera_params_data = camera.LocalIntrinsicParamsData() + local_image_id * local_param_size;

            if (point2D.InOverlap()) {
                Eigen::Vector4d local_qvec(local_qvec_data);
                Eigen::Vector3d local_tvec(local_tvec_data);
                Eigen::Matrix3d local_R = QuaternionToRotationMatrix(local_qvec);
                Eigen::Vector3d Xc = local_R *(R * mappoint.XYZ() + t) + local_tvec;
                observation_factor = 1.0 + 9 * std::exp(-Xc[2] * Xc[2] / 10.0);
            }

            if (camera.ModelName().compare("OPENCV_FISHEYE") == 0 && constant_intrinsics) {
                double f = camera.LocalMeanFocalLength(local_image_id);
                Eigen::Vector3d bearing = camera.LocalImageToBearing(local_image_id,point2D.XY());
                cost_function = LargeFovRigBundleAdjustmentConstantPoseCostFunction<OpenCVFisheyeCameraModel>::Create(
                    image.Qvec(), image.Tvec(), bearing, f);

                problem->AddResidualBlock(cost_function, loss_function, local_qvec_data, local_tvec_data,
                                           mappoint.XYZ().data());

            } else {
                switch (camera.ModelId()) {
#define CAMERA_MODEL_CASE(CameraModel)                                                    \
    case CameraModel::kModelId:                                                           \
        cost_function = RigBundleAdjustmentConstantPoseCostFunction<CameraModel>::Create( \
            image.Qvec(), image.Tvec(), point2D.XY(), observation_factor);                \
        break;
                    CAMERA_MODEL_SWITCH_CASES
#undef CAMERA_MODEL_CASE
                }

                problem->AddResidualBlock(cost_function, loss_function, local_qvec_data, local_tvec_data,
                                           mappoint.XYZ().data(), local_camera_params_data);
            }

        } else {
            // for monocular camera
            if (camera.ModelName().compare("SPHERICAL") == 0) {
                Eigen::Vector3d bearing = camera.ImageToBearing(point2D.XY());
                double f = camera.FocalLength();

                cost_function = SphericalBundleAdjustmentConstantPoseCostFunction<SphericalCameraModel>::Create(
                    image.Qvec(), image.Tvec(), bearing, f);

                double *local_qvec2_data = camera_params_data + 10;
                double *local_tvec2_data = camera_params_data + 14;

                problem->AddResidualBlock(cost_function, loss_function, mappoint.XYZ().data(),
                                           local_qvec2_data, local_tvec2_data);
            } else if (camera.ModelName().compare("OPENCV_FISHEYE") == 0 && constant_intrinsics){
                Eigen::Vector3d bearing = camera.ImageToBearing(point2D.XY());
                double f = camera.MeanFocalLength();
                cost_function = LargeFovBundleAdjustmentConstantPoseCostFunction<OpenCVFisheyeCameraModel>::Create(
                    image.Qvec(), image.Tvec(), bearing, f);

                problem->AddResidualBlock(cost_function, loss_function, mappoint.XYZ().data());
            } else {
                switch (camera.ModelId()) {
#define CAMERA_MODEL_CASE(CameraModel)                                                                               \
    case CameraModel::kModelId:                                                                                      \
        cost_function =                                                                                              \
            BundleAdjustmentConstantPoseCostFunction<CameraModel>::Create(image.Qvec(), image.Tvec(), point2D.XY()); \
        break;
                    CAMERA_MODEL_SWITCH_CASES
#undef CAMERA_MODEL_CASE
                }
                problem->AddResidualBlock(cost_function, loss_function, mappoint.XYZ().data(), camera.ParamsData());
                if (point3D_mea.z() > 0 && reconstruction->depth_enabled) {
                    cost_function =
                        BundleAdjustmentConstantPoseDepthCostFunction::Create(
                            image.Qvec(), image.Tvec(), point3D_mea, options_.rgbd_ba_depth_weight * info
                        );
                    problem->AddResidualBlock(cost_function, loss_function, 
                                                mappoint.XYZ().data());
                }
            }
        }
    }
}

void BlockBundleAdjuster::ParameterizeCameras(Reconstruction *reconstruction, ceres::Problem* problem, Block* block) {
    const bool constant_camera =
        !options_.refine_focal_length && !options_.refine_principal_point && !options_.refine_extra_params;
    

    for (const camera_t camera_id : block->camera_ids) {
        Camera &camera = reconstruction->Camera(camera_id);

        if (camera.NumLocalCameras() > 1) {
            
            if (block->local_camera_ids.find(camera_id) == block->local_camera_ids.end()) {
                continue;
            }

            std::unordered_set<int> & local_camera_set = block->local_camera_ids.at(camera_id);

            int local_param_size = camera.LocalParams().size() / camera.NumLocalCameras();

            for (auto local_camera_id : local_camera_set) {
                double *local_qvec_data = camera.LocalQvecsData() + 4 * local_camera_id;
                double *local_tvec_data = camera.LocalTvecsData() + 3 * local_camera_id;
                double *local_camera_params_data =
                    camera.LocalIntrinsicParamsData() + local_camera_id * local_param_size;

                if(!(camera.ModelName().compare("OPENCV_FISHEYE") == 0 && constant_camera)){
                    if (constant_camera || block->config.IsConstantCamera(camera_id)) {
                        problem->SetParameterBlockConstant(local_camera_params_data);
                    } else {
                        std::vector<int> const_camera_params;
                        if (!options_.refine_focal_length) {
                            const std::vector<size_t> &params_idxs = camera.FocalLengthIdxs();
                            const_camera_params.insert(const_camera_params.end(), params_idxs.begin(),
                                                       params_idxs.end());
                        }
                        if (!options_.refine_principal_point) {
                            const std::vector<size_t> &params_idxs = camera.PrincipalPointIdxs();
                            const_camera_params.insert(const_camera_params.end(), params_idxs.begin(),
                                                       params_idxs.end());
                        }
                        if ((!options_.refine_extra_params) && (camera.ModelName().compare("SPHERICAL") != 0)) {
                            const std::vector<size_t> &params_idxs = camera.ExtraParamsIdxs();
                            const_camera_params.insert(const_camera_params.end(), params_idxs.begin(),
                                                       params_idxs.end());
                        }

                        if (const_camera_params.size() > 0) {
                            ceres::SubsetParameterization *camera_params_parameterization =
                                new ceres::SubsetParameterization(static_cast<int>(local_param_size),
                                                                  const_camera_params);
                            SetParameterization(problem, local_camera_params_data, camera_params_parameterization);
                        }
                    }
                }

                if (options_.refine_local_extrinsics && local_camera_id > 0) {
                    ceres::LocalParameterization *quaternion_parameterization = new ceres::QuaternionParameterization;
                    SetParameterization(problem, local_qvec_data, quaternion_parameterization);
                } else {
                    problem->SetParameterBlockConstant(local_qvec_data);
                    problem->SetParameterBlockConstant(local_tvec_data);
                }
            }
            continue;
        }

        if (constant_camera || block->config.IsConstantCamera(camera_id)) {

            if(camera.ModelName().compare("SPHERICAL") != 0){
                problem->SetParameterBlockConstant(camera.ParamsData());
            }
            if (camera.ModelName().compare("SPHERICAL") == 0) {
                double *camera_params_data = camera.ParamsData();

                double *local_qvec2_data = camera_params_data + 10;
                double *local_tvec2_data = camera_params_data + 14;
                problem->SetParameterBlockConstant(local_qvec2_data);
                problem->SetParameterBlockConstant(local_tvec2_data);
            }
            continue;
        } else {
            if(!(camera.ModelName()=="SPHERICAL")){
                std::vector<int> const_camera_params;

                if (!options_.refine_focal_length) {
                    const std::vector<size_t> &params_idxs = camera.FocalLengthIdxs();
                    const_camera_params.insert(const_camera_params.end(), params_idxs.begin(), params_idxs.end());
                }
                if (!options_.refine_principal_point) {
                    const std::vector<size_t> &params_idxs = camera.PrincipalPointIdxs();
                    const_camera_params.insert(const_camera_params.end(), params_idxs.begin(), params_idxs.end());
                }
                if ((!options_.refine_extra_params) && (camera.ModelName().compare("SPHERICAL") != 0)) {
                    const std::vector<size_t> &params_idxs = camera.ExtraParamsIdxs();
                    const_camera_params.insert(const_camera_params.end(), params_idxs.begin(), params_idxs.end());
                }

                for (size_t idx : camera.FocalLengthIdxs()) {
                    double est_focal = camera.ParamsData()[idx];
                    problem->SetParameterLowerBound(camera.ParamsData(), idx,
                                                     options_.lower_bound_focal_length_factor * est_focal);
                    problem->SetParameterUpperBound(camera.ParamsData(), idx,
                                                     options_.upper_bound_focal_length_factor * est_focal);
                }
                for (size_t idx : camera.PrincipalPointIdxs()) {
                    problem->SetParameterLowerBound(camera.ParamsData(), idx, 0);
                    problem->SetParameterUpperBound(camera.ParamsData(), idx,
                                                     std::max(camera.Width(), camera.Height()));
                }

                for (size_t idx : camera.ExtraParamsIdxs()) {
                    problem->SetParameterLowerBound(camera.ParamsData(), idx, -0.99);
                    problem->SetParameterUpperBound(camera.ParamsData(), idx, 0.99);
                }

                if (const_camera_params.size() > 0) {
                    if (camera.ModelName().compare("SPHERICAL") != 0) {
                        ceres::SubsetParameterization *camera_params_parameterization =
                            new ceres::SubsetParameterization(static_cast<int>(camera.NumParams()),
                                                              const_camera_params);
                        SetParameterization(problem, camera.ParamsData(), camera_params_parameterization);
                    } else {
                        problem->SetParameterBlockConstant(camera.ParamsData());
                    }
                }
            }

            if (camera.ModelName().compare("SPHERICAL") == 0) {
                double *camera_params_data = camera.ParamsData();

                double *local_qvec2_data = camera_params_data + 10;
                double *local_tvec2_data = camera_params_data + 14;

                if (options_.refine_extra_params) {
                    ceres::LocalParameterization *quaternion_parameterization2 = new ceres::QuaternionParameterization;
                    SetParameterization(problem, local_qvec2_data, quaternion_parameterization2);
                } else {
                    problem->SetParameterBlockConstant(local_qvec2_data);
                    problem->SetParameterBlockConstant(local_tvec2_data);
                }
            }
        }
    }
}

void BlockBundleAdjuster::ParameterizePoints(Reconstruction *reconstruction, 
                                             ceres::Problem* problem, 
                                             Block* block) {
    for (const auto elem : block->mappoint_num_observations) {
        MapPoint &mappoint = reconstruction->MapPoint(elem.first);
        if (mappoint.Track().Length() > elem.second) {
            problem->SetParameterBlockConstant(mappoint.XYZ().data());
        }
    }

    for (const mappoint_t mappoint_id : block->config.ConstantPoints()) {
        MapPoint &mappoint = reconstruction->MapPoint(mappoint_id);
        problem->SetParameterBlockConstant(mappoint.XYZ().data());
    }
}

void BlockBundleAdjuster::ParameterizePointsWithoutTrack(Reconstruction *reconstruction, 
                                                         ceres::Problem* problem, 
                                                         Block* block) {
    for (const mappoint_t mappoint_id : block->config.ConstantPoints()) {
        MapPoint &mappoint = reconstruction->MapPoint(mappoint_id);
        problem->SetParameterBlockConstant(mappoint.XYZ().data());
    }
}

void BlockBundleAdjuster::ParameterizePoses(Reconstruction *reconstruction, 
                                            ceres::Problem* problem, 
                                            Block* block) {
    std::cout << "invovled image extrinsics num: " << block->invovled_image_extrinsics.size() << std::endl;
    int const_camera_count = 0;
    for (const image_t image_id : block->invovled_image_extrinsics) {
        if (constant_pose_images_.count(image_id) > 0) {
            Image &image = reconstruction->Image(image_id);

            double *qvec_data = image.Qvec().data();
            double *tvec_data = image.Tvec().data();
            problem->SetParameterBlockConstant(qvec_data);
            problem->SetParameterBlockConstant(tvec_data);
            const_camera_count++;
        }
    }
    std::cout << "const camera count: " << const_camera_count << std::endl;
}

}  // namespace sensemap
