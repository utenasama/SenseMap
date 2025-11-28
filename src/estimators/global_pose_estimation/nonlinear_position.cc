//
// Created by sensetime on 2021/2/4.
//


#include "nonlinear_position.h"

#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include <graph/utils.h>
#include "util/random.h"
#include "base/cost_functions.h"
#include "graph/maximum_spanning_tree_graph.h"
#include "optim/global_motions/utils.h"
#include "optim/ransac/loransac.h"
#include "estimators/camera_alignment.h"

namespace sensemap{

namespace {

using Eigen::Matrix3d;
using Eigen::Vector3d;

Vector3d GetRotatedTranslation(const Vector3d& rotation_angle_axis,
                           const Vector3d& translation) {
    Matrix3d rotation;
    ceres::AngleAxisToRotationMatrix(
            rotation_angle_axis.data(),
            ceres::ColumnMajorAdapter3x3(rotation.data()));
    return rotation.transpose() * translation;
}

Vector3d GetRotatedFeatureRay(const Camera& camera,
                          const Eigen::Vector3d& orientation,
                          const Eigen::Vector2d& pixel,
                          local_camera_t local_camera_id) {

    Matrix3d rotation;
    ceres::AngleAxisToRotationMatrix(
            orientation.data(),
            ceres::ColumnMajorAdapter3x3(rotation.data()));

    Eigen::Vector3d direction;
    if (camera.NumLocalCameras() <= 1) {
        // Remove the effect of calibration.
        const Eigen::Vector2d undistorted_pixel = camera.ImageToWorld(pixel);
        const Eigen::Vector3d undistorted_point(undistorted_pixel(0), undistorted_pixel(1), 1.0);
        // Apply rotation.
        direction = rotation.transpose() * undistorted_point;
    } else {
        // Remove the effect of calibration.
        // const Eigen::Vector2d undistorted_pixel = camera.LocalImageToWorld(local_camera_id, pixel);
        // const Eigen::Vector3d undistorted_point(undistorted_pixel(0), undistorted_pixel(1), 1.0);
        const Eigen::Vector3d undistorted_point = camera.LocalImageToBearing(local_camera_id, pixel);
        // Apply rotation.
        Eigen::Vector4d local_qvec;
        Eigen::Vector3d local_tvec;
        camera.GetLocalCameraExtrinsic(local_camera_id, local_qvec, local_tvec);
        Eigen::Matrix3d local_R = QuaternionToRotationMatrix(local_qvec);
        direction = (local_R * rotation).transpose() * undistorted_point;
    }

    return direction.normalized();
}

Eigen::Vector3d MultiplyRotations(const Eigen::Vector3d& rotation1,
                                  const Eigen::Vector3d& rotation2) {
    Eigen::Matrix3d rotation1_mat, rotation2_mat;
    ceres::AngleAxisToRotationMatrix(rotation1.data(), rotation1_mat.data());
    ceres::AngleAxisToRotationMatrix(rotation2.data(), rotation2_mat.data());

    const Eigen::Matrix3d rotation = rotation1_mat * rotation2_mat;
    Eigen::Vector3d rotation_aa;
    ceres::RotationMatrixToAngleAxis(rotation.data(), rotation_aa.data());
    return rotation_aa;
}

}  // namespace

NonlinearPositionEstimator::NonlinearPositionEstimator(
        const NonlinearPositionEstimator::Options& options,
        std::shared_ptr<SceneGraphContainer> scene_graph_container,
        std::unordered_map<image_pair_t , Eigen::Vector3d>& relative_positions)
        : options_(options), scene_graph_container_(scene_graph_container), relative_positions_(relative_positions){
    CHECK_GT(options_.num_threads, 0);
    CHECK_GE(options_.min_num_points_per_view, 0);
    CHECK_GT(options_.point_to_camera_weight, 0);
    CHECK_GT(options_.robust_loss_width, 0);
}

bool NonlinearPositionEstimator::EstimatePositions(
        const EIGEN_STL_UMAP(image_pair_t, struct CorrespondenceGraph::ImagePair) &image_pairs,
        const std::unordered_map<image_t , Vector3d>& orientations,
        std::unordered_map<image_t , Vector3d>* positions) {
    CHECK_NOTNULL(positions);
    if (image_pairs.empty() || orientations.empty()) {
        VLOG(2) << "Number of view_pairs = " << image_pairs.size()
                << " Number of orientations = " << orientations.size();
        return false;
    }
    triangulated_points_.clear();
    problem_.reset(new ceres::Problem());

    for (auto image_pair : image_pairs) {
        relative_positions_weights_[image_pair.first] = image_pair.second.two_view_geometry.confidence;
    }

    // Iterative schur is only used if the problem is large enough, otherwise
    // sparse schur is used.
    static const int kMinNumCamerasForIterativeSolve = 1000;

    // Initialize positions to be random.
    InitializeRandomPositions(orientations, positions);

    // Add the constraints to the problem.
    AddCameraToCameraConstraints(orientations, positions);
    if (options_.min_num_points_per_view > 0) {
        AddPointToCameraConstraints(orientations, positions);
        AddCamerasAndPointsToParameterGroups(positions);
    }

    if(options_.use_position_prior){
        AddPriorPositionsConstraints(positions);
        positions->begin()->second = scene_graph_container_->Image(positions->begin()->first).TvecPrior();
    } else {
        // Set one camera to be at the origin to remove the ambiguity of the origin.
        positions->begin()->second.setZero();
    }
    problem_->SetParameterBlockConstant(positions->begin()->second.data());

    // Set the solver options.
    ceres::Solver::Summary summary;
    solver_options_.num_threads = options_.num_threads;
    solver_options_.max_num_iterations = options_.max_num_iterations;
    solver_options_.minimizer_progress_to_stdout = true;

    // Choose the type of linear solver. For sufficiently large problems, we want
    // to use iterative methods (e.g., Conjugate Gradient or Iterative Schur);
    // however, we only want to use a Schur solver if 3D points are used in the
    // optimization.
    if (positions->size() > kMinNumCamerasForIterativeSolve) {
        if (options_.min_num_points_per_view > 0) {
            // solver_options_.linear_solver_type = ceres::ITERATIVE_SCHUR;
            // solver_options_.preconditioner_type = ceres::SCHUR_JACOBI;
            solver_options_.linear_solver_type = ceres::SPARSE_SCHUR;
        } else {
            solver_options_.linear_solver_type = ceres::CGNR;
            solver_options_.preconditioner_type = ceres::JACOBI;
        }
    } else {
        if (options_.min_num_points_per_view > 0) {
            solver_options_.linear_solver_type = ceres::SPARSE_SCHUR;
        } else {
            solver_options_.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
        }
    }

    ceres::Solve(solver_options_, problem_.get(), &summary);
    LOG(INFO) << summary.FullReport();

    return summary.IsSolutionUsable();
}

bool NonlinearPositionEstimator::RefinePosesWithPrior(
        const EIGEN_STL_UMAP(image_pair_t, struct CorrespondenceGraph::ImagePair) &image_pairs,
        std::unordered_map<image_t , Eigen::Vector3d>& orientations,
        std::unordered_map<image_t , Eigen::Vector3d>* positions) {
        CHECK_NOTNULL(positions);
    if (image_pairs.empty() || orientations.empty()) {
        VLOG(2) << "Number of view_pairs = " << image_pairs.size()
                << " Number of orientations = " << orientations.size();
        return false;
    }

    std::vector<Eigen::Vector3d> points1, points2;
    auto image_ids = scene_graph_container_->GetImageIds();
    for (auto image_id : image_ids) {
        if (positions->find(image_id) == positions->end()) {
            continue;
        }
        if (scene_graph_container_->Image(image_id).HasTvecPrior()) {
            auto pos = positions->at(image_id);
            points1.push_back(pos);
            auto prior = scene_graph_container_->Image(image_id).TvecPrior();
            points2.push_back(prior);
        }
    }
    if (points1.size() == 0) {
        return false;
    }

    RANSACOptions ransac_options;
    ransac_options.max_error = 3; //the maximum error is 3 meters
    ransac_options.confidence = 0.9999;
    ransac_options.min_inlier_ratio = 0.25;
    ransac_options.min_num_trials = 200;
    ransac_options.max_num_trials = 10000;

    LORANSAC<CameraAlignmentEstimator, CameraAlignmentEstimator> ransac(ransac_options);

    auto report = ransac.Estimate(points1, points2);

    std::cout<<"Restimate transform inlier num: "<<report.support.num_inliers<<" transform the location"<<std::endl;
    std::cout<<"Restimate transform: "<<std::endl;
    std::cout<<report.model<<std::endl;

    auto sRT = report.model;

    const SimilarityTransform3 tform(sRT);

    for (auto & orientation : orientations) {
        Eigen::Vector4d qvec;
        ceres::AngleAxisToQuaternion(orientation.second.data(), qvec.data());
        Eigen::Vector3d tvec = -QuaternionToRotationMatrix(qvec) * positions->at(orientation.first);

        tform.TransformPose(&qvec, &tvec);

        ceres::QuaternionToAngleAxis(qvec.data(), orientation.second.data());
        positions->at(orientation.first) = -QuaternionToRotationMatrix(qvec).transpose() * tvec;
    }

    triangulated_points_.clear();
    problem_.reset(new ceres::Problem());

    // auto R = sRT.block<3, 3>(0, 0);
    // // R /= R.col(2).norm();
    // for (auto image_pair : image_pairs) {
    //     relative_positions_weights_[image_pair.first] = image_pair.second.two_view_geometry.confidence;
    //     relative_positions_[image_pair.first] = relative_positions_[image_pair.first];
    // }

    // Iterative schur is only used if the problem is large enough, otherwise
    // sparse schur is used.
    static const int kMinNumCamerasForIterativeSolve = 1000;

    // Add the constraints to the problem.
    AddCameraToCameraConstraints(orientations, positions);
    if (options_.min_num_points_per_view > 0) {
        AddPointToCameraConstraints(orientations, positions);
        AddCamerasAndPointsToParameterGroups(positions);
    }

    if(options_.use_position_prior){
        AddPriorPositionsConstraints(positions);
        positions->begin()->second = scene_graph_container_->Image(positions->begin()->first).TvecPrior();
    } else {
        // Set one camera to be at the origin to remove the ambiguity of the origin.
        positions->begin()->second.setZero();
    }
    problem_->SetParameterBlockConstant(positions->begin()->second.data());

    // Set the solver options.
    ceres::Solver::Summary summary;
    solver_options_.num_threads = options_.num_threads;
    solver_options_.max_num_iterations = options_.max_num_iterations;
    solver_options_.minimizer_progress_to_stdout = true;

    // Choose the type of linear solver. For sufficiently large problems, we want
    // to use iterative methods (e.g., Conjugate Gradient or Iterative Schur);
    // however, we only want to use a Schur solver if 3D points are used in the
    // optimization.
    if (positions->size() > kMinNumCamerasForIterativeSolve) {
        if (options_.min_num_points_per_view > 0) {
            // solver_options_.linear_solver_type = ceres::ITERATIVE_SCHUR;
            // solver_options_.preconditioner_type = ceres::SCHUR_JACOBI;
            solver_options_.linear_solver_type = ceres::SPARSE_SCHUR;
        } else {
            solver_options_.linear_solver_type = ceres::CGNR;
            solver_options_.preconditioner_type = ceres::JACOBI;
        }
    } else {
        if (options_.min_num_points_per_view > 0) {
            solver_options_.linear_solver_type = ceres::SPARSE_SCHUR;
        } else {
            solver_options_.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
        }
    }

    ceres::Solve(solver_options_, problem_.get(), &summary);
    LOG(INFO) << summary.FullReport();

    return summary.IsSolutionUsable();
}

void NonlinearPositionEstimator::InitializeRandomPositions(
        const std::unordered_map<image_t , Vector3d>& orientations,
        std::unordered_map<image_t , Vector3d>* positions) {
    std::unordered_set<image_t> constrained_positions;
    constrained_positions.reserve(orientations.size());
    const auto &image_pairs = scene_graph_container_->CorrespondenceGraph()->ImagePairs();
    for (const auto& image_pair : image_pairs) {
        image_t image_id1, image_id2;
        sensemap::utility::PairIdToImagePair(image_pair.first, &image_id1, &image_id2);
        constrained_positions.insert(image_id1);
        constrained_positions.insert(image_id2);
    }

    positions->reserve(orientations.size());
    for (const auto& orientation : orientations) {
        if (ContainsKey(constrained_positions, orientation.first)) {
            (*positions)[orientation.first] =
                    100.0 * Eigen::Vector3d(RandomReal<float>(-1, 1), RandomReal<float>(-1, 1), RandomReal<float>(-1, 1));

        }
    }
}

void NonlinearPositionEstimator::AddCameraToCameraConstraints(
        const std::unordered_map<image_t , Vector3d>& orientations,
        std::unordered_map<image_t , Vector3d>* positions) {
    const auto &image_pairs = scene_graph_container_->CorrespondenceGraph()->ImagePairs();
    for (const auto& image_pair : image_pairs) {
        image_t image_id1, image_id2;
        sensemap::utility::PairIdToImagePair(image_pair.first, &image_id1, &image_id2);
        Vector3d* position1 = FindOrNull(*positions, image_id1);
        Vector3d* position2 = FindOrNull(*positions, image_id2);

        // Do not add this view pair if one or both of the positions do not exist.
        if (position1 == nullptr || position2 == nullptr) {
            continue;
        }

        double weight = 1.0;
#ifdef TWO_VIEW_CONFIDENCE
         if (options_.use_position_prior) {
             weight = options_.position_prior_weight * relative_positions_weights_.at(image_pair.first);
         } else {
             weight = 1.0 + relative_positions_weights_.at(image_pair.first);
         }
#endif

        // Rotate the relative translation so that it is aligned to the global
        // orientation frame.
        const Vector3d translation_direction = GetRotatedTranslation(
                FindOrDie(orientations, image_id1), relative_positions_.at(image_pair.first));

        ceres::CostFunction* cost_function =
                PairwiseTranslationCostFunction::Create(translation_direction, weight);

        problem_->AddResidualBlock(cost_function,
                                   new ceres::HuberLoss(options_.robust_loss_width),
                                   position1->data(),
                                   position2->data());
    }

    VLOG(2) << problem_->NumResidualBlocks()
            << " camera to camera constraints "
               "were added to the position "
               "estimation problem.";
}

void NonlinearPositionEstimator::AddPointToCameraConstraints(
        const std::unordered_map<image_t , Eigen::Vector3d>& orientations,
        std::unordered_map<image_t , Eigen::Vector3d>* positions) {
    const int num_camera_to_camera_constraints = problem_->NumResidualBlocks();
    std::unordered_set<track_t> tracks_to_add;
    const int num_point_to_camera_constraints =
            FindTracksForProblem(*positions, &tracks_to_add);
    if (num_point_to_camera_constraints == 0) {
        return;
    }

    const double point_to_camera_weight =
            options_.point_to_camera_weight *
            static_cast<double>(num_camera_to_camera_constraints) /
            static_cast<double>(num_point_to_camera_constraints);

    triangulated_points_.reserve(tracks_to_add.size());
    for (const track_t track_id : tracks_to_add) {
        triangulated_points_[track_id] =
                100.0 * Eigen::Vector3d(RandomReal<float>(-1, 1),
                        RandomReal<float>(-1, 1), RandomReal<float>(-1, 1));

        AddTrackToProblem(
                track_id, orientations, point_to_camera_weight, positions);
    }

    VLOG(2) << num_point_to_camera_constraints
            << " point to camera constriants "
               "were added to the position "
               "estimation problem.";
}


void NonlinearPositionEstimator::AddPriorPositionsConstraints(
        std::unordered_map<image_t , Eigen::Vector3d>* positions,
        const Eigen::Matrix3x4d sRT) {

    auto &images = scene_graph_container_->Images();
    for (auto& image : images) {
        if(!image.second.HasTvecPrior())
            continue;
        auto image_id = image.second.ImageId();
        Vector3d* position = FindOrNull(*positions, image_id);

        // Do not add this view pair if one or both of the positions do not exist.
        if (position == nullptr) {
            continue;
        }

        double weight_x(1.), weight_y(1.), weight_z(1.);
        if (image.second.RtkFlag() == 50) {
            weight_x = image.second.HasRtkStd() ? options_.position_prior_weight / image.second.RtkStdLat() : options_.position_prior_weight;
            weight_y = image.second.HasRtkStd() ? options_.position_prior_weight / image.second.RtkStdLon() : options_.position_prior_weight;
            weight_z = image.second.HasRtkStd() ? options_.position_prior_weight / image.second.RtkStdHgt() : options_.position_prior_weight;
        } else if (image.second.HasTvecPrior()) {
            weight_x = options_.position_prior_weak_weight;
            weight_y = options_.position_prior_weak_weight;
            weight_z = options_.position_prior_weak_weight;
        }

        // double weight_x = options_.position_prior_weight / image.second.RtkStdLat();
        // double weight_y = options_.position_prior_weight / image.second.RtkStdLon();
        // double weight_z = options_.position_prior_weight / image.second.RtkStdHgt();

        ceres::CostFunction* cost_function =
                PriorAbsoluteLocationGlobalSfMCostFunction::Create(
                        image.second.TvecPrior(), weight_x, weight_y, weight_z, sRT);

        problem_->AddResidualBlock(cost_function,
                                   new ceres::HuberLoss(options_.robust_loss_width),
                                   position->data());
    }

    VLOG(2) << problem_->NumResidualBlocks()
            << "Prior position  constraints "
               "were added to the position "
               "estimation problem.";
}

int NonlinearPositionEstimator::FindTracksForProblem(
        const std::unordered_map<image_t , Eigen::Vector3d>& positions,
        std::unordered_set<track_t>* tracks_to_add) {
    CHECK_NOTNULL(tracks_to_add)->clear();

    traks_= scene_graph_container_->CorrespondenceGraph()->GenerateTracks();

    std::sort(traks_.begin(), traks_.end(),
              [](const Track& t1, const Track& t2) {
                  return t1.Length() > t2.Length();
              });

    std::vector<unsigned char> inliers(traks_.size(), 1);

    std::unordered_map<image_t, int> cover_view;
    for (size_t i = 0; i < traks_.size(); ++i) {
        if (!inliers.at(i)) {
            continue;
        }
        auto track = traks_.at(i);
        bool selected = false;
        for (auto track_elem : track.Elements()) {
            image_t image_id = track_elem.image_id;
            std::unordered_map<image_t, int>::iterator it =
                    cover_view.find(image_id);
            if (it == cover_view.end() || it->second < options_.min_num_points_per_view) {
                selected = true;
                break;
            }
        }

        inliers[i] = selected;

        if (selected) {
            std::unordered_set<image_t> track_images;
            for (auto track_elem : track.Elements()) {
                track_images.insert(track_elem.image_id);
            }
            for (auto image_id : track_images) {
                cover_view[image_id]++;
            }
        }
    }

    int num_point_to_camera_constraints = 0;
    for (size_t i = 0; i < traks_.size(); ++i) {
        if (inliers.at(i)) {
            tracks_to_add->emplace(i);
            num_point_to_camera_constraints++;
        }
    }

    return num_point_to_camera_constraints;
}


void NonlinearPositionEstimator::AddTrackToProblem(
        const track_t track_id,
        const std::unordered_map<image_t , Vector3d>& orientations,
        const double point_to_camera_weight,
        std::unordered_map<image_t , Vector3d>* positions) {
    // For each view in the track add the point to camera correspondences.
    for (const auto element : traks_[track_id].Elements()) {
        if (!ContainsKey(*positions, element.image_id)) {
            continue;
        }
        Vector3d& camera_position = FindOrDie(*positions, element.image_id);
        Vector3d& point = FindOrDie(triangulated_points_, track_id);

        // Rotate the feature ray to be in the global orientation frame.
        const auto image = scene_graph_container_->Image(element.image_id);
        const auto camera = scene_graph_container_->Camera(image.CameraId());
        const local_camera_t local_camera_id = image.LocalImageIndices()[element.point2D_idx];

        auto orientation = FindOrDie(orientations, element.image_id);
        Eigen::Matrix3d R;
        ceres::AngleAxisToRotationMatrix(orientation.data(), R.data());
            
        Eigen::Vector3d local_C = Eigen::Vector3d::Zero();
        if (camera.NumLocalCameras() > 1) {
            Eigen::Vector4d local_qvec;
            Eigen::Vector3d local_tvec;
            camera.GetLocalCameraExtrinsic(local_camera_id, local_qvec, local_tvec);
            local_C = -QuaternionToRotationMatrix(local_qvec).transpose() * local_tvec;
            local_C = R.transpose() * local_C;

        }

        const Vector3d feature_ray =  GetRotatedFeatureRay(
                scene_graph_container_->Camera(image.CameraId()),
                orientation,
                image.Point2D(element.point2D_idx).XY(),
                local_camera_id);

        // Rotate the relative translation so that it is aligned to the global
        // orientation frame.
        ceres::CostFunction* cost_function =
                RigPairwiseTranslationCostFunction::Create(feature_ray, local_C, point_to_camera_weight);

        // Add the residual block
        problem_->AddResidualBlock(cost_function,
                                   new ceres::HuberLoss(options_.robust_loss_width),
                                   camera_position.data(),
                                   point.data());
    }
}

void NonlinearPositionEstimator::AddCamerasAndPointsToParameterGroups(
        std::unordered_map<image_t , Vector3d>* positions) {
    CHECK_GT(triangulated_points_.size(), 0)
        << "Cannot set the Ceres parameter groups for Schur based solvers "
           "because there are no triangulated points.";

    // Create a custom ordering for Schur-based problems.
    solver_options_.linear_solver_ordering.reset(
            new ceres::ParameterBlockOrdering);
    ceres::ParameterBlockOrdering* parameter_ordering =
            solver_options_.linear_solver_ordering.get();
    // Add point parameters to group 0.
    for (auto& point : triangulated_points_) {
        parameter_ordering->AddElementToGroup(point.second.data(), 0);
    }

    // Add camera parameters to group 1.
    for (auto& position : *positions) {
        parameter_ordering->AddElementToGroup(position.second.data(), 1);
    }
}

} //namespace sensemap
