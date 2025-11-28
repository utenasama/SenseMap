//Copyright (c) 2021, SenseTime Group.
//All rights reserved.

#include "global_mapper.h"

#include <ceres/rotation.h>
#include <estimators/global_pose_estimation/global_rotation.h>
#include <estimators/global_pose_estimation/robust_rotation.h>
#include <util/misc.h>
#include <util/random.h>

#include "base/pose.h"
#include "estimators/global_pose_estimation/global_position.h"
#include "estimators/global_pose_estimation/nonlinear_position.h"
#include "graph/connected_components.h"
#include "graph/minimum_spanning_tree.h"

#include <Eigen/Dense>

namespace sensemap {

namespace {


std::unordered_set<image_t>RemoveDisconnectedImagePairs(std::shared_ptr<SceneGraphContainer> scene_graph_container) {

    std::unordered_set<image_t> removed_iamges;

    // Extract all connected components.
    ConnectedComponents<image_t> cc_extractor;
    const auto &image_pairs = scene_graph_container->CorrespondenceGraph()->ImagePairs();
    for(const auto &image_pair : image_pairs){
        image_t image_id1, image_id2;
        sensemap::utility::PairIdToImagePair(image_pair.first, &image_id1, &image_id2);
        cc_extractor.AddEdge(image_id1, image_id2);
    }
    std::unordered_map<image_t, std::unordered_set<image_t> > connected_components;
    cc_extractor.Extract(&connected_components);

    // Find the largest connected component.
    int max_cc_size = 0;
    image_t largest_cc_root_id = kInvalidImageId;
    for (const auto& connected_component : connected_components) {
        if (connected_component.second.size() > max_cc_size) {
            max_cc_size = connected_component.second.size();
            largest_cc_root_id = connected_component.first;
        }
    }

    // Remove all image pairs containing a image to remove (i.e. the ones that are
    // not in the largest connectedcomponent).
    const int num_image_pairs_before_filtering = scene_graph_container->CorrespondenceGraph()->NumImagePairs();
    for (const auto& connected_component : connected_components) {
        if (connected_component.first == largest_cc_root_id) {
            continue;
        }

        // NOTE: The connected component will contain the root id as well, so we do
        // not explicity have to remove connected_component.first since it will
        // exist in connected_components.second
        for (const auto image_id2 : connected_component.second) {
            scene_graph_container->DeleteImage(image_id2);
            removed_iamges.insert(image_id2);
        }
    }

    const int num_removed_image_pairs =
            num_image_pairs_before_filtering - scene_graph_container->CorrespondenceGraph()->NumImagePairs();
    LOG_IF(INFO, num_removed_image_pairs > 0) << num_removed_image_pairs
        << " image pairs were disconnected from the largest connected component of the image graph and were removed.";
    return removed_iamges;
}

// Use Ceres to perform a stable composition of rotations. This is not as
// efficient as directly composing angle axis vectors (see the old
// implementation commented above) but is more stable.
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

bool AngularDifferenceIsAcceptable(
        const Eigen::Vector3d& orientation1,
        const Eigen::Vector3d& orientation2,
        const Eigen::Vector3d& relative_orientation,
        const double sq_max_relative_rotation_difference_radians) {
    const Eigen::Vector3d composed_relative_rotation = MultiplyRotations(orientation2, -orientation1);
    const Eigen::Vector3d loop_rotation = MultiplyRotations(-relative_orientation, composed_relative_rotation);
    const double sq_rotation_angular_difference_radians = loop_rotation.squaredNorm();
    return sq_rotation_angular_difference_radians <= sq_max_relative_rotation_difference_radians;
}

void FilterImagePairsFromOrientation(
        const std::unordered_map<image_t , Eigen::Vector3d>& orientations,
        const double max_relative_rotation_difference_degrees,
        std::shared_ptr<SceneGraphContainer> scene_graph_container) {

    CHECK_GE(max_relative_rotation_difference_degrees, 0.0);

    // Precompute the squared threshold in radians.
    const double max_relative_rotation_difference_radians = DegToRad(max_relative_rotation_difference_degrees);
    const double sq_max_relative_rotation_difference_radians =
            max_relative_rotation_difference_radians * max_relative_rotation_difference_radians;

    std::unordered_set<image_pair_t > image_pairs_to_remove;
    const auto &image_pairs = scene_graph_container->CorrespondenceGraph()->ImagePairs();
    for(const auto &image_pair : image_pairs){
        image_t image_id1, image_id2;
        sensemap::utility::PairIdToImagePair(image_pair.first, &image_id1, &image_id2);
        const Eigen::Vector3d* orientation1 = FindOrNull(orientations, image_id1);
        const Eigen::Vector3d* orientation2 = FindOrNull(orientations, image_id2);

        // If the image pair contains a image that does not have an orientation then
        // remove it.
        if (orientation1 == nullptr || orientation2 == nullptr) {
            LOG(WARNING) << "Image pair (" << image_id1 << ", " << image_id2
                    << ") contains a image that does not exist! Removing the image pair.";
            image_pairs_to_remove.insert(image_pair.first);
            continue;
        }

        // Remove the image pair if the relative rotation estimate is not within the
        // tolerance.
        Eigen::Vector3d relative_rotation;
        ceres::QuaternionToAngleAxis(image_pair.second.two_view_geometry.qvec.data(), relative_rotation.data());
        if (!AngularDifferenceIsAcceptable( *orientation1, *orientation2, relative_rotation,
                sq_max_relative_rotation_difference_radians)) {
            image_pairs_to_remove.insert(image_pair.first);
        }

        auto image1 = scene_graph_container->Image(image_id1);
        auto image2 = scene_graph_container->Image(image_id2);
        if(!image1.HasQvecPrior() || !image2.HasQvecPrior()){
            continue;
        }
        Eigen::Vector3d  q_prior1, q_prior2;
        ceres::QuaternionToAngleAxis(image1.QvecPrior().data(), q_prior1.data());
        ceres::QuaternionToAngleAxis(image2.QvecPrior().data(), q_prior2.data());
        const Eigen::Vector3d composed_relative_rotation = MultiplyRotations(q_prior2, -q_prior1);
        if (!AngularDifferenceIsAcceptable(*orientation1, *orientation2, composed_relative_rotation,
                                           sq_max_relative_rotation_difference_radians)) {
            image_pairs_to_remove.insert(image_pair.first);
        }
    }



    // Remove all the "bad" relative poses.
    for (const auto image_pair : image_pairs_to_remove) {
        image_t image_id1, image_id2;
        sensemap::utility::PairIdToImagePair(image_pair, &image_id1, &image_id2);
        scene_graph_container->CorrespondenceGraph()->DeleteCorrespondences(image_id1, image_id2);
    }
    LOG(INFO) << "Removed " << image_pairs_to_remove.size()
            << " image pairs by rotation filtering.";
}

// Creates the constraint matrix such that ||A * t|| is minimized, where A is
// R_i * f_i x R_j * f_j. Given known rotations, we can solve for the
// relative translation from this constraint matrix.
void CreateConstraintMatrix(
        const std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d> > & correspondences,
        const Eigen::Vector3d& rotation1,
        const Eigen::Vector3d& rotation2,
        Eigen::MatrixXd* constraint_matrix) {
    constraint_matrix->resize(3, correspondences.size());

    Eigen::Matrix3d rotation_matrix1;
    ceres::AngleAxisToRotationMatrix( rotation1.data(), ceres::ColumnMajorAdapter3x3(rotation_matrix1.data()));
    Eigen::Matrix3d rotation_matrix2;
    ceres::AngleAxisToRotationMatrix( rotation2.data(), ceres::ColumnMajorAdapter3x3(rotation_matrix2.data()));

    for (int i = 0; i < correspondences.size(); i++) {
        const Eigen::Vector3d rotated_feature1 = rotation_matrix1.transpose() * correspondences[i].first;
        const Eigen::Vector3d rotated_feature2 = rotation_matrix2.transpose() * correspondences[i].second;

        constraint_matrix->col(i) =
                rotated_feature2.cross(rotated_feature1).transpose() * rotation_matrix1.transpose();
    }
}

bool IsTriangulatedPointInFrontOfCameras(
        const std::pair<Eigen::Vector3d, Eigen::Vector3d>& correspondence,
        const Eigen::Matrix3d& rotation,
        const Eigen::Vector3d& position) {
    const Eigen::Vector3d dir1 = correspondence.first;
    const Eigen::Vector3d dir2 = rotation.transpose() * correspondence.second;

    const double dir1_sq = dir1.squaredNorm();
    const double dir2_sq = dir2.squaredNorm();
    const double dir1_dir2 = dir1.dot(dir2);
    const double dir1_pos = dir1.dot(position);
    const double dir2_pos = dir2.dot(position);

    return (dir2_sq * dir1_pos - dir1_dir2 * dir2_pos > 0 && dir1_dir2 * dir1_pos - dir1_sq * dir2_pos > 0);
}

// Determines if the majority of the points are in front of the cameras. This is
// useful for determining the sign of the relative position. Returns true if
// more than 50% of correspondences are in front of both cameras and false
// otherwise.
bool MajorityOfPointsInFrontOfCameras(
        const std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d> >& correspondences,
        const Eigen::Vector3d& rotation1,
        const Eigen::Vector3d& rotation2,
        const Eigen::Vector3d& relative_position) {
    // Compose the relative rotation.
    Eigen::Matrix3d rotation_matrix1, rotation_matrix2;
    ceres::AngleAxisToRotationMatrix(rotation1.data(), ceres::ColumnMajorAdapter3x3(rotation_matrix1.data()));
    ceres::AngleAxisToRotationMatrix(rotation2.data(), ceres::ColumnMajorAdapter3x3(rotation_matrix2.data()));
    const Eigen::Matrix3d relative_rotation_matrix = rotation_matrix2 * rotation_matrix1.transpose();

    // Tests all points for cheirality.
    int num_points_in_front_of_cameras = 0;
    for (const auto& match : correspondences) {
        if (IsTriangulatedPointInFrontOfCameras(match, relative_rotation_matrix, relative_position)) {
            ++num_points_in_front_of_cameras;
        }
    }

    return num_points_in_front_of_cameras > (correspondences.size() / 2);
}


// Given known camera rotations and feature correspondences, this method solves
// for the relative translation that optimizes the epipolar error
// f_i * E * f_j^t = 0.
bool OptimizeRelativePositionWithKnownRotation(
        const std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d> > & correspondences,
        const Eigen::Vector3d& rotation1,
        const Eigen::Vector3d& rotation2,
        Eigen::Vector3d* relative_position) {
    CHECK_NOTNULL(relative_position);

    // Set the initial relative position to random. This helps avoid a bad local
    // minima that is achieved from poor initialization.
    relative_position->setRandom(); //TODO

    // Constants used for the IRLS solving.
    const double eps = 1e-5;
    const int kMaxIterations = 100;
    const int kMaxInnerIterations = 10;
    const double kMinWeight = 1e-7;

    // Create the constraint matrix from the known correspondences and rotations.
    Eigen::MatrixXd constraint_matrix;
    CreateConstraintMatrix(correspondences, rotation1, rotation2, &constraint_matrix);

    // Initialize the weighting terms for each correspondence.
    Eigen::VectorXd weights(correspondences.size());
    weights.setConstant(1.0);

    // Solve for the relative positions using a robust IRLS.
    double cost = 0;
    int num_inner_iterations = 0;
    for (int i = 0; i < kMaxIterations && num_inner_iterations < kMaxInnerIterations; i++) {
        // Limit the minimum weight at kMinWeight.
        weights = (weights.array() < kMinWeight).select(kMinWeight, weights);

        // Apply the weights to the constraint matrix.
        const Eigen::Matrix3d lhs =
                constraint_matrix * weights.asDiagonal().inverse() * constraint_matrix.transpose();

        // Solve for the relative position which is the null vector of the weighted
        // constraints.
        const Eigen::Vector3d new_relative_position =
                lhs.jacobiSvd(Eigen::ComputeFullU).matrixU().rightCols<1>();

        // Update the weights based on the current errors.
        weights = (new_relative_position.transpose() * constraint_matrix).array().abs();

        // Compute the new cost.
        const double new_cost = weights.sum();

        // Check for convergence.
        const double delta = std::max(std::abs(cost - new_cost), 1 - new_relative_position.squaredNorm());

        // If we have good convergence, attempt an inner iteration.
        if (delta <= eps) {
            ++num_inner_iterations;
        } else {
            num_inner_iterations = 0;
        }

        cost = new_cost;
        *relative_position = new_relative_position;
    }

    // The position solver above does not consider the sign of the relative
    // position. We can determine the sign by choosing the sign that puts the most
    // points in front of the camera.
    if (!MajorityOfPointsInFrontOfCameras(correspondences, rotation1, rotation2, *relative_position)) {
        *relative_position *= -1.0;
    }

    return true;
}

bool OptimizeRelativePositionWithKnownRotationRig(
        const std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d> > & correspondences,
        const std::vector<std::pair<uint32_t, uint32_t> > & local_camera_pairs,
        const Camera* camera1, const Camera* camera2,
        const Eigen::Vector3d& rotation1,
        const Eigen::Vector3d& rotation2,
        Eigen::Vector3d* relative_position) {
    CHECK_NOTNULL(relative_position);

#if 0
    Eigen::Matrix3d R1, R2;
    ceres::AngleAxisToRotationMatrix(rotation1.data(), R1.data());
    ceres::AngleAxisToRotationMatrix(rotation2.data(), R2.data());
    Eigen::Matrix3d R = R2 * R1.transpose();

    std::vector<Eigen::Matrix3d> M1(camera1->NumLocalCameras());
    for (local_camera_t local_camera_id = 0; local_camera_id < camera1->NumLocalCameras(); ++local_camera_id) {
        Eigen::Vector4d qvec;
        Eigen::Vector3d tvec;
        camera1->GetLocalCameraExtrinsic(local_camera_id, qvec, tvec);
        Eigen::Vector3d C = -QuaternionToRotationMatrix(qvec).transpose() * tvec;
        Eigen::Matrix3d skew_C;
        skew_C << 0.0, -C(2), C(1), C(2), 0.0, -C(0), -C(1), C(0), 0.0;
        M1[local_camera_id] = skew_C * R;
    }
    std::vector<Eigen::Matrix3d> M2(camera2->NumLocalCameras());
    for (local_camera_t local_camera_id = 0; local_camera_id < camera2->NumLocalCameras(); ++local_camera_id) {
        Eigen::Vector4d qvec;
        Eigen::Vector3d tvec;
        camera2->GetLocalCameraExtrinsic(local_camera_id, qvec, tvec);
        Eigen::Vector3d C = -QuaternionToRotationMatrix(qvec).transpose() * tvec;
        Eigen::Matrix3d skew_C;
        skew_C << 0.0, -C(2), C(1), C(2), 0.0, -C(0), -C(1), C(0), 0.0;
        M2[local_camera_id] = R * skew_C;
    }

    Eigen::MatrixXd A(correspondences.size(), 3);
    Eigen::VectorXd B(correspondences.size());
    for (size_t i = 0; i < correspondences.size(); ++i) {
        Eigen::Vector3d f1 = correspondences.at(i).first;
        Eigen::Vector3d f2 = correspondences.at(i).second;

        Eigen::Vector3d Rf2 = R * f2;
        Eigen::Vector3d G = f1.cross(Rf2);
        A(i, 0) = G(0);
        A(i, 1) = G(1);
        A(i, 2) = G(2);

        uint32_t local_camera_id1 = local_camera_pairs[i].first;
        uint32_t local_camera_id2 = local_camera_pairs[i].second;

        Eigen::Matrix3d M12 = M1[local_camera_id1] - M2[local_camera_id2];
        double b = f1.dot(M12 * f2);
        B(i) = -b;
    }

    // Eigen::Vector3d X = A.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(B);
    Eigen::Vector3d X = A.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(B);
    *relative_position = -R.transpose() * X.normalized();
#endif
    // The position solver above does not consider the sign of the relative
    // position. We can determine the sign by choosing the sign that puts the most
    // points in front of the camera.
    if (!MajorityOfPointsInFrontOfCameras(correspondences, rotation1, rotation2, *relative_position)) {
        *relative_position *= -1.0;
    }

    return true;
}

void RefineRelativeTranslationsWithKnownRotations(
        const Reconstruction& reconstruction,
        const std::unordered_map<image_t , Eigen::Vector3d>& orientations,
        const int num_threads,
        std::shared_ptr<SceneGraphContainer> scene_graph_container,
        std::unordered_map<image_pair_t , Eigen::Vector3d> &relative_positions) {
   // CHECK_GE(num_threads, 1);
    relative_positions.clear();
    auto const &image_pairs = scene_graph_container->CorrespondenceGraph()->ImagePairs();
    for(auto &image_pair : image_pairs){
        image_t image_id1, image_id2;
        sensemap::utility::PairIdToImagePair(image_pair.first, &image_id1, &image_id2);
        const Image& image1 = reconstruction.Image(image_id1);
        const Image& image2 = reconstruction.Image(image_id2);
        const Camera& camera1 = reconstruction.Camera(image1.CameraId());
        const Camera& camera2 = reconstruction.Camera(image2.CameraId());
        const std::vector<uint32_t>& local_image_indices1 = image1.LocalImageIndices();
        const std::vector<uint32_t>& local_image_indices2 = image2.LocalImageIndices();

        if(orientations.count(image_id1) == 0 || orientations.count(image_id2) == 0){
            continue;
        }

        // Get all feature correspondences common to both images.
        auto corrs = scene_graph_container->CorrespondenceGraph()->FindCorrespondencesBetweenImages(
                image_id1, image_id2);

        // std::vector<std::pair<Eigen::Vector2d, Eigen::Vector2d> > matches;
        std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d> > matches;
        matches.reserve(corrs.size());
//#ifdef RIG_TRANSLATION_SOLVER
        std::vector<std::pair<uint32_t, uint32_t> > local_camera_pairs;
        local_camera_pairs.reserve(corrs.size());
//#endif
        for(const auto& cor : corrs) {
            Eigen::Vector3d point1, point2;
            local_camera_t local_camera_id1, local_camera_id2;
            if (camera1.NumLocalCameras() > 1) {
                // Remove the effect of calibration.
                local_camera_id1 = local_image_indices1.at(cor.point2D_idx1);
                const Eigen::Vector3d undistorted_point = camera1.LocalImageToBearing(local_camera_id1, image1.Point2D(cor.point2D_idx1).XY());
                // Apply rotation.
                Eigen::Vector4d local_qvec1;
                Eigen::Vector3d local_tvec1;
                camera1.GetLocalCameraExtrinsic(local_camera_id1, local_qvec1, local_tvec1);
                Eigen::Matrix3d local_R1 = QuaternionToRotationMatrix(local_qvec1);
                point1 = local_R1.transpose() * undistorted_point;
            } else {
                point1 = camera1.ImageToWorld(image1.Point2D(cor.point2D_idx1).XY()).homogeneous();
            }
            if (camera2.NumLocalCameras() > 1) {
                // Remove the effect of calibration.
                local_camera_id2 = local_image_indices2.at(cor.point2D_idx2);
                const Eigen::Vector3d undistorted_point = camera1.LocalImageToBearing(local_camera_id2, image2.Point2D(cor.point2D_idx2).XY());
                // Apply rotation.
                Eigen::Vector4d local_qvec2;
                Eigen::Vector3d local_tvec2;
                camera2.GetLocalCameraExtrinsic(local_camera_id2, local_qvec2, local_tvec2);
                Eigen::Matrix3d local_R2 = QuaternionToRotationMatrix(local_qvec2);
                point2 = local_R2.transpose() * undistorted_point;
            } else {
                point2 = camera2.ImageToWorld(image2.Point2D(cor.point2D_idx2).XY()).homogeneous();
            }
            matches.emplace_back(point1, point2);
//#ifdef RIG_TRANSLATION_SOLVER
            local_camera_pairs.emplace_back(local_camera_id1, local_camera_id2);
//#endif
        }

        // std::cout << image_id1 << " " << image_id2 << ": " << matches.size() << std::endl;

        auto relative_qvec = image_pair.second.two_view_geometry.qvec;
        Eigen::Vector3d relative_position = -QuaternionToRotationMatrix(relative_qvec).transpose() * image_pair.second.two_view_geometry.tvec.normalized();
        // std::cout<<"         "<<relative_position[0]<<" "<<relative_position[1]<<" "<<relative_position[2]<<std::endl;
        if (matches.size() > 8) {
            if (camera1.NumLocalCameras() <= 1 && camera2.NumLocalCameras() <= 1) {
                OptimizeRelativePositionWithKnownRotation(matches, FindOrDie(orientations, image_id1),
                        FindOrDie(orientations, image_id2), &relative_position);
            } else {
//#ifdef RIG_TRANSLATION_SOLVER
                OptimizeRelativePositionWithKnownRotationRig(matches, local_camera_pairs, &camera1, &camera2,
                       FindOrDie(orientations, image_id1), FindOrDie(orientations, image_id2), &relative_position);
//##endif
            }
        }
        relative_positions.emplace(image_pair.first, relative_position);
        // std::cout<<"         "<<relative_position[0]<<" "<<relative_position[1]<<" "<<relative_position[2]<<std::endl;
    }
}

// Rotate the translation direction based on the known orientation such that the
// translation is in the global reference frame.
std::unordered_map<image_pair_t , Eigen::Vector3d>
RotateRelativeTranslationsToGlobalFrame(
        const std::unordered_map<image_t , Eigen::Vector3d>& orientations,
        const std::unordered_map<image_pair_t , Eigen::Vector3d>& relative_positions,
        const EIGEN_STL_UMAP(image_pair_t, struct CorrespondenceGraph::ImagePair)& image_pairs) {
    std::unordered_map<image_pair_t , Eigen::Vector3d> rotated_translations;
    rotated_translations.reserve(orientations.size());

    for (const auto& image_pair : image_pairs) {
        image_t image_id1, image_id2;
        sensemap::utility::PairIdToImagePair(image_pair.first, &image_id1, &image_id2);

        if(orientations.count(image_id1) == 0 || orientations.count(image_id2) == 0){
            continue;
        }

        const Eigen::Vector3d image_to_world_rotation = -1.0 * FindOrDie(orientations, image_id1);
        Eigen::Vector3d rotated_translation;
        ceres::AngleAxisRotatePoint(image_to_world_rotation.data(),
                                    relative_positions.at(image_pair.first).data(),
                                    rotated_translation.data());
        rotated_translations.emplace(image_pair.first, rotated_translation);
    }
    return rotated_translations;
}

// This chooses a random axis based on the given relative translations.
void ComputeMeanVariance(
        const std::unordered_map<image_pair_t , Eigen::Vector3d>& relative_translations,
        Eigen::Vector3d* mean, Eigen::Vector3d* variance) {
    mean->setZero();
    variance->setZero();
    for (const auto& translation : relative_translations) {
        *mean += translation.second;
    }
    *mean /= static_cast<double>(relative_translations.size());

    for (const auto& translation : relative_translations) {
        *variance += (translation.second - *mean).cwiseAbs2();
    }
    *variance /= static_cast<double>(relative_translations.size() - 1);
}

// Projects all the of the translation onto the given axis.
std::unordered_map<image_pair_t , double> ProjectTranslationsOntoAxis(
        const Eigen::Vector3d& axis,
        const std::unordered_map<image_pair_t , Eigen::Vector3d>& relative_translations) {
    std::unordered_map<image_pair_t , double> projection_weights;
    projection_weights.reserve(relative_translations.size());

    for (const auto& relative_translation : relative_translations) {
        const double projection_weight = relative_translation.second.dot(axis);
        projection_weights.emplace(relative_translation.first, projection_weight);
    }
    return projection_weights;
}

// Helper struct to maintain the graph for the translation projection problem.
struct MFASNode {
    std::unordered_map<image_t , double> incoming_nodes;
    std::unordered_map<image_t, double> outgoing_nodes;
    double incoming_weight = 0;
    double outgoing_weight = 0;
};

// Find the next image to add to the order. We attempt to choose a source (i.e.,
// a node with no incoming edges) or choose a node based on a heuristic such
// that it has the most source-like properties.
image_t FindNextImageInOrder(
        const std::unordered_map<image_t , MFASNode>& degrees_for_image) {
    image_t best_choice = kInvalidImageId;
    double best_score = 0;
    for (const auto& image : degrees_for_image) {
        // If the image is a source image, return it.
        if (image.second.incoming_nodes.size() == 0) {
            return image.first;
        }

        // Otherwise, keep track of the max score seen so far.
        const double score = (image.second.outgoing_weight + 1.0) / (image.second.incoming_weight + 1.0);
        if (score > best_score) {
            best_choice = image.first;
            best_score = score;
        }
    }

    return best_choice;
}

// Based on the 1D translation projections, compute an ordering of the
// translations.
std::unordered_map<image_t , int> OrderTranslationsFromProjections(
        const std::unordered_map<image_pair_t , double>&
        translation_direction_projections) {
    // Compute the degrees of all vertices as the sum of weights coming in or out.
    std::unordered_map<image_t , MFASNode> degrees_for_image;
    for (const auto& translation_projection : translation_direction_projections) {
        image_t image_id1, image_id2;
        sensemap::utility::PairIdToImagePair(translation_projection.first, &image_id1, &image_id2);

        const image_pair_t image_id_pair = (translation_projection.second > 0)
                ? translation_projection.first : sensemap::utility::ImagePairToPairId(image_id2, image_id1);

        // Update the MFAS entry.
        const double weight = std::abs(translation_projection.second);
        degrees_for_image[image_id2].incoming_weight += weight;
        degrees_for_image[image_id1].outgoing_weight += weight;
        degrees_for_image[image_id2].incoming_nodes.emplace(image_id1, weight);
        degrees_for_image[image_id1].outgoing_nodes.emplace(image_id2, weight);
    }

    // Compute the ordering.
    const int num_iamges = degrees_for_image.size();
    std::unordered_map<image_t , int> translation_ordering;
    for (int i = 0; i < num_iamges; i++) {
        // Find the next image to add.
        const image_t next_image_in_order = FindNextImageInOrder(degrees_for_image);
        translation_ordering[next_image_in_order] = i;

        // Update the MFAS graph and remove the next image from the degrees_for_image.
        const auto& next_image_info = FindOrDie(degrees_for_image, next_image_in_order);
        for (auto& neighbor_info : next_image_info.incoming_nodes) {
            degrees_for_image[neighbor_info.first].outgoing_weight -= neighbor_info.second;
            degrees_for_image[neighbor_info.first].outgoing_nodes.erase(next_image_in_order);
        }
        for (auto& neighbor_info : next_image_info.outgoing_nodes) {
            degrees_for_image[neighbor_info.first].incoming_weight -= neighbor_info.second;
            degrees_for_image[neighbor_info.first].incoming_nodes.erase(next_image_in_order);
        }
        degrees_for_image.erase(next_image_in_order);
    }

    return translation_ordering;
}

void TranslationFilteringIteration(
        const std::unordered_map<image_pair_t , Eigen::Vector3d>& relative_translations,
        const Eigen::Vector3d& direction_mean,
        const Eigen::Vector3d& direction_variance,
        std::unordered_map<image_pair_t , double>* bad_edge_weight) {


    // Get a random vector to project all relative translations on to.
    const Eigen::Vector3d random_axis =
            Eigen::Vector3d(
                    RandomGaussian(direction_mean[0], direction_variance[0]),
                    RandomGaussian(direction_mean[1], direction_variance[1]),
                    RandomGaussian(direction_mean[2], direction_variance[2])).normalized();

    // Project all vectors.
    const std::unordered_map<image_pair_t , double>& translation_direction_projections =
            ProjectTranslationsOntoAxis(random_axis, relative_translations);

    // Compute ordering.
    const std::unordered_map<image_t , int>& translation_ordering =
            OrderTranslationsFromProjections(translation_direction_projections);

    // Compute bad edge weights.
    for (auto& edge : *bad_edge_weight) {
        image_t image_id1, image_id2;
        sensemap::utility::PairIdToImagePair(edge.first, &image_id1, &image_id2);
        if(translation_ordering.count(image_id1) == 0 || translation_ordering.count(image_id2) == 0){
            continue;
        }
        const int ordering_diff =
                FindOrDie(translation_ordering, image_id2) - FindOrDie(translation_ordering, image_id1);
        const double& projection_weight_of_edge =
                FindOrDieNoPrint(translation_direction_projections, edge.first);

        VLOG(3) << "Edge (" << image_id1 << ", " << image_id2
                << ") has ordering diff of " << ordering_diff
                << " and a projection of " << projection_weight_of_edge << " from "
                << FindOrDieNoPrint(relative_translations, edge.first).transpose();
        // If the ordering is inconsistent, add the absolute value of the bad weight
        // to the aggregate bad weight.
        if ((ordering_diff < 0 && projection_weight_of_edge > 0) ||
            (ordering_diff > 0 && projection_weight_of_edge < 0)) {
            edge.second += std::abs(projection_weight_of_edge);
        }
    }
}

void FilterImagePairsFromRelativeTranslation(
        const GlobalMapper::Options &options,
        const std::unordered_map<image_t, Eigen::Vector3d>& orientations,
        const std::unordered_map<image_pair_t , Eigen::Vector3d>& relative_positions,
        std::shared_ptr<SceneGraphContainer> scene_graph_container) {
    auto const &image_pairs = scene_graph_container->CorrespondenceGraph()->ImagePairs();

    // Weights of edges that have been accumulated throughout the iterations. A
    // higher weight means the edge is more likely to be bad.
    std::unordered_map<image_pair_t , double> bad_edge_weight;
    for (const auto& image_pair : image_pairs) {
        bad_edge_weight[image_pair.first] = 0.0;
    }

    // Compute the adjusted translations so that they are oriented in the global
    // frame.
    const std::unordered_map<image_pair_t , Eigen::Vector3d>& rotated_translations =
            RotateRelativeTranslationsToGlobalFrame(orientations, relative_positions, image_pairs);

    Eigen::Vector3d translation_mean, translation_variance;
    ComputeMeanVariance(rotated_translations, &translation_mean, &translation_variance);

    TranslationFilteringIteration(rotated_translations, translation_mean, translation_variance, &bad_edge_weight);


    // MinimumSpanningTree<image_t, float> mst_extractor;

    // for (auto image_pair : image_pairs) {
    //     image_t image_id1;
    //     image_t image_id2;
    //     utility::PairIdToImagePair(image_pair.first, &image_id1, &image_id2);

    //     mst_extractor.AddEdge(image_id1, image_id2, -image_pair.second.two_view_geometry.confidence);
    //     //sorted_image_pairs.emplace_back(image_pair.first, image_pair.second.num_correspondences);
    // }
    // std::unordered_set<std::pair<image_t, image_t> > minimum_spanning_tree;
    // mst_extractor.Extract(&minimum_spanning_tree);

    // std::unordered_set<image_pair_t> tree_image_pair_ids;
    // for (auto image_pair : minimum_spanning_tree) {
    //     auto pair_id = utility::ImagePairToPairId(image_pair.first, image_pair.second);
    //     tree_image_pair_ids.insert(pair_id);
    // }

    std::unordered_set<image_pair_t> tree_image_pair_ids = *scene_graph_container->GetTreeEdges(true).get();

    // Remove all the bad edges.
    const double max_aggregated_projection_tolerance =
        options.translation_filtering_projection_tolerance * options.translation_filtering_num_iterations;
    int num_image_pairs_removed = 0;
    for (const auto& image_pair : bad_edge_weight) {
        if (tree_image_pair_ids.find(image_pair.first) != tree_image_pair_ids.end()) {
            continue;
        }
        image_t image_id1, image_id2;
        sensemap::utility::PairIdToImagePair(image_pair.first, &image_id1, &image_id2);
        VLOG(3) << "Image pair (" << image_id1 << ", " << image_id2
            << ") projection = " << image_pair.second;
        if (image_pair.second > max_aggregated_projection_tolerance) {
            scene_graph_container->CorrespondenceGraph()->DeleteCorrespondences(image_id1, image_id2);
            ++num_image_pairs_removed;
        }
    }

    LOG(INFO) << "Removed " << num_image_pairs_removed
            << " image pairs by relative translation filtering.";
}


} //unnamed namespace

bool GlobalMapper::Options::Check() const {

    return true;
}

GlobalMapper::GlobalMapper(
    std::shared_ptr<SceneGraphContainer> scene_graph_container)
    : scene_graph_container_(scene_graph_container),
      reconstruction_(nullptr){
}

void GlobalMapper::BeginReconstruction(
        std::shared_ptr<Reconstruction> reconstruction) {
    std::cout << "GlobalMapper::BeginReconstruction" << std::endl;
    CHECK(!reconstruction_.get());
    reconstruction_ = reconstruction;
    scene_graph_container_->CorrespondenceGraph()->CalculateImageNeighbors();
    reconstruction_->SetUp(scene_graph_container_);
    triangulator_.reset(new IncrementalTriangulator(
            scene_graph_container_->CorrespondenceGraph(), reconstruction));
    unlocalized_views_.reserve(reconstruction_->NumImages());
    for (const auto & image : reconstruction_->Images()) {
        if (!image.second.IsRegistered()) {
            unlocalized_views_.insert(image.second.ImageId());
        }
    }
}

void GlobalMapper::EndReconstruction(const bool discard) {
    CHECK_NOTNULL(reconstruction_.get());

    reconstruction_->TearDown();
    reconstruction_ = std::shared_ptr<Reconstruction>();
    triangulator_.reset();
}

bool GlobalMapper::FilterInitialImageGraph(const GlobalMapper::Options &options) {

    std::unordered_set<image_pair_t> image_pairs_to_remove;;
    const auto &image_pairs = scene_graph_container_->CorrespondenceGraph()->ImagePairs();
    for(const auto &image_pair : image_pairs){
        if(image_pair.second.num_correspondences < options.min_num_two_view_inliers){
            image_pairs_to_remove.insert(image_pair.first);
        }
    }
    for(auto image_pair : image_pairs_to_remove){
        image_t image_id1, image_id2;
        sensemap::utility::PairIdToImagePair(image_pair, &image_id1, &image_id2);
        scene_graph_container_->CorrespondenceGraph()->DeleteCorrespondences(image_id1, image_id2);
    }

    // Only reconstruct the largest connected component.
    RemoveDisconnectedImagePairs(scene_graph_container_);
    return scene_graph_container_->CorrespondenceGraph()->NumImagePairs() >= 1;
}

bool GlobalMapper::EstimateGlobalRotations(const GlobalMapper::Options &options) {
    const auto &image_pairs = scene_graph_container_->CorrespondenceGraph()->ImagePairs();
    mst_nodes_.clear();

    // Choose the global rotation estimation type.
    std::unique_ptr<GlobalRotationEstimator> rotation_estimator;
    switch (options.global_rotation_estimator_type) {
        case GlobalRotationEstimatorType::ROBUST_L1L2: {
            // Initialize the orientation estimations by walking along the maximum
            // spanning tree.
            std::unordered_set<std::pair<image_t , image_t> > mst;
            auto view_graph = *scene_graph_container_->CorrespondenceGraph();
            mst_nodes_ = OrientationsFromMaximumSpanningTree(view_graph, &orientations_, mst);
            auto root_id = mst_nodes_[0];
//            auto good_pairs = ViewGraphFilteringFromMaximumSpanningTree(mst,view_graph);
//            std::cout<<"good pairs : "<<good_pairs.size()<<std::endl;
//            for(auto& pair : image_pairs){
//                if(good_pairs.count(pair.first) != 0){
//                    continue;
//                }
//                image_t image_id1, image_id2;
//                sensemap::utility::PairIdToImagePair(pair.first, &image_id1, &image_id2);
//                scene_graph_container_->CorrespondenceGraph()->DeleteCorrespondences(image_id1, image_id2);
//            }
//
//            scene_graph_container_->CorrespondenceGraph()->ExportToGraph(   "./filtered_scene_graph.png");

            RobustRotationEstimator::Options robust_rotation_estimator_options;

            if(options.use_rotation_prior_constrain) {
                image_t refer_id = kInvalidImageId;
                for (auto id : mst_nodes_) {
                    if (scene_graph_container_->Image(id).HasQvecPrior() &&
                        scene_graph_container_->Image(id).PriorQvecGood()) {
                        refer_id = id;
                        break;
                    }
                }
                if (refer_id != kInvalidImageId) {
                    Eigen::Vector3d refer_proir, relative_prior;
                    ceres::QuaternionToAngleAxis(scene_graph_container_->Image(refer_id).QvecPrior().data(),
                                                 refer_proir.data());
                    relative_prior = MultiplyRotations(-orientations_[refer_id], refer_proir);
                    for (auto& orientation : orientations_) {
                        orientation.second = MultiplyRotations(orientation.second, relative_prior);
                    }
                    std::map<camera_t, Eigen::Vector3d> orientation_priors;
                    for (const auto& image_id : scene_graph_container_->GetImageIds()) {
                        const auto& image = scene_graph_container_->Image(image_id);
                        if (image.HasQvecPrior() && image.RtkFlag() == 50 && image.PriorQvecGood()) {
                            Eigen::Vector3d prior;
                            ceres::QuaternionToAngleAxis(image.QvecPrior().data(), prior.data());
                            orientation_priors[image_id] = prior;
                            orientations_[image_id] = prior;
                        }
                    }
                    if (!orientation_priors.empty()) {
                        robust_rotation_estimator_options.has_proir = true;
                        robust_rotation_estimator_options.prior_weight = options.rotation_prior_constrain_weight;
                        robust_rotation_estimator_options.orientation_proirs = orientation_priors;
                        robust_rotation_estimator_options.max_num_l1_iterations = 5;
                    }
                }
            } else {
                robust_rotation_estimator_options.max_num_l1_iterations = 5;
                std::cout<<"Rotation Averaging without priors !"<<std::endl;

            }
            rotation_estimator.reset(
                    new RobustRotationEstimator(robust_rotation_estimator_options));
            break;
        }
        case GlobalRotationEstimatorType::NONLINEAR: {
            // Initialize the orientation estimations by walking along the maximum
            // spanning tree.
//            OrientationsFromMaximumSpanningTree(*view_graph_, &orientations_);
//            rotation_estimator.reset(new NonlinearRotationEstimator());
            break;
        }
        case GlobalRotationEstimatorType::LINEAR: {
//            // Set the constructor variable to true to weigh each term by the inlier
//            // count.
//            rotation_estimator.reset(new LinearRotationEstimator());
            break;
        }
        default: {
            LOG(FATAL) << "Invalid type of global rotation estimation chosen.";
            break;
        }
    }

    // Return false if the rotation estimation does not succeed.
    if (!rotation_estimator->EstimateRotations(image_pairs, &orientations_)) {
        return false;
    }


    // Set the camera orientations of all images that were successfully estimated.
    for (const auto& orientation : orientations_) {
        Eigen::Vector4d qvec;
        ceres::AngleAxisToQuaternion(orientation.second.data(), qvec.data());
        reconstruction_->Image(orientation.first).SetQvec(qvec);
//        Eigen::Vector3d prior_tvec = QuaternionToRotationMatrix(qvec) * - reconstruction_->Image(orientation.first).TvecPrior();
//        reconstruction_->Image(orientation.first).SetTvec(prior_tvec);
//        reconstruction_->RegisterImage(orientation.first);
    }

    return !orientations_.empty();
}



void GlobalMapper::FilterRotations(const GlobalMapper::Options &options) {
// Filter image pairs based on the relative rotation and the estimated global
    // orientations.
    FilterImagePairsFromOrientation(
            orientations_,
            options.rotation_filtering_max_difference_degrees,
            scene_graph_container_);
    // Remove any disconnected images from the estimation.
    const std::unordered_set<image_t> removed_images =
            RemoveDisconnectedImagePairs(scene_graph_container_);
    for (const image_t removed_image : removed_images) {
        orientations_.erase(removed_image);
    }
}

void GlobalMapper::OptimizePairwiseTranslations(const GlobalMapper::Options &options) {
    if (options.refine_relative_translations_after_rotation_estimation) {
        RefineRelativeTranslationsWithKnownRotations(*reconstruction_,
                                                     orientations_,
                                                     options.num_threads,
                                                     scene_graph_container_,
                                                     relative_positions_);
    }
}

void GlobalMapper::FilterRelativeTranslation(const GlobalMapper::Options &options) {
//    if (options.extract_maximal_rigid_subgraph) {
//        LOG(INFO) << "Extracting maximal rigid component of viewing graph to "
//                     "determine which cameras are well-constrained for position "
//                     "estimation.";
//        ExtractMaximallyParallelRigidSubgraph(orientations_, scene_graph_container_);
//    }
    // Filter potentially bad relative translations.
    if (options.filter_relative_translations_with_1dsfm) {
        LOG(INFO) << "Filtering relative translations with 1DSfM filter.";
        FilterImagePairsFromRelativeTranslation(options,
                                               orientations_,
                                               relative_positions_,
                                               scene_graph_container_);
    }

    // Remove any disconnected images from the estimation.
    const std::unordered_set<image_t > removed_images =
            RemoveDisconnectedImagePairs(scene_graph_container_);
    for (const image_t removed_image : removed_images) {
        orientations_.erase(removed_image);
    }



}

bool GlobalMapper::EstimatePosition(const GlobalMapper::Options &options) {

//    for(auto position : orientations_){
//        auto image_id = position.first;
//        reconstruction_->RegisterImage(image_id);
//        Image &image = reconstruction_->Image(image_id);
//        Eigen::Vector3d tvec = QuaternionToRotationMatrix(image.Qvec()) * -image.TvecPrior();
//        image.SetTvec(tvec);
//    }
//    return true;

    // Estimate position.
    const auto& image_pairs = scene_graph_container_->CorrespondenceGraph()->ImagePairs();
    std::unique_ptr<GlobalPositionEstimator> position_estimator;

    // Choose the global position estimation type.
    switch (options.global_position_estimator_type) {
        case GlobalPositionEstimatorType::LEAST_UNSQUARED_DEVIATION: {
//            position_estimator.reset(new LeastUnsquaredDeviationPositionEstimator(
//                    options_.least_unsquared_deviation_position_estimator_options));
            break;
        }
        case GlobalPositionEstimatorType::NONLINEAR: {
            NonlinearPositionEstimator::Options nonlinear_position_estimator_options;
            if(options.use_translation_prior_constrain){
                nonlinear_position_estimator_options.use_position_prior = true;
                nonlinear_position_estimator_options.robust_loss_width = 0.1;
                nonlinear_position_estimator_options.min_num_points_per_view = 100;
                nonlinear_position_estimator_options.point_to_camera_weight = 10;
                nonlinear_position_estimator_options.position_prior_weight = options.translation_prior_constrain_weight;
                nonlinear_position_estimator_options.position_prior_weak_weight = options.translation_prior_weak_constrain_weight;
            }else {
                nonlinear_position_estimator_options.use_position_prior = false;
                nonlinear_position_estimator_options.robust_loss_width = 1.0;
                nonlinear_position_estimator_options.min_num_points_per_view = 100;
                nonlinear_position_estimator_options.point_to_camera_weight = 10;
                nonlinear_position_estimator_options.position_prior_weak_weight = options.translation_prior_weak_constrain_weight;
            }
            // nonlinear_position_estimator_options.max_num_iterations = 100;
            nonlinear_position_estimator_options.num_threads = GetEffectiveNumThreads(-1);
            position_estimator.reset(new NonlinearPositionEstimator(
                    nonlinear_position_estimator_options, scene_graph_container_, relative_positions_));
            break;
        }
        case GlobalPositionEstimatorType::LINEAR_TRIPLET: {
//            position_estimator.reset(new LinearPositionEstimator(
//                    options_.linear_triplet_position_estimator_options,
//                    *reconstruction_));
            break;
        }
        default: {
            LOG(FATAL) << "Invalid type of global position estimation chosen.";
            break;
        }
    }

    bool estimate_success =  position_estimator->EstimatePositions(image_pairs,orientations_,&positions_);
    if(!estimate_success)
        return false;

    LOG(INFO) << positions_.size()
          << " camera positions were estimated successfully.";

    if (options.two_steo_refinement_of_position) {
        NonlinearPositionEstimator::Options nonlinear_position_estimator_options;
        nonlinear_position_estimator_options.use_position_prior = true;
        nonlinear_position_estimator_options.max_num_iterations = 200;
        nonlinear_position_estimator_options.robust_loss_width = 0.1;
        nonlinear_position_estimator_options.min_num_points_per_view = 50;
        nonlinear_position_estimator_options.point_to_camera_weight = 0.5;
        nonlinear_position_estimator_options.position_prior_weight = options.translation_prior_constrain_weight;
        nonlinear_position_estimator_options.position_prior_weak_weight = options.translation_prior_weak_constrain_weight;

        nonlinear_position_estimator_options.num_threads = GetEffectiveNumThreads(-1);
        position_estimator.reset(new NonlinearPositionEstimator(
                nonlinear_position_estimator_options, scene_graph_container_, relative_positions_));
        (reinterpret_cast<NonlinearPositionEstimator*>(position_estimator.get()))->RefinePosesWithPrior(image_pairs,orientations_,&positions_);

        // Set the camera orientations of all images that were successfully estimated.
        for (const auto& orientation : orientations_) {
            Eigen::Vector4d qvec;
            ceres::AngleAxisToQuaternion(orientation.second.data(), qvec.data());
            reconstruction_->Image(orientation.first).SetQvec(qvec);
        }
    }

    for(auto position : positions_) {
        auto image_id = position.first;
        reconstruction_->RegisterImage(image_id);
        Image& image = reconstruction_->Image(image_id);
        Eigen::Vector3d tvec = QuaternionToRotationMatrix(image.Qvec()) * -position.second;
        image.SetTvec(tvec);
        image.SetRegistered(true);
    }

    return true;
}

size_t GlobalMapper::TriangulateImage(const IncrementalTriangulator::Options& tri_options,
                                           const image_t image_id) {
    CHECK_NOTNULL(reconstruction_.get());
    return triangulator_->TriangulateImage(tri_options, image_id);
}

size_t GlobalMapper::Retriangulate(const IncrementalTriangulator::Options& tri_options,
                                        std::unordered_set<image_t>* image_set) {
    CHECK_NOTNULL(reconstruction_.get());
    return triangulator_->Retriangulate(tri_options,image_set);
}

size_t GlobalMapper::Retriangulate(const IncrementalTriangulator::Options& tri_options) {
    CHECK_NOTNULL(reconstruction_.get());
    return triangulator_->Retriangulate(tri_options);
}

size_t GlobalMapper::RetriangulateAllTracks(const IncrementalTriangulator::Options& tri_options) {
    CHECK_NOTNULL(reconstruction_.get());
    return triangulator_->RetriangulateAllTracks(tri_options);
}

size_t GlobalMapper::CompleteTracks(const IncrementalTriangulator::Options& tri_options) {
    CHECK_NOTNULL(reconstruction_.get());
    return triangulator_->CompleteAllTracks(tri_options);
}

size_t GlobalMapper::CompleteTracks(const IncrementalTriangulator::Options& tri_options, const std::unordered_set<mappoint_t>& mappoint_ids) {
    CHECK_NOTNULL(reconstruction_.get());
    return triangulator_->CompleteTracks(tri_options, mappoint_ids);
}

size_t GlobalMapper::MergeTracks(const IncrementalTriangulator::Options& tri_options) {
    CHECK_NOTNULL(reconstruction_.get());
    return triangulator_->MergeAllTracks(tri_options);
}

size_t GlobalMapper::MergeTracks(const IncrementalTriangulator::Options& tri_options, const std::unordered_set<mappoint_t>& mappoint_ids) {
    CHECK_NOTNULL(reconstruction_.get());
    return triangulator_->MergeTracks(tri_options, mappoint_ids);
}


void GlobalMapper::EsitimateStructure(const IncrementalTriangulator::Options& tri_options) {
#if 0
    auto tracks = scene_graph_container_->CorrespondenceGraph()->GenerateTracks(2, true, true);
    std::sort(tracks.begin(), tracks.end(),
              [](const Track& t1, const Track& t2) {
                  return t1.Length() > t2.Length();
              });
    std::vector<unsigned char> inlier_masks(tracks.size(), 1);
    int max_cover_per_view = 800;
    scene_graph_container_->CorrespondenceGraph()->TrackSelection(tracks,
                                                                  inlier_masks, max_cover_per_view);
    for (auto track : tracks) {
        reconstruction_->AddMapPoint(Eigen::Vector3d(0, 0, 0), std::move(track));
    }
   RetriangulateAllTracks(tri_options);
#endif
        for(auto pointid : reconstruction_->MapPointIds()){
            reconstruction_->DeleteMapPoint(pointid);
        }

        int triangulated_image_count = 1;
        std::vector<image_t> image_ids = scene_graph_container_->GetImageIds();
        for (const auto image_id : image_ids) {

            Image &image = reconstruction_->Image(image_id);

            Camera &camera = reconstruction_->Camera(image.CameraId());


            PrintHeading1(StringPrintf("Triangulating image #%d - %s (%d / %d)",
                                       image_id, image.Name().c_str(), triangulated_image_count++,
                                       image_ids.size()));
            const size_t num_existing_points3D = image.NumMapPoints();
            std::cout << "  => Image sees " << num_existing_points3D << " / " << image.NumObservations()
                      << " points"
                      << std::endl;

            TriangulateImage(tri_options, image_id);

            std::cout << "  => Triangulated " << (image.NumMapPoints() - num_existing_points3D) << " points"
                      << std::endl;

        }
}

bool GlobalMapper::AdjustGlobalBundle(const GlobalMapper::Options& options, const BundleAdjustmentOptions& ba_options) {
    CHECK_NOTNULL(reconstruction_.get());

    const std::vector<image_t>& reg_image_ids = reconstruction_->RegisterImageIds();

    CHECK_GE(reg_image_ids.size(), 2) << "At least two images must be "
                                         "registered for global "
                                         "bundle-adjustment";

    if (ba_options.use_prior_absolute_location){
        CHECK(reconstruction_->has_gps_prior);
        reconstruction_->AlignWithPriorLocations(options.max_error_gps);
    }


    // Avoid degeneracies in bundle adjustment.
    reconstruction_->FilterObservationsWithNegativeDepth();

    // Configure bundle adjustment.
    BundleAdjustmentConfig ba_config;
    for (const image_t image_id : reg_image_ids) {
        ba_config.AddImage(image_id);
        const Image& image = reconstruction_->Image(image_id);
        const Camera& camera = reconstruction_->Camera(image.CameraId());
        if (camera.IsCameraConstant() || reg_image_ids.size() <= options.num_fix_camera_first) {
            ba_config.SetConstantCamera(image.CameraId());
        }
    }
    CHECK(reconstruction_->RegisterImageIds().size() > 0);
    const image_t first_image_id = reconstruction_->RegisterImageIds()[0];
    CHECK(reconstruction_->ExistsImage(first_image_id));
    const Image& image = reconstruction_->Image(first_image_id);
    const Camera& camera = reconstruction_->Camera(image.CameraId());

    if (!ba_options.use_prior_absolute_location || !reconstruction_->b_aligned) {
        ba_config.SetConstantPose(reg_image_ids[0]);

        int max_t_index = -1;
        double max_t = -1;
        const Image& second_image = reconstruction_->Image(reg_image_ids[1]);
        for(int j = 0; j < 3; ++j){
            if(abs(second_image.Tvec()[j]) > max_t){
                max_t = abs(second_image.Tvec()[j]);
                max_t_index = j;
            }
        }

        ba_config.SetConstantTvec(reg_image_ids[1], {max_t_index});
    }

    // Run bundle adjustment.
    BundleAdjuster bundle_adjuster(ba_options, ba_config);

    if (!bundle_adjuster.Solve(reconstruction_.get())) {
        return false;
    }


    //   // Normalize scene for numerical stability and
    //   // to avoid large scale changes in viewer.
    if(options.ba_normalize_reconstruction){
        reconstruction_->Normalize();
    }

    return true;
}

bool GlobalMapper::SetOrientationWithQvec() {
    orientations_.clear();
    std::vector<image_t> image_ids = scene_graph_container_->GetImageIds();

    for (const auto image_id : image_ids) {
        Eigen::Vector3d prior;
        ceres::QuaternionToAngleAxis(scene_graph_container_->Image(image_id).Qvec().data(), prior.data());
        orientations_[image_id] = prior;
    }
    return true;
}


} // namespace sensemap
