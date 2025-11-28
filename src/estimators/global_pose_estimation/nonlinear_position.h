//
// Created by sensetime on 2021/2/4.
//

#ifndef SENSEMAP_NONLINEAR_POSITION_H_
#define SENSEMAP_NONLINEAR_POSITION_H_

#include <unordered_map>
#include <Eigen/Eigen>
#include <vector>
#include <ceres/ceres.h>

#include "util/types.h"
#include "global_position.h"
#include "container/scene_graph_container.h"
namespace sensemap {
// Estimates the camera position of views given pairwise relative poses and the
// absolute orientations of cameras. Positions are estimated using a nonlinear
// solver with a robust cost function. This solution strategy closely follows
// the method outlined in "Robust Global Translations with 1DSfM" by Wilson and
// Snavely (ECCV 2014)
class NonlinearPositionEstimator : public GlobalPositionEstimator {
public:
    struct Options {

        // Options for Ceres nonlinear solver.
        int num_threads = 1;
        int max_num_iterations = 400;
        double robust_loss_width = 0.1;

        // Minimum number of 3D points to camera correspondences for each
        // camera. These points can help constrain the problem and add robustness to
        // collinear configurations, but are not necessary to compute the position.
        int min_num_points_per_view = 50;

        // The total weight of all point to camera correspondences compared to
        // camera to camera correspondences.
        double point_to_camera_weight = 0.5;

        bool use_position_prior = false;
        double position_prior_weight = 1.0;
        double position_prior_weak_weight = 0.1;
    };

    NonlinearPositionEstimator(
            const NonlinearPositionEstimator::Options& options,
            std::shared_ptr<SceneGraphContainer> scene_graph_container,
            std::unordered_map<image_pair_t , Eigen::Vector3d>& relative_positions);
            //const Reconstruction& reconstruction);

    // Returns true if the optimization was a success, false if there was a
    // failure.
    bool EstimatePositions(
            const EIGEN_STL_UMAP(image_pair_t, struct CorrespondenceGraph::ImagePair) &image_pairs,
            const std::unordered_map<image_t , Eigen::Vector3d>& orientation,
            std::unordered_map<image_t , Eigen::Vector3d>* positions);

    bool RefinePosesWithPrior(
            const EIGEN_STL_UMAP(image_pair_t, struct CorrespondenceGraph::ImagePair) &image_pairs,
            std::unordered_map<image_t , Eigen::Vector3d>& orientation,
            std::unordered_map<image_t , Eigen::Vector3d>* positions);

private:
    // Initialize all cameras to be random.
    void InitializeRandomPositions(
            const std::unordered_map<image_t , Eigen::Vector3d>& orientations,
            std::unordered_map<image_t , Eigen::Vector3d>* positions);

    // Creates camera to camera constraints from relative translations.
    void AddCameraToCameraConstraints(
            const std::unordered_map<image_t , Eigen::Vector3d>& orientations,
            std::unordered_map<image_t , Eigen::Vector3d>* positions);

    // Creates point to camera constraints.
    void AddPointToCameraConstraints(
            const std::unordered_map<image_t , Eigen::Vector3d>& orientations,
            std::unordered_map<image_t, Eigen::Vector3d>* positions);

    // Creates point to camera constraints.
    void AddPriorPositionsConstraints(
            std::unordered_map<image_t, Eigen::Vector3d>* positions,
            const Eigen::Matrix3x4d sRT = Eigen::Matrix3x4d::Identity());

    // Determines which tracks should be used for point to camera constraints. A
    // greedy approach is used so that the fewest number of tracks are chosen such
    // that all cameras have at least k point to camera constraints.
    int FindTracksForProblem(
            const std::unordered_map<image_t, Eigen::Vector3d>& global_poses,
            std::unordered_set<track_t>* tracks_to_add);

    // Sort the tracks by the number of views that observe them.
    std::vector<track_t> GetTracksSortedByNumViews(
            //const Reconstruction& reconstruction,
            const struct Image& iamge,
            const std::unordered_set<track_t>& existing_tracks);

    // Adds all point to camera constraints for a given track.
    void AddTrackToProblem(
            const track_t track_id,
            const std::unordered_map<image_t , Eigen::Vector3d>& orientations,
            const double point_to_camera_weight,
            std::unordered_map<image_t , Eigen::Vector3d>* positions);

    // Adds the points and cameras to parameters groups 0 and 1 respectively. This
    // allows for the Schur-based methods to take advantage of the sparse block
    // structure of the problem by eliminating points first, then cameras. This
    // method is only called if triangulated points are used when solving the
    // problem.
    void AddCamerasAndPointsToParameterGroups(
            std::unordered_map<image_t , Eigen::Vector3d>* positions);

    const NonlinearPositionEstimator::Options options_;
    std::shared_ptr<SceneGraphContainer> scene_graph_container_;
    //const std::unordered_map<ViewIdPair, TwoViewInfo>* view_pairs_;
    std::unordered_map<image_pair_t , Eigen::Vector3d> relative_positions_;
    std::unordered_map<image_pair_t , double> relative_positions_weights_;
    //std::shared_ptr<RandomNumberGenerator> rng_;
    EIGEN_STL_UMAP(track_t, Eigen::Vector3d) triangulated_points_;
    std::vector<Track> traks_;
    std::unique_ptr<ceres::Problem> problem_;
    ceres::Solver::Options solver_options_;

    friend class EstimatePositionsNonlinearTest;

};


}

#endif //SENSEMAP_NONLINEAR_POSITION_H_
