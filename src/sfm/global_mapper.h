//Copyright (c) 2021, SenseTime Group.
//All rights reserved.


#ifndef SENSEMAP_GLOBAL_MAPPER_H_
#define SENSEMAP_GLOBAL_MAPPER_H_

#include <unordered_set>

#include "util/types.h"
#include "optim/bundle_adjustment.h"
#include "base/reconstruction.h"
#include "incremental_triangulator.h"


namespace sensemap{

class GlobalMapper{

// The recommended type of rotations solver is the Robust L1-L2 method. This
// method is scalable, extremely accurate, and very efficient. See the
// global_pose_estimation directory for more details.
enum class GlobalRotationEstimatorType {
    ROBUST_L1L2 = 0,
    NONLINEAR = 1,
    LINEAR = 2
};


// Global position estimation methods.
//   NONLINEAR: This method minimizes the nonlinear pairwise translation
//     constraint to solve for positions.
//   LINEAR_TRIPLET: This linear method computes camera positions by
//     minimizing an error for image triplets. Essentially, it tries to
//     enforce a loop/triangle constraint for triplets.
//   LEAST_UNSQUARED_DEVIATION: This robust method uses the least unsquared
//     deviation instead of least squares. It is essentially an L1 solver.
enum class GlobalPositionEstimatorType {
    NONLINEAR = 0,
    LINEAR_TRIPLET = 1,
    LEAST_UNSQUARED_DEVIATION = 2,
};

public:
    struct Options{

        GlobalRotationEstimatorType global_rotation_estimator_type =
                GlobalRotationEstimatorType::ROBUST_L1L2;

        GlobalPositionEstimatorType global_position_estimator_type =
                GlobalPositionEstimatorType::NONLINEAR;

        std::string image_path = "";


        // Maximum reprojection error in absolute pose estimation.
        double abs_pose_max_error = 12.0;

        // Minimum number of inliers in absolute pose estimation.
        int abs_pose_min_num_inliers = 30;

        // Minimum inlier ratio in absolute pose estimation.
        double abs_pose_min_inlier_ratio = 0.25;

        // Whether to estimate the focal length in absolute pose estimation.
        bool abs_pose_refine_focal_length = true;

        // Whether to estimate the extra parameters in absolute pose estimation.
        bool abs_pose_refine_extra_params = true;


        bool single_camera = false;

        // Thresholds for bogus camera parameters. Images with bogus camera
        // parameters are filtered and ignored in triangulation.
        double min_focal_length_ratio = 0.1;  // Opening angle of ~130deg
        double max_focal_length_ratio = 10;   // Opening angle of ~5deg
        double max_extra_param = 1;

        // Maximum reprojection error in pixels for observations.
        double filter_max_reproj_error = 4.0;

        // Minimum triangulation angle in degrees for stable Map Points.
        double filter_min_tri_angle = 1.5;

        double filter_min_tri_angle_final = 1.5;
        double filter_max_reproj_error_final = 4.0;
        int filter_min_track_length_final = 2;

        // Number of threads.
        int num_threads = -1;

        int num_fix_camera_first = 5;

        //global sfm prior
        bool use_rotation_prior_constrain = false;
        double rotation_prior_constrain_weight = 1.0;
        bool use_translation_prior_constrain = false;
        double translation_prior_constrain_weight = 0.5;
        double translation_prior_weak_constrain_weight = 0.1;

        double two_steo_refinement_of_position = false;

        bool use_rotation_prior_init = true;

        // prior reconstructio information
        bool have_prior_pose = false;
        std::unordered_map<image_t, Eigen::Vector4d> prior_rotations;
        std::unordered_map<image_t, Eigen::Vector3d> prior_translations;

        std::unordered_map<image_t, std::pair<Eigen::Vector3d,int>> prior_locations_gps;
        bool has_gps_prior = false;

        double max_error_gps = 3.0;
        int min_image_num_for_gps_error = 10;

        // Only use latitude and longtitude for optimization or not
        bool optimization_use_horizontal_gps_only = false;

        // whether to normalize the reconstruction after global ba
        bool ba_normalize_reconstruction = false;

        // Maximum reprojection error. This is the threshold used for filtering
        // outliers after bundle adjustment.
        double max_reprojection_error_in_pixels = 5.0;

        // Any edges in the view graph with fewer than min_num_two_view_inliers will
        // be removed as an initial filtering step.
        int min_num_two_view_inliers = 30;

        // --------------- RANSAC Options --------------- //
        double ransac_confidence = 0.9999;
        int ransac_min_iterations = 50;
        int ransac_max_iterations = 1000;
        bool ransac_use_mle = true;

        // --------------- Rotation Filtering Options --------------- //

        // After orientations are estimated, view pairs may be filtered/removed if the
        // relative rotation of the view pair differs from the relative rotation
        // formed by the global orientation estimations. Adjust this threshold to
        // control the threshold at which rotations are filtered. See
        // theia/sfm/filter_view_pairs_from_orientation.h
        double rotation_filtering_max_difference_degrees = 10.0;

        // --------------- Position Filtering Options --------------- //

        // Refine the relative translations based on the epipolar error and known
        // rotation estimations. This improve the quality of the translation
        // estimation.
        bool refine_relative_translations_after_rotation_estimation = true;

        // If true, the maximal rigid component of the viewing graph will be
        // extracted. This means that only the cameras that are well-constrained for
        // position estimation will be used. This method is somewhat slow, so enabling
        // it will cause a performance hit in terms of efficiency.
        //
        // NOTE: This method does not attempt to remove outlier 2-view geometries, it
        // only determines which cameras are well-conditioned for position estimation.
        bool extract_maximal_rigid_subgraph = false;

        // If true, filter the pairwise translation estimates to remove potentially
        // bad relative poses. Removing potential outliers can increase the
        // performance of position estimation.
        bool filter_relative_translations_with_1dsfm = true;

        // Before the camera positions are estimated, it is wise to remove any
        // relative translations estimates that are low quality. See
        // theia/sfm/filter_view_pairs_from_relative_translation.h
        int translation_filtering_num_iterations = 48;
        double translation_filtering_projection_tolerance = 0.88;

        // --------------- Global Rotation Estimation Options --------------- //

        // Robust loss function scales for nonlinear estimation.
        double rotation_estimation_robust_loss_scale = 0.1;

        // --------------- Global Position Estimation Options --------------- //
//        NonlinearPositionEstimator::Options nonlinear_position_estimator_options;
//        LinearPositionEstimator::Options linear_triplet_position_estimator_options;
//        LeastUnsquaredDeviationPositionEstimator::Options
//                least_unsquared_deviation_position_estimator_options;

        // For global SfM it may be advantageous to run a partial bundle adjustment
        // optimizing only the camera positions and 3d points while holding camera
        // orientation and intrinsics constant.
        bool refine_camera_positions_and_points_after_position_estimation = false;

        bool Check() const;
    };

    explicit GlobalMapper(
            std::shared_ptr<SceneGraphContainer> scene_graph_container);

    // Prepare the mapper for a new reconstruction.
    void BeginReconstruction(std::shared_ptr<Reconstruction> reconstruction);

    // Cleanup the mapper after the current reconstruction is done. If the
    // model is discarded, the number of total and shared registered images will
    // be updated accordingly.
    void EndReconstruction(const bool discard);

    bool FilterInitialImageGraph(const GlobalMapper::Options &options);

    bool EstimateGlobalRotations(const GlobalMapper::Options &options);

    void FilterRotations(const GlobalMapper::Options &options);

    void OptimizePairwiseTranslations(const GlobalMapper::Options &options);

    void FilterRelativeTranslation(const GlobalMapper::Options &options);

    bool EstimatePosition(const GlobalMapper::Options &options);

    void EsitimateStructure(const IncrementalTriangulator::Options& tri_options);

    // Global bundle adjustment using Ceres Solver or PBA.
    bool AdjustGlobalBundle(const Options& options,
                            const BundleAdjustmentOptions& ba_options);

    bool SetOrientationWithQvec();

    std::vector<image_t> GetMSTNodes(){
        return mst_nodes_;
    };

private:


    size_t TriangulateImage(const IncrementalTriangulator::Options& tri_options,
                     const image_t image_id);
    // Retriangulate image pairs that should have common observations according to
    // the scene graph but don't due to drift, etc. To handle drift, the employed
    // reprojection error thresholds should be relatively large. If the thresholds
    // are too large, non-robust bundle adjustment will break down; if the
    // thresholds are too small, we cannot fix drift effectively.
    size_t Retriangulate(const IncrementalTriangulator::Options& tri_options);
    size_t Retriangulate(const IncrementalTriangulator::Options& tri_options, std::unordered_set<image_t>* image_set);

    // Perform triangulation for all mappoints. Just being invoked after GBA.
    size_t RetriangulateAllTracks(const IncrementalTriangulator::Options& tri_options);

    // Complete tracks by transitively following the scene graph correspondences.
    // This is especially effective after bundle adjustment, since many cameras
    // and point locations might have improved. Completion of tracks enables
    // better subsequent registration of new images.
    size_t CompleteTracks(const IncrementalTriangulator::Options& tri_options);
    size_t CompleteTracks(const IncrementalTriangulator::Options& tri_options, const std::unordered_set<mappoint_t>& mappoint_ids);

    // Merge tracks by using scene graph correspondences. Similar to
    // `CompleteTracks`, this is effective after bundle adjustment and improves
    // the redundancy in subsequent bundle adjustments.
    size_t MergeTracks(const IncrementalTriangulator::Options& tri_options);
    size_t MergeTracks(const IncrementalTriangulator::Options& tri_options, const std::unordered_set<mappoint_t>& mappoint_ids);

    std::shared_ptr<SceneGraphContainer> scene_graph_container_;

    // Class that holds data of the reconstruction.
    std::shared_ptr<Reconstruction> reconstruction_;

    // Class that is responsible for incremental triangulation.
    std::unique_ptr<IncrementalTriangulator> triangulator_;

    // The orientations that are solved for through a global rotation estimation
    // technique. These values are then used to simplify the pose estimation
    // problem to just estimating the unknown camera positions.
    std::unordered_map<image_t, Eigen::Vector3d> orientations_;

    std::unordered_map<image_t, Eigen::Vector3d> positions_;
    std::unordered_map<image_pair_t , Eigen::Vector3d> relative_positions_;

    // A container to keep track of which views need to be localized.
    std::unordered_set<image_t> unlocalized_views_;

    std::vector<image_t> mst_nodes_;
//    // An *ordered* container to keep track of which views have been added to the
//    // reconstruction. This is used to determine which views are optimized during
//    // partial BA.
//    std::vector<image_t> reconstructed_views_;

};

}

#endif //SENSEMAP_GLOBAL_MAPPER_H_
