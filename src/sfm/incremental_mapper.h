//Copyright (c) 2019, SenseTime Group.
//All rights reserved.

#ifndef SENSEMAP_SFM_INCREMENTAL_MAPPER_H_
#define SENSEMAP_SFM_INCREMENTAL_MAPPER_H_

#include <unordered_set>
#include <mutex>
#include <opencv2/opencv.hpp>

#include "util/types.h"
#include "estimators/two_view_geometry.h"
#include "container/scene_graph_container.h"
#include "optim/bundle_adjustment.h"
#include "base/reconstruction.h"
#include "base/similarity_transform.h"
#include "incremental_triangulator.h"
#include "lidar/voxel_map.h"

namespace sensemap {

namespace {
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
}

class IncrementalMapper {
public:
    struct Options {

        std::string image_path = "";

        // Minimum number of inliers for initial image pair.
        int init_min_num_inliers = 100;

        // Maximum error in pixels for two-view geometry estimation for initial
        // image pair.
        double init_max_error = 4.0;

        // Maximum angular error for two-view geometry estimation for initial
        // image pair, this is compatible with spherical camera model
        // 0.23 in angular error is equivalent to 4.0 in pixel error when the 
        // focal length is 1000.
        double init_max_angular_error = 0.40;

        // Maximum forward motion for initial image pair.
        double init_max_forward_motion = 0.95;

        // Minimum triangulation angle for initial image pair.
        double init_min_tri_angle = 16.0;

        // Maximum number of trials to use an image for initialization.
        int init_max_reg_trials = 2;

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

        // Number of images to optimize in local bundle adjustment.
        int local_ba_num_images = 6;

        // Minimum triangulation for images to be chosen in local bundle adjustment.
        double local_ba_min_tri_angle = 6;

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

        int init_min_corrs_intra_view = 30;
        double init_max_depth = 4.0;

        // Maximum number of trials to register an image.
        int max_reg_trials = 3;

        // Number of threads.
        int num_threads = -1;

        // Minimum of disparity in initialization stage.
        double init_min_disparity = 16;

        // triangulation angle to prevent initializing with too wide of a baseline
		double max_triangulation_angle_degrees = 30.0;

        // Initializing frame selection from global rotation estimation.
        bool init_from_global_rotation_estimation = false;

        // Initializing frame selection from feature distribution,
        // feature number, and connections with others.
        bool init_from_uncertainty = true;

        // The bin size for computing feature distribution.
        int init_bin_size = 100;

        // The maximum permitted matches in each bin.
        int max_matches_each_bin = 10;

        double gauss_weight_bin = 5.0;

        double gauss_weight_hist = 10.0;

        int num_fix_camera_first = 5;

        int min_visible_map_point_kf = 100;

        int min_keyframe_step = 1;
        
        int min_pose_inlier_kf = 200;

        double min_distance_kf = 0.8;

        // Whether to exploit depth map to improve reconstruction.
	    bool with_depth = false;

        // Whether to perform subimage matching, it is valid only if camera-rig mode.
        bool sub_matching = false;

        // Whether to support self matching.
        bool self_matching = false;

        // Extrinsics of RGBD camera.
	    std::string rgbd_camera_params = "";

        // Options delayed RGBD constraints. 
        // When enabled, RGBD constraints will be more robust if 3D init fails.
        bool rgbd_delayed_start = false;
        double rgbd_delayed_start_weights = 100.0;

        // Depth weight for filtering 3D points
	    // S.A. filter_max_reproj_error and filter_max_reproj_error_final
        // This function is enabled when `with_depth` is true and this value > 0
        double rgbd_filter_depth_weight = 0.0;

        // Maximum allowed reprojection depth for RGBD
        double rgbd_max_reproj_depth = 0.0;

        // Depth weight for pose refinement after estimation
        // This function is enabled when `with_depth` is true and this value > 0
        double rgbd_pose_refine_depth_weight = 0.0;

        double min_inlier_ratio_verification_with_prior_pose = 0.7;

        // The maximum distance to judge whether a camera point is on the 
        // estimated plane. This threshold is the ralative to the baseline 
        // distance of the initial reconstruction
        double max_distance_to_plane  = 0.05;
        int max_plane_count = 3;

        //======================================================================
        // preserve a sample model if the ratio of its inlier number to that of
        // the best model is above a threshold, this is used in multiple 
        // solution detection of initialization
  	    double min_inlier_ratio_to_best_model = 0.8; 

        bool batched_sfm = false;

        double avg_min_dist_kf_factor = 0.8;

        double mean_max_disparity_kf = 24.0;

        int abs_diff_kf = 50;

        bool robust_camera_pose_estimate = false;

        int consecutive_camera_pose_top_k = 2;

        int consecutive_neighbor_ori = 2;
        
        int consecutive_neighbor_t = 1;

        double consecutive_camera_pose_orientation = 5.0;
        
        double consecutive_camera_pose_t = 20.0;

        int local_region_repetitive = 100;

        double loop_image_weight = 1.0;
        int loop_image_min_id_difference = 20;

        bool offline_slam = false;
        int  initial_two_frame_interval_offline_slam = 20;
        // Two image are deemed as covisible if the number of the covisible 
        // mappoints is larger than the threshold
        int min_covisible_mappoint_num = 30;

        // Two image are deemed as covisible if the ratio of the covisible 
        // mappoints to the correspondences is larger than the threshold
        double min_covisible_mappoint_ratio = 0.2;

        // max number of next candidate images to register for offline slam
        int offline_slam_max_next_image_num = 5;

        // whether to normalize the reconstruction after global ba
        bool ba_normalize_reconstruction = false;

        // Overlap images with other clusters.
        std::unordered_set<image_t> overlap_image_ids;

        // prior reconstructio information
	    bool have_prior_pose = false;
        bool use_prior_aggressively = false;
	    bool prior_force_keyframe = false;
	    std::unordered_map<image_t, Eigen::Vector4d> prior_rotations;
        std::unordered_map<image_t, Eigen::Vector3d> prior_translations;

        std::unordered_map<image_t, std::pair<Eigen::Vector3d,int>> prior_locations_gps;
        std::vector<std::pair<unsigned long, std::pair<Eigen::Vector3d,int>>> original_gps_locations;
	    bool has_gps_prior = false;

        double max_error_gps = 3.0;
        int min_image_num_for_gps_error = 10;
        double max_error_horizontal_gps = 3.0;
        long max_gps_time_offset = 60000;

        // Only use latitude and longtitude for optimization or not
        bool optimization_use_horizontal_gps_only = false;
        
        // Explicit loop
        int max_id_difference_for_loop = 10;
        double min_inconsistent_corr_ratio_for_loop = 0.9;
        int min_loop_pose_inlier_num = 30;
        double loop_weight = 1.0;
        int normal_edge_count_per_image = 5;
        int max_loop_edge_count = 1;
        bool optimize_sim3 = true;
        int loop_closure_max_iter_num = 100;
        int normal_edge_min_common_points = 50;
        double loop_distance_factor_wrt_averge_baseline = 6.0;
        double neighbor_distance_factor_wrt_averge_baseline = 1.5;

        bool debug_info = true;

        // check whether to update original map
        bool map_update = false;
        bool update_old_map = false;
        bool update_with_sequential_mode = false;

        // Method to find and select next best image to register.
        enum class ImageSelectionMethod {
            MAX_VISIBLE_POINTS_NUM,
            MAX_VISIBLE_POINTS_RATIO,
            MIN_UNCERTAINTY,
            WEIGHTED_MIN_UNCERTAINTY,
            TIME_WEIGHT_MIN_UNCERTAINTY
        };

        int direct_mapper_type = 1;

        int min_track_length = 3;

        ImageSelectionMethod image_selection_method = ImageSelectionMethod::MIN_UNCERTAINTY;
        // ImageSelectionMethod image_selection_method = ImageSelectionMethod::TIME_WEIGHT_MIN_UNCERTAINTY;

        // lidar option.
        bool lidar_sfm = false;
        
        std::string lidar_path = "";

        std::string lidar_prior_pose_file = "";

        Eigen::Matrix3x4d lidar_to_cam_matrix = Eigen::Matrix3x4d::Identity();

        bool Check() const;
    };

    struct LocalBundleAdjustmentReport {
        size_t num_merged_observations = 0;
        size_t num_completed_observations = 0;
        size_t num_filtered_observations = 0;
        size_t num_adjusted_observations = 0;
    };

    explicit IncrementalMapper(std::shared_ptr<SceneGraphContainer> scene_graph_container);

    // Prepare the mapper for a new reconstruction.
    void BeginReconstruction(std::shared_ptr<Reconstruction> reconstruction);

    // Cleanup the mapper after the current reconstruction is done. If the
    // model is discarded, the number of total and shared registered images will
    // be updated accordingly.
    void EndReconstruction(const bool discard);

    // Estimate the camera orientations from the relative rotations using a 
    // global rotation estimation algorithm.
    bool EstimateCameraOrientations(const Options &options);

    void LidarSetUp(const Options &options);

    bool FindInitialLidarPair(const image_t image_id1, const image_t image_id2,
                              sweep_t * sweep_id1, sweep_t * sweep_id2);

    sweep_t FindNextSweep(const image_t image_id);

    bool RegisterInitialLidarPair(const Options & options,
                                  const BundleAdjustmentOptions& ba_options, 
                                  const image_t image_id1,
                                  const image_t image_id2,
                                  const sweep_t sweep_id1,
                                  const sweep_t sweep_id2);

    void ImageLidarAlignment(const Options & options, const image_t image_id1, const image_t image_id2, 
                             const sweep_t sweep_id1, const sweep_t sweep_id2);

    void RefineImageLidarAlignment(const Options & options);

    // Find initial image pair to seed the incremental reconstruction. The image
    // pairs should be passed to `RegisterInitialImagePair`. This function
    // automatically ignores image pairs that failed to register previously.
    bool FindInitialImagePair(const Options & options, 
        image_t * image_id1, image_t * image_id2);

    bool FindInitialImagePairUncertainty(const Options & options, 
        image_t * image_id1, image_t * image_id2);

    // Choose two cameras to use as the seed for incremental reconstruction. 
    // These cameras should observe 3D points that are well-conditioned. We 
    // determine the conditioning of 3D points by examining the median viewing 
    // angle of the correspondences between the views.
    bool FindInitialImagePairWithKnownOrientation(
        const Options& options,
        image_t* image_id1,
        image_t* image_id2);

    // Find initial image pair to seed the SLAM-like incrmental reconstruction.
    // The two images should be neighbor in time, and are iteratively selected
    // from the head to the tail of the image sequence.

    bool FindInitialImagePairOfflineSLAM(const Options & options, 
        image_t * image_id1, image_t * image_id2);


    // Find best next image to register in the incremental reconstruction. The
    // images should be passed to `RegisterNextImage`. This function automatically
    // ignores images that failed to registered for `max_reg_trials`.
    std::vector<std::pair<image_t, float>> FindNextImages(const Options & options);

    // find the next image register in the offline slam mode, the image is
    // simply selected sequentially
    std::vector<std::pair<image_t, float>> FindNextImagesOfflineSLAM(const Options & options,
                                                   bool prev_normal_mode = true, bool jump_to_backward = false);

    std::vector<sweep_t> FindNextSweeps(const Options & options, const std::vector<std::pair<image_t, float>> & next_images);

    // Attempt to register image to the existing model. This requires that
    // a previous call to `RegisterInitialImagePair` was successful.
    bool RegisterInitialImagePair(const Options & options, const image_t image_id1, const image_t image_id2);

    bool EstimateCameraPose(const Options & options, const image_t image_id, 
                            std::vector<std::pair<point2D_t, mappoint_t> >& tri_corrs,
                            std::vector<char>& inlier_mask, size_t* inlier_num = NULL);

    bool EstimateCameraPose(const Options & options,  Camera camera, 
                            const std::vector<Eigen::Vector2d>&  tri_points2D, 
                            const std::vector<Eigen::Vector3d>&  tri_points3D, 
                            Eigen::Vector3d& pose_tvec,
                            Eigen::Vector4d& pose_qvec,
                            std::vector<char>& inlier_mask, size_t* inlier_num = NULL) const;
    
    bool EstimateCameraPoseRig(const Options & options, const image_t image_id, 
                            std::vector<std::pair<point2D_t, mappoint_t> >& tri_corrs,
                            std::vector<char>& inlier_mask,size_t* inlier_num = NULL);

    bool EstimateCameraPoseRig(const Options & options, Camera camera, 
                            const std::vector<Eigen::Vector2d>&  tri_points2D, 
                            const std::vector<int>& tri_camera_indices,
                            const std::vector<Eigen::Vector3d>&  tri_points3D, 
                            Eigen::Vector3d& pose_tvec,
                            Eigen::Vector4d& pose_qvec,
                            std::vector<char>& inlier_mask,size_t* inlier_num = NULL) const;

    bool EstimateCameraPoseWithPrior(const Options & options, const image_t image_id, 
                            std::vector<std::pair<point2D_t, mappoint_t> >& tri_corrs,
                            std::vector<char>& inlier_mask,size_t* inlier_num = NULL);


    bool EstimateCameraPoseWithLocalMap(const Options & options, 
            const image_t image_id, 
            std::vector<std::pair<point2D_t, mappoint_t> >& tri_corrs,
            std::vector<char>& inlier_mask,size_t* inlier_num = NULL);

    bool EstimateCameraPoseRigWithLocalMap(const Options & options, 
            const image_t image_id, 
            std::vector<std::pair<point2D_t, mappoint_t> >& tri_corrs,
            std::vector<char>& inlier_mask,size_t* inlier_num = NULL);

    bool EstimateCameraPoseWithRTK(const Options & options, 
            const image_t image_id, 
            std::vector<std::pair<point2D_t, mappoint_t> >& tri_corrs,
            std::vector<char>& inlier_mask,size_t* inlier_num = NULL);

    bool EstimateSweepPose(const Options & options, 
                           const BundleAdjustmentOptions& ba_options,
                           const sweep_t sweep_id, 
                           const image_t image_id);

    bool EstimateSweepPosesBetweenFramesSequence(const Options & options, 
                                                 const sweep_t sweep_id1, 
                                                 const sweep_t sweep_id2);

    bool TryGetRelativePriorPose(
        const Options & options, image_t image_id, 
        Eigen::Vector4d & qvec, Eigen::Vector3d & tvec,
        bool is_sequential = true);

    int InlierWithPriorPose( const Options & options,
        const std::vector<Eigen::Vector2d>& tri_points2D,
        const std::vector<Eigen::Vector3d>& tri_points3D,
        const Camera& camera,
        const Eigen::Vector4d prior_qvec,
        const Eigen::Vector3d prior_tvec);


    bool CheckLocalPoseConsistency(const Options & options, const image_t image_id);

    // Attempt to register image to the existing model. This requires that
    // a previous call to `RegisterInitialImagePair` was successful.
    bool RegisterNextImage(const Options& options, const image_t image_id,
                           std::vector<std::pair<point2D_t, mappoint_t> >& tri_corrs,
                           std::vector<char>& inlier_mask, 
                           size_t* inlier_num = NULL);

    bool AddKeyFrame(const Options& options, const image_t image_id, 
                     std::vector<std::pair<point2D_t, mappoint_t> >& tri_corrs,
                     std::vector<char>& inlier_mask,bool force = false,
                     bool unordered = false);

    bool AddKeyFrameUpdate(const Options& options, const image_t image_id, 
                     std::vector<std::pair<point2D_t, mappoint_t> >& tri_corrs,
                     std::vector<char>& inlier_mask,bool force = false);

    bool RegisterNonKeyFrame(const Options& options, const image_t image_id);
    bool RegisterNonKeyFrameRig(const Options& options, const image_t image_id);

    bool RegisterNonKeyFrameLidar(const Options& options, const sweep_t sweep_id);

    bool ClosureDetection(const Options& options,
                          const image_t image_id1,
                          const image_t image_id2,
                          double baseline_distance = 1.0);
    
    bool IsolatedImage(const Options& options, const image_t image_id, double baseline_distance);

    bool AdjustCameraByLoopClosure(const Options& options);

    double ComputeDisparityBetweenImages(const image_t image_id1,
                                         const image_t image_id2);

    // Triangulate observations of image.
    size_t TriangulateImage(const IncrementalTriangulator::Options& tri_options,
                            const image_t image_id);

    // Triangulate mappoint for AprilTag detection result
    size_t TriangulateMappoint(const IncrementalTriangulator::Options& tri_options,
                               const mappoint_t image_id);

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

    // Adjust locally connected images and points of a reference image. In
    // addition, refine the provided Map Points. Only images connected to the
    // reference image are optimized. If the provided Map Points are not locally
    // connected to the reference image, their observing images are set as
    // constant in the adjustment.
    LocalBundleAdjustmentReport AdjustLocalBundle(
        const Options& options, const BundleAdjustmentOptions& ba_options,
        const IncrementalTriangulator::Options& tri_options,
        const image_t image_id, const std::unordered_set<mappoint_t>& mappoint_ids,
        const sweep_t next_sweep_id = -1);

    LocalBundleAdjustmentReport AdjustLocalBundle(
        const Options& options, const BundleAdjustmentOptions& ba_options,
        const IncrementalTriangulator::Options& tri_options,
        const image_t image_id, const std::unordered_set<mappoint_t>& mappoint_ids, 
        const std::unordered_set<image_t>& fixed_images);

    // Local BA in batched incremental SfM
    LocalBundleAdjustmentReport AdjustBatchedLocalBundle(
        const Options& options, const BundleAdjustmentOptions& ba_options,
        const IncrementalTriangulator::Options& tri_options,
        const std::vector<image_t>& image_ids, 
        const std::unordered_set<mappoint_t>& mappoint_ids);

    // GBA of updated images in map update.
    void AdjustUpdatedBundle(const Options& options, const BundleAdjustmentOptions& ba_options);

    // Global bundle adjustment using Ceres Solver or PBA.
    bool AdjustGlobalBundle(const Options& options,
                            const BundleAdjustmentOptions& ba_options);
    // bool AdjustParallelGlobalBundle(
    //     const BundleAdjustmentOptions& ba_options,
    //     const ParallelBundleAdjuster::Options& parallel_ba_options);

    bool AdjustGlobalBundleNonKeyFrames(
        const Options& options,
        const BundleAdjustmentOptions& ba_options,
        const std::unordered_set<mappoint_t>& const_mappoint_ids);

    bool AdjustFrame2FrameBundle(
        const Options& options, 
        const BundleAdjustmentOptions& ba_options,
        const std::vector<sweep_t> sweep_ids, 
        const std::unordered_set<sweep_t>& fixed_sweeps,
        double & final_cost);

    bool RefineSceneScale(const Options& options);

    // Filter images and point observations.
    size_t FilterImages(const Options& options);
    size_t FilterImages(const Options& options, const std::unordered_set<image_t>& addressed_images);

    size_t FilterPoints(const Options& options);
    size_t FilterPoints(const Options& options,int min_track_length);
    size_t FilterPoints(const Options& options, const std::unordered_set<mappoint_t>& addressed_points);
    size_t FilterPointsFinal(const Options& options);

    const Reconstruction& GetReconstruction() const;

    // Number of images that are registered in at least on reconstruction.
    size_t NumTotalRegImages() const;

    // Number of shared images between current reconstruction and all other
    // previous reconstructions.
    size_t NumSharedRegImages() const;

    // Get changed Map Points, since the last call to `ClearModifiedMapPoints`.
    const std::unordered_set<mappoint_t>& GetModifiedMapPoints();

    void AddModifiedMapPoint(const mappoint_t mappoint_id);

    // Clear the collection of changed Map Points.
    void ClearModifiedMapPoints();
    
    void DecreaseNumRegTrials(const image_t image_id);

    void ComputeDepthInfo(const Options& options);
    void ComputeDepthInfo(const Options& options, const image_t image_id);

    std::mutex k_mtx_;
    Eigen::Matrix3f small_warped_rgb_K_ = Eigen::Matrix3f::Identity();
    void GetSmallGrayAndDepth(const Options& options, const image_t image_id, cv::Mat &gray, cv::Mat &depth);

    bool ComputeICPLink(const Options& options, const image_t src_id, const image_t dst_id);
    std::pair<bool, ICPLink> ComputeICPLink2(const Options& options, const image_t src_id, const image_t dst_id);
    std::vector<image_t> FindCovisibleImagesForICP(const Options& options,
                                                const image_t reference_image_id);

    bool EstimateRelativePoseRig(const Options& options,
                                  const image_t image_id1,
                                  const image_t image_id2,
                                  Eigen::Vector4d& qvec,
                                  Eigen::Vector3d& tvec);

    // control blockba
    void AccGlobalAdjustmentCount() { ++ global_ba_count_; }
    void SetGlobalAdjustmentCount(int count) { global_ba_count_ = count; }
    int GlobalAdjustmentCount() { return global_ba_count_; }

    void SetWorkspacePath(const std::string &path) { workspace_path_ = path; }
    std::string GetWorkspacePath() { return workspace_path_;}
    
    // Register / De-register image in current reconstruction and update
    // the number of shared images between all reconstructions.
    void RegisterImageEvent(const image_t image_id);
    void DeRegisterImageEvent(const image_t image_id);

    void AppendToVoxelMap(const Options& options, const sweep_t sweep_id);
    void AbstractFeatureVoxels(const Options& options, const std::string plane_path = "", 
                               const std::string line_path = "", const bool force = false);
    void AbstractPlaneFeatureVoxels(const std::vector<lidar::OctoTree::Point> & points,
        const Options& options, const std::string plane_path = "", const std::string line_path = "");

    std::vector<image_t> keyframe_ids_;

private:

    std::vector<image_pair_t> OrderInitialImagePair(const Options& options);

    double ComputeMedianTriangulationAngle(const image_t image_id1,
                                           const image_t image_id2);

    // Find seed images for incremental reconstruction. Suitable seed images have
    // a large number of correspondences and have camera calibration priors. The
    // returned list is ordered such that most suitable images are in the front.
    std::vector<image_t> FindFirstInitialImage(const Options& options) const;

    // For a given first seed image, find other images that are connected to the
    // first image. Suitable second images have a large number of correspondences
    // to the first image and have camera calibration priors. The returned list is
    // ordered such that most suitable images are in the front.
    std::vector<image_t> FindSecondInitialImage(const Options& options,
                                                const image_t image_id1) const;


    int FindConsecutiveCameraPoseIndex(
                    const Options & options, 
                    const image_t image_id,
                    const std::vector<double>& estimated_focal_length_factors,
                    const std::vector<Eigen::Vector4d>& qvecs,
                    const std::vector<Eigen::Vector3d>& tvecs);

    // Find local bundle for given image in the reconstruction. The local bundle
    // is defined as the images that are most connected, i.e. maximum number of
    // shared Map Points, to the given image.
    std::vector<image_t> FindLocalBundle(const Options& options,
                                        const image_t image_id) const;

    bool EstimateInitialTwoViewGeometry(const Options& options,
                                        const image_t image_id1,
                                        const image_t image_id2); 

    bool EstimateRelativePoseBy3D(const Options& options,
                                  const image_t image_id1,
                                  const image_t image_id2,
                                  Eigen::Vector4d& qvec,
                                  Eigen::Vector3d& tvec);
    
    // Find covisible image with the reference image, the mappoints in these 
    // image compose a local map, which can be used to register the next image.
    std::set<image_t> FindCovisibleImages(const Options& options, 
                                             const image_t reference_image_id);

private:
    std::shared_ptr<SceneGraphContainer> scene_graph_container_;

    // Class that holds data of the reconstruction.
    std::shared_ptr<Reconstruction> reconstruction_;

    // Class that is responsible for incremental triangulation.
    std::unique_ptr<IncrementalTriangulator> triangulator_;

    // The orientations that are solved for through a global rotation estimation
	// technique. These values are then used to simplify the pose estimation
	// problem to just estimating the unknown camera positions.
	std::unordered_map<image_t, Eigen::Vector3d> orientations_;

    // Number of images that are registered in at least on reconstruction.
    size_t num_total_reg_images_;

    // Number of shared images between current reconstruction and all other
    // previous reconstructions.
    size_t num_shared_reg_images_;

    // Estimated two-view geometry of last call to `FindFirstInitialImage`,
    // used as a cache for a subsequent call to `RegisterInitialImagePair`.
    image_pair_t prev_init_image_pair_id_;
    TwoViewGeometry prev_init_two_view_geometry_;

    // Images and image pairs that have been used for initialization. Each image
    // and image pair is only tried once for initialization.
    std::unordered_map<image_t, size_t> init_num_reg_trials_;
    std::unordered_set<image_pair_t> init_image_pairs_;

    // Cameras whose parameters have been refined in pose refinement. Used
    // to avoid duplicate refinement of camera parameters or degradation of
    // already refined camera parameters when multiple images share intrinsics.
    std::unordered_set<camera_t> refined_cameras_;

    // The number of reconstructions in which images are registered.
    std::unordered_map<image_t, size_t> num_registrations_;

    // Images that have been filtered in current reconstruction.
    std::unordered_set<image_t> filtered_images_;

    // Images that need to fixed in current reconstruction. Being used in map update.
    std::vector<bool> fixed_images_;

    // Number of trials to register image in current reconstruction. Used to set
    // an upper bound to the number of trials to register an image.
    std::unordered_map<image_t, size_t> num_reg_trials_;

    // Mappoints of previous frames. Being used in map update with sequential mode.
    std::unordered_set<mappoint_t> prev_mappoint_ids_;

    std::unordered_map<image_t, sweep_t> image_to_lidar_map_;
    std::unordered_map<sweep_t, Eigen::Vector3d> lidar_velocities;

    std::vector<std::pair<sweep_t, long long> > sweep_timestamps_;
    std::vector<std::pair<image_t, long long> > image_timestamps_;
    
    float acc_min_dist_kf_;
    int seq_num_kf_;

    int last_keyframe_idx;

    image_t init_image_id1;
    image_t init_image_id2;

    image_t offline_slam_last_frame_id_;
    image_t offline_slam_last_keyframe_id_;
    image_t offline_slam_start_id_;
    bool offline_slam_forward_;

    int global_ba_count_ = 0;
    std::string workspace_path_ = "";
};

}

#endif