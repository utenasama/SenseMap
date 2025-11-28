//Copyright (c) 2019, SenseTime Group.
//All rights reserved.

#ifndef SENSEMAP_CONTROLLER_MAPPER_OPTION_H
#define SENSEMAP_CONTROLLER_MAPPER_OPTION_H

#include <set>

#include "util/types.h"
#include "util/threading.h"
#include "sfm/incremental_mapper.h"
#include "sfm/global_mapper.h"
#include "base/reconstruction_manager.h"
#include "graph/scene_clustering.h"
#include "container/scene_graph_container.h"
#include "container/feature_data_container.h"
#include "optim/cluster_merge/cluster_merge_optimizer.h"

namespace sensemap {

enum class MapperType {
	INDEPENDENT = 0,
	CLUSTER = 1
};

enum class IndependentMapperType {
	INCREMENTAL = 0,
	GLOBAL = 1,
	HYBRID = 2,
	DIRECTED = 3
};

struct IndependentMapperOptions {
public:

	// Type of reconstruction estimation to use outside.
	MapperType outside_mapper_type = MapperType::INDEPENDENT;

	// Type of reconstruction estimation to use.
	IndependentMapperType independent_mapper_type =
			IndependentMapperType::INCREMENTAL;

	std::string match_method = "";

	// Write text model result
	bool write_binary_model = true;

	// The minimum number of matches for inlier matches to be considered.
	int min_num_matches = 15;

	// Whether to ignore the inlier matches of watermark image pairs.
	bool ignore_watermarks = false;

	// Whether to reconstruct multiple sub-models.
	bool multiple_models = true;

	// The number of sub-models to reconstruct.
	int max_num_models = 50;

	// The maximum number of overlapping images between sub-models. If the
	// current sub-models shares more than this number of images with another
	// model, then the reconstruction is stopped.
	int max_model_overlap = 20;

	// The minimum number of registered images of a sub-model, otherwise the
	// sub-model is discarded.
	int min_model_size = 5;

	// The image identifiers used to initialize the reconstruction. Note that
	// only one or both image identifiers can be specified. In the former case,
	// the second image is automatically determined.
	int init_image_id1 = -1;
	int init_image_id2 = -1;

	// The number of trials to initialize the reconstruction.
	int init_num_trials = 5;

	// The minimal number of inlier 3D point for initialization
	int init_min_num_inliers = 400;
	
	// The minimal value of median angles for intial triangulation
	double init_min_tri_angle = 12.0;

	int init_min_corrs_intra_view = 30;
	double init_max_depth = 4.0;

	// The minimal triangulation angle for filtering 3D points
	double filter_min_tri_angle = 1.5;

	// The maximal reprojection error for filtering 3D points
	double filter_max_reproj_error = 4.0;

	// The final filtering thresholds
	double filter_min_tri_angle_final = 1.5;
    double filter_max_reproj_error_final = 4.0;
    int filter_min_track_length_final = 2;


	double max_triangulation_angle_degrees = 30.0;

	//================Test params should not be distributed====================

	// Maximum angular error to create new triangulations.
	double create_max_angle_error = 2.0;

	// Maximum angular error to continue existing triangulations.
	double continue_max_angle_error = 2.0;

	// Maximum reprojection error to complete an existing triangulation.
	double complete_max_reproj_error = 4.0;
	
	
	// Minimum pairwise triangulation angle for a stable triangulation.
    // Larger threshold should be in favor of accurate triangluation. 
    // Excessive value could cause under-reconstruction.
    double min_tri_angle = 1.5;

	// Minimum number of inliers in absolute pose estimation.
    int abs_pose_min_num_inliers = 30;
	
	double abs_pose_min_inlier_ratio = 0.25;
	// Maximum transitivity to search for correspondences.
	int max_transitivity = 1;

	// preserve a sample model if the ratio of its inlier number to that of the 
  	// best model is above a threshold, this is used in multiple solution 
	// detection of initialization
  	double min_inlier_ratio_to_best_model = 0.8;

	// whether to use batched manner for reconstruction
	bool batched_sfm = false;
	// whether to perform local BA in batched reconstruction
	bool local_ba_batched = true;

	// accept a pose estimated in batched manner if the ratio of its inlier 
	// number to that of the best pose is above a threshold. 
	double min_inlier_ratio_to_best_pose = 0.7;

	// when there is only a single camera, its intrinsic params will be 
	// optimized (calibrated) in the bundle adjustment. After registering a 
	// certain number of images, we think these params are accurate enough and
	// the optimization is stopped.  
	
	// Maximum reprojection error in pixels to merge triangulations.
    double merge_max_reproj_error = 4.0;

	int num_images_for_self_calibration = 200;

	bool camera_fixed = false;

	std::string ba_global_loss_function = "trival";
	
	int num_fix_camera_first = 5;

	// whether to register the images sequentially, perform like a slam 
	bool offline_slam = false;
	int initial_two_frame_interval_offline_slam = 20;

	// Two image are deemed as covisible if the number of the covisible
	// mappoints is larger than the threshold
	int min_covisible_mappoint_num = 30;

	// Two image are deemed as covisible if the ratio of the covisible
	// mappoints to the correspondences is larger than the threshold
	double min_covisible_mappoint_ratio = 0.2;

	// max number of next candidate images to register for offline slam
    int offline_slam_max_next_image_num = 5;


	// Minimum ratio of common triangulations between an image pair over the
    // number of correspondences between that image pair to be considered
    // as under-reconstructed.
    double re_min_ratio = 0.2;

	// prior reconstructio information
	bool have_prior_pose = false;
	std::unordered_map<image_t, Eigen::Vector4d> prior_rotations;
    std::unordered_map<image_t, Eigen::Vector3d> prior_translations;

	bool use_prior_translation_only = false;
	bool use_prior_distance_only = false;
	bool use_prior_align_only = true;
    bool use_prior_aggressively = false;
	bool prior_force_keyframe = false;
    double prior_pose_weight = 1.0;

	std::unordered_map<image_t, std::pair<Eigen::Vector3d,int>> prior_locations_gps;

	//global sfm prior
    bool use_rotation_prior_constrain = false;
    double rotation_prior_constrain_weight = 1.0;
    bool use_translation_prior_constrain = false;
    double translation_prior_constrain_weight = 0.5;

    bool use_rotation_prior_init = true;

	std::vector<std::pair<unsigned long, std::pair<Eigen::Vector3d,int>>> original_gps_locations;
	bool has_gps_prior = false;
	double prior_absolute_location_weight = 1.0;
	double prior_absolute_orientation_weight = 0.0;

	double max_error_gps = 3.0;
    int min_image_num_for_gps_error = 10;
	double max_error_horizontal_gps = 3.0;
	long max_gps_time_offset = 60000;

	// Only use latitude and longtitude for optimization or not
    bool optimization_use_horizontal_gps_only = false;

	//================Test params should not be distributed====================

	double min_inlier_ratio_verification_with_prior_pose = 0.7;

	int min_visible_map_point_kf = 300;

	int min_keyframe_step = 1;

	int min_pose_inlier_kf = 200;

	double avg_min_dist_kf_factor = 1.0;

	double min_visiblility_score_ratio = 0.5f;

	double mean_max_disparity_kf = 20.0;

	int abs_diff_kf = 10;

	// whether to initialization from uncertainty
	bool init_from_uncertainty = true;

	bool extract_keyframe = false;

	bool image_collection = false;

	bool register_nonkeyframe = false;

    int num_first_force_be_keyframe = 10;

	bool optim_inner_cluster = false;

	bool robust_camera_pose_estimate = false;

	int consecutive_camera_pose_top_k = 2;

	int consecutive_neighbor_ori = 2;
	
	int consecutive_neighbor_t = 1;

	double consecutive_camera_pose_orientation = 5.0;
	
	double consecutive_camera_pose_t = 20.0;

    int local_region_repetitive = 0;

	// Loop image will be down-weighted in registration, the sequential image is
	// selected in priority 
	double loop_image_weight = 1.0;
	// If the min id difference from the registrated image is larger than this 
	// value, the current image will be treated as a loop image.
    int loop_image_min_id_difference = 20;

	bool debug_info = true;

	// Whether to exploit depth map to improve reconstruction.
	bool with_depth = false;

	// Whether to support subimage matching.
	bool sub_matching = false;

	// Whether to support self matching, it is valid only if camera-rig mode.
	bool self_matching = false;

	// Extrinsics of RGBD camera.
	std::string rgbd_camera_params = "";

    // icp
    bool use_icp_relative_pose = false;
    double icp_base_weight = 10.0;

	// gravity
    bool use_gravity = false;
    double gravity_base_weight = 10.0;

    // time domain smoothing
    bool use_time_domain_smoothing = false;
    double time_domain_smoothing_weight = 2.0;

	// Options delayed RGBD constraints.
	// When enabled, RGBD constraints will be more robust if 3D init fails.
	bool rgbd_delayed_start = false;
	double rgbd_delayed_start_weights = 100.0;

	// Depth weight in BA
	double rgbd_ba_depth_weight = 10.0;

	// Depth weight for filtering 3D points,
	// S.A. filter_max_reproj_error and filter_max_reproj_error_final
	// This function is enabled when `with_depth` is true and this value > 0
	double rgbd_filter_depth_weight = 0.0;

    // Maximum allowed reprojection depth for RGBD
	double rgbd_max_reproj_depth = 0.0;

	// Depth weight for pose refinement after estimation
	// This function is enabled when `with_depth` is true and this value > 0
	double rgbd_pose_refine_depth_weight = 0.0;

	// Whether to extract colors for reconstructed points.
	bool extract_colors = true;

	// The number of threads to use during reconstruction.
	int num_threads = -1;

	bool single_camera = false;

	// Thresholds for filtering images with degenerate intrinsics.
	double min_focal_length_ratio = 0.1;
	double max_focal_length_ratio = 10.0;
	double max_extra_param = 10.0;

	// Which intrinsic parameters to optimize during the reconstruction.
	bool ba_refine_focal_length = true;
	bool ba_refine_principal_point = false;
	bool ba_refine_principal_point_final = false;
	bool ba_refine_extra_params = true;
	
	bool ba_refine_extrinsics = true;
	bool ba_refine_local_extrinsics = false;
    bool local_relative_translation_constraint = false;

	// The number of images to optimize in local bundle adjustment.
	int ba_local_num_images = 6;

	// The maximum number of local bundle adjustment iterations.
	int ba_local_max_num_iterations = 25;

	// Whether to use PBA in global bundle adjustment.
	bool ba_global_use_pba = true;

	// The GPU index for PBA bundle adjustment.
	int ba_global_pba_gpu_index = -1;

	// The growth rates after which to perform global bundle adjustment.
	// double ba_global_images_ratio = 1.15;
	// double ba_global_points_ratio = 1.5;
	double ba_global_images_ratio = 1.1;
	double ba_global_points_ratio = 1.1;
	int ba_global_images_freq = 500;
	int ba_global_points_freq = 250000;

	// The maximum number of global bundle adjustment iterations.
	int ba_global_max_num_iterations = 50;

	bool refine_separate_cameras = false;

	// Whether to use plane constrain
	bool ba_plane_constrain = false; 
	int plane_constrain_start_image_num = 50;
	double ba_plane_weight = 0.1;

	// The maximum distance to judge whether a camera point is on the 
    // estimated plane. This threshold is the ralative to the baseline 
    // distance of the initial reconstruction
    double max_distance_to_plane  = 0.05;
	int max_plane_count = 3;

	bool gba_weighted = false;
	
	// whether to normalize the reconstruction after global ba
	bool ba_normalize_reconstruction = false;

	// choose points having large angles in ba
	double ba_min_tri_angle = 1.5;

	// The thresholds for iterative bundle adjustment refinements.
	int ba_local_max_refinements = 2;
	double ba_local_max_refinement_change = 0.001;
	int ba_global_max_refinements = 5;
	double ba_global_max_refinement_change = 0.0005;

	// Path to a folder with reconstruction snapshots during incremental
	// reconstruction. Snapshots will be saved according to the specified
	// frequency of registered images.
	std::string snapshot_path = "";
	int snapshot_images_freq = 0;

	// Which images to reconstruct. If no images are specified, all images will
	// be reconstructed by default.
	// std::set<std::string> image_names;
	std::set<image_t> image_ids;

	// Overlap images with other clusters.
	std::unordered_set<image_t> overlap_image_ids;

	bool use_local_ba_retriangulate_all = true;
	bool use_global_ba_update = false;

	bool delete_redundant_old_images = true;

	// block ba
	bool ba_blockba_debug = false;
	int ba_blockba_frequency = 10;
	int ba_block_size = -1;
	int ba_block_common_image_num = 20;
	int ba_block_min_connected_points_for_common_images = 100;

	// loop closure
	bool explicit_loop_closure = false;
	int max_id_difference_for_loop = 10;
	double min_inconsistent_corr_ratio_for_loop = 0.9;
	int min_loop_pose_inlier_num = 30;
	double loop_weight = 1.0;
	int normal_edge_count_per_image = 5;
	int max_loop_edge_count = 1;
	int loop_check_interval = 500;
	bool optimize_sim3 = true;
	int loop_closure_max_iter_num = 100;
	int normal_edge_min_common_points = 50;
	double loop_distance_factor_wrt_averge_baseline = 6.0;
	double neighbor_distance_factor_wrt_averge_baseline = 1.5;


    //0 : only trianglution  1 : tri seq + gba  2 : tri + lba + gba 3. tri all + g
    int direct_mapper_type = 1;

    int min_track_length = 3;

	//check whether update original map
	bool map_update = false;
	bool update_old_map = false;
    bool update_with_sequential_mode = false;

	// lidar option
	bool lidar_sfm = false;
	std::string lidar_path = "";
	std::string lidar_prior_pose_file = "";
	Eigen::Matrix3x4d lidar_to_cam_matrix = Eigen::Matrix3x4d::Identity();

	IncrementalMapper::Options IncrementalMapperOptions() const;
	IncrementalTriangulator::Options Triangulation() const;
	BundleAdjustmentOptions LocalBundleAdjustment() const;
	BundleAdjustmentOptions GlobalBundleAdjustment() const;

    GlobalMapper::Options GlobalMapperOptions() const;
	/*
	//////////////////////////////////////////////////////////////////////////////
	// -------------------------- Global SfM Options -------------------------- //
	//////////////////////////////////////////////////////////////////////////////

	// After orientations are estimated, view pairs may be filtered/removed if the
	// relative rotation of the view pair differs from the relative rotation
	// formed by the global orientation estimations. Adjust this threshold to
	// control the threshold at which rotations are filtered. See
	// theia/sfm/filter_view_pairs_from_orientation.h
	double rotation_filtering_max_difference_degrees = 5.0;

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
	double translation_filtering_projection_tolerance = 0.1;

	// --------------- Global Rotation Estimation Options --------------- //

	// Robust loss function scales for nonlinear estimation.
	double rotation_estimation_robust_loss_scale = 0.1;

	// --------------- Global Position Estimation Options --------------- //
//	NonlinearPositionEstimator::Options nonlinear_position_estimator_options;
//	LinearPositionEstimator::Options linear_triplet_position_estimator_options;
//	LeastUnsquaredDeviationPositionEstimator::Options
//			least_unsquared_deviation_position_estimator_options;

	// For global SfM it may be advantageous to run a partial bundle adjustment
	// optimizing only the camera positions and 3d points while holding camera
	// orientation and intrinsics constant.
	bool refine_camera_positions_and_points_after_position_estimation = true;

	//////////////////////////////////////////////////////////////////////////////
	// -------------------------- Hybrid SfM Options -------------------------- //
	//////////////////////////////////////////////////////////////////////////////
	// The relative position of the initial pair used for the incremental portion
	// of hybrid sfm is re-estimated using a simplified relative translations
	// solver (assuming known rotation). The relative position is re-estimated
	// using a RANSAC procedure with the inlier threshold defined by this
	// parameter.
	double relative_position_estimation_max_sampson_error_pixels = 4.0;
*/
	bool IncrementalMapperCheck() const;
	bool GlobalMapperCheck() const;
	bool HybridMapperCheck() const;

private:
	IncrementalMapper::Options incremental_mapper;
	IncrementalTriangulator::Options triangulation;
	GlobalMapper::Options global_mapper;
//	hybridMapper::Options hybrid_mapper;
};

struct ClusterMapperOptions{
public:
	// The path to the image folder which are used as input.
	std::string image_path;

	// The path to the database file which is used as input.
	std::string database_path;

	// The maximum number of trials to initialize a cluster.
	int init_num_trials = 10;

	// The number of workers used to reconstruct clusters in parallel.
	int num_workers = -1;

	// Whether to reconstruct multiple sub-models.
	bool multiple_models = true;

	//Enable using image lable to generate cluster
	bool enable_image_label_cluster = false;

	std::string reconstruct_image_name_list = "";

	bool enable_cluster_mapper_with_coarse_label = true;

	bool enable_pose_graph_optimization = true;

	bool enable_motion_averager = true;

	// Only Merge 
	bool only_merge_cluster = false;

	int keyframe_gap = 5;

	SceneClustering::Options clustering_options;
	IndependentMapperOptions mapper_options;
	ClusterMergeOptimizer::ClusterMergeOptions merge_options;

	SceneClustering::Options ClusterMapper() const;
	IndependentMapperOptions IndependentMapper() const;

	bool Check() const;
};

struct MapperOptions {
	// Type of reconstruction estimation to use.
	MapperType mapper_type = MapperType::INDEPENDENT;

	IndependentMapperOptions independent_mapper_options;
	ClusterMapperOptions cluster_mapper_options;
};

} // namespace sensemap

#endif