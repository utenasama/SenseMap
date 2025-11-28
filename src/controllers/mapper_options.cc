//Copyright (c) 2019, SenseTime Group.
//All rights reserved.

#include "mapper_options.h"

#include "util/misc.h"

namespace sensemap {

bool ClusterMapperOptions::Check() const {
	CHECK_OPTION_GT(init_num_trials, -1);
	CHECK_OPTION_GE(num_workers, -1);
	return true;
}

SceneClustering::Options ClusterMapperOptions::ClusterMapper() const {
	return clustering_options;
}

IndependentMapperOptions ClusterMapperOptions::IndependentMapper() const {
	return mapper_options;
}

GlobalMapper::Options IndependentMapperOptions::GlobalMapperOptions() const{
    GlobalMapper::Options options = global_mapper;

    options.abs_pose_refine_focal_length = ba_refine_focal_length;
    options.abs_pose_refine_extra_params = ba_refine_extra_params;
    options.min_focal_length_ratio = min_focal_length_ratio;
    options.max_focal_length_ratio = max_focal_length_ratio;
    options.max_extra_param = max_extra_param;
    options.num_threads = num_threads;
    options.filter_min_tri_angle = filter_min_tri_angle;
    options.filter_max_reproj_error = filter_max_reproj_error;


    options.filter_max_reproj_error_final = filter_max_reproj_error_final;
    options.filter_min_track_length_final = filter_min_track_length_final;
    options.filter_min_tri_angle_final = filter_min_tri_angle_final;

    options.abs_pose_min_num_inliers = abs_pose_min_num_inliers;
    options.abs_pose_min_inlier_ratio = abs_pose_min_inlier_ratio;
    options.abs_pose_refine_extra_params = ba_refine_extra_params;

    options.single_camera = single_camera;
    options.num_fix_camera_first = num_fix_camera_first;

    options.use_rotation_prior_constrain = use_rotation_prior_constrain;
    options.rotation_prior_constrain_weight = rotation_prior_constrain_weight;
    options.use_translation_prior_constrain = use_translation_prior_constrain;
    options.translation_prior_constrain_weight = translation_prior_constrain_weight;

    options.use_rotation_prior_init =  use_rotation_prior_init;

    options.have_prior_pose = have_prior_pose;
    options.prior_rotations = prior_rotations;
    options.prior_translations = prior_translations;
    options.ba_normalize_reconstruction = ba_normalize_reconstruction;

    options.has_gps_prior = has_gps_prior;
    options.prior_locations_gps = prior_locations_gps;
    options.optimization_use_horizontal_gps_only = optimization_use_horizontal_gps_only;

    options.max_error_gps = max_error_gps;
    options.min_image_num_for_gps_error = min_image_num_for_gps_error;
    return options;

}

IncrementalMapper::Options IndependentMapperOptions::IncrementalMapperOptions() const {
	IncrementalMapper::Options options = incremental_mapper;
	options.abs_pose_refine_focal_length = ba_refine_focal_length;
	options.abs_pose_refine_extra_params = ba_refine_extra_params;
	options.min_focal_length_ratio = min_focal_length_ratio;
	options.max_focal_length_ratio = max_focal_length_ratio;
	options.max_extra_param = max_extra_param;
	options.num_threads = num_threads;
	options.local_ba_num_images = ba_local_num_images;
	options.init_from_global_rotation_estimation = false;
	options.init_from_uncertainty = init_from_uncertainty;
	options.init_min_num_inliers = init_min_num_inliers;
	options.init_min_tri_angle = init_min_tri_angle;
	options.init_min_corrs_intra_view = init_min_corrs_intra_view;
	options.init_max_depth = init_max_depth;
	options.filter_min_tri_angle = filter_min_tri_angle;
	options.filter_max_reproj_error = filter_max_reproj_error;
	options.max_triangulation_angle_degrees = max_triangulation_angle_degrees;
	options.num_fix_camera_first = num_fix_camera_first;
	options.min_visible_map_point_kf = min_visible_map_point_kf;
	options.min_keyframe_step = min_keyframe_step;
	options.min_pose_inlier_kf = min_pose_inlier_kf;
	options.min_distance_kf = 0.8;
	options.min_inlier_ratio_verification_with_prior_pose = 
		min_inlier_ratio_verification_with_prior_pose;

	options.filter_max_reproj_error_final = filter_max_reproj_error_final;
	options.filter_min_track_length_final = filter_min_track_length_final;
	options.filter_min_tri_angle_final = filter_min_tri_angle_final;

	options.abs_pose_min_num_inliers = abs_pose_min_num_inliers;
	options.abs_pose_min_inlier_ratio = abs_pose_min_inlier_ratio;
	options.abs_pose_refine_extra_params = ba_refine_extra_params;
	options.min_inlier_ratio_to_best_model = min_inlier_ratio_to_best_model;	
	options.batched_sfm = batched_sfm;
	options.single_camera = single_camera;
	options.avg_min_dist_kf_factor = avg_min_dist_kf_factor;
	options.mean_max_disparity_kf = mean_max_disparity_kf;
	options.abs_diff_kf = abs_diff_kf;
    options.with_depth = with_depth;
	options.rgbd_camera_params = rgbd_camera_params;
	options.rgbd_delayed_start = rgbd_delayed_start;
	options.rgbd_delayed_start_weights = rgbd_delayed_start_weights;
	options.rgbd_filter_depth_weight = rgbd_filter_depth_weight;
	options.rgbd_max_reproj_depth = rgbd_max_reproj_depth;
	options.rgbd_pose_refine_depth_weight = rgbd_pose_refine_depth_weight;
	
	options.num_fix_camera_first = num_fix_camera_first;
	options.robust_camera_pose_estimate = robust_camera_pose_estimate;
	options.consecutive_camera_pose_top_k = consecutive_camera_pose_top_k;
	options.consecutive_neighbor_ori = consecutive_neighbor_ori;
	options.consecutive_neighbor_t = consecutive_neighbor_t;
	options.consecutive_camera_pose_orientation = consecutive_camera_pose_orientation;
	options.consecutive_camera_pose_t = consecutive_camera_pose_t;
	options.local_region_repetitive = local_region_repetitive;
	options.loop_image_weight = loop_image_weight;
	options.loop_image_min_id_difference = loop_image_min_id_difference;
	options.overlap_image_ids = overlap_image_ids;

	options.max_distance_to_plane = max_distance_to_plane;
	options.max_plane_count = max_plane_count;

	options.ba_normalize_reconstruction = ba_normalize_reconstruction;
	
	if (match_method.compare("sequential") == 0 ||
		match_method.compare("hybrid") == 0) {
		options.image_selection_method = 
			IncrementalMapper::Options::ImageSelectionMethod::WEIGHTED_MIN_UNCERTAINTY;
	}
	options.offline_slam = offline_slam;
	options.min_covisible_mappoint_num = min_covisible_mappoint_num;
	options.min_covisible_mappoint_ratio = min_covisible_mappoint_ratio;
	options.offline_slam_max_next_image_num = offline_slam_max_next_image_num;
	options.initial_two_frame_interval_offline_slam = initial_two_frame_interval_offline_slam;

	options.have_prior_pose = have_prior_pose;
	options.use_prior_aggressively = use_prior_aggressively;
	options.prior_force_keyframe = prior_force_keyframe;
	options.prior_rotations = prior_rotations;
	options.prior_translations = prior_translations;

	options.has_gps_prior = has_gps_prior;
	options.prior_locations_gps = prior_locations_gps;
	options.original_gps_locations = original_gps_locations;
	options.optimization_use_horizontal_gps_only = optimization_use_horizontal_gps_only;

	options.max_error_gps = max_error_gps;
	options.min_image_num_for_gps_error = min_image_num_for_gps_error;
	options.max_error_horizontal_gps = max_error_horizontal_gps;
	options.max_gps_time_offset = max_gps_time_offset;


	options.max_id_difference_for_loop = max_id_difference_for_loop;
	options.min_inconsistent_corr_ratio_for_loop = min_inconsistent_corr_ratio_for_loop;
	options.min_loop_pose_inlier_num = min_loop_pose_inlier_num;
	options.loop_weight = loop_weight;
	options.normal_edge_count_per_image = normal_edge_count_per_image;
	options.max_loop_edge_count = max_loop_edge_count;
	options.optimize_sim3 = optimize_sim3;
	options.loop_closure_max_iter_num = loop_closure_max_iter_num;
	options.normal_edge_min_common_points = normal_edge_min_common_points;
	options.loop_distance_factor_wrt_averge_baseline = loop_distance_factor_wrt_averge_baseline;
	options.neighbor_distance_factor_wrt_averge_baseline = neighbor_distance_factor_wrt_averge_baseline;

	options.debug_info = debug_info;
	options.direct_mapper_type = direct_mapper_type;
    options.min_track_length = min_track_length;

	options.sub_matching = sub_matching;
	options.self_matching = self_matching;

	options.map_update = map_update;
	options.update_old_map = update_old_map;
	options.update_with_sequential_mode = update_with_sequential_mode;

	options.lidar_path = lidar_path;
	options.lidar_to_cam_matrix = lidar_to_cam_matrix;

	return options;
}

IncrementalTriangulator::Options IndependentMapperOptions::Triangulation() const
{
	IncrementalTriangulator::Options options = triangulation;
	options.min_focal_length_ratio = min_focal_length_ratio;
	options.max_focal_length_ratio = max_focal_length_ratio;
	options.max_extra_param = max_extra_param;

	//=============Test params should not be distributed==================
	options.max_transitivity = max_transitivity;
	options.create_max_angle_error = create_max_angle_error;
	options.continue_max_angle_error = continue_max_angle_error;
	options.complete_max_reproj_error = complete_max_reproj_error;
	options.merge_max_reproj_error = merge_max_reproj_error; 
	options.min_angle = min_tri_angle;
	options.re_min_ratio = re_min_ratio;

	return options;
}

BundleAdjustmentOptions IndependentMapperOptions::LocalBundleAdjustment() const
{
	BundleAdjustmentOptions options;
	options.solver_options.function_tolerance = 0.0;
	options.solver_options.gradient_tolerance = 10.0;
	options.solver_options.parameter_tolerance = 0.0;
	options.solver_options.max_num_iterations = ba_local_max_num_iterations;
	options.solver_options.max_linear_solver_iterations = 100;
	options.solver_options.minimizer_progress_to_stdout = false;
	options.solver_options.num_threads = num_threads;
#if CERES_VERSION_MAJOR < 2
	options.solver_options.num_linear_solver_threads = num_threads;
#endif  // CERES_VERSION_MAJOR
	options.print_summary = true;
	options.refine_focal_length = ba_refine_focal_length;
	options.refine_principal_point = ba_refine_principal_point;
	options.refine_extra_params = ba_refine_extra_params;
	options.loss_function_scale = 1.0;
	// options.loss_function_type =
	//  		BundleAdjustmentOptions::LossFunctionType::SOFT_L1;
	options.loss_function_type =
			BundleAdjustmentOptions::LossFunctionType::Huber;

	options.use_prior_relative_pose = have_prior_pose;
	options.use_prior_translation_only = use_prior_translation_only;
	options.use_prior_distance_only = use_prior_distance_only;
	options.use_prior_aggressively = use_prior_aggressively;
	options.prior_pose_weight = prior_pose_weight;

	options.refine_extrinsics = ba_refine_extrinsics;

	options.rgbd_ba_depth_weight = rgbd_ba_depth_weight;
    options.use_icp_relative_pose = use_icp_relative_pose;
    options.icp_base_weight = icp_base_weight;

    options.use_gravity = use_gravity;
    options.gravity_base_weight = gravity_base_weight;

	options.use_time_domain_smoothing = use_time_domain_smoothing;
	options.time_domain_smoothing_weight = time_domain_smoothing_weight;

	return options;
}

BundleAdjustmentOptions IndependentMapperOptions::GlobalBundleAdjustment() const
{
	BundleAdjustmentOptions options;
	options.solver_options.function_tolerance = 0.0;
	options.solver_options.gradient_tolerance = 1.0;
	options.solver_options.parameter_tolerance = 0.0;
	options.solver_options.max_num_iterations = ba_global_max_num_iterations;
	options.solver_options.max_linear_solver_iterations = 100;
	options.solver_options.minimizer_progress_to_stdout = false;
	options.solver_options.num_threads = num_threads;
#if CERES_VERSION_MAJOR < 2
	options.solver_options.num_linear_solver_threads = num_threads;
#endif  // CERES_VERSION_MAJOR
	options.print_summary = true;
	options.refine_focal_length = ba_refine_focal_length;
	options.refine_principal_point = ba_refine_principal_point;
	options.refine_extra_params = ba_refine_extra_params;
	options.refine_local_extrinsics = ba_refine_local_extrinsics;
	options.local_relative_translation_constraint = local_relative_translation_constraint;
	options.plane_constrain = ba_plane_constrain;
	options.plane_weight = ba_plane_weight;
	options.gba_weighted = gba_weighted;

	options.use_prior_relative_pose = have_prior_pose;
	options.use_prior_translation_only = use_prior_translation_only;
	options.use_prior_distance_only = use_prior_distance_only;
	options.use_prior_aggressively = use_prior_aggressively;
	options.prior_pose_weight = prior_pose_weight;
	
	options.use_prior_absolute_location = has_gps_prior;
	options.prior_absolute_location_weight = prior_absolute_location_weight;
	options.optimization_use_horizontal_gps_only = optimization_use_horizontal_gps_only;
	options.prior_absolute_orientation_weight = prior_absolute_orientation_weight;
	
	options.refine_extrinsics = ba_refine_extrinsics;

	options.rgbd_ba_depth_weight = rgbd_ba_depth_weight;
    options.use_icp_relative_pose = use_icp_relative_pose;
    options.icp_base_weight = icp_base_weight;

    options.use_gravity = use_gravity;
    options.gravity_base_weight = gravity_base_weight;

	options.use_time_domain_smoothing = use_time_domain_smoothing;
	options.time_domain_smoothing_weight = time_domain_smoothing_weight;

	options.ba_min_tri_angle = ba_min_tri_angle;

	if(ba_global_loss_function.compare("trival")==0){
		options.loss_function_type =
	 			BundleAdjustmentOptions::LossFunctionType::TRIVIAL;
	}
	else if(ba_global_loss_function.compare("huber")==0){
		options.loss_function_type =
	 			BundleAdjustmentOptions::LossFunctionType::Huber;
	}

	options.debug_info = ba_blockba_debug;
	options.block_ba_frequency = ba_blockba_frequency;
	options.block_size = ba_block_size;	
	options.block_common_image_num = ba_block_common_image_num;
	options.min_connected_points_for_common_images =  ba_block_min_connected_points_for_common_images;

	return options;
}


bool IndependentMapperOptions::IncrementalMapperCheck() const {
	CHECK_OPTION_GT(min_num_matches, 0);
	CHECK_OPTION_GT(max_num_models, 0);
	CHECK_OPTION_GT(max_model_overlap, 0);
	CHECK_OPTION_GE(min_model_size, 0);
	CHECK_OPTION_GT(init_num_trials, 0);
	CHECK_OPTION_GT(min_focal_length_ratio, 0);
	CHECK_OPTION_GT(max_focal_length_ratio, 0);
	CHECK_OPTION_GE(max_extra_param, 0);
	CHECK_OPTION_GE(ba_local_num_images, 2);
	CHECK_OPTION_GE(ba_local_max_num_iterations, 0);
	CHECK_OPTION_GT(ba_global_images_ratio, 1.0);
	CHECK_OPTION_GT(ba_global_points_ratio, 1.0);
	CHECK_OPTION_GT(ba_global_images_freq, 0);
	CHECK_OPTION_GT(ba_global_points_freq, 0);
	CHECK_OPTION_GT(ba_global_max_num_iterations, 0);
	CHECK_OPTION_GT(ba_local_max_refinements, 0);
	CHECK_OPTION_GE(ba_local_max_refinement_change, 0);
	CHECK_OPTION_GT(ba_global_max_refinements, 0);
	CHECK_OPTION_GE(ba_global_max_refinement_change, 0);
	CHECK_OPTION_GE(snapshot_images_freq, 0);
	CHECK_OPTION(IncrementalMapperOptions().Check());
	CHECK_OPTION(Triangulation().Check());
	return true;
}

bool IndependentMapperOptions::GlobalMapperCheck() const {
	return true;
}

bool IndependentMapperOptions::HybridMapperCheck() const {
	return true;
}

} // namespace sensemap
