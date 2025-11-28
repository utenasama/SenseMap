// Copyright (c) 2019, SenseTime Group.
// All rights reserved.

#include <boost/filesystem.hpp>
#include <boost/filesystem/path.hpp>
#include <chrono>
#include <fstream>
#include <iostream>

#include "incremental_mapper_controller.h"
#include "optim/bundle_adjustment.h"
#include "util/misc.h"
#include "base/pose.h"
#include "base/projection.h"

namespace sensemap {

size_t FilterPoints(const IndependentMapperOptions& options, std::shared_ptr<IncrementalMapper> mapper) {
    const size_t num_filtered_observations = mapper->FilterPoints(options.IncrementalMapperOptions());
    std::cout << "  => Filtered observations: " << num_filtered_observations << std::endl;
    return num_filtered_observations;
}

size_t FilterPoints(const IndependentMapperOptions& options, std::shared_ptr<IncrementalMapper> mapper,
                    int min_track_length) {
    const size_t num_filtered_observations = mapper->FilterPoints(options.IncrementalMapperOptions(), min_track_length);
    std::cout << "  => Filtered observations: " << num_filtered_observations << std::endl;
    return num_filtered_observations;
}

size_t FilterPointsFinal(const IndependentMapperOptions& options, std::shared_ptr<IncrementalMapper> mapper) {
    const size_t num_filtered_observations = mapper->FilterPointsFinal(options.IncrementalMapperOptions());
    std::cout << "  => Filtered observations: " << num_filtered_observations << std::endl;
    return num_filtered_observations;
}

size_t FilterImages(const IndependentMapperOptions& options, std::shared_ptr<IncrementalMapper> mapper) {
    const size_t num_filtered_images = mapper->FilterImages(options.IncrementalMapperOptions());
    std::cout << "  => Filtered images: " << num_filtered_images << std::endl;
    return num_filtered_images;
}

size_t CompleteAndMergeTracks(const IndependentMapperOptions& options, std::shared_ptr<IncrementalMapper> mapper) {
    const size_t num_completed_observations = mapper->CompleteTracks(options.Triangulation());
    std::cout << "  => Completed observations: " << num_completed_observations << std::endl;
    const size_t num_merged_observations = mapper->MergeTracks(options.Triangulation());
    std::cout << "  => Merged observations: " << num_merged_observations << std::endl;
    return num_completed_observations + num_merged_observations;
}

size_t TriangulateImage(const IndependentMapperOptions& options, const Image& image,
                        std::shared_ptr<IncrementalMapper> mapper) {
    std::chrono::high_resolution_clock::time_point start_time = std::chrono::high_resolution_clock::now();

    std::cout << "  => Continued observations: " << image.NumMapPoints() << std::endl;
    const size_t num_tris = mapper->TriangulateImage(options.Triangulation(), image.ImageId());
    std::cout << "  => Added observations: " << num_tris << std::endl;

    std::chrono::high_resolution_clock::time_point end_time = std::chrono::high_resolution_clock::now();
    std::cout << StringPrintf(
                     "  => TriangulateImage Elapsed time: %.3f [second]",
                     std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count() / 1e6)
                     .c_str()
              << std::endl;

    return num_tris;
}

void ExtractColors(const std::string& image_path, const image_t image_id,
                   std::shared_ptr<Reconstruction> reconstruction) {
    if (!reconstruction->ExtractColorsForImage(image_id, image_path)) {
        std::cout << StringPrintf("WARNING: Could not read image %s at path %s.",
                                  reconstruction->Image(image_id).Name().c_str(), image_path.c_str())
                  << std::endl;
    }
}

// i = 0 front camera, i = 1 left camera, i = 2 right camera;
Eigen::Matrix3x4d GetLidarTrans(int i){
    std::vector<Eigen::Matrix4d> cam_to_front_vec;
    Eigen::Matrix4d cam0_to_front = Eigen::Matrix4d::Identity();
    Eigen::Matrix4d cam_left_to_front = Eigen::Matrix4d::Identity();
    Eigen::Matrix4d cam_right_to_front = Eigen::Matrix4d::Identity();
    cam_left_to_front<<   -0.003417538338, -0.00089147387 , -0.999993762834 ,-0.042917186469,
                        -0.02046564317,   0.999790219413, -0.000821349858,  0.000342557085,
                        0.999784715767 , 0.020462708528, -0.003435065991 ,-0.049072652799,
                        0.     ,         0.  ,            0.  ,            1.  ;
    std::cout << "cam_front_to_left: " << cam_left_to_front.inverse() << std::endl;
    cam_right_to_front<< 0.001001133243, -0.013031389532,  0.999914586662,  0.051744013617,
                        0.007862121995,  0.999884285959,  0.013023122934 , 0.00033177879,
                        -0.999968591892,  0.007848412584,  0.001103471772, -0.043182895903,
                        0.             , 0.             , 0.             , 1.       ;
    std::cout << "cam_front_to_right: " << cam_right_to_front.inverse() << std::endl;
    cam_to_front_vec.push_back(cam0_to_front);
    cam_to_front_vec.push_back(cam_left_to_front);
    cam_to_front_vec.push_back(cam_right_to_front);  

    Eigen::Matrix4d T_cam0_to_lidar;
    T_cam0_to_lidar << -0.008360209691, -0.016227256072,  0.999833377646,  0.051347537513,
                        -0.999963222325, -0.001777492671, -0.008390144037,  0.004504013824,
                        0.001913345517, -0.999866749462, -0.01621179906 , -0.032669067397,
                        0.            ,  0.            ,  0.            ,  1.           ;

    //lidar to img
    Eigen::Matrix4d T_lidar_to_cam0 = T_cam0_to_lidar.inverse();
        
    Eigen::Matrix4d cam_to_front = cam_to_front_vec[i];
    Eigen::Matrix4d T_lidar_to_cam = cam_to_front.inverse()  * T_lidar_to_cam0;
    return T_lidar_to_cam.topRows(3);
}

namespace {

void InitialAdjustGlobalBundle(const IndependentMapperOptions& options, std::shared_ptr<IncrementalMapper> mapper) {
    BundleAdjustmentOptions custom_options = options.GlobalBundleAdjustment();
    const size_t num_reg_images = mapper->GetReconstruction().NumRegisterImages();

    // Use stricter convergence criteria for first registered images.
    const size_t kMinNumRegImages = 10;
    if (num_reg_images < kMinNumRegImages) {
        custom_options.solver_options.function_tolerance /= 10;
        custom_options.solver_options.gradient_tolerance /= 10;
        custom_options.solver_options.parameter_tolerance /= 10;
        custom_options.solver_options.max_num_iterations *= 2;
        custom_options.solver_options.max_linear_solver_iterations = 200;
    }

    custom_options.refine_focal_length = false;
    custom_options.refine_extra_params = false;
    custom_options.refine_local_extrinsics = false;
    custom_options.refine_extrinsics = true;

    PrintHeading1("Initial Global bundle adjustment");
    for (int iter = 0; iter < 4; ++iter) {
        mapper->AdjustGlobalBundle(options.IncrementalMapperOptions(), custom_options);

        custom_options.refine_focal_length = !custom_options.refine_focal_length;
        custom_options.refine_extrinsics = !custom_options.refine_extrinsics;
    }
}

void AdjustGlobalBundle(const IndependentMapperOptions& options, std::shared_ptr<IncrementalMapper> mapper,
                        bool force_full_ba = true) {
    BundleAdjustmentOptions custom_options = options.GlobalBundleAdjustment();
    // custom_options.refine_extrinsics = false;
    const size_t num_reg_images = mapper->GetReconstruction().NumRegisterImages();

    if (options.single_camera && (num_reg_images > options.num_images_for_self_calibration || options.camera_fixed) ||
        num_reg_images < options.num_fix_camera_first) {
        custom_options.refine_focal_length = false;
        custom_options.refine_extra_params = false;
        custom_options.refine_principal_point = false;
        if(num_reg_images < options.num_fix_camera_first ){
            custom_options.refine_local_extrinsics = false;
        }
    }

    if (num_reg_images < options.plane_constrain_start_image_num) {
        custom_options.plane_constrain = false;
    }

    if(num_reg_images < options.min_image_num_for_gps_error || 
       options.use_prior_align_only){
        custom_options.use_prior_absolute_location = false;
    }

    // Use stricter convergence criteria for first registered images.
    const size_t kMinNumRegImages = 10;
    if (num_reg_images < kMinNumRegImages) {
        custom_options.solver_options.function_tolerance /= 10;
        custom_options.solver_options.gradient_tolerance /= 10;
        custom_options.solver_options.parameter_tolerance /= 10;
        custom_options.solver_options.max_num_iterations *= 3;
        custom_options.solver_options.max_linear_solver_iterations = 200;
    }

    PrintHeading1("Global bundle adjustment");

    custom_options.workspace_path = mapper->GetWorkspacePath();
    if (force_full_ba) {
        custom_options.block_size = 0;
    }

    custom_options.force_full_ba = true;
    mapper->AdjustGlobalBundle(options.IncrementalMapperOptions(), custom_options);
}

void IterativeLocalRefinement(const IndependentMapperOptions& options, const image_t image_id,
                              std::shared_ptr<IncrementalMapper> mapper) {
    auto ba_options = options.LocalBundleAdjustment();
    
    if (options.single_camera) {
        ba_options.refine_extra_params = false;
        ba_options.refine_focal_length = false;
        ba_options.refine_principal_point = false;
        ba_options.refine_local_extrinsics = false;
    }

    for (int i = 0; i < options.ba_local_max_refinements; ++i) {
        const auto report = mapper->AdjustLocalBundle(options.IncrementalMapperOptions(), ba_options, options.Triangulation(), image_id,
                                                      mapper->GetModifiedMapPoints());
        std::cout << "  => Merged observations: " << report.num_merged_observations << std::endl;
        std::cout << "  => Completed observations: " << report.num_completed_observations << std::endl;
        std::cout << "  => Filtered observations: " << report.num_filtered_observations << std::endl;

        const double changed =
            (report.num_merged_observations + report.num_completed_observations + report.num_filtered_observations) /
            static_cast<double>(report.num_adjusted_observations);
        std::cout << StringPrintf("  => Changed observations: %.6f", changed) << std::endl;
        if (changed < options.ba_local_max_refinement_change) {
            break;
        }

    }
    mapper->ClearModifiedMapPoints();
}

void IterativeLocalRefinement(const IndependentMapperOptions& options, const image_t image_id, const sweep_t sweep_id,
                              std::shared_ptr<IncrementalMapper> mapper) {
    auto ba_options = options.LocalBundleAdjustment();
    
    if (options.single_camera) {
        ba_options.refine_extra_params = false;
        ba_options.refine_focal_length = false;
        ba_options.refine_principal_point = false;
        ba_options.refine_local_extrinsics = false;
    }

    for (int i = 0; i < options.ba_local_max_refinements; ++i) {
        const auto report = mapper->AdjustLocalBundle(options.IncrementalMapperOptions(), ba_options, options.Triangulation(), image_id,
                                                      mapper->GetModifiedMapPoints(), sweep_id);
        std::cout << "  => Merged observations: " << report.num_merged_observations << std::endl;
        std::cout << "  => Completed observations: " << report.num_completed_observations << std::endl;
        std::cout << "  => Filtered observations: " << report.num_filtered_observations << std::endl;

        const double changed =
            (report.num_merged_observations + report.num_completed_observations + report.num_filtered_observations) /
            static_cast<double>(report.num_adjusted_observations);
        std::cout << StringPrintf("  => Changed observations: %.6f", changed) << std::endl;
        if (changed < options.ba_local_max_refinement_change) {
            break;
        }

    }
    mapper->ClearModifiedMapPoints();
}

void IterativeGlobalRefinement(const IndependentMapperOptions& options, std::shared_ptr<IncrementalMapper> mapper,
                               std::string workspace_path = "", bool loop_check = false) {

    int ba_count_after_loop_closure = 0;

    if (options.explicit_loop_closure && loop_check) {
        PrintHeading1("Explicit loop closure");
        if (mapper->AdjustCameraByLoopClosure(options.IncrementalMapperOptions())) {
            mapper->RetriangulateAllTracks(options.Triangulation());
            if (options.debug_info) {
                std::string recon_path = StringPrintf("%s/after_loop_%d/", workspace_path.c_str(),
                                                      mapper->GetReconstruction().NumRegisterImages());
                boost::filesystem::create_directories(recon_path);
                mapper->GetReconstruction().WriteReconstruction(recon_path, true);
            }

            PrintHeading1("CompleteAndMergeTracks");
            CompleteAndMergeTracks(options, mapper);
            PrintHeading1("Retriangulation");
            std::cout << "  => Retriangulated observations: " << mapper->Retriangulate(options.Triangulation()) << std::endl;

            AdjustGlobalBundle(options, mapper);
            FilterPoints(options, mapper);
            FilterImages(options, mapper);
            ba_count_after_loop_closure ++;
        }
    }

    PrintHeading1("CompleteAndMergeTracks");
    CompleteAndMergeTracks(options, mapper);
    PrintHeading1("Retriangulation");
    std::cout << "  => Retriangulated observations: " << mapper->Retriangulate(options.Triangulation()) << std::endl;

    for (int i = 0; i < options.ba_global_max_refinements - ba_count_after_loop_closure; ++i) {
        const size_t num_observations = mapper->GetReconstruction().ComputeNumObservations();
        size_t num_changed_observations = 0;
        AdjustGlobalBundle(options, mapper);

        num_changed_observations += CompleteAndMergeTracks(options, mapper);
        num_changed_observations += FilterPoints(options, mapper);
        
        FilterImages(options, mapper);

        const double changed = static_cast<double>(num_changed_observations) / num_observations;
        std::cout << StringPrintf("  => Changed observations: %.6f", changed) << std::endl;
        if (changed < options.ba_global_max_refinement_change) {
            break;
        }
    }
    
}

void BatchedRegisterAndTriangulate(
					const std::shared_ptr<IndependentMapperOptions> options,
                    std::shared_ptr<IncrementalMapper> mapper,
					std::shared_ptr<Reconstruction> reconstruction,
					const std::string& image_path,
					const std::vector<std::pair<image_t, float>>& next_images,
					bool& reg_next_success,
					size_t& ba_prev_num_reg_images,
					size_t& ba_prev_num_points,
					const std::string& workspace_path){

    IncrementalMapper::Options mapper_options = options->IncrementalMapperOptions();
    mapper_options.image_path = image_path;

	std::vector<image_t> calibrated_next_images;
	std::vector<size_t> inlier_nums;
	size_t max_inlier_num = 0;

	std::unordered_map<image_t, std::vector<std::pair<point2D_t, mappoint_t> > >
		tri_corrs_map;
	std::unordered_map<image_t, std::vector<char>> inlier_mask_map;
	
	// estimate poses of all the next images
	bool best_candidate_reg_fail = true;
	for (size_t reg_trial = 0; reg_trial < next_images.size(); ++reg_trial){
		const image_t next_image_id = next_images[reg_trial].first;
		const Image &next_image = reconstruction->Image(next_image_id);
		const Camera& next_camera = reconstruction->Camera(next_image.CameraId());

		PrintHeading1(
			StringPrintf("Calibrating image #%d (%d) name: (%s)",
						 next_image_id,
						 reconstruction->NumRegisterImages() + 1,
						 reconstruction->Image(next_image_id).Name().c_str()));
		std::cout << StringPrintf("  => Image sees %d / %d points",
								  next_image.NumVisibleMapPoints(),
								  next_image.NumObservations())
				  << std::endl;
		
		std::vector<std::pair<point2D_t, mappoint_t> > tri_corrs;
		std::vector<char> inlier_mask;
		size_t inlier_num;
		bool estimate_pose_success;
		
		if(next_camera.NumLocalCameras()>1){
            estimate_pose_success = 
			mapper->EstimateCameraPoseRig(mapper_options,
									   	  next_image_id,
									   	  tri_corrs,
									   	  inlier_mask,  
									   	  &inlier_num);
		}
		else{
            estimate_pose_success = 
			mapper->EstimateCameraPose(mapper_options,
									   next_image_id,
									   tri_corrs,
									   inlier_mask,  
									   &inlier_num);
		}
		
		
		tri_corrs_map.emplace(next_image_id,tri_corrs);
		inlier_mask_map.emplace(next_image_id,inlier_mask);

		if (estimate_pose_success){
			calibrated_next_images.push_back(next_image_id);
			inlier_nums.push_back(inlier_num);

			if (inlier_num > max_inlier_num){
				max_inlier_num = inlier_num;
			}
			best_candidate_reg_fail = false;
		}

		// num_reg_trials for an image will not be recorded if it is not ranked
		// in the front 
		if(!best_candidate_reg_fail){
			mapper->DecreaseNumRegTrials(next_image_id);
		}
	}
	std::cout<<"pose estimated images: "<<calibrated_next_images.size()
			 <<std::endl;
	if (calibrated_next_images.size() == 0){
		reg_next_success = false;
		return;
	}
	else{
		reg_next_success = true;
	}

	// rank the calibrated images according to the uncertainty of the poses, 
	// i.e. the distribution score of the inlier matches

	std::vector<std::pair<float, image_t> >  calibrated_image_ranks;
	std::cout<<"inlier nums: "<<std::endl;
	for (size_t i = 0; i<calibrated_next_images.size(); ++i){
		const Image& calibrated_image = 
							reconstruction->Image(calibrated_next_images[i]);
		const Camera& calibrated_camera = 
							reconstruction->Camera(calibrated_image.CameraId());

		Image score_image;
		score_image.SetPoints2D(calibrated_image.Points2D());
		score_image.SetNumObservations(calibrated_image.NumObservations());
		score_image.SetCameraId(calibrated_image.CameraId());
		score_image.SetUp(calibrated_camera);

		std::vector<std::pair<point2D_t, mappoint_t> > tri_corrs = 
									tri_corrs_map.at(calibrated_next_images[i]);

		std::vector<char> inlier_mask = 
								inlier_mask_map.at(calibrated_next_images[i]);

		int num_inlier=0;
		for(size_t j = 0; j<inlier_mask.size(); ++j){

			if(!inlier_mask[j]){ 
				continue;
			}
			score_image.IncrementCorrespondenceHasMapPoint(tri_corrs[j].first);
			num_inlier ++;
		}
		std::cout<<num_inlier<<" ";
		const float score = score_image.MapPointVisibilityScore();

		calibrated_image_ranks.emplace_back(score,calibrated_next_images[i]);
	}
	std::cout<<std::endl;

	CHECK_EQ(calibrated_image_ranks.size(),calibrated_next_images.size());
	std::sort(calibrated_image_ranks.begin(),calibrated_image_ranks.end(), 
				[](const std::pair<float, image_t> & image1,
                   const std::pair<float, image_t> & image2) {
                    return image1.first > image2.first;
                });

	float max_score = calibrated_image_ranks[0].first;
	std::cout<<"max_score: "<<max_score<<std::endl;

	std::vector<image_t> keyframe_candidates;
	for (size_t i = 0; i < calibrated_image_ranks.size(); ++i){
	
		if (calibrated_image_ranks[i].first > 
			max_score*options->min_inlier_ratio_to_best_pose){

			// images which has sufficient certainty in pose estimation
			// are selected for keyframe candidates.		
			keyframe_candidates.push_back(calibrated_image_ranks[i].second);
		}
		else{
			// otherwise these image will be addressed in the next iteration.
			class Image & image = 
		 				reconstruction->Image(calibrated_image_ranks[i].second);		
			image.SetPoseFlag(false);
		}
	}
	std::cout<<std::endl;
	std::cout<<"keyframe candidates: "<<keyframe_candidates.size()
			 <<std::endl<<std::endl;
	for(const auto candidate_id:keyframe_candidates){
		std::cout<<candidate_id<<" ";
	}
	std::cout<<std::endl;
	

	PrintHeading1("ADD keyframe and batched triangulation");

	std::vector<image_t> registered_keyframes;
	for (const auto &candidate_id : keyframe_candidates){
		
		bool extract_keyframe = options->extract_keyframe;
		
		bool not_extract_keyframe = (!options->extract_keyframe) ||
						(reconstruction->NumRegisterImages() < 
						 options->num_first_force_be_keyframe);

		// Only triangulation on KeyFrame.
		if (!mapper->AddKeyFrame(mapper_options, candidate_id,
								 tri_corrs_map.at(candidate_id),
								 inlier_mask_map.at(candidate_id), 
								 not_extract_keyframe)) {
			continue;
		}

		const Image &registered_keyframe = 
								reconstruction->Image(candidate_id);
		TriangulateImage(*options.get(), registered_keyframe, mapper);
		
		registered_keyframes.push_back(candidate_id);
	}
	std::cout<<"Registered keyframe num: "<<registered_keyframes.size()
			 <<std::endl;
	for(const auto keyframe_id:registered_keyframes){
		std::cout<<keyframe_id<<" ";
	}
	std::cout<<std::endl;

	for (const auto &keyframe_id : registered_keyframes){
		if(options->local_ba_batched){
			IterativeLocalRefinement(*options.get(), keyframe_id, mapper);	
		}
	}

	std::cout << "image_ratio = "
			  << reconstruction->NumRegisterImages() * 1.0f / 
			  	 ba_prev_num_reg_images
			  << std::endl;
	std::cout << "image_freq = "
			  << reconstruction->NumRegisterImages() - ba_prev_num_reg_images
			  << std::endl;
	std::cout << "points_ratio = "
			  << reconstruction->NumMapPoints() * 1.0f / ba_prev_num_points
			  << std::endl;
	std::cout << "points_freq = "
			  << reconstruction->NumMapPoints() - ba_prev_num_points
			  << std::endl;
	std::cout << std::endl;

	if (reconstruction->NumRegisterImages() >=
			options->ba_global_images_ratio * ba_prev_num_reg_images ||
		reconstruction->NumRegisterImages() >=
			options->ba_global_images_freq + ba_prev_num_reg_images ||
		reconstruction->NumMapPoints() >=
			options->ba_global_points_ratio * ba_prev_num_points ||
		reconstruction->NumMapPoints() >=
			options->ba_global_points_freq + ba_prev_num_points){

		for (const auto &keyframe_id : registered_keyframes){
			std::cout << "Image ID#     " << keyframe_id << ", param = "
					  << reconstruction->Camera(
						 reconstruction->Image(keyframe_id).CameraId())
						 .ParamsToString()
					  << std::endl;
		}

		IterativeGlobalRefinement(*options.get(), mapper, workspace_path);

		for (const auto &keyframe_id : registered_keyframes){
			std::cout << "Image ID(GBA)#" << keyframe_id << ", param = "
					  << reconstruction->Camera(
						 reconstruction->Image(keyframe_id).CameraId())
						 .ParamsToString()
					  << std::endl;
		}
		std::cout<<std::endl;

		ba_prev_num_points = reconstruction->NumMapPoints();
		ba_prev_num_reg_images = reconstruction->NumRegisterImages();

		// write the intermediate results

		if (options->debug_info) {

			std::string rec_path = StringPrintf("%s/%d_ba",workspace_path.c_str(),
										reconstruction->NumRegisterImages());
			if (boost::filesystem::exists(rec_path)){
				boost::filesystem::remove_all(rec_path);
			}
			boost::filesystem::create_directories(rec_path);
			reconstruction->WriteBinary(rec_path);
		}
	}
}
}  // namespace

IncrementalMapperController::IncrementalMapperController(
    const std::shared_ptr<IndependentMapperOptions> options, const std::string& image_path,
    const std::string& workspace_path, const std::shared_ptr<SceneGraphContainer> scene_graph_container,
    std::shared_ptr<class ReconstructionManager> reconstruction_manager)
    : IndependentMapperController(options, image_path, workspace_path, scene_graph_container, reconstruction_manager) {
    CHECK(options->IncrementalMapperCheck());
    RegisterCallback(INITIAL_IMAGE_PAIR_REG_CALLBACK);
    RegisterCallback(NEXT_IMAGE_REG_CALLBACK);
    RegisterCallback(LAST_IMAGE_REG_CALLBACK);
    PrintHeading1("Incremental Mapper");
}

void IncrementalMapperController::Run() {
    init_mapper_options_ = options_->IncrementalMapperOptions();
    init_mapper_options_.image_path = image_path_;
    if (options_->lidar_sfm) {
        init_mapper_options_.lidar_sfm = true;
        init_mapper_options_.lidar_path = options_->lidar_path;
        init_mapper_options_.lidar_prior_pose_file = options_->lidar_prior_pose_file;

        std::vector<class Camera> cameras;
        cameras.reserve(scene_graph_container_->Cameras().size());
        for (auto & camera : scene_graph_container_->Cameras()) {
            cameras.push_back(camera.second);
        }
        if (cameras[0].NumLocalCameras() > 1) {
            Eigen::Vector4d local_qvec;
            Eigen::Vector3d local_tvec;
            cameras[0].GetLocalCameraExtrinsic(0, local_qvec, local_tvec);

            Eigen::Matrix4d ref_to_cam0 = Eigen::Matrix4d::Identity();
            ref_to_cam0.block<3, 3>(0, 0) = QuaternionToRotationMatrix(local_qvec);
            ref_to_cam0.block<3, 1>(0, 3) = local_tvec;

            //lidar to img
            Eigen::Matrix4d lidar_to_cam0 = Eigen::Matrix4d::Identity();
            lidar_to_cam0.topRows(3) = options_->lidar_to_cam_matrix;
            init_mapper_options_.lidar_to_cam_matrix = (ref_to_cam0.inverse() * lidar_to_cam0).topRows(3);
            // std::cout << "lidar_to_cam: " << init_mapper_options_.lidar_to_cam_matrix << std::endl;
        } else {
            // options_->lidar_to_cam_matrix = GetLidarTrans(2);
            init_mapper_options_.lidar_to_cam_matrix = options_->lidar_to_cam_matrix;
        }
        std::cout << "lidar_to_cam_matrix: " << init_mapper_options_.lidar_to_cam_matrix << std::endl;
    }

    Reconstruct();

    const size_t kNumInitRelaxations = 2;
    for (size_t i = 0; i < kNumInitRelaxations; ++i) {
        if (reconstruction_manager_->Size() > 0 || IsStopped()) {
            break;
        }
        std::cout << "  => Relaxing the initialization constraints." << std::endl;
        init_mapper_options_.init_min_num_inliers /= 2;
        // init_mapper_options_.init_max_depth *= 2;
        Reconstruct();

        if (reconstruction_manager_->Size() > 0 || IsStopped()) {
            break;
        }
        std::cout << "  => Relaxing the initialization constraints." << std::endl;
        init_mapper_options_.init_min_tri_angle /= 2;
        Reconstruct();
    }

    std::cout << std::endl;
    GetTimer().PrintMinutes();
}

void IncrementalMapperController::Reconstruct() {
    const bool kDiscardReconstruction = true;

    // Main Loop

    std::shared_ptr<IncrementalMapper> mapper = std::make_shared<IncrementalMapper>(scene_graph_container_);

    mapper->SetWorkspacePath(workspace_path_);


    // // not yet support reconstruction from an existing reconstruction
    // CHECK(reconstruction_manager_->Size() == 0);
    bool update_with_optimization = init_mapper_options_.map_update && init_mapper_options_.update_with_sequential_mode;

    for (int num_trials = 0; num_trials < options_->init_num_trials; ++num_trials) {
        BlockIfPaused();
        if (IsStopped()) {
            break;
        }

        size_t reconstruction_idx;
        if (init_mapper_options_.map_update) {
            // only support one original map to update
            reconstruction_idx = 0;
        } else {
            reconstruction_idx = reconstruction_manager_->Add();
        }
        
        std::shared_ptr<Reconstruction> reconstruction = reconstruction_manager_->Get(reconstruction_idx);

        mapper->BeginReconstruction(reconstruction);

        if (options_->lidar_sfm) {
            mapper->LidarSetUp(init_mapper_options_);

            VoxelMap::Option voxel_option;
            std::vector<size_t> layer_point_size{60, 30, 15, 10, 10};
            std::vector<size_t> min_layer_point_size{100, 80, 50, 30, 20};
            std::vector<size_t> max_layer_point_size{2500, 1500, 800, 500, 200};
            voxel_option.layer_point_size = layer_point_size;
            voxel_option.min_layer_point_size = min_layer_point_size;
            voxel_option.max_layer_point_size = max_layer_point_size;
            voxel_option.max_layer = 3;
            voxel_option.voxel_size = 5.0;
            voxel_option.line_min_max_eigen_ratio = 0.03;
            voxel_option.line_min_mid_eigen_ratio = 0.8;
            voxel_option.verbose = false;
            voxel_option.plane_min_max_eigen_ratio = 0.05;
            voxel_option.plane_min_mid_eigen_ratio = 0.05;
            voxel_option.plane_mid_max_eigen_ratio = 0.1;
            // mapper->InitVoxelMap(voxel_option);
            auto & voxel_map = reconstruction->VoxelMap();
            voxel_map = std::make_shared<VoxelMap>(voxel_option, reconstruction->NumRegisterImages());

            reconstruction->lidar_to_cam_matrix = init_mapper_options_.lidar_to_cam_matrix;
        }

        std::cout<<"reconstruction images size: "<<reconstruction->Images().size()<<std::endl;

        if(init_mapper_options_.have_prior_pose){
            reconstruction->have_prior_pose = true;
            reconstruction->prior_force_keyframe = init_mapper_options_.prior_force_keyframe;
            reconstruction->prior_rotations = init_mapper_options_.prior_rotations;
            reconstruction->prior_translations = init_mapper_options_.prior_translations;
        }

        if(init_mapper_options_.has_gps_prior){
            reconstruction->has_gps_prior = true;
            reconstruction->prior_locations_gps = init_mapper_options_.prior_locations_gps;
            reconstruction->original_gps_locations = init_mapper_options_.original_gps_locations;
            reconstruction->optimization_use_horizontal_gps_only = init_mapper_options_.optimization_use_horizontal_gps_only;
        }

        if(init_mapper_options_.with_depth){
            reconstruction->rgbd_filter_depth_weight = init_mapper_options_.rgbd_filter_depth_weight;
            reconstruction->rgbd_max_reproj_depth = init_mapper_options_.rgbd_max_reproj_depth;
            mapper->ComputeDepthInfo(init_mapper_options_);
        }

        if (init_mapper_options_.init_from_global_rotation_estimation) {
            // Estimate camera orientations using a global rotation estimator.
            if (!mapper->EstimateCameraOrientations(init_mapper_options_)) {
                LOG(ERROR) << "Could not estimate camera rotations for Hybrid SfM.";
                break;
            }
        }

        bool next_loop = false;
        // Register init pair
        if (reconstruction->NumRegisterImages() == 0) {
            for (int iter = 0; iter < 100; ++iter) {
                image_t image_id1 = static_cast<image_t>(options_->init_image_id1);
                image_t image_id2 = static_cast<image_t>(options_->init_image_id2);
                std::cout << "initial image id: " << image_id1 << " " << image_id2 << std::endl;
                // Try to find good initial pair.
                if (options_->init_image_id1 == -1 || options_->init_image_id2 == -1) {
                    bool find_init_success;
                    if (init_mapper_options_.offline_slam) {
                        find_init_success =
                            mapper->FindInitialImagePairOfflineSLAM(init_mapper_options_, &image_id1, &image_id2);
                    } else if (init_mapper_options_.init_from_global_rotation_estimation) {
                        find_init_success =
                            mapper->FindInitialImagePairWithKnownOrientation(init_mapper_options_, &image_id1, &image_id2);
                    } else if (init_mapper_options_.init_from_uncertainty) {
                        find_init_success =
                            mapper->FindInitialImagePairUncertainty(init_mapper_options_, &image_id1, &image_id2);
                    } else {
                        find_init_success = mapper->FindInitialImagePair(init_mapper_options_, &image_id1, &image_id2);
                    }
                    if (!find_init_success) {
                        std::cout << "  => No good initial image pair found." << std::endl;
                        mapper->EndReconstruction(kDiscardReconstruction);
                        reconstruction_manager_->Delete(reconstruction_idx);
                        return;
                    }
                } else {
                    if (!reconstruction->ExistsImage(image_id1) || !reconstruction->ExistsImage(image_id2)) {
                        std::cout << StringPrintf(
                                        "  => Initial image pair #%d and #%d do not"
                                        " exist.",
                                        image_id1, image_id2)
                                << std::endl;
                        return;
                    }
                    const auto& camera_1 = scene_graph_container_->Camera(scene_graph_container_->Image(image_id1).CameraId());
                    const auto& camera_2 = scene_graph_container_->Camera(scene_graph_container_->Image(image_id2).CameraId());
                    bool rig_flag = camera_1.NumLocalCameras() > 1 && camera_2.NumLocalCameras() > 1 &&
                                    (camera_1.NumLocalCameras() == camera_2.NumLocalCameras());
                    bool pano_flag = camera_1.ModelName().compare("SPHERICAL") == 0 && camera_2.ModelName().compare("SPHERICAL") == 0;
                    bool perspect_flag = camera_1.NumLocalCameras() == 1 && camera_2.NumLocalCameras() == 1;
                    if (!(rig_flag || pano_flag || perspect_flag)){
                        std::cout << StringPrintf(
                                        "  => Initial image pair #%d(%s-%d) and #%d(%s-%d)"
                                        "  CameraModel different.",
                                        image_id1, camera_1.ModelName().c_str(), camera_1.NumLocalCameras(),
                                        image_id2, camera_2.ModelName().c_str(), camera_2.NumLocalCameras())
                                << std::endl;
                        return;
                    }
                }

                PrintHeading1(StringPrintf("Initializing with image pair #%d [%s] and #%d [%s]", image_id1,
                                        reconstruction->Image(image_id1).Name().c_str(), image_id2,
                                        reconstruction->Image(image_id2).Name().c_str()));
                point2D_t num_match =
                    scene_graph_container_->CorrespondenceGraph()->NumCorrespondencesBetweenImages(image_id1, image_id2);
                std::cout << "Initial matching number: " << num_match << std::endl;

                const bool reg_init_success = mapper->RegisterInitialImagePair(init_mapper_options_, image_id1, image_id2);
                if (!reg_init_success) {
                    std::cout << "  => Initialization failed - possible solutions:" << std::endl
                            << "     - try to relax the initialization constraints" << std::endl
                            << "     - manually select an initial image pair" << std::endl;
                    mapper->EndReconstruction(kDiscardReconstruction);
                    reconstruction_manager_->Delete(reconstruction_idx);
                    return;
                }

                if (options_->debug_info) {
                    std::string rec_path = StringPrintf("%s/init_%s_%s", workspace_path_.c_str(),
                                                        reconstruction->Image(image_id1).Name().c_str(),
                                                        reconstruction->Image(image_id2).Name().c_str());
                    if (boost::filesystem::exists(rec_path)) {
                        boost::filesystem::remove_all(rec_path);
                    }
                    boost::filesystem::create_directories(rec_path);
                    Reconstruction rig_reconstruction;
                    reconstruction->ConvertRigReconstruction(rig_reconstruction);
                    rig_reconstruction.WriteReconstruction(rec_path, options_->write_binary_model);
                }

                std::cout << " before BA mappoints.size() = " << reconstruction->NumMapPoints() << std::endl;
                AdjustGlobalBundle(*options_.get(), mapper);
                std::cout << " After BA mappoints.size() = " << reconstruction->NumMapPoints() << std::endl;


                if (options_->debug_info) {
                    std::string rec_path = StringPrintf("%s/init_gba_%s_%s", workspace_path_.c_str(),
                                                        reconstruction->Image(image_id1).Name().c_str(),
                                                        reconstruction->Image(image_id2).Name().c_str());
                    if (boost::filesystem::exists(rec_path)) {
                        boost::filesystem::remove_all(rec_path);
                    }
                    boost::filesystem::create_directories(rec_path);

                    Reconstruction reconstruction1 = *reconstruction.get();
                    Reconstruction rig_reconstruction1;
                    reconstruction1.ConvertRigReconstruction(rig_reconstruction1);
                    rig_reconstruction1.WriteReconstruction(rec_path, options_->write_binary_model);
                }

                FilterPoints(*options_.get(), mapper);
                FilterImages(*options_.get(), mapper);

                std::cout << " after Filtering mappoints.size() = " << reconstruction->NumMapPoints() << std::endl;
                // Initial image pair failed to register.
                if (reconstruction->NumRegisterImages() == 0 || reconstruction->NumMapPoints() == 0) {
                    std::cout << "Initialization failed, NumRegisterImages = " << reconstruction->NumRegisterImages()
                                << ", "
                                << ", NumMapPoints = " << reconstruction->NumMapPoints() << std::endl;

                    mapper->EndReconstruction(kDiscardReconstruction);
                    reconstruction_manager_->Delete(reconstruction_idx);
                    // If both initial images are manually specified, there is no need for
                    // further initialization trials.
                    if (options_->init_image_id1 != -1 && options_->init_image_id2 != -1) {
                        return;
                    } else {
                        next_loop = true;
                        break;
                    }
                } else {
                    break;
                }
            }
        }
        if (next_loop) continue;

        Callback(INITIAL_IMAGE_PAIR_REG_CALLBACK);

        // Incremental mapping

        size_t ba_prev_num_reg_images = reconstruction->NumRegisterImages();
        size_t ba_prev_num_points = reconstruction->NumMapPoints();
        size_t loop_prev_num_reg_images = reconstruction->NumRegisterImages();

        bool reg_next_success = true;
        bool prev_reg_next_success = true;
        bool offline_slam_have_next_images = true;
        bool prev_normal_mode = true;
        int normal_mode_failure_time = 0;

        size_t total_num_image = reconstruction->NumImages();
        size_t progress = reconstruction->NumRegisterImages();
        bool lidar_init = false;

        while (reg_next_success) {
            BlockIfPaused();
            if (IsStopped()) {
                break;
            }

            if (options_->debug_info) {
                if (reconstruction->NumRegisterImages() % 50 == 0 || (reconstruction->NumRegisterImages() > 0 && reconstruction->NumRegisterImages() <= 40)) {
                    // std::string ori_rec_path =
                    //     StringPrintf("%s/%d-org", workspace_path_.c_str(), reconstruction->NumRegisterImages());
                    // if (boost::filesystem::exists(ori_rec_path)) {
                    //     boost::filesystem::remove_all(ori_rec_path);
                    // }
                    // boost::filesystem::create_directories(ori_rec_path);

                    // reconstruction->WriteReconstruction(ori_rec_path, options_->write_binary_model);
                    // reconstruction->ExportMapPoints(ori_rec_path + "/sparse.ply");

                    std::string rec_path =
                        StringPrintf("%s/%d", workspace_path_.c_str(), reconstruction->NumRegisterImages());
                    if (boost::filesystem::exists(rec_path)) {
                        boost::filesystem::remove_all(rec_path);
                    }
                    boost::filesystem::create_directories(rec_path);

                    Reconstruction rig_rec;
                    reconstruction->ConvertRigReconstruction(rig_rec);

                    if (options_->lidar_sfm) {
                        image_t start_image_id = 1;
                        for(auto image: rig_rec.Images()){
                            if(image.first > start_image_id){
                                start_image_id = image.first;
                            }
                        }
                        start_image_id ++;

                        image_t new_idx = 1;

                        std::vector<sweep_t> registered_sweep_ids = reconstruction->RegisterSweepIds();
                        for (sweep_t sweep_id : registered_sweep_ids) {
                            class LidarSweep lidarsweep = reconstruction->LidarSweep(sweep_id);

                            class Image image_sweep;
                            image_sweep.SetImageId(start_image_id+new_idx); 
                            image_sweep.SetName("prior_" + lidarsweep.Name());
                            image_sweep.SetCameraId(1);

                            Eigen::Matrix4d hworld2lidar = Eigen::Matrix4d::Identity();
                            hworld2lidar.topRows(3) = lidarsweep.ProjectionMatrix();
                            Eigen::Matrix3x4d world2cam = options_->lidar_to_cam_matrix * hworld2lidar;

                            image_sweep.SetTvec(lidarsweep.Tvec());
                            image_sweep.SetQvec(lidarsweep.Qvec());
                            // image_sweep.SetTvec(world2cam.block<3, 1>(0, 3));
                            // image_sweep.SetQvec(RotationMatrixToQuaternion(world2cam.block<3, 3>(0, 0)));

                            rig_rec.AddImage(image_sweep);
                            rig_rec.RegisterImage(image_sweep.ImageId());

                            new_idx++;
                        }

                        if (lidar_init) {
                            std::string cloud_path = StringPrintf("%s/lidar/lidar_cloud_%d.ply", workspace_path_.c_str(), reconstruction->NumRegisterImages());
                            if (!ExistsPath(GetParentDir(cloud_path))) {
                                boost::filesystem::create_directories(GetParentDir(cloud_path));
                            }

                            std::cout << "Write to " << cloud_path << std::endl;
                            
                            LidarPointCloud lidar_sweeps_cloud;

                            std::vector<sweep_t> register_sweep_ids = reconstruction->RegisterSweepIds();
                            std::cout << "register_sweep_ids: " << register_sweep_ids.size() << std::endl;
                            for (sweep_t sweep_id : register_sweep_ids) {
                                class LidarSweep & lidar_sweep = reconstruction->LidarSweep(sweep_id);
                                Eigen::Matrix4d htrans = Eigen::Matrix4d::Identity();
                                htrans.topRows(3) = lidar_sweep.ProjectionMatrix();
                                Eigen::Matrix4d T = htrans.inverse();

                                LidarPointCloud ref_less_features, ref_less_features_t;
                                LidarPointCloud ref_less_surfs = lidar_sweep.GetSurfPointsLessFlat();
                                LidarPointCloud ref_less_corners = lidar_sweep.GetCornerPointsLessSharp();
                                ref_less_features = ref_less_surfs;
                                ref_less_features += ref_less_corners;
                                LidarPointCloud::TransfromPlyPointCloud (ref_less_features, ref_less_features_t, T);

                                lidar_sweeps_cloud += ref_less_features_t;
                            }
                            WriteBinaryPlyPoints(cloud_path.c_str(), lidar_sweeps_cloud.Convert2Ply(), false, true);

                            // {
                            //     std::vector<Eigen::Vector3d> corner_points;
                            //     std::vector<lidar::OctoTree*> octree_list = reconstruction->VoxelMap()->AbstractTreeVoxels();
                            //     for (auto octree : octree_list) {
                            //         std::vector<Eigen::Vector3d> points;
                            //         octree->GetTreePoints(points);
                            //         corner_points.insert(corner_points.end(), points.begin(), points.end());
                            //     }
                                
                            //     std::string corner_path = StringPrintf("%s/planes/corner_%d.obj", workspace_path_.c_str(), reconstruction->NumRegisterImages());
                            //     FILE * fp = fopen(corner_path.c_str(), "w");
                            //     for (auto point : corner_points) {
                            //         fprintf(fp, "v %f %f %f\n", point[0], point[1], point[2]);
                            //     }
                            //     fclose(fp);

                            //     std::string output_path = JoinPaths(workspace_path_, "planes");
                            //     mapper->AbstractFeatureVoxels(init_mapper_options_, 
                            //         JoinPaths(output_path, StringPrintf("%d-plane.obj", reconstruction->NumRegisterImages())));
                            // }
                        }
                    }

                    rig_rec.WriteReconstruction(rec_path, options_->write_binary_model);
                    rig_rec.ExportMapPoints(rec_path + "/sparse.ply");
                }
            }

            reg_next_success = false;

            std::vector<std::pair<image_t, float>> next_images =
                (prev_reg_next_success && init_mapper_options_.offline_slam)
                    ? mapper->FindNextImagesOfflineSLAM(init_mapper_options_, prev_normal_mode)
                    : mapper->FindNextImages(init_mapper_options_);
                    
            std::cout << "next_images.size() = " << next_images.size()
                      << ", mappoints.size() = " << reconstruction->NumMapPoints() << std::endl;


            if(prev_reg_next_success && init_mapper_options_.offline_slam && next_images.empty()) {
                next_images = mapper->FindNextImages(init_mapper_options_);
                offline_slam_have_next_images = false;
            }
            else if(prev_reg_next_success && init_mapper_options_.offline_slam && !next_images.empty()){
                offline_slam_have_next_images = true;
            }

            if (next_images.empty()) {
                std::cout << "Cannot find candidate images to register" << std::endl;
                break;
            }

            std::vector<sweep_t> sweep_ids;
            if (options_->lidar_sfm) {
                sweep_ids = mapper->FindNextSweeps(init_mapper_options_, next_images);
            }

            if (options_->batched_sfm && !init_mapper_options_.offline_slam) {
                BatchedRegisterAndTriangulate(options_, mapper, reconstruction, image_path_, next_images,
                                              reg_next_success, ba_prev_num_reg_images, ba_prev_num_points,
                                              workspace_path_);

                Callback(NEXT_IMAGE_REG_CALLBACK);

                std::cout << "Reg success: " << reg_next_success << std::endl << std::endl;

                const size_t max_model_overlap = static_cast<size_t>(options_->max_model_overlap);
                if (mapper->NumSharedRegImages() >= max_model_overlap) {
                    break;
                }

                // If no image could be registered, try a single final global
                // iterative bundle adjustment and try again to register one
                // image. If this fails once, then exit the incremental mapping.
                if (!reg_next_success && prev_reg_next_success) {
                    reg_next_success = true;
                    prev_reg_next_success = false;
                    IterativeGlobalRefinement(*options_.get(), mapper, workspace_path_);
                } else {
                    prev_reg_next_success = reg_next_success;
                }

                continue;
            }

            for (size_t reg_trial = 0; reg_trial < next_images.size(); ++reg_trial) {
                const image_t next_image_id = next_images[reg_trial].first;
                Image& next_image = reconstruction->Image(next_image_id);
                Camera& next_camera = reconstruction->Camera(next_image.CameraId());

                bool rtk_ready = false && (options_->has_gps_prior && next_image.RtkFlag() == 50);

                PrintHeading1(StringPrintf("Registering image #%d (%d) name: (%s)(progress: %.2f%)", 
                    next_image_id, reconstruction->NumRegisterImages() + 1,
                    reconstruction->Image(next_image_id).Name().c_str(),
                    (progress + 1) * 100.0 / total_num_image));
                std::cout << StringPrintf("  => Image sees %d / %d points", next_image.NumVisibleMapPoints(),
                                          next_image.NumObservations())
                          << std::endl;

                std::vector<std::pair<point2D_t, mappoint_t>> tri_corrs;
                std::vector<char> inlier_mask;

                if(rtk_ready){
                    reg_next_success = mapper->EstimateCameraPoseWithRTK(
                        init_mapper_options_, next_image_id, tri_corrs, inlier_mask);
                } else if (prev_reg_next_success && init_mapper_options_.offline_slam && offline_slam_have_next_images) {
                    if(next_camera.NumLocalCameras()>1){
                        reg_next_success = mapper->EstimateCameraPoseRigWithLocalMap(
                            init_mapper_options_, next_image_id, tri_corrs, inlier_mask);
                    }
                    else{
                        reg_next_success = mapper->EstimateCameraPoseWithLocalMap(init_mapper_options_, next_image_id,
                                                                                  tri_corrs, inlier_mask);
                    }
                    prev_normal_mode = false;
                } else if (init_mapper_options_.have_prior_pose && init_mapper_options_.use_prior_aggressively) {
                    reg_next_success = mapper->EstimateCameraPoseWithPrior(
											init_mapper_options_,
											next_image_id,
											tri_corrs,
											inlier_mask);
                    prev_normal_mode = true; //true or false
                } else {
                    if(next_camera.NumLocalCameras()>1){
					    reg_next_success = mapper->EstimateCameraPoseRig(
												init_mapper_options_,
												next_image_id,
												tri_corrs,
												inlier_mask);
				    }else{
					    reg_next_success = mapper->EstimateCameraPose(
											init_mapper_options_, 
											next_image_id,
											tri_corrs,
											inlier_mask);
				    }
                    prev_normal_mode = true;
                }

                if (reg_next_success) {
                    progress++;

                    int num_first_force_be_keyframe = options_->num_first_force_be_keyframe;
                    bool force_keyframe = (!options_->extract_keyframe) ||
                                          (reconstruction->NumRegisterImages() < num_first_force_be_keyframe) ||
                                          (options_->overlap_image_ids.count(next_image_id) > 0);
                                        
                    if(init_mapper_options_.offline_slam && (!prev_reg_next_success || !offline_slam_have_next_images)){
                        force_keyframe = true;
                    }

                    if(reg_trial + 1 >= next_images.size()){
                        force_keyframe = true;
                    }
                    else{
                        const image_t next_after_next_image_id = next_images[reg_trial + 1].first;
                        const Image& next_after_next_image = reconstruction->Image(next_after_next_image_id);                         
                        if(next_after_next_image.NumVisibleMapPoints() <=init_mapper_options_.min_visible_map_point_kf){
                            force_keyframe = true;
                            std::cout << "  => next after next image visible map points: " 
                                      << next_after_next_image.NumVisibleMapPoints() << std::endl;
                        }
                    }

                    if (next_images[reg_trial].second < next_images[0].second * options_->min_visiblility_score_ratio) {
                        force_keyframe = true;
                        std::cout << "  => min visiblility score ratio is lower than: " 
                                  << options_->min_visiblility_score_ratio << ", force to be keyframe!" << std::endl;
                    }

                    // Only triangulation on KeyFrame.
                    // if (rtk_ready) {
                    //     auto init_mapper_options_tmp = init_mapper_options_;
                    //     init_mapper_options_tmp.min_pose_inlier_kf = 0;
                    //     if (!mapper->AddKeyFrame(init_mapper_options_tmp, 
                    //         next_image_id, tri_corrs, inlier_mask,
                    //         force_keyframe, true)) {
                    //         continue;
                    //     }
                    // } else {
                    if (init_mapper_options_.map_update) {
                        if (!mapper->AddKeyFrameUpdate(init_mapper_options_, next_image_id, tri_corrs, inlier_mask,
                                                       force_keyframe)) {
                            continue;
                        }
                    } else {
                        if (!mapper->AddKeyFrame(init_mapper_options_, next_image_id, tri_corrs, inlier_mask,
                                                 force_keyframe, options_->image_collection)) {
                            continue;
                        }
                    }

                    if (!update_with_optimization) {

                        TriangulateImage(*options_.get(), next_image, mapper);

                        // ICP Constraint
                        if (options_->use_icp_relative_pose) {
                            std::vector<image_t> icp_candi = mapper->FindCovisibleImagesForICP(init_mapper_options_, next_image_id);
                            printf("FindCovisibleImagesForICP for keyframe %u is %lu\n", next_image_id,  icp_candi.size());
                            if (!icp_candi.empty()) {
                                std::vector<std::pair<bool, ICPLink>> icp_results(10, std::make_pair(false, ICPLink()));

                                // auto st = std::chrono::steady_clock::now();

                                {
                                    #pragma omp parallel for schedule(dynamic)
                                    for(int i = 0; i<icp_candi.size(); i++){
                                        auto &img_id = icp_candi[i];
                                        icp_results[i] = mapper->ComputeICPLink2(init_mapper_options_, next_image_id, img_id);
                                        }
                                }

                                // auto ed = std::chrono::steady_clock::now();
                                // printf("ICP time %d  %f\n",icp_candi.size(),
                                //     std::chrono::duration<float, std::milli>(ed - st).count());

                                Image& src_image = reconstruction->Image(next_image_id);
                                for(int i = 0; i<icp_results.size(); i++){
                                    if(icp_results[i].first==false) continue;
                                    src_image.icp_links_.push_back(icp_results[i].second);
                                }
                            }
                        }

                        // if (reconstruction->NumRegisterImages() < 40) {
                        //     std::string rec_path =
                        //         StringPrintf("%s/%d-bf", workspace_path_.c_str(), reconstruction->NumRegisterImages());
                        //     if (boost::filesystem::exists(rec_path)) {
                        //         boost::filesystem::remove_all(rec_path);
                        //     }
                        //     boost::filesystem::create_directories(rec_path);

                        //     Reconstruction rig_rec;
                        //     reconstruction->ConvertRigReconstruction(rig_rec);

                        //     rig_rec.WriteReconstruction(rec_path, options_->write_binary_model);
                        //     rig_rec.ExportMapPoints(rec_path + "/sparse.ply");
                        // }

                        // if (options_->lidar_sfm && sweep_reg_next_success) {
                        //     IterativeLocalRefinement(*options_.get(), next_image_id, next_sweep_id, mapper);
                        // } else {
                        IterativeLocalRefinement(*options_.get(), next_image_id, mapper);
                        // }

                        std::cout << "image_ratio = " << reconstruction->NumRegisterImages() * 1.0f / ba_prev_num_reg_images
                                << std::endl;
                        std::cout << "image_freq = " << reconstruction->NumRegisterImages() - ba_prev_num_reg_images
                                << std::endl;
                        std::cout << "points_ratio = " << reconstruction->NumMapPoints() * 1.0f / ba_prev_num_points
                                << std::endl;
                        std::cout << "points_freq = " << reconstruction->NumMapPoints() - ba_prev_num_points << std::endl;

                        if (reconstruction->NumRegisterImages() >=
                                options_->ba_global_images_ratio * ba_prev_num_reg_images ||
                            reconstruction->NumRegisterImages() >=
                                options_->ba_global_images_freq + ba_prev_num_reg_images ||
                            reconstruction->NumMapPoints() >= options_->ba_global_points_ratio * ba_prev_num_points ||
                            reconstruction->NumMapPoints() >= options_->ba_global_points_freq + ba_prev_num_points) {
                            for (auto next_camera : scene_graph_container_->Cameras()) {
                                if(next_camera.second.NumLocalCameras()>1){
                                    std::cout << "Camera ID# " << next_camera.first
                                        << ", local param = " <<VectorToCSV(next_camera.second.LocalParams())
                                        << std::endl;
                                    std::cout << "local qvecs = "<<VectorToCSV(next_camera.second.LocalQvecs())<<std::endl
                                        << "local tvecs = " << VectorToCSV(next_camera.second.LocalTvecs())<<std::endl;     
                                }
                                else{
                                    std::cout << "Camera ID# " << next_camera.first << ", param = " << next_camera.second.ParamsToString()
                                              << std::endl;
                                }
                            }
                            bool loop_check = reconstruction->NumRegisterImages() > loop_prev_num_reg_images + options_->loop_check_interval;
                            if(loop_check){
                                loop_prev_num_reg_images = reconstruction->NumRegisterImages();
                            }

                            if (options_->lidar_sfm && lidar_init && reconstruction->NumRegisterLidarSweep() <= 100) {
                            // if (options_->lidar_sfm) {
                                mapper->RefineImageLidarAlignment(init_mapper_options_);
                            }

                            IterativeGlobalRefinement(*options_.get(), mapper, workspace_path_,loop_check);
                            for (auto next_camera : scene_graph_container_->Cameras()) {
                                if(next_camera.second.NumLocalCameras()>1){
                                    std::cout << "Camera ID(GBA)# " << next_camera.first
                                        << ", local param = " <<VectorToCSV(next_camera.second.LocalParams())
                                        << std::endl;
                                    std::cout << "local qvecs = "<<VectorToCSV(next_camera.second.LocalQvecs())<<std::endl
                                        << "local tvecs = " << VectorToCSV(next_camera.second.LocalTvecs())<<std::endl; 
                                            
                                }
                                else{
                                    std::cout << "Camera ID(GBA)# " << next_camera.first << ", param = " << next_camera.second.ParamsToString()
                                              << std::endl;
                                }
                            }
                            class Camera& next_camera = reconstruction->Camera(next_image.CameraId());
                            if (next_camera.ModelName().compare("SPHERICAL") != 0 &&
                                next_camera.ModelName().compare("UNIFIED") != 0 &&
                                next_camera.ModelName().compare("OPENCV_FISHEYE") != 0 &&
                                next_camera.HasBogusParams(init_mapper_options_.min_focal_length_ratio,
                                                        init_mapper_options_.max_focal_length_ratio,
                                                        init_mapper_options_.max_extra_param)) {
                                std::cout << "Camera has bogus params, reset intrinsic parmeters" << std::endl;

                                next_camera.SetParams(scene_graph_container_->Camera(next_image.CameraId()).Params());

                                BundleAdjustmentOptions ba_options = options_->GlobalBundleAdjustment();
                                ba_options.refine_focal_length = true;
                                ba_options.refine_extra_params = true;
                                ba_options.refine_principal_point = false;
                                mapper->AdjustGlobalBundle(init_mapper_options_, ba_options);
                                if (reconstruction->ExistsCamera(next_image.CameraId())) {
                                    std::cout << "Image ID(Reset)#" << next_image_id
                                        << ", param = " << reconstruction->Camera(next_image.CameraId()).ParamsToString()
                                        << std::endl;
                                }
                            }

                            ba_prev_num_points = reconstruction->NumMapPoints();
                            ba_prev_num_reg_images = reconstruction->NumRegisterImages();
                        }

                        std::cout << "keyframe ids: " << mapper->keyframe_ids_.size() << std::endl;

                        bool sweep_reg_next_success = false;
                        sweep_t next_sweep_id = -1;
                        // Initializing Lidar Sweep.
                        if (options_->lidar_sfm && !lidar_init && mapper->keyframe_ids_.size() >= 20) {
                            image_t image_id1 = mapper->keyframe_ids_[0];
                            image_t image_id2 = mapper->keyframe_ids_[1];
                            sweep_t sweep_id1 = mapper->FindNextSweep(image_id1);
                            sweep_t sweep_id2 = mapper->FindNextSweep(image_id2);
                            mapper->keyframe_ids_.erase(mapper->keyframe_ids_.begin(), mapper->keyframe_ids_.begin() + 2);

                            std::cout << "LiDAR&image pair1: " << image_id1 << " " << sweep_id1 << std::endl;
                            std::cout << "LiDAR&image pair2: " << image_id2 << " " << sweep_id2 << std::endl;

                            // sweep_t sweep_id1, sweep_id2;
                            // bool find_init_sweep_success =
                            // mapper->FindInitialLidarPair(image_id1, image_id2, &sweep_id1, &sweep_id2);
                            // std::cout << "find_init_sweep_success: " << find_init_sweep_success << std::endl;
                            // if (!find_init_sweep_success) {
                            //     std::cout << "  => No good initial lidar pair found. DeRegisterImage " << image_id1 << "&" << image_id2 << std::endl;
                            //     reconstruction->DeRegisterImage(image_id1);
                            //     reconstruction->DeRegisterImage(image_id2);
                            //     continue;
                            // }

                            auto & lidar_sweep1 = reconstruction->LidarSweep(sweep_id1);
                            auto & lidar_sweep2 = reconstruction->LidarSweep(sweep_id2);

                            std::cout << StringPrintf("Initializing with lidar pair #%d [%s] and #%d [%s]\n", 
                                        sweep_id1, lidar_sweep1.Name().c_str(),
                                        sweep_id2, lidar_sweep2.Name().c_str()) << std::endl;
                        
                            auto ba_option = options_->LocalBundleAdjustment();
                            ba_option.max_num_iteration_frame2frame = 10;
                            {
                                if (!(lidar_sweep1.HasQvecPrior() && lidar_sweep1.HasTvecPrior() && 
                                    lidar_sweep2.HasQvecPrior() && lidar_sweep2.HasTvecPrior())) {
                                    Eigen::Matrix4d lidar2cam = Eigen::Matrix4d::Identity();
                                    lidar2cam.topRows(3) = init_mapper_options_.lidar_to_cam_matrix;

                                    class Image & image1 = reconstruction->Image(image_id1);
                                    Eigen::Matrix4d world2cam1 = Eigen::Matrix4d::Identity();
                                    world2cam1.topRows(3) = image1.ProjectionMatrix();
                                    Eigen::Matrix3x4d world2lidar1 = (lidar2cam.inverse() * world2cam1).topRows(3);
                                    Eigen::Vector4d qvec1 = RotationMatrixToQuaternion(world2lidar1.block<3, 3>(0, 0));
                                    lidar_sweep1.SetQvecPrior(qvec1);
                                    lidar_sweep1.SetTvecPrior(Eigen::Vector3d(0, 0, 0));


                                    class Image & image2 = reconstruction->Image(image_id2);
                                    Eigen::Matrix4d world2cam2 = Eigen::Matrix4d::Identity();
                                    world2cam2.topRows(3) = image2.ProjectionMatrix();
                                    Eigen::Matrix3x4d world2lidar2 = (lidar2cam.inverse() * world2cam2).topRows(3);
                                    Eigen::Vector4d qvec2 = RotationMatrixToQuaternion(world2lidar2.block<3, 3>(0, 0));
                                    lidar_sweep2.SetQvecPrior(qvec2);
                                    lidar_sweep2.SetTvecPrior(Eigen::Vector3d(0, 0, 0));
                                }
                            }
                            bool lidar_init_success = mapper->RegisterInitialLidarPair(init_mapper_options_, ba_option, 
                                                                                    image_id1, image_id2, sweep_id1, sweep_id2);
                            // if (!lidar_init_success) {
                            //     std::cout << "  => Lidar initialization failed - possible solutions:" << std::endl
                            //         << "     - try to relax the initialization constraints" << std::endl;
                            //     mapper->EndReconstruction(kDiscardReconstruction);
                            //     reconstruction_manager_->Delete(reconstruction_idx);
                            //     return;
                            // }

                            lidar_init = true;

                            mapper->ImageLidarAlignment(init_mapper_options_, image_id1, image_id2, sweep_id1, sweep_id2);

                            // mapper->RefineImageLidarAlignment(init_mapper_options_);

                            mapper->AppendToVoxelMap(init_mapper_options_, sweep_id1);
                            mapper->AppendToVoxelMap(init_mapper_options_, sweep_id2);

                            if (options_->debug_info) {
                                std::string output_path = JoinPaths(workspace_path_, "planes");
                                if (!ExistsPath(output_path)) {
                                    boost::filesystem::create_directories(output_path);
                                }
                                mapper->AbstractFeatureVoxels(init_mapper_options_, JoinPaths(output_path, "plane-init.obj"));

                                Eigen::Matrix4d htrans1 = Eigen::Matrix4d::Identity();
                                htrans1.topRows(3) = lidar_sweep1.ProjectionMatrix();
                                Eigen::Matrix4d T1 = htrans1.inverse();

                                Eigen::Matrix4d htrans2 = Eigen::Matrix4d::Identity();
                                htrans2.topRows(3) = lidar_sweep2.ProjectionMatrix();
                                Eigen::Matrix4d T2 = htrans2.inverse();

                                LidarPointCloud ref_less_features1, ref_less_features_t1;
                                LidarPointCloud ref_less_features2, ref_less_features_t2;

                                LidarPointCloud ref_less_surfs1 = lidar_sweep1.GetSurfPointsLessFlat();
                                LidarPointCloud ref_less_corners1 = lidar_sweep1.GetCornerPointsLessSharp();
                                ref_less_features1 = ref_less_surfs1;
                                ref_less_features1 += ref_less_corners1;
                                LidarPointCloud::TransfromPlyPointCloud (ref_less_features1, ref_less_features_t1, T1);

                                LidarPointCloud ref_less_surfs2 = lidar_sweep2.GetSurfPointsLessFlat();
                                LidarPointCloud ref_less_corners2 = lidar_sweep2.GetCornerPointsLessSharp();
                                ref_less_features2 = ref_less_surfs2;
                                ref_less_features2 += ref_less_corners2;
                                LidarPointCloud::TransfromPlyPointCloud (ref_less_features2, ref_less_features_t2, T2);

                                std::string ref_les_name = workspace_path_ + "/frame" + lidar_sweep1.Name() + ".ply";
                                std::cout << ref_les_name << std::endl;
                                std::string parent_path = GetParentDir(ref_les_name);
                                if (!ExistsPath(parent_path)) {
                                    boost::filesystem::create_directories(parent_path);
                                }
                                WriteBinaryPlyPoints(ref_les_name.c_str(), ref_less_features_t1.Convert2Ply(), false, false);
                            
                                ref_les_name = workspace_path_ + "/frame" + lidar_sweep2.Name() + ".ply";
                                std::cout << ref_les_name << std::endl;
                                WriteBinaryPlyPoints(ref_les_name.c_str(), ref_less_features_t2.Convert2Ply(), false, false);

                                Eigen::Matrix3x4d world2lidar1 = lidar_sweep1.ProjectionMatrix();
                                Eigen::Matrix4d h_world2lidar1 = Eigen::Matrix4d::Identity();
                                h_world2lidar1.topRows(3) = world2lidar1;
                                Eigen::Matrix3x4d world2cam1 = reconstruction->Image(image_id1).ProjectionMatrix();
                                Eigen::Matrix4d h_world2cam1 = Eigen::Matrix4d::Identity();
                                h_world2cam1.topRows(3) = world2cam1;

                                Eigen::Matrix4d est_lidar2cam = h_world2cam1 * h_world2lidar1.inverse();
                                Eigen::Matrix4d calib_lidar2cam = Eigen::Matrix4d::Identity();
                                calib_lidar2cam.topRows(3) = init_mapper_options_.lidar_to_cam_matrix;
                                std::cout << "estimate lidar2cam: " << std::endl;
                                std::cout << est_lidar2cam << std::endl;
                                std::cout << "calib lidar2cam: " << std::endl;
                                std::cout << calib_lidar2cam << std::endl;
                                Eigen::Matrix3d dR = est_lidar2cam.block<3, 3>(0, 0).transpose() * calib_lidar2cam.block<3, 3>(0, 0);
                                Eigen::AngleAxisd angle_axis(dR);
                                double R_angle = angle_axis.angle();
                                std::cout << "R_diff: " << RAD2DEG(R_angle) << std::endl;
                                std::cout << "t_diff: " << (est_lidar2cam.block<3, 1>(0, 3) - calib_lidar2cam.block<3, 1>(0, 3)).transpose() << std::endl;

                                std::string rec_path = StringPrintf("%s/init_gba_%s_%s_af", workspace_path_.c_str(),
                                                                    reconstruction->Image(image_id1).Name().c_str(),
                                                                    reconstruction->Image(image_id2).Name().c_str());
                                if (boost::filesystem::exists(rec_path)) {
                                    boost::filesystem::remove_all(rec_path);
                                }
                                boost::filesystem::create_directories(rec_path);

                                Reconstruction rig_reconstruction;
                                reconstruction->ConvertRigReconstruction(rig_reconstruction);

                                rig_reconstruction.WriteReconstruction(rec_path, options_->write_binary_model);
                                rig_reconstruction.ExportMapPoints(rec_path + "/sparse.ply");
                            }
                        } else if (options_->lidar_sfm && reg_next_success && lidar_init) {
                            std::cout << "keyframes queue: ";
                            for (size_t i = 0; i < mapper->keyframe_ids_.size(); ++i) {
                                std::cout << mapper->keyframe_ids_[i] << " ";
                            }
                            std::cout << std::endl;
                            image_t pool_image_id = mapper->keyframe_ids_[0];
                            sweep_t pool_sweep_id = mapper->FindNextSweep(pool_image_id);
                            mapper->keyframe_ids_.erase(mapper->keyframe_ids_.begin(), mapper->keyframe_ids_.begin() + 1);

                            std::cout << "LiDAR&image pair: " << pool_image_id << " " << pool_sweep_id << std::endl;

                            auto & pool_image = reconstruction->Image(pool_image_id);
                            std::cout << StringPrintf("image#%d: %s", pool_image_id, pool_image.Name().c_str()) << std::endl;

                            if (pool_sweep_id != -1) {
                                auto & lidar_sweep = reconstruction->LidarSweep(pool_sweep_id);
                                if (!lidar_sweep.IsRegistered()) {
                                    sweep_reg_next_success = 
                                    mapper->EstimateSweepPose(init_mapper_options_, options_->LocalBundleAdjustment(), pool_sweep_id, pool_image_id);
                                }
                                if (sweep_reg_next_success) {
                                    next_sweep_id = pool_sweep_id;
                                    mapper->AppendToVoxelMap(init_mapper_options_, pool_sweep_id);
                                }
                            } else {
                                std::cout << "Warnning! Can not find synchronized lidar scan." << std::endl;
                            }
                        }

                        if (sweep_reg_next_success && next_sweep_id != -1 && options_->debug_info) {
                            class LidarSweep & lidar_sweep = reconstruction->LidarSweep(next_sweep_id);

                            Eigen::Matrix4d htrans = Eigen::Matrix4d::Identity();
                            // htrans.topRows(3) = lidar_sweep.ProjectionMatrix();
                            Eigen::Matrix4d T = htrans.inverse();

                            LidarPointCloud ref_less_features, ref_less_features_t;
                            LidarPointCloud ref_less_surfs = lidar_sweep.GetSurfPointsLessFlat();
                            LidarPointCloud ref_less_corners = lidar_sweep.GetCornerPointsLessSharp();
                            // ref_less_features = ref_less_surfs;
                            ref_less_features = ref_less_corners;
                            ref_less_features += ref_less_surfs;
                            LidarPointCloud::TransfromPlyPointCloud (ref_less_features, ref_less_features_t, T);

                            std::string ref_les_name = workspace_path_ + "/frame/" + lidar_sweep.Name() + "-lba.ply";
                            std::cout << ref_les_name << std::endl;
                            std::cout << next_image.Name() << std::endl;
                            std::string parent_path = GetParentDir(ref_les_name);
                            if (!ExistsPath(parent_path)) {
                                boost::filesystem::create_directories(parent_path);
                            }
                            WriteBinaryPlyPoints(ref_les_name.c_str(), ref_less_features_t.Convert2Ply(), false, false);
                            // lidar_sweep.OutputCornerPly(workspace_path_ + "/frame/" + lidar_sweep.Name());

                            // if (lidar_sweep.Name().compare("1722482987.500376.pcd") == 0) {
                            //     std::string output_path = JoinPaths(workspace_path_, "planes");
                            //     mapper->AbstractFeatureVoxels(init_mapper_options_, 
                            //         JoinPaths(output_path, StringPrintf("%s-plane.obj", lidar_sweep.Name().c_str())));
                            // }
                        }
                    }

                    Callback(NEXT_IMAGE_REG_CALLBACK);

                    break;
                } else {
                    std::cout << "  => Could not register, trying another image." << std::endl;
     
                    const size_t kMinNumInitialRegTrials = 30;
                    if (reg_trial >= kMinNumInitialRegTrials &&
                        reconstruction->NumRegisterImages() < static_cast<size_t>(options_->min_model_size)) {
                        break;
                    }
                }
            }

            const size_t max_model_overlap = static_cast<size_t>(options_->max_model_overlap);
            if (mapper->NumSharedRegImages() >= max_model_overlap) {
                std::cout << StringPrintf("shared registered images %d exceed the max model overlap %d\n", mapper->NumSharedRegImages(), max_model_overlap);
                break;
            }

            // If no image could be registered, try a single final global iterative
            // bundle adjustment and try again to register one image. If this fails
            // once, then exit the incremental mapping.

            if(!reg_next_success && prev_normal_mode){
                normal_mode_failure_time ++;
            }
            else if(reg_next_success && prev_normal_mode){
                normal_mode_failure_time = 0;
            }

            if(!reg_next_success && normal_mode_failure_time <= 1){
                reg_next_success = true;
                prev_reg_next_success = false;
                if(normal_mode_failure_time == 1){
                    IterativeGlobalRefinement(*options_.get(), mapper, workspace_path_);
                }
            }
            else{
                prev_reg_next_success = reg_next_success;
            }

        }

        if (IsStopped()) {
            const bool kDiscardReconstruction = false;
            mapper->EndReconstruction(kDiscardReconstruction);
            break;
        }

        if (options_->have_prior_pose && options_->prior_force_keyframe && reconstruction->NumRegisterImages() < options_->prior_rotations.size()) {
            mapper->EndReconstruction(false);
            reconstruction_manager_->Delete(reconstruction_idx);
            std::cout<< "The current reconstruction is too small compared with prior, try another one" << std::endl;
            continue;
        }

        // Only run final global BA, if last incremental BA was not global.
        if ((reconstruction->NumRegisterImages() >= 2 && reconstruction->NumRegisterImages() != ba_prev_num_reg_images &&
            reconstruction->NumMapPoints() != ba_prev_num_points)) {
            mapper->SetGlobalAdjustmentCount(1000 * options_->ba_blockba_frequency);
            IterativeGlobalRefinement(*options_.get(), mapper, workspace_path_);
            for (const auto &camera : reconstruction->Cameras()){
                if(camera.second.NumLocalCameras()>1){
                    std::cout << "Camera ID# " << camera.first
                        << ", local param = " <<VectorToCSV(camera.second.LocalParams())
                        << std::endl;     
                }
                else{
                    std::cout << "Camera ID# " << camera.first
                        << ", param = " << reconstruction->Camera(camera.first).ParamsToString()
                        << std::endl;
                }
            }
        }

        //process sweep id left
        int left_sweeps = 0;
        if (options_->lidar_sfm && lidar_init) {
            std::cout << "left keyframes: " << mapper->keyframe_ids_.size() << std::endl;
            bool sweep_reg_next_success = false;
            for (int i = 0; i < mapper->keyframe_ids_.size(); ++i) {
                image_t pool_image_id = mapper->keyframe_ids_[i];
                sweep_t pool_sweep_id = mapper->FindNextSweep(pool_image_id);
                // mapper->keyframe_ids_.erase(mapper->keyframe_ids_.begin(), mapper->keyframe_ids_.begin() + 1);
                
                std::cout << "LiDAR&image pair: " << pool_image_id << " " << pool_sweep_id << std::endl;

                if (pool_sweep_id != -1) {
                    if (!reconstruction->LidarSweep(pool_sweep_id).IsRegistered()) {
                        sweep_reg_next_success = 
                        mapper->EstimateSweepPose(init_mapper_options_, options_->LocalBundleAdjustment(), pool_sweep_id, pool_image_id);
                    }
                    if (sweep_reg_next_success) {
                        mapper->AppendToVoxelMap(init_mapper_options_, pool_sweep_id);
                        left_sweeps++;
                    }
                } else {
                    std::cout << "Warnning! Can not find synchronized lidar scan." << std::endl;
                }
            }
        }

        // Only run final global BA, if last incremental BA was not global.
        if (reconstruction->NumRegisterImages() >= 2 && left_sweeps > 0) {
            mapper->SetGlobalAdjustmentCount(1000 * options_->ba_blockba_frequency);
            IterativeGlobalRefinement(*options_.get(), mapper, workspace_path_);
            for (const auto &camera : reconstruction->Cameras()){
                if(camera.second.NumLocalCameras()>1){
                    std::cout << "Camera ID# " << camera.first
                        << ", local param = " <<VectorToCSV(camera.second.LocalParams())
                        << std::endl;     
                }
                else{
                    std::cout << "Camera ID# " << camera.first
                        << ", param = " << reconstruction->Camera(camera.first).ParamsToString()
                        << std::endl;
                }
            }
        }

        if (!update_with_optimization) {
            // reconstruction->ResetPointStatus();
            FilterPointsFinal(*options_.get(), mapper);
            FilterImages(*options_.get(), mapper);
        }

        if (options_->register_nonkeyframe) {
            progress = 0;
            total_num_image -= reconstruction->NumRegisterImages();

            const std::unordered_set<mappoint_t>& const_mappoint_ids = reconstruction->MapPointIds();

            for (const auto& image : reconstruction->Images()) {
                if (image.second.IsRegistered()) {
                    continue;
                }
                PrintHeading1(StringPrintf("Registering NonKeyFrame #%d (%d) name: (%s)(progress: %.2f%)", 
                    image.first, reconstruction->NumRegisterImages() + 1,
                    reconstruction->Image(image.first).Name().c_str(),
                    (++progress) * 100.0 / total_num_image));
                
                const Camera& camera = reconstruction->Camera(image.second.CameraId());
                if(camera.NumLocalCameras() == 1){
                    mapper->RegisterNonKeyFrame(init_mapper_options_, image.first);
                }
                else{
                    mapper->RegisterNonKeyFrameRig(init_mapper_options_,image.first);
                }

                if (!update_with_optimization) {
                    TriangulateImage(*options_.get(), image.second, mapper);

                    // ICP Constraint
                    if (options_->use_icp_relative_pose) {
                        std::vector<image_t> icp_candi = mapper->FindCovisibleImagesForICP(init_mapper_options_, image.first);
                        printf("FindCovisibleImagesForICP for nonkeyframe %u is %lu\n",image.first,  icp_candi.size());
                        if(!icp_candi.empty()){
                            std::vector<std::pair<bool, ICPLink>> icp_results(10, std::make_pair(false, ICPLink()));

                            auto st = std::chrono::steady_clock::now();

                            {
                                #pragma omp parallel for schedule(dynamic)
                                for(int i = 0; i<icp_candi.size(); i++){
                                    auto &img_id = icp_candi[i];
                                    icp_results[i] = mapper->ComputeICPLink2(init_mapper_options_, image.first, img_id);
                                }
                            }

                            auto ed = std::chrono::steady_clock::now();
                            printf("ICP time %lu  %f\n",icp_candi.size(),
                                std::chrono::duration<float, std::milli>(ed - st).count());

                            Image& src_image = reconstruction->Image(image.first);
                            for(int i = 0; i<icp_results.size(); i++){
                                if(icp_results[i].first==false) continue;
                                src_image.icp_links_.push_back(icp_results[i].second);
                            }
                        }
                    }
                }
            }

            size_t total_num_lidar = reconstruction->NumLidarSweep();
            total_num_lidar -= reconstruction->NumRegisterLidarSweep();
            progress = 0;
            for (const auto& lidar_sweep : reconstruction->LidarSweeps()) {
                if (lidar_sweep.second.IsRegistered()) {
                    continue;
                }
                PrintHeading1(StringPrintf("Registering NonKeyFrameLidar #%d (%d) name: (%s)(progress: %.2f%)", 
                    lidar_sweep.first, reconstruction->NumRegisterLidarSweep() + 1,
                    reconstruction->LidarSweep(lidar_sweep.first).Name().c_str(),
                    (++progress) * 100.0 / total_num_lidar));
                
                bool reg_success =
                mapper->RegisterNonKeyFrameLidar(init_mapper_options_, lidar_sweep.first);
                if (reg_success) {
                    mapper->AppendToVoxelMap(init_mapper_options_, lidar_sweep.first);
                    // if (options_->debug_info) {
                    
                    //     const auto & reg_lidar_sweep = reconstruction->LidarSweep(lidar_sweep.first);

                    //     Eigen::Matrix4d htrans = Eigen::Matrix4d::Identity();
                    //     htrans.topRows(3) = reg_lidar_sweep.ProjectionMatrix();
                    //     Eigen::Matrix4d T = htrans.inverse();

                    //     LidarPointCloud ref_less_features, ref_less_features_t;
                    //     // LidarPointCloud ref_less_surfs = reg_lidar_sweep.GetSurfPointsLessFlat();
                    //     LidarPointCloud ref_less_corners = reg_lidar_sweep.GetCornerPointsLessSharp();
                    //     // ref_less_features = ref_less_surfs;
                    //     ref_less_features = ref_less_corners;
                    //     LidarPointCloud::TransfromPlyPointCloud (ref_less_features, ref_less_features_t, T);

                    //     std::string ref_les_name = workspace_path_ + "/frame/" + reg_lidar_sweep.Name() + "-nonkf.ply";
                    //     std::cout << ref_les_name << std::endl;
                    //     std::string parent_path = GetParentDir(ref_les_name);
                    //     if (!ExistsPath(parent_path)) {
                    //         boost::filesystem::create_directories(parent_path);
                    //     }
                    //     WriteBinaryPlyPoints(ref_les_name.c_str(), ref_less_features_t.Convert2Ply(), false, false);
                    // }
                }
            }

            if (reconstruction->NumRegisterImages() >= 2) {
                if (update_with_optimization && reconstruction->GetNewImageIds().size() > 0) {
                    BundleAdjustmentOptions ba_options = options_->GlobalBundleAdjustment();
                    mapper->AdjustUpdatedBundle(init_mapper_options_, ba_options);
                } else {
                    FilterPoints(*options_.get(), mapper);
                    FilterImages(*options_.get(), mapper);

                    if (options_->explicit_loop_closure) {
                        PrintHeading1("Explicit loop closure");
                        if (mapper->AdjustCameraByLoopClosure(options_->IncrementalMapperOptions())) {
                            mapper->RetriangulateAllTracks(options_->Triangulation());
                            if (options_->debug_info) {
                                std::string recon_path = StringPrintf("%s/after_loop_%d/", workspace_path_.c_str(),
                                                                    reconstruction->NumRegisterImages());
                                boost::filesystem::create_directories(recon_path);
                                mapper->GetReconstruction().WriteReconstruction(recon_path, true);
                            }
                        }
                    }

                    PrintHeading1("Retriangulation");
                    CompleteAndMergeTracks(*options_.get(), mapper);
                    std::cout << "  => Retriangulated observations: " << mapper->Retriangulate(options_->Triangulation())
                            << std::endl;

                    PrintHeading1("Final Global BA");
                    BundleAdjustmentOptions ba_options = options_->GlobalBundleAdjustment();
                    if(options_->use_prior_align_only){
                        ba_options.use_prior_absolute_location = false;
                    }
                    ba_options.solver_options.minimizer_progress_to_stdout = true;
                    ba_options.solver_options.max_num_iterations = 20;
                    ba_options.solver_options.max_linear_solver_iterations = 100;

                    if (reconstruction->NumRegisterImages() < init_mapper_options_.num_fix_camera_first) {
                        ba_options.refine_focal_length = false;
                        ba_options.refine_principal_point = false;
                        ba_options.refine_extra_params = false;
                        ba_options.refine_local_extrinsics = false;
                    }

                    // ba_options.refine_focal_length = false;
                    // ba_options.refine_extra_params = false;
                    // ba_options.refine_principal_point = false;
                    // ba_options.refine_local_extrinsics = false;

                    mapper->AdjustGlobalBundle(init_mapper_options_, ba_options);

                    if (init_mapper_options_.has_gps_prior && !init_mapper_options_.map_update) {
                        CHECK(reconstruction->has_gps_prior);
                        reconstruction->AlignWithPriorLocations(init_mapper_options_.max_error_gps,
                                                                init_mapper_options_.max_error_horizontal_gps,
                                                                init_mapper_options_.max_gps_time_offset);
                    }
                }
                for (const auto &camera : reconstruction->Cameras()){
                    if(camera.second.NumLocalCameras()>1){
                        std::cout << "Camera ID# " << camera.first
                            << ", local param = " <<VectorToCSV(camera.second.LocalParams())
                            << std::endl;     
                    }
                    else{
                        std::cout << "Camera ID# " << camera.first
                            << ", param = " << reconstruction->Camera(camera.first).ParamsToString()
                            << std::endl;
                    }
                }
            }
        }

        std::unordered_set<image_t> loop_image_set;
        std::unordered_set<sweep_t> loop_lidar_set;
        if (options_->lidar_sfm) {

            reconstruction->ComputeBaselineDistance();
            double baseline_distance = reconstruction->baseline_distance;
            std::cout << "baseline_distance: " << baseline_distance << std::endl;

            // detect image loop.
            std::vector<std::pair<image_t, image_t> > loop_images;
            auto register_image_ids = reconstruction->RegisterImageIds();
            std::vector<std::pair<image_t, uint64_t> > image_timestamps;
            image_timestamps.reserve(register_image_ids.size());
            for (auto image_id : register_image_ids) {
                class Image & image = reconstruction->Image(image_id);
                image_timestamps.emplace_back(image_id, image.timestamp_);
            }
            std::cout << "register images: " << register_image_ids.size() << std::endl;
            std::cout << "image_timestamps: " << image_timestamps.size() << std::endl;
            std::sort(image_timestamps.begin(), image_timestamps.end(), 
                [&](const std::pair<image_t, uint64_t> & image1, const std::pair<image_t, uint64_t> & image2) {
                return image1.second < image2.second;
            });
            std::unordered_map<image_t, Eigen::Vector3d> velocities;
            velocities.reserve(image_timestamps.size());
            for (int i = 0; i < image_timestamps.size() - 1; ++i) {
                auto image_timestamp1 = image_timestamps[i];
                auto image_timestamp2 = image_timestamps[i + 1];

                image_t image_id1 = image_timestamp1.first;
                image_t image_id2 = image_timestamp2.first;
                class Image & image1 = reconstruction->Image(image_id1);
                class Image & image2 = reconstruction->Image(image_id2);
                Eigen::Vector3d C1 = image1.ProjectionCenter();
                Eigen::Vector3d C2 = image2.ProjectionCenter();

                double time_diff = ((double)image_timestamp2.second - (double)image_timestamp1.second) / 1e9;
                if (time_diff < std::numeric_limits<double>::epsilon()) {
                    continue;
                }
                Eigen::Vector3d velocity;
                velocity = (C2 - C1) / time_diff;
                velocities[image_id1] = velocity;
            }
            for (int i = 0; i < image_timestamps.size() - 1; ++i) {
                auto image_timestamp1 = image_timestamps[i];
                auto image_timestamp2 = image_timestamps[i + 1];
                image_t image_id1 = image_timestamp1.first;
                image_t image_id2 = image_timestamp2.first;
                if (velocities.find(image_id1) != velocities.end() && velocities.find(image_id2) != velocities.end()) {
                    class Image & image1 = reconstruction->Image(image_id1);
                    class Image & image2 = reconstruction->Image(image_id2);

                    uint64_t time_diff = std::abs((long long)image1.create_time_ - (long long)image2.create_time_);
                    std::cout << "image name: " << image1.Name() << ", " << image2.Name() << std::endl;
                    std::cout << "create time: " << image1.create_time_ << ", " << image2.create_time_ << std::endl;
                    double dist = (image1.ProjectionCenter() - image2.ProjectionCenter()).norm();
                    std::cout << "dist: " << dist << "/" << baseline_distance << std::endl;
                    
                    Eigen::Vector3d velocity1 = velocities[image_id1];
                    Eigen::Vector3d velocity2 = velocities[image_id2];
                    double s1 = velocity1.norm();
                    double s2 = velocity2.norm();
                    double norm_ratio = (s1 > s2) ? s1 / s2 : s2 / s1;
                    double cos_diff = velocity1.dot(velocity2) / (s1 * s2);
                    double angle_diff = RAD2DEG(std::acos(cos_diff));
                    std::cout << "norm_ratio: " << norm_ratio << std::endl;
                    std::cout << "angle_diff: " << angle_diff << std::endl;

                    int consistent_corr_count = 0;
                    FeatureMatches corrs = scene_graph_container_->CorrespondenceGraph()->FindCorrespondencesBetweenImages(image_id1, image_id2);
                    for (auto corr : corrs) {
                        point2D_t point2D_idx1 = corr.point2D_idx1;
                        point2D_t point2D_idx2 = corr.point2D_idx2;
                        class Point2D & point2D1 = image1.Point2D(point2D_idx1);
                        class Point2D & point2D2 = image2.Point2D(point2D_idx2);

                        if (point2D1.HasMapPoint() && point2D2.HasMapPoint() &&
                            point2D1.MapPointId() == point2D2.MapPointId()) {
                            consistent_corr_count++;
                        }
                    }
                    double inconsistent_corr_ratio = 1.0 - static_cast<double>(consistent_corr_count)/ static_cast<double>(corrs.size());
                    std::cout << "corrs: " << consistent_corr_count << "/" << corrs.size() << std::endl;
                    
                    if(inconsistent_corr_ratio >= 0.7 && consistent_corr_count < 100 && 
                        (norm_ratio > 2.0 || angle_diff > 20.0)) {
                        loop_image_set.insert(image_id1);
                        loop_image_set.insert(image_id2);
                        loop_images.emplace_back(image_id1, image_id2);
                        std::cout << "loop image: " << image_id1 << ", " << image_id2 << std::endl;
                    }
                }

            }
            if (loop_images.size() > 0) {
                std::string loop_image_str = JoinPaths(workspace_path_, "loop-images.txt");
                std::ofstream file(loop_image_str, std::ofstream::out);
                std::cout << "loop image path: " << loop_image_str << std::endl;
                file << loop_images.size() << std::endl;
                for (auto loop_image : loop_images) {
                    file << loop_image.first << " " << loop_image.second << std::endl;
                }
                file.close();
            } else {
                std::cout << "No visual loop being detected!" << std::endl;
            }

            // detect lidar loop.
            std::vector<std::pair<sweep_t, sweep_t> > loop_lidars;
            auto register_sweep_ids = reconstruction->RegisterSweepIds();
            std::vector<std::pair<sweep_t, uint64_t> > sweep_timestamps;
            sweep_timestamps.reserve(register_sweep_ids.size());
            for (auto sweep_id : register_sweep_ids) {
                class LidarSweep & lidar_sweep = reconstruction->LidarSweep(sweep_id);
                sweep_timestamps.emplace_back(sweep_id, lidar_sweep.timestamp_);
            }
            std::sort(sweep_timestamps.begin(), sweep_timestamps.end(), 
                [&](const std::pair<sweep_t, uint64_t> & lidar_sweep1, const std::pair<sweep_t, uint64_t> & lidar_sweep2) {
                return lidar_sweep1.second < lidar_sweep2.second;
            });

            std::unordered_map<sweep_t, Eigen::Vector3d> lidar_velocities;
            lidar_velocities.reserve(sweep_timestamps.size());
            for (int i = 0; i < sweep_timestamps.size() - 1; ++i) {
                auto sweep_timestamp1 = sweep_timestamps[i];
                auto sweep_timestamp2 = sweep_timestamps[i + 1];
                sweep_t sweep_id1 = sweep_timestamp1.first;
                sweep_t sweep_id2 = sweep_timestamp2.first;
                class LidarSweep & lidar_sweep1 = reconstruction->LidarSweep(sweep_id1);
                class LidarSweep & lidar_sweep2 = reconstruction->LidarSweep(sweep_id2);
                Eigen::Vector3d C1 = lidar_sweep1.ProjectionCenter();
                Eigen::Vector3d C2 = lidar_sweep2.ProjectionCenter();

                double time_diff = ((double)sweep_timestamp2.second - (double)sweep_timestamp1.second) / 1e9;
                if (time_diff < std::numeric_limits<double>::epsilon()) {
                    continue;
                }
                Eigen::Vector3d velocity;
                velocity = (C2 - C1) / time_diff;
                lidar_velocities[sweep_id1] = velocity;
            }
            for (int i = 0; i < sweep_timestamps.size() - 1; ++i) {
                auto sweep_timestamp1 = sweep_timestamps[i];
                auto sweep_timestamp2 = sweep_timestamps[i + 1];
                sweep_t sweep_id1 = sweep_timestamp1.first;
                sweep_t sweep_id2 = sweep_timestamp2.first;
                if (lidar_velocities.find(sweep_id1) != lidar_velocities.end() && 
                    lidar_velocities.find(sweep_id2) != lidar_velocities.end()) {
                    class LidarSweep & lidar_sweep1 = reconstruction->LidarSweep(sweep_id1);
                    class LidarSweep & lidar_sweep2 = reconstruction->LidarSweep(sweep_id2);
                    
                    double dist = (lidar_sweep1.ProjectionCenter() - lidar_sweep2.ProjectionCenter()).norm();
                    
                    std::cout << "lidar name: " << lidar_sweep1.Name() << ", " << lidar_sweep2.Name() << std::endl;
                    std::cout << "create time: " << lidar_sweep1.create_time_ << ", " << lidar_sweep2.create_time_ << std::endl;
                    std::cout << "dist: " << dist << "/" << baseline_distance << std::endl;
                    
                    Eigen::Vector3d velocity1 = lidar_velocities[sweep_id1];
                    Eigen::Vector3d velocity2 = lidar_velocities[sweep_id2];
                    double s1 = velocity1.norm();
                    double s2 = velocity2.norm();
                    std::cout << "velocity1: " << velocity1.transpose() << ", " << s1 << std::endl;
                    std::cout << "velocity2: " << velocity2.transpose() << ", " << s2 << std::endl;
                    double norm_ratio = (s1 > s2) ? s1 / s2 : s2 / s1;
                    double cos_diff = velocity1.dot(velocity2) / (s1 * s2);
                    double angle_diff = RAD2DEG(std::acos(cos_diff));
                    std::cout << "norm_ratio: " << norm_ratio << std::endl;
                    std::cout << "angle_diff: " << angle_diff << std::endl;

                    if (dist > 2.0 * baseline_distance && (norm_ratio > 2.0 || angle_diff > 30.0)) {
                        loop_lidar_set.insert(sweep_id1);
                        loop_lidar_set.insert(sweep_id2);
                        loop_lidars.emplace_back(sweep_id1, sweep_id2);
                        std::cout << "loop lidar: " << sweep_id1 << ", " << sweep_id2 << std::endl;
                    }
                }
            }
            if (loop_lidars.size() > 0) {
                std::string loop_lidar_str = JoinPaths(workspace_path_, "loop-lidars.txt");
                std::ofstream file(loop_lidar_str, std::ofstream::out);
                std::cout << "loop lidar path: " << loop_lidar_str << std::endl;
                file << loop_lidars.size() << std::endl;
                for (auto loop_lidar : loop_lidars) {
                    file << loop_lidar.first << " " << loop_lidar.second << std::endl;
                }
                file.close();
            } else {
                std::cout << "No lidar loop being detected!" << std::endl;
            }
        }

        // if(!init_mapper_options_.has_gps_prior){
        //     mapper->RefineSceneScale(init_mapper_options_);
        //     reconstruction->ResetPointStatus();
        // }
        FilterPointsFinal(*options_.get(), mapper);
        FilterImages(*options_.get(), mapper);
        reconstruction->FilterAllFarawayImages();

        Callback(LAST_IMAGE_REG_CALLBACK);

        if (init_mapper_options_.map_update) {
            std::cout << StringPrintf("New Register images: %d/%d\n", reconstruction->GetNewImageIds().size(), 
                                                                      scene_graph_container_->GetNewImageIds().size());
        }

        std::cout << StringPrintf("Total Register images: %d/%d\n", reconstruction->NumRegisterImages(), scene_graph_container_->NumImages());

        if (options_->lidar_sfm && options_->debug_info) {
            std::string cloud_path = JoinPaths(workspace_path_, "lidar/lidar_cloud.ply");
            if (!ExistsPath(GetParentDir(cloud_path))) {
                boost::filesystem::create_directories(GetParentDir(cloud_path));
            }

            std::cout << "Write to " << cloud_path << std::endl;
            
            LidarPointCloud lidar_sweeps_cloud;

            std::vector<sweep_t> register_sweep_ids = reconstruction->RegisterSweepIds();
            std::cout << "register_sweep_ids: " << register_sweep_ids.size() << std::endl;
            for (sweep_t sweep_id : register_sweep_ids) {
                class LidarSweep & lidar_sweep = reconstruction->LidarSweep(sweep_id);
                Eigen::Matrix4d htrans = Eigen::Matrix4d::Identity();
                htrans.topRows(3) = lidar_sweep.ProjectionMatrix();
                Eigen::Matrix4d T = htrans.inverse();

                LidarPointCloud ref_less_features, ref_less_features_t;
                LidarPointCloud ref_less_surfs = lidar_sweep.GetSurfPointsLessFlat();
                LidarPointCloud ref_less_corners = lidar_sweep.GetCornerPointsLessSharp();
                ref_less_features = ref_less_surfs;
                ref_less_features += ref_less_corners;
                LidarPointCloud::TransfromPlyPointCloud (ref_less_features, ref_less_features_t, T);
                lidar_sweeps_cloud += ref_less_features_t;
            }
            WriteBinaryPlyPoints(cloud_path.c_str(), lidar_sweeps_cloud.Convert2Ply(), false, true);

            std::string output_path = JoinPaths(workspace_path_, "planes");
            if (!ExistsPath(output_path)) {
                boost::filesystem::create_directories(output_path);
            }
            mapper->AbstractFeatureVoxels(init_mapper_options_, JoinPaths(output_path, "plane.obj"));

            // {
            //     std::vector<Eigen::Vector3d> corner_points;
            //     std::vector<lidar::OctoTree*> octree_list = reconstruction->VoxelMap()->AbstractTreeVoxels();
            //     for (auto octree : octree_list) {
            //         std::vector<Eigen::Vector3d> points;
            //         octree->GetTreePoints(points);
            //         corner_points.insert(corner_points.end(), points.begin(), points.end());
            //     }
                
            //     FILE * fp = fopen(JoinPaths(output_path, "corner.obj").c_str(), "w");
            //     for (auto point : corner_points) {
            //         fprintf(fp, "v %f %f %f\n", point[0], point[1], point[2]);
            //     }
            //     fclose(fp);
            // }

            if (true || loop_image_set.size() > 0 || loop_lidar_set.size() > 0) {
                Reconstruction reconstruction_tmp = *reconstruction.get();
                image_t start_image_id = 1;
                camera_t start_camera_id = 1;
                for(auto image: reconstruction_tmp.Images()){
                    if(image.first > start_image_id){
                        start_image_id = image.first;
                    }
                    if (start_camera_id < image.second.CameraId()){
                        start_camera_id = image.second.CameraId();
                    }
                }

                for (auto cam : reconstruction_tmp.Cameras()){
                    if (start_camera_id < cam.first){
                        start_camera_id = cam.first;
                    }
                }
                start_image_id ++;
                start_camera_id ++;

                image_t new_idx = 1;
                std::cout << "Start: " << start_image_id << ", " << start_camera_id << std::endl;

                class Camera loop_image_camera = reconstruction_tmp.Camera(1);
                loop_image_camera.Rescale(2);
                loop_image_camera.SetCameraId(start_camera_id);
                reconstruction_tmp.AddCamera(loop_image_camera);
                for (auto image_id : loop_image_set) {
                    class Image & image = reconstruction_tmp.Image(image_id);
                    image.SetCameraId(start_camera_id);
                }

                std::vector<sweep_t> registered_sweep_ids = reconstruction_tmp.RegisterSweepIds();
                for (sweep_t sweep_id : registered_sweep_ids) {
                    if (loop_lidar_set.find(sweep_id) == loop_lidar_set.end()) {
                        class LidarSweep lidarsweep = reconstruction_tmp.LidarSweep(sweep_id);

                        class Image image_sweep;
                        image_sweep.SetImageId(start_image_id+new_idx); 
                        image_sweep.SetName("prior_" + lidarsweep.Name());
                        image_sweep.SetCameraId(1);

                        image_sweep.SetTvec(lidarsweep.Tvec());
                        image_sweep.SetQvec(lidarsweep.Qvec());

                        reconstruction_tmp.AddImage(image_sweep);
                        reconstruction_tmp.RegisterImage(image_sweep.ImageId());

                        new_idx++;
                    }
                }
                for (sweep_t sweep_id : loop_lidar_set) {
                    class LidarSweep lidarsweep = reconstruction_tmp.LidarSweep(sweep_id);

                    class Image image_sweep;
                    image_sweep.SetImageId(start_image_id+new_idx); 
                    image_sweep.SetName("prior_" + lidarsweep.Name());
                    image_sweep.SetCameraId(start_camera_id);

                    image_sweep.SetTvec(lidarsweep.Tvec());
                    image_sweep.SetQvec(lidarsweep.Qvec());

                    reconstruction_tmp.AddImage(image_sweep);
                    reconstruction_tmp.RegisterImage(image_sweep.ImageId());

                    new_idx++;
                }

                std::string rec_path = JoinPaths(workspace_path_, "lidar/0-org");
                if (!ExistsPath(rec_path)) {
                    boost::filesystem::create_directories(rec_path);
                }

                reconstruction_tmp.WriteReconstruction(rec_path, true);
                reconstruction_tmp.ExportMapPoints(rec_path + "/sparse.ply");
            }

            if (true){
                std::string rec_path = JoinPaths(workspace_path_, "lidar/0");
                if (!ExistsPath(rec_path)) {
                    boost::filesystem::create_directories(rec_path);
                }
                Reconstruction rig_rec;
                reconstruction->ConvertRigReconstruction(rig_rec);

                image_t start_image_id = 1;
                camera_t start_camera_id = 1;
                for(auto image: rig_rec.Images()){
                    if(image.first > start_image_id){
                        start_image_id = image.first;
                    }
                    if (start_camera_id < image.second.CameraId()){
                        start_camera_id = image.second.CameraId();
                    }
                }

                for (auto cam : rig_rec.Cameras()){
                    if (start_camera_id < cam.first){
                        start_camera_id = cam.first;
                    }
                }
                start_image_id ++;
                start_camera_id ++;

                image_t new_idx = 1;
                std::cout << "Start: " << start_image_id << ", " << start_camera_id << std::endl;
                std::vector<sweep_t> registered_sweep_ids = reconstruction->RegisterSweepIds();
                for (sweep_t sweep_id : registered_sweep_ids) {
                    class LidarSweep lidarsweep = reconstruction->LidarSweep(sweep_id);

                    class Image image_sweep;
                    image_sweep.SetImageId(start_image_id+new_idx); 
                    image_sweep.SetName("prior_" + lidarsweep.Name());
                    image_sweep.SetCameraId(1);

                    image_sweep.SetTvec(lidarsweep.Tvec());
                    image_sweep.SetQvec(lidarsweep.Qvec());

                    rig_rec.AddImage(image_sweep);
                    rig_rec.RegisterImage(image_sweep.ImageId());

                    new_idx++;
                }

                rig_rec.WriteReconstruction(rec_path, true);
                rig_rec.ExportMapPoints(rec_path + "/sparse.ply");
            }
        }


        //Decide whether this is a valid reconstruction, if not, try next reconstrcution
        bool rec_success = (reconstruction->NumRegisterImages() >= scene_graph_container_->NumImages() * 0.1);
        if (rec_success) {
            mapper->EndReconstruction(false);
        } else {
            mapper->EndReconstruction(true);
            reconstruction_manager_->Delete(reconstruction_idx);
            std::cout<<"The current reconstruction is too small, try another one"<<std::endl;
        }


        if (rec_success && options_->refine_separate_cameras) {
            std::cout << "BundleAdjust Optimization of separate cameras" << std::endl;

            const auto & image_ids = reconstruction->RegisterImageSortIds();
            int num_local_camera = 1;
            for (auto image_id : image_ids) {
                const class Image & image = reconstruction->Image(image_id);
                const class Camera & camera = reconstruction->Camera(image.CameraId());
                num_local_camera = std::max(camera.NumLocalCameras(), num_local_camera);
            }
            if (num_local_camera > 1) {
                std::shared_ptr<SceneGraphContainer> scene_graph = std::make_shared<SceneGraphContainer>();
                scene_graph_container_->ConvertRigSceneGraphContainer(*scene_graph.get(), reconstruction->RegisterImageSortIds());

                std::cout << "Rig SceneGraph: " << scene_graph->NumImages() << std::endl;

                // scene_graph->CorrespondenceGraph()->ExportToGraph(workspace_path_ + "/scene_graph_rig.png");

                std::shared_ptr<IncrementalMapper> rig_mapper = std::make_shared<IncrementalMapper>(scene_graph);
                rig_mapper->SetWorkspacePath(workspace_path_);

                reconstruction_manager_->Delete(reconstruction_idx);
                size_t rec_idx = reconstruction_manager_->Add();
                std::shared_ptr<Reconstruction> rig_reconstruction = reconstruction_manager_->Get(rec_idx);

                reconstruction->ConvertRigReconstruction(*rig_reconstruction.get());
                    
                std::cout << "Register Images: " << rig_reconstruction->NumRegisterImages() << std::endl;

                rig_mapper->BeginReconstruction(rig_reconstruction);

                IndependentMapperOptions options = *options_.get();
                options.complete_max_reproj_error = 12.0;
                options.merge_max_reproj_error = 12;

                for (int iter = 0; iter < 2; ++iter) {

                    PrintHeading1("Retriangulation");
                    CompleteAndMergeTracks(options, rig_mapper);
                    std::cout << "  => Retriangulated observations: " << rig_mapper->Retriangulate(options.Triangulation())
                            << std::endl;

                    PrintHeading1("BundleAdjust Optimization of Separate Cameras");
                    BundleAdjustmentOptions ba_options = options.GlobalBundleAdjustment();
                    if(options.use_prior_align_only){
                        ba_options.use_prior_absolute_location = false;
                    }
                    ba_options.prior_absolute_location_weight = 0.05;

                    ba_options.solver_options.minimizer_progress_to_stdout = true;
                    ba_options.solver_options.max_num_iterations = 20;
                    ba_options.solver_options.max_linear_solver_iterations = 100;

                    // if (rig_reconstruction->NumRegisterImages() < init_mapper_options_.num_fix_camera_first) {
                    ba_options.refine_focal_length = false;
                    ba_options.refine_principal_point = false;
                    ba_options.refine_extra_params = false;
                    ba_options.refine_local_extrinsics = false;
                    // }

                    std::cout << "Register Images: " << rig_reconstruction->NumRegisterImages() << std::endl;

                    rig_mapper->AdjustGlobalBundle(init_mapper_options_, ba_options);

                    if (init_mapper_options_.has_gps_prior && !init_mapper_options_.map_update) {
                        CHECK(rig_reconstruction->has_gps_prior);
                        rig_reconstruction->AlignWithPriorLocations(init_mapper_options_.max_error_gps,
                                                                    init_mapper_options_.max_error_horizontal_gps,
                                                                    init_mapper_options_.max_gps_time_offset);
                    }
                }

                FilterPointsFinal(options, rig_mapper);
                FilterImages(options, rig_mapper);
                rig_reconstruction->FilterAllFarawayImages();

                std::cout << "Mean Track Length: " << rig_reconstruction->ComputeMeanTrackLength() << std::endl;
                std::cout << "Mean Reprojection Error: " << rig_reconstruction->ComputeMeanReprojectionError() << std::endl;
                std::cout << "Mean Observation Per Register Image: " << rig_reconstruction->ComputeMeanObservationsPerRegImage() << std::endl;

                // std::string rig_path = JoinPaths(workspace_path_, "0-rig");
                // CreateDirIfNotExists(rig_path);
                // rig_reconstruction->WriteBinary(rig_path);
            }
        }

        // if ((options_->multiple_models && 
        //     reconstruction->NumRegisterImages() < scene_graph_container_->NumImages() * 0.1) ||
        //     reconstruction->NumRegisterImages() == 0) {
        //     mapper->EndReconstruction(true);
        //     reconstruction_manager_->Delete(reconstruction_idx);
        //     std::cout<<"The current reconstruction is too small, try another one"<<std::endl;
        // } else {
        //     mapper->EndReconstruction(false);
        // }

        std::cout << "multiple_models: " << options_->multiple_models << std::endl;
        std::cout << StringPrintf("reconstruction manager size: %d/%d\n", reconstruction_manager_->Size(), options_->max_num_models) << std::endl;
        std::cout << StringPrintf("total register images: %d/%d\n", mapper->NumTotalRegImages(), scene_graph_container_->NumImages()) << std::endl;

        if (!options_->multiple_models ||
            reconstruction_manager_->Size() >= options_->max_num_models ||
            mapper->NumTotalRegImages() >= scene_graph_container_->NumImages() - 1) {
            printf("%d %d %d\n", options_->multiple_models, reconstruction_manager_->Size(), mapper->NumTotalRegImages());
            break;
        }  
    }
}

}  // namespace sensemap
