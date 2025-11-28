// Copyright (c) 2019, SenseTime Group.
// All rights reserved.
#ifndef _OPTION_PARSER_H_
#define _OPTION_PARSER_H_


#include "Configurator_yaml.h"
#include "controllers/mapper_options.h"
#include "feature/extraction.h"
#include "feature/feature_matcher.h"
#include "feature/global_feature_extraction.h"
#include "util/misc.h"
#include "util/rgbd_helper.h"
#include "camera_rig_params.h"

extern "C" {
#include "apriltag3/apriltag.h"
#include "apriltag3/tag36h11.h"
#include "apriltag3/tag25h9.h"
#include "apriltag3/tag16h5.h"
#include "apriltag3/tagCircle21h7.h"
#include "apriltag3/tagCircle49h12.h"
#include "apriltag3/tagCustom48h12.h"
#include "apriltag3/tagStandard41h12.h"
#include "apriltag3/tagStandard52h13.h"
#include "apriltag3/common/getopt.h"
}

class OptionParser {
public:
    inline void GetImageReaderOptions(sensemap::ImageReaderOptions &reader_options, Configurator &param) {
        reader_options.image_path = param.GetArgument("image_path", "");
        reader_options.mask_path = param.GetArgument("mask_path", "");
        reader_options.num_local_cameras = static_cast<int>(param.GetArgument("num_local_cameras", 1));

        std::string image_type = param.GetArgument("image_type", "perspective");
        bool with_depth = (image_type.compare("rgbd") == 0);

        reader_options.with_depth = with_depth;
        reader_options.image_selection = static_cast<std::string>(param.GetArgument("image_selection", ""));
        reader_options.single_camera = static_cast<bool>(param.GetArgument("single_camera", 1));
        reader_options.single_camera_per_folder = static_cast<bool>(param.GetArgument("single_camera_per_folder", 0));
        reader_options.fixed_camera = static_cast<bool>(param.GetArgument("fixed_camera", 0));

        reader_options.gps_origin = param.GetArgument("gps_origin", "");

        reader_options.camera_model = param.GetArgument("camera_model", "SIMPLE_RADIAL");
        std::string camera_params = param.GetArgument("camera_params", "");
        if (!camera_params.empty()) {
            reader_options.camera_params = camera_params;
        }

        std::string camera_rig_params_file = param.GetArgument("rig_params_file", "");
        if (!camera_rig_params_file.empty()) {
            reader_options.local_camera_models.clear();
            reader_options.local_camera_params.clear();
            reader_options.local_camera_extrinsics.clear();

            if (sensemap::HasFileExtension(camera_rig_params_file, ".yaml")) {
                std::string device_id_str = camera_rig_params_file.substr(camera_rig_params_file.find("sfmrig_")+7);
                device_id_str = device_id_str.substr(0,device_id_str.size()-5);

                sensemap::CameraRigParams rig_params;
                if (rig_params.LoadParams(camera_rig_params_file)) {
                    std::cout << "num local cameras: " << rig_params.num_local_cameras << std::endl;
                    std::cout << "camera model: " << rig_params.camera_model << std::endl;
                    std::cout << "Local intrinsic params: " << rig_params.local_intrinsics_str << std::endl;
                    std::cout << "Local extrinsic params: " << rig_params.local_extrinsics_str << std::endl;

                    CHECK_EQ(reader_options.num_local_cameras, rig_params.num_local_cameras);
                    reader_options.num_local_cameras_devices.emplace(device_id_str,rig_params.num_local_cameras);
                    reader_options.local_camera_models.emplace(device_id_str,rig_params.camera_model);
                    reader_options.local_camera_params.emplace(device_id_str,rig_params.local_intrinsics_str);
                    reader_options.local_camera_extrinsics.emplace(device_id_str,rig_params.local_extrinsics_str);
                    reader_options.camera_model = rig_params.camera_model;
                } else {
                    std::cout << "failed to read rig params" << std::endl;
                    exit(-1);
                }
            }
            else{
                std::vector<std::string> rig_param_file_list = sensemap::GetRecursiveFileList(camera_rig_params_file);
                for (const auto file : rig_param_file_list) {
                    if (sensemap::HasFileExtension(file, ".yaml")) {
                        std::string device_id_str = file.substr(file.find("sfmrig_") + 7);
                        device_id_str = device_id_str.substr(0, device_id_str.size() - 5);

                        sensemap::CameraRigParams rig_params;
                        if (rig_params.LoadParams(file)) {
                            // CHECK_EQ(reader_options.num_local_cameras, rig_params.num_local_cameras);
                            if (reader_options.num_local_cameras != rig_params.num_local_cameras) {
                                continue;
                            }
                            
                            std::cout << "num local cameras: " << rig_params.num_local_cameras << std::endl;
                            std::cout << "camera model: " << rig_params.camera_model << std::endl;
                            std::cout << "Local intrinsic params: " << rig_params.local_intrinsics_str << std::endl;
                            std::cout << "Local extrinsic params: " << rig_params.local_extrinsics_str << std::endl;

                            reader_options.num_local_cameras_devices.emplace(device_id_str,
                                                                             rig_params.num_local_cameras);
                            reader_options.local_camera_models.emplace(device_id_str, rig_params.camera_model);
                            reader_options.local_camera_params.emplace(device_id_str, rig_params.local_intrinsics_str);
                            reader_options.local_camera_extrinsics.emplace(device_id_str,
                                                                           rig_params.local_extrinsics_str);
                            reader_options.camera_model = rig_params.camera_model;
                        } else {
                            std::cout << "failed to read rig params" << std::endl;
                            exit(-1);
                        }
                    }
                }
            }
        }
        reader_options.read_image_info_first = static_cast<bool>(param.GetArgument("read_image_info_first", 0));
        reader_options.bitmap_read_num_threads = param.GetArgument("bitmap_read_num_threads", 4);
    }

    inline void GetImageReaderOptions(sensemap::ImageReaderOptions &reader_options, 
        Configurator &param, int child_id) {
        std::string image_path = param.GetArgument("image_path", "");
        std::string mask_path = param.GetArgument("mask_path", "");
        std::string cameras_param_file = param.GetArgument("camera_param_file", "");
        Configurator camera_param;
        camera_param.Load(cameras_param_file.c_str());

        reader_options.child_path = camera_param.GetArgument("sub_path_" + std::to_string(child_id), "");
        reader_options.image_path = image_path;
        reader_options.mask_path = mask_path;
        reader_options.num_local_cameras = static_cast<int>(camera_param.GetArgument("num_local_cameras_" + std::to_string(child_id), 1));

        std::string image_type = camera_param.GetArgument("image_type_" + std::to_string(child_id), "perspective");
        bool with_depth = (image_type.compare("rgbd") == 0);

        reader_options.with_depth = with_depth;
        reader_options.image_selection = static_cast<std::string>(camera_param.GetArgument("image_selection_" + std::to_string(child_id), ""));
        reader_options.single_camera = static_cast<bool>(camera_param.GetArgument("single_camera_" + std::to_string(child_id), 0));
        reader_options.single_camera_per_folder = static_cast<bool>(camera_param.GetArgument("single_camera_per_folder_" + std::to_string(child_id), 0));
        reader_options.fixed_camera = static_cast<bool>(camera_param.GetArgument("fixed_camera_" + std::to_string(child_id), 0));

        reader_options.gps_origin = param.GetArgument("gps_origin", "");

        reader_options.camera_model = camera_param.GetArgument("camera_model_" + std::to_string(child_id), "SIMPLE_RADIAL");
        std::string camera_params = camera_param.GetArgument("camera_params_" + std::to_string(child_id), "");
        if (!camera_params.empty()) {
            reader_options.camera_params = camera_params;
        }

        std::string camera_rig_params_file = camera_param.GetArgument("rig_params_file_" + std::to_string(child_id), "");
        if (!camera_rig_params_file.empty()) {
            reader_options.local_camera_models.clear();
            reader_options.local_camera_params.clear();
            reader_options.local_camera_extrinsics.clear();

            if (sensemap::HasFileExtension(camera_rig_params_file, ".yaml")) {
                std::string device_id_str = camera_rig_params_file.substr(camera_rig_params_file.find("sfmrig_")+7);
                device_id_str = device_id_str.substr(0,device_id_str.size()-5);

                sensemap::CameraRigParams rig_params;
                if (rig_params.LoadParams(camera_rig_params_file)) {
                    std::cout << "num local cameras: " << rig_params.num_local_cameras << std::endl;
                    std::cout << "camera model: " << rig_params.camera_model << std::endl;
                    std::cout << "Local intrinsic params: " << rig_params.local_intrinsics_str << std::endl;
                    std::cout << "Local extrinsic params: " << rig_params.local_extrinsics_str << std::endl;

                    CHECK_EQ(reader_options.num_local_cameras, rig_params.num_local_cameras);
                    reader_options.num_local_cameras_devices.emplace(device_id_str,rig_params.num_local_cameras);
                    reader_options.local_camera_models.emplace(device_id_str,rig_params.camera_model);
                    reader_options.local_camera_params.emplace(device_id_str,rig_params.local_intrinsics_str);
                    reader_options.local_camera_extrinsics.emplace(device_id_str,rig_params.local_extrinsics_str);
                    reader_options.camera_model = rig_params.camera_model;
                } else {
                    std::cout << "failed to read rig params" << std::endl;
                    exit(-1);
                }
            }
            else{
                std::vector<std::string> rig_param_file_list = sensemap::GetRecursiveFileList(camera_rig_params_file);
                for (const auto file : rig_param_file_list) {
                    if (sensemap::HasFileExtension(file, ".yaml")) {
                        std::string device_id_str = file.substr(file.find("sfmrig_") + 7);
                        device_id_str = device_id_str.substr(0, device_id_str.size() - 5);
                        sensemap::CameraRigParams rig_params;
                        if (rig_params.LoadParams(file)) {
                            std::cout << "num local cameras: " << rig_params.num_local_cameras << std::endl;
                            std::cout << "camera model: " << rig_params.camera_model << std::endl;
                            std::cout << "Local intrinsic params: " << rig_params.local_intrinsics_str << std::endl;
                            std::cout << "Local extrinsic params: " << rig_params.local_extrinsics_str << std::endl;

                            CHECK_EQ(reader_options.num_local_cameras, rig_params.num_local_cameras);
                            reader_options.num_local_cameras_devices.emplace(device_id_str,
                                                                             rig_params.num_local_cameras);
                            reader_options.local_camera_models.emplace(device_id_str, rig_params.camera_model);
                            reader_options.local_camera_params.emplace(device_id_str, rig_params.local_intrinsics_str);
                            reader_options.local_camera_extrinsics.emplace(device_id_str,
                                                                           rig_params.local_extrinsics_str);
                            reader_options.camera_model = rig_params.camera_model;
                        } else {
                            std::cout << "failed to read rig params" << std::endl;
                            exit(-1);
                        }
                    }
                }
            }
        }
        reader_options.read_image_info_first = static_cast<bool>(param.GetArgument("read_image_info_first", 0));
        reader_options.bitmap_read_num_threads = param.GetArgument("bitmap_read_num_threads", 4);
    }

    inline void GetFeatureExtractionOptions(sensemap::SiftExtractionOptions &sift_extraction, Configurator &param) {
        sift_extraction.num_threads = param.GetArgument("feature_extraction_num_threads", -1);
        sift_extraction.use_gpu = static_cast<bool>(param.GetArgument("feature_extraction_use_gpu", 1));
        sift_extraction.gpu_index = param.GetArgument("feature_extraction_gpu_index", "-1");
        sift_extraction.first_octave = param.GetArgument("feature_extraction_first_octave", -1);
        sift_extraction.num_octaves = param.GetArgument("feature_extraction_num_octaves", 4);
        sift_extraction.octave_resolution = param.GetArgument("feature_extraction_octave_resolution", 3);
        sift_extraction.peak_threshold = param.GetArgument("sift_peak_threshold", 0.00666666666667f);
        sift_extraction.min_num_features_customized = param.GetArgument("min_num_features_customized", 1024);
        sift_extraction.max_num_features_customized = param.GetArgument("max_num_features_customized", 4096);
        sift_extraction.max_image_size = param.GetArgument("max_image_size", 6144);
        sift_extraction.estimate_affine_shape = param.GetArgument("estimate_affine_shape", 0);
        sift_extraction.domain_size_pooling = param.GetArgument("domain_size_pooling", 0);

        sift_extraction.convert_to_perspective_image =
            static_cast<bool>(param.GetArgument("convert_to_perspective_image", 1));
        sift_extraction.perspective_image_count = static_cast<int>(param.GetArgument("perspective_image_count", 6));
        sift_extraction.perspective_image_width = static_cast<int>(param.GetArgument("perspective_image_width", 600));
        sift_extraction.perspective_image_height = static_cast<int>(param.GetArgument("perspective_image_height", 600));
        if (param.GetArgument("camera_model", "SIMPLE_RADIAL") == "SPHERICAL") {
            std::vector<double> panorama_camera_param =
                sensemap::CSVToVector<double>(param.GetArgument("camera_params", ""));
            sift_extraction.panorama_image_width = (int)panorama_camera_param[1];
            sift_extraction.panorama_image_height = (int)panorama_camera_param[2];
        }

        sift_extraction.fov_w = static_cast<int>(param.GetArgument("fov_w", 60));

        sift_extraction.binary_descriptor_pattern = param.GetArgument("binary_descriptor_pattern","");
        sift_extraction.pca_matrix_path = param.GetArgument("pca_matrix_path","");
        // AprilTag Related
        sift_extraction.detect_apriltag = static_cast<bool>(param.GetArgument("detect_apriltag", 0));
        const std::string tag_code = param.GetArgument("apriltag_code", "36h11");

        if (tag_code == "25h9") {
            sift_extraction.apriltag_family = tag25h9_create();
        } else if (tag_code == "36h11") {
            sift_extraction.apriltag_family = tag36h11_create();
        } else {
            std::cout << "Invalid tag family specified" << std::endl;
            exit(-1);
        }
    }

    inline void GetFeatureExtractionOptions(sensemap::SiftExtractionOptions &sift_extraction, Configurator &param, int child_id) {
        std::string cameras_param_file = param.GetArgument("camera_param_file", "");
        Configurator camera_param;
        camera_param.Load(cameras_param_file.c_str());

        sift_extraction.first_octave = camera_param.GetArgument("feature_extraction_first_octave_" + std::to_string(child_id), 
            sift_extraction.first_octave);
        sift_extraction.num_octaves = camera_param.GetArgument("feature_extraction_num_octaves_" + std::to_string(child_id), 
            sift_extraction.num_octaves);
        sift_extraction.octave_resolution = camera_param.GetArgument("feature_extraction_octave_resolution_" + std::to_string(child_id),  
            sift_extraction.octave_resolution);
        sift_extraction.peak_threshold = camera_param.GetArgument("sift_peak_threshold_" + std::to_string(child_id), 
            (float)sift_extraction.peak_threshold);
        sift_extraction.min_num_features_customized = camera_param.GetArgument("min_num_features_customized_" + std::to_string(child_id), 
            sift_extraction.min_num_features_customized);
        sift_extraction.max_num_features_customized = camera_param.GetArgument("max_num_features_customized_" + std::to_string(child_id), 
            sift_extraction.max_num_features_customized);
        sift_extraction.max_image_size = camera_param.GetArgument("max_image_size_" + std::to_string(child_id),
            sift_extraction.max_image_size);
        sift_extraction.estimate_affine_shape = camera_param.GetArgument("estimate_affine_shape_" + std::to_string(child_id), 
            sift_extraction.estimate_affine_shape);
        sift_extraction.domain_size_pooling = camera_param.GetArgument("domain_size_pooling_" + std::to_string(child_id), 
            sift_extraction.domain_size_pooling);
    }

    inline void GetGlobalFeatureExtractionOptions(sensemap::GlobalFeatureExtractionOptions &options, Configurator &param){

        options.vlad_code_book_path = param.GetArgument("vlad_code_book_path","");
    } 


    inline void GetFeatureMatchingOptions(sensemap::FeatureMatchingOptions &options, Configurator &param, int child_id = -1) {

        typedef sensemap::FeatureMatchingOptions::RetrieveType RetrieveType;
        RetrieveType retrieve_type = (RetrieveType)param.GetArgument("retrieve_type", 0);
        options.retrieve_type = retrieve_type;

        options.min_track_degree = static_cast<int>(param.GetArgument("min_track_degree", 3));

        options.max_cover_per_view = static_cast<int>(param.GetArgument("max_cover_per_view", 800));
        options.select_range_match_point = static_cast<int>(param.GetArgument("select_range_match_point", 0));
        options.track_preoperation = static_cast<bool>(param.GetArgument("track_preoperation", 1));
        options.detect_apriltag_ = static_cast<bool>(param.GetArgument("detect_apriltag", 0));
        options.delete_duplicated_images_ = static_cast<bool>(param.GetArgument("delete_duplicated_images", 1));

        options.track_block_radius = static_cast<int>(param.GetArgument("track_block_radius", 200));;
        options.track_max_per_block = static_cast<int>(param.GetArgument("track_max_per_block", 30));
        options.track_min_per_block = static_cast<int>(param.GetArgument("track_min_per_block", 20));
        options.track_max_cover_per_view = static_cast<int>(param.GetArgument("track_max_cover_per_view", 5000));

        options.max_match_distance = param.GetArgument("max_match_distance", 80.0f);

        std::string method = param.GetArgument("matching_method", "exhaustive");
        if (method.compare("exhaustive") == 0) {
            options.method_ = sensemap::FeatureMatchingOptions::MatchMethod::EXHAUSTIVE;
        } else if (method.compare("sequential") == 0) {
            options.method_ = sensemap::FeatureMatchingOptions::MatchMethod::SEQUENTIAL;
        } else if (method.compare("vocabtree") == 0) {
            options.method_ = sensemap::FeatureMatchingOptions::MatchMethod::VOCABTREE;
        } else if (method.compare("spatial") == 0) {
            options.method_ = sensemap::FeatureMatchingOptions::MatchMethod::SPATIAL;
        } else if (method.compare("hybrid") == 0) {
            options.method_ = sensemap::FeatureMatchingOptions::MatchMethod::HYBRID;

            std::string inner_method = param.GetArgument("matching_method_inside_cluster", "sequential");
            if (inner_method.compare("sequential") == 0) {
                options.hybrid_matching_.method_inside_cluster = sensemap::FeatureMatchingOptions::MatchMethod::SEQUENTIAL;
            } else if (inner_method.compare("exhaustive") == 0) {
                options.hybrid_matching_.method_inside_cluster = sensemap::FeatureMatchingOptions::MatchMethod::EXHAUSTIVE;
            } else if (inner_method.compare("vocab_tree") == 0) {
                options.hybrid_matching_.method_inside_cluster = sensemap::FeatureMatchingOptions::MatchMethod::VOCABTREE;
            } else {
                CHECK(false) << "invalid matching method inside cluster";
            }
        } else if (method.compare("hybrid_input") == 0) {
            options.method_ = sensemap::FeatureMatchingOptions::MatchMethod::HYBRID_INPUT;
        } else {
            CHECK(false) << "invalid matching method";
        }

        // Spatial Matching
        options.spatial_matching_.max_num_neighbors = param.GetArgument("spatial_max_num_neighbors", 25);
        options.spatial_matching_.max_distance = param.GetArgument("spatial_max_distance", 50.0f);

        // Vocabtree Matching
        options.vocabtree_matching_.vocab_tree_path = param.GetArgument("vocab_path", "");
        options.vocabtree_matching_.vlad_code_book_path = param.GetArgument("vlad_code_book_path", "");
        options.vocabtree_matching_.num_images = param.GetArgument("vocab_matching_num_images", 50);
        options.vocabtree_matching_.num_nearest_neighbors =
            param.GetArgument("vocab_matching_num_nearest_neighbors", 10);
        options.vocabtree_matching_.max_score_factor = param.GetArgument("max_score_factor", 0.0f);
        options.vocabtree_matching_.vocab_tree_max_num_features = param.GetArgument("vocab_tree_max_num_features", 6144);

        // Sequential Matching
        options.sequential_matching_.vocab_tree_path = param.GetArgument("vocab_path", "");
        options.sequential_matching_.vlad_code_book_path = param.GetArgument("vlad_code_book_path", "");
        options.sequential_matching_.loop_detection_num_threads = param.GetArgument("loop_detection_num_threads", -1);
        options.sequential_matching_.loop_detection = static_cast<bool>(param.GetArgument("loop_detection", 0));
        options.sequential_matching_.loop_detection_max_num_features =
            param.GetArgument("loop_detection_max_num_features", 6144);
        options.sequential_matching_.robust_loop_detection =
            static_cast<bool>(param.GetArgument("robust_loop_detection", 0));
        options.sequential_matching_.loop_detection_before_sequential_matching =
            static_cast<bool>(param.GetArgument("loop_detection_before_sequential_matching",0));
        
        options.sequential_matching_.loop_detection_period = param.GetArgument("loop_detection_period", 1);
        options.sequential_matching_.loop_detection_num_images = param.GetArgument("loop_detection_num_images", 50);
        options.sequential_matching_.overlap = param.GetArgument("overlap", 10);
        options.sequential_matching_.loop_consistency_threshold = param.GetArgument("loop_consistency_threshold", 3);
        options.sequential_matching_.max_recent_score_factor = param.GetArgument("max_recent_score_factor", 0.4f);
        options.sequential_matching_.best_acc_score_factor = param.GetArgument("best_acc_score_factor", 0.2f);



        options.sequential_matching_.local_max_recent_score_factor =
            param.GetArgument("local_max_recent_score_factor", 0.8f);
        options.sequential_matching_.local_best_acc_score_factor =
            param.GetArgument("local_best_acc_score_factor", 0.75f);
        options.sequential_matching_.local_region_repetitive = param.GetArgument("local_region_repetitive", 0);

        options.sequential_matching_.local_triplet_checking = param.GetArgument("local_triplet_checking", 0);
        options.global_triplet_checking = param.GetArgument("global_triplet_checking", 0);
        options.local_invalid_theta_dis = param.GetArgument("local_invalid_theta_dis", 10.0f);
        options.global_median_invalid_theta_dis = param.GetArgument("global_median_invalid_theta_dis", 10.0f);
        options.global_mean_invalid_theta_dis = param.GetArgument("global_mean_invalid_theta_dis", 10.0f);
        options.ambiguous_triple_count = param.GetArgument("ambiguous_triple_count", 3);

        options.pair_matching_.min_covered_sub_image_ratio = param.GetArgument("min_covered_sub_image_ratio", 0.5f);
        options.pair_matching_.covered_sub_image_ratio_strong_loop =
            param.GetArgument("covered_sub_image_ratio_strong_loop", 0.9f);
        options.pair_matching_.strong_loop_check_neighbor_count_src =
            param.GetArgument("strong_loop_check_neighbor_count_src", 1);
        options.pair_matching_.strong_loop_check_neighbor_count_dst =
            param.GetArgument("strong_loop_check_neighbor_count_dst", 10);
        options.pair_matching_.min_matched_feature_per_piece = param.GetArgument("ls", 5);
        options.pair_matching_.strong_loop_transitivity = param.GetArgument("strong_loop_transitivity", 0);
        options.pair_matching_.transitive_strong_loop_neighbor_count =
            param.GetArgument("transitive_strong_loop_neighbor_count", 5);

        options.pair_matching_.perspective_image_count = param.GetArgument("perspective_image_count", 8);
        options.pair_matching_.convert_to_perspective_image =
            static_cast<bool>(param.GetArgument("convert_to_perspective_image", 0));

        options.pair_matching_.num_threads = param.GetArgument("matching_num_threads", -1);
        options.pair_matching_.use_gpu = static_cast<bool>(param.GetArgument("matching_use_gpu", 1));
        options.pair_matching_.gpu_index = param.GetArgument("matching_gpu_index", "-1");
        options.pair_matching_.guided_matching = static_cast<bool>(param.GetArgument("guided_matching", 1));
        options.pair_matching_.sub_matching = static_cast<bool>(param.GetArgument("sub_matching", 0));
        options.pair_matching_.self_matching = static_cast<bool>(param.GetArgument("self_matching", 0));
        // if (options.pair_matching_.sub_matching) {
        //     size_t num_local_cameras;
        //     if (child_id < 0){
        //         num_local_cameras = param.GetArgument("num_local_cameras", 1);
        //     } else {      
        //         std::string cameras_param_file = param.GetArgument("camera_param_file", "");
        //         Configurator camera_param;
        //         camera_param.Load(cameras_param_file.c_str());
        //         num_local_cameras = static_cast<int>(camera_param.GetArgument("num_local_cameras_" + std::to_string(child_id), 1));
        //     }
        //     // if (num_local_cameras < 6) {
        //     //     options.pair_matching_.sub_matching = 0;
        //     //     std::cout << "Warning! force to false of sub_matching because of non pro2 sources input!" << std::endl;
        //     // }
        // }

        options.pair_matching_.multiple_models = static_cast<bool>(param.GetArgument("matching_multiple_models", 0));

        options.pair_matching_.guided_matching_multi_homography =
            static_cast<bool>(param.GetArgument("guided_matching_multi_homography", 0));

        options.pair_matching_.max_num_matches = param.GetArgument("max_num_matches", 30000);
        options.pair_matching_.min_num_inliers = param.GetArgument("min_num_inliers", 60);

        options.pair_matching_.max_distance = static_cast<double>(param.GetArgument("max_distance",0.7f));
        options.pair_matching_.max_ratio = static_cast<double>(param.GetArgument("max_ratio",0.8f));
        options.pair_matching_.guided_match_max_ratio = static_cast<double>(param.GetArgument("guided_match_max_ratio",0.8f));

        options.pair_matching_.max_error = static_cast<double>(param.GetArgument("check_max_error", 4.0f));

        if (param.GetArgument("camera_model", "SIMPLE_RADIAL") == "SPHERICAL") {
            options.pair_matching_.max_angular_error = param.GetArgument("matching_max_angular_error", 0.4f);
            options.pair_matching_.guided_match_max_ratio = param.GetArgument("guided_match_max_ratio", 0.8f);
        }
    }

    inline void GetMapperOptions(sensemap::IndependentMapperOptions& mapper_options, Configurator &param, int child_id = -1) {
        
        std::string image_type = param.GetArgument("image_type", "perspective");
        bool with_depth = (image_type.compare("rgbd") == 0);
        mapper_options.with_depth = with_depth;

        mapper_options.sub_matching = param.GetArgument("sub_matching", 0);

        mapper_options.self_matching = param.GetArgument("self_matching", 0);

        mapper_options.extract_colors = param.GetArgument("extract_colors", 0);

        if(with_depth){
            mapper_options.use_icp_relative_pose =
                    static_cast<bool>(param.GetArgument("use_icp_relative_pose", 0));
        }

        mapper_options.rgbd_delayed_start = 
            static_cast<bool>(param.GetArgument("rgbd_delayed_start", 0));
        mapper_options.rgbd_delayed_start_weights = 
            static_cast<double>(param.GetArgument("rgbd_delayed_start_weights", 500.0f));
        mapper_options.rgbd_ba_depth_weight = 
            static_cast<double>(param.GetArgument("rgbd_ba_depth_weight", 0.0f));
        mapper_options.rgbd_filter_depth_weight = 
            static_cast<double>(param.GetArgument("rgbd_filter_depth_weight", 0.0f));
        mapper_options.rgbd_max_reproj_depth =
            static_cast<double>(param.GetArgument("rgbd_max_reproj_depth", 0.0f));
        mapper_options.rgbd_pose_refine_depth_weight = 
            static_cast<double>(param.GetArgument("rgbd_pose_refine_depth_weight", 0.0f));

        mapper_options.icp_base_weight =
                static_cast<double>(param.GetArgument("icp_base_weight", 1.0f));

        mapper_options.use_gravity =
                static_cast<bool>(param.GetArgument("use_gravity", 0));

        mapper_options.gravity_base_weight =
                static_cast<double>(param.GetArgument("gravity_base_weight", 1.0f));

        mapper_options.use_time_domain_smoothing =
                static_cast<bool>(param.GetArgument("use_time_domain_smoothing", 0));

        mapper_options.time_domain_smoothing_weight =
                static_cast<double>(param.GetArgument("time_domain_smoothing_weight", 2.0f));

        std::string rgbd_params_file = param.GetArgument("rgbd_params_file", "");
        if (!rgbd_params_file.empty()) {
            auto calib_reader = sensemap::GetCalibBinReaderFromName(rgbd_params_file);
            calib_reader->ReadCalib(rgbd_params_file);
            mapper_options.rgbd_camera_params = calib_reader->ToParamString();
            std::cout << "rgbd_camera_params: " << mapper_options.rgbd_camera_params << std::endl;
        }

        std::string match_method = param.GetArgument("matching_method");
        mapper_options.match_method = match_method;

        mapper_options.offline_slam =
            static_cast<bool>(param.GetArgument("offline_slam", 0));

        mapper_options.min_covisible_mappoint_num =
            static_cast<int>(param.GetArgument("min_covisible_mappoint_num", 30));

        mapper_options.min_covisible_mappoint_ratio =
            static_cast<double>(param.GetArgument("min_covisible_mappoint_ratio", 0.2f));

        mapper_options.offline_slam_max_next_image_num =
            static_cast<int>(param.GetArgument("offline_slam_max_next_image_num", 10));

        mapper_options.initial_two_frame_interval_offline_slam =
            static_cast<int>(param.GetArgument("initial_two_frame_interval_offline_slam", 20));

        mapper_options.ba_refine_principal_point =
            static_cast<bool>(param.GetArgument("ba_refine_principal_point", 0));
        mapper_options.ba_refine_principal_point_final =
            static_cast<bool>(param.GetArgument("ba_refine_principal_point_final", mapper_options.ba_refine_principal_point));
        mapper_options.ba_refine_focal_length =
            static_cast<bool>(param.GetArgument("ba_refine_focal_length", 1));
        mapper_options.ba_refine_extra_params =
            static_cast<bool>(param.GetArgument("ba_refine_extra_params", 1));
        mapper_options.ba_refine_local_extrinsics =
            static_cast<bool>(param.GetArgument("ba_refine_local_extrinsics", 1));
        mapper_options.ba_refine_extrinsics =
            static_cast<bool>(param.GetArgument("ba_refine_extrinsics", 1));
        mapper_options.local_relative_translation_constraint = 
            static_cast<bool>(param.GetArgument("local_relative_translation_constraint", 0));

        mapper_options.ba_global_use_pba =
            static_cast<bool>(param.GetArgument("ba_global_use_pba", 0));
        mapper_options.ba_global_pba_gpu_index =
            static_cast<int>(param.GetArgument("ba_global_pba_gpu_index", -1));
        mapper_options.ba_global_max_num_iterations =
            static_cast<int>(param.GetArgument("ba_global_max_num_iterations", 30));
        mapper_options.ba_global_max_refinements =
            static_cast<int>(param.GetArgument("ba_global_max_refinements", 2));

        mapper_options.refine_separate_cameras = static_cast<bool>(param.GetArgument("refine_separate_cameras", 0));

        mapper_options.ba_global_loss_function =
            param.GetArgument("ba_global_loss_function", "huber");

        mapper_options.ba_global_images_ratio =
            param.GetArgument("ba_global_images_ratio", 1.1f);
        
        mapper_options.ba_global_points_ratio =
            param.GetArgument("ba_global_points_ratio", 1.1f);
        
        mapper_options.ba_local_num_images =
            param.GetArgument("ba_local_num_images", 6);
        
        mapper_options.ba_local_max_num_iterations =
            param.GetArgument("ba_local_max_num_iterations", 25);
        mapper_options.ba_local_max_refinements = 
            param.GetArgument("ba_local_max_refinements",2);

        mapper_options.ba_normalize_reconstruction = 
            static_cast<bool>(param.GetArgument("ba_normalize_reconstruction",0));


        mapper_options.delete_redundant_old_images = 
            static_cast<bool>(param.GetArgument("delete_redundant_old_images", 0)); 

        mapper_options.ba_plane_constrain =
            static_cast<bool>(param.GetArgument("ba_plane_constrain", 0));

        mapper_options.plane_constrain_start_image_num =
            param.GetArgument("plane_constrain_start_image_num", 50);

        mapper_options.ba_plane_weight = param.GetArgument("ba_plane_weight", 0.1f);

        mapper_options.gba_weighted =
            static_cast<bool>(param.GetArgument("gba_weighted", 0));

        mapper_options.max_distance_to_plane =
            param.GetArgument("max_distance_to_plane", 0.05f);

        mapper_options.max_plane_count = param.GetArgument("max_plane_count", 3);


        mapper_options.num_images_for_self_calibration =
            param.GetArgument("num_images_for_self_calibration", 10000);

        mapper_options.batched_sfm =
            static_cast<bool>(param.GetArgument("batched_sfm", 0));

        mapper_options.local_ba_batched =
            static_cast<bool>(param.GetArgument("local_ba_batched", 1));

        mapper_options.use_global_ba_update = static_cast<bool>(param.GetArgument("use_global_ba_update",1));

        mapper_options.init_image_id1 =
            static_cast<int>(param.GetArgument("init_image_id1", -1));
        mapper_options.init_image_id2 =
            static_cast<int>(param.GetArgument("init_image_id2", -1));
        mapper_options.init_from_uncertainty =
            static_cast<bool>(param.GetArgument("init_from_uncertainty", 0));
        mapper_options.init_min_num_inliers =
            param.GetArgument("init_min_num_inliers", 250);
        mapper_options.init_min_tri_angle =
            param.GetArgument("init_min_tri_angle", 4.0f);

        mapper_options.init_max_depth = param.GetArgument("init_max_depth", 4.0f);
        mapper_options.init_min_corrs_intra_view = param.GetArgument("init_min_corrs_intra_view", 30);

        mapper_options.filter_max_reproj_error =
            param.GetArgument("filter_max_reproj_error", 12.0f);

        mapper_options.merge_max_reproj_error =
            param.GetArgument("merge_max_reproj_error", 12.0f);
        mapper_options.complete_max_reproj_error =
            param.GetArgument("complete_max_reproj_error", 12.0f);

        mapper_options.re_min_ratio = param.GetArgument("re_min_ratio", 0.2f);

        mapper_options.filter_max_reproj_error_final =
            param.GetArgument("filter_max_reproj_error_final", 4.0f);
        mapper_options.filter_min_tri_angle_final =
            param.GetArgument("filter_min_tri_angle_final", 4.0f);
        mapper_options.filter_min_track_length_final =
            param.GetArgument("filter_min_track_length_final", 2);

        mapper_options.filter_min_tri_angle =
            param.GetArgument("filter_min_tri_angle", 4.0f);
        mapper_options.min_tri_angle = param.GetArgument("min_tri_angle", 4.0f);

        mapper_options.abs_pose_min_num_inliers =
            param.GetArgument("abs_pose_min_num_inliers", 12);
        mapper_options.abs_pose_min_inlier_ratio =
            param.GetArgument("abs_pose_min_inlier_ratio", 0.6f);

        mapper_options.min_inlier_ratio_to_best_pose =
            param.GetArgument("min_inlier_ratio_to_best_pose", 0.7f);

        mapper_options.min_inlier_ratio_verification_with_prior_pose =
            param.GetArgument("min_inlier_ratio_verification_with_prior_pose", 0.7f);

        mapper_options.extract_keyframe =
            static_cast<bool>(param.GetArgument("extract_keyframe", 0));
        mapper_options.register_nonkeyframe =
            static_cast<bool>(param.GetArgument("register_nonkeyframe", 1));

        std::string matching_method_inside_cluster = param.GetArgument("matching_method_inside_cluster","sequential");
        if (match_method.compare("sequential") != 0 &&
            (match_method.compare("hybrid") != 0 || matching_method_inside_cluster.compare("sequential") != 0)) {
            std::cout << "Warning! Can't extract keyframe because the variable is enable only in sequential"
                      << std::endl;
            mapper_options.image_collection = true;
        }

        mapper_options.num_first_force_be_keyframe =
            static_cast<int>(param.GetArgument("num_first_force_be_keyframe", 10));
        mapper_options.optim_inner_cluster =
            static_cast<bool>(param.GetArgument("optim_inner_cluster", 0));
        mapper_options.robust_camera_pose_estimate =
            static_cast<bool>(param.GetArgument("robust_camera_pose_estimate", 0));
        mapper_options.consecutive_camera_pose_top_k =
            static_cast<int>(param.GetArgument("consecutive_camera_pose_top_k", 2));
        mapper_options.consecutive_neighbor_ori =
            static_cast<int>(param.GetArgument("consecutive_neighbor_ori", 2));
        mapper_options.consecutive_neighbor_t =
            static_cast<int>(param.GetArgument("consecutive_neighbor_t", 1));
        mapper_options.consecutive_camera_pose_orientation =
            param.GetArgument("consecutive_camera_pose_orientation", 5.0f);
        mapper_options.consecutive_camera_pose_t =
            param.GetArgument("consecutive_camera_pose_t", 20.0f);

        mapper_options.min_inlier_ratio_to_best_model =
            param.GetArgument("min_inlier_ratio_to_best_model", 0.75f);
        mapper_options.local_region_repetitive =
            param.GetArgument("local_region_repetitive", 0);

        mapper_options.loop_image_weight = param.GetArgument("loop_image_weight", 1.0f);

        mapper_options.loop_image_min_id_difference =
            param.GetArgument("loop_image_min_id_difference", 20);

        mapper_options.num_fix_camera_first =
            static_cast<int>(param.GetArgument("num_fix_camera_first", 5));

        bool single_camera = static_cast<bool>(param.GetArgument("single_camera", 1));
        bool single_camera_per_folder = static_cast<bool>(param.GetArgument("single_camera_per_folder", 1));
        mapper_options.single_camera = single_camera || single_camera_per_folder;

        if (!mapper_options.single_camera) {
            std::cout << "Warning! Can't fix camera because single camera is disable, forcing the variable to be zero"
                      << std::endl;
            mapper_options.num_fix_camera_first = 0;
        }

        mapper_options.camera_fixed = param.GetArgument("fixed_camera", 0);

        mapper_options.max_triangulation_angle_degrees =
            static_cast<double>(param.GetArgument("max_triangulation_angle_degrees", 30.0f));

        mapper_options.min_visible_map_point_kf =
            static_cast<int>(param.GetArgument("min_visible_map_point_kf", 300));
        mapper_options.min_keyframe_step =
            static_cast<int>(param.GetArgument("min_keyframe_step", 1));

        mapper_options.min_pose_inlier_kf =
            static_cast<int>(param.GetArgument("min_pose_inlier_kf", 200));
        mapper_options.avg_min_dist_kf_factor =
            static_cast<double>(param.GetArgument("avg_min_dist_kf_factor", 1.0f));
        mapper_options.min_visiblility_score_ratio = 
            static_cast<double>(param.GetArgument("min_visiblility_score_ratio", 0.5f));
        mapper_options.mean_max_disparity_kf =
            static_cast<double>(param.GetArgument("mean_max_disparity_kf", 20.0f));
        mapper_options.abs_diff_kf =
            static_cast<int>(param.GetArgument("abs_diff_kf", 1000000));

        mapper_options.debug_info =
            static_cast<bool>(param.GetArgument("debug_info", 0));

        mapper_options.write_binary_model =
            static_cast<bool>(param.GetArgument("write_binary", 1));
        
        mapper_options.use_local_ba_retriangulate_all = 
            static_cast<bool>(param.GetArgument("use_local_ba_retriangulate_all", 1));

        mapper_options.max_error_gps = 
            static_cast<double>(param.GetArgument("max_error_gps", 3.0f));
        mapper_options.min_image_num_for_gps_error = param.GetArgument("min_image_num_for_gps_error", 10);

        mapper_options.max_error_horizontal_gps = 
            static_cast<double>(param.GetArgument("max_error_horizontal_gps", 3.0f));

        mapper_options.optimization_use_horizontal_gps_only =
            static_cast<bool>(param.GetArgument("optimization_use_horizontal_gps_only",0));
        mapper_options.max_gps_time_offset =
            static_cast<long>(param.GetArgument("max_gps_time_offset", 60000));
    
    
        int num_local_cameras = param.GetArgument("num_local_cameras", 1);
        std::string camera_model = param.GetArgument("camera_model", "SIMPLE_RADIAL");
        if (child_id >= 0) {      
            std::string cameras_param_file = param.GetArgument("camera_param_file", "");
            Configurator camera_param;
            camera_param.Load(cameras_param_file.c_str());
            num_local_cameras = camera_param.GetArgument("num_local_cameras_" + std::to_string(child_id), num_local_cameras);
            camera_model = camera_param.GetArgument ("camera_model_" + std::to_string(child_id), camera_model);
        }
        // Optimizing the instrisics is forbidden for spherical camera
        if (camera_model == "SPHERICAL" &&
            num_local_cameras == 1) {
            mapper_options.ba_refine_principal_point = false;
            mapper_options.ba_refine_focal_length = false;
            mapper_options.single_camera = true;
        }
        mapper_options.has_gps_prior = 
            static_cast<bool>(param.GetArgument("use_gps_prior", 0));

        mapper_options.use_prior_translation_only =
            static_cast<bool>(param.GetArgument("use_prior_translation_only", 0));

        mapper_options.use_prior_distance_only =
            static_cast<bool>(param.GetArgument("use_prior_distance_only", 0));

        mapper_options.use_prior_aggressively =
            static_cast<bool>(param.GetArgument("use_prior_aggressively", 0));

        mapper_options.prior_force_keyframe =
            static_cast<bool>(param.GetArgument("prior_force_keyframe", false));

        mapper_options.prior_pose_weight =
            static_cast<double>(param.GetArgument("prior_pose_weight", 1.0f));

        mapper_options.has_gps_prior = 
            static_cast<bool>(param.GetArgument("use_gps_prior", 0));

        mapper_options.use_prior_align_only = 
            static_cast<bool>(param.GetArgument("use_prior_align_only", 1));

        mapper_options.prior_absolute_location_weight =  
            static_cast<double>(param.GetArgument("prior_absolute_location_weight", 1.0f));
        
        mapper_options.prior_absolute_orientation_weight =  
            static_cast<double>(param.GetArgument("prior_absolute_orientation_weight", 0.0f));
        
        mapper_options.ba_blockba_debug = 
            static_cast<bool>(param.GetArgument("ba_blockba_debug", 0));
        mapper_options.ba_block_size = 
            static_cast<int>(param.GetArgument("ba_block_size", -1));
        mapper_options.ba_block_common_image_num =
            static_cast<int>(param.GetArgument("ba_block_common_image_num", 40));
        mapper_options.ba_blockba_frequency = 
            static_cast<int>(param.GetArgument("ba_blockba_frequency", 10));

        mapper_options. ba_block_min_connected_points_for_common_images =
            static_cast<int>(param.GetArgument("ba_block_min_connected_points_for_common_images", 100));


        mapper_options.explicit_loop_closure = static_cast<bool>(param.GetArgument("explicit_loop_closure", 0));
        mapper_options.max_id_difference_for_loop =
            static_cast<int>(param.GetArgument("max_id_difference_for_loop", 10));
        mapper_options.min_inconsistent_corr_ratio_for_loop =
            static_cast<double>(param.GetArgument("min_inconsistent_corr_ratio_for_loop", 0.9f));
        mapper_options.min_loop_pose_inlier_num = 
        static_cast<int>(param.GetArgument("min_loop_pose_inlier_num", 30));
        mapper_options.loop_weight =
            static_cast<double>(param.GetArgument("loop_weight", 1.0f));
        mapper_options.normal_edge_count_per_image =
            static_cast<int>(param.GetArgument("normal_edge_count_per_image", 5));
        mapper_options.max_loop_edge_count =
            static_cast<int>(param.GetArgument("max_loop_edge_count", 1));
        mapper_options.loop_check_interval =
            static_cast<int>(param.GetArgument("loop_check_interval", 500));
        mapper_options.optimize_sim3 =
            static_cast<bool>(param.GetArgument("optimize_sim3", 1));
        mapper_options.loop_closure_max_iter_num =
            static_cast<int>(param.GetArgument("loop_closure_max_iter_num", 100));
        mapper_options.normal_edge_min_common_points = 
            static_cast<int>(param.GetArgument("normal_edge_min_common_points", 50));
        mapper_options.loop_distance_factor_wrt_averge_baseline = 
            static_cast<double>(param.GetArgument("loop_distance_factor_wrt_averge_baseline", 6.0f));

        mapper_options.neighbor_distance_factor_wrt_averge_baseline = 
            static_cast<double>(param.GetArgument("neighbor_distance_factor_wrt_averge_baseline", 6.0f));   

        mapper_options.map_update = static_cast<bool>(param.GetArgument("map_update", 0));
        mapper_options.update_old_map = static_cast<bool>(param.GetArgument("update_old_map", 0));
        mapper_options.update_with_sequential_mode = static_cast<bool>(param.GetArgument("update_with_sequential_mode", 0));

        mapper_options.direct_mapper_type =  param.GetArgument("direct_mapper_type", 1);
        mapper_options.min_track_length = static_cast<int>(param.GetArgument("min_track_length", 3));

        mapper_options.create_max_angle_error = param.GetArgument("create_max_angle_error", 2.0f);
        mapper_options.continue_max_angle_error = param.GetArgument("continue_max_angle_error", 2.0f);
        mapper_options.complete_max_reproj_error = param.GetArgument("complete_max_reproj_error", 4.0f);

        mapper_options.multiple_models = static_cast<bool>(param.GetArgument("multiple_models", 0));

        // lidar option.
        mapper_options.lidar_sfm = static_cast<bool>(param.GetArgument("lidar_sfm", 0));
        mapper_options.lidar_path = param.GetArgument("lidar_path", "");
        mapper_options.lidar_prior_pose_file = param.GetArgument("lidar_prior_pose_file", "");

        std::string lidar_to_cam_str = param.GetArgument("lidar_to_cam_matrix", "");
        std::vector<double> lidar_to_cam_matrix = sensemap::CSVToVector<double>(lidar_to_cam_str);
        std::cout << "lidar_to_cam_matrix[Option]: " << lidar_to_cam_str << std::endl;
        std::cout << "param size: " << lidar_to_cam_matrix.size() << std::endl;
        if (lidar_to_cam_matrix.size() == 12) {
            mapper_options.lidar_to_cam_matrix << lidar_to_cam_matrix[0], lidar_to_cam_matrix[1], lidar_to_cam_matrix[2],
                lidar_to_cam_matrix[3], lidar_to_cam_matrix[4], lidar_to_cam_matrix[5], lidar_to_cam_matrix[6],
                lidar_to_cam_matrix[7], lidar_to_cam_matrix[8], lidar_to_cam_matrix[9], lidar_to_cam_matrix[10],
                lidar_to_cam_matrix[11];
            std::cout << "lidar_to_cam[Option]: " << mapper_options.lidar_to_cam_matrix << std::endl;
        }
    }
};

#endif
