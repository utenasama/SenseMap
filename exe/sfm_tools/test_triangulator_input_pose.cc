// Copyright (c) 2019, SenseTime Group.
// All rights reserved.

#include <boost/filesystem/path.hpp>
#include <opencv2/opencv.hpp>

#include "base/camera.h"
#include "base/image.h"
#include "base/image_reader.h"
#include "base/reconstruction_manager.h"
#include "graph/correspondence_graph.h"
#include "util/misc.h"

#include "cluster/fast_community.h"
#include "container/feature_data_container.h"
#include "feature/extraction.h"
#include "feature/feature_matcher.h"
#include "feature/utils.h"

#include "controllers/incremental_mapper_controller.h"

#include "../Configurator.h"

#include <dirent.h>
#include <sys/stat.h>
#include <fstream>
#include "base/pose.h"
#include "base/projection.h"

using namespace sensemap;

std::string configuration_file_path;
FILE* fs;

bool dirExists(const std::string& dirName_in) { return access(dirName_in.c_str(), F_OK) == 0; }

void FeatureExtraction(FeatureDataContainer& feature_data_container, Configurator& param) {
    std::string workspace_path = param.GetArgument("workspace_path", "");
    CHECK(!workspace_path.empty());

    ImageReaderOptions reader_options;
    reader_options.image_path = param.GetArgument("image_path", "");
    bool camera_rig = (reader_options.num_local_cameras > 1);

    bool exist_feature_file = false;
    if (boost::filesystem::exists(JoinPaths(workspace_path, "/cameras.bin")) &&
        boost::filesystem::exists(JoinPaths(workspace_path, "/features.bin"))) {
        if (boost::filesystem::exists(JoinPaths(workspace_path, "/local_cameras.bin"))) {
            feature_data_container.ReadCamerasBinaryData(JoinPaths(workspace_path, "/cameras.bin"), false);
            feature_data_container.ReadLocalCamerasBinaryData(JoinPaths(workspace_path, "/local_cameras.bin"));
        } else {
            feature_data_container.ReadCamerasBinaryData(JoinPaths(workspace_path, "/cameras.bin"), camera_rig);
        }
        feature_data_container.ReadImagesBinaryData(JoinPaths(workspace_path, "/features.bin"));
        exist_feature_file = true;
    } else if (boost::filesystem::exists(JoinPaths(workspace_path, "/cameras.txt")) &&
               boost::filesystem::exists(JoinPaths(workspace_path, "/features.txt"))) {
        if (boost::filesystem::exists(JoinPaths(workspace_path, "/local_cameras.txt"))) {
            feature_data_container.ReadCameras(JoinPaths(workspace_path, "/cameras.txt"), false);
            feature_data_container.ReadLocalCameras(JoinPaths(workspace_path, "/local_cameras.txt"));
        } else {
            feature_data_container.ReadCameras(JoinPaths(workspace_path, "/cameras.txt"), camera_rig);
        }
        feature_data_container.ReadImagesData(JoinPaths(workspace_path, "/features.txt"));
        exist_feature_file = true;
    } else if (boost::filesystem::exists(JoinPaths(workspace_path, "/cameras.txt")) &&
               boost::filesystem::exists(JoinPaths(workspace_path, "/features.bin"))) {
        if (boost::filesystem::exists(JoinPaths(workspace_path, "/local_cameras.txt"))) {
            feature_data_container.ReadCameras(JoinPaths(workspace_path, "/cameras.txt"), false);
            feature_data_container.ReadLocalCameras(JoinPaths(workspace_path, "/local_cameras.txt"));
        } else {
            feature_data_container.ReadCameras(JoinPaths(workspace_path, "/cameras.txt"), camera_rig);
        }
        feature_data_container.ReadImagesBinaryData(JoinPaths(workspace_path, "/features.bin"));
        exist_feature_file = true;
    }

    // If the camera model is Spherical
    if (param.GetArgument("camera_model", "SIMPLE_RADIAL") == "SPHERICAL" && exist_feature_file) {
        if (boost::filesystem::exists(workspace_path + "/sub_panorama.bin")) {
            feature_data_container.ReadSubPanoramaBinaryData(JoinPaths(workspace_path, "/sub_panorama.bin"));
        } else if (boost::filesystem::exists(workspace_path + "/sub_panorama.txt")) {
            feature_data_container.ReadSubPanoramaData(JoinPaths(workspace_path, "/sub_panorama.txt"));
        } else {
            // FIXME:
            std::cout << " Warning! Existing feature data do not contain sub_panorama data" << std::endl;
        }
        return;
    } else if (exist_feature_file) {
        return;
    }

    reader_options.single_camera = static_cast<bool>(param.GetArgument("single_camera", 0));
    reader_options.single_camera_per_folder = static_cast<bool>(param.GetArgument("single_camera_per_folder", 0));
    reader_options.fixed_camera = static_cast<bool>(param.GetArgument("fixed_camera", 0));

    reader_options.camera_model = param.GetArgument("camera_model", "SIMPLE_RADIAL");
    std::string camera_params = param.GetArgument("camera_params", "");
    if (!camera_params.empty()) {
        reader_options.camera_params = camera_params;
    }

    SiftExtractionOptions sift_extraction;
    sift_extraction.num_threads = param.GetArgument("feature_extraction_num_threads", -1);
    sift_extraction.use_gpu = static_cast<bool>(param.GetArgument("feature_extraction_use_gpu", 1));
    sift_extraction.peak_threshold = param.GetArgument("sift_peak_threshold", 0.00666666666667f);
    sift_extraction.min_num_features_customized = param.GetArgument("min_num_features_customized", 1024);
    sift_extraction.max_num_features_customized = param.GetArgument("max_num_features_customized", 4096);
    sift_extraction.max_image_size = param.GetArgument("max_image_size", 6144);

    sift_extraction.convert_to_perspective_image =
        static_cast<bool>(param.GetArgument("convert_to_perspective_image", 1));
    sift_extraction.perspective_image_count = static_cast<int>(param.GetArgument("perspective_image_count", 8));
    sift_extraction.perspective_image_width = static_cast<int>(param.GetArgument("perspective_image_width", 600));
    sift_extraction.perspective_image_height = static_cast<int>(param.GetArgument("perspective_image_height", 600));
    sift_extraction.fov_w = static_cast<int>(param.GetArgument("fov_w", 90));

    SiftFeatureExtractor feature_extractor(reader_options, sift_extraction, &feature_data_container);
    feature_extractor.Start();
    feature_extractor.Wait();

    fprintf(
        fs, "%s\n",
        StringPrintf("Feature Extraction Elapsed time: %.3f [minutes]", feature_extractor.GetTimer().ElapsedMinutes())
            .c_str());
    fflush(fs);

    bool write_feature = static_cast<bool>(param.GetArgument("write_feature", 0));
    bool write_binary = static_cast<bool>(param.GetArgument("write_binary", 1));
    if (write_feature) {
        if (write_binary) {
            feature_data_container.WriteImagesBinaryData(JoinPaths(workspace_path, "/features.bin"));
            feature_data_container.WriteCamerasBinaryData(JoinPaths(workspace_path, "/cameras.bin"));
            feature_data_container.WriteLocalCamerasBinaryData(JoinPaths(workspace_path, "/local_cameras.bin"));

            if (param.GetArgument("camera_model", "SIMPLE_RADIAL") == "SPHERICAL") {
                feature_data_container.WriteSubPanoramaBinaryData(JoinPaths(workspace_path, "/sub_panorama.bin"));
            }
        } else {
            feature_data_container.WriteImagesData(JoinPaths(workspace_path, "/features.txt"));
            feature_data_container.WriteCameras(JoinPaths(workspace_path, "/cameras.txt"));
            feature_data_container.WriteLocalCameras(JoinPaths(workspace_path, "/local_cameras.txt"));
            if (param.GetArgument("camera_model", "SIMPLE_RADIAL") == "SPHERICAL") {
                feature_data_container.ReadSubPanoramaData(JoinPaths(workspace_path, "/sub_panorama.txt"));
            }
        }
    }
}

void FeatureMatching(FeatureDataContainer& feature_data_container, SceneGraphContainer& scene_graph,
                     Configurator& param) {
    using namespace std::chrono;

    high_resolution_clock::time_point start_time = high_resolution_clock::now();

    std::string workspace_path = param.GetArgument("workspace_path", "");
    CHECK(!workspace_path.empty());
    bool load_scene_graph = false;
    if (boost::filesystem::exists(JoinPaths(workspace_path, "/scene_graph.bin"))) {
        scene_graph.ReadSceneGraphBinaryData(JoinPaths(workspace_path, "/scene_graph.bin"));
        load_scene_graph = true;
    } else if (boost::filesystem::exists(JoinPaths(workspace_path, "/scene_graph.txt"))) {
        scene_graph.ReadSceneGraphData(JoinPaths(workspace_path, "/scene_graph.txt"));
        load_scene_graph = true;
    }

    if (load_scene_graph) {
        EIGEN_STL_UMAP(image_t, class Image)& images = scene_graph.Images();
        EIGEN_STL_UMAP(camera_t, class Camera)& cameras = scene_graph.Cameras();

        std::vector<image_t> image_ids = feature_data_container.GetImageIds();

        for (const auto image_id : image_ids) {
            const Image& image = feature_data_container.GetImage(image_id);
            if (!scene_graph.CorrespondenceGraph()->ExistsImage(image_id)) {
                continue;
            }

            images[image_id] = image;

            const FeatureKeypoints& keypoints = feature_data_container.GetKeypoints(image_id);
            images[image_id].SetPoints2D(keypoints);

            const Camera& camera = feature_data_container.GetCamera(image.CameraId());

            if (!scene_graph.ExistsCamera(image.CameraId())) {
                cameras[image.CameraId()] = camera;
            }

            if (scene_graph.CorrespondenceGraph()->ExistsImage(image_id)) {
                images[image_id].SetNumObservations(
                    scene_graph.CorrespondenceGraph()->NumObservationsForImage(image_id));
                images[image_id].SetNumCorrespondences(
                    scene_graph.CorrespondenceGraph()->NumCorrespondencesForImage(image_id));
            } else {
                std::cout << "Do not contain ImageId = " << image_id << ", in the correspondence graph." << std::endl;
            }
        }

        scene_graph.CorrespondenceGraph()->Finalize();
        return;
    }

    FeatureMatchingOptions options;

    std::string method = param.GetArgument("matching_method", "exhaustive");
    if (method.compare("exhaustive") == 0) {
        options.method_ = FeatureMatchingOptions::MatchMethod::EXHAUSTIVE;
    } else if (method.compare("sequential") == 0) {
        options.method_ = FeatureMatchingOptions::MatchMethod::SEQUENTIAL;
    } else if (method.compare("vocabtree") == 0) {
        options.method_ = FeatureMatchingOptions::MatchMethod::VOCABTREE;
    } else if (method.compare("hybrid") == 0) {
        options.method_ = FeatureMatchingOptions::MatchMethod::HYBRID;

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
    } else {
        CHECK(false) << "invalid matching method";
    }

    options.vocabtree_matching_.vocab_tree_path = param.GetArgument("vocab_path", "");
    options.vocabtree_matching_.num_images = param.GetArgument("vocab_matching_num_images", 50);
    options.vocabtree_matching_.num_nearest_neighbors = param.GetArgument("vocab_matching_num_nearest_neighbors", 15);
    options.vocabtree_matching_.max_score_factor = param.GetArgument("max_score_factor", 0.0f);

    options.sequential_matching_.vocab_tree_path = param.GetArgument("vocab_path", "");

    options.sequential_matching_.loop_detection_num_threads = param.GetArgument("loop_detection_num_threads", -1);
    options.sequential_matching_.loop_detection = static_cast<bool>(param.GetArgument("loop_detection", 0));
    options.sequential_matching_.robust_loop_detection =
        static_cast<bool>(param.GetArgument("robust_loop_detection", 0));
    options.sequential_matching_.loop_detection_period = param.GetArgument("loop_detection_period", 1);
    options.sequential_matching_.loop_detection_num_images = param.GetArgument("loop_detection_num_images", 50);
    options.sequential_matching_.overlap = param.GetArgument("overlap", 10);
    options.sequential_matching_.loop_consistency_threshold = param.GetArgument("loop_consistency_threshold", 3);
    options.sequential_matching_.max_recent_score_factor = param.GetArgument("max_recent_score_factor", 0.4f);
    options.sequential_matching_.best_acc_score_factor = param.GetArgument("best_acc_score_factor", 0.2f);

    options.sequential_matching_.local_max_recent_score_factor =
        param.GetArgument("local_max_recent_score_factor", 0.8f);
    options.sequential_matching_.local_best_acc_score_factor = param.GetArgument("local_best_acc_score_factor", 0.75f);
    options.sequential_matching_.local_region_repetitive = param.GetArgument("local_region_repetitive", 100);

    options.sequential_matching_.local_triplet_checking = param.GetArgument("local_triplet_checking", 0);
    options.global_triplet_checking = param.GetArgument("global_triplet_checking", 0);
    options.local_invalid_theta_dis = param.GetArgument("local_invalid_theta_dis", 10.0f);
    options.global_median_invalid_theta_dis = param.GetArgument("global_median_invalid_theta_dis", 10.0f);
    options.global_mean_invalid_theta_dis = param.GetArgument("global_mean_invalid_theta_dis", 10.0f);
    options.ambiguous_triple_count = param.GetArgument("ambiguous_triple_count", 3);

    options.pair_matching_.num_threads = param.GetArgument("matching_num_threads", -1);
    options.pair_matching_.use_gpu = static_cast<bool>(param.GetArgument("matching_use_gpu", 1));
    options.pair_matching_.gpu_index = param.GetArgument("matching_gpu_index", "-1");
    options.pair_matching_.guided_matching = static_cast<bool>(param.GetArgument("guided_matching", 1));

    options.pair_matching_.multiple_models = static_cast<bool>(param.GetArgument("multiple_models", 0));

    options.pair_matching_.guided_matching_multi_homography =
        static_cast<bool>(param.GetArgument("guided_matching_multi_homography", 0));

    options.pair_matching_.max_num_matches = param.GetArgument("max_num_matches", 20000);
    options.pair_matching_.min_num_inliers = param.GetArgument("min_num_inliers", 15);

    if (param.GetArgument("camera_model", "SIMPLE_RADIAL") == "SPHERICAL") {
        options.pair_matching_.guided_matching = false;
    }

    MatchDataContainer match_data;
    FeatureMatcher matcher(options, &feature_data_container, &match_data, &scene_graph);
    std::cout << "matching ....." << std::endl;
    matcher.Run();
    std::cout << "matching done" << std::endl;
    std::cout << "build graph" << std::endl;
    matcher.BuildSceneGraph();
    std::cout << "build graph done" << std::endl;

    high_resolution_clock::time_point end_time = high_resolution_clock::now();

    fprintf(fs, "%s\n",
            StringPrintf("Feature Matching Elapsed time: %.3f [minutes]",
                         duration_cast<microseconds>(end_time - start_time).count() / 6e7)
                .c_str());
    fflush(fs);

    scene_graph.CorrespondenceGraph()->ExportToGraph(workspace_path + "/scene_graph.png");
    std::cout << "ExportToGraph done!" << std::endl;

    bool write_match = static_cast<bool>(param.GetArgument("write_match", 0));
    bool write_binary = static_cast<bool>(param.GetArgument("write_binary", 1));

    if (write_match) {
        if (write_binary) {
            scene_graph.WriteSceneGraphBinaryData(workspace_path + "/scene_graph.bin");
        } else {
            scene_graph.WriteSceneGraphData(workspace_path + "/scene_graph.txt");
        }
    }
}

int main(int argc, char* argv[]) {
    PrintHeading("SenseMap.  Copyright(c) 2019, SenseTime Group.");

    configuration_file_path = std::string(argv[1]);
    Configurator param;
    param.Load(configuration_file_path.c_str());

    std::string workspace_path = param.GetArgument("workspace_path", "");
    CHECK(!workspace_path.empty()) << "image path empty";

    std::string image_path = param.GetArgument("image_path", "");
    CHECK(!workspace_path.empty()) << "image path empty";

    if (!boost::filesystem::exists(workspace_path)) {
        CHECK(boost::filesystem::create_directories(workspace_path)) << "Create workspace failed";
    }

    std::string input_pose_path = param.GetArgument("input_pose_path", "");
    CHECK(!input_pose_path.empty()) << "input pose path empty";

    std::ifstream file_input_pose(input_pose_path);
    CHECK(file_input_pose.is_open());

    std::unordered_map<std::string, Eigen::Matrix3x4d> map_image_name_to_pose;
    std::unordered_map<std::string, double> map_image_name_to_focal_length;

    std::string image_name;
    while (file_input_pose >> image_name) {
        // Eigen::Matrix3x4d pose;
        // for (int i = 0; i < 3; ++i) {
        //     for (int j = 0; j < 4; ++j) {
        //         file_input_pose >> pose(i, j);
        //     }
        // }
        // double f;
        // file_input_pose >> f;
        // map_image_name_to_pose.emplace(image_name, pose);
        // map_image_name_to_focal_length.emplace(image_name, f);
        // double distortion;
        // file_input_pose >> distortion;

        Eigen::Vector4d qvec;
        Eigen::Vector3d tvec;

        for (int i = 0; i < 3; ++i) {
            file_input_pose >> tvec(i);
        }
        for (int i = 0; i < 4; ++i) {
            file_input_pose >> qvec(i);
        }

        Eigen::Matrix3x4d pose;
        pose = ComposeProjectionMatrix(qvec, tvec);
        map_image_name_to_pose.emplace(image_name, pose);

        double f;
        file_input_pose >> f;
        map_image_name_to_focal_length.emplace(image_name, f);

        double k1;
        file_input_pose >> k1;

        double k2;
        file_input_pose >> k2;

        int width;
        file_input_pose >> width;
        int height;
        file_input_pose >> height;
    }
    file_input_pose.close();

    fs = fopen((workspace_path + "/time.txt").c_str(), "w");

    auto reconstruction_manager = std::make_shared<ReconstructionManager>();
    auto scene_graph_container = std::make_shared<SceneGraphContainer>();
    auto feature_data_container = std::make_shared<FeatureDataContainer>();

    FeatureExtraction(*feature_data_container.get(), param);

    FeatureMatching(*feature_data_container.get(), *scene_graph_container.get(), param);

    std::shared_ptr<Reconstruction> reconstruction = std::make_shared<Reconstruction>();
    std::shared_ptr<IncrementalMapper> mapper = std::make_shared<IncrementalMapper>(scene_graph_container);
    mapper->BeginReconstruction(reconstruction);

    //////////////////////////////////////////////////////////////////////////////
    // Triangulation
    //////////////////////////////////////////////////////////////////////////////
    IndependentMapperOptions mapper_options;

    mapper_options.single_camera = static_cast<bool>(param.GetArgument("single_camera", 0));
    mapper_options.ba_global_use_pba = static_cast<bool>(param.GetArgument("ba_global_use_pba", 0));
    mapper_options.ba_refine_focal_length = static_cast<bool>(param.GetArgument("ba_refine_focal_length", 1));
    mapper_options.ba_refine_extra_params = static_cast<bool>(param.GetArgument("ba_refine_extra_params", 1));
    mapper_options.ba_refine_principal_point = static_cast<bool>(param.GetArgument("ba_refine_principal_point", 0));
    mapper_options.ba_global_pba_gpu_index = static_cast<int>(param.GetArgument("ba_global_pba_gpu_index", -1));
    mapper_options.ba_global_max_num_iterations =
        static_cast<int>(param.GetArgument("ba_global_max_num_iterations", 30));
    mapper_options.ba_global_max_refinements = static_cast<int>(param.GetArgument("ba_global_max_refinements", 2));

    mapper_options.ba_global_loss_function = param.GetArgument("ba_global_loss_function", "trival");

    mapper_options.ba_global_images_ratio = param.GetArgument("ba_global_images_ratio", 1.1f);

    mapper_options.ba_global_points_ratio = param.GetArgument("ba_global_points_ratio", 1.1f);

    mapper_options.batched_sfm = static_cast<bool>(param.GetArgument("batched_sfm", 0));

    mapper_options.local_ba_batched = static_cast<bool>(param.GetArgument("local_ba_batched", 1));

    mapper_options.ba_local_num_images = param.GetArgument("ba_local_num_images", 6);

    mapper_options.init_image_id1 = static_cast<int>(param.GetArgument("init_image_id1", -1));
    mapper_options.init_image_id2 = static_cast<int>(param.GetArgument("init_image_id2", -1));
    mapper_options.init_from_uncertainty = static_cast<bool>(param.GetArgument("init_from_uncertainty", 1));
    mapper_options.init_min_num_inliers = param.GetArgument("init_min_num_inliers", 200);
    mapper_options.init_min_tri_angle = param.GetArgument("init_min_tri_angle", 12.0f);
    mapper_options.filter_max_reproj_error = param.GetArgument("filter_max_reproj_error", 4.0f);
    mapper_options.merge_max_reproj_error =
        param.GetArgument("merge_max_reproj_error", 4.0f);
    mapper_options.complete_max_reproj_error =
        param.GetArgument("complete_max_reproj_error", 4.0f);


    mapper_options.min_tri_angle = param.GetArgument("min_tri_angle", 1.5f);

    mapper_options.abs_pose_min_num_inliers = param.GetArgument("abs_pose_min_num_inliers", 30);

    mapper_options.min_inlier_ratio_to_best_pose = param.GetArgument("min_inlier_ratio_to_best_pose", 0.7f);

    mapper_options.min_inlier_ratio_verification_with_prior_pose =
        param.GetArgument("min_inlier_ratio_verification_with_prior_pose", 0.7f);

    mapper_options.num_images_for_self_calibration = param.GetArgument("num_images_for_self_calibration", 200);

    mapper_options.extract_keyframe = static_cast<bool>(param.GetArgument("extract_keyframe", 0));
    mapper_options.register_nonkeyframe = static_cast<bool>(param.GetArgument("register_nonkeyframe", 0));

    mapper_options.num_first_force_be_keyframe = static_cast<int>(param.GetArgument("num_first_force_be_keyframe", 10));
    mapper_options.optim_inner_cluster = static_cast<bool>(param.GetArgument("optim_inner_cluster", 0));
    mapper_options.robust_camera_pose_estimate = static_cast<bool>(param.GetArgument("robust_camera_pose_estimate", 0));
    mapper_options.consecutive_camera_pose_top_k =
        static_cast<int>(param.GetArgument("consecutive_camera_pose_top_k", 2));
    mapper_options.consecutive_neighbor_ori = static_cast<int>(param.GetArgument("consecutive_neighbor_ori", 2));
    mapper_options.consecutive_neighbor_t = static_cast<int>(param.GetArgument("consecutive_neighbor_t", 1));
    mapper_options.consecutive_camera_pose_orientation = param.GetArgument("consecutive_camera_pose_orientation", 5.0f);
    mapper_options.consecutive_camera_pose_t = param.GetArgument("consecutive_camera_pose_t", 20.0f);

    mapper_options.min_inlier_ratio_to_best_model = param.GetArgument("min_inlier_ratio_to_best_model", 0.8f);
    mapper_options.local_region_repetitive = param.GetArgument("local_region_repetitive", 100);

    mapper_options.num_fix_camera_first = static_cast<int>(param.GetArgument("num_fix_camera_first", 5));

    bool single_camera = static_cast<bool>(param.GetArgument("single_camera", 0));
    if (!single_camera) {
        std::cout << "Warning! Can't fix camera because single camera is disable, forcing the variable to be zero"
                  << std::endl;
        mapper_options.num_fix_camera_first = 0;
    }

    mapper_options.max_triangulation_angle_degrees =
        static_cast<double>(param.GetArgument("max_triangulation_angle_degrees", 30.0f));

    mapper_options.min_visible_map_point_kf = static_cast<int>(param.GetArgument("min_visible_map_point_kf", 300));
    mapper_options.min_pose_inlier_kf = static_cast<int>(param.GetArgument("min_pose_inlier_kf", 200));

    mapper_options.avg_min_dist_kf_factor = static_cast<double>(param.GetArgument("avg_min_dist_kf_factor", 1.0f));
    mapper_options.mean_max_disparity_kf = static_cast<double>(param.GetArgument("mean_max_disparity_kf", 20.0f));
    mapper_options.abs_diff_kf = static_cast<int>(param.GetArgument("abs_diff_kf", 10));

    mapper_options.debug_info = static_cast<bool>(param.GetArgument("debug_info", 0));

    mapper_options.write_binary_model = static_cast<bool>(param.GetArgument("write_binary", 1));

    int min_track_length = static_cast<int>(param.GetArgument("min_track_length", 3));

    const auto tri_options = mapper_options.Triangulation();

    const auto& rec_image_ids = reconstruction->Images();

    using namespace std::chrono;
    high_resolution_clock::time_point start_time = high_resolution_clock::now();

    int triangulated_image_count = 1;
    for (const auto& rec_image : rec_image_ids) {
        const image_t image_id = rec_image.first;

        // Check the image is in the scene graph or not
        if (!scene_graph_container->CorrespondenceGraph()->ExistsImage(image_id)) {
            continue;
        }

        auto& image = reconstruction->Image(image_id);
        std::string image_name = image.Name();
        if (map_image_name_to_pose.find(image_name) == map_image_name_to_pose.end()) {
            // no prior pose, skipped
            continue;
        }
        Eigen::Matrix3x4d pose = map_image_name_to_pose.at(image_name);
        image.Qvec() = RotationMatrixToQuaternion(pose.block<3, 3>(0, 0));
        image.Tvec() = pose.block<3, 1>(0, 3);

        reconstruction->RegisterImage(image_id);
        auto& camera = reconstruction->Camera(image.CameraId());
        CHECK(map_image_name_to_focal_length.find(image_name) != map_image_name_to_focal_length.end());

        camera.SetFocalLength(map_image_name_to_focal_length.at(image_name));

        PrintHeading1(StringPrintf("Triangulating image #%d (%d)", image_id, triangulated_image_count++));

        const size_t num_existing_points3D = image.NumMapPoints();
        std::cout << "  => Image sees " << num_existing_points3D << " / " << image.NumObservations() << " points"
                  << std::endl;

        mapper->TriangulateImage(tri_options, image_id);

        std::cout << "  => Triangulated " << (image.NumMapPoints() - num_existing_points3D) << " points" << std::endl;

        auto ba_options = mapper_options.LocalBundleAdjustment();

        ba_options.refine_focal_length = false;
        ba_options.refine_principal_point = false;
        ba_options.refine_extra_params = false;
        ba_options.refine_extrinsics = false;

        for (int i = 0; i < mapper_options.ba_local_max_refinements; ++i) {
            const auto report =
                mapper->AdjustLocalBundle(mapper_options.IncrementalMapperOptions(), ba_options, mapper_options.Triangulation(), image_id,
                                          mapper->GetModifiedMapPoints());
            std::cout << "  => Merged observations: " << report.num_merged_observations << std::endl;
            std::cout << "  => Completed observations: " << report.num_completed_observations << std::endl;
            std::cout << "  => Filtered observations: " << report.num_filtered_observations << std::endl;

            const double changed = (report.num_merged_observations + report.num_completed_observations +
                                    report.num_filtered_observations) /
                                   static_cast<double>(report.num_adjusted_observations);
            std::cout << StringPrintf("  => Changed observations: %.6f", changed) << std::endl;
            if (changed < mapper_options.ba_local_max_refinement_change) {
                break;
            }
        }
        mapper->ClearModifiedMapPoints();
    }

    //////////////////////////////////////////////////////////////////////////////
    // Retriangulation
    //////////////////////////////////////////////////////////////////////////////

    PrintHeading1("Retriangulation");

    CompleteAndMergeTracks(mapper_options, mapper);

    //////////////////////////////////////////////////////////////////////////////
    // Bundle adjustment
    //////////////////////////////////////////////////////////////////////////////

    auto ba_options = mapper_options.GlobalBundleAdjustment();
    ba_options.refine_focal_length = false;
    ba_options.refine_principal_point = false;
    ba_options.refine_extra_params = false;
    ba_options.refine_extrinsics = false;

    // Configure bundle adjustment.
    BundleAdjustmentConfig ba_config;
    std::vector<image_t> reg_image_ids = reconstruction->RegisterImageIds();

    for (size_t i = 0; i < reg_image_ids.size(); ++i) {
        const image_t image_id = reg_image_ids[i];
        if (!scene_graph_container->CorrespondenceGraph()->ExistsImage(image_id)) {
            continue;
        }
        ba_config.AddImage(image_id);
    }

    for (int i = 0; i < mapper_options.ba_global_max_refinements; ++i) {
        // Avoid degeneracies in bundle adjustment.
        reconstruction->FilterObservationsWithNegativeDepth();

        const size_t num_observations = reconstruction->ComputeNumObservations();

        PrintHeading1("Bundle adjustment");
        BundleAdjuster bundle_adjuster(ba_options, ba_config);
        CHECK(bundle_adjuster.Solve(reconstruction.get()));

        size_t num_changed_observations = 0;
        num_changed_observations += CompleteAndMergeTracks(mapper_options, mapper);
        num_changed_observations += FilterPoints(mapper_options, mapper, min_track_length);
        const double changed = static_cast<double>(num_changed_observations) / num_observations;
        std::cout << StringPrintf("  => Changed observations: %.6f", changed) << std::endl;
        if (changed < mapper_options.ba_global_max_refinement_change) {
            break;
        }
    }

    for (size_t i = 0; i < reg_image_ids.size(); ++i) {
        if (mapper_options.extract_colors) {
            ExtractColors(image_path, reg_image_ids[i], reconstruction);
        }
    }

    const bool kDiscardReconstruction = false;
    mapper->EndReconstruction(kDiscardReconstruction);

    std::string rec_path = StringPrintf("%s/%d", workspace_path.c_str(), 0);
    if (boost::filesystem::exists(rec_path)) {
        boost::filesystem::remove_all(rec_path);
    }
    boost::filesystem::create_directories(rec_path);
    reconstruction->WriteReconstruction(rec_path, true);

    high_resolution_clock::time_point end_time = high_resolution_clock::now();

    fprintf(fs, "%s\n",
            StringPrintf("Feature Matching Elapsed time: %.3f [minutes]",
                         duration_cast<microseconds>(end_time - start_time).count() / 6e7)
                .c_str());
    fflush(fs);

    return 0;
}