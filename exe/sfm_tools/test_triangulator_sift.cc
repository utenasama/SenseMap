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

#include "../Configurator_yaml.h"

#include <dirent.h>
#include <sys/stat.h>
#include <fstream>
#include "base/pose.h"

using namespace sensemap;

std::string configuration_file_path;
FILE *fs;

bool dirExists(const std::string &dirName_in) { return access(dirName_in.c_str(), F_OK) == 0; }

void FeatureExtraction(FeatureDataContainer &feature_data_container,
                       std::unordered_map<std::string, image_t> image_name_id_map, Configurator &param) {
    std::string workspace_path = param.GetArgument("workspace_path", "");
    CHECK(!workspace_path.empty());

    ImageReaderOptions reader_options;
    reader_options.image_path = param.GetArgument("image_path", "");

    bool exist_feature_file = false;
    if (boost::filesystem::exists(JoinPaths(workspace_path, "/cameras.bin")) &&
        boost::filesystem::exists(JoinPaths(workspace_path, "/features.bin"))) {
        feature_data_container.ReadCamerasBinaryData(JoinPaths(workspace_path, "/cameras.bin"));
        feature_data_container.ReadImagesBinaryData(JoinPaths(workspace_path, "/features.bin"));
        exist_feature_file = true;
    } else if (boost::filesystem::exists(JoinPaths(workspace_path, "/cameras.txt")) &&
               boost::filesystem::exists(JoinPaths(workspace_path, "/features.txt"))) {
        feature_data_container.ReadCameras(JoinPaths(workspace_path, "/cameras.txt"));
        feature_data_container.ReadImagesData(JoinPaths(workspace_path, "/features.txt"));
        exist_feature_file = true;
    } else if (boost::filesystem::exists(JoinPaths(workspace_path, "/cameras.txt")) &&
               boost::filesystem::exists(JoinPaths(workspace_path, "/features.bin"))) {
        feature_data_container.ReadCameras(JoinPaths(workspace_path, "/cameras.txt"));
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
    reader_options.image_name_id_map = image_name_id_map;

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
            StringPrintf("Feature Extraction Elapsed time: %.3f [minutes]",
                         feature_extractor.GetTimer().ElapsedMinutes())
                    .c_str());
    fflush(fs);
}

void FeatureMatching(FeatureDataContainer &feature_data_container, SceneGraphContainer &scene_graph,
                     std::vector<std::pair<image_t, image_t>> prior_image_pairs, Configurator &param) {
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
        EIGEN_STL_UMAP(image_t, class Image) &images = scene_graph.Images();
        EIGEN_STL_UMAP(camera_t, class Camera) &cameras = scene_graph.Cameras();

        std::vector<image_t> image_ids = feature_data_container.GetImageIds();

        for (const auto image_id : image_ids) {
            const Image &image = feature_data_container.GetImage(image_id);
            if (!scene_graph.CorrespondenceGraph()->ExistsImage(image_id)) {
                continue;
            }

            images[image_id] = image;

            const FeatureKeypoints &keypoints = feature_data_container.GetKeypoints(image_id);
            images[image_id].SetPoints2D(keypoints);

            const Camera &camera = feature_data_container.GetCamera(image.CameraId());

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
    options.method_ = FeatureMatchingOptions::MatchMethod::SEQUENTIAL;
    options.have_prior_image_pairs_ = true;
    options.prior_image_pairs_ = prior_image_pairs;

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

int main(int argc, char *argv[]) {
    PrintHeading("SenseMap.  Copyright(c) 2019, SenseTime Group.");

    configuration_file_path = std::string(argv[1]);
    Configurator param;
    param.Load(configuration_file_path.c_str());

    std::string workspace_path = param.GetArgument("workspace_path", "");
    CHECK(!workspace_path.empty()) << "image path empty";

    std::string image_path = param.GetArgument("image_path", "");
    CHECK(!image_path.empty()) << "image path empty";

    std::string old_workspace_path = param.GetArgument("old_workspace_path", "");
    CHECK(!old_workspace_path.empty()) << "old workspace path empty";

    if (!boost::filesystem::exists(workspace_path)) {
        CHECK(boost::filesystem::create_directories(workspace_path)) << "Create workspace failed";
    }

    fs = fopen((workspace_path + "/time.txt").c_str(), "w");

    auto reconstruction_manager = std::make_shared<ReconstructionManager>();
    auto scene_graph_container = std::make_shared<SceneGraphContainer>();
    auto feature_data_container = std::make_shared<FeatureDataContainer>();

    // Store the old feature container
    auto old_scene_graph_container = std::make_shared<SceneGraphContainer>();
    auto old_feature_data_container = std::make_shared<FeatureDataContainer>();

    // Read old feature
    old_feature_data_container->ReadImagesBinaryData(old_workspace_path + "/features.bin");
    old_feature_data_container->ReadCamerasBinaryData(old_workspace_path + "/cameras.bin");
    // Read old scene graph
    std::vector<std::pair<image_t, image_t>> old_iamge_pairs = old_scene_graph_container->CorrespondenceGraph()->ReadCorrespondencePairBinaryData(
            old_workspace_path + "/scene_graph.bin");



    // Get image name id map from the old feature data container
    auto old_image_ids = old_feature_data_container->GetImageIds();
    std::unordered_map<std::string, image_t> image_name_id_map;
    for (auto old_image_id : old_image_ids) {
        std::string old_image_name = old_feature_data_container->GetImage(old_image_id).Name();
        image_name_id_map[old_image_name] = old_image_id;
    }

    // Featrue Extraction
    FeatureExtraction(*feature_data_container.get(), image_name_id_map, param);

    // Modified all the camera id of the new feature_data_container to the old feature_data container
//    for (auto old_image_id : old_image_ids) {
//        auto old_camera_id = old_feature_data_container->GetImage(old_image_id).CameraId();
//        feature_data_container->GetImage(feature_data_container->GetImage(old_image_id).Name())
//                .SetCameraId(old_camera_id);
//    }
//
//    for (auto old_image_id : old_image_ids) {
//        auto old_camera_id = old_feature_data_container->GetImage(old_image_id).CameraId();
//        feature_data_container->GetImage(feature_data_container->GetImage(old_image_id).Name())
//                .SetCameraId(old_camera_id);
//    }

    // TODO: Modified the feature data container camera to the old one
    // for (auto old_image_id : old_image_ids) {

    // }


    // save feature_data_container 
    bool write_feature = static_cast<bool>(param.GetArgument("write_feature", 0));
    bool write_binary = static_cast<bool>(param.GetArgument("write_binary", 1));
    if (write_feature) {
        if (write_binary) {
            feature_data_container->WriteImagesBinaryData(JoinPaths(workspace_path, "/features.bin"));
            feature_data_container->WriteCamerasBinaryData(JoinPaths(workspace_path, "/cameras.bin"));

            if (param.GetArgument("camera_model", "SIMPLE_RADIAL") == "SPHERICAL") {
                feature_data_container->WriteSubPanoramaBinaryData(JoinPaths(workspace_path, "/sub_panorama.bin"));
            }
        } else {
            feature_data_container->WriteImagesData(JoinPaths(workspace_path, "/features.txt"));
            feature_data_container->WriteCameras(JoinPaths(workspace_path, "/cameras.txt"));
            if (param.GetArgument("camera_model", "SIMPLE_RADIAL") == "SPHERICAL") {
                feature_data_container->ReadSubPanoramaData(JoinPaths(workspace_path, "/sub_panorama.txt"));
            }
        }
    }


    // Get image match pair from the old scene graph container
    std::vector<std::pair<image_t, image_t>> prior_image_pairs;
    for (auto old_iamge_pair : old_iamge_pairs) {
        if (!feature_data_container->ExistImage(old_iamge_pair.second)) {
            std::cout << "Image number do not exist = " << old_iamge_pair.second << std::endl;
            continue;
        }
        if (!feature_data_container->ExistImage(old_iamge_pair.first)) {
            std::cout << "Image number do not exist = " << old_iamge_pair.first << std::endl;
            continue;
        }
        prior_image_pairs.emplace_back(
                std::make_pair(old_iamge_pair.first, old_iamge_pair.second));
    }



    // Feature Matching
    FeatureMatching(*feature_data_container.get(), *scene_graph_container.get(), prior_image_pairs, param);

    std::shared_ptr<Reconstruction> reconstruction = std::make_shared<Reconstruction>();
    std::shared_ptr<IncrementalMapper> mapper = std::make_shared<IncrementalMapper>(scene_graph_container);

    // Check the old reconstruction exist or not
    if (!ExistsDir(old_workspace_path + "/0/")) {
        std::cerr << "ERROR: Input reconstruction path is not a directory  " << old_workspace_path + "/0/" << std::endl;
        return 0;
    }

    // Create output path
    // if (!boost::filesystem::exists(workspace_path)) {
    //     boost::filesystem::create_directories(workspace_path);
    // }

    PrintHeading1("Loading model");

    Timer timer;
    timer.Start();
    std::cout << "model path : " << old_workspace_path + "/0/" << std::endl;
    reconstruction->ReadReconstruction(old_workspace_path + "/0/");

    // if (clear_points) {
    std::cout << "Clear points" << std::endl;
    // Delete all the 2D point and 3D point
    reconstruction->DeleteAllPoints2DAndPoints3D();
    // Update image id using image name check
    reconstruction->TranscribeImageIdsToDatabase(feature_data_container);
    // }

    std::cout << std::endl;
    timer.PrintMinutes();
    std::cout << std::endl;

    CHECK_GE(reconstruction->NumImages(), 2) << "Need at least two images for triangulation";

    mapper->BeginReconstruction(reconstruction);

    //////////////////////////////////////////////////////////////////////////////
    // Triangulation
    //////////////////////////////////////////////////////////////////////////////
    IndependentMapperOptions mapper_options;

    const auto tri_options = mapper_options.Triangulation();

    const auto &reg_image_ids = reconstruction->RegisterImageIds();

    for (size_t i = 0; i < reg_image_ids.size(); ++i) {
        const image_t image_id = reg_image_ids[i];

        // Check the image is in the scene graph or not
        if (!scene_graph_container->CorrespondenceGraph()->ExistsImage(image_id)) {
            continue;
        }

        const auto &image = reconstruction->Image(image_id);

        PrintHeading1(StringPrintf("Triangulating image #%d (%d)", image_id, i));

        const size_t num_existing_points3D = image.NumMapPoints();

        std::cout << "  => Image sees " << num_existing_points3D << " / " << image.NumObservations() << " points"
                  << std::endl;

        mapper->TriangulateImage(tri_options, image_id);

        std::cout << "  => Triangulated " << (image.NumMapPoints() - num_existing_points3D) << " points" << std::endl;
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
    ba_options.loss_function_type = BundleAdjustmentOptions::LossFunctionType::Huber;

    // Configure bundle adjustment.
    BundleAdjustmentConfig ba_config;

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
        num_changed_observations += FilterPoints(mapper_options, mapper);
        const double changed = static_cast<double>(num_changed_observations) / num_observations;
        std::cout << StringPrintf("  => Changed observations: %.6f", changed) << std::endl;
        if (changed < mapper_options.ba_global_max_refinement_change) {
            break;
        }
    }

    if (static_cast<bool>(param.GetArgument("extract_color", 0))) {
        PrintHeading1("Extracting colors");
        reconstruction->ExtractColorsForAllImages(image_path);
    }

    const bool kDiscardReconstruction = false;
    mapper->EndReconstruction(kDiscardReconstruction);

    std::string rec_path = workspace_path + "/0/";
    if (boost::filesystem::exists(rec_path)) {
        boost::filesystem::remove_all(rec_path);
    }
    boost::filesystem::create_directories(rec_path);

    reconstruction->WriteReconstruction(rec_path, true);

    return 0;
}