// Copyright (c) 2019, SenseTime Group.
// All rights reserved.

#include <dirent.h>
#include <sys/stat.h>

#include <boost/filesystem/path.hpp>
#include <opencv2/opencv.hpp>

#include "../Configurator_yaml.h"
#include "../option_parsing.h"
#include "../system_io.h"
#include "base/camera.h"
#include "base/image.h"
#include "base/image_reader.h"
#include "base/reconstruction_manager.h"
#include "container/feature_data_container.h"
#include "controllers/cluster_mapper_controller.h"
#include "estimators/scale_gravity.h"
#include "feature/extraction.h"
#include "feature/feature_matcher.h"
#include "feature/utils.h"
#include "graph/correspondence_graph.h"
#include "util/misc.h"
#include "util/timer.h"
#include <fstream>
#include "base/version.h"

using namespace sensemap;

std::string configuration_file_path;

FILE *fs;
OptionParser option_parser;

bool dirExists(const std::string &dirName_in) { return access(dirName_in.c_str(), F_OK) == 0; }

void PrintReconSummary(const std::string &flog_name, const size_t num_total_image,
                       const std::shared_ptr<ReconstructionManager> &reconstruction_manager) {
    if (reconstruction_manager->Size() == 0) {
        return;
    }
    std::shared_ptr<Reconstruction> best_rec;
    for (int i = 0; i < reconstruction_manager->Size(); ++i) {
        const std::shared_ptr<Reconstruction> &rec = reconstruction_manager->Get(i);
        if (!best_rec || best_rec->NumRegisterImages() < rec->NumRegisterImages()) {
            best_rec = rec;
        }
    }
    FILE *fp = fopen(flog_name.c_str(), "w");

    size_t num_reg_image = best_rec->NumRegisterImages();
    fprintf(fp, "Registered / Total: %zu / %zu\n", num_reg_image, num_total_image);
    fprintf(fp, "Mean Track Length: %f\n", best_rec->ComputeMeanTrackLength());
    fprintf(fp, "Mean Reprojection Error: %f\n", best_rec->ComputeMeanReprojectionError());
    fprintf(fp, "Mean Observation Per Register Image: %f\n", best_rec->ComputeMeanObservationsPerRegImage());

    fclose(fp);
}

void FeatureExtraction(FeatureDataContainer &feature_data_container, Configurator &param,
                       const std::vector<double> &intrinsics = std::vector<double>()) {
    std::string workspace_path = param.GetArgument("sfm_export_path", "");
    CHECK(!workspace_path.empty());

    Timer timer;
    timer.Start();
    ImageReaderOptions reader_options;
    option_parser.GetImageReaderOptions(reader_options, param);

    std::vector<double> camera_params = intrinsics;
    std::string camera_params_string = std::to_string(camera_params[0]);
    camera_params_string = camera_params_string + ", " + std::to_string(camera_params[2]);
    camera_params_string = camera_params_string + ", " + std::to_string(camera_params[3]);
    camera_params_string = camera_params_string + ", 0, 0";
    reader_options.camera_params = camera_params_string;
    reader_options.camera_model = "RADIAL";

    std::cout << "Input camera params:" << reader_options.camera_params << std::endl;

    bool camera_rig = (reader_options.num_local_cameras > 1);
    bool with_depth = reader_options.with_depth;

    // Check is map update mode
    bool exist_map_update_feature = false;
    if (static_cast<bool>(param.GetArgument("map_update", 0)) &&
        boost::filesystem::exists(JoinPaths(workspace_path, "/map_update/"))) {
        workspace_path = workspace_path + "/map_update/";
        exist_map_update_feature = true;
    }

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
    }

    // If the camera model is spherical or camera rig
    if (exist_feature_file) {
        if (boost::filesystem::exists(JoinPaths(workspace_path, "/sub_panorama.bin"))) {
            feature_data_container.ReadSubPanoramaBinaryData(JoinPaths(workspace_path, "/sub_panorama.bin"));
        } else {
            std::cout << " Warning! Existing feature data do not contain sub_panorama data" << std::endl;
        }
    }

    // two camera rig
    if (exist_feature_file) {
        if (boost::filesystem::exists(JoinPaths(workspace_path, "/piece_indices.bin"))) {
            feature_data_container.ReadPieceIndicesBinaryData(JoinPaths(workspace_path, "/piece_indices.bin"));
        } else {
            std::cout << " Warning! Existing feature data do not contain piece_indices data" << std::endl;
        }
    }

    std::cout << std::endl;
    timer.PrintMinutes();
    std::cout << std::endl;
    fprintf(fs, "%s\n",
            StringPrintf("Feature Extraction Read Elapsed time: %.3f [minutes]", timer.ElapsedMinutes()).c_str());
    fflush(fs);

    if (exist_map_update_feature && exist_feature_file) {
        return;
    }

    if (static_cast<bool>(param.GetArgument("map_update", 0))) {
        if (!exist_feature_file) {
            std::cout << "ERROR! Map Update Need previous feature data... " << std::endl;
            exit(-1);
        }

        boost::filesystem::create_directories(JoinPaths(workspace_path, "/map_update/"));
        workspace_path = workspace_path + "/map_update/";

        const auto &image_ids = feature_data_container.GetImageIds();

        // Set all the exist image label as 0
        for (const auto image_id : image_ids) {
            std::string image_name = feature_data_container.GetImage(image_id).Name();
            feature_data_container.GetImage(image_name).SetLabelId(0);
        }
    }

    SiftExtractionOptions sift_extraction;
    option_parser.GetFeatureExtractionOptions(sift_extraction, param);

    image_t start_image_id = 0;
    camera_t start_camera_id = 0;
    if (static_cast<bool>(param.GetArgument("map_update", 0))) {
        for (const auto old_image_id : feature_data_container.GetImageIds()) {
            if (start_image_id < old_image_id) {
                start_image_id = old_image_id;
            }
        }
        ++start_image_id;
        start_camera_id = feature_data_container.NumCamera();
    }

    SiftFeatureExtractor feature_extractor(reader_options, sift_extraction, &feature_data_container, start_image_id,
                                           start_camera_id);

    feature_extractor.Start();
    feature_extractor.Wait();
    std::cout << "Feature Extraction total image: " << feature_data_container.GetImageIds().size() << std::endl;
    std::cout << "Feature Extraction new image: " << feature_data_container.GetNewImageIds().size() << std::endl;

    fprintf(
        fs, "%s\n",
        StringPrintf("Feature Extraction Elapsed time: %.3f [minutes]", feature_extractor.GetTimer().ElapsedMinutes())
            .c_str());
    fflush(fs);

    const std::vector<image_t> &new_image_ids = feature_data_container.GetNewImageIds();

    typedef FeatureMatchingOptions::RetrieveType RetrieveType;
    RetrieveType retrieve_type = (RetrieveType)param.GetArgument("retrieve_type", 0);
    if (retrieve_type != RetrieveType::SIFT) {
        // compress descriptors using the trained PCA matrix
        std::cout << "Compressing descriptors " << std::endl;
        int compressed_feature_dimension = static_cast<int>(param.GetArgument("compressed_feature_dimension", 128));

        if (retrieve_type == RetrieveType::SIFT && compressed_feature_dimension != 128) {
            compressed_feature_dimension = 128;
            std::cout << StringPrintf("Warning! Feature dimension is not customized for SIFT\n");
        }

        CHECK(compressed_feature_dimension == 128 || compressed_feature_dimension == 64 ||
            compressed_feature_dimension == 32);

        Eigen::Matrix<double, 128, 128> pca_matrix;
        Eigen::Matrix<double, 128, 1> embedding_thresholds;
        std::string pca_matrix_path = param.GetArgument("pca_matrix_path", "");
        ReadPcaProjectionMatrix(pca_matrix, embedding_thresholds, pca_matrix_path);

        timer.Start();
        for (int i = 0; i < new_image_ids.size(); ++i) {
            image_t current_id = new_image_ids[i];
            auto &descriptors = feature_data_container.GetDescriptors(current_id);
            auto &compressed_descriptors = feature_data_container.GetCompressedDescriptors(current_id);
            
            if(compressed_feature_dimension == 128){
                compressed_descriptors = descriptors;    
            }
            else{
                CompressFeatureDescriptors(descriptors, compressed_descriptors, pca_matrix, embedding_thresholds,
                                        compressed_feature_dimension);
            }
            descriptors.resize(0, 0);
        }
        std::cout << StringPrintf("Compressing descriptors in %.3f min", timer.ElapsedMinutes()) << std::endl;
        fprintf(fs, "%s\n", StringPrintf("Compressing descriptors in %.3f [minutes]", timer.ElapsedMinutes()).c_str());
        fflush(fs);
    } else {
        for (int i = 0; i < new_image_ids.size(); ++i) {
            image_t current_id = new_image_ids[i];
            auto &descriptors = feature_data_container.GetDescriptors(current_id);
            auto &compressed_descriptors = feature_data_container.GetCompressedDescriptors(current_id);
            std::swap(descriptors, compressed_descriptors);
        }
    }

    bool write_feature = static_cast<bool>(param.GetArgument("write_feature", 1));

    timer.Restart();
    if (write_feature) {
        feature_data_container.WriteImagesBinaryData(JoinPaths(workspace_path, "/features.bin"));
        feature_data_container.WriteCamerasBinaryData(JoinPaths(workspace_path, "/cameras.bin"));
        feature_data_container.WriteLocalCamerasBinaryData(JoinPaths(workspace_path, "/local_cameras.bin"));
        if (param.GetArgument("camera_model", "SIMPLE_RADIAL") == "SPHERICAL" || reader_options.num_local_cameras > 1) {
            feature_data_container.WriteSubPanoramaBinaryData(JoinPaths(workspace_path, "/sub_panorama.bin"));
        }
        if (reader_options.num_local_cameras == 2 && reader_options.camera_model == "OPENCV_FISHEYE") {
            feature_data_container.WritePieceIndicesBinaryData(JoinPaths(workspace_path, "/piece_indices.bin"));
        }
    }
    std::cout << std::endl;
    timer.PrintMinutes();
    std::cout << std::endl;
    fprintf(fs, "%s\n",
            StringPrintf("Feature Extraction Write Elapsed time: %.3f [minutes]", timer.ElapsedMinutes()).c_str());
    fflush(fs);
}

void GlobalFeatureExtraction(FeatureDataContainer &feature_data_container, Configurator &param) {
    std::string sfm_export_path = param.GetArgument("sfm_export_path", "");
    CHECK(!sfm_export_path.empty());

    std::string new_workspace_path = sfm_export_path + "/map_update";
    if (boost::filesystem::exists(JoinPaths(new_workspace_path, "/vlad_vectors.bin"))) {
        feature_data_container.ReadGlobalFeaturesBinaryData(new_workspace_path + "/vlad_vectors.bin");
        std::cout << "New global feature already exists, skip feature extraction" << std::endl;
        return;
    }

    
    bool exist_old_global_features = false;
    if (boost::filesystem::exists(JoinPaths(sfm_export_path, "/vlad_vectors.bin"))) {
        feature_data_container.ReadGlobalFeaturesBinaryData(sfm_export_path + "/vlad_vectors.bin");
        std::cout << "Old global feature already exists" << std::endl;
        exist_old_global_features = true;
    }

    OptionParser option_parser;
    GlobalFeatureExtractionOptions options;
    option_parser.GetGlobalFeatureExtractionOptions(options, param);

    GlobalFeatureExtractor global_feature_extractor(options, &feature_data_container, exist_old_global_features);
    global_feature_extractor.Run();
    feature_data_container.WriteGlobalFeaturesBinaryData(new_workspace_path + "/vlad_vectors.bin");

    return;
}

void FeatureMatching(FeatureDataContainer &feature_data_container, SceneGraphContainer &scene_graph,
                     Configurator &param) {
    Timer timer;
    timer.Start();

    std::string workspace_path = param.GetArgument("sfm_export_path", "");

    // Check is map update mode
    bool exist_map_update_correspondence = false;
    if (static_cast<bool>(param.GetArgument("map_update", 0)) &&
        boost::filesystem::exists(JoinPaths(workspace_path, "/map_update/"))) {
        if (boost::filesystem::exists(JoinPaths(workspace_path, "/map_update/scene_graph.bin"))) {
            workspace_path = workspace_path + "/map_update/";
            exist_map_update_correspondence = true;
        } else {
            std::cout << "Not exist correspondence ... " << std::endl;
        }
    }

    CHECK(!workspace_path.empty());
    bool load_scene_graph = false;
    if (boost::filesystem::exists(JoinPaths(workspace_path, "/scene_graph.bin"))) {
        scene_graph.ReadSceneGraphBinaryData(JoinPaths(workspace_path, "/scene_graph.bin"));
        load_scene_graph = true;
    }

    FeatureMatchingOptions options;
    option_parser.GetFeatureMatchingOptions(options, param);

    ImageReaderOptions reader_options;
    option_parser.GetImageReaderOptions(reader_options, param);

    if (load_scene_graph) {
        EIGEN_STL_UMAP(image_t, class Image) &images = scene_graph.Images();
        EIGEN_STL_UMAP(camera_t, class Camera) &cameras = scene_graph.Cameras();

        std::vector<image_t> image_ids = feature_data_container.GetImageIds();

        for (const auto image_id : image_ids) {
            const Image &image = feature_data_container.GetImage(image_id);
            const Camera &camera = feature_data_container.GetCamera(image.CameraId());
            if (!scene_graph.CorrespondenceGraph()->ExistsImage(image_id)) {
                continue;
            }

            images[image_id] = image;

            const FeatureKeypoints &keypoints = feature_data_container.GetKeypoints(image_id);
            images[image_id].SetPoints2D(keypoints);

            const PanoramaIndexs &panorama_indices = feature_data_container.GetPanoramaIndexs(image_id);
            std::vector<uint32_t> local_image_indices(keypoints.size());
            for (size_t i = 0; i < keypoints.size(); ++i) {
                if (panorama_indices.size() == 0 && camera.NumLocalCameras() == 1) {
                    local_image_indices[i] = image_id;
                } else {
                    local_image_indices[i] = panorama_indices[i].sub_image_id;
                }
            }
            images[image_id].SetLocalImageIndices(local_image_indices);

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
        if (!static_cast<bool>(param.GetArgument("map_update", 0)) || exist_map_update_correspondence) {
            return;
        } else {
            workspace_path = workspace_path + "/map_update/";
            options.match_between_reconstructions_ = true;
        }
    }

    std::cout << std::endl;
    timer.PrintMinutes();
    std::cout << std::endl;
    fprintf(fs, "%s\n",
            StringPrintf("Feature Matching Read Elapsed time: %.3f [minutes]", timer.ElapsedMinutes()).c_str());
    fflush(fs);
    timer.Restart();

    std::cout << "fm new image: " << feature_data_container.GetNewImageIds().size() << std::endl;

    MatchDataContainer match_data;
    FeatureMatcher matcher(options, &feature_data_container, &match_data, &scene_graph);
    std::cout << "matching ....." << std::endl;
    matcher.Run();
    std::cout << "matching done" << std::endl;
    std::cout << "build graph" << std::endl;
    matcher.BuildSceneGraph();
    std::cout << "build graph done" << std::endl;

    std::cout << std::endl;
    timer.PrintMinutes();
    std::cout << std::endl;
    fprintf(fs, "%s\n", StringPrintf("Feature Matching Elapsed time: %.3f [minutes]", timer.ElapsedMinutes()).c_str());
    fflush(fs);
    timer.Restart();

    bool write_match = static_cast<bool>(param.GetArgument("write_match", 1));
    bool write_binary = static_cast<bool>(param.GetArgument("write_binary", 1));

    if (write_match) {
        scene_graph.WriteSceneGraphBinaryData(workspace_path + "/scene_graph.bin");
    }

    std::cout << std::endl;
    timer.PrintMinutes();
    std::cout << std::endl;
    fprintf(fs, "%s\n",
            StringPrintf("Feature Matching Write Elapsed time: %.3f [minutes]", timer.ElapsedMinutes()).c_str());
    fflush(fs);
}

size_t CompleteAndMergeTracks(const IndependentMapperOptions &options, std::shared_ptr<IncrementalMapper> mapper,
                              const std::unordered_set<mappoint_t> &mappoints) {
    size_t num_merged_observations = mapper->MergeTracks(options.Triangulation(), mappoints);
    std::cout << "  => Merged observations: " << num_merged_observations << std::endl;

    size_t num_completed_observations = mapper->CompleteTracks(options.Triangulation(), mappoints);
    std::cout << "  => Completed observations: " << num_completed_observations << std::endl;

    return num_completed_observations + num_merged_observations;
}

size_t FilterPoints(const IndependentMapperOptions &options, std::shared_ptr<IncrementalMapper> mapper,
                    const std::unordered_set<mappoint_t> &addressed_points) {
    const size_t num_filtered_observations = mapper->FilterPoints(options.IncrementalMapperOptions(), addressed_points);
    std::cout << "  => Filtered observations: " << num_filtered_observations << std::endl;
    return num_filtered_observations;
}

size_t FilterImages(const IndependentMapperOptions &options, std::shared_ptr<IncrementalMapper> mapper,
                    const std::unordered_set<image_t> &addressed_images) {
    const size_t num_filtered_images = mapper->FilterImages(options.IncrementalMapperOptions(), addressed_images);
    std::cout << "  => Filtered images: " << num_filtered_images << std::endl;
    return num_filtered_images;
}

void IterativeGlobalRefinement(
    const IndependentMapperOptions &mapper_options, std::shared_ptr<IncrementalMapper> mapper,
    std::shared_ptr<Reconstruction> reconstruction, std::unordered_set<mappoint_t> &all_variable_mappoints,
    std::unordered_set<image_t> &reg_new_image_ids, const std::unordered_set<image_t> &fixed_image_ids,
    const std::unordered_set<mappoint_t> &fixed_mappoint_ids, Timer &map_update_timer, double &merge_time_cost,
    double &filter_time_cost, double &ba_time_cost, const size_t &ba_new_num_reg_images) {
    BundleAdjustmentOptions ba_options = mapper_options.GlobalBundleAdjustment();

    map_update_timer.Restart();
    CompleteAndMergeTracks(mapper_options, mapper, all_variable_mappoints);
    std::cout << "  => Retriangulated observations: "
              << mapper->Retriangulate(mapper_options.Triangulation(), &reg_new_image_ids) << std::endl;
    merge_time_cost += map_update_timer.ElapsedMicroSeconds();

    std::unordered_set<mappoint_t> modified_points = mapper->GetModifiedMapPoints();
    for (auto mappoint_id : modified_points) {
        if (all_variable_mappoints.count(mappoint_id) == 0) {
            all_variable_mappoints.insert(mappoint_id);
        }
    }
    mapper->ClearModifiedMapPoints();

    auto &registered_images = mapper->GetReconstruction().RegisterImageIds();
    CHECK_GE(registered_images.size(), 2) << "At least two images must be "
                                             "registered for global "
                                             "bundle-adjustment";

    for (size_t iter = 0; iter < mapper_options.ba_global_max_refinements; iter++) {
        const size_t num_observations = mapper->GetReconstruction().ComputeNumObservations();
        // Avoid degeneracies in bundle adjustment.
        reconstruction->FilterObservationsWithNegativeDepth();

        if (mapper_options.single_camera && (ba_new_num_reg_images < mapper_options.num_fix_camera_first)) {
            ba_options.refine_focal_length = false;
            ba_options.refine_extra_params = false;
            ba_options.refine_principal_point = false;
            ba_options.refine_local_extrinsics = false;
        }

        BundleAdjustmentConfig ba_config;
        std::unordered_set<image_t> all_variable_images = reconstruction->FindImageForMapPoints(all_variable_mappoints);

        for (auto &image_id : reg_new_image_ids) {
            if (reconstruction->IsImageRegistered(image_id)) {
                ba_config.AddImage(image_id);
            }
        }

        std::cout << "GBA image count: " << ba_config.NumImages() << std::endl;

        map_update_timer.Restart();
        if (ba_config.NumImages() > 1) {
            BundleAdjuster bundle_adjuster(ba_options, ba_config);
            if (!bundle_adjuster.Solve(reconstruction.get())) {
                std::cout << "Bundle Adjustment Failed!" << std::endl;
            }
        }
        ba_time_cost += map_update_timer.ElapsedMicroSeconds();

        map_update_timer.Restart();
        size_t num_merged_and_completed_observations =
            CompleteAndMergeTracks(mapper_options, mapper, all_variable_mappoints);
        merge_time_cost += map_update_timer.ElapsedMicroSeconds();

        std::unordered_set<mappoint_t> new_mappoints;
        for (const auto &mappoint_id : all_variable_mappoints) {
            if (fixed_mappoint_ids.count(mappoint_id) == 0) {
                new_mappoints.insert(mappoint_id);
            }
        }

        map_update_timer.Restart();
        const size_t num_filtered_observations = FilterPoints(mapper_options, mapper, new_mappoints);
        const size_t num_filtered_images = FilterImages(mapper_options, mapper, reg_new_image_ids);
        filter_time_cost += map_update_timer.ElapsedMicroSeconds();

        std::unordered_set<mappoint_t> modified_points = mapper->GetModifiedMapPoints();
        for (auto mappoint_id : modified_points) {
            if (all_variable_mappoints.count(mappoint_id) == 0) {
                all_variable_mappoints.insert(mappoint_id);
            }
        }
        mapper->ClearModifiedMapPoints();

        size_t num_changed_observations = num_merged_and_completed_observations + num_filtered_observations;

        const double changed = static_cast<double>(num_changed_observations) / num_observations;
        std::cout << StringPrintf("  => Changed observations: %.6f", changed) << std::endl;
        if (changed < 0.0005) {
            break;
        }
    }
}

size_t TriangulateImage(const IndependentMapperOptions &options, const Image &image,
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

void MapUpdate(const std::shared_ptr<sensemap::FeatureDataContainer> &feature_data_container,
               const std::shared_ptr<sensemap::SceneGraphContainer> &scene_graph_container,
               std::shared_ptr<sensemap::ReconstructionManager> &reconstruction_manager, Configurator &param) {
    Timer timer;
    timer.Start();

    ImageReaderOptions reader_options;
    option_parser.GetImageReaderOptions(reader_options, param);

    // Load reconstructions
    auto reconstruction = std::make_shared<Reconstruction>();

    std::string workspace_path = param.GetArgument("sfm_export_path", "");
    CHECK(!workspace_path.empty()) << "workspace path empty";

    std::string image_path = param.GetArgument("image_path", "");
    CHECK(!image_path.empty()) << "image path empty";

    // Get all the reconstruction file
    std::vector<std::string> file_list = sensemap::GetDirList(workspace_path);

    bool load_reconstruction_success = false;

    // Check the reconstruction file exisit or not
    if (boost::filesystem::is_directory(workspace_path)) {
        if (dirExists(workspace_path + "/cameras.bin") && dirExists(workspace_path + "/images.bin") &&
            dirExists(workspace_path + "/points3D.bin")) {
            load_reconstruction_success = true;
        }
    }

    // Check reconsutrction file load
    if (!load_reconstruction_success) {
        std::cout << " Load Reconstruction failed !!" << std::endl;
        return;
    }

    // Load reconstruction
    bool camera_rig = reader_options.num_local_cameras > 1;
    std::cout << "camera rig: " << camera_rig << std::endl;
    reconstruction->ReadReconstruction(workspace_path, camera_rig);

    std::cout << std::endl;
    timer.PrintMinutes();
    std::cout << std::endl;
    fprintf(fs, "%s\n", StringPrintf("Map Update Read Elapsed time: %.3f [minutes]", timer.ElapsedMinutes()).c_str());
    fflush(fs);
    timer.Restart();

    // Convert all the image id to 0
    auto old_image_ids = reconstruction->RegisterImageIds();
    std::unordered_set<image_t> fixed_image_ids;
    std::unordered_set<camera_t> fixed_camera_ids;
    camera_t initial_reconstruction_camera_id;
    bool initial_reconstruction_camera_set = false;
    for (auto old_image_id : old_image_ids) {
        reconstruction->Image(old_image_id).SetLabelId(0);
        fixed_image_ids.insert(old_image_id);
        fixed_camera_ids.insert(reconstruction->Image(old_image_id).CameraId());
        if (!initial_reconstruction_camera_set) {
            initial_reconstruction_camera_id = reconstruction->Image(old_image_id).CameraId();
            initial_reconstruction_camera_set = true;
        }
    }
    auto fixed_point_ids = reconstruction->MapPointIds();

    PrintHeading1("Update new images");
    IndependentMapperOptions mapper_options;
    option_parser.GetMapperOptions(mapper_options, param);

    std::shared_ptr<IncrementalMapper> mapper = std::make_shared<IncrementalMapper>(scene_graph_container);

    std::unordered_set<mappoint_t> const_mappoint_ids = reconstruction->MapPointIds();

    std::cout << "Current reconstruction image number before setup = " << reconstruction->Images().size() << std::endl;
    std::cout << "Current reconstruction registed image number = " << reconstruction->RegisterImageIds().size()
              << std::endl;
    std::cout << "Current reconstruction mappoint number = " << reconstruction->MapPointIds().size() << std::endl;
    mapper->BeginReconstruction(reconstruction);
    std::cout << "Current reconstruction image number after setup = " << reconstruction->Images().size() << std::endl;
    std::cout << "Current reconstruction registed image number = " << reconstruction->RegisterImageIds().size()
              << std::endl;
    std::cout << "Current reconstruction mappoint number = " << reconstruction->MapPointIds().size() << std::endl;

    auto cameras = reconstruction->Cameras();
    std::cout << "Camera number = " << cameras.size() << std::endl;

    for (auto camera : cameras) {
        if (camera.second.NumLocalCameras() == 1) {
            std::cout << "  Camera index = " << camera.first << std::endl;
            std::cout << "  Camera model = " << camera.second.ModelName() << std::endl;
            std::cout << "  Camera param = ";
            for (auto param : camera.second.Params()) {
                std::cout << "  " << param;
            }
            std::cout << std::endl;
        } else {
            std::cout << "camera id: " << camera.first << std::endl;
            std::cout << "local camera param = ";
            for (auto param : camera.second.LocalParams()) {
                std::cout << "  " << param;
            }
            std::cout << std::endl;
            for (auto qvec : camera.second.LocalQvecs()) {
                std::cout << " " << qvec;
            }
            std::cout << std::endl;
            for (auto tvec : camera.second.LocalTvecs()) {
                std::cout << " " << tvec;
            }
            std::cout << std::endl;

            // this is a new camera
            if (fixed_camera_ids.count(camera.first) == 0) {
                reconstruction->Camera(camera.first).LocalQvecs() =
                    reconstruction->Camera(initial_reconstruction_camera_id).LocalQvecs();
                reconstruction->Camera(camera.first).LocalTvecs() =
                    reconstruction->Camera(initial_reconstruction_camera_id).LocalTvecs();

                std::cout << "rectified local extrinsics: " << std::endl;
                for (auto qvec : reconstruction->Camera(camera.first).LocalQvecs()) {
                    std::cout << " " << qvec;
                }
                std::cout << std::endl;
                for (auto tvec : reconstruction->Camera(camera.first).LocalTvecs()) {
                    std::cout << " " << tvec;
                }
                std::cout << std::endl;
            }
        }
    }

    int update_image_counter = 0;
    Timer map_update_timer;
    map_update_timer.Start();
    double merge_time_cost = 0;
    double filter_time_cost = 0;
    double image_update_cost = 0;
    double ba_time_cost = 0;

    size_t ba_old_num_reg_images = reconstruction->NumRegisterImages();
    size_t ba_old_num_points = reconstruction->NumMapPoints();

    size_t ba_new_num_reg_images;
    size_t ba_new_num_points;

    size_t ba_prev_num_reg_images = 1;
    size_t ba_prev_num_points = 1;

    mapper->ClearModifiedMapPoints();
    std::unordered_set<mappoint_t> all_variable_mappoints;
    std::unordered_set<image_t> all_variable_images;

    std::unordered_set<image_t> new_image_set;
    for (auto image_id : feature_data_container->GetNewImageIds()) {
        new_image_set.insert(image_id);
    }
    std::cout << "New image count: " << new_image_set.size() << std::endl;
    std::unordered_set<image_t> reg_new_image_ids;

    bool reg_next_success = true;

    while (reg_next_success) {
        std::vector<std::pair<image_t, float>> next_images;
        next_images = mapper->FindNextImages(mapper_options.IncrementalMapperOptions());

        if (next_images.empty()) {
            std::cout << "Could not find next images" << std::endl;
            break;
        }
        bool have_new_image = false;
        for (size_t i = 0; i < next_images.size(); ++i) {
            if (new_image_set.count(next_images[i].first) > 0) {
                have_new_image = true;
                break;
            }
        }
        if (!have_new_image) {
            std::cout << "no new image to register" << std::endl;
            break;
        }

        for (int image_id = 0; image_id < next_images.size(); image_id++) {
            image_t next_image_id = next_images[image_id].first;
            if (!new_image_set.count(next_image_id)) {
                continue;
            }
            if (!reconstruction->ExistsImage(next_image_id)) {
                continue;
            }
            const class Image &image = reconstruction->Image(next_image_id);
            if (image.IsRegistered()) {
                continue;
            }

            PrintHeading1(StringPrintf("Registering #%d (%d / %d) name: (%s), total: %d", next_image_id,
                                       reg_new_image_ids.size() + 1, new_image_set.size(),
                                       reconstruction->Image(next_image_id).Name().c_str(),
                                       reconstruction->NumRegisterImages() + 1));

            auto cur_options = mapper_options.IncrementalMapperOptions();
            cur_options.single_camera = true;

            std::vector<std::pair<point2D_t, mappoint_t>> tri_corrs;
            std::vector<char> inlier_mask;

            const Image &next_image = reconstruction->Image(next_image_id);
            const Camera &next_camera = reconstruction->Camera(next_image.CameraId());

            if (next_camera.NumLocalCameras() > 1) {
                reg_next_success = mapper->EstimateCameraPoseRig(mapper_options.IncrementalMapperOptions(), next_image_id,
                                                                 tri_corrs, inlier_mask);
            } else {
                reg_next_success = mapper->EstimateCameraPose(mapper_options.IncrementalMapperOptions(), next_image_id,
                                                              tri_corrs, inlier_mask);
            }
            if (!reg_next_success) {
                continue;
            }

            int num_first_force_be_keyframe = mapper_options.num_first_force_be_keyframe;
            bool force = (!mapper_options.extract_keyframe) || (reg_new_image_ids.size() < num_first_force_be_keyframe);

            // Only triangulation on KeyFrame.
            if (!mapper->AddKeyFrameUpdate(mapper_options.IncrementalMapperOptions(), next_image_id, tri_corrs, inlier_mask,
                                           force)) {
                continue;
            }

            // Triangulation and local BA
            map_update_timer.Restart();
            TriangulateImage(mapper_options, image, mapper);

            update_image_counter++;
            reg_new_image_ids.insert(next_image_id);

            auto local_ba_options = mapper_options.LocalBundleAdjustment();

            if (mapper_options.single_camera) {
                local_ba_options.refine_extra_params = false;
                local_ba_options.refine_focal_length = false;
                local_ba_options.refine_principal_point = false;
                local_ba_options.refine_local_extrinsics = false;
            }

            PrintHeading1("Local BA");
            for (int i = 0; i < mapper_options.ba_local_max_refinements; ++i) {
                const auto report = mapper->AdjustLocalBundle(mapper_options.IncrementalMapperOptions(), local_ba_options,
                                                              mapper_options.Triangulation(), next_image_id,
                                                              mapper->GetModifiedMapPoints(), fixed_image_ids);
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

            if (mapper_options.extract_colors) {
                reconstruction->ExtractColorsForImage(next_image_id, image_path);
            }
            image_update_cost += map_update_timer.ElapsedMicroSeconds();

            std::unordered_set<mappoint_t> modified_points = mapper->GetModifiedMapPoints();
            for (auto mappoint_id : modified_points) {
                if (all_variable_mappoints.count(mappoint_id) == 0) {
                    all_variable_mappoints.insert(mappoint_id);
                }
            }
            mapper->ClearModifiedMapPoints();

            // Global BA when neccessary
            ba_new_num_reg_images = reconstruction->NumRegisterImages() > ba_old_num_reg_images
                                        ? (reconstruction->NumRegisterImages() - ba_old_num_reg_images)
                                        : 1;
            ba_new_num_points = reconstruction->NumMapPoints() > ba_old_num_points
                                    ? (reconstruction->NumMapPoints() - ba_old_num_points)
                                    : 1;

            double image_ratio = ba_new_num_reg_images * 1.0f / ba_prev_num_reg_images;
            double image_freq = ba_new_num_reg_images - ba_prev_num_reg_images;
            double points_ratio = ba_new_num_points * 1.0f / ba_prev_num_points;
            double points_freq = ba_new_num_points - ba_prev_num_points;

            if (mapper_options.use_global_ba_update &&
                (ba_new_num_reg_images >= mapper_options.ba_global_images_ratio * ba_prev_num_reg_images ||
                 ba_new_num_reg_images >= mapper_options.ba_global_images_freq + ba_prev_num_reg_images ||
                 ba_new_num_points >= mapper_options.ba_global_points_ratio * ba_prev_num_points ||
                 ba_new_num_points >= mapper_options.ba_global_points_freq + ba_prev_num_points)) {
                PrintHeading1("Global BA");

                IterativeGlobalRefinement(mapper_options, mapper, reconstruction, all_variable_mappoints,
                                          reg_new_image_ids, fixed_image_ids, fixed_point_ids, map_update_timer,
                                          merge_time_cost, filter_time_cost, ba_time_cost, ba_new_num_reg_images);

                ba_prev_num_reg_images = reconstruction->NumRegisterImages() > ba_old_num_reg_images
                                             ? (reconstruction->NumRegisterImages() - ba_old_num_reg_images)
                                             : 1;
                ba_prev_num_points = reconstruction->NumMapPoints() > ba_old_num_points
                                         ? (reconstruction->NumMapPoints() - ba_old_num_points)
                                         : 1;
            }
            if (reg_next_success) {
                break;
            }
        }
    }

    if (mapper_options.register_nonkeyframe) {
        for (const auto &image_id : new_image_set) {
            if (!reconstruction->ExistsImage(image_id)) {
                continue;
            }

            class Image &image = reconstruction->Image(image_id);
            const Camera &camera = reconstruction->Camera(image.CameraId());
            if (image.IsRegistered()) {
                continue;
            }
            PrintHeading1(StringPrintf("Registering NonKeyFrame #%d (%d / %d) name: (%s)", image_id,
                                       reg_new_image_ids.size() + 1, new_image_set.size(), image.Name().c_str()));

            mapper->ClearModifiedMapPoints();

            if (camera.NumLocalCameras() == 1) {
                if (!mapper->RegisterNonKeyFrame(mapper_options.IncrementalMapperOptions(), image_id)) {
                    continue;
                }
            } else {
                if (!mapper->RegisterNonKeyFrameRig(mapper_options.IncrementalMapperOptions(), image_id)) {
                    continue;
                }
            }

            reg_new_image_ids.insert(image_id);

            TriangulateImage(mapper_options, image, mapper);
            if (mapper_options.extract_colors) {
                reconstruction->ExtractColorsForImage(image_id, image_path);
            }

            update_image_counter++;

            std::unordered_set<mappoint_t> modified_points = mapper->GetModifiedMapPoints();
            for (auto mappoint_id : modified_points) {
                if (all_variable_mappoints.count(mappoint_id) == 0) {
                    all_variable_mappoints.insert(mappoint_id);
                }
            }
        }
    }

    ba_new_num_reg_images = reconstruction->NumRegisterImages() > ba_old_num_reg_images
                                ? (reconstruction->NumRegisterImages() - ba_old_num_reg_images)
                                : 1;
    PrintHeading1("Final Global BA");
    IterativeGlobalRefinement(mapper_options, mapper, reconstruction, all_variable_mappoints, reg_new_image_ids,
                              fixed_image_ids, fixed_point_ids, map_update_timer, merge_time_cost, filter_time_cost,
                              ba_time_cost, ba_new_num_reg_images);

    std::cout
        << StringPrintf("Map Update Merge Track Elapsed time: %.3f [minutes]", merge_time_cost / (1e6 * 60)).c_str()
        << std::endl;
    std::cout
        << StringPrintf("Map Update Filte Point Elapsed time: %.3f [minutes]", filter_time_cost / (1e6 * 60)).c_str()
        << std::endl;
    std::cout << StringPrintf("Map Update Register and LBA Elapsed time: %.3f [minutes]",
                              image_update_cost / (1e6 * 60)).c_str()
              << std::endl;
    std::cout << StringPrintf("Map Update GBA Elapsed time: %.3f [minutes]", ba_time_cost / (1e6 * 60)).c_str()
              << std::endl;

    fprintf(fs, "%s\n",
            StringPrintf("Map Update Merge Track Elapsed time: %.3f [minutes]", merge_time_cost / (1e6 * 60)).c_str());
    fflush(fs);
    fprintf(fs, "%s\n",
            StringPrintf("Map Update Filte Point Elapsed time: %.3f [minutes]", filter_time_cost / (1e6 * 60)).c_str());
    fflush(fs);
    ~fprintf(fs, "%s\n",
             StringPrintf("Map Update Register and LBA Elapsed time: %.3f [minutes]", image_update_cost / (1e6 * 60))
                 .c_str());
    fflush(fs);
    fprintf(fs, "%s\n", StringPrintf("Map Update GBA Elapsed time: %.3f [minutes]", ba_time_cost / (1e6 * 60)).c_str());
    fflush(fs);
    std::cout << "Update image number = " << update_image_counter << std::endl;

    std::cout << std::endl;
    timer.PrintMinutes();
    std::cout << std::endl;
    fprintf(fs, "%s\n", StringPrintf("Map Update Elapsed time: %.3f [minutes]", timer.ElapsedMinutes()).c_str());
    fflush(fs);
    timer.Restart();

    workspace_path = workspace_path + "/map_update";

    mapper->EndReconstruction(false);

    timer.Restart();
    std::string reconstruction_path = workspace_path + "/0/";
    if (!boost::filesystem::exists(reconstruction_path)) {
        boost::filesystem::create_directories(reconstruction_path);
    }
    reconstruction->WriteBinary(reconstruction_path);

    if (camera_rig) {
        std::string export_path = JoinPaths(workspace_path, "0-export");
        if (!boost::filesystem::exists(export_path)) {
            boost::filesystem::create_directories(export_path);
        }
        Reconstruction rig_reconstruction;
        reconstruction->ConvertRigReconstruction(rig_reconstruction);
        rig_reconstruction.WriteReconstruction(export_path);
    }

    std::cout << std::endl;
    timer.PrintMinutes();
    std::cout << std::endl;
    fprintf(fs, "%s\n", StringPrintf("Map Update Write Elapsed time: %.3f [minutes]", timer.ElapsedMinutes()).c_str());
    fflush(fs);
}

int main(int argc, char *argv[]) {
    using namespace sensemap;

    PrintHeading("SenseMap.  Copyright(c) 2019, SenseTime Group.");
    PrintHeading(std::string("Version:Scale-Gravity-Recovery-")+__VERSION__);

    Timer timer;
    timer.Start();

    configuration_file_path = std::string(argv[1]);
    Configurator param;
    param.Load(configuration_file_path.c_str());

    std::string workspace_path = param.GetArgument("sfm_export_path", "");
    CHECK(!workspace_path.empty()) << "image path empty";

    std::string image_path = param.GetArgument("image_path", "");
    CHECK(!workspace_path.empty()) << "image path empty";

    if (!boost::filesystem::exists(workspace_path)) {
        CHECK(boost::filesystem::create_directories(workspace_path)) << "Create workspace failed";
    }

    fs = fopen((workspace_path + "/time.txt").c_str(), "w");

    auto reconstruction_manager = std::make_shared<ReconstructionManager>();
    auto scene_graph_container = std::make_shared<SceneGraphContainer>();
    auto feature_data_container = std::make_shared<FeatureDataContainer>();

    std::string measured_sequence_path = param.GetArgument("measured_sequence_path", "");
    std::vector<std::unordered_map<std::string, std::pair<Eigen::Vector4d, Eigen::Vector3d>>> sequence_poses;
    std::vector<double> intrinsics;

    LoadViSlamFrames(measured_sequence_path, sequence_poses, intrinsics);

    if(!boost::filesystem::exists(workspace_path + "/map_update/0")){
        FeatureExtraction(*feature_data_container.get(), param, intrinsics);

        typedef FeatureMatchingOptions::RetrieveType RetrieveType;
        RetrieveType retrieve_type = (RetrieveType)param.GetArgument("retrieve_type", 0);
        if (retrieve_type == RetrieveType::VLAD) {
            GlobalFeatureExtraction(*feature_data_container.get(), param);
        }

        FeatureMatching(*feature_data_container.get(), *scene_graph_container.get(), param);

        // Map update
        if (static_cast<bool>(param.GetArgument("map_update", 0))) {
            MapUpdate(feature_data_container, scene_graph_container, reconstruction_manager, param);
        }
        fclose(fs);
    }

    auto reconstruction = std::make_shared<Reconstruction>();

    CHECK(boost::filesystem::exists(workspace_path + "/map_update/0"));
    reconstruction->ReadReconstruction(workspace_path + "/map_update/0", false);

    std::vector<std::unordered_map<image_t, std::pair<Eigen::Vector4d, Eigen::Vector3d>>> group_poses;
    group_poses.resize(sequence_poses.size());

    const std::vector<image_t> image_ids = reconstruction->RegisterImageIds();

    for (const auto &image_id : image_ids) {
        const Image &image = reconstruction->Image(image_id);
        std::string image_name = image.Name();

        for (size_t i = 0; i < sequence_poses.size(); ++i) {
            if (sequence_poses[i].find(image_name) != sequence_poses[i].end()) {
                group_poses[i].emplace(image_id, sequence_poses[i].at(image_name));
            }
        }
    }

    ScaleGravityEstimator scale_gravity_estimator(reconstruction);

    if (scale_gravity_estimator.Estimate(group_poses)) {
        Eigen::Matrix<double, 3, 4> transform = scale_gravity_estimator.GetTransform();

        reconstruction->TransformReconstruction(transform, false);

        std::string transformed_reconstruction_path = workspace_path + "/map_update/0-trans";
        if (!boost::filesystem::exists(transformed_reconstruction_path)) {
            boost::filesystem::create_directories(transformed_reconstruction_path);
        }
        reconstruction->WriteBinary(transformed_reconstruction_path);


        auto export_reconstruction = std::make_shared<Reconstruction>();
        export_reconstruction->ReadReconstruction(workspace_path);
        export_reconstruction->TransformReconstruction(transform,false);
        export_reconstruction->WriteBinary(workspace_path);
        export_reconstruction.reset();


        std::string org_workspace_path = param.GetArgument("workspace_path", "");
        CHECK(!workspace_path.empty()) << "image path empty";

        auto org_reconstruction = std::make_shared<Reconstruction>();
        CHECK(boost::filesystem::exists(org_workspace_path + "/0"));
        org_reconstruction->ReadReconstruction(org_workspace_path + "/0", false);
        org_reconstruction->TransformReconstruction(transform, false);
        org_reconstruction->WriteBinary(org_workspace_path + "/0");
        org_reconstruction.reset();

        if(boost::filesystem::exists(org_workspace_path + "/0-export")){
            auto org_rig_reconstruction = std::make_shared<Reconstruction>();
            org_rig_reconstruction->ReadReconstruction(org_workspace_path + "/0-export", false);
            org_rig_reconstruction->TransformReconstruction(transform, false);
            org_rig_reconstruction->WriteBinary(org_workspace_path + "/0-export");
            org_rig_reconstruction.reset();
        }

        auto keyframe_reconstruction = std::make_shared<Reconstruction>();
        CHECK(boost::filesystem::exists(org_workspace_path + "/0/KeyFrames"));
        keyframe_reconstruction->ReadReconstruction(org_workspace_path + "/0/KeyFrames", false);
        keyframe_reconstruction->TransformReconstruction(transform, false);
        keyframe_reconstruction->WriteBinary(org_workspace_path + "/0/KeyFrames");
        keyframe_reconstruction.reset();

        if(boost::filesystem::exists(org_workspace_path + "/0/KeyFrames-export")){
            auto keyframe_rig_reconstruction = std::make_shared<Reconstruction>();
            keyframe_rig_reconstruction->ReadReconstruction(org_workspace_path + "/0/KeyFrames-export", false);
            keyframe_rig_reconstruction->TransformReconstruction(transform, false);
            keyframe_rig_reconstruction->WriteBinary(org_workspace_path + "/0/KeyFrames-export");
            keyframe_rig_reconstruction.reset();
        }

        std::string scale_gravity_trans_path = org_workspace_path + "/scale_gravity.txt";
        std::ofstream file_scale_gravity_trans_path(scale_gravity_trans_path,std::ios::trunc);
        CHECK(file_scale_gravity_trans_path.is_open());
        for(size_t m = 0; m<3; m++){
            for(size_t n = 0; n<3; n++){
               file_scale_gravity_trans_path<< transform(m,n)<<" ";
            }
            file_scale_gravity_trans_path<<std::endl;
        }
        file_scale_gravity_trans_path.close();
    }
    else{
        std::cout<<"Estimate scale and gravity failed"<<std::endl;
    }

    std::cout << std::endl;
    timer.PrintMinutes();
    return 0;
}