// Copyright (c) 2019, SenseTime Group.
// All rights reserved.

#include <boost/filesystem/path.hpp>
#include <opencv2/opencv.hpp>

#include "base/camera.h"
#include "base/image.h"
#include "base/image_reader.h"
#include "base/reconstruction_manager.h"
#include "base/pose.h"
#include "graph/correspondence_graph.h"
#include "util/misc.h"

#include "container/feature_data_container.h"
#include "feature/extraction.h"
#include "feature/feature_matcher.h"
#include "feature/utils.h"

#include "controllers/cluster_mapper_controller.h"

#include "../Configurator_yaml.h"
#include "../option_parsing.h"

#include <dirent.h>
#include <sys/stat.h>

using namespace sensemap;

std::string configuration_file_path;

FILE *fs;

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

void FeatureExtraction(FeatureDataContainer &feature_data_container, Configurator &param) {
    std::string workspace_path = param.GetArgument("workspace_path", "");
    CHECK(!workspace_path.empty());
    OptionParser option_parser;

    ImageReaderOptions reader_options;
    option_parser.GetImageReaderOptions(reader_options,param);

    bool camera_rig = (reader_options.num_local_cameras > 1);
    bool with_depth = reader_options.with_depth;

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

    // Check the AprilTag detection file exist or not
    if (static_cast<bool>(param.GetArgument("detect_apriltag", 0)) && exist_feature_file) {
        if (boost::filesystem::exists(JoinPaths(workspace_path, "/apriltags.bin"))) {
            feature_data_container.ReadAprilTagBinaryData(JoinPaths(workspace_path, "/apriltags.bin"));
        } else {
            // FIXME:
            std::cout << " Warning! Existing feature data do not contain AprilTags data" << std::endl;
        }
    }

    // If the camera model is Spherical
    if ((reader_options.camera_model == "SPHERICAL" || reader_options.num_local_cameras >1)&& exist_feature_file) {
        if (boost::filesystem::exists(JoinPaths(workspace_path, "/sub_panorama.bin"))) {
            feature_data_container.ReadSubPanoramaBinaryData(JoinPaths(workspace_path, "/sub_panorama.bin"));
        } else if (boost::filesystem::exists(JoinPaths(workspace_path, "/sub_panorama.txt"))) {
            feature_data_container.ReadSubPanoramaData(JoinPaths(workspace_path, "/sub_panorama.txt"));
        } else {
            // FIXME:
            std::cout << " Warning! Existing feature data do not contain sub_panorama data" << std::endl;
        }

        if (boost::filesystem::exists(JoinPaths(workspace_path, "/piece_indices.bin"))) {
            feature_data_container.ReadPieceIndicesBinaryData(JoinPaths(workspace_path, "/piece_indices.bin"));
        } else if (boost::filesystem::exists(JoinPaths(workspace_path, "/piece_indices.txt"))) {
            feature_data_container.ReadPieceIndicesData(JoinPaths(workspace_path, "/piece_indices.txt"));
        } else {
            std::cout << " Warning! Existing feature data do not contain piece_indices data" << std::endl;
        }

        return;
    } else if (exist_feature_file) {
        return;
    }

    SiftExtractionOptions sift_extraction;
    option_parser.GetFeatureExtractionOptions(sift_extraction,param);

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
            if (reader_options.camera_model == "SPHERICAL" || reader_options.num_local_cameras>1) {
                feature_data_container.WriteSubPanoramaBinaryData(JoinPaths(workspace_path, "/sub_panorama.bin"));
            }
            if (sift_extraction.detect_apriltag) {
                // Check the Arpiltag Detect Result
                if(feature_data_container.ExistAprilTagDetection()){
                    feature_data_container.WriteAprilTagBinaryData(JoinPaths(workspace_path, "/apriltags.bin"));
                }else{
                    std::cout << "Warning: No Apriltag Detection has been found ... " << std::endl;
                }
            }
            if(reader_options.num_local_cameras == 2 && reader_options.camera_model == "OPENCV_FISHEYE"){
                feature_data_container.WritePieceIndicesBinaryData(JoinPaths(workspace_path, "/piece_indices.bin"));
            }

        } else {
            feature_data_container.WriteImagesData(JoinPaths(workspace_path, "/features.txt"));
            feature_data_container.WriteCameras(JoinPaths(workspace_path, "/cameras.txt"));
            feature_data_container.WriteLocalCameras(JoinPaths(workspace_path, "/local_cameras.txt"));
            if (reader_options.camera_model == "SPHERICAL" || reader_options.num_local_cameras>1) {
                feature_data_container.ReadSubPanoramaData(JoinPaths(workspace_path, "/sub_panorama.txt"));
            }
            if(reader_options.num_local_cameras == 2 && reader_options.camera_model == "OPENCV_FISHEYE"){
                feature_data_container.WritePieceIndicesData(JoinPaths(workspace_path, "/piece_indices.txt"));
            }
        }
    }
}

void FeatureMatching(FeatureDataContainer &feature_data_container, SceneGraphContainer &scene_graph,
                     Configurator &param) {
    using namespace std::chrono;
    high_resolution_clock::time_point start_time = high_resolution_clock::now();

    OptionParser option_parser;
    ImageReaderOptions reader_options;
    option_parser.GetImageReaderOptions(reader_options,param);


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

    int num_local_cameras = reader_options.num_local_cameras;

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
            const PanoramaIndexs & panorama_indices = feature_data_container.GetPanoramaIndexs(image_id);

            std::vector<uint32_t> local_image_indices(keypoints.size());
		    for(size_t i = 0; i<keypoints.size(); ++i){
                if (panorama_indices.size() == 0 && num_local_cameras == 1) {
                    local_image_indices[i] = image_id;
                } else {
			        local_image_indices[i] = panorama_indices[i].sub_image_id;
                }
		    }
		    images[image_id].SetLocalImageIndices(local_image_indices);

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
        // draw matches
        return;
    }

    const auto &image_ids = feature_data_container.GetImageIds();

    // if the map is updated, set the old image label to 1, so all the images can be matched.
    for (const auto image_id : image_ids) {
        std::string image_name = feature_data_container.GetImage(image_id).Name();
        if(feature_data_container.GetImage(image_name).LabelId() == 0){
            feature_data_container.GetImage(image_name).SetLabelId(1);
        }
    }

    FeatureMatchingOptions options;
    option_parser.GetFeatureMatchingOptions(options,param);

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

size_t CompleteAndMergeTracks(const IndependentMapperOptions& options, std::shared_ptr<IncrementalMapper> mapper) {
    const size_t num_completed_observations = mapper->CompleteTracks(options.Triangulation());
    std::cout << "  => Completed observations: " << num_completed_observations << std::endl;
    const size_t num_merged_observations = mapper->MergeTracks(options.Triangulation());
    std::cout << "  => Merged observations: " << num_merged_observations << std::endl;
    return num_completed_observations + num_merged_observations;
}


size_t FilterPoints(const IndependentMapperOptions& options, std::shared_ptr<IncrementalMapper> mapper,
                    int min_track_length) {
    const size_t num_filtered_observations = mapper->FilterPoints(options.IncrementalMapperOptions(), min_track_length);
    std::cout << "  => Filtered observations: " << num_filtered_observations << std::endl;
    return num_filtered_observations;
}


void ExtractColors(const std::string& image_path, const image_t image_id,
                   std::shared_ptr<Reconstruction> reconstruction) {
    if (!reconstruction->ExtractColorsForImage(image_id, image_path)) {
        std::cout << StringPrintf("WARNING: Could not read image %s at path %s.",
                                  reconstruction->Image(image_id).Name().c_str(), image_path.c_str())
                  << std::endl;
    }
}

void Retriangulate( const std::shared_ptr<sensemap::SceneGraphContainer> &scene_graph_container,
                    std::shared_ptr<sensemap::ReconstructionManager> &reconstruction_manager, Configurator &param) {
    using namespace sensemap;

    PrintHeading1("Incremental Mapping");

    IndependentMapperOptions mapper_options;
    OptionParser option_parser;
    option_parser.GetMapperOptions(mapper_options,param);


    std::string workspace_path = param.GetArgument("workspace_path", "");
    CHECK(!workspace_path.empty()) << "image path empty";

    std::string image_path = param.GetArgument("image_path", "");
    CHECK(!workspace_path.empty()) << "image path empty";
   
    bool use_initial_sfm = static_cast<bool>(param.GetArgument("use_initial_sfm", 0));

    bool has_initial_sfm = false;
    if(boost::filesystem::is_directory(workspace_path + "/initial_sfm")&&
       boost::filesystem::exists(workspace_path + "/initial_sfm/cameras.bin")&&
       boost::filesystem::exists(workspace_path + "/initial_sfm/images.bin")&&
       boost::filesystem::exists(workspace_path + "/initial_sfm/points3D.bin")){
       has_initial_sfm = true;
    }
    CHECK(has_initial_sfm);

    std::unordered_map<std::string, Eigen::Vector4d> prior_rotations;
    std::unordered_map<std::string, Eigen::Vector3d> prior_translations;

    auto initial_reconstruction = std::make_shared<Reconstruction>();
    initial_reconstruction->ReadReconstruction(workspace_path + "/initial_sfm");

    std::vector<image_t> image_ids = initial_reconstruction->RegisterImageIds();

    for (const auto image_id : image_ids) {
        const class Image& image = initial_reconstruction->Image(image_id);
        prior_rotations.emplace(image.Name(),image.Qvec());
        prior_translations.emplace(image.Name(),image.Tvec());
    }

    std::shared_ptr<Reconstruction> reconstruction = std::make_shared<Reconstruction>();
    std::shared_ptr<IncrementalMapper> mapper = std::make_shared<IncrementalMapper>(scene_graph_container);
    mapper->BeginReconstruction(reconstruction);

    const auto tri_options = mapper_options.Triangulation();

    const auto& rec_image_ids = reconstruction->Images();

    using namespace std::chrono;
    high_resolution_clock::time_point start_time = high_resolution_clock::now();

    for (const auto &rec_image : rec_image_ids) {
        const image_t image_id = rec_image.first;

        // Check the image is in the scene graph or not
        if (!scene_graph_container->CorrespondenceGraph()->ExistsImage(image_id)) {
            continue;
        }

        auto &image = reconstruction->Image(image_id);
        std::string image_name = image.Name();
        if (prior_rotations.find(image_name) == prior_rotations.end()) {
            // no prior pose, skipped
            continue;
        }

        image.Qvec() = prior_rotations.at(image_name);
        image.Tvec() = prior_translations.at(image_name);

        reconstruction->RegisterImage(image_id);
    }

    int triangulated_image_count = 1;
    for (const auto& rec_image : rec_image_ids) {
        const image_t image_id = rec_image.first;

        // Check the image is in the scene graph or not
        if (!scene_graph_container->CorrespondenceGraph()->ExistsImage(image_id)) {
            continue;
        }

        auto& image = reconstruction->Image(image_id);
        
        if (!image.IsRegistered()) {
            // no prior pose, skipped
            continue;
        }

        PrintHeading1(StringPrintf("Triangulating image #%d (%d)", image_id, triangulated_image_count++));

        const size_t num_existing_points3D = image.NumMapPoints();
        std::cout << "  => Image sees " << num_existing_points3D << " / " << image.NumObservations() << " points"
                  << std::endl;

        high_resolution_clock::time_point start_time_tri = high_resolution_clock::now();
        mapper->TriangulateImage(tri_options, image_id);
        high_resolution_clock::time_point end_time_tri = high_resolution_clock::now();
        std::cout << "  => Triangulated " << (image.NumMapPoints() - num_existing_points3D) << " points" << std::endl;

        std::cout << "Time cost: "
                  << duration_cast<microseconds>(end_time_tri - start_time_tri).count() << std::endl;

        if (mapper_options.use_local_ba_retriangulate_all) {
            auto ba_options = mapper_options.LocalBundleAdjustment();

            ba_options.refine_focal_length = false;
            ba_options.refine_principal_point = false;
            ba_options.refine_extra_params = false;
            ba_options.refine_extrinsics = false;
            ba_options.plane_constrain = false;

            start_time_tri = high_resolution_clock::now();
            for (int i = 0; i < mapper_options.ba_local_max_refinements; ++i) {
                const auto report =
                    mapper->AdjustLocalBundle(mapper_options.IncrementalMapperOptions(), ba_options, mapper_options.Triangulation(),
                                              image_id, mapper->GetModifiedMapPoints());
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
            end_time_tri = high_resolution_clock::now();
            std::cout << " Local BA Time cost: " << duration_cast<microseconds>(end_time_tri - start_time_tri).count()
                      << std::endl;
        }
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
    ba_options.plane_constrain = false;
    int min_track_length = mapper_options.filter_min_track_length_final;

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

    return ;

}





int main(int argc, char *argv[]) {
    using namespace sensemap;

    PrintHeading("SenseMap.  Copyright(c) 2019, SenseTime Group.");
    PrintHeading("Version: map-retriangulate-1.6.2");

    Timer timer;
    timer.Start();

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

    fs = fopen((workspace_path + "/time.txt").c_str(), "w");

    auto reconstruction_manager = std::make_shared<ReconstructionManager>();
    auto scene_graph_container = std::make_shared<SceneGraphContainer>();
    auto feature_data_container = std::make_shared<FeatureDataContainer>();

    FeatureExtraction(*feature_data_container.get(), param);

    FeatureMatching(*feature_data_container.get(), *scene_graph_container.get(), param);

    feature_data_container.reset();

    
    std::string mapper_method = param.GetArgument("mapper_method", "incremental");


    Retriangulate(scene_graph_container, reconstruction_manager, param);

    std::cout << StringPrintf("Reconstruct in %.3fs", timer.ElapsedSeconds()) << std::endl;

    PrintReconSummary(workspace_path + "/statistic.txt", scene_graph_container->NumImages(), reconstruction_manager);

    fclose(fs);

    return 0;
}
