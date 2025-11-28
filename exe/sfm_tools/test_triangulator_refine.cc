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

using namespace sensemap;

std::string image_path;
std::string workspace_path;
std::string input_path;
std::string output_path;

bool dirExists(const std::string& dirName_in) { return access(dirName_in.c_str(), F_OK) == 0; }


// size_t CompleteAndMergeTracks(const IndependentMapperOptions& options, std::shared_ptr<IncrementalMapper> mapper) {
//     const size_t num_completed_observations = mapper->CompleteTracks(options.Triangulation());
//     std::cout << "  => Completed observations: " << num_completed_observations << std::endl;
//     const size_t num_merged_observations = mapper->MergeTracks(options.Triangulation());
//     std::cout << "  => Merged observations: " << num_merged_observations << std::endl;
//     return num_completed_observations + num_merged_observations;
// }


void Retriangulate(const std::shared_ptr<sensemap::FeatureDataContainer> &feature_data_container,
                    const std::shared_ptr<sensemap::SceneGraphContainer> &scene_graph_container,
                    std::shared_ptr<Reconstruction> &reconstruction) {

    PrintHeading1("Incremental Mapping");

    IndependentMapperOptions mapper_options;

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

        // if (mapper_options.use_local_ba_retriangulate_all) {
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
        // }
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

    std::cout << "Feature Matching Elapsed time: " << duration_cast<microseconds>(end_time - start_time).count() / 6e7 << " [minutes]" << std::endl;

    return ;

}

int main(int argc, char* argv[]) {
    workspace_path = std::string(argv[1]);
    input_path = std::string(argv[2]);

    PrintHeading("SenseMap.  Copyright(c) 2019, SenseTime Group.");

    auto scene_graph_container = std::make_shared<SceneGraphContainer>();
    auto feature_data_container = std::make_shared<FeatureDataContainer>();

    // Read feature file
    PrintHeading1("Loading feature data ");
    Timer timer;
    timer.Start();
    if (dirExists(workspace_path + "/features.bin")) {
        feature_data_container->ReadImagesBinaryDataWithoutDescriptor(workspace_path + "/features.bin");
        feature_data_container->ReadCamerasBinaryData(input_path + "/cameras.bin", false);
    } else {
        std::cerr << "ERROR: Current workspace do not contain features.bin " << workspace_path << std::endl;
        return 0;
    }
    std::cout << std::endl;
    timer.PrintMinutes();
    std::cout << std::endl;

    // Read scene graph .txt or .bin
    PrintHeading1("Loading scene graph matching data");
    timer.Start();
    if (dirExists(workspace_path + "/scene_graph.bin")) {
        scene_graph_container->ReadSceneGraphBinaryData(workspace_path + "/scene_graph.bin");
    } else if (dirExists(workspace_path + "/scene_graph.txt")) {
        scene_graph_container->ReadSceneGraphData(workspace_path + "/scene_graph.txt");
    } else {
        std::cerr << "ERROR: Current workspace do not contain scene_graph.bin or scene_graph.txt " << workspace_path
                  << std::endl;
        return 0;
    }
    std::cout << std::endl;
    timer.PrintMinutes();
    std::cout << std::endl;

    auto correspondence_graph = scene_graph_container->CorrespondenceGraph();

    EIGEN_STL_UMAP(image_t, class Image)& images = scene_graph_container->Images();
    EIGEN_STL_UMAP(camera_t, class Camera)& cameras = scene_graph_container->Cameras();

    // FeatureDataContainer data_container;
    std::vector<image_t> image_ids = feature_data_container->GetImageIds();

    std::cout << "image_ids.size() = " << image_ids.size() << std::endl;

    for (const auto image_id : image_ids) {
        const Image& image = feature_data_container->GetImage(image_id);
        images[image_id] = image;
        const FeatureKeypoints& keypoints = feature_data_container->GetKeypoints(image_id);
        // const std::vector<Eigen::Vector2d> points = FeatureKeypointsToPointsVector(keypoints);
        // images[image_id].SetPoints2D(points);
        images[image_id].SetPoints2D(keypoints);
        // std::cout << image.CameraId() << std::endl;
        const Camera& camera = feature_data_container->GetCamera(image.CameraId());
        // std::cout << image.CameraId() << std::endl;
        if (!scene_graph_container->ExistsCamera(image.CameraId())) {
            cameras[image.CameraId()] = camera;
            // cameras[image.CameraId()].SetCameraConstant(true);
        }
        if (correspondence_graph->ExistsImage(image_id)) {
            images[image_id].SetNumObservations(correspondence_graph->NumObservationsForImage(image_id));
            images[image_id].SetNumCorrespondences(correspondence_graph->NumCorrespondencesForImage(image_id));
        } else {
            std::cout << "Do not contain this image" << std::endl;
        }
    }
    std::cout << "Load correspondece graph finished" << std::endl;
    correspondence_graph->Finalize();

    // Check the old reconstruction exist or not
    if (!ExistsDir(input_path)) {
        std::cerr << "ERROR: Input reconstruction path is not a directory  " << input_path << std::endl;
        return 0;
    }

    // Create output path
    // if (!boost::filesystem::exists(workspace_path)) {
    //     boost::filesystem::create_directories(workspace_path);
    // }

    PrintHeading1("Loading model");

    std::shared_ptr<Reconstruction> reconstruction = std::make_shared<Reconstruction>();
    timer.Start();
    std::cout << "model path : " << input_path << std::endl;
    reconstruction->ReadReconstruction(input_path);

    // if (clear_points) {
    std::cout << "Clear points" << std::endl;
    // Delete all the 2D point and 3D point
    reconstruction->DeleteAllPoints2DAndPoints3D();
    // Update image id using image name check
    // reconstruction->TranscribeImageIdsToDatabase(feature_data_container);
    // }

    std::cout << std::endl;
    timer.PrintMinutes();
    std::cout << std::endl;

    Retriangulate(feature_data_container, scene_graph_container, reconstruction);

    return 0;
}