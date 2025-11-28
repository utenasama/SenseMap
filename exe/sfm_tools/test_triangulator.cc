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

int main(int argc, char* argv[]) {
    workspace_path = std::string(argv[1]);
    input_path = std::string(argv[2]);

    PrintHeading("SenseMap.  Copyright(c) 2019, SenseTime Group.");

    auto reconstruction_manager = std::make_shared<ReconstructionManager>();
    auto scene_graph_container = std::make_shared<SceneGraphContainer>();
    auto feature_data_container = std::make_shared<FeatureDataContainer>();

    // Read feature file
    PrintHeading1("Loading feature data ");
    Timer timer;
    timer.Start();
    if (dirExists(workspace_path + "/features.bin")) {
        feature_data_container->ReadImagesBinaryDataWithoutDescriptor(workspace_path + "/features.bin");
        feature_data_container->ReadCamerasBinaryData(input_path + "/cameras.bin");
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

    CHECK_GE(reconstruction->NumImages(), 2) << "Need at least two images for triangulation";

    std::shared_ptr<IncrementalMapper> mapper = std::make_shared<IncrementalMapper>(scene_graph_container);
    mapper->BeginReconstruction(reconstruction);

    //////////////////////////////////////////////////////////////////////////////
    // Triangulation
    //////////////////////////////////////////////////////////////////////////////
    IndependentMapperOptions mapper_options;

    const auto tri_options = mapper_options.Triangulation();

    const auto& reg_image_ids = reconstruction->RegisterImageIds();

    for (size_t i = 0; i < reg_image_ids.size(); ++i) {
        const image_t image_id = reg_image_ids[i];

        // Check the image is in the scene graph or not
        if (!correspondence_graph->ExistsImage(image_id)) {
            continue;
        }

        const auto& image = reconstruction->Image(image_id);

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

    // Configure bundle adjustment.
    BundleAdjustmentConfig ba_config;

    for (size_t i = 0; i < reg_image_ids.size(); ++i) {
        const image_t image_id = reg_image_ids[i];
        if (!correspondence_graph->ExistsImage(image_id)) {
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

    // PrintHeading1("Extracting colors");
    // reconstruction->ExtractColorsForAllImages(image_path);

    const bool kDiscardReconstruction = false;
    mapper->EndReconstruction(kDiscardReconstruction);

    reconstruction->WriteReconstruction(workspace_path, true);

    // Export ORB map
    // ExportORBMap(reconstruction, scene_graph_container, feature_data_container);

    return 0;
}