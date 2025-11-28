// Copyright (c) 2019, SenseTime Group.
// All rights reserved.

#include <dirent.h>
#include <sys/stat.h>

#include <boost/filesystem/path.hpp>
#include <numeric>
#include <opencv2/opencv.hpp>

#include "../Configurator.h"
#include "base/camera.h"
#include "base/image.h"
#include "base/image_reader.h"
#include "base/reconstruction_manager.h"
#include "base/similarity_transform.h"
#include "cluster/fast_community.h"
#include "container/feature_data_container.h"
#include "estimators/scale_selection.h"
#include "feature/extraction.h"
#include "feature/feature_matcher.h"
#include "feature/utils.h"
#include "graph/correspondence_graph.h"
#include "optim/ransac/loransac.h"
#include "util/misc.h"
#include "optim/pose_graph_optimizer.h"

using namespace sensemap;

std::string image_path;
std::string workspace_path;

bool dirExists(const std::string &dirName_in) { return access(dirName_in.c_str(), F_OK) == 0; }

int main(int argc, char *argv[]) {

    std::string workspace_path = std::string(argv[1]);
    std::string recon_path = std::string(argv[2]);

    bool camera_rig = true;

    auto reconstruction_manager = std::make_shared<ReconstructionManager>();
    auto scene_graph_container = std::make_shared<SceneGraphContainer>();
    auto feature_data_container = std::make_shared<FeatureDataContainer>();

    std::cout << "Camera Rig = " << camera_rig << std::endl;

    // Read feature file
    PrintHeading1("Loading feature data ");
    Timer timer;
    timer.Start();
    if (dirExists(workspace_path + "/features.bin")) {
        feature_data_container->ReadImagesBinaryDataWithoutDescriptor(workspace_path + "/features.bin");
        if (dirExists(JoinPaths(workspace_path, "/local_cameras.bin"))) {
            feature_data_container->ReadCamerasBinaryData(JoinPaths(workspace_path, "/cameras.bin"), false);
            feature_data_container->ReadLocalCamerasBinaryData(JoinPaths(workspace_path, "/local_cameras.bin"));
        } else {
            feature_data_container->ReadCamerasBinaryData(JoinPaths(workspace_path, "/cameras.bin"), camera_rig);
        }
        feature_data_container->ReadAprilTagBinaryData(workspace_path + "/apriltags.bin");
        feature_data_container->ReadSubPanoramaBinaryData(workspace_path + "/sub_panorama.bin");
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

    EIGEN_STL_UMAP(image_t, class Image) &images = scene_graph_container->Images();
    EIGEN_STL_UMAP(camera_t, class Camera) &cameras = scene_graph_container->Cameras();

    // FeatureDataContainer data_container;
    std::vector<image_t> image_ids = feature_data_container->GetImageIds();

    std::cout << "image_ids.size() = " << image_ids.size() << std::endl;

    for (const auto image_id : image_ids) {
        const Image &image = feature_data_container->GetImage(image_id);
        if (!scene_graph_container->CorrespondenceGraph()->ExistsImage(image_id)) {
            continue;
        }

        images[image_id] = image;
        const FeatureKeypoints &keypoints = feature_data_container->GetKeypoints(image_id);
        // const std::vector<Eigen::Vector2d> points = FeatureKeypointsToPointsVector(keypoints);
        // images[image_id].SetPoints2D(points);
        images[image_id].SetPoints2D(keypoints);

        // std::cout << image.CameraId() << std::endl;
        const Camera &camera = feature_data_container->GetCamera(image.CameraId());
        // std::cout << image.CameraId() << std::endl;
        if (!scene_graph_container->ExistsCamera(image.CameraId())) {
            cameras[image.CameraId()] = camera;
        }

        const PanoramaIndexs &panorama_indices = feature_data_container->GetPanoramaIndexs(image_id);
        std::vector<uint32_t> local_image_indices(keypoints.size());
        for (size_t i = 0; i < keypoints.size(); ++i) {
            if (panorama_indices.size() == 0 && camera.NumLocalCameras() == 1) {
                local_image_indices[i] = image_id;
            } else {
                local_image_indices[i] = panorama_indices[i].sub_image_id;
            }
        }
        images[image_id].SetLocalImageIndices(local_image_indices);

        if (correspondence_graph->ExistsImage(image_id)) {
            images[image_id].SetNumObservations(correspondence_graph->NumObservationsForImage(image_id));
            images[image_id].SetNumCorrespondences(correspondence_graph->NumCorrespondencesForImage(image_id));
        } else {
            // std::cout << "Do not contain this image" << std::endl;
        }
    }
    std::cout << "Load correspondece graph finished" << std::endl;
    correspondence_graph->Finalize();

    // Check the old reconstruction exist or not
    if (!ExistsDir(workspace_path + "/0/")) {
        std::cerr << "ERROR: Input reconstruction path is not a directory  " << workspace_path + "/0/" << std::endl;
        return 0;
    }

    // Create output path
    // if (!boost::filesystem::exists(workspace_path)) {
    //     boost::filesystem::create_directories(workspace_path);
    // }

    PrintHeading1("Loading model");

    std::shared_ptr<Reconstruction> reconstruction = std::make_shared<Reconstruction>();
    timer.Start();
    std::cout << "model path : " << workspace_path + "/0/" << std::endl;
    reconstruction->ReadReconstruction(workspace_path + "/0/", camera_rig);

    // Add local camera index for reconstruction
    for (const auto &image_id : reconstruction->RegisterImageIds()) {
        const PanoramaIndexs &panorama_indices = feature_data_container->GetPanoramaIndexs(image_id);
        std::vector<uint32_t> local_image_indices(feature_data_container->GetKeypoints(image_id).size());
        for (size_t i = 0; i < feature_data_container->GetKeypoints(image_id).size(); ++i) {
            if (panorama_indices.size() == 0 &&
                feature_data_container->GetCamera(feature_data_container->GetImage(image_id).CameraId())
                        .NumLocalCameras() == 1) {
                local_image_indices[i] = image_id;
            } else {
                local_image_indices[i] = panorama_indices[i].sub_image_id;
            }
        }
        reconstruction->Image(image_id).SetLocalImageIndices(local_image_indices);
    }


    // Get all the edges

    // Get Normal Edges

    // Get Correct Loop Edges


    // Get relative pose for loop edges


    // Add Constrain to all these edges

    // Run the Pose Graph Optimization


    // Output Reconstruction pose



}