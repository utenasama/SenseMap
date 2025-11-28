//
// Created by sh on 3/28/19.
//

// Copyright (c) 2019, SenseTime Group.
// All rights reserved.

#include <boost/filesystem/path.hpp>
#include <iostream>
#include <opencv2/opencv.hpp>

#include "base/reconstruction_manager.h"
#include "cluster/fast_community.h"
#include "container/feature_data_container.h"
#include "feature/utils.h"
#include "graph/correspondence_graph.h"
#include "graph/maximum_spanning_tree_graph.h"
#include "graph/scene_clustering.h"
#include "util/misc.h"
#include "util/types.h"

int main(int argc, char* argv[]) {
    using namespace sensemap;

    std::string image_path(argv[1]);
    std::string workspace_path(argv[2]);

    PrintHeading("SenseMap.  Copyright(c) 2019, SenseTime Group.");

    auto reconstruction_manager = std::make_shared<ReconstructionManager>();
    auto scene_graph_container = std::make_shared<SceneGraphContainer>();
    auto feature_data_container = std::make_shared<FeatureDataContainer>();

    scene_graph_container->ReadSceneGraphData(workspace_path + "/scene_graph.txt");

    auto correspondence_graph = scene_graph_container->CorrespondenceGraph();

    EIGEN_STL_UMAP(image_t, class Image)& images = scene_graph_container->Images();
    EIGEN_STL_UMAP(camera_t, class Camera)& cameras = scene_graph_container->Cameras();

    for (auto& camera : cameras) {
        if (camera.second.Params().size() > 0) {
            camera.second.SetPriorFocalLength(true);
            camera.second.SetCameraConstant(true);
        }
    }

    // FeatureDataContainer data_container;

    feature_data_container->ReadCameras(workspace_path + "/cameras.txt");
    feature_data_container->ReadImagesData(workspace_path + "/features.txt");

    std::vector<image_t> image_ids = feature_data_container->GetImageIds();

    std::cout << "image_ids.size() = " << image_ids.size() << std::endl;

    // Usage:
    for (const auto image_id : image_ids) {
        const Image& image = feature_data_container->GetImage(image_id);
        images[image_id] = image;

        const FeatureKeypoints& keypoints = feature_data_container->GetKeypoints(image_id);
        const std::vector<Eigen::Vector2d> points = FeatureKeypointsToPointsVector(keypoints);
        // images[image_id].SetPoints2D(points);
        images[image_id].SetPoints2D(keypoints);

        const Camera& camera = feature_data_container->GetCamera(image.CameraId());
        if (!scene_graph_container->ExistsCamera(image.CameraId())) {
            cameras[image.CameraId()] = camera;
            // cameras[image.CameraId()].SetCameraConstant(true);
        }

        images[image_id].SetNumObservations(correspondence_graph->NumObservationsForImage(image_id));
        images[image_id].SetNumCorrespondences(correspondence_graph->NumCorrespondencesForImage(image_id));
    }

    correspondence_graph->Finalize();

    std::cout << "ExportToGraph" << std::endl;
    correspondence_graph->ExportToGraph(workspace_path + "/scene_graph.png");
    std::cout << "ExportToGraph done!" << std::endl;

    PrintHeading1("Test Graph");

    /// Mst
    std::unordered_map<image_t, Eigen::Vector3d> orientations;
    std::cout << "MST Detection..." << std::endl;
    OrientationsFromMaximumSpanningTree(*correspondence_graph, &orientations);

    /// community
    std::unordered_map<image_pair_t, point2D_t> corrs_between_images =
        correspondence_graph->NumCorrespondencesBetweenImages();

    // std::vector<std::tuple<image_t, image_t, double> > image_pairs;
    // for (const auto corres : corrs_between_images) {
    // 	image_pair_t image_pair_id = corres.first;
    // 	point2D_t num_corr = corres.second;
    // 	image_t image_id1, image_id2;
    // 	sensemap::utility::PairIdToImagePair(
    // 			image_pair_id,
    // 			&image_id1,
    // 			&image_id2);
    // 	image_pairs.emplace_back(image_id1, image_id2, sqrt(num_corr));
    // }

    SceneClustering::Options options;

    std::cout << "Community Detection..." << std::endl;
    std::vector<std::vector<image_t> > communities;
    std::vector<std::unordered_set<image_t> > overlaps;
    fastcommunity::CommunityDetection(options, correspondence_graph, communities, overlaps);

    //	auto options = std::make_shared<MapperOptions>();
    //	options->mapper_type = MapperType::CLUSTER;
    //	MapperController *mapper = MapperController::Create(
    //			options,
    //			workspace_path,
    //			image_path,
    //			feature_data_container,
    //			scene_graph_container,
    //			reconstruction_manager);
    //	mapper->Start();
    //	mapper->Wait();
    //
    //	for (size_t i = 0; i < reconstruction_manager->Size(); ++i) {
    //		std::string rec_path = StringPrintf("%s/%d", workspace_path.c_str(), i);
    //		std::cout << rec_path << std::endl;
    //		reconstruction_manager->Get(i)->WriteReconstruction(rec_path,
    //			options->cluster_mapper_options.mapper_options.write_binary_model);
    //	}

    return 0;
}