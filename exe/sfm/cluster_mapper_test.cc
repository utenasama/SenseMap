//Copyright (c) 2019, SenseTime Group.
//All rights reserved.

#include <opencv2/opencv.hpp>
#include <boost/filesystem/path.hpp>

#include "util/misc.h"
#include "base/reconstruction_manager.h"
#include "graph/correspondence_graph.h"
#include "container/feature_data_container.h"
#include "feature/utils.h"

#include "controllers/cluster_mapper_controller.h"
#include "base/similarity_transform.h"

int main(int argc, char *argv[]) {
      using namespace sensemap;

	std::string image_path(argv[1]);
	std::string workspace_path(argv[2]);

	PrintHeading("SenseMap.  Copyright(c) 2019, SenseTime Group.");

	auto reconstruction_manager = std::make_shared<ReconstructionManager>();
	auto scene_graph_container = std::make_shared<SceneGraphContainer>();
    auto feature_data_container = std::make_shared<FeatureDataContainer>();

	scene_graph_container->ReadSceneGraphData(workspace_path + "/scene_graph.txt");

    auto correspondence_graph = scene_graph_container->CorrespondenceGraph();

	EIGEN_STL_UMAP(image_t, class Image) & images = scene_graph_container->Images();
	EIGEN_STL_UMAP(camera_t, class Camera) & cameras = scene_graph_container->Cameras();

	for (auto & camera : cameras) {
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

		//draw matches
	#if 0
	if (!boost::filesystem::exists(workspace_path + "/matches")) {
		boost::filesystem::create_directories(workspace_path + "/matches");
	}

	for(image_t id1 = 0; id1 < image_ids.size(); id1++) {
		for(image_t id2 = id1 + 1; id2 < image_ids.size(); id2++) {
			auto matches =scene_graph_container->CorrespondenceGraph()->
					FindCorrespondencesBetweenImages(id1,id2);
			if(matches.empty()){
				continue;
			}
			auto image1 = feature_data_container->GetImage(id1);
			auto image2 = feature_data_container->GetImage(id2);
			const std::string input_image_path1 =
					JoinPaths(feature_data_container->GetImagePath(), image1.Name());
			const std::string input_image_path2 =
					JoinPaths(feature_data_container->GetImagePath(), image2.Name());
			cv
			::Mat mat1 = cv::imread(input_image_path1);
			cv::Mat mat2 = cv::imread(input_image_path2);
			std::vector<cv::KeyPoint> keypoints_show1, keypoints_show2;
			std::vector<cv::DMatch> matches_show;
			for(int i = 0; i < matches.size(); ++i){
				auto keypoint  =
						feature_data_container->GetKeypoints(id1)[matches[i].point2D_idx1];
				keypoints_show1.emplace_back(keypoint.x, keypoint.y,
				                             keypoint.ComputeScale(),
				                             keypoint.ComputeOrientation());
				keypoint  = feature_data_container->GetKeypoints(id2)[matches[i].point2D_idx2];
				keypoints_show2.emplace_back(keypoint.x, keypoint.y,
				                             keypoint.ComputeScale(),
				                             keypoint.ComputeOrientation());
				matches_show.emplace_back(i, i, 1);
			}
			cv::Mat first_match;
			cv::drawMatches(mat1, keypoints_show1, mat2, keypoints_show2,
			                matches_show, first_match);
			const std::string ouput_image_path = JoinPaths(
					workspace_path + "/matches",
					image1.Name()+ "+" + image2.Name());
			cv::imwrite(ouput_image_path, first_match);
//			cv::imshow("first_match ", first_match);
//			cv::waitKey(0);
		}
	}
	#endif

	//Usage:
	for (const auto image_id : image_ids) {
		const Image & image = 
            feature_data_container->GetImage(image_id);
		images[image_id] = image;
		
		const FeatureKeypoints& keypoints = 
            feature_data_container->GetKeypoints(image_id);
        const std::vector<Eigen::Vector2d> points = FeatureKeypointsToPointsVector(keypoints);
        // images[image_id].SetPoints2D(points);
        images[image_id].SetPoints2D(keypoints);

		const Camera & camera = 
            feature_data_container->GetCamera(image.CameraId());
        if (!scene_graph_container->ExistsCamera(image.CameraId())) {
			cameras[image.CameraId()] = camera;
			// cameras[image.CameraId()].SetCameraConstant(true);
		}

		images[image_id].SetNumObservations(
            correspondence_graph->NumObservationsForImage(image_id));

        images[image_id].SetNumCorrespondences(
            correspondence_graph->NumCorrespondencesForImage(image_id));
	}

	correspondence_graph->Finalize();
	/*
	std::cout << "ExportToGraph" << std::endl;
	correspondence_graph->ExportToGraph(workspace_path + "/scene_graph.png");
	std::cout << "ExportToGraph done!" << std::endl;
	*/
	//feature_data_container.reset();
	PrintHeading1("Cluster Mapping");

    auto options = std::make_shared<MapperOptions>();
	options->mapper_type = MapperType::CLUSTER;
	options->cluster_mapper_options.clustering_options.image_overlap=0;
	options->cluster_mapper_options.clustering_options.leaf_max_num_images=300;
	options->cluster_mapper_options.mapper_options.ba_refine_principal_point=false;
	options->cluster_mapper_options.mapper_options.ba_refine_focal_length = true;
	options->cluster_mapper_options.mapper_options.ba_refine_extra_params = true;
	options->cluster_mapper_options.mapper_options.ba_global_use_pba=false;
	
	options->cluster_mapper_options.enable_image_label_cluster = false; 
	options->cluster_mapper_options.mapper_options.write_binary_model = true;

	MapperController *mapper = MapperController::Create(
			options, 
			workspace_path, 
			image_path,
            feature_data_container,
			scene_graph_container,
            reconstruction_manager);
	mapper->Start();
	mapper->Wait();

	for (size_t i = 0; i < reconstruction_manager->Size(); ++i) {
		std::string rec_path = StringPrintf("%s/%d", workspace_path.c_str(), i);
		if (boost::filesystem::exists(rec_path)) {
			boost::filesystem::remove_all(rec_path);
		}
        boost::filesystem::create_directories(rec_path);
		reconstruction_manager->Get(i)->WriteReconstruction(rec_path,
			options->cluster_mapper_options.mapper_options.write_binary_model);
	}

    return 0;
}