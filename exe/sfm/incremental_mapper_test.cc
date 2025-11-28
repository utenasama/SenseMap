//Copyright (c) 2019, SenseTime Group.
//All rights reserved.

#include <opencv2/opencv.hpp>
#include "util/misc.h"
#include "base/reconstruction_manager.h"
#include "controllers/incremental_mapper_controller.h"
#include "container/scene_graph_container.h"
#include "container/feature_data_container.h"
#include "feature/utils.h"

int main(int argc, char *argv[]) {
  using namespace sensemap;

	std::string img_path(argv[1]);
	std::string workspace_path(argv[2]);

	PrintHeading("SenseMap.  Copyright(c) 2019, SenseTime Group.");

	auto reconstruction_manager = std::make_shared<ReconstructionManager>();
	auto scene_graph_container = std::make_shared<SceneGraphContainer>();

	scene_graph_container->ReadSceneGraphData(workspace_path + "/scene_graph.txt");

	EIGEN_STL_UMAP(image_t, class Image) & images = scene_graph_container->Images();
	EIGEN_STL_UMAP(camera_t, class Camera) & cameras = scene_graph_container->Cameras();

	FeatureDataContainer data_container;

	data_container.ReadCameras(workspace_path + "/cameras.txt");
	// data_container.ReadImagesData(workspace_path + "/features.txt");
	data_container.ReadImagesBinaryData(workspace_path + "/features.bin");

	std::vector<image_t> image_ids = data_container.GetImageIds();

	//////////////////////////////////////////////////////////////////////////////
	//draw matches
	#if 0
	for(image_t id1 = 0; id1 < image_ids.size(); id1++) {
		for(image_t id2 = id1 + 1; id2 < image_ids.size(); id2++) {
			auto matches =scene_graph_container->CorrespondenceGraph()->
					FindCorrespondencesBetweenImages(id1,id2);
			if(matches.empty()){
				continue;
			}
			auto image1 = data_container.GetImage(id1);
			auto image2 = data_container.GetImage(id2);
			const std::string input_image_path1 =
					JoinPaths(data_container.GetImagePath(), image1.Name());
			const std::string input_image_path2 =
					JoinPaths(data_container.GetImagePath(), image2.Name());
			cv
			::Mat mat1 = cv::imread(input_image_path1);
			cv::Mat mat2 = cv::imread(input_image_path2);
			std::vector<cv::KeyPoint> keypoints_show1, keypoints_show2;
			std::vector<cv::DMatch> matches_show;
			for(int i = 0; i < matches.size(); ++i){
				auto keypoint  =
						data_container.GetKeypoints(id1)[matches[i].point2D_idx1];
				keypoints_show1.emplace_back(keypoint.x, keypoint.y,
				                             keypoint.ComputeScale(),
				                             keypoint.ComputeOrientation());
				keypoint  = data_container.GetKeypoints(id2)[matches[i].point2D_idx2];
				keypoints_show2.emplace_back(keypoint.x, keypoint.y,
				                             keypoint.ComputeScale(),
				                             keypoint.ComputeOrientation());
				matches_show.emplace_back(i, i, 1);
			}
			cv::Mat first_match;
			//cv::drawMatches(mat1, keypoints_show1, mat2, keypoints_show2,
			//                matches_show, first_match);
			cv::drawKeypoints(mat2,keypoints_show2,first_match);

			const std::string ouput_image_path = JoinPaths(
					workspace_path + "/matches",
					image1.Name()+ "+" + image2.Name()+"_1.jpg");
			cv::imwrite(ouput_image_path, first_match);
			//cv::imshow("first_match ", first_match);
			//cv::waitKey(0);
		}
	}
	#endif
	
	//////////////////////////////////////////////////////////////////////////////
	//Usage:
	for (const auto image_id : image_ids) {
		const Image & image = data_container.GetImage(image_id);
		images[image_id] = image;
		
		const FeatureKeypoints& keypoints = data_container.GetKeypoints(image_id);
        const std::vector<Eigen::Vector2d> points = FeatureKeypointsToPointsVector(keypoints);
        // images[image_id].SetPoints2D(points);
        images[image_id].SetPoints2D(keypoints);

		const Camera & camera = data_container.GetCamera(image.CameraId());
        if (!scene_graph_container->ExistsCamera(image.CameraId())) {
			cameras[image.CameraId()] = camera;
		}

		images[image_id].SetNumObservations(
            scene_graph_container->CorrespondenceGraph()->NumObservationsForImage(
                                                    image_id));
        images[image_id].SetNumCorrespondences(
            scene_graph_container->CorrespondenceGraph()->NumCorrespondencesForImage(
                                                    image_id));
	}

	scene_graph_container->CorrespondenceGraph()->Finalize();

	// PrintHeading1("Test 1");
	// auto incremental_options = std::make_shared<IndependentMapperOptions>();
	// IncrementalMapperController incremental_mapper(incremental_options,
	// 												"",
	// 												scene_graph_container,
	// 												reconstruction_manager);
	// incremental_mapper.Start();
	// incremental_mapper.Wait();

	PrintHeading1("Test 2");
	auto options = std::make_shared<MapperOptions>();
	options->mapper_type = MapperType::INDEPENDENT;
	options->independent_mapper_options.independent_mapper_type =
			IndependentMapperType::INCREMENTAL;
  	options->independent_mapper_options.ba_global_use_pba=false;
	options->independent_mapper_options.ba_refine_focal_length=true;
	options->independent_mapper_options.ba_refine_extra_params=true;
	options->independent_mapper_options.ba_refine_principal_point=false;
	options->independent_mapper_options.write_binary_model = true;

	MapperController *mapper = MapperController::Create(
			options, workspace_path, img_path,
			std::make_shared<FeatureDataContainer>(data_container),
			scene_graph_container, reconstruction_manager);
	mapper->Start();
	mapper->Wait();

	for (size_t i = 0; i < reconstruction_manager->Size(); ++i) {
		std::string rec_path = StringPrintf("%s/%d", workspace_path.c_str(), i);
		if (boost::filesystem::exists(rec_path)) {
			boost::filesystem::remove_all(rec_path);
		}
        boost::filesystem::create_directories(rec_path);
		reconstruction_manager->Get(i)->WriteReconstruction(rec_path,
			options->independent_mapper_options.write_binary_model);
	}

	return 0;
}