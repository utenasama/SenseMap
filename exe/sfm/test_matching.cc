// Copyright (c) 2019, SenseTime Group.
// All rights reserved.

#include <Eigen/Dense>
#include <iostream>
#include <opencv2/opencv.hpp>

#include "base/camera.h"
#include "base/image.h"
#include "base/image_reader.h"
#include "feature/extraction.h"
#include "feature/feature_matcher.h"
#include "util/misc.h"

using namespace sensemap;

int main(int argc, char *argv[]) {
    using namespace sensemap;

    PrintHeading("SenseMap.  Copyright(c) 2019, SenseTime Group.");

    std::string workspace_path(argv[1]);
    std::string vocab_path(argv[2]);
    std::string image_path;
    if (argc >= 4) {
        image_path = argv[3];
    }
    FeatureDataContainer data_container;

    // data_container.ReadCameras(workspace_path + "/cameras.txt");
    data_container.ReadCamerasBinaryData(workspace_path + "/cameras.bin");
    // data_container.ReadImagesData(workspace_path + "/features.txt");
    data_container.ReadImagesBinaryData(workspace_path + "/features.bin");

    //////////////////////////////////////////////////////////////////////////////
    // Test:
    auto camera = data_container.GetCamera(1);
    std::cout << StringPrintf("  Dimensions:      %d x %d", camera.Width(), camera.Height()) << std::endl;
    std::cout << StringPrintf("  Camera:          #%d - %s", camera.CameraId(), camera.ModelName().c_str())
              << std::endl;
    std::cout << StringPrintf("  Focal Length:    %.2fpx", camera.MeanFocalLength()) << std::endl;

#if 0
	auto ids = data_container.GetImageIds();
	for(auto id : ids)
	{
		std::cout<<id<<std::endl;
		auto image = data_container.GetImage(id);

		const std::string input_image_path = JoinPaths(data_container.GetImagePath(),
		                                               image.Name());
		Bitmap bitmap;
		if(bitmap.Read(input_image_path)) {
			std::cout << bitmap.Width() << " " << bitmap.Height() << std::endl;
		}
		////////////////////////////////////////////////////////////////////////////
		//draw keypoints
		cv::Mat mat = cv::imread(input_image_path);
		auto keypoints  = data_container.GetKeypoints(id);
		std::vector<cv::KeyPoint> keypoints_show;
		for(auto keypoint : keypoints) {
			keypoints_show.emplace_back(keypoint.x, keypoint.y,
			                            keypoint.ComputeScale() * 10,
			                            keypoint.ComputeOrientation());
		}
		drawKeypoints(mat, keypoints_show, mat, cv::Scalar::all(-1),
		              cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
		const std::string ouput_image_path = JoinPaths(
				workspace_path + "/features", image.Name());
		cv::imwrite(ouput_image_path, mat);
//		cv::imshow("test", mat);
//		cv::waitKey(0);
	}
#endif
    //////////////////////////////////////////////////////////////////////////////
    // Usage:

    FeatureMatchingOptions options;
    // options.method_ = FeatureMatchingOptions::MatchMethod::SEQUENTIAL;
    // options.method_ = FeatureMatchingOptions::MatchMethod::EXHAUSTIVE;
    // options.method_ = FeatureMatchingOptions::MatchMethod::VOCABTREE;
    options.method_ = FeatureMatchingOptions::MatchMethod::SEQUENTIAL;
    // options.vocabtree_matching_.vocab_tree_path = vocab_path;
    // options.vocabtree_matching_.num_images=50;

    options.sequential_matching_.loop_detection_num_threads = 4;
    options.sequential_matching_.loop_detection = true;
    options.sequential_matching_.loop_detection_period = 2;
    options.sequential_matching_.overlap = 15;
    options.sequential_matching_.vocab_tree_path = vocab_path;
    options.sequential_matching_.loop_detection_num_images = 10;

    options.pair_matching_.num_threads = -1;
    options.pair_matching_.use_gpu = true;
    options.pair_matching_.gpu_index = "0";
    options.pair_matching_.guided_matching = false;
    options.pair_matching_.max_num_matches = 16000;

    MatchDataContainer match_data;
    SceneGraphContainer scene_graph;

    FeatureMatcher matcher(options, &data_container, &match_data, &scene_graph);
    std::cout << "matching ....." << std::endl;
    matcher.Run();
    std::cout << "matching done" << std::endl;
    std::cout << "build graph" << std::endl;
    matcher.BuildSceneGraph();
    std::cout << "build graph done" << std::endl;

    scene_graph.WriteSceneGraphData(workspace_path + "/scene_graph.txt");

    // draw matches
#if 0

	if (!boost::filesystem::exists(workspace_path + "/matches")) {
		boost::filesystem::create_directories(workspace_path + "/matches");
	}
    int write_count = 0;    
	for(image_t id1 = 1; id1 < image_ids.size(); id1++) {
        std::cout << "Image#" << id1 << std::endl;
		for(image_t id2 = id1 + 1; id2 <= image_ids.size(); id2++) {
			auto matches =scene_graph.CorrespondenceGraph()->
					FindCorrespondencesBetweenImages(id1,id2);
			if(matches.empty()){
				continue;
			}
			std::string image_path = param.GetArgument("image_path","");

			auto image1 = feature_data_container.GetImage(id1);
			auto image2 = feature_data_container.GetImage(id2);
			const std::string input_image_path1 =
					JoinPaths(image_path, image1.Name());
			const std::string input_image_path2 =
					JoinPaths(image_path, image2.Name());
			std::cout<<input_image_path1<<std::endl;
			cv::Mat mat1 = cv::imread(input_image_path1);
			cv::Mat mat2 = cv::imread(input_image_path2);
			std::vector<cv::KeyPoint> keypoints_show1, keypoints_show2;
			std::vector<cv::DMatch> matches_show;
			for(int i = 0; i < matches.size(); ++i){
				auto keypoint  =
						feature_data_container.GetKeypoints(id1)[matches[i].point2D_idx1];
				keypoints_show1.emplace_back(keypoint.x, keypoint.y,
				                             keypoint.ComputeScale(),
				                             keypoint.ComputeOrientation());
				keypoint  = feature_data_container.GetKeypoints(id2)[matches[i].point2D_idx2];
				keypoints_show2.emplace_back(keypoint.x, keypoint.y,
				                             keypoint.ComputeScale(),
				                             keypoint.ComputeOrientation());
				matches_show.emplace_back(i, i, 1);
			}
			
            // cv::Mat first_match;
			// cv::drawMatches(mat1, keypoints_show1, mat2, keypoints_show2,
			//                 matches_show, first_match);
			// const std::string ouput_image_path = JoinPaths(
			// 		workspace_path + "/matches/",
			// 		std::to_string(id1)+ "_" + std::to_string(id2)+".jpg");
            // std::cout<<"match path: "<<ouput_image_path<<std::endl;
			// cv::imwrite(ouput_image_path, first_match);
            
		    std::cout<<"keypoints_show1 size: "<<keypoints_show1.size()<<std::endl;
            cv::Mat first_match;
			cv::drawKeypoints(mat1,keypoints_show1,first_match);

			std::string ouput_image_path = JoinPaths(
					workspace_path + "/matches",
					image1.Name()+ "+" + image2.Name()+"_1.jpg");
			cv::imwrite(ouput_image_path, first_match);

            cv::Mat second_match;
			cv::drawKeypoints(mat2,keypoints_show2,second_match);

			ouput_image_path = JoinPaths(
					workspace_path + "/matches",
					image1.Name()+ "+" + image2.Name()+"_2.jpg");
			cv::imwrite(ouput_image_path, second_match);
            write_count++;
            if(write_count>=100) break;
		}
	}
#endif

    return 0;
}