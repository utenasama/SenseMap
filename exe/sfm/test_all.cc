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

#include "container/feature_data_container.h"
#include "feature/extraction.h"
#include "feature/feature_matcher.h"
#include "feature/utils.h"

#include "controllers/cluster_mapper_controller.h"

using namespace sensemap;

std::string image_path;
std::string workspace_path;
std::string vocab_path;

bool dirExists(const std::string &dirName_in) { return access(dirName_in.c_str(), F_OK) == 0; }

void FeatureExtraction(FeatureDataContainer &feature_data_container) {
    ImageReaderOptions reader_options;
    reader_options.image_path = image_path;
    reader_options.single_camera = false;
    reader_options.single_camera_per_folder = false;
    reader_options.fixed_camera = false;
    // reader_options.camera_model = "PINHOLE";
    // reader_options.camera_params = "1117.956055,;1117.468384,;730.657471,;550.304504";
    reader_options.camera_model = "SIMPLE_RADIAL";
    // reader_options.camera_params = "1117.956055,;730.657471,;550.304504,;0.0";
    // reader_options.camera_params = "1152.000000, 960.000000, 540.000000,0.000000";//Church
    // reader_options.camera_params = "1520.000000, 960, 540, 0.000000";//guobo
    // reader_options.camera_params = "1216, 960, 540, 0.000000";//guobo_new
    // reader_options.camera_params = "1117.956055,730.657471,550.304504,0.0"; //shanxi
    // reader_options.camera_params = "1820, 540.0, 960.0, 0.000000";//jiangcun

    SiftExtractionOptions sift_extraction;
    sift_extraction.num_threads = -1;
    sift_extraction.use_gpu = true;
    sift_extraction.peak_threshold = 0.02 / sift_extraction.octave_resolution;

    SiftFeatureExtractor feature_extractor(reader_options, sift_extraction, &feature_data_container);
    feature_extractor.Start();
    feature_extractor.Wait();

    // feature_data_container.WriteCameras(workspace_path + "/cameras.txt");
    // feature_data_container.WriteImagesData(workspace_path + "/features.txt");

#if 0
	auto ids = feature_data_container.GetImageIds();
	for(auto id : ids)
	{
		std::cout<<id<<std::endl;
		auto image = feature_data_container.GetImage(id);

		const std::string input_image_path = JoinPaths(feature_data_container.GetImagePath(),
		                                               image.Name());
		Bitmap bitmap;
		if(bitmap.Read(input_image_path)) {
			std::cout << bitmap.Width() << " " << bitmap.Height() << std::endl;
		}
		////////////////////////////////////////////////////////////////////////////
		//draw keypoints
		cv::Mat mat = cv::imread(input_image_path);
		auto keypoints  = feature_data_container.GetKeypoints(id);
		std::vector<cv::KeyPoint> keypoints_show;
		for(auto keypoint : keypoints) {
			keypoints_show.emplace_back(keypoint.x, keypoint.y,
			                            keypoint.ComputeScale() * 10,
			                            keypoint.ComputeOrientation());
		}
		drawKeypoints(mat, keypoints_show, mat, cv::Scalar::all(-1),
		              cv::DrawMatchesFlags::DEFAULT);
		const std::string ouput_image_path = JoinPaths(
				workspace_path + "/features", image.Name());
		cv::imwrite(ouput_image_path, mat);
	}
#endif
}

void FeatureMatching(FeatureDataContainer &feature_data_container, SceneGraphContainer &scene_graph) {
    FeatureMatchingOptions options;
    // options.method_ = FeatureMatchingOptions::MatchMethod::EXHAUSTIVE;
    // options.method_ = FeatureMatchingOptions::MatchMethod::VOCABTREE;
    options.method_ = FeatureMatchingOptions::MatchMethod::SEQUENTIAL;
    options.vocabtree_matching_.vocab_tree_path = vocab_path;
    options.vocabtree_matching_.num_images = 50;
    options.vocabtree_matching_.num_nearest_neighbors = 50;
    options.vocabtree_matching_.num_checks = 128;

    options.sequential_matching_.vocab_tree_path = vocab_path;
    options.sequential_matching_.loop_detection_num_threads = 8;
    options.sequential_matching_.loop_detection = false;
    options.sequential_matching_.robust_loop_detection = true;
    options.sequential_matching_.loop_detection_period = 1;
    options.sequential_matching_.loop_detection_num_images = 50;
    options.sequential_matching_.overlap = 20;

    options.pair_matching_.num_threads = -1;
    options.pair_matching_.use_gpu = true;
    options.pair_matching_.gpu_index = "-1";
    options.pair_matching_.guided_matching = true;
    options.pair_matching_.multiple_models = false;
    options.pair_matching_.guided_matching_multi_homography = false;
    options.pair_matching_.max_num_matches = 20000;
    MatchDataContainer match_data;
    FeatureMatcher matcher(options, &feature_data_container, &match_data, &scene_graph);
    std::cout << "matching ....." << std::endl;
    matcher.Run();
    std::cout << "matching done" << std::endl;
    std::cout << "build graph" << std::endl;
    matcher.BuildSceneGraph();
    std::cout << "build graph done" << std::endl;

    scene_graph.CorrespondenceGraph()->ExportToGraph(workspace_path + "/scene_graph.png");
    std::cout << "ExportToGraph done!" << std::endl;
    scene_graph.WriteSceneGraphData(workspace_path + "/scene_graph.txt");
    // feature_data_container.WriteImagesData(workspace_path + "/features_new.txt");
}

void ClusterMapper(const std::shared_ptr<sensemap::FeatureDataContainer> &feature_data_container,
                   const std::shared_ptr<sensemap::SceneGraphContainer> &scene_graph_container,
                   std::shared_ptr<sensemap::ReconstructionManager> &reconstruction_manager) {
    using namespace sensemap;

    PrintHeading1("Cluster Mapping");

    auto options = std::make_shared<MapperOptions>();
    options->mapper_type = MapperType::CLUSTER;
    options->cluster_mapper_options.clustering_options.image_overlap = 0;
    options->cluster_mapper_options.clustering_options.leaf_max_num_images = 300;
    options->cluster_mapper_options.mapper_options.ba_refine_principal_point = false;
    options->cluster_mapper_options.mapper_options.ba_refine_focal_length = true;
    options->cluster_mapper_options.mapper_options.ba_refine_extra_params = true;
    options->cluster_mapper_options.mapper_options.ba_global_use_pba = false;
    options->cluster_mapper_options.mapper_options.write_binary_model = true;

    MapperController *mapper = MapperController::Create(options, workspace_path, image_path, feature_data_container,
                                                        scene_graph_container, reconstruction_manager);
    mapper->Start();
    mapper->Wait();

    for (size_t i = 0; i < reconstruction_manager->Size(); ++i) {
        std::string rec_path = StringPrintf("%s/%d", workspace_path.c_str(), i);
        if (boost::filesystem::exists(rec_path)) {
            boost::filesystem::remove_all(rec_path);
        }
        boost::filesystem::create_directories(rec_path);
        reconstruction_manager->Get(i)->WriteReconstruction(
            rec_path, options->cluster_mapper_options.mapper_options.write_binary_model);
    }
}

void IncrementalSFM(const std::shared_ptr<sensemap::FeatureDataContainer> &feature_data_container,
                    const std::shared_ptr<sensemap::SceneGraphContainer> &scene_graph_container,
                    std::shared_ptr<sensemap::ReconstructionManager> &reconstruction_manager) {
    using namespace sensemap;

    PrintHeading1("Incremental Mapping");

    auto options = std::make_shared<MapperOptions>();
    options->mapper_type = MapperType::INDEPENDENT;
    options->independent_mapper_options.independent_mapper_type = IndependentMapperType::INCREMENTAL;
    options->independent_mapper_options.ba_global_use_pba = false;
    options->independent_mapper_options.ba_refine_focal_length = true;
    options->independent_mapper_options.ba_refine_extra_params = true;
    options->independent_mapper_options.ba_refine_principal_point = false;

    MapperController *mapper = MapperController::Create(options, workspace_path, image_path, feature_data_container,
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
}

int main(int argc, char *argv[]) {
    using namespace sensemap;

    image_path = std::string(argv[1]);
    workspace_path = std::string(argv[2]);
    vocab_path = std::string(argv[3]);
    PrintHeading("SenseMap.  Copyright(c) 2019, SenseTime Group.");

    auto reconstruction_manager = std::make_shared<ReconstructionManager>();
    auto scene_graph_container = std::make_shared<SceneGraphContainer>();
    auto feature_data_container = std::make_shared<FeatureDataContainer>();

    FeatureExtraction(*feature_data_container.get());

    FeatureMatching(*feature_data_container.get(), *scene_graph_container.get());
    /*
ClusterMapper(feature_data_container,
              scene_graph_container,
              reconstruction_manager);*/

    IncrementalSFM(feature_data_container, scene_graph_container, reconstruction_manager);

    return 0;
}