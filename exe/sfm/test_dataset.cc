//
// Created by sh on 3/28/19.
//

// Copyright (c) 2019, SenseTime Group.
// All rights reserved.

#include <cluster/fast_community.h>
#include <boost/filesystem/path.hpp>
#include <iostream>
#include <opencv2/opencv.hpp>

#include "base/reconstruction_manager.h"
#include "container/feature_data_container.h"
#include "controllers/cluster_mapper_controller.h"
#include "feature/extraction.h"
#include "feature/feature_matcher.h"
#include "feature/utils.h"
#include "graph/correspondence_graph.h"
#include "graph/maximum_spanning_tree_graph.h"
#include "util/misc.h"

std::string image_path;
std::string workspace_path;
std::string vocab_path;

using namespace sensemap;

void test(const int &method_id, std::shared_ptr<sensemap::FeatureDataContainer> &feature_data_container)

{
    std::string method;

    auto reconstruction_manager = std::make_shared<ReconstructionManager>();
    auto scene_graph_container = std::make_shared<SceneGraphContainer>();
    // FeatureMatching
    FeatureMatchingOptions options;
    switch (method_id) {
        case 0: {
            method = std::string("seq_check");
            options.method_ = FeatureMatchingOptions::MatchMethod::SEQUENTIAL;
            break;
        }
        case 1: {
            method = std::string("seq");
            options.method_ = FeatureMatchingOptions::MatchMethod::SEQUENTIAL;
            break;
        }
        case 2: {
            method = std::string("exh");
            options.method_ = FeatureMatchingOptions::MatchMethod::EXHAUSTIVE;
            break;
        }
        default: {
            std::cout << "Warning: unknown method !" << std::endl;
            return;
        }
    }
    PrintHeading1(method);

    if (method_id == 0) {
        options.global_triplet_checking = true;
    }

    options.sequential_matching_.loop_detection_num_threads = -1;
    options.sequential_matching_.robust_loop_detection = true;
    options.sequential_matching_.vocab_tree_path = vocab_path;
    options.sequential_matching_.loop_detection_period = 1;
    options.sequential_matching_.overlap = 10;
    options.sequential_matching_.max_recent_score_factor = 0.6f;
    options.sequential_matching_.best_acc_score_factor = 0.5f;
    options.sequential_matching_.loop_consistency_threshold = 3;

    options.pair_matching_.num_threads = -1;
    options.pair_matching_.use_gpu = true;
    options.pair_matching_.guided_matching = true;
    options.pair_matching_.max_num_matches = 20000;

    MatchDataContainer match_data;
    FeatureMatcher matcher(options, feature_data_container.get(), &match_data, scene_graph_container.get());
    std::cout << method << ": matching ....." << std::endl;
    matcher.Run();
    std::cout << method << ": build graph" << std::endl;
    matcher.BuildSceneGraph();

    auto correspondence_graph = scene_graph_container->CorrespondenceGraph();

    std::cout << method << ": ExportToGraph" << std::endl;
    correspondence_graph->ExportToGraph(workspace_path + "/" + method + "_scene_graph.png");

    //	///Mst
    //	std::unordered_map<image_t, Eigen::Vector3d> orientations;
    //	std::cout<< method << ": MST Detection..." << std::endl;
    //	OrientationsFromMaximumSpanningTree(*correspondence_graph, &orientations);
    //
    //	///community
    //	std::cout<< method << ": Community Detection..." << std::endl;
    //	std::unordered_map<image_pair_t, point2D_t> corrs_between_images =
    //			correspondence_graph->NumCorrespondencesBetweenImages();

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

    //	std::vector<std::vector<image_t> > communities;
    //	fastcommunity::CommunityDetection(
    //		scene_graph_container->CorrespondenceGraph(), communities);
    //
    //	boost::filesystem::remove(workspace_path + "/"
    //	                          + method + "_graph_structure.dot");
    //	boost::filesystem::copy_file("./graph.dot",
    //	                             workspace_path + "/"
    //	                             + method + "_graph_structure.dot");
    //	boost::filesystem::remove("./graph.dot");

    // Incremental sfm
    std::cout << method << ": Incremental SfM..." << std::endl;
    auto sfm_options = std::make_shared<MapperOptions>();
    sfm_options->mapper_type = MapperType::INDEPENDENT;
    sfm_options->independent_mapper_options.independent_mapper_type = IndependentMapperType::INCREMENTAL;
    MapperController *mapper = MapperController::Create(sfm_options, workspace_path, image_path, feature_data_container,
                                                        scene_graph_container, reconstruction_manager);
    mapper->Start();
    mapper->Wait();

    for (size_t i = 0; i < reconstruction_manager->Size(); ++i) {
        std::string rec_path = StringPrintf("%s/%s_%d", workspace_path.c_str(), method.c_str(), i);
        if (boost::filesystem::exists(rec_path)) {
            boost::filesystem::remove_all(rec_path);
        }
        boost::filesystem::create_directories(rec_path);
        reconstruction_manager->Get(i)->WriteReconstruction(rec_path,
                                                            sfm_options->independent_mapper_options.write_binary_model);
    }
}

int main(int argc, char *argv[]) {
    using namespace sensemap;

    image_path = std::string(argv[1]);
    workspace_path = std::string(argv[2]);
    vocab_path = std::string(argv[3]);

    PrintHeading("SenseMap.  Copyright(c) 2019, SenseTime Group.");

    auto feature_data_container = std::make_shared<FeatureDataContainer>();

    // FeatureExtraction
    ImageReaderOptions reader_options;
    reader_options.image_path = image_path;
    reader_options.single_camera = false;
    reader_options.single_camera_per_folder = false;
    // reader_options.fixed_camera = true;
    //	reader_options.camera_model = "PINHOLE";
    //	reader_options.camera_params = "1117.956055,;1117.468384,;730.657471,;550.304504";
    SiftExtractionOptions sift_extraction;
    sift_extraction.num_threads = -1;
    sift_extraction.use_gpu = true;
    SiftFeatureExtractor feature_extractor(reader_options, sift_extraction, feature_data_container.get());
    feature_extractor.Start();
    feature_extractor.Wait();

    test(0, feature_data_container);
    // test(1,feature_data_container);
    // test(2,feature_data_container);

    return 0;
}
