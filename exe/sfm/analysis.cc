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

std::string image_path;
std::string workspace_path;
std::string vocab_path;

FILE *fs;

void FeatureExtraction(sensemap::FeatureDataContainer &feature_data_container) {
    using namespace sensemap;

    ImageReaderOptions reader_options;
    reader_options.image_path = image_path;
    reader_options.single_camera = false;
    // reader_options.single_camera_per_folder = true;
    reader_options.fixed_camera = false;
    // reader_options.camera_model = "PINHOLE";
    // reader_options.camera_params = "1117.956055,;1117.468384,;730.657471,;550.304504";
    reader_options.camera_model = "SIMPLE_RADIAL";
    // reader_options.camera_params = "1117.956055,;730.657471,;550.304504,;0.0";
    // reader_options.camera_params = "1152.000000, 960.000000, 540.000000,0.000000";//Church

    SiftExtractionOptions sift_extraction;
    sift_extraction.num_threads = -1;
    sift_extraction.use_gpu = true;

    SiftFeatureExtractor feature_extractor(reader_options, sift_extraction, &feature_data_container);
    feature_extractor.Start();
    feature_extractor.Wait();

    fprintf(
        fs, "%s\n",
        StringPrintf("Feature Extraction Elapsed time: %.3f [minutes]", feature_extractor.GetTimer().ElapsedMinutes())
            .c_str());
    fflush(fs);
}

void FeatureMatching(sensemap::FeatureDataContainer &feature_data_container,
                     sensemap::SceneGraphContainer &scene_graph) {
    using namespace sensemap;
    using namespace std::chrono;

    high_resolution_clock::time_point start_time = high_resolution_clock::now();

    FeatureMatchingOptions options;
    // options.method_ = FeatureMatchingOptions::MatchMethod::EXHAUSTIVE;
    options.method_ = FeatureMatchingOptions::MatchMethod::VOCABTREE;
    // options.method_ = FeatureMatchingOptions::MatchMethod::SEQUENTIAL;
    options.vocabtree_matching_.vocab_tree_path = vocab_path;
    options.vocabtree_matching_.num_images = 20;

    options.sequential_matching_.vocab_tree_path = vocab_path;
    options.sequential_matching_.loop_detection_num_threads = 4;
    options.sequential_matching_.loop_detection = false;
    options.sequential_matching_.loop_detection_period = 1;
    options.sequential_matching_.overlap = 15;

    options.pair_matching_.num_threads = -1;
    options.pair_matching_.use_gpu = true;
    options.pair_matching_.gpu_index = "-1";
    options.pair_matching_.guided_matching = true;
    options.pair_matching_.max_num_matches = 20000;
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
}

void ClusterMapper(const std::shared_ptr<sensemap::FeatureDataContainer> &feature_data_container,
                   const std::shared_ptr<sensemap::SceneGraphContainer> &scene_graph_container,
                   std::shared_ptr<sensemap::ReconstructionManager> &reconstruction_manager) {
    using namespace sensemap;

    PrintHeading1("Cluster Mapping");

    auto options = std::make_shared<MapperOptions>();
    options->mapper_type = MapperType::CLUSTER;
    options->cluster_mapper_options.mapper_options.ba_refine_principal_point = false;
    options->cluster_mapper_options.mapper_options.ba_refine_focal_length = true;
    options->cluster_mapper_options.mapper_options.ba_refine_extra_params = true;
    options->cluster_mapper_options.mapper_options.ba_global_use_pba = true;

    options->cluster_mapper_options.mapper_options.write_binary_model = true;

    MapperController *mapper = MapperController::Create(options, workspace_path, image_path, feature_data_container,
                                                        scene_graph_container, reconstruction_manager);
    mapper->Start();
    mapper->Wait();

    fprintf(fs, "%s\n",
            StringPrintf("Cluster Mapper Elapsed time: %.3f [minutes]", mapper->GetTimer().ElapsedMinutes()).c_str());
    fflush(fs);

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
    options->independent_mapper_options.write_binary_model = true;
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

    fs = fopen((workspace_path + "/time.txt").c_str(), "w");

    PrintHeading("SenseMap.  Copyright(c) 2019, SenseTime Group.");

    auto reconstruction_manager = std::make_shared<ReconstructionManager>();
    auto scene_graph_container = std::make_shared<SceneGraphContainer>();
    auto feature_data_container = std::make_shared<FeatureDataContainer>();

    FeatureExtraction(*feature_data_container.get());

    FeatureMatching(*feature_data_container.get(), *scene_graph_container.get());

    ClusterMapper(feature_data_container, scene_graph_container, reconstruction_manager);

    fclose(fs);
    return 0;
}