// Copyright (c) 2019, SenseTime Group.
// All rights reserved.

#include <dirent.h>
#include <sys/stat.h>

#include <boost/filesystem/path.hpp>
#include <opencv2/opencv.hpp>
#include <unordered_set>

#include "../Configurator_yaml.h"
#include "../camera_rig_params.h"
#include "../option_parsing.h"
#include "../system_io.h"
#include "container/feature_data_container.h"
#include "feature/extraction.h"
#include "feature/feature_matcher.h"
#include "feature/utils.h"
#include "graph/correspondence_graph.h"
#include "util/mat.h"
#include "util/misc.h"
#include "util/ply.h"
#include "util/rgbd_helper.h"

#if defined(_OPENMP)
#include <omp.h>
#endif

using namespace sensemap;
FILE *fs;

bool LoadFeatures(FeatureDataContainer &feature_data_container, Configurator &param) {
    std::string workspace_path = param.GetArgument("workspace_path", "");
    CHECK(!workspace_path.empty());

    OptionParser option_parser;
    ImageReaderOptions reader_options;
    option_parser.GetImageReaderOptions(reader_options, param);

    bool camera_rig = (reader_options.num_local_cameras > 1);
    bool with_depth = reader_options.with_depth;

    std::string rgbd_parmas_file = param.GetArgument("rgbd_params_file", "");
    bool use_gps_prior = param.GetArgument("use_gps_prior", 0);

    bool have_matched = boost::filesystem::exists(JoinPaths(workspace_path, "/scene_graph.bin")) ||
                        boost::filesystem::exists(JoinPaths(workspace_path, "/scene_graph.bin"));

    bool exist_feature_file = false;
    if (boost::filesystem::exists(JoinPaths(workspace_path, "/cameras.bin")) &&
        boost::filesystem::exists(JoinPaths(workspace_path, "/features.bin"))) {
        if (boost::filesystem::exists(JoinPaths(workspace_path, "/local_cameras.bin"))) {
            feature_data_container.ReadCamerasBinaryData(JoinPaths(workspace_path, "/cameras.bin"), false);
            feature_data_container.ReadLocalCamerasBinaryData(JoinPaths(workspace_path, "/local_cameras.bin"));
        } else {
            feature_data_container.ReadCamerasBinaryData(JoinPaths(workspace_path, "/cameras.bin"), camera_rig);
        }
       
        feature_data_container.ReadImagesBinaryDataWithoutDescriptor(JoinPaths(workspace_path, "/features.bin"));
        
        exist_feature_file = true;
    }

    // If the camera model is Spherical
    if (exist_feature_file) {
        if (boost::filesystem::exists(JoinPaths(workspace_path, "/sub_panorama.bin"))) {
            feature_data_container.ReadSubPanoramaBinaryData(JoinPaths(workspace_path, "/sub_panorama.bin"));
        } else {
            // FIXME:
            std::cout << " Warning! Existing feature data do not contain sub_panorama data" << std::endl;
        }
    }

    if (exist_feature_file) {
        if (boost::filesystem::exists(JoinPaths(workspace_path, "/piece_indices.bin"))) {
            feature_data_container.ReadPieceIndicesBinaryData(JoinPaths(workspace_path, "/piece_indices.bin"));
        } else {
            // FIXME:
            std::cout << " Warning! Existing feature data do not contain piece_indices data" << std::endl;
        }
    }

    // Check the AprilTag detection file exist or not
    if (exist_feature_file && static_cast<bool>(param.GetArgument("detect_apriltag", 0))) {
        if (boost::filesystem::exists(JoinPaths(workspace_path, "/apriltags.bin"))) {
            feature_data_container.ReadAprilTagBinaryData(JoinPaths(workspace_path, "/apriltags.bin"));
        } else {
            // FIXME:
            std::cout << " Warning! Existing feature data do not contain AprilTags data" << std::endl;
        }
    }

    // Check the GPS file exist or not.
    if (exist_feature_file && use_gps_prior) {
        if (boost::filesystem::exists(JoinPaths(workspace_path, "/gps.bin"))) {
            feature_data_container.ReadGPSBinaryData(JoinPaths(workspace_path, "/gps.bin"));
        }
    }

    if (exist_feature_file) {
        return true;
    } else {
        return false;
    }
}

bool LoadGlobalFeatures(FeatureDataContainer &feature_data_container, Configurator &param){
    std::string workspace_path = param.GetArgument("workspace_path", "");
    CHECK(!workspace_path.empty());

    if (boost::filesystem::exists(JoinPaths(workspace_path, "/vlad_vectors.bin"))){
        feature_data_container.ReadGlobalFeaturesBinaryData(workspace_path + "/vlad_vectors.bin");
        return true;
    }
    else{
        return false;
    }
}


void RetrievalTest(FeatureDataContainer &feature_data_container, SceneGraphContainer &scene_graph,
                     Configurator &param) {
    using namespace std::chrono;

    OptionParser option_parser;
    ImageReaderOptions reader_options;
    option_parser.GetImageReaderOptions(reader_options, param);

    std::string workspace_path = param.GetArgument("workspace_path", "");
    CHECK(!workspace_path.empty());

  
    FeatureMatchingOptions options;
    option_parser.GetFeatureMatchingOptions(options, param);


    high_resolution_clock::time_point start_time = high_resolution_clock::now();

    SequentialMatchingOptions sqm_options = options.SequentialMatching();

  
    VladVisualIndex vlad_visual_index;
    vlad_visual_index.LoadCodeBook(sqm_options.vlad_code_book_path);
    
    Timer timer;
    timer.Start();
    std::cout<<"Load vlad vectors from data container to vlad visual index:"<<std::endl;
    std::vector<image_t> image_ids = feature_data_container.GetImageIds();
    //Load vlad vectors from data container to vlad visual index
    for(int i = 0; i < image_ids.size(); ++i){
        image_t current_id = image_ids[i];        
        const auto& vlad = feature_data_container.GetVladVector(current_id);
        vlad_visual_index.ADD(vlad,current_id);
    }
    std::cout << StringPrintf("Indexing in %.3f seconds", timer.ElapsedSeconds()) << std::endl;

    VladVisualIndex::QueryOptions query_option;
    query_option.max_num_images = sqm_options.loop_detection_num_images;


    std::vector<std::vector<retrieval::ImageScore>> image_image_scores(image_ids.size());
    timer.Start();
    std::cout << "Query " << std::endl;
// #ifdef _OPENMP
// #pragma omp parallel for schedule(static)
// #endif
    for (int i = 0; i < image_ids.size(); ++i) {
        image_t current_id = image_ids[i];
        /// query by voc tree
        std::cout<<"query ["<<i<<"/"<<image_ids.size()<<"] image"<<std::endl;
        std::vector<retrieval::ImageScore> image_scores;

        const VladVisualIndex::VLAD& current_vlad = feature_data_container.GetVladVector(current_id); 
        vlad_visual_index.Query(query_option,current_vlad,&image_scores);

        { image_image_scores[i].swap(image_scores); }
    }

    std::cout << StringPrintf("Query in %.3fmin", timer.ElapsedMinutes()) << std::endl;


    high_resolution_clock::time_point end_time = high_resolution_clock::now();
    fprintf(fs, "%s\n",
            StringPrintf("Image Retrieval Elapsed time: %.3f [minutes]",
                         duration_cast<microseconds>(end_time - start_time).count() / 6e7)
                .c_str());
    fflush(fs);
}

int main(int argc, char *argv[]) {
    using namespace sensemap;

    PrintHeading("SenseMap.  Copyright(c) 2019, SenseTime Group.");
    PrintHeading("Version: sfm-feature-match-1.6.7");
    Timer timer;
    timer.Start();

    std::string configuration_file_path = std::string(argv[1]);
    Configurator param;
    param.Load(configuration_file_path.c_str());

    std::string workspace_path = param.GetArgument("workspace_path", "");
    CHECK(!workspace_path.empty()) << "workspace path empty";

    std::string image_path = param.GetArgument("image_path", "");
    CHECK(!image_path.empty()) << "image path empty";

    if (!boost::filesystem::exists(workspace_path)) {
        CHECK(boost::filesystem::create_directories(workspace_path)) << "Create workspace failed";
    }

    fs = fopen((workspace_path + "/time_feature_match.txt").c_str(), "w");

    auto scene_graph_container = std::make_shared<SceneGraphContainer>();
    auto feature_data_container = std::make_shared<FeatureDataContainer>();

    CHECK(LoadFeatures(*feature_data_container.get(), param)) << "Load features failed";
    CHECK(LoadGlobalFeatures(*feature_data_container.get(), param)) << "Load global features failed";

    RetrievalTest(*feature_data_container.get(), *scene_graph_container.get(), param);


    fclose(fs);
    return 0;
}