// Copyright (c) 2021, SenseTime Group.
// All rights reserved.

#include <boost/filesystem/path.hpp>
#include <opencv2/opencv.hpp>

#include "base/camera.h"
#include "base/image.h"
#include "base/image_reader.h"

#include "util/misc.h"
#include "util/mat.h"
#include "util/rgbd_helper.h"

#include "container/feature_data_container.h"
#include "feature/extraction.h"
#include "feature/feature_matcher.h"
#include "feature/utils.h"


#include "../Configurator_yaml.h"
#include "../camera_rig_params.h"
#include "../option_parsing.h"

#include <dirent.h>
#include <sys/stat.h>

#include <unordered_set>
#include "../system_io.h"

using namespace sensemap;
FILE *fs;

typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> DoubleDescriptors;

void FeatureExtraction(FeatureDataContainer &feature_data_container, Configurator &param) {
    std::string workspace_path = param.GetArgument("workspace_path", "");
    CHECK(!workspace_path.empty());

    OptionParser option_parser;
    ImageReaderOptions reader_options;
    option_parser.GetImageReaderOptions(reader_options,param);

    bool camera_rig = (reader_options.num_local_cameras > 1);
    bool with_depth = reader_options.with_depth;

    std::string rgbd_parmas_file = param.GetArgument("rgbd_params_file", "");

    bool exist_feature_file = false;
    if (boost::filesystem::exists(JoinPaths(workspace_path, "/cameras.bin")) &&
        boost::filesystem::exists(JoinPaths(workspace_path, "/features.bin"))) {
        if (boost::filesystem::exists(JoinPaths(workspace_path, "/local_cameras.bin"))) {
            feature_data_container.ReadCamerasBinaryData(JoinPaths(workspace_path, "/cameras.bin"), false);
            feature_data_container.ReadLocalCamerasBinaryData(JoinPaths(workspace_path, "/local_cameras.bin"));
        } else {
            feature_data_container.ReadCamerasBinaryData(JoinPaths(workspace_path, "/cameras.bin"), camera_rig);
        }
        feature_data_container.ReadImagesBinaryData(JoinPaths(workspace_path, "/features.bin"));

        exist_feature_file = true;
    }

    // If the camera model is Spherical
    if (exist_feature_file) {
        if (boost::filesystem::exists(JoinPaths(workspace_path, "/sub_panorama.bin"))) {
            feature_data_container.ReadSubPanoramaBinaryData(JoinPaths(workspace_path, "/sub_panorama.bin"));
        } else if (boost::filesystem::exists(JoinPaths(workspace_path, "/sub_panorama.txt"))) {
            feature_data_container.ReadSubPanoramaData(JoinPaths(workspace_path, "/sub_panorama.txt"));
        } else {
            // FIXME:
            std::cout << " Warning! Existing feature data do not contain sub_panorama data" << std::endl;
        }
    }

    if (exist_feature_file) {
        if (boost::filesystem::exists(JoinPaths(workspace_path, "/piece_indices.bin"))) {
            feature_data_container.ReadPieceIndicesBinaryData(JoinPaths(workspace_path, "/piece_indices.bin"));
        } else if (boost::filesystem::exists(JoinPaths(workspace_path, "/piece_indices.txt"))) {
            feature_data_container.ReadPieceIndicesData(JoinPaths(workspace_path, "/piece_indices.txt"));
        } else {
            // FIXME:
            std::cout << " Warning! Existing feature data do not contain piece_indices data" << std::endl;
        }
    }
    
    if (exist_feature_file) {
        return;
    }

    SiftExtractionOptions sift_extraction;
    option_parser.GetFeatureExtractionOptions(sift_extraction,param);

    std::string panorama_config_file = param.GetArgument("panorama_config_file", "");
    if(!panorama_config_file.empty()){
        std::vector<PanoramaParam> panorama_params;
        LoadParams(panorama_config_file, panorama_params);
        sift_extraction.panorama_config_params = panorama_params;   
        sift_extraction.use_panorama_config = true;             
    }

    SiftFeatureExtractor feature_extractor(reader_options, sift_extraction, &feature_data_container);
    feature_extractor.Start();
    feature_extractor.Wait();

    fprintf(
        fs, "%s\n",
        StringPrintf("Feature Extraction Elapsed time: %.3f [minutes]", feature_extractor.GetTimer().ElapsedMinutes())
            .c_str());
    fflush(fs);

    bool write_feature = static_cast<bool>(param.GetArgument("write_feature", 0));
    bool write_binary = static_cast<bool>(param.GetArgument("write_binary", 1));
    if (write_feature) {
        if (write_binary) {
            feature_data_container.WriteImagesBinaryData(JoinPaths(workspace_path, "/features.bin"));
            feature_data_container.WriteCamerasBinaryData(JoinPaths(workspace_path, "/cameras.bin"));
            feature_data_container.WriteLocalCamerasBinaryData(JoinPaths(workspace_path, "/local_cameras.bin"));
            feature_data_container.WriteSubPanoramaBinaryData(JoinPaths(workspace_path, "/sub_panorama.bin"));

            if (reader_options.num_local_cameras == 2 && reader_options.camera_model == "OPENCV_FISHEYE" &&
                sift_extraction.convert_to_perspective_image) {
                feature_data_container.WritePieceIndicesBinaryData(JoinPaths(workspace_path, "/piece_indices.bin"));
            }
        } 
    }
}



int main(int argc, char *argv[]) {
     
    PrintHeading("SenseMap.  Copyright(c) 2019, SenseTime Group.");
    PrintHeading("Version: create-vlad-code-book-1.0");
    

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

    fs = fopen((workspace_path + "/time.txt").c_str(), "w");

    auto feature_data_container = std::make_shared<FeatureDataContainer>();

    FeatureExtraction(*feature_data_container.get(), param);


    Timer timer;
    timer.Start();


    std::cout << "Collect training descriptors " << std::endl;
    FeatureDescriptors training_descriptors;
    size_t training_descriptors_count = 0;

    const std::vector<image_t>& image_ids = feature_data_container->GetImageIds();

    int pca_training_feature_count = static_cast<int> (param.GetArgument("pca_training_feature_count", 1000000));
    int total_feature_count = 0;
    for (int i = 0; i < image_ids.size(); ++i){
        image_t current_id = image_ids[i];
        const auto &keypoints = feature_data_container->GetKeypoints(current_id); 

        total_feature_count += keypoints.size();
    }

    int sample_step = 1;
    if(pca_training_feature_count < total_feature_count){
        sample_step = total_feature_count / pca_training_feature_count;
    }

    for (int i = 0; i < image_ids.size(); ++i) {
        image_t current_id = image_ids[i];
        const auto &descriptors = feature_data_container->GetDescriptors(current_id);
        size_t sampled_descriptors_count = descriptors.rows() / sample_step;

        training_descriptors.conservativeResize(training_descriptors_count + sampled_descriptors_count, 128);

        for (size_t j = 0; j < sampled_descriptors_count; j++) {
            training_descriptors.row(training_descriptors_count + j) = descriptors.row(j * sample_step);
        }
        training_descriptors_count += sampled_descriptors_count;
    }
    

    Eigen::Matrix<double, 128, 128> pca_matrix;
    Eigen::Matrix<double, 128, 1> embedding_thresholds;
    
    PcaTraining(training_descriptors,pca_matrix,embedding_thresholds);

    std::string pca_matrix_path = param.GetArgument("pca_matrix_path", "");
    
    std::ofstream file(pca_matrix_path, std::ios::binary);
	CHECK(file.is_open()) << pca_matrix_path;

    for(int i = 0; i< 128; ++i){
        for(int j= 0; j< 128; ++j){
            double elem = pca_matrix(i,j);
            file.write((char*)&elem, sizeof(double));
        }
    }

    for(int i = 0; i < 128; ++i){
        double elem = embedding_thresholds(i);
        file.write((char*)&elem,sizeof(double));
    }

    file.close();
}