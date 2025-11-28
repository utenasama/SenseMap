// Copyright (c) 2019, SenseTime Group.
// All rights reserved.

#include <dirent.h>
#include <sys/stat.h>

#include <boost/filesystem/path.hpp>

#include "../Configurator_yaml.h"
#include "../camera_rig_params.h"
#include "../option_parsing.h"
#include "../system_io.h"
#include "base/camera.h"
#include "base/image.h"
#include "base/image_reader.h"
#include "container/feature_data_container.h"
#include "feature/extraction.h"
#include "feature/utils.h"
#include "util/gps_reader.h"
#include "util/mat.h"
#include "util/misc.h"
#include "util/rgbd_helper.h"
#include "base/version.h"

#ifdef DO_ENCRYPT_CHECK
#include "../check.h"
#endif

#define MAX_PRECISION std::setprecision(std::numeric_limits<double>::digits10 + 1)

using namespace sensemap;

FILE *fs;

bool LoadFeatures(FeatureDataContainer &feature_data_container, Configurator &param, std::string workspace_path) {
    // std::string workspace_path = param.GetArgument("workspace_path", "");
    CHECK(!workspace_path.empty());

    OptionParser option_parser;
    ImageReaderOptions reader_options;
    option_parser.GetImageReaderOptions(reader_options, param);

    bool camera_rig = (reader_options.num_local_cameras > 1);
    bool with_depth = reader_options.with_depth;


    bool use_gps_prior = param.GetArgument("use_gps_prior", 0);

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

bool FeatureExtraction(FeatureDataContainer &feature_data_container, Configurator &param) {
    std::string workspace_path = param.GetArgument("workspace_path", "");
    CHECK(!workspace_path.empty());

    OptionParser option_parser;
    ImageReaderOptions reader_options;
    option_parser.GetImageReaderOptions(reader_options, param);

    bool camera_rig = (reader_options.num_local_cameras > 1);
    bool with_depth = reader_options.with_depth;

    std::string rgbd_parmas_file = param.GetArgument("rgbd_params_file", "");
    bool use_gps_prior = param.GetArgument("use_gps_prior", 0);

    if (boost::filesystem::exists(JoinPaths(workspace_path, "/cameras.bin")) &&
        boost::filesystem::exists(JoinPaths(workspace_path, "/features.bin"))) {
        // only quit not in update mode
        if (!static_cast<bool>(param.GetArgument("map_update", 0))){
            std::cout << "Feature already exists, skip feature extraction" << std::endl;
            return false;
        }
    }

    // read original data when map update
    std::string map_update_workspace_path = JoinPaths(workspace_path, "/map_update");
    if (static_cast<bool>(param.GetArgument("map_update", 0))) {
        if(boost::filesystem::exists(map_update_workspace_path)){
            if (boost::filesystem::exists(JoinPaths(map_update_workspace_path, "/cameras.bin")) &&
                boost::filesystem::exists(JoinPaths(map_update_workspace_path, "/features.bin"))) {
                std::cout << "Updated Map Feature already exists, skip feature extraction" << std::endl;
                return false;
            }
        }else{
            boost::filesystem::create_directories(map_update_workspace_path);
        }
    }

    SiftExtractionOptions sift_extraction;
    option_parser.GetFeatureExtractionOptions(sift_extraction, param);

    std::string panorama_config_file = param.GetArgument("panorama_config_file", "");
    if (!panorama_config_file.empty()) {
        std::vector<PanoramaParam> panorama_params;
        if (LoadParams(panorama_config_file, panorama_params)) {
            sift_extraction.panorama_config_params = panorama_params;
            sift_extraction.use_panorama_config = true;
        }
    }

    //TODO: get start image_id & camera_id
    image_t start_image_id = 0;
    camera_t start_camera_id = 0;
    if (static_cast<bool>(param.GetArgument("map_update", 0))) {
        LoadFeatures(feature_data_container, param, workspace_path);
        std::string gps_origin_str = JoinPaths(workspace_path, "/gps_origin.txt");
        std::vector<double> vec_gps_origin;
        if (boost::filesystem::exists(gps_origin_str) &&
            LoadGpsOrigin(gps_origin_str, vec_gps_origin)){
            std::stringstream ss;
            ss << MAX_PRECISION << vec_gps_origin[0] << "," << vec_gps_origin[1] << "," << vec_gps_origin[2] << std::endl;
            reader_options.gps_origin = ss.str();

            std::cout << "ReaderOptions set ori_gps_origin: " << reader_options.gps_origin << std::endl;
        }
        for(const auto old_image_id: feature_data_container.GetImageIds()){
            if(start_image_id < old_image_id){
                start_image_id = old_image_id;
            }
            std::string image_name = feature_data_container.GetImage(old_image_id).Name();
            feature_data_container.GetImage(image_name).SetLabelId(0);
        }
        ++start_image_id;
        start_camera_id = feature_data_container.NumCamera();
    }


    SiftFeatureExtractor feature_extractor(reader_options, sift_extraction, &feature_data_container, start_image_id, start_camera_id);
    feature_extractor.Start();
    feature_extractor.Wait();
    fprintf(
        fs, "%s\n",
        StringPrintf("Feature Extraction Elapsed time: %.3f [minutes]", feature_extractor.GetTimer().ElapsedMinutes())
            .c_str());
    fflush(fs);

    Timer timer;

    typedef FeatureMatchingOptions::RetrieveType RetrieveType;
    RetrieveType retrieve_type = (RetrieveType)param.GetArgument("retrieve_type", 0);
    if (retrieve_type != RetrieveType::SIFT) {
        // Pca training using the extrated feature descriptors
        timer.Start();

        std::cout << "Collect training descriptors " << std::endl;
        FeatureDescriptors training_descriptors;
        size_t training_descriptors_count = 0;

        const std::vector<image_t>& whole_image_ids = feature_data_container.GetImageIds();
        const std::vector<image_t>& new_image_ids = feature_data_container.GetNewImageIds();
        std::vector<image_t> old_image_ids;
        std::set_difference(whole_image_ids.begin(), whole_image_ids.end(), new_image_ids.begin(), new_image_ids.end(),
                            std::inserter(old_image_ids,old_image_ids.begin()));

        std::cout<<"whole_image_ids: " << whole_image_ids.size() << " new_image_ids: " << new_image_ids.size() << 
                    " old_image_ids: " << old_image_ids.size() << std::endl;

        Eigen::Matrix<double, 128, 128> pca_matrix;
        Eigen::Matrix<double, 128, 1> embedding_thresholds;
        std::string pca_matrix_path = param.GetArgument("pca_matrix_path", "");

        //TODO: if exist pca matrix already & map update, read matrix
        bool load_pca = false;
        bool enable_whole_pca = true;
        size_t existed_feature_dimension;
        if(old_image_ids.size() == 0){
            existed_feature_dimension=static_cast<int>(param.GetArgument("compressed_feature_dimension", 128));;
        }else{
            existed_feature_dimension = feature_data_container.GetCompressedDescriptors(old_image_ids[0]).cols();
        } 
        std::cout<<"existed_feature_dimension: "<<existed_feature_dimension<<std::endl;
        if (static_cast<bool>(param.GetArgument("map_update", 0))) {
            if(boost::filesystem::exists(pca_matrix_path)){
                load_pca = true;
                ReadPcaProjectionMatrix(pca_matrix, embedding_thresholds, pca_matrix_path);
            }else{
                if(existed_feature_dimension != 128){
                    std::cout << "Original Map Feature already compressed, Cannot training pca with old descriptors" << std::endl;
                    enable_whole_pca = false;
                }
            }
        }

        // CHECK(enable_whole_pca) << existed_feature_dimension;
        // compress descriptors using the trained PCA matrix
        PrintHeading1("Compressing descriptors");

        int compressed_feature_dimension = static_cast<int>(param.GetArgument("compressed_feature_dimension", 128));
        CHECK(compressed_feature_dimension == 128 || compressed_feature_dimension == 64 ||
            compressed_feature_dimension == 32);
        if(compressed_feature_dimension != existed_feature_dimension){
            std::cout<<" WARN!!!! existed_feature_dimension is "<<existed_feature_dimension<<" not equal to yaml setting "<<compressed_feature_dimension<<std::endl;
            compressed_feature_dimension = existed_feature_dimension;
            std::cout<<" set compressed_feature_dimension to "<<compressed_feature_dimension<<std::endl;
        }

        if(!load_pca && enable_whole_pca){
            auto image_ids = whole_image_ids;

            if(compressed_feature_dimension != 128){
                int pca_training_feature_count = static_cast<int> (param.GetArgument("pca_training_feature_count", 1000000));
                uint64_t total_feature_count = 0;
                for (int i = 0; i < image_ids.size(); ++i){
                    image_t current_id = image_ids[i];
                    const auto &keypoints = feature_data_container.GetKeypoints(current_id); 

                    total_feature_count += keypoints.size();
                }

                int sample_step = 1;
                if(pca_training_feature_count < total_feature_count){
                    sample_step = total_feature_count / pca_training_feature_count;
                }

                for (int i = 0; i < image_ids.size(); ++i) {
                    image_t current_id = image_ids[i];
                    const auto &descriptors = feature_data_container.GetDescriptors(current_id);
                    size_t sampled_descriptors_count = descriptors.rows() / sample_step;

                    training_descriptors.conservativeResize(training_descriptors_count + sampled_descriptors_count, descriptors.cols());

                    for (size_t j = 0; j < sampled_descriptors_count; j++) {
                        training_descriptors.row(training_descriptors_count + j) = descriptors.row(j * sample_step);
                    }
                    training_descriptors_count += sampled_descriptors_count;
                }
                
                // Eigen::Matrix<double, 128, 128> pca_matrix;
                // Eigen::Matrix<double, 128, 1> embedding_thresholds;

                std::cout << "PCA training " << std::endl;  
                PcaTraining(training_descriptors,pca_matrix,embedding_thresholds);
                std::cout << StringPrintf("PCA training  in %.3f min", timer.ElapsedMinutes()) << std::endl;

                fprintf(fs, "%s\n", StringPrintf("PCA training Elapsed time: %.3f [minutes]", timer.ElapsedMinutes()).c_str());
                fflush(fs);

                // std::string pca_matrix_path = param.GetArgument("pca_matrix_path", "");
                if (pca_matrix_path.empty()) {
                    pca_matrix_path = workspace_path + "pca_matrix.bin";
                }
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
                CHECK(boost::filesystem::exists(pca_matrix_path));
            }
        
        }
        
        auto image_ids = new_image_ids;

        // int compressed_feature_dimension = static_cast<int>(param.GetArgument("compressed_feature_dimension", 128));
        // CHECK(compressed_feature_dimension == 128 || compressed_feature_dimension == 64 ||
        //       compressed_feature_dimension == 32);

        std::cout<<"compressing images size "<<image_ids.size()<<std::endl;
        for (int i = 0; i < image_ids.size(); ++i) {
            image_t current_id = image_ids[i];
            auto &descriptors = feature_data_container.GetDescriptors(current_id);
            auto &compressed_descriptors = feature_data_container.GetCompressedDescriptors(current_id);

            if (compressed_feature_dimension == 128) {
                compressed_descriptors = descriptors;
            } else {
                CompressFeatureDescriptors(descriptors, compressed_descriptors, pca_matrix, embedding_thresholds,
                                        compressed_feature_dimension);
            }
            descriptors.resize(0, 0);
        }
        std::cout << StringPrintf("Compressing descriptors in %.3f min", timer.ElapsedMinutes()) << std::endl;

        fprintf(fs, "%s\n", StringPrintf("Compressing descriptors in %.3f [minutes]", timer.ElapsedMinutes()).c_str());
        fflush(fs);
    } else {
        const std::vector<image_t>& new_image_ids = feature_data_container.GetNewImageIds();
        for (int i = 0; i < new_image_ids.size(); ++i) {
            image_t current_id = new_image_ids[i];
            auto &descriptors = feature_data_container.GetDescriptors(current_id);
            auto &compressed_descriptors = feature_data_container.GetCompressedDescriptors(current_id);
            std::swap(descriptors, compressed_descriptors);
        }
    }

    timer.Start();
    bool write_feature = static_cast<bool>(param.GetArgument("write_feature", 1));
    bool write_binary = static_cast<bool>(param.GetArgument("write_binary", 1));
    if (write_feature) {
        // if map_update, save features to map_update folder
        if (static_cast<bool>(param.GetArgument("map_update", 0))) {
            workspace_path = map_update_workspace_path;
        }
        feature_data_container.WriteImagesBinaryData(JoinPaths(workspace_path, "/features.bin"));
        feature_data_container.WriteLocalImagesBinaryData(JoinPaths(workspace_path, "/local_images.bin"));
        feature_data_container.WriteCamerasBinaryData(JoinPaths(workspace_path, "/cameras.bin"));
        feature_data_container.WriteLocalCamerasBinaryData(JoinPaths(workspace_path, "/local_cameras.bin"));
        feature_data_container.WriteSubPanoramaBinaryData(JoinPaths(workspace_path, "/sub_panorama.bin"));

        if ((static_cast<bool>(param.GetArgument("map_update", 0)))||
            (reader_options.num_local_cameras == 2 && reader_options.camera_model == "OPENCV_FISHEYE")) {
            feature_data_container.WritePieceIndicesBinaryData(JoinPaths(workspace_path, "/piece_indices.bin"));
        }
        if (sift_extraction.detect_apriltag) {
            // Check the Arpiltag Detect Result
            if (feature_data_container.ExistAprilTagDetection()) {
                feature_data_container.WriteAprilTagBinaryData(JoinPaths(workspace_path, "/apriltags.bin"));
            } else {
                std::cout << "Warning: No Apriltag Detection has been found ... " << std::endl;
            }
        }
        if (use_gps_prior) {
            feature_data_container.WriteGPSBinaryData(JoinPaths(workspace_path, "/gps.bin"));
        }
    }
    fprintf(fs, "%s\n", StringPrintf("Write local descriptors in %.3f [minutes]", timer.ElapsedMinutes()).c_str());
    
    if (use_gps_prior) {
        image_t geo_image_idx = feature_data_container.GetGeoImageIndex();
        if (feature_data_container.ExistImage(geo_image_idx) || !reader_options.gps_origin.empty()) {
            double latitude, longitude, altitude;
            if (feature_data_container.ExistImage(geo_image_idx)){
                const class Image& image = feature_data_container.GetImage(geo_image_idx);
                std::string image_path = JoinPaths(reader_options.image_path, image.Name());
                Bitmap bitmap;
                bitmap.Read(image_path);
                // double latitude, longitude, altitude;
                bitmap.ExifLatitude(&latitude);
                bitmap.ExifLongitude(&longitude);
                bitmap.ExifAltitude(&altitude);
            } else {
                std::vector<double> gps_origin = CSVToVector<double>(reader_options.gps_origin);
                latitude = gps_origin[0];
                longitude = gps_origin[1];
                altitude = gps_origin[2];
            }
            GeodeticConverter geo_converter(latitude, longitude, altitude);
            Eigen::Matrix3x4d M = geo_converter.NedToEcefMatrix();

            std::ofstream file(JoinPaths(workspace_path, "/ned_to_ecef.txt"), std::ofstream::out);
            file << MAX_PRECISION << M(0, 0) << " " << M(0, 1) << " " 
                << M(0, 2) << " " << M(0, 3) << std::endl;
            file << MAX_PRECISION << M(1, 0) << " " << M(1, 1) << " " 
                << M(1, 2) << " " << M(1, 3) << std::endl;
            file << MAX_PRECISION << M(2, 0) << " " << M(2, 1) << " " 
                << M(2, 2) << " " << M(2, 3) << std::endl;
            file.close();
            std::ofstream file1(JoinPaths(workspace_path, "/gps_origin.txt"));
            file1 << MAX_PRECISION << latitude << " " << longitude << " " << altitude << std::endl;
            file1.close();
            std::cout << "Save gps info to ned_to_ecef.txt & gps_origin.txt" << std::endl;
        }
    }

    return true;
}

bool GlobalFeatureExtraction(FeatureDataContainer &feature_data_container, Configurator &param,
                             bool new_local_feature_extraction) {
    std::string workspace_path = param.GetArgument("workspace_path", "");
    CHECK(!workspace_path.empty());

    OptionParser option_parser;
    GlobalFeatureExtractionOptions options;

    option_parser.GetGlobalFeatureExtractionOptions(options, param);

    if (boost::filesystem::exists(JoinPaths(workspace_path, "/vlad_vectors.bin"))) {
        // only quit not in update mode
        if (!static_cast<bool>(param.GetArgument("map_update", 0))){
            std::cout << "Global feature already exists, skip feature extraction" << std::endl;
            return false;
        }
    }

    if(!new_local_feature_extraction){
        std::string feature_data_path;
        if (static_cast<bool>(param.GetArgument("map_update", 0))){
            feature_data_path = JoinPaths(workspace_path, "/map_update");
        }else{
            feature_data_path = workspace_path;
        }
        CHECK(LoadFeatures(feature_data_container,param,feature_data_path));
    }

    const std::vector<image_t> &whole_image_ids = feature_data_container.GetImageIds();

    std::string vlad_code_book_path = options.vlad_code_book_path;

    //TODO: if exist vald matrix already & map update, read matrix
    bool load_vlad = false;
    size_t existed_feature_dimension = feature_data_container.GetCompressedDescriptors(whole_image_ids[0]).cols();
    if (static_cast<bool>(param.GetArgument("map_update", 0))) {
        if(boost::filesystem::exists(vlad_code_book_path)){
            load_vlad = true;
        }
    }


    // vlad code book training using the extracted descriptors
    Timer timer;
    if(!load_vlad){
        timer.Start();
        VladVisualIndex vlad_visual_index;
        VladVisualIndex::CodeBookCreateOptions code_book_create_option;
        code_book_create_option.num_vocabulary = param.GetArgument("vlad_num_vocabulary",256);

        std::cout << "Collect training descriptors " << std::endl;


        CompressedFeatureDescriptors training_descriptors;
        size_t training_descriptors_count = 0;

        // const std::vector<image_t> &image_ids = feature_data_container.GetImageIds();
        auto image_ids = whole_image_ids;

        int vlad_training_feature_count = static_cast<int>(param.GetArgument("vlad_training_feature_count", 1000000));
        int total_feature_count = 0;
        for (int i = 0; i < image_ids.size(); ++i) {
            image_t current_id = image_ids[i];
            const auto &keypoints = feature_data_container.GetKeypoints(current_id);
            total_feature_count += keypoints.size();
        }

        int sample_step = 1;
        if (vlad_training_feature_count < total_feature_count) { 
            sample_step = total_feature_count / vlad_training_feature_count;
        }

        for (int i = 0; i < image_ids.size(); ++i) {
            image_t current_id = image_ids[i];
            const auto &descriptors = feature_data_container.GetCompressedDescriptors(current_id);
            size_t sampled_descriptors_count = descriptors.rows() / sample_step;

            training_descriptors.conservativeResize(training_descriptors_count + sampled_descriptors_count,
                                                    descriptors.cols());

            for (size_t j = 0; j < sampled_descriptors_count; j++) {
                training_descriptors.row(training_descriptors_count + j) = descriptors.row(j * sample_step);
            }
            training_descriptors_count += sampled_descriptors_count;
        }
        VladVisualIndex::Descriptors float_training_descriptors;
        CompressedFeatureDescriptorsTofloat(training_descriptors, float_training_descriptors);
        std::cout << "training descriptor count and dimension: " << float_training_descriptors.rows() << " "
                << float_training_descriptors.cols() << std::endl;


        std::cout << "Kmeans to create code book " << std::endl;
        // create code book
        vlad_visual_index.CreateCodeBook(code_book_create_option, float_training_descriptors);
        std::cout << StringPrintf("Create code book in %.3f min", timer.ElapsedMinutes()) << std::endl;

        fprintf(fs, "%s\n", StringPrintf("Create code book in %.3f [minutes]", timer.ElapsedMinutes()).c_str());
        fflush(fs);
        
        vlad_visual_index.SaveCodeBook(options.vlad_code_book_path);
    }

    timer.Start();
    PrintHeading1("Global feature extraction");
    GlobalFeatureExtractor global_feature_extractor(options, &feature_data_container);
    global_feature_extractor.Run();
    if (static_cast<bool>(param.GetArgument("map_update", 0))){
        feature_data_container.WriteGlobalFeaturesBinaryData(workspace_path + "/map_update/vlad_vectors.bin");
    }else{
        feature_data_container.WriteGlobalFeaturesBinaryData(workspace_path + "/vlad_vectors.bin");
    }

    fprintf(fs, "%s\n", StringPrintf("Extract Global descriptors in %.3f [minutes]", timer.ElapsedMinutes()).c_str());
    fflush(fs);

    return true;
}

int main(int argc, char *argv[]) {
    using namespace sensemap;

    PrintHeading("SenseMap.  Copyright(c) 2019, SenseTime Group.");
    PrintHeading(std::string("Version: sfm-feature-extraction-") + __VERSION__);
    Timer timer;
    timer.Start();

    std::string configuration_file_path;

    int param_idx = 1;
#ifdef DO_ENCRYPT_CHECK
	CHECK(argc >= 5);
	int ret = do_check(5, (const char**)argv);
  	std::cout << "Check Status: " << ret << std::endl;
  	if (ret) return ret;
	param_idx = 5;
#endif
	configuration_file_path = std::string(argv[param_idx]);
	std::cout << "configuration_file_path: " << configuration_file_path << std::endl;

    Configurator param;
    param.Load(configuration_file_path.c_str());

    std::string workspace_path = param.GetArgument("workspace_path", "");
    CHECK(!workspace_path.empty()) << "workspace path empty";

    std::string image_path = param.GetArgument("image_path", "");
    CHECK(!image_path.empty()) << "image path empty";

    if (!boost::filesystem::exists(workspace_path)) {
        CHECK(boost::filesystem::create_directories(workspace_path)) << "Create workspace failed";
    }

    fs = fopen((workspace_path + "/time_feature_extraction.txt").c_str(), "w");

    auto feature_data_container = std::make_shared<FeatureDataContainer>();

    bool new_feature_extraction = FeatureExtraction(*feature_data_container.get(), param);

    typedef FeatureMatchingOptions::RetrieveType RetrieveType;
    RetrieveType retrieve_type = (RetrieveType)param.GetArgument("retrieve_type", 0);
    if (retrieve_type == RetrieveType::VLAD) {
        GlobalFeatureExtraction(*feature_data_container.get(), param, new_feature_extraction);
    }

    fclose(fs);
    timer.PrintMinutes();
    return 0;
}
