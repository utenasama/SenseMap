// Copyright (c) 2019, SenseTime Group.
// All rights reserved.

#include <Eigen/Dense>
#include <boost/filesystem/path.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/eigen.hpp>

#include "ceres/rotation.h"
#include "base/cost_functions.h"
#include "base/camera.h"
#include "base/image.h"
#include "base/image_reader.h"
#include "base/reconstruction_manager.h"
#include "base/pose.h"
#include "base/camera_models.h"
#include "base/common.h"
#include "graph/correspondence_graph.h"
#include "util/misc.h"
#include "util/mat.h"
#include "util/rgbd_helper.h"
#include "util/imageconvert.h"

#include "container/feature_data_container.h"
#include "feature/extraction.h"
#include "feature/feature_matcher.h"
#include "feature/utils.h"
#include "feature/global_feature_extraction.h"

#include "controllers/incremental_mapper_controller.h"
#include "sfm/incremental_mapper.h"

#include "../Configurator_yaml.h"
#include "../camera_rig_params.h"
#include "../option_parsing.h"
#include "base/version.h"

#include <dirent.h>
#include <sys/stat.h>

#include "util/gps_reader.h"
#include <unordered_set>
#include "../system_io.h"
#include "util/ply.h"
#include "util/exception_handler.h"

#define MAX_PRECISION std::setprecision(std::numeric_limits<double>::digits10 + 1)

using namespace sensemap;

std::string configuration_file_path;
std::string tool_name;
float points_threshold = 200;

FILE *fs;

struct HybridOptions{
    bool debug_info = false;
    int child_id = -1;
    bool update_flag = false;
    bool save_flag = true;
    bool read_flag = true;
    std::string image_selection = "";
    std::vector<std::string> image_list;
    bool manual_match = false;
    int manual_match_overlap = 10;
    int manual_match_overlap_step = 1;
    std::pair<image_t, image_t> match_range = { kInvalidImageId, kInvalidImageId };
    HybridOptions(bool de = false, int ch = -1, bool up = false, bool sa = true, 
                  bool re = true, std::pair<int, int> mr = { kInvalidImageId, kInvalidImageId }):debug_info(de), child_id(ch), update_flag(up), 
                    save_flag(sa), read_flag(re){};
    void Print(){
        std::cout << "debug_info: " << debug_info << "\nchild_id: " << child_id
            << "\nupdate_flag: " << update_flag << "\nsave_flag: " << save_flag
            << "\nread_flag: " << read_flag << std::endl;
    };
}; 

struct RGBDInfo
{
    std::string calib_cam;
    std::string sub_path;
    std::string rgbd_camera_params;
    Eigen::Matrix3d extra_R;// extrinsic for pro2 or one-r
    Eigen::Vector3d extra_T;
    int timestamp = -1;//whether file format is timestamp(1) or not(0) or auto detect(-1)

    bool has_force_offset = false;
    int force_offset = 0;
};

Camera target_camera;
std::string target_subpath;
std::vector<std::string> target_subpaths;
std::vector<RGBDInfo> rgbd_infos;
std::vector<std::vector<RGBDInfo>> rgbd_infos_vec;

bool dirExists(const std::string &dirName_in) { return access(dirName_in.c_str(), F_OK) == 0; }

void PrintReconSummary(const std::string &flog_name, const size_t num_total_image,
                       const std::shared_ptr<ReconstructionManager> &reconstruction_manager) {
    if (reconstruction_manager->Size() == 0) {
        return;
    }
    std::shared_ptr<Reconstruction> best_rec;
    for (int i = 0; i < reconstruction_manager->Size(); ++i) {
        const std::shared_ptr<Reconstruction> &rec = reconstruction_manager->Get(i);
        if (!best_rec || best_rec->NumRegisterImages() < rec->NumRegisterImages()) {
            best_rec = rec;
        }
    }
    FILE *fp = fopen(flog_name.c_str(), "w");

    size_t num_reg_image = best_rec->NumRegisterImages();
    fprintf(fp, "Registered / Total: %zu / %zu\n", num_reg_image, num_total_image);
    fprintf(fp, "Mean Track Length: %f\n", best_rec->ComputeMeanTrackLength());
    fprintf(fp, "Mean Reprojection Error: %f\n", best_rec->ComputeMeanReprojectionError());
    fprintf(fp, "Mean Observation Per Register Image: %f\n", best_rec->ComputeMeanObservationsPerRegImage());

    fclose(fp);
}

int64_t ImageNameToIndex(const std::string & name) {
    return std::atoll(boost::filesystem::path(name).stem().string().c_str());
}

double GetAverageIndexDiff(FeatureDataContainer &feature_data_container) {
    std::vector<int64_t> index;
    for (auto & image_name : feature_data_container.GetImageNames()) {
        index.emplace_back(ImageNameToIndex(image_name));
    }
    std::sort(index.begin(), index.end());
    if (index.size() <= 1) return 1.0;

    double diff_sum = 0.0;
    size_t diff_count = 0;
    for (int i = 1; i < index.size(); i++) {
        diff_sum += index[i] - index[i - 1];
        diff_count ++;
    }
    return diff_sum / diff_count;
}

void FeatureExtraction(FeatureDataContainer &feature_data_container, Configurator &param, 
    struct HybridOptions hybrid_options = HybridOptions()) {
    std::string workspace_path = param.GetArgument("workspace_path", "");
    CHECK(!workspace_path.empty());

    OptionParser option_parser;
    ImageReaderOptions reader_options;
    if (hybrid_options.child_id < 0){
        option_parser.GetImageReaderOptions(reader_options, param);
    } else {
        option_parser.GetImageReaderOptions(reader_options, param, hybrid_options.child_id);
        if (hybrid_options.save_flag && hybrid_options.debug_info){
            workspace_path = JoinPaths(workspace_path, reader_options.child_path);
            if (!boost::filesystem::exists(workspace_path)){
                boost::filesystem::create_directories(workspace_path);
            }
        }
    }
    if (!hybrid_options.image_selection.empty()) {
        reader_options.image_selection = hybrid_options.image_selection;
    }

    if (!hybrid_options.image_list.empty()){
        reader_options.image_list = hybrid_options.image_list;
    }

    bool camera_rig = (reader_options.num_local_cameras > 1);
    bool with_depth = reader_options.with_depth;

    std::string rgbd_parmas_file = param.GetArgument("rgbd_params_file", "");
    bool use_gps_prior = param.GetArgument("use_gps_prior", 0);

    std::unordered_set<camera_t> existed_camera_ids;
    {
        for (image_t image_id : feature_data_container.GetImageIds()) {
            existed_camera_ids.insert(feature_data_container.GetImage(image_id).CameraId());
        }
    }

    bool exist_feature_file = false;
    if (boost::filesystem::exists(JoinPaths(workspace_path, "/cameras.bin")) &&
        boost::filesystem::exists(JoinPaths(workspace_path, "/features.bin")) && 
        hybrid_options.read_flag) {
        if (boost::filesystem::exists(JoinPaths(workspace_path, "/local_cameras.bin"))) {
            feature_data_container.ReadCamerasBinaryData(JoinPaths(workspace_path, "/cameras.bin"), false);
            feature_data_container.ReadLocalCamerasBinaryData(JoinPaths(workspace_path, "/local_cameras.bin"));
        } else {
            feature_data_container.ReadCamerasBinaryData(JoinPaths(workspace_path, "/cameras.bin"), camera_rig);
        }
        feature_data_container.ReadImagesBinaryData(JoinPaths(workspace_path, "/features.bin"));
        exist_feature_file = true;
    } else if (boost::filesystem::exists(JoinPaths(workspace_path, "/cameras.txt")) &&
               boost::filesystem::exists(JoinPaths(workspace_path, "/features.txt")) && 
               hybrid_options.read_flag) {
        if (boost::filesystem::exists(JoinPaths(workspace_path, "/local_cameras.txt"))) {
            feature_data_container.ReadCameras(JoinPaths(workspace_path, "/cameras.txt"), false);
            feature_data_container.ReadLocalCameras(JoinPaths(workspace_path, "/local_cameras.txt"));
        } else {
            feature_data_container.ReadCameras(JoinPaths(workspace_path, "/cameras.txt"), camera_rig);
        }
        feature_data_container.ReadImagesData(JoinPaths(workspace_path, "/features.txt"));
        exist_feature_file = true;
    } else {
        exist_feature_file = false;
    }

    // Apply fixed_camera option
    if (exist_feature_file) {
        std::unordered_set<camera_t> camera_ids;
        for (image_t image_id : feature_data_container.GetImageIds()) {
            camera_ids.insert(feature_data_container.GetImage(image_id).CameraId());
        }
        for (camera_t camera_id : camera_ids) {
            if (existed_camera_ids.count(camera_id) == 0) {
                feature_data_container.GetCamera(camera_id).SetCameraConstant(reader_options.fixed_camera);
            }
        }
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

    // Check the AprilTag detection file exist or not
    if (exist_feature_file && 
        static_cast<bool>(param.GetArgument("detect_apriltag", 0))) {
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
        return;
    }

    std::cout << "reader_options.image_path: " << reader_options.image_path << std::endl;
    std::cout << "reader_options.child_path: " << reader_options.child_path << std::endl;
    std::cout << "reader_options.image_list size: " << reader_options.image_list.size() << std::endl;
    for (int k = 0; k < reader_options.image_list.size();k++) {
        std::cout << " " << reader_options.image_list[k] << std::endl;
    }

    SiftExtractionOptions sift_extraction;
    option_parser.GetFeatureExtractionOptions(sift_extraction,param);
    if (hybrid_options.child_id >= 0) {
        // override feature extraction options for specific cameras
        option_parser.GetFeatureExtractionOptions(sift_extraction, param, hybrid_options.child_id);
    }

    std::string panorama_config_file = param.GetArgument("panorama_config_file", "");
    if(!panorama_config_file.empty()){
        std::vector<PanoramaParam> panorama_params;
        LoadParams(panorama_config_file, panorama_params);
        sift_extraction.panorama_config_params = panorama_params;   
        sift_extraction.use_panorama_config = true;             
    }

    image_t start_image_id = 0;
    camera_t start_camera_id = 0;
    label_t start_label_id = 0;
    if (hybrid_options.update_flag){
        for(const auto old_image_id: feature_data_container.GetImageIds()){
            if(start_image_id < old_image_id){
                start_image_id = old_image_id;
            }
            std::string image_name = feature_data_container.GetImage(old_image_id).Name();
            feature_data_container.GetImage(image_name).SetLabelId(0);
        }
        if (hybrid_options.match_range.first != kInvalidImageId && 
            hybrid_options.match_range.second >= hybrid_options.match_range.first
        ) {
            for(const auto old_image_id: feature_data_container.GetImageIds()){
                if (old_image_id < hybrid_options.match_range.first || 
                    old_image_id > hybrid_options.match_range.second
                ) {
                    std::string image_name = feature_data_container.GetImage(old_image_id).Name();
                    feature_data_container.GetImage(image_name).SetLabelId(kInvalidLabelId);
                }
            }
        }
        // ++start_image_id;
        start_camera_id = feature_data_container.NumCamera();
        ++start_label_id;
    } else {
        for(const auto old_image_id: feature_data_container.GetImageIds()){
            if(start_image_id < old_image_id){
                start_image_id = old_image_id;
            }
            label_t label_id = feature_data_container.GetImage(old_image_id).LabelId();
            if(label_id > start_label_id){
                start_label_id = label_id;
            }
        }
        // ++start_image_id;
        start_camera_id = feature_data_container.NumCamera();
        ++start_label_id;
    }
    std::cout << "\n \n \n"<< start_image_id << " " << start_camera_id << " " << start_label_id << std::endl;

    reader_options.subpath_camera_map[target_subpath] = target_camera;
    SiftFeatureExtractor feature_extractor(reader_options, sift_extraction, 
        &feature_data_container, start_image_id, start_camera_id, start_label_id);
    feature_extractor.Start();
    feature_extractor.Wait();

    fprintf(
        fs, "%s\n",
        StringPrintf("Feature Extraction Elapsed time: %.3f [minutes]", feature_extractor.GetTimer().ElapsedMinutes())
            .c_str());
    fflush(fs);

    // Pca training using the extrated feature descriptors
    Timer timer;
    timer.Start();
    
    std::cout << "Collect training descriptors " << std::endl;
    FeatureDescriptors training_descriptors;
    size_t training_descriptors_count = 0;

    const std::vector<image_t>& image_ids = feature_data_container.GetNewImageIds();

    int pca_training_feature_count = static_cast<int> (param.GetArgument("pca_training_feature_count", 1000000));
    int total_feature_count = 0;
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
        std::cout << current_id << "-image descriptors size: " << descriptors.rows() 
            << " * " << descriptors.cols() << std::endl;
        training_descriptors.conservativeResize(training_descriptors_count + sampled_descriptors_count, descriptors.cols());

        for (size_t j = 0; j < sampled_descriptors_count; j++) {
            training_descriptors.row(training_descriptors_count + j) = descriptors.row(j * sample_step);
        }
        training_descriptors_count += sampled_descriptors_count;
    }
    
    Eigen::Matrix<double, 128, 128> pca_matrix;
    Eigen::Matrix<double, 128, 1> embedding_thresholds;

    std::cout << "PCA training " << std::endl;  
    PcaTraining(training_descriptors,pca_matrix,embedding_thresholds);
    std::cout << StringPrintf("PCA training  in %.3f min", timer.ElapsedMinutes()) << std::endl;

    fprintf(fs, "%s\n", StringPrintf("PCA training Elapsed time: %.3f [minutes]", timer.ElapsedMinutes()).c_str());
    fflush(fs);

    std::string pca_matrix_path = param.GetArgument("pca_matrix_path", "");
    if (pca_matrix_path.empty()) {
        pca_matrix_path = JoinPaths(workspace_path, "pca_matrix.bin");
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


    // compress descriptors using the trained PCA matrix
    PrintHeading1("Compressing descriptors");
    int compressed_feature_dimension = static_cast<int>(param.GetArgument("compressed_feature_dimension", 128));

    typedef FeatureMatchingOptions::RetrieveType RetrieveType;
    RetrieveType retrieve_type = (RetrieveType)param.GetArgument("retrieve_type", 0);
    if (retrieve_type == RetrieveType::SIFT && compressed_feature_dimension != 128) {
        compressed_feature_dimension = 128;
        std::cout << StringPrintf("Warning! Feature dimension is not customized for SIFT\n");
    }

    CHECK(compressed_feature_dimension == 128 || compressed_feature_dimension == 64 ||
        compressed_feature_dimension == 32);

    timer.Start();
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
        // descriptors.resize(0, 0);
    }
    std::cout << StringPrintf("Compressing descriptors in %.3f min", timer.ElapsedMinutes()) << std::endl;

    fprintf(fs, "%s\n", StringPrintf("Compressing descriptors in %.3f [minutes]", timer.ElapsedMinutes()).c_str());
    fflush(fs);

    bool write_feature = static_cast<bool>(param.GetArgument("write_feature", 0));
    bool write_binary = static_cast<bool>(param.GetArgument("write_binary", 1));
    if (write_feature && hybrid_options.save_flag) {
        if (write_binary) {
            feature_data_container.WriteImagesBinaryData(JoinPaths(workspace_path, "/features.bin"));
            feature_data_container.WriteCamerasBinaryData(JoinPaths(workspace_path, "/cameras.bin"));
            feature_data_container.WriteLocalCamerasBinaryData(JoinPaths(workspace_path, "/local_cameras.bin"));
            feature_data_container.WriteSubPanoramaBinaryData(JoinPaths(workspace_path, "/sub_panorama.bin"));
            feature_data_container.WritePieceIndicesBinaryData(JoinPaths(workspace_path, "/piece_indices.bin"));

            if (sift_extraction.detect_apriltag) {
                // Check the Arpiltag Detect Result
                if(feature_data_container.ExistAprilTagDetection()){
                    feature_data_container.WriteAprilTagBinaryData(JoinPaths(workspace_path, "/apriltags.bin"));
                }else{
                    std::cout << "Warning: No Apriltag Detection has been found ... " << std::endl;
                }
            }
            if (use_gps_prior) {
                feature_data_container.WriteGPSBinaryData(JoinPaths(workspace_path, "/gps.bin"));
            }
        } else {
            feature_data_container.WriteImagesData(JoinPaths(workspace_path, "/features.txt"));
            feature_data_container.WriteCameras(JoinPaths(workspace_path, "/cameras.txt"));
            feature_data_container.WriteLocalCameras(JoinPaths(workspace_path, "/local_cameras.txt"));
            feature_data_container.WriteSubPanoramaData(JoinPaths(workspace_path, "/sub_panorama.txt"));

            if (reader_options.num_local_cameras == 2 && reader_options.camera_model == "OPENCV_FISHEYE" &&
               sift_extraction.convert_to_perspective_image) {
                feature_data_container.WritePieceIndicesData(JoinPaths(workspace_path, "/piece_indices.txt"));
            }
        }
    }

    if (use_gps_prior) {
        image_t geo_image_idx = feature_data_container.GetGeoImageIndex();
        std::string gps_origin_str = param.GetArgument("gps_origin", "");
        std::vector<double> gps_origin;
        if (!gps_origin_str.empty()){
            gps_origin = CSVToVector<double>(gps_origin_str);
        }
        if (feature_data_container.ExistImage(geo_image_idx) || gps_origin.size() == 3) {
            double latitude, longitude, altitude;
            if (feature_data_container.ExistImage(geo_image_idx)){
                const class Image& image = feature_data_container.GetImage(geo_image_idx);
                std::string image_path = JoinPaths(reader_options.image_path, image.Name());
                Bitmap bitmap;
                bitmap.Read(image_path);
                bitmap.ExifLatitude(&latitude);
                bitmap.ExifLongitude(&longitude);
                bitmap.ExifAltitude(&altitude);
            } else {
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
        }
    }
}

void GlobalFeatureExtraction(FeatureDataContainer &feature_data_container, Configurator &param,
                             struct HybridOptions hybrid_options = HybridOptions()){

    std::string workspace_path = param.GetArgument("workspace_path", "");
    CHECK(!workspace_path.empty());

    if (hybrid_options.child_id > -1 && hybrid_options.debug_info){
        Configurator camera_param;
        std::string cameras_param_file = param.GetArgument("camera_param_file", "");
        camera_param.Load(cameras_param_file.c_str());
        std::string child_name = camera_param.GetArgument("sub_path_" + std::to_string(hybrid_options.child_id), "");
        CHECK(!child_name.empty());

        workspace_path = JoinPaths(workspace_path, child_name);
        CHECK(boost::filesystem::exists(workspace_path));
    }

    OptionParser option_parser;
    GlobalFeatureExtractionOptions options;

    option_parser.GetGlobalFeatureExtractionOptions(options,param);

    if (boost::filesystem::exists(JoinPaths(workspace_path, "/vlad_vectors.bin"))){
        feature_data_container.ReadGlobalFeaturesBinaryData(workspace_path + "/vlad_vectors.bin");
        std::cout << "Global feature already exists, skip feature extraction" << std::endl;
        return;
    }

    typedef FeatureMatchingOptions::RetrieveType RetrieveType;
    RetrieveType retrieve_type = (RetrieveType)param.GetArgument("retrieve_type", 0);
    if (retrieve_type == RetrieveType::VLAD) {
        // vlad code book training using the extracted descriptors
        Timer timer;
        timer.Start();
        VladVisualIndex vlad_visual_index;
        VladVisualIndex::CodeBookCreateOptions code_book_create_option;
        code_book_create_option.num_vocabulary = param.GetArgument("vlad_num_vocabulary",256);

        std::cout << "Collect training descriptors " << std::endl;

        CompressedFeatureDescriptors training_descriptors;
        size_t training_descriptors_count = 0;

        const std::vector<image_t> &image_ids = feature_data_container.GetImageIds();

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

    GlobalFeatureExtractor global_feature_extractor(options,&feature_data_container);
    global_feature_extractor.Run();
    feature_data_container.WriteGlobalFeaturesBinaryData(workspace_path + "/vlad_vectors.bin");
}

void FeatureMatching(FeatureDataContainer &feature_data_container, SceneGraphContainer &scene_graph,
                     Configurator &param, struct HybridOptions hybrid_options = HybridOptions()) {
    using namespace std::chrono;
    high_resolution_clock::time_point start_time = high_resolution_clock::now();

    OptionParser option_parser;
    std::string workspace_path = param.GetArgument("workspace_path", "");
    CHECK(!workspace_path.empty());

    if (hybrid_options.child_id > -1 && hybrid_options.debug_info){
        ImageReaderOptions reader_options;
        option_parser.GetImageReaderOptions(reader_options, param, hybrid_options.child_id);

        workspace_path = JoinPaths(workspace_path, reader_options.child_path);
        CHECK(boost::filesystem::exists(workspace_path));
    }

    bool load_scene_graph = false;
    if (boost::filesystem::exists(JoinPaths(workspace_path, "/scene_graph.bin")) && 
        hybrid_options.read_flag) {
        scene_graph.ReadSceneGraphBinaryData(JoinPaths(workspace_path, "/scene_graph.bin"));
        load_scene_graph = true;
    } else if (boost::filesystem::exists(JoinPaths(workspace_path, "/scene_graph.txt")) && 
        hybrid_options.read_flag) {
        scene_graph.ReadSceneGraphData(JoinPaths(workspace_path, "/scene_graph.txt"));
        load_scene_graph = true;
    }

    if (load_scene_graph) {
        EIGEN_STL_UMAP(image_t, class Image) &images = scene_graph.Images();
        EIGEN_STL_UMAP(camera_t, class Camera) &cameras = scene_graph.Cameras();

        std::vector<image_t> image_ids = feature_data_container.GetImageIds();

        for (const auto image_id : image_ids) {
            const Image &image = feature_data_container.GetImage(image_id);
            if (!scene_graph.CorrespondenceGraph()->ExistsImage(image_id)) {
                continue;
            }

            images[image_id] = image;

            const FeatureKeypoints &keypoints = feature_data_container.GetKeypoints(image_id);
            images[image_id].SetPoints2D(keypoints);
            const PanoramaIndexs & panorama_indices = feature_data_container.GetPanoramaIndexs(image_id);

            const Camera &camera = feature_data_container.GetCamera(image.CameraId());

            std::vector<uint32_t> local_image_indices(keypoints.size());
		    for(size_t i = 0; i<keypoints.size(); ++i){
                if (panorama_indices.size() == 0 && camera.NumLocalCameras() == 1) {
                    local_image_indices[i] = image_id;
                } else {
			        local_image_indices[i] = panorama_indices[i].sub_image_id;
                }
		    }
		    images[image_id].SetLocalImageIndices(local_image_indices);

            if (!scene_graph.ExistsCamera(image.CameraId())) {
                cameras[image.CameraId()] = camera;
            }

            if (scene_graph.CorrespondenceGraph()->ExistsImage(image_id)) {
                images[image_id].SetNumObservations(
                    scene_graph.CorrespondenceGraph()->NumObservationsForImage(image_id));
                images[image_id].SetNumCorrespondences(
                    scene_graph.CorrespondenceGraph()->NumCorrespondencesForImage(image_id));
            } else {
                std::cout << "Do not contain ImageId = " << image_id << ", in the correspondence graph." << std::endl;
            }
        }

        scene_graph.CorrespondenceGraph()->Finalize();

        // Set overlap flag of keypoints.
        for (auto& image : scene_graph.Images()) {
            if (!scene_graph.CorrespondenceGraph()->ExistsImage(image.first)) {
                continue;
            }
            const FeatureMatches& corrs = 
            scene_graph.CorrespondenceGraph()->FindCorrespondencesBetweenImages(image.first, image.first);
            for (const FeatureMatch& corr : corrs) {
                image.second.Point2D(corr.point2D_idx1).SetOverlap(true);
                image.second.Point2D(corr.point2D_idx2).SetOverlap(true);
            }
        }
        return;
    }

    FeatureMatchingOptions options;
    option_parser.GetFeatureMatchingOptions(options,param,hybrid_options.child_id);

    if (hybrid_options.child_id > 0 && options.retrieve_type == FeatureMatchingOptions::RetrieveType::VLAD) {
        // vlad is not good for different cameras
        options.retrieve_type = FeatureMatchingOptions::RetrieveType::SIFT;
    }

    // use intial sfm to filter far image pairs. 
    bool use_initial_sfm = static_cast<bool>(param.GetArgument("use_initial_sfm", 0));

    bool has_initial_sfm = false;
    if(boost::filesystem::is_directory(workspace_path + "/initial_sfm")&&
       boost::filesystem::exists(workspace_path + "/initial_sfm/cameras.bin")&&
       boost::filesystem::exists(workspace_path + "/initial_sfm/images.bin")&&
       boost::filesystem::exists(workspace_path + "/initial_sfm/points3D.bin")){
       has_initial_sfm = true;
    }
    
    if(use_initial_sfm && has_initial_sfm){
        std::unordered_map<image_t, Eigen::Vector4d> prior_rotations;
        std::unordered_map<image_t, Eigen::Vector3d> prior_translations;
        std::unordered_map<image_t, bool> prior_pose_validations;
        auto reconstruction = std::make_shared<Reconstruction>();
        reconstruction->ReadReconstruction(workspace_path+"/initial_sfm");

        std::vector<image_t> image_ids = feature_data_container.GetImageIds();

        for (const auto image_id : image_ids) {
            if (!(reconstruction->ExistsImage(image_id)&&reconstruction->IsImageRegistered(image_id))) {
                prior_pose_validations.emplace(image_id,false);
                continue;
            }
            const auto& image = reconstruction->Image(image_id);
            prior_rotations.emplace(image_id,image.Qvec());
            prior_translations.emplace(image_id,image.Tvec());
            prior_pose_validations.emplace(image_id,true);
        }

        reconstruction->ComputeBaselineDistance();
        for(const auto image_id: image_ids){
            if (!(reconstruction->ExistsImage(image_id)&&reconstruction->IsImageRegistered(image_id))) {     
                continue;
            }

            Eigen::Vector4d qvec = prior_rotations.at(image_id);
			Eigen::Vector3d tvec = prior_translations.at(image_id);
					
			Eigen::Vector3d C = ProjectionCenterFromPose(qvec,tvec); 
            
            bool valid = false;
            if(image_id>1&&prior_rotations.find(image_id-1)!=prior_rotations.end()){

                Eigen::Vector4d qvec_previous = prior_rotations.at(image_id-1);
			    Eigen::Vector3d tvec_previous = prior_translations.at(image_id-1);					
			    Eigen::Vector3d C_previous = ProjectionCenterFromPose(qvec_previous,tvec_previous); 

                double distance = (C-C_previous).norm();
                if(distance < reconstruction->baseline_distance * 30){
                    valid = true;
                }
            }
            if(!valid&&prior_rotations.find(image_id+1)!= prior_rotations.end()){
                Eigen::Vector4d qvec_next = prior_rotations.at(image_id+1);
			    Eigen::Vector3d tvec_next = prior_translations.at(image_id+1);					
			    Eigen::Vector3d C_next = ProjectionCenterFromPose(qvec_next,tvec_next); 

                double distance = (C-C_next).norm();
                if(distance < reconstruction->baseline_distance * 30){
                    valid = true;
                }
            }

            if(!valid){
                CHECK(prior_pose_validations.find(image_id)!=prior_pose_validations.end());
                prior_pose_validations.at(image_id) = false;
            }
        }

        options.prior_neighbor_distance = reconstruction->baseline_distance;
        options.have_prior_pose_ = true;
        options.prior_rotations = prior_rotations;
        options.prior_translations = prior_translations;
        options.prior_pose_validations = prior_pose_validations;
        options.max_match_distance = param.GetArgument("max_match_distance",20.0f);
    }

    if (hybrid_options.update_flag){
        scene_graph.CorrespondenceGraph()->Finalize();
        options.match_between_reconstructions_ = true;

        std::cout<<"fm new image: "<<feature_data_container.GetNewImageIds().size()<<std::endl;
    }

    if (hybrid_options.manual_match) {
        const auto new_image_ids = feature_data_container.GetNewImageIds();
        const auto old_image_ids = feature_data_container.GetOldImageIds();

        int64_t new_image_index_start = std::numeric_limits<int64_t>::max();
        int64_t old_image_index_start = std::numeric_limits<int64_t>::max();
        for (auto & id : new_image_ids) {
            new_image_index_start = std::min(
                new_image_index_start, 
                ImageNameToIndex(feature_data_container.GetImage(id).Name()));
        }
        for (auto & id : old_image_ids) {
            old_image_index_start = std::min(
                old_image_index_start, 
                ImageNameToIndex(feature_data_container.GetImage(id).Name()));
        }

        const int overlap = hybrid_options.manual_match_overlap;
        const int step = hybrid_options.manual_match_overlap_step;

        std::unordered_set<image_t> old_image_set(old_image_ids.begin(), old_image_ids.end());

        options.track_preoperation = false;
        options.match_between_reconstructions_ = false;
        options.method_ = FeatureMatchingOptions::MatchMethod::MANUAL;
        options.manual_matching_.generate_pairs_for_image = [
            overlap, step, 
            new_image_index_start, old_image_index_start, 
            old_image_set
        ](image_t current_image, const FeatureDataContainer & feature_data_container) {
            int64_t current_offset = ImageNameToIndex(feature_data_container.GetImage(current_image).Name()) - new_image_index_start;
            CHECK_GE(current_offset, 0);

            std::vector<image_t> selected_images;
            for (image_t target_image : old_image_set) {
                int64_t target_offset = ImageNameToIndex(feature_data_container.GetImage(target_image).Name()) - old_image_index_start;
                CHECK_GE(target_offset, 0);

                if (target_offset >= current_offset - overlap &&
                    target_offset <= current_offset + overlap
                ) {
                    selected_images.emplace_back(target_image);
                }
            }
            std::sort(selected_images.begin(), selected_images.end());

            std::vector<image_t> images;
            for (int i = 0; i < selected_images.size(); i += step) {
                images.emplace_back(selected_images[i]);
            }
            return images;
        };
    }

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

    if (hybrid_options.save_flag){
        scene_graph.CorrespondenceGraph()->ExportToGraph(workspace_path + "/scene_graph.png");
        std::cout << "ExportToGraph done!" << std::endl;
    }

    bool write_match = static_cast<bool>(param.GetArgument("write_match", 0));
    bool write_binary = static_cast<bool>(param.GetArgument("write_binary", 1));

    if (write_match && hybrid_options.save_flag) {
        if (write_binary) {
            scene_graph.WriteSceneGraphBinaryData(workspace_path + "/scene_graph.bin");
        } else {
            scene_graph.WriteSceneGraphData(workspace_path + "/scene_graph.txt");
        }
    }
}

void IncrementalSFM(const std::shared_ptr<sensemap::SceneGraphContainer> &scene_graph_container,
                    std::shared_ptr<sensemap::ReconstructionManager> &reconstruction_manager, 
                    Configurator &param, struct HybridOptions hybrid_options = HybridOptions()) {
    using namespace sensemap;

    PrintHeading1("Incremental Mapping");

    auto options = std::make_shared<MapperOptions>();
    options->mapper_type = MapperType::INDEPENDENT;
    options->independent_mapper_options.outside_mapper_type = MapperType::INDEPENDENT;
    options->independent_mapper_options.independent_mapper_type = IndependentMapperType::INCREMENTAL;

    OptionParser option_parser;
    ImageReaderOptions reader_options;
    option_parser.GetImageReaderOptions(reader_options,param);
    if (hybrid_options.child_id >= 0) {
        option_parser.GetImageReaderOptions(reader_options,param,hybrid_options.child_id);
    }
    option_parser.GetMapperOptions(options->independent_mapper_options,param,hybrid_options.child_id);

    std::string workspace_path = param.GetArgument("workspace_path", "");
    CHECK(!workspace_path.empty()) << "image path empty";

    std::string image_path = param.GetArgument("image_path", "");
    CHECK(!workspace_path.empty()) << "image path empty";

    if (hybrid_options.child_id > -1) {
        Configurator camera_param;
        std::string cameras_param_file = param.GetArgument("camera_param_file", "");
        camera_param.Load(cameras_param_file.c_str());
        std::string child_name = camera_param.GetArgument("sub_path_" + std::to_string(hybrid_options.child_id), "");
        CHECK(!child_name.empty());

        workspace_path = JoinPaths(workspace_path, child_name);
    }
    CreateDirIfNotExists(workspace_path);

    // use prior pose from slam to constrain SfM
    bool use_slam_graph = static_cast<bool>(param.GetArgument("use_slam_graph", 0));
    std::string preview_pose_file = param.GetArgument("preview_pose_file","");
    if (use_slam_graph && (!preview_pose_file.empty()) && boost::filesystem::exists(preview_pose_file)) {
        std::vector<Keyframe> keyframes;
        if (boost::filesystem::path(preview_pose_file).extension().string() == ".tum") {
            LoadPriorPoseFromTum(preview_pose_file, keyframes, (param.GetArgument("image_type", "").compare("rgbd") == 0));
        }
        else {
            LoadPirorPose(preview_pose_file, keyframes);
        }
        
        std::unordered_map<std::string,Keyframe> keyframe_map;
        for(auto const keyframe:keyframes){
            keyframe_map.emplace(keyframe.name,keyframe);
        }

        std::unordered_map<image_t, Eigen::Vector4d> prior_rotations;
        std::unordered_map<image_t, Eigen::Vector3d> prior_translations;

        std::string camera_rig_params_file = param.GetArgument("rig_params_file", "");
        CameraRigParams rig_params;
        bool camera_rig = false;
        Eigen::Matrix3d R0;
        Eigen::Vector3d t0;

        if (!camera_rig_params_file.empty()) {
            if (rig_params.LoadParams(camera_rig_params_file)) {
                camera_rig = true;
                R0 = rig_params.local_extrinsics[0].block<3,3>(0,0);
                t0 = rig_params.local_extrinsics[0].block<3,1>(0,3);

            } else {
                std::cout << "failed to read rig params" << std::endl;
                exit(-1);
            }
        }

        std::vector<image_t> image_ids = scene_graph_container->GetImageIds();
        for (const auto image_id : image_ids) {
            const Image &image = scene_graph_container->Image(image_id);
            
            std::string name = image.Name();
            if(keyframe_map.find(name)!=keyframe_map.end()){
                Keyframe keyframe = keyframe_map.at(name);
        
                Eigen::Matrix3d r = keyframe.rot;
                Eigen::Vector4d q = RotationMatrixToQuaternion(r);
                Eigen::Vector3d t = keyframe.pos;

                if(camera_rig){
                    Eigen::Matrix3d R_rig = R0.transpose()*r;
                    Eigen::Vector3d t_rig = R0.transpose()*(t-t0);
                    q = RotationMatrixToQuaternion(R_rig);
                    t = t_rig;
                }
                prior_rotations.emplace(image_id,q);
                prior_translations.emplace(image_id,t);
            }
        }

        options->independent_mapper_options.prior_rotations = prior_rotations;
        options->independent_mapper_options.prior_translations = prior_translations;
        options->independent_mapper_options.have_prior_pose = true;
    }

    // use gps location prior to constrain the image
    bool use_gps_prior = static_cast<bool>(param.GetArgument("use_gps_prior", 0));
    bool use_prior_align_only = param.GetArgument("use_prior_align_only", 1);
    std::string gps_prior_file = param.GetArgument("gps_prior_file","");
    std::string gps_trans_file = workspace_path + "/gps_trans.txt";
    if (use_gps_prior){
        if (boost::filesystem::exists(gps_prior_file)) {
            auto image_ids = scene_graph_container->GetImageIds();
            std::vector<std::string> image_names;
            for (const auto image_id : image_ids) {
                const Image &image = scene_graph_container->Image(image_id);
                std::string name = image.Name();
                image_names.push_back(name);
            }

            std::vector<std::pair<unsigned long, std::pair<Eigen::Vector3d,int>>> gps_locations;
            if(options->independent_mapper_options.optimization_use_horizontal_gps_only){
                LoadOriginGPSinfo(gps_prior_file, gps_locations,gps_trans_file, true);
            }
            else{
                LoadOriginGPSinfo(gps_prior_file, gps_locations,gps_trans_file, false);
            }
            std::unordered_map<std::string, std::pair<Eigen::Vector3d,int>> image_locations;
            GPSLocationsToImages(gps_locations, image_names, image_locations);
            std::unordered_map<image_t, std::pair<Eigen::Vector3d,int>> prior_locations_gps;   
            std::cout<<image_locations.size()<<" images have gps prior"<<std::endl;

            std::vector<PlyPoint> gps_locations_ply;
            for (const auto image_id : image_ids) {
                const Image &image = scene_graph_container->Image(image_id);
                std::string name = image.Name();

                if(image_locations.find(name)!=image_locations.end()){
                    prior_locations_gps.emplace(image_id,image_locations.at(name));

                    PlyPoint gps_location_ply;
                    gps_location_ply.r = 255;
                    gps_location_ply.g = 0;
                    gps_location_ply.b = 0;
                    gps_location_ply.x = image_locations.at(name).first[0];
                    gps_location_ply.y = image_locations.at(name).first[1];
                    gps_location_ply.z = image_locations.at(name).first[2];
                    gps_locations_ply.push_back(gps_location_ply);
                }
            }
            options->independent_mapper_options.prior_locations_gps = prior_locations_gps;

            sensemap::WriteBinaryPlyPoints(workspace_path+"/gps.ply", gps_locations_ply, false, true);
        }

        options->independent_mapper_options.has_gps_prior = true;
        options->independent_mapper_options.use_prior_align_only = use_prior_align_only;
        options->independent_mapper_options.min_image_num_for_gps_error = param.GetArgument("min_image_num_for_gps_error", 10);
        double prior_absolute_location_weight = 
            static_cast<double>(param.GetArgument("prior_absolute_location_weight", 1.0f));
        std::cout<<"[exe] prior_absolute_location_weight: "<<prior_absolute_location_weight<<std::endl;
        options->independent_mapper_options.prior_absolute_location_weight = prior_absolute_location_weight;
    } 

    int num_local_cameras = reader_options.num_local_cameras;

    MapperController *mapper = 
        MapperController::Create(options, workspace_path, image_path, scene_graph_container, reconstruction_manager);
    mapper->Start();
    mapper->Wait();

    fprintf(
        fs, "%s\n",
        StringPrintf("Incremental Mapper Elapsed time: %.3f [minutes]", mapper->GetTimer().ElapsedMinutes()).c_str());
    fflush(fs);

    for (size_t i = 0; i < reconstruction_manager->Size(); ++i) {
        auto reconstruction = reconstruction_manager->Get(i);
        CHECK(reconstruction->RegisterImageIds().size()>0);
        // const image_t first_image_id = reconstruction->RegisterImageIds()[0]; 
        // CHECK(reconstruction->ExistsImage(first_image_id));
        // const Image& image = reconstruction->Image(first_image_id);
        // const Camera& camera = reconstruction->Camera(image.CameraId());
        bool camera_rig = false;
        const auto& camera_ids = reconstruction->Cameras();
        for (auto camera : camera_ids){
            if (camera.second.NumLocalCameras() > 1){
                camera_rig = true;
            }
        }
        
        std::string rec_path = StringPrintf("%s/%d", workspace_path.c_str(), i);
        if (!boost::filesystem::exists(rec_path)) {
            boost::filesystem::create_directories(rec_path);
        }

        if (use_gps_prior) {
            Eigen::Matrix3x4d matrix_to_align =
            reconstruction->AlignWithPriorLocations(options->independent_mapper_options.max_error_gps);
            if( hybrid_options.save_flag){
                if( !camera_rig){
                    Reconstruction rec = *reconstruction.get();
                    rec.AddPriorToResult();
                    rec.NormalizeWoScale();
                    
                    std::string trans_rec_path = rec_path + "-gps";
                    if (!boost::filesystem::exists(trans_rec_path)) {
                        boost::filesystem::create_directories(trans_rec_path);
                    }
                    rec.WriteBinary(trans_rec_path);
                }
                // Eigen::Matrix3x4d matrix_to_geo = reconstruction->NormalizeWoScale();
                // Eigen::Matrix3d RT = matrix_to_geo.block<3, 3>(0, 0).transpose();
                // Eigen::Matrix3x4d M = Eigen::Matrix3x4d::Identity();
                // M.block<3, 3>(0, 0) = RT;
                // M.block<3, 1>(0, 3) = -RT * matrix_to_geo.block<3, 1>(0, 3);

                // std::ofstream file(JoinPaths(workspace_path, "/local_to_ned.txt"), std::ofstream::out);
                // file << MAX_PRECISION << M(0, 0) << " " << M(0, 1) << " " 
                //     << M(0, 2) << " " << M(0, 3) << std::endl;
                // file << MAX_PRECISION << M(1, 0) << " " << M(1, 1) << " " 
                //     << M(1, 2) << " " << M(1, 3) << std::endl;
                // file << MAX_PRECISION << M(2, 0) << " " << M(2, 1) << " " 
                //     << M(2, 2) << " " << M(2, 3) << std::endl;
                // file.close();
            }
        }

        if (camera_rig && hybrid_options.save_flag) {
            reconstruction->WriteReconstruction(rec_path,
                options->independent_mapper_options.write_binary_model);

            std::string export_rec_path = rec_path + "-export";
            if (!boost::filesystem::exists(export_rec_path)) {
                boost::filesystem::create_directories(export_rec_path);
            }
            Reconstruction rig_reconstruction;
            reconstruction->ConvertRigReconstruction(rig_reconstruction);

            // BundleAdjustmentOptions custom_options =
            //     options->independent_mapper_options.GlobalBundleAdjustment();
            // PrintHeading1("GBA with free camera rig");
            // FreeGBA(rig_reconstruction, custom_options);

            rig_reconstruction.WriteReconstruction(export_rec_path,
                options->independent_mapper_options.write_binary_model);
        } else if (hybrid_options.save_flag){
            reconstruction->WriteReconstruction(rec_path,
                options->independent_mapper_options.write_binary_model);
        }
    }
}

//////////////////////////// Update SFM/////////////////////////////////////////
size_t CompleteAndMergeTracks(const IndependentMapperOptions &options, std::shared_ptr<IncrementalMapper> mapper,
                              const std::unordered_set<mappoint_t> &mappoints) {
    size_t num_merged_observations = mapper->MergeTracks(options.Triangulation(), mappoints);
    std::cout << "  => Merged observations: " << num_merged_observations << std::endl;

    size_t num_completed_observations = mapper->CompleteTracks(options.Triangulation(), mappoints);
    std::cout << "  => Completed observations: " << num_completed_observations << std::endl;

    return num_completed_observations + num_merged_observations;
}

size_t FilterPoints(const IndependentMapperOptions &options, std::shared_ptr<IncrementalMapper> mapper,
                    const std::unordered_set<mappoint_t>& addressed_points) {
    const size_t num_filtered_observations = mapper->FilterPoints(options.IncrementalMapperOptions(),addressed_points);
    std::cout << "  => Filtered observations: " << num_filtered_observations << std::endl;
    return num_filtered_observations;
}

size_t FilterImages(const IndependentMapperOptions &options, std::shared_ptr<IncrementalMapper> mapper,
                    const std::unordered_set<image_t>& addressed_images) {
    const size_t num_filtered_images = mapper->FilterImages(options.IncrementalMapperOptions(),addressed_images);
    std::cout << "  => Filtered images: " << num_filtered_images << std::endl;
    return num_filtered_images;
}

void IterativeGlobalRefinement(const IndependentMapperOptions &mapper_options,
                               std::shared_ptr<IncrementalMapper> mapper,
                               std::shared_ptr<Reconstruction> reconstruction,
                               std::unordered_set<mappoint_t> &all_variable_mappoints,
                               std::unordered_set<image_t> &reg_new_image_ids,
                               const std::unordered_set<image_t> &fixed_image_ids, 
                               const std::unordered_set<mappoint_t>&fixed_mappoint_ids, Timer &map_update_timer,
                               double &merge_time_cost, double &filter_time_cost, double &ba_time_cost,
                               const size_t &ba_new_num_reg_images) {
    BundleAdjustmentOptions ba_options = mapper_options.GlobalBundleAdjustment();

    map_update_timer.Restart();
    CompleteAndMergeTracks(mapper_options, mapper, all_variable_mappoints);
    std::cout << "  => Retriangulated observations: "
              << mapper->Retriangulate(mapper_options.Triangulation(), &reg_new_image_ids) << std::endl;
    merge_time_cost += map_update_timer.ElapsedMicroSeconds();

    std::unordered_set<mappoint_t> modified_points = mapper->GetModifiedMapPoints();
    for (auto mappoint_id : modified_points) {
        if (all_variable_mappoints.count(mappoint_id) == 0) {
            all_variable_mappoints.insert(mappoint_id);
        }
    }
    mapper->ClearModifiedMapPoints();

    if (mapper_options.with_depth && mapper_options.rgbd_delayed_start && !reconstruction->depth_enabled) {
        reconstruction->depth_enabled = reconstruction->TryScaleAdjustmentWithDepth(mapper_options.rgbd_delayed_start_weights);
    }

    auto &registered_images = mapper->GetReconstruction().RegisterImageIds();
    CHECK_GE(registered_images.size(), 2) << "At least two images must be "
                                             "registered for global "
                                             "bundle-adjustment";

    for (size_t iter = 0; iter < mapper_options.ba_global_max_refinements; iter++) {
        const size_t num_observations = mapper->GetReconstruction().ComputeNumObservations();
        // Avoid degeneracies in bundle adjustment.
        reconstruction->FilterObservationsWithNegativeDepth();

        if (mapper_options.single_camera && (ba_new_num_reg_images < mapper_options.num_fix_camera_first)) {
            ba_options.refine_focal_length = false;
            ba_options.refine_extra_params = false;
            ba_options.refine_principal_point = false;
            ba_options.refine_local_extrinsics = false;
        }

        BundleAdjustmentConfig ba_config;
        std::unordered_set<image_t> all_variable_images = reconstruction->FindImageForMapPoints(all_variable_mappoints);

        for (auto &image_id : reg_new_image_ids) {
            if (reconstruction->IsImageRegistered(image_id)) {
                ba_config.AddImage(image_id);
                const Image& image = reconstruction->Image(image_id);
                const Camera& camera = reconstruction->Camera(image.CameraId());
                if (camera.IsCameraConstant()) {
                    ba_config.SetConstantCamera(image.CameraId());
                }
            }
        }

        std::cout << "GBA image count: " << ba_config.NumImages() << std::endl;

        map_update_timer.Restart();
        if (ba_config.NumImages() > 1) {
            BundleAdjuster bundle_adjuster(ba_options, ba_config);
            if (!bundle_adjuster.Solve(reconstruction.get())) {
                std::cout << "Bundle Adjustment Failed!" << std::endl;
            }
        }
        ba_time_cost += map_update_timer.ElapsedMicroSeconds();

        map_update_timer.Restart();
        size_t num_merged_and_completed_observations =
            CompleteAndMergeTracks(mapper_options, mapper, all_variable_mappoints);
        merge_time_cost += map_update_timer.ElapsedMicroSeconds();

        
        std::unordered_set<mappoint_t> new_mappoints;
        for(const auto& mappoint_id: all_variable_mappoints){
            if(fixed_mappoint_ids.count(mappoint_id)==0){
                new_mappoints.insert(mappoint_id);
            }
        }   

        map_update_timer.Restart();
        const size_t num_filtered_observations = FilterPoints(mapper_options, mapper, new_mappoints);
        const size_t num_filtered_images = FilterImages(mapper_options, mapper, reg_new_image_ids);
        filter_time_cost += map_update_timer.ElapsedMicroSeconds();

        std::unordered_set<mappoint_t> modified_points = mapper->GetModifiedMapPoints();
        for (auto mappoint_id : modified_points) {
            if (all_variable_mappoints.count(mappoint_id) == 0) {
                all_variable_mappoints.insert(mappoint_id);
            }
        }
        mapper->ClearModifiedMapPoints();

        if (mapper_options.with_depth && reconstruction->depth_enabled) {
            reconstruction->TryScaleAdjustmentWithDepth(0.0);
        }

        size_t num_changed_observations = num_merged_and_completed_observations + num_filtered_observations;

        const double changed = static_cast<double>(num_changed_observations) / num_observations;
        std::cout << StringPrintf("  => Changed observations: %.6f", changed) << std::endl;
        if (changed < 0.0005) {
            break;
        }
    }
}

void MapUpdate(const std::shared_ptr<sensemap::FeatureDataContainer> &feature_data_container,
               const std::shared_ptr<sensemap::SceneGraphContainer> &scene_graph_container,
               std::shared_ptr<sensemap::ReconstructionManager> &reconstruction_manager, 
               Configurator &param, struct HybridOptions hybrid_options = HybridOptions()) {
    Timer timer;
    timer.Start();

    ImageReaderOptions reader_options;
    OptionParser option_parser;
    option_parser.GetImageReaderOptions(reader_options,param);

    std::string workspace_path = param.GetArgument("workspace_path", "");
    CHECK(!workspace_path.empty()) << "workspace path empty";

    std::string image_path = param.GetArgument("image_path", "");
    CHECK(!image_path.empty()) << "image path empty";

    if (hybrid_options.child_id > -1 && hybrid_options.debug_info){
        Configurator camera_param;        
        std::string cameras_param_file = param.GetArgument("camera_param_file", "");    
        camera_param.Load(cameras_param_file.c_str());
        std::string child_name = camera_param.GetArgument("sub_path_" + 
                                 std::to_string(hybrid_options.child_id), "");
        CHECK(!child_name.empty());

        workspace_path = JoinPaths(workspace_path, child_name);
        CHECK(boost::filesystem::exists(workspace_path));
    }
    
    // Load reconstructions
    // auto reconstruction = std::make_shared<Reconstruction>();
    auto reconstruction = reconstruction_manager->Get(0);

    std::cout << std::endl;
    timer.PrintMinutes();
    std::cout << std::endl;
    fprintf(fs, "%s\n", StringPrintf("Map Update Read Elapsed time: %.3f [minutes]", timer.ElapsedMinutes()).c_str());
    fflush(fs);
    timer.Restart();

    // Convert all the image id to 0
    auto old_image_ids = reconstruction->RegisterImageIds();
    std::unordered_set<image_t> fixed_image_ids;
    std::unordered_set<camera_t> fixed_camera_ids; 
    camera_t initial_reconstruction_camera_id;
    bool initial_reconstruction_camera_set = false;
    for (auto old_image_id : old_image_ids) {
        reconstruction->Image(old_image_id).SetLabelId(0);
        fixed_image_ids.insert(old_image_id);
        fixed_camera_ids.insert(reconstruction->Image(old_image_id).CameraId());
        if(!initial_reconstruction_camera_set){
            initial_reconstruction_camera_id = reconstruction->Image(old_image_id).CameraId();
            initial_reconstruction_camera_set = true;
        }
    }
    auto fixed_point_ids = reconstruction->MapPointIds();

    PrintHeading1("Update new images");
    IndependentMapperOptions mapper_options;
    option_parser.GetMapperOptions(mapper_options, param, hybrid_options.child_id);

    std::shared_ptr<IncrementalMapper> mapper = std::make_shared<IncrementalMapper>(scene_graph_container);

    std::unordered_set<mappoint_t> const_mappoint_ids = reconstruction->MapPointIds();

    std::cout << "Current reconstruction image number before setup = " << reconstruction->Images().size() << std::endl;
    std::cout << "Current reconstruction registed image number = " << reconstruction->RegisterImageIds().size()
              << std::endl;
    std::cout << "Current reconstruction mappoint number = " << reconstruction->MapPointIds().size() << std::endl;
    mapper->BeginReconstruction(reconstruction);
    std::cout << "Current reconstruction image number after setup = " << reconstruction->Images().size() << std::endl;
    std::cout << "Current reconstruction registed image number = " << reconstruction->RegisterImageIds().size()
              << std::endl;
    std::cout << "Current reconstruction mappoint number = " << reconstruction->MapPointIds().size() << std::endl;

    auto cameras = reconstruction->Cameras();
    std::cout << "Camera number = " << cameras.size() << std::endl;

    for (auto camera : cameras) {
        if(camera.second.NumLocalCameras()==1){
            std::cout << "  Camera index = " << camera.first << std::endl;
            std::cout << "  Camera model = " << camera.second.ModelName() << std::endl;
            std::cout << "  Camera param = ";
            for (auto param : camera.second.Params()) {
                std::cout << "  " << param;
            }
            std::cout << std::endl;
        }
        else{
            std::cout<<"camera id: "<<camera.first<<std::endl;
            std::cout<< "local camera param = ";
            for (auto param : camera.second.LocalParams()) {
                std::cout << "  " << param;
            }
            std::cout<<std::endl;
            for (auto qvec: camera.second.LocalQvecs()){
                std::cout<< " "<<qvec;
            }
            std::cout<<std::endl;
            for (auto tvec: camera.second.LocalTvecs()){
                std::cout<< " "<<tvec;
            }
            std::cout << std::endl;
        }
    }

    int update_image_counter = 0;
    Timer map_update_timer;
    map_update_timer.Start();
    double merge_time_cost = 0;
    double filter_time_cost = 0;
    double image_update_cost = 0;
    double ba_time_cost = 0;


    size_t ba_old_num_reg_images = reconstruction->NumRegisterImages();
    size_t ba_old_num_points = reconstruction->NumMapPoints();

    size_t ba_new_num_reg_images;
    size_t ba_new_num_points;

    size_t ba_prev_num_reg_images = 1;
    size_t ba_prev_num_points = 1;

    mapper->ClearModifiedMapPoints();
    std::unordered_set<mappoint_t> all_variable_mappoints;
    std::unordered_set<image_t> all_variable_images;

    std::unordered_set<image_t> new_image_set;
    for (auto image_id : feature_data_container->GetNewImageIds()) {
        new_image_set.insert(image_id);
    }
    std::cout << "New image count: " << new_image_set.size() << std::endl;
    std::unordered_set<image_t> reg_new_image_ids;

    if (mapper_options.with_depth) {
        reconstruction->rgbd_filter_depth_weight = mapper_options.rgbd_filter_depth_weight;
        reconstruction->rgbd_max_reproj_depth = mapper_options.rgbd_max_reproj_depth;
        auto options = mapper_options.IncrementalMapperOptions();
        options.image_path = image_path;
        mapper->ComputeDepthInfo(options);
        
        if (mapper_options.rgbd_delayed_start) {
            reconstruction->depth_enabled = false;
        }
    }

    bool reg_next_success = true;

    while (reg_next_success) {
        std::vector<std::pair<image_t, float>> next_images;
        next_images = mapper->FindNextImages(mapper_options.IncrementalMapperOptions());

        if (next_images.empty()) {
            std::cout << "Could not find next images" << std::endl;
            break;
        }
        bool have_new_image = false;
        for (size_t i = 0; i < next_images.size(); ++i) {
            if (new_image_set.count(next_images[i].first) > 0) {
                have_new_image = true;
                break;
            }
        }
        if (!have_new_image) {
            std::cout << "no new image to register" << std::endl;
            break;
        }

        for (int image_id = 0; image_id < next_images.size(); image_id++) {
            image_t next_image_id = next_images[image_id].first;
            if (!new_image_set.count(next_image_id)) {
                continue;
            }
            if (!reconstruction->ExistsImage(next_image_id)) {
                continue;
            }
            const class Image &image = reconstruction->Image(next_image_id);
            if (image.IsRegistered()) {
                continue;
            }

            PrintHeading1(StringPrintf("Registering #%d (%d / %d) name: (%s), total: %d", next_image_id,
                                       reg_new_image_ids.size() + 1, new_image_set.size(),
                                       reconstruction->Image(next_image_id).Name().c_str(),
                                       reconstruction->NumRegisterImages() + 1));

            auto cur_options = mapper_options.IncrementalMapperOptions();
            cur_options.single_camera = true;

            std::vector<std::pair<point2D_t, mappoint_t>> tri_corrs;
            std::vector<char> inlier_mask;

            const Image &next_image = reconstruction->Image(next_image_id);
		    const Camera& next_camera = reconstruction->Camera(next_image.CameraId());


            if(next_camera.NumLocalCameras()>1){
			    reg_next_success = 
                    mapper->EstimateCameraPoseRig(mapper_options.IncrementalMapperOptions(), next_image_id, tri_corrs, inlier_mask);
		    }
            else{
                reg_next_success =
                    mapper->EstimateCameraPose(mapper_options.IncrementalMapperOptions(), next_image_id, tri_corrs, inlier_mask);
            }
            if (!reg_next_success) {
                continue;
            }

            int num_first_force_be_keyframe = mapper_options.num_first_force_be_keyframe;
            bool force = (!mapper_options.extract_keyframe) || (reg_new_image_ids.size() < num_first_force_be_keyframe);

            // Only triangulation on KeyFrame.
            if (!mapper->AddKeyFrameUpdate(mapper_options.IncrementalMapperOptions(), next_image_id, tri_corrs, inlier_mask,
                                     force)) {
                continue;
            }

            // Triangulation and local BA
            map_update_timer.Restart();
            TriangulateImage(mapper_options, image, mapper);

            update_image_counter++;
            reg_new_image_ids.insert(next_image_id);

            auto local_ba_options = mapper_options.LocalBundleAdjustment();

            if (mapper_options.single_camera) {
                local_ba_options.refine_extra_params = false;
                local_ba_options.refine_focal_length = false;
                local_ba_options.refine_principal_point = false;
                local_ba_options.refine_local_extrinsics = false;
            }

            PrintHeading1("Local BA");
            for (int i = 0; i < mapper_options.ba_local_max_refinements; ++i) {
                const auto report = mapper->AdjustLocalBundle(mapper_options.IncrementalMapperOptions(), local_ba_options,
                                                              mapper_options.Triangulation(), next_image_id,
                                                              mapper->GetModifiedMapPoints(), fixed_image_ids);
                std::cout << "  => Merged observations: " << report.num_merged_observations << std::endl;
                std::cout << "  => Completed observations: " << report.num_completed_observations << std::endl;
                std::cout << "  => Filtered observations: " << report.num_filtered_observations << std::endl;

                const double changed = (report.num_merged_observations + report.num_completed_observations +
                                        report.num_filtered_observations) /
                                       static_cast<double>(report.num_adjusted_observations);
                std::cout << StringPrintf("  => Changed observations: %.6f", changed) << std::endl;
                if (changed < mapper_options.ba_local_max_refinement_change) {
                    break;
                }
            }

            // if (mapper_options.extract_colors) {
            //     reconstruction->ExtractColorsForImage(next_image_id, image_path);
            // }
            image_update_cost += map_update_timer.ElapsedMicroSeconds();


            std::unordered_set<mappoint_t> modified_points = mapper->GetModifiedMapPoints();
            for (auto mappoint_id : modified_points) {
                if (all_variable_mappoints.count(mappoint_id) == 0) {
                    all_variable_mappoints.insert(mappoint_id);
                }
            }
            mapper->ClearModifiedMapPoints();

            // Global BA when neccessary
            ba_new_num_reg_images = reconstruction->NumRegisterImages() > ba_old_num_reg_images
                                        ? (reconstruction->NumRegisterImages() - ba_old_num_reg_images)
                                        : 1;
            ba_new_num_points = reconstruction->NumMapPoints() > ba_old_num_points
                                    ? (reconstruction->NumMapPoints() - ba_old_num_points)
                                    : 1;

            double image_ratio = ba_new_num_reg_images * 1.0f / ba_prev_num_reg_images;
            double image_freq = ba_new_num_reg_images - ba_prev_num_reg_images;
            double points_ratio = ba_new_num_points * 1.0f / ba_prev_num_points;
            double points_freq = ba_new_num_points - ba_prev_num_points;
            
            if (mapper_options.use_global_ba_update &&
                (ba_new_num_reg_images >= mapper_options.ba_global_images_ratio * ba_prev_num_reg_images ||
                ba_new_num_reg_images >= mapper_options.ba_global_images_freq + ba_prev_num_reg_images ||
                ba_new_num_points >= mapper_options.ba_global_points_ratio * ba_prev_num_points ||
                ba_new_num_points >= mapper_options.ba_global_points_freq + ba_prev_num_points)) {
                PrintHeading1("Global BA");

                IterativeGlobalRefinement(mapper_options, mapper, reconstruction, all_variable_mappoints,
                                          reg_new_image_ids, fixed_image_ids, fixed_point_ids,
                                          map_update_timer, merge_time_cost,
                                          filter_time_cost, ba_time_cost, ba_new_num_reg_images);

                ba_prev_num_reg_images = reconstruction->NumRegisterImages() > ba_old_num_reg_images
                                             ? (reconstruction->NumRegisterImages() - ba_old_num_reg_images)
                                             : 1;
                ba_prev_num_points = reconstruction->NumMapPoints() > ba_old_num_points
                                         ? (reconstruction->NumMapPoints() - ba_old_num_points)
                                         : 1;
            }
            if (reg_next_success) {
                break;
            }
        }
    }

    if (mapper_options.register_nonkeyframe) {
        for (const auto &image_id : new_image_set) {
            if (!reconstruction->ExistsImage(image_id)) {
                continue;
            }                

            class Image& image = reconstruction->Image(image_id);
            const Camera& camera = reconstruction->Camera(image.CameraId());
            if (image.IsRegistered()) {
                continue;
            }
            PrintHeading1(StringPrintf("Registering NonKeyFrame #%d (%d / %d) name: (%s)", image_id,
                                       reg_new_image_ids.size() + 1, new_image_set.size(), 
                                       image.Name().c_str()));
            
            mapper->ClearModifiedMapPoints();

            if(camera.NumLocalCameras() == 1){
                if(!mapper->RegisterNonKeyFrame(mapper_options.IncrementalMapperOptions(), image_id)){
                    continue;
                }
            }
            else{
                if(!mapper->RegisterNonKeyFrameRig(mapper_options.IncrementalMapperOptions(), image_id)){
                    continue;
                }
            }

            reg_new_image_ids.insert(image_id);
        
            TriangulateImage(mapper_options, image, mapper);
            // if (mapper_options.extract_colors) {
            //     reconstruction->ExtractColorsForImage(image_id, image_path);
            // }

            update_image_counter++;

            std::unordered_set<mappoint_t> modified_points = mapper->GetModifiedMapPoints();
            for (auto mappoint_id : modified_points) {
                if (all_variable_mappoints.count(mappoint_id) == 0) {
                    all_variable_mappoints.insert(mappoint_id);
                }
            }
        }
    }

    ba_new_num_reg_images = reconstruction->NumRegisterImages() > ba_old_num_reg_images
                                        ? (reconstruction->NumRegisterImages() - ba_old_num_reg_images)
                                        : 1;
    PrintHeading1("Final Global BA");
    IterativeGlobalRefinement(mapper_options, mapper, reconstruction, all_variable_mappoints, reg_new_image_ids,
                              fixed_image_ids, fixed_point_ids,map_update_timer, merge_time_cost, filter_time_cost, 
                              ba_time_cost, ba_new_num_reg_images);

    std::cout
        << StringPrintf("Map Update Merge Track Elapsed time: %.3f [minutes]", merge_time_cost / (1e6 * 60)).c_str()
        << std::endl;
    std::cout
        << StringPrintf("Map Update Filte Point Elapsed time: %.3f [minutes]", filter_time_cost / (1e6 * 60)).c_str()
        << std::endl;
    std::cout << StringPrintf("Map Update Register and LBA Elapsed time: %.3f [minutes]",
                              image_update_cost / (1e6 * 60))
                     .c_str()
              << std::endl;
    std::cout << StringPrintf("Map Update GBA Elapsed time: %.3f [minutes]", ba_time_cost / (1e6 * 60)).c_str()
              << std::endl;

    fprintf(fs, "%s\n",
            StringPrintf("Map Update Merge Track Elapsed time: %.3f [minutes]", merge_time_cost / (1e6 * 60)).c_str());
    fflush(fs);
    fprintf(fs, "%s\n",
            StringPrintf("Map Update Filte Point Elapsed time: %.3f [minutes]", filter_time_cost / (1e6 * 60)).c_str());
    fflush(fs);
    ~fprintf(fs, "%s\n",
             StringPrintf("Map Update Register and LBA Elapsed time: %.3f [minutes]", image_update_cost / (1e6 * 60))
                 .c_str());
    fflush(fs);
    fprintf(fs, "%s\n", StringPrintf("Map Update GBA Elapsed time: %.3f [minutes]", ba_time_cost / (1e6 * 60)).c_str());
    fflush(fs);
    std::cout << "Update image number = " << update_image_counter << std::endl;

    std::cout << std::endl;
    timer.PrintMinutes();
    std::cout << std::endl;
    fprintf(fs, "%s\n", StringPrintf("Map Update Elapsed time: %.3f [minutes]", timer.ElapsedMinutes()).c_str());
    fflush(fs);
    timer.Restart();

    mapper->EndReconstruction(false);

    if ( hybrid_options.save_flag){
        // re-save all the data
        feature_data_container->WriteImagesBinaryData(JoinPaths(workspace_path, "/features.bin"));
        feature_data_container->WriteCamerasBinaryData(JoinPaths(workspace_path, "/cameras.bin"));
        feature_data_container->WriteLocalCamerasBinaryData(JoinPaths(workspace_path, "/local_cameras.bin"));
        feature_data_container->WriteSubPanoramaBinaryData(JoinPaths(workspace_path, "/sub_panorama.bin"));
        feature_data_container->WritePieceIndicesBinaryData(JoinPaths(workspace_path, "/piece_indices.bin"));

        if (static_cast<bool>(param.GetArgument("detect_apriltag", 0))) {
            feature_data_container->WriteAprilTagBinaryData(JoinPaths(workspace_path, "/apriltags.bin"));
        }
        scene_graph_container->WriteSceneGraphBinaryData(workspace_path + "/scene_graph.bin");
        scene_graph_container->CorrespondenceGraph()->ExportToGraph(workspace_path + "/scene_graph.png");
        std::cout << "ExportToGraph done!" << std::endl;

        timer.Restart();
        std::string reconstruction_path = JoinPaths(workspace_path, "0");
        if (!boost::filesystem::exists(reconstruction_path)) {
            boost::filesystem::create_directories(reconstruction_path);
        }
        reconstruction->WriteBinary(reconstruction_path);

        bool camera_rig = false;
        const auto& camera_ids = reconstruction->Cameras();
        for (auto camera : camera_ids){
            if (camera.second.NumLocalCameras() > 1){
                camera_rig = true;
            }
        }
        std::cout<<"camera rig: "<<camera_rig<<std::endl;

        if(camera_rig){
            std::string export_path = JoinPaths(workspace_path, "0-export");
            if (!boost::filesystem::exists(export_path)) {
                boost::filesystem::create_directories(export_path);
            }
            Reconstruction rig_reconstruction;
            reconstruction->ConvertRigReconstruction(rig_reconstruction);
            rig_reconstruction.WriteReconstruction(export_path);
        }
    }

    std::cout << std::endl;
    timer.PrintMinutes();
    std::cout << std::endl;
    fprintf(fs, "%s\n", StringPrintf("Map Update Write Elapsed time: %.3f [minutes]", timer.ElapsedMinutes()).c_str());
    fflush(fs);
}

template<typename T_VAL>
struct LinearInterpolator {
public:
    std::map<int64_t, T_VAL> data_;

    void Add(int64_t key, const T_VAL & value) {
        data_[key] = value;
    }

    T_VAL Get(int64_t key) const {
        if (data_.empty()) return T_VAL();

        auto it2 = std::find_if(data_.begin(), data_.end(), [&](
            const std::pair<int64_t, T_VAL> & val
        ) {
            return val.first > key;
        });

        if (it2 == data_.end()) {
            it2--;
            return it2->second;
        }

        auto it1 = it2;
        if (it1 == data_.begin()) {
            return it1->second;
        }
        it1--;

        CHECK_LE(it1->first, key);
        CHECK_LT(key, it2->first);
        double w1 = key - it1->first;
        double w2 = it2->first - key;
        return (w1 * it2->second + w2 * it1->second) / (w1 + w2);
    }
};

typedef LinearInterpolator<Eigen::VectorXd> LinearWeightInterpolator;

std::vector<int> SelectBestMatchOffsets(
    const LinearWeightInterpolator & interp,
    const std::vector<std::pair<image_t, image_t>> & scan_ranges,
    FeatureDataContainer& feature_data_container,
    SceneGraphContainer& scene_graph_container,
    int alignment_range
) {
    PrintHeading1("Select Best Match Offsets");
    auto & correspondence_graph = *scene_graph_container.CorrespondenceGraph();
    const int num_scans = scan_ranges.size();
    std::vector<int> retval(num_scans, 0);
    if (num_scans <= 1) return retval;

    // map from image_id to image_index
    std::vector<int64_t> image_index;
    for (auto & range : scan_ranges) {
        image_index.resize(std::max(image_index.size(), (size_t)range.second + 1), -1);
    }
    for (auto & range : scan_ranges) {
        for (image_t image_id = range.first; image_id <= range.second; image_id++) {
            std::string image_name = feature_data_container.GetImage(image_id).Name();
            int64_t index = ImageNameToIndex(image_name);
            image_index[image_id] = index;
        }
    }

    // get scan index starts
    std::vector<int64_t> scan_index_start(num_scans, 0);
    for (int scan = 0; scan < num_scans; scan++) {
        std::vector<int64_t> scan_image_index;
        for (image_t image_id = scan_ranges[scan].first;
             image_id <= scan_ranges[scan].second;
             image_id++
        ) {
            scan_image_index.emplace_back(image_index[image_id]);
        }

        if (scan_image_index.size() >= 1) {
            scan_index_start[scan] = scan_image_index[0];
        }
        std::cout << "Scan " << scan << " index start: " << scan_index_start[scan] << std::endl;
    }

    // get scan delta T (roughly)
    std::vector<int64_t> scan_delta_t(num_scans, 0);
    for (int scan = 1; scan < num_scans; scan++) {
        std::vector<std::pair<int64_t, int64_t>> delta_t_candidates;
        for (image_t image0 = scan_ranges[0].first; image0 <= scan_ranges[0].second; image0++) {
            if (!scene_graph_container.ExistsImage(image0)) continue;

            const int64_t image_index0 = image_index[image0];
            const int64_t index_offset0 = image_index0 - scan_index_start[0];
            if (image_index0 < 0) continue;

            for (auto image1 : correspondence_graph.ImageNeighbor(image0)) {
                if (image1 >= scan_ranges[scan].first && 
                    image1 <= scan_ranges[scan].second
                ) {
                    const int64_t image_index1 = image_index[image1];
                    const int64_t index_offset1 = image_index1 - scan_index_start[scan];
                    if (image_index1 < 0) continue;

                    auto num = correspondence_graph.NumCorrespondencesBetweenImages(image0, image1);
                    int64_t weight = num / 32;
                    delta_t_candidates.emplace_back(index_offset0 - index_offset1, weight);
                }
            }
        }

        int64_t delta_t = 0;
        if (delta_t_candidates.size() > 128) {
            std::sort(delta_t_candidates.begin(), delta_t_candidates.end(), 
            [](const std::pair<int64_t, int64_t> & a, const std::pair<int64_t, int64_t> & b) {
                return a.first < b.first;
            });

            int64_t delta_t_weights = 0;
            for (size_t i = delta_t_candidates.size() * 0.25; i < delta_t_candidates.size() * 0.75; i++) {
                delta_t += delta_t_candidates[i].first * delta_t_candidates[i].second;
                delta_t_weights += delta_t_candidates[i].second;
            }
            if (delta_t_weights > 0) {
                delta_t /= delta_t_weights;
            }
        }

        scan_delta_t[scan] = delta_t + scan_index_start[0] - scan_index_start[scan];
        std::cout << "Scan " << scan << " Delta T (roughly): " << scan_delta_t[scan] << std::endl;
    }

    double scan0_score_max = 0.0;
    const size_t scan0_count = scan_ranges[0].second - scan_ranges[0].first + 1;
    for (int offset0 = 0; offset0 < scan0_count - alignment_range; offset0++) {
        const image_t image_start0 = scan_ranges[0].first + offset0;
        const int64_t index_start0 = image_index[image_start0];
        const image_t image_end0   = scan_ranges[0].first + offset0 + alignment_range;
        const int64_t index_end0   = image_index[image_end0];
        const int scan_index_interval0 = (index_end0 - index_start0 + 1) / alignment_range;
        if (index_start0 < 0) continue;
        if (index_end0 < 0) continue;

        std::vector<int> scan_offsets(num_scans, offset0);
        double scan0_score = std::numeric_limits<double>::max();
        for (int scan = 1; scan < num_scans; scan++) {
            auto & scan_offset = scan_offsets[scan];

            int64_t scan_offset_diff = std::numeric_limits<int64_t>::max();
            const size_t scan_count = scan_ranges[scan].second - scan_ranges[scan].first + 1;
            for (int offset1 = 0; offset1 + alignment_range < scan_count; offset1++) {
                const image_t image_start1 = scan_ranges[scan].first + offset1;
                const int64_t index_start1 = image_index[image_start1];
                if (index_start1 < 0) continue;

                int64_t diff = std::abs((index_start0 - index_start1) - scan_delta_t[scan]);
                if (diff < scan_offset_diff) {
                    scan_offset = offset1;
                    scan_offset_diff = diff;
                }
            }

            double scan_score = 0.0;
            {
                const image_t image_start1 = scan_ranges[scan].first + scan_offset;
                const int64_t index_start1 = image_index[image_start1];

                for (int i = 0; i < alignment_range; i++) {
                    const image_t image1 = image_start1 + i;
                    const int64_t index1 = image_index[image1];
                    if (!scene_graph_container.ExistsImage(image1)) continue;
                    if (index1 < 0) continue;

                    double image_scores = 0.0;
                    double image_scores_weight = 0.0;
                    const int64_t index_offset1 = image_index[image1] - index_start1;
                    for (auto image0 : correspondence_graph.ImageNeighbor(image1)) {
                        if (image0 >= scan_ranges[0].first && 
                            image0 <= scan_ranges[0].second && 
                            image0 >= scan_ranges[0].first + offset0 && 
                            image0 <= scan_ranges[0].first + offset0 + alignment_range - 1
                        ) {
                            const int64_t index0 = image_index[image0];
                            CHECK_GE(index0, 0);

                            const double diff_x = std::abs(index0 - index1 - scan_delta_t[scan]) / (double)scan_index_interval0;
                            const double diff_weight = 1.0 / (0.5 * diff_x * diff_x + 1.0);

                            auto weight = interp.Get(index0);
                            auto num = correspondence_graph.NumCorrespondencesBetweenImages(image1, image0);
                            image_scores += weight[scan - 1] * num * diff_weight;
                            image_scores_weight += diff_weight;
                        }
                    }

                    if (image_scores_weight > 0) {
                        scan_score += image_scores / image_scores_weight;
                    }
                }
            }
        
            // std::cout << "  " << scan_score << std::endl;
            scan0_score = std::min(scan_score, scan0_score);
        }

        std::cout << "Offset " << VectorToCSV(scan_offsets) << " score " << scan0_score << std::endl;
        if (scan0_score > scan0_score_max) {
            scan0_score_max = scan0_score;
            retval = scan_offsets;
        }
    }

    std::cout << "Selected offsets: ";
    for (auto val : retval) {
        std::cout << val << " ";
    }
    std::cout << std::endl;
    return retval;
}

std::shared_ptr<Reconstruction> RGBDRegistration(Configurator & param, 
    const LinearWeightInterpolator & interp) {
    std::cout << "\nRGBDRegistration Begin ..." << std::endl;
    std::string workspace_path = param.GetArgument("workspace_path", "");
    if (false && ExistsPath(JoinPaths(workspace_path, "0"))) {
    // if (ExistsPath(JoinPaths(workspace_path, "0"))) {
        auto reconstruction = std::make_shared<Reconstruction>();
        reconstruction->ReadReconstruction(JoinPaths(workspace_path, "0"), 1);
        return reconstruction;
    }

    fs = fopen((workspace_path + "/time.txt").c_str(), "w");
    Timer timer;
    timer.Start();

    bool debug_info = static_cast<bool>(param.GetArgument("debug_info", 0));
    int num_cameras = static_cast<int>(param.GetArgument("num_cameras", -1));
    
    int alignment_sparse_interval = static_cast<int>(param.GetArgument("alignment_sparse_interval", 20));
    int alignment_range = static_cast<int>(param.GetArgument("alignment_range", 600));
    int alignment_match_step = static_cast<int>(param.GetArgument("alignment_match_step", 2));
    CHECK_GT(alignment_sparse_interval, 0);
    CHECK_GT(alignment_range, 0);
    CHECK_GT(alignment_match_step, 0);

    std::vector<std::string> image_selection;
    {
        auto sparse_scene_graph_container = std::make_shared<SceneGraphContainer>();
        auto sparse_feature_data_container = std::make_shared<FeatureDataContainer>();

        HybridOptions hybrid_options(debug_info);
        hybrid_options.image_selection = StringPrintf(",,%d", alignment_sparse_interval);
        hybrid_options.child_id = 0;
        if (!hybrid_options.debug_info){
            hybrid_options.save_flag = false;
            hybrid_options.read_flag = false;
        }
        FeatureExtraction(*sparse_feature_data_container.get(), param, hybrid_options);

        hybrid_options.manual_match = true;
        hybrid_options.manual_match_overlap = (int)(GetAverageIndexDiff(*sparse_feature_data_container) * alignment_range) / alignment_sparse_interval + 1;
        hybrid_options.manual_match_overlap_step = alignment_match_step;
        std::cout << "Manual-match overlap: " << hybrid_options.manual_match_overlap << std::endl;

        auto first_images = sparse_feature_data_container->GetImageIds();
        auto first_range = std::minmax_element(first_images.begin(), first_images.end());
        std::vector<std::pair<image_t, image_t>> scan_ranges;
        scan_ranges.push_back({ *first_range.first, *first_range.second });
        for (int idx = 1; idx < num_cameras; idx++){
            hybrid_options.child_id = idx;
            hybrid_options.update_flag = true;
            hybrid_options.match_range = { *first_range.first, *first_range.second };
            FeatureExtraction(*sparse_feature_data_container.get(), param, hybrid_options);

            auto current_images = sparse_feature_data_container->GetImageIds();
            auto max_image = std::max_element(current_images.begin(), current_images.end());
            scan_ranges.push_back({ scan_ranges[idx - 1].second + 1, *max_image });

            FeatureMatching(*sparse_feature_data_container.get(), *sparse_scene_graph_container.get(), 
                            param, hybrid_options);
        }

        auto best_offsets = SelectBestMatchOffsets(
            interp, 
            scan_ranges, 
            *sparse_feature_data_container, 
            *sparse_scene_graph_container, 
            std::max(1, alignment_range / alignment_sparse_interval));
        
        std::cout << "Image selection: " << std::endl;
        for (auto & best_offset : best_offsets) {
            image_selection.emplace_back(StringPrintf("%d,%d,1", 
                best_offset * alignment_sparse_interval, 
                best_offset * alignment_sparse_interval + alignment_range));
            std::cout << image_selection.back() << std::endl;
        }
    }

    auto reconstruction_manager = std::make_shared<ReconstructionManager>();
    auto scene_graph_container = std::make_shared<SceneGraphContainer>();
    auto feature_data_container = std::make_shared<FeatureDataContainer>();

    {
        HybridOptions hybrid_options;
        hybrid_options.child_id = 0;
        hybrid_options.image_selection = image_selection[0];
        hybrid_options.debug_info = false;
        hybrid_options.save_flag = false;
        hybrid_options.read_flag = false;
        FeatureExtraction(*feature_data_container.get(), param, hybrid_options);

        typedef FeatureMatchingOptions::RetrieveType RetrieveType;
        RetrieveType retrieve_type = (RetrieveType)param.GetArgument("retrieve_type", 0);
        if (retrieve_type == RetrieveType::VLAD) {
            GlobalFeatureExtraction(*feature_data_container.get(), param, hybrid_options);
        }

        FeatureMatching(*feature_data_container.get(), *scene_graph_container.get(), param, hybrid_options);
        IncrementalSFM(scene_graph_container, reconstruction_manager, param, hybrid_options);

        auto first_images = feature_data_container->GetImageIds();
        auto first_range = std::minmax_element(first_images.begin(), first_images.end());
        for (int idx = 1; idx < num_cameras; idx++){
            hybrid_options.child_id = idx;
            hybrid_options.image_selection = image_selection[idx];
            hybrid_options.update_flag = true;
            hybrid_options.match_range = { *first_range.first, *first_range.second };
            FeatureExtraction(*feature_data_container.get(), param, hybrid_options);

            typedef FeatureMatchingOptions::RetrieveType RetrieveType;
            RetrieveType retrieve_type = (RetrieveType)param.GetArgument("retrieve_type", 0);
            if (retrieve_type == RetrieveType::VLAD) {
                // vlad is not good for different cameras
                // GlobalFeatureExtraction(*feature_data_container.get(), param, hybrid_options);
            }

            FeatureMatching(*feature_data_container.get(), *scene_graph_container.get(), 
                            param, hybrid_options);

            if (idx == num_cameras-1){
                hybrid_options.child_id = -1;
                hybrid_options.save_flag = true;
            }
            MapUpdate(feature_data_container, scene_graph_container, reconstruction_manager, 
                        param, hybrid_options);
        }
    }

    std::cout << StringPrintf("Reconstruct in %.3fs", timer.ElapsedSeconds()) << std::endl;

    PrintReconSummary(workspace_path + "/statistic.txt", scene_graph_container->NumImages(), reconstruction_manager);

    fclose(fs);

    return reconstruction_manager->Get(0);
}

void CalibDebug(
    const std::string& debug_rig, 
    const std::string& debug_rgbd,
    const Eigen::Matrix3d& extra_R,
    const Eigen::Vector3d& extra_T,
    Camera& rig_camera,
    int local_cam_id
) {
    RGBDData data;
    ExtractRGBDData(debug_rgbd, data);

    const int width = data.color.Width();
    const int height = data.color.Height();
    MatXf warped_depthmap(width, height, 1);
    UniversalWarpDepthMap(warped_depthmap, data.depth, data.color_camera, data.depth_camera, data.depth_RT.cast<float>());

    std::cout << debug_rig << std::endl;
    std::cout << debug_rgbd << std::endl;
    std::cout << "=>" << std::endl;
    cv::Mat rig_render = cv::Mat::zeros(rig_camera.Height(), rig_camera.Width(), CV_8UC3);
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float depth = warped_depthmap.Get(y, x);
            if (depth > 0) {
                Eigen::Vector3d pt_rgbd = data.color_camera.ImageToWorld(Eigen::Vector2d(x, y)).homogeneous();
                                pt_rgbd *= depth;
                
                Eigen::Vector3d pt_rig = extra_R * pt_rgbd + extra_T;
                // Eigen::Vector3d pt_rig = extra_R.inverse() * (pt_rgbd - extra_T);
                Eigen::Vector2d uv = rig_camera.WorldToLocalImage(local_cam_id, pt_rig.hnormalized());

                const int u = uv.x(), v = uv.y();
                if (u >= 0 && v >= 0 && u < rig_render.cols && v < rig_render.rows) {
                    auto rgb = data.color.GetPixel(x, y);
                    rig_render.at<cv::Vec3b>(v, u) = cv::Vec3b(rgb.b, rgb.g, rgb.r);
                }
            }
        }
    }

    cv::Mat rig_color = cv::imread(debug_rig);
    for (int y = 0; y < rig_color.rows; y++) {
        for (int x = 0; x < rig_color.cols; x++) {
            auto cr = rig_render.at<cv::Vec3b>(y, x);
            if (cr != cv::Vec3b::zeros()) {
                auto cc = rig_color.at<cv::Vec3b>(y, x);
                int b = ((int)cc[0] + (int)cr[0]) / 2;
                int g = ((int)cc[1] + (int)cr[1]) / 2;
                int r = ((int)cc[2] + (int)cr[2]) / 2;
                rig_color.at<cv::Vec3b>(y, x) = cv::Vec3b(b, g, r);
            }
        }
    }

    std::string output_color = "calib_cam" + std::to_string(local_cam_id) + ".color.jpg";
    std::string output_render = "calib_cam" + std::to_string(local_cam_id) + ".render.jpg";
    cv::imwrite(output_color, rig_color);
    cv::imwrite(output_render, rig_render);
    std::cout << output_color << std::endl;
    std::cout << output_render << std::endl;
}

void CalibDebug(
    const std::string& calib_images,
    Camera& rig_camera
) {
    std::cout << "Rig camera id is " << rig_camera.CameraId() << std::endl;
    auto rig_images = GetRecursiveFileList(JoinPaths(calib_images, target_subpath));
    CHECK(!rig_images.empty());

    for (int i_cam = 0; i_cam < rgbd_infos.size(); i_cam++) {
        std::string camth = rgbd_infos[i_cam].calib_cam;
        int local_cam_id = camth.empty() ? -1 : std::atoi(camth.substr(3, camth.size()).c_str());

        std::cout << "Local camera id is " << local_cam_id << std::endl;
        auto rig_cam_images = rig_images;
        for (auto iter = rig_cam_images.begin(); iter != rig_cam_images.end(); ) {
            std::string image_file = *iter;
            if (!strstr(image_file.c_str(), camth.c_str())) {
                iter = rig_cam_images.erase(iter);
            } else {
                iter++;
            }
        }
        CHECK(!rig_cam_images.empty());

        std::cout << "Sub-path is " << rgbd_infos[i_cam].sub_path << std::endl;
        auto rgbd_images = GetRecursiveFileList(JoinPaths(calib_images, rgbd_infos[i_cam].sub_path));
        CHECK(!rgbd_images.empty());

        std::string debug_rig = rig_cam_images[rig_cam_images.size() / 2];
        std::string debug_rgbd = rgbd_images[rgbd_images.size() / 2];
        CalibDebug(debug_rig, debug_rgbd, rgbd_infos[i_cam].extra_R, rgbd_infos[i_cam].extra_T, rig_camera, local_cam_id);
    }
}

bool OptimizeDeltaTime(
    const std::string & image_path,
    const Reconstruction & align_reconstruction,
    std::unordered_map<image_t, std::pair<image_t, int>> & image_alignment,
    std::map<int64_t, image_t>& map_index_to_image_id,
    std::vector<std::map<int64_t, std::string>>& v_map_rgbd_index_to_name,
    int64_t& target_ms_per_frame
) {
    std::cout << "Align By Reprojection Error" << std::endl;

    int success_count = rgbd_infos.size();
    // std::map<int64_t, image_t> map_index_to_image_id;
    for (auto & item : align_reconstruction.GetImageNames()) {
        if (IsInsideSubpath(item.first, target_subpath)) {
            int64_t index = ImageNameToIndex(item.first);
            CHECK(map_index_to_image_id.count(index) == 0) << "Duplicated image numeric name " << item.first;
            
            map_index_to_image_id[index] = item.second;
        }
    }

    target_ms_per_frame = 1;
    std::map<int64_t, int> ms_per_frame_count;
    for (auto iter1 = map_index_to_image_id.begin(); iter1 != map_index_to_image_id.end(); iter1++) {
        auto iter2 = iter1;
        iter2++;
        if (iter2 != map_index_to_image_id.end()) {
            int64_t diff = iter2->first - iter1->first;
            if (ms_per_frame_count.count(diff)) {
                ms_per_frame_count[diff] += 1;
            } else {
                ms_per_frame_count[diff] = 1;
            }
        }
    }
    for (auto item : ms_per_frame_count) {
        if (item.second < 5) continue;

        target_ms_per_frame = item.first;
        break;
    }

    for(int i_cam = 0; i_cam < rgbd_infos.size(); i_cam++)
    {
        std::string sub_path = rgbd_infos[i_cam].sub_path;
        std::string camth = rgbd_infos[i_cam].calib_cam;
        int local_cam_id = camth.empty() ? -1 : std::atoi(camth.substr(3, camth.size()).c_str());
        std::cout << "local_cam_id:" << local_cam_id<<std::endl;
        Eigen::Matrix3d extra_R = rgbd_infos[i_cam].extra_R;
        Eigen::Vector3d extra_T = rgbd_infos[i_cam].extra_T;

        Camera rgbd_camera;
        for (auto & item : align_reconstruction.Images()) {
            if (IsInsideSubpath(item.second.Name(), sub_path)) {
                rgbd_camera = align_reconstruction.Camera(item.second.CameraId());
                break;
            }
        }
        CHECK(rgbd_camera.ModelId() != kInvalidCameraModelId);

        double dis_c = (-extra_R.inverse() * extra_T).norm();
        std::cout << "dis_c:" << dis_c << std::endl;

        std::string rgbd_path = JoinPaths(image_path, sub_path);
        std::vector<std::string> rgbd_files = GetRecursiveFileList(rgbd_path);
        std::map<int64_t, std::string> map_rgbd_index_to_name;  // must be ordered
        #pragma omp parallel for schedule(dynamic, 1)
        for (int idx = 0; idx < rgbd_files.size(); idx++) {
            const auto & str = rgbd_files[idx];
            int64_t index = ImageNameToIndex(str);
            std::string name = GetRelativePath(image_path, str);
            #pragma omp critical
            {
                map_rgbd_index_to_name[index] = name;
            }
        }
        v_map_rgbd_index_to_name.push_back(map_rgbd_index_to_name);

        const auto register_image_ids = align_reconstruction.RegisterImageIds();

        int64_t rgbd_timestamp_start = map_rgbd_index_to_name.begin()->first;
        int64_t target_timestamp_start = map_index_to_image_id.begin()->first;
        int64_t timstamp_diff = target_timestamp_start - rgbd_timestamp_start;
        bool timestamp = false;
        if (rgbd_infos[i_cam].timestamp < 0) {
            if (rgbd_timestamp_start > 10000000 || target_timestamp_start > 10000000) {
                timestamp = true;
            }
        } else if (rgbd_infos[i_cam].timestamp > 0) {
            timestamp = true;
        }
        if (timestamp) {
            std::cout << "Start timestamp(Target): " << target_timestamp_start << std::endl;
            std::cout << "Start timestamp(RGBD): " << rgbd_timestamp_start << std::endl;
            std::cout << "Estimated " << target_ms_per_frame << " ms per frame" << std::endl;
        }

        int64_t offset = 0;
        double pixel_error = std::numeric_limits<double>::max();
        double overall_error = std::numeric_limits<double>::max();
        Eigen::Matrix3d new_extra_R = extra_R;
        Eigen::Vector3d new_extra_T = extra_T;
        if (!rgbd_infos[i_cam].has_force_offset)
        {
            struct OffsetScore {
                double score;
                double weight;
                Eigen::Vector2d image_point;
                Eigen::Vector3d target_point;

                OffsetScore(double s, double w, Eigen::Vector2d ip, Eigen::Vector3d tp) {
                    score = s;
                    weight = w;
                    image_point = ip;
                    target_point = tp;
                }

                bool operator<(const OffsetScore & a) {
                    return this->score < a.score;
                }
            };

            std::cout << "Gathering candidates... " << std::endl;
            std::map<int64_t, std::vector<OffsetScore>> map_offset_score;
            #pragma omp parallel for schedule(dynamic, 1)
            for (int i = 0; i < register_image_ids.size(); i++) {
                const auto & rgbd_image = align_reconstruction.Image(register_image_ids[i]);
                const auto & rgbd_camera = align_reconstruction.Camera(rgbd_image.CameraId());
                const auto rgbd_R = QuaternionToRotationMatrix(rgbd_image.Qvec());
                const auto rgbd_T = rgbd_image.Tvec();
                const auto rgbd_C = -rgbd_R.transpose() * rgbd_T;
                const int64_t rgbd_index = ImageNameToIndex(rgbd_image.Name());
                if (!IsInsideSubpath(rgbd_image.Name(), sub_path)) continue;

                for (auto & point2d : rgbd_image.Points2D()) 
                {
                    if (!point2d.HasMapPoint()) continue;
                    auto map_pt = align_reconstruction.MapPoint(point2d.MapPointId());

                    for (const auto & track : map_pt.Track().Elements()) 
                    {
                        const auto & target_image = align_reconstruction.Image(track.image_id);
                        if (!IsInsideSubpath(target_image.Name(), target_subpath)) continue;

                        auto target_R = QuaternionToRotationMatrix(target_image.Qvec());
                        auto target_T = target_image.Tvec();
                        if (local_cam_id >= 0) {
                            Eigen::Vector4d local_qvec;
                            Eigen::Vector3d local_tvec;
                            const auto target_camera = align_reconstruction.Camera(target_image.CameraId());
                            target_camera.GetLocalCameraExtrinsic(local_cam_id, local_qvec, local_tvec);
                            target_R = QuaternionToRotationMatrix(local_qvec) * target_R;
                            target_T = QuaternionToRotationMatrix(local_qvec) * target_T + local_tvec;
                        }

                        const auto target_C = -target_R.transpose() * target_T;
                        const auto dis = (rgbd_C - target_C).norm();
                        if (dis < dis_c * 0.5 - 0.2 || dis > dis_c * 2.0 + 0.2) continue;

                        Eigen::Vector3d target_cam_coord  = target_R * map_pt.XYZ() + target_T;
                        Eigen::Vector3d estimated_cam_coord = extra_R.inverse() * (target_cam_coord - extra_T);
                        auto estimated_img_coord = rgbd_camera.WorldToImage(estimated_cam_coord.hnormalized());
                        double error2 = (estimated_img_coord - point2d.XY()).squaredNorm();
                        if (error2 < 1000.0) {
                            const double weight = 1.0;
                            const int64_t target_index = ImageNameToIndex(target_image.Name());
                            int64_t offset = target_index - rgbd_index;
                            #pragma omp critical
                            {
                                map_offset_score[offset].emplace_back(
                                    std::sqrt(error2),
                                    weight,
                                    point2d.XY(),
                                    target_cam_coord
                                );
                            }
                        }
                    }
                }
            }

            std::cout << "Original candidates: " << map_offset_score.size() << std::endl;
            // for (auto & item : map_offset_score) {
            //     std::cout << item.first << ": " << item.second.size() << std::endl;
            // }

            if (timestamp) {
                // merge neighboring timestamps
                int64_t diff_thresh = (target_ms_per_frame + 1) / 2;
                if (diff_thresh > 0) {
                    std::cout << "Merging neighboring timestamps using thresh " << diff_thresh << " ms" << std::endl;

                    std::map<int64_t, std::vector<OffsetScore>> map_offset_score2;
                    for (auto iter1 = map_offset_score.begin(); iter1 != map_offset_score.end(); iter1++) {
                        auto item = iter1->second;

                        int64_t offset = iter1->first * item.size();
                        int64_t count = item.size();
                        for (auto iter2 = iter1; ;iter2++) {
                            if (iter2 == map_offset_score.end()) {
                                break;
                            }
                            if (iter2 != iter1) {
                                if (iter2->first - iter1->first < diff_thresh) {
                                    offset += iter2->first * iter2->second.size();
                                    count += iter2->second.size();
                                    item.insert(item.end(), iter2->second.begin(), iter2->second.end());
                                } else {
                                    break;
                                }
                            }
                        }
                        for (auto iter2 = iter1; ;iter2--) {
                            if (iter2 != iter1) {
                                if (iter1->first - iter2->first < diff_thresh) {
                                    offset += iter2->first * iter2->second.size();
                                    count += iter2->second.size();
                                    item.insert(item.end(), iter2->second.begin(), iter2->second.end());
                                } else {
                                    break;
                                }
                            }
                            if (iter2 == map_offset_score.begin()) {
                                break;
                            }
                        }

                        offset /= count;
                        map_offset_score2[offset] = item;
                    }
                    std::swap(map_offset_score, map_offset_score2);
                }
            }

            // Remove offsets with low counts
            size_t max_count = 0;
            for (const auto & item : map_offset_score) {
                max_count = std::max(item.second.size(), max_count);
            }
            std::set<int64_t> offsets_to_del;
            for (const auto & item : map_offset_score) {
                if (item.second.size() < max_count * 0.5 || item.second.size() < 200) {
                    offsets_to_del.emplace(item.first);
                }
            }
            for (auto offset : offsets_to_del) {
                map_offset_score.erase(offset);
            }

            // Pick offset with smallest error
            if (map_offset_score.size() > 0) {
                std::cout << "Candidates: " << std::endl;
                std::vector<int64_t> offsets;
                for (auto & item : map_offset_score) {
                    std::sort(item.second.begin(), item.second.end());

                    Camera & camera = rgbd_camera;
                    Eigen::Vector4d inv_extra_qvec = RotationMatrixToQuaternion(extra_R.inverse());
                    Eigen::Vector3d inv_extra_tvec = -extra_R.inverse() * extra_T;
                    double *qvec_data = inv_extra_qvec.data();
                    double *tvec_data = inv_extra_tvec.data();
                    double *camera_params_data = camera.ParamsData();
                    ceres::LossFunction * loss_function = new ceres::HuberLoss(1.0);
                    ceres::LocalParameterization *quaternion_parameterization = new ceres::QuaternionParameterization;
                    ceres::Problem problem;

                    double total_score = 0.0;
                    double total_weight = 0.0;
                    size_t count = item.second.size() * 0.8;
                    for (auto iter = item.second.begin(); iter != item.second.begin() + count; ++iter) {
                        total_score += iter->score * iter->weight;
                        total_weight += iter->weight;

                        ceres::CostFunction * cost_function = nullptr;
                        switch (camera.ModelId()) {
                        #define CAMERA_MODEL_CASE(CameraModel)\
                            case CameraModel::kModelId:\
                                cost_function = BundleAdjustmentConstantMapPointCostFunction<CameraModel>::Create(iter->image_point, iter->target_point, iter->weight); \
                                break;
                            CAMERA_MODEL_SWITCH_CASES
                        #undef CAMERA_MODEL_CASE
                        }
                        problem.AddResidualBlock(cost_function, loss_function, qvec_data, tvec_data, camera_params_data);
                    }
                    total_score /= total_weight;
                    problem.SetParameterBlockConstant(camera_params_data);
                    {
                        Eigen::Vector3d inv_extra_pvec = extra_T;
                        ceres::CostFunction * cost_function = PriorAbsolutePoseCostFunction::Create(inv_extra_qvec, inv_extra_pvec, 2e4, 2e4, 2e4, 2e4);
                        problem.AddResidualBlock(cost_function, nullptr, qvec_data, tvec_data);
                    }
#if CERES_VERSION_MAJOR >= 2 && CERES_VERSION_MINOR >= 1
                    problem.SetManifold(qvec_data, quaternion_parameterization);
#else
                    problem.SetParameterization(qvec_data, quaternion_parameterization);
#endif

                    ceres::Solver::Options options;
                    ceres::Solver::Summary summary;
                    options.linear_solver_type = ceres::DENSE_QR;
                    options.minimizer_progress_to_stdout = false;
                    ceres::Solve(options, &problem, &summary);

                    Eigen::Matrix3d refine_extra_R = QuaternionToRotationMatrix(inv_extra_qvec).inverse();
                    Eigen::Vector3d refine_extra_T = -refine_extra_R * inv_extra_tvec;
                    double refined_score = 0.0;
                    double refined_weight = 0.0;
                    for (auto iter = item.second.begin(); iter != item.second.begin() + count; ++iter) {
                        Eigen::Vector3d estimated_cam_coord = refine_extra_R.inverse() * (iter->target_point - refine_extra_T);
                        auto estimated_img_coord = rgbd_camera.WorldToImage(estimated_cam_coord.hnormalized());
                        refined_score += (estimated_img_coord - iter->image_point).norm() * iter->weight;
                        refined_weight += iter->weight;
                    }
                    refined_score /= refined_weight;

                    double final_score = refined_score / std::pow(count, 0.5);
                    std::cout << item.first << " " << total_score << " " << refined_score << " " << final_score << " " << count << std::endl;

                    if (final_score < overall_error * 0.99999) {
                        offsets.clear();
                        offsets.emplace_back(item.first);
                        pixel_error = refined_score;
                        overall_error = final_score;
                        new_extra_R = refine_extra_R;
                        new_extra_T = refine_extra_T;
                    } else if (final_score <= overall_error * 1.00001) {
                        offsets.emplace_back(item.first);
                    }
                }
                std::sort(offsets.begin(), offsets.end());
                offset = (offsets[0] + offsets[offsets.size() - 1]) / 2;

                std::cout << "Offset(s):";
                for (auto offset : offsets) {
                    std::cout << " " << offset;
                }
                std::cout << std::endl;
            } else {
                std::cout<<"ERROR: i_cam:" << i_cam << " failed!"<<std::endl;
                success_count--;
                continue;
            }
        }
        else
        {
            //test
            offset = rgbd_infos[i_cam].force_offset;
            // end test
        }

        rgbd_infos[i_cam].extra_R = new_extra_R;
        rgbd_infos[i_cam].extra_T = new_extra_T;
        rgbd_infos[i_cam].force_offset = offset;

        std::cout << "offset: " << offset << (timestamp ? " ms" : "") <<std::endl;
        std::cout << "pixel error: " << pixel_error << std::endl;
        std::cout << "overall error: " << overall_error << std::endl;
        std::cout << extra_R << std::endl;
        std::cout << extra_T << std::endl;
        std::cout << "=>" << std::endl;
        std::cout << new_extra_R << std::endl;
        std::cout << new_extra_T << std::endl << std::endl;

    }

    return success_count > 0;
}

bool AddRgbdReconstruction(
    const std::string & image_path, Configurator & param,
    const Reconstruction & align_reconstruction, 
    std::shared_ptr<FeatureDataContainer> feature_data_container,
    std::shared_ptr<SceneGraphContainer> scene_graph_container,
    std::shared_ptr<ReconstructionManager> reconstruction_manager,
    // Reconstruction & target_reconstruction, 
    std::unordered_map<image_t, std::pair<image_t, int>> & image_alignment,
    std::map<int64_t, image_t>& map_index_to_image_id,
    std::vector<std::map<int64_t, std::string>>& v_map_rgbd_index_to_name,
    const int64_t& target_ms_per_frame) {
    std::cout << "Add Rgbd to Reconstruction" << std::endl;
    std::cout << "rgbd camera size: " << rgbd_infos.size() << "\n" << std::endl;

    std::string workspace_path = param.GetArgument("workspace_path", "");
    fs = fopen((workspace_path + "/time.txt").c_str(), "w");

    int min_track_length = static_cast<int>(param.GetArgument("min_track_length", 3));

    auto target_reconstruction = reconstruction_manager->Get(0);

    OptionParser option_parser;
    IndependentMapperOptions mapper_options;
    option_parser.GetMapperOptions(mapper_options,param);
    const auto tri_options = mapper_options.Triangulation();

    std::vector<image_t> new_image_ids;
    for(int i_cam = 0; i_cam < rgbd_infos.size(); i_cam++)
    {
        std::string camth = rgbd_infos[i_cam].calib_cam;
        int local_cam_id = camth.empty() ? -1 : std::atoi(camth.substr(3, camth.size()).c_str());
        std::cout << "i_cam: "<< i_cam<<" local_cam_id:" << local_cam_id<<std::endl;

        Eigen::Matrix3d extra_R = rgbd_infos[i_cam].extra_R;
        Eigen::Vector3d extra_T = rgbd_infos[i_cam].extra_T;
        int64_t offset = rgbd_infos[i_cam].force_offset;
        // std::cout << "extra_R: " << std::endl;
        // std::cout << extra_R << std::endl;
        // std::cout << "extra_T: " << std::endl;
        // std::cout << extra_T << std::endl;

        std::string sub_path = rgbd_infos[i_cam].sub_path;
        Camera rgbd_camera;
        for (auto & item : align_reconstruction.Images()) {
            if (IsInsideSubpath(item.second.Name(), sub_path)) {
                rgbd_camera = align_reconstruction.Camera(item.second.CameraId());
                break;
            }
        }
        CHECK(rgbd_camera.ModelId() != kInvalidCameraModelId);

        auto& map_rgbd_index_to_name = v_map_rgbd_index_to_name.at(i_cam);
        int64_t rgbd_timestamp_start = map_rgbd_index_to_name.begin()->first;
        int64_t target_timestamp_start = map_index_to_image_id.begin()->first;
        int64_t timstamp_diff = target_timestamp_start - rgbd_timestamp_start;
        bool timestamp = false;
        if (rgbd_infos[i_cam].timestamp < 0) {
            if (rgbd_timestamp_start > 10000000 || target_timestamp_start > 10000000) {
                timestamp = true;
            }
        } else if (rgbd_infos[i_cam].timestamp > 0) {
            timestamp = true;
        }
        if (timestamp) {
            std::cout << "Start timestamp(Target): " << target_timestamp_start << std::endl;
            std::cout << "Start timestamp(RGBD): " << rgbd_timestamp_start << std::endl;
            // std::cout << "Estimated " << target_ms_per_frame << " ms per frame" << std::endl;
        }

        // first delete old rgbd images and
        camera_t sub_path_camera = 0;
        std::vector<image_t> images_to_delete;
        for (auto & item : target_reconstruction->Images()) {
            if (IsInsideSubpath(item.second.Name(), sub_path)) {
                sub_path_camera = item.second.CameraId();
                images_to_delete.emplace_back(item.first);
            }
        }
        for (auto & image : images_to_delete) {
            target_reconstruction->DeleteImage(image);
        }

        // find max image_id
        image_t max_image_id = 0;
        std::vector<image_t> target_images;
        for (auto & item : target_reconstruction->Images()) {
            max_image_id = std::max(max_image_id, item.first);
            target_images.emplace_back(item.first);
        }
        // std::cout << "Target Reconstruction Image size: " << target_reconstruction.NumImages() << std::endl;
        // std::cout << "Target Images size: " << target_images.size() << std::endl;

        // try find existing camera in target_reconstruction
        if (sub_path_camera == 0) {
            camera_t max_camera_id = 0;
            for (auto & item : target_reconstruction->Cameras()) {
                max_camera_id = std::max(max_camera_id, item.first);
                if (item.second.ModelId() == rgbd_camera.ModelId() && 
                    item.second.Width() == rgbd_camera.Width() &&
                    item.second.Height() == rgbd_camera.Height() &&
                    item.second.ParamsToString() == rgbd_camera.ParamsToString()
                ) {
                    sub_path_camera = item.first;
                }
            }
            CHECK(max_camera_id >= 0);

            // add camera to target_reconstruction
            if (sub_path_camera == 0) {
                sub_path_camera = ++max_camera_id;
                rgbd_camera.SetCameraId(sub_path_camera);
                // target_reconstruction->AddCamera(rgbd_camera);
            }
        }

        std::cout << "Next image id: " << max_image_id + 1 << std::endl;
        std::cout << "RGBD camera id: " << sub_path_camera << std::endl;

        size_t num_aligned_images = 0;
        std::vector<std::pair<std::string, image_t>> rgbd_2_target;
        for (auto & target_image_id : target_images){
            const auto & target_image = target_reconstruction->Image(target_image_id);
            if (!IsInsideSubpath(target_image.Name(), target_subpath)){
                continue;
            }

            std::string rgbd_name;
            if (!timestamp){
                int64_t index = ImageNameToIndex(target_image.Name());
                int64_t rgbd_index = index - offset;
                if (map_rgbd_index_to_name.count(rgbd_index) == 0) continue;
                rgbd_name = map_rgbd_index_to_name[rgbd_index];
            } else {
                int64_t timestamp = ImageNameToIndex(target_image.Name());
                int64_t rgbd_timestamp = timestamp - offset;
                int64_t diff_thresh = (target_ms_per_frame + 1) / 2;
                for (int64_t diff = 0; diff < diff_thresh; diff += 1) {
                    if (map_rgbd_index_to_name.count(rgbd_timestamp + diff)) {
                        rgbd_name = map_rgbd_index_to_name[rgbd_timestamp + diff];
                        break;
                    } else if (map_rgbd_index_to_name.count(rgbd_timestamp - diff)) {
                        rgbd_name = map_rgbd_index_to_name[rgbd_timestamp - diff];
                        break;
                    }
                }
            }
            if (rgbd_name.empty()) {
                int64_t timestamp = ImageNameToIndex(target_image.Name());
                int64_t rgbd_timestamp = timestamp - offset;
                // std::cout << "Failed to match " << target_image.Name() << " <= " << rgbd_timestamp << std::endl;
                continue;
            }
            rgbd_2_target.push_back(std::pair<std::string, image_t> (rgbd_name, target_image_id));
            num_aligned_images++;
        }

        // update feature_data
        HybridOptions hybrid_options;
        hybrid_options.child_id = i_cam + 1;
        hybrid_options.update_flag = true;
        hybrid_options.debug_info = false;
        hybrid_options.save_flag = false;
        hybrid_options.read_flag = false;
        auto& image_list = hybrid_options.image_list;
        image_list.clear();

        size_t split_index = 0;
        if(rgbd_2_target.size()>=1){
            auto &r2t = rgbd_2_target[0];
            split_index = r2t.first.find_first_of("/");
            split_index++;
            std::cout << "split_index: " << split_index << " str: " << r2t.first[split_index] << std::endl;
        }

        for (const auto & r2t : rgbd_2_target){
            std::string name = r2t.first.substr(split_index);

            image_list.push_back(name);
            // std::cout << name << std::endl;
        }

        std::cout << "feature_data_container image, camera size: " << feature_data_container->GetImageIds().size() << ", " << feature_data_container->NumCamera() << std::endl;
        std::cout << "scene_graph_container image, camera size: " << scene_graph_container->NumImages() << ", " << scene_graph_container->NumCameras() << std::endl;
        if (target_reconstruction->ExistsCamera(sub_path_camera)){
            std::cout << "target_reconstruction image, camera size: " << target_reconstruction->NumImages() << ", " << target_reconstruction->NumCameras() 
                            << "(" << target_reconstruction->Camera(sub_path_camera).ParamsToString() << ")" << std::endl;
        } else {
            std::cout << "target_reconstruction image, camera size: " << target_reconstruction->NumImages() << ", " << target_reconstruction->NumCameras()  << std::endl;
        }

        FeatureExtraction(*feature_data_container.get(), param, hybrid_options);

        FeatureMatching(*feature_data_container.get(), *scene_graph_container.get(), 
                        param, hybrid_options);

        // update reconstruction

        std::cout << "feature_data_container image, camera size: " << feature_data_container->GetImageIds().size() << ", " << feature_data_container->NumCamera() << std::endl;
        if (target_reconstruction->ExistsCamera(sub_path_camera)){
             std::cout << "scene_graph_container image, camera size: " << scene_graph_container->NumImages() << ", " << scene_graph_container->NumCameras() 
                        << "(" << scene_graph_container->Camera(sub_path_camera).ParamsToString() << ")" << std::endl;
            std::cout << "target_reconstruction image, camera size: " << target_reconstruction->NumImages() << ", " << target_reconstruction->NumCameras() 
                            << "(" << target_reconstruction->Camera(sub_path_camera).ParamsToString() << ")" << std::endl;
        } else {
            std::cout << "scene_graph_container image, camera size: " << scene_graph_container->NumImages() << ", " << scene_graph_container->NumCameras() << std::endl;
            std::cout << "target_reconstruction image, camera size: " << target_reconstruction->NumImages() << ", " << target_reconstruction->NumCameras()  << std::endl;
        }


        PrintHeading1("Triangulating image");

        std::shared_ptr<IncrementalMapper> mapper = std::make_shared<IncrementalMapper>(scene_graph_container);
        mapper->BeginReconstruction(target_reconstruction);
        std::cout << "Start ..." << std::endl;

        std::cout << "feature_data_container image, camera size: " << feature_data_container->GetImageIds().size() << ", " << feature_data_container->NumCamera() << std::endl;
         if (target_reconstruction->ExistsCamera(sub_path_camera)){
             std::cout << "scene_graph_container image, camera size: " << scene_graph_container->NumImages() << ", " << scene_graph_container->NumCameras() 
                        << "(" << scene_graph_container->Camera(sub_path_camera).ParamsToString() << ")" << std::endl;
            std::cout << "target_reconstruction image, camera size: " << target_reconstruction->NumImages() << ", " << target_reconstruction->NumCameras() 
                            << "(" << target_reconstruction->Camera(sub_path_camera).ParamsToString() << ")" << std::endl;
        } else {
            std::cout << "scene_graph_container image, camera size: " << scene_graph_container->NumImages() << ", " << scene_graph_container->NumCameras() << std::endl;
            std::cout << "target_reconstruction image, camera size: " << target_reconstruction->NumImages() << ", " << target_reconstruction->NumCameras()  << std::endl;
        }

        int64_t triangulated_image_count = 0;
        for (const auto & r2t : rgbd_2_target){
            const std::string rgbd_name = r2t.first;
            const auto target_image_id = r2t.second;

            const auto & target_image = target_reconstruction->Image(target_image_id);
            auto target_qvec = target_image.Qvec();
            auto target_tvec = target_image.Tvec();
            Eigen::Quaterniond target_q(target_qvec(0),target_qvec(1),target_qvec(2),target_qvec(3));
            target_q.normalize();
            Eigen::Matrix3d target_R = target_q.toRotationMatrix();
            if (local_cam_id >= 0)
            {
                Eigen::Vector4d local_qvec;
                Eigen::Vector3d local_tvec;
                const auto & target_camera = target_reconstruction->Camera(target_image.CameraId());
                target_camera.GetLocalCameraExtrinsic(local_cam_id, local_qvec, local_tvec);
                target_R = QuaternionToRotationMatrix(local_qvec) * target_R;
                target_tvec = QuaternionToRotationMatrix(local_qvec) * target_tvec + local_tvec;
            }

            Eigen::Matrix3d RR = extra_R.inverse() * target_R;
            Eigen::Vector3d tt = extra_R.inverse() * (target_tvec - extra_T);

            image_t rgbd_id = feature_data_container->GetImageId(rgbd_name);
            if (!scene_graph_container->ExistsImage(rgbd_id)){
                continue;
            }
            if (!target_reconstruction->ExistsImage(rgbd_id)){
                sensemap::Image& scene_image = scene_graph_container->Image(rgbd_id);
                target_reconstruction->AddImage(scene_image);
            } 
            Image &image = target_reconstruction->Image(rgbd_id);
            image.SetQvec(RotationMatrixToQuaternion(RR));
            image.NormalizeQvec();
            image.SetTvec(tt);
            target_reconstruction->RegisterImage(image.ImageId());

            if (!target_reconstruction->ExistsCamera(image.CameraId()) ){
                 target_reconstruction->AddCamera(rgbd_camera);
            } else {
                rgbd_camera.SetCameraId(image.CameraId());
                target_reconstruction->SetCamera(image.CameraId(), rgbd_camera);
            }

            new_image_ids.push_back(rgbd_id);
            image_alignment[rgbd_id] = std::make_pair(target_image_id, local_cam_id);
            // std::cout << "=> info: " << image.ImageId() << "-" << image.Name() << " / " 
            //     << scene_image.ImageId() << "-" << scene_image.Name() << std::endl;

            PrintHeading2(StringPrintf("Triangulating image #%d - %s (%d / %d)", 
                rgbd_id, image.Name().c_str(), triangulated_image_count++, rgbd_2_target.size()));
            const size_t num_existing_points3D = image.NumMapPoints();
            std::cout << "  => Image sees " << num_existing_points3D << " / " << image.NumObservations() << " points"
                    << std::endl;
            if(image.NumObservations()<=0){
                std::cout<<"image.NumObservations(): "<<image.NumObservations()<<", no observation, skip this frame"<<std::endl;
                continue;
            }
            mapper->TriangulateImage(tri_options, image.ImageId());

            std::cout << "  => Triangulated " << (image.NumMapPoints() - num_existing_points3D) << " points" << std::endl;

            mapper->ClearModifiedMapPoints();
        }

        std::cout << "i_cam:" << i_cam << " finished! " << num_aligned_images << " images aligned (" 
                          << feature_data_container->GetImageIds().size() << ", " << scene_graph_container->NumImages() 
                          << ", " << target_reconstruction->NumImages() << ")" << std::endl;

        std::cout << "feature_data_container image, camera size: " << feature_data_container->GetImageIds().size() << ", " << feature_data_container->NumCamera() << std::endl;
        std::cout << "scene_graph_container image, camera size: " << scene_graph_container->NumImages() << ", " << scene_graph_container->NumCameras() 
                        << "(" << scene_graph_container->Camera(sub_path_camera).ParamsToString() << ")" << std::endl;
        std::cout << "target_reconstruction image, camera size: " << target_reconstruction->NumImages() << ", " << target_reconstruction->NumCameras() 
                        << "(" << target_reconstruction->Camera(sub_path_camera).ParamsToString() << ")" << std::endl;

        {
            std::string rec_path = StringPrintf("%s/%d-add-%d", workspace_path.c_str(), 0, i_cam);
            if (!boost::filesystem::exists(rec_path)) {
                boost::filesystem::create_directories(rec_path);
            } else {
                boost::filesystem::remove_all(rec_path);
                boost::filesystem::create_directories(rec_path);
            }

            target_reconstruction->WriteReconstruction(rec_path, true);
            target_reconstruction->OutputPriorResidualsTxt(rec_path);
        }
    }

    // Bundle adjustment
    {
        std::shared_ptr<IncrementalMapper> mapper = std::make_shared<IncrementalMapper>(scene_graph_container);
        mapper->BeginReconstruction(target_reconstruction);

        PrintHeading1("Retriangulation");
        CompleteAndMergeTracks(mapper_options, mapper);

        auto ba_options = mapper_options.GlobalBundleAdjustment();
        ba_options.refine_focal_length = false;
        ba_options.refine_principal_point = false;
        ba_options.refine_extra_params = false;
        ba_options.refine_extrinsics = false;

        BundleAdjustmentConfig ba_config;
        // std::vector<image_t> reg_image_ids = target_reconstruction->RegisterImageIds();
        for(size_t idx = 0; idx < new_image_ids.size(); idx++){
            const image_t image_id = new_image_ids[idx];
            if (!target_reconstruction->IsImageRegistered(image_id)){
                continue;
            }
            ba_config.AddImage(image_id);
        }
        std::cout << "ba_config: " << ba_config.NumImages() << std::endl;

        for (int i = 0; i < 1; ++i) {
            target_reconstruction->FilterObservationsWithNegativeDepth();

            const size_t num_observations = target_reconstruction->ComputeNumObservations();

            PrintHeading1("Constant Pose Bundle adjustment");
            std::cout << "iter: " <<  i << std::endl;
            BundleAdjuster bundle_adjuster(ba_options, ba_config);
            CHECK(bundle_adjuster.Solve(target_reconstruction.get()));

            size_t num_changed_observations = 0;
            num_changed_observations += CompleteAndMergeTracks(mapper_options, mapper);
            // num_changed_observations += FilterPoints(mapper_options, mapper, min_track_length);
            const double changed = static_cast<double>(num_changed_observations) / num_observations;
            std::cout << StringPrintf("  => Changed observations: %.6f", changed) << std::endl;

            size_t num_retriangulate_observations = 0;
            num_retriangulate_observations = mapper->Retriangulate(mapper_options.Triangulation());
            std::cout << num_observations << " <=> " << num_retriangulate_observations << std::endl;

            if (changed < mapper_options.ba_global_max_refinement_change) {
                break;
            }
        }

        if (mapper_options.extract_colors) {
            for (size_t i = 0; i < new_image_ids.size(); ++i) {
                ExtractColors(image_path, new_image_ids[i], target_reconstruction);
            }
        }
    }

    // DeleteRedundantImages(reconstruction_, image_path);
    std::cout << "Registered: " << target_reconstruction->RegisterImageIds().size() << std::endl;
    // reconstruction_->WriteReconstruction(result_path);

    return 1;
}

double RescaleByStatitics(
    Reconstruction & reconstruction,
    const std::vector<std::pair<double, double>> & scale_candidates
) {
    double scale = GetBestScaleByStatitics(scale_candidates);
    reconstruction.RescaleAll(scale);
    return scale;
}

double RescaleByStatitics(
    const std::string & image_path,
    const std::unordered_map<image_t, std::pair<image_t, int>> & image_alignment,
    Reconstruction & reconstruction
) {
    std::vector<image_t> rgbd_ids;
    for (auto & item : image_alignment) {
        rgbd_ids.emplace_back(item.first);
    }

    std::vector<std::pair<double, double>> scale_candidates;
    std::vector<Eigen::Vector3d> rgbd_relative_t(rgbd_ids.size());
    #pragma omp parallel
    {
        std::vector<std::pair<double, double>> _scale_candidates;

        #pragma omp for schedule(dynamic, 1)
        for (int i = 0; i < rgbd_ids.size(); i++)
        {
            const image_t image_id = rgbd_ids[i];
            const std::string & image_name = reconstruction.Image(image_id).Name();
            const auto & image = reconstruction.Image(image_id);

            RGBDData data;
            ExtractRGBDData(JoinPaths(image_path, image_name), RGBDReadOption::NoColor(), data);
            if (!data.HasRGBDCalibration()) continue;

            Eigen::Matrix4d color_to_depth_RT = data.depth_RT.inverse();
            Eigen::Matrix3d color_R = QuaternionToRotationMatrix(image.Qvec());
            Eigen::Vector3d color_T = image.Tvec();
            Eigen::Matrix3d depth_R = color_to_depth_RT.block<3, 3>(0, 0) * color_R;    // Rd*Ri
            Eigen::Vector3d depth_T = color_to_depth_RT.block<3, 3>(0, 0) * color_T +   // Rd*ti + td
                                      color_to_depth_RT.block<3, 1>(0, 3) / 1000.0;
            std::vector<std::pair<double, double>> image_scale_candidates;
            const image_t target_image_id = image_alignment.at(image_id).first;
            const auto & target_image = reconstruction.Image(target_image_id);
            for (const auto & point2d : target_image.Points2D())
            {
                if (!point2d.HasMapPoint()) continue;
                auto mappoint_id = point2d.MapPointId();

                auto pt_3d = reconstruction.MapPoint(mappoint_id).XYZ();
                auto pt_cam = depth_R * pt_3d + depth_T;
                if (pt_cam.z() <= 0) continue;
                
                auto image_coord = data.depth_camera.WorldToImage(pt_cam.hnormalized());
                int nx = image_coord(0) + 0.5f;
                int ny = image_coord(1) + 0.5f;
                if (nx < 0 || nx >= data.depth.GetWidth() ||
                    ny < 0 || ny >= data.depth.GetHeight()
                ) {
                    continue;    
                }

                double rgbd_depth = data.depth.Get(ny, nx);
                if (rgbd_depth > 0) {
                    image_scale_candidates.emplace_back(
                        pt_cam.z() / rgbd_depth, 
                        1.0
                    );
                }
            }

            const int local_cam_id = image_alignment.at(image_id).second;
            auto target_R = QuaternionToRotationMatrix(target_image.Qvec());
            auto target_T = target_image.Tvec();
            if (local_cam_id >= 0) {
                Eigen::Vector4d local_qvec;
                Eigen::Vector3d local_tvec;
                const auto target_camera = reconstruction.Camera(target_image.CameraId());
                target_camera.GetLocalCameraExtrinsic(local_cam_id, local_qvec, local_tvec);
                target_R = QuaternionToRotationMatrix(local_qvec) * target_R;
                target_T = QuaternionToRotationMatrix(local_qvec) * target_T + local_tvec;
            }
            Eigen::Matrix3d relative_R = color_R * target_R.transpose();
            Eigen::Vector3d relative_t = -relative_R * target_T + color_T;
            rgbd_relative_t[i] = relative_t;

            // {
            //     Eigen::Matrix3d current_R = target_R * color_R.inverse();
            //     Eigen::Vector3d current_T = target_T - current_R * color_T;
            //     if (i == 100) {
            //         std::cout << current_R << std::endl;
            //         std::cout << current_T.transpose() << std::endl;
            //     }
            // }

            const size_t image_scale_count = image_scale_candidates.size();
            if (image_scale_count > 10) {
                std::sort(image_scale_candidates.begin(), image_scale_candidates.end(), ScaleCandidateComparer);
                _scale_candidates.insert(_scale_candidates.end(), 
                    image_scale_candidates.begin() + image_scale_count * 0.2,
                    image_scale_candidates.begin() + image_scale_count * 0.8);
            }
        }
    
        #pragma omp critical
        {
            scale_candidates.reserve(scale_candidates.size() + _scale_candidates.size());
            scale_candidates.insert(scale_candidates.end(), _scale_candidates.begin(), _scale_candidates.end());
        }
    }
    
    double scale = RescaleByStatitics(reconstruction, scale_candidates);

    for (int i = 0; i < rgbd_ids.size(); i++)
    {
        const image_t image_id = rgbd_ids[i];
        auto & image = reconstruction.Image(image_id);
        image.Tvec() += (1.0 - scale) * rgbd_relative_t[i];
    }

    return scale;
}

double RescaleByStatitics(
    const std::string & image_path,
    Reconstruction & reconstruction
) {
    auto image_names = reconstruction.GetImageNames();
    std::vector<std::string> rgbd_names;
    for (auto & image_name : image_names) {
        if (!IsFileRGBD(image_name.first)) continue;
        rgbd_names.emplace_back(image_name.first);
    }

    std::vector<std::pair<double, double>> scale_candidates;
    #pragma omp parallel
    {
        std::vector<std::pair<double, double>> _scale_candidates;

        #pragma omp for schedule(dynamic, 1)
        for(int i = 0; i < rgbd_names.size(); i += 1)
        {
            const std::string & image_name = rgbd_names[i];
            
            const image_t image_id = image_names[image_name];
            if ((!reconstruction.ExistsImage(image_id))) continue;

            const auto & image = reconstruction.Image(image_id);
            const auto & camera = reconstruction.Camera(image.CameraId());
            if (!image.IsRegistered()) continue;

            const int width = camera.Width();
            const int height = camera.Height();
            RGBDData data;
            ExtractRGBDData(JoinPaths(image_path, image_name), RGBDReadOption::NoColor(), data);
            if (!data.HasRGBDCalibration()) continue;

            Eigen::Matrix4d color_to_depth_RT = data.depth_RT.inverse();
            Eigen::Matrix3d color_R = QuaternionToRotationMatrix(image.Qvec());
            Eigen::Vector3d color_T = image.Tvec();
            Eigen::Matrix3d depth_R = color_to_depth_RT.block<3, 3>(0, 0) * color_R;    // Rd*Ri
            Eigen::Vector3d depth_T = color_to_depth_RT.block<3, 3>(0, 0) * color_T +   // Rd*ti + td
                                      color_to_depth_RT.block<3, 1>(0, 3) / 1000.0;
            std::vector<std::pair<double, double>> image_scale_candidates;
            for (const auto & point2d : image.Points2D())
            {
                if (!point2d.HasMapPoint()) continue;
                auto mappoint_id = point2d.MapPointId();

                auto pt_3d = reconstruction.MapPoint(mappoint_id).XYZ();
                auto pt_cam = depth_R * pt_3d + depth_T;
                if (pt_cam.z() <= 0) continue;
                
                auto image_coord = data.depth_camera.WorldToImage(pt_cam.hnormalized());
                int nx = image_coord(0) + 0.5f;
                int ny = image_coord(1) + 0.5f;
                if (nx < 0 || nx >= data.depth.GetWidth() ||
                    ny < 0 || ny >= data.depth.GetHeight()
                ) {
                    continue;    
                }

                double rgbd_depth = data.depth.Get(ny, nx);
                if (rgbd_depth > 0) {
                    image_scale_candidates.emplace_back(
                        pt_cam.z() / rgbd_depth, 
                        1.0
                    );
                }
            }

            const size_t image_scale_count = image_scale_candidates.size();
            if (image_scale_count > 10) {
                std::sort(image_scale_candidates.begin(), image_scale_candidates.end(), ScaleCandidateComparer);
                _scale_candidates.insert(_scale_candidates.end(), 
                    image_scale_candidates.begin() + image_scale_count * 0.1,
                    image_scale_candidates.begin() + image_scale_count * 0.9);
            }
        }
    
        #pragma omp critical
        {
            scale_candidates.reserve(scale_candidates.size() + _scale_candidates.size());
            scale_candidates.insert(scale_candidates.end(), _scale_candidates.begin(), _scale_candidates.end());
        }
    }

    return RescaleByStatitics(reconstruction, scale_candidates);
}

double RescaleByReconstruction(
    const Reconstruction & align_reconstruction,
    Reconstruction & reconstruction
) {
    std::unordered_map<std::string, Eigen::Vector3d> image_positions;
    for (const auto & image : align_reconstruction.Images()) {
        auto R = QuaternionToRotationMatrix(image.second.Qvec());
        auto t = image.second.Tvec();
        auto p = -R.transpose() * t;

        image_positions[image.second.Name()] = p;
    }

    std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> image_position_pairs;
    for (const auto & image : reconstruction.Images()) {
        if (image_positions.count(image.second.Name())) {
            auto R = QuaternionToRotationMatrix(image.second.Qvec());
            auto t = image.second.Tvec();
            auto p = -R.transpose() * t;

            image_position_pairs.emplace_back(p, image_positions[image.second.Name()]);
        }
    }

    double scale = 1.0;
    const size_t count = image_position_pairs.size();
    std::cout << count << " images in reconstruction for scale adjustment" << std::endl;
    if (count > 10) {
        Eigen::Vector3d src_center = Eigen::Vector3d::Zero();
        Eigen::Vector3d dst_center = Eigen::Vector3d::Zero();
        for (auto & pair : image_position_pairs) {
            src_center += pair.first;
            dst_center += pair.second;
        }
        src_center /= count;
        dst_center /= count;

        double src_distance = 0.0;
        double dst_distance = 0.0;
        for (size_t i = 0; i < count; i++) {
            const auto & from = image_position_pairs[i].first;
            const auto & to = image_position_pairs[i].second;
            src_distance += (from - src_center).norm();
            dst_distance += (to - dst_center).norm();
        }

        if (src_distance > 0.00001) {
            scale = dst_distance / src_distance;
        }

        std::cout << "Reconstruction scale: " << scale << std::endl;
    }

    reconstruction.RescaleAll(scale);
    return scale;
}

std::tuple<double, double> GetDepthReprojectionError(
    const std::string & image_path,
    const Reconstruction & reconstruction
) {
    std::vector<double> total_diffs;
    for (int i_cam = 0; i_cam < rgbd_infos.size(); i_cam++)
    {
        std::string sub_path = rgbd_infos[i_cam].sub_path;
        std::vector<image_t> rgbd_images;
        for (auto image_id : reconstruction.RegisterImageIds()) {
            auto & image = reconstruction.Image(image_id);
            if (IsFileRGBD(image.Name()) && IsInsideSubpath(image.Name(), sub_path)) {
                rgbd_images.emplace_back(image_id);
            }
        }
        std::sort(rgbd_images.begin(), rgbd_images.end(), 
        [&](image_t a, image_t b) {
            return reconstruction.Image(a).Name() < reconstruction.Image(b).Name();
        });

        #pragma omp parallel for schedule(dynamic, 1)
        for (int i = 1; i < rgbd_images.size() - 1; i++) {
            std::vector<image_t> src_image_ids {
                rgbd_images[i - 1],
                rgbd_images[i + 1],
            };

            RGBDData ref_image_data;
            auto & ref_image = reconstruction.Image(rgbd_images[i]);
            Eigen::Matrix3d ref_R = QuaternionToRotationMatrix(ref_image.Qvec());
            Eigen::Vector3d ref_t = ref_image.Tvec();
            ExtractRGBDData(JoinPaths(image_path, ref_image.Name()), RGBDReadOption::NoColor(), ref_image_data);

            for (int j = 0; j < src_image_ids.size(); j++) {
                RGBDData src_image_data;
                auto & src_image = reconstruction.Image(src_image_ids[j]);
                Eigen::Matrix3d src_R = QuaternionToRotationMatrix(src_image.Qvec());
                Eigen::Vector3d src_t = src_image.Tvec();
                ExtractRGBDData(JoinPaths(image_path, src_image.Name()), RGBDReadOption::NoColor(), src_image_data);

                Eigen::Matrix3d rel_R = ref_R * src_R.transpose();
                Eigen::Vector3d rel_t = -rel_R * src_t + ref_t;

                std::vector<double> current_diffs;
                auto & src_depthmap = src_image_data.depth;
                auto & ref_depthmap = ref_image_data.depth;
                for (int y = 0; y < src_depthmap.GetHeight(); y++) {
                    for (int x = 0; x < src_depthmap.GetWidth(); x++) {
                        float src_depth = src_depthmap.Get(y, x);
                        if (src_depth > 0) {
                            Eigen::Vector3d src_pt = src_image_data.depth_camera.ImageToWorld(Eigen::Vector2d(x, y)).homogeneous();
                            src_pt *= src_depth;

                            Eigen::Vector3d ref_pt = rel_R * src_pt + rel_t;
                            Eigen::Vector2d ref_uv = ref_image_data.depth_camera.WorldToImage(ref_pt.hnormalized());

                            int u = std::round(ref_uv.x());
                            int v = std::round(ref_uv.y());
                            if (u >= 0 && u < ref_depthmap.GetWidth() && v >= 0 && v < ref_depthmap.GetHeight()) {
                                float ref_depth = ref_depthmap.Get(v, u);
                                if (ref_depth > 0) {
                                    current_diffs.emplace_back(ref_depth - ref_pt.z());
                                }
                            }
                        }
                    }
                }
                std::sort(current_diffs.begin(), current_diffs.end());
                if (current_diffs.size() > 10) {
                    #pragma omp critical 
                    {
                        // total_diffs.emplace_back(current_diffs[current_diffs.size() / 2]);
                        // total_diffs.insert(total_diffs.end(), current_diffs.begin(), current_diffs.end());
                        total_diffs.insert(total_diffs.end(), current_diffs.begin() + current_diffs.size() * 0.2, current_diffs.begin() + current_diffs.size() * 0.8);
                    }
                }
            }
        }
    }

    double rmse = 0.0;
    double mae = 0.0;
    for (int i = 0; i < total_diffs.size(); i++) {
        double diff = total_diffs[i];
        mae += std::abs(diff);
        rmse += diff * diff;
    }
    mae = mae / total_diffs.size();
    rmse = std::sqrt(rmse / total_diffs.size());
    std::cout << mae << " " << rmse << std::endl;

    return std::make_tuple(mae, rmse);
}

std::tuple<double, size_t, size_t> CalculateImageReprojectionDiffByPose(
    const Reconstruction & reconstruction,
    const Camera & rgbd_camera,
    const Eigen::Vector4d & image_qvec,
    const Eigen::Vector3d & image_tvec,
    const Eigen::Vector4d & delta_qvec,
    const Eigen::Vector3d & delta_tvec,
    image_t image_id,
    int local_camera_id = -1
) {
    const auto & image = reconstruction.Image(image_id);
    const auto & camera = reconstruction.Camera(image.CameraId());

    double error = 0.0;
    size_t count = 0;
    size_t point_count = 0;
    size_t total_count = 0;
    for (int point2D_idx = 0; point2D_idx < image.Points2D().size(); point2D_idx++) {
        if (local_camera_id >= 0 && local_camera_id != image.LocalImageIndices()[point2D_idx]) continue;
        total_count++;

        auto & point2D = image.Points2D()[point2D_idx];
        if (!point2D.HasMapPoint()) continue;
        point_count++;

        const auto & mappoint = reconstruction.MapPoint(point2D.MapPointId());
        const Eigen::Vector3d proj_point3D0 =
            QuaternionRotatePoint(image_qvec, mappoint.XYZ()) + image_tvec;
        const Eigen::Vector3d proj_point3D1 =
            QuaternionRotatePoint(delta_qvec, mappoint.XYZ()) + delta_tvec;
        if (proj_point3D0.z() < std::numeric_limits<double>::epsilon()) continue;
        if (proj_point3D1.z() < std::numeric_limits<double>::epsilon()) continue;

        const Eigen::Vector2d proj_point2D0 =
            rgbd_camera.WorldToImage(proj_point3D0.hnormalized());
        const Eigen::Vector2d proj_point2D1 =
            rgbd_camera.WorldToImage(proj_point3D1.hnormalized());
        if (proj_point2D0.x() < 0 || proj_point2D0.y() < 0 || proj_point2D0.x() >= rgbd_camera.Width() || proj_point2D0.y() >= rgbd_camera.Height()) continue;
        if (proj_point2D1.x() < 0 || proj_point2D1.y() < 0 || proj_point2D1.x() >= rgbd_camera.Width() || proj_point2D1.y() >= rgbd_camera.Height()) continue;

        error += (proj_point2D1 - proj_point2D0).norm();
        count++;
    }

    if (count == 0) count = 1;
    return std::make_tuple(error / count, point_count, total_count);
}

LinearWeightInterpolator GetIndexWeightInterpolator(
    Configurator & param,
    Reconstruction & target_reconstruction
) {
    std::vector<std::pair<int, Camera>> local_camera_to_rgbd_camera;
    std::string image_path = param.GetArgument("image_path", "");
    for(int i_cam = 0; i_cam < rgbd_infos.size(); i_cam++)
    {
        std::string sub_path = rgbd_infos[i_cam].sub_path;
        std::string camth = rgbd_infos[i_cam].calib_cam;
        int local_cam_id = camth.empty() ? -1 : std::atoi(camth.substr(3, camth.size()).c_str());
        
        Camera rgbd_camera;
        std::string rgbd_path = JoinPaths(image_path, sub_path);
        std::vector<std::string> rgbd_files = GetRecursiveFileList(rgbd_path);
        for (auto & rgbd_file : rgbd_files) {
            RGBDData data;
            ExtractRGBDData(rgbd_file, RGBDReadOption::NoColorNoDepth(), data);
            if (!data.HasRGBDCalibration()) continue;
            rgbd_camera = data.color_camera;

            break;
        }
        CHECK_NE(rgbd_camera.ModelId(), kInvalidCameraModelId);

        local_camera_to_rgbd_camera.emplace_back(local_cam_id, rgbd_camera);
    }
    CHECK_EQ(rgbd_infos.size(), local_camera_to_rgbd_camera.size());
    CHECK_GT(local_camera_to_rgbd_camera.size(), 0);

    LinearWeightInterpolator interp;
    std::map<int64_t, image_t> target_image_index_map;
    for (auto & image : target_reconstruction.Images()) {
        const std::string image_name = image.second.Name();
        if (IsInsideSubpath(image_name, target_subpath) && image.second.IsRegistered()) {
            int64_t index = ImageNameToIndex(image_name);
            target_image_index_map[index] = image.first;
            target_camera = target_reconstruction.Camera(image.second.CameraId());
        }
    }
    CHECK_NE(target_camera.ModelId(), kInvalidCameraModelId);

    double average_index_diff = 0.0;
    {
        std::vector<double> index_diff;
        for (auto iter = target_image_index_map.begin(); iter != target_image_index_map.end(); ++iter) {
            auto prev = iter; prev--;
            if (prev == target_image_index_map.end()) continue;

            index_diff.emplace_back(iter->first - prev->first);
        }
        std::sort(index_diff.begin(), index_diff.end());
        average_index_diff = index_diff[index_diff.size() / 2];

        std::cout << "Average index diff: " << average_index_diff << std::endl;
    }

    for (auto & image : target_reconstruction.Images()) {
        const std::string image_name = image.second.Name();
        if (IsInsideSubpath(image_name, target_subpath) && image.second.IsRegistered()) {
            Eigen::Vector4d qvec = image.second.Qvec();
            Eigen::Vector3d tvec = image.second.Tvec();
            int64_t index = ImageNameToIndex(image_name);
            auto find = target_image_index_map.find(index);
            CHECK(find != target_image_index_map.end());
            if (--find == target_image_index_map.end()) continue;

            int64_t prev_index = find->first;
            image_t prev_image = find->second;
            Eigen::Vector4d prev_qvec = target_reconstruction.Image(prev_image).Qvec();
            Eigen::Vector3d prev_tvec = target_reconstruction.Image(prev_image).Tvec();
            if (index - prev_index > average_index_diff * 10) continue;

            const Camera & camera = target_reconstruction.Camera(image.second.CameraId());
            const double delta = 0.01 * average_index_diff / (index - prev_index);
            Eigen::VectorXd errors(local_camera_to_rgbd_camera.size());
            errors.setZero();
            for(int i_cam = 0; i_cam < local_camera_to_rgbd_camera.size(); i_cam++) {
                const auto & rgbd_item = local_camera_to_rgbd_camera[i_cam];
                const int local_camera_id = rgbd_item.first;
                Eigen::Vector4d local_qvec;
                Eigen::Vector3d local_tvec;
                camera.GetLocalCameraExtrinsic(local_camera_id, local_qvec, local_tvec);

                Eigen::Vector4d qvec0 = ConcatenateQuaternions(qvec, local_qvec);
                Eigen::Vector4d qvec1 = ConcatenateQuaternions(prev_qvec, local_qvec);
                Eigen::Vector3d tvec0 = 
                    QuaternionToRotationMatrix(local_qvec) * tvec + local_tvec;
                Eigen::Vector3d tvec1 = 
                    QuaternionToRotationMatrix(local_qvec) * prev_tvec + local_tvec;

                Eigen::Matrix3d inv_extra_R = rgbd_infos[i_cam].extra_R.inverse();
                Eigen::Vector3d extra_T = rgbd_infos[i_cam].extra_T;
                qvec0 = ConcatenateQuaternions(qvec0, RotationMatrixToQuaternion(inv_extra_R));
                qvec1 = ConcatenateQuaternions(qvec1, RotationMatrixToQuaternion(inv_extra_R));
                tvec0 = inv_extra_R * (tvec0 - extra_T);
                tvec1 = inv_extra_R * (tvec1 - extra_T);

                double error;
                size_t point_count, total_count;
                std::tie(error, point_count, total_count) = CalculateImageReprojectionDiffByPose(
                    target_reconstruction,
                    rgbd_item.second,
                    qvec0, tvec0,
                    qvec0 + delta * (qvec1 - qvec0), 
                    tvec0 + delta * (tvec1 - tvec0),
                    image.first,
                    local_camera_id
                );
                errors[i_cam] = error / total_count;
            }

            errors /= delta;
            interp.Add(index, errors); 
        }
    }
    CHECK_GT(interp.data_.size(), 0);
    std::cout << "GetIndexWeightInterpolator Done" << std::endl;
    return interp;
}

void UpdateImagePose(Reconstruction & reconstruction){

    const auto& rec_image_ids = reconstruction.RegisterImageIds();
    for(auto image_id : rec_image_ids){
        Image &image = reconstruction.Image(image_id);
        Camera &camera = reconstruction.Camera(image.CameraId());
        if (!(camera.HasDisturb() && image.HasQvecPrior() && image.HasTvecPrior())){
            continue;
        }

        Eigen::Vector4d prior_qvec = image.QvecPrior();
        Eigen::Vector3d prior_tvec = QuaternionToRotationMatrix(prior_qvec) * -image.TvecPrior();;
        Eigen::Vector4d delt_qvec = camera.QvecDisturb();
        Eigen::Vector3d delt_tvec = camera.TvecDisturb();

        Eigen::Vector4d qvec;
        Eigen::Vector3d tvec;
        ConcatenatePoses(prior_qvec, prior_tvec, delt_qvec, delt_tvec, &qvec, &tvec);

        image.SetQvec(qvec);
        image.SetTvec(tvec);
    }
}

void OptimizeExtraParams(
    Reconstruction & reconstruction, 
    const std::unordered_map<image_t, std::pair<image_t, int>> & image_alignment,
    Configurator &param){
    PrintHeading1("OptimizeExtraParams");
    std::vector<camera_t > rgbd_camera_ids;

    // Set alignment Prior Pose
    for (const auto & image_map : image_alignment){
        if (reconstruction.IsImageRegistered(image_map.first)){
            auto& image = reconstruction.Image(image_map.first);
            const auto& qvec = image.Qvec();
            const auto& tvec = image.Tvec();

            image.SetQvecPrior(qvec);
            image.SetTvecPrior(ProjectionCenterFromPose(qvec, tvec));
        }
    }

    // Bundle Adjustment
    IndependentMapperOptions mapper_options;
    OptionParser option_parser;
    option_parser.GetMapperOptions(mapper_options,param);

    BundleAdjustmentConfig ba_config;
    auto ba_options = mapper_options.GlobalBundleAdjustment();
    ba_options.refine_focal_length = false;
    ba_options.refine_principal_point = false;
    ba_options.refine_extra_params = false;
    ba_options.refine_extrinsics = false;
    
    const auto& rec_image_ids = reconstruction.Images();
    for (const auto& rec_image : rec_image_ids){
        const image_t image_id = rec_image.first;
        auto& image = reconstruction.Image(image_id);
        if (image_alignment.find(image_id) == image_alignment.end()){
            ba_config.AddImage(image_id);
            ba_config.SetConstantPose(image_id);
        } else {
            ba_config.AddGNSS(image_id);
            auto& camera = reconstruction.Camera(image.CameraId());
            if (!camera.HasDisturb()){
                camera.SetDisturb();
                rgbd_camera_ids.push_back(image.CameraId());
            }
        }
    }
    for (int i = 0; i < 2; ++i) {
        reconstruction.FilterObservationsWithNegativeDepth();

        const size_t num_observations = reconstruction.ComputeNumObservations();

        PrintHeading1("RefineExtrincParam Bundle adjustment");
        std::cout << "iter: " << i << std::endl;
        BundleAdjuster bundle_adjuster(ba_options, ba_config);
        CHECK(bundle_adjuster.Solve(&reconstruction));
        
        UpdateImagePose(reconstruction);
        std::cout << "UpdateImagePose  ... Done" << std::endl;
        for (int i = 0;i < rgbd_camera_ids.size(); i++){
            const auto camera_id = rgbd_camera_ids.at(i);
            std::cout << "Camera_" << camera_id << " delt Translation: "
                << std::setprecision(9) 
                << reconstruction.Camera(camera_id).QvecDisturb()(0) << " "
                << reconstruction.Camera(camera_id).QvecDisturb()(1) << " "
                << reconstruction.Camera(camera_id).QvecDisturb()(2) << " "
                << reconstruction.Camera(camera_id).QvecDisturb()(3) << "/ "
                << reconstruction.Camera(camera_id).TvecDisturb()(0) << " "
                << reconstruction.Camera(camera_id).TvecDisturb()(1) << " "
                << reconstruction.Camera(camera_id).TvecDisturb()(2) << std::endl;
        }
    }

    std::string workspace_path = param.GetArgument("workspace_path", "");
    CHECK(!workspace_path.empty()) << "workspace path empty";
    std::string rec_path = StringPrintf("%s/%d-optim", workspace_path.c_str(), 0);
    if (!boost::filesystem::exists(rec_path)) {
        boost::filesystem::create_directories(rec_path);
    } else {
        boost::filesystem::remove_all(rec_path);
        boost::filesystem::create_directories(rec_path);
    }

    reconstruction.WriteReconstruction(rec_path, true);
    reconstruction.OutputPriorResidualsTxt(rec_path);

    return;
}


void DistrubGBA(std::shared_ptr<sensemap::SceneGraphContainer> &scene_graph_container,
                std::shared_ptr<sensemap::ReconstructionManager> &reconstruction_manager,
                const std::vector<image_t>& ori_imags_ids,
                Configurator &param) {
    using namespace sensemap;

    auto reconstruction = reconstruction_manager->Get(0);
    std::vector<image_t> reg_image_ids = reconstruction->RegisterImageIds();
    std::unordered_set<image_t > ori_image_ids_set(ori_imags_ids.begin(), ori_imags_ids.end());

    // // Compute Image Numpoint
    // std::unordered_map<camera_t, std::pair<size_t, size_t>> num_verbose_point;
    // for (size_t i = 0; i < reg_image_ids.size(); ++i){
    //     const image_t image_id = reg_image_ids[i];
    //     if(ori_image_ids_set.find(image_id) == ori_image_ids_set.end()){
    //         const auto& image = reconstruction->Image(image_id);
    //         const camera_t camera_id = image.CameraId();
    //         if(num_verbose_point.find(camera_id) == num_verbose_point.end()){
    //             num_verbose_point[camera_id] = std::pair<size_t, size_t>(0, 0);
    //         }

    //         num_verbose_point[camera_id].first++;
    //         num_verbose_point[camera_id].second += image.NumMapPoints();
    //     }
    // }

    // Reset Image Prior Info
    for (size_t i = 0; i < reg_image_ids.size(); ++i) {
        const image_t image_id = reg_image_ids[i];
        if(ori_image_ids_set.find(image_id) == ori_image_ids_set.end()){
            auto& image = reconstruction->Image(image_id);
            image.SetQvecPrior(image.Qvec());
            image.SetTvecPrior(image.ProjectionCenter());

            float num_pints = image.NumMapPoints();
            float weig_factor = std::min((num_pints + 1) / points_threshold, 1.0f);
            float weig_std = 1 - std::cos(weig_factor);
            image.SetRtkStd(weig_std, weig_std, weig_std);
            image.SetOrientStd(weig_std);
            image.RtkFlag() = (int8_t)50;
        }
    }

    
    PrintHeading1("Distrub GBA");

    std::string workspace_path = param.GetArgument("workspace_path", "");
    CHECK(!workspace_path.empty()) << "workspace path empty";

    // for (int rect_id=0; rect_id<reconstruction_manager->Size(); rect_id++){
    if (1){

        std::shared_ptr<IncrementalMapper> mapper = std::make_shared<IncrementalMapper>(scene_graph_container);
        mapper->BeginReconstruction(reconstruction);
        
        IndependentMapperOptions mapper_options;
        OptionParser option_parser;
        option_parser.GetMapperOptions(mapper_options,param);

        auto ba_options = mapper_options.GlobalBundleAdjustment();
        ba_options.refine_focal_length = false;
        ba_options.refine_principal_point = false;
        ba_options.refine_extra_params = false;
        ba_options.refine_extrinsics = true;

        reconstruction->b_aligned = true;
        ba_options.use_prior_absolute_location = true;
        if (ba_options.prior_absolute_orientation_weight < 1e-6){
            ba_options.prior_absolute_orientation_weight = 0.1;
        }
        std::cout << "prior_absolute_location_weight, prior_absolute_orientation_weight: " 
                  << ba_options.prior_absolute_location_weight << ", " 
                  << ba_options.prior_absolute_orientation_weight << std::endl;

        // Configure bundle adjustment.
        BundleAdjustmentConfig ba_config;
        for (size_t i = 0; i < reg_image_ids.size(); ++i) {
            const image_t image_id = reg_image_ids[i];
            ba_config.AddImage(image_id);
            if (ori_image_ids_set.find(image_id) != ori_image_ids_set.end()){
                ba_config.SetConstantPose(image_id);
                const auto& image =  reconstruction->Image(image_id);
                ba_config.SetConstantCamera(image.CameraId());
            }
        }
        std::cout << "ba_config: num_images, num_const_images, num_camrea:" 
            << ba_config.NumImages() << ", " << ba_config.NumConstantPoses() 
            << ", " << ba_config.NumConstantCameras() << std::endl;

        for (int i = 0; i < mapper_options.ba_global_max_refinements; ++i) {
            reconstruction->FilterObservationsWithNegativeDepth();

            const size_t num_observations = reconstruction->ComputeNumObservations();

            PrintHeading1("GBA Bundle adjustment");
            std::cout << "iter: " << i << std::endl;
            BundleAdjuster bundle_adjuster(ba_options, ba_config);
            CHECK(bundle_adjuster.Solve(reconstruction.get()));

            size_t num_changed_observations = 0;
            num_changed_observations += CompleteAndMergeTracks(mapper_options, mapper);
            num_changed_observations += FilterPoints(mapper_options, mapper, mapper_options.min_track_length);
            const double changed = static_cast<double>(num_changed_observations) / num_observations;
            std::cout << StringPrintf("  => Changed observations: %.6f", changed) << std::endl;

            size_t num_retriangulate_observations = 0;
            num_retriangulate_observations = mapper->Retriangulate(mapper_options.Triangulation());
            std::cout << "\nnum_retri_observations / num_ori_observations: "
                        << num_observations << " / "
                        << num_retriangulate_observations << std::endl;

            if (changed < mapper_options.ba_global_max_refinement_change) {
                break;
            }
        }
    }

}

void WriteAlignmentResult(
    const Reconstruction & reconstruction, 
    const std::unordered_map<image_t, std::pair<image_t, int>> & image_alignment,
    const std::string & path
) {
    std::unordered_set<image_t> image_ids;
    for (auto & pair : image_alignment) {
        image_ids.insert(pair.first);
    }
    reconstruction.WriteAlignmentBinary(path, image_ids);
}

void ReadTargetReconstruction(const std::string target_workspace_path,
    std::shared_ptr<FeatureDataContainer> feature_data_container,
    std::shared_ptr<SceneGraphContainer> scene_graph_container,
    Reconstruction& target_reconstruction){
    
    // Load original feature
    if (boost::filesystem::exists(JoinPaths(target_workspace_path, "/local_cameras.bin"))) {
        feature_data_container->ReadCamerasBinaryData(JoinPaths(target_workspace_path, "/cameras.bin"), false);
        feature_data_container->ReadLocalCamerasBinaryData(JoinPaths(target_workspace_path, "/local_cameras.bin"));
    } else {
        feature_data_container->ReadCamerasBinaryData(JoinPaths(target_workspace_path, "/cameras.bin"), true);
    }
    feature_data_container->ReadImagesBinaryData(target_workspace_path + "/features.bin");
    
    // Load Panorama feature
    feature_data_container->ReadSubPanoramaBinaryData(target_workspace_path + "/sub_panorama.bin");
    if (boost::filesystem::exists(JoinPaths(target_workspace_path, "/piece_indices.bin"))) {
        feature_data_container->ReadPieceIndicesBinaryData(JoinPaths(target_workspace_path, "/piece_indices.bin"));
    } else if (boost::filesystem::exists(JoinPaths(target_workspace_path, "/piece_indices.txt"))) {
        feature_data_container->ReadPieceIndicesData(JoinPaths(target_workspace_path, "/piece_indices.txt"));
    }

    // load scenegraph
    bool load_scene_graph = false;
    if (boost::filesystem::exists(JoinPaths(target_workspace_path, "/scene_graph.bin"))) {
        scene_graph_container->ReadSceneGraphBinaryData(JoinPaths(target_workspace_path, "/scene_graph.bin"));
        load_scene_graph = true;
    } else if (boost::filesystem::exists(JoinPaths(target_workspace_path, "/scene_graph.txt"))) {
        scene_graph_container->ReadSceneGraphData(JoinPaths(target_workspace_path, "/scene_graph.txt"));
        load_scene_graph = true;
    }

    if (load_scene_graph) {
        EIGEN_STL_UMAP(image_t, class Image) &images = scene_graph_container->Images();
        EIGEN_STL_UMAP(camera_t, class Camera) &cameras = scene_graph_container->Cameras();

        std::vector<image_t> image_ids = feature_data_container->GetImageIds();

        for (const auto image_id : image_ids) {
            const Image &image = feature_data_container->GetImage(image_id);
            if (!scene_graph_container->CorrespondenceGraph()->ExistsImage(image_id)) {
                continue;
            }

            images[image_id] = image;

            const FeatureKeypoints &keypoints = feature_data_container->GetKeypoints(image_id);
            images[image_id].SetPoints2D(keypoints);
            const PanoramaIndexs & panorama_indices = feature_data_container->GetPanoramaIndexs(image_id);

            const Camera &camera = feature_data_container->GetCamera(image.CameraId());

            std::vector<uint32_t> local_image_indices(keypoints.size());
            for(size_t i = 0; i<keypoints.size(); ++i){
                if (panorama_indices.size() == 0 && camera.NumLocalCameras() == 1) {
                    local_image_indices[i] = image_id;
                } else {
                    local_image_indices[i] = panorama_indices[i].sub_image_id;
                }
            }
            images[image_id].SetLocalImageIndices(local_image_indices);

            if (!scene_graph_container->ExistsCamera(image.CameraId())) {
                cameras[image.CameraId()] = camera;
            }

            if (scene_graph_container->CorrespondenceGraph()->ExistsImage(image_id)) {
                images[image_id].SetNumObservations(
                    scene_graph_container->CorrespondenceGraph()->NumObservationsForImage(image_id));
                images[image_id].SetNumCorrespondences(
                    scene_graph_container->CorrespondenceGraph()->NumCorrespondencesForImage(image_id));
            } else {
                std::cout << "Do not contain ImageId = " << image_id << ", in the correspondence graph." << std::endl;
            }
        }

        scene_graph_container->CorrespondenceGraph()->Finalize();

        // Set overlap flag of keypoints.
        for (auto& image : scene_graph_container->Images()) {
            if (!scene_graph_container->CorrespondenceGraph()->ExistsImage(image.first)) {
                continue;
            }
            const FeatureMatches& corrs = 
            scene_graph_container->CorrespondenceGraph()->FindCorrespondencesBetweenImages(image.first, image.first);
            for (const FeatureMatch& corr : corrs) {
                image.second.Point2D(corr.point2D_idx1).SetOverlap(true);
                image.second.Point2D(corr.point2D_idx2).SetOverlap(true);
            }
        }
    }

    // load reconstruction
    std::string target_workspace_path_0 = JoinPaths(target_workspace_path, "0");
    target_reconstruction.ReadReconstruction(target_workspace_path_0, 1);
    for (auto image_id : target_reconstruction.RegisterImageIds()) {
        class Image& cur_image = target_reconstruction.Image(image_id);
        class Camera& cur_camera = target_reconstruction.Camera(cur_image.CameraId());

        const PanoramaIndexs & panorama_indices = feature_data_container->GetPanoramaIndexs(image_id);

        CHECK_EQ(cur_image.Points2D().size(), panorama_indices.size());
        std::vector<uint32_t> local_image_indices(panorama_indices.size(), 0);
        for(size_t i = 0; i < panorama_indices.size(); ++i){
            local_image_indices[i] = panorama_indices[i].sub_image_id;
        }

        cur_image.SetLocalImageIndices(local_image_indices);
    }

    return;
}



int main(int argc, char *argv[]) {
    using namespace sensemap;

    Timer timer;
    timer.Start();

    PrintHeading("SenseMap.  Copyright(c) 2019, SenseTime Group.");
    PrintHeading(std::string("Version: sfm-rgbd-align-")+__VERSION__);

    tool_name = boost::filesystem::path(argv[0]).filename().string();
    configuration_file_path = std::string(argv[1]);
    std::cout << "configuration_file_path: " << configuration_file_path << std::endl;

    Configurator param;
    param.Load(configuration_file_path.c_str());

    std::string workspace_path = param.GetArgument("workspace_path", "");
    std::string align_workspace_path_0 = JoinPaths(workspace_path, "0");
    CHECK(!workspace_path.empty()) << "workspace path empty";

    std::string image_path = param.GetArgument("image_path", "");
    CHECK(!image_path.empty()) << "image path empty";

	std::string target_workspace_path = param.GetArgument("target_workspace_path", "");
    std::string target_workspace_path_0 = JoinPaths(target_workspace_path, "0");
    CHECK(!target_workspace_path.empty()) << "target workspace path empty";

    std::string rgbd_params_file;

    // std::string camera_param_file = param.GetArgument("camera_param_file","");
    // if (camera_param_file == "") {
    //     ExceptionHandler(INVALID_INPUT_PARAM, 
    //         JoinPaths(GetParentDir(workspace_path), "errors/dense"), tool_name).Dump();
    //     exit(INVALID_INPUT_PARAM);
    // }

    bool rescale_flag = param.GetArgument("rescale", 1);
    std::cout << "rescale_flag: " << rescale_flag << std::endl;
    int num_cameras = param.GetArgument("num_cameras",0);
    int num_to_register = param.GetArgument("num_to_register",0);
    std::cout<<"num to register:"<<num_to_register<<std::endl;
    if (num_to_register <= 0) return StateCode::SUCCESS;

    std::vector<std::string> camera_param_files;
    camera_param_files=CSVToVector<std::string>(param.GetArgument("camera_param_files", ""));
    std::cout<<"camera_param_files size: "<<camera_param_files.size()<<std::endl;
    int cnt = 0;
    target_subpaths.clear();
    for(int j = 0; j < num_to_register; j++)
    {
        std::cout<<"camera_param_file "<<camera_param_files[j]<<std::endl;
        YAML::Node node = YAML::LoadFile(camera_param_files[j]);
        target_subpath = node["sub_path_0"].as<std::string>();
        target_subpaths.push_back(target_subpath);
        std::cout << "rgbd info no." << j << std::endl;
        std::cout << "camera file: " << camera_param_files[j] << std::endl;
        std::cout<<"target_subpath: "<<target_subpath<<std::endl;
        rgbd_infos.clear();
        for (int i = 1; i <3; i++) {
            RGBDInfo tmp;
            tmp.calib_cam = node["calib_cam_" + std::to_string(i)].as<std::string>();
            tmp.sub_path = node["sub_path_" + std::to_string(i)].as<std::string>();
            YAML::Node cv_mat_node = node["extrinsic_" + std::to_string(i)];
            if (cv_mat_node.IsDefined()) {
                std::vector<double> mat_data = cv_mat_node["data"].as<std::vector<double>>();
                int mat_rows = cv_mat_node["rows"].as<int>();
                int mat_cols = cv_mat_node["cols"].as<int>();
                CHECK_EQ(mat_data.size(), mat_rows * mat_cols);
                cv::Mat1d extrinsic(mat_rows, mat_cols, mat_data.data());

                cv::Mat extra_R_mat, extra_T_mat;
                extrinsic(cv::Rect(0, 0, 3, 3)).copyTo(extra_R_mat);
                cv::cv2eigen(extra_R_mat, tmp.extra_R);
                extrinsic(cv::Rect(3, 0, 1, 3)).copyTo(extra_T_mat);
                cv::cv2eigen(extra_T_mat, tmp.extra_T);
                tmp.extra_T /= 1000.f;
            } else {
                std::cerr << "Calib info for camera " << i << " not found!" << std::endl;
                ExceptionHandler(RGBD_CALIB_FAILED, JoinPaths(GetParentDir(workspace_path), "errors/dense"), tool_name)
                    .Dump();
                exit(RGBD_CALIB_FAILED);
            }

            if (node["timestamp_" + std::to_string(i)].IsDefined()) {
                tmp.timestamp = node["timestamp_" + std::to_string(i)].as<int>();
                std::cout << "timestamp: " << tmp.timestamp << std::endl;
            }

            if (node["rgbd_params_file_" + std::to_string(i)].IsDefined()) {
                Eigen::Matrix3f rgb_K, depth_K;
                Eigen::Matrix4f RT;
                rgbd_params_file = node["rgbd_params_file_" + std::to_string(i)].as<std::string>();
                std::cout << "rgbd_params_file: " << rgbd_params_file << std::endl;
                auto calib_reader = GetCalibBinReaderFromName(rgbd_params_file);
                calib_reader->ReadCalib(rgbd_params_file);

                tmp.rgbd_camera_params = calib_reader->ToParamString();
                std::cout << "rgbd_camera_params: " << tmp.rgbd_camera_params << std::endl;
            }

            if (node["offset_" + std::to_string(i)].IsDefined()) {
                tmp.has_force_offset = true;
                tmp.force_offset = node["offset_" + std::to_string(i)].as<int>();

                std::cout << "force_offset: " << tmp.force_offset << std::endl;
            }

            // std::cout << "tmp.extra_R: " << tmp.extra_R << std::endl;
            // std::cout << "tmp.extra_T: " << tmp.extra_T << std::endl;
            rgbd_infos.push_back(tmp);
        }
        rgbd_infos_vec.push_back(rgbd_infos);
    }
    std::cout<<"-----------rgbd info ed------------"<<std::endl;
    // std::cout<<"target_subpaths size: "<<target_subpaths.size()<<std::endl;
    // for(auto str: target_subpaths){
    //     std::cout<<"cnt: "<<cnt++<<" str: "<<str<<std::endl;
    // }

    if (!boost::filesystem::exists(workspace_path)) {
        CHECK(boost::filesystem::create_directories(workspace_path)) << "Create workspace failed";
    }

    auto feature_data_container = std::make_shared<FeatureDataContainer>();
    auto scene_graph_container = std::make_shared<SceneGraphContainer>();
    auto reconstruction_manager = std::make_shared<ReconstructionManager>();
    reconstruction_manager->Add();
    Reconstruction& target_reconstruction = *reconstruction_manager->Get(0);
    ReadTargetReconstruction(target_workspace_path, feature_data_container, 
        scene_graph_container, target_reconstruction);

    if (argc > 3 && std::string(argv[2]) == "--debug") {
        std::cout << "Entered Debug Mode" << std::endl;
        Camera rig_camera;
        for (auto & cam : target_reconstruction.Cameras()) {
            if (rig_camera.CameraId() == kInvalidCameraId) {
                rig_camera = cam.second;
            } else if (rig_camera.CameraId() < cam.second.CameraId()) {
                rig_camera = cam.second;
            }
        }
        CHECK(rig_camera.CameraId() != kInvalidCameraId);
        CalibDebug(argv[3], rig_camera);
        return 0;
    }

    // Create all Feature container
    std::cout << "feature_data_container image size: " << feature_data_container->GetImageIds().size() << std::endl;
    std::cout << "scene_graph_container image size: " << scene_graph_container->NumImages() << std::endl;
    std::cout << "target_reconstruction image size: " << target_reconstruction.NumImages() << std::endl;

    std::unordered_map<image_t, std::pair<image_t, int>> image_alignment;
    double scale_factor = 1.0;
    bool result;
    std::vector<image_t> old_image_ids;
    std::cout << "rgbd_infos_vec size: " << rgbd_infos_vec.size() << " target_subpaths size: " << target_subpaths.size()
              << std::endl;
    for (int i = 0; i < rgbd_infos_vec.size(); i++) {
        target_subpath = target_subpaths[i];
        rgbd_infos=rgbd_infos_vec[i];
        std::cout << "data segment index: " << i << " tarpath: " << target_subpath << std::endl;
        param.SetArgument("camera_param_file", camera_param_files[i]);
        std::string camera_param_file_cur=param.GetArgument("camera_param_file", "");
        std::cout<<"camera_param_file: "<<camera_param_file_cur<<std::endl;
        const auto old_image_ids_tmp = target_reconstruction.RegisterImageIds();
        old_image_ids.insert(old_image_ids.end(), old_image_ids_tmp.begin(), old_image_ids_tmp.end());

        auto interp = GetIndexWeightInterpolator(param, target_reconstruction);
        auto align_reconstruction = RGBDRegistration(param, interp);
        if (param.GetArgument("image_type", "") != "rgbd") {
            double temp_factor = RescaleByStatitics(image_path, *align_reconstruction);
            std::cout << "RescaleByStatitics(image_path, *align_reconstruction): " << temp_factor << std::endl;
        }
        
        if (rescale_flag) {
            scale_factor *= RescaleByReconstruction(*align_reconstruction, target_reconstruction);
        } else {
            std::cout << "Skip scale pre-alignment" << std::endl;
        }
        std::cout << "scale pre-alignment: " << scale_factor << std::endl;

        std::cout << "bf add target_reconstruction: " << target_reconstruction.NumImages() << ", "
                  << target_reconstruction.NumCameras() << "\n"
                  << std::endl;

        std::map<int64_t, image_t> map_index_to_image_id;
        std::vector<std::map<int64_t, std::string>> v_map_rgbd_index_to_name;
        int64_t target_ms_per_frame = 1;
        std::unordered_map<image_t, std::pair<image_t, int>> image_alignment_cur;
        bool result_tmp = OptimizeDeltaTime(image_path, *align_reconstruction, image_alignment_cur, map_index_to_image_id,
                                        v_map_rgbd_index_to_name, target_ms_per_frame);
        
        result_tmp = result_tmp ? AddRgbdReconstruction(image_path, param, *align_reconstruction, feature_data_container,
                                                scene_graph_container, reconstruction_manager, image_alignment_cur,
                                                map_index_to_image_id, v_map_rgbd_index_to_name, target_ms_per_frame)
                        : result_tmp;
        result = result | result_tmp;
        // std::cout << "image_alignment_cur: " << i << std::endl;
        // for (auto &align : image_alignment_cur) {
        //     std::cout << " " << align.first << " " << align.second.first << " " << align.second.second << std::endl;
        // }
        std::cout << "image_alignment size bf: " << image_alignment.size() << std::endl;
        image_alignment.insert(image_alignment_cur.begin(), image_alignment_cur.end());
        std::cout << "image_alignment size ed: " << image_alignment.size() << std::endl;

        std::cout << "target_reconstruction: " << target_reconstruction.NumImages() << ", " 
        << target_reconstruction.NumCameras() << "\n" << std::endl;

        std::cout << "RescaleByStatitics: rescale_flag, scale_factor: " << rescale_flag << ", " << scale_factor << std::endl;
        if (rescale_flag) {
        for (int a = 0; a < 2; a++) {
            double temp_factor = RescaleByStatitics(image_path, image_alignment_cur, target_reconstruction);
            scale_factor *= temp_factor;
            std::cout << "\t=> temp_factor " << a << ": " << temp_factor << std::endl;
        }
        } else {
            std::cout << "Skip scale fine-alignment" << std::endl;
        }
        std::cout << "Final scale factor: " << scale_factor << std::endl;
    }
    std::cout << "image_alignment final size: " << image_alignment.size() << std::endl;
    // for (auto &align : image_alignment) {
    //     std::cout << " " << align.first << " " << align.second.first << " " << align.second.second << std::endl;
    // }

    std::cout << "target_reconstruction: " << target_reconstruction.NumImages() << ", " 
        << target_reconstruction.NumCameras() << "\n" << std::endl;
   
    // OptimizeExtraParams(target_reconstruction, image_alignment, param);
    std::cout << "old_images ids size: " << old_image_ids.size() << std::endl;
    DistrubGBA(scene_graph_container, reconstruction_manager, old_image_ids, param);
    
    {
        std::string rec_path = StringPrintf("%s/%d-trig", workspace_path.c_str(), 0);
        if (!boost::filesystem::exists(rec_path)) {
            boost::filesystem::create_directories(rec_path);
        } else {
            boost::filesystem::remove_all(rec_path);
            boost::filesystem::create_directories(rec_path);
        }

        target_reconstruction.WriteReconstruction(rec_path, true);
        target_reconstruction.OutputPriorResidualsTxt(rec_path);
        std::cout << "Save target_reconstruction (" << rec_path << ")\n" << std::endl;
    }

    // // GetDepthReprojectionError(image_path, target_reconstruction);

    Reconstruction rescaled_reconstruction;
    rescaled_reconstruction.ReadReconstruction(target_workspace_path_0, 1);
    rescaled_reconstruction.RescaleAll(scale_factor);
    rescaled_reconstruction.WriteBinary(target_workspace_path_0);
    rescaled_reconstruction.ExportMapPoints(JoinPaths(target_workspace_path_0, MAPPOINT_NAME));
    WriteAlignmentResult(target_reconstruction, image_alignment, JoinPaths(target_workspace_path_0, ALIGNMENT_POSE_NAME));
    std::cout << "rescale " << target_workspace_path_0 << std::endl;
    if (ExistsDir(JoinPaths(target_workspace_path_0, "KeyFrames"))) {
        Reconstruction rescaled_reconstruction;
        rescaled_reconstruction.ReadReconstruction(JoinPaths(target_workspace_path_0, "KeyFrames"), 1);
        rescaled_reconstruction.RescaleAll(scale_factor);
        rescaled_reconstruction.WriteBinary(JoinPaths(target_workspace_path_0, "KeyFrames"));
        WriteAlignmentResult(target_reconstruction, image_alignment, JoinPaths(target_workspace_path_0, "KeyFrames", ALIGNMENT_POSE_NAME));
        std::cout << "rescale " << JoinPaths(target_workspace_path_0, "KeyFrames") << std::endl;
    }

    if (ExistsDir(target_workspace_path_0 + "-export")){
        Reconstruction rescaled_reconstruction;
        rescaled_reconstruction.ReadReconstruction(target_workspace_path_0 + "-export", 1);
        rescaled_reconstruction.RescaleAll(scale_factor);
        rescaled_reconstruction.WriteBinary(target_workspace_path_0 + "-export");
        std::cout << "rescale " << target_workspace_path_0 + "-export" << std::endl;
    }

    std::cout << StringPrintf("RGBD Reconstruct in %.3fs", timer.ElapsedSeconds()) << std::endl;

    if (!result) {
        ExceptionHandler(RGBD_ALIGNMENT_FAILED, 
            JoinPaths(GetParentDir(workspace_path), "errors/dense"), tool_name).Dump();
        exit(RGBD_ALIGNMENT_FAILED);
    }

    return StateCode::SUCCESS;
}