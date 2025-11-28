// Copyright (c) 2019, SenseTime Group.
// All rights reserved.

#include <boost/filesystem/path.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "base/camera.h"
#include "base/image.h"
#include "base/image_reader.h"
#include "base/reconstruction_manager.h"
#include "base/pose.h"
#include "graph/correspondence_graph.h"
#include "util/misc.h"
#include "util/mat.h"
#include "util/rgbd_helper.h"

#include "container/feature_data_container.h"
#include "feature/extraction.h"
#include "feature/feature_matcher.h"
#include "feature/utils.h"
#include "feature/global_feature_extraction.h"

#include "controllers/incremental_mapper_controller.h"

#include "../Configurator_yaml.h"
#include "../camera_rig_params.h"
#include "../option_parsing.h"
#include "base/version.h"
#ifdef DO_ENCRYPT_CHECK
#include "../check.h"
#endif

#include <dirent.h>
#include <sys/stat.h>

#include "util/gps_reader.h"
#include <unordered_set>
#include "../system_io.h"
#include "util/ply.h"

#define MAX_PRECISION std::setprecision(std::numeric_limits<double>::digits10 + 1)

using namespace sensemap;

std::string configuration_file_path;

FILE *fs;

struct HybridOptions{
    bool debug_info = false;
    int child_id = -1;
    bool update_flag = false;
    bool save_flag = true;
    bool read_flag = true;
    HybridOptions(bool de = false, int ch = -1, bool up = false, bool sa = true, 
                  bool re = true):debug_info(de), child_id(ch), update_flag(up), 
                    save_flag(sa), read_flag(re){};
    void Print(){
        std::cout << "debug_info: " << debug_info << "\nchild_id: " << child_id
            << "\nupdate_flag: " << update_flag << "\nsave_flag: " << save_flag
            << "\nread_flag; " << read_flag << std::endl;
    };
}; 

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

    // bool have_matched = boost::filesystem::exists(JoinPaths(workspace_path, "/scene_graph.bin")) ||
    //     boost::filesystem::exists(JoinPaths(workspace_path, "/scene_graph.bin"));
    bool have_matched = false;

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
        if(!have_matched){
            feature_data_container.ReadImagesBinaryData(JoinPaths(workspace_path, "/features.bin"));
        } else{
            feature_data_container.ReadImagesBinaryDataWithoutDescriptor(JoinPaths(workspace_path, "/features.bin"));
        }
        if (boost::filesystem::exists(JoinPaths(workspace_path, "/local_images.bin"))) {
            feature_data_container.ReadLocalImagesBinaryData(JoinPaths(workspace_path, "/local_images.bin"));
        }
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
            feature_data_container.WriteLocalImagesBinaryData(JoinPaths(workspace_path, "/local_images.bin"));
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
        // draw matches

#if 0
        if (!boost::filesystem::exists(workspace_path + "/matches")) {
            boost::filesystem::create_directories(workspace_path + "/matches");
        }
        //std::vector<image_t> image_ids = feature_data_container.GetImageIds();
        int write_count = 0;
        for(image_t id2 = 1; id2 < image_ids.size(); id2++) {
            std::cout << "Image#" << id2 << std::endl;


            //const PanoramaIndexs & panorama_indices1 = feature_data_container.GetPanoramaIndexs(id1);
            for(image_t id1 = id2 + 1; id1 <= image_ids.size(); id1++) {

                const PanoramaIndexs & panorama_indices2 = feature_data_container.GetPanoramaIndexs(id2);

                auto matches =scene_graph.CorrespondenceGraph()->
                        FindCorrespondencesBetweenImages(id1,id2);
                if(matches.empty()){
                    continue;
                }

                std::string image_path = param.GetArgument("image_path","");

                auto image1 = feature_data_container.GetImage(id1);
                auto image2 = feature_data_container.GetImage(id2);

                std::cout<<matches.size()<<std::endl;

                std::cout<<image1.Name()<<std::endl;
                std::cout<<image2.Name()<<std::endl;


                const std::string input_image_path2 = image2.Name();
                std::string input_image_name2 = input_image_path2.substr(input_image_path2.find_last_of("/")+1);


                std::vector<cv::Mat> mats2;
                cv::Mat mat1 = cv::imread(image_path + "/" + image1.Name());

                for(int i = 0; i < 6; ++i){
                    std::string camera_name = "cam"+std::to_string(i)+"/";

                    std::string full_name2 = image_path +"/RIG/" + camera_name +input_image_name2;
                    cv::Mat mat2 = cv::imread(full_name2);
                    std::cout<<full_name2<<std::endl;
                    mats2.push_back(mat2);
                }



                for(int j=0; j < 6; ++j){
                    std::string camera_name = "cam"+std::to_string(id2)+"/";
                    std::string full_name2 = image_path +"/RIG/" + camera_name +input_image_name2;


                    cv::Mat& mat2 = mats2[j];

                    std::vector<cv::KeyPoint> keypoints_show1, keypoints_show2;
                    std::vector<cv::DMatch> matches_show;

                    int k = 0;
                    for (int m = 0; m < matches.size(); ++m){
                        auto keypoint1 =
                                feature_data_container.GetKeypoints(id1)[matches[m].point2D_idx1];

                        auto keypoint2 =
                                feature_data_container.GetKeypoints(id2)[matches[m].point2D_idx2];

                        if(!(panorama_indices2[matches[m].point2D_idx2].sub_image_id ==j)){
                            continue;
                        }

                        keypoints_show1.emplace_back(keypoint1.x, keypoint1.y,
                                                     keypoint1.ComputeScale(),
                                                     keypoint1.ComputeOrientation());

                        keypoints_show2.emplace_back(keypoint2.x, keypoint2.y,
                                                     keypoint2.ComputeScale(),
                                                     keypoint2.ComputeOrientation());
                        matches_show.emplace_back(k, k, 1);
                        k++;
                    }
                    if(matches_show.size()==0){continue; }


                    cv::Mat first_match;
                    cv::drawMatches(mat1, keypoints_show1, mat2, keypoints_show2,
                                    matches_show, first_match);
                    const std::string ouput_image_path = JoinPaths(
                            workspace_path + "/matches/",
                            std::to_string(id1) + "+" + std::to_string(id2) + "_" + std::to_string(j)+".jpg");
                    cv::imwrite(ouput_image_path, first_match);

                }


                write_count++;
            }
        }

#endif
#if 0
    //std::vector<image_t> image_ids = feature_data_container.GetImageIds();

	if (!boost::filesystem::exists(workspace_path + "/matches")) {
		boost::filesystem::create_directories(workspace_path + "/matches");
	}
    int write_count = 0;    
	for(image_t id1 = 1; id1 < image_ids.size(); id1++) {
        std::cout << "Image#" << id1 << std::endl;
        const PanoramaIndexs & panorama_indices1 = feature_data_container.GetPanoramaIndexs(id1);        
		for(image_t id2 = id1 + 1; id2 <= image_ids.size(); id2++) {

            const PanoramaIndexs & panorama_indices2 = feature_data_container.GetPanoramaIndexs(id2);      

			auto matches =scene_graph.CorrespondenceGraph()->
					FindCorrespondencesBetweenImages(id1,id2);
			if(matches.empty()){
				continue;
			}
			std::string image_path = param.GetArgument("image_path","");

			auto image1 = feature_data_container.GetImage(id1);
			auto image2 = feature_data_container.GetImage(id2);

            auto camera1 = feature_data_container.GetCamera(image1.CameraId());
            auto camera2 = feature_data_container.GetCamera(image2.CameraId());
            if (camera1.ModelId() == camera2.ModelId()){
                continue;
            }

                const std::string input_image_path1 = image1.Name();
                std::string input_image_name1 = input_image_path1.substr(input_image_path1.find("/")+1);
                std::cout<<input_image_name1<<std::endl;

			const std::string input_image_path2 = image2.Name();
            std::string input_image_name2 = input_image_path2.substr(input_image_path2.find("/")+1);
            std::cout<<input_image_name2<<std::endl;

            std::vector<cv::Mat> mats1;
            std::vector<cv::Mat> mats2;

            for(int i = 0; i<1; ++i){
                std::string camera_name = "cam"+std::to_string(i)+"/";

                // std::string full_name1 = image_path+"/"+camera_name + input_image_name1;
                // std::string full_name2 = image_path+"/"+camera_name + input_image_name2;

                std::string full_name1 = image_path+"/"+input_image_path1;
                std::string full_name2 = image_path+"/"+input_image_path2;

                cv::Mat mat1 = cv::imread(full_name1);
                cv::Mat mat2 = cv::imread(full_name2);

                mats1.push_back(mat1);
                mats2.push_back(mat2);
            }

            for(int i=0; i<1; ++i){
                std::string camera_name = "cam"+std::to_string(id1)+"_";
                // std::string full_name1 = camera_name + input_image_name1;
                int pos1 = input_image_name1.find_last_of('/');
                std::string full_name1 = input_image_name1.substr(pos1+1);

                for(int j=0; j<1; ++j){
                    std::string camera_name = "cam"+std::to_string(id2)+"_";
                    // std::string full_name2 = camera_name + input_image_name2;
                    int pos2 = input_image_name2.find_last_of('/');
                    std::string full_name2 = input_image_name2.substr(pos2+1);

                    cv::Mat& mat1 = mats1[i] ;
                    cv::Mat& mat2 = mats2[j];

                    std::vector<cv::KeyPoint> keypoints_show1, keypoints_show2;
                    std::vector<cv::DMatch> matches_show;
                    
                    int k = 0;
                    std::cout << "matches.size(): " << matches.size() << std::endl;
                    for (int m = 0; m < matches.size(); ++m){
                        auto keypoint1 =
                            feature_data_container.GetKeypoints(id1)[matches[m].point2D_idx1];
                        
                        auto keypoint2 = 
                            feature_data_container.GetKeypoints(id2)[matches[m].point2D_idx2];

                        // if(!( panorama_indices1[matches[m].point2D_idx1].sub_image_id == i && 
                        //       panorama_indices2[matches[m].point2D_idx2].sub_image_id ==j)){
                        //     continue;
                        // }

                        keypoints_show1.emplace_back(keypoint1.x, keypoint1.y,
                                                     keypoint1.ComputeScale(),
                                                     keypoint1.ComputeOrientation());
                        
                        keypoints_show2.emplace_back(keypoint2.x, keypoint2.y,
                                                     keypoint2.ComputeScale(),
                                                     keypoint2.ComputeOrientation());
                        matches_show.emplace_back(k, k, 1);
                        k++;
                    }
                    
                    std::cout << "matches_show.size(): " << matches_show.size() << std::endl;
                    if(matches_show.size()==0){continue; }
                    
                    
                    cv::Mat first_match;
                    cv::drawMatches(mat1, keypoints_show1, mat2, keypoints_show2,
                                    matches_show, first_match);
                    const std::string ouput_image_path = JoinPaths(
                        workspace_path + "/matches",
                        full_name1 + "+" + full_name2);
                    cv::imwrite(ouput_image_path, first_match);

                    // std::cout<<"keypoints_show1 size: "<<keypoints_show1.size()<<std::endl;
                    // cv::Mat first_match;
                    // cv::drawKeypoints(mat1,keypoints_show1,first_match);

                    // std::string ouput_image_path = JoinPaths(
                    // 		workspace_path + "/matches",
                    // 		image1.Name()+ "+" + image2.Name()+"_1.jpg");
                    // cv::imwrite(ouput_image_path, first_match);

                    // cv::Mat second_match;
                    // cv::drawKeypoints(mat2,keypoints_show2,second_match);

                    // ouput_image_path = JoinPaths(
                    // 		workspace_path + "/matches",
                    // 		image1.Name()+ "+" + image2.Name()+"_2.jpg");
                    // cv::imwrite(ouput_image_path, second_match);
                }
            }
        
            write_count++;
            if(write_count>=100) break;
		}
	}

#endif
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

#if 0
    if (!boost::filesystem::exists(workspace_path + "/matches")) {
        boost::filesystem::create_directories(workspace_path + "/matches");
    }
    std::vector<image_t> image_ids = feature_data_container.GetImageIds();
    int write_count = 0;
    for(image_t id2 = 1; id2 < image_ids.size(); id2++) {
        std::cout << "Image#" << id2 << std::endl;


        //const PanoramaIndexs & panorama_indices1 = feature_data_container.GetPanoramaIndexs(id1);
        for(image_t id1 = id2 + 1; id1 <= image_ids.size(); id1++) {

            const PanoramaIndexs & panorama_indices2 = feature_data_container.GetPanoramaIndexs(id2);

            auto matches =scene_graph.CorrespondenceGraph()->
                    FindCorrespondencesBetweenImages(id1,id2);
            if(matches.empty()){
                continue;
            }
            std::string image_path = param.GetArgument("image_path","");

            auto image1 = feature_data_container.GetImage(id1);
            auto image2 = feature_data_container.GetImage(id2);

            const std::string input_image_path2 = image2.Name();
            std::string input_image_name2 = input_image_path2.substr(input_image_path2.find_last_of("/")+1);


            std::vector<cv::Mat> mats2;
            cv::Mat mat1 = cv::imread(image_path + "/" + image1.Name());

            for(int i = 0; i < 6; ++i){
                std::string camera_name = "cam"+std::to_string(i)+"/";

                std::string full_name2 = image_path +"/RIG/" + camera_name +input_image_name2;
                cv::Mat mat2 = cv::imread(full_name2);
                std::cout<<full_name2<<std::endl;
                mats2.push_back(mat2);
            }



            for(int j=0; j < 6; ++j){
                std::string camera_name = "cam"+std::to_string(id2)+"/";
                std::string full_name2 = image_path +"/RIG/" + camera_name +input_image_name2;


                cv::Mat& mat2 = mats2[j];

                std::vector<cv::KeyPoint> keypoints_show1, keypoints_show2;
                std::vector<cv::DMatch> matches_show;

                int k = 0;
                for (int m = 0; m < matches.size(); ++m){
                    auto keypoint1 =
                            feature_data_container.GetKeypoints(id1)[matches[m].point2D_idx1];

                    auto keypoint2 =
                            feature_data_container.GetKeypoints(id2)[matches[m].point2D_idx2];

                    if(!(panorama_indices2[matches[m].point2D_idx2].sub_image_id ==j)){
                        continue;
                    }

                    keypoints_show1.emplace_back(keypoint1.x, keypoint1.y,
                                                 keypoint1.ComputeScale(),
                                                 keypoint1.ComputeOrientation());

                    keypoints_show2.emplace_back(keypoint2.x, keypoint2.y,
                                                 keypoint2.ComputeScale(),
                                                 keypoint2.ComputeOrientation());
                    matches_show.emplace_back(k, k, 1);
                    k++;
                }
                if(matches_show.size()==0){continue; }


                cv::Mat first_match;
                cv::drawMatches(mat1, keypoints_show1, mat2, keypoints_show2,
                                matches_show, first_match);
                const std::string ouput_image_path = JoinPaths(
                        workspace_path + "/matches/",
                        std::to_string(id1) + "+" + std::to_string(id2) + "_" + std::to_string(j)+".jpg");
                cv::imwrite(ouput_image_path, first_match);

            }


            write_count++;
        }
    }
#endif
}

void ClusterIncrementalMapperOptions(const std::shared_ptr<sensemap::SceneGraphContainer> &scene_graph_container,
                   std::shared_ptr<sensemap::ReconstructionManager> &reconstruction_manager, Configurator &param) {
    using namespace sensemap;

    PrintHeading1("Cluster Mapping");

    auto options = std::make_shared<MapperOptions>();
    options->mapper_type = MapperType::CLUSTER;
    options->cluster_mapper_options.mapper_options.outside_mapper_type = MapperType::CLUSTER;

    options->cluster_mapper_options.enable_image_label_cluster =
        static_cast<bool>(param.GetArgument("enable_image_label_cluster", 0));
    options->cluster_mapper_options.enable_pose_graph_optimization =
        static_cast<bool>(param.GetArgument("enable_pose_graph_optimization", 1));
    options->cluster_mapper_options.enable_cluster_mapper_with_coarse_label =
        static_cast<bool>(param.GetArgument("enable_cluster_mapper_with_coarse_label", 0));

    options->cluster_mapper_options.clustering_options.min_modularity_count =
        static_cast<int>(param.GetArgument("min_modularity_count", 400));
    options->cluster_mapper_options.clustering_options.max_modularity_count =
        static_cast<int>(param.GetArgument("max_modularity_count", 800));
    options->cluster_mapper_options.clustering_options.min_modularity_thres =
        static_cast<double>(param.GetArgument("min_modularity_thres", 0.3f));
    options->cluster_mapper_options.clustering_options.community_image_overlap =
        static_cast<int>(param.GetArgument("community_image_overlap", 5));
    options->cluster_mapper_options.clustering_options.community_transitivity =
        static_cast<int>(param.GetArgument("community_transitivity", 1));
    options->cluster_mapper_options.clustering_options.image_dist_seq_overlap =
        static_cast<int>(param.GetArgument("image_dist_seq_overlap", 5));
    options->cluster_mapper_options.clustering_options.image_overlap = param.GetArgument("cluster_image_overlap", 0);

    OptionParser option_parser;
    ImageReaderOptions reader_options;
    option_parser.GetImageReaderOptions(reader_options,param);
    option_parser.GetMapperOptions(options->cluster_mapper_options.mapper_options,param);

    int num_local_cameras = reader_options.num_local_cameras;

    std::string workspace_path = param.GetArgument("workspace_path", "");
    CHECK(!workspace_path.empty()) << "image path empty";

    std::string image_path = param.GetArgument("image_path", "");
    CHECK(!workspace_path.empty()) << "image path empty";

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
        options->cluster_mapper_options.mapper_options.prior_rotations = prior_rotations;
        options->cluster_mapper_options.mapper_options.prior_translations = prior_translations;
        options->cluster_mapper_options.mapper_options.have_prior_pose = true;
    }


    // use gps location prior to constrain the image
    bool use_gps_prior = static_cast<bool>(param.GetArgument("use_gps_prior", 0));
    bool use_prior_align_only = param.GetArgument("use_prior_align_only", 1);
    std::string gps_prior_file = param.GetArgument("gps_prior_file","");
    std::string gps_trans_file = workspace_path + "/gps_trans.txt";

    if (use_gps_prior){
        if (boost::filesystem::exists(gps_prior_file)){
            std::vector<image_t> image_ids = scene_graph_container->GetImageIds();
            std::vector<std::string> image_names;
            for (const auto image_id : image_ids) {
                const Image &image = scene_graph_container->Image(image_id);
                std::string name = image.Name();
                image_names.push_back(name);
            }

            std::vector<std::pair<unsigned long, std::pair<Eigen::Vector3d,int>>> gps_locations;
            if(options->cluster_mapper_options.mapper_options.optimization_use_horizontal_gps_only){
                LoadOriginGPSinfo(gps_prior_file, gps_locations,gps_trans_file,true);
            }
            else{
                LoadOriginGPSinfo(gps_prior_file, gps_locations,gps_trans_file, false);
            }
            std::unordered_map<std::string, std::pair<Eigen::Vector3d,int>> image_locations;
            GPSLocationsToImages(gps_locations, image_names, image_locations);

            std::unordered_map<image_t, std::pair<Eigen::Vector3d,int>> prior_locations_gps;
            for (const auto image_id : image_ids) {
                const Image &image = scene_graph_container->Image(image_id);
                std::string name = image.Name();

                if(image_locations.find(name)!=image_locations.end()){
                    prior_locations_gps.emplace(image_id,image_locations.at(name));
                }
            }
            options->cluster_mapper_options.mapper_options.prior_locations_gps = prior_locations_gps;
        }

        options->cluster_mapper_options.mapper_options.has_gps_prior = true;
        options->cluster_mapper_options.mapper_options.use_prior_align_only = use_prior_align_only;
        options->cluster_mapper_options.mapper_options.min_image_num_for_gps_error =
            param.GetArgument("min_image_num_for_gps_error", 10);

        double prior_absolute_location_weight = 
            static_cast<double>(param.GetArgument("prior_absolute_location_weight", 1.0f));
       options->cluster_mapper_options.mapper_options.prior_absolute_location_weight = prior_absolute_location_weight;
    }
     

    MapperController *mapper = MapperController::Create(options, workspace_path, image_path,
                                                        scene_graph_container, reconstruction_manager);
    mapper->Start();
    mapper->Wait();

    fprintf(fs, "%s\n",
            StringPrintf("Cluster Mapper Elapsed time: %.3f [minutes]", mapper->GetTimer().ElapsedMinutes()).c_str());
    fflush(fs);

    std::cout<<"Reconstruction Component Size: "<< reconstruction_manager->Size()<<std::endl;
    for (size_t i = 0; i < reconstruction_manager->Size(); ++i) {
        auto reconstruction = reconstruction_manager->Get(i);
        CHECK(reconstruction->RegisterImageIds().size()>0);
        const image_t first_image_id = reconstruction->RegisterImageIds()[0]; 
        CHECK(reconstruction->ExistsImage(first_image_id));
        const Image& image = reconstruction->Image(first_image_id);
        const Camera& camera = reconstruction->Camera(image.CameraId());
        
        std::string rec_path = StringPrintf("%s/%d", workspace_path.c_str(), i);
        if (!boost::filesystem::exists(rec_path)) {
            boost::filesystem::create_directories(rec_path);
        }

        if (use_gps_prior) {
            // Eigen::Matrix3x4d matrix_to_align =
            reconstruction->AlignWithPriorLocations(options->cluster_mapper_options.mapper_options.max_error_gps);
            if(camera.NumLocalCameras() == 1){
                Reconstruction rec = *reconstruction.get();
                rec.AddPriorToResult();
                // rec.NormalizeWoScale();
                
                std::string trans_rec_path = rec_path + "-gps";
                if (!boost::filesystem::exists(trans_rec_path)) {
                    boost::filesystem::create_directories(trans_rec_path);
                }
                rec.WriteBinary(trans_rec_path);
            }
            // Eigen::Matrix3x4d matrix_to_geo = reconstruction->NormalizeWoScale();
            // Eigen::Matrix4d h_matrix_to_align = Eigen::Matrix4d::Identity();
            // h_matrix_to_align.block<3, 4>(0, 0) = matrix_to_align;
            // Eigen::Matrix3x4d M = matrix_to_geo * h_matrix_to_align;
            
            // std::ofstream file((rec_path + "/matrix_to_gps.txt"), std::ofstream::out);
            // file << MAX_PRECISION << M(0, 0) << " " << M(0, 1) << " " 
            //      << M(0, 2) << " " << M(0, 3) << std::endl;
            // file << MAX_PRECISION << M(1, 0) << " " << M(1, 1) << " " 
            //      << M(1, 2) << " " << M(1, 3) << std::endl;
            // file << MAX_PRECISION << M(2, 0) << " " << M(2, 1) << " " 
            //      << M(2, 2) << " " << M(2, 3) << std::endl;
            // file.close();
        }

        if (camera.NumLocalCameras() > 1) {
            reconstruction->FilterAllMapPoints(2, 4.0, 1.5);
            reconstruction->WriteReconstruction(rec_path,
                options->cluster_mapper_options.mapper_options.write_binary_model);

            std::string export_rec_path = rec_path + "-export";
            if (!boost::filesystem::exists(export_rec_path)) {
                boost::filesystem::create_directories(export_rec_path);
            }
            Reconstruction rig_reconstruction;
            reconstruction->ConvertRigReconstruction(rig_reconstruction);
            rig_reconstruction.WriteReconstruction(export_rec_path,
                options->cluster_mapper_options.mapper_options.write_binary_model);
        } else {
            reconstruction->WriteReconstruction(rec_path,
                options->cluster_mapper_options.mapper_options.write_binary_model);
        }
    }
}


bool FreeGBA(Reconstruction& reconstruction_, const BundleAdjustmentOptions& ba_options){


    const std::vector<image_t>& reg_image_ids = reconstruction_.RegisterImageIds();

    CHECK_GE(reg_image_ids.size(), 2) << "At least two images must be "
                                         "registered for global "
                                         "bundle-adjustment";


    // Avoid degeneracies in bundle adjustment.
    reconstruction_.FilterObservationsWithNegativeDepth();

    // Configure bundle adjustment.
    BundleAdjustmentConfig ba_config;
    for (const image_t image_id : reg_image_ids) {
        ba_config.AddImage(image_id);
        const Image& image = reconstruction_.Image(image_id);
        const Camera& camera = reconstruction_.Camera(image.CameraId());
        if (camera.IsCameraConstant()) {
            ba_config.SetConstantCamera(image.CameraId());
        }
    }
    CHECK(reconstruction_.RegisterImageIds().size() > 0);
    const image_t first_image_id = reconstruction_.RegisterImageIds()[0];
    CHECK(reconstruction_.ExistsImage(first_image_id));
    const Image& image = reconstruction_.Image(first_image_id);
    const Camera& camera = reconstruction_.Camera(image.CameraId());

    if (!ba_options.use_prior_absolute_location || !reconstruction_.b_aligned) {
        ba_config.SetConstantPose(reg_image_ids[0]);
        if (camera.NumLocalCameras() == 1) {
            ba_config.SetConstantTvec(reg_image_ids[1], {0});
        } 
        else {
            // ba_config.SetConstantPose(reg_image_ids[1]);
            ba_config.SetConstantTvec(reg_image_ids[1], {0});
        }
    }
    
    // Run bundle adjustment.
    BundleAdjuster bundle_adjuster(ba_options, ba_config);

    if (!bundle_adjuster.Solve(&reconstruction_)) {
        return false;
    }
   
    return true;
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
    option_parser.GetMapperOptions(options->independent_mapper_options,param,hybrid_options.child_id);

    std::string workspace_path = param.GetArgument("workspace_path", "");
    CHECK(!workspace_path.empty()) << "image path empty";

    std::string image_path = param.GetArgument("image_path", "");
    CHECK(!workspace_path.empty()) << "image path empty";

    if (hybrid_options.child_id > -1 && hybrid_options.debug_info){
        Configurator camera_param;
        std::string cameras_param_file = param.GetArgument("camera_param_file", "");
        camera_param.Load(cameras_param_file.c_str());
        std::string child_name = camera_param.GetArgument("sub_path_" + std::to_string(hybrid_options.child_id), "");
        CHECK(!child_name.empty());

        workspace_path = JoinPaths(workspace_path, child_name);
        CHECK(boost::filesystem::exists(workspace_path));
    }

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
    size_t base_reconstruction_idx = reconstruction_manager->Size();
    std::vector<int> base_reconstruction_ids = reconstruction_manager->getReconstructionIds();

    MapperController *mapper = 
        MapperController::Create(options, workspace_path, image_path, scene_graph_container, reconstruction_manager);
    mapper->Start();
    mapper->Wait();

    fprintf(
        fs, "%s\n",
        StringPrintf("Incremental Mapper Elapsed time: %.3f [minutes]", mapper->GetTimer().ElapsedMinutes()).c_str());
    fflush(fs);

    std::vector<int> reconstruction_ids = reconstruction_manager->getReconstructionIds();
    std::vector<int> remain_reconstruction_ids;
    std::set_difference(reconstruction_ids.begin(), reconstruction_ids.end(), 
                        base_reconstruction_ids.begin(), base_reconstruction_ids.end(), std::back_inserter(remain_reconstruction_ids));
    for (size_t i = 0; i < remain_reconstruction_ids.size(); ++i) {
        auto reconstruction = reconstruction_manager->Get(remain_reconstruction_ids[i]);
        CHECK(reconstruction->RegisterImageIds().size()>0);
        bool camera_rig = false;
        const auto& camera_ids = reconstruction->Cameras();
        for (auto camera : camera_ids){
            if (camera.second.NumLocalCameras() > 1){
                camera_rig = true;
            }
        }
        
        std::string rec_path = StringPrintf("%s/%d", workspace_path.c_str(), base_reconstruction_idx + i);
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

    // Check the reconstruction file exisit or not
    if (boost::filesystem::is_directory(workspace_path + "/0")) {
        if ((dirExists(workspace_path + "/0/cameras.txt") && dirExists(workspace_path + "/0/images.txt") &&
             dirExists(workspace_path + "/0/points3D.txt")) ||
            (dirExists(workspace_path + "/0/cameras.bin") && dirExists(workspace_path + "/0/images.bin") &&
             dirExists(workspace_path + "/0/points3D.bin"))) {
            reconstruction = std::make_shared<Reconstruction>();
            reconstruction->ReadReconstruction(workspace_path + "/0",false);
            return;
        }
    }
    // // Get all the reconstruction file
    // std::vector<std::string> file_list = sensemap::GetDirList(workspace_path);

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

            if (mapper_options.extract_colors) {
                reconstruction->ExtractColorsForImage(next_image_id, image_path);
            }
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
            if (mapper_options.extract_colors) {
                reconstruction->ExtractColorsForImage(image_id, image_path);
            }

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
        std::string reconstruction_path = workspace_path + "/0/";
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

////////////////////////////////////////////////////////////////////////////////
int main(int argc, char *argv[]) {
    using namespace sensemap;

    PrintHeading("SenseMap.  Copyright(c) 2019, SenseTime Group.");
    PrintHeading(std::string("Version: hd-sfm-")+__VERSION__);
    
    Timer timer;
    timer.Start();

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

    fs = fopen((workspace_path + "/time.txt").c_str(), "w");

    auto reconstruction_manager = std::make_shared<ReconstructionManager>();
    auto scene_graph_container = std::make_shared<SceneGraphContainer>();
    auto feature_data_container = std::make_shared<FeatureDataContainer>();

    bool debug_info = static_cast<bool>(param.GetArgument("debug_info", 0));
    int register_sequential = static_cast<int>(param.GetArgument("register_sequential", 0));
    int num_cameras = static_cast<int>(param.GetArgument("num_cameras", -1));
    std::string cameras_param_file = param.GetArgument("camera_param_file", "");
    std::cout << "register_sequential, num_cameras, cameras_param_file: " << register_sequential << ", " << num_cameras << ", " << cameras_param_file << std::endl;

    if (num_cameras <= 1 || cameras_param_file.empty()){
        FeatureExtraction(*feature_data_container.get(), param);

        typedef FeatureMatchingOptions::RetrieveType RetrieveType;
        RetrieveType retrieve_type = (RetrieveType)param.GetArgument("retrieve_type", 0);
        if (retrieve_type == RetrieveType::VLAD) {
            GlobalFeatureExtraction(*feature_data_container.get(), param);
        }
        FeatureMatching(*feature_data_container.get(), *scene_graph_container.get(), param);
        
        feature_data_container.reset();

        std::string mapper_method = param.GetArgument("mapper_method", "incremental");

        if (mapper_method.compare("incremental") == 0) {
            IncrementalSFM(scene_graph_container, reconstruction_manager, param);
        } else if (mapper_method.compare("cluster") == 0) {
            ClusterIncrementalMapperOptions(scene_graph_container, reconstruction_manager, param);
        }
    } else if (!cameras_param_file.empty()) {
        struct HybridOptions hybrid_options = HybridOptions(debug_info);
        if (register_sequential == 0){
            for (int idx = 0; idx < num_cameras; idx++){
                hybrid_options.child_id = idx;
                if (idx == num_cameras-1){
                    hybrid_options.save_flag = true;
                }
                FeatureExtraction(*feature_data_container.get(), param, hybrid_options);
            }

            typedef FeatureMatchingOptions::RetrieveType RetrieveType;
            RetrieveType retrieve_type = (RetrieveType)param.GetArgument("retrieve_type", 0);
            if (retrieve_type == RetrieveType::VLAD) {
                GlobalFeatureExtraction(*feature_data_container.get(), param);
            }

            FeatureMatching(*feature_data_container.get(), *scene_graph_container.get(), param);
            IncrementalSFM(scene_graph_container, reconstruction_manager, param);
        } else if (register_sequential == 2) {
            hybrid_options.child_id = 0;
            if (!hybrid_options.debug_info){
                hybrid_options.save_flag = false;
                hybrid_options.read_flag = false;
            }
            FeatureExtraction(*feature_data_container.get(), param, hybrid_options);
            
            typedef FeatureMatchingOptions::RetrieveType RetrieveType;
            RetrieveType retrieve_type = (RetrieveType)param.GetArgument("retrieve_type", 0);
            if (retrieve_type == RetrieveType::VLAD) {
                GlobalFeatureExtraction(*feature_data_container.get(), param, hybrid_options);
            }

            FeatureMatching(*feature_data_container.get(), *scene_graph_container.get(), param, hybrid_options);            

            for (int idx = 1; idx < num_cameras; idx++){
                hybrid_options.child_id = idx;
                hybrid_options.update_flag = true;
                
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
            }

            IncrementalSFM(scene_graph_container, reconstruction_manager, param, hybrid_options);
        } else {
            hybrid_options.child_id = 0;
            if (!hybrid_options.debug_info){
                hybrid_options.save_flag = false;
                hybrid_options.read_flag = false;
            }
            FeatureExtraction(*feature_data_container.get(), param, hybrid_options);
            
            typedef FeatureMatchingOptions::RetrieveType RetrieveType;
            RetrieveType retrieve_type = (RetrieveType)param.GetArgument("retrieve_type", 0);
            if (retrieve_type == RetrieveType::VLAD) {
                GlobalFeatureExtraction(*feature_data_container.get(), param, hybrid_options);
            }

            FeatureMatching(*feature_data_container.get(), *scene_graph_container.get(), param, hybrid_options);            
            IncrementalSFM(scene_graph_container, reconstruction_manager, param, hybrid_options);

            for (int idx = 1; idx < num_cameras; idx++){
                hybrid_options.child_id = idx;
                hybrid_options.update_flag = true;
                
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
    }

    std::cout << StringPrintf("Reconstruct in %.3fs", timer.ElapsedSeconds()) << std::endl;

    PrintReconSummary(workspace_path + "/statistic.txt", scene_graph_container->NumImages(), reconstruction_manager);

    fclose(fs);

    return 0;
}