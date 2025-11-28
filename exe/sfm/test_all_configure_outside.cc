// Copyright (c) 2019, SenseTime Group.
// All rights reserved.

#include <boost/filesystem/path.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "base/common.h"
#include "base/camera.h"
#include "base/image.h"
#include "base/image_reader.h"
#include "base/reconstruction_manager.h"
#include "base/pose.h"
#include "graph/correspondence_graph.h"
#include "util/misc.h"
#include "util/mat.h"
#include "util/rgbd_helper.h"
#include "util/tag_scale_recover.h"

#include "graph/maximum_spanning_tree_graph.h"
#include "container/feature_data_container.h"
#include "feature/extraction.h"
#include "feature/feature_matcher.h"
#include "feature/utils.h"
#include "feature/global_feature_extraction.h"
#include "controllers/cluster_mapper_controller.h"
#include "controllers/directed_mapper_controller.h"

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

bool must_increamental;

FILE *fs;

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

void FeatureExtraction(FeatureDataContainer &feature_data_container, Configurator &param) {
    std::string workspace_path = param.GetArgument("workspace_path", "");
    CHECK(!workspace_path.empty());

    OptionParser option_parser;
    ImageReaderOptions reader_options;
    option_parser.GetImageReaderOptions(reader_options,param);

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
        if(!have_matched){
            feature_data_container.ReadImagesBinaryData(JoinPaths(workspace_path, "/features.bin"));
        }
        else{
            feature_data_container.ReadImagesBinaryDataWithoutDescriptor(JoinPaths(workspace_path, "/features.bin"));
        }
        if (boost::filesystem::exists(JoinPaths(workspace_path, "/local_images.bin"))) {
            feature_data_container.ReadLocalImagesBinaryData(JoinPaths(workspace_path, "/local_images.bin"));
        }
        exist_feature_file = true;
    } else if (boost::filesystem::exists(JoinPaths(workspace_path, "/cameras.txt")) &&
               boost::filesystem::exists(JoinPaths(workspace_path, "/features.txt"))) {
        if (boost::filesystem::exists(JoinPaths(workspace_path, "/local_cameras.txt"))) {
            feature_data_container.ReadCameras(JoinPaths(workspace_path, "/cameras.txt"), false);
            feature_data_container.ReadLocalCameras(JoinPaths(workspace_path, "/local_cameras.txt"));
        } else {
            feature_data_container.ReadCameras(JoinPaths(workspace_path, "/cameras.txt"), camera_rig);
        }
        feature_data_container.ReadImagesData(JoinPaths(workspace_path, "/features.txt"));
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
            feature_data_container.GetCamera(camera_id).SetCameraConstant(reader_options.fixed_camera);
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

    typedef FeatureMatchingOptions::RetrieveType RetrieveType;
    RetrieveType retrieve_type = (RetrieveType)param.GetArgument("retrieve_type", 0);
    if (retrieve_type != RetrieveType::SIFT) {
        // Pca training using the extrated feature descriptors
        Timer timer;
        timer.Start();

        std::cout << "Collect training descriptors " << std::endl;
        FeatureDescriptors training_descriptors;
        size_t training_descriptors_count = 0;

        const std::vector<image_t>& image_ids = feature_data_container.GetImageIds();

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

        // typedef FeatureMatchingOptions::RetrieveType RetrieveType;
        // RetrieveType retrieve_type = (RetrieveType)param.GetArgument("retrieve_type", 0);
        // if (retrieve_type == RetrieveType::SIFT && compressed_feature_dimension != 128) {
        //     compressed_feature_dimension = 128;
        //     std::cout << StringPrintf("Warning! Feature dimension is not customized for SIFT\n");
        // }

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
            descriptors.resize(0, 0);
        }
        std::cout << StringPrintf("Compressing descriptors in %.3f min", timer.ElapsedMinutes()) << std::endl;

        fprintf(fs, "%s\n", StringPrintf("Compressing descriptors in %.3f [minutes]", timer.ElapsedMinutes()).c_str());
        fflush(fs);
    } else {
        const std::vector<image_t>& image_ids = feature_data_container.GetImageIds();
        for (int i = 0; i < image_ids.size(); ++i) {
            image_t current_id = image_ids[i];
            auto &descriptors = feature_data_container.GetDescriptors(current_id);
            auto &compressed_descriptors = feature_data_container.GetCompressedDescriptors(current_id);
            std::swap(descriptors, compressed_descriptors);
        }
    }

    bool write_feature = static_cast<bool>(param.GetArgument("write_feature", 1));
    bool write_binary = static_cast<bool>(param.GetArgument("write_binary", 1));
    if (write_feature) {
        if (write_binary) {
            feature_data_container.WriteImagesBinaryData(JoinPaths(workspace_path, "/features.bin"));
            feature_data_container.WriteLocalImagesBinaryData(JoinPaths(workspace_path, "/local_images.bin"));
            feature_data_container.WriteCamerasBinaryData(JoinPaths(workspace_path, "/cameras.bin"));
            feature_data_container.WriteLocalCamerasBinaryData(JoinPaths(workspace_path, "/local_cameras.bin"));
            feature_data_container.WriteSubPanoramaBinaryData(JoinPaths(workspace_path, "/sub_panorama.bin"));

            if (reader_options.num_local_cameras == 2 && reader_options.camera_model == "OPENCV_FISHEYE" &&
                sift_extraction.convert_to_perspective_image) {
                feature_data_container.WritePieceIndicesBinaryData(JoinPaths(workspace_path, "/piece_indices.bin"));
            }
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
            std::ofstream file1(JoinPaths(workspace_path, "/gps_origin.txt"));
            file1 << MAX_PRECISION << latitude << " " << longitude << " " << altitude << std::endl;
            file1.close();
            std::cout << "Save gps info to ned_to_ecef.txt & gps_origin.txt" << std::endl;
        }
    }
}

void GlobalFeatureExtraction(FeatureDataContainer &feature_data_container, Configurator &param){

    std::string workspace_path = param.GetArgument("workspace_path", "");
    CHECK(!workspace_path.empty());

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
                     Configurator &param) {
    using namespace std::chrono;
    

    OptionParser option_parser;
    ImageReaderOptions reader_options;
    option_parser.GetImageReaderOptions(reader_options,param);


    std::string workspace_path = param.GetArgument("workspace_path", "");
    CHECK(!workspace_path.empty());
    bool load_scene_graph = false;
    if (boost::filesystem::exists(JoinPaths(workspace_path, "/scene_graph.bin"))) {
        scene_graph.ReadSceneGraphBinaryData(JoinPaths(workspace_path, "/scene_graph.bin"));
        if (boost::filesystem::exists(JoinPaths(workspace_path, "/two_view_geometry.bin"))) {
            scene_graph.ReadImagePairsBinaryData(JoinPaths(workspace_path, "/two_view_geometry.bin"));
        }
        load_scene_graph = true;
    } else if (boost::filesystem::exists(JoinPaths(workspace_path, "/scene_graph.txt"))) {
        scene_graph.ReadSceneGraphData(JoinPaths(workspace_path, "/scene_graph.txt"));
        if (boost::filesystem::exists(JoinPaths(workspace_path, "/two_view_geometry.bin"))) {
            scene_graph.ReadImagePairsBinaryData(JoinPaths(workspace_path, "/two_view_geometry.bin"));
        }
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
    {
        if (!boost::filesystem::exists(workspace_path + "/matches")) {
            boost::filesystem::create_directories(workspace_path + "/matches");
        }
        int write_count = 0;
        std::vector<image_t> image_ids = feature_data_container.GetImageIds();
        for(image_t id1 = 0; id1 < image_ids.size(); id1++) {
            image_t image_id1 = image_ids.at(id1);
            std::cout << "Image#" << image_id1 << std::endl;
            const PanoramaIndexs & panorama_indices1 = feature_data_container.GetPanoramaIndexs(image_id1);        
            for(image_t id2 = id1; id2 < image_ids.size(); id2++) {
                // if (id1 != id2) {
                //     continue;
                // }
                image_t image_id2 = image_ids.at(id2);

                const PanoramaIndexs & panorama_indices2 = feature_data_container.GetPanoramaIndexs(image_id2);      

                auto matches =scene_graph.CorrespondenceGraph()->
                        FindCorrespondencesBetweenImages(image_id1, image_id2);
                if(matches.empty()){
                    continue;
                }
                std::string image_path = param.GetArgument("image_path","");

                auto image1 = feature_data_container.GetImage(image_id1);
                auto image2 = feature_data_container.GetImage(image_id2);

                std::string input_image_path1 = image1.Name();
                std::cout<<input_image_path1<<std::endl;

                std::string input_image_path2 = image2.Name();
                std::cout<<input_image_path2<<std::endl;

                std::vector<cv::Mat> mats1;
                std::vector<cv::Mat> mats2;

                const int num_local_camera = 1;
                std::vector<std::string> local_image_names1;
                std::vector<std::string> local_image_names2;
                for(int i = 0; i < num_local_camera; ++i){
                    size_t pos1 = input_image_path1.find("cam0");
                    std::string local_image_path1 = input_image_path1;
                    if (pos1 != std::string::npos) {
                        local_image_path1.replace(pos1, 4, "cam" + std::to_string(i));
                    }
                    local_image_names1.push_back(local_image_path1);

                    size_t pos2 = input_image_path2.find("cam0");
                    std::string local_image_path2 = input_image_path2;
                    if (pos2 != std::string::npos) {
                        local_image_path2.replace(pos2, 4, "cam" + std::to_string(i));
                    }
                    local_image_names2.push_back(local_image_path2);

                    std::string full_name1, full_name2;
                    full_name1 = image_path + "/" + local_image_path1;
                    full_name2 = image_path + "/" + local_image_path2;

                    // std::cout << "+--------------------------------+" << std::endl;
                    // std::cout << full_name1 << std::endl;
                    // std::cout << full_name2 << std::endl;

                    cv::Mat mat1 = cv::imread(full_name1);
                    cv::Mat mat2 = cv::imread(full_name2);

                    mats1.push_back(mat1);
                    mats2.push_back(mat2);
                }

                for(int i = 0; i < num_local_camera; ++i){
                    for(int j = 0; j < num_local_camera; ++j){
                        auto image_name1 = local_image_names1.at(i);
                        auto image_name2 = local_image_names2.at(j);

                        cv::Mat& mat1 = mats1[i];
                        cv::Mat& mat2 = mats2[j];

                        std::vector<cv::KeyPoint> keypoints_show1, keypoints_show2;
                        std::vector<cv::DMatch> matches_show;
                        
                        int k = 0;
                        for (int m = 0; m < matches.size(); ++m){
                            auto& keypoints1 = feature_data_container.GetKeypoints(image_id1);
                            CHECK_LT(matches[m].point2D_idx1, keypoints1.size());
                            auto keypoint1 =keypoints1[matches[m].point2D_idx1];
                            
                            auto keypoints2 = feature_data_container.GetKeypoints(image_id2);
                            CHECK_LT(matches[m].point2D_idx2, keypoints2.size());
                            auto keypoint2 = keypoints2[matches[m].point2D_idx2];

                            CHECK_LT(matches[m].point2D_idx1, panorama_indices1.size());
                            CHECK_LT(matches[m].point2D_idx2, panorama_indices2.size());
                            if(panorama_indices1[matches[m].point2D_idx1].sub_image_id != i || 
                                panorama_indices2[matches[m].point2D_idx2].sub_image_id != j){
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
                        cv::drawMatches(mat1, keypoints_show1, mat2,
                            keypoints_show2, matches_show, first_match);
                        while(1) {
                            size_t pos = image_name1.find("/");
                            if (pos == std::string::npos) {
                                break;
                            }
                            image_name1.replace(pos, 1, "-");
                        }
                        while(1) {
                            size_t pos = image_name2.find("/");
                            if (pos == std::string::npos) {
                                break;
                            }
                            image_name2.replace(pos, 1, "-");
                        }
                        const std::string ouput_image_path = JoinPaths(
                            workspace_path + "/matches",
                            image_name1 + "+" + image_name2);
                        std::string parent_path = GetParentDir(ouput_image_path);
                        std::cout << ouput_image_path << std::endl;
                        boost::filesystem::create_directories(parent_path);
                        cv::imwrite(ouput_image_path, first_match);
                    }
                }
            
                // write_count++;
                // if(write_count>=100) break;
            }
        }
    }
#endif
        return;
    }

    FeatureMatchingOptions options;
    option_parser.GetFeatureMatchingOptions(options,param);

    if (options.method_ == FeatureMatchingOptions::MatchMethod::SPATIAL) {
        size_t num_image_has_prior = 0;
        auto image_ids = feature_data_container.GetImageIds();
        for (auto image_id : image_ids) {
            Image image = feature_data_container.GetImage(image_id);
            num_image_has_prior += !!image.HasTvecPrior();
        }
        float prior_ratio = num_image_has_prior * 1.0 / image_ids.size();
        if (prior_ratio < 0.999) {
            options.method_ = FeatureMatchingOptions::MatchMethod::VOCABTREE;
            std::cout << StringPrintf("The number of image that has prior pose is %f(<0.999), switch to vocabtree!\n");
        }
    }

    // use gps to filter far image pairs.
    bool use_gps_prior = static_cast<bool>(param.GetArgument("use_gps_prior", 0));
    if (use_gps_prior) {
        std::string trans_path = JoinPaths(workspace_path, "ned_to_ecef.txt");
        if (boost::filesystem::exists(trans_path)) {
            std::ifstream fin(trans_path, std::ifstream::in);
            if (fin.is_open()) {
                Eigen::Matrix3x4d trans;
                fin >> trans(0, 0) >> trans(0, 1) >> trans(0, 2) >> trans(0, 3);
                fin >> trans(1, 0) >> trans(1, 1) >> trans(1, 2) >> trans(1, 3);
                fin >> trans(2, 0) >> trans(2, 1) >> trans(2, 2) >> trans(2, 3);
                options.ned_to_ecef_matrix_ = trans;
            }
            fin.close();
        }
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
    high_resolution_clock::time_point start_time = high_resolution_clock::now();

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

    bool write_match = static_cast<bool>(param.GetArgument("write_match", 1));
    bool write_binary = static_cast<bool>(param.GetArgument("write_binary", 1));

    if (write_match) {
        if (write_binary) {
            scene_graph.WriteSceneGraphBinaryData(workspace_path + "/scene_graph.bin");
        } else {
            scene_graph.WriteSceneGraphData(workspace_path + "/scene_graph.txt");
        }
        // scene_graph.WriteImagePairsBinaryData(workspace_path + "/two_view_geometry.bin");
    }
}

void FeatureExtractionAndMatching(SceneGraphContainer &scene_graph_container,
                                  Configurator &param) {
    auto feature_data_container = std::make_shared<FeatureDataContainer>();
    FeatureExtraction(*feature_data_container.get(), param);

    typedef FeatureMatchingOptions::RetrieveType RetrieveType;
    RetrieveType retrieve_type = (RetrieveType)param.GetArgument("retrieve_type", 0);
    if (retrieve_type == RetrieveType::VLAD) {
        GlobalFeatureExtraction(*feature_data_container.get(), param);
    }

    FeatureMatching(*feature_data_container.get(), scene_graph_container, param);
}

void ClusterMapper(const std::shared_ptr<sensemap::SceneGraphContainer> &scene_graph_container,
                   std::shared_ptr<sensemap::ReconstructionManager> &reconstruction_manager, Configurator &param) {
    using namespace sensemap;

    PrintHeading1("Cluster Mapping");

    auto options = std::make_shared<MapperOptions>();
    options->mapper_type = MapperType::CLUSTER;
    options->cluster_mapper_options.mapper_options.outside_mapper_type = MapperType::CLUSTER;
    
    auto camera_model = param.GetArgument("camera_model", "OPENCV");
    if((camera_model == "PARTIAL_OPENCV" || camera_model == "OPENCV") && !must_increamental) {
        options->cluster_mapper_options.mapper_options.independent_mapper_type = IndependentMapperType::DIRECTED;
        std::cout << "independent_mapper_type = IndependentMapperType::DIRECTED" << std::endl;
    }

    options->cluster_mapper_options.multiple_models = 
        static_cast<bool>(param.GetArgument("multiple_models", 0));

    options->cluster_mapper_options.enable_image_label_cluster =
        static_cast<bool>(param.GetArgument("enable_image_label_cluster", 1));
    options->cluster_mapper_options.enable_pose_graph_optimization =
        static_cast<bool>(param.GetArgument("enable_pose_graph_optimization", 1));
    options->cluster_mapper_options.enable_cluster_mapper_with_coarse_label =
        static_cast<bool>(param.GetArgument("enable_cluster_mapper_with_coarse_label", 0));
    
    // options->cluster_mapper_options.clustering_options.min_modularity_count =
    //     static_cast<int>(param.GetArgument("min_modularity_count", 400));
    // options->cluster_mapper_options.clustering_options.max_modularity_count =
    //     static_cast<int>(param.GetArgument("max_modularity_count", 800));
    int max_modularity_count = static_cast<int>(param.GetArgument("max_modularity_count", -1));

    options->cluster_mapper_options.clustering_options.min_modularity_thres =
        static_cast<double>(param.GetArgument("min_modularity_thres", 0.3f));
    options->cluster_mapper_options.clustering_options.community_image_overlap =
        static_cast<int>(param.GetArgument("community_image_overlap", 0));
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


    options->cluster_mapper_options.only_merge_cluster = param.GetArgument("only_merge_cluster", 0);
    std::cout << "reconstruction_manager: " << reconstruction_manager->Size() << ", " 
              << options->cluster_mapper_options.only_merge_cluster << std::endl;
#if 0
    if (options->cluster_mapper_options.only_merge_cluster){
        for (int rect_idx = 1; ; rect_idx++){
            auto reconstruction_path = JoinPaths(workspace_path, "cluster" + std::to_string(rect_idx), "0");
            if (!ExistsDir(reconstruction_path)){
                break;
            }
            auto rect_id = reconstruction_manager->Add();
            auto rect = reconstruction_manager->Get(rect_id);
            rect->ReadReconstruction(reconstruction_path);

            bool in_scenegraph = true;
            const auto image_ids = rect->Images();
            for (const auto id : image_ids){
                if (!scene_graph_container->ExistsImage(id.first)){
                    in_scenegraph = false;
                    break;
                }
            }
            if (!in_scenegraph){
                reconstruction_manager->Delete(rect_id);
                std::cout << "reconstruction_manager skip " << rect_idx << std::endl;
            }
            std::cout << "Read reconstruction-" << rect_id << std::endl;
        }
    }
    std::cout << "reconstruction_manager: " << reconstruction_manager->Size() << std::endl;
#endif

    if (max_modularity_count < 0){
        if (num_local_cameras > 1){
            options->cluster_mapper_options.clustering_options.max_modularity_count = 4000;
            options->cluster_mapper_options.clustering_options.min_modularity_count = 2000;
            options->cluster_mapper_options.clustering_options.leaf_max_num_images = 200;
        } else {
            options->cluster_mapper_options.clustering_options.max_modularity_count = 10000;
            options->cluster_mapper_options.clustering_options.min_modularity_count = 5000;
            options->cluster_mapper_options.clustering_options.leaf_max_num_images = 500;
        }
    } else {
        options->cluster_mapper_options.clustering_options.max_modularity_count = max_modularity_count;
        options->cluster_mapper_options.clustering_options.min_modularity_count = max_modularity_count / 2;
        options->cluster_mapper_options.clustering_options.leaf_max_num_images = max_modularity_count / 20;
    }

    size_t base_reconstruction_idx = reconstruction_manager->Size();
    std::vector<int> base_reconstruction_ids = reconstruction_manager->getReconstructionIds();

    MapperController *mapper = MapperController::Create(options, workspace_path, image_path,
                                                        scene_graph_container, reconstruction_manager);
    mapper->Start();
    mapper->Wait();

    fprintf(fs, "%s\n",
            StringPrintf("Cluster Mapper Elapsed time: %.3f [minutes]", mapper->GetTimer().ElapsedMinutes()).c_str());
    fflush(fs);

    bool use_apriltag = static_cast<bool>(param.GetArgument("detect_apriltag", 0));
    double apriltag_size = param.GetArgument("apriltag_size", 0.113f);
    bool color_harmonization = param.GetArgument("color_harmonization", 0);

    std::vector<int> reconstruction_ids = reconstruction_manager->getReconstructionIds();
    std::vector<int> remain_reconstruction_ids;
    std::set_difference(reconstruction_ids.begin(), reconstruction_ids.end(), 
                        base_reconstruction_ids.begin(), base_reconstruction_ids.end(), std::back_inserter(remain_reconstruction_ids));

    for (size_t i = 0; i < remain_reconstruction_ids.size(); ++i) {
        auto reconstruction = reconstruction_manager->Get(remain_reconstruction_ids[i]);
        if (reconstruction->RegisterImageIds().size() <= 0) {
            continue;
        }

        if (color_harmonization) {
            reconstruction->ColorHarmonization(image_path);
        }

        const image_t first_image_id = reconstruction->RegisterImageIds()[0]; 
        CHECK(reconstruction->ExistsImage(first_image_id));
        const Image& image = reconstruction->Image(first_image_id);
        const Camera& camera = reconstruction->Camera(image.CameraId());
        
        std::string rec_path = StringPrintf("%s/%d", workspace_path.c_str(), base_reconstruction_idx + i);
        if (!boost::filesystem::exists(rec_path)) {
            boost::filesystem::create_directories(rec_path);
        }

        if (use_gps_prior) {
            // Eigen::Matrix3x4d matrix_to_align =
            reconstruction->AlignWithPriorLocations(options->independent_mapper_options.max_error_gps);
            if(false && camera.NumLocalCameras() == 1){
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
            // Eigen::Matrix3d RT = matrix_to_geo.block<3, 3>(0, 0).transpose();
            // Eigen::Matrix3x4d M = Eigen::Matrix3x4d::Identity();
            // M.block<3, 3>(0, 0) = RT;
            // M.block<3, 1>(0, 3) = -RT * matrix_to_geo.block<3, 1>(0, 3);

            // std::ofstream file(JoinPaths(workspace_path, "/local_to_ned.txt"), std::ofstream::out);
            // file << MAX_PRECISION << M(0, 0) << " " << M(0, 1) << " "
            //      << M(0, 2) << " " << M(0, 3) << std::endl;
            // file << MAX_PRECISION << M(1, 0) << " " << M(1, 1) << " "
            //      << M(1, 2) << " " << M(1, 3) << std::endl;
            // file << MAX_PRECISION << M(2, 0) << " " << M(2, 1) << " "
            //      << M(2, 2) << " " << M(2, 3) << std::endl;
            // file.close();
        }
        if (options->cluster_mapper_options.mapper_options.extract_colors) {
            reconstruction->ExtractColorsForAllImages(image_path);
        }

        std::cout << "export to " << JoinPaths(rec_path, MAPPOINT_NAME) << std::endl;
        reconstruction->ExportMapPoints(JoinPaths(rec_path, MAPPOINT_NAME));

        if (camera.NumLocalCameras() > 1) {
            reconstruction->WriteReconstruction(rec_path,
                options->independent_mapper_options.write_binary_model);

            std::string export_rec_path = rec_path + "-export";
            if (!boost::filesystem::exists(export_rec_path)) {
                boost::filesystem::create_directories(export_rec_path);
            }
            Reconstruction rig_reconstruction;
            reconstruction->ConvertRigReconstruction(rig_reconstruction);
            rig_reconstruction.WriteReconstruction(export_rec_path,
                options->independent_mapper_options.write_binary_model);
        } else {
            reconstruction->WriteReconstruction(rec_path,
                options->independent_mapper_options.write_binary_model);
        }

        if (options->independent_mapper_options.extract_keyframe) {
            auto keyframe_reconstruction = *reconstruction.get();
            auto image_ids = reconstruction->RegisterImageIds();
            for (auto image_id : image_ids) {
                auto image = reconstruction->Image(image_id);
                if (!image.IsKeyFrame()) {
                    keyframe_reconstruction.DeleteImage(image_id);
                }
            }
            std::string keyframe_rec_path = StringPrintf("%s/%d/KeyFrames", workspace_path.c_str(), base_reconstruction_idx + i);
            if (!boost::filesystem::exists(keyframe_rec_path)) {
                boost::filesystem::create_directories(keyframe_rec_path);
            }
            keyframe_reconstruction.WriteBinary(keyframe_rec_path);
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
                    std::shared_ptr<sensemap::ReconstructionManager> &reconstruction_manager, Configurator &param) {
    using namespace sensemap;

    PrintHeading1("Incremental Mapping");

    auto options = std::make_shared<MapperOptions>();
    options->mapper_type = MapperType::INDEPENDENT;
    options->independent_mapper_options.outside_mapper_type = MapperType::INDEPENDENT;
    options->independent_mapper_options.independent_mapper_type = IndependentMapperType::INCREMENTAL;

    OptionParser option_parser;
    ImageReaderOptions reader_options;
    option_parser.GetImageReaderOptions(reader_options,param);
    option_parser.GetMapperOptions(options->independent_mapper_options,param);

    if (scene_graph_container->NumImages() < options->independent_mapper_options.min_model_size) {
        std::cout << StringPrintf("too few images(%d) in scene graph\n");
        return;
    }

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

    // rgbd mode
    int num_local_cameras = reader_options.num_local_cameras;
    bool with_depth = options->independent_mapper_options.with_depth;

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

    bool use_apriltag = static_cast<bool>(param.GetArgument("detect_apriltag", 0));
    double apriltag_size = param.GetArgument("apriltag_size", 0.113f);
    bool color_harmonization = param.GetArgument("color_harmonization", 0);

    std::vector<int> reconstruction_ids = reconstruction_manager->getReconstructionIds();
    std::vector<int> remain_reconstruction_ids;
    std::set_difference(reconstruction_ids.begin(), reconstruction_ids.end(), 
                        base_reconstruction_ids.begin(), base_reconstruction_ids.end(), std::back_inserter(remain_reconstruction_ids));

    for (size_t i = 0; i < remain_reconstruction_ids.size(); ++i) {
        auto reconstruction = reconstruction_manager->Get(remain_reconstruction_ids[i]);
        if (reconstruction->RegisterImageIds().size() <= 0) {
            continue;
        }

        if (color_harmonization) {
            reconstruction->ColorHarmonization(image_path);
        }

        // CHECK(reconstruction->RegisterImageIds().size()>0);
        const image_t first_image_id = reconstruction->RegisterImageIds()[0]; 
        CHECK(reconstruction->ExistsImage(first_image_id));
        const Image& image = reconstruction->Image(first_image_id);
        const Camera& camera = reconstruction->Camera(image.CameraId());
        
        std::string rec_path = StringPrintf("%s/%d", workspace_path.c_str(), base_reconstruction_idx + i);
        if (!boost::filesystem::exists(rec_path)) {
            boost::filesystem::create_directories(rec_path);
        }

        if (use_gps_prior) {
            // Eigen::Matrix3x4d matrix_to_align =
            reconstruction->AlignWithPriorLocations(options->independent_mapper_options.max_error_gps);
            if(false && camera.NumLocalCameras() == 1){
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
            // Eigen::Matrix3d RT = matrix_to_geo.block<3, 3>(0, 0).transpose();
            // Eigen::Matrix3x4d M = Eigen::Matrix3x4d::Identity();
            // M.block<3, 3>(0, 0) = RT;
            // M.block<3, 1>(0, 3) = -RT * matrix_to_geo.block<3, 1>(0, 3);

            // std::ofstream file(JoinPaths(workspace_path, "/local_to_ned.txt"), std::ofstream::out);
            // file << MAX_PRECISION << M(0, 0) << " " << M(0, 1) << " "
            //      << M(0, 2) << " " << M(0, 3) << std::endl;
            // file << MAX_PRECISION << M(1, 0) << " " << M(1, 1) << " "
            //      << M(1, 2) << " " << M(1, 3) << std::endl;
            // file << MAX_PRECISION << M(2, 0) << " " << M(2, 1) << " "
            //      << M(2, 2) << " " << M(2, 3) << std::endl;
            // file.close();
        }
        if (options->independent_mapper_options.extract_colors) {
            reconstruction->ExtractColorsForAllImages(image_path);
        }
        std::cout << "export to " << JoinPaths(rec_path, MAPPOINT_NAME) << std::endl;
        reconstruction->ExportMapPoints(JoinPaths(rec_path, MAPPOINT_NAME));

        if (camera.NumLocalCameras() > 1) {
            reconstruction->WriteReconstruction(rec_path,
                options->independent_mapper_options.write_binary_model);

            std::string export_rec_path = rec_path + "-export";
            if (!boost::filesystem::exists(export_rec_path)) {
                boost::filesystem::create_directories(export_rec_path);
            }
            Reconstruction rig_reconstruction;
            reconstruction->ConvertRigReconstruction(rig_reconstruction);

            rig_reconstruction.WriteReconstruction(export_rec_path,
                options->independent_mapper_options.write_binary_model);
        } else {
            reconstruction->WriteReconstruction(rec_path,
                options->independent_mapper_options.write_binary_model);
        }

        if (options->independent_mapper_options.extract_keyframe) {
            auto keyframe_reconstruction = *reconstruction.get();
            auto image_ids = reconstruction->RegisterImageIds();
            for (auto image_id : image_ids) {
                auto image = reconstruction->Image(image_id);
                if (!image.IsKeyFrame()) {
                    keyframe_reconstruction.DeleteImage(image_id);
                }
            }
            std::string keyframe_rec_path = StringPrintf("%s/%d/KeyFrames", workspace_path.c_str(), base_reconstruction_idx + i);
            if (!boost::filesystem::exists(keyframe_rec_path)) {
                boost::filesystem::create_directories(keyframe_rec_path);
            }
            keyframe_reconstruction.WriteBinary(keyframe_rec_path);
        }
    }
}

void GlobalSFM(std::shared_ptr<sensemap::SceneGraphContainer> &scene_graph_container,
               std::shared_ptr<sensemap::ReconstructionManager> &reconstruction_manager, Configurator &param) {
    using namespace sensemap;

    PrintHeading1("Global Mapping");

    auto options = std::make_shared<MapperOptions>();
    options->mapper_type = MapperType::INDEPENDENT;
    options->independent_mapper_options.outside_mapper_type = MapperType::INDEPENDENT;
    options->independent_mapper_options.independent_mapper_type = IndependentMapperType::GLOBAL;

    OptionParser option_parser;
    ImageReaderOptions reader_options;
    option_parser.GetImageReaderOptions(reader_options,param);
    option_parser.GetMapperOptions(options->independent_mapper_options,param);

    std::string workspace_path = param.GetArgument("workspace_path", "");
    CHECK(!workspace_path.empty()) << "workspace path empty";

    std::string image_path = param.GetArgument("image_path", "");
    CHECK(!image_path.empty()) << "image path empty";


    MapperController *mapper = MapperController::Create(options, workspace_path, image_path,
                                                        scene_graph_container, reconstruction_manager);
    mapper->Start();
    mapper->Wait();

    fprintf(
            fs, "%s\n",
            StringPrintf("Global Mapper Elapsed time: %.3f [minutes]", mapper->GetTimer().ElapsedMinutes()).c_str());
    fflush(fs);

}

bool PriorReady(const std::shared_ptr<sensemap::SceneGraphContainer> &scene_graph_container, int threshold = 0.8) {
    auto image_ids = scene_graph_container->GetImageIds();
    double ready_num = 0;
    for (auto id : image_ids) {
        auto image = scene_graph_container->Image(id);
        if (image.HasQvecPrior() && image.HasTvecPrior()) {
            ready_num++;
        }
    }
    return ready_num / image_ids.size() > threshold;
}


bool RTKReady(const std::shared_ptr<sensemap::SceneGraphContainer> &scene_graph_container, double threshold = 0.8) {
    auto image_ids = scene_graph_container->GetImageIds();
    double ready_num = 0;
    for (auto id : image_ids) {
        auto image = scene_graph_container->Image(id);
        if (image.RtkFlag() == 50) {
            ready_num++;
        }
    }
    std::cout << "RTKReady: " << ready_num << " / " << image_ids.size() << std::endl;
    return ready_num / image_ids.size() > threshold;
}

void DirectedSFM(std::shared_ptr<sensemap::SceneGraphContainer> &scene_graph_container,
                 std::shared_ptr<sensemap::ReconstructionManager> &reconstruction_manager, Configurator &param) {
    using namespace sensemap;

    PrintHeading1("Dierected Mapping");

    auto options = std::make_shared<MapperOptions>();
    options->mapper_type = MapperType::INDEPENDENT;
    options->independent_mapper_options.outside_mapper_type = MapperType::INDEPENDENT;
    options->independent_mapper_options.independent_mapper_type = IndependentMapperType::DIRECTED;

    OptionParser option_parser;
    ImageReaderOptions reader_options;
    option_parser.GetImageReaderOptions(reader_options,param);
    option_parser.GetMapperOptions(options->independent_mapper_options,param);

    std::string workspace_path = param.GetArgument("workspace_path", "");
    CHECK(!workspace_path.empty()) << "image path empty";

    std::string image_path = param.GetArgument("image_path", "");
    CHECK(!workspace_path.empty()) << "image path empty";

    bool color_harmonization = param.GetArgument("color_harmonization", 0);

    size_t base_reconstruction_idx = reconstruction_manager->Size();
    std::vector<int> base_reconstruction_ids = reconstruction_manager->getReconstructionIds();

    auto independent_options = std::make_shared<IndependentMapperOptions>(
            options->independent_mapper_options);

    if(independent_options->direct_mapper_type == 0){
        return ;
    }

    while(true) {

        if(!RTKReady(scene_graph_container, 0.4) &&
            (independent_options->direct_mapper_type == 2 || independent_options->direct_mapper_type == 4)){
            break;
        }

        auto *mapper = new DirectedMapperController(independent_options, image_path, workspace_path,
                scene_graph_container, reconstruction_manager);
        mapper->Start();
        mapper->Wait();
        
        fprintf(
                fs, "%s\n",
                StringPrintf("Directed Mapper Elapsed time: %.3f [minutes]", mapper->GetTimer().ElapsedMinutes()).c_str());
        fflush(fs);

        if (mapper->IsSuccess()) {
            std::set<image_t> clustered_image_ids;
            for (auto & image_id : scene_graph_container->GetImageIds()) {
                if (!scene_graph_container->Image(image_id).IsRegistered()) {
                    clustered_image_ids.insert(image_id);
                }
            }
            std::shared_ptr<SceneGraphContainer> cluster_graph_container =
                std::shared_ptr<SceneGraphContainer>(new SceneGraphContainer());
            scene_graph_container->ClusterSceneGraphContainer(clustered_image_ids, *cluster_graph_container.get());

            std::swap(scene_graph_container, cluster_graph_container);

            std::cout << StringPrintf("Remain %d images to be reconstructed!\n", scene_graph_container->NumImages());
        }
        if (!mapper->IsSuccess() || scene_graph_container->NumImages() == 0 || 
            !options->independent_mapper_options.multiple_models) {
            break;
        }
    }

    std::vector<int> reconstruction_ids = reconstruction_manager->getReconstructionIds();
    std::vector<int> remain_reconstruction_ids;
    std::set_difference(reconstruction_ids.begin(), reconstruction_ids.end(), 
                        base_reconstruction_ids.begin(), base_reconstruction_ids.end(), std::back_inserter(remain_reconstruction_ids));
    for (size_t i = 0; i < remain_reconstruction_ids.size(); ++i) {
        auto reconstruction = reconstruction_manager->Get(remain_reconstruction_ids[i]);
        if (reconstruction->RegisterImageIds().size() <= 0) {
            continue;
        }

        if (color_harmonization) {
            reconstruction->ColorHarmonization(image_path);
        }

        // CHECK(reconstruction->RegisterImageIds().size()>0);
        const image_t first_image_id = reconstruction->RegisterImageIds()[0]; 
        CHECK(reconstruction->ExistsImage(first_image_id));
        const Image& image = reconstruction->Image(first_image_id);
        const Camera& camera = reconstruction->Camera(image.CameraId());
        
        std::string rec_path = StringPrintf("%s/%d", workspace_path.c_str(), base_reconstruction_idx + i);
        if (!boost::filesystem::exists(rec_path)) {
            boost::filesystem::create_directories(rec_path);
        }

        if (options->independent_mapper_options.extract_colors) {
            reconstruction->ExtractColorsForAllImages(image_path);
        }
        std::cout << "export to " << JoinPaths(rec_path, MAPPOINT_NAME) << std::endl;
        reconstruction->ExportMapPoints(JoinPaths(rec_path, MAPPOINT_NAME));

        if (camera.NumLocalCameras() > 1) {
            reconstruction->WriteReconstruction(rec_path,
                options->independent_mapper_options.write_binary_model);

            std::string export_rec_path = rec_path + "-export";
            if (!boost::filesystem::exists(export_rec_path)) {
                boost::filesystem::create_directories(export_rec_path);
            }
            Reconstruction rig_reconstruction;
            reconstruction->ConvertRigReconstruction(rig_reconstruction);
            rig_reconstruction.WriteReconstruction(export_rec_path,
                options->independent_mapper_options.write_binary_model);
        } else {
            reconstruction->WriteReconstruction(rec_path,
                options->independent_mapper_options.write_binary_model);
        }
    }
}

int main(int argc, char *argv[]) {
    using namespace sensemap;

    PrintHeading("SenseMap.  Copyright(c) 2019, SenseTime Group.");    PrintHeading(std::string("Version: hd-sfm-")+__VERSION__);
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

    FeatureExtractionAndMatching(*scene_graph_container.get(), param);

    std::string mapper_method = param.GetArgument("mapper_method", "incremental");
    bool multiple_models = static_cast<bool>(param.GetArgument("multiple_models", 0));
    
    bool refine_separate_cameras = static_cast<bool>(param.GetArgument("refine_separate_cameras", 0));
    must_increamental = static_cast<bool>(param.GetArgument("must_increamental", 0)); 
    std::cout << "must_increamental: " << must_increamental << std::endl;

    size_t num_images = scene_graph_container->NumImages();
    {
        std::unordered_map<image_t, std::unordered_set<image_t> > cc;
        GetAllConnectedComponentIds(*scene_graph_container->CorrespondenceGraph(), cc);

        std::cout << StringPrintf("Get %d component\n", cc.size());

        std::vector<std::pair<image_t, int> > sorted_clustered_images;
        sorted_clustered_images.reserve(cc.size());
        for (auto & clustered_image_ids : cc) {
            sorted_clustered_images.emplace_back(clustered_image_ids.first, clustered_image_ids.second.size());
        }
        std::sort(sorted_clustered_images.begin(), sorted_clustered_images.end(), 
            [&](const auto & image1, const auto & image2) {
                return image1.second > image2.second;
            });

        int component_id = 0;
        for (auto & cluster_image : sorted_clustered_images) {
            auto & clustered_image_ids = cc.at(cluster_image.first);
            PrintHeading1(StringPrintf("Processing component %d", component_id++));
            std::shared_ptr<SceneGraphContainer> cluster_graph_container =
                std::shared_ptr<SceneGraphContainer>(new SceneGraphContainer());

            std::set<image_t> unique_image_ids;
            for (auto & image_id : clustered_image_ids) {
                unique_image_ids.insert(image_id);
            }

            scene_graph_container->ClusterSceneGraphContainer(unique_image_ids, *cluster_graph_container.get());

            size_t cluster_num_images = cluster_graph_container->NumImages();
            if (refine_separate_cameras) {
                cluster_num_images = 0;
                const auto & image_ids = cluster_graph_container->GetImageIds();
                for (auto image_id : image_ids) {
                    auto image = cluster_graph_container->Image(image_id);
                    auto camera = cluster_graph_container->Camera(image.CameraId());
                    cluster_num_images += camera.NumLocalCameras();
                }
            }

            if (static_cast<bool>(param.GetArgument("map_update", 0))) {
                IncrementalSFM(cluster_graph_container, reconstruction_manager, param);
            } else {
                if (mapper_method.compare("incremental") == 0) {
                    auto camera_model = param.GetArgument("camera_model", "OPENCV");
                    if((camera_model == "PARTIAL_OPENCV" || camera_model == "OPENCV") && !must_increamental) {
                        DirectedSFM(cluster_graph_container, reconstruction_manager, param);
                    }
                    if (cluster_graph_container->NumImages() > 0) {
                        IncrementalSFM(cluster_graph_container, reconstruction_manager, param);
                    }
                } else if (mapper_method.compare("cluster") == 0) {
                    ClusterMapper(cluster_graph_container, reconstruction_manager, param);
                }
            }

            if (!multiple_models) {
                bool rec_success = false;
                for (size_t rec_idx = 0; rec_idx < reconstruction_manager->Size(); ++rec_idx) {
                    auto reconstruction = reconstruction_manager->Get(rec_idx);
                    if (reconstruction->NumRegisterImages() >= 0.1f * cluster_num_images) {
                        rec_success = true;
                        break;
                    }
                }
                if (rec_success) {
                    break;
                }
            }
        }
    }
    std::cout << StringPrintf("Reconstruct in %.3fs", timer.ElapsedSeconds()) << std::endl;

    PrintReconSummary(workspace_path + "/statistic.txt", num_images, reconstruction_manager);

    fclose(fs);

    return 0;
}
