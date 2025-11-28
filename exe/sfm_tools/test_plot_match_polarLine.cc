// Copyright (c) 2019, SenseTime Group.
// All rights reserved.

#include <boost/filesystem/path.hpp>
#include <opencv2/opencv.hpp>

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

#include "../Configurator_yaml.h"
#include "../camera_rig_params.h"
#include "../option_parsing.h"

#include <dirent.h>
#include <sys/stat.h>

#include "util/gps_reader.h"
#include <unordered_set>


using namespace sensemap;

std::string configuration_file_path;

FILE *fs;

// Load panorama config file
bool LoadParams(const std::string path, std::vector<PanoramaParam>& panorama_params) {
    std::cout << "Load Panorama Params ..." << std::endl;
    cv::FileStorage fs(path, cv::FileStorage::READ);

    // Check file exist
    if (!fs.isOpened()) {
        fprintf(stderr, "%s:%d:loadParams falied. 'Panorama.yaml' does not exist\n", __FILE__, __LINE__);
        return false;
    }

    // Get number of sub camera
    int n_camera = (int)fs["n_camera"];
    panorama_params.resize(n_camera);

    for (int i = 0; i < n_camera; i++) {
        std::string camera_id = "cam_" + std::to_string(i);
        cv::FileNode node = fs[camera_id]["params"];
        std::vector<double> cam_params;
        node >> cam_params;
        double pitch, yaw, roll, fov_w;
        int pers_w, pers_h;

        pitch = cam_params[0];
        yaw = cam_params[1];
        roll = cam_params[2];
        fov_w = cam_params[3];
        std::cout << "camera_" << i << " pitch = " << cam_params[0];
        std::cout << ", yaw = " << cam_params[1];
        std::cout << ", roll = " << cam_params[2];
        std::cout << ", fov_w = " << cam_params[3];

        // Check the pers_x and pers_y is int
        if (cam_params[4] - floor(cam_params[4]) != 0 || cam_params[5] - floor(cam_params[5]) != 0) {
            std::cout << "Input perspective image size is not int" << std::endl;
            return false;
        }

        pers_w = (int)cam_params[4];
        pers_h = (int)cam_params[5];
        std::cout << ", pers_w = " << (int)cam_params[4];
        std::cout << ", pers_h = " << (int)cam_params[5] << std::endl;

        panorama_params[i] = PanoramaParam(pitch, yaw, roll, fov_w, pers_w, pers_h);
    }

    return true;
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

    std::string camera_params = reader_options.camera_params;

    if (with_depth && !rgbd_parmas_file.empty() && camera_params.empty()) {
        CalibOPPOBinReader calib_reader;
        calib_reader.ReadCalib(rgbd_parmas_file);

        auto filelist = GetRecursiveFileList(reader_options.image_path);
        for (auto filename : filelist) {
            MatXf depthmap;
            Bitmap bitmap;
			ExtractRGBDData(filename, bitmap, depthmap);
            
            Eigen::Matrix3f K;
            int rgb_w, rgb_h;
            calib_reader.GetRGB_K(K, rgb_w, rgb_h);

            float width_scale = bitmap.Width() * 1.0f / rgb_w;
            float height_scale = bitmap.Height() * 1.0f / rgb_h;

            camera_params.append(std::to_string(K(0, 0) * width_scale));
            camera_params.append(",");
            camera_params.append(std::to_string(K(1, 1) * height_scale));
            camera_params.append(",");
            camera_params.append(std::to_string(K(0, 2) * width_scale));
            camera_params.append(",");
            camera_params.append(std::to_string(K(1, 2) * height_scale));
            std::cout << "camera_params: " << camera_params << std::endl;
            break;
        }
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
            
            if(reader_options.num_local_cameras == 2 && reader_options.camera_model == "OPENCV_FISHEYE"){
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
            
            if(reader_options.num_local_cameras == 2 && reader_options.camera_model == "OPENCV_FISHEYE"){
                feature_data_container.WritePieceIndicesData(JoinPaths(workspace_path, "/piece_indices.txt"));
            }
        }
    }
}

void FeatureMatching(FeatureDataContainer &feature_data_container, SceneGraphContainer &scene_graph,
                     Configurator &param) {
    using namespace std::chrono;
    high_resolution_clock::time_point start_time = high_resolution_clock::now();

    OptionParser option_parser;
    ImageReaderOptions reader_options;
    option_parser.GetImageReaderOptions(reader_options,param);


    std::string workspace_path = param.GetArgument("workspace_path", "");
    CHECK(!workspace_path.empty());
    bool load_scene_graph = false;
    if (boost::filesystem::exists(JoinPaths(workspace_path, "/scene_graph.bin"))) {
        scene_graph.ReadSceneGraphBinaryData(JoinPaths(workspace_path, "/scene_graph.bin"));
        load_scene_graph = true;
    } else if (boost::filesystem::exists(JoinPaths(workspace_path, "/scene_graph.txt"))) {
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
    } else {
        FeatureMatchingOptions options;
        option_parser.GetFeatureMatchingOptions(options,param);

        // use intial sfm to filter far image pairs. 
        bool use_initial_sfm = static_cast<bool>(param.GetArgument("use_initial_sfm", 0));

        bool has_initial_sfm = false;
        
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

        bool write_match = static_cast<bool>(param.GetArgument("write_match", 0));
        bool write_binary = static_cast<bool>(param.GetArgument("write_binary", 1));

        if (write_match) {
            if (write_binary) {
                scene_graph.WriteSceneGraphBinaryData(workspace_path + "/scene_graph.bin");
            } else {
                scene_graph.WriteSceneGraphData(workspace_path + "/scene_graph.txt");
            }
        }
    }
}

void DrawMatchPloarLIne(FeatureDataContainer &feature_data_container, 
                        SceneGraphContainer &scene_graph,
                        Configurator &param, const int max_num_plot = 10){
    
    std::string workspace_path = param.GetArgument("workspace_path", "");
    CHECK(!workspace_path.empty());

    
    std::string input_pose_path = param.GetArgument("input_pose_path", "");
    CHECK(!input_pose_path.empty()) << "input pose path empty";

    std::ifstream file_input_pose(input_pose_path);
    CHECK(file_input_pose.is_open());

    std::unordered_map<std::string, Eigen::Matrix3x4d> map_image_name_to_pose;
    std::unordered_map<std::string, double> map_image_name_to_focal_length;

    std::string image_name;
    std::string first_image_name;
    while (file_input_pose >> image_name) {
        if (map_image_name_to_pose.size() < 1){
            first_image_name = image_name;
        }
        Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 4; ++j) {
                file_input_pose >> pose(i, j);
            }
        }
        map_image_name_to_pose.emplace(image_name + ".jpg", pose.inverse().topRows(3));        
    }
    file_input_pose.close();

    // Add prior pose
    std::unordered_map<image_t, Eigen::Vector4d> prior_rotations;
    std::unordered_map<image_t, Eigen::Vector3d> prior_translations;

    std::vector<image_t> image_ids = scene_graph.GetImageIds();
    for (const auto image_id : image_ids) {
        Image &image = scene_graph.Image(image_id);        
        std::string image_name = image.Name();
        if (map_image_name_to_pose.find(image_name) == map_image_name_to_pose.end()) {
            // no prior pose, skipped
            std::cout << image_name << "  no prior pose, skipped" << std::endl;
            continue;
        }

        Eigen::Matrix3x4d pose = map_image_name_to_pose.at(image_name);
        prior_rotations.emplace(image_id, RotationMatrixToQuaternion(pose.block<3, 3>(0, 0)));
        prior_translations.emplace(image_id, pose.block<3, 1>(0, 3));
        image.SetQvecPrior(RotationMatrixToQuaternion(pose.block<3, 3>(0, 0)));
        image.SetTvecPrior(pose.block<3, 1>(0, 3));
    }

    // draw 
    if (!boost::filesystem::exists(workspace_path + "/matches")) {
        boost::filesystem::create_directories(workspace_path + "/matches");
    }
    int write_count = 0;
    // std::vector<image_t> image_ids = feature_data_container.GetImageIds();
    for(image_t id1 = 1; id1 < image_ids.size(); id1++) {
        image_t image_id1 = image_ids.at(id1);
        std::cout << "Image#" << image_id1 << std::endl;
        const PanoramaIndexs & panorama_indices1 = feature_data_container.GetPanoramaIndexs(image_id1);        
        for(image_t id2 = id1 + 1; id2 < image_ids.size(); id2++) {
            image_t image_id2 = image_ids.at(id2);

            const PanoramaIndexs & panorama_indices2 = feature_data_container.GetPanoramaIndexs(image_id2);      

            auto matches =scene_graph.CorrespondenceGraph()->
                    FindCorrespondencesBetweenImages(image_id1, image_id2);
            if(matches.empty()){
                continue;
            }
            std::string image_path = param.GetArgument("image_path","");

            auto image1 = scene_graph.Image(image_id1);
            auto image2 = scene_graph.Image(image_id2);

            const std::string input_image_path1 = image1.Name();
            std::string input_image_name1 = input_image_path1.substr(input_image_path1.find("/")+1);
            
            const std::string input_image_path2 = image2.Name();
            std::string input_image_name2 = input_image_path2.substr(input_image_path2.find("/")+1);
            std::cout<<input_image_name1<<" <=> "<<input_image_name2<<std::endl;

            // std::string name_ext1 = input_image_name1.substr(input_image_name1.size() - 3, 3);
            // std::string name_ext2 = input_image_name2.substr(input_image_name2.size() - 3, 3);

            std::vector<cv::Mat> mats1;
            std::vector<cv::Mat> mats2;

            for(int i = 0; i<1; ++i){
                std::string full_name1, full_name2;
                full_name1 = image_path + "/" + input_image_path1;
                full_name2 = image_path + "/" + input_image_path2;

                cv::Mat mat1 = cv::imread(full_name1);
                cv::Mat mat2 = cv::imread(full_name2);

                mats1.push_back(mat1);
                mats2.push_back(mat2);
            }

            // find match
            int i = 0, j = 0;
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
                if(!( panorama_indices1[matches[m].point2D_idx1].sub_image_id == i && 
                    panorama_indices2[matches[m].point2D_idx2].sub_image_id ==j)){
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
            
            // draw match
            cv::Mat first_match;
            cv::drawMatches(mat1, keypoints_show1, mat2, keypoints_show2,
                            matches_show, first_match);

            // draw polar line
            // compute Fundamental Mat
            Eigen::Matrix3d K1, K2;
            const auto camera1 = scene_graph.Camera(image1.CameraId());
            K1 << camera1.FocalLengthX(), 0.0, camera1.PrincipalPointX(),
                  0.0, camera1.FocalLengthY(), camera1.PrincipalPointY(),
                  0.0, 0.0, 1.0;
                  
            const auto camera2 = scene_graph.Camera(image2.CameraId());
            K2 << camera2.FocalLengthX(), 0.0, camera2.PrincipalPointX(),
                  0.0, camera2.FocalLengthY(), camera2.PrincipalPointY(),
                  0.0, 0.0, 1.0;
            if (!image1.HasQvecPrior() && !image2.HasTvecPrior()){
                std::cout << "no qvec prior "<< image1.ImageId() << " " << image2.ImageId() << std::endl;
            }
            Eigen::Matrix4d T_wc1 = Eigen::Matrix4d::Identity();
            T_wc1.topLeftCorner(3,3) = QuaternionToRotationMatrix(image1.QvecPrior());
            T_wc1.topRightCorner(3,1) = image1.TvecPrior();
            Eigen::Matrix4d T_wc2 = Eigen::Matrix4d::Identity();
            T_wc2.topLeftCorner(3,3) = QuaternionToRotationMatrix(image2.QvecPrior());
            T_wc2.topRightCorner(3,1) = image2.TvecPrior();

            // std::cout << "T_wc1: " << T_wc1 << "\nT_wc2 : " << T_wc2 << std::endl;

            Eigen::Matrix4d T_c1c2 = T_wc2 * T_wc1.inverse();
            Eigen::Matrix3d R_c1c2 = T_c1c2.topLeftCorner(3,3);
            Eigen::Vector3d t_c1c2 = T_c1c2.topRightCorner(3,1);
            Eigen::Matrix3d t_cross_c1c2;
            t_cross_c1c2 << 0.0, -t_c1c2.z(), t_c1c2.y(), 
                            t_c1c2.z(), 0.0, -t_c1c2.x(),
                            -t_c1c2.y(), t_c1c2.x(), 0.0;
            Eigen::Matrix3d F_eigen = K2.inverse().transpose() * t_cross_c1c2 * R_c1c2 * K1.inverse();
            cv::Mat F = cv::Mat(3, 3, CV_64FC1);
            F.at<double>(0,0) = F_eigen(0,0);
            F.at<double>(0,1) = F_eigen(0,1);
            F.at<double>(0,2) = F_eigen(0,2);
            F.at<double>(1,0) = F_eigen(1,0);
            F.at<double>(1,1) = F_eigen(1,1);
            F.at<double>(1,2) = F_eigen(1,2);
            F.at<double>(2,0) = F_eigen(2,0);
            F.at<double>(2,1) = F_eigen(2,1);
            F.at<double>(2,2) = F_eigen(2,2);

            std::vector<cv::Point2f> points1, points2;
            cv::KeyPoint::convert(keypoints_show1, points1);
            cv::KeyPoint::convert(keypoints_show2, points2);

            const cv::Mat K = ( cv::Mat_<double> ( 3,3 ) << K1(0,0), 0.0, K1(0,2), 0.0, K1(1,1), K1(1,2), 0.0, 0.0, 1.0 );
            const cv::Mat D = ( cv::Mat_<double> ( 4,1 ) << camera1.Params().at(4), camera1.Params().at(5), camera1.Params().at(6), camera1.Params().at(7));
            // std::cout << "K" << K << std::endl;
            // std::cout << "D" << D << std::endl;
            // std::cout << points1.at(0) << std::endl;
            cv::undistortPoints(points1, points1, K, D);
            cv::undistortPoints(points2, points2, K, D);

            cv::Mat centerPntsThreeCols1 = cv::Mat::zeros(points1.size(), 3, CV_64FC1);
            cv::Mat centerPntsThreeCols2 = cv::Mat::zeros(points1.size(), 3, CV_64FC1);
            for (size_t i = 0; i < points1.size(); i++)
            {
                centerPntsThreeCols1.at<double>(i, 0) = points1[i].x;
                centerPntsThreeCols1.at<double>(i, 1) = points1[i].y;
                centerPntsThreeCols1.at<double>(i, 2) = 1;

                centerPntsThreeCols2.at<double>(i, 0) = points2[i].x;
                centerPntsThreeCols2.at<double>(i, 1) = points2[i].y;
                centerPntsThreeCols2.at<double>(i, 2) = 1;
            }
            cv::Mat undistortCenterThreeCols1 = K*centerPntsThreeCols1.t();
            cv::Mat undistortCenterThreeCols2 = K*centerPntsThreeCols2.t();
            for (size_t i = 0; i < points1.size(); i++)
            {
                points1[i] = cv::Point2f(undistortCenterThreeCols1.at<double>(0, i), undistortCenterThreeCols1.at<double>(1, i));
                points2[i] = cv::Point2f(undistortCenterThreeCols2.at<double>(0, i), undistortCenterThreeCols2.at<double>(1, i));
            }
            // std::cout << points1.at(0) << std::endl;

            // cv::Mat F_cv = cv::findFundamentalMat(points1, points2, CV_FM_8POINT);
            // std::cout << F_cv << "\n F: " << F << std::endl;
            std::vector<cv::Vec<float, 3>> epilines1, epilines2;
            cv::computeCorrespondEpilines(points1, 1, F, epilines1);
            cv::computeCorrespondEpilines(points2, 2, F, epilines2);
            cv::Mat img1, img2;
            if (mat1.type() == CV_8UC3)
            {
                mat1.copyTo(img1);
                mat2.copyTo(img2);
            }
            else if (mat1.type() == CV_8UC1)
            {
                cvtColor(mat1, img1, cv::COLOR_GRAY2BGR);
                cvtColor(mat2, img2, cv::COLOR_GRAY2BGR);
            }
            else
            {
                cout << "unknow img type\n" << endl;
                exit(0);
            }

            cv::Mat img11, img22;
            cv::undistort(img1, img11, K, D, K);
            cv::undistort(img2, img22, K, D, K);
            img11.copyTo(img1);
            img22.copyTo(img2);

            cv::RNG& rng = cv::theRNG();
            int delt_num = keypoints_show2.size() / max_num_plot + 1;
            for (int i = 0; i < keypoints_show2.size(); i+=delt_num) {
                cv::Scalar color = cv::Scalar(rng(256), rng(256), rng(256));
        
                cv::circle(img2, points2[i], 5, color);
                cv::line(img2, cv::Point(0, -epilines1[i][2] / epilines1[i][1]), cv::Point(img2.cols, -(epilines1[i][2] + epilines1[i][0] * img2.cols) / epilines1[i][1]), color, 1);
                cv::circle(img1, points1[i], 5, color);
                cv::line(img1, cv::Point(0, -epilines2[i][2] / epilines2[i][1]), cv::Point(img1.cols, -(epilines2[i][2] + epilines2[i][0] * img1.cols) / epilines2[i][1]), color, 1);
            }
            // draw image
            const std::string ouput_image_path = JoinPaths(
                workspace_path + "/matches",
                input_image_name1 + "+" + input_image_name2);
            cv::Mat img_12, img_comb;
            cv::hconcat(img1, img2, img_12);
            cv::vconcat(first_match, img_12, img_comb);

            // cv::imwrite(ouput_image_path, img2);
            cv::imwrite(ouput_image_path, img_comb);
        
            // write_count++;
            // if(write_count>=100) break;
        }
    }
}

void DrawMatchPloarLIneWithRect(FeatureDataContainer &feature_data_container, 
                        SceneGraphContainer &scene_graph,
                        Configurator &param, const int max_num_plot = 10){
    
    std::string workspace_path = param.GetArgument("workspace_path", "");
    CHECK(!workspace_path.empty());

    const std::string rec_path = workspace_path + "/0/dense/sparse/";
    Reconstruction reconstruction;
    reconstruction.ReadBinary(rec_path);

    // const std::vector<image_t> registered_image_ids = reconstruction.RegisterImageIds();
    // for (const auto & image_id : registered_image_ids) {
    //     const class Image& image = feature_data_container.GetImage(image_id);
    //     if (image.HasTvecPrior()) {
    //         reconstruction.Image(image_id).SetTvecPrior(image.TvecPrior());
    //     }
    // }

    // Add prior pose
    // std::unordered_map<image_t, Eigen::Vector4d> prior_rotations;
    // std::unordered_map<image_t, Eigen::Vector3d> prior_translations;

    std::vector<image_t> image_ids = scene_graph.GetImageIds();
    for (const auto image_id : image_ids) {
        Image &image = scene_graph.Image(image_id);        
        std::string image_name = image.Name();
        if (!reconstruction.IsImageRegistered(image_id)) {
            // no prior pose, skipped
            std::cout << image_name << "  no prior pose, skipped" << std::endl;
            continue;
        }

        image.SetQvecPrior(reconstruction.Image(image_id).Qvec());
        image.SetTvecPrior(reconstruction.Image(image_id).Tvec());
    }

    // draw 
    if (!boost::filesystem::exists(workspace_path + "/matches")) {
        boost::filesystem::create_directories(workspace_path + "/matches");
    }
    int write_count = 0;
    // std::vector<image_t> image_ids = feature_data_container.GetImageIds();
    for(image_t id1 = 1; id1 < image_ids.size(); id1++) {
        image_t image_id1 = image_ids.at(id1);
        std::cout << "Image#" << image_id1 << std::endl;
        const PanoramaIndexs & panorama_indices1 = feature_data_container.GetPanoramaIndexs(image_id1);        
        for(image_t id2 = id1 + 1; id2 < image_ids.size(); id2++) {
            image_t image_id2 = image_ids.at(id2);

            const PanoramaIndexs & panorama_indices2 = feature_data_container.GetPanoramaIndexs(image_id2);      

            auto matches =scene_graph.CorrespondenceGraph()->
                    FindCorrespondencesBetweenImages(image_id1, image_id2);
            if(matches.empty()){
                continue;
            }
            // std::string image_path = param.GetArgument("image_path","");
            std::string image_path =  workspace_path + "/0/dense/images/";

            auto image1 = scene_graph.Image(image_id1);
            auto image2 = scene_graph.Image(image_id2);

            const std::string input_image_path1 = image1.Name();
            std::string input_image_name1 = input_image_path1.substr(input_image_path1.find("/")+1);
            
            const std::string input_image_path2 = image2.Name();
            std::string input_image_name2 = input_image_path2.substr(input_image_path2.find("/")+1);
            std::cout<<input_image_name1<<" <=> "<<input_image_name2<<std::endl;

            // std::string name_ext1 = input_image_name1.substr(input_image_name1.size() - 3, 3);
            // std::string name_ext2 = input_image_name2.substr(input_image_name2.size() - 3, 3);

            std::vector<cv::Mat> mats1;
            std::vector<cv::Mat> mats2;

            for(int i = 0; i<1; ++i){
                std::string full_name1, full_name2;
                full_name1 = image_path + "/" + input_image_path1;
                full_name2 = image_path + "/" + input_image_path2;

                cv::Mat mat1 = cv::imread(full_name1);
                cv::Mat mat2 = cv::imread(full_name2);

                mats1.push_back(mat1);
                mats2.push_back(mat2);
            }

            // find match
            int i = 0, j = 0;
            cv::Mat& mat1 = mats1[i];
            cv::Mat& mat2 = mats2[j];

            std::vector<cv::KeyPoint> keypoints_show1, keypoints_show2;
            std::vector<cv::DMatch> matches_show;
            
            int k = 0;
            for (int m = 0; m < matches.size(); ++m){
                // auto& keypoints1 = feature_data_container.GetKeypoints(image_id1);
                auto& keypoints1 = reconstruction.Image(image_id1).Points2D();
                CHECK_LT(matches[m].point2D_idx1, keypoints1.size());
                auto keypoint1 =keypoints1[matches[m].point2D_idx1];
                
                // auto keypoints2 = feature_data_container.GetKeypoints(image_id2);
                auto keypoints2 = reconstruction.Image(image_id2).Points2D();
                CHECK_LT(matches[m].point2D_idx2, keypoints2.size());
                auto keypoint2 = keypoints2[matches[m].point2D_idx2];

                CHECK_LT(matches[m].point2D_idx1, panorama_indices1.size());
                CHECK_LT(matches[m].point2D_idx2, panorama_indices2.size());
                if(!( panorama_indices1[matches[m].point2D_idx1].sub_image_id == i && 
                    panorama_indices2[matches[m].point2D_idx2].sub_image_id ==j)){
                    continue;
                }

                keypoints_show1.emplace_back(keypoint1.X(), keypoint1.Y(),
                                            1.0, 1.0);
                
                keypoints_show2.emplace_back(keypoint2.X(), keypoint2.Y(),
                                            1.0, 1.0);
                matches_show.emplace_back(k, k, 1);

                k++;
            }
            if(matches_show.size()==0){continue; }
            
            // draw match
            cv::Mat first_match;
            cv::drawMatches(mat1, keypoints_show1, mat2, keypoints_show2,
                            matches_show, first_match);

            // draw polar line
            // compute Fundamental Mat
            Eigen::Matrix3d K1, K2;
            // const auto camera1 = scene_graph.Camera(image1.CameraId());
            const auto camera1 = reconstruction.Camera(image1.CameraId());
            K1 << camera1.FocalLengthX(), 0.0, camera1.PrincipalPointX(),
                  0.0, camera1.FocalLengthY(), camera1.PrincipalPointY(),
                  0.0, 0.0, 1.0;
                  
            const auto camera2 = reconstruction.Camera(image2.CameraId());
            K2 << camera2.FocalLengthX(), 0.0, camera2.PrincipalPointX(),
                  0.0, camera2.FocalLengthY(), camera2.PrincipalPointY(),
                  0.0, 0.0, 1.0;
            if (!image1.HasQvecPrior() && !image2.HasTvecPrior()){
                std::cout << "no qvec prior "<< image1.ImageId() << " " << image2.ImageId() << std::endl;
            }
            Eigen::Matrix4d T_wc1 = Eigen::Matrix4d::Identity();
            T_wc1.topLeftCorner(3,3) = QuaternionToRotationMatrix(image1.QvecPrior());
            T_wc1.topRightCorner(3,1) = image1.TvecPrior();
            Eigen::Matrix4d T_wc2 = Eigen::Matrix4d::Identity();
            T_wc2.topLeftCorner(3,3) = QuaternionToRotationMatrix(image2.QvecPrior());
            T_wc2.topRightCorner(3,1) = image2.TvecPrior();

            // std::cout << "T_wc1: " << T_wc1 << "\nT_wc2 : " << T_wc2 << std::endl;

            Eigen::Matrix4d T_c1c2 = T_wc2 * T_wc1.inverse();
            Eigen::Matrix3d R_c1c2 = T_c1c2.topLeftCorner(3,3);
            Eigen::Vector3d t_c1c2 = T_c1c2.topRightCorner(3,1);
            Eigen::Matrix3d t_cross_c1c2;
            t_cross_c1c2 << 0.0, -t_c1c2.z(), t_c1c2.y(), 
                            t_c1c2.z(), 0.0, -t_c1c2.x(),
                            -t_c1c2.y(), t_c1c2.x(), 0.0;
            Eigen::Matrix3d F_eigen = K2.inverse().transpose() * t_cross_c1c2 * R_c1c2 * K1.inverse();
            cv::Mat F = cv::Mat(3, 3, CV_64FC1);
            F.at<double>(0,0) = F_eigen(0,0);
            F.at<double>(0,1) = F_eigen(0,1);
            F.at<double>(0,2) = F_eigen(0,2);
            F.at<double>(1,0) = F_eigen(1,0);
            F.at<double>(1,1) = F_eigen(1,1);
            F.at<double>(1,2) = F_eigen(1,2);
            F.at<double>(2,0) = F_eigen(2,0);
            F.at<double>(2,1) = F_eigen(2,1);
            F.at<double>(2,2) = F_eigen(2,2);

            std::vector<cv::Point2f> points1, points2;

            cv::KeyPoint::convert(keypoints_show1, points1);
            cv::KeyPoint::convert(keypoints_show2, points2);

            // cv::Mat F_cv = cv::findFundamentalMat(points1, points2, CV_FM_8POINT);
            // std::cout << F_cv << "\n F: " << F << std::endl;
            std::vector<cv::Vec<float, 3>> epilines1, epilines2;
            cv::computeCorrespondEpilines(points1, 1, F, epilines1);
            cv::computeCorrespondEpilines(points2, 2, F, epilines2);
            cv::Mat img1, img2;
            if (mat1.type() == CV_8UC3)
            {
                mat1.copyTo(img1);
                mat2.copyTo(img2);
            }
            else if (mat1.type() == CV_8UC1)
            {
                cvtColor(mat1, img1, cv::COLOR_GRAY2BGR);
                cvtColor(mat2, img2, cv::COLOR_GRAY2BGR);
            }
            else
            {
                cout << "unknow img type\n" << endl;
                exit(0);
            }
        
            cv::RNG& rng = cv::theRNG();
            int delt_num = keypoints_show2.size() / max_num_plot + 1;
            for (int i = 0; i < keypoints_show2.size(); i+=delt_num) {
                cv::Scalar color = cv::Scalar(rng(256), rng(256), rng(256));
        
                cv::circle(img2, points2[i], 15, color, 3);
                cv::line(img2, cv::Point(0, -epilines1[i][2] / epilines1[i][1]), cv::Point(img2.cols, -(epilines1[i][2] + epilines1[i][0] * img2.cols) / epilines1[i][1]), color, 3);
                cv::circle(img1, points1[i], 15, color, 3);
                cv::line(img1, cv::Point(0, -epilines2[i][2] / epilines2[i][1]), cv::Point(img1.cols, -(epilines2[i][2] + epilines2[i][0] * img1.cols) / epilines2[i][1]), color, 3);
            }
            // draw image
            const std::string ouput_image_path = JoinPaths(
                workspace_path + "/matches",
                input_image_name1 + "+" + input_image_name2);
            cv::Mat img_12, img_comb;
            cv::hconcat(img1, img2, img_12);
            cv::vconcat(first_match, img_12, img_comb);

            // cv::imwrite(ouput_image_path, img2);
            cv::imwrite(ouput_image_path, img_comb);
        
            // write_count++;
            // if(write_count>=100) break;
        }
    }
}

int main(int argc, char *argv[]) {
    using namespace sensemap;

    PrintHeading("SenseMap.  Copyright(c) 2019, SenseTime Group.");
    PrintHeading("Version: sfm-1.6.2");
    Timer timer;
    timer.Start();

    configuration_file_path = std::string(argv[1]);
    Configurator param;
    param.Load(configuration_file_path.c_str());

    std::string workspace_path = param.GetArgument("workspace_path", "");
    CHECK(!workspace_path.empty()) << "workspace path empty";

    std::string image_path = param.GetArgument("image_path", "");
    CHECK(!image_path.empty()) << "image path empty";

    int max_num_points = 10;
    std::cout << " argc: "<< argc << std::endl;
    if (argc == 3) {
        max_num_points = atoi(argv[2]);
    }

    if (!boost::filesystem::exists(workspace_path)) {
        CHECK(boost::filesystem::create_directories(workspace_path)) << "Create workspace failed";
    }

    fs = fopen((workspace_path + "/time.txt").c_str(), "w");

    auto reconstruction_manager = std::make_shared<ReconstructionManager>();
    auto scene_graph_container = std::make_shared<SceneGraphContainer>();
    auto feature_data_container = std::make_shared<FeatureDataContainer>();

    FeatureExtraction(*feature_data_container.get(), param);

    FeatureMatching(*feature_data_container.get(), *scene_graph_container.get(), param);

#if 0
    DrawMatchPloarLIne(*feature_data_container.get(),*scene_graph_container.get(), reconstruction_manager, param, max_num_points);
#else
    DrawMatchPloarLIneWithRect(*feature_data_container.get(),*scene_graph_container.get(), param, max_num_points);
#endif
    
    fclose(fs);

    return 0;
}
