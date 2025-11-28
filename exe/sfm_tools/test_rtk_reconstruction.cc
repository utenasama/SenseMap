// Copyright (c) 2019, SenseTime Group.
// All rights reserved.

#include <boost/filesystem/path.hpp>
#include <opencv2/opencv.hpp>

#include "base/camera.h"
#include "base/image.h"
#include "base/image_reader.h"
#include "base/reconstruction_manager.h"
#include "base/pose.h"
#include "base/projection.h"
#include "graph/correspondence_graph.h"
#include "util/misc.h"
#include "util/mat.h"
#include "util/rgbd_helper.h"
// #include "util/ply.h"
// #include "util/obj.h"

#include "container/feature_data_container.h"
#include "feature/extraction.h"
#include "feature/feature_matcher.h"
#include "feature/utils.h"

#include "controllers/incremental_mapper_controller.h"

#include "../Configurator_yaml.h"
#include "../camera_rig_params.h"
#include "../option_parsing.h"

#include <dirent.h>
#include <sys/stat.h>

#include "util/gps_reader.h"
#include <unordered_set>

#define MAX_PRECISION std::setprecision(std::numeric_limits<double>::digits10 + 1)

using namespace sensemap;

std::string configuration_file_path;

FILE *fs;

Eigen::Vector4d camera_extri_qvec(1.0, 0.0, 0.0, 0.0);
Eigen::Vector3d camera_extri_tvec(0.0, 0.0, 0.0);

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

//void SaveNovatel2CameraExtriParam(std::shared_ptr<Reconstruction> reconstruction,
//    const camera_t camera_id, std::string rec_path){
//    // compute new T
//    Eigen::Vector4d extrinx_qvec(1.0, 0.0, 0.0, 0.0);
//    Eigen::Vector3d extrinx_tvec(0.0, 0.0, 0.0);
//    Eigen::Vector4d disturb_qvec = reconstruction->Camera(camera_id).QvecDisturb();
//    Eigen::Vector3d disturb_tvec = reconstruction->Camera(camera_id).TvecDisturb();
//    ConcatenatePoses(camera_extri_qvec, camera_extri_tvec,
//                     disturb_qvec, disturb_tvec, &extrinx_qvec, &extrinx_tvec);
//    Eigen::Matrix3x4d extrinx_T = ComposeProjectionMatrix(extrinx_qvec, extrinx_tvec);
//
//    std::string delt_path = rec_path + "/delt_T.txt";
//    std::ofstream file(delt_path, std::ios::trunc);
//    CHECK(file.is_open()) << delt_path;
//    file << "# Delt Translation (T_delt) of camera2novatel:" << std::endl;
//    file << "# M_cam = M_novatel * T^{-1} => M_cam' = M_novatel * (T_delt*T)^{-1}" << std::endl;
//    file << "# (T_delt) qw qx qy qz x y z" << std::endl;
//    file << "# (T') projection matrix 3x4" << std::endl;
//    file << "# fx fy u0 v0 k1 k2 p1 p2" << std::endl;
//    file << std::fixed << std::setprecision(9)
//        << reconstruction->Camera(camera_id).QvecDisturb()(0) << " "
//        << reconstruction->Camera(camera_id).QvecDisturb()(1) << " "
//        << reconstruction->Camera(camera_id).QvecDisturb()(2) << " "
//        << reconstruction->Camera(camera_id).QvecDisturb()(3) << " "
//        << reconstruction->Camera(camera_id).TvecDisturb()(0) << " "
//        << reconstruction->Camera(camera_id).TvecDisturb()(1) << " "
//        << reconstruction->Camera(camera_id).TvecDisturb()(2) << " \n"
//        << extrinx_T(0,0) << " " << extrinx_T(0,1) << " " << extrinx_T(0,2) << " " << extrinx_T(0,3) << " "
//        << extrinx_T(1,0) << " " << extrinx_T(1,1) << " " << extrinx_T(1,2) << " " << extrinx_T(1,3) << " "
//        << extrinx_T(2,0) << " " << extrinx_T(2,1) << " " << extrinx_T(2,2) << " " << extrinx_T(2,3) << " "
//        << 0.0 << " " << 0.0 << " " << 0.0 << " " << 1.0 << " " << "\n"
//        << reconstruction->Camera(camera_id).ParamsToString() << " " << std::endl;
//    file.close();
//
//    std::cout << " camera: " << reconstruction->Camera(camera_id).HasDisturb() << "-flag\ndelt_qvec: "
//        << reconstruction->Camera(camera_id).QvecPriorDisturb().transpose() << "  ->  "
//        << reconstruction->Camera(camera_id).QvecDisturb().transpose()
//        << "\ndelt_tvec: " << reconstruction->Camera(camera_id).TvecPriorDisturb().transpose() << "  ->  "
//        << reconstruction->Camera(camera_id).TvecDisturb().transpose() << std::endl;
//    std::cout << "extri_qvec: " << camera_extri_qvec.transpose() << "  ->  "
//        << extrinx_qvec.transpose()
//        << "\nextri_tvec: " << camera_extri_tvec.transpose() << "  ->  "
//        << extrinx_tvec.transpose() << std::endl;
//}

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

    if (use_gps_prior) {
        image_t geo_image_idx = feature_data_container.GetGeoImageIndex();
        if (feature_data_container.ExistImage(geo_image_idx)) {
            const class Image& image = feature_data_container.GetImage(geo_image_idx);
            std::string image_path = JoinPaths(reader_options.image_path, image.Name());
            Bitmap bitmap;
            bitmap.Read(image_path);
            double latitude, longitude, altitude;
            bitmap.ExifLatitude(&latitude);
            bitmap.ExifLongitude(&longitude);
            bitmap.ExifAltitude(&altitude);
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

                auto image1 = feature_data_container.GetImage(image_id1);
                auto image2 = feature_data_container.GetImage(image_id2);

                const std::string input_image_path1 = image1.Name();
                std::string input_image_name1 = input_image_path1.substr(input_image_path1.find("/")+1);
                std::cout<<input_image_name1<<std::endl;

                const std::string input_image_path2 = image2.Name();
                std::string input_image_name2 = input_image_path2.substr(input_image_path2.find("/")+1);
                std::cout<<input_image_name2<<std::endl;

                std::string name_ext1 = input_image_name1.substr(input_image_name1.size() - 3, 3);
                std::string name_ext2 = input_image_name2.substr(input_image_name2.size() - 3, 3);

                std::vector<cv::Mat> mats1;
                std::vector<cv::Mat> mats2;

                // const std::string path1 = "/data/dataset/IMX500_data/update-images";
                // const std::string path2 = "/data/dataset/IMX500_data/onex-images";

                for(int i = 0; i<1; ++i){
                    std::string full_name1, full_name2;
                    full_name1 = image_path + "/" + input_image_path1;
                    full_name2 = image_path + "/" + input_image_path2;

                    // if (name_ext1.compare("bmp") == 0) {
                    //     full_name1 = path1 + "/" + input_image_path1;
                    // } else {
                    //     std::string camera_name = "cam"+std::to_string(i)+"/";
                    //     full_name1 = path2+"/"+input_image_path1;
                    // }
                    // if (name_ext2.compare("bmp") == 0) {
                    //     full_name2 = path1 + "/" + input_image_path2;
                    // } else {
                    //     std::string camera_name = "cam"+std::to_string(i)+"/";
                    //     full_name2 = path2+"/"+input_image_path2;
                    // }
                    std::cout << "+--------------------------------+" << std::endl;
                    std::cout << full_name1 << std::endl;
                    std::cout << full_name2 << std::endl;
                    // std::string camera_name = "cam"+std::to_string(i)+"/";
                    // std::string full_name1 = image_path+"/"+input_image_path1;
                    // std::string full_name2 = image_path+"/"+input_image_path2;

                    cv::Mat mat1 = cv::imread(full_name1);
                    cv::Mat mat2 = cv::imread(full_name2);

                    mats1.push_back(mat1);
                    mats2.push_back(mat2);
                }

                for(int i=0; i<1; ++i){
                    for(int j=0; j<1; ++j){
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
                        
                        
                        cv::Mat first_match;
                        cv::drawMatches(mat1, keypoints_show1, mat2, keypoints_show2,
                                        matches_show, first_match);
                        const std::string ouput_image_path = JoinPaths(
                            workspace_path + "/matches",
                            input_image_name1 + "+" + input_image_name2);
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
//void AddNovatelPosePrior(
//    std::shared_ptr<sensemap::SceneGraphContainer> &scene_graph_container,
//    Configurator &param){
//    using namespace sensemap;
//
//    PrintHeading1("Add Novatel Prior");
//
//    // add camera param
//    std::string gnss2camera_extr_str = param.GetArgument("gnss2cam_extr", "");
//    if (!gnss2camera_extr_str.empty()){
//        std::vector<double> gn2cam_extr_vec = CSVToVector<double>(gnss2camera_extr_str);
//        Eigen::Matrix4d gnss2camera_extr;
//        CHECK(gn2cam_extr_vec.size() == 16) << gn2cam_extr_vec.size();
//        gnss2camera_extr <<
//            gn2cam_extr_vec.at(0),gn2cam_extr_vec.at(1),gn2cam_extr_vec.at(2),gn2cam_extr_vec.at(3),
//            gn2cam_extr_vec.at(4),gn2cam_extr_vec.at(5),gn2cam_extr_vec.at(6),gn2cam_extr_vec.at(7),
//            gn2cam_extr_vec.at(8),gn2cam_extr_vec.at(9),gn2cam_extr_vec.at(10),gn2cam_extr_vec.at(11),
//            gn2cam_extr_vec.at(12),gn2cam_extr_vec.at(13),gn2cam_extr_vec.at(14),gn2cam_extr_vec.at(15);
//        camera_extri_qvec = RotationMatrixToQuaternion(gnss2camera_extr.block<3, 3>(0, 0));
//        camera_extri_tvec = gnss2camera_extr.block<3, 1>(0, 3);
//    }
//    std::cout << camera_extri_qvec << " \n" << camera_extri_tvec << std::endl;
//
//    // add novatel pose
//    std::string novatel_pose_file = param.GetArgument("novatel_pose_file", "");
//    CHECK(!novatel_pose_file.empty());
//
//    std::ifstream file_input_pose(novatel_pose_file);
//    CHECK(file_input_pose.is_open());
//
//    std::unordered_map<std::string, Eigen::Matrix3x4d> map_image_name_to_pose;
//    std::string image_name;
//    while (file_input_pose >> image_name) {
//        Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
//        for (int i = 0; i < 3; ++i) {
//            for (int j = 0; j < 4; ++j) {
//                file_input_pose >> pose(i, j);
//            }
//        }
//        map_image_name_to_pose.emplace(image_name + ".jpg", pose.inverse().topRows(3));
//    }
//    file_input_pose.close();
//
//    int num_prior_images = 0;
//    std::vector<image_t> image_ids = scene_graph_container->GetImageIds();
//    for (const auto image_id : image_ids) {
//        auto& image = scene_graph_container->Image(image_id);
//        auto& camera = scene_graph_container->Camera(image.CameraId());
//
//        std::string image_name = image.Name();
//        if (map_image_name_to_pose.find(image_name) == map_image_name_to_pose.end()) {
//            // no prior pose, skipped
//            // std::cout << image_name << "  no prior pose, skipped" << std::endl;
//            continue;
//        }
//        if (!camera.HasDisturb()){
//            camera.SetDisturb();
//            std::cout << "camera id " << image.CameraId() << " disturb-flag: " << camera.HasDisturb()
//                    << "\ncamera_extri_disturb: " << camera.QvecDisturb().transpose()
//                    << " / " << camera.TvecDisturb().transpose()
//                    << "\nparam: " << camera.ParamsToString() << std::endl;
//        }
//
//        Eigen::Matrix3x4d pose = map_image_name_to_pose.at(image_name);
//        Eigen::Vector4d prior_qvec = RotationMatrixToQuaternion(pose.block<3, 3>(0, 0));
//        Eigen::Vector3d prior_tvec = pose.block<3, 1>(0, 3);
//        // Eigen::Vector4d delt_qvec = camera.QvecDisturb();
//        // Eigen::Vector3d delt_tvec = camera.TvecDisturb();
//        Eigen::Vector4d qvec;
//        Eigen::Vector3d tvec;
//        ConcatenatePoses(prior_qvec, prior_tvec, camera_extri_qvec, camera_extri_tvec, &qvec, &tvec);
//
//        image.SetQvecPrior(qvec);
//        image.SetTvecPrior(tvec);
//
//        num_prior_images++;
//    }
//    std::cout << "Has prior pose : " << num_prior_images << " images." << std::endl;
//}

void IncrementalSFM(std::shared_ptr<sensemap::SceneGraphContainer> &scene_graph_container,
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

    std::string workspace_path = param.GetArgument("workspace_path", "");
    CHECK(!workspace_path.empty()) << "workspace path empty";

    std::string image_path = param.GetArgument("image_path", "");
    CHECK(!image_path.empty()) << "image path empty";

    // rgbd mode
    int num_local_cameras = reader_options.num_local_cameras;
    bool with_depth = options->independent_mapper_options.with_depth;
    options->independent_mapper_options.extract_keyframe = static_cast<bool>(param.GetArgument("extract_keyframe", 0));
    // options->independent_mapper_options.register_nonkeyframe = false;

    MapperController *mapper = MapperController::Create(options, workspace_path, image_path, 
                                                        scene_graph_container, reconstruction_manager);
    mapper->Start();
    mapper->Wait();

    fprintf(
        fs, "%s\n",
        StringPrintf("Incremental Mapper Elapsed time: %.3f [minutes]", mapper->GetTimer().ElapsedMinutes()).c_str());
    fflush(fs);

    for (size_t i = 0; i < reconstruction_manager->Size(); ++i) {
        auto reconstruction_ori = reconstruction_manager->Get(i);
        auto reconstruction = *reconstruction_ori;

        CHECK(reconstruction.RegisterImageIds().size()>0);
        const image_t first_image_id = reconstruction.RegisterImageIds()[0]; 
        CHECK(reconstruction.ExistsImage(first_image_id));
        const Image& image = reconstruction.Image(first_image_id);
        const Camera& camera = reconstruction.Camera(image.CameraId());
        
        std::string rec_path = StringPrintf("%s/%d-sfm", workspace_path.c_str(), i);
        if (!boost::filesystem::exists(rec_path)) {
            boost::filesystem::create_directories(rec_path);
        }

        reconstruction.WriteReconstruction(rec_path,
            options->independent_mapper_options.write_binary_model);
    }
}

IndependentMapperOptions InitMapperOptions(Configurator &param){
    IndependentMapperOptions mapper_options;
    mapper_options.single_camera = static_cast<bool>(param.GetArgument("single_camera", 0));
    mapper_options.ba_global_use_pba = static_cast<bool>(param.GetArgument("ba_global_use_pba", 0));
    mapper_options.ba_refine_focal_length = static_cast<bool>(param.GetArgument("ba_refine_focal_length", 1));
    mapper_options.ba_refine_extra_params = static_cast<bool>(param.GetArgument("ba_refine_extra_params", 1));
    mapper_options.ba_refine_principal_point = static_cast<bool>(param.GetArgument("ba_refine_principal_point", 0));
    mapper_options.ba_global_pba_gpu_index = static_cast<int>(param.GetArgument("ba_global_pba_gpu_index", -1));
    mapper_options.ba_global_max_num_iterations = static_cast<int>(param.GetArgument("ba_global_max_num_iterations", 30));
    mapper_options.ba_global_max_refinements = static_cast<int>(param.GetArgument("ba_global_max_refinements", 2));
    mapper_options.ba_global_loss_function = param.GetArgument("ba_global_loss_function", "trival");
    mapper_options.ba_global_images_ratio = param.GetArgument("ba_global_images_ratio", 1.1f);
    mapper_options.ba_global_points_ratio = param.GetArgument("ba_global_points_ratio", 1.1f);
    mapper_options.batched_sfm = static_cast<bool>(param.GetArgument("batched_sfm", 0));
    mapper_options.local_ba_batched = static_cast<bool>(param.GetArgument("local_ba_batched", 1));
    mapper_options.ba_local_num_images = param.GetArgument("ba_local_num_images", 6);
    mapper_options.init_image_id1 = static_cast<int>(param.GetArgument("init_image_id1", -1));
    mapper_options.init_image_id2 = static_cast<int>(param.GetArgument("init_image_id2", -1));
    mapper_options.init_from_uncertainty = static_cast<bool>(param.GetArgument("init_from_uncertainty", 1));
    mapper_options.init_min_num_inliers = param.GetArgument("init_min_num_inliers", 200);
    mapper_options.init_min_tri_angle = param.GetArgument("init_min_tri_angle", 12.0f);
    mapper_options.filter_max_reproj_error = param.GetArgument("filter_max_reproj_error", 4.0f);
    mapper_options.merge_max_reproj_error = param.GetArgument("merge_max_reproj_error", 4.0f);
    mapper_options.complete_max_reproj_error = param.GetArgument("complete_max_reproj_error", 4.0f);
    mapper_options.min_tri_angle = param.GetArgument("min_tri_angle", 1.5f);
    mapper_options.abs_pose_min_num_inliers = param.GetArgument("abs_pose_min_num_inliers", 30);
    mapper_options.min_inlier_ratio_to_best_pose = param.GetArgument("min_inlier_ratio_to_best_pose", 0.7f);
    mapper_options.min_inlier_ratio_verification_with_prior_pose = param.GetArgument("min_inlier_ratio_verification_with_prior_pose", 0.7f);
    mapper_options.num_images_for_self_calibration = param.GetArgument("num_images_for_self_calibration", 200);
    mapper_options.extract_keyframe = static_cast<bool>(param.GetArgument("extract_keyframe", 0));
    mapper_options.register_nonkeyframe = static_cast<bool>(param.GetArgument("register_nonkeyframe", 0));
    mapper_options.num_first_force_be_keyframe = static_cast<int>(param.GetArgument("num_first_force_be_keyframe", 10));
    mapper_options.optim_inner_cluster = static_cast<bool>(param.GetArgument("optim_inner_cluster", 0));
    mapper_options.robust_camera_pose_estimate = static_cast<bool>(param.GetArgument("robust_camera_pose_estimate", 0));
    mapper_options.consecutive_camera_pose_top_k = static_cast<int>(param.GetArgument("consecutive_camera_pose_top_k", 2));
    mapper_options.consecutive_neighbor_ori = static_cast<int>(param.GetArgument("consecutive_neighbor_ori", 2));
    mapper_options.consecutive_neighbor_t = static_cast<int>(param.GetArgument("consecutive_neighbor_t", 1));
    mapper_options.consecutive_camera_pose_orientation = param.GetArgument("consecutive_camera_pose_orientation", 5.0f);
    mapper_options.consecutive_camera_pose_t = param.GetArgument("consecutive_camera_pose_t", 20.0f);
    mapper_options.min_inlier_ratio_to_best_model = param.GetArgument("min_inlier_ratio_to_best_model", 0.8f);
    mapper_options.local_region_repetitive = param.GetArgument("local_region_repetitive", 100);
    mapper_options.num_fix_camera_first = static_cast<int>(param.GetArgument("num_fix_camera_first", 10));
    mapper_options.max_triangulation_angle_degrees = static_cast<double>(param.GetArgument("max_triangulation_angle_degrees", 30.0f));
    mapper_options.min_visible_map_point_kf = static_cast<int>(param.GetArgument("min_visible_map_point_kf", 300));
    mapper_options.min_pose_inlier_kf = static_cast<int>(param.GetArgument("min_pose_inlier_kf", 200));
    mapper_options.avg_min_dist_kf_factor = static_cast<double>(param.GetArgument("avg_min_dist_kf_factor", 1.0f));
    mapper_options.mean_max_disparity_kf = static_cast<double>(param.GetArgument("mean_max_disparity_kf", 20.0f));
    mapper_options.abs_diff_kf = static_cast<int>(param.GetArgument("abs_diff_kf", 10));
    mapper_options.debug_info = static_cast<bool>(param.GetArgument("debug_info", 0));
    mapper_options.write_binary_model = static_cast<bool>(param.GetArgument("write_binary", 1));
    mapper_options.prior_absolute_location_weight =  static_cast<double>(param.GetArgument("prior_absolute_location_weight", 1.0f));
    
    return mapper_options;
}

//void RefineExtrincParam(const std::shared_ptr<sensemap::SceneGraphContainer> &scene_graph_container,
//                        std::shared_ptr<sensemap::ReconstructionManager> &reconstruction_manager,
//                        Configurator &param) {
//
//    std::string workspace_path = param.GetArgument("workspace_path", "");
//    CHECK(!workspace_path.empty()) << "workspace path empty";
//
//    IndependentMapperOptions mapper_options = InitMapperOptions(param);
//    // int min_track_length = static_cast<int>(param.GetArgument("min_track_length", 3));
//    // const auto tri_options = mapper_options.Triangulation();
//
//    BundleAdjustmentConfig ba_config;
//    auto ba_options = mapper_options.GlobalBundleAdjustment();
//    ba_options.refine_focal_length = false;
//    ba_options.refine_principal_point = false;
//    ba_options.refine_extra_params = false;
//    ba_options.refine_extrinsics = true;
//
//    for (size_t i = 0; i < reconstruction_manager->Size(); ++i) {
//
//        auto ba_options = mapper_options.GlobalBundleAdjustment();
//        ba_options.refine_focal_length = false;
//        ba_options.refine_principal_point = false;
//        ba_options.refine_extra_params = false;
//        ba_options.refine_extrinsics = false;
//
//        auto reconstruction = reconstruction_manager->Get(i);
//        reconstruction->AlignWithPriorLocations(50);
//
//        std::string rec_path_align = StringPrintf("%s/%d-align", workspace_path.c_str(), i);
//        if (!boost::filesystem::exists(rec_path_align)) {
//            boost::filesystem::create_directories(rec_path_align);
//        }
//        auto reconstruction_align = *reconstruction;
//        reconstruction_align.AddPriorToResult();
//        reconstruction_align.WriteReconstruction(rec_path_align, true);
//
//        const auto& rec_image_ids = reconstruction->Images();
//        for (const auto& rec_image : rec_image_ids) {
//            const image_t image_id = rec_image.first;
//            auto& image = reconstruction->Image(image_id);
//            if (!(image.HasQvecPrior() && image.HasTvecPrior())) {
//                // no prior pose, skipped
//                reconstruction->DeRegisterImage(image_id);
//                std::cout << image.Name() << "  no prior pose, skipped" << std::endl;
//                continue;
//            }
//
//            auto& camera = reconstruction->Camera(image.CameraId());
//            CHECK(camera.HasDisturb());
//
//            Eigen::Vector4d prior_qvec = image.QvecPrior();
//            Eigen::Vector3d prior_tvec = image.TvecPrior();
//            Eigen::Vector4d delt_qvec = camera.QvecDisturb();
//            Eigen::Vector3d delt_tvec = camera.TvecDisturb();
//            Eigen::Vector4d qvec;
//            Eigen::Vector3d tvec;
//            ConcatenatePoses(prior_qvec, prior_tvec, delt_qvec, delt_tvec, &qvec, &tvec);
//            image.SetQvec(qvec);
//            image.SetTvec(tvec);
//
//            ba_config.AddGNSS(image_id);
//        }
//
//        // for (int i = 0; i < mapper_options.ba_global_max_refinements; ++i) {
//        for (int i = 0; i < 1; ++i) {
//            reconstruction->FilterObservationsWithNegativeDepth();
//
//            const size_t num_observations = reconstruction->ComputeNumObservations();
//
//            PrintHeading1("Novatel Bundle adjustment");
//            std::cout << "iter: " << i << std::endl;
//            BundleAdjuster bundle_adjuster(ba_options, ba_config);
//            CHECK(bundle_adjuster.Solve(reconstruction.get()));
//
//            std::cout << " camera: " << reconstruction->Camera(1).HasDisturb() << "-flag\nqvec: "
//                << reconstruction->Camera(1).QvecDisturb().transpose() << "  ->  "
//                << reconstruction->Camera(1).QvecPriorDisturb().transpose()
//                << "\ntvec: " << reconstruction->Camera(1).TvecDisturb().transpose() << "  ->  "
//                << reconstruction->Camera(1).TvecPriorDisturb().transpose() << std::endl;
//        }
//
//        std::string rec_path = StringPrintf("%s/%d-refine", workspace_path.c_str(), i);
//        if (!boost::filesystem::exists(rec_path)) {
//            boost::filesystem::create_directories(rec_path);
//        }
//        auto reconstruction_copy = *reconstruction;
//        reconstruction_copy.AddPriorToResult();
//        reconstruction_copy.WriteReconstruction(rec_path, true);
//
//        SaveNovatel2CameraExtriParam(reconstruction, 1, rec_path);
//    }
//
//    return;
//}

void UpdateReconstruction(const std::shared_ptr<sensemap::SceneGraphContainer> &scene_graph_container,
                        std::shared_ptr<sensemap::ReconstructionManager> &reconstruction_manager, 
                        Configurator &param) {

    Timer timer;
    timer.Start();

    std::string workspace_path = param.GetArgument("workspace_path", "");
    CHECK(!workspace_path.empty()) << "workspace path empty";

    std::string image_path = param.GetArgument("image_path", "");
    CHECK(!image_path.empty()) << "image path empty";

    IndependentMapperOptions mapper_options = InitMapperOptions(param);
    int min_track_length = static_cast<int>(param.GetArgument("min_track_length", 3));
    const auto tri_options = mapper_options.Triangulation();


    //0 : only trianglution  1 : tri + gba  2 : tri + lba + gba 3. tri all + gba
    int direct_mapper_type =  param.GetArgument("direct_mapper_type", 0);


    if (reconstruction_manager->Size() == 0){
        reconstruction_manager->Add();
    }
    
    for (size_t i = 0; i < reconstruction_manager->Size(); ++i) {
        auto reconstruction = reconstruction_manager->Get(i);
        std::shared_ptr<IncrementalMapper> mapper = std::make_shared<IncrementalMapper>(scene_graph_container);
        mapper->BeginReconstruction(reconstruction);

        for(auto pointid : reconstruction->MapPointIds()){
            reconstruction->DeleteMapPoint(pointid);
        }

        int triangulated_image_count = 1;
        std::vector<image_t> image_ids = scene_graph_container->GetImageIds();
        for (const auto image_id : image_ids) {
            Image &image_scene = scene_graph_container->Image(image_id);
            if (reconstruction->ExistsImage(image_scene.ImageId()) && 
                reconstruction->IsImageRegistered(image_scene.ImageId())){
                reconstruction->DeRegisterImage(image_scene.ImageId());
                //continue;
            }
            if (!reconstruction->ExistsImage(image_id)){
                reconstruction->AddImage(image_scene);
            }
            reconstruction->RegisterImage(image_id);
            Image &image = reconstruction->Image(image_id);

            Camera &camera = reconstruction->Camera(image.CameraId());
            bool has_prior = image.HasQvecPrior() && image.HasTvecPrior();// && camera.HasDisturb();
            if (!has_prior){
                std::cout << "no find prior!" << std::endl;
                return;
            }
            Eigen::Vector4d prior_qvec = image.QvecPrior();
            Eigen::Vector3d prior_tvec = QuaternionToRotationMatrix(prior_qvec) * -image.TvecPrior();

           image.SetQvec(prior_qvec);
           image.SetTvec(prior_tvec);

            if(direct_mapper_type != 3) {
                PrintHeading1(StringPrintf("Triangulating image #%d - %s (%d / %d)",
                                           image_id, image.Name().c_str(), triangulated_image_count++,
                                           image_ids.size()));
                const size_t num_existing_points3D = image.NumMapPoints();
                std::cout << "  => Image sees " << num_existing_points3D << " / " << image.NumObservations()
                          << " points"
                          << std::endl;

                mapper->TriangulateImage(tri_options, image_id);

                std::cout << "  => Triangulated " << (image.NumMapPoints() - num_existing_points3D) << " points"
                          << std::endl;

                if (direct_mapper_type >= 2) {
                    auto ba_options = mapper_options.LocalBundleAdjustment();

                    ba_options.refine_focal_length = false;
                    ba_options.refine_principal_point = false;
                    ba_options.refine_extra_params = false;
                    ba_options.refine_extrinsics = false;

                    for (int i = 0; i < 1; ++i) {
                        reconstruction->FilterObservationsWithNegativeDepth();
                        const auto report =
                                mapper->AdjustLocalBundle(mapper_options.IncrementalMapperOptions(), ba_options,
                                                          mapper_options.Triangulation(), image_id,
                                                          mapper->GetModifiedMapPoints());
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
                    mapper->ClearModifiedMapPoints();
                }
            }
        }

        if(0){
            std::vector<image_t> reg_image_ids = reconstruction->RegisterImageIds();
            for (size_t i = 0; i < reg_image_ids.size(); ++i) {
                if (mapper_options.extract_colors) {
                    ExtractColors(image_path, reg_image_ids[i], reconstruction);
                }
            }

            std::string rec_path = StringPrintf("%s/%d-tri", workspace_path.c_str(), i);
            if (!boost::filesystem::exists(rec_path)) {
                boost::filesystem::create_directories(rec_path);
            }
            reconstruction->WriteReconstruction(rec_path, true);
            
            std::string prior_rec_path = rec_path + "-tri-prior";
            if (!boost::filesystem::exists(prior_rec_path)) {
                boost::filesystem::create_directories(prior_rec_path);
            }
            auto reconstruction_copy = *reconstruction;
            reconstruction_copy.AddPriorToResult();
            reconstruction_copy.WriteReconstruction(prior_rec_path, true);

        }
        
        //////////////////////////////////////////////////////////////////////////////
        // Retriangulation
        //////////////////////////////////////////////////////////////////////////////
        if(direct_mapper_type == 3) {
            PrintHeading1("Triangulation All");
            int max_cover_per_view = static_cast<int>(param.GetArgument("max_cover_per_view", 800));

            auto tracks = scene_graph_container->CorrespondenceGraph()->GenerateTracks(3, true, true);
            std::vector<unsigned char> inlier_masks(tracks.size(), 1);
            scene_graph_container->CorrespondenceGraph()->TrackSelection(tracks,
                    inlier_masks, max_cover_per_view);
            for (auto track : tracks) {
                reconstruction->AddMapPoint(Eigen::Vector3d(0, 0, 0), std::move(track));
            }
            mapper->RetriangulateAllTracks(tri_options);
        } else {
            PrintHeading1("Retriangulation");
            CompleteAndMergeTracks(mapper_options, mapper);
            if (direct_mapper_type < 1)
                FilterPoints(mapper_options, mapper, min_track_length);
        }
        std::vector<image_t> reg_image_ids = reconstruction->RegisterImageIds();
        //////////////////////////////////////////////////////////////////////////////
        // Bundle adjustment
        //////////////////////////////////////////////////////////////////////////////

        if(direct_mapper_type >= 1) {
            auto ba_options = mapper_options.GlobalBundleAdjustment();
            ba_options.refine_focal_length = true;
            ba_options.refine_principal_point = true;
            ba_options.refine_extra_params = true;
            ba_options.refine_extrinsics = true;

            reconstruction->b_aligned = true;
            ba_options.use_prior_absolute_location = true;
            // ba_options.prior_absolute_location_weight = 0.1;
            std::cout << "prior_absolute_location_weight: " << ba_options.prior_absolute_location_weight 
                      << "\t loss function: " << mapper_options.ba_global_loss_function << std::endl;

            // Configure bundle adjustment.
            BundleAdjustmentConfig ba_config;

            for (size_t i = 0; i < reg_image_ids.size(); ++i) {
                const image_t image_id = reg_image_ids[i];
                if (!scene_graph_container->CorrespondenceGraph()->ExistsImage(image_id)) {
                    continue;
                }
                //ba_config.AddGNSS(image_id);
                ba_config.AddImage(image_id);
            }

            for (int i = 0; i < mapper_options.ba_global_max_refinements; ++i) {
                reconstruction->FilterObservationsWithNegativeDepth();

                const size_t num_observations = reconstruction->ComputeNumObservations();

                PrintHeading1("RTK Bundle adjustment");
                std::cout << "iter: " << i << std::endl;
                BundleAdjuster bundle_adjuster(ba_options, ba_config);
                CHECK(bundle_adjuster.Solve(reconstruction.get()));

                size_t num_changed_observations = 0;
                num_changed_observations += CompleteAndMergeTracks(mapper_options, mapper);
                num_changed_observations += FilterPoints(mapper_options, mapper, min_track_length);
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
//            std::cout << " camera: " << reconstruction->Camera(1).HasDisturb() << "-flag\nqvec: "
//                << reconstruction->Camera(1).QvecDisturb().transpose() << "  ->  "
//                << reconstruction->Camera(1).QvecPriorDisturb().transpose()
//                << "\ntvec: " << reconstruction->Camera(1).TvecDisturb().transpose() << "  ->  "
//                << reconstruction->Camera(1).TvecPriorDisturb().transpose() << std::endl;
            }
        }

        for (size_t i = 0; i < reg_image_ids.size(); ++i) {
            if (mapper_options.extract_colors) {
                ExtractColors(image_path, reg_image_ids[i], reconstruction);
            }
        }

        fprintf(
            fs, "%s\n",
            StringPrintf("SFM Elapsed time: %.3f [minutes]", timer.ElapsedMinutes()).c_str());
        fflush(fs);


        std::string rec_path = StringPrintf("%s/%d", workspace_path.c_str(), i);
        if (!boost::filesystem::exists(rec_path)) {
            boost::filesystem::create_directories(rec_path);
        }

        // Eigen::Matrix3x4d matrix_to_align =
        //         reconstruction->AlignWithPriorLocations(mapper_options.max_error_gps);
        Reconstruction rec = *reconstruction.get();

        reconstruction->WriteReconstruction(rec_path, true);

        rec.AddPriorToResult();
        rec.NormalizeWoScale();

        std::string trans_rec_path = rec_path + "-gps";
        if (!boost::filesystem::exists(trans_rec_path)) {
            boost::filesystem::create_directories(trans_rec_path);
        }
        rec.WriteBinary(trans_rec_path);

        Eigen::Matrix3x4d matrix_to_geo = reconstruction->NormalizeWoScale();
        Eigen::Matrix4d h_matrix_to_align = Eigen::Matrix4d::Identity();
        // h_matrix_to_align.block<3, 4>(0, 0) = matrix_to_align;
        Eigen::Matrix3x4d M = matrix_to_geo * h_matrix_to_align;

//        std::ofstream file((rec_path + "/matrix_to_gps.txt"), std::ofstream::out);
//        file << MAX_PRECISION << M(0, 0) << " " << M(0, 1) << " "
//             << M(0, 2) << " " << M(0, 3) << std::endl;
//        file << MAX_PRECISION << M(1, 0) << " " << M(1, 1) << " "
//             << M(1, 2) << " " << M(1, 3) << std::endl;
//        file << MAX_PRECISION << M(2, 0) << " " << M(2, 1) << " "
//             << M(2, 2) << " " << M(2, 3) << std::endl;
//        file.close();

        rec.WriteReconstruction(trans_rec_path, true);

        fprintf(
            fs, "%s\n",
            StringPrintf("SFM & Save Elapsed time: %.3f [minutes]", timer.ElapsedMinutes()).c_str());
        fflush(fs);

        //SaveNovatel2CameraExtriParam(reconstruction, 1, prior_rec_path);
    }
    std::cout << StringPrintf("sfm :Reconstruct in %.3fs", timer.ElapsedSeconds()) << std::endl;
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

    bool extri_opt = false;
    if (argc == 3){
        extri_opt = atoi(argv[2]);
    }

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

    FeatureExtraction(*feature_data_container.get(), param);

    FeatureMatching(*feature_data_container.get(), *scene_graph_container.get(), param);

    feature_data_container.reset();

//    AddNovatelPosePrior(scene_graph_container, param);
//
//    if (extri_opt){
        // IncrementalSFM(scene_graph_container, reconstruction_manager, param);
//
//        RefineExtrincParam(scene_graph_container, reconstruction_manager, param);
//    }

    UpdateReconstruction(scene_graph_container, reconstruction_manager, param);

    fprintf(
        fs, "%s\n",
        StringPrintf("Reconstruct time: %.3f [minutes]", timer.ElapsedMinutes()).c_str());
    fflush(fs);

    std::cout << StringPrintf("Reconstruct in %.3fs", timer.ElapsedSeconds()) << std::endl;

    PrintReconSummary(workspace_path + "/statistic.txt", scene_graph_container->NumImages(), reconstruction_manager);

    fclose(fs);

    return 0;
}
