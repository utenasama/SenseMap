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

#include "controllers/incremental_mapper_controller.h"

#include "../Configurator_yaml.h"
#include "../camera_rig_params.h"
#include "../option_parsing.h"

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

void ClusterMapper(const std::shared_ptr<sensemap::SceneGraphContainer> &scene_graph_container,
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

    bool with_depth = options->cluster_mapper_options.mapper_options.with_depth;
    std::string rgbd_parmas_file = param.GetArgument("rgbd_params_file", "");
    if (with_depth && !rgbd_parmas_file.empty()) {
        CalibOPPOBinReader calib_reader;
        calib_reader.ReadCalib(rgbd_parmas_file);
        options->cluster_mapper_options.mapper_options.rgbd_camera_params = calib_reader.ToParamString();
        std::cout << "rgbd_camera_params: " << options->cluster_mapper_options.mapper_options.rgbd_camera_params << std::endl;
    }

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
        LoadPirorPose(preview_pose_file, keyframes);

        std::unordered_map<std::string,Keyframe> keyframe_map;
        for(auto const keyframe:keyframes){
            keyframe_map.emplace(keyframe.name,keyframe);
        }

        std::unordered_map<image_t, Eigen::Vector4d> prior_rotations;
        std::unordered_map<image_t, Eigen::Vector3d> prior_translations;

        string camera_rig_params_file = param.GetArgument("rig_params_file", "");
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
            GPSLocationsToImgaes(gps_locations, image_names, image_locations);

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

    std::string workspace_path = param.GetArgument("workspace_path", "");
    CHECK(!workspace_path.empty()) << "image path empty";

    std::string image_path = param.GetArgument("image_path", "");
    CHECK(!workspace_path.empty()) << "image path empty";

    // use prior pose from slam to constrain SfM
    bool use_slam_graph = static_cast<bool>(param.GetArgument("use_slam_graph", 0));
    std::string preview_pose_file = param.GetArgument("preview_pose_file","");
    if (use_slam_graph && (!preview_pose_file.empty()) && boost::filesystem::exists(preview_pose_file)) {
        std::vector<Keyframe> keyframes;
        LoadPirorPose(preview_pose_file, keyframes);

        std::unordered_map<std::string,Keyframe> keyframe_map;
        for(auto const keyframe:keyframes){
            keyframe_map.emplace(keyframe.name,keyframe);
        }

        std::unordered_map<image_t, Eigen::Vector4d> prior_rotations;
        std::unordered_map<image_t, Eigen::Vector3d> prior_translations;

        string camera_rig_params_file = param.GetArgument("rig_params_file", "");
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
            GPSLocationsToImgaes(gps_locations, image_names, image_locations);
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

            sensemap::WriteBinaryPlyPoints(workspace_path+"/gps.ply", gps_locations_ply);
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

    std::string rgbd_parmas_file = param.GetArgument("rgbd_params_file", "");
    if (with_depth && !rgbd_parmas_file.empty()) {
        CalibOPPOBinReader calib_reader;
        calib_reader.ReadCalib(rgbd_parmas_file);
        options->independent_mapper_options.rgbd_camera_params = calib_reader.ToParamString();
        std::cout << "rgbd_camera_params: " << options->independent_mapper_options.rgbd_camera_params << std::endl;
    }

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
            reconstruction->AlignWithPriorLocations(options->independent_mapper_options.max_error_gps);
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

        if (camera.NumLocalCameras() > 1) {
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
        } else {
            reconstruction->WriteReconstruction(rec_path,
                                                options->independent_mapper_options.write_binary_model);
        }
    }
}

void DirectedSFM(const std::shared_ptr<sensemap::SceneGraphContainer> &scene_graph_container,
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

    MapperController *mapper =
            MapperController::Create(options, workspace_path, image_path, scene_graph_container, reconstruction_manager);
    mapper->Start();
    mapper->Wait();

    fprintf(
            fs, "%s\n",
            StringPrintf("Directed Mapper Elapsed time: %.3f [minutes]", mapper->GetTimer().ElapsedMinutes()).c_str());
    fflush(fs);

}

bool RTKReady(const std::shared_ptr<sensemap::SceneGraphContainer> &scene_graph_container, int threshold = 0.8) {
    auto image_ids = scene_graph_container->GetImageIds();
    double ready_num = 0;
    for (auto id : image_ids) {
        auto image = scene_graph_container->Image(id);
        if (image.RtkFlag() == 50) {
            ready_num++;
        }
    }
    return ready_num / image_ids.size() > threshold;
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
//    int i =0;
//    for (const auto image_id : image_ids) {
//        Image &image = scene_graph.Image(image_id);
//        std::string image_name = image.Name();
//        std::cout << image_name <<" "<< image_id <<" / "<< image_ids.size() << std::endl;
//        if (!reconstruction.IsImageRegistered(image_id)) {
//            // no prior pose, skipped
//            std::cout << image_name << "  no prior pose, skipped" << std::endl;
//            continue;
//        }
//
//        if(image_id == 71)
//            break;
//
//        image.SetQvecPrior(reconstruction.Image(image_id).Qvec());
//        image.SetTvecPrior(reconstruction.Image(image_id).Tvec());
//    }

    // draw
    if (!boost::filesystem::exists(workspace_path + "/matches")) {
        boost::filesystem::create_directories(workspace_path + "/matches");
    }
    int write_count = 0;
    // std::vector<image_t> image_ids = feature_data_container.GetImageIds();
    for(image_t id1 = 1; id1 < image_ids.size(); id1++) {
        image_t image_id1 = image_ids.at(id1);
        std::cout << "Image#" << image_id1 << std::endl;
        //const PanoramaIndexs & panorama_indices1 = feature_data_container.GetPanoramaIndexs(image_id1);
        for(image_t id2 = id1 + 1; id2 < image_ids.size(); id2++) {
            image_t image_id2 = image_ids.at(id2);

            //const PanoramaIndexs & panorama_indices2 = feature_data_container.GetPanoramaIndexs(image_id2);

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
               if(matches[m].point2D_idx1 >=keypoints1.size())
                   continue;
                auto keypoint1 =keypoints1[matches[m].point2D_idx1];

                // auto keypoints2 = feature_data_container.GetKeypoints(image_id2);
                auto keypoints2 = reconstruction.Image(image_id2).Points2D();
                if(matches[m].point2D_idx2 >=keypoints2.size())
                    continue;
                auto keypoint2 = keypoints2[matches[m].point2D_idx2];

//                CHECK_LT(matches[m].point2D_idx1, panorama_indices1.size());
//                CHECK_LT(matches[m].point2D_idx2, panorama_indices2.size());
//                if(!( panorama_indices1[matches[m].point2D_idx1].sub_image_id == i &&
//                      panorama_indices2[matches[m].point2D_idx2].sub_image_id ==j)){
//                    continue;
//                }

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
//            if (!image1.HasQvecPrior() && !image2.HasTvecPrior()){
//                std::cout << "no qvec prior "<< image1.ImageId() << " " << image2.ImageId() << std::endl;
//            }
            Eigen::Matrix4d T_wc1 = Eigen::Matrix4d::Identity();
            T_wc1.topLeftCorner(3,3) = QuaternionToRotationMatrix(image1.Qvec());
            T_wc1.topRightCorner(3,1) = image1.Tvec();
            Eigen::Matrix4d T_wc2 = Eigen::Matrix4d::Identity();
            T_wc2.topLeftCorner(3,3) = QuaternionToRotationMatrix(image2.Qvec());
            T_wc2.topRightCorner(3,1) = image2.Tvec();

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
                std::cout<< epilines1[i][1]<<" "<<epilines1[i][1] <<std::endl;

            }
            // draw image
            std::string sub_input_image_name1 = input_image_name1.substr(input_image_name1.find("/")+1);
            std::string sub_input_image_name2 = input_image_name2.substr(input_image_name2.find("/")+1);
            const std::string ouput_image_path = JoinPaths(
                    workspace_path + "/matches",
                    sub_input_image_name1 + "+" + sub_input_image_name2);
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

            //image.SetQvec(prior_qvec);
            //image.SetTvec(prior_tvec);

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
            ba_options.use_prior_absolute_location = false;
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
    PrintHeading("Version: sfm-1.6.3");
    Timer timer;
    timer.Start();


    // std::string camera_rig_params_file(argv[1]);
    // sensemap::CameraRigParams rig_params;
    // if (rig_params.LoadParams(camera_rig_params_file)) {
    //     const int width = 3840;
    //     const int height = 2880;
    //     for(size_t i = 0; i < rig_params.num_local_cameras; ++i){
    //         double fx = rig_params.local_intrinsics[i][0];
    //         double fy = rig_params.local_intrinsics[i][1];
    //         double cx = rig_params.local_intrinsics[i][2];
    //         double cy = rig_params.local_intrinsics[i][3];
    //         // local_intrinsics[i][0] = fy;
    //         // local_intrinsics[i][1] = fx;
    //         // local_intrinsics[i][2] = cy;
    //         // local_intrinsics[i][3] = width - cx;

    //         Eigen::Matrix3d R = rig_params.local_extrinsics[i].block<3,3>(0,0);
    //         Eigen::Vector3d t = rig_params.local_extrinsics[i].block<3,1>(0,3);
    //         Eigen::Vector3d r0 = R.row(0);
    //         R.row(0) = R.row
    //         en (1);
    //         R.row(1) = -r0;
    //         std::swap(t(0), t(1));
    //         t(1) = -t(1);

    //         printf("local_camera: %d\n", i);
    //         for (int j = 0; j < 3; ++j) {
    //             printf("   - [%lf, %lf, %lf, %lf]\n", R(j, 0), R(j, 1), R(j, 2), t(j));
    //         }
    //         printf("[%lf, %lf, %lf, %lf]\n", fy, fx, cy, width - cx);
    //     }
    // }

    // return 0;

#if 0
    std::vector<std::string> filelist = GetRecursiveFileList("/data/dataset/B5/pro2_image_0624");
    std::string output_image_path = "/data/dataset/B5/B5-indoor-0624-static/images/";
    for (size_t i = 0; i < filelist.size(); ++i) {
        auto image_name = filelist.at(i);
        if (image_name.find("origin") == std::string::npos) {
            continue;
        }
        Bitmap bitmap;
        bitmap.Read(image_name);

        std::cout << image_name << std::endl;
        int local_camera_idx = std::atoi(image_name.substr(image_name.length() - 5, 1).c_str()) - 1;
        size_t pos = image_name.find("PIC_");
        std::string base_name = image_name.substr(pos, image_name.length() - pos - 13);
        std::string image_path = output_image_path + "cam" + std::to_string(local_camera_idx) + "/" + base_name + ".jpg";
        std::cout << image_path << std::endl;
        std::cout << bitmap.Height() << " " << bitmap.Width() << std::endl;
        Bitmap rotated_bitmap;
        rotated_bitmap.Allocate(bitmap.Height(), bitmap.Width(), true);
        for (int r = 0; r < bitmap.Height(); ++r) {
            for (int c = 0; c < bitmap.Width(); ++c) {
                BitmapColor<uint8_t> color;
                bitmap.GetPixel(c, r, &color);
                rotated_bitmap.SetPixel(r, bitmap.Width() - c, color);
            }
        }
     
        rotated_bitmap.Write(image_path);
    }
    return 0;
#endif
    #if 0
    Reconstruction reconstruction;
    reconstruction.ReadBinary("");
    image_t image_id = reconstruction.RegisterImageIds().at(0);
    camera_t camera_id = reconstruction.Image(image_id).CameraId();
    class Camera& camera = reconstruction.Camera(camera_id);

    int num_local_camera = camera.NumLocalCameras();
    for (int i = 0; i < num_local_camera; ++i) {
        std::cout << "Local Camera " << i << std::endl;
        Eigen::Vector4d qvec;
        Eigen::Vector3d tvec;
        camera.GetLocalCameraExtrinsic(i, qvec, tvec);
        Eigen::Matrix3d R = QuaternionToRotationMatrix(qvec);

        std::cout << R(0, 0) << ", " << R(0, 1) << ", " << R(0, 2) << ", " << tvec(0) << std::endl;
        std::cout << R(1, 0) << ", " << R(1, 1) << ", " << R(1, 2) << ", " << tvec(1) << std::endl;
        std::cout << R(2, 0) << ", " << R(2, 1) << ", " << R(2, 2) << ", " << tvec(2) << std::endl;

        std::vector<double> intrinsic;
        camera.GetLocalCameraIntrisic(i, intrinsic);
        for (int j = 0; j < intrinsic.size(); ++j) {
            std::cout << intrinsic.at(j) << ", ";
        }
        std::cout << std::endl;
    }
    return 0;
    #endif

    configuration_file_path = std::string(argv[1]);
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

    FeatureExtraction(*feature_data_container.get(), param);

    FeatureMatching(*feature_data_container.get(), *scene_graph_container.get(), param);

    //DrawMatchPloarLIneWithRect(*feature_data_container.get(),*scene_graph_container.get(), param, 10);
    feature_data_container.reset();

    std::string mapper_method = param.GetArgument("mapper_method", "incremental");

//    if (mapper_method.compare("incremental") == 0) {
//        IncrementalSFM(scene_graph_container, reconstruction_manager, param);
//    } else if (mapper_method.compare("cluster") == 0) {
//        ClusterMapper(scene_graph_container, reconstruction_manager, param);
//    } else if (mapper_method.compare("global") == 0) {
        GlobalSFM(scene_graph_container, reconstruction_manager, param);
//    }

    //UpdateReconstruction(scene_graph_container, reconstruction_manager, param);

    std::cout << StringPrintf("Reconstruct in %.3fs", timer.ElapsedSeconds()) << std::endl;

    PrintReconSummary(workspace_path + "/statistic.txt", scene_graph_container->NumImages(), reconstruction_manager);

    fclose(fs);

    return 0;
}
