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
#include "util/proc.h"
#include "util/mat.h"
#include "util/misc.h"
#include "util/ply.h"
#include "util/rgbd_helper.h"
#include "base/version.h"

using namespace sensemap;
FILE *fs;

bool LoadFeatures(FeatureDataContainer &feature_data_container, Configurator &param) {
    std::string workspace_path = param.GetArgument("workspace_path", "");
    if (static_cast<bool>(param.GetArgument("map_update", 0))){
        workspace_path = JoinPaths(workspace_path, "/map_update");
    }
    CHECK(!workspace_path.empty());

    OptionParser option_parser;
    ImageReaderOptions reader_options;
    option_parser.GetImageReaderOptions(reader_options, param);

    bool camera_rig = (reader_options.num_local_cameras > 1);
    bool with_depth = reader_options.with_depth;

    std::string rgbd_parmas_file = param.GetArgument("rgbd_params_file", "");
    bool use_gps_prior = param.GetArgument("use_gps_prior", 0);

    bool have_matched = boost::filesystem::exists(JoinPaths(workspace_path, "/scene_graph.bin"));

    bool exist_feature_file = false;
    if (boost::filesystem::exists(JoinPaths(workspace_path, "/cameras.bin")) &&
        boost::filesystem::exists(JoinPaths(workspace_path, "/features.bin"))) {
        if (boost::filesystem::exists(JoinPaths(workspace_path, "/local_cameras.bin"))) {
            feature_data_container.ReadCamerasBinaryData(JoinPaths(workspace_path, "/cameras.bin"), false);
            feature_data_container.ReadLocalCamerasBinaryData(JoinPaths(workspace_path, "/local_cameras.bin"));
        } else {
            feature_data_container.ReadCamerasBinaryData(JoinPaths(workspace_path, "/cameras.bin"), camera_rig);
        }
        if (!have_matched) {
            feature_data_container.ReadImagesBinaryData(JoinPaths(workspace_path, "/features.bin"));
        } else {
            feature_data_container.ReadImagesBinaryDataWithoutDescriptor(JoinPaths(workspace_path, "/features.bin"));
        }
        if (boost::filesystem::exists(JoinPaths(workspace_path, "/local_images.bin"))) {
            feature_data_container.ReadLocalImagesBinaryData(JoinPaths(workspace_path, "/local_images.bin"));
        }
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
    if (static_cast<bool>(param.GetArgument("map_update", 0))){
        workspace_path = JoinPaths(workspace_path, "/map_update");
    }
    CHECK(!workspace_path.empty());

    if (boost::filesystem::exists(JoinPaths(workspace_path, "/vlad_vectors.bin"))){
        feature_data_container.ReadGlobalFeaturesBinaryData(workspace_path + "/vlad_vectors.bin");
        return true;
    }
    else{
        return false;
    }
}

void ShowFeatureMatching(FeatureDataContainer &feature_data_container, SceneGraphContainer &scene_graph,
                     Configurator &param) {

    std::string workspace_path = param.GetArgument("workspace_path", "");
    if (static_cast<bool>(param.GetArgument("map_update", 0))){
        workspace_path = JoinPaths(workspace_path, "/map_update");
    }

    if (!boost::filesystem::exists(workspace_path + "/matches")) {
        boost::filesystem::create_directories(workspace_path + "/matches");
    }
    int write_count = 0;
    std::vector<image_t> image_ids = feature_data_container.GetImageIds();
    for (image_t id1 = 0; id1 < image_ids.size(); id1++) {
        image_t image_id1 = image_ids[id1];
        std::cout << "Image#" << image_id1 << " ";
        const PanoramaIndexs &panorama_indices1 = feature_data_container.GetPanoramaIndexs(image_id1);
        const Camera& camera1  = feature_data_container.GetCamera(feature_data_container.GetImage(image_id1).CameraId());
        for (image_t id2 = id1 + 1; id2 < image_ids.size(); id2++) {
            image_t image_id2 = image_ids[id2];
            std::cout << "Image#" << image_id2 << std::endl;
            const PanoramaIndexs &panorama_indices2 = feature_data_container.GetPanoramaIndexs(image_id2);

            auto matches = scene_graph.CorrespondenceGraph()->FindCorrespondencesBetweenImages(image_id1, image_id2);
            if (matches.empty()) {
                continue;
            }
            std::string image_path = param.GetArgument("image_path", "");

            auto image1 = feature_data_container.GetImage(image_id1);
            auto image2 = feature_data_container.GetImage(image_id2);

            std::vector<cv::Mat> mats1;
            std::vector<cv::Mat> mats2;

            for (int i = 0; i < camera1.NumLocalCameras(); ++i) {
                
                const std::string input_image_path1 = image1.LocalName(i);
                std::string input_image_name1 = input_image_path1.substr(input_image_path1.find("/") + 1);

                const std::string input_image_path2 = image2.LocalName(i);
                std::string input_image_name2 = input_image_path2.substr(input_image_path2.find("/") + 1);

                std::string full_name1, full_name2;
                full_name1 = image_path + "/" + input_image_path1;
                full_name2 = image_path + "/" + input_image_path2;

                // if(camera1.NumLocalCameras() > 1){
                
                //     std::string camera_name = "cam"+std::to_string(i)+"/";
                //     full_name1 = image_path + "/" + camera_name + input_image_name1;
                //     full_name2 = image_path + "/" + camera_name + input_image_name2;
                // }
                
                std::cout << "+--------------------------------+" << std::endl;
                std::cout << full_name1 << std::endl;
                std::cout << full_name2 << std::endl;
             

                cv::Mat mat1 = cv::imread(full_name1);
                cv::Mat mat2 = cv::imread(full_name2);

                mats1.push_back(mat1);
                mats2.push_back(mat2);
            }

            for (int i = 0; i < camera1.NumLocalCameras(); ++i) {
                const std::string input_image_path1 = image1.LocalName(i);
                std::string input_image_name1 = input_image_path1.substr(input_image_path1.find("/") + 1);

                for (int j = 0; j < camera1.NumLocalCameras(); ++j) {
                    cv::Mat &mat1 = mats1[i];
                    cv::Mat &mat2 = mats2[j];

                    std::vector<cv::KeyPoint> keypoints_show1, keypoints_show2;
                    std::vector<cv::DMatch> matches_show;

                    int k = 0;
                    for (int m = 0; m < matches.size(); ++m) {
                        auto &keypoints1 = feature_data_container.GetKeypoints(image_id1);
                        CHECK_LT(matches[m].point2D_idx1, keypoints1.size());
                        auto keypoint1 = keypoints1[matches[m].point2D_idx1];

                        auto keypoints2 = feature_data_container.GetKeypoints(image_id2);
                        CHECK_LT(matches[m].point2D_idx2, keypoints2.size());
                        auto keypoint2 = keypoints2[matches[m].point2D_idx2];

                        CHECK_LT(matches[m].point2D_idx1, panorama_indices1.size());
                        CHECK_LT(matches[m].point2D_idx2, panorama_indices2.size());
                        
                        if (!(panorama_indices1[matches[m].point2D_idx1].sub_image_id == i &&
                              panorama_indices2[matches[m].point2D_idx2].sub_image_id == j)) {
                            continue;
                        }

                        keypoints_show1.emplace_back(keypoint1.x, keypoint1.y, keypoint1.ComputeScale(),
                                                     keypoint1.ComputeOrientation());

                        keypoints_show2.emplace_back(keypoint2.x, keypoint2.y, keypoint2.ComputeScale(),
                                                     keypoint2.ComputeOrientation());
                        matches_show.emplace_back(k, k, 1);
                        k++;
                    }
                    if (matches_show.size() == 0) {
                        continue;
                    }

                    const std::string input_image_path2 = image2.LocalName(j);
                    std::string input_image_name2 = input_image_path2.substr(input_image_path2.find("/") + 1);

                    cv::Mat first_match;
                    cv::drawMatches(mat1, keypoints_show1, mat2, keypoints_show2, matches_show, first_match);
                    const std::string ouput_image_path = JoinPaths(
                        workspace_path + "/matches", input_image_name1.substr(0, input_image_name1.size() - 4) + "_" +
                                                         input_image_name2.substr(0, input_image_name2.size() - 4) +
                                                         "_" + std::to_string(i) + "_" + std::to_string(j) + ".jpg");
                    std::cout<<"output_image_path: "<<ouput_image_path<<std::endl;

                    if (!boost::filesystem::exists(GetParentDir(ouput_image_path))) {
                        boost::filesystem::create_directories(GetParentDir(ouput_image_path));
                    }

                    cv::putText(first_match, StringPrintf("%d,%f", matches.size(), 
                        scene_graph.CorrespondenceGraph()->ImagePair(image_id1, image_id2).two_view_geometry.confidence), cv::Point(200, 200), cv::FONT_HERSHEY_COMPLEX, 8, (200, 0, 0), 2, 8, 0);

                    cv::imwrite(ouput_image_path, first_match);
                }
            }
        }
    }
}

bool IsBluetoothFile(std::string filePath) {
    std::ifstream infile;
    infile.open(filePath, std::ios::in);
    if (!infile.is_open()) {
        std::cout << "Cant to Open file: " << filePath << std::endl;
        return false;
    }
    std::vector<std::string> items = StringSplit(filePath, ".");
    if (items.back() != "txt" && items.back() != "csv") {
        std::cout << filePath << "is not .txt or csv file" << std::endl;
        return false;
    }
    int count = 0;
    std::string line;
    while (getline(infile, line)) {
        if (line.size() == 0) {
            continue;
        } 
        items = StringSplit(line, ",");
        if (items.size() < 9) {
            std::cout << filePath << " is not Bluetooth file" << std::endl;
            return false;
        }
        if (std::stoi(items[8]) > -200 && std::stoi(items[8]) < 200){
             count++;
        }

        if (count > 5) {
            return true;
        }
    }
    return false;
}

std::vector<std::string> GetRecursiveTMPFileList(const std::string& path) {
	std::vector<std::string> file_list;
	for (auto it = boost::filesystem::recursive_directory_iterator(path);
	     it != boost::filesystem::recursive_directory_iterator(); ++it) {
		if (boost::filesystem::is_regular_file(*it)) {
			const boost::filesystem::path file_path = *it;
            file_list.push_back(file_path.string());	
		}
	}
	return file_list;
}

std::string TimeStampFromStr(const std::string &time_str) {
    int year, month, day, hour, minute, second;
    std::vector<std::string> split_elems = StringSplit(time_str, " ");
    if (split_elems.size() == 2) {
        std::vector<std::string> ymd_elems = StringSplit(split_elems[0], "-");

        // std::cout << "ymd_elems[0] = " << ymd_elems[0] << std::endl;
        year = std::stoi(ymd_elems[0]) - 1900;
        // std::cout << "ymd_elems[1] = " << ymd_elems[1] << std::endl;
        month = std::stoi(ymd_elems[1]) - 1;
        // std::cout << "ymd_elems[2] = " << ymd_elems[2] << std::endl;
        day = std::stoi(ymd_elems[2]);


        // std::cout << "split_elems[1].substr(0, 2) = " << split_elems[1].substr(0, 2) << std::endl;
        hour = std::stoi(split_elems[1].substr(0, 2));
        // std::cout << "split_elems[1].substr(3, 2) = " << split_elems[1].substr(3, 2) << std::endl;
        minute = std::stoi(split_elems[1].substr(3, 2));
        // std::cout << "split_elems[1].substr(6, 2) = " << split_elems[1].substr(6, 2) << std::endl;
        second = std::stoi(split_elems[1].substr(6, 2));


        // std::cout << "split_elems[1].substr(9, 3) = " << split_elems[1].substr(9, 3) << std::endl;
        std::string mcro_sceond = split_elems[1].substr(9, 3);

        struct tm timeinfo;
        timeinfo.tm_year = year;
        timeinfo.tm_mon = month;
        timeinfo.tm_mday = day;
        timeinfo.tm_hour = hour;
        timeinfo.tm_min = minute;
        timeinfo.tm_sec = second;
        timeinfo.tm_isdst = 0;
        time_t t = mktime(&timeinfo);
        return std::to_string(t) + mcro_sceond;
    } else {
        std::cout << "Convert Unix time stamp failed" << std::endl;
    }

    return "1609430400000";  //  20210101_000000
}

bool LoadBlueToothFolder(const std::string& path, 
                         std::unordered_map<double, std::vector<std::pair<std::string, int>>>& prior_bluetooth_time_signal) {
    prior_bluetooth_time_signal.clear();

    std::string bluetooth_files_path = path;
    std::vector<std::string> bluetooth_files;
    std::vector<std::string> items = StringSplit(bluetooth_files_path, ".");
    if (items.back() == "txt" || items.back() == "csv") {
        if (IsBluetoothFile(bluetooth_files_path))
            bluetooth_files.push_back(bluetooth_files_path);
        else {
            std::cout << "Error: its not Bluetooth txt ,Check its path :" << bluetooth_files_path << std::endl;
            return false;
        }
    } else {
        std::string endchar = bluetooth_files_path.substr(bluetooth_files_path.length() - 1);
        if (endchar == "/") {
            bluetooth_files_path.pop_back();
        }
        std::vector<std::string> files = GetRecursiveTMPFileList(bluetooth_files_path);
        for (const auto& file : files) {
            std::cout <<"file = " << file<<std::endl;
            items = StringSplit(file, ".");
            if (items.back() == "txt" || items.back() == "csv") {
                if (IsBluetoothFile(file)) {
                    bluetooth_files.push_back(file);
                }
            }
        }
        if (bluetooth_files.size() == 0) {
            std::cout << "Error: no bluetooth file in path" << bluetooth_files_path << std::endl;
        }
    }

    
    if (bluetooth_files.empty()) {
        return false;
    }


    for (const auto& file_path : bluetooth_files) {
        if (!IsBluetoothFile(file_path)) {
            std::cout << "Warning: its not bluetooth file " << file_path << std::endl;
            continue;
        }

        std::ifstream testdatafile(file_path);
        std::string linedata;
        std::vector<std::string> one_signal_line;
        while (getline(testdatafile, linedata)) {
            one_signal_line.clear();
            one_signal_line = StringSplit(linedata, ",");
            //      BeaconSignaldata beacondata;
            //      GetdataFromLinedata(linedata ,beacondata);
            double timestamp = std::stod(TimeStampFromStr(one_signal_line[0]));
            // double timestamp = one_signal_line.size() == 9 ? std::stod(one_signal_line[1]) : std::stod(one_signal_line[9]);
            std::string major_minor = one_signal_line[4] + "-" + one_signal_line[5];
            int rssi = std::stoi(one_signal_line[8]);
            if (rssi == 0) {
                rssi = -97;
            }

            // if (prior_bluetooth_time_signal.count(timestamp)) {
            //     std::cout << "Warning: Dumplicated BlueTooth Collection!" << std::endl;
            // }

            prior_bluetooth_time_signal[timestamp].emplace_back(std::make_pair(major_minor, rssi));
        }
    }

    return true;
}

void FeatureMatching(FeatureDataContainer &feature_data_container, SceneGraphContainer &scene_graph,
                     Configurator &param) {
    using namespace std::chrono;

    OptionParser option_parser;
    ImageReaderOptions reader_options;
    option_parser.GetImageReaderOptions(reader_options, param);

    std::string workspace_path = param.GetArgument("workspace_path", "");
    if (static_cast<bool>(param.GetArgument("map_update", 0))){
        // load original scene_graph
        scene_graph.ReadSceneGraphBinaryData(JoinPaths(workspace_path, "/scene_graph.bin"));

        // setup scene_graph
        EIGEN_STL_UMAP(image_t, class Image) &images = scene_graph.Images();
        EIGEN_STL_UMAP(camera_t, class Camera) &cameras = scene_graph.Cameras();

        std::vector<image_t> image_ids = feature_data_container.GetImageIds();

        for (const auto image_id : image_ids) {
            const Image &image = feature_data_container.GetImage(image_id);
            const Camera &camera = feature_data_container.GetCamera(image.CameraId());
            if (!scene_graph.CorrespondenceGraph()->ExistsImage(image_id)) {
                continue;
            }

            images[image_id] = image;

            const FeatureKeypoints &keypoints = feature_data_container.GetKeypoints(image_id);
            images[image_id].SetPoints2D(keypoints);

            const PanoramaIndexs &panorama_indices = feature_data_container.GetPanoramaIndexs(image_id);
            std::vector<uint32_t> local_image_indices(keypoints.size());
            for (size_t i = 0; i < keypoints.size(); ++i) {
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
        workspace_path = JoinPaths(workspace_path, "/map_update");
    }
    CHECK(!workspace_path.empty());

    if (boost::filesystem::exists(JoinPaths(workspace_path, "/scene_graph.bin"))) {
        std::cout << "Scene graph already exists, skip feature matching" << std::endl;
        // scene_graph.ReadSceneGraphBinaryData(JoinPaths(workspace_path, "/scene_graph.bin"));
        // scene_graph.CorrespondenceGraph()->Finalize();
        // ShowFeatureMatching(feature_data_container,scene_graph,param);
        return;
    }

    FeatureMatchingOptions options;
    option_parser.GetFeatureMatchingOptions(options, param);

    // bool lidar_sfm = param.GetArgument("lidar_sfm", false);
    // options.pair_matching_.is_sphere = !lidar_sfm;
    // // options.pair_matching_.is_sphere = false;
    // std::cout << StringPrintf("is_sphere: \n", options.pair_matching_.is_sphere);

    if (static_cast<bool>(param.GetArgument("map_update", 0))){
        options.match_between_reconstructions_ = true;
        options.delete_duplicated_images_ = false;
        options.map_update = true;
    }

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
    if (boost::filesystem::is_directory(workspace_path + "/initial_sfm") &&
        boost::filesystem::exists(workspace_path + "/initial_sfm/cameras.bin") &&
        boost::filesystem::exists(workspace_path + "/initial_sfm/images.bin") &&
        boost::filesystem::exists(workspace_path + "/initial_sfm/points3D.bin")) {
        has_initial_sfm = true;
    }

    if (use_initial_sfm && has_initial_sfm) {
        std::unordered_map<image_t, Eigen::Vector4d> prior_rotations;
        std::unordered_map<image_t, Eigen::Vector3d> prior_translations;
        std::unordered_map<image_t, bool> prior_pose_validations;
        auto reconstruction = std::make_shared<Reconstruction>();
        reconstruction->ReadReconstruction(workspace_path + "/initial_sfm");

        std::vector<image_t> image_ids = feature_data_container.GetImageIds();

        for (const auto image_id : image_ids) {
            if (!(reconstruction->ExistsImage(image_id) && reconstruction->IsImageRegistered(image_id))) {
                prior_pose_validations.emplace(image_id, false);
                continue;
            }
            const auto &image = reconstruction->Image(image_id);
            prior_rotations.emplace(image_id, image.Qvec());
            prior_translations.emplace(image_id, image.Tvec());
            prior_pose_validations.emplace(image_id, true);
        }

        reconstruction->ComputeBaselineDistance();
        for (const auto image_id : image_ids) {
            if (!(reconstruction->ExistsImage(image_id) && reconstruction->IsImageRegistered(image_id))) {
                continue;
            }

            Eigen::Vector4d qvec = prior_rotations.at(image_id);
            Eigen::Vector3d tvec = prior_translations.at(image_id);

            Eigen::Vector3d C = ProjectionCenterFromPose(qvec, tvec);

            bool valid = false;
            if (image_id > 1 && prior_rotations.find(image_id - 1) != prior_rotations.end()) {
                Eigen::Vector4d qvec_previous = prior_rotations.at(image_id - 1);
                Eigen::Vector3d tvec_previous = prior_translations.at(image_id - 1);
                Eigen::Vector3d C_previous = ProjectionCenterFromPose(qvec_previous, tvec_previous);

                double distance = (C - C_previous).norm();
                if (distance < reconstruction->baseline_distance * 30) {
                    valid = true;
                }
            }
            if (!valid && prior_rotations.find(image_id + 1) != prior_rotations.end()) {
                Eigen::Vector4d qvec_next = prior_rotations.at(image_id + 1);
                Eigen::Vector3d tvec_next = prior_translations.at(image_id + 1);
                Eigen::Vector3d C_next = ProjectionCenterFromPose(qvec_next, tvec_next);

                double distance = (C - C_next).norm();
                if (distance < reconstruction->baseline_distance * 30) {
                    valid = true;
                }
            }

            if (!valid) {
                CHECK(prior_pose_validations.find(image_id) != prior_pose_validations.end());
                prior_pose_validations.at(image_id) = false;
            }
        }

        options.prior_neighbor_distance = reconstruction->baseline_distance;
        options.have_prior_pose_ = true;
        options.prior_rotations = prior_rotations;
        options.prior_translations = prior_translations;
        options.prior_pose_validations = prior_pose_validations;
        options.max_match_distance = param.GetArgument("max_match_distance", 20.0f);
    }
    
    // use bluetooth signal filter far image pairs.
    bool use_prior_bluetooth = static_cast<bool>(param.GetArgument("use_prior_bluetooth", 0));
    std::string bluetooth_prior_folder = param.GetArgument("prior_bluetooth_path", "");
    double prior_bluetooth_threshold_inside = param.GetArgument("prior_bluetooth_threshold_inside", 3.0f);
    double prior_bluetooth_threshold_outside = param.GetArgument("prior_bluetooth_threshold_outside", 7.0f);
    double prior_bluetooth_threshold_outlier = param.GetArgument("prior_bluetooth_threshold_outlier", 9.0f);
    
    if (use_prior_bluetooth) {
        options.have_prior_bluetooth_ = true;
        options.prior_bluetooth_threshold_inside = prior_bluetooth_threshold_inside;
        options.prior_bluetooth_threshold_outside = prior_bluetooth_threshold_outside;
        options.prior_bluetooth_threshold_outlier = prior_bluetooth_threshold_outlier;
        if (!LoadBlueToothFolder(bluetooth_prior_folder, options.prior_bluetooth_time_signal)){
            std::cout << "ERROR: Bluetooth Config Failed" << std::endl;
            // FIXME: May not need exit
            exit(-1);
        }
    }

    // If gps prior is available
    std::string gps_prior_file = param.GetArgument("gps_prior_file", "");
    std::string gps_trans_file = workspace_path + "/gps_trans.txt";
    if (use_gps_prior) {
        if (boost::filesystem::exists(gps_prior_file)) {
            auto image_ids = feature_data_container.GetImageIds();
            std::vector<std::string> image_names;
            for (const auto image_id : image_ids) {
                const Image &image = feature_data_container.GetImage(image_id);
                std::string name = image.Name();
                image_names.push_back(name);
            }

            std::vector<std::pair<unsigned long, std::pair<Eigen::Vector3d,int>>> gps_locations;
            LoadOriginGPSinfo(gps_prior_file, gps_locations, gps_trans_file, true);
            
            std::unordered_map<std::string, std::pair<Eigen::Vector3d,int>> image_locations;
            GPSLocationsToImages(gps_locations, image_names, image_locations);
            std::unordered_map<image_t, std::pair<Eigen::Vector3d,int>> prior_locations_gps;
            std::cout << image_locations.size() << " images have gps prior" << std::endl;

            std::vector<PlyPoint> gps_locations_ply;
            for (const auto image_id : image_ids) {
                const Image &image = feature_data_container.GetImage(image_id);
                std::string name = image.Name();

                if (image_locations.find(name) != image_locations.end()) {
                    prior_locations_gps.emplace(image_id, image_locations.at(name));

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
            options.prior_locations = prior_locations_gps;
            sensemap::WriteBinaryPlyPoints(workspace_path + "/gps-match.ply", gps_locations_ply);
            options.have_prior_location_ = true;
            options.max_distance_for_loop = param.GetArgument("max_distance_for_loop", 20.0f);
        }
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
        scene_graph.WriteSceneGraphBinaryData(workspace_path + "/scene_graph.bin");
        // scene_graph.WriteImagePairsBinaryData(workspace_path + "/two_view_geometry.bin");
        scene_graph.WriteBlueToothPairsInfoBinaryData(workspace_path + "/bluetooth_info.bin");
        scene_graph.WriteLoopPairsInfoBinaryData(workspace_path + "/loop_pairs.bin");
        scene_graph.WriteNormalPairsBinaryData(workspace_path + "/normal_pairs.bin");
        scene_graph.WriteStrongLoopsBinaryData(workspace_path + "/strong_loops.bin");
    }

}

int main(int argc, char *argv[]) {
    using namespace sensemap;

    PrintHeading("SenseMap.  Copyright(c) 2019, SenseTime Group.");
    PrintHeading(std::string("Version: sfm-feature-match-")+__VERSION__);
    Timer timer;
    timer.Start();

    std::string configuration_file_path = std::string(argv[1]);
    Configurator param;
    param.Load(configuration_file_path.c_str());

    std::string workspace_path = param.GetArgument("workspace_path", "");
    if (static_cast<bool>(param.GetArgument("map_update", 0))){
        workspace_path = JoinPaths(workspace_path, "/map_update");
    }
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

    if (0) {
        auto image_ids = feature_data_container->GetImageIds();
        auto image = feature_data_container->GetImage(image_ids[0]);
        auto camera = feature_data_container->GetCamera(image.CameraId());

        // Convert Camera.
        std::unordered_map<camera_t, std::vector<camera_t> > camera_ids_map;
        size_t local_camera_id = 1;
        int num_local_camera = camera.NumLocalCameras();
        const bool exist_camera = camera_ids_map.count(image.CameraId()) != 0 &&
            camera_ids_map.at(image.CameraId()).size() == num_local_camera;

        Reconstruction reconstruction;

        for (local_camera_t camera_id = 0; camera_id < num_local_camera; ++camera_id){
            std::string model_name = camera.ModelName();
            int width = camera.Width();
            int height = camera.Height();

            std::vector<double> params;
            if (num_local_camera > 1) {
                camera.GetLocalCameraIntrisic(camera_id, params);
            } else {
                params = camera.Params();
            }

            class Camera local_camera;
            if (exist_camera) {
                local_camera.SetCameraId(camera_ids_map.at(image.CameraId())[camera_id]);
            } else {
                local_camera.SetCameraId(local_camera_id);
            }
            local_camera.SetModelIdFromName(model_name);
            local_camera.SetWidth(width);
            local_camera.SetHeight(height);
            local_camera.SetNumLocalCameras(1);
            local_camera.SetParams(params);

            reconstruction.AddCamera(local_camera);
            if (!exist_camera) {
                camera_ids_map[image.CameraId()].emplace_back(local_camera_id);
                local_camera_id++;
            }
        }

        EIGEN_STL_UMAP(mappoint_t, class MapPoint) rig_mappoints_;
        // Convert Image.
        image_t image_rig_id = 1;
        
        Eigen::Vector4d local_qvec;
        Eigen::Vector3d local_tvec;
        for (local_camera_t local_id = 0; local_id < num_local_camera; local_id++){
            Eigen::Vector4d normalized_qvec;
            Eigen::Vector3d normalized_tvec;
            if (num_local_camera > 1) {
                camera.GetLocalCameraExtrinsic(local_id, local_qvec, local_tvec);

                normalized_qvec =
                NormalizeQuaternion(RotationMatrixToQuaternion(
                QuaternionToRotationMatrix(local_qvec) * QuaternionToRotationMatrix(image.Qvec())));

                normalized_tvec =
                QuaternionToRotationMatrix(local_qvec)*image.Tvec() + local_tvec;
            } else {
                normalized_qvec = image.Qvec();
                normalized_tvec = image.Tvec();
            }

            std::string image_name = image.Name();
            if (num_local_camera > 1) {
                auto pos = image_name.find("cam0", 0);
                image_name.replace(pos, 4, "cam" + std::to_string(local_id));
            }

            // std::vector<Point2D> local_point2Ds;
            // point2D_t rig_point2D_idx = 0;
            // for(size_t i = 0; i< image.Points2D().size(); ++i){    
            //     Point2D point2D = image.Points2D()[i];
            //     local_camera_t local_camera_id = image.LocalImageIndices()[i];
            //     if (local_camera_id == local_id){
            //         if (point2D.HasMapPoint()) {
            //             if (rig_mappoints_.find(point2D.MapPointId()) == rig_mappoints_.end()){
            //                 class MapPoint mappoint_rig = 
            //                     mappoints_.at(point2D.MapPointId());
            //                 class TrackElement rig_trackelement(image_rig_id, rig_point2D_idx);
            //                 class Track rig_track;
            //                 rig_track.AddElement(rig_trackelement);
            //                 mappoint_rig.SetTrack(rig_track);
            //                 rig_mappoints_.emplace(point2D.MapPointId(), mappoint_rig);
            //             } else{
            //                 class TrackElement rig_trackelement(image_rig_id, rig_point2D_idx);
            //                 rig_mappoints_.at(point2D.MapPointId()).Track().AddElement(rig_trackelement);
            //             }
            //         }
            //         point2D.SetMapPointId(kInvalidMapPointId);
            //         local_point2Ds.push_back(point2D);
            //         rig_point2D_idx++;
            //     }
            // }

            // size_t local_camera_id = (image.second.CameraId()-1) * num_local_camera + local_id + 1;
            size_t local_camera_id = camera_ids_map.at(camera.CameraId())[local_id];

            class Image local_image;
            local_image.SetCameraId(local_camera_id);
            local_image.SetImageId(image_rig_id);
            local_image.SetName(image_name);
            local_image.SetQvec(normalized_qvec);
            local_image.SetTvec(normalized_tvec);
            // local_image.SetRegistered(true);
            local_image.SetPoseFlag(true);
            // local_image.SetPoints2D(local_point2Ds);
            local_image.SetLabelId(image.LabelId());
            reconstruction.AddImage(local_image);
            reconstruction.RegisterImage(image_rig_id);
            image_rig_id++;
        }

        // // Convert MapPoint.
        // for (auto mappoint : rig_mappoints_) {
        //     reconstruction.AddMapPoint(mappoint.first, 
        //                             mappoint.second.XYZ(), 
        //                             mappoint.second.Track(),
        //                             mappoint.second.Color());
        // }

        CreateDirIfNotExists(argv[2]);
        reconstruction.WriteBinary(argv[2]);
        // reconstruction.WriteText(argv[2]);
        return 0;
    }
    
    typedef FeatureMatchingOptions::RetrieveType RetrieveType;
    RetrieveType retrieve_type = (RetrieveType)param.GetArgument("retrieve_type", 0);
    if (retrieve_type == RetrieveType::VLAD) {
        CHECK(LoadGlobalFeatures(*feature_data_container.get(), param)) << "Load global features failed";
    }
    FeatureMatching(*feature_data_container.get(), *scene_graph_container.get(), param);


    fclose(fs);
    timer.PrintMinutes();
    return 0;
}