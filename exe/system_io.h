// Copyright (c) 2019, SenseTime Group.
// All rights reserved.
#ifndef _SYSTEM_IO_H_
#define _SYSTEM_IO_H_


#include <iostream>
#include <string.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <fstream>

#include "util/panorama.h"
#include "util/gps_reader.h"
#include <stdio.h>
#include "base/pose.h"

namespace sensemap{

struct Keyframe {
    unsigned int id;
    std::string name;
    Eigen::Quaterniond q;
    Eigen::Matrix3d rot;
    Eigen::Vector3d pos;

    std::vector<int> covisibilities;
};


// load prior pose from sensemap-preview
bool LoadPirorPose(const std::string &file_path, std::vector<Keyframe> &keyfrms){
    if (file_path.empty()) return false;
    std::cout << "load keyframes" << std::endl;
    /// open file
    std::ifstream infile;
    infile.open(file_path, std::ios::binary | std::ios::in);
    if (!infile.is_open()) {
        return false;
    }

    Keyframe keyfrm;
    /// load keyframes pose and graph
    float x, y, z, qx, qy, qz, qw;

    std::string image_name;
    int id = 1;
    while(infile >> image_name){
        keyfrm.id = id ++;
        keyfrm.name = "cam0/"+image_name+".jpg";

        infile >> x >> y >> z >> qw >> qx >> qy >> qz;

        keyfrm.q = Eigen::Quaterniond(qw, qx, qy, qz);
        keyfrm.rot = keyfrm.q.matrix();
        
        // load pos
        keyfrm.pos << x, y, z;

        Eigen::Matrix3d R = keyfrm.rot.transpose();
        Eigen::Vector3d T = -keyfrm.rot.transpose()*keyfrm.pos;

        keyfrm.rot = R;
        keyfrm.pos = T;
        keyfrm.q = Eigen::Quaterniond(R);

        keyfrm.covisibilities.clear();

        keyfrms.push_back(keyfrm);
    }
    return true;
}

// load prior pose from tum file
bool LoadPriorPoseFromTum(const std::string &file_path, std::vector<Keyframe> &keyfrms, bool as_rgbd = false){
    if (file_path.empty()) return false;
    std::cout << "load tum" << std::endl;

    /// open file
    std::ifstream infile;
    infile.open(file_path, std::ios::binary | std::ios::in);
    if (!infile.is_open()) {
        return false;
    }

    Keyframe keyfrm;
    /// load keyframes pose and graph
    double x, y, z, qx, qy, qz, qw;

    std::vector<char> line(1024);
    std::vector<char> name(1024);
    int id = 1;
    while (infile.getline(line.data(), line.size())) {
        if (sscanf(line.data(), "%s %lf %lf %lf %lf %lf %lf %lf", 
            name.data(), &x, &y, &z, &qx, &qy, &qz, &qw
        ) == 8) {
            keyfrm.id = id ++;

            keyfrm.q = Eigen::Quaterniond(qw, qx, qy, qz);
            keyfrm.rot = keyfrm.q.matrix();
            
            // load pos
            keyfrm.pos << x, y, z;

            Eigen::Matrix3d R = keyfrm.rot.transpose();
            Eigen::Vector3d T = -keyfrm.rot.transpose()*keyfrm.pos;

            keyfrm.rot = R;
            keyfrm.pos = T;
            keyfrm.q = Eigen::Quaterniond(R);

            keyfrm.covisibilities.clear();

            if (as_rgbd) {
                keyfrm.name = std::string(name.data()) + ".bin";
                keyfrms.push_back(keyfrm);
                keyfrm.name = std::string(name.data()) + ".binx";
                keyfrms.push_back(keyfrm);
            }
            else {
                keyfrm.name = std::string(name.data()) + ".jpg";
                keyfrms.push_back(keyfrm);
            }
        }
    }

    return true;
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

bool LoadViSlamFrames(
    const std::string& path,
    std::vector<std::unordered_map<std::string, std::pair<Eigen::Vector4d, Eigen::Vector3d>>>& sequence_poses,
    std::vector<double>& intrinsics) {
    std::vector<std::string> dir_list = GetDirList(path);
    if (dir_list.size() <= 0) {
        std::cout << "ViSlam results not found" << std::endl;
        return false;
    }
    std::cout<<"Load frames and poses ..."<<std::endl;

    bool intrinsics_extracted = false;
    intrinsics.clear();
    sequence_poses.clear();

    for (const auto& dir : dir_list) {
        CHECK(boost::filesystem::exists(dir + "/Frames.txt") && boost::filesystem::exists(dir + "/ARposes.txt"));
        std::string dir_name = dir.substr(path.size() + 1);
        std::cout<<"Dir: "<<dir<<" "<<dir_name<<std::endl;

        std::cout<<"Read frame.txt"<<std::endl;
        std::ifstream file_frame(dir + "/Frames.txt");
        CHECK(file_frame.is_open());

        std::string line;
        std::string item;

        std::unordered_map<std::string, std::string> timestamp_frame_name_map;

        while (std::getline(file_frame, line)) {
            StringTrim(&line);

            std::stringstream line_stream(line);

            std::getline(line_stream, item, ',');
            std::string time_stamp = item;

            std::getline(line_stream, item, ',');
            int frame_id = std::stoi(item);

            char formated_frame_id[32];
            sprintf(formated_frame_id, "%06d", frame_id + 1);
            std::string image_name = formated_frame_id;

            image_name = dir_name + "/" + formated_frame_id + ".jpg";

            timestamp_frame_name_map.emplace(time_stamp, image_name);

            //std::cout<<time_stamp<<" "<<image_name<<std::endl;

            if (!intrinsics_extracted) {
                for (int i = 0; i < 4; ++i) {
                    std::getline(line_stream, item, ',');
                    intrinsics.push_back(std::stod(item));
                }
            }
        }
        file_frame.close();


        std::unordered_map<std::string, std::pair<Eigen::Vector4d, Eigen::Vector3d>> frame_poses;

        std::cout<<"Read ARposes.txt"<<std::endl;
        std::ifstream file_pose(dir + "/ARposes.txt");
        CHECK(file_pose.is_open());
        while (std::getline(file_pose, line)) {
            StringTrim(&line);

            std::stringstream line_stream(line);

            std::getline(line_stream, item, ',');
            std::string time_stamp = item;

            //std::cout<<time_stamp<<std::endl;
            if (timestamp_frame_name_map.find(time_stamp) == timestamp_frame_name_map.end()) {
                continue;
            }

            Eigen::Vector3d t;
            Eigen::Vector4d q;

            for (int i = 0; i < 3; ++i) {
                std::getline(line_stream, item, ',');
                t(i) = std::stod(item);
            }

            for (int i = 0; i < 4; ++i) {
                std::getline(line_stream, item, ',');
                q(i) = std::stod(item);
            }

            q = InvertQuaternion(q);
            t = -QuaternionToRotationMatrix(q) * t;

            std::pair<Eigen::Vector4d, Eigen::Vector3d> pose(q, t);

            frame_poses.emplace(timestamp_frame_name_map.at(time_stamp), pose);
        }

        file_pose.close();

        sequence_poses.push_back(frame_poses);
    }
    return true;
}


}

#endif