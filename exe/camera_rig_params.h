// Copyright (c) 2019, SenseTime Group.
// All rights reserved.

#ifndef _CAMERA_RIG_PARAMS_H_
#define _CAMERA_RIG_PARAMS_H_

#include <string>
#include <opencv2/opencv.hpp>
#include "util/alignment.h"
#include "util/logging.h"
#include "base/pose.h"

namespace sensemap{


class CameraRigParams{

public:
    int num_local_cameras;  
    std::vector<Eigen::Matrix4d> local_extrinsics;
    std::vector<std::vector<double>> local_intrinsics;

    std::string local_intrinsics_str;
    std::string local_extrinsics_str;
    std::string camera_model;

    inline void ParamsToString(){
        CHECK_EQ(local_intrinsics.size(),num_local_cameras);
        local_intrinsics_str = "";        
    
        for(size_t i = 0; i < num_local_cameras; ++i){
            for(size_t j = 0; j<local_intrinsics[i].size(); ++j){
                local_intrinsics_str.append(std::to_string(local_intrinsics[i][j]));
                local_intrinsics_str.append(",");
            }       
        }

        local_extrinsics_str = "";
        for(size_t i = 0; i<local_extrinsics.size(); ++i){
            Eigen::Matrix3d R = local_extrinsics[i].block<3,3>(0,0);
            Eigen::Vector3d t = local_extrinsics[i].block<3,1>(0,3);

            Eigen::Vector4d qvec = RotationMatrixToQuaternion(R);

            for(int j = 0; j<4; j++){
                local_extrinsics_str.append(std::to_string(qvec(j)));
                local_extrinsics_str.append(",");
            }

            for(int j = 0; j<3; j++){
                local_extrinsics_str.append(std::to_string(t(j)));
                local_extrinsics_str.append(",");
            }
        }    
    }

    inline bool LoadParams(const std::string& path){
        std::cout<<"Read camera-rig params file: "<<path<<std::endl;
        cv::FileStorage fs(path, cv::FileStorage::READ);

        if (!fs.isOpened()) {
            std::cout<<"Load camera rig failed"<<std::endl;            
            return false;
        }

        num_local_cameras = (int)fs["n_camera"];    
        local_extrinsics.resize(num_local_cameras);
        local_intrinsics.resize(num_local_cameras);
        
        cv::FileNode node = fs["camera_model"];
        CHECK(node.type() == cv::FileNode::STR);
        node>>camera_model;

        for(int i = 0; i< num_local_cameras; ++i){
            std::string camera_id = "cam_" + std::to_string(i);

            // load local extrinsics
            node = fs[camera_id]["T_cam_imu"];
            if (node.type() != cv::FileNode::SEQ) {
                std::cerr << "T_camera_imu is not a sequence! FAIL" << std::endl;
                exit(-1);
            }
            
            cv::FileNodeIterator it_trans = node.begin(), it_trans_end = node.end(); 
            
            int row = 0;
            for(; it_trans != it_trans_end; it_trans++, row++){
                std::vector<double> trans_row;
                (*it_trans)>>trans_row;

                CHECK_EQ(trans_row.size(),4);

                for(int m = 0; m < 4; m++){
                    local_extrinsics[i](row,m) = trans_row[m];
                }
            }

            node = fs[camera_id]["intrinsics"];
            std::vector<double>  intrinsics;
            node >> intrinsics;   

            std::vector<double> distortion;
            node = fs[camera_id]["distortion_coeffs"];
            node>>distortion;

            local_intrinsics[i].resize(intrinsics.size()+distortion.size());
            for(size_t j = 0; j<intrinsics.size(); ++j){
                local_intrinsics[i][j] = intrinsics[j];
            }
            for(size_t j = 0; j<distortion.size(); ++j){
                local_intrinsics[i][j+intrinsics.size()] = distortion[j];
            }    
        }
        
        ParamsToString();
        return true;
    }

};


}


#endif