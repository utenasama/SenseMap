// Copyright (c) 2019, SenseTime Group.
// All rights reserved.

#include <boost/filesystem/path.hpp>
#include <opencv2/opencv.hpp>

#include "base/camera.h"
#include "base/reconstruction_manager.h"
#include "util/misc.h"
#include "base/pose.h"

using namespace sensemap;

std::string image_path;
std::string workspace_path;
std::string vocab_path;

int main(int argc, char* argv[]) {
    // std::string camera_path = std::string(argv[1]);
    
    // auto reconstruction = new Reconstruction();
    // reconstruction->ReadCamerasText(camera_path);
    // // Display the camera basic info
    // // auto cameras = reconstruction->Cameras();
    // // std::cout << "Camera number = " << cameras.size() << std::endl;

    // // for (auto camera : cameras){
    // //     std::cout << "  Camera index = " << camera.first << std::endl;
    // //     std::cout << "  Camera model = " << camera.second.ModelName() << std::endl;
    // //     std::cout << "  Camera param = ";
    // //     for (auto param : camera.second.Params()){
    // //         std::cout << "  " << param;
    // //     }
    // //     std::cout << std::endl;
    // // }
    // // for(size_t i = 0; i< reconstruction->Cameras().size(); ++i){
    // //     Camera& camera = reconstruction->Camera(i+1);

    // //     for(size_t j = 0; j< 4; ++j){
    // //         camera.Params()[j] *= 3840.0/4000.0;
    // //     }
    // // }

    // std::string camera_txt_path = camera_path.substr(0, camera_path.size()-4) + ".bin2";
    // reconstruction->WriteCamerasBinary(camera_txt_path);

    // std::string camera_txt_path2 = camera_path.substr(0, camera_path.size()-4) + ".txt2";
    // reconstruction->WriteCamerasText(camera_txt_path2);

    Eigen::Vector4d qvec;
    qvec<<-0.001607, 0.007106, 0.999971, -0.002342;
    Eigen::Matrix3d R;
    R = QuaternionToRotationMatrix(qvec);
    std::cout<<"R: "<<std::endl;
    std::cout<<R<<std::endl;

    return 0;
}
