// Copyright (c) 2020, SenseTime Group.
// All rights reserved.

#include <iostream>

#include "util/rgbd_helper.h"
#include "../Configurator_yaml.h"

using namespace sensemap;

std::string configuration_file_path;

int main(int argc, char* argv[]) {

    // configuration_file_path = std::string(argv[1]);
    // Configurator param;
    // param.Load(configuration_file_path.c_str());

    // std::string rgbd_parmas_file = param.GetArgument("rgbd_params_file", "");
    std::string rgbd_parmas_file(argv[1]);

    CalibOPPOBinReader calib_reader;
    calib_reader.ReadCalib(rgbd_parmas_file);

    Eigen::Vector4f rgb_K, tof_K;
    int rgb_w, rgb_h, tof_w, tof_h;
    calib_reader.GetRGB_K(rgb_K, rgb_w, rgb_h);
    calib_reader.GetToF_K(tof_K, tof_w, tof_h);
    
    std::cout << "RGB image size: " << rgb_w << " " << rgb_h << std::endl;
    std::cout << "RGB camera intrinsic matrix: " << std::endl;
    std::cout << rgb_K.transpose() << std::endl;

    std::cout << "ToF image size: " << tof_w << " " << tof_h << std::endl;
    std::cout << "ToF camera intrinsic matrix: " << std::endl;
    std::cout << tof_K.transpose() << std::endl;

    return 0;
}