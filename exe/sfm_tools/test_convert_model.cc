// Copyright (c) 2019, SenseTime Group.
// All rights reserved.

#include <boost/filesystem/path.hpp>
#include <opencv2/opencv.hpp>

#include "base/reconstruction_manager.h"
#include "util/misc.h"

using namespace sensemap;

std::string image_path;
std::string workspace_path;
std::string vocab_path;

int main(int argc, char* argv[]) {
    PrintHeading("SenseMap.  Copyright(c) 2019, SenseTime Group.");
    PrintHeading("Version: convert-model-1.6.9");

    if (argc < 4) {
        std::cout
            << "Usage: test_convert_model 1.trans_yaml_file_path 2.old_reconstruction_0_path 3.new_reconstruction_0_path"
            << std::endl;
        return 1;
    }

    const std::string trans_file_path = std::string(argv[1]);
    const std::string old_recon_path = std::string(argv[2]);
    const std::string new_recon_path = std::string(argv[3]);

    auto reconstruction = std::make_shared<Reconstruction>();
    std::cout << "Start Read reconstruction ... " << std::endl;
    reconstruction->ReadBinary(old_recon_path);
    std::cout << "Read reconstruction finished ..." << std::endl;

    // Load transform matrix
    cv::FileStorage fs;
    fs.open(trans_file_path, cv::FileStorage::READ);
    cv::Mat trans_mat;
    // std::cout << "Type = " << fs["transMatrix"].type() << std::endl;
    if(fs["transMatrix"].type() != cv::FileNode::MAP){
        std::cout << "ERROR: Input yaml error !!" << std::endl;
        exit(-1);
    }
    fs["transMatrix"] >> trans_mat;
    // std::cout << "trans = \n" << " " << trans_mat << std::endl << std::endl;

    Eigen::Matrix3x4d transform;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 4; j++) {
            transform(i, j) = trans_mat.at<double>(i, j);
        }
    }    


    std::cout << transform << std::endl;
    reconstruction->TransformReconstruction(transform, false);

    // Create output path
    if (!boost::filesystem::exists(new_recon_path)) {
        boost::filesystem::create_directories(new_recon_path);
    }

    reconstruction->WriteReconstruction(new_recon_path, true);

    return 0;
}
