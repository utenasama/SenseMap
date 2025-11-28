// Copyright (c) 2019, SenseTime Group.
// All rights reserved.

#include <boost/filesystem/path.hpp>
#include <opencv2/opencv.hpp>

#include "base/reconstruction_manager.h"
#include "util/misc.h"

using namespace sensemap;


int main(int argc, char* argv[]) {
    PrintHeading("SenseMap.  Copyright(c) 2019, SenseTime Group.");
    PrintHeading("Version: convert-to-tum-1.6.9");

    if (argc < 4) {
        std::cout
            << "Usage: convert_to_tum 1.workspace/0/images.bin \n"
            << "                      2.output_pose.txt \n"
            << "                      3.primary key use: image_id--0, image_name--1"
            << std::endl;
        return -1;
    }

    std::string images_bin_path = std::string(argv[1]);
    std::string output_pose_path = std::string(argv[2]);
    int primary_key_option = std::stoi(argv[3]);

    if (!ExistsFile(images_bin_path)){
        std::cout << "Error: Images.bin not exist in path: " << images_bin_path << std::endl;
        exit(-1);
    }

    auto reconstruction = std::make_shared<Reconstruction>();
    reconstruction->ReadImagesBinary(images_bin_path);
    std::cout << "Read Images.bin finished ..." << std::endl;

    if (primary_key_option == 0) {
        reconstruction->WritePoseText(output_pose_path);
    } else if (primary_key_option == 1) {
        reconstruction->WriteLocText(output_pose_path);
    } else{
        std::cout << "Error: Invalid primary key option : " << primary_key_option << std::endl;
        exit(-1);
    }

    return 0;
}
