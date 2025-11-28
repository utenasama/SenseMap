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
   
    const std::string old_recon_path = std::string(argv[1]);
    const std::string new_recon_path = std::string(argv[2]);
    const std::string image_name_patten = std::string(argv[3]);

    std::cout << "Keep Image Patten : " << image_name_patten << std::endl;

    auto reconstruction = std::make_shared<Reconstruction>();
    std::cout << "Start Read reconstruction ... " << std::endl;
    reconstruction->ReadBinary(old_recon_path);
    std::cout << "Read reconstruction finished ..." << std::endl;

    
    auto regist_image_id = reconstruction->RegisterImageIds();

    for (image_t image_id : regist_image_id) {
        std::string image_name = reconstruction->Image(image_id).Name();

        if (image_name.find(image_name_patten) == std::string::npos){
            reconstruction->DeRegisterImage(image_id);
        }
    }

    // Create output path
    if (!boost::filesystem::exists(new_recon_path)) {
        boost::filesystem::create_directories(new_recon_path);
    }

    reconstruction->WriteReconstruction(new_recon_path, true);

    return 0;
}
