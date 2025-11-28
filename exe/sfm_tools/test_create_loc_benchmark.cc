// Copyright (c) 2021, SenseTime Group.
// All rights reserved.

#include <dirent.h>
#include <sys/stat.h>

#include <boost/filesystem/path.hpp>
#include "../Configurator_yaml.h"
#include "../option_parsing.h"
#include "base/reconstruction_manager.h"
#include "base/version.h"
#include "util/misc.h"
#include<stdlib.h>

using namespace sensemap;

int main(int argc, char *argv[]) {
    using namespace sensemap;

    PrintHeading("SenseMap.  Copyright(c) 2021, SenseTime Group.");
    PrintHeading(std::string("Version: create-benchmark-")+__VERSION__);
    

    std::string configuration_file_path = std::string(argv[1]);
    Configurator param;
    param.Load(configuration_file_path.c_str());

    std::string workspace_path = param.GetArgument("workspace_path", "");
    CHECK(!workspace_path.empty()) << "image path empty";

    std::string image_path = param.GetArgument("image_path", "");
    CHECK(!workspace_path.empty()) << "image path empty";

    auto reconstruction = std::make_shared<Reconstruction>();
    auto keyframe_reconstruction = std::make_shared<Reconstruction>();

    if (boost::filesystem::exists(workspace_path + "/0")) {
        reconstruction->ReadReconstruction(workspace_path + "/0");
        keyframe_reconstruction->ReadReconstruction(workspace_path+"/0/KeyFrames");
    }
    else{
        std::cout<<"read reconstruction fail"<<std::endl;
        return -1;
    }

    std::vector<image_t> keyframe_images_v = keyframe_reconstruction->RegisterImageIds();
    std::unordered_set<image_t> keyframe_images(keyframe_images_v.begin(),keyframe_images_v.end());

    // if(!boost::filesystem::exists(workspace_path + "/train")){
    //     boost::filesystem::create_directory(workspace_path + "/train");
    // }

    if(!boost::filesystem::exists(workspace_path + "/test")){
        boost::filesystem::create_directory(workspace_path + "/test");
    }
    std::ofstream file_test_poses(workspace_path + "/test"+"/gt_poses.txt");
    CHECK(file_test_poses.is_open());

    std::shared_ptr<Reconstruction> train_reconstruction = std::make_shared<Reconstruction>();
    std::shared_ptr<Reconstruction> train_keyframe_reconstruction = std::make_shared<Reconstruction>();

    std::unordered_set<image_t> training_images;
    std::unordered_set<image_t> training_keyframe_images;
    
    auto image_ids = reconstruction->RegisterImageIds();
    for (auto image_id : image_ids) {
        const auto& image = reconstruction->Image(image_id);
        std::string image_name = image.Name();
        if(image_name.substr(0,5)=="train"){
            training_images.insert(image_id);
            if(keyframe_images.count(image_id)>0){
                training_keyframe_images.insert(image_id);
            }
        }
        else if(image_name.substr(0,4)=="test"){
            file_test_poses << image_id << " " << image_name << " " << image.Qvec()[0] << " " << image.Qvec()[1] << " "
                            << image.Qvec()[2] << " " << image.Qvec()[3] << " " << image.Tvec()[0] << " "
                            << image.Tvec()[1] << " " << image.Tvec()[2] << std::endl;
        }
    }
    if(training_images.size()==0){
        std::cout<<"cannot find training images"<<std::endl;
        return -1;
    }

    std::string cmd = "mv " + workspace_path + "/0 " + workspace_path + "/0-full";

    system(cmd.c_str());
    
    std::unordered_set<mappoint_t> all_mappoints;
    all_mappoints = reconstruction->MapPointIds();
    
    reconstruction->Copy(training_images,all_mappoints, train_reconstruction);

    std::string train_rec_path = StringPrintf("%s/0", workspace_path.c_str());
    if (!boost::filesystem::exists(train_rec_path)) {
        boost::filesystem::create_directories(train_rec_path);
    }
    train_reconstruction->WriteReconstruction(train_rec_path);


    if(training_keyframe_images.size()==0){
        std::cout<<"cannot find training keyframe images"<<std::endl;
        return -1;
    }

    reconstruction->Copy(training_keyframe_images,all_mappoints, train_keyframe_reconstruction);
    std::string train_keyframe_rec_path = StringPrintf("%s/0/KeyFrames", workspace_path.c_str());
    if (!boost::filesystem::exists(train_keyframe_rec_path)) {
        boost::filesystem::create_directories(train_keyframe_rec_path);
    }
    train_keyframe_reconstruction->WriteReconstruction(train_keyframe_rec_path);



    // std::string org_rec_path = StringPrintf("%s/0-org", workspace_path.c_str());
    // if (!boost::filesystem::exists(org_rec_path)) {
    //     boost::filesystem::create_directories(org_rec_path);
    // }
    // reconstruction->WriteReconstruction(org_rec_path);


    // std::string org_rec_keyframe_path = StringPrintf("%s/0-org/KeyFrames", workspace_path.c_str());
    // if (!boost::filesystem::exists(org_rec_keyframe_path)) {
    //     boost::filesystem::create_directories(org_rec_keyframe_path);
    // }
    // keyframe_reconstruction->WriteReconstruction(org_rec_keyframe_path);



    return 0;
}