// Copyright (c) 2019, SenseTime Group.
// All rights reserved.

#include <boost/filesystem/path.hpp>
#include <chrono>

#include "../Configurator_yaml.h"
#include "../camera_rig_params.h"
#include "../option_parsing.h"
#include "base/image.h"
#include "base/image_reader.h"
#include "util/misc.h"
#include "util/panorama.h"
#include "util/rgbd_helper.h"
#include "base/version.h"

#if defined(_OPENMP)
#include <omp.h>
#endif

using namespace std;
using namespace sensemap;

string img_input_path;
string img_output_path;

// Read the config yaml
bool LoadParams(const std::string& path, std::vector<PanoramaParam>& panorama_params) {
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

int main(int argc, char** argv) {
    //---------------------------
    PrintHeading("SenseMap.  Copyright(c) 2019, SenseTime Group.");
    PrintHeading(std::string("Version: generate-normal-images-") + __VERSION__);

    std::string config_file_path;
    std::vector<PanoramaParam> panorama_params;
    std::string sfm_config_file_path;

    int perspective_image_count, perspective_image_width, perspective_image_height;
    double fov_w;

    if (argc == 9) {
        img_input_path = argv[1];
        img_output_path = argv[2];
        config_file_path = std::string(argv[3]);
        sfm_config_file_path = std::string(argv[4]);
        perspective_image_count = atoi(argv[5]);
        fov_w = strtod(argv[6], nullptr);
        perspective_image_width = atoi(argv[7]);
        perspective_image_height = atoi(argv[8]);
    } else {
        cout << "Input arg number error" << endl;
        cout << "Input_image_path,Output_image_path,config.yaml,sfm-config.yaml,perspective_image_count,fov_w,"
                "perspective_image_width,perspective_image_height"
             << endl;
        return -1;
    }

    LoadParams(config_file_path, panorama_params);

    // load camera params
    Configurator param;
    param.Load(sfm_config_file_path.c_str());
    std::string workspace_path = param.GetArgument("workspace_path", "");
    std::string camera_param_file = param.GetArgument("camera_param_file", "");
    int num_cameras = param.GetArgument("num_cameras", -1);

    OptionParser option_parser;
    ImageReaderOptions reader_options;
    option_parser.GetImageReaderOptions(reader_options, param);
    if (num_cameras >= 1 && ExistsPath(camera_param_file)) {
        option_parser.GetImageReaderOptions(reader_options, param, 0);
    }

    Camera camera;
    camera.SetCameraId(kInvalidCameraId);
    camera.SetModelIdFromName(reader_options.camera_model);
    camera.SetNumLocalCameras(reader_options.num_local_cameras);
    if (reader_options.num_local_cameras > 1) {
        camera.SetLocalCameraIntrinsicParamsFromString(reader_options.local_camera_params.begin()->second);
    } else {
        camera.SetParamsFromString(reader_options.camera_params);
    }

    // Remove the last '/' in the input path and output path
    if (img_input_path.substr(img_input_path.size() - 1) == "/") {
        img_input_path = img_input_path.substr(0, img_input_path.size() - 1);
    }

    if (img_output_path.substr(img_output_path.size() - 1) == "/") {
        img_output_path = img_output_path.substr(0, img_output_path.size() - 1);
    }

    // std::string child_path = reader_options.child_path;
    // std::cout<<"child_path "<<child_path<<std::endl;
    // if (!child_path.empty() && child_path.substr(child_path.size() - 1) == "/") {
    //     child_path = child_path.substr(0, child_path.size() - 1);
    // }
    std::vector<std::string> child_paths;
    if (num_cameras >= 1 && ExistsPath(camera_param_file)) {
        for (int child_idx = 0; child_idx<num_cameras; child_idx++){
            option_parser.GetImageReaderOptions(reader_options,param,child_idx);
            std::string child_path = reader_options.child_path;
            if (!child_path.empty() && child_path.substr(child_path.size() - 1) == "/") {
                child_path = child_path.substr(0, child_path.size() - 1);
            }
            child_paths.push_back(child_path);
        }
        
    }
    for(auto child_path : child_paths){
        std::cout<<"child_path "<<child_path<<std::endl;
    }

    // load reconstruction
    std::unordered_map<std::string, image_t> map_name_camera_id;
    auto reconstruction = std::make_shared<Reconstruction>();

    if (boost::filesystem::exists(workspace_path + "/0")) {
        reconstruction->ReadReconstruction(workspace_path + "/0");
    }
    std::unordered_set<camera_t> cameras_in_reconstruction;
    std::cout << "registered num images: " << reconstruction->RegisterImageIds().size() << std::endl;
    for (const auto image_id : reconstruction->RegisterImageIds()) {
        const auto& image = reconstruction->Image(image_id);
        // std::cout << "image.Name() = " << image.Name() << std::endl;
        map_name_camera_id.emplace(image.Name(), image.CameraId());
        cameras_in_reconstruction.insert(image.CameraId());
    }
    std::cout << "cameras in reconstruction count: " << cameras_in_reconstruction.size() << std::endl;

    // Display the camera basic info
    auto cameras = reconstruction->Cameras();
    std::cout << "Camera number = " << cameras.size() << std::endl;

    for (auto camera : cameras) {
        std::cout << "  Camera index = " << camera.first << std::endl;
        std::cout << "  Camera model = " << camera.second.ModelName() << std::endl;
        std::cout << "  Camera Height = " << camera.second.Height() << std::endl;
        std::cout << "  Camera Width = " << camera.second.Width() << std::endl;
        std::cout << "  Camera param = ";
        for (auto param : camera.second.Params()) {
            std::cout << "  " << param;
        }
        std::cout << std::endl;

        // Sub Camera
        std::cout << "  Sub Camera Number = " << camera.second.NumLocalCameras() << std::endl;
        if (camera.second.NumLocalCameras() > 1) {
            for (size_t local_camera_id = 0; local_camera_id < camera.second.NumLocalCameras(); ++local_camera_id) {
                std::cout << "    Local Camera Id = " << local_camera_id << std::endl;
                std::vector<double> params;
                camera.second.GetLocalCameraIntrisic(local_camera_id, params);
                std::cout << "    Sub Camera intrinsic = ";
                for (auto param : params) {
                    std::cout << "  " << param;
                }
                std::cout << std::endl;

                Eigen::Vector4d qvec;
                Eigen::Vector3d tvec;
                camera.second.GetLocalCameraExtrinsic(local_camera_id, qvec, tvec);
                std::cout << "    Sub Camera extrinsic = ";
                std::cout << "  qvec = " << qvec[0] << " " << qvec[1] << " " << qvec[2] << " " << qvec[3]
                          << " , tvec = " << tvec[0] << " " << tvec[1] << " " << tvec[2] << " ";
                std::cout << std::endl;
            }
        }

        std::cout << "\n";
    }

    std::cout << "\n\n";

    // Load image folder list
    std::cout << "Get image file list ..." << std::endl;
    auto image_list = GetRecursiveFileList(img_input_path);
    std::cout << "Original Image Number = " << image_list.size() << std::endl;
    if(child_paths.size()){
        std::vector<std::string> new_image_list;
        for(auto child_path : child_paths){
            std::cout << "child_path  = " << child_path << std::endl;
            // std::cout << "JoinPaths(img_input_path, child_path) = " << JoinPaths(img_input_path, child_path) <<
            // std::endl;
            for (auto& image : image_list) {
                if (IsInsideSubpath(image, JoinPaths(img_input_path, child_path))) {
                    // std::cout << "image path = " << image << std::endl;
                    new_image_list.emplace_back(std::move(image));
                }
            }
        }
        std::cout << " => " << new_image_list.size() << std::endl;
        std::swap(image_list, new_image_list);
    }

    // Convert from panorama image
    Panorama panorama;
    // Convert from large fov image
    // LargeFovImage large_fov_image;
    // // Convert from large fov image and split in piecewise
    // PiecewiseImage piecewise_image;

    std::unordered_map<camera_t, std::shared_ptr<LargeFovImage>> large_fov_images;
    std::unordered_map<camera_t, std::shared_ptr<PiecewiseImage>> piecewise_images;

    // Process each image
    size_t image_counter = 0;

    // FIXME: Protential Problem
    std::string image_path;
    for (auto image_pth : image_list) {
        if (!IsFileRGBD(image_pth)) {
            image_path = image_pth;
            break;
        }
    }

    // auto image_path = image_list[0];
    // Load panorama images
    Bitmap img_input_test;
    if (!img_input_test.Read(image_path, true)) {
        std::cerr << "seg image read fail. " << std::endl;
        std::cout << "image_path = " << image_path << std::endl;
        exit(-1);
    }
    int input_image_width = img_input_test.Width();
    int input_image_height = img_input_test.Height();

    // Initialize the image converter
    panorama.PerspectiveParamsProcess(input_image_width, input_image_height, panorama_params);

    if(camera.ModelName().compare("OPENCV") == 0){

    }else if (camera.ModelName().compare("SPHERICAL") != 0) {
        if (camera.NumLocalCameras() == 2) {
            CHECK(perspective_image_count == 6 || perspective_image_count == 8 || perspective_image_count == 10);
        }

        large_fov_images[0] = std::make_shared<LargeFovImage>();
        large_fov_images[0]->SetCamera(camera);
        large_fov_images[0]->ParamPreprocess(perspective_image_width, perspective_image_height, fov_w,
                                             input_image_width, input_image_height);

        piecewise_images[0] = std::make_shared<PiecewiseImage>();
        piecewise_images[0]->SetCamera(camera);
        piecewise_images[0]->ParamPreprocess(perspective_image_width, perspective_image_height, fov_w,
                                             input_image_width, input_image_height, perspective_image_count / 2);

        for (const auto camera_id : cameras_in_reconstruction) {
            const auto& camera_in_reconstruction = reconstruction->Camera(camera_id);

            large_fov_images[camera_id] = std::make_shared<LargeFovImage>();
            large_fov_images[camera_id]->SetCamera(camera_in_reconstruction);
            large_fov_images[camera_id]->ParamPreprocess(perspective_image_width, perspective_image_height, fov_w,
                                                         input_image_width, input_image_height);

            piecewise_images[camera_id] = std::make_shared<PiecewiseImage>();
            piecewise_images[camera_id]->SetCamera(camera_in_reconstruction);
            piecewise_images[camera_id]->ParamPreprocess(perspective_image_width, perspective_image_height, fov_w,
                                                         input_image_width, input_image_height,
                                                         perspective_image_count / 2);
        }
    }

    auto start = std::chrono::high_resolution_clock::now();

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for (size_t i = 0; i < image_list.size(); i++) {
        auto image_path = image_list[i];
        // Load panorama images
        Bitmap img_input;
        std::string image_type;
        int use_recon_pose = 0;
        if (IsFileRGBD(image_path)) {
            image_type = "RGBD";
            if (!ExtractRGBDData(image_path, img_input, true)) {
                std::cerr << "RGBD image read fail. " << std::endl;
                std::cout << " Image Path = " << image_path << std::endl;
                continue;
            }
        } else {
            image_type = "Perspective";
            if (!img_input.Read(image_path, true)) {
                std::cerr << "seg image read fail. " << std::endl;
                std::cout << " Image Path = " << image_path << std::endl;
                continue;
            }
        }

        // Process the original image from panorama to perspective projection
        std::vector<Bitmap> img_outs;
        camera_t cur_camera_id = 0;

        if (camera.ModelName().compare("SPHERICAL") == 0 && camera.NumLocalCameras() == 1) {
            image_type = "Panorama";
            panorama.PanoramaToPerspectives(&img_input, img_outs);

            std::string image_name =
                image_path.substr(img_input_path.size(), image_path.size() - img_input_path.size() - 4);
            std::string image_folder_name = image_name.substr(0, image_name.rfind("/"));

            std::string device_folder_name = image_folder_name.substr(0, image_folder_name.rfind("/"));
            std::string result_device_folder_path = img_output_path + "/" + device_folder_name;
            if (!boost::filesystem::exists(result_device_folder_path)) {
                boost::filesystem::create_directories(result_device_folder_path);
            }

            std::string result_image_folder_path = img_output_path + image_folder_name;
            if (!boost::filesystem::exists(result_image_folder_path)) {
                boost::filesystem::create_directories(result_image_folder_path);
            }

            std::string result_image_path = img_output_path + image_name;
            std::string img_path = result_image_path + "_";
            int counter = 0;

            for (const auto& img_out : img_outs) {
                std::string cur_img_path = img_path + std::to_string(counter) + ".jpg";
                img_out.Write(cur_img_path, FIF_JPEG);
                counter++;
            }
        } else if (image_type != "RGBD" && camera.NumLocalCameras() <= 2) {
            image_type = "Insta One";
            // std::cout << "image_path = " << image_path << std::endl;
            std::string local_camera = image_path.substr(image_path.rfind("cam"), 4);
            camera_t local_camera_index = std::stoi(local_camera.substr(3));

            size_t cam_pos = image_path.rfind("cam");
            std::string image_name = image_path;
            std::string image_name_in_reconstruction = image_path;
            image_name_in_reconstruction.replace(cam_pos, 5, "cam0/");
            image_name.replace(cam_pos, 5, "");

            image_name =
                image_name.substr(img_input_path.size() + 1, image_name.size() - img_input_path.size() - 4 - 1);
            image_name_in_reconstruction = image_name_in_reconstruction.substr(
                img_input_path.size() + 1, image_name_in_reconstruction.size() - img_input_path.size() - 1);

            std::string image_folder_name = image_name;
            image_folder_name = image_name.substr(0, image_name.rfind("/"));
            std::string device_folder_name = image_folder_name.substr(0, image_folder_name.rfind("/"));

            std::string result_device_folder_path = img_output_path + "/" + device_folder_name;
            if (!boost::filesystem::exists(result_device_folder_path)) {
                boost::filesystem::create_directories(result_device_folder_path);
            }

            std::string result_image_folder_path = img_output_path + "/" + image_folder_name;
            if (!boost::filesystem::exists(result_image_folder_path)) {
                boost::filesystem::create_directories(result_image_folder_path);
            }

            std::string result_image_path = img_output_path + "/" + image_name;

            if (map_name_camera_id.find(image_name_in_reconstruction) != map_name_camera_id.end()) {
                cur_camera_id = map_name_camera_id.at(image_name_in_reconstruction);
            }

            piecewise_images[cur_camera_id]->ToSplitedPerspectives(img_input, img_outs, local_camera_index);
            // std::cout<<"cur_camera_id: "<<cur_camera_id<<std::endl;

            std::string img_path = result_image_path + "_";
            int counter = 0;
            for (const auto& img_out : img_outs) {
                std::string cur_img_path =
                    img_path + std::to_string(counter + local_camera_index * img_outs.size()) + ".jpg";
                img_out.Write(cur_img_path, FIF_JPEG);
                counter++;
            }
        } else if (camera.NumLocalCameras() > 2) {
            image_type = "Insta Pro2";
            std::string local_camera = image_path.substr(image_path.rfind("cam"), 4);
            camera_t local_camera_index = std::stoi(local_camera.substr(3));
            Bitmap img_out;

            size_t cam_pos = image_path.rfind("cam");
            std::string image_name = image_path;
            std::string image_name_in_reconstruction = image_path;
            image_name_in_reconstruction.replace(cam_pos, 5, "cam0/");
            image_name.replace(cam_pos, 5, "");

            image_name =
                image_name.substr(img_input_path.size() + 1, image_name.size() - img_input_path.size() - 4 - 1);
            image_name_in_reconstruction = image_name_in_reconstruction.substr(
                img_input_path.size() + 1, image_name_in_reconstruction.size() - img_input_path.size() - 1);

            std::string image_folder_name = image_name;
            image_folder_name = image_name.substr(0, image_name.rfind("/"));
            std::string device_folder_name = image_folder_name.substr(0, image_folder_name.rfind("/"));

            std::string result_device_folder_path = img_output_path + "/" + device_folder_name;
            if (!boost::filesystem::exists(result_device_folder_path)) {
                boost::filesystem::create_directories(result_device_folder_path);
            }

            std::string result_image_folder_path = img_output_path + "/" + image_folder_name;
            if (!boost::filesystem::exists(result_image_folder_path)) {
                boost::filesystem::create_directories(result_image_folder_path);
            }

            std::string result_image_path = img_output_path + "/" + image_name;

            if (map_name_camera_id.find(image_name_in_reconstruction) != map_name_camera_id.end()) {
                cur_camera_id = map_name_camera_id.at(image_name_in_reconstruction);
            }

            large_fov_images[cur_camera_id]->ToPerspective(img_input, img_out, local_camera_index);
            // std::cout<<"cur_camera_id: "<<cur_camera_id<<std::endl;

            std::string img_path = result_image_path + "_";

            std::string cur_img_path = img_path + std::to_string(local_camera_index) + ".jpg";
            img_out.Write(cur_img_path, FIF_JPEG);
        } else if (image_type == "RGBD") {
            // std::cout << "RGDB Image Process" << std::endl;
            std::string image_name =
                image_path.substr(img_input_path.size(), image_path.size() - img_input_path.size() - 4);
            std::string image_folder_name = image_name.substr(0, image_name.rfind("/"));
            // std::cout << "image_name = " << image_name << std::endl;
            // std::cout << "image_folder_name = " << image_folder_name << std::endl;

            std::string device_folder_name = image_folder_name.substr(0, image_folder_name.rfind("/"));
            std::string result_device_folder_path = img_output_path + "/" + device_folder_name;
            if (!boost::filesystem::exists(result_device_folder_path)) {
                boost::filesystem::create_directories(result_device_folder_path);
            }

            std::string result_image_folder_path = img_output_path + image_folder_name;
            if (!boost::filesystem::exists(result_image_folder_path)) {
                boost::filesystem::create_directories(result_image_folder_path);
            }

            std::string result_image_path = img_output_path + image_name;
            std::string img_path = result_image_path;

            std::string cur_img_path = img_path + "jpg";
            img_input.Write(cur_img_path, FIF_JPEG);
        }else if (image_type == "Perspective") {
            // std::cout << "RGDB Image Process" << std::endl;
            std::string image_name =
                image_path.substr(img_input_path.size(), image_path.size() - img_input_path.size() - 4);
            std::string image_folder_name = image_name.substr(0, image_name.rfind("/"));
            // std::cout << "image_name = " << image_name << std::endl;
            // std::cout << "image_folder_name = " << image_folder_name << std::endl;

            std::string device_folder_name = image_folder_name.substr(0, image_folder_name.rfind("/"));
            std::string result_device_folder_path = img_output_path + "/" + device_folder_name;
            if (!boost::filesystem::exists(result_device_folder_path)) {
                boost::filesystem::create_directories(result_device_folder_path);
            }

            std::string result_image_folder_path = img_output_path + image_folder_name;
            if (!boost::filesystem::exists(result_image_folder_path)) {
                boost::filesystem::create_directories(result_image_folder_path);
            }

            std::string result_image_path = img_output_path + image_name;
            std::string img_path = result_image_path;

            std::string cur_img_path = img_path + ".jpg";
            img_input.Write(cur_img_path, FIF_JPEG);
        }

#ifdef _OPENMP
#pragma omp critical(all_save_time)
#endif
        {
            bool use_reconstruction_pose = false;
            if (cur_camera_id != 0) {
                use_reconstruction_pose = true;
            }
            // Print the progress
            std::cout << "\r";
            std::cout << "Process Images [" << image_counter + 1 << " / " << image_list.size()
                      << "] Image Type : " << image_type
                      << " , Use Prior Intrinsic : " << std::to_string(use_reconstruction_pose) << std::flush;

            image_counter++;
        }
    }
    std::cout << std::endl;

    std::cout << "Cost time = "
              << (double)std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() -
                                                                          start)
                         .count() /
                     60
              << " min" << std::endl;

    return 0;
}