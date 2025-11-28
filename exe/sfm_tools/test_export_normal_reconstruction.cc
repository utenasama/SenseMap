// Copyright (c) 2019, SenseTime Group.
// All rights reserved.

#include <boost/filesystem/path.hpp>
#include <opencv2/opencv.hpp>

#include "base/camera.h"
#include "base/image.h"
#include "base/image_reader.h"
#include "base/point2d.h"
#include "base/pose.h"
#include "base/reconstruction_manager.h"
#include "graph/correspondence_graph.h"
#include "util/misc.h"

#include "container/feature_data_container.h"
#include "feature/extraction.h"
#include "feature/feature_matcher.h"
#include "feature/utils.h"

#include "controllers/cluster_mapper_controller.h"

#include "../Configurator_yaml.h"
#include "../option_parsing.h"
#include "base/version.h"

using namespace sensemap;

Eigen::Matrix3d EulerToRotationMatrix(double roll, double pitch, double yaw) {
    Eigen::Quaterniond q = Eigen::AngleAxisd(pitch / 180 * M_PI, Eigen::Vector3d::UnitY()) *
                           Eigen::AngleAxisd(roll / 180 * M_PI, Eigen::Vector3d::UnitX()) *
                           Eigen::AngleAxisd(yaw / 180 * M_PI, Eigen::Vector3d::UnitZ());

    return q.matrix();
}

bool IsImageExtValid(std::string& ext) {
    std::string EXT = ext;
    StringToUpper(&EXT);
    if (EXT == ".PNG" || EXT == ".JPG") {
        return true;
    }
    return false;
}

int main(int argc, char* argv[]) {

    PrintHeading("SenseMap.  Copyright(c) 2019, SenseTime Group.");
    PrintHeading(std::string("Version: sfm-export-")+__VERSION__);
    
    if (argc < 5) {
        std::cout
            << "Usage: convert-sfm-export 1.old_workspace_path 2.new_workspace_path 3.sfm-config.yaml 4.simplify_correspondence"
            << std::endl;
        return 1;
    }

    std::string out_path;
    std::string workspace_path;

    workspace_path = std::string(argv[1]);
    out_path = std::string(argv[2]);
    // Set camera number
    
    std::string sfm_config_file_path = std::string(argv[3]);
    Configurator param;
    param.Load(sfm_config_file_path.c_str());
    
    std::string camera_param_file = param.GetArgument("camera_param_file", "");
    int num_cameras = param.GetArgument("num_cameras", -1);
    int divide_camera_num = static_cast<int>(param.GetArgument("perspective_image_count", 6));
    const int perspective_width = static_cast<int>(param.GetArgument("perspective_image_width", 600));
    const int perspective_height = static_cast<int>(param.GetArgument("perspective_image_height", 600));
    const int fov_w = static_cast<int>(param.GetArgument("fov_w", 60));

    const int simplify_correspondence = argc == 5 ? atoi(argv[4]) : 0;
    
    bool camera_rig = false;
    
    OptionParser option_parser;
    ImageReaderOptions reader_options;
    option_parser.GetImageReaderOptions(reader_options,param);
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

    if(reader_options.num_local_cameras > 1){
        camera_rig = true;
    }


    std::cout << "simplified correspondence = " << simplify_correspondence << std::endl;

    // Create all Feature container
    auto feature_data_container = std::make_shared<FeatureDataContainer>();
    auto update_feature_data_container = std::make_shared<FeatureDataContainer>();
    auto reconstruction = std::make_shared<Reconstruction>();
    auto update_reconstruction = std::make_shared<Reconstruction>();
    auto scene_graph_container = std::make_shared<SceneGraphContainer>();
    auto update_scene_graph_container = std::make_shared<SceneGraphContainer>();

    
    // Load original feature
    if (boost::filesystem::exists(JoinPaths(workspace_path, "/local_cameras.bin"))) {
        feature_data_container->ReadCamerasBinaryData(JoinPaths(workspace_path, "/cameras.bin"), false);
        feature_data_container->ReadLocalCamerasBinaryData(JoinPaths(workspace_path, "/local_cameras.bin"));
    } 
    else {
        feature_data_container->ReadCamerasBinaryData(JoinPaths(workspace_path, "/cameras.bin"), camera_rig);
    }
    feature_data_container->ReadImagesBinaryData(workspace_path + "/features.bin");
    // Load Panorama feature
    feature_data_container->ReadSubPanoramaBinaryData(workspace_path + "/sub_panorama.bin");

    if (boost::filesystem::exists(JoinPaths(workspace_path, "/piece_indices.bin"))) {
        feature_data_container->ReadPieceIndicesBinaryData(JoinPaths(workspace_path, "/piece_indices.bin"));
    } 
    else if (boost::filesystem::exists(JoinPaths(workspace_path, "/piece_indices.txt"))) {
        feature_data_container->ReadPieceIndicesData(JoinPaths(workspace_path, "/piece_indices.txt"));
    } 
    

    // Load original scene graph
    scene_graph_container->ReadSceneGraphBinaryData(workspace_path + "/scene_graph.bin");
    const auto& correspondence_graph = scene_graph_container->CorrespondenceGraph();
    auto& update_correspondence_graph = update_scene_graph_container->CorrespondenceGraph();

    // Set new feature
    update_feature_data_container->SetImagePath(feature_data_container->GetImagePath());

    // Load reconstruction
    reconstruction->ReadReconstruction(workspace_path + "/0/");

    // Display the camera basic info
    auto cameras = reconstruction->Cameras();
    std::cout << "Camera number = " << cameras.size() << std::endl;

    for (auto camera : cameras){
        std::cout << "  Camera index = " << camera.first << std::endl;
        std::cout << "  Camera model = " << camera.second.ModelName() << std::endl;
        std::cout << "  Camera Height = " << camera.second.Height() << std::endl;
        std::cout << "  Camera Width = " << camera.second.Width() << std::endl;
        std::cout << "  Camera param = ";
        for (auto param : camera.second.Params()){
            std::cout << "  " << param;
        }
        std::cout << std::endl;

        // Sub Camera
        std::cout << " Sub Camera Number = " << camera.second.NumLocalCameras() << std::endl;
        if (camera.second.NumLocalCameras() > 1) {

            for (size_t local_camera_id = 0; local_camera_id < camera.second.NumLocalCameras(); ++local_camera_id) {
                std::cout <<"    Local Camera Id = " << local_camera_id << std::endl;
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
                std::cout << "  qvec = " << qvec[0] << " " << qvec[1] << " " << qvec[2] << " " << qvec[3] << " , tvec = "
                << tvec[0] << " " << tvec[1] << " " << tvec[2] << " ";
                std::cout << std::endl;
            }
        }
    }
    
    
    // Identify panorama or camera-rig
    std::unordered_map<camera_t, std::vector<Eigen::Matrix3d>> camera_rotations;
    for (auto camera : cameras){ 
        size_t local_camera_num = camera.second.NumLocalCameras();
        std::cout<<"local_camera_num: "<<local_camera_num<<std::endl;
        // Load rotation_matrixs
        std::vector<Eigen::Matrix3d> rotation_matrixs;
        rotation_matrixs.resize(divide_camera_num);
        if (camera.second.ModelName() == "SPHERICAL") {
            const double disturbation_roll = 0;
            const double disturbation_pitch = 0;
            const double disturbation_yaw = 0;

            double pitch_interval = 360.0 / static_cast<double>(divide_camera_num);

            for (size_t i = 0; i < divide_camera_num; ++i) {
                double roll, pitch, yaw;
                roll = disturbation_roll;
                pitch = disturbation_pitch + i * pitch_interval;
                yaw = disturbation_yaw;
                rotation_matrixs[i] = EulerToRotationMatrix(roll, pitch, yaw);
                rotation_matrixs[i].transposeInPlace();
            }
        } else if(camera.second.NumLocalCameras() == 2 && camera.second.ModelName() == "OPENCV_FISHEYE") { // panorama camera in rig

            double roll[5] =  {0,0,0,0,0};
            double yaw[5]  =  {-60,0,60,0,0};
            double pitch[5] = {0,0,0,-60,60};

            // get piecewise transforms    
            for (size_t i = 0; i < divide_camera_num / 2 ; ++i) {
                Eigen::Matrix3d transform;
                transform = Eigen::AngleAxisd(yaw[i] / 180 * M_PI, Eigen::Vector3d::UnitY()) *
                        Eigen::AngleAxisd(pitch[i] / 180 * M_PI, Eigen::Vector3d::UnitX()) *
                        Eigen::AngleAxisd(roll[i] / 180 * M_PI, Eigen::Vector3d::UnitZ());
                rotation_matrixs[i] = transform.transpose();
            }
        }
        else{
            rotation_matrixs[0]<< 1, 0, 0, 0, 1, 0, 0, 0, 1;
        }
        
        if(local_camera_num>2 || (local_camera_num == 2 && camera.second.ModelName() != "OPENCV_FISHEYE")){
            divide_camera_num = camera.second.NumLocalCameras();
        }


        camera_rotations[camera.second.CameraId()] = rotation_matrixs;
    }

    std::cout<<"divide_camera_num: "<<divide_camera_num<<std::endl;

    ///////////////////////////////////////////////////////////////////////////////////////
    // 1. Set new camera bin
    ///////////////////////////////////////////////////////////////////////////////////////
    std::cout << "1. Set new camera bin ... " << std::endl;
    std::shared_ptr<Camera> new_camera = std::make_shared<Camera>();
    new_camera->SetCameraId(1);
    new_camera->SetModelId(0);
    new_camera->SetWidth(perspective_width);
    new_camera->SetHeight(perspective_height);
    // PARAMS
    new_camera->Params().clear();
    double folcal_length = perspective_width * 0.5 / tan(fov_w / 360.0 * M_PI);
    new_camera->Params().push_back(folcal_length);
    new_camera->Params().push_back(perspective_width * 0.5);
    new_camera->Params().push_back(perspective_height * 0.5);
    update_feature_data_container->emplace(new_camera->CameraId(), new_camera);
    update_reconstruction->AddCamera(*new_camera.get());

    // Add Camera From previous which is not a spheric camera or rig
    std::unordered_map<camera_t, camera_t> camera_id_map;
    int camera_id_counter = 2;
    for (auto camera : cameras){
        if (camera.second.ModelName() != "SPHERICAL" && camera.second.NumLocalCameras() == 1) {
            
            std::shared_ptr<Camera> new_camera = std::make_shared<Camera>();
            new_camera->SetCameraId(camera_id_counter);
            new_camera->SetModelId(camera.second.ModelId());
            new_camera->SetWidth(camera.second.Width());
            new_camera->SetHeight(camera.second.Height());
            // PARAMS
            new_camera->Params().clear();
            new_camera->Params() = camera.second.Params();
            update_feature_data_container->emplace(new_camera->CameraId(), new_camera);
            update_reconstruction->AddCamera(*new_camera.get());

            camera_id_map[camera.second.CameraId()] = camera_id_counter;

            camera_id_counter++;
        }
    }


    ///////////////////////////////////////////////////////////////////////////////////////
    // 2. Divide panorama camera and create new image id
    ///////////////////////////////////////////////////////////////////////////////////////
    std::cout << "2. Divide panorama camera and create new image id ...." << std::endl;
    std::unordered_map<image_t, std::unordered_map<int, image_t>> image_id_map;
    std::unordered_map<image_t, std::unordered_map<point2D_t, std::pair<image_t, point2D_t>>> image_point2d_map;

    const auto& image_ids = feature_data_container->GetImageIds();
    image_t new_image_counter = 1;
    for (const auto& image_id : image_ids) {
        // Load feature data for current image
        const auto& image = feature_data_container->GetImage(image_id);
        const auto& camera = feature_data_container->GetCamera(image.CameraId());
        const auto& keypoints = feature_data_container->GetKeypoints(image_id);
        const auto& descriptors = feature_data_container->GetCompressedDescriptors(image_id);
        const auto& panorama_indexs = feature_data_container->GetPanoramaIndexs(image_id);
        const auto& piece_indexs = feature_data_container->GetPieceIndexs(image_id);

        size_t local_camera_num = camera.NumLocalCameras();

        // Create new FeatureDataPtr
        std::vector<FeatureDataPtr> feature_data_ptrs;

        if(child_paths.size()){
            int continue_count = 0;
            for(auto child_path : child_paths){
                if (!child_path.empty() && !IsInsideSubpath(image.Name(), child_path)) {
                    continue_count ++;
                }
            }
            if(continue_count == child_paths.size()){
                std::cout<<"continue "<<image.Name()<<std::endl;
                continue;
            }
        }
        

        std::string image_name = image.Name();
        std::string image_pth, image_ext;
        SplitFileExtension(image_name, &image_pth, &image_ext);
        if (!IsImageExtValid(image_ext)) {
            image_name = image_pth +".jpg";
        }
        
        if (camera.ModelName() != "SPHERICAL" && camera.NumLocalCameras() == 1) {
            // Genertate new image id
            image_id_map[image_id][0] = new_image_counter;
            FeatureDataPtr tmp_feature_ptr = std::make_shared<FeatureData>();
            feature_data_ptrs.push_back(std::move(tmp_feature_ptr));
            feature_data_ptrs[0]->image.SetImageId(new_image_counter);
            feature_data_ptrs[0]->image.SetCameraId(camera_id_map[image.CameraId()]);
            feature_data_ptrs[0]->image.SetName(image_name);
            feature_data_ptrs[0]->image.SetLabelId(image.LabelId());
            new_image_counter++;

            // 2D KeyPoint 
            feature_data_ptrs[0]->keypoints = keypoints;
            feature_data_ptrs[0]->compressed_descriptors = descriptors;

            for (size_t keypoint_id = 0; keypoint_id < keypoints.size(); keypoint_id++) {
                image_point2d_map[image_id][keypoint_id] = std::make_pair(image_id_map[image_id][0], keypoint_id);
            }

        } else {
            // Genertate new image id
            for (size_t perspective_image_id = 0; perspective_image_id < divide_camera_num; perspective_image_id++) {
                // image_id_map[image_id][perspective_image_id] = image_id * divide_camera_num + perspective_image_id;
                image_id_map[image_id][perspective_image_id] = new_image_counter;
                FeatureDataPtr tmp_feature_ptr = std::make_shared<FeatureData>();
                feature_data_ptrs.push_back(std::move(tmp_feature_ptr));
                // feature_data_ptrs[perspective_image_id]->image.SetImageId(image_id * divide_camera_num + perspective_image_id);
                feature_data_ptrs[perspective_image_id]->image.SetImageId(new_image_counter);
                feature_data_ptrs[perspective_image_id]->image.SetCameraId(1);  // Only one camera
                std::string new_image_name = StringPrintf(
                    "%s_%d.jpg", image.Name().substr(0, image.Name().size() - 4).c_str(), perspective_image_id);
                if (local_camera_num > 1){
                    new_image_name.replace(new_image_name.rfind("cam"),5,"");
                }
                feature_data_ptrs[perspective_image_id]->image.SetName(new_image_name);
                feature_data_ptrs[perspective_image_id]->image.SetLabelId(image.LabelId());
                new_image_counter++;
            }

            // 2D KeyPoint perspective image distribution
            std::vector<size_t> point2d_counters(divide_camera_num,0);
            for (size_t keypoint_id = 0; keypoint_id < keypoints.size(); keypoint_id++) {
                
                int perspective_image_id = panorama_indexs[keypoint_id].sub_image_id;

                if (local_camera_num == 2 && camera.ModelName() == "OPENCV_FISHEYE") {
                    perspective_image_id = panorama_indexs[keypoint_id].sub_image_id * (divide_camera_num / 2) +
                                        piece_indexs[keypoint_id].piece_id;
                }

                image_point2d_map[image_id][keypoint_id] =
                    std::make_pair(image_id_map[image_id][perspective_image_id], point2d_counters[perspective_image_id]);
                

                point2d_counters[perspective_image_id]++;

                auto new_keypoint = keypoints[keypoint_id];

                if(local_camera_num == 2 && camera.ModelName() == "OPENCV_FISHEYE"){
                    new_keypoint.x = piece_indexs[keypoint_id].piece_x;
                    new_keypoint.y = piece_indexs[keypoint_id].piece_y;
                }
                else{
                    new_keypoint.x = panorama_indexs[keypoint_id].sub_x;
                    new_keypoint.y = panorama_indexs[keypoint_id].sub_y;
                }

                feature_data_ptrs[perspective_image_id]->keypoints.emplace_back(new_keypoint);

                feature_data_ptrs[perspective_image_id]->compressed_descriptors.conservativeResize(
                    feature_data_ptrs[perspective_image_id]->keypoints.size(),descriptors.cols());
                feature_data_ptrs[perspective_image_id]->compressed_descriptors.row(
                    feature_data_ptrs[perspective_image_id]->keypoints.size() - 1) = descriptors.row(keypoint_id);
            }
        }

        
   
        // Emplace new feature data
        for (auto feature_data_ptr : feature_data_ptrs) {
            update_feature_data_container->emplace(feature_data_ptr->image.ImageId(), feature_data_ptr);
        }
    }

    ///////////////////////////////////////////////////////////////////////////////////////
    // 3. Update Image pose for all new images
    ///////////////////////////////////////////////////////////////////////////////////////
    std::cout << "3. Update Image pose for all new images ..." << std::endl;
    const auto& reconsutrction_image_ids = reconstruction->RegisterImageIds();
    for (const auto& image_id : reconsutrction_image_ids) {
        const auto cur_image = reconstruction->Image(image_id);
        const auto cur_camera = reconstruction->Camera(cur_image.CameraId());

        size_t local_camera_num = cur_camera.NumLocalCameras();

        if(child_paths.size()){
            int continue_count = 0;
            for(auto child_path : child_paths){
                if (!child_path.empty() && !IsInsideSubpath(cur_image.Name(), child_path)) {
                    continue_count++;
                }
            }
            if(continue_count == child_paths.size()){
                std::cout<<"continue "<<cur_image.Name()<<std::endl;
                continue;
            }
        }

        std::string image_name = cur_image.Name();
        std::string image_pth, image_ext;
        SplitFileExtension(image_name, &image_pth, &image_ext);
        if (!IsImageExtValid(image_ext)) {
            image_name = image_pth +".jpg";
        }
        
        if (cur_camera.ModelName() != "SPHERICAL" && cur_camera.NumLocalCameras() == 1) {
            Image new_image;

            // Update image id
            new_image.SetImageId(image_id_map[image_id][0]);
            new_image.SetCameraId(camera_id_map[cur_image.CameraId()]);

            new_image.SetQvec(cur_image.Qvec());
            new_image.SetTvec(cur_image.Tvec());

            new_image.SetName(image_name);
            // Reset point2d
            auto cur_point2ds = cur_image.Points2D();
            std::vector<class Point2D> new_point2d;
            for (int i = 0; i < cur_point2ds.size(); i++) {
                class Point2D new_point2D;
                new_point2D.SetXY(cur_point2ds[i].XY());
                new_point2d.emplace_back(new_point2D);
            }
            new_image.SetPoints2D(new_point2d);

            // Update reconstruction
            update_reconstruction->AddImage(new_image);
            update_reconstruction->RegisterImage(new_image.ImageId());

        } else {
            // Modified point2ds
            const auto& old_point2ds = cur_image.Points2D();
            std::unordered_map<int, std::vector<class Point2D>> new_point2Ds;  // Store new point2d
            for (size_t point_id = 0; point_id < old_point2ds.size(); point_id++) {
                class Point2D new_point2D;
                // Get new image id and point id
                auto new_image_to_point = image_point2d_map[image_id][point_id];
                auto new_image_id = new_image_to_point.first;
                auto new_point_id = new_image_to_point.second;

                // Skip the point2d not exist ???  CHECK instead
                CHECK(update_feature_data_container->ExistImage(new_image_id));
                CHECK(update_feature_data_container->GetKeypoints(new_image_id).size() > new_point_id);


                Eigen::Vector2d new_point2d_XY(update_feature_data_container->GetKeypoints(new_image_id)[new_point_id].x,
                                            update_feature_data_container->GetKeypoints(new_image_id)[new_point_id].y);

                int perspective_image_id = feature_data_container->GetPanoramaIndexs(image_id)[point_id].sub_image_id;
                if (local_camera_num == 2) {
                    perspective_image_id = feature_data_container->GetPanoramaIndexs(image_id)[point_id].sub_image_id *
                                            (divide_camera_num / 2) +
                                        feature_data_container->GetPieceIndexs(image_id)[point_id].piece_id;
                }

                new_point2D.SetXY(new_point2d_XY);
                new_point2Ds[perspective_image_id].emplace_back(new_point2D);
            }

            std::vector<Image> new_images;
            new_images.reserve(divide_camera_num);
            for (size_t perspective_image_id = 0; perspective_image_id < divide_camera_num; perspective_image_id++) {
                Image new_image;
                // Update image id
                new_image.SetImageId(image_id_map[image_id][perspective_image_id]);
                new_image.SetCameraId(1);

                // Update the camera rotation
                auto old_tvec = cur_image.Tvec();
                auto old_rot = cur_image.RotationMatrix();

                Eigen::Matrix3d new_rot;
                Eigen::Vector3d new_tvec;
                
                if(local_camera_num == 1){
                    const double *camera_params_data = cur_camera.ParamsData();
                    const double * local_qvec2_data = camera_params_data + 10;
                    const double * local_tvec2_data = camera_params_data + 14;
                    Eigen::Vector4d local_qvec2;
                    for(int i = 0; i<4; ++i){
                        local_qvec2[i]=local_qvec2_data[i];
                    }
                    Eigen::Vector3d local_tvec2;
                    for(int i = 0; i<3; ++i){
                        local_tvec2[i]=local_tvec2_data[i];
                    }
                    if(perspective_image_id == 2 || perspective_image_id == 3 ||perspective_image_id == 4){
                        old_rot = QuaternionToRotationMatrix(local_qvec2) * old_rot;
                        old_tvec = QuaternionToRotationMatrix(local_qvec2) * old_tvec + local_tvec2;
                    }

                    new_rot = camera_rotations[cur_image.CameraId()][perspective_image_id] * old_rot;  
                    new_tvec = camera_rotations[cur_image.CameraId()][perspective_image_id] * old_tvec;
                }
                else if(local_camera_num > 1){
                    int local_camera_id =
                        (local_camera_num == 2) ? (perspective_image_id / (divide_camera_num / 2)) : perspective_image_id;
                    int piece_id = (local_camera_num == 2) ? (perspective_image_id % (divide_camera_num / 2)) : 0;

                    Eigen::Vector4d local_qvec;
                    Eigen::Vector3d local_tvec;

                    cur_camera.GetLocalCameraExtrinsic(local_camera_id, local_qvec, local_tvec);

                    const Eigen::Matrix3d local_camera_R =
                    QuaternionToRotationMatrix(local_qvec) * old_rot;

                    const Eigen::Vector3d local_camera_T =
                    QuaternionToRotationMatrix(local_qvec) * old_tvec + local_tvec;

                    if(local_camera_num == 2 && cur_camera.ModelName() == "OPENCV_FISHEYE"){
                        new_rot = camera_rotations[cur_image.CameraId()][piece_id] * local_camera_R;
                        new_tvec = camera_rotations[cur_image.CameraId()][piece_id] * local_camera_T; 
                    }
                    else{
                        new_rot = local_camera_R;
                        new_tvec = local_camera_T;
                    }               
                }


                auto new_qvec = RotationMatrixToQuaternion(new_rot);
                new_image.SetQvec(new_qvec);
                new_image.SetTvec(new_tvec);

                std::string new_image_name = StringPrintf(
                    "%s_%d.jpg", cur_image.Name().substr(0, cur_image.Name().size() - 4).c_str(), perspective_image_id);
                if (local_camera_num > 1){
                    new_image_name.replace(new_image_name.rfind("cam"),5,"");
                }
                new_image.SetName(new_image_name);
                new_image.SetPoints2D(new_point2Ds[perspective_image_id]);

                // Update reconstruction
                update_reconstruction->AddImage(new_image);
                update_reconstruction->RegisterImage(new_image.ImageId());
            }
        }


        
    }

    ///////////////////////////////////////////////////////////////////////////////////////
    // 4. Update 3d point track id
    ///////////////////////////////////////////////////////////////////////////////////////
    std::cout << "4. Update 3d point track id ... " << std::endl;
    const auto& mappoint_ids = reconstruction->MapPointIds();
    for (const auto& mappoint_id : mappoint_ids) {
        class MapPoint new_mappoint;
        // Get old mappoint
        const auto old_mappoint = reconstruction->MapPoint(mappoint_id);

        // Use the old mappoint position
        new_mappoint.SetXYZ(old_mappoint.XYZ());
        new_mappoint.SetColor(old_mappoint.Color());
        new_mappoint.SetError(old_mappoint.Error());

        // Update the old mappoint track with new image id and point2d id
        class Track new_track;
        for (const auto& track_el : old_mappoint.Track().Elements()) {
            if(child_paths.size()){
                int continue_count = 0;
                for(auto child_path : child_paths){
                    if (!child_path.empty() && !IsInsideSubpath(reconstruction->Image(track_el.image_id).Name(), child_path)) {
                        continue_count++;
                    }
                }
                if(continue_count == child_paths.size()){
                    continue;
                }
            }
            // if (!child_path.empty() && !IsInsideSubpath(reconstruction->Image(track_el.image_id).Name(), child_path)) {
            //     continue;
            // }


            auto new_image_to_point = image_point2d_map[track_el.image_id][track_el.point2D_idx];
            
            CHECK(update_feature_data_container->ExistImage(new_image_to_point.first));
            CHECK(update_feature_data_container->GetKeypoints(new_image_to_point.first).size() > new_image_to_point.second);
            
            new_track.AddElement(new_image_to_point.first, new_image_to_point.second);
        }
        // Check track size
        if(new_track.Length() <= 2){
            continue;
        }

        new_mappoint.SetTrack(new_track);

        // FIXME:  Error not written
        // Update reconstruction
        update_reconstruction->AddMapPointWithError(new_mappoint.XYZ(), std::move(new_mappoint.Track()), new_mappoint.Color(),
                                                    new_mappoint.Error());
    }

    ///////////////////////////////////////////////////////////////////////////////////////
    // 5. Update scene graph
    ///////////////////////////////////////////////////////////////////////////////////////
    std::cout << "5. Update scene graph ... " << std::endl;

    auto image_pairs = correspondence_graph->ImagePairs();

    // Structure to store the new feature coorespondence data
    std::unordered_map<image_pair_t, FeatureMatches> new_matches;

    std::unordered_map<image_t, struct CorrespondenceGraph::Image> corr_images;
    std::unordered_map<image_pair_t, int> new_image_pairs;
    int skip_counter = 0;
    for (const auto& image_pair : image_pairs) {
        const auto image_pair_id = image_pair.first;
        const auto& image_pair_struct = image_pair.second;
        // Skip some image pair
        if (image_pair_struct.num_correspondences < 30) {
            skip_counter++;
            continue;
        }

        // Get two image ids
        image_t image_id_1, image_id_2;
        utility::PairIdToImagePair(image_pair_id, &image_id_1, &image_id_2);

        if(child_paths.size()){
            int continue_count = 1;
            for(auto child_path : child_paths){
                if (!IsInsideSubpath(feature_data_container->GetImage(image_id_1).Name(), child_path) ||
                    !IsInsideSubpath(feature_data_container->GetImage(image_id_2).Name(), child_path)) {
                    continue_count = 1;
                }
            }
            if(continue_count == child_paths.size()){
                continue;
            }
        }

        // if (!child_path.empty()) {
        //     if (!IsInsideSubpath(feature_data_container->GetImage(image_id_1).Name(), child_path) ||
        //         !IsInsideSubpath(feature_data_container->GetImage(image_id_2).Name(), child_path)) {
        //             continue;
        //         }
        // }


        // Calculate the original matchs
        const auto& original_matchs = correspondence_graph->FindCorrespondencesBetweenImages(image_id_1, image_id_2);

        for (const auto& original_match : original_matchs) {
            const auto original_point_id_1 = original_match.point2D_idx1;
            const auto original_point_id_2 = original_match.point2D_idx2;

            image_t new_image_id_1 = image_point2d_map[image_id_1][original_point_id_1].first;
            image_t new_image_id_2 = image_point2d_map[image_id_2][original_point_id_2].first;

            point2D_t new_point2d_id_1 = image_point2d_map[image_id_1][original_point_id_1].second;
            point2D_t new_point2d_id_2 = image_point2d_map[image_id_2][original_point_id_2].second;

            image_pair_t new_image_pair_id = utility::ImagePairToPairId(new_image_id_1, new_image_id_2);
            if (!new_image_pairs.count(new_image_pair_id)) {
                new_image_pairs[new_image_pair_id] = 0;
            }
            new_image_pairs[new_image_pair_id]++;
            if (!corr_images.count(new_image_id_1)) {
                struct CorrespondenceGraph::Image image_1;
                image_1.num_observations = update_feature_data_container->GetKeypoints(new_image_id_1).size();
                image_1.corrs.resize(image_1.num_observations);
                corr_images.emplace(new_image_id_1, image_1);
            }

            if (!corr_images.count(new_image_id_2)) {
                struct CorrespondenceGraph::Image image_2;
                image_2.num_observations = update_feature_data_container->GetKeypoints(new_image_id_2).size();
                image_2.corrs.resize(image_2.num_observations);
                corr_images.emplace(new_image_id_2, image_2);
            }

            // Corresponding images.
            struct CorrespondenceGraph::Image& image1 = corr_images.at(new_image_id_1);
            struct CorrespondenceGraph::Image& image2 = corr_images.at(new_image_id_2);

            // Store number of correspondences for each image to find good initial pair.
            image1.num_correspondences += 1;
            image2.num_correspondences += 1;

            const bool valid_idx1 = new_point2d_id_1 < image1.corrs.size();
            const bool valid_idx2 = new_point2d_id_2 < image2.corrs.size();

            if (valid_idx1 && valid_idx2) {
                auto& corrs1 = image1.corrs[new_point2d_id_1];
                auto& corrs2 = image2.corrs[new_point2d_id_2];

                const bool duplicate1 = std::find_if(corrs1.begin(), corrs1.end(),
                                                     [new_image_id_2](const CorrespondenceGraph::Correspondence& corr) {
                                                         return corr.image_id == new_image_id_2;
                                                     }) != corrs1.end();

                const bool duplicate2 = std::find_if(corrs2.begin(), corrs2.end(),
                                                     [new_image_id_1](const CorrespondenceGraph::Correspondence& corr) {
                                                         return corr.image_id == new_image_id_1;
                                                     }) != corrs2.end();

                if (duplicate1 || duplicate2) {
                    image1.num_correspondences -= 1;
                    image2.num_correspondences -= 1;
                    std::cout << StringPrintf(
                                     "WARNING: Duplicate correspondence between "
                                     "point2D_idx=%d in image_id=%d and point2D_idx=%d in "
                                     "image_id=%d",
                                     new_point2d_id_1, new_image_id_1, new_point2d_id_2, new_point2d_id_2)
                              << std::endl;
                } else {
                    corrs1.emplace_back(new_image_id_2, new_point2d_id_2);
                    corrs2.emplace_back(new_image_id_1, new_point2d_id_1);
                }
            } else {
                image1.num_correspondences -= 1;
                image2.num_correspondences -= 1;
                if (!valid_idx1) {
                    std::cout << StringPrintf(
                                     "WARNING: point2D_idx=%d in image_id=%d does not exist, the corr size = %d",
                                     new_point2d_id_1, new_image_id_1, image1.corrs.size())
                              << std::endl;
                }
                if (!valid_idx2) {
                    std::cout << StringPrintf(
                                     "WARNING: point2D_idx=%d in image_id=%d does not exist, the corr size = %d",
                                     new_point2d_id_2, new_image_id_2, image2.corrs.size())
                              << std::endl;
                }
            }
        }
    }

    for (const auto& image : corr_images) {
        update_correspondence_graph->AddImage(image.first, image.second);
    }

    for (const auto& image_pair : new_image_pairs) {
        struct CorrespondenceGraph::ImagePair pair;
        image_t image_id_1, image_id_2;
        utility::PairIdToImagePair(image_pair.first, &image_id_1, &image_id_2);
        pair.num_correspondences = image_pair.second;
        pair.image_id1 = image_id_1;
        pair.image_id2 = image_id_2;
        if (simplify_correspondence == 1 && pair.num_correspondences < 30) {
            skip_counter++;
            continue;
        }
        update_correspondence_graph->AddCorrespondences(image_id_1, image_id_2, pair);
    }
    std::cout << "Original panorama image pair number = " << image_pairs.size() << std::endl;
    std::cout << "Original perspective image pair number = " << new_image_pairs.size() << std::endl;
    std::cout << "skip perspective image pair number = " << skip_counter << std::endl;

    update_correspondence_graph->Finalize();

    if (boost::filesystem::exists(out_path)) {
        boost::filesystem::remove_all(out_path);
    }
    boost::filesystem::create_directories(out_path);

    
    update_feature_data_container->WriteImagesBinaryData(out_path + "/features.bin",true);
    update_reconstruction->WriteReconstruction(out_path, true);
    update_scene_graph_container->WriteSceneGraphBinaryData(out_path + "/scene_graph.bin");

    return 0;
}
