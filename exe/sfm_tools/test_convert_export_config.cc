// Copyright (c) 2019, SenseTime Group.
// All rights reserved.

#include <boost/filesystem/path.hpp>
#include <opencv2/opencv.hpp>
// #include <unordered_map>

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
#include "../system_io.h"

using namespace sensemap;

std::string out_path;
std::string workspace_path;

#define debug

int round_double(double number) { return (number > 0.0) ? floor(number + 0.5) : ceil(number - 0.5); }

Eigen::Matrix3d EulerToRotationMatrix(const double yaw, double pitch, double roll) {
    Eigen::Quaterniond q = Eigen::AngleAxisd(yaw / 180 * M_PI, Eigen::Vector3d::UnitY()) *
                           Eigen::AngleAxisd(pitch / 180 * M_PI, Eigen::Vector3d::UnitX()) *
                           Eigen::AngleAxisd(roll / 180 * M_PI, Eigen::Vector3d::UnitZ());

    return q.matrix();
}

int main(int argc, char* argv[]) {

    PrintHeading("SenseMap.  Copyright(c) 2020, SenseTime Group.");
    PrintHeading("Version: convert-sfm-export-config-1.6.3");

    if (argc < 5) {
        std::cout << "Usage: convert_panorama 1.old_workspace_path 2.new_workspace_path 3.sfm-config.yaml 4.simplify_correspondence"
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
    
    
    int divide_camera_num = static_cast<int>(param.GetArgument("perspective_image_count", 6));
    int perspective_width = static_cast<int>(param.GetArgument("perspective_image_width", 600));
    int perspective_height = static_cast<int>(param.GetArgument("perspective_image_height", 600));
    int fov_w = static_cast<int>(param.GetArgument("fov_w", 90));

    const int simplify_correspondence = argc == 5 ? atoi(argv[4]) : 1;
    
    bool camera_rig = false;
    
    OptionParser option_parser;
    ImageReaderOptions reader_options;
    option_parser.GetImageReaderOptions(reader_options,param);
    if(reader_options.num_local_cameras > 1){
        camera_rig = true;
    }


    std::cout << "simplified correspondence = " << simplify_correspondence << std::endl;
    
    std::string panorama_config_file = param.GetArgument("panorama_config_file", "");
    if(panorama_config_file.empty()){
        std::cout << "Panoamra Config File is empty ... " << std::endl;
    }
    std::vector<PanoramaParam> panorama_params;
    LoadParams(panorama_config_file, panorama_params);


    // Calculate the map between the old panorama index to the new panorama index
    std::unordered_map<int, std::vector<int>> panorama_index_map;
    double old_yaw_interval = 360.0 / static_cast<double>(divide_camera_num);
    for (size_t i = 0; i < panorama_params.size(); i++) {
        int old_index = round_double(panorama_params[i].yaw / old_yaw_interval);
        panorama_index_map[old_index].emplace_back(i);
    }

#ifdef debug
    for (const auto& panorama_index : panorama_index_map) {
        std::cout << "old_panorama_index = " << panorama_index.first << std::endl;
        for (const auto& n_panorama_index : panorama_index.second) {
            std::cout << " " << n_panorama_index << std::endl;
        }
    }
#endif

    // Create reconstruction container
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
    } else {
        feature_data_container->ReadCamerasBinaryData(JoinPaths(workspace_path, "/cameras.bin"), camera_rig);
    }
    feature_data_container->ReadImagesBinaryData(workspace_path + "/features.bin");
    // Load Panorama feature
    feature_data_container->ReadSubPanoramaBinaryData(workspace_path + "/sub_panorama.bin");

    if (boost::filesystem::exists(JoinPaths(workspace_path, "/piece_indices.bin"))) {
        feature_data_container->ReadPieceIndicesBinaryData(JoinPaths(workspace_path, "/piece_indices.bin"));
    } else if (boost::filesystem::exists(JoinPaths(workspace_path, "/piece_indices.txt"))) {
        feature_data_container->ReadPieceIndicesData(JoinPaths(workspace_path, "/piece_indices.txt"));
    } 
    

    // Load original scene graph
    scene_graph_container->ReadSceneGraphBinaryData(workspace_path + "/scene_graph.bin");
    const auto& correspondence_graph = scene_graph_container->CorrespondenceGraph();
    auto update_correspondence_graph = update_scene_graph_container->CorrespondenceGraph();

    // Set new feature
    update_feature_data_container->SetImagePath(feature_data_container->GetImagePath());

    // Load reconstruction
    reconstruction->ReadReconstruction(workspace_path + "/0/");

    
    // Identify panorama or camera-rig
    const Camera& camera = feature_data_container->GetCamera(1); 
    size_t local_camera_num = camera.NumLocalCameras();
    std::cout<<"local_camera_num: "<<local_camera_num<<std::endl;

    if (camera.ModelName() == "SPHERICAL") {
        divide_camera_num = panorama_params.size();
    } else if(camera.NumLocalCameras() > 2 && camera.ModelName() == "OPENCV_FISHEYE") {
        divide_camera_num = camera.NumLocalCameras();
    }

    // Load rotation_matrixs
    Eigen::Matrix3d rotation_matrixs[divide_camera_num];
    
    if (camera.ModelName() == "SPHERICAL") {
        for (size_t i = 0; i < divide_camera_num; i++) {
            double roll, pitch, yaw;
            roll = panorama_params[i].roll;
            pitch = panorama_params[i].pitch;
            yaw = panorama_params[i].yaw;
            rotation_matrixs[i] = EulerToRotationMatrix(yaw, pitch, roll);
            rotation_matrixs[i].transposeInPlace();
        }

    } else if(camera.NumLocalCameras() == 2 && camera.ModelName() == "OPENCV_FISHEYE"){ // panorama camera in OneX rig
        double roll[5] =  {0,0,0,0,0};
        double yaw[5]  =  {-60,0,60,0,0};
        double pitch[5] = {0,0,0,-60,60};

        // get piecewise transforms    
        for (size_t i = 0; i < divide_camera_num / 2 ; ++i) {
            Eigen::Matrix3d transform;
            transform = EulerToRotationMatrix(yaw[i], pitch[i], roll[i]);
            rotation_matrixs[i] = transform.transpose();
        }
    } else {
        rotation_matrixs[0]<< 1, 0, 0, 0, 1, 0, 0, 0, 1;
    }

    std::cout<<"divide_camera_num: "<<divide_camera_num<<std::endl;

    ///////////////////////////////////////////////////////////////////////////////////////
    // 1. Set new camera bin
    ///////////////////////////////////////////////////////////////////////////////////////
    std::cout << "1. Set new camera bin ... " << std::endl;
    for (size_t i = 0; i < panorama_params.size(); i++) {
        std::shared_ptr<Camera> new_camera = std::make_shared<Camera>();
        new_camera->SetCameraId(i+1);
        new_camera->SetModelId(0);

        if (camera.ModelName() == "SPHERICAL") {
            perspective_width = panorama_params[i].pers_w;
            perspective_height = panorama_params[i].pers_h;
            fov_w = panorama_params[i].fov_w;
        }

        new_camera->SetWidth(perspective_width);
        new_camera->SetHeight(perspective_height);

        // PARAMS
        new_camera->Params().clear();
        double folcal_length;
        folcal_length = perspective_width * 0.5 / tan(fov_w / 360.0 * M_PI);
        new_camera->Params().push_back(folcal_length);
        new_camera->Params().push_back(perspective_width * 0.5);
        new_camera->Params().push_back(perspective_height * 0.5);
        update_reconstruction->AddCamera(*new_camera.get());
    }

    ///////////////////////////////////////////////////////////////////////////////////////
    // 2. Update Image id using feature data
    ///////////////////////////////////////////////////////////////////////////////////////
    std::cout << "2. Update Feature data and generate new image id  ... " << std::endl;
    std::unordered_map<image_t, std::unordered_map<int, image_t>> image_id_map;
    std::unordered_map<image_t, std::unordered_map<point2D_t, std::vector<image_t>>> image_point2d_map;
    
    const auto& image_ids = feature_data_container->GetImageIds();
    image_t new_image_counter = 1;
    for (const auto& image_id : image_ids) {
        // Load feature data for current image
        const auto image = feature_data_container->GetImage(image_id);
        const auto keypoints = feature_data_container->GetKeypoints(image_id);
        const auto descriptors = feature_data_container->GetDescriptors(image_id);
        const auto panorama_indexs = feature_data_container->GetPanoramaIndexs(image_id);
        const auto piece_indexs = feature_data_container->GetPieceIndexs(image_id);

        // Create new FeatureDataPtr
        std::vector<FeatureDataPtr> feature_data_ptrs;

        // Genertate new image id
        for (size_t perspective_image_id = 0; perspective_image_id < divide_camera_num; perspective_image_id++) {
            image_id_map[image_id][perspective_image_id] = new_image_counter;
            FeatureDataPtr tmp_feature_ptr = std::make_shared<FeatureData>();
            feature_data_ptrs.push_back(std::move(tmp_feature_ptr));
            feature_data_ptrs[perspective_image_id]->image.SetImageId(new_image_counter);
            feature_data_ptrs[perspective_image_id]->image.SetCameraId(perspective_image_id + 1);
            std::string new_image_name = StringPrintf(
                "%s_%d.jpg", image.Name().substr(0, image.Name().size() - 4).c_str(), perspective_image_id);
            feature_data_ptrs[perspective_image_id]->image.SetName(new_image_name);
            feature_data_ptrs[perspective_image_id]->image.SetLabelId(image.LabelId());

            // Create empty keypoint and empty descriptor
            feature_data_ptrs[perspective_image_id]->descriptors.resize(1, 128);
            new_image_counter++;
        }

        // 2D KeyPoint perspective image distribution
        for (size_t keypoint_id = 0; keypoint_id < keypoints.size(); keypoint_id++) {
            int sub_image_id = panorama_indexs[keypoint_id].sub_image_id;
            
            if (camera.ModelName() == "SPHERICAL") {
                for (const auto& new_perspective_image_id : panorama_index_map[sub_image_id]) {
                    image_point2d_map[image_id][keypoint_id].emplace_back(image_id_map[image_id][new_perspective_image_id]);
                }
            } else if (local_camera_num == 2 && camera.ModelName() == "OPENCV_FISHEYE") {
                sub_image_id = panorama_indexs[keypoint_id].sub_image_id * (divide_camera_num / 2) + piece_indexs[keypoint_id].piece_id;
                image_point2d_map[image_id][keypoint_id].emplace_back(image_id_map[image_id][sub_image_id]);
            } else {
                image_point2d_map[image_id][keypoint_id].emplace_back(image_id_map[image_id][sub_image_id]);
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

        std::vector<Image> new_images;
        new_images.reserve(divide_camera_num);
        for (size_t perspective_image_id = 0; perspective_image_id < divide_camera_num; perspective_image_id++) {
            Image new_image;
            // Update image id
            new_image.SetImageId(image_id_map[image_id][perspective_image_id]);
            new_image.SetCameraId(perspective_image_id + 1);

            // Update the camera rotation
            auto old_tvec = cur_image.Tvec();
            auto old_rot = cur_image.RotationMatrix();

            Eigen::Matrix3d new_rot;
            Eigen::Vector3d new_tvec;

            if (local_camera_num == 1) {
                new_rot = rotation_matrixs[perspective_image_id] * old_rot;  
                new_tvec = rotation_matrixs[perspective_image_id] * old_tvec;
            } else if(local_camera_num > 1) {
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

                if(local_camera_num == 2 && camera.ModelName() == "OPENCV_FISHEYE"){
                    new_rot = rotation_matrixs[piece_id] * local_camera_R;
                    new_tvec = rotation_matrixs[piece_id] * local_camera_T; 
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
            new_image.SetName(new_image_name);

            // Update reconstruction
            update_reconstruction->AddImage(new_image);
            update_reconstruction->RegisterImage(new_image.ImageId());
        }
    }

    ///////////////////////////////////////////////////////////////////////////////////////
    // 3. Generate empty point3d
    ///////////////////////////////////////////////////////////////////////////////////////

    // FIXME:
    ///////////////////////////////////////////////////////////////////////////////////////
    // 4. Update scene graph
    ///////////////////////////////////////////////////////////////////////////////////////
    std::cout << "4. Update scene graph ... " << std::endl;

    auto image_pairs = correspondence_graph->ImagePairs();

    std::unordered_map<image_t, struct Image> images;
    std::unordered_map<image_pair_t, int> new_image_pairs;

    for (const auto& image_pair : image_pairs) {
        const auto image_pair_id = image_pair.first;
        const auto& image_pair_struct = image_pair.second;

        // Get two image ids
        image_t image_id_1, image_id_2;
        utility::PairIdToImagePair(image_pair_id, &image_id_1, &image_id_2);

        // Calculate the original matchs
        const auto& original_matchs = correspondence_graph->FindCorrespondencesBetweenImages(image_id_1, image_id_2);

        for (const auto& original_match : original_matchs) {
            const point2D_t original_point_id_1 = original_match.point2D_idx1;
            const point2D_t original_point_id_2 = original_match.point2D_idx2;

            auto new_image_ids_1 = image_point2d_map[image_id_1][original_point_id_1];
            auto new_image_ids_2 = image_point2d_map[image_id_2][original_point_id_2];

            for (auto new_image_id_1 : new_image_ids_1) {
                for (auto new_image_id_2 : new_image_ids_2) {
                    if (new_image_id_1 % panorama_params.size() != new_image_id_2 % panorama_params.size()) {
                        continue;
                    }

                    if (!update_correspondence_graph->ExistsImage(new_image_id_1)) {
                        struct CorrespondenceGraph::Image image;
                        image.num_observations = 0;
                        image.corrs.resize(0);
                        update_correspondence_graph->AddImage(new_image_id_1, image);
                    }

                    if (!update_correspondence_graph->ExistsImage(new_image_id_2)) {
                        struct CorrespondenceGraph::Image image;
                        image.num_observations = 0;
                        image.corrs.resize(0);
                        update_correspondence_graph->AddImage(new_image_id_2, image);
                    }

                    // Calculate new image pair id
                    image_pair_t new_image_pair_id = utility::ImagePairToPairId(new_image_id_1, new_image_id_2);
                    if (!new_image_pairs.count(new_image_pair_id)) {
                        new_image_pairs[new_image_pair_id] = 0;
                    }
                    new_image_pairs[new_image_pair_id]++;
                }
            }
        }
    }

    int skip_counter = 0;
    for (const auto& image_pair : new_image_pairs) {
        struct CorrespondenceGraph::ImagePair pair;
        image_t image_id_1, image_id_2;
        utility::PairIdToImagePair(image_pair.first, &image_id_1, &image_id_2);
        pair.num_correspondences = image_pair.second;
        pair.image_id1 = image_id_1;
        pair.image_id2 = image_id_2;
        if(pair.num_correspondences < 30){
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

    update_feature_data_container->WriteImagesBinaryData(out_path + "/features.bin");
    update_reconstruction->WriteReconstruction(out_path, true);
    update_scene_graph_container->WriteSceneGraphBinaryData(out_path + "/scene_graph.bin");

    return 0;
}
