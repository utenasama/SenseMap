//Copyright (c) 2021, SenseTime Group.
//All rights reserved.
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <vector>

#include "../Configurator_yaml.h"
#include "../camera_rig_params.h"
#include "../option_parsing.h"
#include "../system_io.h"

#include "base/common.h"
#include "base/camera.h"
#include "base/image.h"
#include "base/image_reader.h"
#include "base/point2d.h"
#include "base/pose.h"
#include "base/camera_models.h"
#include "base/undistortion.h"
#include "base/reconstruction_manager.h"
#include "base/projection.h"

#include "lidar/voxel_map.h"

#include "graph/correspondence_graph.h"
#include "controllers/incremental_mapper_controller.h"

#include "lidar/pcd.h"
#include "lidar/lidar_sweep.h"

#include "util/ply.h"
#include "util/proc.h"
#include "util/misc.h"
#include "util/threading.h"
#include "util/timer.h"
#include "util/exception_handler.h"

#include "base/version.h"

using namespace sensemap;

std::string workspace_path;
static std::unordered_map<std::string, image_t> image_name_map;

void ColorMap(const float intensity, uint8_t &r, uint8_t &g, uint8_t &b){
    if(intensity <= 0.25) {
        r = 0;
        g = 0;
        b = (intensity) * 4 * 255;
        return;
    }

    if(intensity <= 0.5) {
        r = 0;
        g = (intensity - 0.25) * 4 * 255;
        b = 255;
        return;
    }

    if(intensity <= 0.75) {
        r = 0;
        g = 255;
        b = (0.75 - intensity) * 4 * 255;
        return;
    }

    if(intensity <= 1.0) {
        r = (intensity - 0.75) * 4 * 255;
        g = (1.0 - intensity) * 4 * 255;
        b = 0;
        return;
    }
};

void ReadRigList(std::vector<RigNames>& list, const std::string file_path){

    if (!ExistsFile(file_path)){
        std::cout << "file is empty, " << file_path << std::endl;
        return;
    }
    std::ifstream ifs;
    //打开文件
    ifs.open(file_path.c_str(), std::ios::in);
    //定义一个字符串
    std::string str;
    //从文件中读取数据
    while(getline(ifs, str))
    {
        // std::cout << str << std::endl;
        std::string pcd = str;

        std::string img1,img2,img3;
        getline(ifs, img1);
        getline(ifs, img2);
        getline(ifs, img3);
        
        RigNames rig_name;
        rig_name.Init("points/" + GetPathBaseName(pcd), "cam0/" + GetPathBaseName(img1));

        list.push_back(rig_name);
    }
    std::cout << "read in " << list.size() << " frame." << std::endl;
    return;
}

bool LoadFeaturesAndMatches(FeatureDataContainer &feature_data_container, SceneGraphContainer &scene_graph,
                            Configurator &param) {
    std::string workspace_path = param.GetArgument("workspace_path", "");
    CHECK(!workspace_path.empty());
    if (static_cast<bool>(param.GetArgument("map_update", 0))){
        workspace_path = JoinPaths(workspace_path, "/map_update");
    }

    OptionParser option_parser;
    ImageReaderOptions reader_options;
    option_parser.GetImageReaderOptions(reader_options, param);

    bool camera_rig = (reader_options.num_local_cameras > 1);
    bool use_gps_prior = param.GetArgument("use_gps_prior", 0);

    if (!boost::filesystem::exists(JoinPaths(workspace_path, "/cameras.bin")) ||
        !boost::filesystem::exists(JoinPaths(workspace_path, "/features.bin")) ||
        !boost::filesystem::exists(JoinPaths(workspace_path, "/scene_graph.bin"))) {
        return false;
    }

    float availabel_memeory;
    GetAvailableMemory(availabel_memeory);
    std::cout << "Available Memory: " << availabel_memeory << "GB" << std::endl;

    // load feature data
    if (boost::filesystem::exists(JoinPaths(workspace_path, "/local_cameras.bin"))) {
        feature_data_container.ReadCamerasBinaryData(JoinPaths(workspace_path, "/cameras.bin"), false);
        feature_data_container.ReadLocalCamerasBinaryData(JoinPaths(workspace_path, "/local_cameras.bin"));
    } else {
        feature_data_container.ReadCamerasBinaryData(JoinPaths(workspace_path, "/cameras.bin"), camera_rig);
    }

    feature_data_container.ReadImagesBinaryDataWithoutDescriptor(JoinPaths(workspace_path, "/features.bin"));
    if (boost::filesystem::exists(JoinPaths(workspace_path, "/local_images.bin"))) {
        feature_data_container.ReadLocalImagesBinaryData(JoinPaths(workspace_path, "/local_images.bin"));
    }

    if (boost::filesystem::exists(JoinPaths(workspace_path, "/sub_panorama.bin"))) {
        feature_data_container.ReadSubPanoramaBinaryData(JoinPaths(workspace_path, "/sub_panorama.bin"));
    } else {
        std::cout << " Warning! Existing feature data do not contain sub_panorama data" << std::endl;
    }

    if (boost::filesystem::exists(JoinPaths(workspace_path, "/piece_indices.bin"))) {
        feature_data_container.ReadPieceIndicesBinaryData(JoinPaths(workspace_path, "/piece_indices.bin"));
    } else {
        std::cout << " Warning! Existing feature data do not contain piece_indices data" << std::endl;
    }

    if (static_cast<bool>(param.GetArgument("detect_apriltag", 0))) {
        if (boost::filesystem::exists(JoinPaths(workspace_path, "/apriltags.bin"))) {
            feature_data_container.ReadAprilTagBinaryData(JoinPaths(workspace_path, "/apriltags.bin"));
        } else {
            std::cout << " Warning! Existing feature data do not contain AprilTags data" << std::endl;
        }
    }

    if (use_gps_prior) {
        if (boost::filesystem::exists(JoinPaths(workspace_path, "/gps.bin"))) {
            feature_data_container.ReadGPSBinaryData(JoinPaths(workspace_path, "/gps.bin"));
        }
    }

    GetAvailableMemory(availabel_memeory);
    std::cout << "Available Memory[LoadFeatures]: " << availabel_memeory << "GB" << std::endl;

    // load match data
    scene_graph.ReadSceneGraphBinaryData(JoinPaths(workspace_path, "/scene_graph.bin"));

    GetAvailableMemory(availabel_memeory);
    std::cout << "Available Memory[LoadSceneGraph]: " << availabel_memeory << "GB" << std::endl;

    if (boost::filesystem::exists(JoinPaths(workspace_path, "/two_view_geometry.bin"))) {
        scene_graph.ReadImagePairsBinaryData(JoinPaths(workspace_path, "/two_view_geometry.bin"));
    }
    if (boost::filesystem::exists(JoinPaths(workspace_path, "/strong_loops.bin"))) {
        std::cout<<"read strong_loops file "<<std::endl;
        scene_graph.ReadStrongLoopsBinaryData(JoinPaths(workspace_path, "/strong_loops.bin"));
    }
    if (boost::filesystem::exists(JoinPaths(workspace_path, "/loop_pairs.bin"))) {
        std::cout<<"read loop_pairs file "<<std::endl;
        scene_graph.ReadLoopPairsInfoBinaryData(JoinPaths(workspace_path, "/loop_pairs.bin"));
    }
    if (boost::filesystem::exists(JoinPaths(workspace_path, "/normal_pairs.bin"))) {
        std::cout<<"read normal_pairs file "<<std::endl;
        scene_graph.ReadNormalPairsBinaryData(JoinPaths(workspace_path, "/normal_pairs.bin"));
    }

    EIGEN_STL_UMAP(image_t, class Image) &images = scene_graph.Images();
    EIGEN_STL_UMAP(camera_t, class Camera) &cameras = scene_graph.Cameras();

    std::vector<image_t> image_ids = feature_data_container.GetImageIds();

    for (const auto image_id : image_ids) {
        const Image &image = feature_data_container.GetImage(image_id);
        if (!scene_graph.CorrespondenceGraph()->ExistsImage(image_id)) {
            continue;
        }

        images[image_id] = image;

        const FeatureKeypoints &keypoints = feature_data_container.GetKeypoints(image_id);
        images[image_id].SetPoints2D(keypoints);
        const PanoramaIndexs &panorama_indices = feature_data_container.GetPanoramaIndexs(image_id);

        const Camera &camera = feature_data_container.GetCamera(image.CameraId());

        std::vector<uint32_t> local_image_indices(keypoints.size());
        for (size_t i = 0; i < keypoints.size(); ++i) {
            if (panorama_indices.size() == 0 && camera.NumLocalCameras() == 1) {
                local_image_indices[i] = image_id;
            } else {
                local_image_indices[i] = panorama_indices[i].sub_image_id;
            }
        }
        images[image_id].SetLocalImageIndices(local_image_indices);
        images[image_id].SetLocalNames(image.LocalNames());
        images[image_id].SetLocalQvecsPrior(image.LocalQvecsPrior());
        images[image_id].SetLocalTvecsPrior(image.LocalTvecsPrior());

        if (!scene_graph.ExistsCamera(image.CameraId())) {
            cameras[image.CameraId()] = camera;
        }

        if (scene_graph.CorrespondenceGraph()->ExistsImage(image_id)) {
            images[image_id].SetNumObservations(scene_graph.CorrespondenceGraph()->NumObservationsForImage(image_id));
            images[image_id].SetNumCorrespondences(
                scene_graph.CorrespondenceGraph()->NumCorrespondencesForImage(image_id));
        } else {
            std::cout << "Do not contain ImageId = " << image_id << ", in the correspondence graph." << std::endl;
        }
    }
    scene_graph.CorrespondenceGraph()->Finalize();

    GetAvailableMemory(availabel_memeory);
    std::cout << "Available Memory[FinalizeSceneGraph]: " << availabel_memeory << "GB" << std::endl;

    return true;
}

// i = 0 front camera, i = 1 left camera, i = 2 right camera;
Eigen::Matrix3x4d GetLidarTrans(int i){
    std::vector<Eigen::Matrix4d> cam_to_front_vec;
    Eigen::Matrix4d cam0_to_front = Eigen::Matrix4d::Identity();
    Eigen::Matrix4d cam_left_to_front = Eigen::Matrix4d::Identity();
    Eigen::Matrix4d cam_right_to_front = Eigen::Matrix4d::Identity();
    cam_left_to_front<<   -0.003417538338, -0.00089147387 , -0.999993762834 ,-0.042917186469,
                        -0.02046564317,   0.999790219413, -0.000821349858,  0.000342557085,
                        0.999784715767 , 0.020462708528, -0.003435065991 ,-0.049072652799,
                        0.     ,         0.  ,            0.  ,            1.  ;
    std::cout << "cam_left_to_front: " << cam_left_to_front << std::endl;
    cam_right_to_front<< 0.001001133243, -0.013031389532,  0.999914586662,  0.051744013617,
                        0.007862121995,  0.999884285959,  0.013023122934 , 0.00033177879,
                        -0.999968591892,  0.007848412584,  0.001103471772, -0.043182895903,
                        0.             , 0.             , 0.             , 1.       ;
    std::cout << "cam_right_to_front: " << cam_right_to_front << std::endl;
    cam_to_front_vec.push_back(cam0_to_front);
    cam_to_front_vec.push_back(cam_left_to_front);
    cam_to_front_vec.push_back(cam_right_to_front);  

    Eigen::Matrix4d T_cam0_to_lidar;
    T_cam0_to_lidar << -0.008360209691, -0.016227256072,  0.999833377646,  0.051347537513,
                        -0.999963222325, -0.001777492671, -0.008390144037,  0.004504013824,
                        0.001913345517, -0.999866749462, -0.01621179906 , -0.032669067397,
                        0.            ,  0.            ,  0.            ,  1.           ;

    //lidar to img
    Eigen::Matrix4d T_lidar_to_cam0 = T_cam0_to_lidar.inverse();
        
    Eigen::Matrix4d cam_to_front = cam_to_front_vec[i];
    Eigen::Matrix4d T_lidar_to_cam = cam_to_front.inverse()  * T_lidar_to_cam0;
    return T_lidar_to_cam.topRows(3);
}
void ReconstructionBA(std::shared_ptr<SceneGraphContainer> scene_graph_container,
                      std::shared_ptr<Reconstruction> reconstruction, 
                      Configurator param = Configurator(), bool sweep_init = false){
    
    std::shared_ptr<IncrementalMapper> mapper = 
        std::make_shared<IncrementalMapper>(scene_graph_container);
    mapper->BeginReconstruction(reconstruction);
    mapper->SetWorkspacePath(workspace_path);

    IndependentMapperOptions mapper_options;
    OptionParser option_parser;
    option_parser.GetMapperOptions(mapper_options,param);
    const auto inc_options = mapper_options.IncrementalMapperOptions();

    BundleAdjustmentConfig ba_config;
    auto ba_options = mapper_options.LocalBundleAdjustment();
    ba_options.max_num_iteration_frame2frame = 10;
    
    std::unordered_set<sweep_t> fix_sweep_ids;
    fix_sweep_ids.insert(reconstruction->RegisterSweepIds().at(0));
    double final_cost;
    mapper->AdjustFrame2FrameBundle(
        inc_options, 
        ba_options, 
        reconstruction->RegisterSweepIds(), 
        fix_sweep_ids,
        final_cost);

}


std::vector<sensemap::PlyPoint> RebuildPcd(PCDPointCloud& pc){
  std::vector<sensemap::PlyPoint> ply_points;
  ply_points.reserve(pc.info.num_points);

  float max_intensity = 0.0f;
  // std::vector<float> intensity_v;
  for (size_t i = 0; i < pc.info.num_points; i++){
    // max_curv = std::max(max_curv, points[i].curv);
    long int pnt_height = i % pc.info.height;
    long int pnt_width = i / pc.info.height;
    max_intensity = std::max(max_intensity, pc.point_cloud[pnt_height][pnt_width].intensity);
    // intensity_v.push_back(pc.point_cloud[pnt_height][pnt_width].intensity);
  }
  const double max_value = std::log1p(max_intensity);
  std::cout << "\t=> max_intensity: " << max_intensity << ", " << pc.info.height << " x " << pc.info.width << std::endl;

  std::vector<int> scan_ids;
  scan_ids.reserve(pc.info.num_points);
  int num_scan = 0;
  int num_per_scan = 4;
  int max_scan = 0;
  for (int i = 0; i < pc.info.num_points; i++){
    int line_id = i % num_per_scan;
    scan_ids.push_back(num_scan + line_id);
    if (num_scan + line_id > max_scan){
        max_scan = num_scan + line_id;
    }

    if (i < num_per_scan || i % num_per_scan != 0){
        continue;
    }

    PlyPoint pnt_new, pnt_back;
    {
        long int pnt_height = i % pc.info.height;
        long int pnt_width = i / pc.info.height;
        if (!pc.point_cloud[pnt_height][pnt_width].is_valid){
            continue;
        }
        pnt_new.x = pc.point_cloud[pnt_height][pnt_width].x;
        pnt_new.y = pc.point_cloud[pnt_height][pnt_width].y;
        pnt_new.z = pc.point_cloud[pnt_height][pnt_width].z;
    }
    {
        long int pnt_height = (i - num_per_scan)% pc.info.height;
        long int pnt_width = (i - num_per_scan) / pc.info.height;
        if (!pc.point_cloud[pnt_height][pnt_width].is_valid){
            continue;
        }
        pnt_back.x = pc.point_cloud[pnt_height][pnt_width].x;
        pnt_back.y = pc.point_cloud[pnt_height][pnt_width].y;
        pnt_back.z = pc.point_cloud[pnt_height][pnt_width].z;
    }
    if (pnt_new.x > 0 && pnt_new.y * pnt_back.y < 0){
        num_scan += 4;
    }
  }

  std::cout << "Pc scan = " << num_scan << ", " << scan_ids.size() << std::endl;

  std::vector<std::vector<sensemap::PlyPoint>> scan_points;
  scan_points.resize(max_scan + 1);
  for (int i = 1; i < pc.info.num_points; i++){
    PlyPoint pnt;
    long int pnt_height = (i) % pc.info.height;
    long int pnt_width = (i) / pc.info.height;
    if (!pc.point_cloud[pnt_height][pnt_width].is_valid){
        continue;
    }
    pnt.x = pc.point_cloud[pnt_height][pnt_width].x;
    pnt.y = pc.point_cloud[pnt_height][pnt_width].y;
    pnt.z = pc.point_cloud[pnt_height][pnt_width].z;
    // ColorMap(pc.point_cloud[pnt_height][pnt_width].intensity,
    //          pnt.r, pnt.g, pnt.b);
    // ColorMap(float(pnt_height) / 32,
    //           pnt.r, pnt.g, pnt.b);

    // const double value = std::log1p(pc.point_cloud[pnt_height][pnt_width].intensity) / max_value;
    // const double value = double(pnt_width) / pc.info.num_points;
    // const double value = double(int(i / 4) % 48) / 48;
    // const double value = double(i) / pc.info.num_points;
    const double value = double(scan_ids.at(i)) / num_scan;
    uint8_t r = 255 * JetColormap::Red(value);
    uint8_t g = 255 * JetColormap::Green(value);
    uint8_t b = 255 * JetColormap::Blue(value);

    pnt.r = r;
    pnt.g = g;
    pnt.b = b;
    ply_points.push_back(pnt);
    scan_points.at(scan_ids.at(i)).push_back(pnt);
  }

  for (int i = 0; i < num_scan; i++){
    std::cout << "\t=> C-" << i << ", " << scan_points.at(i).size() << std::endl;
    WriteBinaryPlyPoints("/home/SENSETIME/zhangzhuang/data/lidar-test/static/points-new/0/" + std::to_string(i) + ".ply", scan_points.at(i));
  }

  return ply_points;
};

int main(int argc, char** argv) {

    PrintHeading("SenseMap.  Copyright(c) 2021, SenseTime Group.");
    PrintHeading(std::string("Version: ") + __VERSION__);

    std::string configuration_file_path = std::string(argv[1]);
    Configurator param;
    param.Load(configuration_file_path.c_str());


    std::string image_path = param.GetArgument("image_path", "");
    workspace_path = param.GetArgument("workspace_path", "");
    CHECK(!workspace_path.empty()) << "workspace path empty";
    std::string lidar_path = param.GetArgument("lidar_path", "");
    CHECK(!lidar_path.empty()) << "lidar_path path empty";



    // auto reconstruction_path = JoinPaths(workspace_path, std::to_string(0));
    // if (!ExistsDir(reconstruction_path)){
    //     return 0;
    // }   
    // auto reconstruction = std::make_shared<Reconstruction>();
    // reconstruction->ReadReconstruction(reconstruction_path);
    // {

    //     // Create all Feature container
    //     auto feature_data_container = std::make_shared<FeatureDataContainer>();
    //     if (boost::filesystem::exists(JoinPaths(workspace_path, "/local_cameras.bin"))) {
    //         feature_data_container->ReadCamerasBinaryData(JoinPaths(workspace_path, "/cameras.bin"), false);
    //         feature_data_container->ReadLocalCamerasBinaryData(JoinPaths(workspace_path, "/local_cameras.bin"));
    //     } else {
    //         feature_data_container->ReadCamerasBinaryData(JoinPaths(workspace_path, "/cameras.bin"), true);
    //     }
    //     feature_data_container->ReadImagesBinaryData(workspace_path + "/features.bin");
    //     // Load Panorama feature
    //     feature_data_container->ReadSubPanoramaBinaryData(workspace_path + "/sub_panorama.bin");
    //     if (boost::filesystem::exists(JoinPaths(workspace_path, "/piece_indices.bin"))) {
    //         feature_data_container->ReadPieceIndicesBinaryData(JoinPaths(workspace_path, "/piece_indices.bin"));
    //     } else if (boost::filesystem::exists(JoinPaths(workspace_path, "/piece_indices.txt"))) {
    //         feature_data_container->ReadPieceIndicesData(JoinPaths(workspace_path, "/piece_indices.txt"));
    //     }

    //     std::cout << "Convert Rig cameras to Normal" << std::endl;
    //     const auto image_sort_ids = reconstruction->RegisterImageSortIds();
    //     for (auto image_id : image_sort_ids) {
    //         class Image& cur_image = reconstruction->Image(image_id);
    //         class Camera& cur_camera = reconstruction->Camera(cur_image.CameraId());

    //         const FeatureKeypoints &keypoints = feature_data_container->GetKeypoints(image_id);
    //         const PanoramaIndexs & panorama_indices = feature_data_container->GetPanoramaIndexs(image_id);

    //         std::vector<uint32_t> local_image_indices(keypoints.size(), 0);
    //         for(size_t i = 0; i < keypoints.size(); ++i){
    //             local_image_indices[i] = panorama_indices[i].sub_image_id;
    //         }

    //         cur_image.SetLocalImageIndices(local_image_indices);
    //     }
    // }

    // reconstruction->ExtractColorsForAllImages(image_path);

    // {
    //     std::string rec_path = JoinPaths(workspace_path, "lidar/0");
    //     if (!ExistsPath(GetParentDir(rec_path))) {
    //         boost::filesystem::create_directories(GetParentDir(rec_path));
    //     }
    //     Reconstruction rig_rec;
    //     reconstruction->ConvertRigReconstruction(rig_rec);
    //     std::cout << "rig_rec done " << std::endl;

    //     image_t start_image_id = 1;
    //     camera_t start_camera_id = 1;
    //     for(auto image: rig_rec.Images()){
    //         if(image.first > start_image_id){
    //             start_image_id = image.first;
    //         }
    //         if (start_camera_id < image.second.CameraId()){
    //             start_camera_id = image.second.CameraId();
    //         }
    //     }

    //     for (auto cam : rig_rec.Cameras()){
    //         if (start_camera_id < cam.first){
    //             start_camera_id = cam.first;
    //         }
    //     }
    //     start_image_id ++;
    //     start_camera_id ++;

    //     Camera camera_new = reconstruction->Camera(1);
    //     camera_new.SetCameraId(start_camera_id);
    //     rig_rec.AddCamera(camera_new);

    //     image_t new_idx = 1;
    //     std::cout << "Start: " << start_image_id << ", " << start_camera_id << std::endl;
    //     std::vector<sweep_t> registered_sweep_ids = reconstruction->RegisterSweepIds();
    //     for (sweep_t sweep_id : registered_sweep_ids) {
    //         class LidarSweep lidarsweep = reconstruction->LidarSweep(sweep_id);

    //         class Image image_sweep;
    //         image_sweep.SetImageId(start_image_id+new_idx); 
    //         image_sweep.SetName("prior_" + lidarsweep.Name());
    //         image_sweep.SetCameraId(start_camera_id);

    //         image_sweep.SetTvec(lidarsweep.Tvec());
    //         image_sweep.SetQvec(lidarsweep.Qvec());

    //         rig_rec.AddImage(image_sweep);
    //         rig_rec.RegisterImage(image_sweep.ImageId());

    //         new_idx++;
    //     }

    //     rig_rec.WriteReconstruction(rec_path, true);
    //     rig_rec.ExportMapPoints(rec_path + "/sparse.ply");
    // }

    // return 0;


    // std::string lidar_prior_pose_file = param.GetArgument("lidar_prior_pose_file", "");


    auto scene_graph_container = std::make_shared<SceneGraphContainer>();

    // auto file_names = GetRecursiveFileList(lidar_path);
    // std::vector<std::string> file_names;
    // file_names.push_back("/home/SENSETIME/zhangzhuang/data/lidar-test/test/points/1718695990.300101.pcd");
    // // file_names.push_back("/home/SENSETIME/zhangzhuang/data/lidar-test/test/points/1718695991.500118.pcd");
    // auto tmp_reconstruction = std::make_shared<Reconstruction>();
    // std::cout << "Fine num files : " << file_names.size() << std::endl;
    // int num_read = 0;
    // for(int i = 0; i < file_names.size(); i++){
    //     if (file_names.at(i).substr(file_names.at(i).length() - 4) != ".pcd"){
    //         continue;
    //     }
    //     std::cout << "=> Add Lidar (" << num_read++ << ") " << file_names.at(i) << std::endl;
    //     // tmp_reconstruction->AddLidarData(lidar_path, GetPathBaseName(file_names.at(i)));
    //     auto pc = ReadPCD(file_names.at(i));
    //     std::cout << "read pc: " << pc.info.num_points << std::endl;
    //     RebuildLivoxMid360(pc, 4);
    //     std::cout << "RebuildLivoxMid360, new_pc: " << pc.info.num_points << std::endl;
    //     // auto ply = RebuildPcd(new_pc);
    //     auto ply = Convert2Ply(pc);
    //     WriteBinaryPlyPoints(file_names.at(i) + "-1.ply", ply);
    //     std::cout << "Write to " << file_names.at(i) + "-1.ply" << std::endl;
    // }
    
    // return 0;
    workspace_path = "/home/SENSETIME/zhangzhuang/Project/SenseMap/0";
    std::vector<std::shared_ptr<Reconstruction>> reconstructions;
    // // 1
    // {
    //     auto tmp_reconstruction = std::make_shared<Reconstruction>();
    //     tmp_reconstruction->AddLidarData(GetParentDir(workspace_path), "/data/points/1703124196.967064.pcd");
    //     tmp_reconstruction->AddLidarData(GetParentDir(workspace_path), "/data/points/1703124198.964255.pcd");
    //     reconstructions.push_back(tmp_reconstruction);
    // }
    {
        auto tmp_reconstruction = std::make_shared<Reconstruction>();

        const auto sweep_id1 = tmp_reconstruction->AddLidarData(GetParentDir(workspace_path), "/0/1722482822.500025.pcd");
        tmp_reconstruction->LidarSweep(sweep_id1).SetQvec(Eigen::Vector4d(0.702257, -0.666851, 0.00764485, -0.279485));
        tmp_reconstruction->LidarSweep(sweep_id1).SetTvec(Eigen::Vector3d(1.07214, 0.382698, 0.952677));
        tmp_reconstruction->RegisterLidarSweep(sweep_id1);

        const auto sweep_id2 = tmp_reconstruction->AddLidarData(GetParentDir(workspace_path), "/0/1722482859.300220.pcd");
        tmp_reconstruction->LidarSweep(sweep_id2).SetQvec(Eigen::Vector4d(0.557199, -0.784536, -0.261304, 0.0758427));
        tmp_reconstruction->LidarSweep(sweep_id2).SetTvec(Eigen::Vector3d(0.467712, 1.28053, 0.1926));
        tmp_reconstruction->RegisterLidarSweep(sweep_id2);

        reconstructions.push_back(tmp_reconstruction);
    }

    {
        auto tmp_reconstruction = std::make_shared<Reconstruction>();

        const auto sweep_id2 = tmp_reconstruction->AddLidarData(GetParentDir(workspace_path), "/0/1722482859.300220.pcd");
        tmp_reconstruction->LidarSweep(sweep_id2).SetQvec(Eigen::Vector4d(0.557199, -0.784536, -0.261304, 0.0758427));
        tmp_reconstruction->LidarSweep(sweep_id2).SetTvec(Eigen::Vector3d(0.467712, 1.28053, 0.1926));
        tmp_reconstruction->RegisterLidarSweep(sweep_id2);

        const auto sweep_id1 = tmp_reconstruction->AddLidarData(GetParentDir(workspace_path), "/0/1722482822.500025.pcd");
        tmp_reconstruction->LidarSweep(sweep_id1).SetQvec(Eigen::Vector4d(0.702257, -0.666851, 0.00764485, -0.279485));
        tmp_reconstruction->LidarSweep(sweep_id1).SetTvec(Eigen::Vector3d(1.07214, 0.382698, 0.952677));
        tmp_reconstruction->RegisterLidarSweep(sweep_id1);

        reconstructions.push_back(tmp_reconstruction);
    }
    
    for (int i = 0; i < reconstructions.size(); i++){
        auto reconstruction = reconstructions.at(i);

        std::string pair_name = "";
        for (const auto& sweep : reconstruction->LidarSweeps()){
            auto& sweep_rect = reconstruction->LidarSweep(sweep.first);
        //     sweep_rect.SetQvec(Eigen::Vector4d(1, 0, 0, 0));
        //     sweep_rect.SetTvec(Eigen::Vector3d(0, 0, 0));
        //     reconstruction->RegisterLidarSweep(sweep.first);
            std::cout << "RegisterLidarSweep: " << sweep.first << ", " 
                << sweep_rect.Name() << "("
                << reconstruction->LidarSweep(sweep.first).Qvec().transpose() 
                << ", " << reconstruction->LidarSweep(sweep.first).Tvec().transpose()
                << ")" << std::endl;
            pair_name += GetPathBaseName(sweep_rect.Name()) + "_";
        }

        reconstruction->OutputLidarPointCloud2World(
            workspace_path+"/lidar_before/" + pair_name, true);

        ReconstructionBA(scene_graph_container, reconstruction, param, true);

        reconstruction->OutputLidarPointCloud2World(
            workspace_path+"/lidar_after/" + pair_name, true);
    }


    //////////////////////////////////////////////////////////////////
    /*
    const auto sweep_ids = reconstruction->RegisterSweepIds();
    if(sweep_ids.size() != 2) {
        std::cout << "sweep_ids.size(): " << sweep_ids.size() << std::endl;
        return 0;
    }
    auto& sweep_ref = reconstruction->LidarSweep(sweep_ids.at(0));
    sweep_ref.NormalizeQvec();
    Eigen::Matrix4d Tr_ref_inv_4d = Eigen::Matrix4d::Identity();
    Tr_ref_inv_4d.topRows(3) = sweep_ref.InverseProjectionMatrix();

    auto& sweep_src = reconstruction->LidarSweep(sweep_ids.at(1));
    sweep_src.NormalizeQvec();
    Eigen::Matrix4d Tr_src_inv_4d = Eigen::Matrix4d::Identity();
    Tr_src_inv_4d.topRows(3) = sweep_src.InverseProjectionMatrix();
    Eigen::Matrix4d Tr_ref2src_4d = Tr_src_inv_4d.inverse() * Tr_ref_inv_4d;

    std::vector<struct LidarEdgeCorrespondence> ref_edge_point_corr;
    std::vector<struct LidarEdgeCorrespondence> src_edge_point_corr;
    if (CornerCorrespondence(sweep_ref, sweep_src, Tr_ref2src_4d, 
                            ref_edge_point_corr) < 15 || 
        CornerCorrespondence(sweep_src, sweep_ref, Tr_ref2src_4d, 
                            src_edge_point_corr) < 15){
        std::cout << "edge_point_corr size: " << ref_edge_point_corr.size() 
            << ", " << src_edge_point_corr.size() << std::endl;
        return 1;
    };
    std::cout << "edge_point_corr size: " << ref_edge_point_corr.size() 
            << ", " << src_edge_point_corr.size() << std::endl;

    const auto ref_corners = sweep_ref.GetCornerPointsSharp();
    const auto ref_less_corners = sweep_ref.GetCornerPointsLessSharp();

    Eigen::Matrix4f T_ref = Eigen::Matrix4f::Identity();
    T_ref.topRows(3) = sweep_ref.InverseProjectionMatrix().cast<float>();

    const auto src_corners = sweep_src.GetCornerPointsSharp();
    const auto src_less_corners = sweep_src.GetCornerPointsLessSharp();
    Eigen::Matrix4f T_src = Eigen::Matrix4f::Identity();
    T_src.topRows(3) = sweep_src.InverseProjectionMatrix().cast<float>();;

    int num_edge_ref = ref_edge_point_corr.size();
    std::vector<float> dists(num_edge_ref);
    std::vector<Eigen::Vector3f> normals(num_edge_ref);
    double mean = 0;
    for (int i = 0; i < num_edge_ref; i++){
        const auto cor_id = ref_edge_point_corr.at(i);
        const auto &ref = ref_corners.points.at(cor_id.ref_idx);
        Eigen::Vector4f pnt_ref_4d(ref.x, ref.y, ref.z, 1.0);
        Eigen::Vector3f pnt_ref = (T_ref * pnt_ref_4d).topRows(3);

        const auto& src_a = src_less_corners.points.at(cor_id.src_a_idx);
        Eigen::Vector4f pnt_src_a_4d(src_a.x, src_a.y, src_a.z, 1.0);
        Eigen::Vector3f pnt_src_a = (T_src * pnt_src_a_4d).topRows(3);

        const auto& src_b = src_less_corners.points.at(cor_id.src_b_idx);
        Eigen::Vector4f pnt_src_b_4d(src_b.x, src_b.y, src_b.z, 1.0);
        Eigen::Vector3f pnt_src_b = (T_src * pnt_src_b_4d).topRows(3);

        normals.at(i) = (pnt_src_a - pnt_src_b).normalized();
        auto nu = (pnt_ref - pnt_src_a).cross(pnt_ref - pnt_src_b);
        dists.at(i) = nu.norm() / (pnt_src_b - pnt_src_a).norm();
        mean += dists.at(i);
    }
    mean /= num_edge_ref;
    std::cout << "mean: " << mean << std::endl;
    
    std::vector<std::unordered_set<int>> lines;
    std::unordered_map<int, size_t> points_lineid;

    std::unordered_map<int, std::unordered_set<int>> line_ids;
    for (int i = 0; i < num_edge_ref; i++){
        const auto cor_id = ref_edge_point_corr.at(i);
        const auto &ref = ref_corners.points.at(cor_id.ref_idx);
        Eigen::Vector4f pnt_ref_4d(ref.x, ref.y, ref.z, 1.0);
        Eigen::Vector3f pnt_ref = (T_ref * pnt_ref_4d).topRows(3);

        const auto& src_a = src_less_corners.points.at(cor_id.src_a_idx);
        Eigen::Vector4f pnt_src_a_4d(src_a.x, src_a.y, src_a.z, 1.0);
        Eigen::Vector3f pnt_src_a = (T_src * pnt_src_a_4d).topRows(3);

        const auto& src_b = src_less_corners.points.at(cor_id.src_b_idx);
        Eigen::Vector4f pnt_src_b_4d(src_b.x, src_b.y, src_b.z, 1.0);
        Eigen::Vector3f pnt_src_b = (T_src * pnt_src_b_4d).topRows(3);

        bool is_similar = false;
        int point_id = -1;
        for (int j = 0; j < i; j++){
            float theat = std::abs(normals.at(i).dot(normals.at(j)));
            if (theat < 0.9){
                continue;
            }

            const auto cor_id_j = ref_edge_point_corr.at(j);
            const auto &ref_j = ref_corners.points.at(cor_id_j.ref_idx);
            Eigen::Vector4f pnt_ref_4d_j(ref_j.x, ref_j.y, ref_j.z, 1.0);
            Eigen::Vector3f pnt_ref_j = (T_ref * pnt_ref_4d_j).topRows(3);
            if ((pnt_ref_j-pnt_ref).norm() > 1){
                continue;
            }

            auto nu = (pnt_ref_j - pnt_src_a).cross(pnt_ref_j - pnt_src_b);
            auto dist = nu.norm() / (pnt_src_b - pnt_src_a).norm();
            if (dist > mean * 2){
                continue;
            }
            is_similar = true;
            point_id = j;
            // line_ids[i].insert(j);
            break;
        }
        if (is_similar){
            auto line_id = points_lineid[point_id];
            lines.at(line_id).insert(i);
        } else {
            std::unordered_set<int> temp_line;
            temp_line.insert(i);
            points_lineid[i] = lines.size();
            lines.push_back(temp_line);
        }
    }

    std::cout << "line size: " << lines.size() << std::endl;

    std::vector<PlyPoint> ref_corner_ply, src_corner_ply;
    ref_corner_ply.reserve(ref_edge_point_corr.size());
    src_corner_ply.reserve(ref_edge_point_corr.size() * 2);

    int valid_lines = 0;
    cv::RNG rng(12345);
    int nr_color = rng.uniform(0, 255);
    int ng_color = rng.uniform(0, 255);
    int nb_color = rng.uniform(0, 255);
    for (size_t i = 0; i < lines.size(); i++){
        int r_color = 0;
        int g_color = 0;
        int b_color = 0;

        if (lines.at(i).size() > 3){
            r_color = nr_color;
            g_color = ng_color;
            b_color = nb_color;
            valid_lines++;
        }

        for (const auto line : lines.at(i)){
            PlyPoint a_pnt;
            const auto& a = ref_edge_point_corr.at(line).ref_point_;
            Eigen::Vector4f a_v4(a.x(), a.y(), a.z(), 1.0);
            a_v4 = T_ref * a_v4;
            a_pnt.x = a_v4.x();
            a_pnt.y = a_v4.y();
            a_pnt.z = a_v4.z();
            a_pnt.r = r_color;
            a_pnt.g = g_color;
            a_pnt.b = b_color;
            ref_corner_ply.push_back(a_pnt);


            PlyPoint b1_pnt, b2_pnt;
            const auto& b1 = ref_edge_point_corr.at(line).src_point_a_;
            Eigen::Vector4f b1_v4(b1.x(), b1.y(), b1.z(), 1.0);
            b1_v4 = T_src * b1_v4;
            b1_pnt.x = b1_v4.x();
            b1_pnt.y = b1_v4.y();
            b1_pnt.z = b1_v4.z();
            b1_pnt.r = r_color;
            b1_pnt.g = g_color;
            b1_pnt.b = b_color;
            src_corner_ply.push_back(b1_pnt);

            const auto& b2 = ref_edge_point_corr.at(line).src_point_b_;
            Eigen::Vector4f b2_v4(b2.x(), b2.y(), b2.z(), 1.0);
            b2_v4 = T_src * b2_v4;
            b2_pnt.x = b2_v4.x();
            b2_pnt.y = b2_v4.y();
            b2_pnt.z = b2_v4.z();
            b2_pnt.r = r_color;
            b2_pnt.g = g_color;
            b2_pnt.b = b_color;
            src_corner_ply.push_back(b2_pnt);
        }
    }

    WriteBinaryPlyPoints(workspace_path + "/corner_ref.ply", ref_corner_ply, false, true);
    WriteBinaryPlyPoints(workspace_path + "/corner_src.ply", src_corner_ply, false, true);

    std::cout << "valid_lines: " << valid_lines << std::endl;
    // std::unordered_set<int> rebuild_ids;
    // for (const auto& line : line_ids){

    // }

    // LidarPointsCorrespondence();
    

    */
    return 0;
}