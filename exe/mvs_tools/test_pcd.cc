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
#if 1

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
                      Configurator param = Configurator()){
    
    std::shared_ptr<IncrementalMapper> mapper = 
        std::make_shared<IncrementalMapper>(scene_graph_container);
    mapper->BeginReconstruction(reconstruction);

    IndependentMapperOptions mapper_options;
    OptionParser option_parser;
    option_parser.GetMapperOptions(mapper_options,param);

    BundleAdjustmentConfig ba_config;
    auto ba_options = mapper_options.GlobalBundleAdjustment();
    ba_options.refine_focal_length = false;
    ba_options.refine_principal_point = false;
    ba_options.refine_extra_params = false;
    ba_options.refine_local_extrinsics = false;
    ba_options.refine_extrinsics = true;
    
    std::unordered_set<image_t> const_image_ids;
    const auto & correspondence_graph = scene_graph_container->CorrespondenceGraph();
    for (auto image_id : reconstruction->RegisterImageIds()) {
        const auto & image_neighbors = correspondence_graph->ImageNeighbor(image_id);

        if (!reconstruction->ExistsLidarSweep(image_id) || !reconstruction->IsImageRegistered(image_id)){
            continue;
        }
        ba_config.AddImage(image_id);
        ba_config.AddSweep(image_id);
        const Image& image = reconstruction->Image(image_id);
        ba_config.SetConstantCamera(image.CameraId());
        if (const_image_ids.find(image_id) != const_image_ids.end() || ba_config.NumImages() == 1){
            ba_config.SetConstantPose(image_id);
        }
    }
    std::cout << "ba_config: num_images, num_const_images, num_camrea:" 
            << ba_config.NumImages() << ", " << ba_config.NumConstantPoses() 
            << ", " << ba_config.NumConstantCameras() << std::endl;

    
    size_t num_retriangulate_observations = mapper->Retriangulate(mapper_options.Triangulation());
    std::cout << "Retriangulate observation: " << num_retriangulate_observations << std::endl;
    for (int i = 0; i < mapper_options.ba_global_max_refinements; ++i) {
        reconstruction->FilterObservationsWithNegativeDepth();

        const size_t num_observations = reconstruction->ComputeNumObservations();

        PrintHeading1("GBA Bundle adjustment");
        std::cout << "iter: " << i << std::endl;
        if (mapper_options.has_gps_prior){
            reconstruction->AlignWithPriorLocations();
        }

        BundleAdjuster bundle_adjuster(ba_options, ba_config);
        CHECK(bundle_adjuster.Solve(reconstruction.get()));

        size_t num_changed_observations = 0;
        num_changed_observations += CompleteAndMergeTracks(mapper_options, mapper);
        num_changed_observations += FilterPoints(mapper_options, mapper, mapper_options.min_track_length);
        const double changed = static_cast<double>(num_changed_observations) / num_observations;
        std::cout << StringPrintf("  => Changed observations: %.6f", changed) << std::endl;

        num_retriangulate_observations = 0;
        num_retriangulate_observations = mapper->Retriangulate(mapper_options.Triangulation());
        std::cout << "\nnum_retri_observations / num_ori_observations: "
                    << num_retriangulate_observations << " / "
                    << num_observations << std::endl;

        if (changed < mapper_options.ba_global_max_refinement_change) {
            break;
        }
    }
}

int main(int argc, char** argv) {

    PrintHeading("SenseMap.  Copyright(c) 2021, SenseTime Group.");
    PrintHeading(std::string("Version: ") + __VERSION__);

    std::string configuration_file_path = std::string(argv[1]);
    Configurator param;
    param.Load(configuration_file_path.c_str());


    std::string workspace_path = param.GetArgument("workspace_path", "");
    CHECK(!workspace_path.empty()) << "workspace path empty";

    std::string lidar_prior_pose_file = param.GetArgument("lidar_prior_pose_file", "");

    std::vector<RigNames> rig_list;
    // ReadRigList(rig_list, GetParentDir(workspace_path) + "/file_list.txt");
    ReadRigList(lidar_prior_pose_file, rig_list);

    auto scene_graph_container = std::make_shared<SceneGraphContainer>();
    auto feature_data_container = std::make_shared<FeatureDataContainer>();

    CHECK(LoadFeaturesAndMatches(*feature_data_container.get(), *scene_graph_container.get(), param))
        << "Load features or matches failed";

    auto reconstruction = std::make_shared<Reconstruction>();
    reconstruction->ReadReconstruction(workspace_path + "/0");
    reconstruction->SetUp(scene_graph_container, 0);
    std::cout << "reconstruction: " << reconstruction->NumImages() << std::endl;
    reconstruction->Rescale(1.406);

    auto image_names = reconstruction->GetImageNames();
    // for (auto image_name : image_names) {
    //     std::cout << image_name.first << std::endl;
    // }

    std::vector<std::pair<image_t, long long> > image_timestamps;
    image_timestamps.reserve(rig_list.size());
    for (auto & lidar_image : rig_list) {
        auto image_name = GetPathBaseName(lidar_image.img);
        auto image_name_t = "cam0/" + image_name;
        // auto image_name_t = image_name;
        lidar_image.img = image_name_t;
        if (image_names.find(image_name_t) != image_names.end()) {
            const auto image_id = image_names.at(image_name_t);
            long long image_timestamp = std::stoll(image_name.c_str());
            image_timestamps.emplace_back(image_id, image_timestamp);
            std::cout << image_name << " " << image_timestamp << std::endl;
        }
    }
    std::vector<std::pair<image_t, long long> > image_timestamps_dec = image_timestamps;

    std::sort(image_timestamps.begin(), image_timestamps.end(), 
        [&](const std::pair<image_t, long long> & a, const std::pair<image_t, long long> & b){
            return a.second < b.second;
        });
    std::sort(image_timestamps_dec.begin(), image_timestamps_dec.end(), 
        [&](const std::pair<image_t, long long> & a, const std::pair<image_t, long long> & b){
            return a.second >= b.second;
        });

    // std::unordered_map<std::string, image_t> image_name_map = reconstruction->GetImageNames();
    // std::vector<RigNames> rig_list_new;
    // for (size_t i = 0; i < rig_list.size(); i++){
    //     if(image_name_map.find(rig_list.at(i).img) != image_name_map.end()){
    //         rig_list_new.push_back(rig_list.at(i));
    //     }
    // }
    // rig_list.clear();
    // rig_list.shrink_to_fit();
    // rig_list_new.shrink_to_fit();
    // // std::sort(rig_list_new.begin(),rig_list_new.end(),[&](const RigNames& a ,const RigNames& b){
	// // 	return (size_t)image_name_map.at(a.img) > (size_t)image_name_map.at(b.img);
	// // });

    Eigen::Matrix3x4d Tr = GetLidarTrans(0);
    // Eigen::Matrix3x4d Tr = Eigen::Matrix3x4d::Identity();
    reconstruction->LoadLidar(/*rig_list_new*/rig_list, GetParentDir(workspace_path));

    const auto max_precision{std::numeric_limits<long double>::digits10 + 1};

    std::unordered_map<sweep_t, Eigen::Matrix4d> sweep_pose_map;
    auto & lidar_sweeps = reconstruction->LidarSweeps();
    for (auto lidar_sweep : lidar_sweeps) {
        if (lidar_sweep.second.HasQvecPrior() && lidar_sweep.second.HasTvecPrior()) {
            Eigen::Matrix4d proj_matrix = Eigen::Matrix4d::Identity();
            auto R = QuaternionToRotationMatrix(lidar_sweep.second.QvecPrior());
            auto tvec = lidar_sweep.second.TvecPrior();
            proj_matrix.block<3, 3>(0, 0) = R;
            proj_matrix.block<3, 1>(0, 3) = -R * tvec;
            sweep_pose_map[lidar_sweep.first] = proj_matrix;
        } else {
            auto lidar_name = GetPathBaseName(lidar_sweep.second.Name());
            long long lidar_timestamp = std::atof(lidar_name.c_str()) * 1e9;
            const std::pair<int, long long> lidar_timestamp_t({lidar_sweep.first, lidar_timestamp});

            auto next_lidar_image = std::lower_bound(image_timestamps.begin(), image_timestamps.end(), lidar_timestamp_t, 
                [&](const std::pair<image_t, long long> & a, const std::pair<image_t, long long> & b) {
                    return a.second < b.second;
                });
            auto prev_lidar_image = std::lower_bound(image_timestamps_dec.begin(), image_timestamps_dec.end(), lidar_timestamp_t, 
                [&](const std::pair<image_t, long long> & a, const std::pair<image_t, long long> & b) {
                    return a.second >= b.second;
                });

            if (!reconstruction->ExistsImage(prev_lidar_image->first) ||
                !reconstruction->ExistsImage(next_lidar_image->first)) {
                continue;
            }

            std::cout << "lidar_name:       " << std::setprecision(max_precision) << lidar_name << " " << lidar_timestamp << std::endl;
            std::cout << "prev_lidar_image: " << std::setprecision(max_precision) << prev_lidar_image->second << " " << prev_lidar_image->first << std::endl;
            std::cout << "next_lidar_image: " << std::setprecision(max_precision) << next_lidar_image->second << " " << next_lidar_image->first << std::endl;

            double t = (lidar_timestamp - prev_lidar_image->second) * 1.0 / (next_lidar_image->second - prev_lidar_image->second);

            auto prev_image = reconstruction->Image(prev_lidar_image->first);
            auto next_image = reconstruction->Image(next_lidar_image->first);

            Eigen::Vector4d qvec;
            Eigen::Vector3d tvec;
            InterpolatePose(prev_image.Qvec(), prev_image.Tvec(), next_image.Qvec(), next_image.Tvec(), t, &qvec, &tvec);

            Eigen::Matrix4d proj_matrix = Eigen::Matrix4d::Identity();
            proj_matrix.block<3, 3>(0, 0) = QuaternionToRotationMatrix(qvec);
            proj_matrix.block<3, 1>(0, 3) = tvec;

            sweep_pose_map[lidar_sweep.first] = proj_matrix;
        }
    }

    // reconstruction->OutputLidarPointCloud2World(workspace_path+"/lidar", sweep_pose_map, true);

    bool downsample_flag = true;
    std::string lidar_path = workspace_path + "/lidar";
    LidarPointCloud lidar_sweeps_cloud, laser_cloud_temp;

    VoxelMap::Option option;
    std::vector<size_t> layer_point_size{60, 30, 15, 10, 10};
    std::vector<size_t> max_layer_point_size{200, 100, 80, 80, 80};
    option.layer_point_size = layer_point_size;
    option.max_layer_point_size = max_layer_point_size;
    option.max_layer = 3;
    option.voxel_size = 5.0;
    option.line_min_max_eigen_ratio = 0.02;
    option.line_min_mid_eigen_ratio = 0.8;
    option.verbose = true;

    option.plane_min_mid_eigen_ratio = 0.03;

    VoxelMap voxel_map(option, 0);

    bool init = false;

    int num_images = 0;
    for (auto lidar_sweep: lidar_sweeps){
        num_images++;
        image_t sweep_id = lidar_sweep.first;
        if (sweep_pose_map.find(sweep_id) == sweep_pose_map.end()) {
            continue;
        }
#if 0
        class Image image = reconstruction->Image(sweep_id);
        Eigen::Matrix3x4d Tr_ref_inv = image.InverseProjectionMatrix();
        Eigen::Matrix4d Tr_ref_inv_4d = Eigen::Matrix4d::Identity();
        Tr_ref_inv_4d.topRows(3) = Tr_ref_inv;
#else
        const Eigen::Matrix4d Tr_ref_inv_4d = sweep_pose_map.at(sweep_id).inverse();
#endif

        class LidarSweep sweep_ref = lidar_sweeps.at(sweep_id);

        Eigen::Matrix3x4d Tr_lidar2camera = Tr;
        Eigen::Matrix4d Tr_lidar2camera_4d = Eigen::Matrix4d::Identity();
        Tr_lidar2camera_4d.topRows(3) = Tr_lidar2camera;
        Eigen::Matrix4d T = Tr_lidar2camera_4d.inverse() * Tr_ref_inv_4d;

        LidarPointCloud ref_less_features, ref_less_features_t; 
        LidarPointCloud ref_less_surfs = sweep_ref.GetSurfPointsLessFlat();
        LidarPointCloud ref_less_corners = sweep_ref.GetCornerPointsLessSharp();
        ref_less_features = ref_less_surfs;
        ref_less_features += ref_less_corners;
        LidarPointCloud::TransfromPlyPointCloud (ref_less_features, ref_less_features_t, T);

        lidar_sweeps_cloud += ref_less_features_t;

        std::vector<lidar::OctoTree::Point> points;
        points.reserve(ref_less_features_t.points.size());
        for (auto point : ref_less_features_t.points) {
            lidar::OctoTree::Point X;
            X.x = point.x;
            X.y = point.y;
            X.z = point.z;
            X.type = 0;
            points.push_back(X);
        }

        // if (!init) {
        //     voxel_map.BuildVoxelMap(points);
        //     init = true;
        // } else {
            voxel_map.AppendToVoxelMap(points, num_images);
        // }

        if (downsample_flag){
            // PlyPointCloud laserCloudTemp;
            // laser_cloud_temp += ref_less_features_t;
            // GridSimplifyPlyPointCloud(lidar_sweeps_cloud, laserCloudTemp, 0.2);
            // laser_cloud_temp = laserCloudTemp;
        // } else {

            std::string ref_les_name = lidar_path + "/frame/" + sweep_ref.Name() + ".ply";
            std::string parent_path = GetParentDir(ref_les_name);
            if (!ExistsPath(parent_path)) {
                boost::filesystem::create_directories(parent_path);
            }
            WriteBinaryPlyPoints(ref_les_name.c_str(), ref_less_features_t.Convert2Ply(),
                                false, true);
        }
    }

    // // TODO downsample
    // PlyPointCloud laserCloudStack = lidar_sweeps_cloud;
    std::string cloud_path = lidar_path + "/lidar_cloud.ply";
    if (!ExistsPath(GetParentDir(cloud_path))) {
        boost::filesystem::create_directories(GetParentDir(cloud_path));
    }
    WriteBinaryPlyPoints(cloud_path.c_str(), lidar_sweeps_cloud.Convert2Ply(), false, true);

    std::cout << "Abstract all feature voxels" << std::endl;

    std::srand(0);

    std::vector<Eigen::Vector3d> plane_points;
    std::vector<Eigen::Vector3i> plane_colors;
    std::vector<Eigen::Vector3d> line_points;
    std::vector<Eigen::Vector3i> line_colors;

    std::vector<lidar::OctoTree*> octree_list = voxel_map.AbstractFeatureVoxels();
    for (auto octree : octree_list) {
        int r = (0.5 + 0.5 * std::rand() / RAND_MAX) * 255;
        int g = (0.5 + 0.5 * std::rand() / RAND_MAX) * 255;
        int b = (0.5 + 0.5 * std::rand() / RAND_MAX) * 255;
        std::vector<Eigen::Vector3d> points;
        octree->GetGridPoints(points);
        if (octree->voxel_->FeatureType() == Voxel::FeatureType::PLANE) {
            for (size_t i = 0; i < points.size(); ++i) {
                double w = 1.0;
                plane_colors.push_back(Eigen::Vector3i(r * w, g * w, b * w));
            }
            plane_points.insert(plane_points.end(), points.begin(), points.end());
        } else if (octree->voxel_->FeatureType() == Voxel::FeatureType::LINE) {
            line_points.insert(line_points.end(), points.begin(), points.end());
            line_colors.insert(line_colors.end(), points.size(), Eigen::Vector3i(r, g, b));
        }
    }

    std::ofstream file(lidar_path + "/planes.obj", std::ios::out);
    for (size_t i = 0; i < plane_points.size(); ++i) {
        auto & p = plane_points[i];
        auto & c = plane_colors[i];
        file << "v " << p[0] << " " << p[1] << " " << p[2] << " " << c[0] << " " << c[1] << " " << c[2] << std::endl;
    }
    file.close();
    
    std::ofstream file2(lidar_path + "/lines.obj", std::ios::out);
    for (size_t i = 0; i < line_points.size(); ++i) {
        auto & p = line_points[i];
        auto & c = line_colors[i];
        file2 << "v " << p[0] << " " << p[1] << " " << p[2] << " " << c[0] << " " << c[1] << " " << c[2] << std::endl;
    }
    file2.close();

    // if(ExistsDir(workspace_path+"/lidar_before")){
    //     boost::filesystem::remove_all(workspace_path+"/lidar_before");
    //     std::cout << "output path no empty, clear " << workspace_path+"/lidar_before" << std::endl;
    // }
    // reconstruction->OutputLidarPointCloud2World(workspace_path+"/lidar_before", std::unordered_map<sweep_t, Eigen::Matrix4d>(), true);

    // ReconstructionBA(scene_graph_container, reconstruction, param);

    // if(ExistsDir(workspace_path+"/lidar")){
    //     boost::filesystem::remove_all(workspace_path+"/lidar");
    //     std::cout << "output path no empty, clear " << workspace_path+"/lidar" << std::endl;
    // }
    // reconstruction->OutputLidarPointCloud2World(workspace_path+"/lidar", std::unordered_map<sweep_t, Eigen::Matrix4d>(), true);
    return 0;
}
#else
int main(int argc, char** argv) {

	PrintHeading("SenseMap.  Copyright(c) 2021, SenseTime Group.");	
	PrintHeading(std::string("Version: ") + __VERSION__);
    
    if (argc < 2) {
        std::cout << "Please enter ./test_ply2pnts input.pcd" << std::endl;
		return StateCode::NO_MATCHING_INPUT_PARAM;
    }

    const std::string in_pcd_path = std::string(argv[1]);

    auto pc = ReadPCD(in_pcd_path);
    std::vector<sensemap::PlyPoint> ply_points = Convert2Ply(pc);
    WriteTextPlyPoints(in_pcd_path+".ply", ply_points, false, true);

    LidarSweep lidar_pc;
    lidar_pc.Setup(pc, 0);

    lidar_pc.OutputCornerPly(GetParentDir(in_pcd_path));

    return 0;
}
#endif