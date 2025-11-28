// Copyright (c) 2019, SenseTime Group.
// All rights reserved.

#include <dirent.h>
#include <sys/stat.h>
#include <malloc.h>

#include <boost/filesystem/path.hpp>
#include <unordered_set>

#include "../Configurator_yaml.h"
#include "../camera_rig_params.h"
#include "../option_parsing.h"
#include "../system_io.h"
#include "base/common.h"
#include "base/reconstruction_manager.h"
#include "container/feature_data_container.h"
#include "controllers/incremental_mapper_controller.h"
#include "controllers/directed_mapper_controller.h"
#include "feature/extraction.h"
#include "feature/feature_matcher.h"
#include "feature/utils.h"
#include "graph/correspondence_graph.h"
#include "graph/maximum_spanning_tree_graph.h"
#include "util/gps_reader.h"
#include "util/mat.h"
#include "util/misc.h"
#include "util/ply.h"
#include "util/rgbd_helper.h"
#include "base/version.h"
#include "util/tag_scale_recover.h"
#include "base/common.h"
#include "util/proc.h"

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/convex_hull_2.h>
#include <CGAL/Convex_hull_traits_adapter_2.h>
#include <CGAL/property_map.h>
#include <vector>
#include <numeric>
typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef K::Point_2 Point_2;
typedef CGAL::Convex_hull_traits_adapter_2<K,
          CGAL::Pointer_property_map<Point_2>::type > Convex_hull_traits_2;


#define MAX_PRECISION std::setprecision(std::numeric_limits<double>::digits10 + 1)

using namespace sensemap;
FILE *fs;

bool must_increamental;

void PrintReconSummary(const std::string &flog_name, const size_t num_total_image,
                       const std::shared_ptr<ReconstructionManager> &reconstruction_manager) {
    if (reconstruction_manager->Size() == 0) {
        return;
    }
    std::shared_ptr<Reconstruction> best_rec;
    for (int i = 0; i < reconstruction_manager->Size(); ++i) {
        const std::shared_ptr<Reconstruction> &rec = reconstruction_manager->Get(i);
        if (!best_rec || best_rec->NumRegisterImages() < rec->NumRegisterImages()) {
            best_rec = rec;
        }
    }
    FILE *fp = fopen(flog_name.c_str(), "w");

    size_t num_reg_image = best_rec->NumRegisterImages();
    fprintf(fp, "Registered / Total: %zu / %zu\n", num_reg_image, num_total_image);
    fprintf(fp, "Mean Track Length: %f\n", best_rec->ComputeMeanTrackLength());
    fprintf(fp, "Mean Reprojection Error: %f\n", best_rec->ComputeMeanReprojectionError());
    fprintf(fp, "Mean Observation Per Register Image: %f\n", best_rec->ComputeMeanObservationsPerRegImage());

    fclose(fp);
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

    auto tracks = scene_graph.CorrespondenceGraph()->GenerateTracks(5, false, true);

    GetAvailableMemory(availabel_memeory);
    std::cout << "Available Memory[FinalizeSceneGraph]: " << availabel_memeory << "GB" << std::endl;

    return true;
}

void ReadSlamPoses(const std::string path, std::map<std::string,std::vector<double>>& slam_poses){
    std::cout<<"Read "<<path<<std::endl;
    std::ifstream infile(path);
    CHECK(infile.is_open());

    std::string line;
    std::string item;

    while (std::getline(infile, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }
        StringTrim(&line);

        std::stringstream line_stream(line);

        std::getline(line_stream, item, ' ');
        std::string img_name = item;

        std::vector<double> pose;

        std::getline(line_stream, item, ' ');
        // std::cout<<"tx "<<item<<std::endl;
        double tx = std::stold(item);
        // std::cout<<std::setprecision(17)<<"tx "<<tx<<std::endl;
        pose.push_back(tx);
        std::getline(line_stream, item, ' ');
        double ty = std::stold(item);
        // std::cout<<"ty "<<ty<<std::endl;
        pose.push_back(ty);
        std::getline(line_stream, item, ' ');
        double tz = std::stold(item);
        // std::cout<<"tz "<<tz<<std::endl;
        pose.push_back(tz);


        std::getline(line_stream, item, ' ');
        double rw = std::stold(item);
        // std::cout<<"rw "<<rw<<std::endl;
        pose.push_back(rw);
        std::getline(line_stream, item, ' ');
        double rx = std::stold(item);
        // std::cout<<"rx "<<rx<<std::endl;
        pose.push_back(rx);
        std::getline(line_stream, item, ' ');
        double ry = std::stold(item);
        // std::cout<<"ry "<<ry<<std::endl;
        pose.push_back(ry);
        std::getline(line_stream, item, ' ');
        double rz = std::stold(item);
        // std::cout<<"rz "<<rz<<std::endl;
        pose.push_back(rz);

        slam_poses[img_name] = pose;
    }
    infile.close();
}

void ConstructSLAMReconstruction(const std::shared_ptr<sensemap::SceneGraphContainer> &scene_graph_container,
                                  std::map<std::string,std::vector<double>> &slam_poses,std::shared_ptr<Reconstruction> &slam_reconstruction,
                                  std::vector<image_t> &image_ids, const std::string workspace_path){
    // build slam cluster scene_graph
    std::set<image_t> cur_image_ids;
    std::map<std::string, image_t> slam_img_name_map;
    std::string video_name;
    for(const auto image_id : image_ids){
        // std::cout<<image_id<<"/"<<image_ids.size()<<std::flush;
        if(!scene_graph_container->ExistsImage(image_id)){
            std::cout<<image_id<<" not exist in scene_graph_container"<<std::endl;
            continue;
        }
        const Image &image = scene_graph_container->Image(image_id);
        
        std::string name = image.Name();
        // std::cout << "name = " << name  <<" "<<name.substr(name.find('/'))<< std::endl;
        if(slam_poses.count(name)){
            cur_image_ids.insert(image_id);
            slam_img_name_map[name] = image_id;
            video_name = name.substr(0, name.find('/'));
        }
    }
    std::cout << "SLAM Cluster Image Number = " << cur_image_ids.size() << std::endl;

    std::shared_ptr<SceneGraphContainer> cur_cluster_graph_container =
        std::shared_ptr<SceneGraphContainer>(new SceneGraphContainer());
    auto slam_scene_graph_container(scene_graph_container);
    slam_scene_graph_container->ClusterSceneGraphContainer(cur_image_ids, *cur_cluster_graph_container.get());
            

    std::shared_ptr<IncrementalMapper> mapper = std::make_shared<IncrementalMapper>(cur_cluster_graph_container);

    // simply triangulate slam mappoint
    mapper->BeginReconstruction(slam_reconstruction);
    std::unordered_map<image_t, Eigen::Vector4d> prior_rotations;
    std::unordered_map<image_t, Eigen::Vector3d> prior_translations;
    

    // auto angle_axis2=EulerAnglesToRotationMatrix(M_PI,0, 0);
    auto angle_axis2=EulerAnglesToRotationMatrix(0,0, 0);
    // std::cout<<angle_axis2<<std::endl;
    for(const auto slam_img_name_set : slam_img_name_map){
        Eigen::Vector3d t;
        t << slam_poses[slam_img_name_set.first][0],
            slam_poses[slam_img_name_set.first][1],
            slam_poses[slam_img_name_set.first][2];
        Eigen::Vector4d q;
        q << slam_poses[slam_img_name_set.first][3],
            slam_poses[slam_img_name_set.first][4],
            slam_poses[slam_img_name_set.first][5],
            slam_poses[slam_img_name_set.first][6];
        const Eigen::Quaterniond q_quat(q(0), q(1), q(2), q(3));

        auto q_cw=InvertQuaternion(q);
        const Eigen::Quaterniond quat(q_cw(0), q_cw(1),
                                q_cw(2), q_cw(3));
        Eigen::Vector3d t_cw = quat * -t;
        Eigen::Matrix3x4d proj_matrix;
        proj_matrix.leftCols<3>() = quat.toRotationMatrix();
        proj_matrix.rightCols<1>() = t_cw;
        // std::cout<<"prior projection matrix "<<std::endl<<ProjectionCenterFromMatrix(proj_matrix)<<std::endl;
        auto self_proj_matrix = angle_axis2 * proj_matrix;
        // std::cout<<"after projection matrix "<<std::endl<<ProjectionCenterFromMatrix(self_proj_matrix)<<std::endl;

        Eigen::Matrix3d q_tmp = (self_proj_matrix.leftCols<3>());
        Eigen::Vector3d t_tmp = self_proj_matrix.rightCols<1>();
        Eigen::Vector3d t_final;
        t_final<< t_tmp[0],t_tmp[1],t_tmp[2];

        
        slam_reconstruction->Image(slam_img_name_set.second).Tvec() = t_final ;
        slam_reconstruction->Image(slam_img_name_set.second).Qvec() = RotationMatrixToQuaternion(q_tmp);
        slam_reconstruction->Image(slam_img_name_set.second).SetPoseFlag(true);

        prior_rotations.emplace(slam_img_name_set.second,slam_reconstruction->Image(slam_img_name_set.second).Qvec());
        prior_translations.emplace(slam_img_name_set.second,slam_reconstruction->Image(slam_img_name_set.second).Tvec());
        slam_reconstruction->RegisterImage(slam_img_name_set.second);
    }
            

    //////////////////////////////////////////////////////////////////////////////
    // Triangulation
    //////////////////////////////////////////////////////////////////////////////
    IndependentMapperOptions mapper_options;
    mapper_options.have_prior_pose = true;
    mapper_options.prior_rotations = prior_rotations;
    mapper_options.prior_translations = prior_translations;

    auto tri_options = mapper_options.Triangulation();
    tri_options.min_angle = 4.0;

    const auto& reg_image_ids = slam_reconstruction->RegisterImageIds();
    // std::cout<<"reg_image_ids.size() "<<reg_image_ids.size()<<std::endl;

    for (size_t i = 0; i < reg_image_ids.size(); ++i) {
        const image_t image_id = reg_image_ids[i];

        // Check the image is in the scene graph or not
        if (!slam_scene_graph_container->ExistsImage(image_id)) {
            continue;
        }

        const auto& image = slam_reconstruction->Image(image_id);

        PrintHeading1(StringPrintf("Triangulating image #%d (%d)", image_id, i));

        const size_t num_existing_points3D = image.NumMapPoints();

        std::cout << "  => Image sees " << num_existing_points3D << " / " << image.NumObservations() << " points"
                << std::endl;

        mapper->TriangulateImage(tri_options, image_id);

        std::cout << "  => Triangulated " << (image.NumMapPoints() - num_existing_points3D) << " points" << std::endl;
    }

    //////////////////////////////////////////////////////////////////////////////
    // Retriangulation
    //////////////////////////////////////////////////////////////////////////////

    PrintHeading1("Retriangulation");

    const size_t num_completed_observations = mapper->CompleteTracks(mapper_options.Triangulation());
    std::cout << "  => Completed observations: " << num_completed_observations << std::endl;
    const size_t num_merged_observations = mapper->MergeTracks(mapper_options.Triangulation());
    std::cout << "  => Merged observations: " << num_merged_observations << std::endl;

    std::string slam_reconstruction_path1 = workspace_path+"/map_update/slam"+video_name;
    CreateDirIfNotExists(slam_reconstruction_path1);
    slam_reconstruction->WriteReconstruction(slam_reconstruction_path1, true);

    slam_reconstruction->prior_rotations = prior_rotations;
    slam_reconstruction->prior_translations = prior_translations;
    
    //////////////////////////////////////////////////////////////////////////////
    // Bundle adjustment
    //////////////////////////////////////////////////////////////////////////////
    auto ba_options = mapper_options.GlobalBundleAdjustment();
    ba_options.refine_focal_length = true;
    ba_options.refine_principal_point = true;
    ba_options.refine_extra_params = true;
    ba_options.refine_extrinsics = true;
    ba_options.refine_local_extrinsics = false;
    ba_options.use_prior_relative_pose = true;

    // Configure bundle adjustment.
    BundleAdjustmentConfig ba_config;
    
    for (size_t i = 0; i < reg_image_ids.size(); ++i) {
        const image_t image_id = reg_image_ids[i];
        if (!slam_scene_graph_container->ExistsImage(image_id)) {
            continue;
        }
        ba_config.AddImage(image_id);
        // ba_config.SetConstantPose(image_id);
    }
    
    for (int i = 0; i < 3; ++i) {
        // Avoid degeneracies in bundle adjustment.
        slam_reconstruction->FilterObservationsWithNegativeDepth();

        const size_t num_observations = slam_reconstruction->ComputeNumObservations();

        auto print_cameras = slam_reconstruction->Cameras();
        std::cout << "Camera number = " << print_cameras.size() << std::endl;

        for (auto camera : print_cameras) {
            std::cout << "  Camera index = " << camera.first << std::endl;
            std::cout << "  Camera model = " << camera.second.ModelName() << std::endl;
            std::cout << "  Camera param = ";
            for (auto param : camera.second.Params()) {
                std::cout << "  " << param;
            }
            std::cout << std::endl;

            std::vector<Eigen::Vector4d> local_qvecs;
            std::vector<Eigen::Vector3d> local_tvecs;
            std::vector<double *> local_camera_params_data;

            if(camera.second.NumLocalCameras() > 1){
                int local_param_size = camera.second.LocalParams().size() / camera.second.NumLocalCameras();
                Eigen::Vector4d local_qvec;
                Eigen::Vector3d local_tvec;
                for(size_t i = 0; i < camera.second.NumLocalCameras(); ++i){
                    camera.second.GetLocalCameraExtrinsic(i,local_qvec,local_tvec);   
                    std::cout << " Local Camera " << i << " Tvec " << local_tvec[0] << " " << local_tvec[1] << " " << local_tvec[2] << std::endl;
                    std::cout << " Local Camera " << i << " Qvec " << local_qvec[0] << " " << local_qvec[1] << " " << local_qvec[2] << " " << local_qvec[3] << std::endl;
                    double * param_data = camera.second.LocalIntrinsicParamsData() + i * local_param_size;
                    for (int i = 0; i < camera.second.Params().size(); i++) {
                        std::cout << " " << *(param_data+i) << std::endl;
                    }     
                }
            }
        }


        PrintHeading1("Bundle adjustment");
        BundleAdjuster bundle_adjuster(ba_options, ba_config);
        CHECK(bundle_adjuster.Solve(slam_reconstruction.get()));

        size_t num_changed_observations = 0;
        const size_t num_completed_observations = mapper->CompleteTracks(mapper_options.Triangulation());
        std::cout << "  => Completed observations: " << num_completed_observations << std::endl;
        num_changed_observations += num_completed_observations;
        const size_t num_merged_observations = mapper->MergeTracks(mapper_options.Triangulation());
        std::cout << "  => Merged observations: " << num_merged_observations << std::endl;
        num_changed_observations += num_merged_observations;
        const size_t num_filtered_observations = mapper->FilterPoints(mapper_options.IncrementalMapperOptions());
        std::cout << "  => Filtered observations: " << num_filtered_observations << std::endl;
        num_changed_observations += num_filtered_observations;
        const double changed = static_cast<double>(num_changed_observations) / num_observations;
        std::cout << StringPrintf("  => Changed observations: %.6f", changed) << std::endl;

        print_cameras = slam_reconstruction->Cameras();
        std::cout << "Camera number = " << print_cameras.size() << std::endl;

        for (auto camera : print_cameras) {
            std::cout << "  Camera index = " << camera.first << std::endl;
            std::cout << "  Camera model = " << camera.second.ModelName() << std::endl;
            std::cout << "  Camera param = ";
            for (auto param : camera.second.Params()) {
                std::cout << "  " << param;
            }
            std::cout << std::endl;

            std::vector<Eigen::Vector4d> local_qvecs;
            std::vector<Eigen::Vector3d> local_tvecs;
            std::vector<double *> local_camera_params_data;

            if(camera.second.NumLocalCameras() > 1){
                int local_param_size = camera.second.LocalParams().size() / camera.second.NumLocalCameras();
                Eigen::Vector4d local_qvec;
                Eigen::Vector3d local_tvec;
                for(size_t i = 0; i < camera.second.NumLocalCameras(); ++i){
                    camera.second.GetLocalCameraExtrinsic(i,local_qvec,local_tvec);   
                    std::cout << " Local Camera " << i << " Tvec " << local_tvec[0] << " " << local_tvec[1] << " " << local_tvec[2] << std::endl;
                    std::cout << " Local Camera " << i << " Qvec " << local_qvec[0] << " " << local_qvec[1] << " " << local_qvec[2] << " " << local_qvec[3] << std::endl;
                    double * param_data = camera.second.LocalIntrinsicParamsData() + i * local_param_size;
                    for (int i = 0; i < camera.second.Params().size(); i++) {
                        std::cout << " " << *(param_data+i) << std::endl;
                    }     
                }
            }
        }

        if (changed < mapper_options.ba_global_max_refinement_change) {
            break;
        }
        std::string slam_reconstruction_path = workspace_path+"/map_update/slam"+video_name+"/after"+std::to_string(i);
        CreateDirIfNotExists(slam_reconstruction_path);
        slam_reconstruction->WriteReconstruction(slam_reconstruction_path, true);
    }


    const bool kDiscardReconstruction = false;
    mapper->EndReconstruction(kDiscardReconstruction);

    std::string slam_reconstruction_path = workspace_path+"/map_update/slam"+video_name+"/after";
    CreateDirIfNotExists(slam_reconstruction_path);
    slam_reconstruction->WriteReconstruction(slam_reconstruction_path, true);
        
    std::cout<<"total slam reconsturction images "<<slam_reconstruction->RegisterImageIds().size()<<std::endl;
}


void IncrementalSFM(const std::shared_ptr<sensemap::SceneGraphContainer> &scene_graph_container,
                    std::shared_ptr<sensemap::ReconstructionManager> &reconstruction_manager, Configurator &param) {
    using namespace sensemap;

    PrintHeading1("Incremental Mapping");

    auto options = std::make_shared<MapperOptions>();
    options->mapper_type = MapperType::INDEPENDENT;
    options->independent_mapper_options.outside_mapper_type = MapperType::INDEPENDENT;
    options->independent_mapper_options.independent_mapper_type = IndependentMapperType::INCREMENTAL;

    OptionParser option_parser;
    ImageReaderOptions reader_options;
    option_parser.GetImageReaderOptions(reader_options, param);
    option_parser.GetMapperOptions(options->independent_mapper_options, param);

    std::string workspace_path = param.GetArgument("workspace_path", "");
    CHECK(!workspace_path.empty()) << "workspace path empty";

    size_t base_reconstruction_idx = reconstruction_manager->Size();
    std::vector<int> base_reconstruction_ids = reconstruction_manager->getReconstructionIds();

    if (static_cast<bool>(param.GetArgument("map_update", 0))){
        options->independent_mapper_options.map_update = true;
        auto reconstruction_idx = reconstruction_manager->Add();
        std::shared_ptr<Reconstruction> reconstruction = reconstruction_manager->Get(reconstruction_idx);
        auto cameras = scene_graph_container->Cameras();
        bool camera_rig = false;
        for(auto camera : cameras){
            if(camera.second.NumLocalCameras()>1){
                camera_rig = true;
            }
        }
        reconstruction->ReadReconstruction(workspace_path + "/0",camera_rig);

        std::cout << "images in old map: " << reconstruction->NumRegisterImages() << std::endl;
        
        // set original image to label 0
        auto old_image_ids = reconstruction->RegisterImageIds();
        for (auto old_image_id : old_image_ids) {

            class Image & old_image = reconstruction->Image(old_image_id);
            old_image.SetLocalImageIndices(scene_graph_container->Image(old_image_id).LocalImageIndices());
            old_image.SetLabelId(0);
            old_image.SetPoseFlag(true);
        }
        workspace_path = JoinPaths(workspace_path, "/map_update");
    }

    std::string image_path = param.GetArgument("image_path", "");
    CHECK(!workspace_path.empty()) << "image path empty";

    // use prior pose from slam to constrain SfM
    bool use_slam_graph = static_cast<bool>(param.GetArgument("use_slam_graph", 0));
    std::string preview_pose_file = param.GetArgument("preview_pose_file", "");
    if (use_slam_graph && (!preview_pose_file.empty()) && boost::filesystem::exists(preview_pose_file)) {
        std::vector<Keyframe> keyframes;
        if (boost::filesystem::path(preview_pose_file).extension().string() == ".tum") {
            LoadPriorPoseFromTum(preview_pose_file, keyframes, (param.GetArgument("image_type", "").compare("rgbd") == 0));
        }
        else {
            LoadPirorPose(preview_pose_file, keyframes);
        }

        std::unordered_map<std::string, Keyframe> keyframe_map;
        for (auto const keyframe : keyframes) {
            keyframe_map.emplace(keyframe.name, keyframe);
        }

        std::unordered_map<image_t, Eigen::Vector4d> prior_rotations;
        std::unordered_map<image_t, Eigen::Vector3d> prior_translations;

        // std::string camera_rig_params_file = param.GetArgument("rig_params_file", "");
        // CameraRigParams rig_params;
        // bool camera_rig = false;
        // Eigen::Matrix3d R0;
        // Eigen::Vector3d t0;

        // if (!camera_rig_params_file.empty()) {
        //     if (rig_params.LoadParams(camera_rig_params_file)) {
        //         camera_rig = true;
        //         R0 = rig_params.local_extrinsics[0].block<3, 3>(0, 0);
        //         t0 = rig_params.local_extrinsics[0].block<3, 1>(0, 3);

        //     } else {
        //         std::cout << "failed to read rig params" << std::endl;
        //         exit(-1);
        //     }
        // }

        std::vector<image_t> image_ids = scene_graph_container->GetImageIds();
        for (const auto image_id : image_ids) {
            const Image &image = scene_graph_container->Image(image_id);
            const Camera& camera = scene_graph_container->Camera(image.CameraId());
            bool camera_rig = false;
            Eigen::Matrix3d R0;
            Eigen::Vector3d t0;
            if(camera.NumLocalCameras()>1){
                camera_rig = true;
                Eigen::Vector4d qvec;
                Eigen::Vector3d tvec;
                camera.GetLocalCameraExtrinsic(0,qvec,tvec);
                R0 = QuaternionToRotationMatrix(qvec);
                t0 = tvec;
            }

            std::string name = image.Name();
            if (keyframe_map.find(name) != keyframe_map.end()) {
                Keyframe keyframe = keyframe_map.at(name);

                Eigen::Matrix3d r = keyframe.rot;
                Eigen::Vector4d q = RotationMatrixToQuaternion(r);
                Eigen::Vector3d t = keyframe.pos;

                if (camera_rig) {
                    Eigen::Matrix3d R_rig = R0.transpose() * r;
                    Eigen::Vector3d t_rig = R0.transpose() * (t - t0);
                    q = RotationMatrixToQuaternion(R_rig);
                    t = t_rig;
                }
                prior_rotations.emplace(image_id, q);
                prior_translations.emplace(image_id, t);
            }
        }

        options->independent_mapper_options.prior_rotations = prior_rotations;
        options->independent_mapper_options.prior_translations = prior_translations;
        options->independent_mapper_options.have_prior_pose = true;
    }

    // use gps location prior to constrain the image
    bool use_gps_prior = static_cast<bool>(param.GetArgument("use_gps_prior", 0));
    std::string gps_prior_file = param.GetArgument("gps_prior_file", "");
    std::string gps_trans_file = workspace_path + "/gps_trans.txt";
    if (use_gps_prior) {
        if (boost::filesystem::exists(gps_prior_file)) {
            auto image_ids = scene_graph_container->GetImageIds();
            std::vector<std::string> image_names;
            for (const auto image_id : image_ids) {
                const Image &image = scene_graph_container->Image(image_id);
                std::string name = image.Name();
                image_names.push_back(name);
            }

            std::vector<std::pair<unsigned long, std::pair<Eigen::Vector3d,int>>> gps_locations;
            if (options->independent_mapper_options.optimization_use_horizontal_gps_only) {
                LoadOriginGPSinfo(gps_prior_file, gps_locations, gps_trans_file, true);
            } else {
                LoadOriginGPSinfo(gps_prior_file, gps_locations, gps_trans_file, false);
            }
            std::unordered_map<std::string, std::pair<Eigen::Vector3d,int>> image_locations;
            GPSLocationsToImages(gps_locations, image_names, image_locations);
            std::unordered_map<image_t, std::pair<Eigen::Vector3d,int>> prior_locations_gps;
            std::cout << image_locations.size() << " images have gps prior" << std::endl;
            
            bool set_gps_info = true;
            if(image_locations.size() == 0){
                std::cout<<" WARNING!!!!! There has no gps related images!"<<std::endl;
                std::cout<<" WARNING!!!!! Set use_gps_prior to 0!"<<std::endl;
                use_gps_prior = false;
                set_gps_info = false;
            }

            if(set_gps_info){
                std::vector<PlyPoint> gps_locations_ply;
                for (const auto image_id : image_ids) {
                    const Image &image = scene_graph_container->Image(image_id);
                    std::string name = image.Name();

                    if (image_locations.find(name) != image_locations.end()) {
                        prior_locations_gps.emplace(image_id, image_locations.at(name));

                        PlyPoint gps_location_ply;
                        gps_location_ply.r = 255;
                        gps_location_ply.g = 0;
                        gps_location_ply.b = 0;
                        gps_location_ply.x = image_locations.at(name).first[0];
                        gps_location_ply.y = image_locations.at(name).first[1];
                        gps_location_ply.z = image_locations.at(name).first[2];
                        gps_locations_ply.push_back(gps_location_ply);
                    }
                }
                options->independent_mapper_options.prior_locations_gps = prior_locations_gps;
                options->independent_mapper_options.original_gps_locations = gps_locations;
                sensemap::WriteBinaryPlyPoints(workspace_path + "/gps.ply", gps_locations_ply, false, true);
                options->independent_mapper_options.has_gps_prior = true;
            }
        }

    }

    // rgbd mode
    int num_local_cameras = reader_options.num_local_cameras;
    bool with_depth = options->independent_mapper_options.with_depth;

    MapperController *mapper =
        MapperController::Create(options, workspace_path, image_path, scene_graph_container, reconstruction_manager);
    mapper->Start();
    mapper->Wait();

    fprintf(
        fs, "%s\n",
        StringPrintf("Incremental Mapper Elapsed time: %.3f [minutes]", mapper->GetTimer().ElapsedMinutes()).c_str());
    fflush(fs);

    bool use_apriltag = static_cast<bool>(param.GetArgument("detect_apriltag", 0));
    double apriltag_size = param.GetArgument("apriltag_size", 0.113f);
    bool color_harmonization = param.GetArgument("color_harmonization", 0);

    std::vector<int> reconstruction_ids = reconstruction_manager->getReconstructionIds();
    std::vector<int> remain_reconstruction_ids;
    std::set_difference(reconstruction_ids.begin(), reconstruction_ids.end(), 
                        base_reconstruction_ids.begin(), base_reconstruction_ids.end(), std::back_inserter(remain_reconstruction_ids));

    for (size_t i = 0; i < remain_reconstruction_ids.size(); ++i) {
        auto reconstruction = reconstruction_manager->Get(remain_reconstruction_ids[i]);
        CHECK(reconstruction->RegisterImageIds().size() > 0);
        const image_t first_image_id = reconstruction->RegisterImageIds()[0];
        CHECK(reconstruction->ExistsImage(first_image_id));
        const Image &image = reconstruction->Image(first_image_id);
        const Camera &camera = reconstruction->Camera(image.CameraId());

        std::string rec_path = StringPrintf("%s/%d", workspace_path.c_str(), base_reconstruction_idx + i);
        if (!boost::filesystem::exists(rec_path)) {
            boost::filesystem::create_directories(rec_path);
        }

        if (color_harmonization) {
            reconstruction->ColorHarmonization(image_path);
        }

        reconstruction->ExportMapPoints(JoinPaths(rec_path, MAPPOINT_NAME));

        if (options->independent_mapper_options.debug_info && use_gps_prior && !static_cast<bool>(param.GetArgument("map_update", 0))) {
            // reconstruction->AlignWithPriorLocations(options->independent_mapper_options.max_error_gps);
            if(camera.NumLocalCameras() <= 1){
                Reconstruction rec = *reconstruction.get();
                rec.AddPriorToResult();
                // rec.NormalizeWoScale();
                
                std::string trans_rec_path = rec_path + "-gps";
                if (!boost::filesystem::exists(trans_rec_path)) {
                    boost::filesystem::create_directories(trans_rec_path);
                }
                // rec.WriteBinary(trans_rec_path);
                std::shared_ptr<Reconstruction> filtered_reconstruction = std::make_shared<Reconstruction>();
                rec.FilterUselessPoint2D(filtered_reconstruction);

                filtered_reconstruction->WriteBinary(trans_rec_path);
            } else {
                Reconstruction rig_reconstruction;
                reconstruction->ConvertRigReconstruction(rig_reconstruction);

                rig_reconstruction.AddPriorToResult();

                std::shared_ptr<Reconstruction> filtered_reconstruction = std::make_shared<Reconstruction>();
                rig_reconstruction.FilterUselessPoint2D(filtered_reconstruction);

                CreateDirIfNotExists(rec_path + "-gps");
                filtered_reconstruction->WriteBinary(rec_path + "-gps");
            }
        }

        if (options->independent_mapper_options.extract_colors) {
            reconstruction->ExtractColorsForAllImages(image_path);
        }
        reconstruction->ExportMapPoints(JoinPaths(rec_path, MAPPOINT_NAME));
        
        if (camera.NumLocalCameras() > 1) {
            reconstruction->WriteReconstruction(rec_path, options->independent_mapper_options.write_binary_model);

            if (options->independent_mapper_options.debug_info) {
                std::shared_ptr<Reconstruction> filtered_reconstruction = std::make_shared<Reconstruction>();
                reconstruction->FilterUselessPoint2D(filtered_reconstruction);

                CreateDirIfNotExists(rec_path + "-filtered");
                filtered_reconstruction->WriteBinary(rec_path + "-filtered");
            }

            std::string export_rec_path = rec_path + "-export";
            if (!boost::filesystem::exists(export_rec_path)) {
                boost::filesystem::create_directories(export_rec_path);
            }
            Reconstruction rig_reconstruction;
            reconstruction->ConvertRigReconstruction(rig_reconstruction);

            rig_reconstruction.WriteReconstruction(export_rec_path,
                                                   options->independent_mapper_options.write_binary_model);

            if (options->independent_mapper_options.debug_info) {
                std::shared_ptr<Reconstruction> filtered_reconstruction = std::make_shared<Reconstruction>();
                rig_reconstruction.FilterUselessPoint2D(filtered_reconstruction);

                CreateDirIfNotExists(export_rec_path + "-filtered");
                filtered_reconstruction->WriteBinary(export_rec_path + "-filtered");
            }
        } else {
            reconstruction->WriteReconstruction(rec_path, options->independent_mapper_options.write_binary_model);
            
            if (options->independent_mapper_options.debug_info) {
                std::shared_ptr<Reconstruction> filtered_reconstruction = std::make_shared<Reconstruction>();
                reconstruction->FilterUselessPoint2D(filtered_reconstruction);

                CreateDirIfNotExists(rec_path + "-filtered");
                filtered_reconstruction->WriteBinary(rec_path + "-filtered");
            }
        }
    
        if (options->independent_mapper_options.extract_keyframe) {
            auto keyframe_reconstruction = *reconstruction.get();
            auto image_ids = reconstruction->RegisterImageIds();
            for (auto image_id : image_ids) {
                auto image = reconstruction->Image(image_id);
                if (!image.IsKeyFrame()) {
                    keyframe_reconstruction.DeleteImage(image_id);
                }
            }
            std::string keyframe_rec_path = StringPrintf("%s/%d/KeyFrames", workspace_path.c_str(), base_reconstruction_idx + i);
            if (!boost::filesystem::exists(keyframe_rec_path)) {
                boost::filesystem::create_directories(keyframe_rec_path);
            }
            keyframe_reconstruction.WriteBinary(keyframe_rec_path);
        }
    }
}

bool RTKReady(const std::shared_ptr<sensemap::SceneGraphContainer> &scene_graph_container, double threshold = 0.8) {
    auto image_ids = scene_graph_container->GetImageIds();
    double ready_num = 0;
    for (auto id : image_ids) {
        auto image = scene_graph_container->Image(id);
        if (image.RtkFlag() == 50) {
            ready_num++;
        }
    }
    std::cout << "RTKReady: " << ready_num << " / " << image_ids.size() << std::endl;
    return ready_num / image_ids.size() > threshold;
}

void DirectedSFM(std::shared_ptr<sensemap::SceneGraphContainer> &scene_graph_container,
                 std::shared_ptr<sensemap::ReconstructionManager> &reconstruction_manager, Configurator &param) {
    using namespace sensemap;

    PrintHeading1("Dierected Mapping");

    auto options = std::make_shared<MapperOptions>();
    options->mapper_type = MapperType::INDEPENDENT;
    options->independent_mapper_options.outside_mapper_type = MapperType::INDEPENDENT;
    options->independent_mapper_options.independent_mapper_type = IndependentMapperType::DIRECTED;

    OptionParser option_parser;
    ImageReaderOptions reader_options;
    option_parser.GetImageReaderOptions(reader_options,param);
    option_parser.GetMapperOptions(options->independent_mapper_options,param);

    std::string workspace_path = param.GetArgument("workspace_path", "");
    CHECK(!workspace_path.empty()) << "workspace path empty";
    // if (static_cast<bool>(param.GetArgument("map_update", 0))){
    //     options->independent_mapper_options.map_update = true;
    //     auto reconstruction_idx = reconstruction_manager->Add();
    //     std::shared_ptr<Reconstruction> reconstruction = reconstruction_manager->Get(reconstruction_idx);
    //     auto cameras = scene_graph_container->Cameras();
    //     bool camera_rig = false;
    //     for(auto camera : cameras){
    //         if(camera.second.NumLocalCameras()>1){
    //             camera_rig = true;
    //         }
    //     }
    //     reconstruction->ReadReconstruction(workspace_path + "/0",camera_rig);
    //     // set original image to label 0
    //     auto old_image_ids = reconstruction->RegisterImageIds();
    //     for (auto old_image_id : old_image_ids) {
    //         reconstruction->Image(old_image_id).SetLabelId(0);
    //         reconstruction->Image(old_image_id).SetPoseFlag(true);
    //     }
    //     workspace_path = JoinPaths(workspace_path, "/map_update");
    // }

    std::string image_path = param.GetArgument("image_path", "");
    CHECK(!workspace_path.empty()) << "image path empty";

    bool color_harmonization = param.GetArgument("color_harmonization", 0);
    bool use_gps_prior = static_cast<bool>(param.GetArgument("use_gps_prior", 0));
    
    size_t base_reconstruction_idx = reconstruction_manager->Size();
    std::vector<int> base_reconstruction_ids = reconstruction_manager->getReconstructionIds();

    auto independent_options = std::make_shared<IndependentMapperOptions>(
            options->independent_mapper_options);

    if(independent_options->direct_mapper_type == 0){
        return ;
    }

    while(true) {

        if(!RTKReady(scene_graph_container, 0.4) &&
            (independent_options->direct_mapper_type == 2 || independent_options->direct_mapper_type == 4)){
            break;
        }

        auto *mapper = new DirectedMapperController(independent_options, image_path, workspace_path,
                scene_graph_container, reconstruction_manager);
        mapper->Start();
        mapper->Wait();
        
        fprintf(
                fs, "%s\n",
                StringPrintf("Directed Mapper Elapsed time: %.3f [minutes]", mapper->GetTimer().ElapsedMinutes()).c_str());
        fflush(fs);

        if (mapper->IsSuccess()) {
            std::set<image_t> clustered_image_ids;
            for (auto & image_id : scene_graph_container->GetImageIds()) {
                if (!scene_graph_container->Image(image_id).IsRegistered()) {
                    clustered_image_ids.insert(image_id);
                }
            }
            std::shared_ptr<SceneGraphContainer> cluster_graph_container =
                std::shared_ptr<SceneGraphContainer>(new SceneGraphContainer());
            scene_graph_container->ClusterSceneGraphContainer(clustered_image_ids, *cluster_graph_container.get());

            std::swap(scene_graph_container, cluster_graph_container);

            std::cout << StringPrintf("Remain %d images to be reconstructed!\n", scene_graph_container->NumImages());
        }
        if (!mapper->IsSuccess() || scene_graph_container->NumImages() == 0 || 
            !options->independent_mapper_options.multiple_models) {
            break;
        }
    }
    std::vector<int> reconstruction_ids = reconstruction_manager->getReconstructionIds();
    std::vector<int> remain_reconstruction_ids;
    std::set_difference(reconstruction_ids.begin(), reconstruction_ids.end(), 
                        base_reconstruction_ids.begin(), base_reconstruction_ids.end(), std::back_inserter(remain_reconstruction_ids));
    for (size_t i = 0; i < remain_reconstruction_ids.size(); ++i) {
        auto reconstruction = reconstruction_manager->Get(remain_reconstruction_ids[i]);
        if (reconstruction->RegisterImageIds().size() <= 0) {
            continue;
        }

        if (color_harmonization) {
            reconstruction->ColorHarmonization(image_path);
        }

        // CHECK(reconstruction->RegisterImageIds().size()>0);
        const image_t first_image_id = reconstruction->RegisterImageIds()[0]; 
        CHECK(reconstruction->ExistsImage(first_image_id));
        const Image& image = reconstruction->Image(first_image_id);
        const Camera& camera = reconstruction->Camera(image.CameraId());
        
        std::string rec_path = StringPrintf("%s/%d", workspace_path.c_str(), base_reconstruction_idx + i);
        if (!boost::filesystem::exists(rec_path)) {
            boost::filesystem::create_directories(rec_path);
        }

        if (options->independent_mapper_options.extract_colors) {
            reconstruction->ExtractColorsForAllImages(image_path);
        }
        std::cout << "export to " << JoinPaths(rec_path, MAPPOINT_NAME) << std::endl;
        reconstruction->ExportMapPoints(JoinPaths(rec_path, MAPPOINT_NAME));

        if (camera.NumLocalCameras() > 1) {
            reconstruction->WriteReconstruction(rec_path,
                options->independent_mapper_options.write_binary_model);

            if (options->independent_mapper_options.debug_info) {
                std::shared_ptr<Reconstruction> filtered_reconstruction = std::make_shared<Reconstruction>();
                reconstruction->FilterUselessPoint2D(filtered_reconstruction);

                CreateDirIfNotExists(rec_path + "-filtered");
                filtered_reconstruction->WriteBinary(rec_path + "-filtered");
            }

            std::string export_rec_path = rec_path + "-export";
            if (!boost::filesystem::exists(export_rec_path)) {
                boost::filesystem::create_directories(export_rec_path);
            }
            Reconstruction rig_reconstruction;
            reconstruction->ConvertRigReconstruction(rig_reconstruction);
            rig_reconstruction.WriteReconstruction(export_rec_path,
                options->independent_mapper_options.write_binary_model);

            if (options->independent_mapper_options.debug_info) {
                std::shared_ptr<Reconstruction> filtered_reconstruction = std::make_shared<Reconstruction>();
                rig_reconstruction.FilterUselessPoint2D(filtered_reconstruction);

                CreateDirIfNotExists(export_rec_path + "-filtered");
                filtered_reconstruction->WriteBinary(export_rec_path + "-filtered");
            }
        } else {
            reconstruction->WriteReconstruction(rec_path,
                options->independent_mapper_options.write_binary_model);

            if (options->independent_mapper_options.debug_info) {
                std::shared_ptr<Reconstruction> filtered_reconstruction = std::make_shared<Reconstruction>();
                reconstruction->FilterUselessPoint2D(filtered_reconstruction);

                CreateDirIfNotExists(rec_path + "-filtered");
                filtered_reconstruction->WriteBinary(rec_path + "-filtered");
            }
        }

        if (options->independent_mapper_options.debug_info && use_gps_prior && !static_cast<bool>(param.GetArgument("map_update", 0))) {
            // reconstruction->AlignWithPriorLocations(options->independent_mapper_options.max_error_gps);
            if(camera.NumLocalCameras() <= 1){
                Reconstruction rec = *reconstruction.get();
                rec.AddPriorToResult();
                // rec.NormalizeWoScale();
                
                std::string trans_rec_path = rec_path + "-gps";
                if (!boost::filesystem::exists(trans_rec_path)) {
                    boost::filesystem::create_directories(trans_rec_path);
                }
                // rec.WriteBinary(trans_rec_path);

                std::shared_ptr<Reconstruction> filtered_reconstruction = std::make_shared<Reconstruction>();
                rec.FilterUselessPoint2D(filtered_reconstruction);

                filtered_reconstruction->WriteBinary(trans_rec_path);
            } else {
                Reconstruction rig_reconstruction;
                reconstruction->ConvertRigReconstruction(rig_reconstruction);

                rig_reconstruction.AddPriorToResult();

                std::shared_ptr<Reconstruction> filtered_reconstruction = std::make_shared<Reconstruction>();
                rig_reconstruction.FilterUselessPoint2D(filtered_reconstruction);

                CreateDirIfNotExists(rec_path + "-gps");
                filtered_reconstruction->WriteBinary(rec_path + "-gps");
            }
        }
    }
}

void SetUpReconstructions(std::string slam_poses_path,const std::shared_ptr<sensemap::SceneGraphContainer> &scene_graph_container, 
                          std::shared_ptr<sensemap::ReconstructionManager> &reconstruction_manager, Configurator &param,
                          std::vector<image_t> &image_ids){
    std::cout<<"Setting up Reconstructions... "<<std::endl;
    std::string workspace_path = param.GetArgument("workspace_path", "");

    std::shared_ptr<Reconstruction> slam_reconstruction = std::make_shared<Reconstruction>();
    std::cout<<"slam_poses_path "<<slam_poses_path<<std::endl;
    if(boost::filesystem::is_directory(slam_poses_path)){
        if (boost::filesystem::exists(slam_poses_path)){
            slam_reconstruction->ReadReconstruction(slam_poses_path);
        }
    }else{
        std::map<std::string,std::vector<double>> slam_poses;
        ReadSlamPoses(slam_poses_path,slam_poses);
        ConstructSLAMReconstruction(scene_graph_container, slam_poses,slam_reconstruction,image_ids,workspace_path);
    }
    
    size_t reconstruction_idx;
    if(reconstruction_manager->Size()==0){
        reconstruction_idx = reconstruction_manager->Add();
        std::shared_ptr<Reconstruction> reconstruction_old = reconstruction_manager->Get(reconstruction_idx);

        OptionParser option_parser;
        ImageReaderOptions reader_options;
        option_parser.GetImageReaderOptions(reader_options,param);
        bool camera_rig = reader_options.num_local_cameras > 1;
        std::cout<<"camera rig: "<<camera_rig<<std::endl;
        reconstruction_old->ReadReconstruction(workspace_path + "/0",camera_rig);

        const std::vector<image_t>& image_ids = scene_graph_container->GetImageIds();
        for (const auto image_id : image_ids) {
            if (!reconstruction_old->ExistsImage(image_id)) {
                continue;
            }
            const std::vector<uint32_t>& local_image_indices = scene_graph_container->Image(image_id).LocalImageIndices();
            reconstruction_old->Image(image_id).SetLocalImageIndices(local_image_indices);
        }
    }

    reconstruction_idx = reconstruction_manager->Add();
    std::cout<<"Add Reconstruction "<<reconstruction_idx<<std::endl;
    std::shared_ptr<Reconstruction> reconstruction_slam = reconstruction_manager->Get(reconstruction_idx);
    *reconstruction_slam.get() = std::move(*slam_reconstruction);

}


void ClusterMapper(const std::shared_ptr<sensemap::SceneGraphContainer> &scene_graph_container,
                   std::shared_ptr<sensemap::ReconstructionManager> &reconstruction_manager, Configurator &param) {
    using namespace sensemap;

    PrintHeading1("Cluster Mapping");

    auto options = std::make_shared<MapperOptions>();
    options->mapper_type = MapperType::CLUSTER;
    options->cluster_mapper_options.mapper_options.outside_mapper_type = MapperType::CLUSTER;
    
    auto camera_model = param.GetArgument("camera_model", "OPENCV");
    if((camera_model == "PARTIAL_OPENCV" || camera_model == "OPENCV") && !must_increamental) {
        options->cluster_mapper_options.mapper_options.independent_mapper_type = IndependentMapperType::DIRECTED;
        std::cout << "independent_mapper_type = IndependentMapperType::DIRECTED" << std::endl;
    }

    options->cluster_mapper_options.multiple_models = 
        static_cast<bool>(param.GetArgument("multiple_models", 0));

    options->cluster_mapper_options.enable_image_label_cluster =
        static_cast<bool>(param.GetArgument("enable_image_label_cluster", 1));
    options->cluster_mapper_options.enable_pose_graph_optimization =
        static_cast<bool>(param.GetArgument("enable_pose_graph_optimization", 1));
    options->cluster_mapper_options.enable_cluster_mapper_with_coarse_label =
        static_cast<bool>(param.GetArgument("enable_cluster_mapper_with_coarse_label", 0));
    
    // options->cluster_mapper_options.clustering_options.min_modularity_count =
    //     static_cast<int>(param.GetArgument("min_modularity_count", 5000));
    // options->cluster_mapper_options.clustering_options.max_modularity_count =
    //     static_cast<int>(param.GetArgument("max_modularity_count", -1));
    int max_modularity_count = static_cast<int>(param.GetArgument("max_modularity_count", -1));

    options->cluster_mapper_options.clustering_options.min_modularity_thres =
        static_cast<double>(param.GetArgument("min_modularity_thres", 0.3f));
    options->cluster_mapper_options.clustering_options.community_image_overlap =
        static_cast<int>(param.GetArgument("community_image_overlap", 0));
    options->cluster_mapper_options.clustering_options.community_transitivity =
        static_cast<int>(param.GetArgument("community_transitivity", 1));
    options->cluster_mapper_options.clustering_options.image_dist_seq_overlap =
        static_cast<int>(param.GetArgument("image_dist_seq_overlap", 5));
    options->cluster_mapper_options.clustering_options.image_overlap = param.GetArgument("cluster_image_overlap", 0);

    bool color_harmonization = param.GetArgument("color_harmonization", 0);

    OptionParser option_parser;
    ImageReaderOptions reader_options;
    option_parser.GetImageReaderOptions(reader_options,param);
    option_parser.GetMapperOptions(options->cluster_mapper_options.mapper_options,param);

    int num_local_cameras = reader_options.num_local_cameras;

    std::string workspace_path = param.GetArgument("workspace_path", "");
    CHECK(!workspace_path.empty()) << "image path empty";

    std::string image_path = param.GetArgument("image_path", "");
    CHECK(!workspace_path.empty()) << "image path empty";

    // use prior pose from slam to constrain SfM
    bool use_slam_graph = static_cast<bool>(param.GetArgument("use_slam_graph", 0));
    std::string preview_pose_file = param.GetArgument("preview_pose_file","");
    if (use_slam_graph && (!preview_pose_file.empty()) && boost::filesystem::exists(preview_pose_file)) {
        std::vector<Keyframe> keyframes;
        if (boost::filesystem::path(preview_pose_file).extension().string() == ".tum") {
            LoadPriorPoseFromTum(preview_pose_file, keyframes, (param.GetArgument("image_type", "").compare("rgbd") == 0));
        }
        else {
            LoadPirorPose(preview_pose_file, keyframes);
        }
        
        std::unordered_map<std::string,Keyframe> keyframe_map;
        for(auto const keyframe:keyframes){
            keyframe_map.emplace(keyframe.name,keyframe);
        }

        std::unordered_map<image_t, Eigen::Vector4d> prior_rotations;
        std::unordered_map<image_t, Eigen::Vector3d> prior_translations;

        std::string camera_rig_params_file = param.GetArgument("rig_params_file", "");
        CameraRigParams rig_params;
        bool camera_rig = false;
        Eigen::Matrix3d R0;
        Eigen::Vector3d t0;

        if (!camera_rig_params_file.empty()) {
            if (rig_params.LoadParams(camera_rig_params_file)) {
                camera_rig = true;
                R0 = rig_params.local_extrinsics[0].block<3,3>(0,0);
                t0 = rig_params.local_extrinsics[0].block<3,1>(0,3);

            } else {
                std::cout << "failed to read rig params" << std::endl;
                exit(-1);
            }
        }
        std::vector<image_t> image_ids = scene_graph_container->GetImageIds();
        for (const auto image_id : image_ids) {
            const Image &image = scene_graph_container->Image(image_id);
            
            std::string name = image.Name();
            if(keyframe_map.find(name)!=keyframe_map.end()){
                Keyframe keyframe = keyframe_map.at(name);
                
                Eigen::Matrix3d r = keyframe.rot;

                Eigen::Vector4d q = RotationMatrixToQuaternion(r);
                Eigen::Vector3d t = keyframe.pos;

                if(camera_rig){
                    Eigen::Matrix3d R_rig = R0.transpose()*r;
                    Eigen::Vector3d t_rig = R0.transpose()*(t-t0);
                    q = RotationMatrixToQuaternion(R_rig);
                    t = t_rig;
                }
                prior_rotations.emplace(image_id,q);
                prior_translations.emplace(image_id,t);
            }
        }
        options->cluster_mapper_options.mapper_options.prior_rotations = prior_rotations;
        options->cluster_mapper_options.mapper_options.prior_translations = prior_translations;
        options->cluster_mapper_options.mapper_options.have_prior_pose = true;
    }


    // use gps location prior to constrain the image
    bool use_gps_prior = static_cast<bool>(param.GetArgument("use_gps_prior", 0));
    bool use_prior_align_only = param.GetArgument("use_prior_align_only", 1);
    std::string gps_prior_file = param.GetArgument("gps_prior_file","");
    std::string gps_trans_file = workspace_path + "/gps_trans.txt";

    if (use_gps_prior){
        if (boost::filesystem::exists(gps_prior_file)){
            std::vector<image_t> image_ids = scene_graph_container->GetImageIds();
            std::vector<std::string> image_names;
            for (const auto image_id : image_ids) {
                const Image &image = scene_graph_container->Image(image_id);
                std::string name = image.Name();
                image_names.push_back(name);
            }

            std::vector<std::pair<unsigned long, std::pair<Eigen::Vector3d,int>>> gps_locations;
            if(options->cluster_mapper_options.mapper_options.optimization_use_horizontal_gps_only){
                LoadOriginGPSinfo(gps_prior_file, gps_locations,gps_trans_file,true);
            }
            else{
                LoadOriginGPSinfo(gps_prior_file, gps_locations,gps_trans_file, false);
            }
            std::unordered_map<std::string, std::pair<Eigen::Vector3d,int>> image_locations;
            GPSLocationsToImages(gps_locations, image_names, image_locations);

            std::unordered_map<image_t, std::pair<Eigen::Vector3d,int>> prior_locations_gps;
            for (const auto image_id : image_ids) {
                const Image &image = scene_graph_container->Image(image_id);
                std::string name = image.Name();

                if(image_locations.find(name)!=image_locations.end()){
                    prior_locations_gps.emplace(image_id,image_locations.at(name));
                }
            }
            options->cluster_mapper_options.mapper_options.prior_locations_gps = prior_locations_gps;
        }

        options->cluster_mapper_options.mapper_options.has_gps_prior = true;
        options->cluster_mapper_options.mapper_options.use_prior_align_only = use_prior_align_only;
        options->cluster_mapper_options.mapper_options.min_image_num_for_gps_error =
            param.GetArgument("min_image_num_for_gps_error", 10);

        double prior_absolute_location_weight = 
            static_cast<double>(param.GetArgument("prior_absolute_location_weight", 1.0f));
       options->cluster_mapper_options.mapper_options.prior_absolute_location_weight = prior_absolute_location_weight;
    }

    size_t base_reconstruction_idx = reconstruction_manager->Size();
    std::vector<int> base_reconstruction_ids = reconstruction_manager->getReconstructionIds();

    options->cluster_mapper_options.only_merge_cluster = param.GetArgument("only_merge_cluster", 0);
    std::cout << "reconstruction_manager: " << reconstruction_manager->Size() << ", " 
              << options->cluster_mapper_options.only_merge_cluster << std::endl;
#if 0
    if (options->cluster_mapper_options.only_merge_cluster){
        for (int rect_idx = 1; ; rect_idx++){
            auto reconstruction_path = JoinPaths(workspace_path, "cluster" + std::to_string(rect_idx), "0");
            if (!ExistsDir(reconstruction_path)){
                break;
            }
            auto rect_id = reconstruction_manager->Add();
            auto rect = reconstruction_manager->Get(rect_id);
            rect->ReadReconstruction(reconstruction_path);
            std::cout << "Read reconstruction-" << rect_id << std::endl;
        }
    }
    std::cout << "reconstruction_manager: " << reconstruction_manager->Size() << std::endl;
#endif

    if (max_modularity_count < 0){
        if (num_local_cameras > 1){
            options->cluster_mapper_options.clustering_options.max_modularity_count = 4000;
            options->cluster_mapper_options.clustering_options.min_modularity_count = 2000;
            options->cluster_mapper_options.clustering_options.leaf_max_num_images = 200;
        } else {
            options->cluster_mapper_options.clustering_options.max_modularity_count = 10000;
            options->cluster_mapper_options.clustering_options.min_modularity_count = 5000;
            options->cluster_mapper_options.clustering_options.leaf_max_num_images = 500;
        }
    } else {
        options->cluster_mapper_options.clustering_options.max_modularity_count = max_modularity_count;
        options->cluster_mapper_options.clustering_options.min_modularity_count = max_modularity_count / 2;
        options->cluster_mapper_options.clustering_options.leaf_max_num_images = max_modularity_count / 20;
    }

    MapperController *mapper = MapperController::Create(options, workspace_path, image_path,
                                                        scene_graph_container, reconstruction_manager);
    mapper->Start();
    mapper->Wait();

    fprintf(fs, "%s\n",
            StringPrintf("Cluster Mapper Elapsed time: %.3f [minutes]", mapper->GetTimer().ElapsedMinutes()).c_str());
    fflush(fs);

    std::cout<<"Reconstruction Component Size: "<< reconstruction_manager->Size()<<std::endl;

    std::vector<int> reconstruction_ids = reconstruction_manager->getReconstructionIds();
    std::vector<int> remain_reconstruction_ids;
    std::set_difference(reconstruction_ids.begin(), reconstruction_ids.end(), 
                        base_reconstruction_ids.begin(), base_reconstruction_ids.end(), std::back_inserter(remain_reconstruction_ids));

    for (size_t i = 0; i < remain_reconstruction_ids.size(); ++i) {
        auto reconstruction = reconstruction_manager->Get(remain_reconstruction_ids[i]);
        if (reconstruction->RegisterImageIds().size() <= 0) {
            continue;
        }

        if (color_harmonization) {
            reconstruction->ColorHarmonization(image_path);
        }

        if (use_gps_prior){
            reconstruction->AlignWithPriorLocations(
                options->cluster_mapper_options.IndependentMapper().max_error_gps);
        }

        const image_t first_image_id = reconstruction->RegisterImageIds()[0]; 
        CHECK(reconstruction->ExistsImage(first_image_id));
        const Image& image = reconstruction->Image(first_image_id);
        const Camera& camera = reconstruction->Camera(image.CameraId());
        
        std::string rec_path = StringPrintf("%s/%d", workspace_path.c_str(), base_reconstruction_idx + i);
        if (!boost::filesystem::exists(rec_path)) {
            boost::filesystem::create_directories(rec_path);
        }
        if (options->cluster_mapper_options.mapper_options.extract_colors) {
            reconstruction->ExtractColorsForAllImages(image_path);
        }

        std::cout << "export to " << JoinPaths(rec_path, MAPPOINT_NAME) << std::endl;
        reconstruction->ExportMapPoints(JoinPaths(rec_path, MAPPOINT_NAME));

        if (camera.NumLocalCameras() > 1) {
            reconstruction->WriteReconstruction(rec_path,
                options->independent_mapper_options.write_binary_model);

            std::string export_rec_path = rec_path + "-export";
            if (!boost::filesystem::exists(export_rec_path)) {
                boost::filesystem::create_directories(export_rec_path);
            }
            Reconstruction rig_reconstruction;
            reconstruction->ConvertRigReconstruction(rig_reconstruction);
            rig_reconstruction.WriteReconstruction(export_rec_path,
                options->independent_mapper_options.write_binary_model);
            
            if (options->independent_mapper_options.debug_info) {
                std::shared_ptr<Reconstruction> filtered_reconstruction = std::make_shared<Reconstruction>();
                rig_reconstruction.FilterUselessPoint2D(filtered_reconstruction);

                CreateDirIfNotExists(export_rec_path + "-filtered");
                filtered_reconstruction->WriteBinary(export_rec_path + "-filtered");
            }
        } else {
            reconstruction->WriteReconstruction(rec_path,
                options->independent_mapper_options.write_binary_model);
            
            if (options->independent_mapper_options.debug_info) {
                std::shared_ptr<Reconstruction> filtered_reconstruction = std::make_shared<Reconstruction>();
                reconstruction->FilterUselessPoint2D(filtered_reconstruction);

                CreateDirIfNotExists(rec_path + "-filtered");
                filtered_reconstruction->WriteBinary(rec_path + "-filtered");
            }
        }
    }
}

int main(int argc, char *argv[]) {
    using namespace sensemap;

    PrintHeading("SenseMap.  Copyright(c) 2019, SenseTime Group.");
    PrintHeading(std::string("Version: hd-sfm-")+__VERSION__);
    Timer timer;
    timer.Start();

    // {
    //     Reconstruction reconstruction;
    //     reconstruction.ReadBinary(argv[1]);
    //     auto image_ids = reconstruction.RegisterImageIds();
    //     // std::vector<int> selected_image_ids;
    //     // selected_image_ids.push_back(295);
    //     // selected_image_ids.push_back(296);

    //     Eigen::Vector4d qvec0;
    //     Eigen::Vector3d tvec0;
    //     for (int i = 0; i < image_ids.size(); ++i) {
    //         auto image_id = image_ids[i];
    //         class Image & image = reconstruction.Image(image_id);
    //         class Camera & camera = reconstruction.Camera(image.CameraId());

    //         if (camera.NumLocalCameras() > 1) {
    //             for (int local_camera_id = 0; local_camera_id < camera.NumLocalCameras(); ++local_camera_id) {
    //                 Eigen::Vector4d local_qvec;
    //                 Eigen::Vector3d local_tvec;
    //                 camera.GetLocalCameraExtrinsic(local_camera_id,local_qvec,local_tvec);
                    
    //                 // Eigen::Vector4d qvec = ConcatenateQuaternions(image.Qvec(),local_qvec);
    //                 // Eigen::Vector3d tvec = 
    //                 //     QuaternionToRotationMatrix(local_qvec)*image.Tvec() + local_tvec;

    //                 Eigen::Matrix3x4d T = Eigen::Matrix3x4d::Identity();
    //                 T.block<3, 3>(0, 0) = QuaternionToRotationMatrix(local_qvec);
    //                 T.block<3, 1>(0, 3) = local_tvec;
    //                 std::cout << "local camera id " << local_camera_id << std::endl;
    //                 std::cout << T << std::endl;
    //             }
    //         } else {
    //             if (i == 0) {
    //                 qvec0 = image.Qvec();
    //                 tvec0 = image.Tvec();
    //             } else {
    //                 Eigen::Vector4d qvec0_inverse = InvertQuaternion(qvec0);
    //                 Eigen::Vector4d qvec = ConcatenateQuaternions(image.Qvec(), qvec0_inverse);
    //                 Eigen::Vector3d tvec = -QuaternionToRotationMatrix(qvec) * image.Tvec() + tvec0;
                    
    //                 Eigen::Matrix3x4d T = Eigen::Matrix3x4d::Identity();
    //                 T.block<3, 3>(0, 0) = QuaternionToRotationMatrix(qvec);
    //                 T.block<3, 1>(0, 3) = tvec;
    //                 std::cout << T << std::endl;
    //             }
    //         }
    //         std::string intr_str = camera.ParamsToString();
    //         std::cout << "intrinsic: " << std::endl;
    //         std::cout << intr_str << std::endl;
    //         break;
    //     }

    //     return 0;
    // }

    if (0) {
        Voxel::Option option1;
        Voxel voxel1(option1);

        std::vector<Eigen::Vector3d> points;
        for (int i = 0; i < 100; ++i) {
            Eigen::Vector3d X;
            X[0] = (rand() % 100) / 100000.0;
            X[1] = (rand() % 100) / 100000.0;
            X[2] = (rand() % 100) / 100000.0;
            points.push_back(X);
            voxel1.Add(X);
        }
        for (int i = 50; i < 100; ++i) {
            voxel1.Sub(points[i]);
        }
        for (int i = 50; i < 70; ++i) {
            voxel1.Add(points[i]);
        }
        for (int i = 30; i < 70; ++i) {
            voxel1.Sub(points[i]);
        }
        for (int i = 30; i < 70; ++i) {
            voxel1.Add(points[i]);
        }
        for (int i = 0; i < 70; ++i) {
            voxel1.Sub(points[i]);
        }
        for (int i = 50; i < 100; ++i) {
            voxel1.Add(points[i]);
        }
        voxel1.ComputeFeature();

        points.resize(50);

        Voxel::Option option;
        Voxel voxel(option);
        voxel.Init(points);

        Eigen::Vector3d m_var = voxel.GetEx();
        Eigen::Matrix3d m_cov = voxel.GetCov();
        Eigen::Matrix3d m_inv_cov = voxel.GetInvCov();
        Eigen::Vector3d m_pivot = voxel.GetPivot();
        std::cout << "m_var: " << m_var.transpose() << std::endl;
        std::cout << "m_pivot: " << m_pivot.transpose() << std::endl;
        std::cout << "m_cov: " << m_cov.transpose() << std::endl;
        std::cout << "m_det: " << m_cov.determinant() << std::endl;
        std::cout << "m_inv_cov: " << m_inv_cov.transpose() << std::endl;

        Eigen::Vector3d m_var1 = voxel1.GetEx();
        Eigen::Matrix3d m_cov1 = voxel1.GetCov();
        Eigen::Matrix3d m_inv_cov1 = voxel1.GetInvCov();
        Eigen::Vector3d m_pivot1 = voxel1.GetPivot();
        std::cout << "m_var1: " << m_var1.transpose() << std::endl;
        std::cout << "m_pivot1: " << m_pivot1.transpose() << std::endl;
        std::cout << "m_cov1: " << m_cov1.transpose() << std::endl;
        std::cout << "m_det1: " << m_cov1.determinant() << std::endl;
        std::cout << "m_inv_cov1: " << m_inv_cov1.transpose() << std::endl;

        for (int i = 0; i < 1000; ++i) {
            Eigen::Vector3d X;
            X[0] = (rand() % 100) / 10000.0;
            X[1] = (rand() % 100) / 10000.0;
            X[2] = (rand() % 100) / 10000.0;

            Eigen::Vector3d ray = X - m_var1;
            double proj_len = ray.dot(m_pivot1);
            Eigen::Vector3d residual_vec = m_pivot1 * proj_len;

            std::cout << "X: " << X.transpose() << std::endl;
            std::cout << "residual_vec: " << residual_vec.transpose() << std::endl;
            std::cout << "m_pivot1: " << m_pivot1.transpose() << std::endl;

            double y = 0.5 * (m_inv_cov1(0, 0) * residual_vec[0] * residual_vec[0] + 
                                m_inv_cov1(1, 1) * residual_vec[1] * residual_vec[1] + 
                                m_inv_cov1(2, 2) * residual_vec[2] * residual_vec[2] +
                                (m_inv_cov1(0, 1) + m_inv_cov1(1, 0)) * residual_vec[0] * residual_vec[1] +
                                (m_inv_cov1(1, 2) + m_inv_cov1(2, 1)) * residual_vec[1] * residual_vec[2] +
                                (m_inv_cov1(0, 2) + m_inv_cov1(2, 0)) * residual_vec[0] * residual_vec[2]);
            double error = 1 - ceres::exp(-y);
            // double error = voxel1.Error(X);
            std::cout << "error: " << error << std::endl;
        }
        return 0;
    }

    std::string configuration_file_path = std::string(argv[1]);
    Configurator param;
    param.Load(configuration_file_path.c_str());

    std::string workspace_path = param.GetArgument("workspace_path", "");
    CHECK(!workspace_path.empty()) << "workspace path empty";
    if (static_cast<bool>(param.GetArgument("map_update", 0))){
        workspace_path = JoinPaths(workspace_path, "/map_update");
    }

    std::string image_path = param.GetArgument("image_path", "");
    CHECK(!image_path.empty()) << "image path empty";

    if (!boost::filesystem::exists(workspace_path)) {
        CHECK(boost::filesystem::create_directories(workspace_path)) << "Create workspace failed";
    }

    fs = fopen((workspace_path + "/time_mapping.txt").c_str(), "w");

    auto reconstruction_manager = std::make_shared<ReconstructionManager>();
    auto scene_graph_container = std::make_shared<SceneGraphContainer>();
    auto feature_data_container = std::make_shared<FeatureDataContainer>();

    CHECK(LoadFeaturesAndMatches(*feature_data_container.get(), *scene_graph_container.get(), param))
        << "Load features or matches failed";

    std::vector<image_t> image_ids = feature_data_container->GetImageIds();
    std::cout<<"total image_ids size " << image_ids.size() << std::endl;

    feature_data_container.reset();
    std::cout << "malloc_trim: " << malloc_trim(0) << std::endl;
    float available_memory;
    GetAvailableMemory(available_memory);
    std::cout << "Available Memory[ReleaseFeatures]: " << available_memory << "GB" << std::endl;

    std::string mapper_method = param.GetArgument("mapper_method", "incremental");
    bool multiple_models = static_cast<bool>(param.GetArgument("multiple_models", 0));
    
    bool refine_separate_cameras = static_cast<bool>(param.GetArgument("refine_separate_cameras", 0));
    must_increamental = static_cast<bool>(param.GetArgument("must_increamental", 0)); 
    std::cout << "must_increamental: " << must_increamental << std::endl;

    size_t num_images = scene_graph_container->NumImages();
    if (refine_separate_cameras) {
        num_images = 0;
        const auto & image_ids = scene_graph_container->GetImageIds();
        for (auto image_id : image_ids) {
            auto image = scene_graph_container->Image(image_id);
            auto camera = scene_graph_container->Camera(image.CameraId());
            num_images += camera.NumLocalCameras();
        }
    }

    if(static_cast<bool>(param.GetArgument("slam_update", 0))) {
        std::cout<<"use slam update map"<<std::endl;
        std::string slam_poses_path =  param.GetArgument("slam_poses_path", "");
        std::cout<<"slam_poses_path "<<slam_poses_path<<std::endl;
        std::vector<std::string> path_list = GetFileList(slam_poses_path);
        for(auto file_path : path_list){
            if (file_path.substr(file_path.length() - 4, file_path.length()) != ".txt"){
                continue;	
            }
            SetUpReconstructions(file_path,scene_graph_container, reconstruction_manager, param, image_ids);
            ClusterMapper(scene_graph_container, reconstruction_manager, param);
        }
    } else {
        std::unordered_map<image_t, std::unordered_set<image_t> > cc;
        GetAllConnectedComponentIds(*scene_graph_container->CorrespondenceGraph(), cc);

        std::cout << StringPrintf("Get %d component\n", cc.size());

        std::vector<std::pair<image_t, int> > sorted_clustered_images;
        sorted_clustered_images.reserve(cc.size());
        for (auto & clustered_image_ids : cc) {
            sorted_clustered_images.emplace_back(clustered_image_ids.first, clustered_image_ids.second.size());
        }
        std::sort(sorted_clustered_images.begin(), sorted_clustered_images.end(), 
            [&](const auto & image1, const auto & image2) {
                return image1.second > image2.second;
            });

        int component_id = 0;
        for (auto & cluster_image : sorted_clustered_images) {
            auto & clustered_image_ids = cc.at(cluster_image.first);
            PrintHeading1(StringPrintf("Processing component %d", component_id++));
            std::shared_ptr<SceneGraphContainer> cluster_graph_container =
                std::shared_ptr<SceneGraphContainer>(new SceneGraphContainer());

            std::set<image_t> unique_image_ids;
            for (auto & image_id : clustered_image_ids) {
                unique_image_ids.insert(image_id);
            }

            scene_graph_container->ClusterSceneGraphContainer(unique_image_ids, *cluster_graph_container.get());
            std::string camera_model = "";
            for (auto camera : scene_graph_container->Cameras()) {
                camera_model = camera.second.ModelName();
            }
            
            size_t cluster_num_images = cluster_graph_container->NumImages();
            if (refine_separate_cameras) {
                cluster_num_images = 0;
                const auto & image_ids = cluster_graph_container->GetImageIds();
                for (auto image_id : image_ids) {
                    auto image = cluster_graph_container->Image(image_id);
                    auto camera = cluster_graph_container->Camera(image.CameraId());
                    cluster_num_images += camera.NumLocalCameras();
                }
            }
            if (static_cast<bool>(param.GetArgument("map_update", 0))) {
                IncrementalSFM(cluster_graph_container, reconstruction_manager, param);
            } else {
                if (mapper_method.compare("incremental") == 0) {
                    // auto camera_model = param.GetArgument("camera_model", "OPENCV");
                    if((camera_model == "PARTIAL_OPENCV" || camera_model == "OPENCV") && !must_increamental) {
                        DirectedSFM(cluster_graph_container, reconstruction_manager, param);
                    }
                    if (cluster_graph_container->NumImages() > 0) {
                        IncrementalSFM(cluster_graph_container, reconstruction_manager, param);
                    }
                } else if (mapper_method.compare("cluster") == 0) {
                    ClusterMapper(cluster_graph_container, reconstruction_manager, param);
                }
            }
            if (!multiple_models) {
                bool rec_success = false;
                for (size_t rec_idx = 0; rec_idx < reconstruction_manager->Size(); ++rec_idx) {
                    auto reconstruction = reconstruction_manager->Get(rec_idx);
                    if (reconstruction->NumRegisterImages() >= 0.1f * cluster_num_images) {
                        rec_success = true;
                        break;
                    }
                }
                if (rec_success) {
                    break;
                }
            }
        }
    }
    std::cout << StringPrintf("Reconstruct in %.3fs", timer.ElapsedSeconds()) << std::endl;
    PrintReconSummary(workspace_path + "/statistic.txt", num_images, reconstruction_manager);

    fclose(fs);
    return 0;
}
