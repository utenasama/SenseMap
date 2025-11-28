// Copyright (c) 2019, SenseTime Group.
// All rights reserved.

#include <boost/filesystem/path.hpp>
#include <opencv2/opencv.hpp>

#include "base/reconstruction_manager.h"
#include "container/feature_data_container.h"
#include "feature/utils.h"
#include "graph/correspondence_graph.h"
#include "util/misc.h"

#include "base/similarity_transform.h"
#include "controllers/cluster_mapper_controller.h"

std::string image_path;
std::string workspace_path;

bool dirExists(const std::string& dirName_in) { return access(dirName_in.c_str(), F_OK) == 0; }

void GetReconstructionDir(std::vector<std::string>& reconstruct_dirs) {
    reconstruct_dirs.clear();

    // Get all the reconstruction file
    std::vector<std::string> file_list = sensemap::GetDirList(workspace_path);
    for (auto file_dir : file_list) {
        auto parent_dir = sensemap::GetParentDir(file_dir);
        auto len = parent_dir.length();
        auto filename = file_dir.substr(len + 1, file_dir.length() - len);
        if (filename.length() <= 7 ||
            filename.substr(0, 7).compare("cluster") != 0) {
            continue;
        }
        // Check the reconstruction file exisit or not
        if (boost::filesystem::is_directory(file_dir + "/0")) {
            file_dir = file_dir + "/0";
            if ((dirExists(file_dir + "/cameras.txt") && dirExists(file_dir + "/images.txt") &&
                 dirExists(file_dir + "/points3D.txt")) ||
                (dirExists(file_dir + "/cameras.bin") && dirExists(file_dir + "/images.bin") &&
                 dirExists(file_dir + "/points3D.bin"))) {
                reconstruct_dirs.emplace_back(file_dir);
            }
        }
    }

    // Sort the reconstruct by name
    std::sort(reconstruct_dirs.begin(), reconstruct_dirs.end(),
              [](const std::string& v1, const std::string& v2) { return v1 < v2; });
}

int main(int argc, char* argv[]) {
    using namespace sensemap;

    image_path = std::string(argv[1]);
    workspace_path = std::string(argv[2]);

    PrintHeading("SenseMap.  Copyright(c) 2019, SenseTime Group.");

    auto reconstruction_manager = std::make_shared<ReconstructionManager>();
    auto scene_graph_container = std::make_shared<SceneGraphContainer>();
    auto feature_data_container = std::make_shared<FeatureDataContainer>();

    scene_graph_container->ReadSceneGraphBinaryData(workspace_path + "/scene_graph.bin");
    if (boost::filesystem::exists(JoinPaths(workspace_path, "/strong_loops.bin"))) {
        std::cout<<"read strong_loops file "<<std::endl;
        scene_graph_container->ReadStrongLoopsBinaryData(JoinPaths(workspace_path, "/strong_loops.bin"));
    }

    auto correspondence_graph = scene_graph_container->CorrespondenceGraph();

    EIGEN_STL_UMAP(image_t, class Image)& images = scene_graph_container->Images();
    EIGEN_STL_UMAP(camera_t, class Camera)& cameras = scene_graph_container->Cameras();

    // FeatureDataContainer data_container;

    feature_data_container->ReadCamerasBinaryData(workspace_path + "/cameras.bin");
    feature_data_container->ReadImagesBinaryData(workspace_path + "/features.bin");

    std::vector<image_t> image_ids = feature_data_container->GetImageIds();

    std::cout << "image_ids.size() = " << image_ids.size() << std::endl;

    for (const auto image_id : image_ids) {
        const Image& image = feature_data_container->GetImage(image_id);
        images[image_id] = image;
        const FeatureKeypoints& keypoints = feature_data_container->GetKeypoints(image_id);
        const std::vector<Eigen::Vector2d> points = FeatureKeypointsToPointsVector(keypoints);
        // images[image_id].SetPoints2D(points);
        images[image_id].SetPoints2D(keypoints);
        const Camera& camera = feature_data_container->GetCamera(image.CameraId());
        if (!scene_graph_container->ExistsCamera(image.CameraId())) {
            cameras[image.CameraId()] = camera;
            // cameras[image.CameraId()].SetCameraConstant(true);
        }
        if (correspondence_graph->ExistsImage(image_id)) {
            images[image_id].SetNumObservations(correspondence_graph->NumObservationsForImage(image_id));
            images[image_id].SetNumCorrespondences(correspondence_graph->NumCorrespondencesForImage(image_id));
        } else {
            std::cout << "Do not contain this image" << std::endl;
        }
    }
    std::cout << "Load correspondece graph finished" << std::endl;
    correspondence_graph->Finalize();

    std::cout << "ExportToGraph" << std::endl;

    // Read exisiting reconstruction list
    std::vector<std::string> reconstruction_dirs;
    size_t reconstruction_idx = 0;
    GetReconstructionDir(reconstruction_dirs);
    for (const auto& reconstruction_dir : reconstruction_dirs) {
        std::cout << reconstruction_dir << std::endl;
        // Read exisiting reconstructions
        reconstruction_idx = reconstruction_manager->Add();
        std::shared_ptr<Reconstruction> reconstruction_tmp = reconstruction_manager->Get(reconstruction_idx);
        reconstruction_tmp->ReadReconstruction(reconstruction_dir);
    }

    PrintHeading1("Cluster Mapping");

    auto options = std::make_shared<MapperOptions>();

    options->mapper_type = MapperType::CLUSTER;
    options->cluster_mapper_options.clustering_options.image_overlap = 0;
    options->cluster_mapper_options.clustering_options.leaf_max_num_images = 150;
    options->cluster_mapper_options.mapper_options.ba_refine_principal_point = false;
    options->cluster_mapper_options.mapper_options.ba_refine_focal_length = false;
    options->cluster_mapper_options.mapper_options.ba_refine_extra_params = false;
    options->cluster_mapper_options.mapper_options.ba_global_use_pba = true;

    // -- Enable only merge
    options->cluster_mapper_options.only_merge_cluster = true;

    options->cluster_mapper_options.mapper_options.debug_info = true;

    // Merge cluster related
    // options->cluster_mapper_options.merge_options.OverlapCheckPnP = false;
    // options->cluster_mapper_options.merge_options.LoopVerifiedBySIM3 = false;

    // // // Change to get more loop edge
    // options->cluster_mapper_options.merge_options.OnlyAddCloseImageID = true;
    // options->cluster_mapper_options.merge_options.OnlyAddSameImageID = false;

    // // options->cluster_mapper_options.merge_options.Corespondence2DThreshold = 50;  // 50 30
    // // options->cluster_mapper_options.merge_options.Corespondence3DThreshold = 45;  // 45 20
    // options->cluster_mapper_options.merge_options.PnPInlierTreshold = 0;  // 40 20

    // options->cluster_mapper_options.merge_options.lossfunction_enable = false;
    // options->cluster_mapper_options.merge_options.CheckInlierByDistance = false;  // false
    // options->cluster_mapper_options.merge_options.enable_final_ba = false;

    // // FIXME: Tmp
    // options->cluster_mapper_options.merge_options.EnableLoopVerified = false;
    // options->cluster_mapper_options.merge_options.Corespondence2DThreshold = 0;
    // options->cluster_mapper_options.merge_options.Corespondence3DThreshold = 0;
    // options->cluster_mapper_options.merge_options.attempt_pnp_after_sim3_fail = true;

    MapperController* mapper = MapperController::Create(options, workspace_path,
                                                        image_path, 
                                                        scene_graph_container, reconstruction_manager);
    mapper->Start();
    mapper->Wait();

    std::string s = StringPrintf("Cluster Merge Elapsed time: %.3f [minutes]", mapper->GetTimer().ElapsedMinutes());
    std::cout << s << std::endl;

    if (reconstruction_manager->Size() > 0) {
        std::string rec_path = StringPrintf("%s/0", workspace_path.c_str());
        if (boost::filesystem::exists(rec_path)) {
            boost::filesystem::remove_all(rec_path);
        }
        boost::filesystem::create_directories(rec_path);
        reconstruction_manager->Get(0)->WriteReconstruction(
            rec_path, options->cluster_mapper_options.mapper_options.write_binary_model);
    }
    return 0;
}
