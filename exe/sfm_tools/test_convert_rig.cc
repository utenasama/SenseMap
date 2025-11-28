// Copyright (c) 2019, SenseTime Group.
// All rights reserved.

#include <dirent.h>
#include <sys/stat.h>

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

#define MAX_PRECISION std::setprecision(std::numeric_limits<double>::digits10 + 1)

using namespace sensemap;
FILE *fs;

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
        std::cout << "..." << std::endl;
        return false;
    }

    // load feature data
    if (boost::filesystem::exists(JoinPaths(workspace_path, "/local_cameras.bin"))) {
        feature_data_container.ReadCamerasBinaryData(JoinPaths(workspace_path, "/cameras.bin"), false);
        feature_data_container.ReadLocalCamerasBinaryData(JoinPaths(workspace_path, "/local_cameras.bin"));
    } else {
        feature_data_container.ReadCamerasBinaryData(JoinPaths(workspace_path, "/cameras.bin"), camera_rig);
    }

    feature_data_container.ReadImagesBinaryDataWithoutDescriptor(JoinPaths(workspace_path, "/features.bin"));

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

    // load match data
    scene_graph.ReadSceneGraphBinaryData(JoinPaths(workspace_path, "/scene_graph.bin"));
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

    return true;
}

int main(int argc, char *argv[]) {
    using namespace sensemap;

    PrintHeading("SenseMap.  Copyright(c) 2019, SenseTime Group.");
    PrintHeading(std::string("Version: hd-sfm-")+__VERSION__);
    Timer timer;
    timer.Start();

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

    auto reconstruction_manager = std::make_shared<ReconstructionManager>();
    auto scene_graph_container = std::make_shared<SceneGraphContainer>();
    auto feature_data_container = std::make_shared<FeatureDataContainer>();

    CHECK(LoadFeaturesAndMatches(*feature_data_container.get(), *scene_graph_container.get(), param))
        << "Load features or matches failed";
        
    reconstruction_manager->Add();
    Reconstruction& reconstruction = *reconstruction_manager->Get(0);
    reconstruction.ReadReconstruction(workspace_path + "/0", 1);
    std::cout << "reconstruction: " << reconstruction.NumRegisterImages() << std::endl;

    const auto& image_ids = reconstruction.RegisterImageIds();
    for (int i = 0; i < image_ids.size(); i++){
        const auto image_id = image_ids.at(i);
        auto& rect_image = reconstruction.Image(image_id);
        auto feature_image = scene_graph_container->Image(image_id);
        rect_image.SetLocalImageIndices(feature_image.LocalImageIndices());
    }
    
    Reconstruction rig_reconstruction;
    reconstruction.ConvertRigReconstruction(rig_reconstruction);
    std::string export_rec_path = workspace_path + "/0-export";
    if (!boost::filesystem::exists(export_rec_path)) {
        boost::filesystem::create_directories(export_rec_path);
    }
    rig_reconstruction.WriteReconstruction(export_rec_path, 1);

    std::cout << StringPrintf("Convert Reconstruct in %.3fs", timer.ElapsedSeconds()) << std::endl;
    
    return 0;
}
