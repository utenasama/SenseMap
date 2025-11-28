// Copyright (c) 2019, SenseTime Group.
// All rights reserved.

#include <dirent.h>
#include <sys/stat.h>

#include <boost/filesystem/path.hpp>
#include <unordered_set>

#include "../Configurator_yaml.h"
#include "../camera_rig_params.h"
#include "../option_parsing.h"
#include "base/version.h"
#include "../system_io.h"
#include "base/reconstruction_manager.h"
#include "container/feature_data_container.h"
#include "controllers/incremental_mapper_controller.h"
#include "feature/extraction.h"
#include "feature/feature_matcher.h"
#include "feature/utils.h"
#include "graph/correspondence_graph.h"
#include "util/gps_reader.h"
#include "util/mat.h"
#include "util/misc.h"
#include "util/ply.h"
#include "util/rgbd_helper.h"

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

void ClusterMapper(const std::shared_ptr<sensemap::SceneGraphContainer> &scene_graph_container,
                   std::shared_ptr<sensemap::ReconstructionManager> &reconstruction_manager, Configurator &param) {
    using namespace sensemap;

    PrintHeading1("Cluster Mapping");

    auto options = std::make_shared<MapperOptions>();
    options->mapper_type = MapperType::CLUSTER;
    options->cluster_mapper_options.mapper_options.outside_mapper_type = MapperType::CLUSTER;

    options->cluster_mapper_options.enable_image_label_cluster =
        static_cast<bool>(param.GetArgument("enable_image_label_cluster", 1));
    options->cluster_mapper_options.enable_pose_graph_optimization =
        static_cast<bool>(param.GetArgument("enable_pose_graph_optimization", 1));
    options->cluster_mapper_options.enable_cluster_mapper_with_coarse_label =
        static_cast<bool>(param.GetArgument("enable_cluster_mapper_with_coarse_label", 0));

    options->cluster_mapper_options.clustering_options.image_overlap = param.GetArgument("cluster_image_overlap", 0);

    OptionParser option_parser;
    ImageReaderOptions reader_options;
    option_parser.GetImageReaderOptions(reader_options, param);
    option_parser.GetMapperOptions(options->cluster_mapper_options.mapper_options, param);

    int num_local_cameras = reader_options.num_local_cameras;

    std::string workspace_path = param.GetArgument("workspace_path", "");
    CHECK(!workspace_path.empty()) << "image path empty";

    std::string image_path = param.GetArgument("image_path", "");
    CHECK(!workspace_path.empty()) << "image path empty";

    // use gps location prior to constrain the image
    bool use_gps_prior = static_cast<bool>(param.GetArgument("use_gps_prior", 0));
    std::string gps_prior_file = param.GetArgument("gps_prior_file", "");
    std::string gps_trans_file = workspace_path + "/gps_trans.txt";

    if (use_gps_prior) {
        if (boost::filesystem::exists(gps_prior_file)) {
            std::vector<image_t> image_ids = scene_graph_container->GetImageIds();
            std::vector<std::string> image_names;
            for (const auto image_id : image_ids) {
                const Image &image = scene_graph_container->Image(image_id);
                std::string name = image.Name();
                image_names.push_back(name);
            }

            std::vector<std::pair<unsigned long, std::pair<Eigen::Vector3d, int>>> gps_locations;
            if (options->cluster_mapper_options.mapper_options.optimization_use_horizontal_gps_only) {
                LoadOriginGPSinfo(gps_prior_file, gps_locations, gps_trans_file, true);
            } else {
                LoadOriginGPSinfo(gps_prior_file, gps_locations, gps_trans_file, false);
            }
            std::unordered_map<std::string, std::pair<Eigen::Vector3d, int>> image_locations;
            GPSLocationsToImages(gps_locations, image_names, image_locations);

            std::unordered_map<image_t, std::pair<Eigen::Vector3d, int>> prior_locations_gps;
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
            options->cluster_mapper_options.mapper_options.prior_locations_gps = prior_locations_gps;
            options->independent_mapper_options.original_gps_locations = gps_locations;
            sensemap::WriteBinaryPlyPoints(workspace_path + "/gps.ply", gps_locations_ply);
            options->independent_mapper_options.has_gps_prior = true;
        }
        options->cluster_mapper_options.mapper_options.min_image_num_for_gps_error =
            param.GetArgument("min_image_num_for_gps_error", 10);

        double prior_absolute_location_weight =
            static_cast<double>(param.GetArgument("prior_absolute_location_weight", 1.0f));
        options->cluster_mapper_options.mapper_options.prior_absolute_location_weight = prior_absolute_location_weight;
    }

    MapperController *mapper =
        MapperController::Create(options, workspace_path, image_path, scene_graph_container, reconstruction_manager);
    mapper->Start();
    mapper->Wait();

    fprintf(fs, "%s\n",
            StringPrintf("Cluster Mapper Elapsed time: %.3f [minutes]", mapper->GetTimer().ElapsedMinutes()).c_str());
    fflush(fs);

    std::cout << "Reconstruction Component Size: " << reconstruction_manager->Size() << std::endl;
    for (size_t i = 0; i < reconstruction_manager->Size(); ++i) {
        auto reconstruction = reconstruction_manager->Get(i);
        CHECK(reconstruction->RegisterImageIds().size() > 0);
        const image_t first_image_id = reconstruction->RegisterImageIds()[0];
        CHECK(reconstruction->ExistsImage(first_image_id));
        const Image &image = reconstruction->Image(first_image_id);
        const Camera &camera = reconstruction->Camera(image.CameraId());

        std::string rec_path = StringPrintf("%s/%d", workspace_path.c_str(), i);
        if (!boost::filesystem::exists(rec_path)) {
            boost::filesystem::create_directories(rec_path);
        }

        if (camera.NumLocalCameras() > 1) {
            reconstruction->FilterAllMapPoints(2, 4.0, 1.5);
            reconstruction->WriteReconstruction(rec_path,
                                                options->cluster_mapper_options.mapper_options.write_binary_model);

            std::string export_rec_path = rec_path + "-export";
            if (!boost::filesystem::exists(export_rec_path)) {
                boost::filesystem::create_directories(export_rec_path);
            }
            Reconstruction rig_reconstruction;
            reconstruction->ConvertRigReconstruction(rig_reconstruction);
            rig_reconstruction.WriteReconstruction(export_rec_path,
                                                   options->cluster_mapper_options.mapper_options.write_binary_model);
        } else {
            reconstruction->WriteReconstruction(rec_path,
                                                options->cluster_mapper_options.mapper_options.write_binary_model);
        }
    }
}

int main(int argc, char *argv[]) {
    using namespace sensemap;

    PrintHeading("SenseMap.  Copyright(c) 2019, SenseTime Group.");
    PrintHeading(std::string("Version: sfm-mapping-") + __VERSION__);
    Timer timer;
    timer.Start();

    std::string configuration_file_path = std::string(argv[1]);
    Configurator param;
    param.Load(configuration_file_path.c_str());

    std::string workspace_path = param.GetArgument("workspace_path", "");
    CHECK(!workspace_path.empty()) << "workspace path empty";

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

    feature_data_container.reset();
    std::string mapper_method = param.GetArgument("mapper_method", "incremental");

    ClusterMapper(scene_graph_container, reconstruction_manager, param);

    std::cout << StringPrintf("Reconstruct in %.3fs", timer.ElapsedSeconds()) << std::endl;
    PrintReconSummary(workspace_path + "/statistic.txt", scene_graph_container->NumImages(), reconstruction_manager);

    fclose(fs);
    return 0;
}
