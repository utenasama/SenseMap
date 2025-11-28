// Copyright (c) 2019, SenseTime Group.
// All rights reserved.

#include <boost/filesystem/path.hpp>
#include <opencv2/opencv.hpp>

#include "base/camera.h"
#include "base/image.h"
#include "base/image_reader.h"
#include "base/reconstruction_manager.h"
#include "graph/correspondence_graph.h"
#include "util/misc.h"
#include "util/sparsification/sampler-base.h"
#include "util/sparsification/sampler-factory.h"

#include "cluster/fast_community.h"
#include "container/feature_data_container.h"
#include "feature/extraction.h"
#include "feature/feature_matcher.h"
#include "feature/utils.h"

#include <dirent.h>
#include <sys/stat.h>

using namespace sensemap;

std::string image_path;
std::string workspace_path;
std::string input_path;
std::string output_path;

bool dirExists(const std::string &dirName_in) { return access(dirName_in.c_str(), F_OK) == 0; }

// Remove all the point2d which do not has mappoint
void RemoveUselessPoint2D(std::shared_ptr<FeatureDataContainer> feature_data_in,
                          std::shared_ptr<FeatureDataContainer> feature_data_out,
                          std::shared_ptr<SceneGraphContainer> scene_graph_in,
                          std::shared_ptr<SceneGraphContainer> scene_graph_out,
                          std::shared_ptr<Reconstruction> reconstruction_in,
                          std::shared_ptr<Reconstruction> reconstruction_out) {
    std::unordered_map<image_t, std::unordered_map<point2D_t, point2D_t>> update_point2d_map;

    ///////////////////////////////////////////////////////////////////////////////////////
    // 1. Set camera bin
    ///////////////////////////////////////////////////////////////////////////////////////
    std::cout << "1. Set camera bin ... " << std::endl;
    for (const auto& cur_camera : reconstruction_in->Cameras()) {
        reconstruction_out->AddCamera(cur_camera.second);
    }

    ///////////////////////////////////////////////////////////////////////////////////////
    // 2. Update feature, remove all the unresgisted image and point2d
    ///////////////////////////////////////////////////////////////////////////////////////
    // Reset feature_out
    std::cout << "2. Update feature, remove all the unresgisted image and point2d ... " << std::endl;
    size_t bad_image_counter = 0;
    size_t bad_point_2d_counter = 0;

    // Get all the image id for feature in
    std::vector<image_t> image_ids_input = feature_data_in->GetImageIds();
    for (const auto &image_id_in : image_ids_input) {
        if (!reconstruction_in->ExistsImage(image_id_in)) {
            // std::cout << " Image not exist in reconstruction ... " << std::endl;
            bad_image_counter++;
            continue;
        }

        // Check the image id is registed or not
        if (!reconstruction_in->IsImageRegistered(image_id_in)) {
            // std::cout << " Image not registed ... " << std::endl;
            bad_image_counter++;
            continue;
        }
        // Get current image
        const auto cur_image = reconstruction_in->Image(image_id_in);
        const auto old_point2ds = cur_image.Points2D();

        // Set old Keypoint and descriptor
        const auto &old_keypoints = feature_data_in->GetKeypoints(image_id_in);
        const auto &old_descriptors = feature_data_in->GetDescriptors(image_id_in);

        // Create new Keypoint and descriptor

        // Create new FeatureDataPtr
        FeatureDataPtr feature_data_ptr = std::make_shared<FeatureData>();;

        // FIXME: Copy the old image data
        auto cur_old_image = feature_data_in->GetImage(image_id_in);
        feature_data_ptr->image.SetImageId(cur_old_image.ImageId());
        feature_data_ptr->image.SetCameraId(cur_old_image.CameraId());  // Only one camera
        feature_data_ptr->image.SetName(cur_old_image.Name());
        feature_data_ptr->image.SetLabelId(cur_old_image.LabelId());
        feature_data_ptr->descriptors.resize(old_keypoints.size(), old_descriptors.cols());

        point2D_t new_point_id = 0;
        for (point2D_t point_id = 0; point_id < old_keypoints.size(); point_id++) {
            // Skip all the point2d do not has mappoint
            if (!old_point2ds[point_id].HasMapPoint()) {
                continue;
            }
            feature_data_ptr->keypoints.emplace_back(old_keypoints[point_id]);
            feature_data_ptr->descriptors.row(feature_data_ptr->keypoints.size() - 1) = old_descriptors.row(point_id);

            update_point2d_map[image_id_in][point_id] = new_point_id;

            new_point_id++;
        }

        if (feature_data_ptr->keypoints.empty()) {
            // std::cout << "Find image has not mappoint..." << std::endl;
            bad_image_counter++;
            update_point2d_map.erase(image_id_in);
            continue;
        }
        feature_data_out->emplace(feature_data_ptr->image.ImageId(), feature_data_ptr);
    }

    std::cout << " Image number = " << feature_data_out->GetImageIds().size()
              << " ,  bad image number = " << bad_image_counter << std::endl;

    // ///////////////////////////////////////////////////////////////////////////////////////
    // // 3. Update Scene Graph
    // ///////////////////////////////////////////////////////////////////////////////////////
    // std::cout << "3. Update Scene graph data ... " << std::endl;

    // const auto scene_graph_images = scene_graph_in->Images();

    // for (const auto scene_graph_image : scene_graph_images) {
    //     // Check image exist in the feature
    //     if (!feature_data_out->ExistImage(scene_graph_image.first)) {
    //         continue;
    //     }
    //     scene_graph_out->AddImage(scene_graph_image.second);
    // }

    // const auto &correspondence_graph = scene_graph_in->CorrespondenceGraph();

    // const auto &all_image_pairs = correspondence_graph->ImagePairs();

    // for (const auto &image_pair : all_image_pairs) {
    //     const auto image_pair_id = image_pair.first;
    //     const auto &image_pair_struct = image_pair.second;

    //     // Get two image ids
    //     image_t image_id_1, image_id_2;
    //     utility::PairIdToImagePair(image_pair_id, &image_id_1, &image_id_2);

    //     if (!scene_graph_out->ExistsImage(image_id_1) || !scene_graph_out->ExistsImage(image_id_2)) {
    //         continue;
    //     }

    // }

    ///////////////////////////////////////////////////////////////////////////////////////
    // 4. Update Point2d
    ///////////////////////////////////////////////////////////////////////////////////////
    std::cout << "4. Update Point2d ... " << std::endl;
    const auto &image_ids = reconstruction_in->RegisterImageIds();
    for (const auto &image_id : image_ids) {
        const auto cur_image = reconstruction_in->Image(image_id);
        // Get all the old point2ds
        const auto old_point2ds = cur_image.Points2D();
        std::vector<class Point2D> new_point2Ds;  // Store new point2d

        for (point2D_t point_id = 0; point_id < old_point2ds.size(); point_id++) {
            if (!old_point2ds[point_id].HasMapPoint()) {
                continue;
            }

            class Point2D new_point2D;
            new_point2D.SetXY(old_point2ds[point_id].XY());
            auto point2d_index = update_point2d_map[image_id][point_id];
            new_point2Ds.emplace_back(new_point2D);
        }

        if (new_point2Ds.empty()) {
            continue;
        }

        Image new_image;
        // Update image id
        new_image.SetImageId(cur_image.ImageId());
        new_image.SetCameraId(cur_image.CameraId());

        // Update the camera rotation
        new_image.SetQvec(cur_image.Qvec());
        new_image.SetTvec(cur_image.Tvec());

        new_image.SetName(cur_image.Name());
        new_image.SetPoints2D(new_point2Ds);

        // Update reconstruction
        reconstruction_out->AddImage(new_image);
        reconstruction_out->RegisterImage(new_image.ImageId());
    }

    ///////////////////////////////////////////////////////////////////////////////////////
    // 5. Update 3d point track id
    ///////////////////////////////////////////////////////////////////////////////////////
    std::cout << "5. Update 3d point track id ... " << std::endl;
    const auto &mappoint_ids = reconstruction_in->MapPointIds();
    for (const auto &mappoint_id : mappoint_ids) {
        class MapPoint new_mappoint;
        // Get old mappoint
        const auto old_mappoint = reconstruction_in->MapPoint(mappoint_id);

        // Use the old mappoint position
        new_mappoint.SetXYZ(old_mappoint.XYZ());
        new_mappoint.SetColor(old_mappoint.Color());
        new_mappoint.SetError(old_mappoint.Error());

        // Update the old mappoint track with new image id and point2d id
        class Track new_track;
        for (const auto &track_el : old_mappoint.Track().Elements()) {
            if (!update_point2d_map.count(track_el.image_id)) {
                continue;
            }

            if (!update_point2d_map[track_el.image_id].count(track_el.point2D_idx)) {
                continue;
            }
            auto new_point2d_id = update_point2d_map[track_el.image_id][track_el.point2D_idx];

            // std::cout << "Has mappoint ? " <<
            // update_reconstruction->Image(new_image_to_point.first).Point2D(new_image_to_point.second).HasMapPoint()
            // << " image id = " <<new_image_to_point.first << " , point id = " << new_image_to_point.second <<
            // std::endl;
            new_track.AddElement(track_el.image_id, new_point2d_id);
        }
        new_mappoint.SetTrack(new_track);

        reconstruction_out->AddMapPointWithError(new_mappoint.XYZ(), std::move(new_mappoint.Track()), new_mappoint.Color(),
                                                 new_mappoint.Error());
    }
}

int main(int argc, char *argv[]) {
    workspace_path = std::string(argv[1]);

    PrintHeading("SenseMap.  Copyright(c) 2019, SenseTime Group.");

    auto feature_data_container = std::make_shared<FeatureDataContainer>();
    auto scene_graph_container = std::make_shared<SceneGraphContainer>();
    auto reconstruction = std::make_shared<Reconstruction>();

    // Load Feature
    feature_data_container->ReadCamerasBinaryData(JoinPaths(workspace_path, "/cameras.bin"));
    feature_data_container->ReadImagesBinaryData(JoinPaths(workspace_path, "/features.bin"));

    // // Load Correspondence
    // scene_graph_container->ReadSceneGraphBinaryData(JoinPaths(workspace_path, "/scene_graph.bin"));

    // EIGEN_STL_UMAP(image_t, class Image) &images = scene_graph_container->Images();
    // EIGEN_STL_UMAP(camera_t, class Camera) &cameras = scene_graph_container->Cameras();

    // std::vector<image_t> image_ids = feature_data_container->GetImageIds();

    // for (const auto image_id : image_ids) {
    //     const Image &image = feature_data_container->GetImage(image_id);
    //     if (!scene_graph_container->CorrespondenceGraph()->ExistsImage(image_id)) {
    //         continue;
    //     }

    //     images[image_id] = image;

    //     const FeatureKeypoints &keypoints = feature_data_container->GetKeypoints(image_id);
    //     images[image_id].SetPoints2D(keypoints);

    //     const Camera &camera = feature_data_container->GetCamera(image.CameraId());

    //     if (!scene_graph_container->ExistsCamera(image.CameraId())) {
    //         cameras[image.CameraId()] = camera;
    //     }

    //     if (scene_graph_container->CorrespondenceGraph()->ExistsImage(image_id)) {
    //         images[image_id].SetNumObservations(
    //             scene_graph_container->CorrespondenceGraph()->NumObservationsForImage(image_id));
    //         images[image_id].SetNumCorrespondences(
    //             scene_graph_container->CorrespondenceGraph()->NumCorrespondencesForImage(image_id));
    //     } else {
    //         std::cout << "Do not contain ImageId = " << image_id << ", in the correspondence graph." << std::endl;
    //     }
    // }

    // scene_graph_container->CorrespondenceGraph()->Finalize();

    std::cout << "Load Reconstruction ... " << std::endl;
    reconstruction->ReadReconstruction(JoinPaths(workspace_path, "/0/"));

    auto feature_data_container_out = std::make_shared<FeatureDataContainer>();
    auto scene_graph_container_out = std::make_shared<SceneGraphContainer>();
    auto reconstruction_out = std::make_shared<Reconstruction>();
    // Remove useless point 2d from reconstruction, scene graph  and feature container
    RemoveUselessPoint2D(feature_data_container, feature_data_container_out, scene_graph_container,
                         scene_graph_container_out, reconstruction, reconstruction_out);


    if (boost::filesystem::exists(workspace_path + "/simplified/")) {
        boost::filesystem::remove_all(workspace_path + "/simplified/");
    }
    boost::filesystem::create_directories(workspace_path + "/simplified/");
    reconstruction_out->WriteReconstruction(JoinPaths(workspace_path, "/simplified/"), true);
    feature_data_container_out->WriteImagesBinaryData(JoinPaths(workspace_path, "/simplified/features.bin"));
    
    return 0;
}