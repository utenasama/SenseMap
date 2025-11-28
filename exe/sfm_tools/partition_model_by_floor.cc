// Copyright (c) 2021, SenseTime Group.
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

using namespace sensemap;

void LoadAlignParam(const std::string align_param_path, std::vector<Eigen::Vector4d>& planes) {
    std::ifstream file(align_param_path.c_str());
	CHECK(file.is_open()) << align_param_path;

	std::string line;
	std::string item;

    int floor_number = -1;
	while (std::getline(file, line)) {
        // Skip Empty Line
        if (line.empty()) {
			continue;
		}

        // Load Floor Number
        if (line.find("# Number of floor") != std::string::npos) {
            std::getline(file, line);
            floor_number = std::stoi(line);
            std::cout << " Floor Num: " << floor_number << std::endl;
            planes.resize(floor_number);
        }

        // Load Plane Equations
        if (line.find("# Plane Equation: ax + by + cz + d = 0") != std::string::npos) {
            if (floor_number == -1) {
                std::cout << "Error: Align Param File Load Error, Please Check the file content ... " << std::endl;
                std::cout << " File Path: " << align_param_path << std::endl;
                exit(-1); 
            }

            for (int cur_floor_number = 0; cur_floor_number < floor_number; cur_floor_number++) {
                std::getline(file, line);
                
                std::stringstream line_stream(line);

                // a
                std::getline(line_stream, item, ' ');
                planes[cur_floor_number][0] = std::stod(item);

                // b
                std::getline(line_stream, item, ' ');
                planes[cur_floor_number][1] = std::stod(item);

                // c
                std::getline(line_stream, item, ' ');
                planes[cur_floor_number][2] = std::stod(item);

                // d
                std::getline(line_stream, item, ' ');
                planes[cur_floor_number][3] = std::stod(item);
                std::cout << "  " << cur_floor_number  << " : " 
                    << planes[cur_floor_number][0] << " , " 
                    << planes[cur_floor_number][1] << " , "
                    << planes[cur_floor_number][2] << " , "
                    << planes[cur_floor_number][3]<< std::endl;
            }
        }
	}
}

bool IsCameraBeyondPlane(const Eigen::Vector3d& camera_pose, const Eigen::Vector4d& plane) {
    double value = plane[0] * camera_pose[0] + plane[1] * camera_pose[1] + plane[2] * camera_pose[2] + plane[3];
    return value > 0;
}

void DivideReconstructionByPlanes(std::shared_ptr<Reconstruction> reconstruction,
                                  const std::vector<Eigen::Vector4d>& planes,
                                  std::unordered_map<int, std::unordered_set<image_t>>& floor_image_map) {
    // 
    // Inverted Order add image
    std::unordered_set<image_t> added_images;
    std::vector<image_t> registed_image_ids = reconstruction->RegisterImageIds();
    for (int cur_floor_number = planes.size(); cur_floor_number > 0; cur_floor_number--) {
        const Eigen::Vector4d& cur_plane = planes[cur_floor_number-1];

        for (const image_t image_id : registed_image_ids) {
            // Skip added images
            if (added_images.count(image_id)){
                continue;
            }

            if (IsCameraBeyondPlane(reconstruction->Image(image_id).ProjectionCenter(), cur_plane)){
                floor_image_map[cur_floor_number-1].insert(image_id);
                added_images.insert(image_id);
            }
        }
    }
}

void ObtainSubFeatureDataContainer(std::shared_ptr<FeatureDataContainer> feature_data_container,
                                   std::shared_ptr<FeatureDataContainer> sub_feature_data_container,
                                   const std::unordered_set<image_t>& floor_images) {
    for (const image_t& image_id : floor_images) {
        // Load feature data for current image
        const auto image = feature_data_container->GetImage(image_id);
        const auto keypoints = feature_data_container->GetKeypoints(image_id);
        const auto descriptors = feature_data_container->GetDescriptors(image_id);

        // Create new FeatureDataPtr
        FeatureDataPtr feature_data_ptr = std::make_shared<FeatureData>();

        feature_data_ptr->image = std::move(image);   
        feature_data_ptr->keypoints = std::move(keypoints);
        feature_data_ptr->descriptors = std::move(descriptors);
        
        sub_feature_data_container->emplace(feature_data_ptr->image.ImageId(), feature_data_ptr);
    }
     
}

void ObtainSubSceneGraphContainer(std::shared_ptr<SceneGraphContainer> scene_graph_container,
                                  std::shared_ptr<SceneGraphContainer> sub_scene_graph_container,
                                  const std::unordered_set<image_t>& floor_images) {

    std::shared_ptr<class CorrespondenceGraph> sub_correspondence_graph = sub_scene_graph_container->CorrespondenceGraph();
    std::shared_ptr<class CorrespondenceGraph> correspondence_graph = scene_graph_container->CorrespondenceGraph();

    for (const auto& image_id : floor_images) {
        if (!correspondence_graph->ExistsImage(image_id)) {
            continue;
        }
        const auto& image = correspondence_graph->Image(image_id);
        sub_correspondence_graph->AddImage(image_id, image);
    }

    const auto& image_pairs = correspondence_graph->ImagePairs();
    for (auto image_pair : image_pairs) {
        struct CorrespondenceGraph::ImagePair pair;
        image_t image_id1, image_id2;
        utility::PairIdToImagePair(image_pair.first, &image_id1, &image_id2);
        if (floor_images.count(image_id1) && floor_images.count(image_id2)) {
            const FeatureMatches &feature_matches = correspondence_graph->FindCorrespondencesBetweenImages(image_id1, image_id2);
            pair.num_correspondences = image_pair.second.num_correspondences;
            pair.image_id1 = image_id1;
            pair.image_id2 = image_id2;
            sub_correspondence_graph->AddCorrespondences(image_id1, image_id2, pair);
        }
    }
    sub_correspondence_graph->Finalize();
}

void ObtainSubReconstruction(std::shared_ptr<Reconstruction> reconstruction,
                             std::shared_ptr<Reconstruction> sub_reconstruction,
                             const std::unordered_set<image_t>& floor_images) {
    std::unordered_set<mappoint_t> add_mappoint_set;
    for (const image_t& image_id : floor_images) {
        if (!reconstruction->ExistsImage(image_id)) {
            std::cout << " Warning: Reconstruction not contained image id" << std::endl;
            continue;
        }

        Image image = reconstruction->Image(image_id);
        const Camera& camera = reconstruction->Camera(image.CameraId());

        // 1. Add Camera
        if (!sub_reconstruction->ExistsCamera(image.CameraId())) {
            sub_reconstruction->AddCamera(camera);
        }

        // 2. Add Image
        if (!sub_reconstruction->ExistsImage(image_id)) {
            // Go throught all the point2d with point3d
            for (point2D_t i = 0; i < image.Points2D().size(); i++){
                class Point2D& point2d = image.Point2D(i); 
                if (point2d.HasMapPoint()){
                    add_mappoint_set.insert(point2d.MapPointId());
                    image.ResetMapPointForPoint2D(i);
                }
            }
            image.SetRegistered(false); 
            sub_reconstruction->AddImage(image);
            sub_reconstruction->RegisterImage(image.ImageId());
        }
    }

    // 3. Add Mappoint
    for (const mappoint_t& mappoint_id : add_mappoint_set) {
        class MapPoint new_mappoint;

        // Get old mappoint
        const auto old_mappoint = reconstruction->MapPoint(mappoint_id);

        // Use the old mappoint position
        new_mappoint.SetXYZ(old_mappoint.XYZ());
        new_mappoint.SetColor(old_mappoint.Color());
        new_mappoint.SetError(old_mappoint.Error());

        int observation_num = old_mappoint.Track().Elements().size();
        // Update the old mappoint track with new image id and point2d id
        class Track new_track;
        for (int j = 0; j < observation_num; j++) {
            int frame_id = old_mappoint.Track().Elements()[j].image_id;
            int point_idx = old_mappoint.Track().Elements()[j].point2D_idx;

            if (!sub_reconstruction->ExistsImage(frame_id)) {
                continue;
            }
            new_track.AddElement(frame_id, point_idx);
        }

        // Check track size
        if(new_track.Length() < 2){
            continue;
        }

        new_mappoint.SetTrack(new_track);

        // Update reconstruction
        sub_reconstruction->AddMapPointWithError(new_mappoint.XYZ(), std::move(new_mappoint.Track()), new_mappoint.Color(),
                                                    new_mappoint.Error());
    }

    std::cout << " Regist sub image : " << sub_reconstruction->RegisterImageIds().size() << std::endl;
}


int main(int argc, char* argv[]) {

    PrintHeading("SenseMap.  Copyright(c) 2019, SenseTime Group.");
    PrintHeading("Version: partition-export-model-by-floor-1.6.7");
    
    if (argc < 4) {
        std::cout
            << "Usage: partition_export_model_by_floor 1.multi_floor_sfm_export_path 2.output_sfm_exports_path 3.align_param_path "
            << std::endl;
        return 1;
    }

    std::string workspace_path = std::string(argv[1]);
    std::string output_workspace_path = std::string(argv[2]); 
    std::string align_param_path = std::string(argv[3]);

    // Load align param 
    std::vector<Eigen::Vector4d> planes;
    LoadAlignParam(align_param_path, planes);

    if (planes.empty()){
        std::cout << "Error: Empty Plane in align param file: " << align_param_path << std::endl;
        exit(-1); 
    }

    // Load Reconstruction
    auto reconstruction = std::make_shared<Reconstruction>();
    // Note: This tool is only used for perspective camera model reconstruction
    std::cout << "\nLoad Reconstruction ... " << std::endl;
    reconstruction->ReadReconstruction(workspace_path, false);
    std::cout << "Finished Load Reconstruction \n" << std::endl;

    std::cout << "Reconstruction Image Number = " << reconstruction->RegisterImageIds().size() << std::endl;
    // Divide Reconstruction
    std::unordered_map<int, std::unordered_set<image_t>> floor_image_map;
    DivideReconstructionByPlanes(reconstruction, planes, floor_image_map);

    std::cout << "Divide Floor number: " <<  floor_image_map.size() << std::endl;
    for (const auto& floor_image : floor_image_map) {
        std::cout << "  floor number: " << floor_image.first << "  image number:  " << floor_image.second.size() << std::endl;
    }

    // According to Divided Image ids to Divide other data
    // Load Feature 
    auto feature_data_container = std::make_shared<FeatureDataContainer>();
    auto scene_graph_container = std::make_shared<SceneGraphContainer>();

    std::cout << "\nLoad Feature Binary ..." << std::endl;

    // Load original feature
    feature_data_container->ReadCamerasBinaryData(JoinPaths(workspace_path, "/cameras.bin"), false);
    feature_data_container->ReadImagesBinaryData(workspace_path + "/features.bin");

    std::cout << "Finished Load Feature Binary\n" << std::endl;

    // Load Scene Graph
    std::cout << "\nLoad Scene Graph Binary ..." << std::endl;
    scene_graph_container->ReadSceneGraphBinaryData(workspace_path + "/scene_graph.bin");
    std::cout << "Finished Load Scene Graph Binary\n" << std::endl;

    for (const auto& floor_image_pair : floor_image_map) {
        std::cout << "\n\nDivide floor number: " << floor_image_pair.first << std::endl; 
        auto sub_feature_data_container = std::make_shared<FeatureDataContainer>();
        auto sub_reconstruction = std::make_shared<Reconstruction>();
        auto sub_scene_graph_container = std::make_shared<SceneGraphContainer>();
        
        // 1. Divide Features.bin
        ObtainSubFeatureDataContainer(feature_data_container, sub_feature_data_container, floor_image_pair.second);

        // 2. Divide Scene Graph.bin
        ObtainSubSceneGraphContainer(scene_graph_container, sub_scene_graph_container, floor_image_pair.second);

        // 3. Divide Reconstruction
        ObtainSubReconstruction(reconstruction, sub_reconstruction, floor_image_pair.second);

        // Save Data
        if (boost::filesystem::exists(output_workspace_path+"/"+std::to_string(floor_image_pair.first))) {
            boost::filesystem::remove_all(output_workspace_path+"/"+std::to_string(floor_image_pair.first));
        }
        boost::filesystem::create_directories(output_workspace_path+"/"+std::to_string(floor_image_pair.first));
        // Write Feature
        sub_feature_data_container->WriteImagesBinaryData(output_workspace_path+"/"+std::to_string(floor_image_pair.first) + "/features.bin",true);
        // Write Reconstruction
        sub_reconstruction->WriteReconstruction(output_workspace_path+"/"+std::to_string(floor_image_pair.first));
        // Write Scene Graph
        sub_scene_graph_container->WriteSceneGraphBinaryData(output_workspace_path+"/"+std::to_string(floor_image_pair.first) + "/scene_graph.bin");
        std::cout << "\n";
    }
}