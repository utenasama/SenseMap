// Copyright (c) 2022, SenseTime Group.
// All rights reserved.

#include <dirent.h>
#include <sys/stat.h>
#include <malloc.h>

#include <boost/filesystem/path.hpp>
#include <gflags/gflags.h>
#include <unordered_set>

#include "../Configurator_yaml.h"
#include "../camera_rig_params.h"
#include "../option_parsing.h"
#include "../system_io.h"
#include "base/version.h"
#include "base/reconstruction_manager.h"
#include "base/common.h"
#include "base/similarity_transform.h"
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
#include "util/proc.h"
#include "util/exception_handler.h"
#include "optim/cluster_merge/cluster_merge_optimizer.h"
#include "optim/cluster_motion_averager.h"
#include "estimators/reconstruction_aligner.h"

// #define DEBUG_INFO

#define MAX_PRECISION std::setprecision(std::numeric_limits<double>::digits10 + 1)

// DEFINE_string(rec_path1, "", "reconstruction path for the first scene, e.g. scene1/sfm-workspace/0");
// DEFINE_string(rec_path2, "", "reconstruction path for the second scene, e.g. scene2/sfm-workspace/0");
DEFINE_string(config1, "", "configuration for the first scene, e.g. scene1/sfm.yaml");
DEFINE_string(rectID1, "", "first reconstruction id");
DEFINE_string(config2, "", "configuration for the second scene, e.g. scene2/sfm.yaml");
DEFINE_string(rectID2, "", "second reconstruction ");
using namespace sensemap;
bool gps_prior_align = false;
std::string align_path;

struct GpsInfo{
    Eigen::Matrix3x4d ned_ecef = Eigen::Matrix3x4d::Zero();
    std::vector<double> gps_origin;

    bool isvalid(){
        return !((ned_ecef == Eigen::Matrix3x4d::Zero()) || (gps_origin.size() != 3));
    }
};

bool LoadFeatures(FeatureDataContainer &feature_data_container, 
                  GpsInfo& gps_info,
                  Configurator &param,
                  std::string workspace_path = "") {
    if (workspace_path.empty()){
        workspace_path = param.GetArgument("workspace_path", "");
    }
    CHECK(!workspace_path.empty());

    OptionParser option_parser;
    ImageReaderOptions reader_options;
    option_parser.GetImageReaderOptions(reader_options, param);

    bool camera_rig = (reader_options.num_local_cameras > 1);
    bool with_depth = reader_options.with_depth;


    bool use_gps_prior = param.GetArgument("use_gps_prior", 0);

    bool exist_feature_file = false;
    if (boost::filesystem::exists(JoinPaths(workspace_path, "/cameras.bin")) &&
        boost::filesystem::exists(JoinPaths(workspace_path, "/features.bin"))) {
        if (boost::filesystem::exists(JoinPaths(workspace_path, "/local_cameras.bin"))) {
            feature_data_container.ReadCamerasBinaryData(JoinPaths(workspace_path, "/cameras.bin"), false);
            feature_data_container.ReadLocalCamerasBinaryData(JoinPaths(workspace_path, "/local_cameras.bin"));
        } else {
            feature_data_container.ReadCamerasBinaryData(JoinPaths(workspace_path, "/cameras.bin"), camera_rig);
        }
        feature_data_container.ReadImagesBinaryData(JoinPaths(workspace_path, "/features.bin"));
        exist_feature_file = true;
    }

    // If the camera model is Spherical
    if (exist_feature_file) {
        if (boost::filesystem::exists(JoinPaths(workspace_path, "/sub_panorama.bin"))) {
            feature_data_container.ReadSubPanoramaBinaryData(JoinPaths(workspace_path, "/sub_panorama.bin"));
        } else {
            // FIXME:
            std::cout << " Warning! Existing feature data do not contain sub_panorama data" << std::endl;
        }
    }

    if (exist_feature_file) {
        if (boost::filesystem::exists(JoinPaths(workspace_path, "/piece_indices.bin"))) {
            feature_data_container.ReadPieceIndicesBinaryData(JoinPaths(workspace_path, "/piece_indices.bin"));
        } else {
            // FIXME:
            std::cout << " Warning! Existing feature data do not contain piece_indices data" << std::endl;
        }
    }

    // Check the AprilTag detection file exist or not
    if (exist_feature_file && static_cast<bool>(param.GetArgument("detect_apriltag", 0))) {
        if (boost::filesystem::exists(JoinPaths(workspace_path, "/apriltags.bin"))) {
            feature_data_container.ReadAprilTagBinaryData(JoinPaths(workspace_path, "/apriltags.bin"));
        } else {
            // FIXME:
            std::cout << " Warning! Existing feature data do not contain AprilTags data" << std::endl;
        }
    }

    // Check the GPS file exist or not.
    if (exist_feature_file && use_gps_prior) {
        if (boost::filesystem::exists(JoinPaths(workspace_path, "/gps.bin"))) {
            feature_data_container.ReadGPSBinaryData(JoinPaths(workspace_path, "/gps.bin"));
        }
        if (boost::filesystem::exists(JoinPaths(workspace_path, "/gps_origin.txt"))){
            LoadGpsOrigin(JoinPaths(workspace_path, "/gps_origin.txt"), gps_info.gps_origin);
        }
        if (boost::filesystem::exists(JoinPaths(workspace_path, "/ned_to_ecef.txt"))){
            std::ifstream fin(JoinPaths(workspace_path, "ned_to_ecef.txt"), std::ifstream::in);
            if (fin.is_open()) {
                fin >> gps_info.ned_ecef(0, 0) >> gps_info.ned_ecef(0, 1) >> gps_info.ned_ecef(0, 2) >> gps_info.ned_ecef(0, 3);
                fin >> gps_info.ned_ecef(1, 0) >> gps_info.ned_ecef(1, 1) >> gps_info.ned_ecef(1, 2) >> gps_info.ned_ecef(1, 3);
                fin >> gps_info.ned_ecef(2, 0) >> gps_info.ned_ecef(2, 1) >> gps_info.ned_ecef(2, 2) >> gps_info.ned_ecef(2, 3);
            }
            fin.close();
        }
        if (gps_info.isvalid()){
            std::cout << "gps origin: " << gps_info.gps_origin[0] << ", " 
                << gps_info.gps_origin[1] << ", " << gps_info.gps_origin[2] << std::endl;
            std::cout << "ned to ecef: \n" << gps_info.ned_ecef << std::endl;
        }
    }

    if (exist_feature_file) {
        return true;
    } else {
        return false;
    }
}

bool LoadMatch(FeatureDataContainer &feature_data_container,
               SceneGraphContainer &scene_graph_container,
               Configurator &param,
               std::string workspace_path = ""){
    if (workspace_path.empty()){
        workspace_path = param.GetArgument("workspace_path", "");
    }
    CHECK(!workspace_path.empty());

    // load match data
    scene_graph_container.ReadSceneGraphBinaryData(JoinPaths(workspace_path, "/scene_graph.bin"));
    if (boost::filesystem::exists(JoinPaths(workspace_path, "/two_view_geometry.bin"))) {
        scene_graph_container.ReadImagePairsBinaryData(JoinPaths(workspace_path, "/two_view_geometry.bin"));
    }
    if (boost::filesystem::exists(JoinPaths(workspace_path, "/strong_loops.bin"))) {
        std::cout<<"read strong_loops file "<<std::endl;
        scene_graph_container.ReadStrongLoopsBinaryData(JoinPaths(workspace_path, "/strong_loops.bin"));
    }
    if (boost::filesystem::exists(JoinPaths(workspace_path, "/loop_pairs.bin"))) {
        std::cout<<"read loop_pairs file "<<std::endl;
        scene_graph_container.ReadLoopPairsInfoBinaryData(JoinPaths(workspace_path, "/loop_pairs.bin"));
    }
    if (boost::filesystem::exists(JoinPaths(workspace_path, "/normal_pairs.bin"))) {
        std::cout<<"read normal_pairs file "<<std::endl;
        scene_graph_container.ReadNormalPairsBinaryData(JoinPaths(workspace_path, "/normal_pairs.bin"));
    }

    EIGEN_STL_UMAP(image_t, class Image) &images = scene_graph_container.Images();
    EIGEN_STL_UMAP(camera_t, class Camera) &cameras = scene_graph_container.Cameras();

    std::vector<image_t> image_ids = feature_data_container.GetImageIds();

    for (const auto image_id : image_ids) {
        const Image &image = feature_data_container.GetImage(image_id);
        if (!scene_graph_container.CorrespondenceGraph()->ExistsImage(image_id)) {
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

        if (!scene_graph_container.ExistsCamera(image.CameraId())) {
            cameras[image.CameraId()] = camera;
        }

        if (scene_graph_container.CorrespondenceGraph()->ExistsImage(image_id)) {
            images[image_id].SetNumObservations(scene_graph_container.CorrespondenceGraph()->NumObservationsForImage(image_id));
            images[image_id].SetNumCorrespondences(
                scene_graph_container.CorrespondenceGraph()->NumCorrespondencesForImage(image_id));
        } else {
            std::cout << "Do not contain ImageId = " << image_id << ", in the correspondence graph." << std::endl;
        }
    }
    scene_graph_container.CorrespondenceGraph()->Finalize();
    return true;

}
void MergeFeatureAndMatches(FeatureDataContainer &feature_data_container1,
                            FeatureDataContainer &feature_data_container2,
                            GpsInfo & gps_info1,
                            GpsInfo & gps_info2,
                            SceneGraphContainer &scene_graph_container1,
                            SceneGraphContainer &scene_graph_container2,
                            FeatureDataContainer &feature_data_container,
                            GpsInfo &gps_info,
                            SceneGraphContainer &scene_graph_container,
                            std::unordered_map<image_t, image_t> &image_id_map,
                            std::unordered_map<camera_t, camera_t> &camera_id_map) {
    feature_data_container = feature_data_container1;
    gps_info = gps_info1;
    std::vector<image_t> image_ids = feature_data_container.GetImageIds();
    image_t max_image_id1 = 0;
    camera_t max_camera_id1 = 0;
    for (auto image_id : image_ids) {
        const auto &keypoints = feature_data_container.GetKeypoints(image_id);
        Image &image = feature_data_container.GetFeatureData().at(image_id)->image;
        image.SetPoints2D(keypoints);
        image.SetLabelId(0);

        max_image_id1 = std::max(max_image_id1, image_id);
        max_camera_id1 = std::max(feature_data_container.GetImage(image_id).CameraId(), max_camera_id1);
    }

    std::cout << "old_image info: " << std::endl;
    for (auto image_id : image_ids){
        const auto& image = feature_data_container.GetImage(image_id);
        std::cout << "\t" << image_id << ", " << image.Name() << std::endl;
    }
    std::cout << std::endl;

    std::cout << StringPrintf("Update Feature Container\n");
    //gps_trans
    SimilarityTransform3 tform;
    bool gps_ok = false;
    if (gps_info1.isvalid() && gps_info2.isvalid()){
        Eigen::Matrix4d trans_1_ecef = Eigen::Matrix4d::Identity();
        trans_1_ecef.topRows(3) = gps_info1.ned_ecef; 
        Eigen::Matrix4d trans_2_ecef = Eigen::Matrix4d::Identity();
        trans_2_ecef.topRows(3) = gps_info2.ned_ecef; 

        Eigen::Matrix4d trans_21t = trans_1_ecef.inverse() * trans_2_ecef;
        Eigen::Matrix3x4d trans_21 = trans_21t.topRows(3);
        std::cout << "trans_21: \n" << trans_21 << std::endl;

        tform = SimilarityTransform3(trans_21);
        gps_ok = true;
        gps_prior_align = true;
    }

    std::vector<image_t> new_image_ids;
    new_image_ids.reserve(feature_data_container2.GetImageIds().size());
    image_t new_image_id = max_image_id1 + 1;
    camera_t new_camera_id = max_camera_id1 + 1;
    FeatureDataPtrUmap feature_datas = feature_data_container2.GetFeatureData();
    auto image_ids_2 = feature_data_container2.GetImageIds();
    std::sort(image_ids_2.begin(), image_ids_2.end());
    for (image_t image_id : image_ids_2) {
        FeatureDataPtr feature_data_ptr = feature_datas.at(image_id);
        Image& new_image = feature_data_ptr->image;
        camera_t camera_id = new_image.CameraId();
        Camera new_camera = feature_data_container2.GetCamera(camera_id);

        const auto &keypoints = feature_data_container2.GetKeypoints(image_id);
        const auto &panorama_indices = feature_data_container2.GetPanoramaIndexs(image_id);
        std::vector<uint32_t> local_image_indices(keypoints.size());
        for(size_t i = 0; i<keypoints.size(); ++i){
            if (panorama_indices.size() == 0 && new_camera.NumLocalCameras() == 1) {
                local_image_indices[i] = new_image_id;
            } else{
			    local_image_indices[i] = panorama_indices[i].sub_image_id;
            }
		}
        new_image.SetLocalImageIndices(local_image_indices);

        if (camera_id_map.find(camera_id) == camera_id_map.end()) {
            camera_id_map[camera_id] = new_camera_id++;
        }

        new_image.SetImageId(new_image_id);
        new_image.SetCameraId(camera_id_map[camera_id]);
        new_image.SetPoints2D(feature_data_ptr->keypoints);
        new_image.SetLabelId(1);
        if (gps_ok){
            if (new_image.HasQvecPrior()){
                Eigen::Vector3d tvec = Eigen::Vector3d::Zero();
                tform.TransformPose(&new_image.QvecPrior(), &tvec);
            }
            if (new_image.HasTvecPrior()){
                tform.TransformPoint(&new_image.TvecPrior());
            }
        }

        new_camera.SetCameraId(new_image.CameraId());

        new_image_ids.push_back(new_image_id);
        image_id_map[image_id] = new_image_id++;

        feature_data_container.emplace(new_image.ImageId(), feature_data_ptr);
        feature_data_container.emplace(new_camera.CameraId(), std::make_shared<Camera>(new_camera));
        feature_data_container.emplace(new_image.Name(), new_image.ImageId());
        std::cout << "new image: " << new_image.Name() << ", " << new_image.ImageId() << std::endl;
    }

    std::cout << StringPrintf("Update SceneGraph Container\n");

    // Append feature data1 to scene_graph.
    auto correspondence_graph = scene_graph_container.CorrespondenceGraph();
    for (auto image_id : feature_data_container1.GetImageIds()) {
        const auto& image = feature_data_container.GetImage(image_id);
        if (!scene_graph_container.ExistsImage(image_id)) {
            scene_graph_container.AddImage(image);
        }
        
        const auto& camera = feature_data_container.GetCamera(image.CameraId());
        if (!scene_graph_container.ExistsCamera(image.CameraId())) {
            scene_graph_container.AddCamera(camera);
        }
    }

    auto correspondence_graph1 = scene_graph_container1.CorrespondenceGraph();
    const auto &image_pairs1 = correspondence_graph1->ImagePairs();
    for (auto image_pair : image_pairs1) {
        image_t image_id1, image_id2;
        utility::PairIdToImagePair(image_pair.first, &image_id1, &image_id2);
        auto feature_matches = correspondence_graph1->FindCorrespondencesBetweenImages(image_id1, image_id2);
        image_pair.second.two_view_geometry.inlier_matches = feature_matches;
        
        const class Image &image1 = scene_graph_container.Image(image_id1);
        const class Camera &camera1 = scene_graph_container.Camera(image1.CameraId());
        const class Image &image2 = scene_graph_container.Image(image_id2);
        const class Camera &camera2 = scene_graph_container.Camera(image2.CameraId());

        const bool remove_redundant = !(camera1.NumLocalCameras() > 2 && camera2.NumLocalCameras() > 2);

        correspondence_graph->AddCorrespondences(image_id1, image_id2, 
            image_pair.second.two_view_geometry, remove_redundant);
    }
    std::cout << "images in correspondence before finalize " << correspondence_graph->NumImages() << std::endl;
    correspondence_graph->Finalize();
    std::cout << "images in correspondence after finalize " << correspondence_graph->NumImages() << std::endl;

    for (const auto &image_id : feature_data_container1.GetImageIds()) {
        if (!correspondence_graph->ExistsImage(image_id)) {
            continue;
        }
        scene_graph_container.Image(image_id).SetNumObservations(correspondence_graph1->NumObservationsForImage(image_id));
        scene_graph_container.Image(image_id).SetNumCorrespondences(correspondence_graph1->NumCorrespondencesForImage(image_id));
    }

    // Append feature data2 to scene_graph.
    for (auto image_id : feature_data_container2.GetImageIds()) {
        image_t new_image_id = image_id_map.at(image_id);
        const auto& image = feature_data_container.GetImage(new_image_id);
        if (!scene_graph_container.ExistsImage(new_image_id)) {
            scene_graph_container.AddImage(image);
        }
        
        const auto& camera = feature_data_container.GetCamera(image.CameraId());
        if (!scene_graph_container.ExistsCamera(image.CameraId())) {
            scene_graph_container.AddCamera(camera);
        }
    }

    auto correspondence_graph2 = scene_graph_container2.CorrespondenceGraph();
    const auto &image_pairs2 = correspondence_graph2->ImagePairs();
    for (auto image_pair : image_pairs2) {
        image_t image_id1, image_id2;
        utility::PairIdToImagePair(image_pair.first, &image_id1, &image_id2);
        auto feature_matches = correspondence_graph2->FindCorrespondencesBetweenImages(image_id1, image_id2);
        image_pair.second.two_view_geometry.inlier_matches = feature_matches;
        
        image_t new_image_id1 = image_id_map.at(image_id1);
        image_t new_image_id2 = image_id_map.at(image_id2);
        const class Image &image1 = scene_graph_container.Image(new_image_id1);
        const class Camera &camera1 = scene_graph_container.Camera(image1.CameraId());
        const class Image &image2 = scene_graph_container.Image(new_image_id2);
        const class Camera &camera2 = scene_graph_container.Camera(image2.CameraId());

        const bool remove_redundant = !(camera1.NumLocalCameras() > 2 && camera2.NumLocalCameras() > 2);

        correspondence_graph->AddCorrespondences(new_image_id1, new_image_id2, 
            image_pair.second.two_view_geometry, remove_redundant);
    }
    std::cout << "images in correspondence before finalize " << correspondence_graph->NumImages() << std::endl;
    correspondence_graph->Finalize();
    std::cout << "images in correspondence after finalize " << correspondence_graph->NumImages() << std::endl;

    EIGEN_STL_UMAP(image_t, class Image) &images = scene_graph_container.Images();
    for (const auto &image_id : new_image_ids) {
        if (!correspondence_graph->ExistsImage(image_id)) {
            continue;
        }
        scene_graph_container.Image(image_id).SetNumObservations(correspondence_graph->NumObservationsForImage(image_id));
        scene_graph_container.Image(image_id).SetNumCorrespondences(correspondence_graph->NumCorrespondencesForImage(image_id));
    }
    std::cout << "scene graph: " << scene_graph_container.NumImages() << std::endl;
}

void MergeSeneGraph(FeatureDataContainer &feature_data_container,
                    SceneGraphContainer &scene_graph_container,
                    std::unordered_map<image_t, image_t>& image_id_map,
                    std::unordered_map<camera_t, camera_t>& camera_id_map,
                    Configurator& param1, Configurator& param2){

    std::string workspace_path1 = param1.GetArgument("workspace_path", "");
    std::string workspace_path2 = param2.GetArgument("workspace_path", "");
    GpsInfo gps_info;

    align_path = JoinPaths(GetParentDir(workspace_path2), "sfm-workspace-align");
    if (!boost::filesystem::exists(align_path)) {
        boost::filesystem::create_directories(align_path);
    }

    if (boost::filesystem::exists(JoinPaths(align_path, "/cameras.bin")) &&
        boost::filesystem::exists(JoinPaths(align_path, "/features.bin"))&& 
        boost::filesystem::exists(JoinPaths(align_path, "/scene_graph.bin"))){
        LoadFeatures(feature_data_container, gps_info, param2, align_path);

        {
            float mem;
            sensemap::GetAvailableMemory(mem);
            std::cout << "begin memory left : " << mem << std::endl;
            const auto image_ids = feature_data_container.GetImageIds();
            for (auto image_id : image_ids){
                feature_data_container.GetDescriptors(image_id).resize(0,0);
                feature_data_container.GetCompressedDescriptors(image_id).resize(0,0);
            }
            std::cout << malloc_trim(0) << std::endl;
            sensemap::GetAvailableMemory(mem);
            std::cout << "clear Descriptors memory left : " << mem << std::endl;
        }

        // reset id map
        {
            image_id_map.clear();
            camera_id_map.clear();

            FeatureDataContainer feature_data_container2;
            GpsInfo gps2;
            LoadFeatures(feature_data_container2, gps2, param2);

            const auto new_image_ids = feature_data_container.GetNewImageIds();
            const auto image2_ids = feature_data_container2.GetImageIds();
            CHECK_EQ(new_image_ids.size(), image2_ids.size());
            for (auto new_id : new_image_ids){
                const auto& new_image = feature_data_container.GetImage(new_id);
                const auto& image_name = new_image.Name();

                const auto image2_id = feature_data_container2.GetImageId(image_name);
                image_id_map[image2_id] = new_id;

                const auto camera2_id = feature_data_container2.GetImage(image2_id).CameraId();
                if (camera_id_map.find(camera2_id) == camera_id_map.end()){
                    camera_id_map[camera2_id] = new_image.CameraId();
                }
            }
        }

        LoadMatch(feature_data_container, scene_graph_container, param2, align_path);

        std::cout << "Load Features & Match... Done" << std::endl;
        return ;
    }

    {
        FeatureDataContainer feature_data_container1, feature_data_container2;
        GpsInfo gps1, gps2;
        LoadFeatures(feature_data_container1, gps1, param1);
        LoadFeatures(feature_data_container2, gps2, param2);

        SceneGraphContainer scene_graph_container1, scene_graph_container2;
        scene_graph_container1.ReadSceneGraphBinaryData(JoinPaths(workspace_path1, "/scene_graph.bin"));
        scene_graph_container2.ReadSceneGraphBinaryData(JoinPaths(workspace_path2, "/scene_graph.bin"));

        MergeFeatureAndMatches(feature_data_container1, feature_data_container2,
                            gps1, gps2, scene_graph_container1, scene_graph_container2,
                            feature_data_container, gps_info, scene_graph_container,
                            image_id_map, camera_id_map);
    }

    {
        feature_data_container.WriteImagesBinaryData(JoinPaths(align_path, "/features.bin"));
        feature_data_container.WriteCamerasBinaryData(JoinPaths(align_path, "/cameras.bin"));
        feature_data_container.WriteLocalCamerasBinaryData(JoinPaths(align_path, "/local_cameras.bin"));
        feature_data_container.WriteSubPanoramaBinaryData(JoinPaths(align_path, "/sub_panorama.bin"));
        feature_data_container.WritePieceIndicesBinaryData(JoinPaths(align_path, "/piece_indices.bin"));

        // Check the Arpiltag Detect Result
        if (feature_data_container.ExistAprilTagDetection()) {
            feature_data_container.WriteAprilTagBinaryData(JoinPaths(align_path, "/apriltags.bin"));
        } else {
            std::cout << "Warning: No Apriltag Detection has been found ... " << std::endl;
        }
        
        if (gps_info.isvalid()) {
            feature_data_container.WriteGPSBinaryData(JoinPaths(align_path, "/gps.bin"));

            std::ofstream file(JoinPaths(align_path, "/ned_to_ecef.txt"), std::ofstream::out);
            file << MAX_PRECISION << gps_info.ned_ecef(0, 0) << " " << gps_info.ned_ecef(0, 1) << " " 
                << gps_info.ned_ecef(0, 2) << " " << gps_info.ned_ecef(0, 3) << std::endl;
            file << MAX_PRECISION << gps_info.ned_ecef(1, 0) << " " << gps_info.ned_ecef(1, 1) << " " 
                << gps_info.ned_ecef(1, 2) << " " << gps_info.ned_ecef(1, 3) << std::endl;
            file << MAX_PRECISION << gps_info.ned_ecef(2, 0) << " " << gps_info.ned_ecef(2, 1) << " " 
                << gps_info.ned_ecef(2, 2) << " " << gps_info.ned_ecef(2, 3) << std::endl;
            file.close();

            std::ofstream file1(JoinPaths(align_path, "/gps_origin.txt"));
            file1 << MAX_PRECISION << gps_info.gps_origin.at(0) << " " 
                << gps_info.gps_origin.at(1) << " " 
                << gps_info.gps_origin.at(2) << std::endl;
            file1.close();
        }
    }

    {
        float mem;
        sensemap::GetAvailableMemory(mem);
        std::cout << "memory left : " << mem << std::endl;
        std::cout << malloc_trim(0) << std::endl;
        sensemap::GetAvailableMemory(mem);
        std::cout << "memory left : " << mem << std::endl;
    }

    std::cout << "images in scene graph: " << scene_graph_container.NumImages() << std::endl;
    std::cout << "images in correspondence graph: " << scene_graph_container.CorrespondenceGraph()->NumImages() << std::endl;

    {

        std::cout << "Matching between reconstructions." << std::endl;
        MatchDataContainer match_data;

        FeatureMatchingOptions options;
        OptionParser option_parser;
        option_parser.GetFeatureMatchingOptions(options, param1);
        options.delete_duplicated_images_ = false;

        options.method_ = FeatureMatchingOptions::MatchMethod::NONE;
        options.match_between_reconstructions_ = true;
        FeatureMatcher matcher(options, &feature_data_container, &match_data, &scene_graph_container);
        std::cout << "matching ....." << std::endl;
        matcher.Run();
        std::cout << "matching done" << std::endl;
        std::cout << "build graph" << std::endl;
        matcher.BuildSceneGraph();
        std::cout << "build graph done" << std::endl;
        {
            float mem;
            sensemap::GetAvailableMemory(mem);
            std::cout << "begin memory left : " << mem << std::endl;
            const auto image_ids = feature_data_container.GetImageIds();
            for (auto image_id : image_ids){
                feature_data_container.GetDescriptors(image_id).resize(0,0);
                feature_data_container.GetCompressedDescriptors(image_id).resize(0,0);
            }
            std::cout << malloc_trim(0) << std::endl;
            sensemap::GetAvailableMemory(mem);
            std::cout << "clear Descriptors memory left : " << mem << std::endl;
        }

        scene_graph_container.WriteSceneGraphBinaryData(align_path + "/scene_graph.bin");
        scene_graph_container.CorrespondenceGraph()->ExportToGraph(align_path + "/scene_graph.png");
        std::cout << "ExportToGraph done!" << std::endl;
    }
}

void PretreatReconstruction(FeatureDataContainer &feature_data_container,
                            SceneGraphContainer &scene_graph_container,
                            std::unordered_map<image_t, image_t>& image_id_map,
                            std::unordered_map<camera_t, camera_t>& camera_id_map,
                            std::string rect_path1, std::string rect_path2,
                            Reconstruction& reconstruction1,
                            Reconstruction& reconstruction2){
    
    reconstruction1.ReadBinary(rect_path1, false);
    // reconstruction1.ComputeBaselineDistance();
    std::vector<image_t> registered_images1 = reconstruction1.RegisterImageIds();
    std::set<image_t> registered_image_id_set1(registered_images1.begin(), registered_images1.end());
    std::shared_ptr<SceneGraphContainer> cluster_graph_container1 =
    std::shared_ptr<SceneGraphContainer>(new SceneGraphContainer());
    scene_graph_container.ClusterSceneGraphContainer(registered_image_id_set1, *cluster_graph_container1.get());
    reconstruction1.SetUp(cluster_graph_container1);
    reconstruction1.TearDown();
    for (auto image_id : reconstruction1.RegisterImageIds()) {
        Image & image = reconstruction1.Image(image_id);
        Camera & camera = reconstruction1.Camera(image.CameraId());

        const auto &keypoints = feature_data_container.GetKeypoints(image_id);
        const auto &panorama_indices = feature_data_container.GetPanoramaIndexs(image_id);
        std::vector<uint32_t> local_image_indices(keypoints.size());
        for(size_t i = 0; i<keypoints.size(); ++i){
            if (panorama_indices.size() == 0 && camera.NumLocalCameras() == 1) {
                local_image_indices[i] = image_id;
            } else{
                local_image_indices[i] = panorama_indices[i].sub_image_id;
            }
        }
        image.SetLocalImageIndices(local_image_indices);
    }

    Reconstruction reconstruction_t;
    reconstruction_t.ReadBinary(rect_path2, false);

    auto register_image_ids = reconstruction_t.RegisterImageIds();
    for (auto image_id : register_image_ids) {
        class Image image = reconstruction_t.Image(image_id);
        class Camera camera = reconstruction_t.Camera(image.CameraId());
        if (camera_id_map.find(camera.CameraId()) != camera_id_map.end()) {
            camera.SetCameraId(camera_id_map.at(camera.CameraId()));
        }
        if (!reconstruction2.ExistsCamera(camera.CameraId())) {
            reconstruction2.AddCamera(camera);
        }

        image.SetImageId(image_id_map.at(image_id));
        image.SetCameraId(camera.CameraId());
        for (size_t point2D_idx = 0; point2D_idx < image.NumPoints2D(); ++point2D_idx) {
            image.Point2D(point2D_idx).SetMapPointId(kInvalidMapPointId);
        }
        image.SetRegistered(false);

        reconstruction2.AddImage(image);
        reconstruction2.RegisterImage(image.ImageId());
    }
    auto mappoint_ids = reconstruction_t.MapPointIds();
    for (auto mappoint_id : mappoint_ids) {
        class MapPoint mappoint = reconstruction_t.MapPoint(mappoint_id);
        class Track& track = mappoint.Track();
        for (auto & track_elem : track.Elements()) {
            track_elem.image_id = image_id_map.at(track_elem.image_id);
        }
        reconstruction2.AddMapPoint(mappoint_id, mappoint.XYZ(), std::move(track), mappoint.Color());
    }

    std::vector<image_t> registered_images2 = reconstruction2.RegisterImageIds();
    std::set<image_t> registered_image_id_set2(registered_images2.begin(), registered_images2.end());
    std::shared_ptr<SceneGraphContainer> cluster_graph_container2 =
        std::shared_ptr<SceneGraphContainer>(new SceneGraphContainer());
    scene_graph_container.ClusterSceneGraphContainer(registered_image_id_set2, *cluster_graph_container2.get());
    reconstruction2.SetUp(cluster_graph_container2);
    reconstruction2.TearDown();
    for (auto image_id : reconstruction2.RegisterImageIds()) {
        Image & image = reconstruction2.Image(image_id);
        Camera & camera = reconstruction2.Camera(image.CameraId());

        const auto &keypoints = feature_data_container.GetKeypoints(image_id);
        const auto &panorama_indices = feature_data_container.GetPanoramaIndexs(image_id);
        std::vector<uint32_t> local_image_indices(keypoints.size());
        for(size_t i = 0; i<keypoints.size(); ++i){
            if (panorama_indices.size() == 0 && camera.NumLocalCameras() == 1) {
                local_image_indices[i] = image_id;
            } else{
                local_image_indices[i] = panorama_indices[i].sub_image_id;
            }
        }
        image.SetLocalImageIndices(local_image_indices);
    }
}

void ClusterFinalBundleAdjust(const sensemap::SceneGraphContainer &scene_graph_container,
                         const std::unordered_map<image_t, std::set<image_t>> image_neighbor_between_cluster,
                         std::shared_ptr<Reconstruction> reconstruction){
    const std::vector<image_t>& reg_image_ids = reconstruction->RegisterImageIds();
    std::set<image_t> neighbor_image_ids;
    for (const auto& image_neighbor : image_neighbor_between_cluster){
        neighbor_image_ids.insert(image_neighbor.first);
    }

    IndependentMapperOptions mapper_options;
    // OptionParser option_parser;
    // option_parser.GetMapperOptions(mapper_options,param);

    auto ba_options = mapper_options.GlobalBundleAdjustment();
    ba_options.refine_focal_length = false;
    ba_options.refine_principal_point = false;
    ba_options.refine_extra_params = false;
    ba_options.refine_local_extrinsics = false;
    ba_options.refine_extrinsics = true;

    const auto & correspondence_graph = scene_graph_container.CorrespondenceGraph();
    BundleAdjustmentConfig ba_config;
    for (auto image_id : neighbor_image_ids) {
        const auto & image_neighbors = correspondence_graph->ImageNeighbor(image_id);

        ba_config.AddImage(image_id);
        const Image& image = reconstruction->Image(image_id);
        ba_config.SetConstantCamera(image.CameraId());
        for (auto neighbor_id : image_neighbors){
            if (neighbor_image_ids.find(neighbor_id) == neighbor_image_ids.end()){
                ba_config.AddImage(neighbor_id);
                ba_config.SetConstantPose(neighbor_id);
                if (!reconstruction->ExistsImage(neighbor_id)){
                    continue;
                }
                const Image& image_neighbor = reconstruction->Image(neighbor_id);
                ba_config.SetConstantCamera(image_neighbor.CameraId());
            }
        }
    }
    std::cout << "ba_config: num_images, num_const_images, num_camrea:" 
            << ba_config.NumImages() << ", " << ba_config.NumConstantPoses() 
            << ", " << ba_config.NumConstantCameras() << std::endl;
    if (gps_prior_align){
        ba_options.prior_absolute_location_weight = 0.01;
        ba_options.use_prior_absolute_location = true;
        reconstruction->AlignWithPriorLocations();
    }
    std::shared_ptr<IncrementalMapper> mapper = 
        std::make_shared<IncrementalMapper>(std::make_shared<SceneGraphContainer>(scene_graph_container));
    mapper->BeginReconstruction(reconstruction);

    size_t num_retriangulate_observations = mapper->Retriangulate(mapper_options.Triangulation());
    std::cout << "Retriangulate observation: " << num_retriangulate_observations << std::endl;
    for (int i = 0; i < mapper_options.ba_global_max_refinements; ++i) {
        reconstruction->FilterObservationsWithNegativeDepth();

        const size_t num_observations = reconstruction->ComputeNumObservations();

        PrintHeading1("GBA Bundle adjustment");
        std::cout << "iter: " << i << std::endl;
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
};

void ClusterNeighborsFinalBundleAdjust(const std::shared_ptr<SceneGraphContainer> scene_graph_container,
                         const std::unordered_map<image_t, std::set<image_t>> &image_neighbor_between_cluster,
                         const IndependentMapperOptions& mapper_options,
                         std::shared_ptr<Reconstruction> reconstruction,
                         const std::unordered_set<image_t>& const_image_ids){
    
    PrintHeading1("Final Cluster Neighbors Bundle Adjust");
    const std::vector<image_t>& reg_image_ids = reconstruction->RegisterImageIds();
    std::set<image_t> neighbor_image_ids;
    for (const auto& image_neighbor : image_neighbor_between_cluster){
        neighbor_image_ids.insert(image_neighbor.first);
    }

    auto ba_options = mapper_options.GlobalBundleAdjustment();
    ba_options.refine_focal_length = false;
    ba_options.refine_principal_point = false;
    ba_options.refine_extra_params = false;
    ba_options.refine_local_extrinsics = false;
    ba_options.refine_extrinsics = true;

    std::shared_ptr<IncrementalMapper> mapper = 
        std::make_shared<IncrementalMapper>(scene_graph_container);
    mapper->BeginReconstruction(reconstruction);

    const auto & correspondence_graph = scene_graph_container->CorrespondenceGraph();
    BundleAdjustmentConfig ba_config;
    for (auto image_id : neighbor_image_ids) {
        const auto & image_neighbors = correspondence_graph->ImageNeighbor(image_id);

        ba_config.AddImage(image_id);
        const Image& image = reconstruction->Image(image_id);
        ba_config.SetConstantCamera(image.CameraId());
        if (const_image_ids.find(image_id) != const_image_ids.end()){
            ba_config.SetConstantPose(image_id);
        }
        for (auto neighbor_id : image_neighbors){
            if (neighbor_image_ids.find(neighbor_id) == neighbor_image_ids.end()){
                if (!reconstruction->ExistsImage(neighbor_id)){
                    continue;
                }
                ba_config.AddImage(neighbor_id);
                ba_config.SetConstantPose(neighbor_id);
                const Image& image_neighbor = reconstruction->Image(neighbor_id);
                ba_config.SetConstantCamera(image_neighbor.CameraId());
            }
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
};

void ClusterIsolatedFinalBundleAdjust(const std::shared_ptr<SceneGraphContainer> scene_graph_container,
                         const std::vector<std::vector<image_t>> &cluster_image_ids,
                         const std::unordered_map<image_t, std::set<image_t>>& image_neighbor_between_cluster,
                         const IndependentMapperOptions& mapper_options,
                         std::shared_ptr<Reconstruction> reconstruction,
                         const std::unordered_set<image_t>& const_image_ids){
    
    PrintHeading1("Final Cluster Isolated Bundle Adjust");
    const auto & correspondence_graph = scene_graph_container->CorrespondenceGraph();

    std::set<image_t> neighbor_image_ids;
    for (const auto& image_neighbor : image_neighbor_between_cluster){
        neighbor_image_ids.insert(image_neighbor.first);
    }

    auto ba_options = mapper_options.GlobalBundleAdjustment();
    ba_options.refine_focal_length = false;
    ba_options.refine_principal_point = false;
    ba_options.refine_extra_params = false;
    ba_options.refine_local_extrinsics = false;
    ba_options.refine_extrinsics = true;

    std::shared_ptr<IncrementalMapper> mapper = 
        std::make_shared<IncrementalMapper>(scene_graph_container);
    mapper->BeginReconstruction(reconstruction);

    size_t num_retriangulate_observations = mapper->Retriangulate(mapper_options.Triangulation());
    std::cout << "Retriangulate observation(Begin BA): " << num_retriangulate_observations << std::endl;
    for (int i = 0; i < cluster_image_ids.size(); i++){
        PrintHeading1(StringPrintf("Final Cluster-%d Isolated Bundle Adjust", i));
        BundleAdjustmentConfig ba_config;
        for (auto image_id : cluster_image_ids[i]){
            if (!reconstruction->ExistsImage(image_id)){
                continue;
            }

            const Image& image = reconstruction->Image(image_id);
            ba_config.AddImage(image_id);
            ba_config.SetConstantCamera(image.CameraId());
            if (const_image_ids.find(image_id) != const_image_ids.end() || 
                neighbor_image_ids.find(image_id) != neighbor_image_ids.end()){
                ba_config.SetConstantPose(image_id);
            }
        }
        std::cout << "ba_config: num_images, num_const_images, num_camrea:" 
            << ba_config.NumImages() << ", " << ba_config.NumConstantPoses() 
            << ", " << ba_config.NumConstantCameras() << std::endl;
        
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
};

void MergeReconstruction(Reconstruction reconstruction1, Reconstruction reconstruction2,
                         SceneGraphContainer scene_graph_container,
                         EIGEN_STL_UMAP(cluster_t, Eigen::Matrix3x4d) &global_transforms,
                         std::unordered_map<cluster_t, Reconstruction>& merged_reconstructions,
                         Configurator param = Configurator()){
    std::cout << "Merge Reconstructions" << std::endl;
    merged_reconstructions.clear();
    // merged_reconstructions[0] = reconstruction1;
    // merged_reconstructions[1] = reconstruction2;
    std::vector<std::shared_ptr<Reconstruction>> reconstructions;
    reconstructions.resize(2);
    reconstructions[0] = std::make_shared<Reconstruction>(reconstruction1);
    reconstructions[1] = std::make_shared<Reconstruction>(reconstruction2);

    ClusterMotionAverager cluster_motion_averager(true);
    cluster_motion_averager.SetGraphs(scene_graph_container.CorrespondenceGraph().get());

    EIGEN_STL_UMAP(cluster_pair_t, Eigen::Matrix3x4d) relative_transforms;
    std::vector<std::vector<cluster_t>> clusters_ordered;
    clusters_ordered.clear();
    Timer motion_averager_timer;
    motion_averager_timer.Start();
    cluster_motion_averager.ClusterMotionAverage(
            reconstructions,  // -- Calculate the global transform using the given reconstructions
            global_transforms, relative_transforms, clusters_ordered);
    std::cout << "clusters_ordered size: " << clusters_ordered.size() << std::endl;
    std::cout << "relative_transforms size: " << relative_transforms.size() << std::endl;
    std::cout << "Motion Average Cost " << motion_averager_timer.ElapsedMinutes() << " [min]" << std::endl;

#ifdef DEBUG_INFO
    std::cout << "Debug Info[motion average]: clusters_ordered size: " << clusters_ordered.size() << std::endl;
    for (size_t component_idx = 0; component_idx < clusters_ordered.size(); ++component_idx) {
        Reconstruction rect_t = *reconstructions[clusters_ordered[component_idx][0]];
        std::shared_ptr<Reconstruction> ref_reconstruction = std::make_shared<Reconstruction>(rect_t);

        for (size_t i = 1; i < clusters_ordered[component_idx].size(); ++i) {
            CHECK(global_transforms.find(clusters_ordered[component_idx][i]) != global_transforms.end());
            ref_reconstruction->Merge(*(reconstructions[clusters_ordered[component_idx][i]]),
                                        global_transforms.at(clusters_ordered[component_idx][i]), 8.0);
        }
        std::string save_path = JoinPaths(align_path, "align_motion_averge_" + std::to_string(component_idx));
        if (!boost::filesystem::exists(save_path)) {
            boost::filesystem::create_directories(save_path);
        }
        ref_reconstruction->WriteBinary(save_path);
        std::cout << "save MotionAverge Reconstruction-" << component_idx << " to " << save_path << std::endl;
    }
#endif

    std::shared_ptr<Reconstruction> reconstruction;
    ClusterMergeOptimizer::ClusterMergeOptions merge_options;

    ClusterMergeOptimizer clustermerge(
        std::make_shared<ClusterMergeOptimizer::ClusterMergeOptions>(merge_options),
        scene_graph_container.CorrespondenceGraph().get());

    // Sort the clusters_ordered by cluster size
    std::sort(
        clusters_ordered.begin(), clusters_ordered.end(),
        [](const std::vector<cluster_t>& v1, const std::vector<cluster_t>& v2) { return v1.size() > v2.size(); });

    auto cluster_ordered = clusters_ordered[0];
    Timer cluster_merge_timer;
    cluster_merge_timer.Start();
    clustermerge.MergeByPoseGraph(reconstructions, global_transforms, relative_transforms, cluster_ordered,
                                reconstruction);

    const auto& image_neighbor_between_cluster = clustermerge.GetImageNeighborBetweenCluster();
    std::vector<std::vector<image_t>> cluster_image_ids = clustermerge.GetClusterImageIds(true);
    std::unordered_set<image_t> const_image_ids = clustermerge.GetConstImageIds();

    std::cout << "GetClusterImageIds: \n";
    for (size_t i = 0; i < cluster_image_ids.size(); i++){
        std::cout << "cluster: " << i << ": ";
        for (size_t j = 0; j < cluster_image_ids.at(i).size(); j++){
            std::cout << cluster_image_ids.at(i).at(j) << ", ";
        }
        std::cout << std::endl;
    }
    std::cout << "...done" << std::endl;


    std::cout << "GetConstImageIds size: " << const_image_ids.size() << std::endl;
    for (auto id : const_image_ids){
        std::cout << id << ", ";
    }
    std::cout << std::endl;
    std::cout <<"...done" << std::endl; 

    std::cout << "Pose Graph cluster find " << image_neighbor_between_cluster.size() 
        << "neighbor images, merge cost " << cluster_merge_timer.ElapsedMinutes() << " [min]" << std::endl;
    
    {
        float mem;
        sensemap::GetAvailableMemory(mem);
        std::cout << "memory left : " << mem << std::endl;
        std::cout << malloc_trim(0) << std::endl;
        sensemap::GetAvailableMemory(mem);
        std::cout << "memory left : " << mem << std::endl;
    }

#ifdef DEBUG_INFO
    std::string save_path = JoinPaths(align_path, "align_pose_graph_0");
    if (!boost::filesystem::exists(save_path)) {
        boost::filesystem::create_directories(save_path);
    }
    reconstruction->WriteBinary(save_path);
    std::cout << "save PoseGraph Reconstruction-" << 0 << " to " << save_path << std::endl;
#endif
    merged_reconstructions[0] = *reconstruction;

    // ClusterFinalBundleAdjust(scene_graph_container, image_neighbor_between_cluster, reconstruction);
    
    {
        float mem;
        sensemap::GetAvailableMemory(mem);
        std::cout << "memory left : " << mem << std::endl;
        std::cout << malloc_trim(0) << std::endl;
        sensemap::GetAvailableMemory(mem);
        std::cout << "memory left : " << mem << std::endl;
    }

    const std::shared_ptr<SceneGraphContainer> scene_graph_container_ptr 
        = std::make_shared<SceneGraphContainer>(scene_graph_container);
    IndependentMapperOptions mapper_options;
    OptionParser option_parser;
    option_parser.GetMapperOptions(mapper_options,param);

    ClusterNeighborsFinalBundleAdjust(scene_graph_container_ptr, 
                                      image_neighbor_between_cluster, 
                                      mapper_options,
                                      reconstruction,
                                      const_image_ids);

    ClusterIsolatedFinalBundleAdjust(scene_graph_container_ptr,
                                     cluster_image_ids,
                                     image_neighbor_between_cluster,
                                     mapper_options,
                                     reconstruction,
                                     const_image_ids);
    
    ClusterNeighborsFinalBundleAdjust(scene_graph_container_ptr, 
                                      image_neighbor_between_cluster, 
                                      mapper_options,
                                      reconstruction,
                                      const_image_ids);
    reconstruction->AlignWithPriorLocations();
    merged_reconstructions[0] = *reconstruction;

    return;
}

int main(int argc, char *argv[]) {
    using namespace sensemap;

    PrintHeading("SenseMap.  Copyright(c) 2022, SenseTime Group.");
    PrintHeading(std::string("Version: align-reconstruction-") + __VERSION__);
    Timer timer;
    timer.Start();

    std::string help_info = StringPrintf("Usage: \n" \
        "./align_reconstruction --config1=./sfm1.yaml\n" \
        "                       --rectID1=0\n"
        "                       --config2=./sfm2.yaml\n"
        "                       --rectID2=0\n\n");
    google::SetUsageMessage(help_info.c_str());

    google::ParseCommandLineFlags(&argc, &argv, false);

    if (argc != 5) {
        std::cout << google::ProgramUsage() << std::endl;
		return StateCode::NO_MATCHING_INPUT_PARAM;
    }

    std::cout << "config1: " << FLAGS_config1 << ", rect id: " << FLAGS_rectID1 << std::endl;
    std::cout << "config2: " << FLAGS_config2 << ", rect id: " << FLAGS_rectID2 << std::endl;

    Configurator param1, param2;
    param1.Load(FLAGS_config1.c_str());
    param2.Load(FLAGS_config2.c_str());

    std::string workspace_path1 = param1.GetArgument("workspace_path", "");
    std::string workspace_path2 = param2.GetArgument("workspace_path", "");
    int rect_id1 = std::atoi(FLAGS_rectID1.c_str()); 
    int rect_id2 = std::atoi(FLAGS_rectID2.c_str()); 
    std::string rect_path1 = JoinPaths(workspace_path1, std::to_string(rect_id1));
    std::string rect_path2 = JoinPaths(workspace_path2, std::to_string(rect_id2));

    align_path = JoinPaths(GetParentDir(workspace_path2), "sfm-workspace-align");
    
    std::unordered_map<image_t, image_t> image_id_map;
    std::unordered_map<camera_t, camera_t> camera_id_map;
    FeatureDataContainer feature_data_container;
    SceneGraphContainer scene_graph_container;

    MergeSeneGraph(feature_data_container, scene_graph_container, image_id_map, camera_id_map, param1, param2);

    {
        float mem;
        sensemap::GetAvailableMemory(mem);
        std::cout << "begin memory left : " << mem << std::endl;
        const auto image_ids = feature_data_container.GetImageIds();
        for (auto image_id : image_ids){
            feature_data_container.GetDescriptors(image_id).resize(0,0);
            feature_data_container.GetCompressedDescriptors(image_id).resize(0,0);
        }
        std::cout << malloc_trim(0) << std::endl;
        sensemap::GetAvailableMemory(mem);
        std::cout << "clear Descriptors memory left : " << mem << std::endl;
    }

    Reconstruction reconstruction1, reconstruction2;
    PretreatReconstruction(feature_data_container, scene_graph_container, 
        image_id_map, camera_id_map, rect_path1, rect_path2, 
        reconstruction1, reconstruction2);

    {
        float mem;
        sensemap::GetAvailableMemory(mem);
        std::cout << "memory left : " << mem << std::endl;
        std::cout << malloc_trim(0) << std::endl;
        sensemap::GetAvailableMemory(mem);
        std::cout << "memory left : " << mem << std::endl;
    }

    std::unordered_map<cluster_t, Reconstruction> merged_reconstructions;
    EIGEN_STL_UMAP(cluster_t, Eigen::Matrix3x4d) global_transforms;
    MergeReconstruction(reconstruction1, reconstruction2, scene_graph_container, 
                        global_transforms, merged_reconstructions);
    
    Reconstruction new_reconstruction = merged_reconstructions[0];
    {
        auto align_rect_path = JoinPaths(align_path, "0");
        if (!boost::filesystem::exists(align_rect_path)) {
            boost::filesystem::create_directories(align_rect_path);
        }
        new_reconstruction.WriteBinary(align_rect_path);
        std::cout << "save reconstruction : " << align_rect_path << std::endl;

        const std::string gps_rec_path = JoinPaths(align_path, "0-gps");
        if (!boost::filesystem::exists(gps_rec_path)) {
            boost::filesystem::create_directories(gps_rec_path);
        }
        new_reconstruction.AlignWithPriorLocations();
        new_reconstruction.WriteBinary(gps_rec_path);
        std::cout << "save gps prior reconstruction : " << gps_rec_path << std::endl;
    }

    std::cout << StringPrintf("Align Reconstruction in %.3fs", timer.ElapsedSeconds()) << std::endl;

    return 0;
}