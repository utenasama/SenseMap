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

#define MAX_PRECISION std::setprecision(std::numeric_limits<double>::digits10 + 1)

// DEFINE_string(rec_path1, "", "reconstruction path for the first scene, e.g. scene1/sfm-workspace/0");
DEFINE_string(config1, "", "configuration for the first scene, e.g. scene1/sfm.yaml");
// DEFINE_string(rec_path2, "", "reconstruction path for the second scene, e.g. scene2/sfm-workspace/0");
DEFINE_string(config2, "", "configuration for the second scene, e.g. scene2/sfm.yaml");

using namespace sensemap;

bool LoadFeatures(FeatureDataContainer &feature_data_container, Configurator &param) {
    std::string workspace_path = param.GetArgument("workspace_path", "");
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
        if (boost::filesystem::exists(JoinPaths(workspace_path, "/local_images.bin"))) {
            feature_data_container.ReadLocalImagesBinaryData(JoinPaths(workspace_path, "/local_images.bin"));
        }
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
    }

    if (exist_feature_file) {
        return true;
    } else {
        return false;
    }
}

void MergeFeatureAndMatches(FeatureDataContainer &feature_data_container1,
                            FeatureDataContainer &feature_data_container2,
                            SceneGraphContainer &scene_graph_container1,
                            SceneGraphContainer &scene_graph_container2,
                            FeatureDataContainer &feature_data_container,
                            SceneGraphContainer &scene_graph_container,
                            std::unordered_map<image_t, image_t> &image_id_map,
                            std::unordered_map<camera_t, camera_t> &camera_id_map) {
    feature_data_container = feature_data_container1;
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

    std::cout << StringPrintf("Update Feature Container\n");

    // Update feature data container.
    // std::unordered_map<image_t, image_t> image_id_map;
    // std::unordered_map<camera_t, camera_t> camera_id_map;

    std::vector<image_t> new_image_ids;
    new_image_ids.reserve(feature_data_container2.GetImageIds().size());
    image_t new_image_id = max_image_id1 + 1;
    camera_t new_camera_id = max_camera_id1 + 1;
    FeatureDataPtrUmap feature_datas = feature_data_container2.GetFeatureData();
    for (image_t image_id : feature_data_container2.GetImageIds()) {
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

        new_camera.SetCameraId(new_image.CameraId());

        new_image_ids.push_back(new_image_id);
        image_id_map[image_id] = new_image_id++;

        feature_data_container.emplace(new_image.ImageId(), feature_data_ptr);
        feature_data_container.emplace(new_camera.CameraId(), std::make_shared<Camera>(new_camera));
        feature_data_container.emplace(new_image.Name(), new_image.ImageId());
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

// Merge Communities using max spanning tree and cluster motion average
void EstimateTransformationBetweenReconstructions(
    const std::unordered_map<int, std::shared_ptr<ReconstructionManager>>& reconstruction_managers,
    const CorrespondenceGraph* full_correspondence_graph,
    std::shared_ptr<ReconstructionManager>& root_reconstruction_manager,
    ClusterMergeOptimizer::ClusterMergeOptions merge_options,
    std::unordered_map<cluster_t, Reconstruction>& merged_reconstructions,
    EIGEN_STL_UMAP(cluster_t, Eigen::Matrix3x4d) & global_transforms) {

    std::cout << "Estimate Transformation" << std::endl;
    merged_reconstructions.clear();
    std::vector<std::shared_ptr<Reconstruction>> reconstructions;
    std::map<int, cluster_t> cluster_ids_map;
    size_t cluster_id = 0;
    reconstructions.resize(reconstruction_managers.size());
    // For Reconstruction Manager of each community
    for (const auto& recon_manager : reconstruction_managers) {
        // For each reconstruction in the manager
        for (size_t i = 0; i < recon_manager.second->Size(); i++) {

            
            // -- Skip small reconstruction
            if (recon_manager.second->Get(i)->NumMapPoints() < 1000) {
                reconstructions[recon_manager.first] = std::shared_ptr<Reconstruction>(new Reconstruction());
                continue;
            }
            // -- Push all the reconstruction results in the reconstructions
            // reconstructions.emplace_back(recon_manager.second->Get(i));
            reconstructions[recon_manager.first] = recon_manager.second->Get(i);
            cluster_ids_map[reconstructions.size() - 1] = recon_manager.first;
            merged_reconstructions[recon_manager.first] = *reconstructions[recon_manager.first].get();
        }
    }

    if (reconstructions.size() == 0) {
        return;
    }

    ClusterMotionAverager cluster_motion_averager(merge_options.debug_info);
    cluster_motion_averager.SetGraphs(full_correspondence_graph);

    EIGEN_STL_UMAP(cluster_pair_t, Eigen::Matrix3x4d) relative_transforms;
    std::vector<std::vector<cluster_t>> clusters_ordered;

    clusters_ordered.clear();
    Timer motion_averager_timer;
    motion_averager_timer.Start();
    cluster_motion_averager.ClusterMotionAverage(
        reconstructions,  // -- Calculate the global transform using the given reconstructions
        global_transforms, relative_transforms, clusters_ordered);
    std::cout << "Motion Average Cost " << motion_averager_timer.ElapsedMinutes() << " [min]" << std::endl;
    // if(merge_options.save_initial_transform){
    //     std::cout << "Save Initial Transform" << std::endl;
    //     WriteInitialTransform(merge_options.initial_transform_path, global_transforms, relative_transforms,
    //                           clusters_ordered);
    // }

    // merge reconstructions with global transform from motion average directly
    for (size_t component_idx = 0; component_idx < clusters_ordered.size(); ++component_idx) {
        std::shared_ptr<Reconstruction> ref_reconstruction = reconstructions[clusters_ordered[component_idx][0]];

        for (size_t i = 1; i < clusters_ordered[component_idx].size(); ++i) {
            CHECK(global_transforms.find(clusters_ordered[component_idx][i]) != global_transforms.end());
            ref_reconstruction->Merge(*(reconstructions[clusters_ordered[component_idx][i]]),
                                        global_transforms.at(clusters_ordered[component_idx][i]), 8.0);
        }
        root_reconstruction_manager->Add();
        *root_reconstruction_manager->Get(component_idx).get() = std::move(*ref_reconstruction.get());
    }
}

size_t EstimateAlignError(const Reconstruction reconstruction1, const Reconstruction reconstruction2,
                          const SceneGraphContainer scene_graph_container,
                          const cluster_t cluster_id1, const cluster_t cluster_id2,
                          EIGEN_STL_UMAP(cluster_t, Eigen::Matrix3x4d) global_transforms,
                          const float max_error) {
    ReconstructionAlignerOptions options;        
    ReconstructionAligner reconstruction_aligner(options);
    
    reconstruction_aligner.SetGraphs(scene_graph_container.CorrespondenceGraph().get());
    
    std::vector<std::pair<std::pair<image_t, image_t>, point2D_t> > loaded_image_pairs;
    size_t loaded_max_num_image_pair = 0;

    std::vector<mappoint_t> common_points1; 
    std::vector<mappoint_t> common_points2;
    reconstruction_aligner.FindCommonPointsByCorrespondence(reconstruction1, reconstruction2, loaded_image_pairs,
                                                            loaded_max_num_image_pair, common_points1, common_points2, false);

    std::vector<double> residuals;
	residuals.resize(common_points1.size());

    const Eigen::Matrix3x4d alignment21 = SimilarityTransform3(global_transforms.at(cluster_id2)).Inverse().Matrix().topRows<3>();
    const Eigen::Matrix3x4d alignment12 = global_transforms.at(cluster_id2);

	for(size_t i = 0; i < common_points1.size(); ++i){
		const MapPoint& mappoint1 = reconstruction1.MapPoint(common_points1[i]);
		const MapPoint& mappoint2 = reconstruction2.MapPoint(common_points2[i]);

	    std::vector<TrackElement> track1 = mappoint1.Track().Elements();
		std::vector<TrackElement> track2 = mappoint2.Track().Elements();
		CHECK((!track1.empty())&&(!track2.empty()));
		
		double median_error = 0;
		int num_elems = track1.size() + track2.size();
		std::vector<double> reproj_errors1, reproj_errors2;
        //reproject points1 into reconstruction_dst using alignment12  
		const Eigen::Vector3d xyz12 = alignment12 * mappoint1.XYZ().homogeneous();  
		for(const auto elem : track2){
			const Image& image2 = reconstruction2.Image(elem.image_id);
			const Camera& camera2 = reconstruction2.Camera(image2.CameraId());
			const Point2D& point2d2 = image2.Point2D(elem.point2D_idx);
			const Eigen::Matrix3x4d proj_matrix2 = image2.ProjectionMatrix();
			
			if(camera2.NumLocalCameras()>1){
				camera_t  local_image_id2 = image2.LocalImageIndices()[elem.point2D_idx];
				double error = CalculateSquaredReprojectionErrorRig(point2d2.XY(), xyz12, proj_matrix2, 
                                                                    local_image_id2, camera2);
				reproj_errors1.push_back(error);
			}
			else{
				double error = CalculateSquaredReprojectionError(point2d2.XY(), xyz12,
													             proj_matrix2, camera2);
				reproj_errors1.push_back(error);
			}
		}
        //reproject points2 into reconstruction_src using alignment21  
		const Eigen::Vector3d xyz21 = alignment21 * mappoint2.XYZ().homogeneous();
		for(const auto elem : track1){
			const Image& image1 = reconstruction1.Image(elem.image_id);
			const Camera& camera1 = reconstruction1.Camera(image1.CameraId());
			const Point2D& point2d1=image1.Point2D(elem.point2D_idx);
			const Eigen::Matrix3x4d proj_matrix1 = image1.ProjectionMatrix();

			if(camera1.NumLocalCameras()>1){
				camera_t  local_image_id1 = image1.LocalImageIndices()[elem.point2D_idx];
				double error = CalculateSquaredReprojectionErrorRig(point2d1.XY(), xyz21, proj_matrix1, 
                                                                    local_image_id1, camera1);
				reproj_errors2.push_back(error);
			}
			else{
				double error = CalculateSquaredReprojectionError(point2d1.XY(),xyz21,
															     proj_matrix1,camera1);
				reproj_errors2.push_back(error);
			}
		}
		//mean_error/=num_elems;
		int nth1 = reproj_errors1.size() / 2;
        std::nth_element(reproj_errors1.begin(), reproj_errors1.begin() + nth1, 
						 reproj_errors1.end());
        int nth2 = reproj_errors2.size() / 2;
        std::nth_element(reproj_errors2.begin(), reproj_errors2.begin() + nth2, 
						 reproj_errors2.end());	
		median_error = std::max(reproj_errors1[nth1],reproj_errors2[nth2]);

		residuals[i] = median_error;
	}

    const double max_residual = max_error * max_error;
    size_t num_inliers = 0;
    for (size_t i = 0; i < common_points1.size(); ++i) {
        if (residuals[i] < max_residual) {
            num_inliers++;
        }
    }
    return num_inliers;
}

void AlignReconstruction(Reconstruction reconstruction1, Reconstruction reconstruction2,
                         SceneGraphContainer scene_graph_container,
                         EIGEN_STL_UMAP(cluster_t, Eigen::Matrix3x4d) &global_transforms) {
    std::unordered_map<int, std::shared_ptr<ReconstructionManager>> reconstruction_managers;
    
    std::vector<image_t> registered_images1 = reconstruction1.RegisterImageIds();
    std::cout<<"Reconstruction "<< 0 <<" has "<<registered_images1.size()<<" images"<<std::endl;
    std::set<image_t> registered_image_id_set1(registered_images1.begin(), registered_images1.end());

    std::cout << "Create cluster graph container" << std::endl;
    std::shared_ptr<SceneGraphContainer> cluster_graph_container1 =
        std::shared_ptr<SceneGraphContainer>(new SceneGraphContainer());
    scene_graph_container.ClusterSceneGraphContainer(registered_image_id_set1, *cluster_graph_container1.get());
    std::cout << "Create cluster graph container done " << std::endl;

    auto reconstruction_manager1 = std::make_shared<ReconstructionManager>();
    auto rec_idx1 = reconstruction_manager1->Add();
    *reconstruction_manager1->Get(rec_idx1).get() = reconstruction1;

    reconstruction_manager1->Get(rec_idx1)->SetUp(cluster_graph_container1);
    reconstruction_manager1->Get(rec_idx1)->TearDown();
    reconstruction_managers.emplace(0, reconstruction_manager1);

    std::vector<image_t> registered_images2 = reconstruction2.RegisterImageIds();
    std::cout<<"Reconstruction "<< 1 <<" has "<<registered_images2.size()<<" images"<<std::endl;
    std::set<image_t> registered_image_id_set2(registered_images2.begin(), registered_images2.end());

    std::cout << "Create cluster graph container" << std::endl;
    std::shared_ptr<SceneGraphContainer> cluster_graph_container2 =
        std::shared_ptr<SceneGraphContainer>(new SceneGraphContainer());
    scene_graph_container.ClusterSceneGraphContainer(registered_image_id_set2, *cluster_graph_container2.get());
    std::cout << "Create cluster graph container done " << std::endl;
    
    auto reconstruction_manager2 = std::make_shared<ReconstructionManager>();
    auto rec_idx2 = reconstruction_manager2->Add();
    *reconstruction_manager2->Get(rec_idx2).get() = reconstruction2;
    reconstruction_manager2->Get(rec_idx2)->SetUp(cluster_graph_container2);
    reconstruction_manager2->Get(rec_idx2)->TearDown();
    reconstruction_managers.emplace(1, reconstruction_manager2);

    // EIGEN_STL_UMAP(cluster_t, Eigen::Matrix3x4d) global_transforms;
    std::unordered_map<cluster_t, Reconstruction> reconstructions;

    std::shared_ptr<ReconstructionManager> merged_reconstruction_manager = std::make_shared<ReconstructionManager>();
    ClusterMergeOptimizer::ClusterMergeOptions merge_options;
    EstimateTransformationBetweenReconstructions(reconstruction_managers, scene_graph_container.CorrespondenceGraph().get(),
                    merged_reconstruction_manager, merge_options, reconstructions, global_transforms);

    // // Transform cluster.
    for (auto reconstruction_map : reconstructions) {
        size_t community_id = reconstruction_map.first;
        auto reconstruction = reconstruction_map.second;

        std::cout << "cluster_id " << community_id << "  has Images " << reconstruction.NumImages() << std::endl;

        auto trans = global_transforms.at(community_id);
        std::cout<<"transfromation is "<<std::endl<<trans<<std::endl;
        // reconstruction.TransformReconstruction(trans);
    }
}

int main(int argc, char *argv[]) {
    using namespace sensemap;

    PrintHeading("SenseMap.  Copyright(c) 2022, SenseTime Group.");
    PrintHeading(std::string("Version: align-reconstruction-") + __VERSION__);
    Timer timer;
    timer.Start();

    std::string help_info = StringPrintf("Usage: \n" \
        "./align_reconstruction --config1=./sfm1.yaml\n" \
        "                       --config2=./sfm2.yaml\n\n");
    google::SetUsageMessage(help_info.c_str());

    google::ParseCommandLineFlags(&argc, &argv, false);

    if (argc != 3) {
        std::cout << google::ProgramUsage() << std::endl;
		return StateCode::NO_MATCHING_INPUT_PARAM;
    }

    std::cout << "config1: " << FLAGS_config1 << std::endl;
    std::cout << "config2: " << FLAGS_config2 << std::endl;

    Configurator param1, param2;
    param1.Load(FLAGS_config1.c_str());
    param2.Load(FLAGS_config2.c_str());

    std::string workspace_path1 = param1.GetArgument("workspace_path", "");
    std::string workspace_path2 = param2.GetArgument("workspace_path", "");

    std::unordered_map<image_t, image_t> image_id_map;
    std::unordered_map<camera_t, camera_t> camera_id_map;
    FeatureDataContainer feature_data_container;
    SceneGraphContainer scene_graph_container;
    {

        FeatureDataContainer feature_data_container1, feature_data_container2;
        LoadFeatures(feature_data_container1, param1);
        LoadFeatures(feature_data_container2, param2);

        SceneGraphContainer scene_graph_container1, scene_graph_container2;
        scene_graph_container1.ReadSceneGraphBinaryData(JoinPaths(workspace_path1, "/scene_graph.bin"));
        scene_graph_container2.ReadSceneGraphBinaryData(JoinPaths(workspace_path2, "/scene_graph.bin"));

        MergeFeatureAndMatches(feature_data_container1, feature_data_container2,
                            scene_graph_container1, scene_graph_container2,
                            feature_data_container, scene_graph_container,
                            image_id_map, camera_id_map);

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
        std::cout << "memory left : " << mem << std::endl;
        const auto image_ids = feature_data_container.GetImageIds();
        for (auto image_id : image_ids){
            feature_data_container.GetDescriptors(image_id).resize(0,0);
            feature_data_container.GetCompressedDescriptors(image_id).resize(0,0);
        }
        sensemap::GetAvailableMemory(mem);
        std::cout << "clear Descriptors memory left : " << mem << std::endl;
    }
    {
        float mem;
        sensemap::GetAvailableMemory(mem);
        std::cout << "memory left : " << mem << std::endl;
        std::cout << malloc_trim(0) << std::endl;
        sensemap::GetAvailableMemory(mem);
        std::cout << "memory left : " << mem << std::endl;
    }
    // if (!boost::filesystem::exists(FLAGS_merge_path)) {
    //     boost::filesystem::create_directories(FLAGS_merge_path);
    // }

    // scene_graph_container.WriteSceneGraphBinaryData(FLAGS_merge_path + "/scene_graph.bin");
    // scene_graph_container.CorrespondenceGraph()->ExportToGraph(FLAGS_merge_path + "/scene_graph.png");
    // std::cout << "ExportToGraph done!" << std::endl;
    size_t num_reconstruction1 = 0;
    for (size_t i = 0; ;i++) {
        const auto& reconstruction_path = JoinPaths(workspace_path1, std::to_string(i));
        if (!ExistsDir(reconstruction_path)) {
            break;
        }
        num_reconstruction1++;
    }

    size_t num_reconstruction2 = 0;
    for (size_t i = 0; ;i++) {
        const auto& reconstruction_path = JoinPaths(workspace_path2, std::to_string(i));
        if (!ExistsDir(reconstruction_path)) {
            break;
        }
        num_reconstruction2++;
    }

    EIGEN_STL_UMAP(cluster_t, Eigen::Matrix3x4d) best_transform;
    std::size_t best_num_inliers = 0;
    for (size_t i = 0; i < num_reconstruction1; ++i) {
        Reconstruction reconstruction1;
        reconstruction1.ReadBinary(JoinPaths(workspace_path1, std::to_string(i)), false);
        // reconstruction1.ComputeBaselineDistance();

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

        for (size_t j = 0; j < num_reconstruction2; ++j) {
            Reconstruction reconstruction2;
            reconstruction2.ReadBinary(JoinPaths(workspace_path2, std::to_string(j)), false);

            Reconstruction new_reconstruction;

            auto register_image_ids = reconstruction2.RegisterImageIds();
            for (auto image_id : register_image_ids) {
                class Image image = reconstruction2.Image(image_id);
                class Camera camera = reconstruction2.Camera(image.CameraId());
                if (camera_id_map.find(camera.CameraId()) != camera_id_map.end()) {
                    camera.SetCameraId(camera_id_map.at(camera.CameraId()));
                }
                if (!new_reconstruction.ExistsCamera(camera.CameraId())) {
                    new_reconstruction.AddCamera(camera);
                }

                image.SetImageId(image_id_map.at(image_id));
                image.SetCameraId(camera.CameraId());
                for (size_t point2D_idx = 0; point2D_idx < image.NumPoints2D(); ++point2D_idx) {
                    image.Point2D(point2D_idx).SetMapPointId(kInvalidMapPointId);
                }
                image.SetRegistered(false);

                const auto &keypoints = feature_data_container.GetKeypoints(image.ImageId());
                const auto &panorama_indices = feature_data_container.GetPanoramaIndexs(image.ImageId());
                std::vector<uint32_t> local_image_indices(keypoints.size());
                for(size_t i = 0; i<keypoints.size(); ++i){
                    if (panorama_indices.size() == 0 && camera.NumLocalCameras() == 1) {
                        local_image_indices[i] = image.ImageId();
                    } else{
                        local_image_indices[i] = panorama_indices[i].sub_image_id;
                    }
                }
                image.SetLocalImageIndices(local_image_indices);

                new_reconstruction.AddImage(image);
                new_reconstruction.RegisterImage(image.ImageId());
            }

            auto mappoint_ids = reconstruction2.MapPointIds();
            for (auto mappoint_id : mappoint_ids) {
                class MapPoint mappoint = reconstruction2.MapPoint(mappoint_id);
                class Track& track = mappoint.Track();
                for (auto & track_elem : track.Elements()) {
                    track_elem.image_id = image_id_map.at(track_elem.image_id);
                }
                new_reconstruction.AddMapPoint(mappoint_id, mappoint.XYZ(), std::move(track), mappoint.Color());
            }
            // new_reconstruction.ComputeBaselineDistance();

            std::cout << "Estimate Transformation Between Reconstructions." << std::endl;
            EIGEN_STL_UMAP(cluster_t, Eigen::Matrix3x4d) global_transforms;
            AlignReconstruction(reconstruction1, new_reconstruction, scene_graph_container, global_transforms);
            size_t num_inliers =
            EstimateAlignError(reconstruction1, new_reconstruction, scene_graph_container, 
                               0, 1, global_transforms, 8.0f);

            if (num_inliers > best_num_inliers) {
                best_num_inliers = num_inliers;
                best_transform = global_transforms;
            }
            std::cout << StringPrintf("Number of inliers between Reconstruction %d and %d(align): %d\n", i, j, num_inliers);
        }
    }

    // std::cout << StringPrintf("Choose the transformation between Reconstruction %d and %d\n", best_cluster_i, best_cluster_j);

    {
        float mem;
        sensemap::GetAvailableMemory(mem);
        std::cout << "memory left : " << mem << std::endl;
        std::cout << malloc_trim(0) << std::endl;
        sensemap::GetAvailableMemory(mem);
        std::cout << "memory left : " << mem << std::endl;
    }
    // Transform cluster.
    Eigen::Matrix3x4d trans = (best_transform.size() > 1) ? best_transform.at(1) : Eigen::Matrix3x4d::Identity();    
    std::cout<<"transfromation is "<<std::endl<<trans<<std::endl;
    for (size_t j = 0; j < num_reconstruction2; ++j) {
        Reconstruction reconstruction2;
        reconstruction2.ReadBinary(JoinPaths(workspace_path2, std::to_string(j)), false);
        reconstruction2.TransformReconstruction(trans);

        auto align_path = JoinPaths(GetParentDir(workspace_path2), StringPrintf("sfm-workspace-align/%d", j));
        if (!boost::filesystem::exists(align_path)) {
            boost::filesystem::create_directories(align_path);
        }
        reconstruction2.WriteBinary(align_path);
    }

    // Reconstruction reconstruction1, reconstruction2;
    // reconstruction1.ReadBinary(JoinPaths(workspace_path1, "0"), false);
    // reconstruction2.ReadBinary(JoinPaths(workspace_path2, "0"), false);

    // Reconstruction new_reconstruction;

    // auto register_image_ids = reconstruction2.RegisterImageIds();
    // for (auto image_id : register_image_ids) {
    //     class Image image = reconstruction2.Image(image_id);
    //     class Camera camera = reconstruction2.Camera(image.CameraId());
    //     if (camera_id_map.find(camera.CameraId()) != camera_id_map.end()) {
    //         camera.SetCameraId(camera_id_map.at(camera.CameraId()));
    //     }
    //     if (!new_reconstruction.ExistsCamera(camera.CameraId())) {
    //         new_reconstruction.AddCamera(camera);
    //     }

    //     image.SetImageId(image_id_map.at(image_id));
    //     image.SetCameraId(camera.CameraId());
    //     for (size_t point2D_idx = 0; point2D_idx < image.NumPoints2D(); ++point2D_idx) {
    //         image.Point2D(point2D_idx).SetMapPointId(kInvalidMapPointId);
    //     }
    //     image.SetRegistered(false);
    //     new_reconstruction.AddImage(image);
    //     new_reconstruction.RegisterImage(image.ImageId());
    // }

    // auto mappoint_ids = reconstruction2.MapPointIds();
    // for (auto mappoint_id : mappoint_ids) {
    //     class MapPoint mappoint = reconstruction2.MapPoint(mappoint_id);
    //     class Track& track = mappoint.Track();
    //     for (auto & track_elem : track.Elements()) {
    //         track_elem.image_id = image_id_map.at(track_elem.image_id);
    //     }
    //     new_reconstruction.AddMapPoint(mappoint_id, mappoint.XYZ(), track, mappoint.Color());
    // }

    // std::cout << "Estimate Transformation Between Reconstructions." << std::endl;
    // EIGEN_STL_UMAP(cluster_t, Eigen::Matrix3x4d) global_transforms;
    // AlignReconstruction(reconstruction1, new_reconstruction, scene_graph_container, global_transforms);
    
    // // Transform cluster.
    // auto trans = global_transforms.at(1);
    // std::cout<<"transfromation is "<<std::endl<<trans<<std::endl;
    // // new_reconstruction.TransformReconstruction(trans);
    // reconstruction2.TransformReconstruction(trans);

    // std::string rec_path = StringPrintf("%s", JoinPaths(workspace_path2, "aligned/0").c_str());
    // if (!boost::filesystem::exists(rec_path)) {
    //     boost::filesystem::create_directories(rec_path);
    // }
    // reconstruction2.WriteBinary(rec_path);

    std::cout << StringPrintf("Align Reconstruction in %.3fs", timer.ElapsedSeconds()) << std::endl;

    return 0;
}
