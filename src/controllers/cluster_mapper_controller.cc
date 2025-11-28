// Copyright (c) 2019, SenseTime Group.
// All rights reserved.

#include "cluster_mapper_controller.h"

#include <boost/filesystem/path.hpp>

#include "base/pose.h"
#include "cluster/fast_community.h"
#include "controllers/incremental_mapper_controller.h"
#include "controllers/directed_mapper_controller.h"
// #include "optim/block_bundle_adjustment.h"
#include "optim/cluster_merge/cluster_merge_optimizer.h"
#include "optim/cluster_motion_averager.h"
#include "optim/pose_graph_optimizer.h"
#include "util/bitmap.h"
#include "util/misc.h"
#include "util/ply.h"

namespace sensemap {
namespace {
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
}

// FIXME:
// Callback functor called after each bundle adjustment iteration.
class BundleAdjustmentIterationCallback : public ceres::IterationCallback {
   public:
    explicit BundleAdjustmentIterationCallback(Thread* thread) : thread_(thread) {}

    virtual ceres::CallbackReturnType operator()(const ceres::IterationSummary& summary) {
        CHECK_NOTNULL(thread_);
        thread_->BlockIfPaused();
        if (thread_->IsStopped()) {
            return ceres::SOLVER_TERMINATE_SUCCESSFULLY;
        } else {
            return ceres::SOLVER_CONTINUE;
        }
    }

   private:
    Thread* thread_;
};

namespace {

// Function to reconstruct one cluster using incremental mapping.
void ReconstructSingleInstance(const std::string& workspace_path, const std::string& image_path,
                               const IndependentMapperOptions& options,
                               const std::shared_ptr<SceneGraphContainer>& scene_graph_container, const int cluster_id,
                               std::shared_ptr<ReconstructionManager>& reconstruction_manager) {
    std::string cluster_path = StringPrintf("%s/cluster%d", workspace_path.c_str(), cluster_id);
    if (boost::filesystem::exists(cluster_path)) {
        boost::filesystem::remove_all(cluster_path);
    }
    boost::filesystem::create_directories(cluster_path);

    std::shared_ptr<IndependentMapperOptions> mapper_options = std::make_shared<IndependentMapperOptions>(options);

    if (mapper_options->independent_mapper_type == IndependentMapperType::DIRECTED){
        DirectedMapperController direct_mapper(mapper_options, image_path, cluster_path, scene_graph_container,
                                                    reconstruction_manager);
        direct_mapper.Start();
        direct_mapper.Wait();
    } else {
        IncrementalMapperController incremental_mapper(mapper_options, image_path, cluster_path, scene_graph_container,
                                                    reconstruction_manager);

        incremental_mapper.Start();
        incremental_mapper.Wait();
    }

    if (reconstruction_manager && reconstruction_manager->Size() > 0) {
        for (size_t i = 0; i < reconstruction_manager->Size(); ++i) {
            std::string rec_path = StringPrintf("%s/%d", cluster_path.c_str(), i);
            if (!boost::filesystem::exists(rec_path)) {
                boost::filesystem::create_directories(rec_path);
            }
            auto reconstruction = reconstruction_manager->Get(i);
            reconstruction->WriteReconstruction(rec_path, options.write_binary_model);
        }
    }

}

void ClusterNeighborsFinalBundleAdjust(const std::shared_ptr<SceneGraphContainer> scene_graph_container,
                         const std::unordered_map<image_t, std::set<image_t>> &image_neighbor_between_cluster,
                         const IndependentMapperOptions& mapper_options,
                         std::shared_ptr<Reconstruction>& reconstruction){
    
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
                         std::shared_ptr<Reconstruction>& reconstruction){
    
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
            if (neighbor_image_ids.find(image_id) != neighbor_image_ids.end()){
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

void RefineSepareteRig(const std::shared_ptr<SceneGraphContainer> scene_graph_container,
                       std::shared_ptr<Reconstruction>& reconstruction,
                       const IndependentMapperOptions& mapper_options,
                       const std::vector<std::vector<image_t>> &cluster_image_ids,
                       const std::string workspace_path){
    PrintHeading1("\nRefineSepareteRig\n");
    int num_local_camera = 1;
    for (auto image_id : reconstruction->RegisterImageIds()) {
        const class Image & image = reconstruction->Image(image_id);
        const class Camera & camera = reconstruction->Camera(image.CameraId());
        num_local_camera = std::max(camera.NumLocalCameras(), num_local_camera);
    }
    if(num_local_camera == 1){
        return;
    }
    std::shared_ptr<SceneGraphContainer> rig_scene_graph = std::make_shared<SceneGraphContainer>();
    scene_graph_container->ConvertRigSceneGraphContainer(*rig_scene_graph.get(), reconstruction->RegisterImageSortIds());
    std::cout << "Rig SceneGraph: " << rig_scene_graph->NumImages() << std::endl;

    auto rig_reconstruction = std::make_shared<Reconstruction>();
    const auto image_ids_map = reconstruction->ConvertRigReconstruction(*rig_reconstruction.get());
    std::cout << "Register Images: " << rig_reconstruction->NumRegisterImages() 
        << ", Ori_Images: " << reconstruction->NumRegisterImages() << std::endl;
    
    const auto neighbors = rig_scene_graph->CorrespondenceGraph()->ImageNeighbors();

    std::vector<std::vector<image_t>> rig_cluster_image_ids;
    rig_cluster_image_ids.resize(cluster_image_ids.size());
    std::cout << "rig_cluster_image_ids (" << rig_cluster_image_ids.size() << "): " << std::endl;;
    for (size_t cluster_id = 0; cluster_id < cluster_image_ids.size(); cluster_id++){
        rig_cluster_image_ids.at(cluster_id).reserve(cluster_image_ids.at(cluster_id).size() * num_local_camera);
        for (const auto image_id : cluster_image_ids.at(cluster_id)){
            for (const auto rig_id : image_ids_map.at(image_id)){
                rig_cluster_image_ids.at(cluster_id).push_back(rig_id);
            }
        }
        rig_cluster_image_ids.at(cluster_id).shrink_to_fit();
        std::cout << "=> cluster - " << cluster_id << "("<< rig_cluster_image_ids.at(cluster_id).size() << "): ";
        for (size_t i = 0; i < rig_cluster_image_ids.at(cluster_id).size(); i++){
            std::cout << rig_cluster_image_ids.at(cluster_id).at(i) << ", ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    //convert rtk
    struct GpsInfo{
        Eigen::Vector4d qvec;
        Eigen::Vector3d center;
        double std_lon, std_lat, std_hgt, std_ori;
        bool flag;
    };
    std::unordered_map<image_t, GpsInfo> rtk_image;
    bool rect_has_gps_prior = reconstruction->has_gps_prior;
    std::cout << "reconstruction->has_gps_prior: " << reconstruction->has_gps_prior << std::endl;
    {
        std::cout << "set rtk info ..." << std::endl;
        float points_threshold = 300;
        for (const auto image_id : rig_reconstruction->RegisterImageIds()){
            auto& image = rig_reconstruction->Image(image_id);
            
            rtk_image[image_id].qvec = image.QvecPrior();
            rtk_image[image_id].center = image.TvecPrior();
            rtk_image[image_id].std_lon = image.RtkStdLon();
            rtk_image[image_id].std_lat = image.RtkStdLat();
            rtk_image[image_id].std_hgt = image.RtkStdHgt();
            rtk_image[image_id].std_ori = image.OrientStd();
            rtk_image[image_id].flag = image.RtkFlag();

            image.SetQvecPrior(image.Qvec());
            image.SetTvecPrior(image.ProjectionCenter());

            float num_pints = image.NumMapPoints();
            float weig_factor = std::min((num_pints + 1) / points_threshold, 1.0f);
            float weig_std = 1 - std::cos(weig_factor);
            image.SetRtkStd(weig_std, weig_std, weig_std);
            image.SetOrientStd(weig_std);
            image.RtkFlag() = (int8_t)50;
        }
        rig_reconstruction->has_gps_prior = true;
        rig_reconstruction->AlignWithPriorLocations();
        std::cout << "Done" << std::endl;
    }

    // mapper
    IndependentMapperOptions inde_options = mapper_options;
    inde_options.complete_max_reproj_error = 12.0;
    inde_options.merge_max_reproj_error = 12;
    std::shared_ptr<IncrementalMapper> rig_mapper = std::make_shared<IncrementalMapper>(rig_scene_graph);
    rig_mapper->SetWorkspacePath(workspace_path);
    rig_mapper->BeginReconstruction(rig_reconstruction);

    BundleAdjustmentOptions ba_options = inde_options.GlobalBundleAdjustment();
    ba_options.use_prior_absolute_location = true;
    if (ba_options.prior_absolute_orientation_weight < 1e-6){
        ba_options.prior_absolute_orientation_weight = 0.01;
    }

    ba_options.solver_options.minimizer_progress_to_stdout = true;
    ba_options.solver_options.max_num_iterations = 20;
    ba_options.solver_options.max_linear_solver_iterations = 100;

    ba_options.refine_focal_length = false;
    ba_options.refine_principal_point = false;
    ba_options.refine_extra_params = false;
    ba_options.refine_local_extrinsics = false;
    for (size_t cluster_id = 0; cluster_id < rig_cluster_image_ids.size(); cluster_id++){
        std::cout << "\n cluster - " << cluster_id << std::endl;
        PrintHeading1("Retriangulation");
        CompleteAndMergeTracks(inde_options, rig_mapper);

        size_t num_retriangulate_observations = 
            rig_mapper->Retriangulate(mapper_options.Triangulation());
        std::cout << "  => Retriangulated observations: " << num_retriangulate_observations
                << std::endl;

        // PrintHeading1("BundleAdjust Optimization of Separate Cameras");
        std::unordered_set<image_t> t_image_ids(
            rig_cluster_image_ids[cluster_id].begin(), 
            rig_cluster_image_ids[cluster_id].end());

        BundleAdjustmentConfig ba_config;
        for (auto image_id : rig_cluster_image_ids[cluster_id]){
            if (!rig_reconstruction->ExistsImage(image_id)){
                continue;
            }
            const Image& image = rig_reconstruction->Image(image_id);
            ba_config.AddImage(image_id);
            ba_config.SetConstantCamera(image.CameraId());
            for (auto neighbor : neighbors.at(image_id)){
                if (!rig_reconstruction->ExistsImage(neighbor) || 
                    t_image_ids.find(neighbor) != t_image_ids.end()){
                    continue;
                }
                const Image& neighbor_image = rig_reconstruction->Image(neighbor);
                ba_config.AddImage(neighbor);
                ba_config.SetConstantPose(neighbor);
                ba_config.SetConstantCamera(neighbor_image.CameraId());
            }
        }
        std::cout << "ba_config: num_images, num_const_images, num_camrea:" 
            << ba_config.NumImages() << ", " << ba_config.NumConstantPoses() 
            << ", " << ba_config.NumConstantCameras() << std::endl;

        for (int iter = 0; iter < 2; ++iter) {
            rig_reconstruction->FilterObservationsWithNegativeDepth();
            const size_t num_observations = rig_reconstruction->ComputeNumObservations();

            PrintHeading1("GBA Bundle adjustment");
            std::cout << "iter: " << iter << std::endl;
            rig_reconstruction->AlignWithPriorLocations();

            BundleAdjuster bundle_adjuster(ba_options, ba_config);
            CHECK(bundle_adjuster.Solve(rig_reconstruction.get()));

            size_t num_changed_observations = 0;
            num_changed_observations += CompleteAndMergeTracks(mapper_options, rig_mapper);
            num_changed_observations += FilterPoints(mapper_options, rig_mapper, mapper_options.min_track_length);
            const double changed = static_cast<double>(num_changed_observations) / num_observations;
            std::cout << StringPrintf("  => Changed observations: %.6f", changed) << std::endl;

            num_retriangulate_observations = 0;
            num_retriangulate_observations = rig_mapper->Retriangulate(mapper_options.Triangulation());
            std::cout << "\nnum_retri_observations / num_ori_observations: "
                        << num_retriangulate_observations << " / "
                        << num_observations << std::endl;

            if (changed < mapper_options.ba_global_max_refinement_change) {
                break;
            }
        }
    }

    // reset rtk
    {
        std::cout << "reset rtk info ..." << std::endl;
        for (const auto image_id : rig_reconstruction->RegisterImageIds()){
            auto& image = rig_reconstruction->Image(image_id);
            image.SetQvecPrior(rtk_image[image_id].qvec);
            image.SetTvecPrior(rtk_image[image_id].center);
            image.SetRtkStd(rtk_image[image_id].std_lon, 
                            rtk_image[image_id].std_lat, 
                            rtk_image[image_id].std_hgt);
            image.SetOrientStd(rtk_image[image_id].std_ori);
            image.RtkFlag() = rtk_image[image_id].flag;
        }
        rig_reconstruction->has_gps_prior = rect_has_gps_prior;
        std::cout << "Done" << std::endl;
    }

    reconstruction = rig_reconstruction;
    std::cout << "rig_reconstruction, reconstruction: " << rig_reconstruction->NumRegisterImages() 
        << ", " << reconstruction->NumRegisterImages() << std::endl;
}

// Write initial global transforms, relative transforms, ordered clusters from motion averager
void WriteInitialTransform(const std::string path, 
                           EIGEN_STL_UMAP(cluster_t, Eigen::Matrix3x4d) &global_transforms, 
                           EIGEN_STL_UMAP(cluster_pair_t, Eigen::Matrix3x4d)& relative_transforms,
                           std::vector<std::vector<cluster_t>>&  clusters_ordered){
    std::ofstream file(path, std::ios::binary);
	CHECK(file.is_open()) << path;

    std::cout << "WriteInitialTransform path: " << path << std::endl;

    // write clusters_ordered
    size_t num_clusters_ordered = clusters_ordered.size();
    std::cout << "write num_clusters_ordered: " << num_clusters_ordered << std::endl;
    file.write((char*)&num_clusters_ordered, sizeof(size_t));
    for (auto clusters : clusters_ordered) {
        size_t num_clusters = clusters.size();
        std::cout << "write num_clusters: " << num_clusters << std::endl;
        file.write((char*)&num_clusters, sizeof(size_t));
        for (auto cluster_id : clusters) {
            file.write((char*)&cluster_id, sizeof(cluster_t));
            std::cout << "write cluster_id: " << cluster_id << std::endl;
        }
    }

    // write global transforms
    size_t num_global_transforms = global_transforms.size();
    std::cout << "write num_global_transforms: " << num_global_transforms << std::endl;
    file.write((char*)&num_global_transforms, sizeof(size_t));
    for (auto& cluster : global_transforms) {
        std::cout << "write cluster.first: " << cluster.first << std::endl;
        file.write((char*)&cluster.first, sizeof(cluster_t));
        for (int i = 0; i < 3; ++i) {
            std::cout << "write cluster.second: " << cluster.second(i, 0) << " " << cluster.second(i, 1) << " "
                      << cluster.second(i, 2) << " " << cluster.second(i, 3) << " " << std::endl;
            file.write((char*)&cluster.second(i, 0), sizeof(double));
            file.write((char*)&cluster.second(i, 1), sizeof(double));
            file.write((char*)&cluster.second(i, 2), sizeof(double));
            file.write((char*)&cluster.second(i, 3), sizeof(double));
        }
    }

    // write relative transforms
    size_t num_relative_transforms = relative_transforms.size();
    std::cout << "write num_relative_transforms: " << num_relative_transforms << std::endl;
    file.write((char*)&num_relative_transforms, sizeof(size_t));
    for (auto& cluster : relative_transforms) {
        std::cout << "write cluster.first: " << cluster.first << std::endl;
        file.write((char*)&cluster.first, sizeof(cluster_pair_t));
        for (int i = 0; i < 3; ++i) {
             std::cout << "write cluster.second: " << cluster.second(i, 0) << " " << cluster.second(i, 1) << " "
                      << cluster.second(i, 2) << " " << cluster.second(i, 3) << " " << std::endl;
            file.write((char*)&cluster.second(i, 0), sizeof(double));
            file.write((char*)&cluster.second(i, 1), sizeof(double));
            file.write((char*)&cluster.second(i, 2), sizeof(double));
            file.write((char*)&cluster.second(i, 3), sizeof(double));
        }
    }
}

// Read initial global transforms, relative transforms, ordered clusters from motion averager
void ReadInitialTransform(const std::string path, 
                           EIGEN_STL_UMAP(cluster_t, Eigen::Matrix3x4d) &global_transforms, 
                           EIGEN_STL_UMAP(cluster_pair_t, Eigen::Matrix3x4d)& relative_transforms,
                           std::vector<std::vector<cluster_t>>&  clusters_ordered){
    std::ifstream file(path, std::ios::binary);
	CHECK(file.is_open()) << path;

    std::cout << "ReadInitialTransform path: " << path << std::endl;

    // read clusters_ordered
    size_t num_clusters_ordered;
    file.read(reinterpret_cast<char*>(&num_clusters_ordered), sizeof(size_t));
    std::cout << "read num_clusters_ordered: " << num_clusters_ordered << std::endl;
    clusters_ordered.resize(num_clusters_ordered);
    for (size_t i = 0; i < num_clusters_ordered; ++i) {
        size_t num_clusters;
        file.read(reinterpret_cast<char*>(&num_clusters), sizeof(size_t));
        std::cout << "read num_clusters: " << num_clusters << std::endl;
        clusters_ordered[i].resize(num_clusters);
        for (size_t j = 0; j < num_clusters; ++j) {
            file.read(reinterpret_cast<char*>(&clusters_ordered[i][j]), sizeof(cluster_t));
            std::cout << "read cluster_id: " << clusters_ordered[i][j] << std::endl;
        }
    }

    // read global_transforms
    size_t num_global_transforms;
    file.read(reinterpret_cast<char*>(&num_global_transforms), sizeof(size_t));
    // std::cout << "read num_global_transforms: " << num_global_transforms << std::endl;
    for (size_t i = 0; i < num_global_transforms; ++i) {
        cluster_t cluster_id;
        file.read(reinterpret_cast<char*>(&cluster_id), sizeof(cluster_t));
        // std::cout << "write cluster.first: " << cluster_id << std::endl;
        Eigen::Matrix3x4d transform = Eigen::MatrixXd::Identity(3, 4);
        for (int i = 0; i < 3; ++i) {
            file.read(reinterpret_cast<char*>(&transform(i, 0)), sizeof(double));
            file.read(reinterpret_cast<char*>(&transform(i, 1)), sizeof(double));
            file.read(reinterpret_cast<char*>(&transform(i, 2)), sizeof(double));
            file.read(reinterpret_cast<char*>(&transform(i, 3)), sizeof(double));
            // std::cout << "read cluster.second: " << transform(i, 0) << " " << transform(i, 1) << " "
            //           << transform(i, 2) << " " << transform(i, 3) << " " << std::endl;
        }
        global_transforms[cluster_id] = transform;
    }

    // read relative_transforms
    size_t num_relative_transforms;
    file.read(reinterpret_cast<char*>(&num_relative_transforms), sizeof(size_t));
    // std::cout << "read num_relative_transforms: " << num_relative_transforms << std::endl;
    for (size_t i = 0; i < num_relative_transforms; ++i) {
        cluster_pair_t cluster_pair_id;
        file.read(reinterpret_cast<char*>(&cluster_pair_id), sizeof(cluster_pair_t));
        // std::cout << "read cluster.first: " << cluster_pair_id << std::endl;
        Eigen::Matrix3x4d transform = Eigen::MatrixXd::Identity(3, 4);
        for (int i = 0; i < 3; ++i) {
            file.read(reinterpret_cast<char*>(&transform(i, 0)), sizeof(double));
            file.read(reinterpret_cast<char*>(&transform(i, 1)), sizeof(double));
            file.read(reinterpret_cast<char*>(&transform(i, 2)), sizeof(double));
            file.read(reinterpret_cast<char*>(&transform(i, 3)), sizeof(double));
            // std::cout << "read cluster.second: " << transform(i, 0) << " " << transform(i, 1) << " "
            //           << transform(i, 2) << " " << transform(i, 3) << " " << std::endl;
        }
        relative_transforms[cluster_pair_id] = transform;
    }
}

void WriteOptimizedReconstructions(const std::string path, const int index, 
                                   const std::shared_ptr<Reconstruction>& reconstruction){
    std::string pose_save_path = JoinPaths(JoinPaths(path,"/cluster" + std::to_string(index)+"/0"),"/pose_trans.txt");
    std::cout << "write pose path: " << pose_save_path << std::endl;
    reconstruction->WritePoseText(pose_save_path);
}

// Merge Communities using max spanning tree and cluster motion average
void MergeCommunities(const std::unordered_map<int, std::shared_ptr<ReconstructionManager>>& reconstruction_managers,
                      const std::shared_ptr<SceneGraphContainer> scene_graph_container,
                      std::shared_ptr<ReconstructionManager>& root_reconstruction_manager,
                      ClusterMergeOptimizer::ClusterMergeOptions merge_options,
                      const std::shared_ptr<ClusterMapperOptions> options,
                      std::unordered_map<cluster_t, Reconstruction>& merged_reconstructions,
                      EIGEN_STL_UMAP(cluster_t, Eigen::Matrix3x4d) & global_transforms,
                      const std::string workspace_path) {
    std::cout << "Merge Conmunities" << std::endl<<std::endl;
    merged_reconstructions.clear();
    std::vector<std::shared_ptr<Reconstruction>> reconstructions;
    reconstructions.resize(reconstruction_managers.size());
    // For Reconstruction Manager of each community
    for (const auto& recon_manager : reconstruction_managers) {
        // For each reconstruction in the manager
        for (size_t i = 0; i < recon_manager.second->Size(); i++) {
            // // -- Skip small reconstruction
            // if (recon_manager.second->Get(i)->NumMapPoints() < 1000) {
            //     reconstructions[recon_manager.first] = std::shared_ptr<Reconstruction>(new Reconstruction());
            //     cluster_image_ids[recon_manager.first] = std::vector<image_t>();
            //     continue;
            // }
            // -- Push all the reconstruction results in the reconstructions
            reconstructions[recon_manager.first] = recon_manager.second->Get(i);
            merged_reconstructions[recon_manager.first] = *reconstructions[recon_manager.first].get();
        }
    }
    if (reconstructions.size() == 0) {
        return;
    }

    ClusterMotionAverager cluster_motion_averager(merge_options.debug_info, merge_options.save_strong_pairs, 
                                                merge_options.strong_pairs_path, merge_options.candidate_strong_pairs_num,
                                                merge_options.load_strong_pairs);
    const CorrespondenceGraph* full_correspondence_graph = scene_graph_container->CorrespondenceGraph().get();
    cluster_motion_averager.SetGraphs(full_correspondence_graph);

    EIGEN_STL_UMAP(cluster_pair_t, Eigen::Matrix3x4d) relative_transforms;
    std::vector<std::vector<cluster_t>> clusters_ordered;

    if(merge_options.load_initial_transform){
        std::cout << "Load Initial Transform" << std::endl;
        ReadInitialTransform(merge_options.initial_transform_path, global_transforms, relative_transforms,
                             clusters_ordered);
    }
    if(options->enable_motion_averager){
        clusters_ordered.clear();
        Timer motion_averager_timer;
        motion_averager_timer.Start();
        cluster_motion_averager.ClusterMotionAverage(
            reconstructions,  // -- Calculate the global transform using the given reconstructions
            global_transforms, relative_transforms, clusters_ordered);
        std::cout << "Motion Average Cost " << motion_averager_timer.ElapsedMinutes() << " [min]" << std::endl;
    }
    if(merge_options.save_initial_transform){
        std::cout << "Save Initial Transform" << std::endl;
        WriteInitialTransform(merge_options.initial_transform_path, global_transforms, relative_transforms,
                              clusters_ordered);
    }

    if (merge_options.debug_info){
        std::cout << "Debug Info[motion average]: clusters_ordered size: " << clusters_ordered.size() << std::endl;
        for (size_t component_idx = 0; component_idx < clusters_ordered.size(); ++component_idx) {
            Reconstruction rect_t = *reconstructions[clusters_ordered[component_idx][0]];
            std::shared_ptr<Reconstruction> ref_reconstruction = std::make_shared<Reconstruction>(rect_t);

            for (size_t i = 1; i < clusters_ordered[component_idx].size(); ++i) {
                CHECK(global_transforms.find(clusters_ordered[component_idx][i]) != global_transforms.end());
                ref_reconstruction->Merge(*(reconstructions[clusters_ordered[component_idx][i]]),
                                          global_transforms.at(clusters_ordered[component_idx][i]), 8.0);
            }
            std::string save_path = JoinPaths(workspace_path, "motion_averge_" + std::to_string(component_idx));
            if (!boost::filesystem::exists(save_path)) {
                boost::filesystem::create_directories(save_path);
            }
            ref_reconstruction->WriteBinary(save_path);
            std::cout << "save MotionAverge Reconstruction-" << component_idx << " to " << save_path << std::endl;
        }
    }

    if (options->enable_pose_graph_optimization) {
        std::shared_ptr<Reconstruction> reconstruction;

        // if(!merge_options.only_load_optimized_reconstructions){
            // Pose graph optimization
        merge_options.optimized_reconsturctions_path = workspace_path;
        std::cout<<"merge_options->clusters_close_images_distance "<<merge_options.clusters_close_images_distance<<std::endl;
        std::cout<<"optimerge_optionsons_->max_iter_time "<<merge_options.max_iter_time<<std::endl;
        std::cout<<"optimerge_optionsons_->use_prior_relative_pose "<<merge_options.use_prior_relative_pose<<std::endl;
        std::cout<<"optimerge_optionsons_->optimized_reconsturctions_path "<<merge_options.optimized_reconsturctions_path<<std::endl;
        ClusterMergeOptimizer clustermerge(std::make_shared<ClusterMergeOptimizer::ClusterMergeOptions>(merge_options),
                                        full_correspondence_graph);

        // Sort the clusters_ordered by cluster size
        std::sort(
            clusters_ordered.begin(), clusters_ordered.end(),
            [](const std::vector<cluster_t>& v1, const std::vector<cluster_t>& v2) { return v1.size() > v2.size(); });
        
        auto MergeRecon = [&](size_t clusters_index) {
            PrintHeading1(StringPrintf("Merge Reconcstruction Index = %d", clusters_index));

            auto cluster_ordered = clusters_ordered[clusters_index];
            if (cluster_ordered.size() == 1){
                auto rect_idx = root_reconstruction_manager->Add();
                *root_reconstruction_manager->Get(rect_idx).get() = *reconstructions[cluster_ordered[0]];
                return;
            }

            std::vector<std::vector<image_t>> cluster_image_ids;
            cluster_image_ids.resize(cluster_ordered.size());

            std::cout << "cluster_ordered (" << cluster_ordered.size() << "): " << std::endl;
            for (int i = 0; i < cluster_ordered.size(); i++){
                cluster_image_ids[i] = reconstructions[cluster_ordered[i]]->RegisterImageIds();
                std::cout << "cluster-" << cluster_ordered[i] << cluster_image_ids[i].size() << " images, gps:" 
                    << reconstructions[cluster_ordered[i]]->has_gps_prior 
                    << ", aligned:" << reconstructions[cluster_ordered[i]]->b_aligned << std::endl;
            }
            std::cout << std::endl;

            Timer cluster_merge_timer;
            cluster_merge_timer.Start();
            clustermerge.MergeByPoseGraph(reconstructions, global_transforms, relative_transforms, cluster_ordered,
                                        reconstruction);
            std::cout << "Pose Graph cluster merge cost " << cluster_merge_timer.ElapsedMinutes() << " [min]" << std::endl;
            
            if (merge_options.debug_info){
                std::string save_path = JoinPaths(workspace_path, "pose_graph_" + std::to_string(clusters_index));
                if (!boost::filesystem::exists(save_path)) {
                    boost::filesystem::create_directories(save_path);
                }
                reconstruction->WriteBinary(save_path);
                std::cout << "save PoseGraph Reconstruction-" << clusters_index << " to " << save_path << std::endl;
            }

            if (merge_options.enable_final_ba) {
                //  Preform Global BA optimization
                PrintHeading1("Final Global bundle adjustment");

                std::cout << "filter_max_reproj_error:" << options->IndependentMapper().filter_max_reproj_error << std::endl;
                std::cout << "filter_min_tri_angle:" << options->IndependentMapper().filter_min_tri_angle << std::endl;
                std::cout << "filter_max_reproj_error_final:" << options->IndependentMapper().filter_max_reproj_error_final << std::endl;
                std::cout << "filter_min_tri_angle_final:" << options->IndependentMapper().filter_min_tri_angle_final << std::endl;
                std::cout << "filter_min_track_length_final:" << options->IndependentMapper().filter_min_track_length_final << std::endl;

                {
                    float mem;
                    sensemap::GetAvailableMemory(mem);
                    std::cout << "memory left : " << mem << std::endl;
                    std::cout << malloc_trim(0) << std::endl;
                    sensemap::GetAvailableMemory(mem);
                    std::cout << "memory left : " << mem << std::endl;
                }
                
                const auto& image_neighbor_between_cluster = clustermerge.GetImageNeighborBetweenCluster();
                ClusterNeighborsFinalBundleAdjust(
                    scene_graph_container, 
                    image_neighbor_between_cluster, 
                    options->IndependentMapper(),
                    reconstruction);
                ClusterIsolatedFinalBundleAdjust(
                    scene_graph_container, 
                    cluster_image_ids, 
                    image_neighbor_between_cluster, 
                    options->IndependentMapper(),
                    reconstruction);
                ClusterNeighborsFinalBundleAdjust(
                    scene_graph_container, 
                    image_neighbor_between_cluster, 
                    options->IndependentMapper(),
                    reconstruction);
                if (options->merge_options.refine_separate_cameras){
                    if (merge_options.debug_info){
                        std::string save_path = 
                            JoinPaths(workspace_path, "before_separete_rig" + std::to_string(clusters_index));
                        if (!boost::filesystem::exists(save_path)) {
                            boost::filesystem::create_directories(save_path);
                        }
                        reconstruction->WriteBinary(save_path);
                        std::cout << "save Before Separete Rig Reconstruction-" << clusters_index << " to " << save_path << std::endl;
                    }
                    RefineSepareteRig(scene_graph_container, 
                        reconstruction, 
                        options->IndependentMapper(),
                        cluster_image_ids, 
                        workspace_path);
                }
                std::cout << "\nAlignWithPriorLocations..." << std::endl;
                reconstruction->AlignWithPriorLocations();
            }
            
            std::cout << "reconstruction: " << reconstruction->NumRegisterImages() << std::endl;
            auto rect_idx = root_reconstruction_manager->Add();
            *root_reconstruction_manager->Get(rect_idx).get() = std::move(*reconstruction.get());
        };

        for (size_t idx = 0; idx < clusters_ordered.size(); idx++){
            MergeRecon(idx);
            if (!options->multiple_models){
                std::cout << "multiple_models : 0, break..." << std::endl;
                break;
            }
        }

    } else {  // merge reconstructions with global transform from motion average directly
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
}

void WriteTransforms(const std::string workspace_path,
                     const EIGEN_STL_UMAP(cluster_t, Eigen::Matrix3x4d) global_transforms) {
    std::string trans_path = JoinPaths(workspace_path, "cluster-trans.txt");
    std::ofstream file(trans_path, std::ios::out);
    if (!file.is_open()) {
        std::cerr << StringPrintf("Fail to create file %s\n", trans_path);
        return;
    }

    for (auto& cluster : global_transforms) {
        file << cluster.first << std::endl;
        for (int i = 0; i < 3; ++i) {
            file << cluster.second(i, 0) << " " << cluster.second(i, 1) << " " << cluster.second(i, 2) << " "
                 << cluster.second(i, 3) << std::endl;
        }
    }

    file.close();
}

}  // namespace

ClusterMapperController::ClusterMapperController(const std::shared_ptr<ClusterMapperOptions> options,
                                                 const std::string& workspace_path, const std::string& image_path,
                                                 const std::shared_ptr<SceneGraphContainer> scene_graph_container,
                                                 std::shared_ptr<ReconstructionManager> reconstruction_manager)
    : options_(options),
      workspace_path_(workspace_path),
      image_path_(image_path),
      scene_graph_container_(scene_graph_container),
      reconstruction_manager_(reconstruction_manager) {}

void ClusterMapperController::Run() {
    scene_clustering_ = std::shared_ptr<SceneClustering>(new SceneClustering(options_->ClusterMapper()));

    std::shared_ptr<CorrespondenceGraph> correspondence_graph = scene_graph_container_->CorrespondenceGraph();

    // Merge the cluster only
    if (options_->only_merge_cluster) {

        std::unordered_map<int, std::shared_ptr<ReconstructionManager>> reconstruction_managers;
        std::cout<<"Total reconstructions size "<<reconstruction_manager_->Size() << std::endl;
        cluster_t cluster_id = 0;
        for (size_t i = 0; i < reconstruction_manager_->Size(); ++i) {
            auto reconstruction_manager = std::make_shared<ReconstructionManager>();
            if (reconstruction_manager_->Get(i)->NumMapPoints() < 1000) {
                continue;
            }
            auto component_idx = reconstruction_manager->Add();
            *reconstruction_manager->Get(component_idx).get() = std::move(*reconstruction_manager_->Get(i));

            std::vector<image_t> registered_images = reconstruction_manager->Get(component_idx)->RegisterImageIds();

            std::cout<<"Reconstruction "<<cluster_id<<" has "<<registered_images.size()<<" images"<<std::endl;
            std::set<image_t> registered_image_id_set(registered_images.begin(), registered_images.end());

            std::cout << "Create cluster graph container" << std::endl;
            std::shared_ptr<SceneGraphContainer> cluster_graph_container =
                std::shared_ptr<SceneGraphContainer>(new SceneGraphContainer());
            scene_graph_container_->ClusterSceneGraphContainer(registered_image_id_set, *cluster_graph_container.get());
            std::cout << "Create cluster graph container done " << std::endl;
            reconstruction_manager->Get(component_idx)->SetUp(cluster_graph_container);
            reconstruction_manager->Get(component_idx)->TearDown();
            cluster_graph_container.reset();
            reconstruction_managers.emplace(cluster_id, reconstruction_manager);
            cluster_id++;
        }
        reconstruction_manager_->Clear();

        options_->merge_options.debug_info = options_->mapper_options.debug_info;

        EIGEN_STL_UMAP(cluster_t, Eigen::Matrix3x4d) global_transforms;
        std::unordered_map<cluster_t, Reconstruction> reconstructions;

        MergeCommunities(reconstruction_managers, scene_graph_container_,
                         reconstruction_manager_, options_->merge_options, options_, reconstructions, global_transforms,
                         workspace_path_);

        WriteTransforms(workspace_path_, global_transforms);
#if 0
        // Transform cluster.
        for (auto reconstruction_map : reconstructions) {
            size_t community_id = reconstruction_map.first;
            auto reconstruction = reconstruction_map.second;

            std::cout << "cluster_id " << community_id << "  has Images " << reconstruction.NumImages() << std::endl;

            auto trans = global_transforms.at(community_id);
            std::cout<<"transfromation is "<<std::endl<<trans<<std::endl;
            reconstruction.TransformReconstruction(trans);

            std::shared_ptr<Reconstruction> keyframe_reconstruction = std::make_shared<Reconstruction>();
            std::unordered_set<image_t> keyframe_images; 
            std::unordered_set<mappoint_t> all_mappoints;

            if( options_->merge_options.fixed_original_reconstruction && community_id == options_->merge_options.original_reconstruction_id){
                auto old_keyframe_reconstruction = std::make_shared<Reconstruction>();
                std::string old_keyframe_path = JoinPaths(workspace_path_,"/0/KeyFrames");
                old_keyframe_reconstruction->ReadReconstruction(old_keyframe_path);
                std::vector<image_t> old_keyframes_v = old_keyframe_reconstruction->RegisterImageIds();
                std::unordered_set<image_t> old_keyframes(old_keyframes_v.begin(),old_keyframes_v.end());

                const std::vector<image_t> registered_image_ids =  reconstruction.RegisterImageIds();
                for(size_t i = 0; i<registered_image_ids.size(); ++i){
                    CHECK(reconstruction.ExistsImage(registered_image_ids[i]));
                    const auto& image = reconstruction.Image(registered_image_ids[i]);
                    if (old_keyframes.count(registered_image_ids[i])>0 ) {
                        keyframe_images.insert(registered_image_ids[i]);
                    }
                }
            }else{

                const std::vector<image_t> registered_image_ids =  reconstruction.RegisterImageIds();
                int keyframes_size = 0;
                for(size_t i = 0; i<registered_image_ids.size(); ++i){
                    CHECK(reconstruction.ExistsImage(registered_image_ids[i]));
                    const auto& image = reconstruction.Image(registered_image_ids[i]);
                    if(image.IsKeyFrame()){
                        keyframes_size++;
                    }
                    
                }
                if(keyframes_size){
                    for(size_t i = 0; i<registered_image_ids.size(); ++i){
                        CHECK(reconstruction.ExistsImage(registered_image_ids[i]));
                        const auto& image = reconstruction.Image(registered_image_ids[i]);
                        if(image.IsKeyFrame()){
                            keyframe_images.insert(registered_image_ids[i]);
                        }
                        
                    }
                }else{
                    for(size_t i = 0; i<registered_image_ids.size(); ++i){
                        CHECK(reconstruction.ExistsImage(registered_image_ids[i]));
                        const auto& image = reconstruction.Image(registered_image_ids[i]);
                        if(registered_image_ids[i]%options_->keyframe_gap ==0){
                            keyframe_images.insert(registered_image_ids[i]);
                        }
                    }
                }
                

            }

            all_mappoints = reconstruction.MapPointIds();

            reconstruction.Copy(keyframe_images,all_mappoints, keyframe_reconstruction);

            std::string align_rec_path = StringPrintf("%s/aligned-rec/%d", workspace_path_.c_str(), community_id);
            if (!boost::filesystem::exists(align_rec_path)) {
                boost::filesystem::create_directories(align_rec_path);
            }
            reconstruction.WriteBinary(align_rec_path);

            std::string rec_path = StringPrintf("%s/aligned-rec/%d/KeyFrames", workspace_path_.c_str(),community_id);  
            std::cout<<"write keyframe "<<rec_path<<std::endl;
            if (!boost::filesystem::exists(rec_path)) {
                boost::filesystem::create_directories(rec_path);
            }
            keyframe_reconstruction->WriteReconstruction(rec_path);
        }

        std::string after_ba_rec_path = StringPrintf("%s/map_update/0/", workspace_path_.c_str());
        if (!boost::filesystem::exists(after_ba_rec_path)) {
            boost::filesystem::create_directories(after_ba_rec_path);
        }
        reconstruction_manager_->Get(0)->WriteBinary(after_ba_rec_path);

        std::shared_ptr<Reconstruction> keyframe_reconstruction = std::make_shared<Reconstruction>();
        std::unordered_set<image_t> keyframe_images; 
        std::unordered_set<mappoint_t> all_mappoints;

        const std::vector<image_t> registered_image_ids =  reconstruction_manager_->Get(0)->RegisterImageIds();
        for(size_t i = 0; i<registered_image_ids.size(); ++i){
            CHECK(reconstruction_manager_->Get(0)->ExistsImage(registered_image_ids[i]));
            const auto& image = reconstruction_manager_->Get(0)->Image(registered_image_ids[i]);
            if(registered_image_ids[i]%options_->keyframe_gap ==0){
                keyframe_images.insert(registered_image_ids[i]);
            }
        }
        all_mappoints = reconstruction_manager_->Get(0)->MapPointIds();

        reconstruction_manager_->Get(0)->Copy(keyframe_images,all_mappoints, keyframe_reconstruction);
        std::string rec_path = StringPrintf("%s/map_update/0/KeyFrames", workspace_path_.c_str());  
            std::cout<<"write keyframe "<<rec_path<<std::endl;
        if (!boost::filesystem::exists(rec_path)) {
            boost::filesystem::create_directories(rec_path);
        }

        keyframe_reconstruction->WriteReconstruction(rec_path);
#endif
        return;
    }

    if (scene_graph_container_->NumImages() < options_->clustering_options.max_modularity_count) {
        std::vector<std::vector<image_t>> label_cluster(1);
        const auto image_ids = scene_graph_container_->GetImageIds();
        label_cluster.at(0).insert(label_cluster.at(0).begin(), image_ids.begin(), image_ids.end());

        std::vector<label_t> label_cluster_id(1,0);
        std::vector<std::unordered_set<uint32_t>> overlap_image_ids;
        std::cout << StringPrintf("Community %d have %d images", 1, label_cluster.at(0).size()) << std::endl;

        ReconstructCommunities(label_cluster, overlap_image_ids,label_cluster_id);
    }else if (options_->enable_image_label_cluster) {
        std::cout << "Cluster scene according to prior DirId..." << std::endl;
        // Generate cluster using the image label
        const EIGEN_STL_UMAP(image_t, class Image)& images = scene_graph_container_->Images();

        std::map<std::string, label_t> coarse_label_list;
        label_t coarse_label_index = 1;

        // Only Reconstruct Selected Cluster
        std::vector<std::string> reconstruct_image_name_list;
        if (options_->reconstruct_image_name_list != "" ) {
            reconstruct_image_name_list = StringSplit(options_->reconstruct_image_name_list, ",");

            std::cout << "Only Reconstruction Video Name: " << std::endl;
            for (const auto& image_name_pattern : reconstruct_image_name_list) {
                std::cout << "  " << image_name_pattern << std::endl;
            }
        }

        std::map<label_t, std::vector<image_t>> labelled_image_clusters;
        for (auto image_elem : images) {
            // Check the image has label or not
            // feature_data_container_->GetImage(image_id);
            auto image = image_elem.second;
            image_t image_id = image_elem.first;

            bool skip_image = false;
            for (const auto& image_name_pattern : reconstruct_image_name_list) {
                if (image.Name().find(image_name_pattern) == std::string::npos) {
                    skip_image = true;
                }
            }

            if (skip_image) {
                continue;
            }

            if (options_->enable_cluster_mapper_with_coarse_label) {
                std::string coarse_dir = image.Name().substr(0, image.Name().find("/"));
                if (coarse_label_list.find(coarse_dir) == coarse_label_list.end()) {
                    coarse_label_list.emplace(coarse_dir, coarse_label_index);
                    coarse_label_index++;
                }
                label_t current_label = coarse_label_list.at(coarse_dir);
                labelled_image_clusters[current_label].emplace_back(image_id);
            } else {
                if (!image.HasLabel()) {
                    std::cout << "Image do not have label !!!" << std::endl;
                    std::cout << "Image id : " << image.ImageId() << ", Image name : " << image.Name()
                              << ", Image label : " << image.LabelId() << std::endl;
                    CHECK(!image.HasLabel());
                }
                labelled_image_clusters[image.LabelId()].emplace_back(image_id);
            }
        }

        std::vector<std::pair<label_t, size_t>>  labeled_image_size;
        for (auto labelled_image_cluster : labelled_image_clusters) {
            auto pair = std::pair<label_t, size_t>(labelled_image_cluster.first, labelled_image_cluster.second.size());
            labeled_image_size.push_back(pair);
        }
        std::sort(labeled_image_size.begin(),labeled_image_size.end(),
            [&](const std::pair<label_t, size_t>& a,const std::pair<label_t, size_t>& b)->bool{
            return a.second > b.second;
        });

        std::vector<std::vector<image_t>> label_cluster;
        std::vector<label_t> label_cluster_id;
        int cluster_number = 0;
        for (size_t label_idx = 0; label_idx < labeled_image_size.size(); label_idx++){
            label_cluster.emplace_back(labelled_image_clusters[labeled_image_size[label_idx].first]);
            label_cluster_id.emplace_back(label_idx);
            cluster_number++;
            std::cout << "labeled size: " << label_idx << ", " << labelled_image_clusters[labeled_image_size[label_idx].first].size() << std::endl;
        }

        std::cout << "Cluster number  = " << cluster_number << std::endl;

        std::vector<std::unordered_set<uint32_t>> overlap_image_ids;

        size_t total_num_images = 0;
        for (size_t i = 0; i < label_cluster.size(); ++i) {
            total_num_images += label_cluster[i].size();
            std::cout << StringPrintf("  Community %d with %d images", i + 1, label_cluster[i].size()) << std::endl;
        }
        std::cout << StringPrintf("Communities have %d images", total_num_images) << std::endl;

        ReconstructCommunities(label_cluster, overlap_image_ids,label_cluster_id);
    } else {
        std::cout << "Community Detection..." << std::endl;
        std::vector<std::vector<image_t>> communities;
        std::vector<std::unordered_set<uint32_t>> overlap_image_ids;
        std::vector<label_t> label_cluster_id;

        options_->clustering_options.Print();
        std::set<uint32_t> valid_image_ids;
        fastcommunity::CommunityDetection(options_->clustering_options,
                                          scene_graph_container_,
                                          valid_image_ids,
                                          communities, 
                                          overlap_image_ids);
        std::sort(communities.begin(), communities.end(), 
            [](const std::vector<image_t> a, const std::vector<image_t> b){return a.size() > b.size(); });

        size_t total_num_images = 0;
        for (size_t i = 0; i < communities.size(); ++i) {
            total_num_images += communities[i].size();
            std::cout << StringPrintf("  Community %d with %d images", i + 1, communities[i].size()) << std::endl;
            label_cluster_id.emplace_back( i + 1);
        }
        std::cout << StringPrintf("Communities have %d images", total_num_images) << std::endl;

        // std::unordered_map<camera_t, std::unordered_set<size_t>> camera_cluster_map;
        std::unordered_map<camera_t, std::unordered_map<size_t, std::vector<image_t>>> camera_image_map;
        for (size_t i = 0; i < communities.size(); ++i){
            for (size_t j = 0; j < communities.at(i).size(); ++j){
                const auto& image = scene_graph_container_->Image(communities[i][j]);
                // camera_cluster_map[image.CameraId()].insert(i);
                camera_image_map[image.CameraId()][i].push_back(image.ImageId());
            }
        }

        // std::unordered_map<camera_t, std::vector<size_t>> camera_cluster_vec;
        for (const auto camera_cluster : camera_image_map){
            if (camera_cluster.second.size() < 2){
                continue;
            }

            std::vector<std::pair<size_t, size_t>> cluster_size;
            std::cout << "camera: " << camera_cluster.first << std::endl;
            for (const auto cluster_image : camera_cluster.second){
                cluster_size.push_back(std::make_pair(cluster_image.first, cluster_image.second.size()));
                std::cout << "\t=> cluster_image: " << cluster_image.first << ", image size: " << cluster_image.second.size() << std::endl;
            }
            std::sort(cluster_size.begin(), cluster_size.end(), 
                [](const std::pair<size_t, size_t> a, const std::pair<size_t, size_t> b){
                return a.second > b.second; });
            for (size_t k = 0; k < cluster_size.size() - 1; k++){
                if (cluster_size.at(k).first < cluster_size.at(k+1).first){
                    continue;
                }
                communities.at(cluster_size.at(k).first).swap(communities.at(cluster_size.at(k+1).first));
                std::cout << "\t=> swap " << cluster_size.at(k).first << " -> "<< cluster_size.at(k+1).first << std::endl;
            }
        }

        // for (size_t cluster_id = 0; cluster_id < communities.size(); ++cluster_id) {
        //     std::cout << "Cluster#" << cluster_id << " overlap image ids" << std::endl;
        //     for (const auto& overlap_image_id : overlap_image_ids[cluster_id]) {
        //         std::cout << overlap_image_id << " ";
        //     }
        //     std::cout << std::endl;
        // }

#if 0
        {
            auto image_ids  = scene_graph_container_->GetImageIds();
            std::sort(image_ids.begin(), image_ids.end());
            std::unordered_map<size_t, size_t> map_id2idx;

            std::vector<PlyPoint> locations;
            locations.reserve(image_ids.size());
            size_t max_id = 0;
            for (int i = 0; i < image_ids.size(); i++){
                map_id2idx[image_ids[i]] = i;
                const auto& image = scene_graph_container_->Image(image_ids[i]); 
                auto tvec = image.TvecPrior();
                Eigen::Matrix3d rot = QuaternionToRotationMatrix(image.QvecPrior());
                Eigen::Vector3d cam_ray(rot.row(2));

                PlyPoint point;
                point.x = tvec.x();
                point.y = tvec.y();
                point.z = tvec.z();
                point.nx = cam_ray.x();
                point.ny = cam_ray.y();
                point.nz = cam_ray.z();
                point.r = 0;
                point.g = 0;
                point.b = 0;
                locations.emplace_back(point);
                if (max_id < image_ids[i]){
                    max_id = image_ids[i];
                }
                // std::cout << "idx-id: " << i << ", " << image_ids[i] << std::endl;
            }
            std::cout << "num_images, max_id: " << image_ids.size() << ", " << max_id << std::endl;
            std::string SaveNeighborPlyPath = workspace_path_ + "/neighbors";
            
            const std::unordered_map<uint32_t, std::unordered_set<uint32_t> >&
                neighbors = correspondence_graph->ImageNeighbors();
            std::unordered_map<uint64_t, uint32_t> corrs_between_images =
                correspondence_graph->NumCorrespondencesBetweenImages();
            uint32_t step = image_ids.size() / 11 + 1;
            for (int i = 0; i < image_ids.size(); i += step){
                const auto image_id = image_ids[i];
                size_t num_max_corrs = 0;
                for (const auto neighbor : neighbors.at(image_id)){
                    auto pair_id = sensemap::utility::ImagePairToPairId(image_id, neighbor);
                    size_t num_corr = corrs_between_images.at(pair_id);
                    if (num_max_corrs < num_corr){
                        num_max_corrs = num_corr;
                    }
                }

                std::vector<PlyPoint> neighbor_locations = locations;
                std::vector<PlyPoint> neighbor_locations2 = locations;
                {
                    auto& point = neighbor_locations[map_id2idx[image_id]];
                    point.r = point.g = point.b = 255;
                    auto& point2 = neighbor_locations[map_id2idx[image_id]];
                    point2.r = point2.g = point2.b = 255;
                }
                
                for (const auto neighbor : neighbors.at(image_id)){
                    auto pair_id = sensemap::utility::ImagePairToPairId(image_id, neighbor);
                    size_t num_corr = corrs_between_images.at(pair_id);

                    auto& point = neighbor_locations[map_id2idx[neighbor]];
                    double grey = (double)num_corr / (double)num_max_corrs;
                    ColorMap(grey, point.r, point.g, point.b);


                    auto& point2 = neighbor_locations2[map_id2idx[neighbor]];
                    ColorMap(1.0, point2.r, point2.g, point2.b);
                }

                const auto& image = scene_graph_container_->Image(image_id); 
                const auto& base_name = GetPathBaseName(image.Name());
                std::string SavePlyPath = SaveNeighborPlyPath + "/" + std::to_string(image_id) + "_" + base_name + ".ply";
                if (!boost::filesystem::exists(GetParentDir(SavePlyPath))) {
                    boost::filesystem::create_directories(GetParentDir(SavePlyPath));
                }
                sensemap::WriteBinaryPlyPoints(SavePlyPath, neighbor_locations, true, true);

                std::string SavePlyPath2 = SaveNeighborPlyPath + "/" + std::to_string(image_id) + "_" + base_name + "_n.ply";
                sensemap::WriteBinaryPlyPoints(SavePlyPath2, neighbor_locations2, true, true);
                // std::cout << "WriteNeighborPlyPoints :" << SavePlyPath << std::endl;
            }
        }
#endif
	
        {   
            auto ExportClusterGraph = [&](std::vector<std::vector<image_t>>& com, std::string name){

                cv::RNG rng(12345);
                std::vector<std::vector<int>> cluster_color(com.size());
                std::map<image_t, size_t> id_2_cluster;
                for (size_t community_id = 0; community_id < com.size(); ++community_id){
                    int r_color = rng.uniform(0, 255);
                    int g_color = rng.uniform(0, 255);
                    int b_color = rng.uniform(0, 255);
                    cluster_color.at(community_id).push_back(r_color);
                    cluster_color.at(community_id).push_back(g_color);
                    cluster_color.at(community_id).push_back(b_color);
                    for (const auto image_id : com[community_id]){
                        id_2_cluster[image_id] = community_id;
                    }
                }
        
                const int max_graph_node = 50000;

                size_t min_val = std::numeric_limits<size_t>::max();
                size_t max_val = 0;
                for (auto & image_pair : correspondence_graph->ImagePairs()) {
                    size_t num_corres = correspondence_graph->NumCorrespondencesBetweenImages(image_pair.first);
                    min_val = std::min(min_val, num_corres);
                    max_val = std::max(max_val, num_corres);
                }

                size_t num_images = correspondence_graph->NumImages();
                std::vector<image_t> image_ids;
                image_ids.reserve(num_images);
                const auto& images = scene_graph_container_->Images();
                image_t max_image_id = 0;
                for (const auto & image : images) {
                    image_ids.push_back(image.first);
                    max_image_id = std::max(max_image_id, image.first);
                }
                std::sort(image_ids.begin(), image_ids.end());

                int step = 1;
                if (num_images > max_graph_node) {
                    step = (num_images - 1) / max_graph_node + 1;
                }

                std::unordered_map<image_t, int> image_id_to_idx;
                std::vector<bool> selected(max_image_id + 1, false);
                int num_image_pick = 0;
                for (int i = 0; i < num_images; i += step) {
                    selected[image_ids[i]] = true;
                    image_id_to_idx[image_ids[i]] = num_image_pick++;
                }

                if (num_images > max_graph_node) {
                    std::cout << StringPrintf("Compress Graph Node from %d to %d\n", num_images, num_image_pick);
                }

                Bitmap bitmap;
                bitmap.Allocate(num_image_pick, num_image_pick, true);

                const double max_value = std::log1p(max_val);
                const double bk_value = std::log1p(0) / max_value;
                const BitmapColor<float> bk_color(255 * JetColormap::Red(bk_value),
                                                255 * JetColormap::Green(bk_value),
                                                255 * JetColormap::Blue(bk_value));
                            
                bitmap.Fill(bk_color.Cast<uint8_t>());
                for (auto & image_pair : correspondence_graph->ImagePairs()) {
                    image_t image_id1, image_id2;
                    sensemap::utility::PairIdToImagePair(image_pair.first, 
                                                        &image_id1, 
                                                        &image_id2);
                    if (!selected[image_id1] || !selected[image_id2]) {
                        continue;
                    }
                    int idx1 = image_id_to_idx.at(image_id1);
                    int idx2 = image_id_to_idx.at(image_id2);

                    size_t num_corres = correspondence_graph->NumCorrespondencesBetweenImages(image_pair.first);

                    const double value = std::log1p(num_corres) / max_value;
                    const BitmapColor<float> color(255 * JetColormap::Red(value),
                                                    255 * JetColormap::Green(value),
                                                    255 * JetColormap::Blue(value));

                    if (id_2_cluster[idx1] = id_2_cluster[idx2]){
                        size_t community_id = id_2_cluster[idx1];
                        // const BitmapColor<float> color1(cluster_color.at(community_id)[0],
                        //                                cluster_color.at(community_id)[1],
                        //                                cluster_color.at(community_id)[2]);
                        
                        if (idx1 < idx2){
                            // bitmap.SetPixel(idx1, idx2, color1.Cast<uint8_t>());
                            bitmap.SetPixel(idx2, idx1, color.Cast<uint8_t>());
                        } else {
                            bitmap.SetPixel(idx1, idx2, color.Cast<uint8_t>());
                            // bitmap.SetPixel(idx2, idx1, color1.Cast<uint8_t>());
                        }
                    // } else {
                    //     size_t community_id1 = id_2_cluster[idx1];
                    //     size_t community_id2 = id_2_cluster[idx2];
                    //     const BitmapColor<float> color1(0,0,0);

                    //     if (idx1 < idx2){
                    //         // bitmap.SetPixel(idx1, idx2, color1.Cast<uint8_t>());
                    //         bitmap.SetPixel(idx2, idx1, color.Cast<uint8_t>());
                    //     } else {
                    //         bitmap.SetPixel(idx1, idx2, color.Cast<uint8_t>());
                    //         // bitmap.SetPixel(idx2, idx1, color1.Cast<uint8_t>());
                    //     }
                    }

                }
                
                // write cluster block
                for (size_t community_id = 0; community_id < com.size(); ++community_id){
                    const BitmapColor<float> color1(cluster_color.at(community_id)[0],
                                                    cluster_color.at(community_id)[1],
                                                    cluster_color.at(community_id)[2]);
                        
                    for (const auto image_id1 : com[community_id]){
                        int idx1 = image_id_to_idx.at(image_id1);

                        for (const auto image_id2 : com[community_id]){
                            if (image_id1 > image_id2){
                                continue;
                            }

                            int idx2 = image_id_to_idx.at(image_id2);
                            bitmap.SetPixel(idx1, idx2, color1.Cast<uint8_t>());
                        }
                    }
                }

                // std::string SaveWholePngPath = workspace_path_ + "/" + name + "/" + name + "-scenegraph.png";
                std::string SaveWholePngPath = workspace_path_ + "/" + name + "-scenegraph.png";
                if (!boost::filesystem::exists(GetParentDir(SaveWholePngPath))) {
                    boost::filesystem::create_directories(GetParentDir(SaveWholePngPath));
                }
                bitmap.Write(SaveWholePngPath);
            };
            ExportClusterGraph(communities, "merge-cluster");
        }

        {
            auto SaveCom = [&](std::vector<std::vector<image_t>>& com, std::string name){
                cv::RNG rng(12345);
                std::vector<PlyPoint> locations;

                for (size_t community_id = 0; community_id < com.size(); ++community_id) {
                    int r_color = rng.uniform(0, 255);
                    int g_color = rng.uniform(0, 255);
                    int b_color = rng.uniform(0, 255);
                    std::cout << "cluster - " << community_id << " color : " << r_color << ", "
                        << g_color << ", " << b_color << std::endl;
                    std::vector<PlyPoint> locations_cluster;
                    for (const auto image_id : com[community_id]){
                        const auto& image = scene_graph_container_->Image(image_id); 
                        if (image.HasTvecPrior() && image.HasTvecPrior()){
                            PlyPoint point;
                            auto tvec = image.TvecPrior();
                            Eigen::Matrix3d rot = QuaternionToRotationMatrix(image.QvecPrior());
                            Eigen::Vector3d cam_ray(rot.row(2));
                            point.x = tvec.x();
                            point.y = tvec.y();
                            point.z = tvec.z();
                            point.nx = cam_ray.x();
                            point.ny = cam_ray.y();
                            point.nz = cam_ray.z();
                            point.r = r_color;
                            point.g = g_color;
                            point.b = b_color;
                            locations_cluster.emplace_back(point);
                            locations.emplace_back(point);
                        }
                    }
                            
                    // std::string SavePlyPath = workspace_path_ + "/" + name + "/" + std::to_string(community_id) + ".ply";
                    // if (!boost::filesystem::exists(GetParentDir(SavePlyPath))) {
                    //     boost::filesystem::create_directories(GetParentDir(SavePlyPath));
                    // }
                    // sensemap::WriteBinaryPlyPoints(SavePlyPath, locations_cluster, true, true);
                    // std::cout << "WriteBinaryPlyPoints :" << SavePlyPath << std::endl;

                    // std::string SavePngPath = workspace_path_ + "/" + name + "/" + std::to_string(community_id) + ".png";
                    // std::set<uint32_t> scene_cluster_ids;
                    // scene_cluster_ids.insert(com[community_id].begin(), com[community_id].end());
                    // std::shared_ptr<SceneGraphContainer> cluster_graph_container =
                    //     std::shared_ptr<SceneGraphContainer>(new SceneGraphContainer());
                    // scene_graph_container_->ClusterSceneGraphContainer(scene_cluster_ids, *cluster_graph_container.get());
                    // cluster_graph_container->CorrespondenceGraph()->ExportToGraph(SavePngPath.c_str());
                }

                std::string SaveWholePlyPath = workspace_path_ + "/" + name + ".ply";
                sensemap::WriteBinaryPlyPoints(SaveWholePlyPath, locations, true, true);
                std::cout << "WriteBinaryPlyPoints :" << SaveWholePlyPath << std::endl;

            };
            SaveCom(communities, "merge-cluster");
        }

        ReconstructCommunities(communities, overlap_image_ids,label_cluster_id);
    }

    IndependentMapperOptions mapper_options = options_->mapper_options;
    // Final filtering
    for (size_t i = 0; i < reconstruction_manager_->Size(); ++i) {
        std::shared_ptr<Reconstruction> reconstruction = reconstruction_manager_->Get(i);
        reconstruction->FilterAllMapPoints(mapper_options.filter_min_track_length_final,
                                           mapper_options.filter_max_reproj_error_final,
                                           mapper_options.filter_min_tri_angle_final);
        reconstruction->FilterImages(mapper_options.min_focal_length_ratio, mapper_options.max_focal_length_ratio,
                                     mapper_options.max_extra_param);
    }

    GetTimer().PrintMinutes();
    std::cout << "Cluster mapper Run function return" << std::endl;
}

void ClusterMapperController::ReconstructCommunities(
    const std::vector<std::vector<image_t>>& communities,
    const std::vector<std::unordered_set<image_t>>& overlap_image_ids,
    const std::vector<label_t>& communitie_ids) {
    // Start the reconstruction workers.

    std::unordered_map<int, std::shared_ptr<ReconstructionManager>> reconstruction_managers;
    reconstruction_managers.reserve(communities.size());

    bool camera_calibrated = false;
    std::unordered_set<camera_t> set_fixed_camera_ids;
    for (int community_id = 0; community_id < communities.size(); ++community_id) {

        std::shared_ptr<ReconstructionManager>& reconstruction_manager = reconstruction_managers[community_id];

        reconstruction_manager = std::shared_ptr<ReconstructionManager>(new ReconstructionManager());

        std::set<uint32_t> image_ids;
        for (const auto image_id : communities[community_id]) {
            image_ids.insert(image_id);
        }

        std::cout << "image_ids.size() = " << image_ids.size() << std::endl;

        if (image_ids.empty()) {
            continue;
        }

        std::cout << "Create cluster graph container" << std::endl;
        std::shared_ptr<SceneGraphContainer> cluster_graph_container;
        if (communities.size() == 1){
            cluster_graph_container = scene_graph_container_;
            std::cout << "Copy cluster graph container done " << std::endl;
        } else {
            cluster_graph_container=
                std::shared_ptr<SceneGraphContainer>(new SceneGraphContainer());
            scene_graph_container_->ClusterSceneGraphContainer(image_ids, *cluster_graph_container.get());
            std::cout << "Create cluster graph container done " << std::endl;
        }
        
        std::cout << "Start Export to Graph" << std::endl;
        cluster_graph_container->CorrespondenceGraph()->ExportToGraph(
            StringPrintf("%s/cluster_graph%04d.png", workspace_path_.c_str(), community_id));

        IndependentMapperOptions custom_options = options_->IndependentMapper();
        custom_options.max_model_overlap = 3;
        custom_options.init_num_trials = options_->init_num_trials;
        custom_options.image_ids = image_ids;
        if (overlap_image_ids.size() > community_id) {
            custom_options.overlap_image_ids = overlap_image_ids[community_id];
        }

        if (custom_options.single_camera && camera_calibrated) {
            custom_options.camera_fixed = true;
        }

        if (custom_options.refine_separate_cameras) {
            custom_options.refine_separate_cameras = false;
            options_->merge_options.refine_separate_cameras = true;
        }

        ReconstructSingleInstance(workspace_path_, image_path_, custom_options, cluster_graph_container, communitie_ids[community_id],
                                  reconstruction_manager);
        
        const auto rect_ids = reconstruction_manager->getReconstructionIds();
        for (int rect_idx = 0; rect_idx < rect_ids.size(); rect_idx++){
            const auto rect = reconstruction_manager->Get(rect_ids.at(rect_idx));
            const auto& rect_cameras = rect->Cameras();
            for (const auto& rect_camera : rect_cameras){
                if(!scene_graph_container_->ExistsCamera(rect_camera.first)){
                    std::cout << "Error: " << "scene_graph_container_->ExistsCamera(" << rect_camera.first << ")" << std::endl;
                    exit(-1);
                }
                auto& scene_camera = scene_graph_container_->Camera(rect_camera.first);
                if (scene_camera.IsCameraConstant()){
                    continue;
                }
                scene_camera = rect_camera.second;
                scene_camera.SetCameraConstant(true);
                set_fixed_camera_ids.insert(rect_camera.first);
                std::cout << "set camera-" << rect_camera.first << " fixed, param:" << scene_camera.ParamsToString() << std::endl;
            }
        }
    }

    std::cout << "\nRestore Cameras Set: " << std::endl;
    for (const auto fixed_camera_id : set_fixed_camera_ids){
        auto& scene_camera = scene_graph_container_->Camera(fixed_camera_id);
        scene_camera.SetCameraConstant(false);
        std::cout << "\t => camera-" << fixed_camera_id << "(param:" << scene_camera.ParamsToString() << "), in "; 
        for (int community_id = 0; community_id < communities.size(); ++community_id) {
            std::shared_ptr<ReconstructionManager>& reconstruction_manager = reconstruction_managers[community_id];
            auto rect_ids = reconstruction_manager->getReconstructionIds();
            std::cout << "community-" << community_id << "(";
            for (int rect_idx = 0; rect_idx < rect_ids.size(); rect_idx++){
                auto rect = reconstruction_manager->Get(rect_ids.at(rect_idx));
                if (rect->ExistsCamera(fixed_camera_id)){
                    rect->Camera(fixed_camera_id).SetCameraConstant(false);
                    std::cout << community_id << ", ";
                }
            }
            std::cout << ") |";
        }
        std::cout << std::endl;
    }

    std::cout << "reconstruct communities done\n" << std::endl;

    {
        float mem_b, mem_a;
        sensemap::GetAvailableMemory(mem_b);
        sensemap::GetAvailableMemory(mem_a);
        std::cout << "malloc_trim available: " << mem_a << "G (" << mem_a - mem_b << ")" << std::endl;
    }

    //////////////////////////////////////////////////////////////////////////
    //  Merge Clusters
    //////////////////////////////////////////////////////////////////////////
    if (communities.size() == 1) {
        *reconstruction_manager_.get() = std::move(*reconstruction_managers.begin()->second.get());
    } else if (communities.size() > 1) {

        // merge with max spanning tree and motion average
        options_->merge_options.debug_info = options_->mapper_options.debug_info;

        EIGEN_STL_UMAP(cluster_t, Eigen::Matrix3x4d) global_transforms;
        std::unordered_map<cluster_t, Reconstruction> reconstructions;

        MergeCommunities(reconstruction_managers, scene_graph_container_,
                         reconstruction_manager_, options_->merge_options, options_, reconstructions,
                         global_transforms,workspace_path_);

        WriteTransforms(workspace_path_, global_transforms);

        std::cout << "reconstructions: " << reconstructions.size() << std::endl;
    }
    std::cout << "merge communities done" << std::endl;
}

// void ClusterMapperController::FinalBundleAdjustment(Reconstruction* reconstruction) {
//     CHECK_NOTNULL(reconstruction);

//     PrintHeading1("Final Global bundle adjustment");

//     const std::vector<image_t>& reg_image_ids = reconstruction->RegisterImageIds();

//     if (reg_image_ids.size() < 2) {
//         std::cout << "ERROR: Need at least two views." << std::endl;
//         return;
//     }

//     // Avoid degeneracies in bundle adjustment.
//     reconstruction->FilterObservationsWithNegativeDepth();

//     BundleAdjustmentOptions ba_options;
//     ba_options.refine_extra_params = true;
//     ba_options.refine_focal_length = true;

//     ba_options.solver_options.minimizer_progress_to_stdout = true;
//     ba_options.solver_options.max_num_iterations = 50;

//     BundleAdjustmentIterationCallback iteration_callback(this);
//     ba_options.solver_options.callbacks.push_back(&iteration_callback);

//     // Configure bundle adjustment.
//     BundleAdjustmentConfig ba_config;
//     for (const image_t image_id : reg_image_ids) {
//         ba_config.AddImage(image_id);
//     }
//     ba_config.SetConstantPose(reg_image_ids[0]);
//     ba_config.SetConstantTvec(reg_image_ids[1], {0});

//     // Run bundle adjustment.
//     BundleAdjuster bundle_adjuster(ba_options, ba_config);
//     bundle_adjuster.Solve(reconstruction);

//     GetTimer().PrintMinutes();
// }

}  // namespace sensemap
