// Copyright (c) 2019, SenseTime Group.
// All rights reserved.

#ifndef SENSEMAP_OPTIMIZER_CLUSTERMERGE_CLUSTERMERGEOPTIMIZER_H_
#define SENSEMAP_OPTIMIZER_CLUSTERMERGE_CLUSTERMERGEOPTIMIZER_H_

#include <malloc.h>

#include "optim/pose_graph_optimizer.h"
#include "util/alignment.h"
#include "util/threading.h"
namespace sensemap {

class ClusterMergeOptimizer {
public:
    struct ClusterMergeOptions {
        //////////////////////////////////////////////////////
        // Loop Edge Selection
        //////////////////////////////////////////////////////

        // Debug info
        bool debug_info = false;

        // Loop image candidate verification
        bool LoopVerifiedBySIM3 = false;

        bool OverlapCheckPnP = false;

        // The max error for pnp estimation
        float max_error = 0.4;

        bool EnableLoopVerified = true;

        bool OnlyAddCloseImageID = false;

        bool OnlyAddSameImageID = false;

        bool CheckInlierByDistance = true;

        double Corespondence2DThreshold = 30;

        double Corespondence3DThreshold = 20;

        double PnPInlierTreshold = 15;

        double SIM3InlierTreshold = 15;

        //////////////////////////////////////////////////////
        // Construct Normal Edge
        //////////////////////////////////////////////////////
        // Neighbor threshold
        size_t NeighborNumberThreshold = 10;

        //////////////////////////////////////////////////////
        // Cluster Motion Averager related
        //////////////////////////////////////////////////////
        bool save_initial_transform = false;

        std::string initial_transform_path = "";

        bool load_initial_transform = false;

        bool save_strong_pairs = false;

        std::string strong_pairs_path = "";

        size_t candidate_strong_pairs_num = 50;

        bool load_strong_pairs = false;

        //////////////////////////////////////////////////////
        // Pose Graph related
        //////////////////////////////////////////////////////
        PoseGraphOptimizer::OPTIMIZATION_METHOD optimization_method = PoseGraphOptimizer::OPTIMIZATION_METHOD::SIM3;

        bool lossfunction_enable = true;

        // Cluster Track merge
        double merge_max_reproj_error = 3;

        // Check neighbor correspondence threshold
        size_t NeighborCorrespondece = 40;

        // Set the Pose graph optimization iteration number
        size_t max_optimization_iteration_num = 50;

        // Check loop within cluster
        size_t LoopCheckThreshold = 20;

        // Only process the image with same image id
        bool only_process_same_image = false;

        // If 3d point number which used to calculate the final sim3 transform
        //  is larger than this threshold, it will be down sampled
        size_t sim3_downsample_threshold = 400;

        // Enable the final ba or not
        bool enable_final_ba = true;

        // Only pick mappoint which track length is lager than threshold to calculate sim3
        double track_length_threshold = 4;

        // Enable to calculate pnp when sim3 estimation is fail
        bool attempt_pnp_after_sim3_fail = false;

        double neighbor_distance_factor_wrt_averge_baseline = 1.5;
        size_t normal_edge_min_common_points = 30;
        size_t loop_edge_min_pose_inlier_num = 25;
        size_t max_loop_image_between_clusters = 30;

        bool fixed_original_reconstruction = true;
        int original_reconstruction_id = 0;
        bool fixed_reconstruction_pose = false;
        int fixed_reconstruction_pose_id = 0;
        bool fixed_recon_scale = true;
        int fixed_reconstruction_scale_id = 0;

        bool use_prior_relative_pose = true;
        bool use_prior_relative_pose_id = 1;

        bool only_load_optimized_reconstructions = false;

        bool save_optimized_reconsturctions = false;

        std::string optimized_reconsturctions_path = "";

        bool save_loop_image_pairs = false;

        bool load_loop_image_pairs = false;

        std::string loop_image_pairs_path = "";

        bool save_normal_image_pairs = false;

        bool load_normal_image_pairs = false;

        std::string normal_image_pairs_path = "";

        bool detect_strong_loop = true;

        double clusters_close_images_distance = 0.5;
        int max_iter_time = 3;
        int current_iteration = 0;

        bool refine_separate_cameras = false;
    };

private:
    struct Vertex {
        Eigen::Quaterniond qvec;
        Eigen::Vector3d tvec;
        double scale;
    };

    struct Edge {
        std::pair<cluster_t, image_t> id_begin;
        std::pair<cluster_t, image_t> id_end;
        size_t correspondence_num;
        size_t label;
        Vertex relative_pose;
        double weight = 1.0;
    };


    // Data for a correspondence / element of a track, used to store all
    // relevant data for triangulation, in order to avoid duplicate lookup
    // in the underlying unordered_map's in the Reconstruction
    struct CorrData {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        image_t image_id;
        point2D_t point2D_idx;
        const class Image* image;
        const class Camera* camera;
        const class Point2D* point2D;
    };

public:
    ClusterMergeOptimizer(std::shared_ptr<ClusterMergeOptions> options,
                          const CorrespondenceGraph* correspondence_graph);

    // optimize the global motions of all the clusters based on an initial guess
    void MergeByPoseGraph(const std::vector<std::shared_ptr<Reconstruction>>& reconstructions,
                          const EIGEN_STL_UMAP(cluster_t, Eigen::Matrix3x4d)& global_transforms,
                          const EIGEN_STL_UMAP(cluster_pair_t, Eigen::Matrix3x4d)& relative_transforms,
                          const std::vector<cluster_t>& cluster_ordered,
                          std::shared_ptr<Reconstruction>& reconstruction);

    const std::unordered_map<image_t, std::set<image_t>>& GetImageNeighborBetweenCluster();
    std::vector<std::vector<image_t>> GetClusterImageIds(bool is_true);
    const std::unordered_set<image_t>& GetConstImageIds();

private:
    // Write the Merge Result without any transform
    void WriteMergeResult(const std::string& filename);

    void WriteReconstructionResult(const std::string& filename);

    void FindImageNeighbor();

    size_t MergeTracks(const mappoint_t mappoint);

    void MergeReconstructionTracks();

    size_t Triangulate(const mappoint_t mappoint_id);

    void TriangulateReconstruction();

    size_t FindCorrespondences(const image_t image_id, const point2D_t point2D_idx, 
                const size_t transitivity, std::vector<CorrData>* corrs_data);

    size_t CompleteLoopPoint(const image_t image_id, const point2D_t point2D_idx);
    size_t CompleteImage(const image_t image_id);

    void CompleteLoopImages();

    void WritePoseGraphResult(const std::string& folder_name);

    void InitialPoseGraph(const std::vector<std::shared_ptr<Reconstruction>>& reconstructions,
                          const EIGEN_STL_UMAP(cluster_t, Eigen::Matrix3x4d)& global_transforms,
                          const std::vector<cluster_t>& cluster_ordered);

    void RefineRelativeTransform(const EIGEN_STL_UMAP(cluster_pair_t, Eigen::Matrix3x4d)& relative_transforms,
                                 const std::vector<cluster_t>& cluster_ordered);

    // Find the Loop Image pair Candidate
    void FindLoopImageCandidate();
    void FindCommonViewImageCandidate();

    //
    bool PnPVerification(std::vector<Eigen::Vector2d> point2d, std::vector<Eigen::Vector3d> point3d, double max_error,
                         size_t& inlier_num);
    // Construct Pose Graph Vertex
    void ConstructVertex();

    // Construct Pose Graph Edge
    void ConstructNormalEdge();

    // Construct Pose Graph Edge
    void ConstructLoopEdge();

    // Pose Graph Ceres Optimization
    void OptimizationSE3();

    // Sim3 Pose Graph Ceres Optimization
    void OptimizationSim3();

    // Update image pose after sim3 optimization
    void UpdateSim3ImagePose();

    // Update image pose
    void UpdateSE3ImagePose();

    // Update map point pose after sim3 optimization
    void UpdateSim3MapPoint();

    // Update mappoint
    void UpdateSE3MapPoint();

    // Output all the candidate correspondence loop
    void OutputCorrespondence();

    bool EstimatePnP(cluster_t cluster_1, cluster_t cluster_2, image_t image_src, image_t image_dst, Edge& edge,
                     bool optimize_pose = false);

private:
    // Store the pose graph optimizer
    PoseGraphOptimizer* optimizer_{};

    const CorrespondenceGraph* full_correspondence_graph_;

    std::shared_ptr<ReconstructionManager> reconstruction_manager_;

    std::shared_ptr<Reconstruction> reconstruction_;

    // Store the image neighbor
    std::unordered_map<cluster_t, std::unordered_map<image_t, std::set<image_t>>> all_image_neighbor_;

    std::unordered_map<cluster_t, std::unordered_map<image_t, std::set<image_t>>> normal_edge_image_neighbor_;
    std::unordered_map<cluster_t,std::vector<std::pair<image_pair_t, size_t>>> normal_edge_candidate_image_neighbor_;

    EIGEN_STL_UMAP(cluster_pair_t, Eigen::Matrix3x4d) relative_transforms_;

    // Store the loop candidate image pair id for each cluster pair
    std::unordered_map<cluster_pair_t, std::set<image_pair_t>> all_loop_image_ids_;

    std::unordered_map<cluster_pair_t, std::vector<std::pair<image_pair_t, point2D_t>>> all_candidate_loop_image_ids_;

    std::shared_ptr<ClusterMergeOptions> options_;

    std::unordered_map<cluster_pair_t, EIGEN_STL_UMAP(image_t, Vertex)> image_transform_;

    // Cache for tried track merges to avoid duplicate merge trials.
    std::unordered_map<mappoint_t, std::unordered_set<mappoint_t>> merge_trials_;

    // Changed Map Points, i.e. if a Map Point is modified (created, continued,
    // deleted, merged, etc.). Cleared once `ModifiedMapPoints` is called.
    std::unordered_set<mappoint_t> modified_mappoint_ids_;

    // Store the image pose before pose graph optimization
    std::unordered_map<cluster_t, EIGEN_STL_UMAP(image_t, Vertex)> poses_;

    std::vector<Edge,Eigen::aligned_allocator<Edge> > edges_;

    std::unordered_map<cluster_t, cluster_t> manager_id_map_;

    // Store the correspondence
    std::unordered_map<cluster_pair_t, std::unordered_map<image_pair_t, FeatureMatches>> image_pair_correspondences_;

    //
    std::unordered_map<image_t, std::set<image_t>> image_neighbor_between_cluster_;
    std::unordered_set<image_t> image_const_pose_ids_;

    // ThreadPool
    std::unique_ptr<ThreadPool> thread_pool;
    std::mutex add_mappoint_mutex_;
};  // ClusterMergeOptimizer

}  // namespace sensemap
#endif  // SENSEMAP_OPTIMIZER_CLUSTERMERGE_CLUSTERMERGEOPTIMIZER_H_
