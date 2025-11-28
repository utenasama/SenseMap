//Copyright (c) 2019, SenseTime Group.
//All rights reserved.

#ifndef SENSEMAP_CONTROLLERS_PATCH_MATCH_CONTROLLER_H_
#define SENSEMAP_CONTROLLERS_PATCH_MATCH_CONTROLLER_H_

#include "util/types.h"
#include "util/threading.h"
#include "util/obj.h"
#include "util/ply.h"
#include "util/octree.h"
#include "util/roi_box.h"
#include "util/semantic_table.h"
#include "controllers/patch_match_options.h"
#include "mvs/workspace.h"
#include "mvs/mvs_cluster.h"

#define CACHED_DEPTH_MAP

namespace sensemap {
namespace mvs {

class PatchMatchController : public Thread {
public:
    struct VertInfo {
        Eigen::Vector3f X;
        Eigen::Vector3f color;
        std::vector<uint32_t> view_ids;
    };
public:
    PatchMatchController(const PatchMatchOptions& options,
			             const std::string& workspace_path,
                         const std::string& lidar_path = "",
                         const int reconstrction_idx = -1,
                         const int cluster_idx = -1);

    void AddBlackList(const std::vector<uint8_t>& black_list);

protected:
	void Run() override;
	void Reconstruct();
    void ClusterReconstruction(
        const int cluster_id, 
        const std::vector<std::vector<int>>& cluster_image_map, 
        const std::vector<std::vector<int> >& overlapping_images,
        const std::string dense_reconstruction_path,
        // const std::string cluster_reconstruction_path,
        PatchMatchOptions options,
        bool prior_ply);

    void BuildPriorModel(const std::string& workspace_path);
    void BuildLidarModel(const std::string& lidar_path);

    void ConvertVisibility2Dense(
        const std::unordered_map<size_t, image_t>& cluster_image_idx_to_id,
        std::vector<std::vector<uint32_t> >& fused_vis);
    void MergeFusedPly(const std::string& workspace_path);

    void ProcessProblem(PatchMatchOptions options,
                        const int problem_idx,
                        const std::string &dense_path,
                        const std::string &workspace_path,
                        const int level,
                        const bool fill_flag,
                        bool save_flag);
    void ProcessFilterProblem(PatchMatchOptions options,
                              const int problem_idx,
                              const std::string &workspace_path,
                              const int level,
                              const bool ProcessFilterProblem = false);
    int CrossFilterImageCluster(
            std::vector<std::unordered_set<int >> &cluster_problem_ids,
            std::vector<std::unordered_set<int >> &cluster_whole_ids,
            std::unordered_map<image_t, bool>& images_exclusive,
            std::vector<int >& cluster_gpu_idx,
            float memory_factor = 0.6f);
    void ProcessCrossFilter(
            const std::vector<std::unordered_set<int >> &cluster_ref_ids,
            const std::vector<std::unordered_set<int >> &cluster_whole_ids,
            const std::unordered_map<image_t, bool>& images_exclusive,
            const std::vector<int > &cluster_gpu_idx,
            const std::vector<std::vector<int> >& overlapping_images,
            PatchMatchOptions options, const std::string &workspace_path,
            const int cluster_id, const int level, const bool prior_ply,
            const bool process_rgbd_prior);

    // void ReadWorkspace();
    void ReadGpuIndices();
    void ReadCrossFilterGpuIndices();
    void GetGpuProp();
    void EstimateThreadsPerGPU();
    int ReadClusterBox();

    std::pair<float, float> 
    InitDepthMap(const PatchMatchOptions& options,
                 const size_t problem_idx,
                 const Image& image,
                 const std::string& dense_path,
                 const std::string& workspace_path);
    
    void PyramidRefineDepthNormal(const std::string &workspace_path,
                                  const int level, bool fill_flag,
                                  std::string input_type,
                                  PatchMatchOptions options);

    void RefineDepthAndNormalMap(const PatchMatchOptions options,
                                 const std::string& workspace_path,
                                 const int image_idx,
                                 DepthMap& ref_depth_map,
                                 NormalMap& ref_nromal_map,
                                 const int level);

    float ScoreRemoveOutlier(const int cluster_id);

    float DistRemoveOutlier(float remove_factor, const int cluster_id);

    void VoxelRemoveOutlier(const double average_dist, 
                            const float factor,
                            const int cluster_id);

    void PlaneScoreCompute(const double average_dist,
                            float factor,
                            const int cluster_id);

    void ComputeSemanticLabel(const std::vector<int> &cluster_images,
                              const std::string semantic_path);

    PatchMatchOptions options_;
	const std::string workspace_path_;  // path of workspace
    const std::string lidar_path_;
    std::unordered_set<int> used_images_;

    TriangleMesh prior_model_;
    std::vector<PlyPoint> prior_points_;

    std::vector<VertInfo> lidar_samps_;
    
#ifdef CACHED_DEPTH_MAP
    std::vector<Bitmap> bitmaps_;
    std::vector<Bitmap> semantic_maps_;
    std::vector<DepthMap> depth_maps_;
    std::vector<NormalMap> normal_maps_;
    std::vector<Bitmap> mask_maps_;
    std::vector<Mat<float> > conf_maps_;
    std::vector<DepthMap> depth_maps_temp_;
    std::vector<NormalMap> normal_maps_temp_;
    std::vector<DepthMapInfo> depth_maps_info_;

    std::vector<DepthMap> prior_wgt_maps_;
    
    std::mutex workspace_mutex_;
#endif
    std::vector<bool> flag_depth_maps_;

    std::unique_ptr<ThreadPool> thread_pool_;
    std::unique_ptr<Workspace> workspace_;
    std::vector<Problem> problems_;
    std::vector<std::pair<float, float> > depth_ranges_;

    std::vector<int> gpu_indices_;
    std::vector<int> threads_per_gpu_;
    std::vector<int> cross_filter_gpu_indices_;
    std::vector<float> max_gpu_memory_array_;
    std::vector<int> max_gpu_cudacore_array_;
    std::vector<uint64_t > max_gpu_texture_layered_0_;
    std::vector<uint64_t > max_gpu_texture_layered_1_;
    std::vector<uint64_t > max_gpu_texture_layered_2_;

    int num_cluster_ = -1;
    int cluster_step_ = 1;
    int select_reconstruction_idx_;
    int select_cluster_idx_;
    
    std::unique_ptr<ThreadPool> cross_filter_thread_pool_;

    std::unordered_set<int> locked_images_;
    std::mutex deduplication_mutex_;

    std::unordered_map<int, std::vector<int>> src_2_refs_map_;
    std::unordered_map<int, std::vector<int>> ref_2_srcs_map_;
    std::unordered_map<int, int> ref_2_problem_map_;
    std::vector<Eigen::Vector3f> images_c_;

    std::vector<std::vector<PlyPoint> > cluster_points_;
    std::vector<std::vector<float> > cluster_points_score_;
    std::vector<std::vector<std::vector<uint32_t> >> cluster_points_visibility_;

    std::vector<PlyPoint> cross_fused_points_; 
    std::vector<float> cross_fused_points_score_; 
    std::vector<std::vector<uint32_t> > cross_fused_points_visibility_;
    std::vector<std::vector<float> > cross_fused_points_vis_weight_;

    uint64_t num_save_fused_points_ = 0;

    int num_all_images_ = 0;
    int num_processed_images_ = 0;

    int num_box_ = -1;
    Eigen::Matrix3f box_rot_ = Eigen::Matrix3f::Identity();
    Box roi_box_;
    std::vector<Box> roi_child_boxs_;
    std::vector<uint8_t> semantic_label_black_list_;
    std::vector<std::unordered_map<size_t, image_t>> sfm_cluster_image_idx_to_id_;

    double depth_elapsed_time_ = 0.0;
    double cross_elapsed_time_ = 0.0;
    double fusion_elapsed_time_ = 0.0;
};

class PanoramaPatchMatchController : public Thread {
public:
    PanoramaPatchMatchController(const PatchMatchOptions& options,
                                 const std::string& image_path,
                                 const std::string& workspace_path,
                                 const int reconstrction_idx = -1);

private:
    void Run();
    void ReadGpuIndices();
    void ReadWorkspace();
    void PrepareData();
    void ReadProblems();
    void ProcessProblem(const PatchMatchOptions& options,
                        const size_t problem_idx,
                        const int level,
                        const bool ref_flag,
                        std::string input_type);
    void ProcessFilterProblem(const PatchMatchOptions& options,
                              const size_t problem_idx,
                              const int level,
                              const bool ref_flag,
                              std::string input_type);

    std::pair<float, float> 
    InitDepthMap(const PatchMatchOptions& options,
                 const size_t problem_idx,
                 const Image& image,
                 const std::string& workspace_path);
    void BuildPriorModel(const std::string& workspace_path);


    void PyramidRefineDepthNormal(const PatchMatchOptions& options,
                                  const int level,
                                  bool fill_flag);

    const PatchMatchOptions options_;
    const std::string image_path_;
    const std::string workspace_path_;
    std::string component_path_;
    std::unordered_set<int> used_images_;

    TriangleMesh prior_model_;

    std::vector<std::string> perspective_image_names_;
    std::vector<mvs::Image> perspective_images_;
    std::vector<image_t> perspective_image_ids_;
    std::vector<std::vector<int> > perspective_src_images_idx_;
    std::vector<std::pair<float, float>> depth_ranges_;

#ifdef CACHED_DEPTH_MAP
    std::vector<DepthMap> depth_maps_;
    std::vector<NormalMap> normal_maps_;
    std::vector<Mat<float>> conf_maps_;
    std::mutex workspace_mutex_;
#endif

    std::unique_ptr<ThreadPool> thread_pool_;
    std::unique_ptr<Workspace> workspace_;
    std::vector<Problem> problems_;
    std::vector<int> gpu_indices_;
    std::vector<int> rmap_idx_;

    int num_reconstruction_;
    int select_reconstruction_idx_;
};

} // namespace mvs
} // namespace sensemap

#endif
