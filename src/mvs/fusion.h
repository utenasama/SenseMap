//Copyright (c) 2019, SenseTime Group.
//All rights reserved.

#ifndef SENSEMAP_MVS_FUSION_H_
#define SENSEMAP_MVS_FUSION_H_

#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/StdVector>
#include "util/types.h"
#include "util/math.h"
#include "util/ply.h"
#include "util/threading.h"
#include "util/bitmap.h"
#include "util/mat.h"
#include "util/semantic_table.h"
#include "util/roi_box.h"
#include "base/common.h"

#include "mvs/workspace.h"
#include "mvs/depth_map.h"
#include "mvs/normal_map.h"

// #define OLD_PIPELINE

namespace sensemap {
namespace mvs {

struct FusionData {
  int image_idx = kInvalidImageId;
  int row = 0;
  int col = 0;
  int traversal_depth = -1;
  bool operator()(const FusionData& data1, const FusionData& data2) {
    return data1.image_idx > data2.image_idx;
  }
};

class StereoFusion : public Thread {
 public:
  // EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  struct Options {
    
    // Camera format.
    std::string format = "perspective";

    // Maximum image size in either dimension.
    int max_image_size = -1;

    // Step size of depth map fusion.
    int step_size = 1;

    // Minimum number of fused pixels to produce a point.
    int min_num_pixels = 5;

    // Maximum number of pixels to fuse into a single point.
    int max_num_pixels = 10000;

    // Minimum number of visible images to produce a point.
    int min_num_visible_images = 3;

    // Maximum depth in consistency graph traversal.
    int max_traversal_depth = 100;

    // Maximum relative difference between measured and projected pixel.
    double max_reproj_error = 2.0f;

    // Maximum relative difference between measured and projected depth.
    double max_depth_error = 0.01f;

    // Minimum relative difference between occluded measured and projected depth.
    double min_occluded_depth_error = 0.03f;

    // Whether to fuse normal map.
    bool with_normal = true;

    // Maximum angular difference in degrees of normals of pixels to be fused.
    double max_normal_error = 10.0f;

    // Number of overlapping images to transitively check for fusing points.
    int check_num_images = 50;

    // Cache size in gigabytes for fusion. The fusion keeps the bitmaps, depth
    // maps, normal maps, and consistency graphs of this number of images in
    // memory. A higher value leads to less disk access and faster fusion, while
    // a lower value leads to reduced memory usage. Note that a single image can
    // consume a lot of memory, if the consistency graph is dense.
    double cache_size = 32.0;

    // Whether to fuse semantic info.
    double num_consistent_semantic_ratio = 0.3;

    double min_inlier_ratio_to_best_model = 0.8;

    bool fit_ground = false;

    bool cache_depth = true;


    bool remove_duplicate_pnts = true;

    bool outlier_removal = false;
    float outlier_deviation_factor = 3.0f;
    float outlier_voxel_factor = 25.0f;

    int nb_neighbors = 6;    
    double max_spacing_factor = 6.0;
    
    bool roi_fuse = false;
    float roi_box_width = -1.f;
    float roi_box_factor = -1.f;
    bool roi_box = false;

    // Plane Fitting parameters.
    double dist_to_best_plane_model = 12.0;
    double angle_diff_to_main_plane = 3.0;

    // max memory capacity in gigabytes for fusion.  -1.0f means infinite
    float max_ram = -1.0f;
    float ram_eff_factor = 0.75;
    float fuse_common_persent = 0.06;

    // delaunay insert param
    float dist_insert = 5.0f;
    float diff_depth = 0.01f;

    bool fused_delaunay_sample = false;
    bool map_update = false;

    RANSACOptions ransac_options;

    // Check the options for validity.
    bool Check() const;

    // Print the options to stdout.
    void Print() const;
  };

  StereoFusion(const Options& options,
               const std::string& workspace_path,
               const std::string& input_type = GEOMETRIC_TYPE,
               const int reconstrction_idx = -1,
               const int cluster_idx = -1);

  const std::vector<PlyPoint>& GetFusedPoints() const;
  const std::vector<std::vector<uint32_t>>& GetFusedPointsVisibility() const;
  void AddBlackList(const std::vector<uint8_t>& black_list);
  // pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
  void SetROIBox(const Box& box);

 protected:
  void ReadWorkspace();

  void Run();
  // void Fuse(const Options &options, std::vector<FusionData> fusion_queue);
  void Fuse(const Options &options,
            std::vector<FusionData> fusion_queue,
            std::vector<float> &fused_point_x,
            std::vector<float> &fused_point_y,
            std::vector<float> &fused_point_z,
            std::vector<float> &fused_point_nx,
            std::vector<float> &fused_point_ny,
            std::vector<float> &fused_point_nz,
            std::vector<float> &fused_point_c,
            std::vector<uint8_t> &fused_point_r,
            std::vector<uint8_t> &fused_point_g,
            std::vector<uint8_t> &fused_point_b,
            std::unordered_set<int> &fused_point_visibility,
            std::vector<uint8_t> &fused_point_seg_id);

  void FuseImage(const Options &options, const int thread_id, 
                 const int image_idx, const int num_eff_threads, 
                 const bool fuse_plane = false);

  void Clear();

  void RemoveDuplicatePoints(const Options& options,
                             std::vector<int>& cluster_image_ids,
                             const size_t num_pre_fused_points);
  void AdjustDuplicatePoints(const Options& options,
                            std::vector<int>& cluster_image_ids,
                            const std::vector<short int> cluster_ids, 
                            const size_t num_pre_fused_points);
  
  int RemoveCommonDuplicatePoints();

  void RemoveOutliers(const Options& options);
  void VoxelRemoveOutlier(const double average_dist,
                          const Options& options);

  void ComplementPlanes(const Options& options, 
                        std::vector<int>& cluster_image_ids,
                        size_t num_pre_fused_points,
                        size_t num_pre_fused_images,
                        size_t all_fusion_image_num);
  void ComplementPlanesPanorama(const Options& options, 
                        std::vector<int>& cluster_image_ids,
                        size_t num_pre_fused_points,
                        size_t num_pre_fused_images,
                        size_t all_fusion_image_num);

  void Read(const std::string& path);
  int ReadClusterBox();

  int FindNextImage(const int prev_image_idx) ;

  void ConvertVisibility2Dense(
    const std::unordered_map<size_t, image_t> & cluster_image_idx_to_id,
    std::vector<std::vector<uint32_t> >& fused_vis);
  void MergeFusedPly(const std::string& workspace_path);

  int num_cluster_;
  int cluster_step_ = 1;
  int select_reconstruction_idx_;
  int select_cluster_idx_;

  std::unique_ptr<Workspace> workspace_;

  const Options options_;
  const std::string workspace_path_;  // path of workspace
  std::string active_dense_path_;
  std::string active_cluster_stereo_path_;
  const std::string input_type_;

  std::unique_ptr<ThreadPool> thread_pool_;

  const float max_squared_reproj_error_;
  const float min_cos_normal_error_;

  std::vector<std::string> image_names_;
  std::vector<mvs::Image> images_;
  std::vector<std::unique_ptr<Bitmap> > semantic_maps_; 
  std::vector<std::unique_ptr<DepthMap> > depth_maps_;
  std::vector<std::unique_ptr<NormalMap> > normal_maps_;
  std::vector<std::unique_ptr<MatXf> > conf_maps_;
  std::vector<unsigned char> used_images_;
  std::vector<unsigned char> fused_images_;
  std::vector<std::vector<int> > overlapping_images_;
  std::vector<std::unique_ptr<Mat<bool> > > fused_pixel_masks_;
  std::vector<Eigen::RowMatrix3x4f> P_;
  std::vector<Eigen::RowMatrix3x4f> inv_P_;
  std::vector<Eigen::RowMatrix3f> inv_R_;

  // Next points to fuse.
  std::vector<FusionData> fusion_queue_;

  // Already fused points.
  std::vector<PlyPoint> fused_points_;
  std::vector<std::vector<uint32_t> > fused_points_visibility_;
  std::vector<std::vector<float> > fused_points_vis_weight_;
  std::vector<float> fused_points_score_;

  std::vector<int64_t> merge_points_;
  std::vector<std::vector<PlyPoint>> cluster_points_;
  std::vector<std::vector<std::vector<uint32_t> >> cluster_points_visibility_;

  std::vector<uint8_t> semantic_label_black_list_;

  Box roi_box_;
  int num_box_ = -1;
  Eigen::Matrix3f box_rot_ = Eigen::Matrix3f::Identity();
  std::vector<Box> roi_child_boxs_;
  std::vector<std::unordered_map<size_t, image_t>> sfm_cluster_image_idx_to_id_;

  std::mutex push_mutex_;
  std::mutex depthmap_mutex_;
  std::mutex normalmap_mutex_;
  std::mutex image_mutex_;
  std::mutex mask_mutex_;
};

}  // namespace mvs
}  // namespace sensemap

#endif  // SENSEMAP_MVS_FUSION_H_
