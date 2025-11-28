//Copyright (c) 2019, SenseTime Group.
//All rights reserved.

#ifndef SENSEMAP_MVS_INTEGRATION_H_
#define SENSEMAP_MVS_INTEGRATION_H_

#include <unordered_set>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/StdVector>
// #include "util/alignment.h"
#include "util/types.h"
#include "util/math.h"
#include "util/ply.h"
#include "util/threading.h"
#include "util/bitmap.h"
#include "util/mat.h"
#include "base/common.h"

#include "mvs/workspace.h"
#include "mvs/depth_map.h"
#include "mvs/normal_map.h"

namespace sensemap {
namespace mvs {

class StereoIntegration : public Thread {
 public:
  // EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  struct Options {
    
    // Camera format.
    std::string format = "rgbd";

    // Maximum image size in either dimension.
    int max_image_size = -1;

    // Number of overlapping images to transitively check for fusing points.
    int check_num_images = 50;

    // Cache size in gigabytes for fusion. The fusion keeps the bitmaps, depth
    // maps, normal maps, and consistency graphs of this number of images in
    // memory. A higher value leads to less disk access and faster fusion, while
    // a lower value leads to reduced memory usage. Note that a single image can
    // consume a lot of memory, if the consistency graph is dense.
    double cache_size = 32.0;

    bool cache_depth = true;

    // Maximum spacing factor for noisy isolated faces filtering (6.0 is a normal choice).
    // Faces with spacing larger than max_spacing_factor * average_spacing will be removed.
    // If max_spacing_factor < 1.0, no filtering is done.
    double max_spacing_factor = 0.0; // = 6.0;

    // Voxel length for TSDF fusion (0.004f is a normal choice for small object).
    float voxel_length = 0.02f;

    // Truncation precision for TSDF fusion (0.03f is a normal choice for small object).
    float sdf_trunc_precision = 0.06f;

    // Weight threshold for mesh extraction. If set larger the mesh contains with (only)
    // more confident faces, and vice versa. 
    float extract_mesh_threshold = 1.0f;

    // Min depth allowed
    float min_depth = 0.1f;

    // Max depth allowed
    float max_depth = 10.0f;

    // Whether and weight to use tof in RGBD
    float tof_weight = 0.0f;

    // Integrate with ROI
    bool roi_fuse = false;

    // Depth filter options
    bool do_filter = false;
    float noise_filter_depth_thresh = 0.05f;
    int noise_filter_count_thresh = 15;
    float connectivity_filter_thresh = 0.05f;
    bool do_joint_filter = false;
    float joint_filter_completion_thresh = -1;

    // Components with faces less than this will be removed
    int num_isolated_pieces = 0;

    // Check the options for validity.
    bool Check() const;

    // Print the options to stdout.
    void Print() const;
  };

  StereoIntegration(const Options& options,
               const std::string& workspace_path,
               const std::string& input_type);

  void AddBlackList(const std::vector<uint8_t>& black_list);

 protected:
  void ReadWorkspace();

  void Run();
  void Fuse(const Options &options);
  void Clear();

  void Read(const std::string& path);

  void NoiseFilter(
    cv::Mat & mask,
    cv::Mat & depth,
    float depth_error_thresh,
    int statistic_count_thresh
  );
  void ComponentFilter(
    cv::Mat & mask,
    cv::Mat & depth,
    float connectivity_thresh,
    int min_piece_thresh,
    bool with_depth
  );
  void JointTrilateralFilter(
    const cv::Mat & joint, const cv::Mat & src, cv::Mat & dst, 
    int d, double sigmaColor, double sigmaDepth, double sigmaSpace, double completionThresh, 
    int sampleStep, int borderType
  );
  void DepthMapFilter(cv::Mat &depth, const cv::Mat &gray);

  int num_reconstruction_;

  const Options options_;
  const std::string workspace_path_;  // path of workspace
  std::string active_dense_path_;
  const std::string input_type_;

  std::vector<std::string> image_names_;
  std::vector<mvs::Image> images_;
  std::vector<std::unique_ptr<Bitmap> > semantic_maps_; 
  std::vector<std::unique_ptr<DepthMap> > depth_maps_;
  std::vector<std::unique_ptr<DepthMap> > tof_depth_maps_;
  std::vector<unsigned char> used_images_;
  std::vector<std::vector<int> > overlapping_images_;
  std::vector<Eigen::RowMatrix3x4f> P_;
  std::vector<Eigen::RowMatrix3x4f> inv_P_;
  std::vector<Eigen::RowMatrix3f> inv_R_;

  std::vector<uint8_t> semantic_label_black_list_;
};

}  // namespace mvs
}  // namespace sensemap

#endif  // SENSEMAP_MVS_INTEGRATION_H_
