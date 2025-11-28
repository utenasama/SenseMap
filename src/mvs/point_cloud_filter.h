//Copyright (c) 2019, SenseTime Group.
//All rights reserved.

#ifndef SENSEMAP_MVS_POINT_CLOUD_FILTER_H_
#define SENSEMAP_MVS_POINT_CLOUD_FILTER_H_

#include <unordered_set>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/StdVector>
// #include "util/alignment.h"
#include "util/types.h"
#include "util/math.h"
#include "util/ply.h"
#include "util/obj.h"
#include "util/threading.h"
#include "util/bitmap.h"
#include "util/mat.h"
#include "mvs/image.h"
#include "mvs/model.h"
#include "mvs/workspace.h"
#include "base/common.h"


namespace sensemap {
namespace mvs {

class PointCloudFilter : public Thread {
 public:
  // EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  enum FilterMethod {
    STATISTIC_FILTER = 0,
    MODEL_FILTER = 1,
    CONFIDENCE_FILTER = 2
  };

  struct Options {
    
    FilterMethod method = STATISTIC_FILTER;

    // Camera format.
    std::string format = "perspective";

    double min_grad_thres = 1.2;

    double conf_thres = 0.7;

    int win_size = 3;

    int trust_region = 7;

    int nb_neighbors = 6;

    // Maximum spacing factor for noisy isolated points filtering (6.0 is a normal choice).
    // Points with spacing larger than max_spacing_factor * average_spacing will be removed.
    // If max_spacing_factor < 1.0, no filtering is done.
    double max_spacing_factor = 6.0;

    // Check the options for validity.
    bool Check() const;

    // Print the options to stdout.
    void Print() const;
  };

  PointCloudFilter(const Options& options,
                   const std::string& workspace_path);

 protected:
  void ReadWorkspace();

  void Run();

  bool FilterWithNCC(
              std::vector<mvs::Image>& images,
              std::vector<PlyPoint>& fused_points,
              std::vector<std::vector<uint32_t>>& fused_points_visibility);

  bool Filter(std::vector<PlyPoint> &fused_points,
              std::vector<std::vector<uint32_t> > &fused_points_visibility,
              const double max_spacing_factor = 6.0,
              const unsigned int nb_neighbors = 6);

  bool Filter(std::vector<PlyPoint> &fused_points,
              std::vector<std::vector<uint32_t> > &fused_points_visibility,
              const TriangleMesh &mesh);

  void Read(const std::string& path);

  int num_reconstruction_;

  const Options options_;
  const std::string workspace_path_;  // path of workspace
};

}  // namespace mvs
}  // namespace sensemap

#endif  // SENSEMAP_MVS_POINT_CLOUD_FILTER_H_
