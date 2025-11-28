//Copyright (c) 2019, SenseTime Group.
//All rights reserved.

#ifndef SENSEMAP_MVS_CAMERA_CLUSTER_H_
#define SENSEMAP_MVS_CAMERA_CLUSTER_H_

#include <iostream>
#include <queue>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <dirent.h>
#include <sys/stat.h>
#include <boost/filesystem/path.hpp>

#include <Eigen/Dense>

#include "util/threading.h"
#include "base/reconstruction_manager.h"
#include "util/ply.h"


namespace sensemap {
namespace mvs {

class CameraCluster : public Thread {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  struct Options {
    int min_pts_per_cluster = 50000;   // Minimum number of fused points for each camera cluster.
    int max_pts_per_cluster = 100000;   // Maximum number of fused points for each camera cluster.
    int min_mappts_per_cluster = 5000;   // Minimum number of mappoints for each camera cluster.
    int max_mappts_per_cluster = 10000;   // Maximum number of mappoints for each camera cluster.

    // Minimum number of fused pixels to produce a point.
    int min_num_pixels = 5;

    // Check the options for validity.
    bool Check() const;

    // Print the options to stdout.
    void Print() const;
  };

  CameraCluster(const Options& options,
                    const std::string& workspace_path);

 protected:
  int num_reconstruction_;
  const CameraCluster::Options options_;
  const std::string workspace_path_;

  void Run();
  void ReadWorkspace();

  std::shared_ptr<ReconstructionManager>
    Cluster(std::shared_ptr<Reconstruction> reconstruction);

  std::shared_ptr<ReconstructionManager>
    Cluster(std::vector<std::vector<std::size_t> > &fused_point_idxs_clusters,
            std::vector<std::vector<std::vector<uint32_t> > > &fused_points_visibility_clusters,
            std::shared_ptr<Reconstruction> reconstruction,
            const std::vector<PlyPoint> &fused_points,
            const std::vector<std::vector<uint32_t> > &fused_points_visibility);
};

}  // namespace mvs
}  // namespace sensemap

#endif  // SENSEMAP_MVS_CAMERA_CLUSTER_H_
