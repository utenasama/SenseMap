//Copyright (c) 2019, SenseTime Group.
//All rights reserved.

#ifndef SENSEMAP_MVS_MESH_CLUSTER_H_
#define SENSEMAP_MVS_MESH_CLUSTER_H_

#include <iostream>
#include <queue>
#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <dirent.h>
#include <sys/stat.h>
#include <boost/filesystem/path.hpp>

#include <Eigen/Dense>

#include "util/threading.h"

namespace sensemap {
namespace mvs {

class MeshCluster : public Thread {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  struct Options {
    int min_faces_per_cluster = 50000;   // Minimum number of faces for each mesh cluster.
    int max_faces_per_cluster = 100000;   // Maximum number of faces for each mesh cluster.
    // cell_size_factor * [average spacing] is the size of cell in each dimension, usually for cases without scale.
    // If cell_size_factor <= 0, cell_size is used; else, compute cell_size using cell_size_factor.
    float cell_size_factor = 100.0f;
    // Size of cell in each dimension, usually for cases with scale and dimension.
    float cell_size = 6.0f;

    double valid_spacing_factor = 2.5;

    // Check the options for validity.
    bool Check() const;

    // Print the options to stdout.
    void Print() const;
  };

  MeshCluster(const Options& options,
                    const std::string& workspace_path);

 protected:
  int num_reconstruction_;
  const MeshCluster::Options options_;
  const std::string workspace_path_;

  void Run();
  void ReadWorkspace();

  std::size_t GridCluster(std::vector<int> &cell_cluster_map,
                          const std::vector<std::size_t> &cell_point_count,
                          const std::vector<float> &cell_average_spacing,
                          const std::size_t grid_size_x,
                          const std::size_t grid_size_y,
                          const float valid_spacing);

  std::size_t Cluster(std::vector<int> &face_cluster_map,
                      const TriangleMesh &mesh);
};

}  // namespace mvs
}  // namespace sensemap

#endif  // SENSEMAP_MVS_MESH_CLUSTER_H_
