//Copyright (c) 2019, SenseTime Group.
//All rights reserved.

#ifndef SENSEMAP_MVS_DELAUNAY_MESHING_H_
#define SENSEMAP_MVS_DELAUNAY_MESHING_H_

#include <iostream>
#include <dirent.h>
#include <sys/stat.h>
#include <boost/filesystem/path.hpp>
#include <util/mesh_info.h>

#include "util/misc.h"
#include "util/ply.h"
#include "util/obj.h"
#include "util/endian.h"
#include "util/timer.h"
#include "util/roi_box.h"
#include "util/proc.h"
#include "util/threading.h"
#include "controllers/patch_match_options.h"
#include "base/common.h"
#include "mvs/workspace.h"
#include "mvs/delaunay/delaunay_triangulation.h"

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/compute_average_spacing.h>
#include <CGAL/Search_traits_2.h>
#include <CGAL/Search_traits_3.h>
#include <CGAL/Orthogonal_k_neighbor_search.h>

namespace sensemap {
namespace mvs {

class DelaMeshing : public Thread {
 public:
  // EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  struct Options {
    bool sampInsert = true;
    int num_isolated_pieces = 0;
    float dist_insert = 5.0f;
    float diff_depth = 0.01f;
    float plane_insert_factor = 2.0;
    float plane_score_thred = 0.9;
    float decimate_mesh = 1.0f;
    float remove_spurious = 2.0f;
    float sigma = 1.0f;
    bool only_remove_edge_spurious = false;
    unsigned remove_spikes = 1;
    unsigned close_holes = 30;
    unsigned smooth_mesh = 10;
    unsigned fix_mesh = 0;

    // adaptive delaunay parameter.
    unsigned adaptive_insert = 0;
    
    bool roi_mesh = false;
    float roi_box_width = -1.f;
    float roi_box_factor = -1.f;

    // max memory capacity in gigabytes for fusion.  -1.0f means infinite
    bool mesh_cluster = false;
    float max_ram = -1.0f;
    float ram_eff_factor = 0.75;

    float overlap_factor = 0.015;

    // Check the options for validity.
    bool Check() const;

    // Print the options to stdout.
    void Print() const;
  };

  DelaMeshing(const Options& options,
              const std::string& worksapce_patch,
              const std::string& image_type,
              const int reconstruciotn_idx = -1,
              const int cluster_idx = -1);

 protected:
  void ReadWorkspace();

  void Run();

  void DelaunayRecon(const std::string &dense_reconstruction_path,
                     const std::string &dense_fused_path,
                     const std::string &undistort_image_path,
                     const std::string &output_path);

  void AdjustMesh(TriangleMesh& obj_mesh, float fDecimate);
  
  void MergeMeshing(const std::string &workspace_path);

  void ColorizingMesh(TriangleMesh& obj_mesh, 
                      const std::vector<mvs::Image>& images);

  // int PointsCluster(std::vector<std::set<std::size_t> > &point_cluster_map_set,
  //                   std::vector<struct Box> &cluster_bound_box,
  //                   const std::vector<PlyPoint> &ply_points,
  //                   const Model& model, const int max_num_points);

  int num_cluster_;
  int select_reconstruction_idx_;
  int select_cluster_idx_;

  const Options options_;
  const std::string workspace_path_;
  std::string image_type_;

  std::unique_ptr<Workspace> workspace_;
};

}  // namespace mvs
}  // namespace sensemap

#endif  // SENSEMAP_MVS_DELAUNAY_MESHING_H_
