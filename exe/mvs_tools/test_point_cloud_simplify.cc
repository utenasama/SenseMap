//Copyright (c) 2019, SenseTime Group.
//All rights reserved.

#include <boost/filesystem/path.hpp>
#include <iostream>
#include <unordered_map>
#include <string>
#include <set>
#include <dirent.h>
#include <sys/stat.h>
#include <gflags/gflags.h>

#include "util/obj.h"
#include "util/misc.h"
#include "util/ply.h"
#include "util/timer.h"
#include "base/common.h"
#include "util/exception_handler.h"
#include "mvs/workspace.h"

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/compute_average_spacing.h>
#include <CGAL/Search_traits_3.h>
#include <CGAL/squared_distance_3.h>
#include <CGAL/Orthogonal_k_neighbor_search.h>
#include <CGAL/license/Point_set_processing_3.h>

#include <CGAL/AABB_tree.h>
#include <CGAL/AABB_traits.h>
#include <CGAL/AABB_face_graph_triangle_primitive.h>
#include <CGAL/AABB_triangle_primitive.h>

#include <functional>
#include <vector>
#include <fstream>
#include <boost/tuple/tuple.hpp>

#ifdef CGAL_LINKED_WITH_TBB
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#endif

#include "base/version.h"

// Types
typedef CGAL::Exact_predicates_inexact_constructions_kernel kernel_t;
typedef kernel_t::Point_3 point_t;
typedef kernel_t::FT FT;
typedef CGAL::Search_traits_3<kernel_t> tree_traits_t;
typedef CGAL::Orthogonal_k_neighbor_search<tree_traits_t> neighbor_search_t;
typedef neighbor_search_t::iterator search_iterator_t;
typedef neighbor_search_t::Tree tree_t;
typedef neighbor_search_t::Distance Distance;
// Data type := index, followed by the point, followed by three integers that
// define the Red Green Blue color of the point.
// typedef boost::tuple<int, Point, int, int, int> IndexedPointWithColorTuple;
typedef boost::tuple<int, point_t> indexed_point_tuple_t;

DEFINE_string(workspace_path, "", "the path of workspace");
DEFINE_int32(filter_size, 6, "the diameter of kernel size");
DEFINE_double(diff_depth, 0.001, "the depth different thresthold");
// DEFINE_string(filter_x, "", "the factor of x-axis simple ratio");
// DEFINE_string(filter_y, "", "the factor of y-axis simple ratio");
// DEFINE_string(filter_z, "", "the factor of z-axis simple ratio");

using namespace sensemap;

const int dirs[6][3] = {
    {-1, 0, 0}, {1, 0, 0}, {0, -1, 0}, {0, 1, 0}, {0, 0, -1}, {0, 0, 1}
    // {-1, -1, -1}, {1, -1, -1}, {1, 1, -1}, {-1, 1, -1},
    // {-1, -1, 1}, {1, -1, 1}, {1, 1, 1}, {-1, 1, 1}
};

void Simplify(const std::vector<PlyPoint>& fused_points,
              const std::vector<std::vector<uint32_t> >& fused_viss, 
              std::vector<PlyPoint>& simplified_points,
              std::vector<std::vector<uint32_t> >& simplified_fused_viss,
              const double filter_x, const double filter_y, const double filter_z) {
    size_t i, j;
    size_t num_point = fused_points.size();
    bool has_vis = !fused_viss.empty();
    std::cout << "Number of points: " << num_point << std::endl;

    std::cout << "Compute Bounding Box" << std::endl;
    Eigen::Vector3f lt, rb;
    lt.setConstant(std::numeric_limits<float>::max());
    rb.setConstant(std::numeric_limits<float>::lowest());
    for (i = 0; i < num_point; ++i) {
        const PlyPoint& point = fused_points.at(i);
        lt.x() = std::min(lt.x(), point.x);
        lt.y() = std::min(lt.y(), point.y);
        lt.z() = std::min(lt.z(), point.z);
        rb.x() = std::max(rb.x(), point.x);
        rb.y() = std::max(rb.y(), point.y);
        rb.z() = std::max(rb.z(), point.z);
    }
    std::cout << "LT: " << lt.transpose() << std::endl;
    std::cout << "RB: " << rb.transpose() << std::endl;

    size_t lenx = (rb.x() - lt.x()) / filter_x + 1;
    size_t leny = (rb.y() - lt.y()) / filter_y + 1;
    size_t lenz = (rb.z() - lt.z()) / filter_z + 1;
    uint64_t slide = lenx * leny;

    std::unordered_map<uint64_t, std::vector<size_t> > m_voxels_map;
    for (i = 0; i < num_point; ++i) {
        const PlyPoint& point = fused_points.at(i);
        if (point.x < lt.x() || point.y < lt.y() || point.z < lt.z() ||
            point.x > rb.x() || point.y > rb.y() || point.z > rb.z()) {
            continue;
        }
        uint64_t ix = (point.x - lt.x()) / filter_x;
        uint64_t iy = (point.y - lt.y()) / filter_y;
        uint64_t iz = (point.z - lt.z()) / filter_z;

        uint64_t key = iz * slide + iy * lenx + ix;
        m_voxels_map[key].push_back(i);
    }

    std::vector<uint64_t> voxel_map_index;
    for (auto voxel_map : m_voxels_map) {
        // if (voxel_map.second.size() > 1) {
        voxel_map_index.push_back(voxel_map.first);
        // }
    }
    std::cout << "voxel map: " << voxel_map_index.size() << std::endl;

    // simplified_points.clear();
    // simplified_fused_viss.clear();
    simplified_points.resize(voxel_map_index.size());
    // simplified_fused_viss.resize(m_voxels_map.size());
    
    auto SamplePoint = [&](uint64_t voxel_key, PlyPoint *samp_point) {
        auto voxel_map = m_voxels_map[voxel_key];

        Eigen::Vector3f X(0, 0, 0);
        Eigen::Vector3f Xn(0, 0, 0);
        Eigen::Vector3f color(0, 0, 0);
        float sum_w(0.0);
        int max_samples(0);
        uint8_t best_label(-1), samps_per_label[256];
        memset(samps_per_label, 0, sizeof(uint8_t) * 256);

        for (int k = 0; k < voxel_map.size(); ++k) {
            size_t point_idx = voxel_map.at(k);
	          CHECK_LT(point_idx, num_point);
            const PlyPoint& point = fused_points.at(point_idx);
            float w;
            if (has_vis) {
                w = fused_viss.at(point_idx).size();
            } else {
                w = 1.0f;
            }
            X += w * Eigen::Vector3f(&point.x);
            Xn += w * Eigen::Vector3f(&point.nx);
            color[0] += w * point.r;
            color[1] += w * point.g;
            color[2] += w * point.b;
            sum_w += w;

            uint8_t sid = (point.s_id + 256) % 256;
            samps_per_label[sid]++;
            if (samps_per_label[sid] > max_samples) {
            max_samples = samps_per_label[sid];
            best_label = point.s_id;
            }
        }
        X /= sum_w;
        Xn = (Xn / sum_w).normalized();
        color /= sum_w;

        // PlyPoint point;
        (*samp_point).x = X[0];
        (*samp_point).y = X[1];
        (*samp_point).z = X[2];
        (*samp_point).nx = Xn[0];
        (*samp_point).ny = Xn[1];
        (*samp_point).nz = Xn[2];
        (*samp_point).r = color[0];
        (*samp_point).g = color[1];
        (*samp_point).b = color[2];
        (*samp_point).s_id = best_label;
    };
    for (size_t i = 0; i < voxel_map_index.size(); ++i) {
        SamplePoint(voxel_map_index.at(i), &simplified_points[i]);
    }
}


bool Filter(std::vector<PlyPoint> &fused_points, 
            const double max_spacing_factor,
            const unsigned int nb_neighbors) {
  if (fused_points.empty()) {
    return false;
  }

  std::size_t fused_point_num = fused_points.size();
  std::vector<point_t> points(fused_point_num);
  for (std::size_t i = 0; i < fused_point_num; i++) {
    const auto &fused_point = fused_points[i];
    points[i] = point_t(fused_point.x, fused_point.y, fused_point.z);
  }

  // Instantiate a KD-tree search.
  std::cout << "Instantiating a search tree for point cloud spacing ..." << std::endl;
  tree_t tree(points.begin(), points.end());
#ifdef CGAL_LINKED_WITH_TBB
  tree.build<CGAL::Parallel_tag>();
#endif

  // iterate over input points, compute and output point spacings
  FT average_spacing = (FT)0.0;
  std::vector<FT> point_spacings(fused_point_num);
#ifdef CGAL_LINKED_WITH_TBB
  {
    std::cout << "Starting average spacing computation(Parallel) ..." << std::endl;
    tbb::parallel_for (tbb::blocked_range<std::size_t> (0, fused_point_num),
                     [&](const tbb::blocked_range<std::size_t>& r) {
                       for (std::size_t s = r.begin(); s != r.end(); ++ s) {
                        // Neighbor search can be instantiated from
                        // several threads at the same time
                        const auto &query = points[s];
                        neighbor_search_t search(tree, query, nb_neighbors + 1);

                        auto &point_spacing = point_spacings[s];
                        point_spacing = 0.0f;
                        std::size_t k = 0;
                        Distance tr_dist;
                        for (neighbor_search_t::iterator it = search.begin(); it != search.end() && k <= nb_neighbors; it++, k++)
                        {
                          float dist = tr_dist.inverse_of_transformed_distance(it->second);
                          point_spacing += dist;
                        }
                        // output point spacing
                        if (k > 1) {
                          point_spacing /= (k - 1);
                        }
                       }
                     });
    for (auto & point_spacing : point_spacings) {
      average_spacing += point_spacing;
    }
  }
#else
  {
    std::cout << "Starting average spacing computation ..." << std::endl;
    for (std::size_t i = 0; i < fused_point_num; i++) {
      const auto &query = points[i];
      // performs k + 1 queries (if unique the query point is
      // output first). search may be aborted when k is greater
      // than number of input points
      neighbor_search_t search(tree, query, nb_neighbors + 1);
      auto &point_spacing = point_spacings[i];
      point_spacing = (FT)0.0;
      std::size_t k = 0;
      for (search_iterator_t search_iterator = search.begin(); search_iterator != search.end() && k <= nb_neighbors; search_iterator++, k++)
      {
        const auto &p = search_iterator->first;
        // point_spacing += std::sqrt(CGAL::squared_distance(query, p));
        point_spacing += std::sqrt(search_iterator->second);
      }
      // output point spacing
      if (k > 1) {
        point_spacing /= (FT)(k - 1);
      }

      average_spacing += point_spacing;
      // std::cout << "\r";
      // std::cout << "Point spacings computed [" << i + 1 << " / " << fused_point_num << "]" << std::flush;
    }
    // std::cout << std::endl;
  }
#endif
  average_spacing /= (FT)fused_point_num;
  std::cout << "Average spacing: " << average_spacing << std::endl;

  std::size_t point_count = 0;
  FT max_point_spacing = average_spacing * (FT)max_spacing_factor;
  for (std::size_t i = 0; i < fused_point_num; i++) {
      if (point_spacings[i] > max_point_spacing) {
      continue;
      }
      fused_points[point_count] = fused_points[i];
      point_count++;
  }
  fused_points.resize(point_count);
  std::cout << "Filter " << fused_point_num << " fused points to " << point_count << " ones." << std::endl;

  return true;
}

int main(int argc, char *argv[]) {

	PrintHeading("SenseMap.  Copyright(c) 2021, SenseTime Group.");	
	PrintHeading(std::string("Version: ") + __VERSION__);

    std::string help_info = StringPrintf("Usage: \n" \
        "./test_point_cloud_simplify --workspace_path=./workspace_path" \
        "                            --filter_size=8(pixel)"\
        "                            --diff_depth=0.005\n");
    google::SetUsageMessage(help_info.c_str());

    google::ParseCommandLineFlags(&argc, &argv, false);

    if (argc < 2) {
        std::cout << google::ProgramUsage() << std::endl;
		    return StateCode::NO_MATCHING_INPUT_PARAM;
    }

    std::cout << "workspace_path: " << FLAGS_workspace_path << std::endl;
    std::cout << "filter_size: " << FLAGS_filter_size << std::endl;
    std::cout << "diff_depth: " << FLAGS_diff_depth << std::endl;
    // std::cout << "filter_x: " << FLAGS_filter_x << std::endl;
    // std::cout << "filter_y: " << FLAGS_filter_y << std::endl;
    // std::cout << "filter_z: " << FLAGS_filter_z << std::endl;

    // double filter_x = std::atof(FLAGS_filter_x.c_str());
    // double filter_y = std::atof(FLAGS_filter_y.c_str());
    // double filter_z = std::atof(FLAGS_filter_z.c_str());

    const int filter_radius = FLAGS_filter_size / 2;

    Timer timer;
    timer.Start();

    for (int rect_id = 0; ; rect_id++){
        auto reconstruction_path =
            JoinPaths(FLAGS_workspace_path.c_str(), std::to_string(rect_id));
        if (!ExistsDir(reconstruction_path)){
            break;
        }

        auto dense_reconstruction_path = JoinPaths(reconstruction_path.c_str(), DENSE_DIR);
        auto undistort_image_path = JoinPaths(dense_reconstruction_path, IMAGES_DIR);
        auto input_path = JoinPaths(dense_reconstruction_path, FUSION_NAME);

        if (!ExistsFile(input_path + ".vis")) {
          continue;
        }
            
        std::vector<PlyPoint> fused_points = ReadPly(input_path);
        std::vector<std::vector<uint32_t> > fused_points_visibility;
        ReadPointsVisibility(input_path + ".vis", fused_points_visibility);
        if (ExistsFile(input_path + ".sem")) {
            ReadPointsSemantic(input_path + ".sem", fused_points);
        }

        std::cout << "fused_points: " << fused_points.size() << std::endl;

        mvs::Workspace::Options workspace_options;

        workspace_options.max_image_size = -1;
        workspace_options.image_as_rgb = false;
        workspace_options.image_path = undistort_image_path;
        workspace_options.workspace_path = dense_reconstruction_path;
        workspace_options.workspace_format = "perspective";

        mvs::Workspace workspace(workspace_options);
        const mvs::Model model = workspace.GetModel();
        const std::vector<mvs::Image>& images = model.images;

        size_t i, j;

        std::vector<std::vector<uint32_t> > image_points(images.size());
        for (i = 0; i < fused_points_visibility.size(); ++i) {
          std::vector<uint32_t>& viss = fused_points_visibility.at(i);
          for (auto vis : viss) {
              image_points.at(vis).push_back(i);
          }
        }

        std::vector<bool> removed(fused_points.size());
        std::fill(removed.begin(), removed.end(), false);
        for (i = 0; i < image_points.size(); ++i) {
          const mvs::Image& image = images.at(i);
          const int width = image.GetWidth();
          const int height = image.GetHeight();

          MatXu mask(width, height, 1);
          mask.Fill(0);
          MatXf zBuffer(width, height, 1);
          zBuffer.Fill(FLT_MAX);

          Eigen::RowMatrix3x4f P(image.GetP());
          Eigen::RowMatrix3x4f invP(image.GetInvP());
          for (auto point_idx : image_points.at(i)) {
            if (removed.at(point_idx)) {
              continue;
            }
            PlyPoint& point = fused_points.at(point_idx);
            Eigen::Vector3f X(&point.x);
            Eigen::Vector3f proj = P * X.homogeneous();
            int u = proj.x() / proj.z();
            int v = proj.y() / proj.z();
            if (u < 0 || u >= width || v < 0 || v >= height || proj.z() <= 0) {
              removed.at(point_idx) = true;
              continue;
            }

            float d = zBuffer.Get(v, u);
            float diff_depth = std::fabs(d - proj.z()) / proj.z();
            if (mask.Get(v, u) && diff_depth <= FLAGS_diff_depth) {
              removed.at(point_idx) = true;
              continue;
            }
            int u_min = std::max(0, u - filter_radius);
            int v_min = std::max(0, v - filter_radius);
            int u_max = std::min(width - 1, u + filter_radius);
            int v_max = std::min(height - 1, v + filter_radius);
            for (int r = v_min; r <= v_max; ++r) {
              for (int c = u_min; c <= u_max; ++c){
                mask.Set(r, c, 1);
                if (proj.z() < d) {
                  zBuffer.Set(r, c, proj.z());
                }
              }
            }
          }
          std::cout << std::flush << StringPrintf("\rProcess Image#%d/%d", i, images.size());
        }
        std::cout << std::endl;

        for (i = 0, j = 0; i < fused_points.size(); ++i) {
          if (!removed.at(i)) {
            fused_points.at(j) = fused_points.at(i);
            fused_points_visibility.at(j) = fused_points_visibility.at(i);
            j = j + 1;
          }
        }
        std::cout << StringPrintf("Simplying fused points to %d / %d\n", j, fused_points.size());

        fused_points.resize(j);
        fused_points.shrink_to_fit();
        fused_points_visibility.resize(j);
        fused_points_visibility.shrink_to_fit();

        Filter(fused_points, 6, 6);

        std::string output_path = JoinPaths(dense_reconstruction_path, "fused_samp.ply");
        WriteBinaryPlyPoints(output_path, fused_points, false, true);
        // WritePointsVisibility(output_path + ".vis", fused_points_visibility);

        if (ExistsFile(input_path + ".sem")) {
            WritePointsSemantic(output_path + ".sem", fused_points);
            output_path = JoinPaths(dense_reconstruction_path, "fused_samp_semvis.ply");
            WritePointsSemanticColor(output_path, fused_points);
        }

        // std::vector<PlyPoint> simplified_points;
        // std::vector<std::vector<uint32_t> > simplified_fused_points_visibility;
        // Simplify(fused_points, fused_points_visibility, simplified_points,
        //         simplified_fused_points_visibility, filter_x, filter_y, filter_z);

        // // Filter(simplified_points, 6, 6);

        // std::string output_path = JoinPaths(reconstruction_path.c_str(), "dense/fused_samp.ply");
        // WriteBinaryPlyPoints(output_path, simplified_points);

        // if (ExistsFile(input_path + ".sem")) {
        //     WritePointsSemantic(output_path + ".sem", simplified_points);
        //     output_path = JoinPaths(reconstruction_path.c_str(), "dense/fused_samp_semvis.ply");
        //     WritePointsSemanticColor(output_path, simplified_points);
        // }

    }

    timer.PrintMinutes();

    return 0;
}