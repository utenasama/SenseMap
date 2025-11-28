#include <iomanip>
#include <numeric>
#include <list>

#include "util/misc.h"
#include "util/threading.h"
#include "util/string.h"

#include "mvs/utils.h"
#include "mvs/depth_map.h"
#include "mvs/point_cloud_filter.h"
#include "mvs/point_cloud_filter_cuda.h"

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

namespace sensemap {
namespace mvs {

using namespace utility;

namespace {
float ComputeNCCScore(const std::vector<float>& grays1,
                      const std::vector<float>& grays2) {
  const float kMaxCost = 2.0f;
  const int num_pixel = grays1.size();

  float color_sum1 = 0.0f, color_sum2 = 0.0f;
  float color_squared_sum1 = 0.0f, color_squared_sum2 = 0.0f;
  float color_sum12 = 0.0f;
  
  for (int i = 0; i < num_pixel; ++i) {
    float gray1 = grays1.at(i);
    color_sum1 += gray1;
    color_squared_sum1 += gray1 * gray1;

    float gray2 = grays2.at(i);
    color_sum2 += gray2;
    color_squared_sum2 += gray2 * gray2;

    color_sum12 += gray1 * gray2;
  }

  color_sum1 /= num_pixel;
  color_squared_sum1 /= num_pixel;
  color_sum2 /= num_pixel;
  color_squared_sum2 /= num_pixel;
  color_sum12 /= num_pixel;

  const float color_var1 = color_squared_sum1 - color_sum1 * color_sum1;
  const float color_var2 = color_squared_sum2 - color_sum2 * color_sum2;

  if (color_var1 < 1e-5 || color_var2 < 1e-5) {
      return kMaxCost;
  }

  const float color_cover12 = color_sum12 - color_sum1 * color_sum2;
  const float color_var12 = std::sqrt((color_var1 * color_var2));
  const float val = 1.0f - color_cover12 / color_var12;
  float score = std::max(0.0f, std::min(kMaxCost, val));

  return score;
}
}

void PointCloudFilter::Options::Print() const {
  PrintHeading2("PointCloudFilter::Options");
  PrintOption(max_spacing_factor);
}

bool PointCloudFilter::Options::Check() const {
  CHECK_OPTION_GE(max_spacing_factor, 0.0);
  return true;
}

PointCloudFilter::PointCloudFilter(const Options& options,
                                   const std::string& workspace_path)
    : num_reconstruction_(0),
      options_(options),
      workspace_path_(workspace_path) {
  CHECK(options_.Check());
}

void PointCloudFilter::ReadWorkspace() {
  num_reconstruction_ = 0;
  std::cout << "Reading workspace..." << std::endl;
    for (size_t reconstruction_idx = 0; ;reconstruction_idx++) {
        const auto& reconstruction_path = 
            JoinPaths(workspace_path_, std::to_string(reconstruction_idx));
        const auto& dense_reconstruction_path = 
            JoinPaths(reconstruction_path, DENSE_DIR);
        if (!ExistsDir(dense_reconstruction_path)) {
            break;
        }

        num_reconstruction_++;
    }
}

void PointCloudFilter::Run() {
  options_.Print();
  std::cout << std::endl;

  ReadWorkspace();

  for (size_t reconstruction_idx = 0; reconstruction_idx < num_reconstruction_; 
       reconstruction_idx++) {

    PrintHeading1(StringPrintf("Filtering# %d", reconstruction_idx));
    
    if (IsStopped()) {
      GetTimer().PrintMinutes();
      return;
    }

    auto reconstruction_path = 
      JoinPaths(workspace_path_, std::to_string(reconstruction_idx));
    auto dense_reconstruction_path = JoinPaths(reconstruction_path, DENSE_DIR);
    if (!ExistsDir(dense_reconstruction_path)) {
      continue;
    }
    auto undistort_image_path = JoinPaths(dense_reconstruction_path, IMAGES_DIR);

    auto fused_path = JoinPaths(dense_reconstruction_path, FUSION_NAME);
    if (!ExistsFile(fused_path)) {
      continue;
    }

    std::vector<std::string> image_names;
    std::vector<mvs::Image> images;
    std::vector<std::vector<int> > overlapping_images;
    if (options_.method == FilterMethod::MODEL_FILTER) {
      if (options_.format.compare("panorama") == 0) {
        std::vector<std::pair<float, float> > depth_ranges;
        std::vector<image_t> image_ids;
        ImportPanoramaWorkspace(dense_reconstruction_path, image_names, images, 
          image_ids, overlapping_images, depth_ranges, false);
        std::cout << "done!" << std::endl;
      } else {
        Workspace::Options workspace_options;
        workspace_options.max_image_size = -1;
        workspace_options.image_as_rgb = true;
        workspace_options.workspace_path = dense_reconstruction_path;
        workspace_options.workspace_format = options_.format;
        workspace_options.image_path = undistort_image_path;

        std::unique_ptr<Workspace> workspace;
        workspace.reset(new Workspace(workspace_options));
        const Model& model = workspace->GetModel();

        images = model.images;
      }
    }

    std::vector<PlyPoint> fused_points = ReadPly(fused_path);
    std::vector<std::vector<uint32_t> > fused_points_visibility;
    if (ExistsFile(fused_path + ".vis")) {
      ReadPointsVisibility(fused_path + ".vis", fused_points_visibility);
    }
    const std::string fused_sem_path = fused_path + ".sem";
    if (ExistsFile(fused_sem_path)) {
      ReadPointsSemantic(fused_sem_path, fused_points, false);
    }

    if (fused_points.empty()) {
        std::cout << "WARNING: Could not fuse any points. This is likely "
                     "caused by incorrect settings - filtering must be enabled "
                     "for the last call to patch match stereo."
                  << std::endl;
        continue;
    }

    std::cout << "Number of fused points: " << fused_points.size() 
              << std::endl;

    if (options_.method == FilterMethod::CONFIDENCE_FILTER) {
      FilterWithNCC(images, fused_points, fused_points_visibility);
    } else if (options_.method == FilterMethod::MODEL_FILTER) {
      const auto model_path = JoinPaths(dense_reconstruction_path, MODEL_NAME);
      TriangleMesh mesh;
      if (ExistsFile(model_path)) {
        ReadTriangleMeshObj(model_path, mesh, true, false);
      }
      if (!mesh.faces_.empty()) {
        std::cout << "Filtering isolated points(Model Prior)" << std::endl;
        Filter(fused_points, fused_points_visibility, mesh);
      }
    } else if (options_.method == FilterMethod::STATISTIC_FILTER) {
      if (options_.max_spacing_factor >= 1.0) {
        std::cout << "Filtering isolated points" << std::endl;
        Filter(fused_points, fused_points_visibility, 
          options_.max_spacing_factor, options_.nb_neighbors);
      }
    }

    GetTimer().PrintMinutes();

    auto filtered_fused_path = JoinPaths(dense_reconstruction_path, FILTER_FUSION_NAME);
    const std::string filtered_fused_sem_path = filtered_fused_path + ".sem";

    std::cout << "Writing fusion output: " << filtered_fused_path << std::endl;
    WriteBinaryPlyPoints(filtered_fused_path, fused_points, false, true);
    if (ExistsFile(fused_path + ".vis")) {
      WritePointsVisibility(filtered_fused_path + ".vis", fused_points_visibility);
    }
    if (ExistsFile(fused_sem_path)) {
      WritePointsSemantic(filtered_fused_sem_path, fused_points, false);
      WritePointsSemanticColor(JoinPaths(dense_reconstruction_path, FILTER_FUSION_SEM_NAME), fused_points);
    }
  }
}

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

bool PointCloudFilter::FilterWithNCC(
  std::vector<mvs::Image>& images,
  std::vector<PlyPoint>& fused_points,
  std::vector<std::vector<uint32_t>>& fused_points_visibility) {

  size_t i, j;

  // Load bitmap.
  size_t num_image = images.size();
  std::vector<Mat<float> > dist_maps(num_image);

  int interval = 100;
  while(num_image < interval) {
    interval /= 10;
  }

  for (i = 0; i < num_image; ++i) {
    if (i % interval == 0) {
      std::cout << StringPrintf("\rLoading Image %d / %d", i + 1, num_image);
    }
    Bitmap bitmap;
    bitmap.Read(images.at(i).GetPath(), false);
    images.at(i).SetBitmap(bitmap);

    ComputeDistanceMap(bitmap, dist_maps.at(i), options_.min_grad_thres);
    // DepthMap(dist_maps.at(i), -1, -1).ToBitmap().Write(image_path + "-dist.jpg");

  }
  std::cout << std::endl;

  const int win_r = options_.win_size / 2;
  std::vector<float> scores(fused_points.size());

  for (i = 0; i < fused_points.size(); ++i) {
    if (i % 1000 == 0) {
      std::cout << StringPrintf("\rProcess point %d / %d", i + 1, fused_points.size());
    }
    const PlyPoint& point = fused_points.at(i);
    const Eigen::Vector3f X(&point.x);
    const std::vector<uint32_t>& vvis = fused_points_visibility.at(i);

    std::vector<std::vector<float> > vgrays;
    for (auto& image_id : vvis) {
      const mvs::Image& image = images.at(image_id);
      const Bitmap& bitmap = image.GetBitmap();
      Eigen::RowMatrix3x4f P(image.GetP());
      Eigen::Vector3f proj = P * X.homogeneous();
      int u = proj.x() / proj.z();
      int v = proj.y() / proj.z();
      if (u < win_r || u >= image.GetWidth() - win_r ||
          v < win_r || v >= image.GetHeight() - win_r) {
        continue;
      }

      float dist = dist_maps.at(image_id).Get(v, u);
      if (dist > 0 && dist < options_.trust_region) { 
        std::vector<float> grays;
        for (int x = u - win_r; x <= u + win_r; ++x) {
          for (int y = v - win_r; y <= v + win_r; ++y) {
            BitmapColor<uint8_t> val = bitmap.GetPixel(x, y);
            grays.push_back(val.r / 255.0f);
          }
        }
        vgrays.emplace_back(grays);
      }
    }
    if (vgrays.size() < 2) {
      continue;
    }

    std::vector<float> image_pair_scores;
    for (int j1 = 0; j1 < vgrays.size(); ++j1) {
      for (int j2 = j1 + 1; j2 < vgrays.size(); ++j2) {
        float score = ComputeNCCScore(vgrays.at(j1), vgrays.at(j2));
        image_pair_scores.push_back(score);
      }
    }
    size_t nth = image_pair_scores.size() / 2;
    std::nth_element(image_pair_scores.begin(), image_pair_scores.begin() + nth, image_pair_scores.end());
    scores[i] = image_pair_scores.at(nth);
  }
  std::cout << std::endl;

  for (i = 0, j = 0; i < fused_points.size(); ++i) {
    if (scores[i] < options_.conf_thres) {
      fused_points.at(j) = fused_points.at(i);
      fused_points_visibility.at(j) = fused_points_visibility.at(i);
      j = j + 1;
    }
  }
  std::cout << StringPrintf("Remove %d points\n", fused_points.size() - j);
  fused_points.resize(j);
  fused_points_visibility.resize(j);

  return true;
}

bool PointCloudFilter::Filter(std::vector<PlyPoint> &fused_points,
                              std::vector<std::vector<uint32_t> > &fused_points_visibility,
                              const double max_spacing_factor,
                              const unsigned int nb_neighbors) {
  if (fused_points.empty()) {
    return false;
  }

  std::size_t fused_point_num = fused_points.size();
  bool has_vis = !fused_points_visibility.empty();

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
    std::cout << "Starting average spacing computation ..." << std::endl;
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
    if (has_vis) {
      fused_points_visibility[point_count] = fused_points_visibility[i];
    }
    point_count++;
  }
  fused_points.resize(point_count);
  if (has_vis) {
    fused_points_visibility.resize(point_count);
  }
  std::cout << "Filter " << fused_point_num << " fused points to " << point_count << " ones." << std::endl;

  return true;
}

typedef CGAL::Simple_cartesian<double>                            Simple_Kernel;
typedef Simple_Kernel::Point_3                                    Point;
typedef Simple_Kernel::Vector_3                                   Vector;
typedef Simple_Kernel::Segment_3                                  Segment;
typedef Simple_Kernel::Ray_3                                      Ray;
typedef Simple_Kernel::Triangle_3                                 Triangle;
typedef std::list<Triangle>::iterator                             Iterator;
typedef CGAL::AABB_triangle_primitive<Simple_Kernel, Iterator>    Primitive;
typedef CGAL::AABB_traits<Simple_Kernel, Primitive>               Traits;
typedef CGAL::AABB_tree<Traits>                                   Tree;
typedef Tree::Primitive_id                                        Primitive_id;

bool PointCloudFilter::Filter(std::vector<PlyPoint> &fused_points,
                              std::vector<std::vector<uint32_t> > &fused_points_visibility,
                              const TriangleMesh &mesh) {
  if (fused_points.empty() || mesh.faces_.empty()) {
    return false;
  }

  std::size_t fused_point_num = fused_points.size();

  size_t i, j;
  double median_dist = 0;
  std::vector<double> dists;
  std::list<Triangle> triangles;

  for (auto facet : mesh.faces_) {
      auto a = mesh.vertices_.at(facet[0]);
      auto b = mesh.vertices_.at(facet[1]);
      auto c = mesh.vertices_.at(facet[2]);
      if ((a - b).norm() < 1e-6 || (b - c).norm() < 1e-6 || (a - c).norm() < 1e-6) {
          continue;
      }
      Triangle tri(Point(a[0], a[1], a[2]), Point(b[0], b[1], b[2]), 
                  Point(c[0], c[1], c[2]));
      triangles.emplace_back(tri);
  }
  std::cout << "Construct AABB Tree" << std::endl;
  // Contruct AABB tree.
  Tree tree(triangles.begin(), triangles.end());

  tree.accelerate_distance_queries();
  
  dists.resize(fused_points.size());
  for (i = 0; i < fused_points.size(); ++i) {
      auto point = fused_points.at(i);
      Point query(point.x, point.y, point.z);
      double dist = tree.squared_distance(query);
      dists[i] = dist;
  }
  std::vector<double> dists_sort = dists;
  size_t nth = dists_sort.size() * 0.85;
  std::nth_element(dists_sort.begin(), dists_sort.begin() + nth,
      dists_sort.end());
  median_dist = dists_sort[nth];
  std::cout << "Median distance of point to mesh: " << median_dist 
            << std::endl;

  bool has_vis = !fused_points_visibility.empty();

  for (i = 0, j = 0; i < fused_points.size(); ++i) {
      auto point = fused_points.at(i);
      Eigen::Vector3d X = Eigen::Vector3f((float*)&point).cast<double>();
      
      if (dists[i] > 1.2 * median_dist) {
          continue;
      }

      fused_points[j] = fused_points[i];
      if (has_vis) {
        fused_points_visibility[j] = fused_points_visibility[i];
      }
      j = j + 1;
  }
  
  fused_points.resize(j);
  if (has_vis) {
    fused_points_visibility.resize(j);
  }

  return true;
}

}  // namespace mvs
}  // namespace sensemap
