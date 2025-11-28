#include <iomanip>
#include <numeric>
#include <algorithm>

#include <malloc.h>
#include "yaml-cpp/yaml.h"

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Search_traits_3.h>
#include <CGAL/Search_traits_adapter.h>
#include <CGAL/point_generators_3.h>
#include <CGAL/Orthogonal_k_neighbor_search.h>
#include <CGAL/property_map.h>
#include <CGAL/tags.h>
#include <boost/iterator/zip_iterator.hpp>
#ifdef CGAL_LINKED_WITH_TBB
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#endif

#include "util/misc.h"
#include "util/threading.h"
#include "util/kmeans.h"
#include "util/math.h"

#include "estimators/plane_estimator.h"
#include "optim/ransac/loransac.h"

#include "mvs/utils.h"
#include "mvs/fusion.h"
#include "util/proc.h"
#include "mvs/mvs_cluster.h"

#include "mvs/delaunay/delaunay_triangulation.h"

#include "util/exception_handler.h"

namespace sensemap {
namespace mvs {

const int num_perspective_per_image = 6;

using namespace utility;

namespace internal {

template <typename T>
float Median(std::vector<T>* elems) {
  CHECK(!elems->empty());
  const size_t mid_idx = elems->size() / 2;
  std::nth_element(elems->begin(), elems->begin() + mid_idx, elems->end());
  if (elems->size() % 2 == 0) {
    const float mid_element1 = static_cast<float>((*elems)[mid_idx]);
    const float mid_element2 = static_cast<float>(
        *std::max_element(elems->begin(), elems->begin() + mid_idx));
    return (mid_element1 + mid_element2) / 2.0f;
  } else {
    return static_cast<float>((*elems)[mid_idx]);
  }
}

template <typename T>
float Var(std::vector<T>* elems) {
  float sum(0);
  sum = std::accumulate(elems->begin(), elems->end(), 0.0f);
  float mean(sum / (int)elems->size());
  float stdev(0);
  for (auto val : *elems) {
    stdev += (val - mean) * (val - mean);
  }
  return std::sqrt(stdev / ((int)elems->size() - 1));
}

template <typename T>
void MeanStdev(std::vector<T>* elems, float* m, float *stdev) {
  float sum(0);
  sum = std::accumulate(elems->begin(), elems->end(), 0.0f);
  *m = sum / (int)elems->size();

  *stdev = 0;
  for (auto val : *elems) {
    *stdev += (val - *m) * (val - *m);
  }
  *stdev = std::sqrt(*stdev / ((int)elems->size() - 1));
}

//// Use the sparse model to find most connected image that has not yet been
//// fused. This is used as a heuristic to ensure that the workspace cache reuses
//// already cached images as efficient as possible.
//int FindNextImage(const std::vector<std::vector<int> >& overlapping_images,
//                  const std::vector<unsigned char>& used_images,
//                  const std::vector<unsigned char>& fused_images,
//                  const int prev_image_idx) {
//  CHECK_EQ(used_images.size(), fused_images.size());
//
//  for (const auto image_idx : overlapping_images.at(prev_image_idx)) {
//    if (used_images.at(image_idx) && !fused_images.at(image_idx)) {
//      return image_idx;
//    }
//  }
//
//  // If none of the overlapping images are not yet fused, simply return the
//  // first image that has not yet been fused.
//  for (size_t image_idx = 0; image_idx < fused_images.size(); ++image_idx) {
//    if (used_images[image_idx] && !fused_images[image_idx]) {
//      return image_idx;
//    }
//  }
//
//  return -1;
//}

const static float BLACK = 20.0;
const static float YELLOW = 70.0;

void Lab2RGB(const float lab0, const float lab1, const float lab2,
             float &rgb0, float &rgb1, float &rgb2) {
  float l, a, b;
  l = lab0 * 100.0f / 255;
  a = lab1 - 128;
  b = lab2 - 128;
   
   float x, y, z;
   float fx, fy, fz;
   
   fy = (l+16.0)/116.0;
   
   if(fy > 0.206893f) {
    y=fy * fy * fy;
   } else {
    y = l/903.3;
    fy = 7.787*y+16.0/116.0;
   }
   
   fx = a/500.0 + fy;
   if(fx > 0.206893) {
    x = pow(fx,3.0);
   } else {
    x = (fx-16.0/116.0)/7.787;
   }
   
   fz = fy - b/200.0;
   if(fz > 0.206893) {
    z = pow(fz,3);
   } else {
    z = (fz-16.0/116.0)/7.787;
   }
   
   x = x*0.950456*255.0;
   y = y*255.0;
   z = z*1.088754*255.0;
   
   float dr, dg, db;
   // [ R ]   [  3.240479 -1.537150 -0.498535 ]   [ X ]
   // [ G ] = [ -0.969256  1.875992  0.041556 ] * [ Y ]
   // [ B ]   [  0.055648 -0.204043  1.057311 ]   [ Z ]
   dr =  3.240479*x  - 1.537150*y - 0.498535*z;
   dg =  -0.969256*x + 1.875992*y + 0.041556*z;
   db =  0.055648*x  - 0.204043*y + 1.057311*z;
   
   // 防止溢出
   if(dr<0.0) {
    rgb0 = 0;
   } else if(dr>255.0) {
    rgb0 = 255;
   } else {
    rgb0 = dr;
   }
   
   if(dg<0.0) {
    rgb1 = 0;
   } else if(dg>255.0) {
    rgb1 = 255;
   } else {
    rgb1 = dg;
   }
   
   if(db<0.0) {
    rgb2 = 0;
   } else if(db>255.0) {
    rgb2 = 255;
   } else {
    rgb2 = db;
   }
}

void RGB2Lab(const float r, const float g, const float b,
             float &lab0, float &lab1, float &lab2) {
  float x, y, z;
  float fx, fy, fz;
   // 转至X-Y-Z
   //[ X ]   [ 0.412453  0.357580  0.180423 ]   [ R ]
   //[ Y ] = [ 0.212671  0.715160  0.072169 ] * [ G ]
   //[ Z ]   [ 0.019334  0.119193  0.950227 ]   [ B ]
   x = 0.412453*r + 0.357580*g + 0.180423*b;
   y = 0.212671*r + 0.715160*g + 0.072169*b;
   z = 0.019334*r + 0.119193*g + 0.950227*b;
   
   // 除255即归一化
   x = x/(255.0*0.950456);
   y = y/255.0;
   z = z/(255.0*1.088754);
   
   if(y>0.008856) {
    fy = pow(y,1.0/3.0);
    lab0 = 116.0*fy-16.0;
   } else {
    fy = 7.787*y + 16.0/116.0;
    lab0 = 903.3*y;
   }
   
   if(x>0.008856) {
    fx = pow(x,1.0/3.0);
   } else {
    fx = 7.787*x + 16.0/116.0;
   }
   
   if(z>0.008856) {
    fz = pow(z,1.0/3.0);
   } else {
    fz = 7.787*z + 16.0/116.0;
   }
   
   lab1 = 500.0*(fx-fy);
   lab2 = 200.0*(fy-fz);
   
   // 这里不加时出现颜色饱和的情况(见上图)
   // 参考出处http://c.chinaitlab.com/cc/ccjq/200806/752572.html
   if (lab0 < BLACK) {
    lab1 *= exp((lab0 - BLACK) / (BLACK/ 4));
    lab2 *= exp((lab0 - BLACK) / (BLACK/ 4));
    lab0 = 20;
   }
   if (lab2 > YELLOW)lab2 = YELLOW;
   
   // 归一化值Lab
   lab0 = lab0 * 255.0 / 100;    // L
   lab1 = (lab1 + 128.0); // a
   lab2 = (lab2 + 128.0); // b
}

bool FitPlane(const std::vector<Eigen::Vector3d> &points,
              const std::vector<Eigen::Vector3d> &cameras_center,
              Eigen::Vector4d &plane) {
  if (points.size() < 3) {
    std::cout << "No enough points to fit a plane!" << std::endl;
    return false;
  }

  PlaneLocalEstimator local_estimator;
  std::vector<Eigen::Vector4d> camera_models = local_estimator.Estimate(cameras_center);
  
  // Correct Plane Normal.
  float sum_d = 0.0f;
  for (auto & C : cameras_center) {
    sum_d += camera_models[0].dot(C.homogeneous());
  }
  if (sum_d < 0) {
    camera_models[0] = -camera_models[0];
  }
  std::cout << "camera model: " << camera_models[0].transpose() << std::endl;

  Eigen::Vector3d cam_plane_normal = camera_models[0].head<3>();

  std::vector<Eigen::Vector6d> vecs;
  vecs.reserve(points.size());
  for (int i = 0; i < points.size(); ++i) {
    Eigen::Vector6d vec_t;
    vec_t << points[i][0], points[i][1], points[i][2], cam_plane_normal[0], cam_plane_normal[1], cam_plane_normal[2];
    vecs.emplace_back(vec_t);
  }
  // Estimate planar equation.
  RANSACOptions ransac_options;
  ransac_options.max_error = 1e-3;

  LORANSAC<WeightedPlaneEstimator, WeightedPlaneLocalEstimator> P_ransac(ransac_options);
  const auto report = P_ransac.Estimate(vecs);
  Eigen::Vector4d model = report.model;
  std::cout << "model: " << model.transpose() << std::endl;
  
  // Correct Plane Normal.
  // float sum_d = 0.0f;
  // for (auto & C : cameras_center) {
  //   sum_d += model.dot(C.homogeneous());
  // }
  double plane_diff = camera_models[0].head<3>().dot(model.head<3>());
  if (std::fabs(plane_diff) < std::cos(DEG2RAD(30))) {
    std::cout << StringPrintf("plane from point has %f angle difference large than 30", RadToDeg(std::acos(plane_diff))) << std::endl;
    return false;
  }
  if (plane_diff < 0) {
    model = -model;
  }

  double norm = model.head<3>().norm();
  plane = model / norm;
  return true;
}

bool FitConsistentPlane(const std::vector<Eigen::Vector3d> &points,
                        const Eigen::Vector4d &primary_plane,
                        const double dist_to_best_model,
                        const double angle_diff_to_main_plane,
                        const double min_inlier_ratio_to_best_model,
                        Eigen::Vector4d &plane) {
  if (points.size() < 3) {
    std::cout << "No enough points to fit a plane!" << std::endl;
    return false;
  }

  const double angle_consistent_thres = std::cos(angle_diff_to_main_plane / 180.0 * M_PI);
  const Eigen::Vector3d primary_normal = primary_plane.head<3>();

    // Estimate planar equation.
  RANSACOptions ransac_options;
  ransac_options.max_error = 1e-2;
  ransac_options.min_inlier_ratio_to_best_model = min_inlier_ratio_to_best_model;
    
  LORANSAC<PlaneEstimator, PlaneLocalEstimator> P_ransac(ransac_options);
    const auto report = P_ransac.EstimateMultiple(points);
    Eigen::Vector4d model = report.model;
  std::vector<Eigen::Vector4d> models = report.multiple_models;

  // Correct Plane Normal.
  Eigen::Vector3d m_vNormal = model.head<3>();
  if (m_vNormal.dot(primary_normal) < 0) {
    model = -model;
  }

  double norm = model.head<3>().norm();
  plane = model / norm;

  double angle_consistent = std::abs(plane.head<3>().dot(primary_normal));
  if (angle_consistent < angle_consistent_thres) {
    std::cout << StringPrintf("Angle diff(%f) is greater than %f, "
                              "Conflict with the primary plane.", 
      std::acos(angle_consistent) / M_PI * 180.0, angle_diff_to_main_plane) << std::endl;
    return false;
  }

  // Consistent Check.
  for (auto sub_model : models) {
    Eigen::Vector3d normal = sub_model.head<3>().normalized();
    double angle_consistent = std::fabs(normal.dot(plane.head<3>()));
    if (angle_consistent > angle_consistent_thres) {
      double dist;
      if (sub_model[3] * plane[3] <= 0) {
        dist = std::fabs(sub_model[3] + plane[3]);
      } else {
        dist = std::fabs(sub_model[3] - plane[3]);
      }
      if (dist < dist_to_best_model) {
        return true;
      }
    }
  }
  std::cout << "Plane from best model and good model not Consistent." << std::endl;
  return false;
}

int PanoramaImageCluster(std::vector<std::vector<int>> &cluster_image_map,
                 std::vector<int>& common_image_ids,
                 uint64_t& max_images_num,
                 const std::vector<mvs::Image> &images,
                 bool has_fused, float max_ram, float ram_eff_factor,
                 const float common_persent,
                 const std::vector<std::vector<int> >& overlapping_images_,
                 const bool with_normal){
    const int all_image_num = images.size();
    if (max_ram < 0 || has_fused){
      cluster_image_map.resize(1);
      for (int i = 0; i < all_image_num; i++){
        cluster_image_map[0].push_back(i);
      }
      return 1;
    }
    
    // compute max_num_images 
    float width = images.at(0).GetWidth();
    float height = images.at(0).GetHeight();
    for (int i = 0; i < images.size(); i++){
      float tmp_width = images.at(i).GetWidth();
      float tmp_height = images.at(i).GetHeight();
      if (width * height < tmp_width * tmp_height){
        width = tmp_width;
        height = tmp_height;
      }
    }
    // width * height * float size * (image + depth_map + normal_map 
    // + semantic_map + mask) * coefficient_of_fluctuation
    float image_memory;
    if (with_normal) {
      image_memory = width * height * num_perspective_per_image 
                         * (3 + 4 + 12 + 1 + 1);
    } else {
      image_memory = width * height * num_perspective_per_image 
                         * (3 + 4 + 1 + 1);;
    }
    uint64_t G_byte = 1.0e9;
    max_images_num = max_ram * G_byte * ram_eff_factor / image_memory * (1 - 4 * common_persent);
    int num_cluster = all_image_num / (max_images_num * num_perspective_per_image) + 1;
    // int num_cluster = 2;
    const int fixed_size = all_image_num /(num_perspective_per_image * num_cluster) + 1;
    num_cluster = std::ceil((float)all_image_num/(num_perspective_per_image*fixed_size));
    
    std::cout << "Num_max_images: " << int(fixed_size * num_perspective_per_image) 
              << "\tNum_all_images: " << int(all_image_num) 
              << "\tNum_cluster: " << num_cluster << std::endl;
    cluster_image_map.resize(num_cluster);

    if (num_cluster == 1){
      for (int i = 0; i < all_image_num; i++){
        cluster_image_map[0].push_back(i);
      }
      return 1;
    }
    
    KMeans kmeans;
    for (int image_idx = 0; image_idx < all_image_num; image_idx += num_perspective_per_image){
      auto& image = images.at(image_idx);
      Eigen::Vector3f location = Eigen::Map<const Eigen::Vector3f>(image.GetC());
      Tuple tuple;
      tuple.location = location.cast <double> ();
      tuple.id = image_idx;
      tuple.name = std::to_string(image_idx);
      kmeans.mv_pntcloud.emplace_back(tuple);
    }
    kmeans.SetK(num_cluster);
    kmeans.max_point_size = max_images_num;
    kmeans.fixed_size = fixed_size;
    // kmeans.SameSizeCluster();
    kmeans.SameSizeClusterWithConnection(overlapping_images_);
    int common_images_size = kmeans.fixed_size * common_persent + 1;
    std::vector<std::unordered_map<int, std::vector<Tuple>>> neighbors_points;
    std::vector<std::vector<int>> neighbors;
    // kmeans.FindNeighborsAndCommonPoints(neighbors_points, neighbors);
    kmeans.FindNeighborsAndCommonPointsWithConnection(neighbors_points, 
      neighbors, overlapping_images_);

    for (size_t i = 0; i < kmeans.m_k; i++) {
      for (size_t j = 0; j < kmeans.m_grp_pntcloud[i].size(); j++){
        for (int per_id = 0; per_id < num_perspective_per_image; ++per_id){
          cluster_image_map[i].push_back(kmeans.m_grp_pntcloud[i].at(j).id + per_id);
        }
      }
      int num_neighbors = neighbors[i].size();
      common_images_size = (num_neighbors <= 4 ? common_images_size : 
        common_images_size * 4 / num_neighbors + 1);
      for (size_t j = 0; j < neighbors[i].size(); j++) {
        for (size_t k = 0; k < neighbors_points[i][neighbors[i][j]].size() 
            && k < common_images_size; k++) {
          for (int per_id = 0; per_id < num_perspective_per_image; ++per_id){
            cluster_image_map[neighbors[i][j]].push_back(
              neighbors_points[i][neighbors[i][j]][k].id + per_id);
            common_image_ids.push_back(
              neighbors_points[i][neighbors[i][j]][k].id + per_id);
          }
        }
      }
    }
    std::set<int> st(common_image_ids.begin(), common_image_ids.end());
    common_image_ids.assign(st.begin(), st.end());
    max_images_num = max_images_num * num_perspective_per_image;
    
    return num_cluster;
};

void MutiThreadSaveCrossPointCloud(
    const std::string ply_path, const bool has_sem,
    const std::vector<PlyPoint>& points,
    const  std::vector<std::vector<uint32_t> >& points_visibility,
    const std::vector<std::vector<float> >& points_vis_weight = std::vector<std::vector<float>>()){

    int num_eff_threads = std::min(GetEffectiveNumThreads(-1), 5);
    std::unique_ptr<ThreadPool> wirte_thread_pool;
    wirte_thread_pool.reset(new ThreadPool(num_eff_threads));
    wirte_thread_pool->AddTask([&](){
        WriteBinaryPlyPoints(ply_path, points, true, true);
    });
    wirte_thread_pool->AddTask([&](){
        WritePointsVisibility(ply_path + ".vis", points_visibility);
    });
    if (has_sem) {
        wirte_thread_pool->AddTask([&](){
            WritePointsSemantic(ply_path + ".sem", points, false);
        });
        wirte_thread_pool->AddTask([&](){
            WritePointsSemanticColor(
                JoinPaths(GetParentDir(ply_path), FUSION_SEM_NAME), 
                points);
        });
    }
    if (points_vis_weight.size() == points.size()){
        wirte_thread_pool->AddTask([&](){
            WritePointsWeight(ply_path + ".wgt", points_vis_weight);
        });
    }
    wirte_thread_pool->Wait();

    std::cout << "MutiThreadSaveCrossPointCloud(" << points.size() 
                    << ", " << has_sem << ", " << ply_path  << " ) ... Done" << std::endl;
    return;
}

void MutiThreadAppendCrossPointCloud(
    const std::string ply_path, const bool has_sem,
    const std::vector<PlyPoint>& points,
    const  std::vector<std::vector<uint32_t> >& points_visibility,
    const std::vector<std::vector<float> >& points_vis_weight = std::vector<std::vector<float>>()){
    int num_eff_threads = std::min(GetEffectiveNumThreads(-1), 5);

    std::unique_ptr<ThreadPool> append_thread_pool;
    append_thread_pool.reset(new ThreadPool(num_eff_threads));
    append_thread_pool->AddTask([&](){
        AppendWriteBinaryPlyPoints(ply_path, points);
    });
    append_thread_pool->AddTask([&](){
        AppendWritePointsVisibility(ply_path + ".vis", points_visibility);
    });
    if (has_sem) {
        append_thread_pool->AddTask([&](){
            AppendWritePointsSemantic(ply_path + ".sem", points, false);
        });
        append_thread_pool->AddTask([&](){
            AppendWritePointsSemanticColor(
                JoinPaths(GetParentDir(ply_path), FUSION_SEM_NAME), 
                points);
        });
    }
    if (points_vis_weight.size() == points.size()){
        append_thread_pool->AddTask([&](){
            AppendWritePointsWeight(ply_path + ".wgt", points_vis_weight);
        });
    }
    append_thread_pool->Wait();

    std::cout << "MutiThreadAppendCrossPointCloud (" << points.size() 
                    << ", " << has_sem << ", " << ply_path  << " ) ... Done" << std::endl;
    return;
}
}  // namespace internal

void StereoFusion::Options::Print() const {
  PrintHeading2("StereoFusion::Options");
  PrintOption(max_image_size);
  PrintOption(min_num_pixels);
  PrintOption(max_num_pixels);
  PrintOption(min_num_visible_images);
  PrintOption(max_traversal_depth);
  PrintOption(max_reproj_error);
  PrintOption(max_depth_error);
  PrintOption(min_occluded_depth_error);
  PrintOption(max_normal_error);
  PrintOption(check_num_images);
  PrintOption(cache_size);
  PrintOption(max_ram);
  PrintOption(ram_eff_factor);
  PrintOption(fuse_common_persent);
  PrintOption(fit_ground);
  PrintOption(cache_depth);
  PrintOption(roi_fuse);
  PrintOption(roi_box_width);
  PrintOption(roi_box_factor);
  PrintOption(fused_delaunay_sample);
  PrintOption(dist_insert);
  PrintOption(diff_depth);
  PrintOption(map_update);
}

bool StereoFusion::Options::Check() const {
  CHECK_OPTION_GE(min_num_pixels, 0);
  CHECK_OPTION_LE(min_num_pixels, max_num_pixels);
  CHECK_OPTION_GE(min_num_visible_images, 0);
  CHECK_OPTION_GT(max_traversal_depth, 0);
  CHECK_OPTION_GE(max_reproj_error, 0);
  CHECK_OPTION_GE(max_depth_error, 0);
  CHECK_OPTION_GE(min_occluded_depth_error, 0);
  CHECK_OPTION_GE(max_normal_error, 0);
  CHECK_OPTION_GT(check_num_images, 0);
  CHECK_OPTION_GT(cache_size, 0);
  return true;
}

int  StereoFusion::ReadClusterBox() {
    num_box_ = -1;
    const auto reconstruction_path = 
        JoinPaths(workspace_path_, std::to_string(select_reconstruction_idx_));
    const auto box_path = 
        JoinPaths(reconstruction_path, DENSE_DIR, BOX_YAML);
    
    if (ExistsFile(box_path)) {
        YAML::Node box_node = YAML::LoadFile(box_path);
        num_box_ = box_node["num_clusters"].as<int>();

        int vec_id = 0;
        for (YAML::const_iterator it= box_node["transformation"].begin(); 
            it != box_node["transformation"].end();++it){
            box_rot_(vec_id / 3, vec_id %3) = it->as<float>();
            vec_id++;
        }
        if (vec_id != 9){
            std::cout << "Yaml transformation is bad!" << std::endl;
        }
        // std::cout << box_rot_ << std::endl;
        float roi_box_factor = options_.roi_box_factor;
        if (options_.roi_box_width < 0 && options_.roi_box_factor < 0){
            roi_box_factor = 0.01;
        }
        roi_box_.x_min = box_node["box"]["x_min"].as<float>();
        roi_box_.y_min = box_node["box"]["y_min"].as<float>();
        roi_box_.z_min = box_node["box"]["z_min"].as<float>();
        roi_box_.x_max = box_node["box"]["x_max"].as<float>();
        roi_box_.y_max = box_node["box"]["y_max"].as<float>();
        roi_box_.z_max = box_node["box"]["z_max"].as<float>();
        roi_box_.SetBoundary(options_.roi_box_width * 2, roi_box_factor * 2);
        roi_box_.z_box_min = -FLT_MAX;
        roi_box_.z_box_max = FLT_MAX;
        roi_box_.rot = box_rot_;
        std::cout << StringPrintf("ROI(box.yaml): [%f %f %f] -> [%f %f %f]\n", 
                    roi_box_.x_box_min, roi_box_.y_box_min, roi_box_.z_box_min,
                    roi_box_.x_box_max, roi_box_.y_box_max, roi_box_.z_box_max);

        for (int idx = 0; idx < num_box_; idx++){
            Box box;
            box.x_min = box_node[std::to_string(idx)]["x_min"].as<float>();
            box.y_min = box_node[std::to_string(idx)]["y_min"].as<float>();
            box.z_min = box_node[std::to_string(idx)]["z_min"].as<float>();
            // box.z_min = -FLT_MAX;
            box.x_max = box_node[std::to_string(idx)]["x_max"].as<float>();
            box.y_max = box_node[std::to_string(idx)]["y_max"].as<float>();
            box.z_max = box_node[std::to_string(idx)]["z_max"].as<float>();
            // box.z_max = FLT_MAX;
            box.SetBoundary(options_.roi_box_width * 2, roi_box_factor * 2);
            box.z_box_min = -FLT_MAX;
            box.z_box_max = FLT_MAX;
            box.rot = box_rot_;
            roi_child_boxs_.push_back(box);
            std::cout << StringPrintf("ROI-%d(box.txt): [%f %f %f] -> [%f %f %f]\n", 
                    idx, box.x_box_min, box.y_box_min, box.z_box_min,
                    box.x_box_max, box.y_box_max, box.z_box_max);
        }
    }

    std::cout << "Reading Box Yaml (" << num_box_ << " cluster)..." << std::endl;
    return num_box_;
}

StereoFusion::StereoFusion(const Options& options,
                           const std::string& workspace_path,
                           const std::string& input_type,
                           const int reconstrction_idx,
                           const int cluster_idx)
    : num_cluster_(0),
      options_(options),
      workspace_path_(workspace_path),
      input_type_(input_type),
      max_squared_reproj_error_(options_.max_reproj_error *
                                options_.max_reproj_error),
      min_cos_normal_error_(std::cos(DegToRad(options_.max_normal_error))),
      select_reconstruction_idx_(reconstrction_idx),
      select_cluster_idx_(cluster_idx) {
  CHECK(options_.Check());
}

const std::vector<PlyPoint>& StereoFusion::GetFusedPoints() const {
  return fused_points_;
}

const std::vector<std::vector<uint32_t> >& StereoFusion::GetFusedPointsVisibility()
    const {
  return fused_points_visibility_;
}

void StereoFusion::ReadWorkspace() {
    for (size_t cluster_idx = 0; ;cluster_idx++) {
        const auto& reconstruction_path = 
            JoinPaths(workspace_path_, std::to_string(select_reconstruction_idx_));
        const auto& cluster_reconstruction_path = 
            JoinPaths(reconstruction_path, std::to_string(cluster_idx));
        if (!ExistsDir(cluster_reconstruction_path) && cluster_idx != 0) {
            break;
        }

        num_cluster_++;
    }
    std::cout << "Reading workspace (" << num_cluster_ << " cluster)..." << std::endl;
}

void StereoFusion::AddBlackList(const std::vector<uint8_t>& black_list) {
  semantic_label_black_list_ = black_list;
}

void StereoFusion::SetROIBox(const Box& box) {
  roi_box_ = box;
}

int StereoFusion::FindNextImage(const int prev_image_idx) {

    for (const auto image_idx : overlapping_images_.at(prev_image_idx)) {
        if (used_images_.at(image_idx) && !fused_images_.at(image_idx) && !images_[image_idx].IsRig()) {
            return image_idx;
        }
    }

    for (const auto image_idx : overlapping_images_.at(prev_image_idx)) {
        if (used_images_.at(image_idx) && !fused_images_.at(image_idx)) {
            return image_idx;
        }
    }

    // If none of the overlapping images are not yet fused, simply return the
    // first image that has not yet been fused.
    for (size_t image_idx = 0; image_idx < fused_images_.size(); ++image_idx) {
        if (used_images_[image_idx] && !fused_images_[image_idx] &&  !images_[image_idx].IsRig()) {
            return image_idx;
        }
    }

    for (size_t image_idx = 0; image_idx < fused_images_.size(); ++image_idx) {
        if (used_images_[image_idx] && !fused_images_[image_idx]) {
            return image_idx;
        }
    }

    return -1;
}


void StereoFusion::Run() {
  options_.Print();
  std::cout << std::endl;

  size_t reconstruction_idx = select_reconstruction_idx_;
  PrintHeading1(StringPrintf("Fusing# %d", reconstruction_idx));
  // ReadWorkspace();
  
  if (IsStopped()) {
    GetTimer().PrintMinutes();
    return;
  }

  auto reconstruction_path = 
    JoinPaths(workspace_path_, std::to_string(reconstruction_idx));

  auto dense_reconstruction_path = JoinPaths(reconstruction_path, DENSE_DIR);
  // if (!ExistsDir(dense_reconstruction_path) || 
  //     ExistsFile(JoinPaths(dense_reconstruction_path, FUSION_NAME))) {
  //   return;
  // }
  active_dense_path_ = dense_reconstruction_path;

  auto undistort_image_path =
    JoinPaths(dense_reconstruction_path, IMAGES_DIR);
  if (!ExistsDir(undistort_image_path)) {
    return;
  }

  auto undistort_sparse_path = 
    JoinPaths(dense_reconstruction_path, SPARSE_DIR);
  if (!ExistsDir(undistort_sparse_path)) {
    return;
  }

  auto options = options_;
  Workspace::Options workspace_options;
  workspace_options.max_image_size = options.max_image_size;
  workspace_options.image_as_rgb = true;
  workspace_options.cache_size = options.cache_size;
  workspace_options.workspace_path = dense_reconstruction_path;
  workspace_options.workspace_format = options.format;
  workspace_options.image_path = undistort_image_path;
  workspace_options.input_type = input_type_;
  workspace_.reset(new Workspace(workspace_options));

  if (1){
    std::string cluster_roi_path = dense_reconstruction_path;

    auto fused_path = JoinPaths(dense_reconstruction_path, FUSION_NAME);
    bool has_fused = ExistsFile(fused_path);
    if (has_fused) {
      std::cout << fused_path << " has ply file and no process." << std::endl;
      // // continue;
      // boost::filesystem::remove(fused_path);
      // if (ExistsFile(fused_path + ".vis")){
      //   boost::filesystem::remove(fused_path + ".vis");
      // }
      // if (ExistsFile(fused_path + ".wgt")){
      //   boost::filesystem::remove(fused_path + ".wgt");
      // }
      // if (ExistsFile(fused_path + ".sem")){
      //   boost::filesystem::remove(fused_path + ".sem");
      // }
      has_fused = false;
    }
    // if (has_fused) {
    //   options.cache_depth = false;
    // }

    auto stereo_reconstruction_path = JoinPaths(dense_reconstruction_path, STEREO_DIR);
    active_cluster_stereo_path_ = stereo_reconstruction_path;

    auto depth_maps_path = JoinPaths(stereo_reconstruction_path, DEPTHS_DIR);
    if (!ExistsDir(depth_maps_path) && !has_fused) {
      return;
    }

    auto normal_maps_path = JoinPaths(stereo_reconstruction_path, NORMALS_DIR);
    if (options.with_normal && !ExistsDir(normal_maps_path) && !has_fused) {
      return;
    }

    auto conf_maps_path = JoinPaths(stereo_reconstruction_path, CONFS_DIR);
    auto semantic_maps_path = JoinPaths(dense_reconstruction_path, SEMANTICS_DIR);

    Clear();

    // workspace_->SetModel(cluster_rect_path);
    const Model& model = workspace_->GetModel();
    sfm_cluster_image_idx_to_id_.push_back(model.GetImageIdx2Ids());

    std::string update_path = JoinPaths(undistort_sparse_path, "update_images.txt");
    std::unordered_set<int> update_image_idxs;
    if (options_.map_update && ExistsFile(update_path)){
        model.GetUpdateImageidxs(update_path, update_image_idxs);
    } else {
        for (int i = 0; i < model.images.size(); i++){
            update_image_idxs.insert(i);
        }
    }

    const double kMinTriangulationAngle = 0;
    overlapping_images_ = model.GetMaxOverlappingImages(
        options.check_num_images, kMinTriangulationAngle, update_image_idxs);

    images_ = model.images;
    image_names_.resize(images_.size());
    for (int image_idx = 0; image_idx < images_.size(); ++image_idx) {
      const std::string image_name = model.GetImageName(image_idx);
      image_names_[image_idx] = image_name;
    }

    if (ReadClusterBox() < 0){
      if (!ExistsFile(JoinPaths(cluster_roi_path, ROI_BOX_NAME)) && 
        ExistsFile(JoinPaths(reconstruction_path, ROI_BOX_NAME))){
        cluster_roi_path = reconstruction_path;
      }
      auto ori_box_path = JoinPaths(cluster_roi_path, ROI_BOX_NAME);

      if (ExistsFile(ori_box_path)){
        ReadBoundBoxText(ori_box_path, roi_box_);
        roi_box_.SetBoundary(options.roi_box_width * 1.2, options.roi_box_factor * 1.2);
        roi_box_.z_box_min = -FLT_MAX;
        roi_box_.z_box_max = FLT_MAX;
      } else {
        std::vector<Eigen::Vector3f> points;
        for (const Model::Point & point : model.points) {
          Eigen::Vector3f p(&point.x);
          points.emplace_back(p);
        }
        for (const mvs::Image & image : model.images) {
          Eigen::Vector3f C(image.GetC());
          points.emplace_back(C);
        }
        roi_box_.x_min = roi_box_.y_min = roi_box_.z_min = FLT_MAX;
        roi_box_.x_max = roi_box_.y_max = roi_box_.z_max = -FLT_MAX;
        for (auto point : points) {
          roi_box_.x_min = std::min(roi_box_.x_min, point.x());
          roi_box_.y_min = std::min(roi_box_.y_min, point.y());
          roi_box_.z_min = std::min(roi_box_.z_min, point.z());
          roi_box_.x_max = std::max(roi_box_.x_max, point.x());
          roi_box_.y_max = std::max(roi_box_.y_max, point.y());
          roi_box_.z_max = std::max(roi_box_.z_max, point.z());
        }
        float x_offset = (roi_box_.x_max - roi_box_.x_min) * 0.05;
        float y_offset = (roi_box_.y_max - roi_box_.y_min) * 0.05;
        float z_offset = (roi_box_.z_max - roi_box_.z_min) * 0.05;
        roi_box_.x_box_min = roi_box_.x_min - x_offset;
        roi_box_.x_box_max = roi_box_.x_max + x_offset;
        roi_box_.y_box_min = roi_box_.y_min - y_offset;
        roi_box_.y_box_max = roi_box_.y_max + y_offset;
        roi_box_.z_box_min = roi_box_.z_min - z_offset;
        roi_box_.z_box_max = roi_box_.z_max + z_offset;
        roi_box_.rot = Eigen::Matrix3f::Identity();
        std::cout << StringPrintf("ROI: [%f %f %f] -> [%f %f %f]\n", 
          roi_box_.x_box_min, roi_box_.y_box_min, roi_box_.z_box_min,
          roi_box_.x_box_max, roi_box_.y_box_max, roi_box_.z_box_max);
      }
    }

    const size_t image_num = images_.size();

    options.min_num_pixels =
          std::min((int)image_num + 1, options.min_num_pixels);
    options.min_num_visible_images =
          std::min((int)image_num, options.min_num_visible_images);

    semantic_maps_.resize(images_.size());
    depth_maps_.resize(images_.size());
    normal_maps_.resize(images_.size());
    conf_maps_.resize(images_.size());
    used_images_.resize(image_num, 0);
    fused_images_.resize(image_num, 0);
    fused_pixel_masks_.resize(image_num);
    // valid_pixel_masks_.resize(image_num);
    P_.resize(image_num);
    inv_P_.resize(image_num);
    inv_R_.resize(image_num);

    int interval = 100;
    while(image_num < interval) {
      interval /= 10;
    }

    const int num_eff_threads = GetEffectiveNumThreads(-1);
    std::cout << "num_eff_threads: " << num_eff_threads << std::endl;
    thread_pool_.reset(new ThreadPool(num_eff_threads));

    std::vector<std::vector<int>> cluster_image_map;
    std::vector<int> common_image_ids;
    uint64_t max_images_num;

    Timer cluster_timer;
    cluster_timer.Start();
    if (options.format.compare("panorama") == 0){
      num_cluster_ = internal::PanoramaImageCluster(cluster_image_map, 
                                           common_image_ids, max_images_num,
                                           images_, has_fused, options_.max_ram,
                                           options_.ram_eff_factor,
                                           options_.fuse_common_persent,
                                           overlapping_images_, 
                                           options.with_normal);
    } else {
      const std::string cluster_yaml_path = JoinPaths(
          dense_reconstruction_path, FUSION_CLUATER_YAML);
        if (ExistsFile(cluster_yaml_path)){
            mvs::MVSCluster mvs_cluster;
            mvs_cluster.ReadImageClusterYaml(cluster_yaml_path, 
                                    cluster_image_map, common_image_ids, cluster_step_);
            num_cluster_ = cluster_image_map.size();
        } else {
            mvs::MVSCluster mvs_cluster;
            mvs_cluster.FusionImageCluster(options_, cluster_image_map, 
                                      common_image_ids, max_images_num, images_,
                                      overlapping_images_, update_image_idxs);
            num_cluster_ = cluster_image_map.size();
        }
    }

    int cluster_begin = select_cluster_idx_ < 0 ? 0 : select_cluster_idx_ * cluster_step_;
    int cluster_end = select_cluster_idx_ < 0 ? num_cluster_ : select_cluster_idx_ * cluster_step_ + cluster_step_;
    cluster_end = std::min(num_cluster_, cluster_end);
    if (cluster_begin >= num_cluster_){
        std::cout << "crash bug: cluster_begin(" << select_cluster_idx_ << " * " <<  cluster_step_ 
            << ") >= num_cluster_(" <<  num_cluster_ << ")" << std::endl;
            ExceptionHandler(INVALID_INPUT_PARAM, 
                JoinPaths(GetParentDir(workspace_path_), "errors/dense"), "DenseFusion").Dump();
            exit(INVALID_INPUT_PARAM);
        return;
    }
    
    int all_fusion_image_num = 0;
    for (int cluster_id = cluster_begin; cluster_id < cluster_end; cluster_id++){
      all_fusion_image_num += cluster_image_map.at(cluster_id).size();
    }

    std::cout << "num_cluster, cluster_begin, cluster_end, num_all_images: " 
        << num_cluster_ << ", " << cluster_begin << ", " << cluster_end << ", " << all_fusion_image_num << std::endl;
    std::cout << StringPrintf(" in %.3fs (image cluster)", cluster_timer.ElapsedSeconds())
              << std::endl;

    auto Init = [&](int image_idx, int num_image) {
      // for (int image_idx = start; image_idx < end; ++image_idx) {
        const std::string image_name = image_names_[image_idx];

        // progress++;
        if (image_idx % interval == 0) {
          std::cout << "\rconfigure image " << image_idx << "/"
                    << all_fusion_image_num << std::flush;
        }

        auto& image = images_.at(image_idx);
        const std::string file_name = StringPrintf("%s.%s.%s", 
            image_name.c_str(), input_type_.c_str(), DEPTH_EXT);
        
        auto image_path = JoinPaths(undistort_image_path, image_name);
        auto semantic_path = JoinPaths(semantic_maps_path, image_name);
        auto depth_map_path = JoinPaths(depth_maps_path, file_name);
        auto normal_map_path = JoinPaths(normal_maps_path, file_name);
        auto conf_map_path = JoinPaths(conf_maps_path, file_name);
        if (!ExistsFile(depth_map_path) || 
           (options.with_normal && !ExistsFile(normal_map_path))) {
            std::cout << StringPrintf(
                    "WARNING: Ignoring image %s, because input does not "
                    "exist.", image_name.c_str())
                      << std::endl;
            return;
        }

        if (options.cache_depth) {
          std::unique_ptr<DepthMap> depth_map_ptr;
          depth_map_ptr = std::unique_ptr<DepthMap>(new DepthMap);
          depth_map_ptr->Read(depth_map_path);
          std::unique_ptr<NormalMap> normal_map_ptr;
          if (options.with_normal) {
            normal_map_ptr = std::unique_ptr<NormalMap>(new NormalMap);
            normal_map_ptr->Read(normal_map_path);
          }

          if (!depth_map_ptr->IsValid() || 
             (options.with_normal && !normal_map_ptr->IsValid())) {
            std::cout << StringPrintf(
                      "WARNING: Ignoring image %s, because input is empty."
                      , image_name.c_str())
                        << std::endl;
            return;
          }

          depth_maps_.at(image_idx).swap(depth_map_ptr);
          normal_maps_.at(image_idx).swap(normal_map_ptr);

          if (ExistsFile(conf_map_path)) {
            auto conf_map_ptr = std::unique_ptr<MatXf>(new MatXf);
            conf_map_ptr->Read(conf_map_path);
            conf_maps_.at(image_idx).swap(conf_map_ptr);
          }

          Bitmap bitmap;
          bitmap.Read(image_path, true);
          image.SetBitmap(bitmap);

          const auto semantic_base_name = 
            semantic_path.substr(0, semantic_path.size() - 3);
          if (ExistsFile(semantic_base_name + "png")) {
            semantic_maps_.at(image_idx).reset(new Bitmap);
            semantic_maps_.at(image_idx)->Read(semantic_base_name + "png", false);
          } else if (ExistsFile(semantic_base_name + "jpg")) {
            semantic_maps_.at(image_idx).reset(new Bitmap);
            semantic_maps_.at(image_idx)->Read(semantic_base_name + "jpg", false);
          }

          fused_pixel_masks_.at(image_idx).reset(new Mat<bool>(image.GetWidth(), image.GetHeight(), 1));
          fused_pixel_masks_.at(image_idx)->Fill(false);
        }

        used_images_.at(image_idx) = true;

        Eigen::RowMatrix3f K = 
          Eigen::Map<const Eigen::RowMatrix3f>(image.GetK());
        ComposeProjectionMatrix(K.data(), image.GetR(), image.GetT(),
                                P_.at(image_idx).data());
        ComposeInverseProjectionMatrix(K.data(), image.GetR(), image.GetT(),
                                      inv_P_.at(image_idx).data());
        inv_R_.at(image_idx) =
            Eigen::Map<const Eigen::RowMatrix3f>(image.GetR()).transpose();
      // }
    };

    if (has_fused) {
      fused_points_ = ReadPly(fused_path);
      ReadPointsVisibility(fused_path + ".vis", fused_points_visibility_);
      const std::string fused_sem_path = fused_path + ".sem";
      if (ExistsFile(fused_sem_path)) {
        ReadPointsSemantic(fused_sem_path, fused_points_, false);
      } else if (ExistsDir(semantic_maps_path)) {
        size_t i = 0, j = 0;
        for (i = 0; i < fused_points_.size(); ++i) {
          if (i % 1000 == 0) {
            std::cout << StringPrintf("\rProcess Fused Points#%d / %d", i, fused_points_.size());
          }
          Eigen::Vector3f point((float*)&fused_points_[i]);
          auto fused_point_vis = fused_points_visibility_.at(i);
          
          int max_samples = 0;
          int best_label = -1;
          size_t num_fused_point = 0;

          int samps_per_label[256];
          memset(samps_per_label, 0, sizeof(int) * 256);

          for (auto vis : fused_point_vis) {
            const mvs::Image& image = images_.at(vis);
            const Eigen::RowMatrix3f K(image.GetK());
            const Eigen::RowMatrix3f R(image.GetR());
            const Eigen::Vector3f T(image.GetT());
            Eigen::Vector3f proj = K * (R * point + T);
            int u = std::round(proj[0] / proj[2]);
            int v = std::round(proj[1] / proj[2]);
            if (u < 0 || u >= image.GetWidth() ||
                v < 0 || v >= image.GetHeight()) {
              continue;
            }

            if (!semantic_maps_.at(vis)) {
              const auto image_name = image_names_[vis];
              const auto semantic_name = JoinPaths(semantic_maps_path, image_name);
              const auto semantic_base_name = semantic_name.substr(0, semantic_name.size() - 3);
              if (ExistsFile(semantic_base_name + "png")) {
                semantic_maps_.at(vis).reset(new Bitmap);
                semantic_maps_.at(vis)->Read(semantic_base_name + "png", false);
              } else if (ExistsFile(semantic_base_name + "jpg")) {
                semantic_maps_.at(vis).reset(new Bitmap);
                semantic_maps_.at(vis)->Read(semantic_base_name + "jpg", false);
              }
            }

            BitmapColor<uint8_t> semantic;
            semantic_maps_.at(vis)->GetPixel(u, v, &semantic);
            uint8_t sid = (semantic.r + 256) % 256;
            samps_per_label[sid]++;
            if (samps_per_label[sid] > max_samples) {
              max_samples = samps_per_label[sid];
              best_label = semantic.r;
            }
            num_fused_point++;
          }
          bool remove = false;
          for (auto id : semantic_label_black_list_) {
            float ratio = samps_per_label[id] / (float)num_fused_point;
            if (ratio > options.num_consistent_semantic_ratio) {
              remove = true;
              break;
            }
          }
          if (!remove) {
            fused_points_[i].s_id = best_label;
            fused_points_[j] = fused_points_[i];
            fused_points_visibility_[j] = fused_points_visibility_[i];
            j++;
          }
        }
        fused_points_.resize(j);
        fused_points_visibility_.resize(j);
      }
    }

    size_t num_fused_images = 0;
    size_t num_pre_fused_images = 0;
    int num_init_images = 1;
    size_t num_pre_fused_points = 0;
    size_t num_fused_points = 0;
    int64_t num_fused_vis_num = 0;
    std::vector<short int> points_cluster_id;


    for (int cluster_id = cluster_begin; cluster_id < cluster_end; cluster_id++){
      for (auto image_idx : cluster_image_map.at(cluster_id)) {
        thread_pool_->AddTask(Init, image_idx, num_init_images);
        num_init_images++;
      }
      thread_pool_->Wait();

      if (!has_fused) {
          int start_image_id = 0;
          for (int image_idx = 0; image_idx < images_.size(); ++ image_idx){
              if(!images_[image_idx].IsRig()){
                  start_image_id = image_idx;
                  break;
              }
          }

          for (int image_idx = start_image_id; image_idx >= 0; image_idx = FindNextImage(image_idx)) {
            if (IsStopped()) {
              break;
            }

            if (!used_images_.at(image_idx)) {
              continue;
            }

            Timer timer;
            timer.Start();

            std::cout << StringPrintf("Fusing image [%d/%d]", num_fused_images + 1,
                                      all_fusion_image_num)
                      << std::flush;

            auto &fused_pixel_mask = fused_pixel_masks_.at(image_idx);
            if (!fused_pixel_mask) {
              auto image = images_.at(image_idx);
              fused_pixel_mask.reset(new Mat<bool>(image.GetWidth(), image.GetHeight(), 1));
              fused_pixel_mask->Fill(false);
            }

            const auto semantic_path = JoinPaths(active_dense_path_, SEMANTICS_DIR);
            bool has_semantic = ExistsDir(semantic_path);
            std::unique_ptr<Bitmap>& semantic_map = semantic_maps_.at(image_idx);
            if (has_semantic && !semantic_map) {
              // Load semantic.
              const auto image_name = image_names_[image_idx];
              const auto semantic_name = JoinPaths(semantic_path, image_name);
              const auto semantic_base_name = semantic_name.substr(0, semantic_name.size() - 3);
              if (ExistsFile(semantic_base_name + "png")) {
                semantic_maps_.at(image_idx).reset(new Bitmap);
                semantic_maps_.at(image_idx)->Read(semantic_base_name + "png", false);
              } else if (ExistsFile(semantic_base_name + "jpg")) {
                semantic_maps_.at(image_idx).reset(new Bitmap);
                semantic_maps_.at(image_idx)->Read(semantic_base_name + "jpg", false);
              }
            }

            for(int thread_idx = 0; thread_idx < num_eff_threads; ++thread_idx) {
              thread_pool_->AddTask(&StereoFusion::FuseImage, this, options,
                                    thread_idx, image_idx, num_eff_threads, false);
            };
            thread_pool_->Wait();

            num_fused_images += 1;
            fused_images_.at(image_idx) = 255;

            depth_maps_.at(image_idx).reset(nullptr);
            normal_maps_.at(image_idx).reset(nullptr);
            conf_maps_.at(image_idx).reset(nullptr);
            Bitmap &bitmap = images_.at(image_idx).GetBitmap();
            bitmap.Deallocate();
            semantic_maps_.at(image_idx).reset(nullptr);
            fused_pixel_masks_.at(image_idx).reset(nullptr);

            std::cout << StringPrintf(" in %.3fs (%d points)", 
                                      timer.ElapsedSeconds(), fused_points_.size())
                      << std::endl;
        }

        fused_points_.shrink_to_fit();
        fused_points_visibility_.shrink_to_fit();
      }

      if (fused_points_.empty()) {
          std::cout << "WARNING: Could not fuse any points. This is likely "
                      "caused by incorrect settings - filtering must be enabled "
                      "for the last call to patch match stereo."
                    << std::endl;
      }

      std::cout << "Cluster " << cluster_id << " number of fused points: " 
                << fused_points_.size() << std::endl;

      if (options.fit_ground && ExistsDir(semantic_maps_path)) {
        if (options.format.compare("panorama") == 0) {
          PrintHeading1("Fitting Planes(panorama)");
          ComplementPlanesPanorama(options, cluster_image_map.at(cluster_id), 
                           num_pre_fused_points, num_pre_fused_images,
                           all_fusion_image_num);
        } else {
          PrintHeading1("Fitting Planes");
          ComplementPlanes(options, cluster_image_map.at(cluster_id), 
                           num_pre_fused_points, num_pre_fused_images,
                           all_fusion_image_num);
        }
      }

      semantic_maps_.clear();
      semantic_maps_.resize(images_.size());
      depth_maps_.clear();
      depth_maps_.resize(images_.size());
      conf_maps_.clear();
      conf_maps_.resize(images_.size());
      normal_maps_.clear();
      normal_maps_.resize(images_.size());
      used_images_.clear();
      used_images_.resize(image_num, 0);
      fused_images_.clear();
      fused_images_.resize(image_num, 0);
      fused_pixel_masks_.clear();
      fused_pixel_masks_.resize(image_num);
      P_.clear();
      P_.resize(image_num);
      inv_P_.clear();
      inv_P_.resize(image_num);
      inv_R_.clear();
      inv_R_.resize(image_num);

      if (options_.remove_duplicate_pnts){
        std::cout << "Cluster " << cluster_id << " Remove Duplicate points" << std::endl;
        RemoveDuplicatePoints(options, cluster_image_map.at(cluster_id), num_pre_fused_points);
      }

      if (options.outlier_removal) {
        RemoveOutliers(options);
      }

#ifndef FUSED_PC_MEMORY
      if (!has_fused) {
        std::vector<short int> temp_cluster_id(int(fused_points_.size() - num_pre_fused_points), cluster_id);
        points_cluster_id.insert(points_cluster_id.end(), temp_cluster_id.begin(), temp_cluster_id.end());
        num_pre_fused_points = fused_points_.size();
        num_fused_points += fused_points_.size();
        for (int64_t pnt_id = 0; pnt_id < fused_points_.size(); pnt_id++){
          num_fused_vis_num += fused_points_visibility_.at(pnt_id).size();
        }
      }
#else
      if (options_.fused_delaunay_sample){
        const Model& model_fusion = workspace_->GetModel();
        PointsSample(fused_points_, fused_points_score_, 
                       fused_points_visibility_, 
                       fused_points_vis_weight_, 
                       model_fusion, 
                       options_.dist_insert, 
                       options_.diff_depth);
        std::cout << "DelaunaySample visibility/weight size: " << fused_points_visibility_.size() 
          << " / " << fused_points_vis_weight_.size() << std::endl;
      }

      num_fused_points += fused_points_.size();
      for (int64_t pnt_id = 0; pnt_id < fused_points_.size(); pnt_id++){
        num_fused_vis_num += fused_points_visibility_.at(pnt_id).size();
      }

      if (!has_fused) {
#ifdef OLD_PIPELINE
        if (num_cluster_ > 1) {
          // std::vector<short int> temp_cluster_id(int(fused_points_.size() - num_pre_fused_points), cluster_id);
          // points_cluster_id.insert(points_cluster_id.end(), temp_cluster_id.begin(), temp_cluster_id.end());
          auto cluster_fused_path = JoinPaths(dense_reconstruction_path, std::to_string(cluster_id) + "-" + FUSION_NAME);
          std::cout << "Writing fusion output: " << cluster_fused_path << std::endl;
          WriteBinaryPlyPoints(cluster_fused_path, fused_points_, true, true);

          WritePointsVisibility(cluster_fused_path + ".vis", fused_points_visibility_);
          WritePointsWeight(cluster_fused_path + ".wgt", fused_points_vis_weight_);
          if (ExistsDir(semantic_maps_path)) {
            WritePointsSemantic(cluster_fused_path + ".sem", fused_points_, false);
            // WritePointsSemanticColor(JoinPaths(dense_reconstruction_path, FUSION_SEM_NAME), fused_points_);
          }
        }
#else
        std::cout << "Write Points ..." << std::endl;
        const std::string ply_path = JoinPaths(dense_reconstruction_path, FUSION_NAME);
        bool has_sem = ExistsDir(semantic_maps_path);
        if (cluster_id == 0 && !options_.map_update ){
            internal::MutiThreadSaveCrossPointCloud(ply_path, has_sem, fused_points_, 
                fused_points_visibility_,  fused_points_vis_weight_);
        } else {
            internal::MutiThreadAppendCrossPointCloud(ply_path, has_sem, fused_points_, 
                fused_points_visibility_,  fused_points_vis_weight_);
        }
  
        if (num_box_ > 1) {
          std::size_t num_points = fused_points_.size();
          std::vector<std::set<int>> pnts_2_cluster(num_points);
          for (std::size_t i = 0; i < num_points; i++){
              const auto& pnt = fused_points_.at(i);
              const Eigen::Vector3f xyz(pnt.x, pnt.y, pnt.z);
              const Eigen::Vector3f rot_xyz = box_rot_ * xyz;
              for (int j = 0; j < num_box_; j++){
                  if (rot_xyz(0) < roi_child_boxs_.at(j).x_box_min || 
                      rot_xyz(0) > roi_child_boxs_.at(j).x_box_max ||
                      rot_xyz(1) < roi_child_boxs_.at(j).y_box_min || 
                      rot_xyz(1) > roi_child_boxs_.at(j).y_box_max ||
                      rot_xyz(2) < roi_child_boxs_.at(j).z_box_min || 
                      rot_xyz(2) > roi_child_boxs_.at(j).z_box_max){
                      continue;
                  }
                  pnts_2_cluster.at(i).insert(j);
              }
          }

          std::vector<std::size_t> num_points_per_cluster(num_box_, 0);
          std::vector<std::vector<std::size_t>> cluster_2_pnts(num_box_);
          for (std::size_t i = 0; i < num_points; i++){
              for (const auto& box_id : pnts_2_cluster.at(i)){
                  cluster_2_pnts.at(box_id).push_back(i);
                  num_points_per_cluster.at(box_id)++;
              }
          }

          for (int i = 0; i < num_box_; i++){
            std::cout << "num_points_per_cluster.at(i): " << num_points_per_cluster.at(i) << std::endl;
          }
          bool has_weight = (fused_points_.size() == fused_points_vis_weight_.size());
          std::cout << "has_weight: " << has_weight << std::endl;

          std::vector<std::vector<PlyPoint>> cluster_fused_points(num_box_);
          std::vector<std::vector<std::vector<uint32_t> >> cluster_points_visibility(num_box_);
          std::vector<std::vector<std::vector<float> >> cluster_points_vis_weight(num_box_);
          for (int i = 0; i < num_box_; i++){
            const std::string cluster_path = JoinPaths(
                    dense_reconstruction_path, "..", std::to_string(i), FUSION_NAME);
            const std::string cluster_box_path = JoinPaths(
                    dense_reconstruction_path, "..", std::to_string(i), ROI_BOX_NAME);
            CreateDirIfNotExists(GetParentDir(cluster_path));
            if (!ExistsFile(cluster_box_path)){
                WriteBoundBoxText(cluster_box_path, roi_child_boxs_.at(i));
            }
            // if (num_points_per_cluster.at(i) < 1){
            //     continue;
            //   }
              cluster_fused_points.at(i).reserve(num_points_per_cluster.at(i));
              cluster_points_visibility.at(i).reserve(num_points_per_cluster.at(i));
              if (has_weight){
                  cluster_points_vis_weight.at(i).reserve(num_points_per_cluster.at(i));
              }

              for (std::size_t j = 0; j < num_points_per_cluster.at(i); j++){
                  std::size_t pnt_id = cluster_2_pnts.at(i).at(j);
                  cluster_fused_points.at(i).push_back(fused_points_.at(pnt_id));
                  cluster_points_visibility.at(i).push_back(fused_points_visibility_.at(pnt_id));
                  if (has_weight){
                      cluster_points_vis_weight.at(i).push_back(fused_points_vis_weight_.at(pnt_id));
                  }
              }

              if (cluster_id == 0){
                  internal::MutiThreadSaveCrossPointCloud(cluster_path, has_sem, cluster_fused_points.at(i), 
                      cluster_points_visibility.at(i),  cluster_points_vis_weight.at(i));
              } else {
                  internal::MutiThreadAppendCrossPointCloud(cluster_path, has_sem, cluster_fused_points.at(i), 
                      cluster_points_visibility.at(i),  cluster_points_vis_weight.at(i));
              }
          }
        }
#endif

        fused_points_.clear();
        fused_points_visibility_.clear();
        fused_points_vis_weight_.clear();
        fused_points_score_.clear();
        fused_points_.shrink_to_fit();
        fused_points_visibility_.shrink_to_fit();
        fused_points_vis_weight_.shrink_to_fit();
        fused_points_score_.shrink_to_fit();

        num_pre_fused_points = fused_points_.size();

      }

#endif
      num_pre_fused_images += cluster_image_map.at(cluster_id).size();

        malloc_trim(0);
    }

    if (cluster_end != num_cluster_){
      return;
    }

#ifdef OLD_PIPELINE
    Timer merge_timer;
    merge_timer.Start();
    uint64_t G_byte = 1.0e9;
    const int num_eff_threads_merge = GetEffectiveNumThreads(-1);
    const float estimated_memory = options_.max_ram * options_.ram_eff_factor * G_byte;
    const float points_memory = num_fused_points * (30.0f + 8.0f);
    const float visiblity_memory = (sizeof(std::vector<uint32_t>) * num_fused_points + 
      num_fused_vis_num * sizeof(uint32_t));
    float eff_ram_factor = std::max(1.0 - 
      (points_memory + visiblity_memory * 1.5f) / estimated_memory, 0.01);
    {
      float now_memory;
      GetAvailableMemory(now_memory);
      std::cout << "Points size: " << num_fused_points << "  Visbility num: " 
                << num_fused_vis_num
                << "\tpoint, visibilty, all memory: " << points_memory << ", " 
                << visiblity_memory * 1.5f << ", " 
                << points_memory + visiblity_memory * 1.5f
                << "  Available, EstimatedTotal Memory: " << now_memory * G_byte << ", " 
                << estimated_memory << std::endl;
    }

    int merge_max_mun_images = 
      std::min((int)(max_images_num * eff_ram_factor * 4), (int)common_image_ids.size());
    int num_common = 1;
    std::unique_ptr<ThreadPool> merge_thread_pool_;
#ifdef FUSED_PC_MEMORY
    if (!has_fused  && num_cluster_ > 1){
      float free_memory_factor = 
        eff_ram_factor - (float)merge_max_mun_images / (max_images_num * 4);
      int num_free_threads = 
        free_memory_factor / ((num_fused_points * 4.0f * 1.5f)/ estimated_memory);

      int merge_eff_thread = 
        std::min (std::max(num_free_threads, 1), num_eff_threads_merge);
      merge_max_mun_images = std::ceil((float)merge_max_mun_images / merge_eff_thread);
      num_common = 
        std::ceil((float)common_image_ids.size() / (float)merge_max_mun_images);

      merge_thread_pool_.reset(new ThreadPool(merge_eff_thread));

      std::cout << "Merge " << num_common << " Duplicate points by " 
                << merge_eff_thread << " threads. " 
                << "common_image_num, merge_max_mun_images, eff_ram_factor, free_memory_factor, num_free_threads: " 
                << common_image_ids.size() << ", " << merge_max_mun_images 
                << ", " << eff_ram_factor << ", " << free_memory_factor 
                << ", " << num_free_threads << std::endl;

      cluster_points_.resize(num_cluster_);
      cluster_points_visibility_.resize(num_cluster_);

      auto Read = [&](int cluster_id){
        auto cluster_fused_path = JoinPaths(dense_reconstruction_path, 
          std::to_string(cluster_id) + "-" + FUSION_NAME);
        
        cluster_points_.at(cluster_id) = ReadPly(cluster_fused_path);
        ReadPointsVisibility(cluster_fused_path + ".vis", 
          cluster_points_visibility_.at(cluster_id));
        const std::string fused_sem_path = cluster_fused_path + ".sem";
        if (ExistsFile(fused_sem_path)) {
          ReadPointsSemantic(fused_sem_path, cluster_points_.at(cluster_id), false);
        }

        CHECK_EQ(cluster_points_.at(cluster_id).size(), 
          cluster_points_visibility_.at(cluster_id).size());
      };

      const int num_read_thread = std::min(num_eff_threads, num_cluster_);
      thread_pool_.reset(new ThreadPool(num_read_thread));
      for (int cluster_id = 0; cluster_id < num_cluster_; cluster_id++){
        thread_pool_->AddTask(Read, cluster_id);
      }
      thread_pool_->Wait();

      {
        float now_memory;
        GetAvailableMemory(now_memory);
        std::cout << "1 read after  GetAvailableMemory: " << now_memory << std::endl;
      }

      for (int cluster_id = 0; cluster_id < num_cluster_; cluster_id++){
        float memory1, memory5, memory7;
        GetAvailableMemory(memory1);
        fused_points_.insert(fused_points_.end(), 
          cluster_points_.at(cluster_id).begin(),
          cluster_points_.at(cluster_id).end());
        fused_points_visibility_.insert(fused_points_visibility_.end(),
          cluster_points_visibility_.at(cluster_id).begin(), 
          cluster_points_visibility_.at(cluster_id).end());
        std::vector<short int> temp_cluster_id(
          cluster_points_.at(cluster_id).size(), cluster_id);
        points_cluster_id.insert(points_cluster_id.end(), 
          temp_cluster_id.begin(), temp_cluster_id.end());
        
        GetAvailableMemory(memory5);

        cluster_points_.at(cluster_id).clear();
        cluster_points_visibility_.at(cluster_id).clear();
        cluster_points_.at(cluster_id).shrink_to_fit();
        cluster_points_visibility_.at(cluster_id).shrink_to_fit();
        GetAvailableMemory(memory7);

        std::cout << "\tAdd " << cluster_id << " cluster_points (" 
                  << temp_cluster_id.size() << ") "
                  << "to all fused_points (" << fused_points_.size() << ")\t" 
                  << "Start memory: " << memory1
                  << "  add memory" << memory1 - memory5 
                  << "  Free memory" << ", " << memory7 - memory5 
                  << "  End memory: " << memory7 << std::endl;

        auto cluster_fused_path = JoinPaths(dense_reconstruction_path, 
          std::to_string(cluster_id) + "-" + FUSION_NAME);
        boost::filesystem::remove(cluster_fused_path);
        boost::filesystem::remove(cluster_fused_path + ".vis");
        if (ExistsFile(cluster_fused_path + ".sem")) {
          boost::filesystem::remove(cluster_fused_path + ".sem");
        }
      }
      fused_points_.shrink_to_fit();
      fused_points_visibility_.shrink_to_fit();
      
      int64_t vis_size = 0;
      for (int64_t pnt_id = 0; pnt_id < fused_points_.size(); pnt_id++){
        vis_size += fused_points_visibility_.at(pnt_id).size();
      }

      {
        float now_memory;
        GetAvailableMemory(now_memory);
        std::cout << "2 Add " << fused_points_.size() << " Points and " 
          << vis_size << " PointsVis,\t AvailableMemory: " << now_memory << std::endl;
      }
    }
#endif
    if (fused_points_.empty()) {
        std::cout << "WARNING: Could not fuse any points. This is likely "
                     "caused by incorrect settings - filtering must be enabled "
                     "for the last call to patch match stereo."
                  << std::endl;
    }

    if (num_cluster_ > 1 && !has_fused && options_.remove_duplicate_pnts){
      {
        std::vector<int64_t> temp_v(fused_points_.size(), 0);
        merge_points_.swap(temp_v);
      }
      std::cout << "AdjustDuplicatePoints..." << std::endl;

      for (int num = 0; num < num_common; num++){
        std::vector<int> temp_ids;
        if ((num + 1) * merge_max_mun_images >= std::distance(
            common_image_ids.begin(), common_image_ids.end())){
          temp_ids.assign(common_image_ids.begin() + num * merge_max_mun_images, 
                          common_image_ids.end());
        } else {
          temp_ids.assign(common_image_ids.begin() + num * merge_max_mun_images, 
                          common_image_ids.begin() + (num + 1) * merge_max_mun_images);
        }
        merge_thread_pool_->AddTask(&StereoFusion::AdjustDuplicatePoints, 
                              this, options, temp_ids, points_cluster_id, 0);
      }
      merge_thread_pool_->Wait();
      
      {
        std::cout << "Read PointsWeight... " << std::endl;
        bool has_weight = true;
        for (int cluster_id = 0; cluster_id < num_cluster_; cluster_id++){
          std::cout << "\t fused_points_vis_weight_size: " << fused_points_vis_weight_.size() << std::endl;
          const std::string fused_wgt_path = JoinPaths(dense_reconstruction_path, 
            std::to_string(cluster_id) + "-" + FUSION_NAME) + ".wgt";
          if (!ExistsFile(fused_wgt_path)) {
            has_weight = false;
            break;
          }
          std::vector<std::vector<float> > temp_vis_weight_;
          ReadPointsWeight(fused_wgt_path, temp_vis_weight_);
          fused_points_vis_weight_.insert(fused_points_vis_weight_.end(),
            temp_vis_weight_.begin(), temp_vis_weight_.end());

          boost::filesystem::remove(fused_wgt_path);
        }
      }

      std::cout << "RemoveCommonDuplicatePoints..." << std::endl;
      int duplicate_point = RemoveCommonDuplicatePoints();
      std::cout << StringPrintf("Merge %d duplicate points in %.3fs\n", 
                            duplicate_point, merge_timer.ElapsedSeconds());
      
    }

    GetTimer().PrintMinutes();

    std::cout << "Writing fusion output: " << fused_path << std::endl;
    WriteBinaryPlyPoints(fused_path, fused_points_, true, true);
    if (ExistsDir(semantic_maps_path)) {
      WritePointsSemantic(fused_path + ".sem", fused_points_, false);
      WritePointsSemanticColor(JoinPaths(dense_reconstruction_path, FUSION_SEM_NAME), fused_points_);
    }
    WritePointsVisibility(fused_path + ".vis", fused_points_visibility_);
    std::cout << "fused_points_vis_weight_: " << fused_points_vis_weight_.size() 
      << "\tfused_points_visibility_: " << fused_points_visibility_.size() << std::endl; 
    if (fused_points_vis_weight_.size() == fused_points_visibility_.size()){
      WritePointsWeight(fused_path + ".wgt", fused_points_vis_weight_);
    }

    if (num_box_ > 1){
      std::size_t num_points = fused_points_.size();
      std::vector<std::set<int>> pnts_2_cluster(num_points);
      for (std::size_t i = 0; i < num_points; i++){
          const auto& pnt = fused_points_.at(i);
          const Eigen::Vector3f xyz(pnt.x, pnt.y, pnt.z);
          const Eigen::Vector3f rot_xyz = box_rot_ * xyz;
          for (int j = 0; j < num_box_; j++){
              if (rot_xyz(0) < roi_child_boxs_.at(j).x_box_min || 
                  rot_xyz(0) > roi_child_boxs_.at(j).x_box_max ||
                  rot_xyz(1) < roi_child_boxs_.at(j).y_box_min || 
                  rot_xyz(1) > roi_child_boxs_.at(j).y_box_max ||
                  rot_xyz(2) < roi_child_boxs_.at(j).z_box_min || 
                  rot_xyz(2) > roi_child_boxs_.at(j).z_box_max){
                  continue;
              }
              pnts_2_cluster.at(i).insert(j);
              // num_points_per_cluster.at(j)++;
          }
      }

      std::vector<std::size_t> num_points_per_cluster(num_box_, 0);
      std::vector<std::vector<std::size_t>> cluster_2_pnts(num_box_);
      for (std::size_t i = 0; i < num_points; i++){
          for (const auto& box_id : pnts_2_cluster.at(i)){
              cluster_2_pnts.at(box_id).push_back(i);
              num_points_per_cluster.at(box_id)++;
          }
      }

        for (int i = 0; i < num_box_; i++){
          std::cout << "num_points_per_cluster.at(i): " << num_points_per_cluster.at(i) << std::endl;
        }
      bool has_weight = (fused_points_.size() == fused_points_vis_weight_.size());
      std::cout << "has_weight: " << has_weight << std::endl;

      std::vector<std::vector<PlyPoint>> cluster_fused_points(num_box_);
      std::vector<std::vector<std::vector<uint32_t> >> cluster_points_visibility(num_box_);
      std::vector<std::vector<std::vector<float> >> cluster_points_vis_weight(num_box_);
      for (int i = 0; i < num_box_; i++){
          cluster_fused_points.at(i).reserve(num_points_per_cluster.at(i));
          cluster_points_visibility.at(i).reserve(num_points_per_cluster.at(i));
          if (has_weight){
              cluster_points_vis_weight.at(i).reserve(num_points_per_cluster.at(i));
          }

          for (std::size_t j = 0; j < num_points_per_cluster.at(i); j++){
              std::size_t pnt_id = cluster_2_pnts.at(i).at(j);
              cluster_fused_points.at(i).push_back(fused_points_.at(pnt_id));
              cluster_points_visibility.at(i).push_back(fused_points_visibility_.at(pnt_id));
              if (has_weight){
                  cluster_points_vis_weight.at(i).push_back(fused_points_vis_weight_.at(pnt_id));
              }
          }

          const std::string cluster_path = JoinPaths(
                  dense_reconstruction_path, "..", std::to_string(i), FUSION_NAME);
          CreateDirIfNotExists(GetParentDir(cluster_path));
           WriteBinaryPlyPoints(cluster_path, fused_points_, true, true);
          if (ExistsDir(semantic_maps_path)) {
            WritePointsSemantic(cluster_path + ".sem", fused_points_, false);
            WritePointsSemanticColor(JoinPaths(GetParentDir(cluster_path) , FUSION_SEM_NAME), fused_points_);
          }
          WritePointsVisibility(cluster_path + ".vis", fused_points_visibility_);
          std::cout << "fused_points_vis_weight_: " << fused_points_vis_weight_.size() 
            << "\tfused_points_visibility_: " << fused_points_visibility_.size() << std::endl; 
          if (fused_points_vis_weight_.size() == fused_points_visibility_.size()){
            WritePointsWeight(cluster_path + ".wgt", fused_points_vis_weight_);
          }
      }
    }

    fused_points_.clear();
    fused_points_visibility_.clear();
    fused_points_vis_weight_.clear();
    fused_points_score_.clear();
    fused_points_.shrink_to_fit();
    fused_points_visibility_.shrink_to_fit();
    fused_points_vis_weight_.shrink_to_fit();
    fused_points_score_.shrink_to_fit();
    #endif
  }

  // if (cluster_end == num_cluster_){
  //   MergeFusedPly(reconstruction_path);
  // }
}

void StereoFusion::FuseImage(const Options &options, const int thread_id, 
                             const int image_idx, const int num_eff_threads, 
                             const bool fuse_plane) {
  const int width = images_.at(image_idx).GetWidth();
  const int height = images_.at(image_idx).GetHeight();
  const int height_slice = (height + num_eff_threads - 1) / num_eff_threads;

  int height_limit = std::min(height_slice * (thread_id + 1), height);
  FusionData data;
  data.image_idx = image_idx;
  data.traversal_depth = 0;

  std::vector<float> fused_point_x;
  std::vector<float> fused_point_y;
  std::vector<float> fused_point_z;
  std::vector<float> fused_point_nx;
  std::vector<float> fused_point_ny;
  std::vector<float> fused_point_nz;
  std::vector<float> fused_point_c;
  std::vector<uint8_t> fused_point_r;
  std::vector<uint8_t> fused_point_g;
  std::vector<uint8_t> fused_point_b;
  std::unordered_set<int> fused_point_visibility;
  std::vector<uint8_t> fused_point_seg_id;

  auto& fused_pixel_mask = fused_pixel_masks_.at(image_idx);
  auto& semantic_map = semantic_maps_.at(image_idx);

  const int step = fuse_plane ? 4 : options.step_size;
  const int blank_size = 3;

  for (data.row = std::max(height_slice * thread_id, blank_size); 
       data.row < std::min(height_limit, height - blank_size); 
       data.row += step) {
    for (data.col = blank_size; data.col < width - blank_size; data.col += step) {
      if (fused_pixel_mask->Get(data.row, data.col)) {
        continue;
      }

      BitmapColor<uint8_t> semantic;
      if (semantic_map) {
        semantic_map->GetPixel(data.col, data.row, &semantic);
      }
      if (fuse_plane && semantic_map && semantic.r != LABLE_GROUND) {
        continue;
      }
    
      auto fuse_options = options;
      if (semantic_map && (semantic.r == LABEL_RIVER || semantic.r == LABEL_SKY)) {
        fuse_options.min_num_pixels = 3;
        fuse_options.min_num_visible_images = 3;
      }

      std::vector<FusionData> fusion_queue;
      fusion_queue.push_back(data);
      Fuse(fuse_options, fusion_queue, 
          fused_point_x, fused_point_y, fused_point_z, 
          fused_point_nx, fused_point_ny, fused_point_nz, fused_point_c,
          fused_point_r, fused_point_g, fused_point_b, 
          fused_point_visibility, fused_point_seg_id);
    }
  } 
}

void StereoFusion::Fuse(const Options &options,
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
                        std::vector<uint8_t> &fused_point_seg_id) {
  CHECK_EQ(fusion_queue.size(), 1);

  Eigen::Vector4f fused_ref_point = Eigen::Vector4f::Zero();
  Eigen::Vector3f fused_ref_normal = Eigen::Vector3f::Zero();

  fused_point_x.clear();
  fused_point_y.clear();
  fused_point_z.clear();
  fused_point_nx.clear();
  fused_point_ny.clear();
  fused_point_nz.clear();
  fused_point_c.clear();
  fused_point_r.clear();
  fused_point_g.clear();
  fused_point_b.clear();
  fused_point_visibility.clear();

  fused_point_seg_id.clear();
  std::vector<double> fused_depth_errors;

  const int blank_size = 3;

  while (!fusion_queue.empty()) {
    const auto data = fusion_queue.back();
    const auto image_idx = data.image_idx;
    const int row = data.row;
    const int col = data.col;
    const int traversal_depth = data.traversal_depth;

    fusion_queue.pop_back();

    const auto& image = images_.at(image_idx);
    const int width = image.GetWidth();
    const int height = image.GetHeight();

    // Check if pixel already fused.
    auto &fused_pixel_mask = fused_pixel_masks_.at(image_idx);
    if (!fused_pixel_mask) {
        std::unique_lock<std::mutex> mask_lock(mask_mutex_);
        fused_pixel_mask.reset(new Mat<bool>(width, height, 1));
        fused_pixel_mask->Fill(false);
    }
    if (fused_pixel_mask->Get(row, col)) {
      continue;
    }

    if (!depth_maps_.at(image_idx)) {
      std::unique_lock<std::mutex> depth_lock(depthmap_mutex_);
      if (!depth_maps_.at(image_idx)) {
        const std::string image_name = image_names_[image_idx];
        const std::string file_name = StringPrintf("%s.%s.%s", 
            image_name.c_str(), input_type_.c_str(), DEPTH_EXT);
        const std::string depth_map_path =
            JoinPaths(active_cluster_stereo_path_, DEPTHS_DIR, file_name);
        
        auto depth_map_ptr = std::unique_ptr<DepthMap>(new DepthMap);
        depth_map_ptr->Read(depth_map_path);

        if (!depth_map_ptr->IsValid()) {
          std::cout << StringPrintf(
                    "WARNING: Ignoring image %s, because input is empty."
                    , image_name.c_str())
                      << std::endl;
          continue;
        }
        depth_maps_.at(image_idx).swap(depth_map_ptr);
      }
    }
    if (!depth_maps_.at(image_idx) || !depth_maps_.at(image_idx)->IsValid()) {
      continue;
    }
    const auto &depth_map = *depth_maps_.at(image_idx);
    const float depth = depth_map.Get(row, col);

    // Pixels with negative depth are filtered.
    if (depth <= 0.0f) {
      continue;
    }

    // If the traversal depth is greater than zero, the initial reference
    // pixel has already been added and we need to check for consistency.
    float depth_error;
    if (traversal_depth > 0) {
      // Project reference point into current view.
      const Eigen::Vector3f proj = P_.at(image_idx) * fused_ref_point;

      // Reprojection error reference point in the current view.
      const float col_diff = proj(0) / proj(2) - col;
      const float row_diff = proj(1) / proj(2) - row;
      const float squared_reproj_error =
          col_diff * col_diff + row_diff * row_diff;
      if (squared_reproj_error > max_squared_reproj_error_) {
        continue;
      }

      // Depth error of reference depth with current depth.
      depth_error = std::fabs(proj(2) - depth) / depth;
      if (depth_error > options.max_depth_error) {
        continue;
      }
    }

    // Determine normal direction in global reference frame.
    Eigen::Vector3f normal;
    if (options.with_normal) {
      if (!normal_maps_.at(image_idx)) {
        std::unique_lock<std::mutex> normal_lock(normalmap_mutex_);
        if (!normal_maps_.at(image_idx)) {
          const std::string image_name = image_names_[image_idx];
          const std::string file_name = StringPrintf("%s.%s.%s", 
              image_name.c_str(), input_type_.c_str(), NORMAL_EXT);
          const std::string normal_map_path =
              JoinPaths(active_cluster_stereo_path_, NORMALS_DIR, file_name);

          auto normal_map_ptr = std::unique_ptr<NormalMap>(new NormalMap);
          normal_map_ptr->Read(normal_map_path);

          if (!normal_map_ptr->IsValid()) {
            std::cout << StringPrintf(
                      "WARNING: Ignoring image %s, because input is empty."
                      , image_name.c_str())
                        << std::endl;
            continue;
          }
          normal_maps_.at(image_idx).swap(normal_map_ptr);
        }
      }
      if (!normal_maps_.at(image_idx) || !normal_maps_.at(image_idx)->IsValid()) {
        continue;
      }
      const auto &normal_map = *normal_maps_.at(image_idx);
      normal = inv_R_.at(image_idx) * Eigen::Vector3f(
                normal_map.Get(row, col, 0), normal_map.Get(row, col, 1),
                normal_map.Get(row, col, 2));

      // Check for consistent normal direction with reference normal.
      if (traversal_depth > 0) {
        const float cos_normal_error = fused_ref_normal.dot(normal);
        if (cos_normal_error < min_cos_normal_error_) {
          continue;
        }
      }
    }

    // Determine 3D location of current depth value.
    const Eigen::Vector3f xyz =
        inv_P_.at(image_idx) *
        Eigen::Vector4f(col * depth, row * depth, depth, 1.0f);

    if (options.roi_fuse){
      const Eigen::Vector3f rot_xyz = roi_box_.rot * xyz;
      if (rot_xyz(0) < roi_box_.x_box_min || rot_xyz(0) > roi_box_.x_box_max ||
          rot_xyz(1) < roi_box_.y_box_min || rot_xyz(1) > roi_box_.y_box_max ||
          rot_xyz(2) < roi_box_.z_box_min || rot_xyz(2) > roi_box_.z_box_max){
          break;
      }
    }

    // Read the color of the pixel.
    BitmapColor<uint8_t> color;
    const Bitmap& bitmap = images_.at(image_idx).GetBitmap();
    auto& conf_map = conf_maps_.at(image_idx);
    if (!bitmap.Data()) {
      std::unique_lock<std::mutex> image_lock(image_mutex_);
      if (!bitmap.Data()) {
        // Load image.
        const auto image_name = image_names_[image_idx];
        const auto image_path = JoinPaths(active_dense_path_, IMAGES_DIR, image_name);
        Bitmap bitmap_;
        bitmap_.Read(image_path);
        images_.at(image_idx).SetBitmap(bitmap_);
        
        // Load semantic.
        const auto semantic_path = JoinPaths(active_dense_path_, SEMANTICS_DIR);
        if (ExistsDir(semantic_path) && !semantic_maps_.at(image_idx)) {
          const auto image_name = image_names_[image_idx];
          const auto semantic_name = JoinPaths(semantic_path, image_name);
          const auto semantic_base_name = semantic_name.substr(0, semantic_name.size() - 3);
          if (ExistsFile(semantic_base_name + "png")) {
            semantic_maps_.at(image_idx).reset(new Bitmap);
            semantic_maps_.at(image_idx)->Read(semantic_base_name + "png", false);
          } else if (ExistsFile(semantic_base_name + "jpg")) {
            semantic_maps_.at(image_idx).reset(new Bitmap);
            semantic_maps_.at(image_idx)->Read(semantic_base_name + "jpg", false);
          }
        }
      }
      if (!conf_map || !conf_map->IsValid()) {
        const auto image_name = image_names_[image_idx];
        const std::string file_name = StringPrintf("%s.%s.%s", 
            image_name.c_str(), input_type_.c_str(), DEPTH_EXT);
        const auto conf_map_path = JoinPaths(active_cluster_stereo_path_, CONFS_DIR, file_name);
        if (ExistsFile(conf_map_path)) {
          conf_map = std::unique_ptr<MatXf>(new MatXf);
          conf_map->Read(conf_map_path);
        }
      }
    }
    bitmap.GetPixel(col, row, &color);

    {
      if (col > 1 && col < width - 1 && row > 1 && row < height - 1) {
        BitmapColor<uint8_t> lcolor, rcolor, tcolor, bcolor;
        bitmap.GetPixel(col - 1, row, &lcolor);
        bitmap.GetPixel(col, row - 1, &tcolor);
        bitmap.GetPixel(col + 1, row, &rcolor);
        bitmap.GetPixel(col, row + 1, &bcolor);
        float gx = (lcolor.r - rcolor.r) / 255.0f;
        float gy = (tcolor.r - bcolor.r) / 255.0f;
        float grad = sqrt(gx * gx + gy * gy);
        fused_point_c.push_back(grad);
      } else {
        fused_point_c.push_back(0);
      }
    }

    // Set the current pixel as visited.
    // pthread_mutex_lock(&mutex);
    fused_pixel_mask->Set(row, col, true);
    // pthread_mutex_unlock(&mutex);

    // Accumulate statistics for fused point.
    fused_point_x.push_back(xyz(0));
    fused_point_y.push_back(xyz(1));
    fused_point_z.push_back(xyz(2));
    if (options.with_normal) {
      fused_point_nx.push_back(normal(0));
      fused_point_ny.push_back(normal(1));
      fused_point_nz.push_back(normal(2));
    }
    fused_point_r.push_back(color.r);
    fused_point_g.push_back(color.g);
    fused_point_b.push_back(color.b);
    fused_point_visibility.insert(image_idx);
    if (traversal_depth > 0) {
      fused_depth_errors.push_back(depth_error);
    }

    if (semantic_maps_.at(image_idx)) {
      BitmapColor<uint8_t> semantic;
      semantic_maps_.at(image_idx)->GetPixel(col, row, &semantic);
      fused_point_seg_id.push_back(semantic.r);
    }
    // if (conf_map && conf_map->IsValid()) {
    //   float conf = conf_map->Get(row, col);
    //   fused_point_c.push_back(conf);
    // }

    // Remember the first pixel as the reference.
    if (traversal_depth == 0) {
      fused_ref_point = Eigen::Vector4f(xyz(0), xyz(1), xyz(2), 1.0f);
      fused_ref_normal = normal;
    }

   if (fused_point_x.size() >= static_cast<size_t>(options.max_num_pixels)) {
     break;
   }

    FusionData next_data;
    next_data.traversal_depth = traversal_depth + 1;

    if (next_data.traversal_depth >= options.max_traversal_depth) {
      continue;
    }

    for (const auto next_image_id : overlapping_images_.at(image_idx)) {
      if (!used_images_.at(next_image_id) ||
          fused_images_.at(next_image_id)) {
        continue;
      }

      next_data.image_idx = next_image_id;

      const Eigen::Vector3f next_proj =
          P_.at(next_image_id) * xyz.homogeneous();
      next_data.col = static_cast<int>(std::round(next_proj(0) / next_proj(2)));
      next_data.row = static_cast<int>(std::round(next_proj(1) / next_proj(2)));

      const auto& next_image = images_.at(next_image_id);
      if (next_data.col < blank_size || next_data.row < blank_size ||
          next_data.col >= next_image.GetWidth() - blank_size ||
          next_data.row >= next_image.GetHeight() - blank_size) {
        continue;
      }

      fusion_queue.push_back(next_data);
    }
  }

  fusion_queue.clear();
  fusion_queue.shrink_to_fit();

  const size_t num_pixels = fused_point_x.size();
  if (num_pixels >= options.min_num_pixels
      && fused_point_visibility.size() >= options.min_num_visible_images) {
    PlyPoint fused_point;
    fused_point.x = internal::Median(&fused_point_x);
    fused_point.y = internal::Median(&fused_point_y);
    fused_point.z = internal::Median(&fused_point_z);

    if (std::isnan(fused_point.x) || std::isinf(fused_point.x) ||
        std::isnan(fused_point.y) || std::isinf(fused_point.y) ||
        std::isnan(fused_point.z) || std::isinf(fused_point.z)) {
      return;
    }

    if (options.with_normal) {
      Eigen::Vector3f fused_normal;
      fused_normal.x() = internal::Median(&fused_point_nx);
      fused_normal.y() = internal::Median(&fused_point_ny);
      fused_normal.z() = internal::Median(&fused_point_nz);
      const float fused_normal_norm = fused_normal.norm();
      if (fused_normal_norm < std::numeric_limits<float>::epsilon()) {
        return;
      }
      fused_point.nx = fused_normal.x() / fused_normal_norm;
      fused_point.ny = fused_normal.y() / fused_normal_norm;
      fused_point.nz = fused_normal.z() / fused_normal_norm;
    }

    if (!fused_point_seg_id.empty()) {
      int max_samples = 0;
      int best_label = -1;
      int samps_per_label[256];
      memset(samps_per_label, 0, sizeof(int) * 256);
      for (auto id : fused_point_seg_id) {
        uint8_t sid = (id + 256) % 256;
        samps_per_label[sid]++;
        if (samps_per_label[sid] > max_samples) {
          max_samples = samps_per_label[sid];
          best_label = id;
        }
      }
      size_t num_fused_point = fused_point_seg_id.size();
      for (auto id : semantic_label_black_list_) {
        float ratio = samps_per_label[id] / (float)num_fused_point;
        if (ratio > options.num_consistent_semantic_ratio) {
          return;
        }
      }
      fused_point.s_id = best_label;
    }

    fused_point.r = TruncateCast<float, uint8_t>(
        std::round(internal::Median(&fused_point_r)));
    fused_point.g = TruncateCast<float, uint8_t>(
        std::round(internal::Median(&fused_point_g)));
    fused_point.b = TruncateCast<float, uint8_t>(
        std::round(internal::Median(&fused_point_b)));

    float m_conf(0.0f);
    for (size_t i = 0; i < fused_point_c.size(); ++i) {
      m_conf += fused_point_c[i];
    }
    m_conf /= fused_point_c.size();
    if (fused_point_visibility.size() >= 3) {
      m_conf = 1.0f;
    }
    m_conf = 1.0f - m_conf;

    float score = internal::Median(&fused_depth_errors) / options.max_depth_error;
    score = std::min(0.6f * score + 0.4f * m_conf, 1.0f);
    if (std::isnan(score) || std::isinf(score)) {
      return;
    }

    std::unique_lock<std::mutex> push_lock(push_mutex_);
    {
      fused_points_.push_back(fused_point);
      fused_points_visibility_.emplace_back(fused_point_visibility.begin(),
                                            fused_point_visibility.end());
      fused_points_score_.push_back(score);
    }
  }
}

void StereoFusion::RemoveDuplicatePoints(const Options& options,
                             std::vector<int>& cluster_image_ids,
                             const size_t num_pre_fused_points) {
  Timer timer;
  timer.Start();
  size_t i, j;
  std::unordered_map<int, int> unoder_image_ids;
  for (int i = 0; i < cluster_image_ids.size(); i++){
    unoder_image_ids.emplace(cluster_image_ids[i], i);
  }
  std::vector<Mat<int> > occupy_maps(cluster_image_ids.size());
  for (i = 0; i < occupy_maps.size(); ++i) {
    auto id = cluster_image_ids[i];
    auto& image = images_.at(id);
    occupy_maps.at(i) = Mat<int>(image.GetWidth(), image.GetHeight(), 1);
    occupy_maps.at(i).Fill(-1);
  }

  int removal_num = fused_points_.size() - num_pre_fused_points;
  std::vector<bool> removal(removal_num, 0);

  for (i = num_pre_fused_points; i < fused_points_.size(); ++i) {
    if (removal.at(i - num_pre_fused_points)) {
      continue;
    }
    Eigen::Vector3f point((float*)&fused_points_[i]);
    auto fused_point_vis = fused_points_visibility_.at(i);
    for (auto vis : fused_point_vis) {
      if (unoder_image_ids.find((int)vis) == unoder_image_ids.end()){
        continue;
      }
      auto occupy_id = unoder_image_ids.find((int)vis)->second;
      const mvs::Image& image = images_.at(vis);
      const Eigen::RowMatrix3f K(image.GetK());
      const Eigen::RowMatrix3f R(image.GetR());
      const Eigen::Vector3f T(image.GetT());
      Eigen::Vector3f proj = K * (R * point + T);
      int u = std::round(proj[0] / proj[2]);
      int v = std::round(proj[1] / proj[2]);
      if (u < 0 || u >= image.GetWidth() ||
          v < 0 || v >= image.GetHeight()) {
        continue;
      }
      int point_idx = occupy_maps.at(occupy_id).Get(v, u);
      if (point_idx == -1) {
        occupy_maps.at(occupy_id).Set(v, u, i);
        continue;
      }
      if (fused_points_visibility_.at(point_idx).size() > fused_point_vis.size()) {
        removal.at(i - num_pre_fused_points) = true;
        break;
      } else {
        occupy_maps.at(occupy_id).Set(v, u, i);
        removal.at(point_idx - num_pre_fused_points) = true;
      }
    }
  }

  std::vector<Mat<int> >().swap(occupy_maps);

  bool has_score = (fused_points_score_.size() == fused_points_.size());
  size_t duplicate_point = 0;
  for (i = num_pre_fused_points, j = num_pre_fused_points; i < fused_points_.size(); ++i) {
    if (!removal.at(i - num_pre_fused_points)) {
      fused_points_.at(j) = fused_points_.at(i);
      fused_points_visibility_.at(j) = fused_points_visibility_.at(i);
      if (has_score) {
        fused_points_score_.at(j) = fused_points_score_.at(i);
      }
      j = j + 1;
    } else {
      duplicate_point++;
    }
  }

  std::vector<bool>().swap(removal);

  fused_points_.resize(j);
  fused_points_visibility_.resize(j);
  if (has_score) {
    fused_points_score_.resize(j);
  }
  fused_points_.shrink_to_fit();
  fused_points_visibility_.shrink_to_fit();
  fused_points_score_.shrink_to_fit();
  std::cout << StringPrintf("Remove %d duplicate points in %.3fs\n", 
                            duplicate_point, timer.ElapsedSeconds());
}


void StereoFusion::AdjustDuplicatePoints(const Options& options,
                            std::vector<int>& cluster_image_ids,
                            const std::vector<short int> cluster_ids, 
                            const size_t num_pre_fused_points) {
  size_t i, j;
  std::unordered_map<int, int> unoder_image_ids;
  for (int i = 0; i < cluster_image_ids.size(); i++){
    unoder_image_ids.emplace(cluster_image_ids[i], i);
  }
  std::vector<Mat<int> > occupy_maps(cluster_image_ids.size());
  for (i = 0; i < occupy_maps.size(); ++i) {
    auto id = cluster_image_ids[i];
    auto& image = images_.at(id);
    occupy_maps.at(i) = Mat<int>(image.GetWidth(), image.GetHeight(), 1);
    occupy_maps.at(i).Fill(-1);
  }
  int merge_num = fused_points_.size() - num_pre_fused_points;
  std::vector<bool> merged(merge_num, 0);

  for (i = num_pre_fused_points; i < fused_points_.size(); ++i) {
    if (merged.at(i - num_pre_fused_points)) {
      continue;
    }
    Eigen::Vector3f point((float*)&fused_points_[i]);
    auto fused_point_vis = fused_points_visibility_.at(i);
    for (auto vis : fused_point_vis) {
      if (unoder_image_ids.find((int)vis) == unoder_image_ids.end()){
        continue;
      }
      auto occupy_id = unoder_image_ids.find((int)vis)->second;
      const mvs::Image& image = images_.at(vis);
      const Eigen::RowMatrix3f K(image.GetK());
      const Eigen::RowMatrix3f R(image.GetR());
      const Eigen::Vector3f T(image.GetT());
      Eigen::Vector3f proj = K * (R * point + T);
      int u = std::round(proj[0] / proj[2]);
      int v = std::round(proj[1] / proj[2]);
      if (u < 0 || u >= image.GetWidth() ||
          v < 0 || v >= image.GetHeight()) {
        continue;
      }
      int point_idx = occupy_maps.at(occupy_id).Get(v, u);
      if (point_idx == -1) {
        occupy_maps.at(occupy_id).Set(v, u, i);
        continue;
      }
      Eigen::Vector3f occupy_point((float*)&fused_points_[point_idx]);
      Eigen::Vector3f occupy_proj = K * (R * occupy_point + T);

      Eigen::Vector3f normal_i(fused_points_[i].nx, 
                               fused_points_[i].ny, 
                               fused_points_[i].nz);
      normal_i.normalize();
      Eigen::Vector3f normal_idx(fused_points_[point_idx].nx, 
                                 fused_points_[point_idx].ny, 
                                 fused_points_[point_idx].nz);
      normal_idx.normalize();

      if (std::abs(occupy_proj[2] - proj[2]) < proj[2] * options.max_depth_error &&
          normal_idx.dot(normal_i) > min_cos_normal_error_ &&
          cluster_ids[i] != cluster_ids[point_idx]){
        merged.at(point_idx - num_pre_fused_points) = true;
        occupy_maps.at(occupy_id).Set(v, u, i);
        merge_points_.at(i) = point_idx + 1;
        merge_points_.at(point_idx) = -1;

      }else if(fused_points_visibility_.at(point_idx).size() > fused_point_vis.size()){
        merged.at(i - num_pre_fused_points) = true;
        merge_points_.at(i) = -1;
        break;
      } else {
        merged.at(point_idx - num_pre_fused_points) = true;
        occupy_maps.at(occupy_id).Set(v, u, i);
        merge_points_.at(point_idx) = -1;
      }
    }
  }

  std::vector<Mat<int> >().swap(occupy_maps);

}

int StereoFusion::RemoveCommonDuplicatePoints() {
  bool has_weight = 
    fused_points_vis_weight_.size() == fused_points_visibility_.size();
  std::cout << "has_weight: " << fused_points_vis_weight_.size() 
    << ", " << fused_points_visibility_.size() << std::endl;
  for (int i = 0; i < fused_points_.size(); i++){
    if (merge_points_.at(i) > 0) {
      Eigen::Vector3f point((float*)&fused_points_[i]);
      int point_idx(merge_points_.at(i) - 1);
      Eigen::Vector3f occupy_point((float*)&fused_points_[point_idx]);
      Eigen::Vector3f merge_point = (point * fused_points_visibility_.at(i).size()
        + occupy_point * fused_points_visibility_.at(point_idx).size()) 
        / (fused_points_visibility_.at(point_idx).size() + fused_points_visibility_.at(i).size());
      fused_points_.at(i).x = merge_point.x();
      fused_points_.at(i).y = merge_point.y();
      fused_points_.at(i).z = merge_point.z();

      std::unordered_set<int> point_vis_set(
        fused_points_visibility_.at(i).begin(), 
        fused_points_visibility_.at(i).end());        
      for (int j = 0; j < fused_points_visibility_.at(point_idx).size(); j++){
        auto vis = fused_points_visibility_.at(point_idx)[j];
        if (point_vis_set.find(vis) == point_vis_set.end()){
          fused_points_visibility_.at(i).emplace_back(vis);
          if (has_weight){
            auto wgt = fused_points_vis_weight_.at(point_idx)[j];
            fused_points_vis_weight_.at(i).emplace_back(wgt);
          }
        } else if (has_weight){
          auto iter = std::find(fused_points_visibility_.at(i).begin(),
                                fused_points_visibility_.at(i).end(), vis);
          if (iter == fused_points_visibility_.at(i).end()){
            continue;
          }
          int pnt_wgt_id = std::distance(fused_points_visibility_.at(i).begin(), iter);
          auto wgt = fused_points_vis_weight_.at(point_idx)[j];
          fused_points_vis_weight_.at(i)[pnt_wgt_id] += wgt;
        }
      }
    }
  }

  int duplicate_point = 0;
  int64_t i, j;
  for (i = 0, j = 0; i < fused_points_.size(); ++i) {
    if (merge_points_.at(i) < 0) {
      duplicate_point++;
    } else {
      fused_points_.at(j) = fused_points_.at(i);
      fused_points_visibility_.at(j) = fused_points_visibility_.at(i);
      if (has_weight){
        fused_points_vis_weight_.at(j) = fused_points_vis_weight_.at(i);
      }
      j = j + 1;
    }
  }

  std::vector<int64_t>().swap(merge_points_);

  fused_points_.resize(j);
  fused_points_visibility_.resize(j);
  fused_points_.shrink_to_fit();
  fused_points_visibility_.shrink_to_fit();
  if (has_weight){
    fused_points_vis_weight_.resize(j);
    fused_points_vis_weight_.shrink_to_fit();
  }

  return duplicate_point;
}

typedef CGAL::Exact_predicates_inexact_constructions_kernel Kernel;
typedef Kernel::Point_3                                     Point_3;
typedef boost::tuple<Point_3,int>                           Point_and_int;
typedef CGAL::Search_traits_3<Kernel>                       Traits_base;
typedef CGAL::Search_traits_adapter<Point_and_int,
  CGAL::Nth_of_tuple_property_map<0, Point_and_int>,
  Traits_base>                                              Traits;
typedef CGAL::Orthogonal_k_neighbor_search<Traits>          K_neighbor_search;
typedef K_neighbor_search::Tree                             Tree;
typedef K_neighbor_search::Distance                         Distance;
typedef K_neighbor_search::Point_with_transformed_distance  Point_with_distance;

void ComputeAverageDistance(const std::vector<PlyPoint>& fused_points,
  std::vector<float> &point_spacings, float* average_spacing,
  const int nb_neighbors) {
  Timer timer;
  timer.Start();

  const size_t num_point = fused_points.size();
  std::vector<Point_3> points(num_point);
  for (std::size_t i = 0; i < num_point; i++) {
    const auto &fused_point = fused_points[i];
    points[i] = Point_3(fused_point.x, fused_point.y, fused_point.z);
  }
  std::vector<int> indices(num_point);
  std::iota(indices.begin(), indices.end(), 0);

  // Instantiate a KD-tree search.
  std::cout << "Instantiating a search tree for point cloud spacing ..." << std::endl;
  Tree tree(
    boost::make_zip_iterator(boost::make_tuple(points.begin(),indices.begin())),
    boost::make_zip_iterator(boost::make_tuple(points.end(),indices.end()))
  );
#ifdef CGAL_LINKED_WITH_TBB
  tree.build<CGAL::Parallel_tag>();
#endif

  *average_spacing = 0.0f;
  point_spacings.resize(num_point);

#ifndef CGAL_LINKED_WITH_TBB
  const int num_eff_threads = GetEffectiveNumThreads(-1);
  std::cout << "num_eff_threads: " << num_eff_threads << std::endl;
  std::unique_ptr<ThreadPool> thread_pool;
  thread_pool.reset(new ThreadPool(num_eff_threads));
  {
    std::cout << "Starting average spacing computation ..." << std::endl;
    auto ComputePointSpace = [&](std::size_t i) {
      const auto &query = points[i];
      const auto &fused_point = fused_points[i];
      Eigen::Vector3f point(&fused_point.x);
      // performs k + 1 queries (if unique the query point is
      // output first). search may be aborted when k is greater
      // than number of input points
      Distance tr_dist;
      K_neighbor_search search(tree, query, nb_neighbors + 1);
      auto &point_spacing = point_spacings[i];
      point_spacing = 0.0f;
      std::size_t k = 0;
      for (K_neighbor_search::iterator it = search.begin(); it != search.end() && k <= nb_neighbors; it++, k++)
      {
        point_spacing += tr_dist.inverse_of_transformed_distance(it->second);
      }
      // output point spacing
      if (k > 1) {
        point_spacing /= (k - 1);
      }
    };

    for (std::size_t i = 0; i < num_point; ++i) {
      thread_pool->AddTask(ComputePointSpace, i);
    }
    thread_pool->Wait();

    for (auto & point_spacing : point_spacings) {
      *average_spacing += point_spacing;
    }
  }
#else
  {
    std::cout << "Starting average spacing computation(Parallel) ..." << std::endl;
    tbb::parallel_for (tbb::blocked_range<std::size_t> (0, num_point),
                     [&](const tbb::blocked_range<std::size_t>& r) {
                       for (std::size_t s = r.begin(); s != r.end(); ++ s) {
                        // Neighbor search can be instantiated from
                        // several threads at the same time
                        const auto &query = points[s];
                        const Eigen::Vector3f point(&fused_points.at(s).x);
                        K_neighbor_search search(tree, query, nb_neighbors + 1);

                        auto &point_spacing = point_spacings[s];
                        point_spacing = 0.0f;
                        std::size_t k = 0;
                        Distance tr_dist;
                        for (K_neighbor_search::iterator it = search.begin(); it != search.end() && k <= nb_neighbors; it++, k++)
                        {
                          point_spacing += tr_dist.inverse_of_transformed_distance(it->second);
                        }
                        // output point spacing
                        if (k > 1) {
                          point_spacing /= (k - 1);
                        }
                       }
                     });
    for (auto & point_spacing : point_spacings) {
      *average_spacing += point_spacing;
    }
  }
  #endif

  *average_spacing /= num_point;
  std::cout << "Average spacing: " << *average_spacing << std::endl;
  timer.PrintMinutes();
}

void StereoFusion::RemoveOutliers(const Options& options) {
  if (fused_points_score_.size() != fused_points_.size()){
    std::cout << "fused_points_score_ is filled by 0" << std::endl;
    std::vector<float> temp_scores(fused_points_.size(), 0.0f);
    fused_points_score_.swap(temp_scores);
  }
  Timer timer;
  timer.Start();

  size_t i, j;

  std::vector<float> dists;
  float average_dist;
  ComputeAverageDistance(fused_points_, dists, &average_dist, options.nb_neighbors);
  if (average_dist > 1e-6){
    for (i = 0; i < fused_points_score_.size(); ++i) {
      auto& score = fused_points_score_.at(i);
      float factor = dists[i] / (options.max_spacing_factor * average_dist);
      score = score * 0.6 + factor * 0.4;
    }
  } else {
    std::cout << "average_dist <= 1e-6\tfactor = 1.0f" << std::endl;
  }

  const float robust_min = Percentile(fused_points_score_, 0);
  const float robust_max = Percentile(fused_points_score_, 100);
  const float robust_range = robust_max - robust_min;

  std::cout << "robust min: " << robust_min << std::endl;
  std::cout << "robust max: " << robust_max << std::endl;

  std::vector<float> ratios(256, 0.0f);
  for (size_t i = 0; i < fused_points_score_.size(); ++i) {
      const float gray = (fused_points_score_[i] - robust_min) / robust_range;
      ratios[gray * 255]++;
  }
  for (size_t i = 0; i < 256; ++i) {
    ratios[i] /= fused_points_score_.size();
  }
  float aum_ratio = 0.0f;
  std::vector<float> acc_ratios(256, 0.0f);
  for (size_t i = 0; i < 256; ++i) {
    aum_ratio += ratios[i];
    acc_ratios[i] = aum_ratio;
  }

  size_t num_outlier(0);
  // const float score_thres = mval + options.outlier_deviation_factor * stdev;
  for (i = 0, j = 0; i < fused_points_score_.size(); ++i) {
    const float gray = (fused_points_score_[i] - robust_min) / robust_range;
    int val = gray * 255;
    if (ratios[val] < 0.01 && acc_ratios[val] >= 0.9) {
      num_outlier++;
    } else {
      fused_points_.at(j) = fused_points_.at(i);
      fused_points_visibility_.at(j) = fused_points_visibility_.at(i);
      fused_points_score_.at(j) = fused_points_score_.at(i);
      j = j + 1;
    }
  }
  fused_points_.resize(j);
  fused_points_visibility_.resize(j);
  fused_points_score_.resize(j);
  fused_points_.shrink_to_fit();
  fused_points_visibility_.shrink_to_fit();
  fused_points_score_.shrink_to_fit();
  std::cout << StringPrintf("Score Remove %d outliers in %.3fs\n", 
                            num_outlier, timer.ElapsedSeconds());
  // VoxelRemoveOutlier(average_dist, options);
}

void StereoFusion::VoxelRemoveOutlier(const double average_dist, const Options& options) {
    float factor = options.outlier_voxel_factor;
    const double filter_x = average_dist * factor;
    const double filter_y = average_dist * factor; 
    const double filter_z = average_dist * factor;

    Timer timer;
    timer.Start();

    size_t i, j;
    size_t num_point = fused_points_.size();
    std::cout << "Compute Bounding Box" << std::endl;
    Eigen::Vector3f lt, rb;
    lt.setConstant(std::numeric_limits<float>::max());
    rb.setConstant(std::numeric_limits<float>::lowest());
    for (i = 0; i < num_point; ++i) {
        const PlyPoint& point = fused_points_.at(i);
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
    const int num_neighbors = 26;
    int neighbor_offs[num_neighbors][3] = 
    { { -1, -1, -1 }, { 0, -1, -1 }, { 1, -1, -1}, 
      { -1, 0, -1 }, { 0, 0, -1 }, { 1, 0, -1}, 
      { -1, 1, -1 }, { 0, 1, -1 }, { 1, 1, -1}, 
      { -1, -1, 0 }, { 0, -1, 0 }, { 1, -1, 0 }, 
      { -1, 0, 0 }, { 1, 0, 0}, 
      { -1, 1, 0 }, { 0, 1, 0 }, { 1, 1, 0 }, 
      { -1, -1, 1 }, { 0, 1, 1 }, { 1, -1, 1}, 
      { -1, 0, 1 }, { 0, 0, 1 }, { 1, 0, 1}, 
      { -1, 1, 1 }, { 0, 1, 1 }, { 1, 1, 1}};

    std::unordered_map<uint64_t, std::vector<size_t> > m_voxels_map;
    for (i = 0; i < num_point; ++i) {
        const PlyPoint& point = fused_points_.at(i);
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

    std::vector<uint64_t > vec_remove_pnts;
    for (auto voxel_map : m_voxels_map) {
        int num_neighbor = 0;
        bool weak_pnts = false;
        if (voxel_map.second.size() < factor) {
            weak_pnts = true;
        }

        num_neighbor++;
        uint64_t key = voxel_map.first;
        uint64_t iz = key / slide;
        uint64_t iy = (key % slide) / lenx;
        uint64_t ix = (key % slide) % lenx;

        for (int i = 0; i < num_neighbors; i++){
            uint64_t neighbor_key = (iz + neighbor_offs[i][2]) * slide +
                                    (iy + neighbor_offs[i][1]) * lenx + 
                                    (ix + neighbor_offs[i][0]);
            if (m_voxels_map.find(neighbor_key) == m_voxels_map.end()){
                continue;
            } 
            uint64_t next_neighbor_key = (iz + neighbor_offs[i][2] * 2) * slide +
                                         (iy + neighbor_offs[i][1] * 2) * lenx + 
                                         (ix + neighbor_offs[i][0] * 2);
            if (m_voxels_map[neighbor_key].size() > factor / 2 + 0.5){
                num_neighbor++;
               if (m_voxels_map.find(next_neighbor_key) == m_voxels_map.end()){
                    continue;
                }
                if (m_voxels_map[next_neighbor_key].size() > factor / 2 + 0.5){
                    num_neighbor++;
                }
            } else if (m_voxels_map[neighbor_key].size() > 0.5){
                if (m_voxels_map.find(next_neighbor_key) == m_voxels_map.end()){
                    continue;
                }
                if (m_voxels_map[neighbor_key].size() + m_voxels_map[neighbor_key].size() > factor){
                    num_neighbor++;
                }
            }
            
            if (num_neighbor > 3){
                break;
            }
        }

        if ((num_neighbor < 3 && weak_pnts) || (num_neighbor == 0)) {
            for (auto pnt_id : voxel_map.second){
                vec_remove_pnts.push_back(pnt_id);
            }
        }
    }

    std::cout << "Number of points, average_dist, factor, voxels_num: " 
              << num_point << ", " << average_dist << ", " << factor 
              << ", " << m_voxels_map.size() << std::endl;

    int num_outlier = 0;
    if (!vec_remove_pnts.empty()){
        std::sort(vec_remove_pnts.begin(), vec_remove_pnts.end());
        size_t i, j, k;
        for (i = 0, j = 0, k = 0; i < fused_points_.size(); ++i){
            if (i == vec_remove_pnts[k]){
                k++;
                num_outlier++;
            }else {
                fused_points_.at(j) = fused_points_.at(i);
                fused_points_score_.at(j) = fused_points_score_.at(i);
                fused_points_visibility_.at(j) = fused_points_visibility_.at(i);
                j = j + 1;
            }
        }
        fused_points_.resize(j);
        fused_points_score_.resize(j);
        fused_points_visibility_.resize(j);
        fused_points_.shrink_to_fit();
        fused_points_score_.shrink_to_fit();
        fused_points_visibility_.shrink_to_fit();
    }
    std::cout << StringPrintf("Voxel Remove %d outliers in %.3fs\n", 
                                num_outlier, timer.ElapsedSeconds());
}

void StereoFusion::ComplementPlanesPanorama(const Options& options,
                                    std::vector<int>& cluster_image_ids,
                                    size_t num_pre_fused_points,
                                    size_t num_pre_fused_images,
                                    size_t all_fusion_image_num) {
  std::unordered_map<int, int> unoder_image_ids;
  for (int i = 0; i < cluster_image_ids.size(); i++){
    unoder_image_ids.emplace(cluster_image_ids[i], i);
  }
  size_t image_num = images_.size();
  for (auto image_idx : cluster_image_ids) {
    auto& fused_pixel_mask = fused_pixel_masks_[image_idx];
    if (!fused_pixel_mask) {
      fused_pixel_mask.reset(new Mat<bool>(images_.at(image_idx).GetWidth(), 
        images_.at(image_idx).GetHeight(), 1));
    }
    fused_pixel_mask->Fill(false);
  }

  std::vector<Eigen::Vector3d> points;
  std::vector<Eigen::Vector3d> cameras_center;
  std::vector<std::vector<size_t> > image_points_idx(image_num);
  std::unordered_set<uint32_t> unique_cameras_idx;
  for (size_t i = num_pre_fused_points; i < fused_points_.size(); ++i) {
    if (fused_points_[i].s_id != LABLE_GROUND) {
      continue;
    }

    Eigen::Vector3d point;
    point.x() = fused_points_[i].x;
    point.y() = fused_points_[i].y;
    point.z() = fused_points_[i].z;
    points.emplace_back(point);

    Eigen::Map<const Eigen::Vector3f> p((float*)&fused_points_[i]);
    auto points_vis = fused_points_visibility_[i];
    for (auto vis : points_vis) {
      if (unoder_image_ids.find((int)vis) == unoder_image_ids.end()){
        continue;
      }
      unique_cameras_idx.insert(vis);
      const mvs::Image& image = images_.at(vis);
      Eigen::Map<const Eigen::RowMatrix3f> K(image.GetK());
      Eigen::Map<const Eigen::RowMatrix3f> R(image.GetR());
      Eigen::Map<const Eigen::Vector3f> T(image.GetT());
      Eigen::Vector3f proj = K * (R * p + T);
      int u = proj[0] / proj[2];
      int v = proj[1] / proj[2];
      if (u < 0 || u >= image.GetWidth() || v < 0 || v >= image.GetHeight()) {
        continue;
      }
      fused_pixel_masks_[vis]->Set(v, u, true);
      image_points_idx[vis].push_back(i);
    }
  }
  for (auto cam_idx : unique_cameras_idx) {
    const mvs::Image& image = images_.at(cam_idx);
    cameras_center.emplace_back(Eigen::Vector3f(image.GetC()).cast<double>());
  }

  std::cout << "Collect " << points.size() << " candidate points" << std::endl;
  std::cout << "Collect " << cameras_center.size() << " cameras" << std::endl;

  // Fit Main plane.
  Eigen::Vector4d main_plane;
  if (!internal::FitPlane(points, cameras_center, main_plane)) {
    return;
  }
  std::cout << "Primary plane parameter: " << main_plane.transpose() << std::endl;

  auto FillDepthMap = [&](int image_idx, int num_image) {
    for (int i = 0; i < num_perspective_per_image; ++i) {
      const mvs::Image& image = images_.at(image_idx + i);

      const auto image_name = image_names_[image_idx + i];
      const auto file_name = StringPrintf("%s.%s.%s", 
          image_name.c_str(), input_type_.c_str(), DEPTH_EXT);
      const auto depth_map_path = JoinPaths(active_cluster_stereo_path_, DEPTHS_DIR, file_name);
      const auto normal_map_path = JoinPaths(active_cluster_stereo_path_, NORMALS_DIR, file_name);

      if (!ExistsFile(depth_map_path) || 
         (options.with_normal && !ExistsFile(normal_map_path))) {
        return;
      }

      if (!depth_maps_.at(image_idx + i)) {
        auto depth_map_ptr = std::unique_ptr<DepthMap>(new DepthMap);
        depth_map_ptr->Read(depth_map_path);
        
        if (!depth_map_ptr->IsValid()) {
          std::cout << StringPrintf(
                    "WARNING: Ignoring image %s, because input is empty."
                    , image_name.c_str())
                      << std::endl;
          return;
        }
        depth_maps_.at(image_idx + i).swap(depth_map_ptr);
      }
      std::unique_ptr<DepthMap>& depth_map = depth_maps_.at(image_idx + i);
      if (!depth_map || !depth_map->IsValid()) {
        return;
      }
      if (options.with_normal) {
        if (!normal_maps_.at(image_idx + i)) {
          auto normal_map_ptr = std::unique_ptr<NormalMap>(new NormalMap);
          normal_map_ptr->Read(normal_map_path);

          if (!normal_map_ptr->IsValid()) {
            std::cout << StringPrintf(
                      "WARNING: Ignoring image %s, because input is empty."
                      , image_name.c_str())
                        << std::endl;
            return;
          }
          normal_maps_.at(image_idx + i).swap(normal_map_ptr);
        }
        std::unique_ptr<NormalMap>& normal_map = normal_maps_.at(image_idx + i);
        if (!normal_map || !normal_map->IsValid()) {
          return;
        }
      }
      std::unique_ptr<Bitmap>& semantic_map = semantic_maps_.at(image_idx + i);
      const auto semantic_path = JoinPaths(active_dense_path_, SEMANTICS_DIR);
      if (ExistsDir(semantic_path) && !semantic_map) {
        const auto image_name = image_names_[image_idx + i];
        const auto semantic_name = JoinPaths(semantic_path, image_name);
        const auto semantic_base_name = semantic_name.substr(0, semantic_name.size() - 3);
        if (ExistsFile(semantic_base_name + "png")) {
          semantic_maps_.at(image_idx + i).reset(new Bitmap);
          semantic_maps_.at(image_idx + i)->Read(semantic_base_name + "png", false);
        } else if (ExistsFile(semantic_base_name + "jpg")) {
          semantic_maps_.at(image_idx + i).reset(new Bitmap);
          semantic_maps_.at(image_idx + i)->Read(semantic_base_name + "jpg", false);
        }
      }
    }

    class mvs::Image &image = images_.at(0);
    const double mean_focal_length = (image.GetK()[0] + image.GetK()[4]) * 0.5;
    const double dist_to_best_model = options.dist_to_best_plane_model / mean_focal_length;

    // Fit plane.
    std::vector<Eigen::Vector3d> image_points;
    for (int i = 0; i < num_perspective_per_image; ++i) {
      for (size_t point_idx : image_points_idx[image_idx + i]) {
        Eigen::Vector3d point;
        point.x() = fused_points_[point_idx].x;
        point.y() = fused_points_[point_idx].y;
        point.z() = fused_points_[point_idx].z;
        image_points.emplace_back(point);
      }
    }
    Eigen::Vector4d plane;
    if (!internal::FitConsistentPlane(image_points, main_plane, 
                                      dist_to_best_model, 
                                      options.angle_diff_to_main_plane,
                                      options.min_inlier_ratio_to_best_model, 
                                      plane)) {
      return;
    }

    Eigen::Vector3f m_fNormal = plane.head<3>().cast<float>();

    for (int i = 0; i < num_perspective_per_image; ++i) {
      const auto image_name = image_names_[image_idx + i];
      const auto file_name = StringPrintf("%s.%s.%s", 
          image_name.c_str(), input_type_.c_str(), DEPTH_EXT);
      const auto depth_map_path = JoinPaths(active_cluster_stereo_path_, DEPTHS_DIR, file_name);
      const auto normal_map_path = JoinPaths(active_cluster_stereo_path_, NORMALS_DIR, file_name);

      const auto& fused_pixel_mask = fused_pixel_masks_.at(image_idx + i);
      const mvs::Image& image = images_.at(image_idx + i);
      std::unique_ptr<DepthMap>& depth_map = depth_maps_.at(image_idx + i);
      std::unique_ptr<NormalMap>& normal_map = normal_maps_.at(image_idx + i);
      std::unique_ptr<Bitmap>& semantic_map = semantic_maps_.at(image_idx + i);
      if (!depth_map || (options.with_normal && !normal_map)) {
        continue;
      }

      const Eigen::RowMatrix3f K(image.GetK());
      const Eigen::RowMatrix3f R(image.GetR());
      const Eigen::Vector3f C(image.GetC());
      const Eigen::RowMatrix3f RTKI = R.transpose() * K.inverse();
      const Eigen::Vector3f m_lNormal = R * m_fNormal;

      for (int r = 0; r < semantic_map->Height(); ++r) {
        for (int c = 0; c < semantic_map->Width(); ++c) {
          BitmapColor<uint8_t> semantic;
          semantic_map->GetPixel(c, r, &semantic);
          if (semantic.r != LABLE_GROUND || fused_pixel_mask->Get(r, c)) {
            continue;
          }
          Eigen::Vector3f ray = (RTKI * Eigen::Vector3f(c, r, 1.0f)).normalized();
          float lambda = - (m_fNormal.dot(C) + plane[3]) / (m_fNormal.dot(ray));
          float d = (lambda * R * ray)[2];
          if (d > 0) {
            depth_map->Set(r, c, d);
            if (options.with_normal) {
              normal_map->Set(r, c, 0, m_lNormal[0]);
              normal_map->Set(r, c, 1, m_lNormal[1]);
              normal_map->Set(r, c, 2, m_lNormal[2]);
            }
          }
        }
      }

      depth_map->Write(depth_map_path);
      if (options.with_normal) {
        normal_map->Write(normal_map_path);
      }
      if (!options.cache_depth) {
        depth_map.reset(nullptr);
        normal_map.reset(nullptr);
        semantic_map.reset(nullptr);
      }
      std::cout << StringPrintf("Fill Depth Map %d/%d\n", num_image + i + 1, all_fusion_image_num);
    }
  };

  // Fill Depth Map.
  for (size_t idx = 0; idx < cluster_image_ids.size(); 
       idx += num_perspective_per_image) {
    int num_id = idx + num_pre_fused_images;
    size_t image_idx = cluster_image_ids.at(idx);
    thread_pool_->AddTask(FillDepthMap, image_idx, num_id);
  }
  thread_pool_->Wait();
  std::cout << std::endl;

  const int num_eff_threads = GetEffectiveNumThreads(-1);

  Options fused_options = options;
  fused_options.min_num_visible_images = 
    std::max(options.min_num_visible_images, 3);
  fused_options.min_num_pixels = std::max(options.min_num_pixels, 5);

  std::fill(fused_images_.begin(), fused_images_.end(), 0);
  size_t num_fused_images = 0;
  for (int image_idx = 0; image_idx >= 0;
      image_idx = image_idx = FindNextImage(image_idx)) {
      if (IsStopped()) {
        break;
      }

      if (!used_images_.at(image_idx)) {
        continue;
      }

      Timer timer;
      timer.Start();

      std::cout << StringPrintf("Fusing image(Fitting planes) [%d/%d]", 
                                num_pre_fused_images + num_fused_images + 1, 
                                all_fusion_image_num)
                << std::flush;

      auto &fused_pixel_mask = fused_pixel_masks_.at(image_idx);
      if (!fused_pixel_mask) {
          auto image = images_.at(image_idx);
          fused_pixel_mask.reset(new Mat<bool>(image.GetWidth(), image.GetHeight(), 1));
      }
      fused_pixel_mask->Fill(false);

      const auto semantic_path = JoinPaths(active_dense_path_, SEMANTICS_DIR);
      bool has_semantic = ExistsDir(semantic_path);
      std::unique_ptr<Bitmap>& semantic_map = semantic_maps_.at(image_idx);
      if (has_semantic && !semantic_map) {
        // Load semantic.
        const auto image_name = image_names_[image_idx];
        const auto semantic_name = JoinPaths(semantic_path, image_name);
        const auto semantic_base_name = semantic_name.substr(0, semantic_name.size() - 3);
        if (ExistsFile(semantic_base_name + "png")) {
          semantic_maps_.at(image_idx).reset(new Bitmap);
          semantic_maps_.at(image_idx)->Read(semantic_base_name + "png", false);
        } else if (ExistsFile(semantic_base_name + "jpg")) {
          semantic_maps_.at(image_idx).reset(new Bitmap);
          semantic_maps_.at(image_idx)->Read(semantic_base_name + "jpg", false);
        }
      }

      for(int thread_idx = 0; thread_idx < num_eff_threads; ++thread_idx) {
        thread_pool_->AddTask(&StereoFusion::FuseImage, this, options,
                              thread_idx, image_idx, num_eff_threads, true);
      };
      thread_pool_->Wait();

      num_fused_images += 1;
      fused_images_.at(image_idx) = 255;

      depth_maps_.at(image_idx).reset(nullptr);
      normal_maps_.at(image_idx).reset(nullptr);
      Bitmap &bitmap = images_.at(image_idx).GetBitmap();
      bitmap.Deallocate();
      semantic_maps_.at(image_idx).reset(nullptr);
      fused_pixel_masks_.at(image_idx).reset(nullptr);

      std::cout << StringPrintf(" in %.3fs (%d points)", 
                                timer.ElapsedSeconds(), fused_points_.size())
                << std::endl;
  }

  fused_points_.shrink_to_fit();
  fused_points_visibility_.shrink_to_fit();
  fused_points_score_.shrink_to_fit();
}

void StereoFusion::ComplementPlanes(const Options& options, 
                                    std::vector<int>& cluster_image_ids,
                                    size_t num_pre_fused_points,
                                    size_t num_pre_fused_images,
                                    size_t all_fusion_image_num) {
  size_t i, j;
  std::unordered_map<int, int> unoder_image_ids;
  for (i = 0; i < cluster_image_ids.size(); i++){
    unoder_image_ids.emplace(cluster_image_ids[i], i);
  }
  size_t image_num = images_.size();
  // for (size_t image_idx = 0; image_idx < image_num; ++image_idx) {
  for (auto image_idx : cluster_image_ids) {
    auto& fused_pixel_mask = fused_pixel_masks_[image_idx];
    if (!fused_pixel_mask) {
      fused_pixel_mask.reset(new Mat<bool>(images_.at(image_idx).GetWidth(), 
        images_.at(image_idx).GetHeight(), 1));
    }
    fused_pixel_mask->Fill(false);
  }

  std::vector<Eigen::Vector3d> points;
  std::vector<Eigen::Vector3d> cameras_center;
  std::vector<std::vector<size_t> > image_points_idx(image_num);
  std::unordered_set<uint32_t> unique_cameras_idx;
  for (i = num_pre_fused_points; i < fused_points_.size(); ++i) {
    if (fused_points_[i].s_id != LABLE_GROUND) {
      continue;
    }

    Eigen::Vector3d point;
    point.x() = fused_points_[i].x;
    point.y() = fused_points_[i].y;
    point.z() = fused_points_[i].z;
    points.emplace_back(point);

    Eigen::Map<const Eigen::Vector3f> p((float*)&fused_points_[i]);
    auto points_vis = fused_points_visibility_[i];
    for (auto vis : points_vis) {
      if (unoder_image_ids.find((int)vis) == unoder_image_ids.end()){
        continue;
      }
      unique_cameras_idx.insert(vis);
      const mvs::Image& image = images_.at(vis);
      Eigen::Map<const Eigen::RowMatrix3f> K(image.GetK());
      Eigen::Map<const Eigen::RowMatrix3f> R(image.GetR());
      Eigen::Map<const Eigen::Vector3f> T(image.GetT());
      Eigen::Vector3f proj = K * (R * p + T);
      int u = proj[0] / proj[2];
      int v = proj[1] / proj[2];
      if (u < 0 || u >= image.GetWidth() || v < 0 || v >= image.GetHeight()) {
        continue;
      }
      fused_pixel_masks_[vis]->Set(v, u, true);
      image_points_idx[vis].push_back(i);
    }
  }
  for (auto cam_idx : unique_cameras_idx) {
    const mvs::Image& image = images_.at(cam_idx);
    cameras_center.emplace_back(Eigen::Vector3f(image.GetC()).cast<double>());
  }

  std::cout << "Collect " << points.size() << " candidate points" << std::endl;
  std::cout << "Collect " << cameras_center.size() << " cameras" << std::endl;

  // Fit Main plane.
  Eigen::Vector4d main_plane;
  if (!internal::FitPlane(points, cameras_center, main_plane)) {
    return;
  }

  std::cout << "Primary plane parameter: " << main_plane.transpose() << std::endl;  

  auto FillDepthMap = [&](int image_idx, int num_image) {
    const mvs::Image& image = images_.at(image_idx);

    const auto image_name = image_names_[image_idx];
    const auto file_name = StringPrintf("%s.%s.%s", 
        image_name.c_str(), input_type_.c_str(), DEPTH_EXT);
    const auto depth_map_path = JoinPaths(active_dense_path_, STEREO_DIR, DEPTHS_DIR, file_name);
    const auto normal_map_path = JoinPaths(active_dense_path_, STEREO_DIR, NORMALS_DIR, file_name);

    if (!ExistsFile(depth_map_path) || 
       (options.with_normal && !ExistsFile(normal_map_path))) {
      return;
    }

    if (!depth_maps_.at(image_idx)) {
      auto depth_map_ptr = std::unique_ptr<DepthMap>(new DepthMap);
      depth_map_ptr->Read(depth_map_path);
      
      if (!depth_map_ptr->IsValid()) {
        std::cout << StringPrintf(
                  "WARNING: Ignoring image %s, because input is empty."
                  , image_name.c_str())
                    << std::endl;
        return;
      }
      depth_maps_.at(image_idx).swap(depth_map_ptr);
    }
    std::unique_ptr<DepthMap>& depth_map = depth_maps_.at(image_idx);
    if (!depth_map || !depth_map->IsValid()) {
      return;
    }
    if (options.with_normal) {
      if (!normal_maps_.at(image_idx)) {
        auto normal_map_ptr = std::unique_ptr<NormalMap>(new NormalMap);
        normal_map_ptr->Read(normal_map_path);

        if (!normal_map_ptr->IsValid()) {
          std::cout << StringPrintf(
                    "WARNING: Ignoring image %s, because input is empty."
                    , image_name.c_str())
                      << std::endl;
          return;
        }
        normal_maps_.at(image_idx).swap(normal_map_ptr);
      }
      std::unique_ptr<NormalMap>& normal_map = normal_maps_.at(image_idx);
      if (!normal_map || !normal_map->IsValid()) {
        return;
      }
    }

    std::unique_ptr<Bitmap>& semantic_map = semantic_maps_.at(image_idx);
    const auto semantic_path = JoinPaths(active_dense_path_, SEMANTICS_DIR);
    if (ExistsDir(semantic_path) && !semantic_map) {
      const auto image_name = image_names_[image_idx];
      const auto semantic_name = JoinPaths(semantic_path, image_name);
      const auto semantic_base_name = semantic_name.substr(0, semantic_name.size() - 3);
      if (ExistsFile(semantic_base_name + "png")) {
        semantic_maps_.at(image_idx).reset(new Bitmap);
        semantic_maps_.at(image_idx)->Read(semantic_base_name + "png", false);
      } else if (ExistsFile(semantic_base_name + "jpg")) {
        semantic_maps_.at(image_idx).reset(new Bitmap);
        semantic_maps_.at(image_idx)->Read(semantic_base_name + "jpg", false);
      }
    }

    const double mean_focal_length = (image.GetK()[0] + image.GetK()[4]) * 0.5;
    const double dist_to_best_model = options.dist_to_best_plane_model / mean_focal_length;

    std::vector<int> overlap_images_idx;
    overlap_images_idx.push_back(image_idx);
    for (auto overlap_image_idx : overlapping_images_.at(image_idx)) {
      overlap_images_idx.push_back(overlap_image_idx);
      if (overlap_images_idx.size() >= 9) {
        break;
      }
    }

    // Fit plane.
    std::vector<Eigen::Vector3d> image_points;
    for (auto overlap_image_idx : overlap_images_idx) {
      for (size_t point_idx : image_points_idx[overlap_image_idx]) {
        Eigen::Vector3d point;
        point.x() = fused_points_[point_idx].x;
        point.y() = fused_points_[point_idx].y;
        point.z() = fused_points_[point_idx].z;
        image_points.emplace_back(point);
      }
    }
    Eigen::Vector4d plane;
    if (!internal::FitConsistentPlane(image_points, main_plane, 
                                      dist_to_best_model, 
                                      options.angle_diff_to_main_plane,
                                      options.min_inlier_ratio_to_best_model, 
                                      plane)) {
      return;
    }

    Eigen::Vector3f m_fNormal = plane.head<3>().cast<float>();
    // Eigen::Vector3f m_fNormal = main_plane.head<3>().cast<float>();

    const auto& fused_pixel_mask = fused_pixel_masks_.at(image_idx);

    const Eigen::RowMatrix3f K(image.GetK());
    const Eigen::RowMatrix3f R(image.GetR());
    const Eigen::Vector3f C(image.GetC());
    const Eigen::RowMatrix3f RTKI = R.transpose() * K.inverse();
    const Eigen::Vector3f m_lNormal = R * m_fNormal;

    std::unique_ptr<NormalMap>& normal_map = normal_maps_.at(image_idx);
    for (int r = 0; r < semantic_map->Height(); ++r) {
      for (int c = 0; c < semantic_map->Width(); ++c) {
        BitmapColor<uint8_t> semantic;
        semantic_map->GetPixel(c, r, &semantic);
        if (semantic.r != LABLE_GROUND || fused_pixel_mask->Get(r, c)) {
          continue;
        }
        Eigen::Vector3f ray = (RTKI * Eigen::Vector3f(c, r, 1.0f)).normalized();
        float lambda = - (m_fNormal.dot(C) + plane[3]) / (m_fNormal.dot(ray));
        float d = (lambda * R * ray)[2];
        if (d > 0) {
          depth_map->Set(r, c, d);
          if (options.with_normal) {
            normal_map->Set(r, c, 0, m_lNormal[0]);
            normal_map->Set(r, c, 1, m_lNormal[1]);
            normal_map->Set(r, c, 2, m_lNormal[2]);
          }
        }
      }
    }

    depth_map->Write(depth_map_path);
    if (options.with_normal) {
      normal_map->Write(normal_map_path);
    }
    if (!options.cache_depth) {
      depth_map.reset(nullptr);
      normal_map.reset(nullptr);
      semantic_map.reset(nullptr);
    }
    std::cout << StringPrintf("\rFill Depth Map %d/%d", num_image + 1, all_fusion_image_num);
  };

  const int num_eff_threads = GetEffectiveNumThreads(-1);

  // Fill Depth Map.
  // for (size_t image_idx = 0; image_idx < image_num; ++image_idx) {
  int num_id = num_pre_fused_images;
  for (auto image_idx : cluster_image_ids) {
    thread_pool_->AddTask(FillDepthMap, image_idx, num_id);
    num_id++;
  }
  thread_pool_->Wait();
  std::cout << std::endl;

  Options fused_options = options;
  fused_options.min_num_visible_images = 
    std::max(options.min_num_visible_images, 3);
  fused_options.min_num_pixels = std::max(options.min_num_pixels, 5);

  std::fill(fused_images_.begin(), fused_images_.end(), 0);
  size_t num_fused_images = 0;
  for (int image_idx = 0; image_idx >= 0; image_idx = FindNextImage(image_idx)) {
      if (IsStopped()) {
        break;
      }

      if (!used_images_.at(image_idx)) {
        continue;
      }

      Timer timer;
      timer.Start();

      std::cout << StringPrintf("Fusing image(ground) [%d/%d]", 
                                num_pre_fused_images + num_fused_images + 1,
                                all_fusion_image_num)
                << std::flush;

      auto &fused_pixel_mask = fused_pixel_masks_.at(image_idx);
      if (!fused_pixel_mask) {
          auto image = images_.at(image_idx);
          fused_pixel_mask.reset(new Mat<bool>(image.GetWidth(), image.GetHeight(), 1));
      }
      fused_pixel_mask->Fill(false);

      const auto semantic_path = JoinPaths(active_dense_path_, SEMANTICS_DIR);
      bool has_semantic = ExistsDir(semantic_path);
      std::unique_ptr<Bitmap>& semantic_map = semantic_maps_.at(image_idx);
      if (has_semantic && !semantic_map) {
        // Load semantic.
        const auto image_name = image_names_[image_idx];
        const auto semantic_name = JoinPaths(semantic_path, image_name);
        const auto semantic_base_name = semantic_name.substr(0, semantic_name.size() - 3);
        if (ExistsFile(semantic_base_name + "png")) {
          semantic_maps_.at(image_idx).reset(new Bitmap);
          semantic_maps_.at(image_idx)->Read(semantic_base_name + "png", false);
        } else if (ExistsFile(semantic_base_name + "jpg")) {
          semantic_maps_.at(image_idx).reset(new Bitmap);
          semantic_maps_.at(image_idx)->Read(semantic_base_name + "jpg", false);
        }
      }

      for(int thread_idx = 0; thread_idx < num_eff_threads; ++thread_idx) {
        thread_pool_->AddTask(&StereoFusion::FuseImage, this, options,
                              thread_idx, image_idx, num_eff_threads, true);
      };
      thread_pool_->Wait();

      num_fused_images += 1;
      fused_images_.at(image_idx) = 255;

      depth_maps_.at(image_idx).reset(nullptr);
      normal_maps_.at(image_idx).reset(nullptr);
      Bitmap &bitmap = images_.at(image_idx).GetBitmap();
      bitmap.Deallocate();
      semantic_maps_.at(image_idx).reset(nullptr);
      fused_pixel_masks_.at(image_idx).reset(nullptr);

      std::cout << StringPrintf(" in %.3fs (%d points)", 
                                timer.ElapsedSeconds(), fused_points_.size())
                << std::endl;
  }
  fused_points_.shrink_to_fit();
  fused_points_visibility_.shrink_to_fit();
  fused_points_score_.shrink_to_fit();
}

void StereoFusion::Clear() {
    image_names_.clear();
    image_names_.shrink_to_fit();
    images_.clear();
    images_.shrink_to_fit();
    depth_maps_.clear();
    depth_maps_.shrink_to_fit();
    normal_maps_.clear();
    normal_maps_.shrink_to_fit();
    conf_maps_.clear();
    conf_maps_.shrink_to_fit();
    used_images_.clear();
    used_images_.shrink_to_fit();
    fused_images_.clear();
    fused_images_.shrink_to_fit();
    overlapping_images_.clear();
    overlapping_images_.shrink_to_fit();
    fused_pixel_masks_.clear();
    fused_pixel_masks_.shrink_to_fit();
    P_.clear();
    P_.shrink_to_fit();
    inv_P_.clear();
    inv_P_.shrink_to_fit();
    inv_R_.clear();
    inv_R_.shrink_to_fit();

    fusion_queue_.clear();
    fusion_queue_.shrink_to_fit();

    fused_points_.clear();
    fused_points_.shrink_to_fit();
    fused_points_visibility_.clear();
    fused_points_visibility_.shrink_to_fit();
    fused_points_score_.clear();
    fused_points_score_.shrink_to_fit();
}

void StereoFusion::ConvertVisibility2Dense(
    const std::unordered_map<size_t, image_t> & cluster_image_idx_to_id,
    std::vector<std::vector<uint32_t> >& fused_vis) {
    
    const auto dense_image_id_to_idx = workspace_->GetDenseImageId2Idx();

    for (size_t i = 0; i < fused_vis.size(); i++){
        for (size_t j = 0; j < fused_vis[i].size(); j++){
            size_t image_idx = fused_vis[i][j];
            image_t image_id = cluster_image_idx_to_id.at(image_idx);
            fused_vis[i][j] = dense_image_id_to_idx.at(image_id);
        }
    }
}

void StereoFusion::MergeFusedPly(const std::string& workspace_path){
    const auto& dense_reconstruction_path = JoinPaths(workspace_path, DENSE_DIR);
    std::string output_path = JoinPaths(dense_reconstruction_path, FUSION_NAME);
    if (ExistsFile(output_path)){
      std::cout << "exist file: " << output_path << std::endl;
      return;
    }
    bool has_sem = false;
    std::vector<PlyPoint> merge_points;
    std::vector<PlyPoint> merge_sem_points;
    std::vector<std::vector<uint32_t> > merge_points_vis;
    std::vector<std::vector<float> > merge_points_wgt;

    std::cout << "Merge Reconstruction: " << workspace_path << std::endl;
    if (sfm_cluster_image_idx_to_id_.size() != num_cluster_){
        sfm_cluster_image_idx_to_id_.clear();
        for (int rect_id = 0; ; rect_id++){
            auto reconstruction_path =
                JoinPaths(workspace_path, std::to_string(rect_id));
            if (!ExistsDir(reconstruction_path)){
                break;
            }
            workspace_->SetModel(reconstruction_path);
            const Model model = workspace_->GetModel();
            sfm_cluster_image_idx_to_id_.push_back(model.GetImageIdx2Ids());
        }
    }

    for (int rect_id = 0; ; rect_id++){
        auto reconstruction_path =
            JoinPaths(workspace_path, std::to_string(rect_id));
        if (!ExistsDir(reconstruction_path)){
            break;
        }
        // std::string dense_path = JoinPaths(reconstruction_path.c_str(), DENSE_DIR);
        std::string fused_input_path = JoinPaths(reconstruction_path.c_str(), FUSION_NAME);
        if (ExistsFile(fused_input_path)){
            std::vector<PlyPoint> fused_points = ReadPly(fused_input_path);
            if (ExistsFile(fused_input_path + ".sem")) {
                ReadPointsSemantic(fused_input_path + ".sem", fused_points);
                has_sem = true;
            }
            merge_points.insert(merge_points.end(), fused_points.begin(), fused_points.end());

            std::vector<std::vector<uint32_t> > fused_points_vis;
            ReadPointsVisibility(fused_input_path + ".vis", fused_points_vis);
            ConvertVisibility2Dense(sfm_cluster_image_idx_to_id_[rect_id], fused_points_vis);
            merge_points_vis.insert(merge_points_vis.end(), fused_points_vis.begin(), fused_points_vis.end());

            std::vector<std::vector<float> > fused_points_wgt;
            if (ExistsFile(fused_input_path + ".wgt")){
              ReadPointsWeight(fused_input_path + ".wgt", fused_points_wgt);
              merge_points_wgt.insert(merge_points_wgt.end(), fused_points_wgt.begin(), fused_points_wgt.end());
            }
            std::cout << "=> merge :" << fused_points.size() << "(" << merge_points.size() 
                      << ") in "<< fused_input_path 
                      << " (with sem, wgt: " << std::to_string(has_sem) 
                      << ", " << bool(!fused_points_wgt.empty())<< ")" << std::endl;
        }
    }

    std::cout << "\nSave merge result to " << dense_reconstruction_path << std::endl;
    if (merge_points.size() > 0) {
        WriteBinaryPlyPoints(output_path, merge_points, true, true);
        WritePointsVisibility(output_path + ".vis", merge_points_vis);
        if (merge_points_wgt.size() == merge_points_vis.size()){
            WritePointsWeight(output_path + ".wgt", merge_points_wgt);
        }

        std::cout << "=> save " << FUSION_NAME;
        if (has_sem) {
            WritePointsSemantic(output_path + ".sem", merge_points);
            output_path = JoinPaths(dense_reconstruction_path, FUSION_SEM_NAME);
            WritePointsSemanticColor(output_path, merge_points);
            std::cout << " & " << FUSION_SEM_NAME;
        }
        std::cout << std::endl;
    }
}
}  // namespace mvs
}  // namespace sensemap
