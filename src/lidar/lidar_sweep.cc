//Copyright (c) 2019, SenseTime Group.
//All rights reserved.

#include <fstream>
#include <iomanip>
#include <Eigen/Core>

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/grid_simplify_point_set.h>
#include <CGAL/property_map.h>

#include "util/logging.h"
#include "util/misc.h"
#include "util/bitmap.h"
#include "base/projection.h"

#include "utils.h"
#include "lidar_sweep.h"

namespace CGAL {
typedef CGAL::Exact_predicates_inexact_constructions_kernel Kernel;
typedef Kernel::Point_3 Point;
};

namespace sensemap {

float cloud_curvature[400000];
int cloud_sort_corner_ind[400000];
int cloud_sort_surf_ind[400000];
int cloud_neighbor_picked[400000];
int cloud_label[400000];

void ComputeLidarPointNormal(LidarPointCloud &cloud_in, int id){
    std::vector<Eigen::Vector3d> nearCorners;
    Eigen::Vector3d center(0, 0, 0);
    double radius = std::sqrt(cloud_in.points[id].x * cloud_in.points[id].x + 
                              cloud_in.points[id].y * cloud_in.points[id].y +
                              cloud_in.points[id].z * cloud_in.points[id].z);

    for (int j = -5; j < 6; j++){
        if (abs(cloud_in.points[id+j].x - cloud_in.points[id].x) >= 0.02 * radius * abs(j) + 0.1 ||
            abs(cloud_in.points[id+j].y - cloud_in.points[id].y) >= 0.02 * radius * abs(j) + 0.1 ||
            abs(cloud_in.points[id+j].x - cloud_in.points[id].x) >= 0.02 * radius * abs(j)  + 0.1){
            continue;
        }
        Eigen::Vector3d tmp(cloud_in.points[id+j].x,
                            cloud_in.points[id+j].y,
                            cloud_in.points[id+j].z);
        center = center + tmp;
        nearCorners.push_back(tmp);
    }

    if (nearCorners.size() < 5){
        cloud_in.points[id].nx = 0.f;
        cloud_in.points[id].ny = 0.f;
        cloud_in.points[id].nz = 0.f;
        return;
    }

    center = center / nearCorners.size();
    Eigen::Matrix3d covMat = Eigen::Matrix3d::Zero();
    for (int j = 0; j < nearCorners.size(); j++) {
        Eigen::Matrix<double, 3, 1> tmpZeroMean = nearCorners[j] - center;
        covMat = covMat + tmpZeroMean * tmpZeroMean.transpose();
    }

    Eigen::Vector3d center_point(cloud_in.points[id].x,
                                 cloud_in.points[id].y,
                                 cloud_in.points[id].z);
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(covMat);
    Eigen::Vector3d unit_direction = saes.eigenvectors().col(2);
    // if (saes.eigenvalues()[2] > 3 * saes.eigenvalues()[1]) { 
    if (1) { 
        Eigen::Vector3d normal = Eigen::Vector3d::Zero();
        normal = unit_direction.cross(unit_direction.cross(center_point));
        normal.normalize();
        if (normal.dot(center_point) > 0){
            cloud_in.points[id].nx = -normal(0);
            cloud_in.points[id].ny = -normal(1);
            cloud_in.points[id].nz = -normal(2);
        } else {
            cloud_in.points[id].nx = normal(0);
            cloud_in.points[id].ny = normal(1);
            cloud_in.points[id].nz = normal(2);
        }
    }
};

std::vector<PlyPoint> LidarPointCloud::Convert2Ply(){
  float max_curv = 0.0f;
  for (size_t i = 0; i < points.size(); i++){
    // max_curv = std::max(max_curv, points[i].curv);
    max_curv = std::max(max_curv, points[i].intensity);
  }
  const double max_value = std::log1p(max_curv);
  
  std::vector<PlyPoint> ply_pc(points.size());
  for (size_t i = 0; i < points.size(); i++){
    ply_pc[i].x = points[i].x;
    ply_pc[i].y = points[i].y;
    ply_pc[i].z = points[i].z;
    ply_pc[i].nx = points[i].nx;
    ply_pc[i].ny = points[i].ny;
    ply_pc[i].nz = points[i].nz;

    // const double value = std::log1p(points[i].curv) / max_value;
    const double value = std::log1p(points[i].intensity) / max_value;
    uint8_t r = 255 * JetColormap::Red(value);
    uint8_t g = 255 * JetColormap::Green(value);
    uint8_t b = 255 * JetColormap::Blue(value);

    ply_pc[i].r = r;
    ply_pc[i].g = g;
    ply_pc[i].b = b;
  }
  return ply_pc;
}

void LidarPointCloud::GridSimplifyPointCloud(const LidarPointCloud & pointcloud_in,
                             LidarPointCloud & pointcloud_out,
                             const double cell_size){
  // pointcloud_out = pointcloud_in;
  std::vector<CGAL::Point> cgal_points_in;
  for (auto scan_point : pointcloud_in.points){
      cgal_points_in.push_back(CGAL::Point(double(scan_point.x), (scan_point.y), (scan_point.z)));
  }
  //parameters
  std::vector<std::size_t> indices(cgal_points_in.size());
  for(std::size_t i = 0; i < cgal_points_in.size(); ++i){
    indices[i] = i;
  }
  // simplification by clustering using erase-remove idiom
  std::vector<std::size_t>::iterator end;
  // end = CGAL::grid_simplify_point_set(indices.begin(), indices.end(), &(cgal_points_in[0]),cell_size);
  end = CGAL::grid_simplify_point_set(indices, cell_size, CGAL::parameters::point_map(
                                      CGAL::make_property_map(cgal_points_in)));

  pointcloud_out.points.clear();
  std::size_t k = end - indices.begin();

  indices.resize(k);
  std::sort(indices.begin(), indices.end());

  for(std::size_t i=0; i<k; ++i){
    LidarPoint point_temp;
    point_temp = pointcloud_in.points.at(indices[i]);
    pointcloud_out.points.push_back(point_temp);
  }
}

void LidarPointCloud::TransfromPlyPointCloud(const LidarPointCloud & pointcloud_in,
                            LidarPointCloud & pointcloud_out,
                            const Eigen::Matrix4d& Tr){
  pointcloud_out.points.clear();
  pointcloud_out.points.reserve(pointcloud_in.points.size());

  for (auto pt : pointcloud_in.points){
    Eigen::Vector4d point3d(pt.x, pt.y, pt.z, 1.0f);
    Eigen::Vector4d point3d_t;
    point3d_t = Tr * point3d;

    Eigen::Matrix3d R = Tr.block<3,3>(0,0);
    Eigen::Vector3d normal_in(pt.nx, pt.ny, pt.nz);
    Eigen::Vector3d normal_t = R * normal_in;

    LidarPoint pt_t = pt;
    pt_t.x = point3d_t(0);
    pt_t.y = point3d_t(1);
    pt_t.z = point3d_t(2);
    pt_t.nx = normal_t(0);
    pt_t.ny = normal_t(1);
    pt_t.nz = normal_t(2);

    pointcloud_out.points.push_back(pt_t);
  }
  return;
}

LidarSweep::LidarSweep()
    : sweep_id_(kInvalidLidarSweepId),
      name_(""),
      registered_(false),
      qvec_(1.0, 0.0, 0.0, 0.0),
      tvec_(0.0, 0.0, 0.0),
      qvec_prior_(kNaN, kNaN, kNaN, kNaN),
      tvec_prior_(kNaN, kNaN, kNaN) {
}

LidarSweep::LidarSweep(sweep_t sweep_id, std::string name)
    : sweep_id_(sweep_id),
      name_(name),
      registered_(false),
      qvec_(1.0, 0.0, 0.0, 0.0),
      tvec_(0.0, 0.0, 0.0),
      qvec_prior_(kNaN, kNaN, kNaN, kNaN),
      tvec_prior_(kNaN, kNaN, kNaN) {}

LidarSweep::LidarSweep(sweep_t sweep_id, std::string name, const PCDPointCloud& pc)
    : sweep_id_(sweep_id),
      name_(name),
      registered_(false),
      qvec_(1.0, 0.0, 0.0, 0.0),
      tvec_(0.0, 0.0, 0.0),
      qvec_prior_(kNaN, kNaN, kNaN, kNaN),
      tvec_prior_(kNaN, kNaN, kNaN){
    Setup(pc, 0);
}

void LidarSweep::FeatureExtraction(const PCDPointCloud& cloud){
  std::vector<int> scanStartInd(n_scans, 0);
  std::vector<int> scanEndInd(n_scans, 0);

  LidarPointCloud laserCloud;
  for (int i = 0; i < n_scans; i++) { 
      LidarPointCloud scan_laserCloud;
      scan_laserCloud.points.reserve(cloud.point_cloud.at(i).size());
      for (size_t j = 0; j < cloud.point_cloud.at(i).size(); j++){
        if (cloud.point_cloud[i][j].is_valid){
          LidarPoint temp_pnt;
          temp_pnt.init(cloud.point_cloud[i][j].x,
                        cloud.point_cloud[i][j].y,
                        cloud.point_cloud[i][j].z,
                        cloud.point_cloud[i][j].intensity);
          temp_pnt.scanid = i;
          scan_laserCloud.points.push_back(temp_pnt);
        }
      }
      scan_laserCloud.points.shrink_to_fit();

      scanStartInd[i] = laserCloud.points.size() + 5;
      laserCloud.points.insert(laserCloud.points.end(), 
                                scan_laserCloud.points.begin(),
                                scan_laserCloud.points.end());
      scanEndInd[i] = laserCloud.points.size() - 6;
    //   std::cout << "laser cloud - " << i << ": " << scanStartInd[i] << "/" << scanEndInd[i] << std::endl;
  }

  for (int i = 5; i < laserCloud.points.size() - 5; i++){ 
      float diffX = laserCloud.points[i - 5].x + laserCloud.points[i - 4].x 
                  + laserCloud.points[i - 3].x + laserCloud.points[i - 2].x 
                  + laserCloud.points[i - 1].x - 10 * laserCloud.points[i].x 
                  + laserCloud.points[i + 1].x + laserCloud.points[i + 2].x 
                  + laserCloud.points[i + 3].x + laserCloud.points[i + 4].x 
                  + laserCloud.points[i + 5].x;
      float diffY = laserCloud.points[i - 5].y + laserCloud.points[i - 4].y 
                  + laserCloud.points[i - 3].y + laserCloud.points[i - 2].y 
                  + laserCloud.points[i - 1].y - 10 * laserCloud.points[i].y 
                  + laserCloud.points[i + 1].y + laserCloud.points[i + 2].y 
                  + laserCloud.points[i + 3].y + laserCloud.points[i + 4].y 
                  + laserCloud.points[i + 5].y;
      float diffZ = laserCloud.points[i - 5].z + laserCloud.points[i - 4].z 
                  + laserCloud.points[i - 3].z + laserCloud.points[i - 2].z 
                  + laserCloud.points[i - 1].z - 10 * laserCloud.points[i].z 
                  + laserCloud.points[i + 1].z + laserCloud.points[i + 2].z 
                  + laserCloud.points[i + 3].z + laserCloud.points[i + 4].z 
                  + laserCloud.points[i + 5].z;

      float radius2 = laserCloud.points[i].x * laserCloud.points[i].x +
                      laserCloud.points[i].y * laserCloud.points[i].y +
                      laserCloud.points[i].z * laserCloud.points[i].z;

      // cloud_curvature[i] = (diffX * diffX + diffY * diffY + diffZ * diffZ);
      cloud_curvature[i] = (diffX * diffX + diffY * diffY + diffZ * diffZ) / radius2;
      laserCloud.points[i].curv = cloud_curvature[i];
      cloud_sort_corner_ind[i] = i;
      cloud_sort_surf_ind[i] = i;
      cloud_neighbor_picked[i] = 0;
      cloud_label[i] = 0;
  }

  float t_q_sort = 0;   
  LidarPointCloud surf_points_less_flat_temp;
  LidarPointCloud corner_points_sharp_temp;
  LidarPointCloud surf_points_flat_temp;

  for (int i = 0; i < n_scans; i++){
    if( scanEndInd[i] - scanStartInd[i] < 6)
        continue;
    LidarPointCloud surf_points_less_flat_Scan;
    // const int num_parts = 6;
    const int num_corner_parts = 6;
    for (int j = 0; j < num_corner_parts; j++){
        int sp = scanStartInd[i] + (scanEndInd[i] - scanStartInd[i]) 
                  * j/num_corner_parts; 
        int ep = scanStartInd[i] + (scanEndInd[i] - scanStartInd[i]) 
                  * (j + 1) / num_corner_parts - 1;
        
        auto cmp = [](int a, int b) { 
                        return (cloud_curvature[a]<cloud_curvature[b]); };
        std::sort (cloud_sort_corner_ind + sp, 
                    cloud_sort_corner_ind + ep + 1, cmp);

        int largestPickedNum = 0;
        for (int k = ep; k >= sp; k--)
        {
            int ind = cloud_sort_corner_ind[k]; 

            if (cloud_neighbor_picked[ind] == 0 &&
                cloud_curvature[ind] > 0.1){
                largestPickedNum++;
                ComputeLidarPointNormal(laserCloud, ind);
                if (largestPickedNum <= 2/* && cloud_curvature[ind] > 1.0*/)
                {                        
                    cloud_label[ind] = 2;
                    corner_points_sharp_temp.points.push_back(
                                                    laserCloud.points[ind]);
                    corner_points_less_sharp_.points.push_back(
                                                    laserCloud.points[ind]);
                }
                else if (largestPickedNum <= 20)
                {                        
                    cloud_label[ind] = 1; 
                    corner_points_less_sharp_.points.push_back(
                                                    laserCloud.points[ind]);
                }
                else
                {
                    break;
                }

                cloud_neighbor_picked[ind] = 1; 

                for (int l = 1; l <= 5; l++)
                {
                    float diffX = laserCloud.points[ind + l].x 
                                  - laserCloud.points[ind + l - 1].x;
                    float diffY = laserCloud.points[ind + l].y 
                                  - laserCloud.points[ind + l - 1].y;
                    float diffZ = laserCloud.points[ind + l].z 
                                  - laserCloud.points[ind + l - 1].z;
                    if(diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
                    {
                        break;
                    }

                    cloud_neighbor_picked[ind + l] = 1;
                }
                for (int l = -1; l >= -5; l--)
                {
                    float diffX = laserCloud.points[ind + l].x 
                                  - laserCloud.points[ind + l + 1].x;
                    float diffY = laserCloud.points[ind + l].y 
                                  - laserCloud.points[ind + l + 1].y;
                    float diffZ = laserCloud.points[ind + l].z 
                                  - laserCloud.points[ind + l + 1].z;
                    if(diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
                    {
                        break;
                    }

                    cloud_neighbor_picked[ind + l] = 1;
                }
            }
        }
    }

    const int num_surf_parts = 16;
    for (int j = 0; j < num_surf_parts; j++)
    {
        int sp = scanStartInd[i] + (scanEndInd[i] - scanStartInd[i]) 
                  * j/num_surf_parts; 
        int ep = scanStartInd[i] + (scanEndInd[i] - scanStartInd[i]) 
                  * (j + 1) / num_surf_parts - 1;
        
        auto cmp = [](int a, int b) { 
                        return (cloud_curvature[a]<cloud_curvature[b]); };
        std::sort (cloud_sort_surf_ind + sp, 
                    cloud_sort_surf_ind + ep + 1, cmp);

        int smallestPickedNum = 0;
        for (int k = sp; k <= ep; k++)
        {
            int ind = cloud_sort_surf_ind[k];

            if (cloud_neighbor_picked[ind] == 0 &&
                cloud_curvature[ind] < 0.02)
            {

                cloud_label[ind] = -1; 
                // surf_points_flat_.points.push_back(laserCloud.points[ind]);
                ComputeLidarPointNormal(laserCloud, ind);
                surf_points_flat_temp.points.push_back(laserCloud.points[ind]);

                smallestPickedNum++;
                if (smallestPickedNum >= 4)
                { 
                    break;
                }

                cloud_neighbor_picked[ind] = 1;
                for (int l = 1; l <= 5; l++)
                { 
                    float diffX = laserCloud.points[ind + l].x 
                                  - laserCloud.points[ind + l - 1].x;
                    float diffY = laserCloud.points[ind + l].y 
                                  - laserCloud.points[ind + l - 1].y;
                    float diffZ = laserCloud.points[ind + l].z 
                                  - laserCloud.points[ind + l - 1].z;
                    if(diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
                    {
                        break;
                    }

                    cloud_neighbor_picked[ind + l] = 1;
                }
                for (int l = -1; l >= -5; l--)
                {
                    float diffX = laserCloud.points[ind + l].x 
                                  - laserCloud.points[ind + l + 1].x;
                    float diffY = laserCloud.points[ind + l].y 
                                  - laserCloud.points[ind + l + 1].y;
                    float diffZ = laserCloud.points[ind + l].z 
                                  - laserCloud.points[ind + l + 1].z;
                    if(diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
                    {
                        break;
                    }

                    cloud_neighbor_picked[ind + l] = 1;
                }
            }
        }

        for (int k = sp; k <= ep; k++)
        {
            int ind = cloud_sort_surf_ind[k];
            if (cloud_label[k] <= 0 && cloud_curvature[ind] < 0.1)
            {
                ComputeLidarPointNormal(laserCloud, k);
                surf_points_less_flat_Scan.points.push_back(
                                                    laserCloud.points[k]);
            }
        }
    }

    surf_points_less_flat_temp.points.insert(
        surf_points_less_flat_temp.points.end(),
        surf_points_less_flat_Scan.points.begin(),
        surf_points_less_flat_Scan.points.end());
  }

  LidarPointCloud::GridSimplifyPointCloud(
    surf_points_less_flat_temp, surf_points_less_flat_,0.2);
  // surf_points_less_flat_ = surf_points_less_flat_temp;

  surf_points_flat_ = surf_points_flat_temp;
  corner_points_sharp_ = corner_points_sharp_temp;
}

void LidarSweep::Setup(const PCDPointCloud& pc, uint64_t lifetime){

  create_time_ = lifetime;
  
  n_scans = pc.info.height;
//   std::vector<LidarPointCloud> laser_scans(n_scans);

//   RemoveClosedPointCloud(pc, min_range, max_range);
  FeatureExtraction(pc);
  // OutputCornerPly("/home/SENSETIME/zhangzhuang/data/wuqiong-test2/sfm-workspace-rig1/lidar_feature/" + GetPathBaseName(Name()) + "/");

  for (auto & point : corner_points_sharp_.points) {
    point.lifetime = lifetime;
  }
  for (auto & point : corner_points_less_sharp_.points) {
    point.lifetime = lifetime;
  }
  for (auto & point : surf_points_flat_.points) {
    point.lifetime = lifetime;
  }
  for (auto & point : surf_points_less_flat_.points) {
    point.lifetime = lifetime;
  }

  printf("corner points size: %d \n", (int)corner_points_sharp_.points.size());
  printf("corner less points size: %d \n", (int)corner_points_less_sharp_.points.size());
  printf("plane points size: %d \n", (int)surf_points_flat_.points.size());
  printf("plane less points size: %d \n", (int)surf_points_less_flat_.points.size());
};

void LidarSweep::OutputCornerPly(const std::string path){
  std::cout << "Output surf & corner points to " << path << std::endl;
  if (!ExistsPath(path)) {
    boost::filesystem::create_directories(path);
  }

  // size_t num_pnt = surf_points_less_flat_.points.size();
  // int max_scan = -1;
  // int id = -1;
  // for(int i = 0 ; i<num_pnt; i++){
  //   auto pnt = surf_points_less_flat_.points.at(i);

  //   if (pnt.scanid > max_scan){
  //     std::cout << (int)pnt.scanid << ", " << max_scan << " / " << id << ", " << i << std::endl;
  //     max_scan = pnt.scanid;
  //     id = i;
  //   }

  //   if (pnt.scanid < max_scan){
  //     std::cout << "!!!" << (int)pnt.scanid << ", " << max_scan << " / " << id << ", " << i << std::endl;
  //   }
  // }

  // exit(-1);

  std::vector<PlyPoint> corner_pc = corner_points_sharp_.Convert2Ply();
  WriteBinaryPlyPoints(path+"/corner_sharp.ply", corner_pc, false, true);

  std::vector<PlyPoint> corner_l_pc = corner_points_less_sharp_.Convert2Ply();
  WriteBinaryPlyPoints(path+"/corner_less_sharp.ply", corner_l_pc, false, true);

  std::vector<PlyPoint> surf_pc = surf_points_flat_.Convert2Ply();
  WriteBinaryPlyPoints(path+"/corner_surf.ply", surf_pc, false, true);

  std::vector<PlyPoint> surf_l_pc = surf_points_less_flat_.Convert2Ply();
  WriteBinaryPlyPoints(path+"/corner_less_surf.ply", surf_l_pc, false, true);

  std::cout << "Output surf & corner points Done." << std::endl;
  return;
}

void LidarSweep::UpdateNeighbor(const sweep_t sweep_id2, 
                                const Eigen::Matrix4d rel_pose){
    auto it = relative_pose_.find(sweep_id2); 
    if(it != relative_pose_.end()) { 
        // relative_pose_[sweep_id2] = rel_pose; 
    } else { 
        relative_pose_.insert(
                    std::pair<sweep_t, Eigen::Matrix4d>{sweep_id2, rel_pose});
    }

    return ;
}

Eigen::Matrix3x4d LidarSweep::ProjectionMatrix() const {
   return ComposeProjectionMatrix(qvec_, tvec_);
}

Eigen::Matrix3x4d LidarSweep::InverseProjectionMatrix() const {
   return InvertProjectionMatrix(ComposeProjectionMatrix(qvec_, tvec_));
}

Eigen::Matrix3d LidarSweep::RotationMatrix() const {
   return QuaternionToRotationMatrix(qvec_);
}

Eigen::Vector3d LidarSweep::ProjectionCenter() const {
   return ProjectionCenterFromPose(qvec_, tvec_);
}

Eigen::Vector3d LidarSweep::ViewingDirection() const {
   return RotationMatrix().row(2);
}

void LidarSweep::FilterPointCloud(const double intensity_threshold) {
    // sharp corners.
    std::vector<LidarPoint> new_corner_points_sharp;
    new_corner_points_sharp.reserve(corner_points_sharp_.points.size());
    for (const auto & point : corner_points_sharp_.points) {
        double exp_weight = 1.0 - std::exp(-point.intensity * point.intensity / 100);
        if (exp_weight >= intensity_threshold) {
            new_corner_points_sharp.push_back(point);
        }
    }
    new_corner_points_sharp.shrink_to_fit();
    std::cout << "shrink lidar corner shapen points from " << corner_points_sharp_.points.size() << " to " << new_corner_points_sharp.size() << std::endl;
    std::swap(corner_points_sharp_.points, new_corner_points_sharp);

    // less sharp corners.
    std::vector<LidarPoint> new_corner_points_less_sharp;
    new_corner_points_less_sharp.reserve(corner_points_less_sharp_.points.size());
    for (const auto & point : corner_points_less_sharp_.points) {
        double exp_weight = 1.0 - std::exp(-point.intensity * point.intensity / 100);
        if (exp_weight >= intensity_threshold) {
            new_corner_points_less_sharp.push_back(point);
        }
    }
    new_corner_points_less_sharp.shrink_to_fit();
    std::cout << "shrink lidar corner shapen points from " << corner_points_less_sharp_.points.size() << " to " << new_corner_points_less_sharp.size() << std::endl;
    std::swap(corner_points_less_sharp_.points, new_corner_points_less_sharp);

    // flat surfs
    std::vector<LidarPoint> new_surf_points_flat;
    new_surf_points_flat.reserve(surf_points_flat_.points.size());
    for (const auto & point : surf_points_flat_.points) {
        double exp_weight = 1.0 - std::exp(-point.intensity * point.intensity / 100);
        if (exp_weight >= intensity_threshold) {
            new_surf_points_flat.push_back(point);
        }
    }
    new_surf_points_flat.shrink_to_fit();
    std::cout << "shrink lidar surf flat points from " << surf_points_flat_.points.size() << " to " << new_surf_points_flat.size() << std::endl;
    std::swap(surf_points_flat_.points, new_surf_points_flat);

    // less flat surfs
    std::vector<LidarPoint> new_surf_points_less_flat;
    new_surf_points_less_flat.reserve(surf_points_less_flat_.points.size());
    for (const auto & point : surf_points_less_flat_.points) {
        double exp_weight = 1.0 - std::exp(-point.intensity * point.intensity / 100);
        if (exp_weight >= intensity_threshold) {
            new_surf_points_less_flat.push_back(point);
        }
    }
    new_surf_points_less_flat.shrink_to_fit();
    std::cout << "shrink lidar surf flat points from " << surf_points_less_flat_.points.size() << " to " << new_surf_points_flat.size() << std::endl;
    std::swap(surf_points_less_flat_.points, new_surf_points_less_flat);
}

}  // namespace sensemap
