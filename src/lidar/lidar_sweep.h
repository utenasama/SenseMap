//Copyright (c) 2023, SenseTime Group.
//All rights reserved.

#ifndef SENSEMAP_UTIL_LIDAR_SWEEP_H_
#define SENSEMAP_UTIL_LIDAR_SWEEP_H_

#include <string>
#include <vector>
#include <unordered_map>

#include <Eigen/Core>
#include <Eigen/Dense>

#include "base/pose.h"
#include "util/math.h"
#include "util/types.h"
#include "util/logging.h"
#include "util/ply.h"

#include "pcd.h"

namespace sensemap {
struct LidarPoint{
  float x = 0.0f;
  float y = 0.0f;
  float z = 0.0f;
  float nx = 0.0f;
  float ny = 0.0f;
  float nz = 0.0f;
  float curv = 0.0f;
  float intensity = 0.0;
  uint64_t lifetime = 0.0;
  int8_t scanid = -1;

  void init(float pnt_x, float pnt_y, float pnt_z, float pnt_intensity){
    x = pnt_x;
    y = pnt_y;
    z = pnt_z;
    intensity = pnt_intensity;
  };
};

struct LidarPointCloud{
  std::vector<LidarPoint> points;

  // Must return the number of data points
  inline size_t kdtree_get_point_count() const { return points.size(); }
  
  // Returns the dim'th component of the idx'th point in the class:
  // Since this is inlined and the "dim" argument is typically an immediate 
  // value, the "if/else's" are actually solved at compile time.
  inline float kdtree_get_pt(const size_t idx, const size_t dim) const
  {
      if (dim == 0) return points[idx].x;
      else if (dim == 1) return points[idx].y;
      else return points[idx].z;
  }
  
  // Optional bounding-box computation: return false to default to a standard 
  // bbox computation loop.  Return true if the BBOX was already computed by 
  // the class and returned in "bb" so it can be avoided to redo it again. 
  // Look at bb.size() to find out the expected dimensionality (e.g. 2 or 3 
  // for point clouds)
  template <class BBOX>
  bool kdtree_get_bbox(BBOX& /* bb */) const { return false; }
  
  LidarPointCloud operator+(const LidarPointCloud &others)
	{
    LidarPointCloud pc;
    pc.points.insert(this->points.end(), others.points.begin(), 
                     others.points.end());
		return pc;
	}

  LidarPointCloud operator+=(const LidarPointCloud &others)
	{
    this->points.insert(this->points.end(), others.points.begin(), 
                        others.points.end());
		return *this;
	}

	std::vector<PlyPoint> Convert2Ply();

  static void GridSimplifyPointCloud(const LidarPointCloud & pointcloud_in,
                                      LidarPointCloud & pointcloud_out,
                                      const double cell_size);
  static void TransfromPlyPointCloud(const LidarPointCloud & pointcloud_in,
                                      LidarPointCloud & pointcloud_out,
                                      const Eigen::Matrix4d& Tr);
};




class LidarSweep {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    LidarSweep();

    LidarSweep(sweep_t sweep_id, std::string name);

    LidarSweep(sweep_t sweep_id, std::string name, const PCDPointCloud& pc);

    // lidar point 
    inline void SetName(const std::string & name);
    inline std::string Name();
    inline const std::string Name() const;
    inline sweep_t SweepID() const;

    inline bool IsRegistered() const;
    inline void SetRegistered(const bool registered);

    // Access quaternion vector as (qw, qx, qy, qz) specifying the rotation of the
    // pose which is defined as the transformation from world to lidar space.
    inline const Eigen::Vector4d& Qvec() const;
    inline Eigen::Vector4d& Qvec();
    inline double Qvec(const size_t idx) const;
    inline double& Qvec(const size_t idx);
    inline void SetQvec(const Eigen::Vector4d& qvec);
    inline void NormalizeQvec();

    // Quaternion prior, e.g. given by EXIF gyroscope tag.
    inline const Eigen::Vector4d& QvecPrior() const;
    inline Eigen::Vector4d& QvecPrior();
    inline double QvecPrior(const size_t idx) const;
    inline double& QvecPrior(const size_t idx);
    inline bool HasQvecPrior() const;
    inline void SetQvecPrior(const Eigen::Vector4d& qvec);

    // Access quaternion vector as (tx, ty, tz) specifying the translation of the
    // pose which is defined as the transformation from world to lidar space.
    inline const Eigen::Vector3d& Tvec() const;
    inline Eigen::Vector3d& Tvec();
    inline double Tvec(const size_t idx) const;
    inline double& Tvec(const size_t idx);
    inline void SetTvec(const Eigen::Vector3d& tvec);

    // Quaternion prior, e.g. given by EXIF GPS tag.
    inline const Eigen::Vector3d& TvecPrior() const;
    inline Eigen::Vector3d& TvecPrior();
    inline double TvecPrior(const size_t idx) const;
    inline double& TvecPrior(const size_t idx);
    inline bool HasTvecPrior() const;
    inline void SetTvecPrior(const Eigen::Vector3d& tvec);

    void Setup(const PCDPointCloud& pc, uint64_t lifetime);

    // Get edge/plane features
    inline const LidarPointCloud& GetCornerPointsSharp() const;
    inline const LidarPointCloud& GetCornerPointsLessSharp() const;
    inline const LidarPointCloud& GetSurfPointsFlat() const;
    inline const LidarPointCloud& GetSurfPointsLessFlat() const;

    void OutputCornerPly(const std::string path);

    // neighbors id and relative poses of camera
    void UpdateNeighbor(const sweep_t sweep_id2, const Eigen::Matrix4d rel_pose);
    inline bool ExistsNeighbor(const sweep_t sweep_id2) const;
    inline const EIGEN_STL_UMAP(sweep_t, Eigen::Matrix4d)& GetNeighbors() const;

    // Compose the projection matrix from world to image space.
    Eigen::Matrix3x4d ProjectionMatrix() const;

    // Compose the inverse projection matrix from image to world space
    Eigen::Matrix3x4d InverseProjectionMatrix() const;

    // Compose rotation matrix from quaternion vector.
    Eigen::Matrix3d RotationMatrix() const;

    Eigen::Vector3d ProjectionCenter() const;

    Eigen::Vector3d ViewingDirection() const;

    void FilterPointCloud(const double intensity_threshold = 0.5);

    uint64_t timestamp_ = 0;
    uint64_t create_time_;

private:
    void FeatureExtraction(const PCDPointCloud& cloud);

private:

    // Identifier of the lidar scan.
    sweep_t sweep_id_;

    // The name of the lidar scan, i.e. the relative path.
    std::string name_;

    // Whether the lidar sweep is successfully registered in the reconstruction.
    bool registered_;

    // The pose of the lidar, defined as the transformation from world to lidar.
    Eigen::Vector4d qvec_;
    Eigen::Vector3d tvec_;

    // The pose prior of the lidar, e.g. extracted from lidar-slam.
    Eigen::Vector4d qvec_prior_;
    Eigen::Vector3d tvec_prior_;

    // // number of lidar scans (16,32,64...)
    int16_t n_scans = -1;
    // int16_t n_scans = 32;

    //Segmentaion param
    int16_t n_horizons = 1800;
    int horizontal_scan_index = 6;
    float ang_res_x = 0.2;
    float ang_res_y = 2.0;
    float ang_bottom = 24.8+0.2;
    int groundScanInd = 50;
    // int N_SCAN = 64;
    // int Horizon_SCAN = 1800; //1028~4500
    // float ang_res_x = 360.0/float(Horizon_SCAN);
    // float ang_res_y = 28.0/float(N_SCAN-1);
    // float ang_bottom = 25.0;
    // int groundScanInd = 60;`

    float sensorMountAngle = 0.0;
    float segmentTheta = 60.0/180.0*M_PI; // decrese this value may improve accuracy
    int segmentValidPointNum = 5;
    int segmentValidLineNum = 3;
    float segmentAlphaX = ang_res_x / 180.0 * M_PI;
    float segmentAlphaY = ang_res_y / 180.0 * M_PI;
    
    // lidar frequence
    double scan_period = 0.1;

    // filter the closed point cloud
    double min_range = 1.0; 
    double max_range = 150.0;

    // feature point extrated by curvature
    LidarPointCloud corner_points_sharp_;
    LidarPointCloud corner_points_less_sharp_;
    LidarPointCloud surf_points_flat_;
    LidarPointCloud surf_points_less_flat_;

    // PlyPointCloud horizontal_scan_;

    // neighbors id and relative poses
    EIGEN_STL_UMAP(sweep_t, Eigen::Matrix4d) relative_pose_;
};


////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////

void LidarSweep::SetName(const std::string & image_name){
    name_ = image_name.substr(0,image_name.find(".")) + ".bin";
}

std::string LidarSweep::Name() {return name_;}

const std::string LidarSweep::Name() const { return name_; }

sweep_t LidarSweep::SweepID() const {return sweep_id_;}

bool LidarSweep::IsRegistered() const { return registered_; }

void LidarSweep::SetRegistered(const bool registered) { registered_ = registered; }

const Eigen::Vector4d& LidarSweep::Qvec() const { return qvec_; }

Eigen::Vector4d& LidarSweep::Qvec() { return qvec_; }

inline double LidarSweep::Qvec(const size_t idx) const { return qvec_(idx); }

inline double& LidarSweep::Qvec(const size_t idx) { return qvec_(idx); }

void LidarSweep::SetQvec(const Eigen::Vector4d& qvec) { qvec_ = qvec; }

void LidarSweep::NormalizeQvec() { qvec_ = NormalizeQuaternion(qvec_); }

const Eigen::Vector4d& LidarSweep::QvecPrior() const { return qvec_prior_; }

Eigen::Vector4d& LidarSweep::QvecPrior() { return qvec_prior_; }

inline double LidarSweep::QvecPrior(const size_t idx) const {
    return qvec_prior_(idx);
}

inline double& LidarSweep::QvecPrior(const size_t idx) { return qvec_prior_(idx); }

inline bool LidarSweep::HasQvecPrior() const { return !IsNaN(qvec_prior_.sum()); }

void LidarSweep::SetQvecPrior(const Eigen::Vector4d& qvec) { qvec_prior_ = qvec; }

const Eigen::Vector3d& LidarSweep::Tvec() const { return tvec_; }

Eigen::Vector3d& LidarSweep::Tvec() { return tvec_; }

inline double LidarSweep::Tvec(const size_t idx) const { return tvec_(idx); }

inline double& LidarSweep::Tvec(const size_t idx) { return tvec_(idx); }

void LidarSweep::SetTvec(const Eigen::Vector3d& tvec) { tvec_ = tvec; }

const Eigen::Vector3d& LidarSweep::TvecPrior() const { return tvec_prior_; }

Eigen::Vector3d& LidarSweep::TvecPrior() { return tvec_prior_; }

inline double LidarSweep::TvecPrior(const size_t idx) const {
    return tvec_prior_(idx);
}

inline double& LidarSweep::TvecPrior(const size_t idx) { return tvec_prior_(idx); }

inline bool LidarSweep::HasTvecPrior() const { return !IsNaN(tvec_prior_.sum()); }

void LidarSweep::SetTvecPrior(const Eigen::Vector3d& tvec) { tvec_prior_ = tvec; }

const LidarPointCloud & LidarSweep::GetCornerPointsSharp() const {
    return corner_points_sharp_;
}

const LidarPointCloud & LidarSweep::GetCornerPointsLessSharp() const {
    return corner_points_less_sharp_;
}

const LidarPointCloud & LidarSweep::GetSurfPointsFlat() const {
    return surf_points_flat_;
}

const LidarPointCloud & LidarSweep::GetSurfPointsLessFlat() const {
    return surf_points_less_flat_;
}

bool LidarSweep::ExistsNeighbor(const sweep_t sweep_id) const { 
    return relative_pose_.find(sweep_id) != relative_pose_.end(); 
}

const EIGEN_STL_UMAP(sweep_t, Eigen::Matrix4d)& LidarSweep::GetNeighbors() const {
    return relative_pose_;
}

}

#endif  // SENSEMAP_UTIL_LIDAR_SWEEP_H_
