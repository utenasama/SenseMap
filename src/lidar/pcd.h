//Copyright (c) 2019, SenseTime Group.
//All rights reserved.

#ifndef SENSEMAP_UTIL_PCD_H_
#define SENSEMAP_UTIL_PCD_H_

#include <string>
#include <vector>

#include "util/types.h"
#include "util/logging.h"

#include "util/ply.h"

namespace sensemap {

struct PCDInfo{
  float Version = 0.0;

  bool xyz_float = false;
  bool intensity_float = false;

  long int height = -1;
  long int width = -1;
  long long int num_points = -1;

  float min_dist = 2.0;
  float max_dist = 120.0;

  bool is_valid(){
    if (Version > 0 && height > 0 && width > 0 && num_points > 0){
      return true;
    }else {
      return false;
    }
  };

  void init(){
    Version = 0.0;

    xyz_float = false;
    intensity_float = false;

    int height = -1;
    int width = -1;
    int num_points = -1;

    min_dist = 2.0;
    max_dist = 120.0;
  }
};

struct PCDPoint {
  float x = 0.0f;
  float y = 0.0f;
  float z = 0.0f;
  float intensity = 0.0;
  // uint32_t t = 0;
  // uint16_t reflectivity = 0;
  // uint8_t rign = 0;
  // uint16_t ambient = 0;
  // uint32_t range = 0;

  bool is_valid = false;

  float norm(){
    return x * x + y * y + z * z;
  }

  PCDPoint operator = (const PCDPoint& A){
    x = A.x;
    y = A.y;
    z = A.z;
    intensity = A.intensity;
    is_valid = A.is_valid;
    return *this;
  }
};

struct PCDPointCloud {
  PCDInfo info;
  std::vector<std::vector<PCDPoint>> point_cloud;

  void Init(){
    CHECK(info.is_valid());
    point_cloud.resize(info.height, std::vector<PCDPoint>(info.width));
  };

  void Clear(){
    PCDInfo tmp_info;
    info = tmp_info;
    std::vector<std::vector<PCDPoint> > empty_pcd;
    std::swap(point_cloud, empty_pcd);
  };
};

PCDPointCloud ReadPCD(const std::string& path);

void RebuildLivoxMid360(PCDPointCloud& pc, int num_synchronous = 4);

// void WritePcd(const std::string& path, PCDPointCloud pc);

void RemoveClosedPointCloud(PCDPointCloud& cloud, float min_thres, float max_thres);

std::vector<PlyPoint> Convert2Ply(PCDPointCloud& pc);

void ConvertPcd2Ply(PCDPointCloud& pc,
                 std::vector<PlyPoint>& ply_pc, 
                 std::vector<float>& intensities);

}

#endif  // SENSEMAP_UTIL_PCD_H_
