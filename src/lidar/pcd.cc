//Copyright (c) 2019, SenseTime Group.
//All rights reserved.

#include "pcd.h"

#include <fstream>
#include <iomanip>
#include <Eigen/Core>
#include <unordered_map>

#include "util/bitmap.h"
#include "util/logging.h"
#include "util/misc.h"

namespace sensemap {
namespace {
void ColorMap(const float intensity, uint8_t &r, uint8_t &g, uint8_t &b){
    if(intensity <= 0.25) {
        r = 0;
        g = 0;
        b = (intensity) * 4 * 255;
        return;
    }

    if(intensity <= 0.5) {
        r = 0;
        g = (intensity - 0.25) * 4 * 255;
        b = 255;
        return;
    }

    if(intensity <= 0.75) {
        r = 0;
        g = 255;
        b = (0.75 - intensity) * 4 * 255;
        return;
    }

    if(intensity <= 1.0) {
        r = (intensity - 0.75) * 4 * 255;
        g = (1.0 - intensity) * 4 * 255;
        b = 0;
        return;
    }
};
}

PCDPointCloud ReadPCD(const std::string& path){
  bool is_binary = false;
  std::vector<int> pc_size;
  std::vector<int> pc_type;
  std::vector<int> pc_cont;

  std::ifstream file(path, std::ios::binary);
  CHECK(file.is_open()) << path;

  PCDPointCloud pc;
  std::string line;

  int index = 0;
  while (std::getline(file, line)) {
    StringTrim(&line);

    if (line.empty()) {
      continue;
    }

    if (line.substr(0,1) == "#"){
      continue;
    }

    if (line == "end_header") {
      break;
    }

    const std::vector<std::string> line_elems = StringSplit(line, " ");

    if (line_elems.size() >= 2 && line_elems[0] == "VERSION") {
      pc.info.Version = std::stof(line_elems[1]);
      // std::cout << "pc.info.Version" << pc.info.Version << std::endl;
      continue;
    }

    if (line_elems.size() >= 2 && line_elems[0] == "FIELDS") {
      pc_size.resize(line_elems.size() - 1);
      pc_type.resize(line_elems.size() - 1);
      pc_cont.resize(line_elems.size() - 1);
      continue;
    }

    if (line_elems.size() >= 2 && line_elems[0] == "SIZE") {
      if (line_elems.size() != pc_size.size() + 1){
        std::cout << "error FIELDS & SIZE is different..." << std::endl;
        exit(-1);
      }
      for (int i = 1; i < line_elems.size(); i++){
        pc_size.at(i-1) = std::stoi(line_elems[i]);
      }
      continue;
    }

    if (line_elems.size() >= 2 && line_elems[0] == "TYPE") {
      if (line_elems.size() != pc_type.size() + 1){
        std::cout << "error FIELDS & TYPE is different..." << std::endl;
        exit(-1);
      }
      for (int i = 1; i < line_elems.size(); i++){
        if (line_elems[i] == "F"){
          pc_type.at(i-1) = 1;
        } else if(line_elems[i] == "U"){
          pc_type.at(i-1) = 0;
        }
      }
      continue;
    }

    if (line_elems.size() >= 2 && line_elems[0] == "COUNT") {
      if (line_elems.size() != pc_size.size() + 1){
        std::cout << "error FIELDS & SIZE is different..." << std::endl;
        exit(-1);
      }
      for (int i = 1; i < line_elems.size(); i++){
        pc_cont.at(i-1) = std::stoi(line_elems[i]);
      }
      continue;
    }

    if (line_elems.size() >= 2 && line_elems[0] == "WIDTH") {
      pc.info.width = std::stol(line_elems[1]);
    }

    if (line_elems.size() >= 2 && line_elems[0] == "HEIGHT") {
      pc.info.height = std::stol(line_elems[1]);
    }

    if (line_elems.size() >= 2 && line_elems[0] == "VIEWPOINT") {
      continue;
    }

    if (line_elems.size() >= 2 && line_elems[0] == "POINTS") {
      pc.info.num_points = std::stoll(line_elems[1]);
      continue;
    }

    if (line_elems.size() >= 2 && line_elems[0] == "DATA") {
      if (line_elems[1].compare("binary") == 0){
        is_binary = true;
      } else if(line_elems[1].compare("ascii") == 0){
        std::cout << "error: data ascii" << std::endl;
        exit(-1);
      } else if (line_elems[1].compare("binary_compressed") == 0){
        std::cout << "error: data binary_compressed" << std::endl;
        exit(-1);
      }
      // continue;
    }

    if (is_binary){
      pc.Init();

      size_t num_bytes_per_line = 0;
      for (int i = 0; i < pc_type.size(); i++){
        num_bytes_per_line += pc_size[i] * pc_cont[i];
      }
      // std::cout << "num_bytes_per_line: " << num_bytes_per_line << std::endl;

      size_t valid_pnts = 0;
      size_t read_pnts = 0;
      std::vector<char> buffer(num_bytes_per_line);
      for (size_t i = 0; i < pc.info.num_points; i++){
        PCDPoint pnt;
        file.read(buffer.data(), num_bytes_per_line);
        // std::cout << std::string(buffer.begin(), buffer.end()) << std::endl;
        // exit(-1);
        pnt.x = LittleEndianToNative(
            *reinterpret_cast<float*>(&buffer[0]));
        pnt.y = LittleEndianToNative(
            *reinterpret_cast<float*>(&buffer[4]));
        pnt.z = LittleEndianToNative(
            *reinterpret_cast<float*>(&buffer[8]));
        pnt.intensity = LittleEndianToNative(
            *reinterpret_cast<float*>(&buffer[12]));
        
        float dist_2 = pnt.x * pnt.x +pnt.y * pnt.y + pnt.z * pnt.z;
        if (dist_2 < pc.info.min_dist * pc.info.min_dist || 
          dist_2 > pc.info.max_dist * pc.info.max_dist){
          pnt.is_valid = false;
        } else {
          pnt.is_valid = true;
        }

        long int pnt_width = i % pc.info.width;
        long int pnt_height = i / pc.info.width;
        pc.point_cloud[pnt_height][pnt_width] = pnt;
        read_pnts++;
      }

      // std::cout << "pc read size: " << read_pnts  << std::endl;
      break;
    }

  }
  return pc;
};

void RebuildLivoxMid360(PCDPointCloud& pc, int num_synchronous){
  if (pc.info.height != 1){
    std::cout << "Warning: Input Pcd file is not Livox-Mid360!" << std::endl;
    return;
  }

  PCDPointCloud line_scan;
  line_scan.info = pc.info;
  line_scan.info.height = num_synchronous;
  line_scan.info.width = std::ceil((float)pc.info.width / num_synchronous);
  line_scan.Init();

  int num_valid_points = 0;
  for (int idx = 0; idx < pc.info.num_points; idx++){
    long int pnt_height = idx % pc.info.height;
    long int pnt_width = idx / pc.info.height;

    int new_height = idx % num_synchronous;
    int new_width = idx / num_synchronous;
    
    line_scan.point_cloud[new_height][new_width] = pc.point_cloud[pnt_height][pnt_width];
    if (line_scan.point_cloud[new_height][new_width].is_valid){
      num_valid_points++;
    }
  }
  std::cout << "\t=> pc info: " << line_scan.info.num_points << ", " << num_valid_points << std::endl;

  std::vector<std::vector<int>> scan_ids;
  scan_ids.resize(line_scan.info.height);
  for (int i = 0; i < scan_ids.size(); i++){
    scan_ids.at(i).resize(line_scan.info.width);
  }

  // // statistics
  // std::vector<float> corners_eval;
  // corners_eval.reserve(line_scan.info.height * line_scan.info.width);
  // double sum_corner = 0;

  int num_scan = 0;
  int max_scan = -1;
  std::vector<int > num_pnts_per_scan(4, 0);
  // std::unordered_map<int, float> scan_theta;
  std::vector<std::pair<int, float>> scan_theta;
  std::vector<int> last_ids(line_scan.info.height, -1);
  for (int j = 0; j < line_scan.info.width; j++){
    for (int i = 0; i < line_scan.info.height; i++){
      PCDPoint pnt, last_pnt;
      // pnt.is_valid = false;
      // last_pnt.is_valid = false;
      pnt = line_scan.point_cloud[i][j];
      scan_ids[i][j] = num_scan + i;
      num_pnts_per_scan.at(num_scan + i)++;

      if (max_scan < scan_ids[i][j]){
        max_scan = scan_ids[i][j];
      }

      if (scan_theta.size() <= scan_ids[i][j]){
        for (int theta_k = scan_theta.size(); theta_k < scan_ids[i][j] + 1; theta_k++){
          scan_theta.push_back(std::pair<int, float>(theta_k, -3));
        }
      }

      if (!pnt.is_valid || pnt.norm() < 1e-6){
        continue;
      }


      if (scan_theta[scan_ids[i][j]].second < -2){
        float theta = atan(pnt.z / std::sqrt(pnt.x * pnt.x + pnt.y * pnt.y));
        scan_theta[scan_ids[i][j]] = std::pair<int, float>(scan_ids[i][j], theta);
      }

      if (i == 0 && last_ids[i] > 0){
        last_pnt = line_scan.point_cloud[i][last_ids[i]];
        if(pnt.x < 0 && last_pnt.y * pnt.y <0){
          num_scan += num_synchronous;
          for (int scan_id = 0; scan_id < line_scan.info.height; scan_id++){
            last_ids[scan_id] = -1;
            num_pnts_per_scan.push_back(0);
          }
        }
      }
    
      // // statistics
      // if (last_ids[i] > 0 && j - last_ids[i] == 1){
      //   Eigen::Vector3f pnt1(pnt.x, pnt.y, pnt.z);
      //   Eigen::Vector3f pnt2(last_pnt.x, last_pnt.y, last_pnt.z);
      //   if (pnt1.norm() > 0.5 && pnt2.norm() > 0.5){
      //     float theat = std::abs(acos(pnt1.normalized().dot(pnt2.normalized())));
      //     corners_eval.push_back(theat);
      //     sum_corner += theat;
      //   }
      // }

      if (line_scan.point_cloud[i][j].is_valid){
        last_ids[i] = j;
      }
    }
  }

  std::sort(scan_theta.begin(), scan_theta.end(), 
    [](const std::pair<int, float> a, const std::pair<int, float> b){
      return a.second > b.second;
    });
  std::unordered_map<int, int> scan_id_maps;
  for (int i = 0; i < scan_theta.size(); i++){
    scan_id_maps[scan_theta[i].first] = i;
  }

  int max_pnts_per_scan = 0;
  for (int i = 0; i < num_pnts_per_scan.size(); i++){
    max_pnts_per_scan = std::max(max_pnts_per_scan, num_pnts_per_scan.at(i));
  }
  std::cout << "\t=> num_pnts_per_scan = " << max_pnts_per_scan << std::endl;

  pc.Clear();
  pc.info.Version = line_scan.info.Version;
  pc.info.height = max_scan + 1;
  pc.info.width = max_pnts_per_scan;
  pc.info.num_points = pc.info.height * pc.info.width;
  pc.Init();

  std::vector<int > new_scan_idxs(pc.info.height, 0);
  int max_new_scan_id = 0;
  for (int j = 0; j < line_scan.info.width; j++){
    for (int i = 0; i < line_scan.info.height; i++){
      int scan_id = scan_ids[i][j];
      int new_idx = new_scan_idxs[scan_id];
      int new_scan_id = scan_id_maps[scan_id];
      pc.point_cloud[new_scan_id][new_idx] = line_scan.point_cloud[i][j];
      new_scan_idxs[scan_id]++;
    }
  }
  
  // std::cout << "RebuildLivoxMid360 Done ..." << pc.info.height << ", " 
  //           << pc.info.width << ", " << pc.info.num_points << std::endl;
  return;
}

void WritePcd(const std::string& path, PCDPointCloud pc){
  return;
}

void RemoveClosedPointCloud(PCDPointCloud& cloud, float min_thres, float max_thres){
    size_t j = 0;
    for (size_t row = 0; row < cloud.info.height; row++){
      for (size_t col = 0; col < cloud.info.width; col++){
        double dist2 = cloud.point_cloud[row][col].x * cloud.point_cloud[row][col].x + 
            cloud.point_cloud[row][col].y * cloud.point_cloud[row][col].y + 
            cloud.point_cloud[row][col].z * cloud.point_cloud[row][col].z;
        if (dist2 < min_thres * min_thres || max_thres * max_thres < dist2){
          cloud.point_cloud[row][col].is_valid = false;
        }
      }
    }
};

std::vector<PlyPoint> Convert2Ply(PCDPointCloud& pc){
  std::vector<sensemap::PlyPoint> ply_points;
  ply_points.reserve(pc.info.num_points);

  float max_intensity = 0.0f;
  // std::vector<float> intensity_v;
  for (size_t i = 0; i < pc.info.num_points; i++){
    // max_curv = std::max(max_curv, points[i].curv);
    long int pnt_height = i % pc.info.height;
    long int pnt_width = i / pc.info.height;
    max_intensity = std::max(max_intensity, pc.point_cloud[pnt_height][pnt_width].intensity);
    // intensity_v.push_back(pc.point_cloud[pnt_height][pnt_width].intensity);
  }
  const double max_value = std::log1p(max_intensity);
  std::cout << "\t=> max_intensity: " << max_intensity << ", " << pc.info.height << " x " << pc.info.width << std::endl;
 
  for (int i = 0; i < pc.info.num_points; i++){
      PlyPoint pnt;
      long int pnt_height = i % pc.info.height;
      long int pnt_width = i / pc.info.height;
      if (!pc.point_cloud[pnt_height][pnt_width].is_valid){
        continue;
      }
      pnt.x = pc.point_cloud[pnt_height][pnt_width].x;
      pnt.y = pc.point_cloud[pnt_height][pnt_width].y;
      pnt.z = pc.point_cloud[pnt_height][pnt_width].z;
      // ColorMap(pc.point_cloud[pnt_height][pnt_width].intensity,
      //          pnt.r, pnt.g, pnt.b);
      // ColorMap(float(pnt_height) / 32,
      //           pnt.r, pnt.g, pnt.b);

      // const double value = std::log1p(pc.point_cloud[pnt_height][pnt_width].intensity) / max_value;
      // const double value = double(pnt_width) / pc.info.num_points;
      double value = double(pnt_height) / pc.info.height;
      uint8_t r = 255 * JetColormap::Red(value);
      uint8_t g = 255 * JetColormap::Green(value);
      uint8_t b = 255 * JetColormap::Blue(value);

      pnt.r = r;
      pnt.g = g;
      pnt.b = b;
      ply_points.push_back(pnt);
  }
  return ply_points;
};

void ConvertPcd2Ply(PCDPointCloud& pc,
  std::vector<PlyPoint>& ply_points, 
  std::vector<float>& intensities){

  const double max_value = std::log1p(1000);

  ply_points.reserve(pc.info.num_points);
  intensities.reserve(pc.info.num_points);
  for (int i = 0; i < pc.info.num_points; i++){
      PlyPoint pnt;
      long int pnt_height = i % pc.info.height;
      long int pnt_width = i / pc.info.height;
      if (!pc.point_cloud[pnt_height][pnt_width].is_valid){
        continue;
      }

      pnt.x = pc.point_cloud[pnt_height][pnt_width].x;
      pnt.y = pc.point_cloud[pnt_height][pnt_width].y;
      pnt.z = pc.point_cloud[pnt_height][pnt_width].z;

      double value = std::log1p(pc.point_cloud[pnt_height][pnt_width].intensity) / max_value;
      value = std::min(value, 1.0);
      uint8_t r = 255 * JetColormap::Red(value);
      uint8_t g = 255 * JetColormap::Green(value);
      uint8_t b = 255 * JetColormap::Blue(value);
      pnt.r = r;
      pnt.g = g;
      pnt.b = b;

      ply_points.push_back(pnt);
      intensities.push_back(pc.point_cloud[pnt_height][pnt_width].intensity);
  }
  return;
}

}  // namespace sensemap
