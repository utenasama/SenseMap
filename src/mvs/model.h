//Copyright (c) 2019, SenseTime Group.
//All rights reserved.

#ifndef SENSEMAP_MVS_MODEL_H_
#define SENSEMAP_MVS_MODEL_H_

#include <cstdint>
#include <fstream>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <vector>
#include <Eigen/Core>

#include "base/reconstruction.h"
#include "util/bb.h"
#include "mvs/image.h"

namespace sensemap {
namespace mvs {

// Simple sparse model class.
struct Model {
  struct Point {
    float x = 0;
    float y = 0;
    float z = 0;
    float error = 0;
    std::vector<int> track;
    std::vector<Eigen::Vector2d> points2d;
  };

  // Read the model from different data formats.
  void Read(const std::string& path, 
            const std::string& images_path,
            const std::string& format,
            bool as_orig_res = false);
  void ReadFromCOLMAP(const std::string& path,
                      const std::string& images_path,
                      const std::string& model_type,
                      bool as_orig_res = false);
  void ReadFromPMVS(const std::string& path);

  void ModelFilter(const std::string& rect_cluster_path,
                   const std::string& images_path,
                   const std::string& model_type);
  void ModelFilter(const std::unordered_set<int >& cluster_ids,
                   const std::string& images_path,
                   const std::string& model_type);

  void GetUpdateImageidxs(const std::string& path,
      std::unordered_set<int>& unique_image_idxs) const;

  // Get the image index for the given image name.
  int GetImageIdx(const std::string& name) const;
  std::string GetImageName(const int image_idx) const;
  image_t GetImageId(const int image_idx) const;
  int GetImageIdx(const image_t image_id) const;

  std::unordered_map<size_t, image_t> GetImageIdx2Ids() const;
  std::unordered_map<image_t, size_t> GetImageId2Idx() const;

  // For each image, determine the maximally overlapping images, sorted based on
  // the number of shared points subject to a minimum robust average
  // triangulation angle of the points.
  std::vector<std::vector<int>> GetMaxOverlappingImages(
      const size_t max_num_images, const double min_triangulation_angle) const;
  std::vector<std::vector<int>> GetMaxOverlappingImages(
      const size_t max_num_images, const double min_triangulation_angle, 
      const std::unordered_set<int>& unique_image_idxs) const;

  // Get the overlapping images defined in the vis.dat file.
  const std::vector<std::vector<int>>& GetMaxOverlappingImagesFromPMVS() const;

  // Compute the robust minimum and maximum depths from the sparse point cloud.
  std::vector<std::pair<float, float>> ComputeDepthRanges() const;

  // Compute the number of shared points between all overlapping images.
  std::vector<std::unordered_map<int, int>> ComputeSharedPoints() const;

  std::vector<int> ComputeOverlappingImages(const int ref_image_idx) const;

  // Compute the median triangulation angles between all overlapping images.
  std::vector<std::unordered_map<int, float>> ComputeTriangulationAngles(
      const float percentile = 50) const;
  
  // Compute the median distance between point to camera
  float ComputeNthDistance(float nth_factor) const;
  float ComputeNthDepthImage(const int ref_image_idx, float nth_factor = 0.2) const;

  float ComputeMeanAngularResolution() const;

  std::unordered_map<image_t, size_t> GetRectImageId2Idx() const;

  // Note that in case the data is read from a SenseMap reconstruction, the index
  // of an image or point does not correspond to its original identifier in the
  // reconstruction, but it corresponds to the position in the
  // images.bin/points3D.bin files. This is mainly done for more efficient
  // access to the data, which is required during the stereo fusion stage.
  std::vector<Image> images;
  std::vector<Point> points;
  std::vector<YCrCbFactor> yrb_factors_;

  BoundingBox bb_;

 private:
  bool ReadFromBundlerPMVS(const std::string& path);
  bool ReadFromRawPMVS(const std::string& path);

  std::vector<std::string> image_names_;
  std::unordered_map<std::string, int> image_name_to_idx_;
  std::unordered_map<image_t, size_t> image_id_to_idx_;
  std::unordered_map<size_t, image_t> image_idx_to_id_;

  Reconstruction* reconstruction_;
  std::unordered_map<image_t, size_t> rect_image_id_to_idx_;

  std::vector<std::vector<int>> pmvs_vis_dat_;
};

}  // namespace mvs
}  // namespace sensemap

#endif  // SENSEMAP_MVS_MODEL_H_
