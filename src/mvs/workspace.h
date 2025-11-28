//Copyright (c) 2019, SenseTime Group.
//All rights reserved.

#ifndef SENSEMAP_MVS_WORKSPACE_H_
#define SENSEMAP_MVS_WORKSPACE_H_

#include "mvs/consistency_graph.h"
#include "mvs/depth_map.h"
#include "mvs/model.h"
#include "mvs/normal_map.h"
#include "util/bitmap.h"
#include "util/cache.h"

namespace sensemap {
namespace mvs {

class Workspace {
 public:
  struct Options {
    // The maximum cache size in gigabytes.
    double cache_size = 32.0;

    // Maximum image size in either dimension.
    int max_image_size = -1;

    // Whether to read image as RGB or gray scale.
    bool image_as_rgb = true;

    bool as_orig_res = false;

    // Location and type of workspace.
    std::string workspace_path;
    std::string workspace_format;
    std::string image_path;
    std::string input_type;
    std::string stereo_folder = "stereo";
  };

  Workspace(const Options& options);

  void ResetInputType(std::string input_type);
  void SetModel(const std::string input_cluster_file_path = "");
  void SetModel(const std::unordered_set<int >& cluster_ids);

  void ClearCache();

  const Options& GetOptions() const;

  const Model& GetModel() const;
  const Bitmap& GetBitmap(const int image_idx);
  const DepthMap& GetDepthMap(const int image_idx);
  const NormalMap& GetNormalMap(const int image_idx);
  const Mat<float>& GetConfhMap(const int image_idx);

  // Get paths to bitmap, depth map, normal map and consistency graph.
  std::string GetBitmapPath(const int image_idx) const;
  std::string GetDepthMapPath(const int image_idx) const;
  std::string GetNormalMapPath(const int image_idx) const;
  std::string GetConfMapPath(const int image_idx) const;

  // Return whether bitmap, depth map, normal map, and consistency graph exist.
  bool HasBitmap(const int image_idx) const;
  bool HasDepthMap(const int image_idx) const;
  bool HasNormalMap(const int image_idx) const;

  std::string GetFileName(const int image_idx) const;
  std::unordered_map<image_t, size_t> GetDenseImageId2Idx() const;

 private:
  class CachedImage {
   public:
    CachedImage();
    CachedImage(CachedImage&& other);
    CachedImage& operator=(CachedImage&& other);
    size_t NumBytes() const;
    size_t num_bytes = 0;
    std::unique_ptr<Bitmap> bitmap;
    std::unique_ptr<DepthMap> depth_map;
    std::unique_ptr<NormalMap> normal_map;
    std::unique_ptr<Mat<float>> conf_map;

   private:
    NON_COPYABLE(CachedImage)
  };

  Options options_;
  Model model_;
  MemoryConstrainedLRUCache<int, CachedImage> cache_;
  std::string depth_map_path_;
  std::string normal_map_path_;
  std::string conf_map_path_;
};

// Import a PMVS workspace into the COLMAP workspace format. Only images in the
// provided option file name will be imported and used for reconstruction.
void ImportPMVSWorkspace(const Workspace& workspace,
                         const std::string& option_name);

bool ImportPanoramaWorkspace(
  const std::string& workspace_path,
  std::vector<std::string>& image_names,
  std::vector<mvs::Image>& images,
  std::vector<image_t>& image_ids,
  std::vector<std::vector<int> >& overlapping_images,
  std::vector<std::pair<float, float> >& depth_ranges,
  const bool load_image = false,
  bool as_orig_res = false);

void ExportPanoramaWorkspace(
  const std::string& workspace_path,
  const std::vector<std::string>& image_names,
  const std::vector<mvs::Image>& images,
  const std::vector<image_t>& image_ids,
  const std::vector<std::vector<int> >& overlapping_images,
  const std::vector<std::pair<float, float> >& depth_ranges,
  bool as_orig_res = false);

}  // namespace mvs
}  // namespace sensemap

#endif  // SENSEMAP_MVS_WORKSPACE_H_
