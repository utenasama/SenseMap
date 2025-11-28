//Copyright (c) 2019, SenseTime Group.
//All rights reserved.

#include <numeric>

#ifdef USE_CV_MAT
#include "util/freeimage2mat.h"
#include "util/mat2freeimage.h"
#endif

#include "util/misc.h"
#include "base/common.h"

#include "mvs/workspace.h"

namespace sensemap {
namespace mvs {

Workspace::CachedImage::CachedImage() {}

Workspace::CachedImage::CachedImage(CachedImage&& other) {
  num_bytes = other.num_bytes;
  bitmap = std::move(other.bitmap);
  depth_map = std::move(other.depth_map);
  normal_map = std::move(other.normal_map);
  conf_map = std::move(other.conf_map);
}

Workspace::CachedImage& Workspace::CachedImage::operator=(CachedImage&& other) {
  if (this != &other) {
    num_bytes = other.num_bytes;
    bitmap = std::move(other.bitmap);
    depth_map = std::move(other.depth_map);
    normal_map = std::move(other.normal_map);
    conf_map = std::move(other.conf_map);
  }
  return *this;
}

size_t Workspace::CachedImage::NumBytes() const { return num_bytes; }

Workspace::Workspace(const Options& options)
    : options_(options),
      cache_(1024 * 1024 * 1024 * options_.cache_size,
             [](const int) { return CachedImage(); }) {
  StringToLower(&options_.input_type);
  model_.Read(options_.workspace_path, options_.image_path, 
              options_.workspace_format, options_.as_orig_res);
  SetModel();
  // std::cout << "max_image_size = " << options_.max_image_size << std::endl;
  // if (options.workspace_format.compare("panorama") != 0 &&
  //     options_.max_image_size > 0) {
  //   for (auto& image : model_.images) {
  //     image.Downsize(options_.max_image_size, options_.max_image_size);
  //   }
  // }

  depth_map_path_ = EnsureTrailingSlash(
      JoinPaths(options_.workspace_path, options_.stereo_folder, DEPTHS_DIR));
  normal_map_path_ = EnsureTrailingSlash(JoinPaths(
      options_.workspace_path, options_.stereo_folder, NORMALS_DIR));
  conf_map_path_ = EnsureTrailingSlash(JoinPaths(
      options_.workspace_path, options_.stereo_folder, CONFS_DIR));
}

void Workspace::ResetInputType(std::string input_type){
  options_.input_type = input_type;
  StringToLower(&options_.input_type);
}

void Workspace::SetModel(const std::string cluster_rect_path) { 
  model_.ModelFilter(cluster_rect_path, 
                    options_.image_path, 
                    options_.workspace_format);

  depth_map_path_ = EnsureTrailingSlash(
    JoinPaths(cluster_rect_path, options_.stereo_folder, DEPTHS_DIR));
  normal_map_path_ = EnsureTrailingSlash(
    JoinPaths(cluster_rect_path, options_.stereo_folder, NORMALS_DIR));
  conf_map_path_ = EnsureTrailingSlash(
    JoinPaths(cluster_rect_path, options_.stereo_folder, CONFS_DIR));
}

void Workspace::SetModel(const std::unordered_set<int >& cluster_ids) { 
  model_.ModelFilter(cluster_ids, 
                    options_.image_path, 
                    options_.workspace_format);
}

void Workspace::ClearCache() { cache_.Clear(); }

const Workspace::Options& Workspace::GetOptions() const { return options_; }

const Model& Workspace::GetModel() const { return model_; }

const Bitmap& Workspace::GetBitmap(const int image_idx) {
  auto& cached_image = cache_.GetMutable(image_idx);
  if (!cached_image.bitmap) {
#ifdef USE_CV_MAT
    // cv::Mat mat = cv::imread(GetBitmapPath(image_idx), 
    //   options_.image_as_rgb ? cv::IMREAD_COLOR : cv::IMREAD_GRAYSCALE);

    // cached_image.bitmap.reset(new Bitmap());
    // cached_image.bitmap->Allocate(mat.cols, mat.rows, options_.image_as_rgb);
    // Mat2FreeImage(mat, cached_image.bitmap.get());
#else
    cached_image.bitmap.reset(new Bitmap());
    cached_image.bitmap->Read(GetBitmapPath(image_idx), options_.image_as_rgb);
#endif
    if (options_.max_image_size > 0) {
      cached_image.bitmap->Rescale(model_.images.at(image_idx).GetWidth(),
                                   model_.images.at(image_idx).GetHeight());
    }
    cached_image.num_bytes += cached_image.bitmap->NumBytes();
    cache_.UpdateNumBytes(image_idx);
  }
  return *cached_image.bitmap;
}

const DepthMap& Workspace::GetDepthMap(const int image_idx) {
  auto& cached_image = cache_.GetMutable(image_idx);
  if (!cached_image.depth_map) {
    cached_image.depth_map.reset(new DepthMap());
    cached_image.depth_map->Read(GetDepthMapPath(image_idx));
    if (options_.max_image_size > 0) {
      cached_image.depth_map->Downsize(model_.images.at(image_idx).GetWidth(),
                                       model_.images.at(image_idx).GetHeight());
    }
    cached_image.num_bytes += cached_image.depth_map->GetNumBytes();
    cache_.UpdateNumBytes(image_idx);
  }
  return *cached_image.depth_map;
}

const NormalMap& Workspace::GetNormalMap(const int image_idx) {
  auto& cached_image = cache_.GetMutable(image_idx);
  if (!cached_image.normal_map) {
    cached_image.normal_map.reset(new NormalMap());
    cached_image.normal_map->Read(GetNormalMapPath(image_idx));
    if (options_.max_image_size > 0) {
      cached_image.normal_map->Downsize(
          model_.images.at(image_idx).GetWidth(),
          model_.images.at(image_idx).GetHeight());
    }
    cached_image.num_bytes += cached_image.normal_map->GetNumBytes();
    cache_.UpdateNumBytes(image_idx);
  }
  return *cached_image.normal_map;
}

const Mat<float>& Workspace::GetConfhMap(const int image_idx) {
  auto& cached_image = cache_.GetMutable(image_idx);
  if (!cached_image.conf_map) {
    cached_image.conf_map.reset(new Mat<float>());
    cached_image.conf_map->Read(GetConfMapPath(image_idx));
    if (options_.max_image_size > 0) {
      DepthMap temp_depth_map(*cached_image.conf_map, -1.0f, -1.0f);
      temp_depth_map.Downsize(model_.images.at(image_idx).GetWidth(),
                                       model_.images.at(image_idx).GetHeight());
      cached_image.conf_map->Set(temp_depth_map.GetData());
    }
    cached_image.num_bytes += cached_image.conf_map->GetNumBytes();
    cache_.UpdateNumBytes(image_idx);
  }
  return *cached_image.conf_map;
}

std::string Workspace::GetBitmapPath(const int image_idx) const {
  return model_.images.at(image_idx).GetPath();
}

std::string Workspace::GetDepthMapPath(const int image_idx) const {
  return depth_map_path_ + GetFileName(image_idx);
}

std::string Workspace::GetNormalMapPath(const int image_idx) const {
  return normal_map_path_ + GetFileName(image_idx);
}

std::string Workspace::GetConfMapPath(const int image_idx) const {
  return conf_map_path_ + GetFileName(image_idx);
}

bool Workspace::HasBitmap(const int image_idx) const {
  return ExistsFile(GetBitmapPath(image_idx));
}

bool Workspace::HasDepthMap(const int image_idx) const {
  return ExistsFile(GetDepthMapPath(image_idx));
}

bool Workspace::HasNormalMap(const int image_idx) const {
  return ExistsFile(GetNormalMapPath(image_idx));
}

std::string Workspace::GetFileName(const int image_idx) const {
  const auto& image_name = model_.GetImageName(image_idx);
  return StringPrintf("%s.%s.%s", image_name.c_str(),
                      options_.input_type.c_str(), DEPTH_EXT);
}

std::unordered_map<image_t, size_t> Workspace::GetDenseImageId2Idx() const{
  return model_.GetRectImageId2Idx();
}

void ImportPMVSWorkspace(const Workspace& workspace,
                         const std::string& option_name) {
  const std::string& workspace_path = workspace.GetOptions().workspace_path;
  const std::string& stereo_folder = workspace.GetOptions().stereo_folder;

  CreateDirIfNotExists(JoinPaths(workspace_path, stereo_folder));
  CreateDirIfNotExists(JoinPaths(workspace_path, stereo_folder, DEPTHS_DIR));
  CreateDirIfNotExists(JoinPaths(workspace_path, stereo_folder, NORMALS_DIR));
  CreateDirIfNotExists(
      JoinPaths(workspace_path, stereo_folder, CONSISTENCY_DIR));

  const auto option_lines =
      ReadTextFileLines(JoinPaths(workspace_path, option_name));
  for (const auto& line : option_lines) {
    if (!StringStartsWith(line, "timages")) {
      continue;
    }

    const auto elems = StringSplit(line, " ");
    int num_images = std::stoull(elems[1]);

    std::vector<int> image_idxs;
    if (num_images == -1) {
      CHECK_EQ(elems.size(), 4);
      const int range_lower = std::stoull(elems[2]);
      const int range_upper = std::stoull(elems[3]);
      CHECK_LT(range_lower, range_upper);
      num_images = range_upper - range_lower;
      image_idxs.resize(num_images);
      std::iota(image_idxs.begin(), image_idxs.end(), range_lower);
    } else {
      CHECK_EQ(num_images + 2, elems.size());
      image_idxs.reserve(num_images);
      for (size_t i = 2; i < elems.size(); ++i) {
        const int image_idx = std::stoull(elems[i]);
        image_idxs.push_back(image_idx);
      }
    }

    std::vector<std::string> image_names;
    image_names.reserve(num_images);
    for (const auto image_idx : image_idxs) {
      const std::string image_name =
          workspace.GetModel().GetImageName(image_idx);
      image_names.push_back(image_name);
    }

    const auto& overlapping_images =
        workspace.GetModel().GetMaxOverlappingImagesFromPMVS();

    const auto patch_match_path =
        JoinPaths(workspace_path, stereo_folder, "patch-match.cfg");
    const auto fusion_path =
        JoinPaths(workspace_path, stereo_folder, "fusion.cfg");
    std::ofstream patch_match_file(patch_match_path, std::ios::trunc);
    std::ofstream fusion_file(fusion_path, std::ios::trunc);
    CHECK(patch_match_file.is_open()) << patch_match_path;
    CHECK(fusion_file.is_open()) << fusion_path;
    for (size_t i = 0; i < image_names.size(); ++i) {
      const auto& ref_image_name = image_names[i];
      patch_match_file << ref_image_name << std::endl;
      if (overlapping_images.empty()) {
        patch_match_file << "__auto__, 20" << std::endl;
      } else {
        for (const int image_idx : overlapping_images[i]) {
          patch_match_file << workspace.GetModel().GetImageName(image_idx)
                           << ", ";
        }
        patch_match_file << std::endl;
      }
      fusion_file << ref_image_name << std::endl;
    }
  }
}

bool ImportPanoramaWorkspace(
  const std::string& workspace_path,
  std::vector<std::string>& image_names,
  std::vector<mvs::Image>& images,
  std::vector<image_t>& image_ids,
  std::vector<std::vector<int> >& overlapping_images,
  std::vector<std::pair<float, float> >& depth_ranges,
  const bool load_image,
  bool as_orig_res) {

  std::cout << "\nImport Panorams Cameras " << std::endl;

  std::string filename = 
    JoinPaths(workspace_path, (as_orig_res ? SPARSE_ORIG_RES_DIR : SPARSE_DIR), "cameras.bin");

  if (!ExistsFile(filename)) {
    filename = JoinPaths(workspace_path, (as_orig_res ? SPARSE_ORIG_RES_DIR : SPARSE_DIR), "cameras.txt");
    std::ifstream file(filename, std::ifstream::in);
    if (!file.is_open()) {
      std::cout << "ERROR! Open file " << filename << " failed" << std::endl;
      return false;
    }

    std::string line;
    std::string item;

    while (std::getline(file, line)){
        StringTrim(&line);

        if (line.empty() || line[0] == '#'){
            continue;
        }

        std::stringstream line_stream1(line);

        // image name
        std::getline(line_stream1, item, ' ');
        const std::string image_name = item;
        image_names.push_back(image_name);

        // image dimension.
        int width, height;
        std::getline(line_stream1, item, ' ');
        width = std::stoi(item);        
        std::getline(line_stream1, item, ' ');
        height = std::stoi(item);

        // camera parameters.
        float K[9], R[9], T[3];
        for (int i = 0; i < 9; ++i) {
            std::getline(line_stream1, item, ' ');
            K[i] = std::stof(item);
        }
        for (int i = 0; i < 9; ++i) {
            std::getline(line_stream1, item, ' ');
            R[i] = std::stof(item);
        }
        for (int i = 0; i < 3; ++i) {
            std::getline(line_stream1, item, ' ');
            T[i] = std::stof(item);
        }

        std::string image_path = 
            JoinPaths(workspace_path, (as_orig_res ? IMAGES_ORIG_RES_DIR : IMAGES_DIR), image_name);
        mvs::Image image(image_path, width, height, K, R, T);
        if (load_image) {
          Bitmap bitmap;
          bitmap.Read(image_path, false);
          image.SetBitmap(bitmap);
        }
        images.emplace_back(image);

        if (!std::getline(file, line)){
            break;
        }
        StringTrim(&line);
        std::stringstream line_stream2(line);
        std::vector<int> src_image_idxs;
        if (!line.empty()) {
            while(!line_stream2.eof()) {
                std::getline(line_stream2, item, ' ');
                int image_idx = std::stoi(item);
                src_image_idxs.push_back(image_idx);
            }
        }
        overlapping_images.emplace_back(src_image_idxs);

        std::pair<float, float> depth_range;
        if (!std::getline(file, line)){
            break;
        }
        StringTrim(&line);
        std::stringstream line_stream3(line);
        if (!line.empty()) {
            std::getline(line_stream3, item, ' ');
            depth_range.first = std::stof(item);            
            std::getline(line_stream3, item, ' ');
            depth_range.second = std::stof(item);
            depth_ranges.emplace_back(depth_range);
        }
    }
    file.close();
  } else {
    std::ifstream file(filename, std::ifstream::in | std::ifstream::binary);
    if (!file.is_open()) {
      std::cout << "ERROR! Open file " << filename << " failed" << std::endl;
      return false;
    }

    image_names.clear();
    images.clear();
    overlapping_images.clear();
    depth_ranges.clear();

    int num_image = 0;
    file.read((char*)&num_image, sizeof(int));

    float K[9], R[9], T[3];

    int interval = 100;
    while(num_image < interval) {
      interval /= 10;
    }

    int progress = 0;
    for (int i = 0; i < num_image; ++i) {
      short len_name = 0;
      file.read((char*)&len_name, sizeof(short));
      std::string name(len_name, '\0');
      file.read((char*)name.data(), len_name);
      image_names.push_back(name);

      int width = 0, height = 0;
      file.read((char*)&width, sizeof(int));
      file.read((char*)&height, sizeof(int));

      file.read((char*)K, sizeof(float) * 9);
      file.read((char*)R, sizeof(float) * 9);
      file.read((char*)T, sizeof(float) * 3);

      std::string image_path = 
          JoinPaths(workspace_path, IMAGES_DIR, name);
      mvs::Image image(image_path, width, height, K, R, T);
      if (load_image) {
        Bitmap bitmap;
        bitmap.Read(image_path, false);
        image.SetBitmap(bitmap);
      }
      images.emplace_back(image);

      int num_overlap_image = 0;
      file.read((char*)&num_overlap_image, sizeof(int));
      std::vector<int> src_image_idxs(num_overlap_image);
      file.read((char*)src_image_idxs.data(), num_overlap_image * sizeof(int));
      overlapping_images.emplace_back(src_image_idxs);

      float depth_min, depth_max;
      file.read((char*)&depth_min, sizeof(float));
      file.read((char*)&depth_max, sizeof(float));
      depth_ranges.emplace_back(depth_min, depth_max);
      ++progress;

      if (progress % interval == 0) {
        std::cout << "\rImport Camera " << progress << "/" << num_image << std::flush; 
      }
    }
    std::cout << std::endl;
    file.close();
  }

  filename = JoinPaths(workspace_path, SPARSE_DIR, "subimage_ids.bin");
  std::ifstream file(filename, std::ifstream::in | std::ifstream::binary);
  if (file.is_open()) {
    image_ids.clear();

    int num_image = 0;
    file.read((char*)&num_image, sizeof(int));
    for (int i = 0; i < num_image; ++i) {
      image_t image_id;
      file.read((char*)&image_id, sizeof(image_t));
      image_ids.push_back(image_id);
    }
  }
  return true;
}

void ExportPanoramaWorkspace(
    const std::string& workspace_path,
    const std::vector<std::string>& image_names,
    const std::vector<mvs::Image>& images,
    const std::vector<image_t>& image_ids,
    const std::vector<std::vector<int> >& overlapping_images,
    const std::vector<std::pair<float, float> >& depth_ranges,
    bool as_orig_res) {
    auto filename = JoinPaths(workspace_path, (as_orig_res ? SPARSE_ORIG_RES_DIR : SPARSE_DIR), "cameras.bin");

    std::ofstream file;
    file.open(filename, std::ofstream::out | std::ofstream::binary);

    int num_image = images.size();
    file.write((char*)&num_image, sizeof(int));
    for (size_t i = 0; i < num_image; ++i) {
        const mvs::Image& image = images[i];

        short len_name = image_names[i].size();
        file.write((char*)&len_name, sizeof(short));
        file.write(image_names[i].data(), image_names[i].size());

        int width = image.GetWidth();
        int height = image.GetHeight();
        file.write((char*)&width, sizeof(int));
        file.write((char*)&height, sizeof(int));

        file.write((char*)image.GetK(), sizeof(float) * 9);
        file.write((char*)image.GetR(), sizeof(float) * 9);
        file.write((char*)image.GetT(), sizeof(float) * 3);

        int num_overlap_image = overlapping_images[i].size();
        file.write((char*)&num_overlap_image, sizeof(int));
        file.write((char*)overlapping_images[i].data(), 
                          num_overlap_image * sizeof(int));

        file.write((char*)&depth_ranges.at(i).first, sizeof(float));
        file.write((char*)&depth_ranges.at(i).second, sizeof(float));
    }
    file.close();

    filename = JoinPaths(workspace_path, SPARSE_DIR, "subimage_ids.bin");
    file.open(filename, std::ofstream::out | std::ofstream::binary);
    file.write((char*)&num_image, sizeof(int));
    for (size_t i = 0; i < num_image; ++i) {
      image_t image_id = image_ids.at(i);
      file.write((char*)&image_id, sizeof(image_t));
    }
    file.close();
}

}  // namespace mvs
}  // namespace sensemap
