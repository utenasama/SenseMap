//Copyright (c) 2019, SenseTime Group.
//All rights reserved.

#include "base/pose.h"
#include "base/projection.h"
#include "base/camera_models.h"
#include "base/reconstruction.h"
#include "base/triangulation.h"
#include "base/common.h"
#include "util/exception_handler.h"
#include "util/misc.h"
#include "util/threading.h"

#include "mvs/utils.h"
#include "mvs/model.h"

namespace sensemap {
namespace mvs {

using namespace utility;

namespace {
std::pair<float, float> ComputeDepthRangeFromBBox(
  const BoundingBox& bb, const Eigen::Matrix3f& R, const Eigen::Vector3f& t) {
  std::pair<float, float> depth_range;
  depth_range.first = depth_range.second = (R * bb.GetCorners(0) + t).z();
  for (size_t i = 1; i < 8; ++i) {
    Eigen::Vector3f X = R * bb.GetCorners(i) + t;
    depth_range.first = std::min(depth_range.first, X.z());
    depth_range.second = std::max(depth_range.second, X.z());
  }
  depth_range.first = std::max(0.000001f, depth_range.first);
  // depth_range.second *= 1.2;
  return depth_range;
}
}

std::vector<int> Model::ComputeOverlappingImages(const int ref_image_idx) const {
  const image_t ref_image_id = image_idx_to_id_.at(ref_image_idx);
  const auto& mappoints = reconstruction_->MapPoints();
  const auto& ref_image = reconstruction_->Image(ref_image_id);
  std::unordered_set<mappoint_t> ref_mappoint_ids;
  for (const auto& point2D : ref_image.Points2D()) {
    if (!point2D.HasMapPoint()) {
      continue;
    }
    mappoint_t mappoint_id = point2D.MapPointId();
    ref_mappoint_ids.insert(mappoint_id);
  }

  std::vector<int> overlap_image_ids;
  for (size_t i = 0; i < reconstruction_->NumRegisterImages(); ++i) {
    const auto image_id = reconstruction_->RegisterImageIds()[i];
    const auto& image = reconstruction_->Image(image_id);
    if (image_id == ref_image_id) {
      continue;
    }
    size_t num_overlap_mappoint = 0;
    for (const auto& point2D : image.Points2D()) {
      if (!point2D.HasMapPoint()) {
        continue;
      }
      mappoint_t mappoint_id = point2D.MapPointId();
      if (ref_mappoint_ids.count(mappoint_id) != 0) {
        num_overlap_mappoint++;
      }
    }
    if (num_overlap_mappoint > 1000) {
      overlap_image_ids.emplace_back(image_id_to_idx_.at(image_id));
    }
  }
  return overlap_image_ids;
}

void Model::Read(const std::string& path, 
                 const std::string& images_path,
                 const std::string& format,
                 bool as_orig_res) {
  auto format_lower_case = format;
  StringToLower(&format_lower_case);
  if (format_lower_case == "perspective" || format_lower_case == "panorama" ||
      format_lower_case == "rgbd") {
    ReadFromCOLMAP(path, images_path, format_lower_case, as_orig_res);
  } else if (format_lower_case == "pmvs") {
    ReadFromPMVS(path);
  } else {
    // LOG(FATAL) << "Invalid input format";
    exit(StateCode::INVALID_INPUT_FORMAT);
  }
}

void Model::ReadFromCOLMAP(const std::string& path,
                           const std::string& images_path,
                           const std::string& model_type,
                           bool as_orig_res) {
  reconstruction_ = new Reconstruction();
  if (model_type == "panorama") {
    reconstruction_->ReadReconstruction(path);
  } else {
    reconstruction_->ReadReconstruction(JoinPaths(path, (as_orig_res ? SPARSE_ORIG_RES_DIR : SPARSE_DIR)));
  }

  const auto& rig_file_path = JoinPaths(path, SPARSE_DIR, "rig.txt");

  bool has_rig_file = ExistsFile(rig_file_path);
  if (has_rig_file) {
      reconstruction_->ReadCameraIsRigText(rig_file_path);
  }

  auto image_ids = reconstruction_->GetNewImageIds();
  if (image_ids.size() == 0) {
    image_ids = reconstruction_->RegisterImageIds();
  }

  size_t num_images = 0;
  for (size_t i = 0; i < image_ids.size(); ++i) {
    const auto image_id = image_ids.at(i);
    rect_image_id_to_idx_.emplace(image_id, num_images);
    num_images++;
  }
}

void Model::ModelFilter(const std::string& rect_cluster_path,
                        const std::string& images_path,
                        const std::string& model_type) {
  std::unordered_set<int > cluster_ids;
  std::unordered_set<std::string > cluster_names;

  const auto cluster_file_path = JoinPaths(rect_cluster_path, RECT_CLUSTER_NAME);
  if (ExistsFile(cluster_file_path)){
    std::ifstream file(cluster_file_path);
    CHECK(file.is_open()) << cluster_file_path;
    std::string line;
    std::string item;

    while (std::getline(file, line)) {
        StringTrim(&line);

        if (line.empty() || line[0] == '#') {
            continue;
        }

        std::stringstream line_stream(line);

        // ID
        std::getline(line_stream, item, ' ');
        cluster_ids.emplace(std::stoi(item));

        // name
        std::getline(line_stream, item, ' ');
        cluster_names.emplace(item);
    }
    file.close();
  }

  {
    images.clear();
    images.shrink_to_fit();

    points.clear();
    points.shrink_to_fit();

    image_names_.clear();
    image_names_.shrink_to_fit();

    image_name_to_idx_.clear();
    image_id_to_idx_.clear();
    image_idx_to_id_.clear();
  }

  auto image_ids = reconstruction_->GetNewImageIds();
  if (image_ids.size() == 0) {
    image_ids = reconstruction_->RegisterImageIds();
  }

  images.reserve(cluster_ids.size());
  size_t num_images = 0;
  for (size_t i = 0; i < image_ids.size(); ++i) {
    // const auto image_id = reconstruction_->RegisterImageIds()[i];
    const auto image_id = image_ids.at(i);
    if (cluster_ids.find(image_id) == cluster_ids.end() && !cluster_ids.empty()){
      continue;
    }
    const auto& image = reconstruction_->Image(image_id);
    const auto& camera = reconstruction_->Camera(image.CameraId());

    const Eigen::RowMatrix3f K = camera.CalibrationMatrix().cast<float>();
    const Eigen::RowMatrix3f R = 
      QuaternionToRotationMatrix(image.Qvec()).cast<float>();
    const Eigen::Vector3f T = image.Tvec().cast<float>();

    std::string image_name = image.Name();
    if (model_type == "rgbd") {
      image_name = image_name.append(".jpg");
    }

    bool is_rig = false;
    // if(has_rig_file)
    is_rig = camera.IsFromRIG();

    images.emplace_back(JoinPaths(images_path, image_name),  
                                  camera.Width(), camera.Height(), 
                                  K.data(), R.data(), T.data(), is_rig);
    image_id_to_idx_.emplace(image_id, num_images);
    image_idx_to_id_.emplace(num_images, image_id);
    image_names_.push_back(image_name);
    image_name_to_idx_.emplace(image_name, num_images);
    num_images++;
  }
  images.shrink_to_fit();

  auto & yrb_factors = reconstruction_->yrb_factors;
  if (!yrb_factors.empty()) {
    yrb_factors_.resize(num_images);
    for (int i = 0; i < num_images; ++i) {
      image_t image_id = image_idx_to_id_.at(i);
      yrb_factors_[i] = yrb_factors.at(image_id);
    }
  }

  bb_.lt.x() = bb_.lt.y() = bb_.lt.z() = FLT_MAX;
  bb_.rb.x() = bb_.rb.y() = bb_.rb.z() = -FLT_MAX;

  points.reserve(reconstruction_->NumMapPoints());
  const auto& mappoints = reconstruction_->MapPoints();
  for (const auto& mappoint : mappoints) {
    Point point;
    point.x = mappoint.second.X();
    point.y = mappoint.second.Y();
    point.z = mappoint.second.Z();
    point.track.reserve(mappoint.second.Track().Length());
    int track_length = 0;
    for (const auto& track_el : mappoint.second.Track().Elements()) {
      if (image_id_to_idx_.count(track_el.image_id) != 0) {
        const auto& image = reconstruction_->Image(track_el.image_id);
        point.error = mappoint.second.Error();
        point.track.push_back(image_id_to_idx_.at(track_el.image_id));
        point.points2d.emplace_back(image.Point2D(track_el.point2D_idx).XY());
        track_length++;
      }
    }
    if (track_length < 2) {
      continue;
    }
    point.track.shrink_to_fit();
    point.points2d.shrink_to_fit();

    bb_.lt[0] = std::min(bb_.lt[0], point.x);
    bb_.lt[1] = std::min(bb_.lt[1], point.y);
    bb_.lt[2] = std::min(bb_.lt[2], point.z);
    bb_.rb[0] = std::max(bb_.rb[0], point.x);
    bb_.rb[1] = std::max(bb_.rb[1], point.y);
    bb_.rb[2] = std::max(bb_.rb[2], point.z);

    points.push_back(point);
  }

  for (const auto& image_id : image_ids) {
    const auto& image = reconstruction_->Image(image_id);
    Eigen::Vector3d C = image.ProjectionCenter();
    bb_.lt[0] = std::min(bb_.lt[0], (float)C[0]);
    bb_.lt[1] = std::min(bb_.lt[1], (float)C[1]);
    bb_.lt[2] = std::min(bb_.lt[2], (float)C[2]);
    bb_.rb[0] = std::max(bb_.rb[0], (float)C[0]);
    bb_.rb[1] = std::max(bb_.rb[1], (float)C[1]);
    bb_.rb[2] = std::max(bb_.rb[2], (float)C[2]);
  }
}

void Model::ModelFilter(const std::unordered_set<int >& cluster_ids,
                        const std::string& images_path,
                        const std::string& model_type) {
  {
    images.clear();
    images.shrink_to_fit();

    points.clear();
    points.shrink_to_fit();

    image_names_.clear();
    image_names_.shrink_to_fit();

    image_name_to_idx_.clear();
    image_id_to_idx_.clear();
    image_idx_to_id_.clear();
  }

  auto image_ids = reconstruction_->GetNewImageIds();
  if (image_ids.size() == 0) {
    image_ids = reconstruction_->RegisterImageIds();
  }

  images.reserve(cluster_ids.size());
  size_t num_images = 0;
  for (size_t i = 0; i < image_ids.size(); ++i) {
    // const auto image_id = reconstruction_->RegisterImageIds()[i];
    const auto image_id = image_ids.at(i);
    if (cluster_ids.find(image_id) == cluster_ids.end() && !cluster_ids.empty()){
      continue;
    }
    const auto& image = reconstruction_->Image(image_id);
    const auto& camera = reconstruction_->Camera(image.CameraId());

    const Eigen::RowMatrix3f K = camera.CalibrationMatrix().cast<float>();
    const Eigen::RowMatrix3f R = 
      QuaternionToRotationMatrix(image.Qvec()).cast<float>();
    const Eigen::Vector3f T = image.Tvec().cast<float>();

    std::string image_name = image.Name();
    if (model_type == "rgbd") {
      image_name = image_name.append(".jpg");
    }

    bool is_rig = false;
    // if(has_rig_file)
    is_rig = camera.IsFromRIG();

    images.emplace_back(JoinPaths(images_path, image_name),  
                                  camera.Width(), camera.Height(), 
                                  K.data(), R.data(), T.data(), is_rig);
    image_id_to_idx_.emplace(image_id, num_images);
    image_idx_to_id_.emplace(num_images, image_id);
    image_names_.push_back(image_name);
    image_name_to_idx_.emplace(image_name, num_images);
    num_images++;
  }
  images.shrink_to_fit();

  auto & yrb_factors = reconstruction_->yrb_factors;
  if (!yrb_factors.empty()) {
    yrb_factors_.resize(num_images);
    for (int i = 0; i < num_images; ++i) {
      image_t image_id = image_idx_to_id_.at(i);
      yrb_factors_[i] = yrb_factors.at(image_id);
    }
  }

  bb_.lt.x() = bb_.lt.y() = bb_.lt.z() = FLT_MAX;
  bb_.rb.x() = bb_.rb.y() = bb_.rb.z() = -FLT_MAX;

  points.reserve(reconstruction_->NumMapPoints());
  const auto& mappoints = reconstruction_->MapPoints();
  for (const auto& mappoint : mappoints) {
    Point point;
    point.x = mappoint.second.X();
    point.y = mappoint.second.Y();
    point.z = mappoint.second.Z();
    point.track.reserve(mappoint.second.Track().Length());
    int track_length = 0;
    for (const auto& track_el : mappoint.second.Track().Elements()) {
      if (image_id_to_idx_.count(track_el.image_id) != 0) {
        const auto& image = reconstruction_->Image(track_el.image_id);
        point.error = mappoint.second.Error();
        point.track.push_back(image_id_to_idx_.at(track_el.image_id));
        point.points2d.emplace_back(image.Point2D(track_el.point2D_idx).XY());
        track_length++;
      }
    }
    if (track_length < 2) {
      continue;
    }
    point.track.shrink_to_fit();
    point.points2d.shrink_to_fit();

    bb_.lt[0] = std::min(bb_.lt[0], point.x);
    bb_.lt[1] = std::min(bb_.lt[1], point.y);
    bb_.lt[2] = std::min(bb_.lt[2], point.z);
    bb_.rb[0] = std::max(bb_.rb[0], point.x);
    bb_.rb[1] = std::max(bb_.rb[1], point.y);
    bb_.rb[2] = std::max(bb_.rb[2], point.z);

    points.push_back(point);
  }

  for (const auto& image_id : image_ids) {
    const auto& image = reconstruction_->Image(image_id);
    Eigen::Vector3d C = image.ProjectionCenter();
    bb_.lt[0] = std::min(bb_.lt[0], (float)C[0]);
    bb_.lt[1] = std::min(bb_.lt[1], (float)C[1]);
    bb_.lt[2] = std::min(bb_.lt[2], (float)C[2]);
    bb_.rb[0] = std::max(bb_.rb[0], (float)C[0]);
    bb_.rb[1] = std::max(bb_.rb[1], (float)C[1]);
    bb_.rb[2] = std::max(bb_.rb[2], (float)C[2]);
  }
}

void Model::ReadFromPMVS(const std::string& path) {
  if (ReadFromBundlerPMVS(path)) {
    return;
  } else if (ReadFromRawPMVS(path)) {
    return;
  } else {
    // LOG(FATAL) << "Invalid PMVS format";
    exit(StateCode::INVALID_PMVS_FORMAT);
  }
}

void Model::GetUpdateImageidxs(const std::string& path, 
  std::unordered_set<int>& unique_image_idxs) const {
  std::cout << "MODEL: GetUpdateImageidxs" << std::endl;
  std::vector<image_t> image_ids;  
  {
    std::ifstream file(path);
    CHECK(file.is_open()) << path;

    std::string line;
    std::string item;

    while (std::getline(file, line)) {
      StringTrim(&line);

      if (line.empty() || line[0] == '#') {
          continue;
      }

      std::stringstream line_stream1(line);

      // IMAGE_ID
      std::getline(line_stream1, item, ' ');
      const image_t image_id = std::stoul(item);
      // IMAGE_NAME
      std::getline(line_stream1, item, ' ');
      const std::string image_name = item;
      image_ids.emplace_back(image_id);
    }
    file.close();
  }

  std::cout << "insert ..." << std::endl;
  // std::unordered_set<int> unique_image_idxs;
  for(int i = 0; i < image_ids.size(); i++){
    int idx = image_id_to_idx_.at(image_ids.at(i));
    unique_image_idxs.insert(idx);
  }
  std::cout << unique_image_idxs.size() << std::endl;

  return;
}

int Model::GetImageIdx(const std::string& name) const {
  CHECK_GT(image_name_to_idx_.count(name), 0)
      << "Image with name `" << name << "` does not exist";
  return image_name_to_idx_.at(name);
}

std::string Model::GetImageName(const int image_idx) const {
  CHECK_GE(image_idx, 0);
  CHECK_LT(image_idx, image_names_.size());
  return image_names_.at(image_idx);
}

image_t Model::GetImageId(const int image_idx) const {
  CHECK_GE(image_idx, 0);
  CHECK_LT(image_idx, images.size());
  return image_idx_to_id_.at(image_idx);
}

int Model::GetImageIdx(const image_t image_id) const {
  return image_id_to_idx_.at(image_id);
}

std::unordered_map<size_t, image_t> Model::GetImageIdx2Ids() const {
  return image_idx_to_id_;
}

std::unordered_map<image_t, size_t> Model::GetImageId2Idx() const {
  return image_id_to_idx_;
}

std::unordered_map<image_t, size_t> Model::GetRectImageId2Idx() const {
  return rect_image_id_to_idx_;
}

std::vector<std::vector<int> > Model::GetMaxOverlappingImages(
    const size_t max_num_images, const double min_triangulation_angle) const {
  const float kTriangulationAnglePercentile = 75;
  const float min_triangulation_angle_rad = DegToRad(min_triangulation_angle);
  const int num_image = images.size();

  std::vector<std::vector<int> > overlapping_images(num_image);

  std::cout << "Compute observation per image." << std::endl;
  std::vector<std::vector<uint32_t> > images_points(num_image);
  for (size_t i = 0; i < points.size(); ++i) {
    auto point = points.at(i);
    for (size_t j = 0; j < point.track.size(); ++j) {
      const int image_idx = point.track[j];
      images_points[image_idx].push_back(i);
    }
  }

  std::cout << "Compute projection center." << std::endl;
  std::vector<Eigen::Vector3d> proj_centers(num_image);
  for (size_t image_idx = 0; image_idx < num_image; ++image_idx) {
    const auto& image = images[image_idx];
    Eigen::Vector3f C;
    ComputeProjectionCenter(image.GetR(), image.GetT(), C.data());
    proj_centers[image_idx] = C.cast<double>();
  }

  std::cout << "Compute minimal mean distance." << std::endl;
  const std::vector<std::unordered_map<int, int>>& shared_points = ComputeSharedPoints();
  float m_dist(0.0f);
  int m_cnt(0);
  for (size_t image_idx = 0; image_idx < num_image; ++image_idx) {
    float min_dist = FLT_MAX;
    for (auto shared_image : shared_points[image_idx]) {
      float dist = (proj_centers[image_idx] - proj_centers[shared_image.first]).norm();
      if (dist <= 1e-5) {
        continue;
      }
      min_dist = std::min(dist, min_dist);
    }
    if (min_dist < FLT_MAX) {
      m_dist += min_dist;
      m_cnt++;
    }
  }
  m_dist /= m_cnt;

  const float min_dist = 0.1 * m_dist;
  const float max_dist = 100 * m_dist;

  const float optimal_angle = DegToRad(15.0);
  const float min_angle = DegToRad(0.0);
  const float max_angle = DegToRad(60.0);

  auto SelectNeighborViews = [&](size_t image_idx1) {
    std::vector<uint32_t>& image_points = images_points.at(image_idx1);
    if (image_points.empty()) {
      return false;
    }

    std::unordered_map<int, int> shared_images;
    std::unordered_map<int, std::vector<float> > all_triangulation_angles;
    for (size_t j = 0; j < image_points.size(); ++j) {
      auto point = points.at(image_points[j]);
      std::vector<int> track = point.track;
      for (size_t k = 0; k < track.size(); ++k) {
        const int image_idx2 = track[k];
        if (image_idx2 != image_idx1) {
          shared_images[image_idx2]++;
          const float angle = CalculateTriangulationAngle(
            proj_centers.at(image_idx1), proj_centers.at(image_idx2),
            Eigen::Vector3d(point.x, point.y, point.z));
          all_triangulation_angles[image_idx2].push_back(angle);
        }
      }
    }
    
    std::unordered_map<int, float> triangulation_angles;
    for (auto tri_angle : all_triangulation_angles) {
      triangulation_angles[tri_angle.first] = 
      Percentile(tri_angle.second, kTriangulationAnglePercentile);
    }

    auto src_is_rig = images[image_idx1].IsRig();

    std::vector<std::pair<int, int> > ordered_images;
    ordered_images.reserve(shared_images.size());
    for (const auto& image : shared_images) {
      if (triangulation_angles.at(image.first) >= min_triangulation_angle_rad) {
        double factor = images[image.first].IsRig() == src_is_rig ? 1.5 : 1.0;
        ordered_images.emplace_back(image.first, image.second * factor);
      }
    }

    const size_t eff_num_images = std::min(ordered_images.size(), max_num_images);
    if (eff_num_images < ordered_images.size()) {
      std::partial_sort(ordered_images.begin(),
                        ordered_images.begin() + eff_num_images,
                        ordered_images.end(),
                        [](const std::pair<int, int> image1,
                           const std::pair<int, int> image2) {
                          return image1.second > image2.second;
                        });
    } else {
      std::sort(ordered_images.begin(), ordered_images.end(),
                [](const std::pair<int, int> image1,
                   const std::pair<int, int> image2) {
                  return image1.second > image2.second;
                });
    }

    overlapping_images[image_idx1].reserve(eff_num_images);
    for (size_t i = 0; i < eff_num_images; ++i) {
      overlapping_images[image_idx1].push_back(ordered_images[i].first);
    }
    return true;
  };
  auto SelectSpatialNeighborViews = [&](size_t image_idx1) {
    size_t i, j;
    const mvs::Image& ref_image = images.at(image_idx1);
    Eigen::Vector3f ref_ray = Eigen::Map<const Eigen::RowMatrix3f>(ref_image.GetR()).row(2);
    Eigen::Vector3d& ref_C = proj_centers[image_idx1];

    std::vector<std::pair<int, mvs::Image> > scored_images;
    for (i = 0; i < num_image; ++i) {
      if (i == image_idx1) {
        continue;
      }
      const mvs::Image& src_image = images.at(i);
      Eigen::Vector3f src_ray = Eigen::Map<const Eigen::RowMatrix3f>(src_image.GetR()).row(2);
      const float dist = (proj_centers[i] - ref_C).norm();
      if (dist < min_dist || dist > max_dist) {
        continue;
      }
      const float angle = std::acos(ref_ray.dot(src_ray));
      if (angle < min_angle || angle > max_angle) {
        continue;
      }
      scored_images.emplace_back(i, src_image);
    }

    struct ViewScore {
        uint32_t view_idx;
        float score;
    };
    std::vector<ViewScore> view_scores;
    view_scores.reserve(scored_images.size());
    for (i = 0; i < scored_images.size(); ++i) {
      const mvs::Image& src_image = scored_images.at(i).second;

      const float* src_R = src_image.GetR();
      Eigen::Vector3d src_C = proj_centers.at(scored_images.at(i).first);
      Eigen::Vector3f src_ray = Eigen::Map<const Eigen::RowMatrix3f>(src_R).row(2);

      const float cos_angle = ref_ray.dot(src_ray);
      const float wangle = 1.1f - cos_angle;
      const float dist = (src_C - ref_C).norm();
      const float wdist = std::max(dist / m_dist, 1.0f);

      ViewScore view_score;
      view_score.view_idx = scored_images.at(i).first;
      view_score.score = wangle * wdist;
      view_scores.emplace_back(view_score);
    }

    std::sort(view_scores.begin(), view_scores.end(), 
        [&](const ViewScore s1, const ViewScore s2) {
        return s1.score < s2.score;
    });

    int eff_num_images = std::min(view_scores.size(), max_num_images);
    if (eff_num_images < 1) {
        return false;
    }

    overlapping_images[image_idx1].push_back(image_idx1);
    while(overlapping_images[image_idx1].size() <= eff_num_images &&
          !view_scores.empty()) {
      size_t k;
      for (k = 0; k < view_scores.size(); ++k) {
        int next_image_id = view_scores[k].view_idx;
        Eigen::Vector3d& next_C = proj_centers.at(next_image_id);
        bool spurious_image = false;
        for (auto overlap_image_id : overlapping_images[image_idx1]) {
          Eigen::Vector3d& overlap_C = proj_centers.at(overlap_image_id);
          double dist = (next_C - overlap_C).norm();
          if (dist < min_dist) {
            spurious_image = true;
            break;
          }
        }
        if (!spurious_image) {
          overlapping_images[image_idx1].push_back(next_image_id);
          view_scores.erase(view_scores.begin() + k);
          break;
        }
      }
      if (k >= view_scores.size()) {
        break;
      }
    }
    overlapping_images[image_idx1].erase(overlapping_images[image_idx1].begin());
    return true;
  };
  
  auto ProcessImage = [&](size_t i) {
    if (!SelectNeighborViews(i)) {
      SelectSpatialNeighborViews(i);
    }
    std::cout << std::flush << StringPrintf("\rProcess Image %.1f%", (i + 1) * 100.0 / num_image);
  };
  std::cout << std::endl;

  const int num_eff_threads = GetEffectiveNumThreads(-1);
  std::unique_ptr<ThreadPool> thread_pool;
  thread_pool.reset(new ThreadPool(std::min(num_eff_threads, num_image)));

  for (size_t image_idx = 0; image_idx < num_image; ++image_idx) {
    thread_pool->AddTask(ProcessImage, image_idx);
    // ProcessImage(image_idx);
  }
  thread_pool->Wait();

  std::cout << std::endl;
  return overlapping_images;
}

std::vector<std::vector<int> > Model::GetMaxOverlappingImages(
    const size_t max_num_images, const double min_triangulation_angle, 
    const std::unordered_set<int>& unique_image_idxs) const {
  PrintHeading3("Model GetMaxOverlappingImages");
  const float kTriangulationAnglePercentile = 75;
  const float min_triangulation_angle_rad = DegToRad(min_triangulation_angle);
  const int num_image = images.size();

  std::vector<std::vector<int> > overlapping_images(num_image);

  std::cout << "Compute observation per image." << std::endl;
  std::vector<std::vector<uint32_t> > images_points(num_image);
  for (size_t i = 0; i < points.size(); ++i) {
    auto point = points.at(i);
    for (size_t j = 0; j < point.track.size(); ++j) {
      const int image_idx = point.track[j];
      images_points[image_idx].push_back(i);
    }
  }

  std::cout << "Compute projection center." << std::endl;
  std::vector<Eigen::Vector3d> proj_centers(num_image);
  for (size_t image_idx = 0; image_idx < num_image; ++image_idx) {
    const auto& image = images[image_idx];
    Eigen::Vector3f C;
    ComputeProjectionCenter(image.GetR(), image.GetT(), C.data());
    proj_centers[image_idx] = C.cast<double>();
  }

  std::cout << "Compute minimal mean distance." << std::endl;
  const std::vector<std::unordered_map<int, int>>& shared_points = ComputeSharedPoints();
  float m_dist(0.0f);
  int m_cnt(0);
  for (size_t image_idx = 0; image_idx < num_image; ++image_idx) {
    float min_dist = FLT_MAX;
    for (auto shared_image : shared_points[image_idx]) {
      float dist = (proj_centers[image_idx] - proj_centers[shared_image.first]).norm();
      if (dist <= 1e-5) {
        continue;
      }
      min_dist = std::min(dist, min_dist);
    }
    if (min_dist < FLT_MAX) {
      m_dist += min_dist;
      m_cnt++;
    }
  }
  m_dist /= m_cnt;

  const float min_dist = 0.1 * m_dist;
  const float max_dist = 100 * m_dist;

  const float optimal_angle = DegToRad(15.0);
  const float min_angle = DegToRad(0.0);
  const float max_angle = DegToRad(60.0);

  auto SelectNeighborViews = [&](size_t image_idx1) {
    std::vector<uint32_t>& image_points = images_points.at(image_idx1);
    if (image_points.empty()) {
      return false;
    }

    std::unordered_map<int, int> shared_images;
    std::unordered_map<int, std::vector<float> > all_triangulation_angles;
    for (size_t j = 0; j < image_points.size(); ++j) {
      auto point = points.at(image_points[j]);
      std::vector<int> track = point.track;
      for (size_t k = 0; k < track.size(); ++k) {
        if (unique_image_idxs.find(track[k]) == unique_image_idxs.end()){
          continue;
        }
        const int image_idx2 = track[k];
        if (image_idx2 != image_idx1) {
          shared_images[image_idx2]++;
          const float angle = CalculateTriangulationAngle(
            proj_centers.at(image_idx1), proj_centers.at(image_idx2),
            Eigen::Vector3d(point.x, point.y, point.z));
          all_triangulation_angles[image_idx2].push_back(angle);
        }
      }
    }
    
    std::unordered_map<int, float> triangulation_angles;
    for (auto tri_angle : all_triangulation_angles) {
      triangulation_angles[tri_angle.first] = 
      Percentile(tri_angle.second, kTriangulationAnglePercentile);
    }

    auto src_is_rig = images[image_idx1].IsRig();

    std::vector<std::pair<int, int> > ordered_images;
    ordered_images.reserve(shared_images.size());
    for (const auto& image : shared_images) {
      if (triangulation_angles.at(image.first) >= min_triangulation_angle_rad) {
        double factor = images[image.first].IsRig() == src_is_rig ? 1.5 : 1.0;
        ordered_images.emplace_back(image.first, image.second * factor);
      }
    }

    const size_t eff_num_images = std::min(ordered_images.size(), max_num_images);
    if (eff_num_images < ordered_images.size()) {
      std::partial_sort(ordered_images.begin(),
                        ordered_images.begin() + eff_num_images,
                        ordered_images.end(),
                        [](const std::pair<int, int> image1,
                           const std::pair<int, int> image2) {
                          return image1.second > image2.second;
                        });
    } else {
      std::sort(ordered_images.begin(), ordered_images.end(),
                [](const std::pair<int, int> image1,
                   const std::pair<int, int> image2) {
                  return image1.second > image2.second;
                });
    }

    overlapping_images[image_idx1].reserve(eff_num_images);
    for (size_t i = 0; i < eff_num_images; ++i) {
      overlapping_images[image_idx1].push_back(ordered_images[i].first);
    }
    return true;
  };
  auto SelectSpatialNeighborViews = [&](size_t image_idx1) {
    size_t i, j;
    const mvs::Image& ref_image = images.at(image_idx1);
    Eigen::Vector3f ref_ray = Eigen::Map<const Eigen::RowMatrix3f>(ref_image.GetR()).row(2);
    Eigen::Vector3d& ref_C = proj_centers[image_idx1];

    std::vector<std::pair<int, mvs::Image> > scored_images;
    for (i = 0; i < num_image; ++i) {
      if (i == image_idx1) {
        continue;
      }
      const mvs::Image& src_image = images.at(i);
      Eigen::Vector3f src_ray = Eigen::Map<const Eigen::RowMatrix3f>(src_image.GetR()).row(2);
      const float dist = (proj_centers[i] - ref_C).norm();
      if (dist < min_dist || dist > max_dist) {
        continue;
      }
      const float angle = std::acos(ref_ray.dot(src_ray));
      if (angle < min_angle || angle > max_angle) {
        continue;
      }
      scored_images.emplace_back(i, src_image);
    }

    struct ViewScore {
        uint32_t view_idx;
        float score;
    };
    std::vector<ViewScore> view_scores;
    view_scores.reserve(scored_images.size());
    for (i = 0; i < scored_images.size(); ++i) {
      const mvs::Image& src_image = scored_images.at(i).second;

      const float* src_R = src_image.GetR();
      Eigen::Vector3d src_C = proj_centers.at(scored_images.at(i).first);
      Eigen::Vector3f src_ray = Eigen::Map<const Eigen::RowMatrix3f>(src_R).row(2);

      const float cos_angle = ref_ray.dot(src_ray);
      const float wangle = 1.1f - cos_angle;
      const float dist = (src_C - ref_C).norm();
      const float wdist = std::max(dist / m_dist, 1.0f);

      ViewScore view_score;
      view_score.view_idx = scored_images.at(i).first;
      view_score.score = wangle * wdist;
      view_scores.emplace_back(view_score);
    }

    std::sort(view_scores.begin(), view_scores.end(), 
        [&](const ViewScore s1, const ViewScore s2) {
        return s1.score < s2.score;
    });

    int eff_num_images = std::min(view_scores.size(), max_num_images);
    if (eff_num_images < 1) {
        return false;
    }

    overlapping_images[image_idx1].push_back(image_idx1);
    while(overlapping_images[image_idx1].size() <= eff_num_images &&
          !view_scores.empty()) {
      size_t k;
      for (k = 0; k < view_scores.size(); ++k) {
        int next_image_id = view_scores[k].view_idx;
        Eigen::Vector3d& next_C = proj_centers.at(next_image_id);
        bool spurious_image = false;
        for (auto overlap_image_id : overlapping_images[image_idx1]) {
          Eigen::Vector3d& overlap_C = proj_centers.at(overlap_image_id);
          double dist = (next_C - overlap_C).norm();
          if (dist < min_dist) {
            spurious_image = true;
            break;
          }
        }
        if (!spurious_image) {
          overlapping_images[image_idx1].push_back(next_image_id);
          view_scores.erase(view_scores.begin() + k);
          break;
        }
      }
      if (k >= view_scores.size()) {
        break;
      }
    }
    overlapping_images[image_idx1].erase(overlapping_images[image_idx1].begin());
    return true;
  };
  
  auto ProcessImage = [&](size_t i) {
    if (!SelectNeighborViews(i)) {
      SelectSpatialNeighborViews(i);
    }
    std::cout << std::flush << StringPrintf("\rProcess Image %.1f%", (i + 1) * 100.0 / num_image);
  };

  const int num_eff_threads = GetEffectiveNumThreads(-1);
  std::unique_ptr<ThreadPool> thread_pool;
  thread_pool.reset(new ThreadPool(std::min(num_eff_threads, num_image)));

  std::cout << "ProcessImage..." << std::endl;
  for (size_t image_idx = 0; image_idx < num_image; ++image_idx) {
    if (unique_image_idxs.find(image_idx) == unique_image_idxs.end()){
      continue;
    }
    thread_pool->AddTask(ProcessImage, image_idx);
    // ProcessImage(image_idx);
  }
  thread_pool->Wait();

  std::cout << std::endl;
  return overlapping_images;
}

const std::vector<std::vector<int>>& Model::GetMaxOverlappingImagesFromPMVS()
    const {
  return pmvs_vis_dat_;
}

std::vector<std::pair<float, float>> Model::ComputeDepthRanges() const {
  std::vector<std::pair<float, float>> depth_ranges(images.size());
  auto ComputeDepthRange = [&](int image_idx) {
      const auto& image = images.at(image_idx);
      Eigen::RowMatrix3f R(image.GetR());
      Eigen::Vector3f t(image.GetT());
      depth_ranges[image_idx] = ComputeDepthRangeFromBBox(bb_, R, t);
  };

  const int num_eff_threads = GetEffectiveNumThreads(-1);
  std::unique_ptr<ThreadPool> thread_pool;
  thread_pool.reset(new ThreadPool(num_eff_threads));

  for (size_t image_idx = 0; image_idx < depth_ranges.size(); ++image_idx) {
    thread_pool->AddTask(ComputeDepthRange, image_idx);
  }
  thread_pool->Wait();

  return depth_ranges;
}

std::vector<std::unordered_map<int, int>> Model::ComputeSharedPoints() const {
  constexpr int lock_batch = 32;
  std::vector<std::unordered_map<int, int>> shared_points(images.size());
  std::vector<std::mutex> locks(images.size() / lock_batch + 1);

  #pragma omp parallel
  {
    std::vector<std::vector<int>> _shared_points(images.size());

    #pragma omp for schedule(dynamic, 1024)
    for (int64_t point_idx = 0; point_idx < points.size(); point_idx++) {
      const auto& point = points[point_idx];
      for (size_t i = 0; i < point.track.size(); ++i) { 
        const int image_idx1 = point.track[i];
        for (size_t j = 0; j < i; ++j) {
          const int image_idx2 = point.track[j];
          if (image_idx1 != image_idx2) {
            _shared_points.at(image_idx1).emplace_back(image_idx2);
            _shared_points.at(image_idx2).emplace_back(image_idx1);
          }
        }
      }
    }

    for (size_t image_idx1 = 0; image_idx1 < images.size(); image_idx1++) {
      if (_shared_points[image_idx1].size()) {
        locks[image_idx1 / lock_batch].lock();
        for (const auto & image_idx2 : _shared_points[image_idx1]) {
          shared_points.at(image_idx1)[image_idx2] += 1;
        }
        locks[image_idx1 / lock_batch].unlock();

        _shared_points[image_idx1].clear();
        _shared_points[image_idx1].shrink_to_fit();
      }
    }
  }
  return shared_points;
}

std::vector<std::unordered_map<int, float>> Model::ComputeTriangulationAngles(
    const float percentile) const {
  std::vector<Eigen::Vector3d> proj_centers(images.size());
  for (size_t image_idx = 0; image_idx < images.size(); ++image_idx) {
    const auto& image = images[image_idx];
    Eigen::Vector3f C;
    ComputeProjectionCenter(image.GetR(), image.GetT(), C.data());
    proj_centers[image_idx] = C.cast<double>();
  }

  constexpr int lock_batch = 32;
  std::vector<std::mutex> locks(images.size() / lock_batch + 1);
  std::vector<std::unordered_map<int, std::vector<float>>>  all_triangulation_angles(images.size());
  #pragma omp parallel
  {
    std::vector<std::vector<std::pair<int, float>>> _all_triangulation_angles(images.size());

    #pragma omp for schedule(dynamic, 1024)
    for (int64_t point_idx = 0; point_idx < points.size(); point_idx++) {
      const auto& point = points[point_idx];
      for (size_t i = 0; i < point.track.size(); ++i) {
        const int image_idx1 = point.track[i];
        for (size_t j = 0; j < i; ++j) {
          const int image_idx2 = point.track[j];
          if (image_idx1 != image_idx2) {
            const float angle = CalculateTriangulationAngle(
                proj_centers.at(image_idx1), proj_centers.at(image_idx2),
                Eigen::Vector3d(point.x, point.y, point.z));
            _all_triangulation_angles.at(image_idx1).emplace_back(image_idx2, angle);
            _all_triangulation_angles.at(image_idx2).emplace_back(image_idx1, angle);
          }
        }
      }
    }

    for (size_t image_idx = 0; image_idx < images.size(); image_idx++) {
      if (_all_triangulation_angles[image_idx].size()) {
        locks[image_idx / lock_batch].lock();
        for (auto & item : _all_triangulation_angles[image_idx]) {
          all_triangulation_angles[image_idx][item.first].emplace_back(item.second);
        }
        locks[image_idx / lock_batch].unlock();

        _all_triangulation_angles[image_idx].clear();
        _all_triangulation_angles[image_idx].shrink_to_fit();
      }
    }
  }

  const int num_eff_threads = GetEffectiveNumThreads(-1);
  std::unique_ptr<ThreadPool> thread_pool;
  thread_pool.reset(new ThreadPool(num_eff_threads));

  std::vector<std::unordered_map<int, float>> triangulation_angles(images.size());
  auto CalcTriAngles = [&](int image_idx) {
    const auto& overlapping_images = all_triangulation_angles[image_idx];
    for (const auto& image : overlapping_images) {
      triangulation_angles[image_idx].emplace(
          image.first, Percentile(image.second, percentile));
    }
  };

  for (size_t image_idx = 0; image_idx < all_triangulation_angles.size();
       ++image_idx) {
    thread_pool->AddTask(CalcTriAngles, image_idx);
  }
  thread_pool->Wait();

  return triangulation_angles;
}

float Model::ComputeNthDistance(float nth_factor = 0.75) const {

  std::vector<Eigen::Vector3d> proj_centers(images.size());
  #pragma omp parallel for
  for (size_t image_idx = 0; image_idx < images.size(); ++image_idx) {
    const auto& image = images[image_idx];
    Eigen::Vector3f C;
    ComputeProjectionCenter(image.GetR(), image.GetT(), C.data());
    proj_centers[image_idx] = C.cast<double>();
  }

  float mean_dist = -1.0f;
#if 1
  std::vector<float> point_dist;
  point_dist.resize(points.size());
  #pragma omp parallel for
  for(int64_t point_idx = 0; point_idx < points.size(); point_idx++){
    double sum_dist = 0;
    const auto& point = points[point_idx];
    for (size_t i = 0; i < point.track.size(); ++i){
      const int image_idx = point.track[i];
      sum_dist += (Eigen::Vector3d(point.x, point.y, point.z) - proj_centers[image_idx]).norm();
    }
    point_dist[point_idx] = (float)sum_dist / (float)point.track.size();
  }

  // int median_idx = point_dist.size() /2;
  int nth_idx = point_dist.size() * nth_factor;
  std::nth_element(point_dist.begin(), point_dist.begin() + nth_idx, point_dist.end());
  mean_dist = point_dist.at(nth_idx);

#else
  double sum_dist = 0;
  double sum_track = 0;
  for(int64_t point_idx = 0; point_idx < points.size(); point_idx++){
    const auto& point = points[point_idx];
    for (size_t i = 0; i < point.track.size(); ++i){
      const int image_idx = point.track[i];
      sum_dist += (Eigen::Vector3d(point.x, point.y, point.z) - proj_centers[image_idx]).norm();
      sum_track ++;
    }
  }  
  
  mean_dist = sum_dist / sum_track;
  std::cout << "\t=>Model Mean Distance: " << mean_dist << std::endl;
#endif

  return mean_dist;
}

float Model::ComputeNthDepthImage(const int ref_image_idx, float nth_factor) const{
  const image_t ref_image_id = image_idx_to_id_.at(ref_image_idx);
  const auto& mappoints = reconstruction_->MapPoints();
  const auto& ref_image = reconstruction_->Image(ref_image_id);
  const Eigen::Vector4d qvec = ref_image.Qvec();
  const Eigen::Vector3d tvec = ref_image.Tvec();

  std::vector<float> pnt_depths;
  std::size_t num_pnt2d = ref_image.Points2D().size();
  pnt_depths.reserve(num_pnt2d);
#pragma omp parallel for
  for (int i = 0; i < num_pnt2d; i++) {
    const auto& point2D = ref_image.Points2D().at(i);
    if (!point2D.HasMapPoint()) {
      continue;
    }

    mappoint_t mappoint_id = point2D.MapPointId();
    const auto& xyz = mappoints.at(mappoint_id).XYZ();
    Eigen::Vector3d proj_point3D =
        QuaternionRotatePoint(qvec, xyz) + tvec;

    #pragma omp critical
    {
      pnt_depths.push_back(proj_point3D.z());
    }
  }
  pnt_depths.shrink_to_fit();
  int nth_idx = pnt_depths.size() * nth_factor;
  std::nth_element(pnt_depths.begin(), pnt_depths.begin() + nth_idx, pnt_depths.end());

  return pnt_depths.at(nth_idx);
}

float Model::ComputeMeanAngularResolution() const {
  double sum_angular_res = 0;
  double num_samp = 0;
  for(size_t i = 0; i < images.size(); i++){
    const float width = images.at(i).GetWidth();
    const float height = images.at(i).GetHeight();
    const float fx = images.at(i).GetK()[0];
    const float fy = images.at(i).GetK()[4];
    float width_angular = 2.0f * std::atan(width / (2.0f * fx));
    float height_angular = 2.0f * std::atan(height / (2.0f * fy));

    float angular_res = 0.5f * width_angular / width + 0.5f * height_angular / height;

    sum_angular_res += (double)angular_res;
    num_samp++;
  }
  float mean_angular_res = (float)(sum_angular_res / num_samp);
  // std::cout << "\t=>Model Mean Angular Resolution: " << RAD2DEG(mean_angular_res) << " degree (" << mean_angular_res << " rad)" << std::endl;
  return mean_angular_res;
}

bool Model::ReadFromBundlerPMVS(const std::string& path) {
  const std::string bundle_file_path = JoinPaths(path, "bundle.rd.out");

  if (!ExistsFile(bundle_file_path)) {
    return false;
  }

  std::ifstream file(bundle_file_path);
  CHECK(file.is_open()) << bundle_file_path;

  // Header line.
  std::string header;
  std::getline(file, header);

  int num_images, num_points;
  file >> num_images >> num_points;

  images.reserve(num_images);
  for (int image_idx = 0; image_idx < num_images; ++image_idx) {
    const std::string image_name = StringPrintf("%08d.jpg", image_idx);
    const std::string image_path = JoinPaths(path, "visualize", image_name);

    float K[9] = {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f};
    file >> K[0];
    K[4] = K[0];

    Bitmap bitmap;
    CHECK(bitmap.Read(image_path));
    K[2] = bitmap.Width() / 2.0f;
    K[5] = bitmap.Height() / 2.0f;

    float k1, k2;
    file >> k1 >> k2;
    CHECK_EQ(k1, 0.0f);
    CHECK_EQ(k2, 0.0f);

    float R[9];
    for (size_t i = 0; i < 9; ++i) {
      file >> R[i];
    }
    for (size_t i = 3; i < 9; ++i) {
      R[i] = -R[i];
    }

    float T[3];
    file >> T[0] >> T[1] >> T[2];
    T[1] = -T[1];
    T[2] = -T[2];

    images.emplace_back(image_path, bitmap.Width(), bitmap.Height(), K, R, T);
    image_names_.push_back(image_name);
    image_name_to_idx_.emplace(image_name, image_idx);
  }

  points.resize(num_points);
  for (int point_id = 0; point_id < num_points; ++point_id) {
    auto& point = points[point_id];

    file >> point.x >> point.y >> point.z;

    int color[3];
    file >> color[0] >> color[1] >> color[2];

    int track_len;
    file >> track_len;
    point.track.resize(track_len);

    for (int i = 0; i < track_len; ++i) {
      int feature_idx;
      float imx, imy;
      file >> point.track[i] >> feature_idx >> imx >> imy;
      CHECK_LT(point.track[i], images.size());
    }
  }

  return true;
}

bool Model::ReadFromRawPMVS(const std::string& path) {
  const std::string vis_dat_path = JoinPaths(path, "vis.dat");
  if (!ExistsFile(vis_dat_path)) {
    return false;
  }

  for (int image_idx = 0;; ++image_idx) {
    const std::string image_name = StringPrintf("%08d.jpg", image_idx);
    const std::string image_path = JoinPaths(path, "visualize", image_name);

    if (!ExistsFile(image_path)) {
      break;
    }

    Bitmap bitmap;
    CHECK(bitmap.Read(image_path));

    const std::string proj_matrix_path =
        JoinPaths(path, "txt", StringPrintf("%08d.txt", image_idx));

    std::ifstream proj_matrix_file(proj_matrix_path);
    CHECK(proj_matrix_file.is_open()) << proj_matrix_path;

    std::string contour;
    proj_matrix_file >> contour;
    CHECK_EQ(contour, "CONTOUR");

    Eigen::Matrix3x4d P;
    for (int i = 0; i < 3; ++i) {
      proj_matrix_file >> P(i, 0) >> P(i, 1) >> P(i, 2) >> P(i, 3);
    }

    Eigen::Matrix3d K;
    Eigen::Matrix3d R;
    Eigen::Vector3d T;
    DecomposeProjectionMatrix(P, &K, &R, &T);

    // The COLMAP patch match algorithm requires that there is no skew.
    K(0, 1) = 0.0f;
    K(1, 0) = 0.0f;
    K(2, 0) = 0.0f;
    K(2, 1) = 0.0f;
    K(2, 2) = 1.0f;

    const Eigen::Matrix<float, 3, 3, Eigen::RowMajor> K_float = K.cast<float>();
    const Eigen::Matrix<float, 3, 3, Eigen::RowMajor> R_float = R.cast<float>();
    const Eigen::Vector3f T_float = T.cast<float>();

    images.emplace_back(image_path, bitmap.Width(), bitmap.Height(),
                        K_float.data(), R_float.data(), T_float.data());
    image_names_.push_back(image_name);
    image_name_to_idx_.emplace(image_name, image_idx);
  }

  std::ifstream vis_dat_file(vis_dat_path);
  CHECK(vis_dat_file.is_open()) << vis_dat_path;

  std::string visdata;
  vis_dat_file >> visdata;
  CHECK_EQ(visdata, "VISDATA");

  int num_images;
  vis_dat_file >> num_images;
  CHECK_GE(num_images, 0);
  CHECK_EQ(num_images, images.size());

  pmvs_vis_dat_.resize(num_images);
  for (int i = 0; i < num_images; ++i) {
    int image_idx;
    vis_dat_file >> image_idx;
    CHECK_GE(image_idx, 0);
    CHECK_LT(image_idx, num_images);

    int num_visible_images;
    vis_dat_file >> num_visible_images;

    auto& visible_image_idxs = pmvs_vis_dat_[image_idx];
    visible_image_idxs.reserve(num_visible_images);

    for (int j = 0; j < num_visible_images; ++j) {
      int visible_image_idx;
      vis_dat_file >> visible_image_idx;
      CHECK_GE(visible_image_idx, 0);
      CHECK_LT(visible_image_idx, num_images);
      if (visible_image_idx != image_idx) {
        visible_image_idxs.push_back(visible_image_idx);
      }
    }
  }

  return true;
}

}  // namespace mvs
}  // namespace sensemap
