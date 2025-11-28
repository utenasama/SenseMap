//Copyright (c) 2019, SenseTime Group.
//All rights reserved.

#include <algorithm>

#include "image.h"
#include "base/pose.h"
#include "base/projection.h"
#include "util/logging.h"

namespace sensemap {

namespace {
    static const double kNaN = std::numeric_limits<double>::quiet_NaN();
}

const int Image::kNumMapPointVisibilityPyramidLevels = 6;

Image::Image()
    : image_id_(kInvalidImageId),
      label_id_(kInvalidLabelId),
      name_(""),
      camera_id_(kInvalidCameraId),
      registered_(false),
      has_pose_(false),
      keyframe_(false),
      num_MapPoints_(0),
      num_observations_(0),
      num_correspondences_(0),
      num_visible_MapPoints_(0),
      qvec_(1.0, 0.0, 0.0, 0.0),
      tvec_(0.0, 0.0, 0.0),
      qvec_prior_(kNaN, kNaN, kNaN, kNaN),
      tvec_prior_(kNaN, kNaN, kNaN),
      prior_inlier_(false),
      prior_qvec_good_(false),
      sim3_pose_(){
}


void Image::SetUp(const Camera & camera) {
    CHECK_EQ(camera_id_, camera.CameraId());
    mappoint_visibility_pyramid_ = VisibilityPyramid(kNumMapPointVisibilityPyramidLevels, camera.Width(), camera.Height());
}

void Image::TearDown() {
    mappoint_visibility_pyramid_ = VisibilityPyramid(0, 0, 0);
}

void Image::SetPoints2D(const std::vector<Eigen::Vector2d>& points) {
    // CHECK(points2D_.empty());
    points2D_.resize(points.size());
    num_correspondences_have_mappoint_.resize(points.size(), 0);
    for (point2D_t point2D_idx = 0; point2D_idx < points.size(); ++point2D_idx) {
        points2D_[point2D_idx].SetXY(points[point2D_idx]);
    }
}

void Image::SetPoints2D(const std::vector<class Point2D>& points) {
    // CHECK(points2D_.empty());
    points2D_ = points;
    num_correspondences_have_mappoint_.resize(points.size(), 0);
}

void Image::SetPoints2D(const FeatureKeypoints& keypoints) {
    // CHECK(points2D_.empty());
    num_correspondences_have_mappoint_.resize(keypoints.size(), 0);
    points2D_.resize(keypoints.size());
    for (size_t i = 0; i < keypoints.size(); ++i) {
        points2D_[i].SetXY(Eigen::Vector2d(keypoints[i].x, keypoints[i].y));
        points2D_[i].SetScale(keypoints[i].ComputeScale());
    }
}

void Image::SetLocalImageIndices( const std::vector<uint32_t>& indices){
    local_image_indices_ = indices;
}

std::vector<point2D_t> Image::AddPoints2D(const std::vector<class Point2D>& points){
    size_t new_points_size = points.size();
    size_t old_points_size = points2D_.size();
    size_t total_points_size = old_points_size + new_points_size;

    points2D_.resize(total_points_size);
    num_correspondences_have_mappoint_.resize(total_points_size);

    std::vector<point2D_t> new_point2d_ids;
    for(size_t i=0; i<new_points_size;++i){
        new_point2d_ids.emplace_back(old_points_size+i);
        points2D_[old_points_size+i]=points[i];
        num_correspondences_have_mappoint_[old_points_size+i] = 0;
    }
    return new_point2d_ids;
}

std::vector<point2D_t> Image::AddPoints2D(const std::vector<class Point2D>& points, const std::vector<uint32_t>& indices){
    CHECK_EQ(points.size(), indices.size());

    size_t new_points_size = points.size();
    size_t old_points_size = points2D_.size();
    size_t total_points_size = old_points_size + new_points_size;

    points2D_.resize(total_points_size);
    num_correspondences_have_mappoint_.resize(total_points_size);

    std::vector<point2D_t> new_point2d_ids;
    for(size_t i=0; i<new_points_size;++i){
        new_point2d_ids.emplace_back(old_points_size+i);
        points2D_[old_points_size+i]=points[i];
        num_correspondences_have_mappoint_[old_points_size+i] = 0;
    }

    // Append local image indices
    local_image_indices_.insert(local_image_indices_.end(), indices.begin(),indices.end());

    return new_point2d_ids;
}

void Image::SetMapPointForPoint2D(const point2D_t point2D_idx,
                                 const mappoint_t mappoint_id) {
    CHECK_NE(mappoint_id, kInvalidMapPointId);
    class Point2D& point2D = points2D_.at(point2D_idx);
    point2D.SetMask(true);
    if (!point2D.HasMapPoint()) {
        num_MapPoints_ += 1;
    }
    point2D.SetMapPointId(mappoint_id);
}

void Image::ResetMapPointForPoint2D(const point2D_t point2D_idx) {
    class Point2D& point2D = points2D_.at(point2D_idx);
    point2D.SetMask(false);
    if (point2D.HasMapPoint()) {
        point2D.SetMapPointId(kInvalidMapPointId);
        num_MapPoints_ -= 1;
    }
}

bool Image::HasMapPoint(const mappoint_t mappoint_id) const {
    return std::find_if(points2D_.begin(), points2D_.end(),
                        [mappoint_id](const class Point2D& point2D) {
                          return point2D.MapPointId() == mappoint_id;
                        }) != points2D_.end();
}

void Image::IncrementCorrespondenceHasMapPoint(const point2D_t point2D_idx) {
    const class Point2D & point2D = points2D_.at(point2D_idx);

    num_correspondences_have_mappoint_[point2D_idx] += 1;
    if (num_correspondences_have_mappoint_[point2D_idx] == 1) {
        num_visible_MapPoints_ += 1;
    }

    mappoint_visibility_pyramid_.SetPoint(point2D.X(), point2D.Y());

    assert(num_visible_MapPoints_ <= num_observations_);
}

void Image::DecrementCorrespondenceHasMapPoint(const point2D_t point2D_idx) {
    const class Point2D & point2D = points2D_.at(point2D_idx);

    num_correspondences_have_mappoint_[point2D_idx] -= 1;
    if (num_correspondences_have_mappoint_[point2D_idx] == 0) {
        num_visible_MapPoints_ -= 1;
    }

    mappoint_visibility_pyramid_.ResetPoint(point2D.X(), point2D.Y());

    assert(num_visible_MapPoints_ <= num_observations_);
}

void Image::NormalizeQvec() { qvec_ = NormalizeQuaternion(qvec_); }

void Image::NormalizePriorQvec() { qvec_prior_ = NormalizeQuaternion(qvec_prior_); }

Eigen::Matrix3x4d Image::ProjectionMatrix() const {
   return ComposeProjectionMatrix(qvec_, tvec_);
}

Eigen::Matrix3x4d Image::InverseProjectionMatrix() const {
   return InvertProjectionMatrix(ComposeProjectionMatrix(qvec_, tvec_));
}

Eigen::Matrix3d Image::RotationMatrix() const {
   return QuaternionToRotationMatrix(qvec_);
}

Eigen::Vector3d Image::ProjectionCenter() const {
   return ProjectionCenterFromPose(qvec_, tvec_);
}

Eigen::Vector3d Image::ViewingDirection() const {
   return RotationMatrix().row(2);
}


} // namespace sensemap

