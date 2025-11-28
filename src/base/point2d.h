//Copyright (c) 2019, SenseTime Group.
//All rights reserved.

#ifndef SENSEMAP_BASE_POINT2D_H_
#define SENSEMAP_BASE_POINT2D_H_

#include <Eigen/Core>

#include "util/types.h"
#include "util/alignment.h"
namespace sensemap {

// 2D point class corresponds to a feature in an image. It may or may not have a
// corresponding map point if it is part of a triangulated track.
class Point2D {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Point2D();

    // The coordinate in image space in pixels.
    inline const Eigen::Vector2d& XY() const;
    inline Eigen::Vector2d& XY();
    inline double X() const;
    inline double Y() const;

    inline void SetXY(const Eigen::Vector2d& xy);

    inline float Scale();
    inline float Scale() const;
    inline void SetScale(const float scale);
    inline bool InMask();
    inline bool InMask() const;
    inline void SetMask(const bool in_mask); 
    inline bool InOverlap();
    inline bool InOverlap() const;
    inline void SetOverlap(const bool in_overlap);

    inline float& DepthWeight();
    inline float DepthWeight() const;

    inline float& Covariance();
    inline float Covariance() const;

    inline float& Depth();
    inline float Depth() const;

    // The identifier of the observed map point. If the image point does not
    // observe a map point, the identifier is `kInvalidMapPointid`.
    inline mappoint_t MapPointId() const;
    inline bool HasMapPoint() const;
    inline void SetMapPointId(const mappoint_t mappoint_id);

private:
    // the measurement from depth map.
    float depth_;

    // the weight of depth in [0, 1]
    float depth_weight_;

    // the covariance of depth at position xy_.
    float covariance_;

    // Whether to triangulate map point.
    bool in_mask_;

    // Whether the point locates at overlap region of two intra-view one frame,
    // valid for camera-rig model.
    bool in_overlap_;

    // the feature scale.
    float scale_;

    // The image coordinates in pixels, starting at upper left corner with 0.
    Eigen::Vector2d xy_;

    // The identifier of the map point. If the 2D point is not part of a map point
    // track the identifier is `kInvalidMapPointid` and `HasMapPoint() = false`.
    mappoint_t mappoint_id_;
};

/////////////////////////////////////////////////////////////////////////////////////
// Implementation
/////////////////////////////////////////////////////////////////////////////////////

const Eigen::Vector2d& Point2D::XY() const { return xy_; }

Eigen::Vector2d& Point2D::XY() { return xy_; }

double Point2D::X() const { return xy_.x(); }

double Point2D::Y() const { return xy_.y(); }

void Point2D::SetXY(const Eigen::Vector2d& xy) { xy_ = xy; }

float Point2D::Scale() { return scale_; }

float Point2D::Scale() const { return scale_; }

void Point2D::SetScale(const float scale) { scale_ = scale; }

bool Point2D::InMask() { return in_mask_; }

bool Point2D::InMask() const { return in_mask_; }

void Point2D::SetMask(const bool in_mask) { in_mask_ = in_mask; }

bool Point2D::InOverlap() { return in_overlap_; }

bool Point2D::InOverlap() const { return in_overlap_; }

void Point2D::SetOverlap(const bool in_overlap) { in_overlap_ = in_overlap; }

float& Point2D::Covariance() { return covariance_; }

float Point2D::Covariance() const { return covariance_; }

float& Point2D::DepthWeight() { return depth_weight_; }

float Point2D::DepthWeight() const { return depth_weight_; }

float& Point2D::Depth() { return depth_; }

float Point2D::Depth() const { return depth_; }

mappoint_t Point2D::MapPointId() const { return mappoint_id_; }

bool Point2D::HasMapPoint() const { return mappoint_id_ != kInvalidMapPointId; }

void Point2D::SetMapPointId(const mappoint_t mappoint_id) { mappoint_id_ = mappoint_id; }

} // namespace sensemap

#endif //SENSEMAP_BASE_POINT2D_H
