//Copyright (c) 2019, SenseTime Group.
//All rights reserved.

#ifndef SENSEMAP_BASE_MAPPOINT_H_
#define SENSEMAP_BASE_MAPPOINT_H_

#include <Eigen/Core>

#include "util/types.h"
#include "util/logging.h"
#include "base/track.h"

namespace sensemap {

// 2D point class corresponds to a feature in an image. It may or may not have a
// corresponding Map Point if it is part of a triangulated track.
class MapPoint {
public:
    MapPoint();
    
    // The point coordinate in world space.
    inline const Eigen::Vector3d& XYZ() const;
    inline Eigen::Vector3d& XYZ();
    inline double XYZ(const size_t idx) const;
    inline double& XYZ(const size_t idx);
    inline double X() const;
    inline double Y() const;
    inline double Z() const;
    inline void SetXYZ(const Eigen::Vector3d& xyz);

    // The RGB color of the point.
    inline const Eigen::Vector3ub& Color() const;
    inline Eigen::Vector3ub& Color();
    inline uint8_t Color(const size_t idx) const;
    inline uint8_t& Color(const size_t idx);
    inline void SetColor(const Eigen::Vector3ub& color);

    // The mean reprojection error in image space.
    inline double Error() const;
    inline bool HasError() const;
    inline void SetError(const double error);

    inline const class Track& Track() const;
    inline class Track& Track();
    inline void SetTrack(class Track track);

    inline void SetTriAngle(double tri_angle);
    inline double TriAngle() const;

    inline void SetCreateTime(uint64_t create_time);
    inline uint64_t CreateTime() const;

private:
    // The 3D position of the point.
    Eigen::Vector3d xyz_;

    // The color of the point in the range [0, 255].
    Eigen::Vector3ub color_;

    // The mean reprojection error in pixels.
    double error_;

    // The max tri angle in degree
    double tri_angle_;

    // The track of the point as a list of image observations.
    class Track track_;

    uint64_t create_time_= 0;
};

//////////////////////////////////////////////////////////////////////////////////////////
// Implementation
//////////////////////////////////////////////////////////////////////////////////////////

const Eigen::Vector3d & MapPoint::XYZ() const { return xyz_; }

Eigen::Vector3d& MapPoint::XYZ() { return xyz_; }

double MapPoint::XYZ(const size_t idx) const { return xyz_(idx); }

double& MapPoint::XYZ(const size_t idx) { return xyz_(idx); }

double MapPoint::X() const { return xyz_.x(); }

double MapPoint::Y() const { return xyz_.y(); }

double MapPoint::Z() const { return xyz_.z(); }

void MapPoint::SetXYZ(const Eigen::Vector3d& xyz) { xyz_ = xyz; }

const Eigen::Vector3ub& MapPoint::Color() const { return color_; }

Eigen::Vector3ub& MapPoint::Color() { return color_; }

uint8_t MapPoint::Color(const size_t idx) const { return color_(idx); }

uint8_t& MapPoint::Color(const size_t idx) { return color_(idx); }

void MapPoint::SetColor(const Eigen::Vector3ub& color) { color_ = color; }

double MapPoint::Error() const { return error_; }

bool MapPoint::HasError() const { return error_ != -1.0; }

void MapPoint::SetError(const double error) { error_ = error; }

const class Track& MapPoint::Track() const { return track_; }

class Track& MapPoint::Track() { return track_; }

void MapPoint::SetTrack(class Track track) { track_ = std::move(track); }

void MapPoint::SetTriAngle(double tri_angle) {tri_angle_ = tri_angle;}

double MapPoint::TriAngle() const {return tri_angle_;}

void MapPoint::SetCreateTime(uint64_t create_time) { create_time_ = create_time; }

uint64_t MapPoint::CreateTime() const { return create_time_; }

} // namespace sensemap

#endif //SENSEMAP_BASE_MAPPOINT_H_
