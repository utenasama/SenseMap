//Copyright (c) 2019, SenseTime Group.
//All rights reserved.

#include "point2d.h"

namespace sensemap {
    Point2D::Point2D()
        : in_mask_(false), 
          in_overlap_(false),
          scale_(0.0f),
          xy_(Eigen::Vector2d::Zero()),
          mappoint_id_(kInvalidMapPointId),
          depth_weight_(0),
          covariance_(-1),
          depth_(0) {}
}
