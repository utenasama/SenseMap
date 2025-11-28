//Copyright (c) 2019, SenseTime Group.
//All rights reserved.

#ifndef SENSEMAP_MVS_DEPTH_MAP_H_
#define SENSEMAP_MVS_DEPTH_MAP_H_

#include <string>
#include <vector>

#include "util/mat.h"
#include "util/bitmap.h"

namespace sensemap {
namespace mvs {

class DepthMap : public Mat<float> {
 public:
  DepthMap();
  DepthMap(const size_t width, const size_t height, const float depth_min,
           const float depth_max);
  DepthMap(const Mat<float>& mat, const float depth_min, const float depth_max);

  inline float GetDepthMin() const;
  inline float GetDepthMax() const;

  inline float Get(const size_t row, const size_t col) const;

  void Rescale(const float factor);
  void Rescale(const float factor_x, const float factor_y);
  void Downsize(const size_t max_width, const size_t max_height);

  Bitmap ToBitmap(const float min_percentile = 2, const float max_percentile = 98) const;

 private:
  float depth_min_ = -1.0f;
  float depth_max_ = -1.0f;
};

////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////

float DepthMap::GetDepthMin() const { return depth_min_; }

float DepthMap::GetDepthMax() const { return depth_max_; }

float DepthMap::Get(const size_t row, const size_t col) const {
  return data_.at(row * width_ + col);
}

} // namespace mvs
} // namespace sensemap

#endif