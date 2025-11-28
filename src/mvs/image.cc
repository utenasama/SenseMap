//Copyright (c) 2019, SenseTime Group.
//All rights reserved.

#include <Eigen/Core>

#include "util/logging.h"

#include "mvs/utils.h"
#include "mvs/image.h"

namespace sensemap {
namespace mvs {

using namespace utility;

Image::Image() {}

Image::Image(const std::string& path, const size_t width, const size_t height,
             const float* K, const float* R, const float* T, 
             const bool is_from_rig, const bool is_from_lidar)
    : path_(path), width_(width), height_(height), is_rig(is_from_rig), is_lidar(is_from_lidar) {
  memcpy(K_, K, 9 * sizeof(float));
  memcpy(R_, R, 9 * sizeof(float));
  memcpy(T_, T, 3 * sizeof(float));

  C_[0] = -(R_[0] * T_[0] + R_[3] * T_[1] + R_[6] * T_[2]);
  C_[1] = -(R_[1] * T_[0] + R_[4] * T_[1] + R_[7] * T_[2]);
  C_[2] = -(R_[2] * T_[0] + R_[5] * T_[1] + R_[8] * T_[2]);

  ComposeProjectionMatrix(K_, R_, T_, P_);
  ComposeInverseProjectionMatrix(K_, R_, T_, inv_P_);
}

Image::~Image() {
  bitmap_.Deallocate();
}

void Image::SetBitmap(const Bitmap& bitmap) {
  bitmap_ = bitmap;
  CHECK_EQ(width_, bitmap_.Width());
  CHECK_EQ(height_, bitmap_.Height());
}

void Image::Rescale(const float factor) { Rescale(factor, factor); }

void Image::Rescale(const float factor_x, const float factor_y) {
  if (factor_x - 1 < 1e-6 && 1 - factor_x < 1e-6 && 
      factor_y - 1 < 1e-6 && 1 - factor_y < 1e-6 ){
    return;
  }

  const size_t new_width = std::round(width_ * factor_x);
  const size_t new_height = std::round(height_ * factor_y);

  if (bitmap_.Data() != nullptr) {
    bitmap_.Rescale(new_width, new_height);
  }

  const float scale_x = new_width / static_cast<float>(width_);
  const float scale_y = new_height / static_cast<float>(height_);
  K_[0] *= scale_x;
  K_[2] *= scale_x;
  K_[4] *= scale_y;
  K_[5] *= scale_y;
  ComposeProjectionMatrix(K_, R_, T_, P_);
  ComposeInverseProjectionMatrix(K_, R_, T_, inv_P_);

  width_ = new_width;
  height_ = new_height;
}

void Image::Downsize(const size_t max_width, const size_t max_height) {
  if (width_ <= max_width && height_ <= max_height) {
    return;
  }
  const float factor_x = static_cast<float>(max_width) / width_;
  const float factor_y = static_cast<float>(max_height) / height_;
  Rescale(std::min(factor_x, factor_y));
}

}  // namespace mvs
}  // namespace sensemap
