//Copyright (c) 2019, SenseTime Group.
//All rights reserved.

#include <Eigen/Dense>

#include "mvs/normal_map.h"

const float decode_factor = 32767.f;
const float decode_inv_factor = 1.0f / decode_factor;

namespace sensemap {
namespace mvs {

NormalMap::NormalMap() : Mat<float>(0, 0, 3) {}

NormalMap::NormalMap(const size_t width, const size_t height)
    : Mat<float>(width, height, 3) {}

NormalMap::NormalMap(const Mat<float>& mat)
    : Mat<float>(mat.GetWidth(), mat.GetHeight(), mat.GetDepth()) {
  CHECK_EQ(mat.GetDepth(), 3);
  data_ = mat.GetData();
}

void NormalMap::Rescale(const float factor) { Rescale(factor, factor); }

void NormalMap::Rescale(const float factor_x, const float factor_y) {
  if (width_ * height_ == 0) {
    return;
  }

  if (factor_x - 1 < 1e-6 && 1 - factor_x < 1e-6 && 
      factor_y - 1 < 1e-6 && 1 - factor_y < 1e-6 ){
    return;
  }

  const size_t new_width = std::round(width_ * factor_x);
  const size_t new_height = std::round(height_ * factor_y);
  std::vector<float> new_data(new_width * new_height * 3);

  // Resample the normal map.
  for (size_t d = 0; d < 3; ++d) {
    const size_t offset = d * width_ * height_;
    const size_t new_offset = d * new_width * new_height;
    // DownsampleImage(data_.data() + offset, height_, width_, new_height,
    //                 new_width, new_data.data() + new_offset);
    InterpolateImage(data_.data() + offset, height_, width_, new_height, 
                     new_width, new_data.data() + new_offset);
  }

  data_ = new_data;
  width_ = new_width;
  height_ = new_height;

  data_.shrink_to_fit();

  // Re-normalize the normal vectors.
#pragma omp parallel for
  for (size_t r = 0; r < height_; ++r) {
    for (size_t c = 0; c < width_; ++c) {
      Eigen::Vector3f normal(Get(r, c, 0), Get(r, c, 1), Get(r, c, 2));
      const float squared_norm = normal.squaredNorm();
      if (squared_norm > 0) {
        normal /= std::sqrt(squared_norm);
      }
      Set(r, c, 0, normal(0));
      Set(r, c, 1, normal(1));
      Set(r, c, 2, normal(2));
    }
  }
}

void NormalMap::Downsize(const size_t max_width, const size_t max_height) {
  if (height_ <= max_height && width_ <= max_width) {
    return;
  }
  const float factor_x = static_cast<float>(max_width) / width_;
  const float factor_y = static_cast<float>(max_height) / height_;
  Rescale(std::min(factor_x, factor_y));
}

Bitmap NormalMap::ToBitmap() const {
  CHECK_GT(width_, 0);
  CHECK_GT(height_, 0);
  CHECK_EQ(depth_, 3);

  Bitmap bitmap;
  bitmap.Allocate(width_, height_, true);

#pragma omp parallel for
  for (size_t y = 0; y < height_; ++y) {
    for (size_t x = 0; x < width_; ++x) {
      float normal[3];
      GetSlice(y, x, normal);
      if (normal[0] != 0 || normal[1] != 0 || normal[2] != 0) {
        const BitmapColor<float> color(127.5f * (-normal[0] + 1),
                                       127.5f * (-normal[1] + 1),
                                       -255.0f * normal[2]);
        bitmap.SetPixel(x, y, color.Cast<uint8_t>());
      } else {
        bitmap.SetPixel(x, y, BitmapColor<uint8_t>(0));
      }
    }
  }

  return bitmap;
}

void NormalMap::Read(const std::string& path) {
  #if 1
  MatXui mat;
  mat.Read(path);

  Decode(mat);

  #else
  Mat<float>::Read(path);
  #endif
}

void NormalMap::Write(const std::string& path) const {
  #if 1
  const double factor = 32767;
  MatXui mat = Encode();
  mat.Write(path);
  #else
  Mat<float>::Write(path);
  #endif
}

void NormalMap::Decode(const MatXui &mat){
  CHECK_EQ(mat.GetDepth(), 1);
  width_ = mat.GetWidth();
  height_ = mat.GetHeight();
  depth_ = 3;
  data_.resize(width_ * height_ * depth_);

#pragma omp parallel for
  for (int y = 0; y < height_; ++y) {
    for (int x = 0; x < width_; ++x) {
      uint32_t code = mat.Get(y, x);
      uint32_t iq1 = (code >> 16);
      uint32_t iq2 = (code & 0x0000ffff);
      float q1 = iq1 * decode_inv_factor - 1;
      float q2 = iq2 * decode_inv_factor - 1;
      float s = q1 * q1 + q2 * q2;

      float normal[3];
      normal[0] = 1.0 - 2.0 * s;
      normal[1] = 2 * q1 * std::sqrt(1 - s);
      normal[2] = 2 * q2 * std::sqrt(1 - s);
      SetSlice(y, x, normal);
    }
  }
}

MatXui NormalMap::Encode() const {
  MatXui mat(width_, height_, 1);
#pragma omp parallel for
  for (int y = 0; y < height_; ++y) {
    for (int x = 0; x < width_; ++x) {
      float normal[3];
      GetSlice(y, x, normal);

      float s = (1 - normal[0]) * 0.5;
      float m = 1.0f / std::sqrt(1 - s);
      float q1 = normal[1] * 0.5 * m;
      float q2 = normal[2] * 0.5 * m;
      q1 = std::min(0.999999f, std::max(q1, -0.999999f));
      q2 = std::min(0.999999f, std::max(q2, -0.999999f));
      uint32_t iq1 = (q1 + 1) * decode_factor;
      uint32_t iq2 = (q2 + 1) * decode_factor;
      uint32_t code = (iq1 << 16) | iq2;
      mat.Set(y, x, code);
    }
  }
  return mat;
}

}  // namespace mvs
}  // namespace sensemap
