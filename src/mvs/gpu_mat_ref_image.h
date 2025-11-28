//Copyright (c) 2019, SenseTime Group.
//All rights reserved.

#ifndef SENSEMAP_MVS_GPU_MAT_REF_IMAGE_H_
#define SENSEMAP_MVS_GPU_MAT_REF_IMAGE_H_

#include <memory>

#include "util/gpu_mat.h"
#include "mvs/cuda_array_wrapper.h"

namespace sensemap {
namespace mvs {

class GpuMatRefImage {
 public:
  GpuMatRefImage(const size_t width, const size_t height);

  // Filter image using sum convolution kernel to compute local sum of
  // intensities. The filtered images can then be used for repeated, efficient
  // NCC computation.
  void Filter(const uint8_t* image_data, const size_t window_radius,
              const size_t window_step, const float sigma_spatial,
              const float sigma_color);

  // Image intensities.
  std::unique_ptr<GpuMat<uint8_t>> image;

  // Local sum of image intensities.
  std::unique_ptr<GpuMat<float>> sum_image;

  // Local sum of squared image intensities.
  std::unique_ptr<GpuMat<float>> squared_sum_image;

 private:
  const static size_t kBlockDimX = 16;
  const static size_t kBlockDimY = 12;

  size_t width_;
  size_t height_;
  cudaTextureObject_t image_texture = 0;
};


}  // namespace mvs
}  // namespace sensemap

#endif  // SENSEMAP_MVS_GPU_MAT_REF_IMAGE_H_
