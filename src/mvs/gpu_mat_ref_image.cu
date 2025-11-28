//Copyright (c) 2019, SenseTime Group.
//All rights reserved.

#include <iostream>

#include "util/cudacc.h"
#include "mvs/utils.h"
#include "gpu_mat_ref_image.h"

namespace sensemap {
namespace mvs {
namespace {

using namespace utility;

__global__ void FilterKernel(GpuMat<uint8_t> image, GpuMat<float> sum_image,
                             GpuMat<float> squared_sum_image,
                             const int window_radius, const int window_step,
                             const float sigma_spatial,
                             const float sigma_color,
                             cudaTextureObject_t image_texture) {
  const size_t row = blockDim.y * blockIdx.y + threadIdx.y;
  const size_t col = blockDim.x * blockIdx.x + threadIdx.x;
  if (row >= image.GetHeight() || col >= image.GetWidth()) {
    return;
  }

  BilateralWeightComputer bilateral_weight_computer(sigma_spatial, sigma_color);

  const float center_color = tex2D<float>(image_texture, col, row);

  float color_sum = 0.0f;
  float color_squared_sum = 0.0f;
  float bilateral_weight_sum = 0.0f;

  for (int window_row = -window_radius; window_row <= window_radius;
       window_row += window_step) {
    for (int window_col = -window_radius; window_col <= window_radius;
         window_col += window_step) {
      const float color =
          tex2D<float>(image_texture, col + window_col, row + window_row);
      const float bilateral_weight = bilateral_weight_computer.Compute(
          window_row, window_col, center_color, color);
      color_sum += bilateral_weight * color;
      color_squared_sum += bilateral_weight * color * color;
      bilateral_weight_sum += bilateral_weight;
    }
  }

  color_sum /= bilateral_weight_sum;
  color_squared_sum /= bilateral_weight_sum;

  image.Set(row, col, static_cast<uint8_t>(255.0f * center_color));
  sum_image.Set(row, col, color_sum);
  squared_sum_image.Set(row, col, color_squared_sum);
}

}  // namespace

GpuMatRefImage::GpuMatRefImage(const size_t width, const size_t height)
    : height_(height), width_(width) {
  image.reset(new GpuMat<uint8_t>(width, height));
  sum_image.reset(new GpuMat<float>(width, height));
  squared_sum_image.reset(new GpuMat<float>(width, height));
}

void GpuMatRefImage::Filter(const uint8_t* image_data,
                            const size_t window_radius,
                            const size_t window_step, const float sigma_spatial,
                            const float sigma_color) {
  CudaArrayWrapper<uint8_t> image_array(width_, height_, 1);
  image_array.CopyToDevice(image_data);

  struct cudaResourceDesc resDesc;
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = cudaResourceTypeArray;
  resDesc.res.array.array = image_array.GetPtr();

  struct cudaTextureDesc texDesc;
  memset(&texDesc, 0, sizeof(texDesc));
  texDesc.addressMode[0] = cudaAddressModeBorder;
  texDesc.addressMode[1] = cudaAddressModeBorder;
  texDesc.addressMode[2] = cudaAddressModeBorder;
  texDesc.filterMode = cudaFilterModePoint;
  texDesc.readMode = cudaReadModeNormalizedFloat;
  texDesc.normalizedCoords = 0;

  CUDA_SAFE_CALL(cudaCreateTextureObject(&image_texture, &resDesc, &texDesc, NULL));

  const dim3 block_size(kBlockDimX, kBlockDimY);
  const dim3 grid_size((width_ - 1) / block_size.x + 1,
                       (height_ - 1) / block_size.y + 1);

  FilterKernel<<<grid_size, block_size>>>(
      *image, *sum_image, *squared_sum_image, window_radius, window_step,
      sigma_spatial, sigma_color, image_texture);
  CUDA_SYNC_AND_CHECK();
  
  CUDA_SAFE_CALL(cudaDestroyTextureObject(image_texture));
}


}  // namespace mvs
}  // namespace colmap
