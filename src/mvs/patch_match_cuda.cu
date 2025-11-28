//Copyright (c) 2019, SenseTime Group.
//All rights reserved.

#define _USE_MATH_DEFINES

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <ctime>
#include <sstream>

#include "util/cuda.h"
#include "util/cudacc.h"
#include "util/logging.h"
#include "util/misc.h"
#include "util/timer.h"

#include "mvs/utils.h"
#include "mvs/patch_match_cuda.h"
#include "kernel_functions.h"

namespace sensemap {
namespace mvs {

using namespace utility;

PatchMatchBase::PatchMatchBase(const PatchMatchOptions &options,
                               const Problem &problem)
: options_(options),
  problem_(problem),
  ref_width_(0),
  ref_height_(0) {
}

PatchMatchBase::~PatchMatchBase() {
  CUDA_SAFE_CALL(cudaDestroyTextureObject(ref_image_texture));
  CUDA_SAFE_CALL(cudaDestroyTextureObject(src_images_texture));
  CUDA_SAFE_CALL(cudaDestroyTextureObject(ref_semantic_texture));
  CUDA_SAFE_CALL(cudaDestroyTextureObject(ref_mask_texture));
  CUDA_SAFE_CALL(cudaDestroyTextureObject(src_semantics_texture));
  if (options_.geom_consistency) {
    CUDA_SAFE_CALL(cudaDestroyTextureObject(src_depth_maps_texture));
  }
}

DepthMap PatchMatchBase::GetDepthMap() const {
  return DepthMap(depth_map_->CopyToMat(), options_.depth_min,
                  options_.depth_max);
}

NormalMap PatchMatchBase::GetNormalMap() const {
  return NormalMap(normal_map_->CopyToMat());
}

std::vector<int> PatchMatchBase::GetConsistentImageIdxs() const {
  int src_image_num = problem_.src_image_idxs.size();
  const Mat<uint8_t> mask = consistency_mask_->CopyToMat();
  std::vector<int> consistent_image_idxs;
  std::vector<int> pixel_consistent_image_idxs;
  pixel_consistent_image_idxs.reserve(src_image_num);
  int height = mask.GetHeight();
  int width = mask.GetWidth();
  for (size_t r = 0; r < height; ++r) {
    for (size_t c = 0; c < width; ++c) {
      pixel_consistent_image_idxs.clear();
      for (size_t d = 0; d < src_image_num; ++d) {
        if (mask.Get(r, c, d)) {
          pixel_consistent_image_idxs.push_back(problem_.src_image_idxs[d]);
        }
      }
      if (pixel_consistent_image_idxs.size() > 0) {
        consistent_image_idxs.push_back(c);
        consistent_image_idxs.push_back(r);
        consistent_image_idxs.push_back(pixel_consistent_image_idxs.size());
        consistent_image_idxs.insert(consistent_image_idxs.end(),
                                     pixel_consistent_image_idxs.begin(),
                                     pixel_consistent_image_idxs.end());
      }
    }
  }
  return consistent_image_idxs;
}

void PatchMatchBase::ComputeCudaConfig() {
  sweep_block_size_.x = THREADS_PER_BLOCK;
  sweep_block_size_.y = 1;
  sweep_block_size_.z = 1;
  sweep_grid_size_.x = (depth_map_->GetWidth() - 1) / THREADS_PER_BLOCK + 1;
  sweep_grid_size_.y = 1;
  sweep_grid_size_.z = 1;

  elem_wise_block_size_.x = THREADS_PER_BLOCK;
  elem_wise_block_size_.y = THREADS_PER_BLOCK;
  elem_wise_block_size_.z = 1;
  elem_wise_grid_size_.x = 
      (depth_map_->GetWidth() - 1) / THREADS_PER_BLOCK + 1;
  elem_wise_grid_size_.y =
      (depth_map_->GetHeight() - 1) / THREADS_PER_BLOCK + 1;
  elem_wise_grid_size_.z = 1;

  box_block_size_.x = THREADS_PER_BLOCK;
  box_block_size_.y = 1;
  box_block_size_.z = 1;
  box_grid_size_.x = (depth_map_->GetWidth() - 1) / THREADS_PER_BLOCK + 1;
  box_grid_size_.y = 1;
  box_grid_size_.z = 1;
}

void PatchMatchBase::InitRefImage() {
  const Image& ref_image = problem_.images->at(problem_.ref_image_idx);
  const Bitmap& ref_semantic = problem_.semantic_maps->at(problem_.ref_image_idx);
  const Bitmap& ref_mask = problem_.mask_maps->at(problem_.ref_image_idx);

  ref_width_ = ref_image.GetWidth();
  ref_height_ = ref_image.GetHeight();

  if (ref_image.GetBitmap().Data()) {
  // Upload to device.
  ref_image_.reset(new GpuMatRefImage(ref_width_, ref_height_));
  const std::vector<uint8_t> ref_image_array =
      ref_image.GetBitmap().ConvertToRowMajorArray();
  ref_image_->Filter(ref_image_array.data(), options_.window_radius,
                     options_.window_step, options_.sigma_spatial,
                     options_.sigma_color);

  ref_image_device_.reset(
      new CudaArrayWrapper<uint8_t>(ref_width_, ref_height_, 1));
  ref_image_device_->CopyFromGpuMat(*ref_image_->image);

  // Create texture.
  struct cudaResourceDesc resDesc;
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = cudaResourceTypeArray;
  resDesc.res.array.array = ref_image_device_->GetPtr();

  struct cudaTextureDesc texDesc;
  memset(&texDesc, 0, sizeof(texDesc));
  texDesc.addressMode[0] = cudaAddressModeBorder;
  texDesc.addressMode[1] = cudaAddressModeBorder;
  texDesc.addressMode[2] = cudaAddressModeBorder;
  texDesc.filterMode = cudaFilterModePoint;
  texDesc.readMode = cudaReadModeNormalizedFloat;
  texDesc.normalizedCoords = 0;

  CUDA_SAFE_CALL(
      cudaCreateTextureObject(&ref_image_texture, &resDesc, &texDesc, NULL));
  }
  if (options_.refine_with_semantic && ref_semantic.Data()) {
    // Mask texture.
    ref_semantic_device_.reset(new CudaArrayWrapper<uint8_t>(ref_width_, ref_height_, 1));
    const std::vector<uint8_t> ref_mask_array = ref_semantic.ConvertToRowMajorArray();
    ref_semantic_device_->CopyToDevice(ref_mask_array.data());

    // Create texture.
    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = ref_semantic_device_->GetPtr();

    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeBorder;
    texDesc.addressMode[1] = cudaAddressModeBorder;
    texDesc.addressMode[2] = cudaAddressModeBorder;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;

    CUDA_SAFE_CALL(
        cudaCreateTextureObject(&ref_semantic_texture, &resDesc, &texDesc, NULL));
  }
  if (ref_mask.Data()) {
    // Mask texture.
    ref_mask_device_.reset(new CudaArrayWrapper<uint8_t>(ref_width_, ref_height_, 1));
    const std::vector<uint8_t> ref_mask_array = ref_mask.ConvertToRowMajorArray();
    ref_mask_device_->CopyToDevice(ref_mask_array.data());

    // Create texture.
    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = ref_mask_device_->GetPtr();

    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeBorder;
    texDesc.addressMode[1] = cudaAddressModeBorder;
    texDesc.addressMode[2] = cudaAddressModeBorder;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;

    CUDA_SAFE_CALL(
        cudaCreateTextureObject(&ref_mask_texture, &resDesc, &texDesc, NULL));
  }
}

void PatchMatchBase::InitSourceImages() {
  // Determine maximum image size.
  size_t max_width = 0;
  size_t max_height = 0;
  std::vector<Bitmap> src_bitmaps, mask_bitmaps;
  for (size_t i = 0; i < problem_.src_image_idxs.size(); ++i) {
    int image_idx = problem_.src_image_idxs[i];
    float image_scale = 1.0f;
    if (problem_.src_image_scales.size() != 0) {
        image_scale = problem_.src_image_scales[i];
    }

    const Image& image = problem_.images->at(image_idx);
    const Bitmap mask = problem_.semantic_maps->at(image_idx);
    if (image.GetWidth() > max_width) {
        max_width = image.GetWidth();
    }
    if (image.GetHeight() > max_height) {
        max_height = image.GetHeight();
    }
    if (!image.GetBitmap().Data()) {
      continue;
    }
    Bitmap bitmap = image.GetBitmap().Clone();
    if (image_scale != 1) {
        const size_t new_width = 
            std::round(bitmap.Width() * image_scale);
        const size_t new_height = 
            std::round(bitmap.Height() * image_scale);
        bitmap.Rescale(new_width, new_height, image_scale > 1 ?
            FILTER_BICUBIC : FILTER_BILINEAR);
    }
    src_bitmaps.emplace_back(bitmap);
    if (mask.Data()) {
      mask_bitmaps.emplace_back(mask);
    }
  }

  // Upload source images to device.
  // if (!options_.filter){
  const uint8_t kDefaultValue = 0;
  const uint32_t max_size = max_width * max_height;
  std::vector<uint8_t> src_images_host_data;
  if (!src_bitmaps.empty()) {
    // Copy source images to contiguous memory block.
    src_images_host_data.resize(max_size * src_bitmaps.size(), kDefaultValue);

#pragma omp parallel for
    for (size_t i = 0; i < src_bitmaps.size(); ++i) {
      const Bitmap& bitmap = src_bitmaps[i];
      uint8_t* dest = src_images_host_data.data() + max_size * i;
      for (size_t r = 0; r < bitmap.Height(); ++r) {
        memcpy(dest, bitmap.GetScanline(r), 
               bitmap.Width() * sizeof(uint8_t));
        dest += max_width;
      }
    }

    // Upload to device.
    src_images_device_.reset(new CudaArrayWrapper<uint8_t>(
        max_width, max_height, src_bitmaps.size()));
    src_images_device_->CopyToDevice(src_images_host_data.data());

    // Create source images texture.
    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = src_images_device_->GetPtr();

    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeBorder;
    texDesc.addressMode[1] = cudaAddressModeBorder;
    texDesc.addressMode[2] = cudaAddressModeBorder;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeNormalizedFloat;
    texDesc.normalizedCoords = 0;

    CUDA_SAFE_CALL(
      cudaCreateTextureObject(&src_images_texture, &resDesc, &texDesc, NULL));
  }

  if (options_.refine_with_semantic && !mask_bitmaps.empty()) {
    src_images_host_data.resize(max_size * mask_bitmaps.size(),kDefaultValue);

#pragma omp parallel for
    for (size_t i = 0; i < mask_bitmaps.size(); ++i) {
      const Bitmap& bitmap = mask_bitmaps[i];
      uint8_t* dest = src_images_host_data.data() + max_size * i;
      for (size_t r = 0; r < bitmap.Height(); ++r) {
        memcpy(dest, bitmap.GetScanline(r), bitmap.Width() * sizeof(uint8_t));
        dest += max_width;
      }
    }
    // Upload to device.
    src_semantics_device_.reset(new CudaArrayWrapper<uint8_t>(
      max_width, max_height, mask_bitmaps.size()));
    src_semantics_device_->CopyToDevice(src_images_host_data.data());

    // Create source images texture.
    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = src_semantics_device_->GetPtr();

    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeBorder;
    texDesc.addressMode[1] = cudaAddressModeBorder;
    texDesc.addressMode[2] = cudaAddressModeBorder;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;

    CUDA_SAFE_CALL(
      cudaCreateTextureObject(&src_semantics_texture, &resDesc, &texDesc, NULL));
  }

  // Upload source depth maps to device.
  if (options_.geom_consistency) {
    const float kDefaultValue = 0.0f;
    std::vector<float> src_depth_maps_host_data(
      static_cast<size_t>(max_width * max_height *
                          problem_.src_image_idxs.size()),
      kDefaultValue);
#pragma omp parallel for
    for (size_t i = 0; i < problem_.src_image_idxs.size(); ++i) {
      DepthMap depth_map =
          problem_.depth_maps->at(problem_.src_image_idxs[i]);
      
      float image_scale = 1.0f;
      if (problem_.src_image_scales.size() != 0) {
          image_scale = problem_.src_image_scales[i];
      }
      if (image_scale != 1) {
          depth_map.Rescale(image_scale);
      }

      float* dest =
          src_depth_maps_host_data.data() + max_width * max_height * i;
      for (size_t r = 0; r < depth_map.GetHeight(); ++r) {
        memcpy(dest, depth_map.GetPtr() + r * depth_map.GetWidth(),
              depth_map.GetWidth() * sizeof(float));
        dest += max_width;
      }
    }

    src_depth_maps_device_.reset(new CudaArrayWrapper<float>(
        max_width, max_height, problem_.src_image_idxs.size()));
    src_depth_maps_device_->CopyToDevice(src_depth_maps_host_data.data());

    // Create source depth maps texture.
    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = src_depth_maps_device_->GetPtr();

    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeBorder;
    texDesc.addressMode[1] = cudaAddressModeBorder;
    texDesc.addressMode[2] = cudaAddressModeBorder;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;

    CUDA_SAFE_CALL(
      cudaCreateTextureObject(&src_depth_maps_texture, &resDesc,
                              &texDesc, NULL));

    // src_depth_maps_texture.addressMode[0] = cudaAddressModeBorder;
    // src_depth_maps_texture.addressMode[1] = cudaAddressModeBorder;
    // src_depth_maps_texture.addressMode[2] = cudaAddressModeBorder;
    // // TODO: Check if linear interpolation improves results or not.
    // src_depth_maps_texture.filterMode = cudaFilterModeLinear;
    // src_depth_maps_texture.normalized = false;
    // CUDA_SAFE_CALL(cudaBindTextureToArray(src_depth_maps_texture,
                                          // src_depth_maps_device_->GetPtr()));
  } else if (options_.propagate_depth) {
    const size_t image_size = max_width * max_height;
    const float kDefaultValue = 0.0f;
    std::vector<float> src_depth_maps_host_data(image_size * problem_.src_image_idxs.size(),
                                                kDefaultValue);
    for (size_t i = 0; i < problem_.src_image_idxs.size(); ++i) {
      if (!problem_.flag_depth_maps->at(problem_.src_image_idxs[i])) {
        continue;
      }
      DepthMap depth_map = problem_.depth_maps->at(problem_.src_image_idxs[i]);
      float* dest = src_depth_maps_host_data.data() + image_size * i;
      for (size_t r = 0; r < depth_map.GetHeight(); ++r) {
        memcpy(dest, depth_map.GetPtr() + r * depth_map.GetWidth(),
              depth_map.GetWidth() * sizeof(float));
        dest += max_width;
      }
    }

    src_depth_maps_device_.reset(new CudaArrayWrapper<float>(
        max_width, max_height, problem_.src_image_idxs.size()));
    src_depth_maps_device_->CopyToDevice(src_depth_maps_host_data.data());

    // Create source depth maps texture.
    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = src_depth_maps_device_->GetPtr();

    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeBorder;
    texDesc.addressMode[1] = cudaAddressModeBorder;
    texDesc.addressMode[2] = cudaAddressModeBorder;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;

    CUDA_SAFE_CALL(
      cudaCreateTextureObject(&src_depth_maps_texture, &resDesc,
                              &texDesc, NULL));
  }
}

////////////////////////////////////////ACMM////////////////////////////////////

PatchMatchACMMCuda::PatchMatchACMMCuda(const PatchMatchOptions &options,
                                       const Problem &problem) 
    : PatchMatchBase(options, problem) {
    SetBestCudaDevice(std::stoi(options.gpu_index));
    InitRefImage();
    InitSourceImages();
    InitTransforms();
    if(options.has_prior_depth) {
      InitPriorDepthAndNormalInfo();
    }
}

PatchMatchACMMCuda::~PatchMatchACMMCuda() {
  CUDA_SAFE_CALL(cudaDestroyTextureObject(ref_sum_image_texture));
  CUDA_SAFE_CALL(cudaDestroyTextureObject(ref_squared_sum_image_texture));
  CUDA_SAFE_CALL(cudaDestroyTextureObject(poses_texture));
}

Mat<float> PatchMatchACMMCuda::GetConfMap() const {
    return conf_map_->CopyToMat();
}

Mat<unsigned short> PatchMatchACMMCuda::GetCurvatureMap() const {
    return grad_curvature_map_->CopyToMat();
}

void PatchMatchACMMCuda::Preprocess() {
    ref_sum_image_device_.reset(
        new CudaArrayWrapper<float>(ref_width_, ref_height_, 1));
    ref_sum_image_device_->CopyFromGpuMat(*ref_image_->sum_image);

    // Create texture.
    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = ref_sum_image_device_->GetPtr();

    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeBorder;
    texDesc.addressMode[1] = cudaAddressModeBorder;
    texDesc.addressMode[2] = cudaAddressModeBorder;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;
    CUDA_SAFE_CALL(
        cudaCreateTextureObject(&ref_sum_image_texture, &resDesc,
                                                  &texDesc, NULL));

    ref_squared_sum_image_device_.reset(
        new CudaArrayWrapper<float>(ref_width_, ref_height_, 1));
    ref_squared_sum_image_device_->CopyFromGpuMat(
        *ref_image_->squared_sum_image);

    // Create texture.
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = ref_squared_sum_image_device_->GetPtr();

    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeBorder;
    texDesc.addressMode[1] = cudaAddressModeBorder;
    texDesc.addressMode[2] = cudaAddressModeBorder;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;
    CUDA_SAFE_CALL(
        cudaCreateTextureObject(&ref_squared_sum_image_texture, &resDesc,
                                                          &texDesc, NULL));
}

void PatchMatchACMMCuda::InitTransforms() {
    const Image& ref_image = problem_.images->at(problem_.ref_image_idx);
    ref_K_host_[0] = ref_image.GetK()[0];
    ref_K_host_[1] = ref_image.GetK()[2];
    ref_K_host_[2] = ref_image.GetK()[4];
    ref_K_host_[3] = ref_image.GetK()[5];

    // Extract 1/fx, -cx/fx, 1/fy, -cy/fy.
    ref_inv_K_host_[0] = 1.0f / ref_K_host_[0];
    ref_inv_K_host_[1] = -ref_K_host_[1] / ref_K_host_[0];
    ref_inv_K_host_[2] = 1.0f / ref_K_host_[2];
    ref_inv_K_host_[3] = -ref_K_host_[3] / ref_K_host_[2];

    // Bind pose to constant global memory.
    SetRefK(ref_K_host_, options_.thread_index);
    SetRefInvK(ref_inv_K_host_, options_.thread_index);

    float rotated_R[9];
    memcpy(rotated_R, ref_image.GetR(), 9 * sizeof(float));

    float rotated_T[3];
    memcpy(rotated_T, ref_image.GetT(), 3 * sizeof(float));

    std::vector<float> poses_host_data;
    poses_host_data.resize(K_NUM_TFORM_PARAMS * problem_.src_image_idxs.size());
    int offset = 0;
    // for (const auto image_idx : problem_.src_image_idxs) {
    for (int i = 0; i < problem_.src_image_idxs.size(); ++i) {
        const int image_idx = problem_.src_image_idxs[i];
        const Image& image = problem_.images->at(image_idx);
        const float scale = problem_.src_image_scales[i];

        float iK[9];
        memcpy(iK, image.GetK(), sizeof(float) * 9);
        if (scale != 1) {
          iK[0] *= scale; iK[2] *= scale;
          iK[4] *= scale; iK[5] *= scale;
        }

        const float K[4] = {iK[0], iK[2], iK[4], iK[5]};
        memcpy(poses_host_data.data() + offset, K, 4 * sizeof(float));
        offset += 4;

        float rel_R[9];
        float rel_T[3];
        ComputeRelativePose(rotated_R, rotated_T, image.GetR(), image.GetT(),
        rel_R, rel_T);
        memcpy(poses_host_data.data() + offset, rel_R, 9 * sizeof(float));
        offset += 9;
        memcpy(poses_host_data.data() + offset, rel_T, 3 * sizeof(float));
        offset += 3;

        float C[3];
        ComputeProjectionCenter(rel_R, rel_T, C);
        memcpy(poses_host_data.data() + offset, C, 3 * sizeof(float));
        offset += 3;

        float P[12];
        ComposeProjectionMatrix(iK, rel_R, rel_T, P);
        memcpy(poses_host_data.data() + offset, P, 12 * sizeof(float));
        offset += 12;

        float inv_P[12];
        ComposeInverseProjectionMatrix(iK, rel_R, rel_T, inv_P);
        memcpy(poses_host_data.data() + offset, inv_P, 12 * sizeof(float));
        offset += 12;
    }
    poses_device_.reset(new CudaArrayWrapper<float>(
        K_NUM_TFORM_PARAMS, problem_.src_image_idxs.size(), 1));
    poses_device_->CopyToDevice(poses_host_data.data());

    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = poses_device_->GetPtr();

    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeBorder;
    texDesc.addressMode[1] = cudaAddressModeBorder;
    texDesc.addressMode[2] = cudaAddressModeBorder;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;

    CUDA_SAFE_CALL(
      cudaCreateTextureObject(&poses_texture, &resDesc, &texDesc, NULL));
}

void PatchMatchACMMCuda::InitWorkspace() {
    rand_state_map_.reset(new GpuMatPRNG(ref_width_, ref_height_));

    conf_map_.reset(new GpuMat<float>(ref_width_, ref_height_));
    depth_map_.reset(new GpuMat<float>(ref_width_, ref_height_));
    normal_map_.reset(new GpuMat<float>(ref_width_, ref_height_, 3));
    if (!options_.has_prior_depth) {
      prior_depth_map_.reset(new GpuMat<float>(0, 0, 0));
      prior_normal_map_.reset(new GpuMat<float>(0, 0, 0));
      prior_wgt_map_.reset(new GpuMat<float>(0, 0, 0));
    }
    if (!options_.init_depth_random) {
      DepthMap init_depth_map = 
          problem_.depth_maps->at(problem_.ref_image_idx);
      depth_map_->CopyToDevice(init_depth_map.GetPtr(),
                                init_depth_map.GetWidth() * sizeof(float));
      NormalMap init_normal_map =
          problem_.normal_maps->at(problem_.ref_image_idx);
      normal_map_->CopyToDevice(init_normal_map.GetPtr(),
                                init_normal_map.GetWidth() * sizeof(float));

      if (options_.propagate_depth) {
        CudaTimer timer;

        size_t max_width = 0;
        size_t max_height = 0;
        std::cout << "neighbor image ids: ";
        for (size_t i = 0; i < problem_.src_image_idxs.size(); ++i) {
          int image_idx = problem_.src_image_idxs[i];
          const Image& image = problem_.images->at(image_idx);
          if (image.GetWidth() > max_width) {
              max_width = image.GetWidth();
          }
          if (image.GetHeight() > max_height) {
              max_height = image.GetHeight();
          }
          if (problem_.flag_depth_maps->at(image_idx)) {
            std::cout << image_idx << " ";
          }
        }
        std::cout << std::endl;

        elem_wise_block_size_.x = THREADS_PER_BLOCK;
        elem_wise_block_size_.y = THREADS_PER_BLOCK;
        elem_wise_grid_size_.x = (max_width - 1) / THREADS_PER_BLOCK + 1;
        elem_wise_grid_size_.y = (max_height - 1) / THREADS_PER_BLOCK + 1;

        const int num_src_image = problem_.src_image_idxs.size();
        GpuMat<float> composite_depth_map(ref_width_, ref_height_, num_src_image);
        composite_depth_map.FillWithScalar(FLT_MAX);
        for (int i = 0; i < num_src_image; ++i) {
          int src_image_idx = problem_.src_image_idxs.at(i);
          if (!problem_.flag_depth_maps->at(src_image_idx)) {
            continue;
          }
          GenerateProjectDepthMaps<<<elem_wise_grid_size_, elem_wise_block_size_>>>(
            max_width, max_height, i, num_src_image, 
            composite_depth_map, /*composite_normal_map,*/
            src_depth_maps_texture, poses_texture, options_.thread_index);
        }
        CUDA_SYNC_AND_CHECK();

        prior_depth_map_.reset(new GpuMat<float>(ref_width_, ref_height_));
        prior_depth_map_->FillWithScalar(0);

        ComputeCudaConfig();
        CompositeInitDepthMap<<<elem_wise_grid_size_, elem_wise_block_size_>>>(
          *rand_state_map_, *depth_map_, *prior_depth_map_, composite_depth_map, 
          options_.depth_min, options_.depth_max);
        CUDA_SYNC_AND_CHECK();

        CompositeInitNormalMap<<<elem_wise_grid_size_, elem_wise_block_size_>>>(
          *depth_map_, *normal_map_, options_.thread_index
        );
        CUDA_SYNC_AND_CHECK();

        timer.Print("Propagate Depth Map");
      }
      
      ComputeCudaConfig();
      InitDepthMap<<<elem_wise_grid_size_, elem_wise_block_size_>>>(
        *depth_map_, *normal_map_, *rand_state_map_, 
        options_.depth_min, options_.depth_max, options_.thread_index
      );
    } else {
      ComputeCudaConfig();
      depth_map_->FillWithRandomNumbers(options_.depth_min, 
                                        options_.depth_max, *rand_state_map_);
      InitNormalMap<<<elem_wise_grid_size_, elem_wise_block_size_>>>(
      *normal_map_, *rand_state_map_, options_.thread_index);
    }
    CUDA_SYNC_AND_CHECK();

    cost_map_.reset(new GpuMat<float>(ref_width_, ref_height_, src_images_device_->GetDepth()));
    view_sel_map_.reset(new GpuMat<uint32_t>(ref_width_, ref_height_, 1));
    view_sel_map_->FillWithScalar(0);

    // selected_images_list_.reset(new GpuMat<uint32_t>(ref_width_, ref_height_, 1));
    // selected_images_list_->FillWithScalar(0);

    if (options_.geom_consistency && options_.est_curvature) {
      grad_curvature_map_.reset(new GpuMat<unsigned short>(ref_width_, ref_height_, 2));
    }

    gradient_map_.reset(new GpuMat<float>(ref_width_, ref_height_));
    ComputeGradientMap<<<elem_wise_grid_size_, elem_wise_block_size_>>>(
      *gradient_map_, ref_image_texture
    );
    CUDA_SYNC_AND_CHECK();

    if (options_.median_filter) {
      geom_gradient_map_.reset(new GpuMat<float>(ref_width_, ref_height_));
    }

    // planarity_map_.reset(new GpuMat<float>(ref_width_, ref_height_));
    // planarity_map_->FillWithScalar(0);
}


void PatchMatchACMMCuda::InitWorkspaceFilter() {
  depth_map_.reset(new GpuMat<float>(ref_width_, ref_height_));
  // normal_map_.reset(new GpuMat<float>(ref_width_, ref_height_, 3));

  DepthMap init_depth_map = 
      problem_.depth_maps->at(problem_.ref_image_idx);
  depth_map_->CopyToDevice(init_depth_map.GetPtr(),
                            init_depth_map.GetWidth() * sizeof(float));
  // NormalMap init_normal_map =
  //     problem_.normal_maps->at(problem_.ref_image_idx);
  // normal_map_->CopyToDevice(init_normal_map.GetPtr(),
  //                           init_normal_map.GetWidth() * sizeof(float));
}

void PatchMatchACMMCuda::InitPriorDepthAndNormalInfo() 
{
  // const auto prior_depth_map = problem_.prior_depth_map;
  const auto prior_depth_map = std::make_shared<DepthMap>(problem_.depth_maps->at(problem_.ref_image_idx));
  const auto prior_normal_map = std::make_shared<NormalMap>(problem_.normal_maps->at(problem_.ref_image_idx));
  
  const size_t prior_depth_map_width_ = prior_depth_map->GetWidth();
  const size_t prior_depth_map_height_ = prior_depth_map->GetHeight();

  // Upload to device.
  prior_depth_map_.reset(new GpuMat<float>(prior_depth_map_width_, prior_depth_map_height_));
  prior_depth_map_->CopyToDevice(prior_depth_map->GetPtr(),
                                 prior_depth_map->GetWidth() * sizeof(float));

  prior_normal_map_.reset(new GpuMat<float>(prior_depth_map_width_, prior_depth_map_height_, 3));
  prior_normal_map_->CopyToDevice(prior_normal_map->GetPtr(),
                                  prior_normal_map->GetWidth() * sizeof(float));
  std::shared_ptr<DepthMap> prior_wgt_map;
  if (problem_.prior_wgt_maps->at(problem_.ref_image_idx).IsValid()){
    prior_wgt_map = std::make_shared<DepthMap>(problem_.prior_wgt_maps->at(problem_.ref_image_idx));
  } else {
    DepthMap prior_wgt(prior_depth_map_width_, prior_depth_map_height_, 0, 1.0);
    prior_wgt.Fill(0);
    prior_wgt_map = std::make_shared<DepthMap>(prior_wgt);
  }
  prior_wgt_map_.reset(new GpuMat<float>(prior_depth_map_width_, prior_depth_map_height_));
  prior_wgt_map_->CopyToDevice(prior_wgt_map->GetPtr(),
                                prior_wgt_map->GetWidth() * sizeof(float));

  std::cout<<"init prior depth and normal end"<<std::endl;

}

bool PatchMatchACMMCuda::Run() {
    std::cout << "PatchMatchACMMCuda::Run" << std::endl;

    BilateralWeightComputer weight_computer(options_.sigma_spatial,
      options_.sigma_color);
    const int num_src_image = src_images_device_->GetDepth();
    const int BLOCK_W = THREADS_PER_BLOCK;
    const int BLOCK_H = THREADS_PER_BLOCK / 2;

    auto PropagationAndOptim = [&](SweepOptions sweep_options) {

      sweep_block_size_.x = BLOCK_W;
      sweep_block_size_.y = BLOCK_H;
      sweep_block_size_.z = 1;
      sweep_grid_size_.x = (ref_width_ + BLOCK_W - 1) / BLOCK_W;
      sweep_grid_size_.y = (ref_height_ + BLOCK_H - 1) / BLOCK_H;
      sweep_grid_size_.z = 1;

      // box_block_size_.x = THREADS_PER_BLOCK;
      // box_block_size_.y = 1;
      // box_block_size_.z = 1;
      // box_grid_size_.x = (depth_map_->GetWidth() - 1) / THREADS_PER_BLOCK + 1;
      // box_grid_size_.y = 1;
      // box_grid_size_.z = 1;

      // size_t max_num_src_images = std::min((size_t)5, options_.max_num_src_images);
      const int max_iteration = options_.num_iterations;
      for (int iter = 0; iter < max_iteration; ++iter) {
          CudaTimer iter_timer;

          sweep_options.th_mc = sweep_options.init_ncc_matching_cost * 
                                exp(-iter * iter / sweep_options.alpha);
          sweep_options.perturbation = 1.0f / std::pow(2.0f, iter);
          
          // // CudaTimer rand_timer;
          // InitSelectedImageList<<<elem_wise_grid_size_, elem_wise_block_size_>>>(
          //   *rand_state_map_, *selected_images_list_, max_num_src_images, options_.max_num_src_images);
          // // rand_timer.Print("  Random Cost");

// #ifdef ACMMPM_SHMEM_CUDA          
//           // shared memory needed for cache ref_image_texture
//           const int row_size = 1 + sweep_options.window_radius * 2; 
//           const int col_size = THREADS_PER_BLOCK * 2 +
//                                sweep_options.window_radius * 2;
//           int shared_size = row_size * col_size * sizeof(float); 
//           BlackPixelUpdateOpt<<<sweep_grid_size_, sweep_block_size_,
//                                 shared_size>>>(
//                                 *rand_state_map_, *depth_map_, *normal_map_, 
//                                 *conf_map_, *cost_map_, *view_sel_map_,
//                                 num_src_image, sweep_options, weight_computer,
//                                 ref_image_texture, ref_sum_image_texture,
//                                 ref_squared_sum_image_texture,
//                                 src_images_texture, src_depth_maps_texture,
//                                 poses_texture);
//           CUDA_SYNC_AND_CHECK();
//           RedPixelUpdateOpt<<<sweep_grid_size_, sweep_block_size_,
//                                 shared_size>>>(
//                                 *rand_state_map_,
//                                 *depth_map_, *normal_map_, *conf_map_, 
//                                 *cost_map_, *view_sel_map_, num_src_image, 
//                                 sweep_options, weight_computer,
//                                 ref_image_texture, ref_sum_image_texture,
//                                 ref_squared_sum_image_texture,
//                                 src_images_texture, src_depth_maps_texture,
//                                 poses_texture);
//           CUDA_SYNC_AND_CHECK();
// #else
          // shared memory needed for cache ref_image_texture
          const int row_size = BLOCK_H * 2 + sweep_options.window_radius * 2; 
          const int col_size = BLOCK_W + sweep_options.window_radius * 2; 
          int shared_size = row_size * col_size * sizeof(float);
          // if (options_.has_prior_depth) {
          //   BlackPixelUpdateOpt<<<sweep_grid_size_, sweep_block_size_, shared_size>>>(
          //                 *rand_state_map_, *depth_map_, *normal_map_, 
          //                 *conf_map_, *gradient_map_, *cost_map_, *view_sel_map_, 
          //                 /**planarity_map_,*/ *prior_depth_map_, // *selected_images_list_,
          //                 num_src_image, sweep_options, weight_computer,
          //                 ref_image_texture, ref_sum_image_texture,
          //                 ref_squared_sum_image_texture, src_images_texture, 
          //                 /*ref_semantic_texture, src_semantics_texture, */src_depth_maps_texture,
          //                 poses_texture);
          //   CUDA_SYNC_AND_CHECK();
          //   RedPixelUpdateOpt<<<sweep_grid_size_, sweep_block_size_, shared_size>>>(
          //                 *rand_state_map_, *depth_map_, *normal_map_, 
          //                 *conf_map_, *gradient_map_, *cost_map_, *view_sel_map_, 
          //                 /**planarity_map_,*/ *prior_depth_map_, // *selected_images_list_,
          //                 num_src_image, sweep_options, weight_computer,
          //                 ref_image_texture, ref_sum_image_texture,
          //                 ref_squared_sum_image_texture, src_images_texture, 
          //                 /*ref_semantic_texture, src_semantics_texture, */src_depth_maps_texture,
          //                 poses_texture);
          //   CUDA_SYNC_AND_CHECK();
          // } else {
            BlackPixelUpdateOpt<<<sweep_grid_size_, sweep_block_size_, shared_size>>>(
                          *rand_state_map_, *depth_map_, *normal_map_, 
                          *conf_map_, *gradient_map_, *cost_map_, *view_sel_map_, 
                          /**planarity_map_,*/ *prior_depth_map_,  *prior_normal_map_, *prior_wgt_map_,// *selected_images_list_,
                          num_src_image, sweep_options, weight_computer,
                          ref_image_texture, ref_sum_image_texture,
                          ref_squared_sum_image_texture, src_images_texture, 
                          /*ref_semantic_texture, src_semantics_texture, */src_depth_maps_texture,
                          poses_texture);
            CUDA_SYNC_AND_CHECK();
            RedPixelUpdateOpt<<<sweep_grid_size_, sweep_block_size_, shared_size>>>(
                          *rand_state_map_, *depth_map_, *normal_map_, 
                          *conf_map_, *gradient_map_, *cost_map_, *view_sel_map_, 
                          /**planarity_map_,*/ *prior_depth_map_, *prior_normal_map_, *prior_wgt_map_,// *selected_images_list_,
                          num_src_image, sweep_options, weight_computer,
                          ref_image_texture, ref_sum_image_texture,
                          ref_squared_sum_image_texture, src_images_texture, 
                          /*ref_semantic_texture, src_semantics_texture, */src_depth_maps_texture,
                          poses_texture);
            CUDA_SYNC_AND_CHECK();
            if (options_.local_optimization && iter > 2 && !options_.has_prior_depth) {
              BlackPixelLocalOptimization<<<sweep_grid_size_, sweep_block_size_, shared_size>>>(
                          *rand_state_map_, *depth_map_, *normal_map_, 
                          *conf_map_, *cost_map_, *view_sel_map_, // *selected_images_list_,
                          num_src_image, sweep_options, weight_computer,
                          ref_image_texture, ref_sum_image_texture,
                          ref_squared_sum_image_texture, src_images_texture, 
                          /*ref_semantic_texture, src_semantics_texture, */src_depth_maps_texture,
                          poses_texture);
              CUDA_SYNC_AND_CHECK();
              RedPixelLocalOptimization<<<sweep_grid_size_, sweep_block_size_, shared_size>>>(
                          *rand_state_map_, *depth_map_, *normal_map_, 
                          *conf_map_, *cost_map_, *view_sel_map_, // *selected_images_list_,
                          num_src_image, sweep_options, weight_computer,
                          ref_image_texture, ref_sum_image_texture,
                          ref_squared_sum_image_texture, src_images_texture, 
                          /*ref_semantic_texture, src_semantics_texture, */src_depth_maps_texture,
                          poses_texture);
              CUDA_SYNC_AND_CHECK();

            }
            // if (options_.window_radius == 3 && options_.window_step == 1) {
            //   EstimateLocalPlanarity<3, 1><<<elem_wise_grid_size_, elem_wise_block_size_>>>(
            //     sweep_options, *depth_map_, *normal_map_, *planarity_map_);
            //   CUDA_SYNC_AND_CHECK();
            // } else if (options_.window_radius == 3 && options_.window_step == 2) {
            //   EstimateLocalPlanarity<3, 2><<<elem_wise_grid_size_, elem_wise_block_size_>>>(
            //     sweep_options, *depth_map_, *normal_map_, *planarity_map_);
            //   CUDA_SYNC_AND_CHECK();
            // } else if (options_.window_radius == 5 && options_.window_step == 1) {
            //   EstimateLocalPlanarity<5, 1><<<elem_wise_grid_size_, elem_wise_block_size_>>>(
            //     sweep_options, *depth_map_, *normal_map_, *planarity_map_);
            //   CUDA_SYNC_AND_CHECK();
            // } else if (options_.window_radius == 5 && options_.window_step == 2) {
            //   EstimateLocalPlanarity<5, 2><<<elem_wise_grid_size_, elem_wise_block_size_>>>(
            //     sweep_options, *depth_map_, *normal_map_, *planarity_map_);
            //   CUDA_SYNC_AND_CHECK();
            // } else if (options_.window_radius == 7 && options_.window_step == 1) {
            //   EstimateLocalPlanarity<7, 1><<<elem_wise_grid_size_, elem_wise_block_size_>>>(
            //     sweep_options, *depth_map_, *normal_map_, *planarity_map_);
            //   CUDA_SYNC_AND_CHECK();
            // } else if (options_.window_radius == 7 && options_.window_step == 2) {
            //   EstimateLocalPlanarity<7, 2><<<elem_wise_grid_size_, elem_wise_block_size_>>>(
            //     sweep_options, *depth_map_, *normal_map_, *planarity_map_);
            //   CUDA_SYNC_AND_CHECK();
            // }
          // }
          iter_timer.Print("  Iteration " + std::to_string(iter + 1));
      }
    };

    CudaTimer total_timer;

    SweepOptions sweep_options;
    sweep_options.thread_index = options_.thread_index;
    sweep_options.depth_min = options_.depth_min;
    sweep_options.depth_max = options_.depth_max;
    sweep_options.window_radius = options_.window_radius;
    sweep_options.window_step = options_.window_step;
    sweep_options.sigma_spatial = options_.sigma_spatial;
    sweep_options.sigma_color = options_.sigma_color;
    sweep_options.geom_consistency_regularizer = options_.geom_consistency_regularizer;
    sweep_options.geom_consistency_max_cost = options_.geom_consistency_max_cost;

    sweep_options.random_depth_ratio = options_.random_depth_ratio;
    sweep_options.random_angle1_range = options_.random_angle1_range;
    sweep_options.random_angle2_range = options_.random_angle2_range;
    sweep_options.random_smooth_bonus = options_.random_smooth_bonus;
    sweep_options.diff_photometric_consistency = 
        options_.diff_photometric_consistency;
    sweep_options.ncc_thres_refine = options_.ncc_thres_refine;
    sweep_options.init_ncc_matching_cost = options_.init_ncc_matching_cost;
    sweep_options.th_mc = options_.th_mc;
    sweep_options.max_ncc_matching_cost = options_.max_ncc_matching_cost;
    sweep_options.alpha = options_.alpha;
    sweep_options.beta = options_.beta;
    sweep_options.num_good_hypothesis = options_.num_good_hypothesis;
    sweep_options.num_bad_hypothesis = options_.num_bad_hypothesis;
    sweep_options.thk = options_.thk;
    sweep_options.plane_regularizer = options_.plane_regularizer;
    sweep_options.geom_consistency = options_.geom_consistency;
    sweep_options.random_optimization = options_.random_optimization;
    sweep_options.propagate_depth = options_.propagate_depth;
    sweep_options.prior_depth_ncc = options_.has_prior_depth;

    // propagate_depth and prior_depth_ncc use the same buffer
    CHECK(!(sweep_options.propagate_depth && sweep_options.prior_depth_ncc));

    sweep_options.thk = 
        min(sweep_options.thk, (int)src_images_device_->GetDepth());

    InitWorkspace();
    CUDA_SYNC_AND_CHECK();
    Preprocess();

    // ComputeCudaConfig();
    // InitializeDistMap(const GpuMat<float> grad_map, 
    //                               GpuMat<float> dist_map,
    //                               const float min_grad_thres);
    // CUDA_SYNC_AND_CHECK();
    
    // Inittializing Confidence Map
    {
      elem_wise_block_size_.x = BLOCK_W;
      elem_wise_block_size_.y = BLOCK_H;
      elem_wise_block_size_.z = 1;
      elem_wise_grid_size_.x = (ref_width_ + BLOCK_W - 1) / BLOCK_W;
      elem_wise_grid_size_.y = (ref_height_ + BLOCK_H - 1) / BLOCK_H;
      elem_wise_grid_size_.z = 1;
      const int row_size = BLOCK_H + sweep_options.window_radius * 2;
      const int col_size = BLOCK_W + sweep_options.window_radius * 2; 
      int shared_size = row_size * col_size * sizeof(float);
      InitConfMapKernel<<<elem_wise_grid_size_, elem_wise_block_size_, shared_size>>>(
          sweep_options, src_images_device_->GetDepth(),
          *depth_map_, *normal_map_, *conf_map_, *cost_map_, *view_sel_map_, 
          *prior_depth_map_, *prior_normal_map_, *prior_wgt_map_,
          weight_computer, ref_image_texture, ref_sum_image_texture,
          ref_squared_sum_image_texture, src_images_texture, /*ref_semantic_texture, 
          src_semantics_texture, */src_depth_maps_texture, poses_texture, options_.thread_index);
      CUDA_SYNC_AND_CHECK();
    }

    PropagationAndOptim(sweep_options);

    if (options_.geom_consistency && options_.est_curvature) {
      ComputeCurvatureMap<<<elem_wise_grid_size_, elem_wise_block_size_>>>(
        ref_image_texture, *normal_map_, *grad_curvature_map_);
      CUDA_SYNC_AND_CHECK();
    }

    if (options_.median_filter) {
      CudaTimer filter_timer;

      ComputeGeomGradientMap<<<elem_wise_grid_size_, elem_wise_block_size_>>>(
        *depth_map_, *geom_gradient_map_
      );
      CUDA_SYNC_AND_CHECK();

      BlackPixelFilter<<<sweep_grid_size_, sweep_block_size_>>>(
        *depth_map_, *normal_map_, *geom_gradient_map_, options_.depth_diff_threshold
      );
      CUDA_SYNC_AND_CHECK();
      RedPixelFilter<<<sweep_grid_size_, sweep_block_size_>>>(
        *depth_map_, *normal_map_, *geom_gradient_map_, options_.depth_diff_threshold
      );
      CUDA_SYNC_AND_CHECK();
      filter_timer.Print("Median Filter");
    }

    // RectifyConfidenceMap<<<elem_wise_grid_size_, elem_wise_block_size_>>>(
    //   *depth_map_, *normal_map_, *conf_map_, *view_sel_map_, num_src_image, 
    //   sweep_options, ref_image_texture, src_images_texture, poses_texture,
    //   options_.depth_diff_threshold, options_.conf_threshold, options_.thread_index);

    if (options_.filter && options_.conf_filter) {
      std::cout << "FilterDepthMap" << std::endl;
      
      int BLOCK_W = THREADS_PER_BLOCK;
      int BLOCK_H = BLOCK_W / 2;
      dim3 grid_size_filter;
      grid_size_filter.x = (ref_width_ + BLOCK_W - 1) / BLOCK_W;
      grid_size_filter.y= ( (ref_height_ / 2) + BLOCK_H - 1) / BLOCK_H;
      grid_size_filter.z = 1;
      dim3 block_size_filter;
      block_size_filter.x = BLOCK_W;
      block_size_filter.y = BLOCK_H;
      block_size_filter.z = 1;
      FilterDepthMap<<<grid_size_filter, block_size_filter>>>(
        *depth_map_, *normal_map_, *conf_map_, options_.depth_diff_threshold,
        options_.conf_threshold, options_.thread_index, 0
      );
      CUDA_SYNC_AND_CHECK();
      FilterDepthMap<<<grid_size_filter, block_size_filter>>>(
        *depth_map_, *normal_map_, *conf_map_, options_.depth_diff_threshold,
        options_.conf_threshold, options_.thread_index, 1
      );
      CUDA_SYNC_AND_CHECK();
    }

    if (problem_.mask_maps->at(problem_.ref_image_idx).Data()) {
      std::cout << "MaskDepthMap" << std::endl;
      CudaTimer mask_timer;
      elem_wise_block_size_.x = BLOCK_W;
      elem_wise_block_size_.y = BLOCK_H;
      elem_wise_block_size_.z = 1;
      elem_wise_grid_size_.x = (ref_width_ + BLOCK_W - 1) / BLOCK_W;
      elem_wise_grid_size_.y = (ref_height_ + BLOCK_H - 1) / BLOCK_H;
      elem_wise_grid_size_.z = 1;
      MaskDepthMapKernel<<<elem_wise_grid_size_, elem_wise_block_size_>>>(*depth_map_, ref_mask_texture);
      CUDA_SYNC_AND_CHECK();
      mask_timer.Print("Mask Filter");
    }

    poses_device_.reset();
    total_timer.Print("Total");
    return true;
}

void PatchMatchACMMCuda::InitConfMapFilter() {

  conf_map_.reset(new GpuMat<float>(ref_width_, ref_height_));
  if (problem_.conf_maps->at(problem_.ref_image_idx).IsValid()){
    Mat<float> init_conf_map = problem_.conf_maps->at(problem_.ref_image_idx);
    conf_map_->CopyToDevice(init_conf_map.GetPtr(),
                            init_conf_map.GetWidth() * sizeof(float));
  } else {
    Mat<float> init_conf_map(ref_width_, ref_height_, 1);
    init_conf_map.Fill(options_.conf_threshold + 0.1);
    conf_map_->CopyToDevice(init_conf_map.GetPtr(),
                            init_conf_map.GetWidth() * sizeof(float));
  }
  CUDA_SYNC_AND_CHECK();
}

bool PatchMatchACMMCuda::RunFilter(bool deduplication_flag) {
  std::cout << "PatchMatchACMMCuda::RunFilter" << std::endl;

  CudaTimer total_timer;

  InitWorkspaceFilter();

  CUDA_SYNC_AND_CHECK();

  // compute cuda configure
  ComputeCudaConfig();

  const int filter_min_num_consistent = min(options_.filter_min_num_consistent, 
    (int)problem_.src_image_idxs.size());

  if (!deduplication_flag){
    std::cout << "FilterDepthMapsKernel" << std::endl;
    // InitConfMapFilter();
    FilterDepthMapsKernel<<<elem_wise_grid_size_, elem_wise_block_size_>>>(
      problem_.src_image_idxs.size(), *depth_map_/*, *normal_map_, *conf_map_*/,
      options_.geom_consistency_max_cost, options_.filter_geom_consistency_max_cost, 
      options_.depth_diff_threshold, filter_min_num_consistent, options_.conf_threshold,
      src_depth_maps_texture, poses_texture, options_.thread_index);
  } else {
    std::cout << "DeduplicDepthMapsKernel" << std::endl;
    DeduplicDepthMapsKernel<<<elem_wise_grid_size_, elem_wise_block_size_>>>(
      problem_.src_image_idxs.size(), *depth_map_/*, *normal_map_*/, 
      options_.geom_consistency_max_cost, options_.filter_geom_consistency_max_cost, 
      options_.depth_diff_threshold, filter_min_num_consistent, 
      src_depth_maps_texture, poses_texture, options_.thread_index);
  }
  total_timer.Print("Filter Total");
  return true;
}


CrossFilterCuda::CrossFilterCuda(
  const PatchMatchOptions &options,
  const std::vector<Problem> &problems,
  const std::vector<int > &whole_ids,
  const std::vector<bool > &ref_flags)
: options_(options),
  problems_(problems),
  all_images_ids_(whole_ids),
  ref_flags_(ref_flags),
  num_images_(whole_ids.size()),
  max_width_(0),
  max_height_(0){
  SetBestCudaDevice(std::stoi(options.gpu_index));
}

CrossFilterCuda::~CrossFilterCuda() {}

void CrossFilterCuda::ComputeCudaConfig() {
  elem_wise_block_size_.x = THREADS_PER_BLOCK;
  elem_wise_block_size_.y = THREADS_PER_BLOCK;
  elem_wise_block_size_.z = 1;
  elem_wise_grid_size_.x = 
      (max_width_ - 1) / THREADS_PER_BLOCK + 1;
  elem_wise_grid_size_.y =
      (max_height_ - 1) / THREADS_PER_BLOCK + 1;
  elem_wise_grid_size_.z = 1;
}

void CrossFilterCuda::InitDepthMaps() {
  // std::cout << "num_images: " << num_images_ << std::endl;
  for (int i = 0; i < num_images_; i++){
    int image_id = all_images_ids_[i];
    const Image& image = problems_.at(0).images->at(image_id);

    if (image.GetWidth() > max_width_) {
      max_width_ = image.GetWidth();
    }
    if (image.GetHeight() > max_height_) {
      max_height_ = image.GetHeight();
    }
  }

  std::cout << "InitDepthMaps size: " << all_images_ids_.size() << "(" 
    << max_width_ << " x " << max_height_ << ")"<< std::endl;

  Timer timer;
  timer.Start();
  const float kDefaultValue = 0.0f;
  std::vector<float> depth_maps_host_data(
    static_cast<size_t>(max_width_ * max_height_ * num_images_),
                        kDefaultValue);
  // std::cout << max_width_ << " " << max_height_ << " " << num_images_ 
  //           << " " << depth_maps_host_data.size() << std::endl;

#pragma omp parallel for
  for (size_t i = 0; i < all_images_ids_.size(); ++i) {
    DepthMap depth_map =
        problems_.at(0).depth_maps->at(all_images_ids_[i]);
    float* dest =
        depth_maps_host_data.data() + max_width_ * max_height_ * i;
    for (size_t r = 0; r < depth_map.GetHeight(); ++r) {
      memcpy(dest, depth_map.GetPtr() + r * depth_map.GetWidth(),
            depth_map.GetWidth() * sizeof(float));
      dest += max_width_;
    }
  }

  std::cout << StringPrintf("InitDepthMaps Cost Time: %.3fs\n", timer.ElapsedSeconds());

  depth_maps_device_.reset(new CudaArrayWrapper<float>(
    max_width_, max_height_, num_images_));
  depth_maps_device_->CopyToDevice(depth_maps_host_data.data());

  // Create source depth maps texture.
  struct cudaResourceDesc resDesc;
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = cudaResourceTypeArray;
  resDesc.res.array.array = depth_maps_device_->GetPtr();

  struct cudaTextureDesc texDesc;
  memset(&texDesc, 0, sizeof(texDesc));
  texDesc.addressMode[0] = cudaAddressModeBorder;
  texDesc.addressMode[1] = cudaAddressModeBorder;
  texDesc.addressMode[2] = cudaAddressModeBorder;
  texDesc.filterMode = cudaFilterModePoint;
  texDesc.readMode = cudaReadModeElementType;
  texDesc.normalizedCoords = 0;

  CUDA_SAFE_CALL(
    cudaCreateTextureObject(&depth_maps_texture, &resDesc,
                            &texDesc, NULL));
  
}

void CrossFilterCuda::InitNormalMaps() {
  // std::cout << "num_images: " << num_images_ << std::endl;
  std::cout << "InitNormalMaps size: " << all_images_ids_.size() << std::endl;

  Timer timer;
  timer.Start();
  const unsigned int kDefaultValue = 0;
  std::vector<unsigned int> normal_maps_host_data(
    static_cast<size_t>(max_width_ * max_height_ * num_images_),
                        kDefaultValue);
  // std::cout << max_width_ << " " << max_height_ << " " << num_images_ 
  //           << " " << depth_maps_host_data.size() << std::endl;

#pragma omp parallel for
  for (size_t i = 0; i < all_images_ids_.size(); ++i) {
    NormalMap normal_map =
        problems_.at(0).normal_maps->at(all_images_ids_[i]);
    MatXui normal_encode_map = normal_map.Encode();
    unsigned int* dest =
        normal_maps_host_data.data() + max_width_ * max_height_ * i;
    for (size_t r = 0; r < normal_encode_map.GetHeight(); ++r) {
      memcpy(dest, normal_encode_map.GetPtr() + r * normal_encode_map.GetWidth(),
      normal_encode_map.GetWidth() * sizeof(unsigned int));
      dest += max_width_;
    }
  }

  std::cout << StringPrintf("InitNormalMaps Cost Time: %.3fs\n", timer.ElapsedSeconds());

  normal_maps_device_.reset(new CudaArrayWrapper<unsigned int>(
    max_width_, max_height_, num_images_));
  normal_maps_device_->CopyToDevice(normal_maps_host_data.data());

  // Create source depth maps texture.
  struct cudaResourceDesc resDesc;
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = cudaResourceTypeArray;
  resDesc.res.array.array = normal_maps_device_->GetPtr();

  struct cudaTextureDesc texDesc;
  memset(&texDesc, 0, sizeof(texDesc));
  texDesc.addressMode[0] = cudaAddressModeBorder;
  texDesc.addressMode[1] = cudaAddressModeBorder;
  texDesc.addressMode[2] = cudaAddressModeBorder;
  texDesc.filterMode = cudaFilterModePoint;
  // texDesc.filterMode = cudaFilterModeLinear;
  texDesc.readMode = cudaReadModeElementType;
  texDesc.normalizedCoords = 0;

  CUDA_SAFE_CALL(
    cudaCreateTextureObject(&normal_maps_texture, &resDesc,
                            &texDesc, NULL));
  
}

void CrossFilterCuda::ResetDepthMaps() {
  // std::cout << "num_images: " << num_images_ << std::endl;
  std::cout << "RefineDepthMaps size: " << all_images_ids_.size() << std::endl;
  
  Timer timer;
  timer.Start();

  const float kDefaultValue = 0.0f;
  std::vector<float> depth_maps_host_data(
    static_cast<size_t>(max_width_ * max_height_ * num_images_),
                        kDefaultValue);
  // std::cout << max_width_ << " " << max_height_ << " " << num_images_ 
  //           << " " << depth_maps_host_data.size() << std::endl;

#pragma omp parallel for
  for (size_t i = 0; i < all_images_ids_.size(); ++i) {
    DepthMap depth_map;
    if (ref_flags_[i]){
      depth_map = DepthMap(refined_depth_maps_.at(i)->CopyToMat(), 
        problems_.at(0).depth_maps->at(all_images_ids_[i]).GetDepthMin(), 
        problems_.at(0).depth_maps->at(all_images_ids_[i]).GetDepthMax());
    } else {
      depth_map = problems_.at(0).depth_maps->at(all_images_ids_[i]);
    }
        
    float* dest =
        depth_maps_host_data.data() + max_width_ * max_height_ * i;
    for (size_t r = 0; r < depth_map.GetHeight(); ++r) {
      memcpy(dest, depth_map.GetPtr() + r * depth_map.GetWidth(),
            depth_map.GetWidth() * sizeof(float));
      dest += max_width_;
    }
  }

  std::cout << StringPrintf("RefineDepthMaps Cost Time: %.3fs\n", timer.ElapsedSeconds());

  depth_maps_device_.reset(new CudaArrayWrapper<float>(
    max_width_, max_height_, num_images_));
  depth_maps_device_->CopyToDevice(depth_maps_host_data.data());

  // Create source depth maps texture.
  struct cudaResourceDesc resDesc;
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = cudaResourceTypeArray;
  resDesc.res.array.array = depth_maps_device_->GetPtr();

  struct cudaTextureDesc texDesc;
  memset(&texDesc, 0, sizeof(texDesc));
  texDesc.addressMode[0] = cudaAddressModeBorder;
  texDesc.addressMode[1] = cudaAddressModeBorder;
  texDesc.addressMode[2] = cudaAddressModeBorder;
  texDesc.filterMode = cudaFilterModePoint;
  texDesc.readMode = cudaReadModeElementType;
  texDesc.normalizedCoords = 0;

  CUDA_SAFE_CALL(
    cudaCreateTextureObject(&depth_maps_texture, &resDesc,
                            &texDesc, NULL));
}

void CrossFilterCuda::InitTransforms(const Problem& problem,
  int &ref_width,
  int &ref_height){
  const Image& ref_image = problem.images->at(problem.ref_image_idx);
  ref_width = ref_image.GetWidth();
  ref_height = ref_image.GetHeight();
  ref_K_host_[0] = ref_image.GetK()[0];
  ref_K_host_[1] = ref_image.GetK()[2];
  ref_K_host_[2] = ref_image.GetK()[4];
  ref_K_host_[3] = ref_image.GetK()[5];

  // Extract 1/fx, -cx/fx, 1/fy, -cy/fy.
  ref_inv_K_host_[0] = 1.0f / ref_K_host_[0];
  ref_inv_K_host_[1] = -ref_K_host_[1] / ref_K_host_[0];
  ref_inv_K_host_[2] = 1.0f / ref_K_host_[2];
  ref_inv_K_host_[3] = -ref_K_host_[3] / ref_K_host_[2];

  // Bind pose to constant global memory.
  SetRefK(ref_K_host_, 0);
  SetRefInvK(ref_inv_K_host_, 0);

  float rotated_R[9];
  memcpy(rotated_R, ref_image.GetR(), 9 * sizeof(float));

  float rotated_T[3];
  memcpy(rotated_T, ref_image.GetT(), 3 * sizeof(float));

  std::vector<float> poses_host_data;
  poses_host_data.resize(K_NUM_TFORM_PARAMS * problem.src_image_extend_idxs.size());
  int offset = 0;
  // for (const auto image_idx : problem.src_image_idxs) {
  for (int i = 0; i < problem.src_image_extend_idxs.size(); ++i) {
    const int image_idx = problem.src_image_extend_idxs[i];
    const Image& image = problem.images->at(image_idx);

    float iK[9];
    memcpy(iK, image.GetK(), sizeof(float) * 9);

    const float K[4] = {iK[0], iK[2], iK[4], iK[5]};
    memcpy(poses_host_data.data() + offset, K, 4 * sizeof(float));
    offset += 4;

    float rel_R[9];
    float rel_T[3];
    ComputeRelativePose(rotated_R, rotated_T, image.GetR(), image.GetT(),
    rel_R, rel_T);
    memcpy(poses_host_data.data() + offset, rel_R, 9 * sizeof(float));
    offset += 9;
    memcpy(poses_host_data.data() + offset, rel_T, 3 * sizeof(float));
    offset += 3;

    float C[3];
    ComputeProjectionCenter(rel_R, rel_T, C);
    memcpy(poses_host_data.data() + offset, C, 3 * sizeof(float));
    offset += 3;

    float P[12];
    ComposeProjectionMatrix(iK, rel_R, rel_T, P);
    memcpy(poses_host_data.data() + offset, P, 12 * sizeof(float));
    offset += 12;

    float inv_P[12];
    ComposeInverseProjectionMatrix(iK, rel_R, rel_T, inv_P);
    memcpy(poses_host_data.data() + offset, inv_P, 12 * sizeof(float));
    offset += 12;
  }

  poses_device_.reset(new CudaArrayWrapper<float>(
    K_NUM_TFORM_PARAMS, problem.src_image_extend_idxs.size(), 1));
  poses_device_->CopyToDevice(poses_host_data.data());

  struct cudaResourceDesc resDesc;
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = cudaResourceTypeArray;
  resDesc.res.array.array = poses_device_->GetPtr();

  struct cudaTextureDesc texDesc;
  memset(&texDesc, 0, sizeof(texDesc));
  texDesc.addressMode[0] = cudaAddressModeBorder;
  texDesc.addressMode[1] = cudaAddressModeBorder;
  texDesc.addressMode[2] = cudaAddressModeBorder;
  texDesc.filterMode = cudaFilterModePoint;
  texDesc.readMode = cudaReadModeElementType;
  texDesc.normalizedCoords = 0;

  CUDA_SAFE_CALL(
  cudaCreateTextureObject(&poses_texture, &resDesc, &texDesc, NULL));
}

void CrossFilterCuda::DestoryTransforms(){
  CUDA_SAFE_CALL(cudaDestroyTextureObject(poses_texture));
}

void CrossFilterCuda::InitWorkspace() {
  refined_depth_maps_.resize(num_images_);
  for (int i = 0; i < num_images_; i++){
    refined_depth_maps_.at(i).reset(new GpuMat<float>(1, 1));
  }
}

void CrossFilterCuda::InitDeltDepthMaps() {
  delt_depth_maps_.resize(num_images_);
  for (int i = 0; i < num_images_; i++){
    delt_depth_maps_.at(i).reset(new GpuMat<float>(1, 1));
  }
}

std::vector<int > CrossFilterCuda::FindSrcImageIds(const Problem& problem) {
  std::vector<int > src_images_idxs;
  int ref_id = problem.ref_image_idx;
  std::vector<int>::iterator iter=std::find(all_images_ids_.begin(),
                                       all_images_ids_.end(),ref_id);
  if(iter == all_images_ids_.end())      {
    std::cout << "no find ref image id " << ref_id 
              << "in all_images_ids" << std::endl;
    return src_images_idxs;
  } else{
    src_images_idxs.push_back(std::distance(all_images_ids_.begin(),iter));
  }

  // for (int i = 0; i < problem.src_image_idxs.size(); i++){
  //   int image_id = problem.src_image_idxs.at(i);
  for (int i = 0; i < problem.src_image_extend_idxs .size(); i++){
    int image_id = problem.src_image_extend_idxs.at(i);
    std::vector<int>::iterator iter=std::find(all_images_ids_.begin(),
                                         all_images_ids_.end(),image_id);
    if(iter == all_images_ids_.end())      {
      std::cout << "no find src image id " << image_id 
                << "in all_images_ids" << std::endl;
      continue;
    } else{
      src_images_idxs.push_back(std::distance(all_images_ids_.begin(),iter));
    }
  }
  return src_images_idxs;
}

std::vector<List > CrossFilterCuda::FindSrcImageLists(const Problem& problem) {
  std::vector<List > src_images_lists;
  int ref_id = problem.ref_image_idx;
  std::vector<int>::iterator iter=std::find(all_images_ids_.begin(),
                                       all_images_ids_.end(),ref_id);
  if(iter == all_images_ids_.end())      {
    std::cout << "no find ref image id " << ref_id 
              << "in all_images_ids" << std::endl;
    return src_images_lists;
  } else{
    int dist = std::distance(all_images_ids_.begin(),iter);
    List src_list;
    src_list.id = dist;
    src_list.flag = ref_flags_.at(dist);
    src_images_lists.push_back(src_list);
  }

  for (int i = 0; i < problem.src_image_extend_idxs.size(); i++){
    int image_id = problem.src_image_extend_idxs.at(i);
    std::vector<int>::iterator iter=std::find(all_images_ids_.begin(),
                                         all_images_ids_.end(),image_id);
    if(iter == all_images_ids_.end())      {
      std::cout << "no find src image id " << image_id 
                << "in all_images_ids" << std::endl;
      continue;
    } else{
      int dist = std::distance(all_images_ids_.begin(),iter);
      List src_list;
      src_list.id = dist;
      src_list.flag = ref_flags_.at(dist);
      src_images_lists.push_back(src_list);
    }
  }
  return src_images_lists;
}

bool CrossFilterCuda::Run() {
  std::cout << "CrossFilterCuda::Run" << std::endl;
  CudaTimer total_timer;

  InitDepthMaps();
  InitNormalMaps();
  InitWorkspace();
  CUDA_SYNC_AND_CHECK();
  // compute cuda configure
  ComputeCudaConfig();
  for(const auto &problem : problems_){
    int ref_width, ref_height;
    InitTransforms(problem, ref_width, ref_height);

    std::vector<int > vec_images_idxs = FindSrcImageIds(problem);
    if (vec_images_idxs.size() < 2){
      continue;
    }

    refined_depth_maps_.at(vec_images_idxs[0]).reset(
        new GpuMat<float>(ref_width, ref_height));
    refined_depth_maps_.at(vec_images_idxs[0])->FillWithScalar(0);

    int num_src_images = (int)vec_images_idxs.size() - 1;
    const int filter_min_num_consistent = min(options_.filter_min_num_consistent, 
      num_src_images);

    ImageIdxs image_idxs;
    memcpy(image_idxs.list, vec_images_idxs.data(),sizeof(int) * vec_images_idxs.size());
    image_idxs.length=vec_images_idxs.size();

    CrossFilterKernel<<<elem_wise_grid_size_, elem_wise_block_size_>>>(
      num_src_images, ref_width, ref_height,
      options_.geom_consistency_max_cost, 
      options_.filter_geom_consistency_max_cost,
      options_.max_normal_error,
      options_.depth_diff_threshold,
      filter_min_num_consistent, image_idxs, 
      *(refined_depth_maps_.at(vec_images_idxs[0])),
      depth_maps_texture, normal_maps_texture, poses_texture, 0);
    CUDA_SYNC_AND_CHECK();

    // DestoryTransforms();

    std::string src_images_str;
    for (int i = 1; i < vec_images_idxs.size(); i++){
      const auto id = vec_images_idxs.at(i);
      src_images_str += " ";
      src_images_str += std::to_string(all_images_ids_.at(id));
    }
    std::cout << "Cross Filter Problem: " << problem.ref_image_idx << " \tsrc(" 
              << num_src_images << "): " << src_images_str << std::endl;
  }

  CUDA_SYNC_AND_CHECK();

  depth_maps_device_.reset();
  normal_maps_device_.reset();
  CUDA_SAFE_CALL(cudaDestroyTextureObject(depth_maps_texture));
  CUDA_SAFE_CALL(cudaDestroyTextureObject(normal_maps_texture));

  poses_device_.reset();
  CUDA_SAFE_CALL(cudaDestroyTextureObject(poses_texture));

  total_timer.Print("CrossFilter Total");
  return true;
}

bool CrossFilterCuda::DedupRun() {
  std::cout << "DeduplicFilterCuda::Run" << std::endl;
  CudaTimer total_timer;

  ResetDepthMaps();
  refined_depth_maps_.clear();
  refined_depth_maps_.shrink_to_fit();

  InitDeltDepthMaps();
  ComputeCudaConfig();
  {
    visbility_maps_.reset(new GpuMat<uint32_t> (
      max_width_, max_height_, all_images_ids_.size()));
    FillGpuMat<<<elem_wise_grid_size_, elem_wise_block_size_>>>(
      max_width_, max_height_, *visbility_maps_);
    CUDA_SYNC_AND_CHECK();
  }

  for(const auto &problem : problems_){
    int ref_width, ref_height;
    InitTransforms(problem, ref_width, ref_height);

    std::vector<List > vec_images_lists = FindSrcImageLists(problem);
    if (vec_images_lists.size() < 2){
      continue;
    }

    int num_src_images = (int)vec_images_lists.size() - 1;
    const int filter_min_num_consistent = min(options_.filter_min_num_consistent, 
      num_src_images);

    ImageLists image_list;
    memcpy(image_list.list, vec_images_lists.data(), sizeof(List) * vec_images_lists.size());
    image_list.length = vec_images_lists.size();
    
    delt_depth_maps_.at(vec_images_lists[0].id).reset(
      new GpuMat<float>(ref_width, ref_height));
    delt_depth_maps_.at(vec_images_lists[0].id)->FillWithScalar(0.0f);

    DeduplicFilterKernel<<<elem_wise_grid_size_, elem_wise_block_size_>>>(
      num_src_images, ref_width, ref_height,
      options_.filter_geom_consistency_max_cost,
      options_.depth_diff_threshold,
      filter_min_num_consistent, image_list, 
      *(delt_depth_maps_.at(vec_images_lists[0].id)),
      *visbility_maps_,
      depth_maps_texture, poses_texture, 0);
    CUDA_SYNC_AND_CHECK();

    std::string src_images_str;
    for (int i = 1; i < vec_images_lists.size(); i++){
      const auto& list = vec_images_lists.at(i);
      src_images_str += " ";
      src_images_str += std::to_string(all_images_ids_.at(list.id));
    }
    std::cout << "Deduplic Filter Problem: " << problem.ref_image_idx << " \tsrc(" 
              << num_src_images << "): " << src_images_str << std::endl;
  }

  poses_device_.reset();
  CUDA_SAFE_CALL(cudaDestroyTextureObject(poses_texture));

  total_timer.Print("DeduplicFilter Total");
  return true;
}

DepthMap CrossFilterCuda::GetRefineDepthMap(int image_id, DepthMapInfo info) {

  int image_idx = -1;
  std::vector<int>::iterator iter=std::find(all_images_ids_.begin(),
                                       all_images_ids_.end(),image_id);
  image_idx = std::distance(all_images_ids_.begin(),iter);
  if (image_idx > all_images_ids_.size() - 1){
    std::cout << "GetRefineDepthMap: nofind image " << image_id << " in all_images_ids" << std::endl;
    return DepthMap(info.width, info.height, info.depth_min, info.depth_max);
  }
  return DepthMap(refined_depth_maps_.at(image_idx)->CopyToMat(), 
                  info.depth_min, info.depth_max);
}

DepthMap CrossFilterCuda::GetDeduplicDepthMap(int image_id, DepthMapInfo info) {

  int image_idx = -1;
  std::vector<int>::iterator iter=std::find(all_images_ids_.begin(),
                                       all_images_ids_.end(),image_id);
  image_idx = std::distance(all_images_ids_.begin(),iter);
  if (image_idx > all_images_ids_.size() - 1){
    std::cout << "GetDeduplicDepthMap: nofind image " << image_id << " in all_images_ids" << std::endl;
    return DepthMap();
  }

  std::unique_ptr<GpuMat<float>> deduplic_depth_map(
    new GpuMat<float>(info.width, info.height));
  deduplic_depth_map->FillWithScalar(0.0f);

  ComputeCudaConfig();
  GetDeduplicDepthMapsLayer<<<elem_wise_grid_size_, elem_wise_block_size_>>>(
    *visbility_maps_, image_idx, info.width, info.height, 
    *deduplic_depth_map, depth_maps_texture);
  CUDA_SYNC_AND_CHECK();

  return DepthMap(deduplic_depth_map->CopyToMat(), 
                  info.depth_min, info.depth_max);
}

DepthMap CrossFilterCuda::GetDeltDepthMap(int image_id) {
  int image_idx = -1;
  std::vector<int>::iterator iter=std::find(all_images_ids_.begin(),
                                       all_images_ids_.end(),image_id);
  image_idx = std::distance(all_images_ids_.begin(),iter);
  if (image_idx > all_images_ids_.size() - 1){
    std::cout << "GetRefineDepthMap: nofind image " << image_id << " in all_images_ids" << std::endl;
    return DepthMap();
  }
  return DepthMap(delt_depth_maps_.at(image_idx)->CopyToMat(), 0.0f, 1.0f);
}

Mat<uint32_t> CrossFilterCuda::GetVisibilyMap(const int image_id){

  const Image& image = problems_[0].images->at(image_id);
  int width = image.GetWidth();
  int height = image.GetHeight();

  std::vector<int>::iterator iter=std::find(all_images_ids_.begin(),
                                       all_images_ids_.end(),image_id);
  int image_cross_id = -1;
  if(iter == all_images_ids_.end()) {
    return Mat<uint32_t>();
  } else{
    image_cross_id = std::distance(all_images_ids_.begin(),iter);
  }

  std::unique_ptr<GpuMat<uint32_t> > visbility_map;
  visbility_map.reset(new GpuMat<uint32_t>(width, height));

  ComputeCudaConfig();
  GetVisbilityMapsLayer<<<elem_wise_grid_size_, elem_wise_block_size_>>>(
    *visbility_maps_, image_cross_id, width, height,
    *visbility_map);
  CUDA_SYNC_AND_CHECK();

  return visbility_map->CopyToMat();
}


void CrossFilterCuda::DestroyDepthMapTexture(){
  depth_maps_device_.reset();
  CUDA_SAFE_CALL(cudaDestroyTextureObject(depth_maps_texture));

}

}  // namespace mvs
}  // namespace colmap
