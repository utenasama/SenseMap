//Copyright (c) 2019, SenseTime Group.
//All rights reserved.

#ifndef SENSEMAP_MVS_PATCH_MATCH_CUDA_H_
#define SENSEMAP_MVS_PATCH_MATCH_CUDA_H_

#include <iostream>
#include <memory>
#include <vector>

#include <cuda_runtime.h>

#include "util/types.h"
#include "util/bitmap.h"
#include "util/gpu_mat.h"
#include "base/image.h"
#include "controllers/patch_match_options.h"

#include "mvs/cuda_array_wrapper.h"
#include "mvs/gpu_mat_prng.h"
#include "mvs/gpu_mat_ref_image.h"
#include "mvs/consistency_graph.h"
#include "mvs/depth_map.h"
#include "mvs/normal_map.h"
#include "mvs/image.h"
#include "mvs/utils.h"

namespace sensemap {
namespace mvs {

using namespace utility;

class PatchMatchBase {
public:
    PatchMatchBase(const PatchMatchOptions &options,
                   const Problem &problem);
    ~PatchMatchBase();

    virtual bool Run() = 0;

    virtual void InitTransforms() = 0;

    DepthMap GetDepthMap() const;
    NormalMap GetNormalMap() const;
    std::vector<int> GetConsistentImageIdxs() const;

protected:

    void ComputeCudaConfig();

    void InitRefImage();
    void InitSourceImages();

    const PatchMatchOptions options_;
    const Problem problem_;

    // Dimensions for sweeping from top to bottom, i.e. one thread per column.
    dim3 sweep_block_size_;
    dim3 sweep_grid_size_;
    // Dimensions for element-wise operations, i.e. one thread per pixel.
    dim3 elem_wise_block_size_;
    dim3 elem_wise_grid_size_;
    // Dimensions for block sweeping, i.e. one thread per column in block.
    dim3 box_block_size_;
    dim3 box_grid_size_;

    // Original (not rotated) dimension of reference image.
    size_t ref_width_;
    size_t ref_height_;

    std::unique_ptr<CudaArrayWrapper<uint8_t> > ref_image_device_;
    std::unique_ptr<CudaArrayWrapper<uint8_t> > src_images_device_;
    std::unique_ptr<CudaArrayWrapper<uint8_t> > ref_semantic_device_;
    std::unique_ptr<CudaArrayWrapper<uint8_t> > src_semantics_device_;
    std::unique_ptr<CudaArrayWrapper<uint8_t> > ref_mask_device_;
    std::unique_ptr<CudaArrayWrapper<uint8_t> > src_masks_device_;
    std::unique_ptr<CudaArrayWrapper<float> > src_depth_maps_device_;

    // Data for reference image.
    std::unique_ptr<GpuMatRefImage> ref_image_;
    std::unique_ptr<GpuMat<float> > depth_map_;
    std::unique_ptr<GpuMat<float> > normal_map_;
    std::unique_ptr<GpuMatPRNG> rand_state_map_;
    std::unique_ptr<GpuMat<uint8_t> > consistency_mask_;

    std::unique_ptr<GpuMat<float> > gradient_map_;

    cudaTextureObject_t ref_image_texture = 0;
    cudaTextureObject_t src_images_texture = 0;
    cudaTextureObject_t ref_semantic_texture = 0;
    cudaTextureObject_t src_semantics_texture = 0;
    cudaTextureObject_t ref_mask_texture = 0;
    cudaTextureObject_t src_depth_maps_texture = 0;
};

class PatchMatchCuda : public PatchMatchBase {
 public:
  PatchMatchCuda(const PatchMatchOptions &options,
                 const Problem &problem);
  ~PatchMatchCuda();

  bool Run() override;

    Mat<float> GetSelProbMap() const;

 protected:

  template <int kWindowSize, int kWindowStep>
  void RunWithWindowSizeAndStep();

  void InitTransforms() override;
  void InitWorkspaceMemory();

  // Rotate reference image by 90 degrees in counter-clockwise direction.
  void Rotate();

protected:

  // Rotation of reference image in pi/2. This is equivalent to the number of
  // calls to `rotate` mod 4.
  int rotation_in_half_pi_;

  // Relative poses from rotated versions of reference image to source images
  // corresponding to _rotationInHalfPi:
  //
  //    [S(1), S(2), S(3), ..., S(n)]
  //
  // where n is the number of source images and:
  //
  //    S(i) = [K_i(0, 0), K_i(0, 2), K_i(1, 1), K_i(1, 2), R_i(:), T_i(:)
  //            C_i(:), P(:), P^-1(:)]
  //
  // where i denotes the index of the source image and K is its calibration.
  // R, T, C, P, P^-1 denote the relative rotation, translation, camera
  // center, projection, and inverse projection from there reference to the
  // i-th source image.
  std::unique_ptr<CudaArrayWrapper<float> > poses_device_[4];
  cudaTextureObject_t poses_texture = 0;

  // Calibration matrix for rotated versions of reference image
  // as {K[0, 0], K[0, 2], K[1, 1], K[1, 2]} corresponding to _rotationInHalfPi.
  float ref_K_host_[4][4];
  float ref_inv_K_host_[4][4];
  
  std::unique_ptr<GpuMat<float> > cost_map_;
  std::unique_ptr<GpuMat<float> > sel_prob_map_;
  std::unique_ptr<GpuMat<float> > prev_sel_prob_map_;
  
  // Shared memory is too small to hold local state for each thread,
  // so this is workspace memory in global memory.
  std::unique_ptr<GpuMat<float> > global_workspace_;
};

class PatchMatchACMMCuda : public PatchMatchBase {
public:
    PatchMatchACMMCuda(const PatchMatchOptions &options,
                       const Problem &problem);
    ~PatchMatchACMMCuda();

    Mat<float> GetConfMap() const;

    Mat<unsigned short> GetCurvatureMap() const;

    bool Run() override;
    bool RunFilter(bool deduplication_flag = false);

protected:
    void Preprocess();
    void InitTransforms() override;
    void InitWorkspace();
    void InitWorkspaceFilter();
    void InitConfMapFilter();
    void InitPriorDepthAndNormalInfo();

protected:

    std::unique_ptr<CudaArrayWrapper<float> > ref_sum_image_device_;
    std::unique_ptr<CudaArrayWrapper<float> > ref_squared_sum_image_device_;
    std::unique_ptr<CudaArrayWrapper<float> > poses_device_;

    //prior_depth_map from rgbd
    std::unique_ptr<GpuMat<float>> prior_depth_map_;
    std::unique_ptr<GpuMat<float>> prior_normal_map_;
    std::unique_ptr<GpuMat<float>> prior_wgt_map_;

    std::unique_ptr<GpuMat<float> > conf_map_;
    std::unique_ptr<GpuMat<unsigned short> > grad_curvature_map_;
    std::unique_ptr<GpuMat<float> > planarity_map_;
    std::unique_ptr<GpuMat<float> > geom_gradient_map_;

    std::unique_ptr<GpuMat<uint32_t> > view_sel_map_;
    // std::unique_ptr<GpuMat<int> > coords_idx_map_;
    // std::unique_ptr<GpuMat<uint32_t> > selected_images_list_;

    std::unique_ptr<GpuMat<float> > cost_map_;

    float ref_K_host_[4];
    float ref_inv_K_host_[4];

    cudaTextureObject_t ref_sum_image_texture = 0;
    cudaTextureObject_t ref_squared_sum_image_texture = 0;
    cudaTextureObject_t poses_texture = 0;
};

class CrossFilterCuda {
public:
    CrossFilterCuda(const PatchMatchOptions &options,
                    const std::vector<Problem> &problems,
                    const std::vector<int > &whole_ids,
                    const std::vector<bool > &ref_flags);
    ~CrossFilterCuda();

    bool Run();
    bool DedupRun();

    DepthMap GetRefineDepthMap(int image_id, DepthMapInfo info);
    DepthMap GetDeduplicDepthMap(int image_id, DepthMapInfo info);
    DepthMap GetDeltDepthMap(int image_id);
    Mat<uint32_t> GetVisibilyMap(int image_id);
    void DestroyDepthMapTexture();


protected:

    void ComputeCudaConfig();

    void InitImages();
    void InitDepthMaps();
    void InitNormalMaps();
    void ResetDepthMaps();
    void InitTransforms(const Problem& problem,
                        int &ref_width,
                        int &ref_height);
    void DestoryTransforms();

    void InitWorkspace();
    void InitDeltDepthMaps();
    std::vector<int > FindSrcImageIds(const Problem& problem);
    std::vector<List > FindSrcImageLists(const Problem& problem);

    const PatchMatchOptions options_;
    const std::vector<Problem> problems_;
    std::vector<int > all_images_ids_;
    std::vector<bool > ref_flags_;

    // Dimensions for element-wise operations, i.e. one thread per pixel.
    dim3 elem_wise_block_size_;
    dim3 elem_wise_grid_size_;

    // Original (not rotated) dimension of reference image.
    int num_images_;
    size_t max_width_;
    size_t max_height_;

    std::vector<std::unique_ptr<GpuMat<float>>> refined_depth_maps_;
    std::vector<std::unique_ptr<GpuMat<float>>> delt_depth_maps_;
    std::unique_ptr<GpuMat<uint32_t>> visbility_maps_;

    float ref_K_host_[4];
    float ref_inv_K_host_[4];
    std::unique_ptr<CudaArrayWrapper<float> > poses_device_;

    std::unique_ptr<CudaArrayWrapper<float> > depth_maps_device_;
    std::unique_ptr<CudaArrayWrapper<unsigned int> > normal_maps_device_;

    cudaTextureObject_t depth_maps_texture = 0;
    cudaTextureObject_t normal_maps_texture = 0;
    cudaTextureObject_t poses_texture = 0;
};

}  // namespace mvs
}  // namespace colmap

#endif  // SENSEMAP_MVS_PATCH_MATCH_CUDA_H_
