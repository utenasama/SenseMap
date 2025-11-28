//Copyright (c) 2019, SenseTime Group.
//All rights reserved.
#ifndef SENSEMAP_MVS_KERNEL_FUNCTIONS_H_
#define SENSEMAP_MVS_KERNEL_FUNCTIONS_H_

#define _USE_MATH_DEFINES

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <sstream>

#include "util/cuda.h"
#include "util/cudacc.h"
#include "util/logging.h"
#include "util/misc.h"
#include "util/gpu_mat.h"
#include "util/semantic_table.h"

#include "mvs/utils.h"

// The number of threads per Cuda thread. Warning: Do not change this value,
// since the templated window sizes rely on this value.
#define THREADS_PER_BLOCK 32
#define SWEEP_BLOCK_SIZE_1 64
#define SWEEP_BLOCK_SIZE_2 73
#define MAX_NUM_SRC_IMAGE 8
#define MAX_NUM_EXTEND_SRC_IMAGE (MAX_NUM_CROSS_SRC + 1)
#define K_NUM_TFORM_PARAMS (4 + 9 + 3 + 3 + 12 + 12)
#define K_NUM_SAMPS 8
#define K_NUM_EACH_SAMP 10
#define MAX_NUM_NEIGHBOR 8
#define MAX_CIRCULAR_NEIGHBOR 4

// We must not include "util/math.h" to avoid any Eigen includes here,
// since Visual Studio cannot compile some of the Eigen/Boost expressions.
#ifndef DEG2RAD
#define DEG2RAD(deg) deg * 0.0174532925199432
#endif
#ifndef RAD2DEG
#define RAD2DEG(rad) rad * 57.295779513082322
#endif
namespace sensemap {
namespace mvs {

using namespace utility;

// const size_t max_num_src_image = 10;
// const size_t max_num_extend_src_image = MAX_NUM_CROSS_SRC + 1;
// const size_t kNumTformParams = 4 + 9 + 3 + 3 + 12 + 12;
// const size_t kNumSamps = 8;
// const size_t kNumEachSamp = 10;
// const size_t max_num_neighbor = 8;
// const size_t max_circular_neighbor = 4;

struct ImageLists{
List list[MAX_NUM_EXTEND_SRC_IMAGE];
int length;
} ;

struct ImageIdxs{
int list[MAX_NUM_EXTEND_SRC_IMAGE];
int length;
} ;

struct SweepOptions {
  int thread_index = 0;
  float perturbation = 1.0f;
  float depth_min = 0.0f;
  float depth_max = 1.0f;
  int num_samples = 15;
  int window_radius = 5;
  int window_step = 1;
  float sigma_spatial = 3.0f;
  float sigma_color = 0.3f;
  float ncc_sigma = 0.6f;
  float min_triangulation_angle = 0.5f;
  float incident_angle_sigma = 0.9f;
  float prev_sel_prob_weight = 0.0f;
  float geom_consistency_regularizer = 0.1f;
  float geom_consistency_max_cost = 5.0f;
  float filter_min_ncc = 0.1f;
  float filter_min_triangulation_angle = 3.0f;
  int filter_min_num_consistent = 2;
  float filter_geom_consistency_max_cost = 1.0f;

  //// ACMH Options.
  float random_depth_ratio = 0.004f;
  float random_angle1_range = DegToRad(20.0f);
  float random_angle2_range = DegToRad(12.0f);
  float random_smooth_bonus = 0.93f;
  float diff_photometric_consistency = 0.1f;
  float ncc_thres_refine = 0.03;
  float init_ncc_matching_cost = 0.8f;
  float th_mc = init_ncc_matching_cost;
  float max_ncc_matching_cost = 1.2f;
  float ncc_thres_keep = 0.5f;
  float conf_thres_small = 0.25 * ncc_thres_keep;
  float conf_thres_big = 0.5 * ncc_thres_keep;
  float alpha = 90.0f;
  float beta = 0.3f;
  int num_good_hypothesis = 2;
  int num_bad_hypothesis = 3;
  int thk = 4;
  bool plane_regularizer = false;
  bool geom_consistency = true;
  bool random_optimization = true;
  bool propagate_depth = false;
  bool prior_depth_ncc = false;
};

// Rotate normals by 90deg around z-axis in counter-clockwise direction.
__global__ void InitNormalMap(GpuMat<float> normal_map,
                              GpuMat<curandState> rand_state_map,
                              const int thread_index);

__global__ void InitDepthMap(GpuMat<float> depth_map,
                             GpuMat<float> normal_map,
                             GpuMat<curandState> rand_state_map,
                             const float depth_min,
                             const float depth_max,
                             const int thread_index);

__global__ void InitSelectedImageList(GpuMat<curandState> rand_state_map,
                                      GpuMat<uint32_t> view_sel_map,
                                      const int num_src_image,
                                      const int max_num_src_image);

__global__ void GenerateProjectDepthMaps(const int max_width,
                                        const int max_height,
                                        const int src_image_idx,
                                        const int num_src_image,
                                        GpuMat<float> composite_depth_map,
                                        // GpuMat<float> composite_normal_map,
                                        cudaTextureObject_t depth_maps_texture,
                                        cudaTextureObject_t poses_texture,
                                        const int thread_index);

__global__ void CompositeInitDepthMap(GpuMat<curandState> rand_state_map,
  GpuMat<float> depth_map, GpuMat<float> prior_depth_map, GpuMat<float> composite_depth_map,
  const float depth_min, const float depth_max);

__global__ void CompositeInitNormalMap(GpuMat<float> depth_map, GpuMat<float> normal_map, const int tid);

__global__ void PreprocessImageKernel(GpuMat<float> ref_sum_image,
                                      GpuMat<float> ref_squared_sum_image,
                                      const int window_radius,
                                      const int window_step,
                                      BilateralWeightComputer weight_computer,
                                      cudaTextureObject_t ref_image_texture);

__global__ void InitConfMapKernel(const SweepOptions options,
                                  const int num_src_image, 
                                  GpuMat<float> depth_map,
                                  GpuMat<float> normal_map,
                                  GpuMat<float> conf_map,
                                  GpuMat<float> cost_map,
                                  GpuMat<uint32_t> view_sel_map,
                                  GpuMat<float> prior_depth_map,
                                  GpuMat<float> prior_normal_map,
                                  GpuMat<float> prior_wgt_map,
                                  BilateralWeightComputer weight_computer,
                                  cudaTextureObject_t ref_image_texture,
                                  cudaTextureObject_t ref_sum_image_texture,
                                  cudaTextureObject_t ref_squared_sum_image_texture,
                                  cudaTextureObject_t src_images_texture,
                                  // cudaTextureObject_t ref_semantic_texture,
                                  // cudaTextureObject_t src_semantics_texture,
                                  cudaTextureObject_t src_depth_maps_texture,
                                  cudaTextureObject_t poses_texture,
                                  const int thread_index);

__global__ void ComputeCurvatureMap(cudaTextureObject_t ref_image_texture,
                                    GpuMat<float> normal_map,
                                    GpuMat<unsigned short> grad_curvature_map);

__global__ void ComputeGradientMap(GpuMat<float> grad_map, cudaTextureObject_t ref_image_texture);

__global__ void DilateGradientMap(GpuMat<float> grad_map, const int r = 1);

__global__ void InitializeDistMap(const GpuMat<float> grad_map, 
                                  GpuMat<float> dist_map,
                                  const float min_grad_thres);

__global__ void ComputeRowDistMap(GpuMat<float> dist_map);

__global__ void ComputeColDistMap(GpuMat<float> dist_map);

__global__ void ComputeCovarianceMap(const GpuMat<float> depth_map, 
                                     GpuMat<float> covariance_map);

#if DENSE_SHARPNESS == DENSE_SHARPNESS_GRAD
__global__ void ComputeGradientsMap(GpuMat<float> grad_maps,
  cudaTextureObject_t src_images_texture);
#endif

__global__ void RectifyConfidenceMap(GpuMat<float> depth_map,
                                    GpuMat<float> normal_map,
                                    GpuMat<float> conf_map,
                                    // GpuMat<float> view_sel_map,
                                    const int num_src_image,
                                    const SweepOptions options,
                                    cudaTextureObject_t ref_image_texture,
                                    cudaTextureObject_t src_images_texture,
                                    cudaTextureObject_t poses_texture,
                                    const float depth_diff_thres,
                                    const float conf_thres,
                                    const int thread_index);

template<int WINDOW_RADIUS, int WINDOW_STEP>
__global__ void EstimateLocalPlanarity(const SweepOptions options,
                                       GpuMat<float> depth_map,
                                       GpuMat<float> normal_map,
                                       GpuMat<float> planarity_map) {
  const int row = blockDim.y * blockIdx.y + threadIdx.y;
  const int col = blockDim.x * blockIdx.x + threadIdx.x;
  const int width = depth_map.GetWidth();
  const int height = depth_map.GetHeight();
  if (col >= width - WINDOW_RADIUS || row >= height - WINDOW_RADIUS ||
      col < WINDOW_RADIUS || row < WINDOW_RADIUS) {
    return;
  }
  int num_consistent = 0, num_pixel = 0;
  const float depth = depth_map.Get(row, col);
  // for (int r = row - window_radius; r <= row + window_radius; ++r) {
  //   for (int c = col - window_radius; c <= col + window_radius; ++c) {
  #pragma unroll
  for (int dr = -WINDOW_RADIUS; dr <= WINDOW_RADIUS; dr += WINDOW_STEP) {
  #pragma unroll
    for (int dc = -WINDOW_RADIUS; dc <= WINDOW_RADIUS; dc += WINDOW_STEP) {
      const float ndepth = depth_map.Get(dr + row , dc + col);
      const float diff_depth = fabs(depth - ndepth) / depth;
      num_consistent += (diff_depth < 0.01);
      num_pixel = num_pixel + 1;
    }
  }
  planarity_map.Set(row, col, num_consistent * 1.0f / num_pixel);
}

__global__ void BlackPixelUpdate(GpuMat<curandState> rand_state_map,
                                GpuMat<float> depth_map,
                                GpuMat<float> normal_map,
                                GpuMat<float> conf_map,
                                const int num_src_image,
                                const SweepOptions options,
                                BilateralWeightComputer weight_computer,
                                cudaTextureObject_t ref_image_texture,
                                cudaTextureObject_t ref_sum_image_texture,
                                cudaTextureObject_t ref_squared_sum_image_texture,
                                cudaTextureObject_t src_images_texture,
                                cudaTextureObject_t src_depth_maps_texture,
                                cudaTextureObject_t poses_texture);

__global__ void RedPixelUpdate(GpuMat<curandState> rand_state_map,
                              GpuMat<float> depth_map,
                              GpuMat<float> normal_map,
                              GpuMat<float> conf_map,
                              const int num_src_image,
                              const SweepOptions options,
                              BilateralWeightComputer weight_computer,
                              cudaTextureObject_t ref_image_texture,
                              cudaTextureObject_t ref_sum_image_texture,
                              cudaTextureObject_t ref_squared_sum_image_texture,
                              cudaTextureObject_t src_images_texture,
                              cudaTextureObject_t src_depth_maps_texture,
                              cudaTextureObject_t poses_texture);

__global__ void BlackPixelUpdateOpt(GpuMat<curandState> rand_state_map,
                                    GpuMat<float> depth_map,
                                    GpuMat<float> normal_map,
                                    GpuMat<float> conf_map,
                                    GpuMat<float> grad_map,
                                    GpuMat<float> cost_map,
                                    GpuMat<uint32_t> view_sel_map,
                                    // GpuMat<float> planarity_map,
                                    GpuMat<float> prior_depth_map,
                                    GpuMat<float> prior_normal_map,
                                    GpuMat<float> prior_wgt_map,
                                    // GpuMat<uint32_t> selected_images_list,
                                    const int num_src_image,
                                    const SweepOptions options,
                                    BilateralWeightComputer weight_computer,
                                    cudaTextureObject_t ref_image_texture,
                                    cudaTextureObject_t ref_sum_image_texture,
                                    cudaTextureObject_t ref_squared_sum_image_texture,
                                    cudaTextureObject_t src_images_texture,
                                    // cudaTextureObject_t ref_semantic_texture,
                                    // cudaTextureObject_t src_semantics_texture,
                                    cudaTextureObject_t src_depth_maps_texture,
                                    cudaTextureObject_t poses_texture);

__global__ void RedPixelUpdateOpt(GpuMat<curandState> rand_state_map,
                                  GpuMat<float> depth_map,
                                  GpuMat<float> normal_map,
                                  GpuMat<float> conf_map,
                                  GpuMat<float> grad_map,
                                  GpuMat<float> cost_map,
                                  GpuMat<uint32_t> view_sel_map,
                                  // GpuMat<float> planarity_map,
                                  GpuMat<float> prior_depth_map,
                                  GpuMat<float> prior_normal_map,
                                  GpuMat<float> prior_wgt_map,
                                  // GpuMat<uint32_t> selected_images_list,
                                  const int num_src_image,
                                  const SweepOptions options,
                                  BilateralWeightComputer weight_computer,
                                  cudaTextureObject_t ref_image_texture,
                                  cudaTextureObject_t ref_sum_image_texture,
                                  cudaTextureObject_t ref_squared_sum_image_texture,
                                  cudaTextureObject_t src_images_texture,
                                  // cudaTextureObject_t ref_semantic_texture,
                                  // cudaTextureObject_t src_semantics_texture,
                                  cudaTextureObject_t src_depth_maps_texture,
                                  cudaTextureObject_t poses_texture);

__global__ void BlackPixelLocalOptimization(GpuMat<curandState> rand_state_map,
                                  GpuMat<float> depth_map,
                                  GpuMat<float> normal_map,
                                  GpuMat<float> conf_map,
                                  GpuMat<float> cost_map,
                                  GpuMat<uint32_t> view_sel_map,
                                  // GpuMat<uint32_t> selected_images_list,
                                  const int num_src_image,
                                  const SweepOptions options,
                                  BilateralWeightComputer weight_computer,
                                  cudaTextureObject_t ref_image_texture,
                                  cudaTextureObject_t ref_sum_image_texture,
                                  cudaTextureObject_t ref_squared_sum_image_texture,
                                  cudaTextureObject_t src_images_texture,
                                  // cudaTextureObject_t ref_semantic_texture,
                                  // cudaTextureObject_t src_semantics_texture,
                                  cudaTextureObject_t src_depth_maps_texture,
                                  cudaTextureObject_t poses_texture);

__global__ void RedPixelLocalOptimization(GpuMat<curandState> rand_state_map,
                                  GpuMat<float> depth_map,
                                  GpuMat<float> normal_map,
                                  GpuMat<float> conf_map,
                                  GpuMat<float> cost_map,
                                  GpuMat<uint32_t> view_sel_map,
                                  // GpuMat<uint32_t> selected_images_list,
                                  const int num_src_image,
                                  const SweepOptions options,
                                  BilateralWeightComputer weight_computer,
                                  cudaTextureObject_t ref_image_texture,
                                  cudaTextureObject_t ref_sum_image_texture,
                                  cudaTextureObject_t ref_squared_sum_image_texture,
                                  cudaTextureObject_t src_images_texture,
                                  // cudaTextureObject_t ref_semantic_texture,
                                  // cudaTextureObject_t src_semantics_texture,
                                  cudaTextureObject_t src_depth_maps_texture,
                                  cudaTextureObject_t poses_texture);

//
/*
                 *     *
              *     *     *
                 *  +  *
              *     *     *
                 *     *
*/
//
__global__ void BilateralFilter(cudaTextureObject_t ref_image_texture,
    GpuMat<float> depth_map, GpuMat<float> normal_map, 
    GpuMat<float> filtered_depth_map, GpuMat<float> filtered_normal_map);

__global__ void MaskDepthMapKernel(GpuMat<float> depth_map, cudaTextureObject_t ref_mask_texture);

__global__ void ComputeGeomGradientMap(GpuMat<float> depth_map, GpuMat<float> grad_map);

__global__ void BlackPixelFilter(GpuMat<float> depth_map, GpuMat<float> normal_map, 
  GpuMat<float> grad_map, const float diff_thres);

__global__ void RedPixelFilter(GpuMat<float> depth_map, GpuMat<float> normal_map, 
  GpuMat<float> grad_map, const float diff_thres);

__global__ void FilterDepthMap(GpuMat<float> depth_map,
                               GpuMat<float> normal_map,
                               GpuMat<float> conf_map,
                               const float depth_diff_thres,
                               const float conf_thres,
                               const int thread_index,
                               const bool red_or_black);

__global__ void FilterDepthMapsKernel(const int num_src_image,
                                  GpuMat<float> depth_map,
                                  // GpuMat<float> normal_map,
                                  // GpuMat<float> conf_map,
                                  float geom_consistency_max_cost,
                                  float filter_geom_consistency_max_cost,
                                  float filter_neighbor_depth_error,
                                  int filter_min_num_consistent,
                                  float fitler_conf_threshold,
                                  cudaTextureObject_t src_depth_maps_texture,
                                  cudaTextureObject_t poses_texture,
                                  const int thread_index);

__global__ void DeduplicDepthMapsKernel(const int num_src_image,
                                  GpuMat<float> depth_map,
                                  // GpuMat<float> normal_map,
                                  float geom_consistency_max_cost,
                                  float filter_geom_consistency_max_cost,
                                  float filter_neighbor_depth_error,
                                  int filter_min_num_consistent,
                                  cudaTextureObject_t src_depth_maps_texture,
                                  cudaTextureObject_t poses_texture,
                                  const int thread_index);

__global__ void CrossFilterKernel(const int num_src_image,
                                  const int ref_width,
                                  const int ref_height,
                                  float geom_consistency_max_cost,
                                  float filter_geom_consistency_max_cost,
                                  float max_normal_error,
                                  float depth_diff_threshold,
                                  int filter_min_num_consistent,
                                  ImageIdxs images_idxs,
                                  GpuMat<float> refine_depth_map,
                                  cudaTextureObject_t depth_maps_texture,
                                  cudaTextureObject_t normal_maps_texture,
                                  cudaTextureObject_t poses_texture,
                                  const int thread_index);

__global__ void DeduplicFilterKernel(const int num_src_image,
                                     const int ref_width,
                                     const int ref_height,
                                     float filter_geom_consistency_max_cost,
                                     float depth_diff_threshold,
                                     int filter_min_num_consistent,
                                     ImageLists images_idxs,
                                     GpuMat<float> delt_depth_map,
                                     GpuMat<uint32_t> visbility_maps,
                                     cudaTextureObject_t depth_maps_texture,
                                     cudaTextureObject_t poses_texture,
                                     const int thread_index);

__host__ void SetRefK(const float * ref_K_host, int thread_index);

__host__ void SetRefInvK(const float * ref_inv_K_host, int thread_index);

__global__ void FillGpuMat(const int max_width, const int max_height,
  GpuMat<uint32_t> visbility_maps);

__global__ void GetDeduplicDepthMapsLayer(
  const GpuMat<uint32_t> visbility_maps,
  const int image_id,
  const int width,
  const int height,
  GpuMat<float > depth_map,
  cudaTextureObject_t depth_maps_texture);

__global__ void GetVisbilityMapsLayer(
  const GpuMat<uint32_t> visbility_maps,
  const int image_cross_id,
  const int width,
  const int height,
  GpuMat<uint32_t> visbility_map);

} // namespace mvs
} // namespace sensemap

#endif
