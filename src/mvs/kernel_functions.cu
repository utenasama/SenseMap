//Copyright (c) 2019, SenseTime Group.
//All rights reserved.
#include "kernel_functions.h"

#define PRIOR_WGT_COST
#define PRIOR_WGT_PROGET
#define PRIOR_LIMIT

const float decode_factor = 32767;

namespace sensemap {
namespace mvs {

using namespace utility;


// Calibration of reference image as {fx, cx, fy, cy}.
__constant__ float ref_K[4 * MAX_THREADS_PER_GPU];
// Calibration of reference image as {1/fx, -cx/fx, 1/fy, -cy/fy}.
__constant__ float ref_inv_K[4 * MAX_THREADS_PER_GPU];

__constant__ float mweights[5] = {0.f, 1.f, 2.01f, 3.03f, 4.06f};
__constant__ int neighbor_offs[MAX_NUM_NEIGHBOR][2] = 
    { { 0, -1 }, { -1, 0 }, { 0, 1 }, { 1, 0 }, 
      { -1, -1}, { -1, 1 }, { 1, 1 }, { 1, -1} };

__constant__ int dirSamples[K_NUM_SAMPS][K_NUM_EACH_SAMP * 2] = {
    /* red    */ 
    {/*0, -1, */-1, -2, 1, -2, -2, -3, 2, -3, -3, -4, 3, -4, -4, -5, 4, -5, -5, -6, 5, -6},
    /* blue   */ 
    {/*0, 1, */-1, 2, 1, 2, -2, 3, 2, 3, -3, 4, 3, 4, -4, 5, 4, 5, -5, 6, 5, 6},
    /* yellow */ 
    {/*-1, 0, */-2, -1, -2, 1, -3, -2, -3, 2, -4, -3, -4, 3, -5, -4, -5, 4, -6, -5, -6, 5},
    /* green  */
    {/*1, 0, */2, -1, 2, 1, 3, -2, 3, 2, 4, -3, 4, 3, 5, -4, 5, 4, 6, -5, 6, 5},
    /* red    */
    {0, -3, 0, -5, 0, -7, 0, -9, 0, -11, 0, -13, 0, -15, 0, -17, 0, -19, 0, -21/*, 0, -23*/},
    /* blue   */
    {0, 3, 0, 5, 0, 7, 0, 9, 0, 11, 0, 13, 0, 15, 0, 17, 0, 19, 0, 21/*, 0, 23*/},
    /* yellow */
    {-3, 0, -5, 0, -7, 0, -9, 0, -11, 0, -13, 0, -15, 0, -17, 0, -19, 0, -21, 0/*,-23, 0*/},
    /* green  */
    {3, 0, 5, 0, 7, 0, 9, 0, 11, 0, 13, 0, 15, 0, 17, 0, 19, 0, 21, 0/*, 23, 0*/}
};
// __constant__ int dirSamples[K_NUM_SAMPS][K_NUM_EACH_SAMP * 2] = {
//     {0, -3, 0, -5, 0, -7, 0, -9, 0, -11, 0, -13, 0, -15, 0, -17, 0, -19, 0, -21/*, 0, -23*/},
//     {0, 3, 0, 5, 0, 7, 0, 9, 0, 11, 0, 13, 0, 15, 0, 17, 0, 19, 0, 21/*, 0, 23*/},
//     {-3, 0, -5, 0, -7, 0, -9, 0, -11, 0, -13, 0, -15, 0, -17, 0, -19, 0, -21, 0/*,-23, 0*/},
//     {3, 0, 5, 0, 7, 0, 9, 0, 11, 0, 13, 0, 15, 0, 17, 0, 19, 0, 21, 0/*, 23, 0*/},
//     {-1, -2, -2, -1, -2, -3, -3, -2, -3, -4, -4, -3, -4, -5, -5, -4, -5, -6, -6, -5},
//     {-1, 2, -2, 1, -2, 3, -3, 2, -3, 4, -4, 3, -4, 5, -5, 4, -5, 6, -6, 5},
//     {1, 2, 2, 1, 2, 3, 3, 2, 3, 4, 4, 3, 4, 5, 5, 4, 5, 6, 6, 5},
//     {1, -2, 2, -1, 2, -3, 3, -2, 3, -4, 4, -3, 4, -5, 5, -4, 5, -6, 6, -5},
// };

__constant__ float filter_kernel_x[3][3] = {
  {-3, 0, 3}, {-10, 0, 10}, {-3, 0 ,3}
};
__constant__ float filter_kernel_y[3][3] = {
  {-3, -10, -3}, {0, 0, 0}, {3, 10 ,3}};

__forceinline__ __device__  void setBit(unsigned int &input, const unsigned int n) {
    input |= (unsigned int)(1 << n);
}

__forceinline__ __device__  int isSet(unsigned int input, const unsigned int n) {
    return (input >> n) & 1;
}

__forceinline__ __device__ float GenerateRandomNumber(const float mean,
                                             const float delta,
                                             curandState* rand_state) {
    return delta * (2 * curand_uniform(rand_state) - 1) + mean;
}

__forceinline__ __device__ float GenerateRandomDepth(const float depth_min,
                                            const float depth_max,
                                            curandState* rand_state) {
  return curand_uniform(rand_state) * (depth_max - depth_min) + depth_min;
}

__forceinline__ __device__ void GenerateRandomNormal(const int row, const int col,
                                            curandState* rand_state,
                                            float normal[3], const int thread_index) {
  // Unbiased sampling of normal, according to George Marsaglia, "Choosing a
  // Point from the Surface of a Sphere", 1972.
  float v1 = 0.0f;
  float v2 = 0.0f;
  float s = 2.0f;
  while (s >= 1.0f) {
    v1 = 2.0f * curand_uniform(rand_state) - 1.0f;
    v2 = 2.0f * curand_uniform(rand_state) - 1.0f;
    s = v1 * v1 + v2 * v2;
  }

  const float s_norm = 2.0f * sqrtf(1.0f - s);
  normal[0] = v1 * s_norm;
  normal[1] = v2 * s_norm;
  normal[2] = 1.0f - 2.0f * s;
  
  // Make sure normal is looking away from camera.
  const int k_idx = (thread_index << 2);
  const float view_ray[3] = {ref_inv_K[k_idx + 0] * col + ref_inv_K[k_idx + 1],
                             ref_inv_K[k_idx + 2] * row + ref_inv_K[k_idx + 3], 1.0f};
  if (DotProduct3(normal, view_ray) > 0) {
  // if (normal[2] > 0) {
    normal[0] = -normal[0];
    normal[1] = -normal[1];
    normal[2] = -normal[2];
  }
}

__forceinline__ __device__ void GenerateRandomNormal(const int row, const int col, 
                                                    curandState* rand_state, const float view_ray[3],
                                                    float normal[3]) {
  // Unbiased sampling of normal, according to George Marsaglia, "Choosing a
  // Point from the Surface of a Sphere", 1972.
  float v1 = 0.0f;
  float v2 = 0.0f;
  float s = 2.0f;
  while (s >= 1.0f) {
    v1 = 2.0f * curand_uniform(rand_state) - 1.0f;
    v2 = 2.0f * curand_uniform(rand_state) - 1.0f;
    s = v1 * v1 + v2 * v2;
  }

  const float s_norm = 2.0f * sqrtf(1.0f - s);
  normal[0] = v1 * s_norm;
  normal[1] = v2 * s_norm;
  normal[2] = 1.0f - 2.0f * s;
  
  // Make sure normal is looking away from camera.
  if (DotProduct3(normal, view_ray) > 0) {
  // if (normal[2] > 0) {
    normal[0] = -normal[0];
    normal[1] = -normal[1];
    normal[2] = -normal[2];
  }
}

__forceinline__ __device__ float PerturbDepth(const float perturbation,
                                     const float depth,
                                     curandState* rand_state) {
  const float depth_min = (1.0f - perturbation) * depth;
  const float depth_max = (1.0f + perturbation) * depth;
  return GenerateRandomDepth(depth_min, depth_max, rand_state);
}

__forceinline__ __device__ void PerturbNormal(const int row, const int col,
                                     const float perturbation,
                                     const float normal[3],
                                     curandState* rand_state,
                                     float perturbed_normal[3],
                                     const int thread_index,
                                     const int num_trials = 0) {
  // Perturbation rotation angles.
  const float a1 = (curand_uniform(rand_state) - 0.5f) * perturbation;
  const float a2 = (curand_uniform(rand_state) - 0.5f) * perturbation;
  const float a3 = (curand_uniform(rand_state) - 0.5f) * perturbation;

  const float sin_a1 = __sinf(a1);
  const float sin_a2 = __sinf(a2);
  const float sin_a3 = __sinf(a3);
  const float cos_a1 = __cosf(a1);
  const float cos_a2 = __cosf(a2);
  const float cos_a3 = __cosf(a3);

  const float sina2a3 = sin_a2 * sin_a3;
  const float cosa1a3 = cos_a1 * cos_a3;
  const float cosa3_sina1 = cos_a3 * sin_a1; 

  // R = Rx * Ry * Rz
  float R[9];
  R[0] = cos_a2 * cos_a3;
  R[1] = -cos_a2 * sin_a3;
  R[2] = sin_a2;
  R[3] = cos_a1 * sin_a3 + cosa3_sina1 * sin_a2;
  R[4] = cosa1a3 - sin_a1 * sina2a3;
  R[5] = -cos_a2 * sin_a1;
  R[6] = sin_a1 * sin_a3 - cosa1a3 * sin_a2;
  R[7] = cosa3_sina1 + cos_a1 * sina2a3;
  R[8] = cos_a1 * cos_a2;

  // Perturb the normal vector.
  Mat33DotVec3(R, normal, perturbed_normal);

  // Make sure the perturbed normal is still looking in the same direction as
  // the viewing direction, otherwise try again but with smaller perturbation.
  const int k_idx = (thread_index << 2);
  const float view_ray[3] = {ref_inv_K[k_idx + 0] * col + ref_inv_K[k_idx + 1],
                             ref_inv_K[k_idx + 2] * row + ref_inv_K[k_idx + 3], 1.0f};
  if (DotProduct3(perturbed_normal, view_ray) >= 0.0f) {
  // if (perturbed_normal[2] > 0) {
    const int kMaxNumTrials = 3;
    if (num_trials < kMaxNumTrials) {
      PerturbNormal(row, col, 0.5f * perturbation, normal, rand_state,
                    perturbed_normal, thread_index, num_trials + 1);
      return;
    } else {
      perturbed_normal[0] = normal[0];
      perturbed_normal[1] = normal[1];
      perturbed_normal[2] = normal[2];
      return;
    }
  }

  // Make sure normal has unit norm.
  const float inv_norm = rsqrtf(DotProduct3(perturbed_normal, perturbed_normal));
  perturbed_normal[0] *= inv_norm;
  perturbed_normal[1] *= inv_norm;
  perturbed_normal[2] *= inv_norm;
}

__forceinline__ __device__ void PerturbNormal(const float normal[3], float pert_normal[3],
                              const float angle1_ratio, 
                              const float angle2_ratio,
                              curandState* rand_state) {
    float nx, ny;
    Normal2Dir(normal, nx, ny);

    nx = GenerateRandomNumber(nx, angle1_ratio, rand_state);
    ny = GenerateRandomNumber(ny, angle2_ratio, rand_state);

    Dir2Normal(nx, ny, pert_normal);
}

__forceinline__ __device__ void CorrectNormal(float normal[3], const float view_dir[3], const float inv_view_len) {
    const float cos_angle_len = DotProduct3(normal, view_dir);
    if (cos_angle_len >= 0) {
        float n_crs[3];
        n_crs[0] = normal[1] * view_dir[2] - normal[2] * view_dir[1];
        n_crs[1] = normal[2] * view_dir[0] - normal[0] * view_dir[2];
        n_crs[2] = normal[0] * view_dir[1] - normal[1] * view_dir[0];
        
        float phi = acosf(cos_angle_len * inv_view_len) - DEG2RAD(90.f);
        phi = min(phi * 1.01f, -0.001f);
        if (phi == 0) {
            return;
        }

        const float inv_crs_len = rsqrtf(n_crs[0] * n_crs[0] +
            n_crs[1] * n_crs[1] + n_crs[2] * n_crs[2]);
        n_crs[0] *= inv_crs_len;
        n_crs[1] *= inv_crs_len;
        n_crs[2] *= inv_crs_len;

        float omega[9];
        omega[0] = 0.0f;
        omega[1] = -n_crs[2];
        omega[2] = n_crs[1];
        omega[3] = n_crs[2];
        omega[4] = 0.0f;
        omega[5] = -n_crs[0];
        omega[6] = -n_crs[1];
        omega[7] = n_crs[0];
        omega[8] = 0.0f;

        const float sin_phi = __sinf(phi);
        float sin_O[9];
        sin_O[0] = omega[0] * sin_phi;
        sin_O[1] = omega[1] * sin_phi;
        sin_O[2] = omega[2] * sin_phi;
        sin_O[3] = omega[3] * sin_phi;
        sin_O[4] = omega[4] * sin_phi;
        sin_O[5] = omega[5] * sin_phi;
        sin_O[6] = omega[6] * sin_phi;
        sin_O[7] = omega[7] * sin_phi;
        sin_O[8] = omega[8] * sin_phi;
        
        const float cos_phi = 1.0f - __cosf(phi);
        float cos_O_O[9];
        cos_O_O[0] = cos_phi * 
          (omega[0] * omega[0] + omega[1] * omega[3] + omega[2] * omega[6]);
        cos_O_O[1] = cos_phi * 
          (omega[0] * omega[1] + omega[1] * omega[4] + omega[2] * omega[7]);
        cos_O_O[2] = cos_phi * 
          (omega[0] * omega[2] + omega[1] * omega[5] + omega[2] * omega[8]);
        cos_O_O[3] = cos_phi * 
          (omega[3] * omega[0] + omega[4] * omega[3] + omega[5] * omega[6]);
        cos_O_O[4] = cos_O_O[3];
        cos_O_O[5] = cos_O_O[3];
        cos_O_O[6] = cos_phi * 
          (omega[6] * omega[0] + omega[7] * omega[3] + omega[8] * omega[6]);
        cos_O_O[7] = cos_O_O[6];
        cos_O_O[8] = cos_O_O[6];

        float *Rot = omega;
        Rot[0] = 1 + sin_O[0] + cos_O_O[0];
        Rot[1] = 0 + sin_O[1] + cos_O_O[1];
        Rot[2] = 0 + sin_O[2] + cos_O_O[2];
        Rot[3] = 0 + sin_O[3] + cos_O_O[3];
        Rot[4] = 1 + sin_O[4] + cos_O_O[4];
        Rot[5] = 0 + sin_O[5] + cos_O_O[5];
        Rot[6] = 0 + sin_O[6] + cos_O_O[6];
        Rot[7] = 0 + sin_O[7] + cos_O_O[7];
        Rot[8] = 1 + sin_O[8] + cos_O_O[8];
        
        float *rnormal = n_crs;
        rnormal[0] = Rot[0] * normal[0] + Rot[1] * normal[1] + Rot[2] * normal[2];
        rnormal[1] = Rot[3] * normal[0] + Rot[4] * normal[1] + Rot[5] * normal[2];
        rnormal[2] = Rot[6] * normal[0] + Rot[7] * normal[1] + Rot[8] * normal[2];
        normal[0] = rnormal[0];
        normal[1] = rnormal[1];
        normal[2] = rnormal[2];
    }
}

__forceinline__ __device__ void ComputePointAtDepth(const float row, const float col,
                                           const float depth, float point[3],
                                           const int thread_index) {
  const int k_idx = (thread_index << 2);
  point[0] = depth * (ref_inv_K[k_idx + 0] * col + ref_inv_K[k_idx + 1]);
  point[1] = depth * (ref_inv_K[k_idx + 2] * row + ref_inv_K[k_idx + 3]);
  point[2] = depth;
}

__forceinline__ __device__ float ComputeNCCWeightFromPrior(const float4 hypothesis, 
  const float4 prior_hypothesis, const float ref_color_var) {
    // return = 1.0 - exp(-(hypoth_depth - prior_depth) * (hypoth_depth - prior_depth) / 0.5) / (2.5 * 0.5);
    // const float x = hypothesis.w / prior_hypothesis.w - 1.0f;
    // return 1.0f - 0.8 * exp(-2 * x * x * hypothesis.w * prior_hypothesis.w);
    
    float regu = 1.0f;
    float sigma2 = 0.5 - 0.4 * exp(-10000.0 * fabs(ref_color_var));
    float delt_depth_2 = max((hypothesis.w - prior_hypothesis.w) * 
                      (hypothesis.w - prior_hypothesis.w), 0.05f * 0.05f);
    regu *= max(1.0 - 0.2 * exp(- delt_depth_2 / sigma2), 0.01);
    // regu *= max(1.0 - 0.2 * exp(- delt_depth_2 / sigma2), 0.01);

    float prior_norm = prior_hypothesis.x * prior_hypothesis.x + prior_hypothesis.y * prior_hypothesis.y + prior_hypothesis.z * prior_hypothesis.z;
    if (prior_norm > 0.8 && prior_norm < 1.25 ){
      float cost_theta = prior_hypothesis.x * hypothesis.x 
                      + prior_hypothesis.y * hypothesis.y
                      + prior_hypothesis.z * hypothesis.z;
      // float cost_theta = 0.1;
      float delt_theta_2 = (1.0f - cost_theta) * (1.0f - cost_theta);
      regu *= max(1.0 - 0.54 * exp(- delt_theta_2 / (sigma2)), 0.01);
    }
    return regu;
}

// Transfer depth on plane from viewing ray at row1 to row2. The returned
// depth is the intersection of the viewing ray through row2 with the plane
// at row1 defined by the given depth and normal.
__forceinline__ __device__ float PropagateDepth(const float depth1,
                                       const float normal1[3], const float row1,
                                       const float row2, const int thread_index) {
  // Extract 1/fx, -cx/fx, 1/fy, -cy/fy.
  const int k_idx = (thread_index << 2);
  const float iK2 = ref_inv_K[k_idx + 2];
  const float iK3 = ref_inv_K[k_idx + 3];
  
  // Point along first viewing ray.
  const float x1 = depth1 * (iK2 * row1 + iK3);
  const float y1 = depth1;
  // Point on plane defined by point along first viewing ray and plane normal1.
  const float x2 = x1 + normal1[2];
  const float y2 = y1 - normal1[1];

  // Origin of second viewing ray.
  // const float x3 = 0.0f;
  // const float y3 = 0.0f;
  // Point on second viewing ray.
  const float x4 = iK2 * row2 + iK3;
  // const float y4 = 1.0f;

  // Intersection of the lines ((x1, y1), (x2, y2)) and ((x3, y3), (x4, y4)).
  const float denom = x2 - x1 + x4 * (y1 - y2);
  const float kEps = 1e-5f;
  if (fabs(denom) < kEps) {
    return depth1;
  }
  const float nom = y1 * x2 - x1 * y2;
  return nom / denom;
}

__forceinline__ __device__ float PropagateDepth(const float depth1, 
                                       const float normal1[3],
                                       const int row1, const int col1, 
                                       const int row2, const int col2,
                                       const int thread_index) {
  // Extract 1/fx, -cx/fx, 1/fy, -cy/fy.
  const int k_idx = (thread_index << 2);
  const float iK0 = ref_inv_K[k_idx + 0];
  const float iK1 = ref_inv_K[k_idx + 1];
  const float iK2 = ref_inv_K[k_idx + 2];
  const float iK3 = ref_inv_K[k_idx + 3];

  const float x1 = depth1 * (iK0 * col1 + iK1);
  const float y1 = depth1 * (iK2 * row1 + iK3);
  const float z1 = depth1;

  // plane parameter.
  const float d = normal1[0] * x1 + normal1[1] * y1 + normal1[2] * z1;

  const float la = iK0 * col2 + iK1;
  const float lb = iK2 * row2 + iK3;
  const float lc = 1.0f;
  const float t = d / (normal1[0] * la + normal1[1] * lb + normal1[2] * lc);

  const float kEps = 1e-5f;
  if (fabs(t) < kEps) {
      return depth1;
  }

  return t;
}

__forceinline__ __device__ float PropagateDepth(const float depth1, 
                                                const float normal1[3],
                                                const float ref_invK[4],
                                                const int row1, const int col1,
                                                const float normalized_point2[3]) {
  const float x1 = depth1 * (ref_invK[0] * col1 + ref_invK[1]);
  const float y1 = depth1 * (ref_invK[2] * row1 + ref_invK[3]);
  const float z1 = depth1;

  // plane parameter.
  const float d = normal1[0] * x1 + normal1[1] * y1 + normal1[2] * z1;
  const float t = d / (normal1[0] * normalized_point2[0] + normal1[1] * normalized_point2[1] 
                      + normal1[2] * normalized_point2[2]);

  const float kEps = 1e-5f;
  if (fabs(t) < kEps) {
      return depth1;
  }

  return t;
}

__forceinline__ __device__ void ComposeHomography(const int image_idx, const int row,
                                                  const int col, const float depth,
                                                  const float normal[3], float H[9],
                                                  //  cudaTextureObject_t poses_texture,
                                                  const float* poses_texture,
                                                  const float* ref_invK) {
  // Calibration of source image.
  const int pose_idx = 43 * image_idx;
  float K[4];
  for (int i = 0; i < 4; ++i) {
    // K[i] = tex2D<float>(poses_texture, i, image_idx);
    K[i] = poses_texture[pose_idx + i];
  }

  // Relative rotation between reference and source image.
  float R[9];
  for (int i = 0; i < 9; ++i) {
    // R[i] = tex2D<float>(poses_texture, i + 4, image_idx);
    R[i] = poses_texture[pose_idx + i + 4];
  }

  // Relative translation between reference and source image.
  float T[3];
  for (int i = 0; i < 3; ++i) {
    // T[i] = tex2D<float>(poses_texture, i + 13, image_idx);
    T[i] = poses_texture[pose_idx + i + 13];
  }

  // Distance to the plane.
  const float dist = depth * (normal[0] * (ref_invK[0] * col + ref_invK[1]) +
                     normal[1] * (ref_invK[2] * row + ref_invK[3]) + normal[2]);
  const float inv_dist = 1.0f / dist;

  const float inv_dist_N0 = inv_dist * normal[0];
  const float inv_dist_N1 = inv_dist * normal[1];
  const float inv_dist_N2 = inv_dist * normal[2];
  const float inv_dist_N0T0 = inv_dist_N0 * T[0];
  const float inv_dist_N0T1 = inv_dist_N0 * T[1];
  const float inv_dist_N0T2 = inv_dist_N0 * T[2];
  const float inv_dist_N1T0 = inv_dist_N1 * T[0];
  const float inv_dist_N1T1 = inv_dist_N1 * T[1];
  const float inv_dist_N1T2 = inv_dist_N1 * T[2];
  const float inv_dist_N2T0 = inv_dist_N2 * T[0];
  const float inv_dist_N2T1 = inv_dist_N2 * T[1];
  const float inv_dist_N2T2 = inv_dist_N2 * T[2];

  const float K0R0N0T0 = K[0] * (R[0] + inv_dist_N0T0);
  const float K0R1N1T0 = K[0] * (R[1] + inv_dist_N1T0);
  const float K2R3N0T1 = K[2] * (R[3] + inv_dist_N0T1);
  const float K2R4N1T1 = K[2] * (R[4] + inv_dist_N1T1);
  const float K1R6N0T2 = K[1] * (R[6] + inv_dist_N0T2);
  const float K1R7N1T2 = K[1] * (R[7] + inv_dist_N1T2);
  const float K3R6N0T2 = K[3] * (R[6] + inv_dist_N0T2);
  const float K3R7N1T2 = K[3] * (R[7] + inv_dist_N1T2);
  const float K0R0N0T0_K1R6N0T2 = K0R0N0T0 + K1R6N0T2;
  const float K0R1N1T0_K1R7N1T2 = K0R1N1T0 + K1R7N1T2;
  const float K2R3N0T1_K3R6N0T2 = K2R3N0T1 + K3R6N0T2;
  const float K2R4N1T1_K3R7N1T2 = K2R4N1T1 + K3R7N1T2;
  const float R8_N2T2 = R[8] + inv_dist_N2T2;

  // Homography as H = K * (R - T * n' / d) * Kref^-1.
  H[0] = ref_invK[0] * K0R0N0T0_K1R6N0T2;
  H[1] = ref_invK[2] * K0R1N1T0_K1R7N1T2;
  H[2] = K[0] * (R[2] + inv_dist_N2T0) + K[1] * R8_N2T2 +
         ref_invK[1] * K0R0N0T0_K1R6N0T2 + ref_invK[3] * K0R1N1T0_K1R7N1T2;
  H[3] = ref_invK[0] * K2R3N0T1_K3R6N0T2;
  H[4] = ref_invK[2] * K2R4N1T1_K3R7N1T2;
  H[5] = K[2] * (R[5] + inv_dist_N2T1) + K[3] * R8_N2T2 +
         ref_invK[1] * K2R3N0T1_K3R6N0T2 + ref_invK[3] * K2R4N1T1_K3R7N1T2;
  H[6] = ref_invK[0] * (R[6] + inv_dist_N0T2);
  H[7] = ref_invK[2] * (R[7] + inv_dist_N1T2);
  H[8] = ref_invK[1] * (R[6] + inv_dist_N0T2) + ref_invK[3] * (R[7] + inv_dist_N1T2) + R8_N2T2;
}

__forceinline__ __device__ float ComputeGeomConsistencyCostShared(const float row,
                                                                  const float col,
                                                                  const float depth,
                                                                  const float normalized_point[3],
                                                                  const int image_idx,
                                                                  const float max_cost,
                                                                  cudaTextureObject_t src_depth_maps_texture,
                                                                  const float* poses_texture,
                                                                  const float K[4]) {
  // Extract projection matrices for source image.
  const int pose_idx = 43 * image_idx;
  float P[12], inv_P[12];
  for (int i = 0; i < 12; ++i) {
    P[i] = poses_texture[pose_idx + i + 19];
    inv_P[i] = poses_texture[pose_idx + i + 31];
  }

  // Project point in reference image to world.
  float forward_point[3];
  forward_point[0] = depth * normalized_point[0];
  forward_point[1] = depth * normalized_point[1];
  forward_point[2] = depth * normalized_point[2];

  // Project world point to source image.
  const float inv_forward_z =
      1.0f / (P[8] * forward_point[0] + P[9] * forward_point[1] +
              P[10] * forward_point[2] + P[11]);
  float src_col =
      inv_forward_z * (P[0] * forward_point[0] + P[1] * forward_point[1] +
                       P[2] * forward_point[2] + P[3]);
  float src_row =
      inv_forward_z * (P[4] * forward_point[0] + P[5] * forward_point[1] +
                       P[6] * forward_point[2] + P[7]);

  // Extract depth in source image.
  const float src_depth = tex2DLayered<float>(src_depth_maps_texture, src_col + 0.5f,
                                       src_row + 0.5f, image_idx);

  // Projection outside of source image.
  if (src_depth == 0.0f) {
    return max_cost;
  }

  // Project point in source image to world.
  src_col *= src_depth;
  src_row *= src_depth;
  const float backward_point_x =
      inv_P[0] * src_col + inv_P[1] * src_row + inv_P[2] * src_depth + inv_P[3];
  const float backward_point_y =
      inv_P[4] * src_col + inv_P[5] * src_row + inv_P[6] * src_depth + inv_P[7];
  const float backward_point_z = inv_P[8] * src_col + inv_P[9] * src_row +
                                 inv_P[10] * src_depth + inv_P[11];
  const float inv_backward_point_z = 1.0f / backward_point_z;

  // Project world point back to reference image.
  const float backward_col = inv_backward_point_z * K[0] * backward_point_x + K[1];
  const float backward_row = inv_backward_point_z * K[2] * backward_point_y + K[3];

  // Return truncated reprojection error between original observation and
  // the forward-backward projected observation.
  const float diff_col = col - backward_col;
  const float diff_row = row - backward_row;
  return min(max_cost, sqrtf(diff_col * diff_col + diff_row * diff_row));
}

__forceinline__ __device__ float ComputeGeomConsistencyCost(const float row,
                                                            const float col,
                                                            const float depth,
                                                            const int image_idx,
                                                            const float max_cost,
                                                            cudaTextureObject_t src_depth_maps_texture,
                                                             cudaTextureObject_t poses_texture,
                                                            const int thread_index) {
  // Extract projection matrices for source image.
  float P[12];
  for (int i = 0; i < 12; ++i) {
    P[i] = tex2D<float>(poses_texture, i + 19, image_idx);
  }
  float inv_P[12];
  for (int i = 0; i < 12; ++i) {
    inv_P[i] = tex2D<float>(poses_texture, i + 31, image_idx);
  }

  // Project point in reference image to world.
  float forward_point[3];
  ComputePointAtDepth(row, col, depth, forward_point, thread_index);

  // Project world point to source image.
  const float inv_forward_z =
      1.0f / (P[8] * forward_point[0] + P[9] * forward_point[1] +
              P[10] * forward_point[2] + P[11]);
  float src_col =
      inv_forward_z * (P[0] * forward_point[0] + P[1] * forward_point[1] +
                       P[2] * forward_point[2] + P[3]);
  float src_row =
      inv_forward_z * (P[4] * forward_point[0] + P[5] * forward_point[1] +
                       P[6] * forward_point[2] + P[7]);

  // Extract depth in source image.
  const float src_depth = tex2DLayered<float>(src_depth_maps_texture, src_col + 0.5f,
                                       src_row + 0.5f, image_idx);

  // Projection outside of source image.
  if (src_depth == 0.0f) {
    return max_cost;
  }

  // Project point in source image to world.
  src_col *= src_depth;
  src_row *= src_depth;
  const float backward_point_x =
      inv_P[0] * src_col + inv_P[1] * src_row + inv_P[2] * src_depth + inv_P[3];
  const float backward_point_y =
      inv_P[4] * src_col + inv_P[5] * src_row + inv_P[6] * src_depth + inv_P[7];
  const float backward_point_z = inv_P[8] * src_col + inv_P[9] * src_row +
                                 inv_P[10] * src_depth + inv_P[11];
  const float inv_backward_point_z = 1.0f / backward_point_z;

  // Project world point back to reference image.
  const int k_idx = (thread_index << 2);
  const float backward_col = inv_backward_point_z * ref_K[k_idx + 0] * backward_point_x + ref_K[k_idx + 1];
  const float backward_row = inv_backward_point_z * ref_K[k_idx + 2] * backward_point_y + ref_K[k_idx + 3];

  // Return truncated reprojection error between original observation and
  // the forward-backward projected observation.
  const float diff_col = col - backward_col;
  const float diff_row = row - backward_row;
  return min(max_cost, sqrtf(diff_col * diff_col + diff_row * diff_row));
}

// Find index of minimum in given values.
template <int kNumCosts>
__forceinline__ __device__ int FindMinCost(const float costs[kNumCosts]) {
  float min_cost = FLT_MAX;
  int min_cost_idx = -1;
  for (int idx = 0; idx < kNumCosts; ++idx) {
    if (costs[idx] <= min_cost) {
      min_cost = costs[idx];
      min_cost_idx = idx;
    }
  }
  return min_cost_idx;
}

__forceinline__ __device__ int FindMinCost(const float *costs, const int length) {
  float min_cost = FLT_MAX;
  int min_cost_idx = -1;
  for (int idx = 0; idx < length; ++idx) {
    if (costs[idx] <= min_cost) {
      min_cost = costs[idx];
      min_cost_idx = idx;
    }
  }
  return min_cost_idx;
}

__forceinline__ __device__ int FindMinCostWithin(const float *costs, const int length, const float4 *hypotheses, float depth_min, float depth_max) {
  float min_cost = FLT_MAX;
  int min_cost_idx = -1;
  for (int idx = 0; idx < length; ++idx) {
    if (costs[idx] <= min_cost && hypotheses[idx].w >= depth_min && hypotheses[idx].w <= depth_max) {
      min_cost = costs[idx];
      min_cost_idx = idx;
    }
  }
  return min_cost_idx;
}

__forceinline__ __device__ void TransformPDFToCDF(float* probs, const int num_probs) {
  float prob_sum = 0.0f;
  for (int i = 0; i < num_probs; ++i) {
    prob_sum += probs[i];
  }
  const float inv_prob_sum = 1.0f / prob_sum;

  float cum_prob = 0.0f;
  for (int i = 0; i < num_probs; ++i) {
    const float prob = probs[i] * inv_prob_sum;
    cum_prob += prob;
    probs[i] = cum_prob;
  }
}

// Rotate normals by 90deg around z-axis in counter-clockwise direction.
__global__ void InitNormalMap(GpuMat<float> normal_map,
                              GpuMat<curandState> rand_state_map,
                              const int thread_index) {
  const int row = blockDim.y * blockIdx.y + threadIdx.y;
  const int col = blockDim.x * blockIdx.x + threadIdx.x;
  if (col < normal_map.GetWidth() && row < normal_map.GetHeight()) {
    curandState rand_state = rand_state_map.Get(row, col);
    float normal[3];
    GenerateRandomNormal(row, col, &rand_state, normal, thread_index);
    normal_map.SetSlice(row, col, normal);
    // rand_state_map.Set(row, col, rand_state);
  }
}

__global__ void InitDepthMap(GpuMat<float> depth_map,
                             GpuMat<float> normal_map,
                             GpuMat<curandState> rand_state_map,
                             const float depth_min,
                             const float depth_max,
                             const int thread_index) {
  const int row = blockDim.y * blockIdx.y + threadIdx.y;
  const int col = blockDim.x * blockIdx.x + threadIdx.x;
  if (col < depth_map.GetWidth() && row < depth_map.GetHeight()) {
    curandState rand_state = rand_state_map.Get(row, col);

    const int k_idx = (thread_index << 2);
    float view_dir[3];
    view_dir[0] = 1.0f * (ref_inv_K[k_idx + 0] * col + ref_inv_K[k_idx + 1]);
    view_dir[1] = 1.0f * (ref_inv_K[k_idx + 2] * row + ref_inv_K[k_idx + 3]);
    view_dir[2] = 1.0f;

    float depth = depth_map.Get(row, col);
    float normal[3];
    normal_map.GetSlice(row, col, normal);

    if (depth <= 0) {
        depth = GenerateRandomDepth(depth_min, depth_max, &rand_state);
        GenerateRandomNormal(row, col, &rand_state, normal, thread_index);
        depth_map.Set(row, col, depth);
        normal_map.SetSlice(row, col, normal);
    }
    // if (normal[2] >= 0) {
    if (DotProduct3(normal, view_dir) >= 0) {
        GenerateRandomNormal(row, col, &rand_state, normal, thread_index);
        normal_map.SetSlice(row, col, normal);
    }
    // rand_state_map.Set(row, col, rand_state);
  }
}

__global__ void InitSelectedImageList(GpuMat<curandState> rand_state_map,
                                      GpuMat<uint32_t> selected_images_list,
                                      const int num_src_image,
                                      const int max_num_src_images) {
  int row = blockDim.y * blockIdx.y + threadIdx.y;
  int col = blockDim.x * blockIdx.x + threadIdx.x;
  const int width = selected_images_list.GetWidth();
  const int height = selected_images_list.GetHeight();
  if (row >= height || col >= width) {
    return;
  }
  // if (row % 200 == 0 && col % 200 == 0) {
    curandState rand_state = rand_state_map.Get(row, col);
    uint32_t ref_view_sel = 0;
    int idxs[MAX_NUM_SRC_IMAGE];
    for (int i = 0; i < max_num_src_images; ++i) {
      idxs[i] = i;
    }
    for (int i = 0; i < num_src_image; ++i) {
      int j = i + curand(&rand_state) % (max_num_src_images - i);
      uint32_t val = idxs[i];
      idxs[i] = idxs[j];
      idxs[j] = val;
    }
    for (int i = 0; i < num_src_image; ++i) {
      ref_view_sel |= (unsigned int)(1 << idxs[i]);
    }
    // for (int y = 0; y < 200; ++y) {
    //   if (row + y >= height) continue;
    //   for (int x = 0; x < 200; ++x) {
    //     if (col + x >= width) continue;
    //     selected_images_list.Set(row + y, col + x, ref_view_sel);
    //   }
    // }
    selected_images_list.Set(row, col, ref_view_sel);
    rand_state_map.Set(row, col, rand_state);
  // }
}

__global__ void GenerateProjectDepthMaps(const int max_width,
                                        const int max_height,
                                        const int src_image_idx,
                                        const int num_src_image,
                                        GpuMat<float> composite_depth_map,
                                        // GpuMat<float> composite_normal_map,
                                        cudaTextureObject_t depth_maps_texture,
                                        cudaTextureObject_t poses_texture,
                                        const int thread_index) {
  int row = blockDim.y * blockIdx.y + threadIdx.y;
  int col = blockDim.x * blockIdx.x + threadIdx.x;
  if (row >= max_height || col >= max_width) {
    return;
  }
  const float depth = tex2DLayered<float>(depth_maps_texture, col, row, src_image_idx);
  if (depth < 1e-6) {
    return;
  }

  float inv_P[12];
  for (int j = 0; j < 12; ++j) {
    inv_P[j] = tex2D<float>(poses_texture, j + 31, src_image_idx);
  }

  const int k_idx = (thread_index << 2);

  // Project point in source image to world.
  col *= depth;
  row *= depth;
  const float backward_point_x = inv_P[0] * col + inv_P[1] * row + inv_P[2] * depth + inv_P[3];
  const float backward_point_y = inv_P[4] * col + inv_P[5] * row + inv_P[6] * depth + inv_P[7];
  float backward_point_z = inv_P[8] * col + inv_P[9] * row + inv_P[10] * depth + inv_P[11];
  float inv_backward_point_z = 1.0f / backward_point_z;

  // Project world point back to reference image.
  const float backward_col =
      inv_backward_point_z * ref_K[k_idx + 0] * backward_point_x + ref_K[k_idx + 1];
  const float backward_row =
      inv_backward_point_z * ref_K[k_idx + 2] * backward_point_y + ref_K[k_idx + 3];

  const int ref_width = composite_depth_map.GetWidth();
  const int ref_height = composite_depth_map.GetHeight();
  if (backward_col < 0 || backward_col >= ref_width || backward_row < 0 || backward_row >= ref_height) {
    return;
  }

  unsigned int* address_as_ull = (unsigned int*)&composite_depth_map.GetRef(backward_row, backward_col, src_image_idx);
  atomicMin(address_as_ull, __float_as_uint(backward_point_z));
}

__global__ void CompositeInitDepthMap(GpuMat<curandState> rand_state_map,
  GpuMat<float> depth_map, GpuMat<float> prior_depth_map, GpuMat<float> composite_depth_map,
  const float depth_min, const float depth_max) {
  const int row = blockDim.y * blockIdx.y + threadIdx.y;
  const int col = blockDim.x * blockIdx.x + threadIdx.x;
  if (col < depth_map.GetWidth() && row < depth_map.GetHeight()) {
    if (depth_map.Get(row, col) > 0) {
      return;
    }
    int num_depth = 0;
    float depths[MAX_NUM_SRC_IMAGE] = {0};
    for (int i = 0; i < composite_depth_map.GetDepth(); ++i) {
      float depth = composite_depth_map.Get(row, col, i);
      if (depth > 0 && depth < FLT_MAX) {
        depths[num_depth] = depth;
        num_depth++;
      }
    }
    if (num_depth > 2) {
      BubbleSort(depths, num_depth);
      float mdepth = depths[num_depth / 2];
      depth_map.Set(row, col, mdepth);

      int num_consistent = 0;
      for (int i = 0; i < num_depth; ++i) {
        float diff_depth = fabs(depths[i] - mdepth) / mdepth;
        num_consistent += (diff_depth <= 0.01) ? 1 : 0;
      }
      if (num_consistent > 2) {
        prior_depth_map.Set(row, col, mdepth);
      }
    } else if (num_depth > 0) {
      float depth = (num_depth > 1) ? min(depths[0], depths[1]) : depths[0];
      depth_map.Set(row, col, depth);
    }
  }
}

__forceinline__ __device__ void EstimatePatchNormal(float *depth_map, const int width, const int height, const int pitch, 
                                    const int row, const int col, float* normal, char* success, const int tid) {
    if (col >= width - 1 || row >= height - 1 || col <= 0 || row <= 0) {
      *success = 0;
      return;
    }

    int row_index = pitch * row;
    int col_index = col;
    float depth, ndepth, point0[3], point1[3], point2[3], point3[3];
    depth = *((float*)((char*)depth_map + row_index) + col_index);
    if (depth <= 0) {
      *success = 0;
      return;
    }

    ndepth = *((float*)((char*)depth_map + row_index - pitch) + col_index);
    if (fabs(ndepth - depth) / depth > 0.02) { *success = 0; return; }
    ComputePointAtDepth(row - 1, col, ndepth, point0, tid);

    ndepth = *((float*)((char*)depth_map + row_index) + col_index + 1);
    if (fabs(ndepth - depth) / depth > 0.02) { *success = 0; return; }
    ComputePointAtDepth(row, col + 1, ndepth, point1, tid);

    ndepth = *((float*)((char*)depth_map + row_index + pitch) + col_index);
    if (fabs(ndepth - depth) / depth > 0.02) { *success = 0; return; }
    ComputePointAtDepth(row + 1, col, ndepth, point2, tid);

    ndepth = *((float*)((char*)depth_map + row_index) + col_index - 1);
    if (fabs(ndepth - depth) / depth > 0.02) { *success = 0; return; }
    ComputePointAtDepth(row, col - 1, ndepth, point3, tid);

    float u[3] = {point2[0] - point0[0], point2[1] - point0[1], point2[2] - point0[2]};
    float v[3] = {point1[0] - point3[0], point1[1] - point3[1], point1[2] - point3[2]};
    CrossProduct3(u, v, normal);

    float inv_norm = rsqrtf(normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]);
    normal[0] *= inv_norm;
    normal[1] *= inv_norm;
    normal[2] *= inv_norm;
    *success = 1;
}

__global__ void CompositeInitNormalMap(GpuMat<float> depth_map, GpuMat<float> normal_map, const int tid) {
  const int row = blockDim.y * blockIdx.y + threadIdx.y;
  const int col = blockDim.x * blockIdx.x + threadIdx.x;
  const int width = depth_map.GetWidth();
  const int height = depth_map.GetHeight();
  const int pitch = depth_map.GetPitch();
  
  int n = 0;
  char success = 0;
  float normals[3][5] = {0.f}, normal[3] = {0.f};

  EstimatePatchNormal(depth_map.GetPtr(), width, height, pitch, row, col, normal, &success, tid);
  if (success) {
    normals[0][n] = normal[0]; normals[1][n] = normal[1], normals[2][n] = normal[2];
    n++;
  }

  EstimatePatchNormal(depth_map.GetPtr(), width, height, pitch, row - 1, col - 1, normal, &success, tid);
  if (success) {
    normals[0][n] = normal[0]; normals[1][n] = normal[1], normals[2][n] = normal[2];
    n++;
  }

  EstimatePatchNormal(depth_map.GetPtr(), width, height, pitch, row - 1, col + 1, normal, &success, tid);
  if (success) {
    normals[0][n] = normal[0]; normals[1][n] = normal[1], normals[2][n] = normal[2];
    n++;
  }

  EstimatePatchNormal(depth_map.GetPtr(), width, height, pitch, row + 1, col + 1, normal, &success, tid);
  if (success) {
    normals[0][n] = normal[0]; normals[1][n] = normal[1], normals[2][n] = normal[2];
    n++;
  }

  EstimatePatchNormal(depth_map.GetPtr(), width, height, pitch, row + 1, col - 1, normal, &success, tid);
  if (success) {
    normals[0][n] = normal[0]; normals[1][n] = normal[1], normals[2][n] = normal[2];
    n++;
  }
  if (n > 0) {
    BubbleSort(normals[0], n);
    BubbleSort(normals[1], n);
    BubbleSort(normals[2], n);

    int nth = (n >> 1);
    normal[0] = normals[0][nth];
    normal[1] = normals[1][nth];
    normal[2] = normals[2][nth];

    float inv_norm = rsqrtf(normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]);
    normal[0] *= inv_norm;
    normal[1] *= inv_norm;
    normal[2] *= inv_norm;

    normal_map.SetSlice(row, col, normal);
  }
}

__global__ void PreprocessImageKernel(GpuMat<float> ref_sum_image,
                                      GpuMat<float> ref_squared_sum_image,
                                      const int window_radius,
                                      const int window_step,
                                      BilateralWeightComputer weight_computer,
                                      cudaTextureObject_t ref_image_texture) {
  const int row = blockDim.y * blockIdx.y + threadIdx.y;
  const int col = blockDim.x * blockIdx.x + threadIdx.x;
  if (col < ref_sum_image.GetWidth() && row < ref_sum_image.GetHeight()) {
    const float center_color = tex2D<float>(ref_image_texture, col, row);
    float color_sum = 0.0f;
    float color_squared_sum = 0.0f;
    float sum_w = 0.0f;
    for (int dr = -window_radius; dr <= window_radius; dr += window_step) {
      for (int dc = -window_radius; dc <= window_radius; dc += window_step) {
        const float color = tex2D<float>(ref_image_texture, col + dc, row + dr);
        const float bilateral_weight = 
          weight_computer.Compute(dr, dc, center_color, color);
        
        color_sum += bilateral_weight * color;
        color_squared_sum += bilateral_weight * color * color;
        sum_w += bilateral_weight;
      }
    }
    color_sum /= sum_w;
    color_squared_sum /= sum_w;

    ref_sum_image.Set(row, col, color_sum);
    ref_squared_sum_image.Set(row, col, color_squared_sum);
  }
}


__device__ float ComputeNCCCostFronto(const int image_idx,
                                      const int row, const int col,
                                      const float4 hypothesis,
                                      const int width, const int height,
                                      const int window_radius, 
                                      const int window_step,
                                      const bool plane_regularizer,
                                      BilateralWeightComputer weight_computer,
                                      cudaTextureObject_t ref_image_texture,
                                      cudaTextureObject_t ref_sum_image_texture,
                                      cudaTextureObject_t ref_squared_sum_image_texture,
                                      cudaTextureObject_t src_images_texture,
                                      cudaTextureObject_t ref_semantic_texture,
                                      cudaTextureObject_t src_semantics_texture,
                                      cudaTextureObject_t poses_texture,
                                      const int thread_index) {

  const float kMaxCost = 2.0f;

  // Extract projection matrices for source image.
  float P[12];
  for (int i = 0; i < 12; ++i) {
    P[i] = tex2D<float>(poses_texture, i + 19, image_idx);
  }

  float point[3];
  ComputePointAtDepth(row, col, hypothesis.w, point, thread_index);

  // Project world point to source image.
  const float inv_z = 1.0f / (P[8] * point[0] + P[9] * point[1] + P[10] * point[2] + P[11]);
  float src_col = inv_z * (P[0] * point[0] + P[1] * point[1] + P[2] * point[2] + P[3]);
  float src_row = inv_z * (P[4] * point[0] + P[5] * point[1] + P[6] * point[2] + P[7]);
  if (src_col < 0 || src_col >= width || src_row < 0 || src_row >= height) {
    return kMaxCost;
  }

  const float center_color = tex2D<float>(ref_image_texture, col, row);
  const float ref_color_sum = tex2D<float>(ref_sum_image_texture, col, row);
  const float ref_color_squared_sum = tex2D<float>(ref_squared_sum_image_texture, col, row);

  float src_color_sum = 0.0f;
  float src_color_squared_sum = 0.0f;
  float ref_src_color_sum = 0.0f;
  float sum_w = 0.0f, inv_sum_w;

  for (int dr = -window_radius; dr <= window_radius; dr += window_step) {
    for (int dc = -window_radius; dc <= window_radius; dc += window_step) {
      const int r = row + dr;
      const int c = col + dc;
      const float fu = src_row + dc + 0.5f;
      const float fv = src_col + dr + 0.5f;

      const float ref_color = tex2D<float>(ref_image_texture, c, r);
      const float src_color = tex2DLayered<float>(src_images_texture, fu, fv, image_idx);

      const float bilateral_weight = weight_computer.Compute(dr, dc, center_color, ref_color);
      const float bilateral_weight_color = bilateral_weight * src_color;

      src_color_sum += bilateral_weight_color;
      src_color_squared_sum += bilateral_weight_color * src_color;
      ref_src_color_sum += bilateral_weight_color * ref_color;
      sum_w += bilateral_weight;
    }
  }

  inv_sum_w = 1.0f / sum_w;
  src_color_sum *= inv_sum_w;
  src_color_squared_sum *= inv_sum_w;
  ref_src_color_sum *= inv_sum_w;

  const float ref_color_var = 
      ref_color_squared_sum - ref_color_sum * ref_color_sum;
  const float src_color_var = 
      src_color_squared_sum - src_color_sum * src_color_sum;

  if (ref_color_var < 1e-5 || src_color_var < 1e-5) {
      return kMaxCost;
  }

  const float ref_src_color_cover = ref_src_color_sum - ref_color_sum * src_color_sum;
  const float ref_src_color_var_inv = rsqrtf((ref_color_var * src_color_var));
  const float val = 1.0f - ref_src_color_cover * ref_src_color_var_inv;
  float score = max(0.0f, min(kMaxCost, val));
  return score;
}

__device__ float ComputeNCCCostWithPrior(const int image_idx,
                                const int row, const int col,
                                const float4 hypothesis,
                                const int width, const int height,
                                const int window_radius, 
                                const int window_step,
                                BilateralWeightComputer weight_computer,
                                cudaTextureObject_t ref_image_texture,
                                cudaTextureObject_t ref_sum_image_texture,
                                cudaTextureObject_t ref_squared_sum_image_texture,
                                cudaTextureObject_t src_images_texture,
                                cudaTextureObject_t ref_semantic_texture,
                                cudaTextureObject_t src_semantics_texture,
                                // cudaTextureObject_t poses_texture,
                                const float* poses_texture,
                                const float* ref_invK) {
  const float kMaxCost = 2.0f;
  const float color_norm = 1.0f / 255.0f;

  float H[9];
  ComposeHomography(image_idx, row, col, hypothesis.w, &hypothesis.x, H, poses_texture, ref_invK);

  float inv_z = 1.0f / (H[6] * col + H[7] * row + H[8]);
  int u = (H[0] * col + H[1] * row + H[2]) * inv_z;
  int v = (H[3] * col + H[4] * row + H[5]) * inv_z;
  if (u < 0 || u >= width || v < 0 || v >= height) {
    return kMaxCost;
  }

  const int row_start = row - window_radius;
  const int col_start = col - window_radius;

  float col_src = H[0] * col_start + H[1] * row_start + H[2];
  float row_src = H[3] * col_start + H[4] * row_start + H[5];
  float z = H[6] * col_start + H[7] * row_start + H[8];
  float base_col_src = col_src;
  float base_row_src = row_src;
  float base_z = z;

  for (int i = 0; i < 9; ++i) {
    H[i] = window_step * H[i];
  }

  const float center_color = tex2D<float>(ref_image_texture, col, row);
  const float ref_color_sum = tex2D<float>(ref_sum_image_texture, col, row);
  const float ref_color_squared_sum = 
    tex2D<float>(ref_squared_sum_image_texture, col, row);

  float src_color_sum = 0.0f;
  float src_color_squared_sum = 0.0f;
  float ref_src_color_sum = 0.0f;

  const uint8_t center_mask_color = tex2D<uint8_t>(ref_semantic_texture, col + 0.5, row + 0.5);
  const bool in_mask = (center_mask_color == LABEL_POWERLINE);

  float m_ref_color_sum = 0;
  float m_ref_color_squared_sum = 0;
  float m_src_color_sum = 0.0f;
  float m_src_color_squared_sum = 0.0f;
  float m_ref_src_color_sum = 0.0f;
  float sum_w = 0.0f, inv_sum_w;

  for (int dr = -window_radius; dr <= window_radius; dr += window_step) {
    for (int dc = -window_radius; dc <= window_radius; dc += window_step) {
      const int r = row + dr;
      const int c = col + dc;

      // TODO: eliminate the offset 0.5
      const float inv_z = 1.0f / z;
      const float fu = inv_z * col_src + 0.5f;
      const float fv = inv_z * row_src + 0.5f;

      const float ref_color = tex2D<float>(ref_image_texture, c, r);
      const float src_color = 
        tex2DLayered<float>(src_images_texture, fu, fv, image_idx);

      const float bilateral_weight = 
        weight_computer.Compute(dr, dc, center_color, ref_color);
      float bilateral_weight_color = bilateral_weight * src_color;

      src_color_sum += bilateral_weight_color;
      src_color_squared_sum += bilateral_weight_color * src_color;
      ref_src_color_sum += bilateral_weight_color * ref_color;
      sum_w += bilateral_weight;

      if (in_mask) {
        const uint8_t i_ref_color = tex2D<uint8_t>(ref_semantic_texture, c + 0.5, r + 0.5);
        const uint8_t i_src_color = tex2DLayered<uint8_t>(src_semantics_texture, fu + 0.5, fv + 0.5, image_idx);
        const float m_ref_color = (i_ref_color == LABEL_POWERLINE) ? i_ref_color * color_norm : 0.0f;
        const float m_src_color = (i_src_color == LABEL_POWERLINE) ? i_src_color * color_norm : 0.0f;

        m_ref_color_sum += bilateral_weight * m_ref_color;
        m_ref_color_squared_sum += bilateral_weight * m_ref_color * m_ref_color;

        bilateral_weight_color = bilateral_weight * m_src_color;
        m_src_color_sum += bilateral_weight_color;
        m_src_color_squared_sum += bilateral_weight_color * m_src_color;
        m_ref_src_color_sum += bilateral_weight_color * m_ref_color;
      }

      col_src += H[0];
      row_src += H[3];
      z += H[6];
    }

    base_col_src += H[1];
    base_row_src += H[4];
    base_z += H[7];

    col_src = base_col_src;
    row_src = base_row_src;
    z = base_z;
  }

  inv_sum_w = 1.0f / sum_w;
  src_color_sum *= inv_sum_w;
  src_color_squared_sum *= inv_sum_w;
  ref_src_color_sum *= inv_sum_w;

  const float ref_color_var = ref_color_squared_sum - ref_color_sum * ref_color_sum;
  const float src_color_var = src_color_squared_sum - src_color_sum * src_color_sum;

  if (ref_color_var < 1e-5 || src_color_var < 1e-5) {
      return kMaxCost;
  }

  const float ref_src_color_cover = ref_src_color_sum - ref_color_sum * src_color_sum;
  const float ref_src_color_var_inv = rsqrtf((ref_color_var * src_color_var));
  const float val = 1.0f - ref_src_color_cover * ref_src_color_var_inv;
  float score = max(0.0f, min(kMaxCost, val));

  if (in_mask) {
    m_ref_color_sum *= inv_sum_w;
    m_ref_color_squared_sum *= inv_sum_w;
    m_src_color_sum *= inv_sum_w;
    m_src_color_squared_sum *= inv_sum_w;
    m_ref_src_color_sum *= inv_sum_w;

    const float m_ref_color_var = 
        m_ref_color_squared_sum - m_ref_color_sum * m_ref_color_sum;
    const float m_src_color_var = 
        m_src_color_squared_sum - m_src_color_sum * m_src_color_sum;

    if (m_ref_color_var < 1e-5 || m_src_color_var < 1e-5) {
        return score;
    }

    const float m_ref_src_color_cover = 
        m_ref_src_color_sum - m_ref_color_sum * m_src_color_sum;
    const float m_ref_src_color_var_inv = rsqrtf((m_ref_color_var * m_src_color_var));
    const float m_val = 1.0f - m_ref_src_color_cover * m_ref_src_color_var_inv;
    float m_score = max(0.0f, min(kMaxCost, m_val));
    score = score * 0.2 + m_score * 0.8;
  }

  return score;
}

template<int WINDOW_RADIUS, int WINDOW_STEP>
__device__ float ComputeNCCCost(const int image_idx,
                                const int row, const int col,
                                const float4 hypothesis,
                                const int width, const int height,
                                const float center_color,
                                const float ref_color_sum,
                                const float ref_color_squared_sum,
                                const float ref_color_var,
                                BilateralWeightComputer weight_computer,
                                const float* ref_image_texture,
                                cudaTextureObject_t src_images_texture,
                                // cudaTextureObject_t poses_texture,
                                const float* poses_texture,
                                const float* ref_invK) {
  const float kMaxCost = 2.0f;

  float H[9];
  ComposeHomography(image_idx, row, col, hypothesis.w, &hypothesis.x, H, poses_texture, ref_invK);

  float inv_z = 1.0f / (H[6] * col + H[7] * row + H[8]);
  int u = (H[0] * col + H[1] * row + H[2]) * inv_z;
  int v = (H[3] * col + H[4] * row + H[5]) * inv_z;
  if (u < 0 || u >= width || v < 0 || v >= height) {
    return kMaxCost;
  }

  const int row_start = row - WINDOW_RADIUS;
  const int col_start = col - WINDOW_RADIUS;

  float col_src = H[0] * col_start + H[1] * row_start + H[2];
  float row_src = H[3] * col_start + H[4] * row_start + H[5];
  float z = H[6] * col_start + H[7] * row_start + H[8];
  float base_col_src = col_src;
  float base_row_src = row_src;
  float base_z = z;

  H[0] = WINDOW_STEP * H[0]; H[1] = WINDOW_STEP * H[1]; H[2] = WINDOW_STEP * H[2];
  H[3] = WINDOW_STEP * H[3]; H[4] = WINDOW_STEP * H[4]; H[5] = WINDOW_STEP * H[5];
  H[6] = WINDOW_STEP * H[6]; H[7] = WINDOW_STEP * H[7]; H[8] = WINDOW_STEP * H[8];

  float src_color_sum = 0.0f;
  float src_color_squared_sum = 0.0f;
  float ref_src_color_sum = 0.0f;
  float sum_w = 0.0f, inv_sum_w;

  const int col_size = blockDim.x + WINDOW_RADIUS * 2;
  for (int dr = -WINDOW_RADIUS; dr <= WINDOW_RADIUS; dr += WINDOW_STEP) {
    for (int dc = -WINDOW_RADIUS; dc <= WINDOW_RADIUS; dc += WINDOW_STEP) {
      // TODO: eliminate the offset 0.5
      const float inv_z = 1.0f / z;
      const float fu = inv_z * col_src + 0.5f;
      const float fv = inv_z * row_src + 0.5f;

      // const float ref_color = tex2D<float>(ref_image_texture, col + dc, row + dr);
      const float ref_color = ref_image_texture[dr * col_size + dc];
      const float src_color = tex2DLayered<float>(src_images_texture, fu, fv, image_idx);

      const float bilateral_weight = weight_computer.Compute(dr, dc, center_color, ref_color);
      const float bilateral_weight_color = bilateral_weight * src_color;

      src_color_sum += bilateral_weight_color;
      src_color_squared_sum += bilateral_weight_color * src_color;
      ref_src_color_sum += bilateral_weight_color * ref_color;
      sum_w += bilateral_weight;

      col_src += H[0];
      row_src += H[3];
      z += H[6];
    }

    base_col_src += H[1];
    base_row_src += H[4];
    base_z += H[7];

    col_src = base_col_src;
    row_src = base_row_src;
    z = base_z;
  }

  inv_sum_w = 1.0f / sum_w;
  src_color_sum *= inv_sum_w;
  src_color_squared_sum *= inv_sum_w;
  ref_src_color_sum *= inv_sum_w;

  const float src_color_var = src_color_squared_sum - src_color_sum * src_color_sum;
  if (ref_color_var < 1e-5 || src_color_var < 1e-5) {
      return kMaxCost;
  }

  const float ref_src_color_cover = ref_src_color_sum - ref_color_sum * src_color_sum;
  const float ref_src_color_var_inv = rsqrtf((ref_color_var * src_color_var));
  const float val = 1.0f - ref_src_color_cover * ref_src_color_var_inv;
  float score = max(0.0f, min(kMaxCost, val));
  return score;
}


__device__ float ComputeNCCCost(const int image_idx,
                                const int row, const int col,
                                const float4 hypothesis,
                                const int width, const int height,
                                const int window_radius, 
                                const int window_step,
                                const float center_color,
                                const float ref_color_sum,
                                const float ref_color_squared_sum,
                                const float ref_color_var,
                                BilateralWeightComputer weight_computer,
                                const float* ref_image_texture,
                                cudaTextureObject_t src_images_texture,
                                // cudaTextureObject_t poses_texture,
                                const float* poses_texture,
                                const float* ref_invK) {
  if (window_radius == 3 && window_step == 1) {
    return ComputeNCCCost<3, 1>(image_idx, row, col, hypothesis, width, height,
                                center_color, ref_color_sum, ref_color_squared_sum, ref_color_var, 
                                weight_computer, ref_image_texture, src_images_texture, 
                                poses_texture, ref_invK);
  } else if (window_radius == 5 && window_step == 1) {
    return ComputeNCCCost<5, 1>(image_idx, row, col, hypothesis, width, height,
                                center_color, ref_color_sum, ref_color_squared_sum, ref_color_var, 
                                weight_computer, ref_image_texture, src_images_texture, 
                                poses_texture, ref_invK);
  } else if (window_radius == 7 && window_step == 1) {
    return ComputeNCCCost<7, 1>(image_idx, row, col, hypothesis, width, height,
                                center_color, ref_color_sum, ref_color_squared_sum, ref_color_var, 
                                weight_computer, ref_image_texture, src_images_texture, 
                                poses_texture, ref_invK);
  } else if (window_radius == 3 && window_step == 2) {
    return ComputeNCCCost<3, 2>(image_idx, row, col, hypothesis, width, height,
                                center_color, ref_color_sum, ref_color_squared_sum, ref_color_var, 
                                weight_computer, ref_image_texture, src_images_texture, 
                                poses_texture, ref_invK);
  } else if (window_radius == 4 && window_step == 2) {
    return ComputeNCCCost<4, 2>(image_idx, row, col, hypothesis, width, height,
                                center_color, ref_color_sum, ref_color_squared_sum, ref_color_var, 
                                weight_computer, ref_image_texture, src_images_texture, 
                                poses_texture, ref_invK);
  } else if (window_radius == 5 && window_step == 2) {
    return ComputeNCCCost<5, 2>(image_idx, row, col, hypothesis, width, height,
                                center_color, ref_color_sum, ref_color_squared_sum, ref_color_var, 
                                weight_computer, ref_image_texture, src_images_texture, 
                                poses_texture, ref_invK);
  } else if (window_radius == 7 && window_step == 2) {
    return ComputeNCCCost<7, 2>(image_idx, row, col, hypothesis, width, height,
                                center_color, ref_color_sum, ref_color_squared_sum, ref_color_var, 
                                weight_computer, ref_image_texture, src_images_texture, 
                                poses_texture, ref_invK);
  } else { // Default.
    return ComputeNCCCost<5, 2>(image_idx, row, col, hypothesis, width, height,
                                center_color, ref_color_sum, ref_color_squared_sum, ref_color_var, 
                                weight_computer, ref_image_texture, src_images_texture, 
                                poses_texture, ref_invK);
  }
}

__device__ float ScorePixel(const SweepOptions options, 
                            const int num_src_image,
                            const int row, const int col,
                            const float4 hypothesis,
                            const float normalized_point[3],
                            const float* ref_invK,
                            const int width, const int height,
                            const float center_color,
                            const float ref_color_sum,
                            const float ref_color_squared_sum,
                            const float ref_color_var,
                            BilateralWeightComputer weight_computer,
                            const float* ref_image_texture,
                            cudaTextureObject_t ref_sum_image_texture,
                            cudaTextureObject_t ref_squared_sum_image_texture,
                            cudaTextureObject_t src_images_texture,
                            cudaTextureObject_t ref_semantic_texture,
                            cudaTextureObject_t src_semantics_texture,
                            cudaTextureObject_t src_depth_maps_texture,
                            // cudaTextureObject_t poses_texture,
                            const float* poses_texture) {
  float costs[MAX_NUM_SRC_IMAGE];
  for (size_t image_idx = 0; image_idx < num_src_image; ++image_idx) {
    float cost = ComputeNCCCost(image_idx, row, col, hypothesis, width, height,
                                options.window_radius, options.window_step,
                                center_color, ref_color_sum, ref_color_squared_sum, 
                                ref_color_var, weight_computer, ref_image_texture,
                                src_images_texture, poses_texture, ref_invK);
    if (options.geom_consistency) {
      cost += options.geom_consistency_regularizer *
              ComputeGeomConsistencyCostShared(row, col, hypothesis.w, normalized_point, image_idx,
                                         options.geom_consistency_max_cost, src_depth_maps_texture, 
                                         poses_texture, &ref_invK[4]);
    }
    costs[image_idx] = cost;
  }
  BubbleSort(costs, num_src_image);
  float conf = 0.0f;
  for (int i = 0; i < options.thk; ++i) {
    conf += costs[i];
  }
  conf /= mweights[options.thk];
  return conf;
}

__device__ void ComputeMultiViewCostVector(const float4 hypothesis,
                                          const int row, const int col,
                                          const float normalized_point[3],
                                          const float* ref_invK,
                                          const int width, const int height,
                                          const int window_radius, 
                                          const int window_step,
                                          const int num_src_image,
                                          const float center_color,
                                          const float ref_color_sum,
                                          const float ref_color_squared_sum,
                                          const float ref_color_var,
                                          const float4 prior_info,
                                          const float prior_wgt,
                                          float *costs,
                                          const SweepOptions options,
                                          BilateralWeightComputer weight_computer,
                                          const float* ref_image_texture,
                                          cudaTextureObject_t src_images_texture,
                                          cudaTextureObject_t src_depth_maps_texture,
                                          // cudaTextureObject_t poses_texture,
                                          const float* poses_texture) {
  
  float prior_weight = 1.0f;
  if (options.prior_depth_ncc && prior_info.w > FLT_EPSILON) {
    prior_weight = ComputeNCCWeightFromPrior(hypothesis, prior_info, ref_color_var);
#ifdef PRIOR_WGT_COST
    if (prior_wgt < 1.0 && prior_wgt > 0.0){
      prior_weight *= 1 - prior_wgt * 0.3;
    }
#endif
  }

  for (int image_idx = 0; image_idx < num_src_image; ++image_idx) {
    float cost = ComputeNCCCost(image_idx, row, col, hypothesis, width, height, 
                                window_radius, window_step, center_color, 
                                ref_color_sum, ref_color_squared_sum, ref_color_var, 
                                weight_computer, ref_image_texture, 
                                src_images_texture, poses_texture, ref_invK);
    cost *= prior_weight;
    if (options.geom_consistency) {
      cost += options.geom_consistency_regularizer *
              ComputeGeomConsistencyCostShared(row, col, hypothesis.w, normalized_point, image_idx,
                                        options.geom_consistency_max_cost, src_depth_maps_texture, 
                                        poses_texture, &ref_invK[4]);
    }
    costs[image_idx] = cost;
  }
}

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
                                  const int thread_index) {
                                  
  __shared__ float shared_pose_addr[43 * MAX_NUM_SRC_IMAGE];
  float *shared_poses_texture = shared_pose_addr;
  for (int image_idx = 0; image_idx < num_src_image; ++image_idx) {
    int pose_index = image_idx * 43;
    for (int i = threadIdx.x; i < 43; i += blockDim.x) {
      shared_pose_addr[i + pose_index] = tex2D<float>(poses_texture, i, image_idx);
    }
  }
  __syncthreads();

  const int row = blockDim.y * blockIdx.y + threadIdx.y;
  const int col = blockDim.x * blockIdx.x + threadIdx.x;
  const int width = depth_map.GetWidth();
  const int height = depth_map.GetHeight();
  if (col < width && row < height) {
    float4 prior_info = {0,0,0,0};
    float prior_wgt = 0;
    if (options.prior_depth_ncc) {
      prior_info.w = prior_depth_map.Get(row, col);

      prior_info.x = prior_normal_map.Get(row, col, 0);
      prior_info.y = prior_normal_map.Get(row, col, 1);
      prior_info.z = prior_normal_map.Get(row, col, 2);

      prior_wgt = prior_wgt_map.Get(row, col);
    }
    int k_idx = (thread_index << 2);
    const float ref_invK[4] = {ref_inv_K[k_idx], ref_inv_K[k_idx + 1], ref_inv_K[k_idx + 2], ref_inv_K[k_idx + 3]};
    const float normalized_point[3] = {
      ref_invK[0] * col + ref_invK[1],
      ref_invK[2] * row + ref_invK[3],
      1.0f
    };

    float4 hypothesis;
    hypothesis.w = depth_map.Get(row, col);
    hypothesis.x = normal_map.Get(row, col, 0);
    hypothesis.y = normal_map.Get(row, col, 1);
    hypothesis.z = normal_map.Get(row, col, 2);

    uint32_t& view_sel = view_sel_map.GetRef(row, col);

    const float center_color = tex2D<float>(ref_image_texture, col, row);
    const float ref_color_sum = tex2D<float>(ref_sum_image_texture, col, row);
    const float ref_color_squared_sum = tex2D<float>(ref_squared_sum_image_texture, col, row);
    const float ref_color_var = ref_color_squared_sum - ref_color_sum * ref_color_sum;

    extern __shared__ float shared_addr[];
    const int row0 = blockIdx.y * blockDim.y;
    const int col0 = blockIdx.x * blockDim.x;
    const int col_size = blockDim.x + options.window_radius * 2;
    float *ref_colors = shared_addr + (options.window_radius + (row - row0)) * col_size
                                    + (options.window_radius + (col - col0));
    for (int dr = -options.window_radius; dr <= options.window_radius; dr += options.window_step) {
      for (int dc = -options.window_radius; dc <= options.window_radius; dc += options.window_step) {
        const float ref_color = tex2D<float>(ref_image_texture, col + dc, row + dr);
        ref_colors[dr * col_size + dc] = ref_color;
      }
    }

    int O[MAX_NUM_SRC_IMAGE];
    float costs[MAX_NUM_SRC_IMAGE];

    ComputeMultiViewCostVector(hypothesis, row, col, normalized_point, ref_invK, width, height,
                               options.window_radius, options.window_step, num_src_image,
                               center_color, ref_color_sum, ref_color_squared_sum, ref_color_var, 
                               prior_info, prior_wgt, costs, options, weight_computer, ref_colors, 
                               src_images_texture, src_depth_maps_texture, shared_poses_texture);

    for (size_t image_idx = 0; image_idx < num_src_image; ++image_idx) {
      float cost = costs[image_idx];
      O[image_idx] = image_idx;
      cost_map.Set(row, col, image_idx, cost);
    }
    BubbleSortByValue(O, costs, num_src_image);
    float conf = 0.0f;
    for (int i = 0; i < options.thk; ++i) {
      conf += costs[i];
      setBit(view_sel, O[i]);
    }
    conf /= mweights[options.thk];
    conf_map.Set(row, col, conf);
  }
}

__global__ void ComputeCurvatureMap(cudaTextureObject_t ref_image_texture,
                                    GpuMat<float> normal_map,
                                    GpuMat<unsigned short> grad_curvature_map) {
  const int row = blockDim.y * blockIdx.y + threadIdx.y;
  const int col = blockDim.x * blockIdx.x + threadIdx.x;
  const int width = normal_map.GetWidth();
  const int height = normal_map.GetHeight();
  if (col < width && row < height) {
    // Compute Gradient.
    float conv_x = 0, conv_y = 0;
    for (int dr = -1; dr <= 1; ++dr) {
      for (int dc = -1; dc <= 1; ++dc) {
        const float color = tex2D<float>(ref_image_texture, col + dc, row + dr);
        conv_x += color * filter_kernel_x[dr + 1][dc + 1];
        conv_y += color * filter_kernel_y[dr + 1][dc + 1];
      }
    }
    const float grad = sqrtf(conv_x * conv_x + conv_y * conv_y);

    // Compute Curvature.
    const int min_x = max(col - 2, 0);
    const int max_x = min(col + 2, width - 1);
    const int min_y = max(row - 2, 0);
    const int max_y = min(row + 2, height - 1);

    float normal[3];
    normal_map.GetSlice(row, col, normal);

    int counter = 0;
    float angles[24];
    for (int y = min_y; y <= max_y; ++y) {
      for (int x = min_x; x <= max_x; ++x) {
        if (y == row && x == col) {
          continue;
        }
        float nnormal[3];
        normal_map.GetSlice(y, x, nnormal);

        float dr = DotProduct3(normal, nnormal);
        angles[counter++] = acos(dr);
      }
    }
    grad_curvature_map.Set(row, col, 0, min(grad * 10000.f, 32767.f));
    if (counter > 0) {
      BubbleSort(angles, counter);
      grad_curvature_map.Set(row, col, 1, angles[counter / 2] * 10000);
    } else {
      grad_curvature_map.Set(row, col, 1, M_PI * 10000);
    }
  }
}

__forceinline__ __device__ float ComputeGradientMapKernel(const int row, const int col, cudaTextureObject_t ref_image_texture) {
  float conv_x = 0, conv_y = 0;
  for (int dr = -1; dr <= 1; ++dr) {
    for (int dc = -1; dc <= 1; ++dc) {
      const float color = tex2D<float>(ref_image_texture, col + dc, row + dr);
      conv_x += color * filter_kernel_x[dr + 1][dc + 1];
      conv_y += color * filter_kernel_y[dr + 1][dc + 1];
    }
  }
  const float mag = sqrtf(conv_x * conv_x + conv_y * conv_y);
  return mag;
}

__global__ void ComputeGradientMap(GpuMat<float> grad_map, cudaTextureObject_t ref_image_texture) {
  const int row = blockDim.y * blockIdx.y + threadIdx.y;
  const int col = blockDim.x * blockIdx.x + threadIdx.x;
  if (col < grad_map.GetWidth() && row < grad_map.GetHeight()) {
    const float grad = ComputeGradientMapKernel(row, col, ref_image_texture);
    grad_map.Set(row, col, grad);
  }
}

__global__ void DilateGradientMap(GpuMat<float> grad_map, const int r) {
  const int row = blockDim.y * blockIdx.y + threadIdx.y;
  const int col = blockDim.x * blockIdx.x + threadIdx.x;
  if (col < grad_map.GetWidth() && row < grad_map.GetHeight()) {
    float grad = grad_map.Get(row, col);
    if (grad < 1.0) {
      return;
    }
    const int min_x = max(col - r, 0);
    const int max_x = min(col + r, (int)grad_map.GetWidth() - 1);
    const int min_y = max(row - r, 0);
    const int max_y = min(row + r, (int)grad_map.GetHeight() - 1);
    for (int y = min_y; y <= max_y; ++y) {
      for (int x = min_x; x <= max_x; ++x) {
        grad_map.Set(y, x, grad);
      }
    }
  }
}

__global__ void InitializeDistMap(const GpuMat<float> grad_map, 
                                  GpuMat<float> dist_map,
                                  const float min_grad_thres) {
  const int row = blockDim.y * blockIdx.y + threadIdx.y;
  const int col = blockDim.x * blockIdx.x + threadIdx.x;
  if (col < grad_map.GetWidth() && row < grad_map.GetHeight()) {
    if (grad_map.Get(row, col) < min_grad_thres || row == 0 || col == 0 ||
        row == grad_map.GetHeight() - 1 || col == grad_map.GetWidth() -1) {
      dist_map.Set(row, col, 100000.0f);
    } else {
      dist_map.Set(row, col, 0);
    }
  }
}

__global__ void ComputeRowDistMap(GpuMat<float> dist_map) {
  const int col = blockDim.x * blockIdx.x + threadIdx.x;
  if (col >= dist_map.GetWidth()) {
    return;
  }
  const float d1 = 1.0f;
  for (int row = 1; row < dist_map.GetHeight(); ++row) {
    float n_dist = dist_map.Get(row, col);
    float nd_tmp;
    if (n_dist != 0 && (nd_tmp = d1 + dist_map.Get(row - 1, col)) < n_dist) {
      dist_map.Set(row, col, nd_tmp);
    }
  }
  for (int row = dist_map.GetHeight() - 2; row >= 0; --row) {
    float n_dist = dist_map.Get(row, col);
    float nd_tmp;
    if (n_dist != 0 && (nd_tmp = d1 + dist_map.Get(row + 1, col)) < n_dist) {
      dist_map.Set(row, col, nd_tmp);
    }
  }
}

__global__ void ComputeColDistMap(GpuMat<float> dist_map) {
  const int row = blockDim.x * blockIdx.x + threadIdx.x;
  if (row >= dist_map.GetHeight()) {
    return;
  }
  const float d1 = 1.0f;
  for (int col = 1; col < dist_map.GetWidth(); ++col) {
    float n_dist = dist_map.Get(row, col);
    float nd_tmp;
    if (n_dist != 0 && (nd_tmp = d1 + dist_map.Get(row, col - 1)) < n_dist) {
      dist_map.Set(row, col, nd_tmp);
    }
  }
  for (int col = dist_map.GetWidth() - 2; col >= 0 ; --col) {
    float n_dist = dist_map.Get(row, col);
    float nd_tmp;
    if (n_dist != 0 && (nd_tmp = d1 + dist_map.Get(row, col + 1)) < n_dist) {
      dist_map.Set(row, col, nd_tmp);
    }
  }
}

__global__ void ComputeCovarianceMap(const GpuMat<float> depth_map, 
                                     GpuMat<float> covariance_map) {
  const int row = blockDim.y * blockIdx.y + threadIdx.y;
  const int col = blockDim.x * blockIdx.x + threadIdx.x;
  const int width = covariance_map.GetWidth();
  const int height = covariance_map.GetHeight();
  if (col < 0 || col >= width || row < 0 || row >= height) {
    return;
  }
  int sx = max(col - 5, 0);
  int ex = min(col + 5, width - 1);
  int sy = max(row - 5, 0);
  int ey = min(row + 5, height - 1);
  float vals[64];
  int num_val = 0;
  float mean(0), stdev(0), covariance(-1);
  for (int iy = sy; iy <= ey; iy += 2) {
    for (int ix = sx; ix <= ex; ix += 2) {
      const float depth = depth_map.Get(iy, ix);
      if (depth > 0) {
        mean += depth;
        vals[num_val] = depth;
        num_val++;
      }
    }
  }
  if (num_val > 0) {
    mean /= num_val;
    for (int i = 0; i < num_val; ++i) {
      stdev += (vals[i] - mean) * (vals[i] - mean);
    }
    covariance = sqrtf(stdev / num_val);
  }
  covariance_map.Set(row, col, covariance);
}

#if DENSE_SHARPNESS == DENSE_SHARPNESS_GRAD
__forceinline__ __device__ float ComputeGradientMapsKernel(const int row, const int col,
                                           const int image_idx,
                                           cudaTextureObject_t src_images_texture) {
  float conv_x = 0, conv_y = 0;
  for (int dr = -1; dr <= 1; ++dr) {
    for (int dc = -1; dc <= 1; ++dc) {
      const float color = 
        tex2DLayered<float>(src_images_texture, col + dc, row + dr, image_idx);
      conv_x += color * filter_kernel_x[dr + 1][dc + 1];
      conv_y += color * filter_kernel_y[dr + 1][dc + 1];
    }
  }
  const float mag = sqrtf(conv_x * conv_x + conv_y * conv_y);
  return mag;
}

__global__ void ComputeGradientsMap(GpuMat<float> grad_maps,
  cudaTextureObject_t src_images_texture) {
  const int row = blockDim.y * blockIdx.y + threadIdx.y;
  const int col = blockDim.x * blockIdx.x + threadIdx.x;
  if (col < grad_maps.GetWidth() && row < grad_maps.GetHeight()) {
    for (int image_idx = 0; image_idx < grad_maps.GetDepth(); ++image_idx) {
      const float grad = ComputeGradientMapsKernel(row, col, image_idx,
                                    src_images_texture);
      grad_maps.Set(row, col, image_idx, grad);
    }
  }
}
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
                                    const int thread_index) {
  const int row = blockDim.y * blockIdx.y + threadIdx.y;
  const int col = blockDim.x * blockIdx.x + threadIdx.x;
  const int tid = options.thread_index;
  const int width = depth_map.GetWidth();
  const int height = depth_map.GetHeight();
  if (col < width && row < height) {
    float depth = depth_map.Get(row, col);
    if (depth <= 0) {
      return;
    }
    float normal[3];
    normal_map.GetSlice(row, col, normal);
    float point[3];
    ComputePointAtDepth(row, col, depth, point, tid);
    
    int num_good_view(0);
    for (int i = 0; i < MAX_CIRCULAR_NEIGHBOR; ++i) {
      int u = neighbor_offs[i][1] + col;
      int v = neighbor_offs[i][0] + row;
      if (u < 0 || u >= width || v < 0 || v >= height) {
        continue;
      }
      float ndepth = depth_map.Get(v, u);
      float nnormal[3];
      normal_map.GetSlice(v, u, nnormal);

      float npoint[3];
      ComputePointAtDepth(v, u, ndepth, npoint, tid);
      float dist1 = fabs((point[0] - npoint[0]) * normal[0]
                      + (point[1] - npoint[1]) * normal[1]
                      + (point[2] - npoint[2]) * normal[2]);
      float dist2 = fabs((point[0] - npoint[0]) * nnormal[0]
                      + (point[1] - npoint[1]) * nnormal[1]
                      + (point[2] - npoint[2]) * nnormal[2]);
      if (dist1 / depth < depth_diff_thres && dist2 / ndepth < depth_diff_thres) {
        num_good_view++;
      }
    }

    float factor = __expf(-num_good_view * num_good_view / 8.0);
    float conf = conf_map.Get(row, col);
    conf_map.Set(row, col, conf * factor);
  }
}

// __global__ void EstimateLocalPlanarity(const SweepOptions options,
//                                        GpuMat<float> depth_map,
//                                        GpuMat<float> normal_map,
//                                        GpuMat<float> planarity_map) {
//   const int row = blockDim.y * blockIdx.y + threadIdx.y;
//   const int col = blockDim.x * blockIdx.x + threadIdx.x;
//   const int width = depth_map.GetWidth();
//   const int height = depth_map.GetHeight();
//   const int window_radius = options.window_radius;
//   const int patch_length = (window_radius << 1) + 1;
//   const int patch_size = patch_length * patch_length;
//   if (col >= width - window_radius || row >= height - window_radius ||
//       col < window_radius || row < window_radius) {
//     return;
//   }
//   int num_consistent = 0;
//   const float depth = depth_map.Get(row, col);
//   for (int r = row - window_radius; r <= row + window_radius; ++r) {
//     for (int c = col - window_radius; c <= col + window_radius; ++c) {
//       const float ndepth = depth_map.Get(r, c);
//       const float diff_depth = fabs(depth - ndepth) / depth;
//       num_consistent += (diff_depth < 0.01);
//     }
//   }
//   planarity_map.Set(row, col, num_consistent * 1.0f / patch_size);
// }

__device__ void SweepFromCheckerBoard(curandState* rand_state_map,
                                      float* depth_map,
                                      float* normal_map,
                                      float* conf_map,
                                      const int row, const int col,
                                      const int width, const int height, 
                                      const int pitch, const int num_src_image,
                                      const SweepOptions options,
                                      BilateralWeightComputer weight_computer,
                                      cudaTextureObject_t ref_image_texture,
                                      cudaTextureObject_t ref_sum_image_texture,
                                      cudaTextureObject_t ref_squared_sum_image_texture,
                                      cudaTextureObject_t src_images_texture,
                                      cudaTextureObject_t src_depth_maps_texture,
                                      cudaTextureObject_t poses_texture) {
  int row_index = pitch * row;
  int row_index1 = pitch * (height + row);
  int row_index2 = pitch * (2 * height + row);
  int col_index = col;

  curandState* rand_state = (curandState*)((char*)rand_state_map + row_index) + col_index;

  float conf = *((float*)((char*)conf_map + row_index) + col_index);
  float depth = *((float*)((char*)depth_map + row_index) + col_index);
  float normal[3];
  normal[0] = *((float*)((char*)normal_map + row_index) + col_index);
  normal[1] = *((float*)((char*)normal_map + row_index1) + col_index);
  normal[2] = *((float*)((char*)normal_map + row_index2) + col_index);
  float ndepth, nconf, nnormal[3];

  const int tid = options.thread_index;
  const float depth_min = options.depth_min;
  const float depth_max = 1.2 * options.depth_max;
  const float random_depth_ratio = options.random_depth_ratio;
  const float random_angle1 = min(options.random_angle1_range, DEG2RAD(180.0f));
  const float random_angle2 = min(options.random_angle2_range, DEG2RAD(90.0f));

  int k_idx = (tid << 2);
  const float ref_invK[8] = {ref_inv_K[k_idx], ref_inv_K[k_idx + 1], ref_inv_K[k_idx + 2], ref_inv_K[k_idx + 3],
                             ref_K[k_idx], ref_K[k_idx + 1], ref_K[k_idx + 2], ref_K[k_idx + 3]};
  float view_dir[3];
  view_dir[0] = ref_invK[0] * col + ref_invK[1];
  view_dir[1] = ref_invK[2] * row + ref_invK[3];
  view_dir[2] = 1.0f;
  const float inv_view_len = rsqrtf(view_dir[0] * view_dir[0] + 
                                    view_dir[1] * view_dir[1] + 
                                    view_dir[2] * view_dir[2]);

  const float normalized_point[3] = {
    ref_invK[0] * col + ref_invK[1], ref_invK[2] * row + ref_invK[3], 1.0f};

  // find hypotheses.
  float4 hypotheses[K_NUM_SAMPS + 2], hypothesis;
  int num_valid_samp = 0;
  for (int idx = 0; idx < K_NUM_SAMPS; ++idx) {
    int best_r, best_c;
    int best_row_index, best_row_index1, best_row_index2, best_col_index;
    float best_conf = FLT_MAX;
    for (int iff = 0; iff < K_NUM_EACH_SAMP; ++iff) {
      int nrow = row + dirSamples[idx][iff * 2];
      int ncol = col + dirSamples[idx][iff * 2 + 1];
      if (nrow < 0 || nrow >= height ||
          ncol < 0 || ncol >= width) {
        continue;
      }
      int nrow_index = pitch * nrow;
      int nrow_index1 = pitch * (height + nrow);
      int nrow_index2 = pitch * (2 * height + nrow);
      int ncol_index = ncol;
      nconf = *((float*)((char*)conf_map + nrow_index) + ncol_index);
      if (best_conf > nconf) {
        best_r = nrow;
        best_c = ncol;
        best_row_index = nrow_index;
        best_row_index1 = nrow_index1;
        best_row_index2 = nrow_index2;
        best_col_index = ncol_index;
        best_conf = nconf;
      }
    }
    if (best_conf < FLT_MAX) {
      ndepth = *((float*)((char*)depth_map + best_row_index) + best_col_index);
      nnormal[0] = *((float*)((char*)normal_map + best_row_index) + best_col_index);
      nnormal[1] = *((float*)((char*)normal_map + best_row_index1) + best_col_index);
      nnormal[2] = *((float*)((char*)normal_map + best_row_index2) + best_col_index);
      hypothesis.w = PropagateDepth(ndepth, nnormal, ref_invK, best_r, best_c, normalized_point);
      hypothesis.x = nnormal[0];
      hypothesis.y = nnormal[1];
      hypothesis.z = nnormal[2];
      hypotheses[num_valid_samp++] = hypothesis;
    }
  }
  {
    int num_valid_neighbor = 0;
    float neigh_confs[MAX_CIRCULAR_NEIGHBOR], O[MAX_CIRCULAR_NEIGHBOR];
    for (int i = 0; i < MAX_CIRCULAR_NEIGHBOR; ++i) {
      O[i] = i;
      int u = neighbor_offs[i][1] + col;
      int v = neighbor_offs[i][0] + row;
      if (u < 0 || u >= width || v < 0 || v >= height) {
        neigh_confs[i] = FLT_MAX;
        continue;
      }
      num_valid_neighbor++;
      int nrow_index = pitch * v;
      int ncol_index = u;
      neigh_confs[i] = *((float*)((char*)conf_map + nrow_index) + ncol_index);
    }

    BubbleSortByValue(O, neigh_confs, MAX_CIRCULAR_NEIGHBOR);
    num_valid_neighbor = min(num_valid_neighbor, 2);
    for (int i = 0; i < num_valid_neighbor; ++i) {
      int idx = O[i];
      int u = neighbor_offs[idx][1] + col;
      int v = neighbor_offs[idx][0] + row;

      int nrow_index = pitch * v;
      int nrow_index1 = pitch * (height + v);
      int nrow_index2 = pitch * (2 * height + v);
      int ncol_index = u;

      ndepth = *((float*)((char*)depth_map + nrow_index) + ncol_index);
      nnormal[0] = *((float*)((char*)normal_map + nrow_index) + ncol_index);
      nnormal[1] = *((float*)((char*)normal_map + nrow_index1) + ncol_index);
      nnormal[2] = *((float*)((char*)normal_map + nrow_index2) + ncol_index);

      hypothesis.w = PropagateDepth(ndepth, nnormal, v, u, row, col, tid);
      hypothesis.x = normal[0];
      hypothesis.y = normal[1];
      hypothesis.z = normal[2];
      hypotheses[num_valid_samp++] = hypothesis;
    }
  }

  const float center_color = tex2D<float>(ref_image_texture, col, row);
  const float ref_color_sum = tex2D<float>(ref_sum_image_texture, col, row);
  const float ref_color_squared_sum = tex2D<float>(ref_squared_sum_image_texture, col, row);
  const float ref_color_var = ref_color_squared_sum - ref_color_sum * ref_color_sum;

  extern __shared__ float shared_addr[];
  const int row0 = blockIdx.y * blockDim.y * 2;
  const int col0 = blockIdx.x * blockDim.x;
  const int col_size = blockDim.x + options.window_radius * 2;
  float *ref_colors = shared_addr + (options.window_radius + (row - row0)) * col_size
                                  + (options.window_radius + (col - col0));
  for (int dr = -options.window_radius; dr <= options.window_radius; dr += options.window_step) {
    for (int dc = -options.window_radius; dc <= options.window_radius; dc += options.window_step) {
      const float ref_color = tex2D<float>(ref_image_texture, col + dc, row + dr);
      ref_colors[dr * col_size + dc] = ref_color;
    }
  }

  __shared__ float shared_pose_addr[43 * MAX_NUM_SRC_IMAGE];
  float *shared_poses_texture = shared_pose_addr;
  for (int image_idx = 0; image_idx < num_src_image; ++image_idx) {
    int pose_index = image_idx * 43;
    for (int i = 0; i < 43; ++i) {
      shared_pose_addr[i + pose_index] = tex2D<float>(poses_texture, i, image_idx);
    }
  }

  for (int i = 0; i < num_valid_samp; ++i) {
    nconf = ScorePixel(options, num_src_image, row, col, hypotheses[i], normalized_point, 
                      ref_invK, width, height, center_color, ref_color_sum, ref_color_squared_sum, 
                      ref_color_var, weight_computer, ref_colors, ref_sum_image_texture, 
                      ref_squared_sum_image_texture, src_images_texture, NULL, NULL, 
                      src_depth_maps_texture, shared_poses_texture);
    if (conf > nconf) {
      depth = hypotheses[i].w;
      normal[0] = hypotheses[i].x;
      normal[1] = hypotheses[i].y;
      normal[2] = hypotheses[i].z;
      conf = nconf;
    }
  }

  // Optimization of depth and normal.
  num_valid_samp = 0;
  ndepth = PerturbDepth(random_depth_ratio, depth, rand_state);
  PerturbNormal(normal, nnormal, random_angle1, random_angle2, rand_state);
  if (nnormal[2] >= 0) {
    nnormal[0] = -nnormal[0];
    nnormal[1] = -nnormal[1];
    nnormal[2] = -nnormal[2];
  }
  hypothesis = make_float4(nnormal[0], nnormal[1], nnormal[2], ndepth);
  hypotheses[num_valid_samp++] = hypothesis;

  hypothesis = make_float4(nnormal[0], nnormal[1], nnormal[2], depth);
  hypotheses[num_valid_samp++] = hypothesis;

  hypothesis = make_float4(normal[0], normal[1], normal[2], ndepth);
  hypotheses[num_valid_samp++] = hypothesis;

  float rdepth = GenerateRandomDepth(depth_min, depth_max, rand_state);
  float rnormal[3];
  GenerateRandomNormal(row, col, rand_state, rnormal, tid);

  hypothesis = make_float4(rnormal[0], rnormal[1], rnormal[2], depth);
  hypotheses[num_valid_samp++] = hypothesis;

  hypothesis = make_float4(normal[0], normal[1], normal[2], rdepth);
  hypotheses[num_valid_samp++] = hypothesis;

  for (int i = 0; i < num_valid_samp; ++i) {

    nconf = ScorePixel(options, num_src_image, row, col, hypotheses[i], normalized_point, 
                      ref_invK, width, height, center_color, ref_color_sum, ref_color_squared_sum, 
                      ref_color_var, weight_computer, ref_colors, ref_sum_image_texture, 
                      ref_squared_sum_image_texture, src_images_texture, NULL, NULL, 
                      src_depth_maps_texture, shared_poses_texture);
                             
    if (conf > nconf) {
      depth = hypotheses[i].w;
      normal[0] = hypotheses[i].x;
      normal[1] = hypotheses[i].y;
      normal[2] = hypotheses[i].z;
      conf = nconf;
    }
  }
  if (depth >= depth_min && depth <= depth_max) {
    *((float*)((char*)conf_map + row_index) + col_index) = conf;
    *((float*)((char*)depth_map + row_index) + col_index) = depth;
    *((float*)((char*)normal_map + row_index) + col_index) = normal[0];
    *((float*)((char*)normal_map + row_index1) + col_index) = normal[1];
    *((float*)((char*)normal_map + row_index2) + col_index) = normal[2];
  }
}

__device__ void SweepFromCheckerBoardOpt(curandState* rand_state_map,
                                         float* depth_map,
                                         float* normal_map,
                                         float* conf_map,
                                         float* grad_map,
                                         float* cost_map,
                                         uint32_t* view_sel_map,
                                        //  float* planarity_map,
                                         float* prior_depth_map,
                                         float* prior_normal_map,
                                         float* prior_wgt_map,
                                        //  uint32_t* selected_images_list,
                                         const int row, const int col,
                                         const int width, const int height, 
                                         const int pitch, 
                                         const int num_src_image,
                                         const SweepOptions options,
                                         BilateralWeightComputer weight_computer,
                                         cudaTextureObject_t ref_image_texture,
                                         cudaTextureObject_t ref_sum_image_texture,
                                         cudaTextureObject_t ref_squared_sum_image_texture,
                                         cudaTextureObject_t src_images_texture,
                                        //  cudaTextureObject_t ref_semantic_texture,
                                        //  cudaTextureObject_t src_semantics_texture,
                                         cudaTextureObject_t src_depth_maps_texture,
                                         cudaTextureObject_t poses_texture) {

  __shared__ float shared_pose_addr[43 * MAX_NUM_SRC_IMAGE];
  float *shared_poses_texture = shared_pose_addr;
  for (int image_idx = 0; image_idx < num_src_image; ++image_idx) {
    int pose_index = image_idx * 43;
    for (int i = threadIdx.x; i < 43; i += blockDim.x) {
      shared_pose_addr[i + pose_index] = tex2D<float>(poses_texture, i, image_idx);
    }
  }
  __syncthreads();

  int pitch_height = pitch * height;
  int row_index = pitch * row;
  int col_index = col;

  curandState* rand_state = (curandState*)((char*)rand_state_map + row_index) + col_index;

  float conf = *((float*)((char*)conf_map + row_index) + col_index);
  float depth = *((float*)((char*)depth_map + row_index) + col_index);
  float normal[3];
  char *row_normal_map = (char*)normal_map + row_index;
  normal[0] = *((float*)row_normal_map + col_index);
  normal[1] = *((float*)(row_normal_map + pitch_height) + col_index);
  normal[2] = *((float*)(row_normal_map + (pitch_height << 1)) + col_index);
  float grad = *((float*)((char*)grad_map + row_index) + col_index);

  float prior_random_factor = 1.0f;
  float4 prior_info = {0,0,0,0};
  float prior_wgt = 0.0;
  if (options.propagate_depth || options.prior_depth_ncc) {
    prior_info.w = *((float*)((char*)prior_depth_map + row_index) + col_index);

    char *prior_row_normal_map = (char*)prior_normal_map + row_index;
    prior_info.x = *((float*)prior_row_normal_map + col_index);
    prior_info.y = *((float*)(prior_row_normal_map + pitch_height) + col_index);
    prior_info.z = *((float*)(prior_row_normal_map + (pitch_height << 1)) + col_index);

    prior_wgt = *((float*)((char*)prior_wgt_map + row_index) + col_index);
  }
  if (options.prior_depth_ncc && prior_info.w > 1e-6){
    prior_random_factor = 0.1;
  }

  uint32_t ref_view_sel = 0;

  float ndepth, nconf, nnormal[3];

  const int tid = options.thread_index;
  const float inv_beta2 = 0.5f / (options.beta * options.beta);
  const float depth_min = options.depth_min;
  const float depth_max = 1.2 * options.depth_max;
  const float random_angle1 = min(options.random_angle1_range * prior_random_factor, DEG2RAD(180.0f));
  const float random_angle2 = min(options.random_angle2_range * prior_random_factor, DEG2RAD(90.0f));
  int num_valid_samp = 0, num_valid_neighbor = 0;
  
  int k_idx = (tid << 2);
  const float ref_invK[8] = {ref_inv_K[k_idx], ref_inv_K[k_idx + 1], ref_inv_K[k_idx + 2], ref_inv_K[k_idx + 3],
                             ref_K[k_idx], ref_K[k_idx + 1], ref_K[k_idx + 2], ref_K[k_idx + 3]};
  float view_dir[3];
  view_dir[0] = ref_invK[0] * col + ref_invK[1];
  view_dir[1] = ref_invK[2] * row + ref_invK[3];
  view_dir[2] = 1.0f;
  const float inv_view_len = rsqrtf(view_dir[0] * view_dir[0] + 
                                    view_dir[1] * view_dir[1] + 
                                    view_dir[2] * view_dir[2]);

  const float normalized_point[3] = {
    ref_invK[0] * col + ref_invK[1], ref_invK[2] * row + ref_invK[3], 1.0f
  };

  // find neighbors.
  float4 hypotheses[K_NUM_SAMPS + 3], hypothesis;
  hypothesis = make_float4(normal[0], normal[1], normal[2], depth);
  hypotheses[num_valid_samp++] = hypothesis;

  if (grad >= 1.0) {
    float neigh_confs[MAX_CIRCULAR_NEIGHBOR], O[MAX_CIRCULAR_NEIGHBOR];
    for (int i = 0; i < MAX_CIRCULAR_NEIGHBOR; ++i) {
      O[i] = i;
      int u = neighbor_offs[i][1] + col;
      int v = neighbor_offs[i][0] + row;
      if (u < 0 || u >= width || v < 0 || v >= height) {
        neigh_confs[i] = FLT_MAX;
        continue;
      }
      num_valid_neighbor++;
      int nrow_index = pitch * v;
      int ncol_index = u;
      neigh_confs[i] = *((float*)((char*)conf_map + nrow_index) + ncol_index);
    }

    BubbleSortByValue(O, neigh_confs, MAX_CIRCULAR_NEIGHBOR);
    num_valid_neighbor = min(num_valid_neighbor, 2);
    for (int i = 0; i < num_valid_neighbor; ++i) {
      int idx = O[i];
      int u = neighbor_offs[idx][1] + col;
      int v = neighbor_offs[idx][0] + row;

      int nrow_index = pitch * v;
      int ncol_index = u;

      char *nrow_normal_map = (char*)normal_map + nrow_index;
      ndepth = *((float*)((char*)depth_map + nrow_index) + ncol_index);
      nnormal[0] = *((float*)nrow_normal_map + ncol_index);
      nnormal[1] = *((float*)(nrow_normal_map + pitch_height) + ncol_index);
      nnormal[2] = *((float*)(nrow_normal_map + (pitch_height << 1)) + ncol_index);
      CorrectNormal(nnormal, view_dir, inv_view_len);

      hypothesis.w = PropagateDepth(ndepth, nnormal, ref_invK, v, u, normalized_point);
      hypothesis.x = nnormal[0];
      hypothesis.y = nnormal[1];
      hypothesis.z = nnormal[2];
      hypotheses[num_valid_samp++] = hypothesis;
    }
  }

  for (int idx = 0; idx < K_NUM_SAMPS; ++idx) {
    int best_r, best_c, *ptr = dirSamples[idx];
    float best_conf = FLT_MAX;
    for (int iff = 0; iff < K_NUM_EACH_SAMP; ++iff) {
      int k = (iff << 1);
      int nrow = row + ptr[k];
      int ncol = col + ptr[k + 1];
      if (nrow < 0 || nrow >= height ||
          ncol < 0 || ncol >= width) {
        continue;
      }
      int nrow_index = pitch * nrow;
      int ncol_index = ncol;

      float prior_weight_samp = 0.0;
      if (options.propagate_depth || options.prior_depth_ncc) {
        float pws = *((float*)((char*)prior_wgt_map + nrow_index) + ncol_index);
        if (pws > 1e-6){
          prior_weight_samp = pws;
        }
      }
      // float w1 = (grad > 1) ? 1.0f : 1.0f - *((float*)((char*)planarity_map + nrow_index) + ncol_index);
      // nconf = *((float*)((char*)conf_map + nrow_index) + ncol_index) * w1;
      nconf = *((float*)((char*)conf_map + nrow_index) + ncol_index);
#ifdef PRIOR_WGT_PROGET
      nconf *= (1 - 0.2 * prior_weight_samp);
#endif
      if (best_conf > nconf) {
        best_r = nrow;
        best_c = ncol;
        best_conf = nconf;
      }
    }
    if (best_conf < FLT_MAX) {
      const int best_row_index = pitch * best_r;
      const int best_col_index = best_c;
      ndepth = *((float*)((char*)depth_map + best_row_index) + best_col_index);
      char *nrow_normal_map = (char*)normal_map + best_row_index;
      nnormal[0] = *((float*)nrow_normal_map + best_col_index);
      nnormal[1] = *((float*)(nrow_normal_map + pitch_height) + best_col_index);
      nnormal[2] = *((float*)(nrow_normal_map + (pitch_height << 1)) + best_col_index);

      CorrectNormal(nnormal, view_dir, inv_view_len);
      hypothesis.w = PropagateDepth(ndepth, nnormal, ref_invK, best_r, best_c, normalized_point);
      hypothesis.x = nnormal[0];
      hypothesis.y = nnormal[1];
      hypothesis.z = nnormal[2];
      hypotheses[num_valid_samp++] = hypothesis;
    }
  }

  const float center_color = tex2D<float>(ref_image_texture, col, row);
  const float ref_color_sum = tex2D<float>(ref_sum_image_texture, col, row);
  const float ref_color_squared_sum = tex2D<float>(ref_squared_sum_image_texture, col, row);
  const float ref_color_var = ref_color_squared_sum - ref_color_sum * ref_color_sum;

  extern __shared__ float shared_addr[];
  const int row0 = blockIdx.y * blockDim.y * 2;
  const int col0 = blockIdx.x * blockDim.x;
  const int col_size = blockDim.x + options.window_radius * 2;
  float *ref_colors = shared_addr + (options.window_radius + (row - row0)) * col_size
                                   + (options.window_radius + (col - col0));
  for (int dr = -options.window_radius; dr <= options.window_radius; dr += options.window_step) {
    for (int dc = -options.window_radius; dc <= options.window_radius; dc += options.window_step) {
      const float ref_color = tex2D<float>(ref_image_texture, col + dc, row + dr);
      ref_colors[dr * col_size + dc] = ref_color;
    }
  }

  uint32_t flags(0);
  float sampling_probs[MAX_NUM_SRC_IMAGE] = {0.0f};
  float M[K_NUM_SAMPS + 3][MAX_NUM_SRC_IMAGE], *ptr, *min_ptr;

  ptr = M[0];
  for (int image_idx = 0; image_idx < num_src_image; ++image_idx) {
    ptr[image_idx] = *((float*)((char*)cost_map + pitch_height * image_idx + row_index) + col_index);
  }
  for (int neigh_idx = 1; neigh_idx < num_valid_samp; ++neigh_idx) {
    ComputeMultiViewCostVector(hypotheses[neigh_idx], row, col, normalized_point, ref_invK, 
                               width, height, options.window_radius, options.window_step, 
                               num_src_image, center_color, ref_color_sum, ref_color_squared_sum,
                               ref_color_var, prior_info, prior_wgt, M[neigh_idx], options,
                               weight_computer, ref_colors, src_images_texture,
                               src_depth_maps_texture, shared_poses_texture);
  }

  for (int image_idx = 0; image_idx < num_src_image; ++image_idx) {
    unsigned int n1(0), n2(0);
    for (int neigh_idx = 0; neigh_idx < num_valid_samp; ++neigh_idx) {
      float cost = M[neigh_idx][image_idx];
      sampling_probs[image_idx] += __expf(-cost * cost * inv_beta2);
      n1 += (cost < options.th_mc);
      n2 += (cost > options.max_ncc_matching_cost);
    }
    if (n1 >= options.num_good_hypothesis && n2 <= options.num_bad_hypothesis) {
      setBit(flags, image_idx);
    }
  }
  for (int image_idx = 0; image_idx < num_src_image; ++image_idx) {
    if (flags != 0 && !isSet(flags, image_idx)) {
      sampling_probs[image_idx] = 0;
    }
  }
  if (flags == 0) {
    return;
  }

  TransformPDFToCDF(sampling_probs, num_src_image);

  float view_weights[MAX_NUM_SRC_IMAGE] = {0.0f}, weight_norm(0.0f), inv_weight_norm(0.0f);
  for (int sample = 0; sample < options.num_samples; ++sample) {
    const float rand_prob = curand_uniform(rand_state) - FLT_EPSILON;
    for (int image_idx = 0; image_idx < num_src_image; ++image_idx) {
      const float prob = sampling_probs[image_idx];
      if (prob > rand_prob) {
        if (options.random_optimization) {
          view_weights[image_idx] = 1.0f;
        } else {
          view_weights[image_idx] += 1.0f;
        }
        break;
      }
    }
  }

  float costs[K_NUM_SAMPS + 3] = {0};
  for (int image_idx = 0; image_idx < num_src_image; ++image_idx) {
    weight_norm += view_weights[image_idx];
  }
  if (weight_norm <= 1e-5) {
    return;
  }
  inv_weight_norm = 1.0f / weight_norm;
  ref_view_sel = 0;
  for (int image_idx = 0; image_idx < num_src_image; ++image_idx) {
    view_weights[image_idx] *= inv_weight_norm;
    if (view_weights[image_idx] > 0) {
      setBit(ref_view_sel, image_idx);
    }
  }

  for (int i = 0; i < num_valid_samp; ++i) {
    ptr = M[i];
    float cost(0);
    for (int image_idx = 0; image_idx < num_src_image; ++image_idx) {
      cost += view_weights[image_idx] * ptr[image_idx];
    }
    costs[i] = cost;
  }
  *((uint32_t*)((char*)view_sel_map + row_index) + col_index) = ref_view_sel;

  // Find the parameters of the minimum cost and update hypothesis.
  int min_cost_idx;
  if (options.prior_depth_ncc && prior_info.w > 0.0f)
  {
    float prior_depth_min = min(prior_info.w * 0.95f - 0.05, 0.0f);
    float prior_depth_max = prior_info.w * 1.05f + 0.05;
    // float prior_depth_min = prior_info.w * 0.95f - 0.05f;
    // float prior_depth_max = prior_info.w * 1.05f + 0.05;
    min_cost_idx = FindMinCostWithin(costs, num_valid_samp, hypotheses, prior_depth_min, prior_depth_max);
  }
  else 
  {
    min_cost_idx = FindMinCost(costs, num_valid_samp);
  }
  depth = hypotheses[min_cost_idx].w;
  normal[0] = hypotheses[min_cost_idx].x;
  normal[1] = hypotheses[min_cost_idx].y;
  normal[2] = hypotheses[min_cost_idx].z;
  conf = costs[min_cost_idx];

  // Optimization of depth and normal.
  num_valid_samp = 0;
  hypothesis = make_float4(normal[0], normal[1], normal[2], depth);
  hypotheses[num_valid_samp++] = hypothesis;

  ndepth = PerturbDepth(options.random_depth_ratio, depth, rand_state);
  PerturbNormal(normal, nnormal, random_angle1, random_angle2, rand_state);
  CorrectNormal(nnormal, view_dir, inv_view_len);

  // hypothesis = make_float4(nnormal[0], nnormal[1], nnormal[2], ndepth);
  // hypotheses[num_valid_samp++] = hypothesis;

  hypothesis = make_float4(nnormal[0], nnormal[1], nnormal[2], depth);
  hypotheses[num_valid_samp++] = hypothesis;

  hypothesis = make_float4(normal[0], normal[1], normal[2], ndepth);
  hypotheses[num_valid_samp++] = hypothesis;

  if (!options.propagate_depth || 
       options.propagate_depth && prior_info.w <= 0) {
    float rdepth = GenerateRandomDepth(depth_min, depth_max, rand_state);
    float rnormal[3];
    GenerateRandomNormal(row, col, rand_state, view_dir, rnormal);
    CorrectNormal(rnormal, view_dir, inv_view_len);

    hypothesis = make_float4(rnormal[0], rnormal[1], rnormal[2], depth);
    hypotheses[num_valid_samp++] = hypothesis;

    hypothesis = make_float4(normal[0], normal[1], normal[2], rdepth);
    hypotheses[num_valid_samp++] = hypothesis;
  } else {
    float rnormal[3];
    GenerateRandomNormal(row, col, rand_state, view_dir, rnormal);
    CorrectNormal(rnormal, view_dir, inv_view_len);
    
    hypothesis = make_float4(rnormal[0], rnormal[1], rnormal[2], depth);
    hypotheses[num_valid_samp++] = hypothesis;
  }

  costs[0] = 0;
  ptr = M[0];
  min_ptr = M[min_cost_idx];
  for (int image_idx = 0; image_idx < num_src_image; ++image_idx) {
    ptr[image_idx] = min_ptr[image_idx];
    costs[0] += view_weights[image_idx] * ptr[image_idx];
  }

  for (int neigh_idx = 1; neigh_idx < num_valid_samp; ++neigh_idx) {
    ComputeMultiViewCostVector(hypotheses[neigh_idx], row, col, normalized_point, ref_invK, 
                               width, height, options.window_radius, options.window_step, 
                               num_src_image, center_color, ref_color_sum, ref_color_squared_sum,
                               ref_color_var, prior_info, prior_wgt, M[neigh_idx], options,
                               weight_computer, ref_colors, src_images_texture,
                               src_depth_maps_texture, shared_poses_texture);
  }

  for (int neigh_idx = 1; neigh_idx < num_valid_samp; ++neigh_idx) {
    costs[neigh_idx] = 0;
    ptr = M[neigh_idx];
    for (int image_idx = 0; image_idx < num_src_image; ++image_idx) {
      float cost = ptr[image_idx];
      costs[neigh_idx] += view_weights[image_idx] * cost;
    }
  }
  
  // Find the parameters of the minimum cost and update hypothesis.
  min_cost_idx = FindMinCost(costs, num_valid_samp);
  depth = hypotheses[min_cost_idx].w;
  normal[0] = hypotheses[min_cost_idx].x;
  normal[1] = hypotheses[min_cost_idx].y;
  normal[2] = hypotheses[min_cost_idx].z;
  conf = costs[min_cost_idx];
  
  if (min_cost_idx != -1) {
    *((float*)((char*)conf_map + row_index) + col_index) = conf;
    *((float*)((char*)depth_map + row_index) + col_index) = depth;
    row_normal_map = (char*)normal_map + row_index;
    *((float*)row_normal_map + col_index) = normal[0];
    *((float*)(row_normal_map + pitch_height) + col_index) = normal[1];
    *((float*)(row_normal_map + (pitch_height << 1)) + col_index) = normal[2];
    min_ptr = M[min_cost_idx];
    // for (int i = 0; i < num_hit_image; ++i) {
    //   int image_idx = image_idx_list[i];
    for (int image_idx = 0; image_idx < num_src_image; ++image_idx) {
      *((float*)((char*)cost_map + pitch_height * image_idx + row_index) + col_index) = min_ptr[image_idx];
    }
    // // for (int i = 0; i < num_miss_image; ++i) {
    // //   int image_idx = miss_image_idx_list[i];
    // for (int image_idx = 0; image_idx < num_src_image; ++image_idx) {
    //   float cost = ComputeNCCCost(image_idx, row, col, hypotheses[min_cost_idx], 
    //                               width, height, window_radius, window_step,
    //                               options.plane_regularizer, weight_computer,
    //                               ref_image_texture, ref_sum_image_texture,
    //                               ref_squared_sum_image_texture, src_images_texture, 
    //                               ref_semantic_texture, src_semantics_texture,
    //                               poses_texture, tid);

    //   if (options.prior_depth_ncc && prior_depth > FLT_EPSILON) {
    //     cost *= ComputeNCCWeightFromPrior(hypotheses[min_cost_idx].w, prior_depth);
    //   }
    //   if (options.geom_consistency) {
    //     cost += options.geom_consistency_regularizer *
    //             ComputeGeomConsistencyCostShared(row, col, hypotheses[min_cost_idx].w, image_idx,
    //                                       options.geom_consistency_max_cost,
    //                                       src_depth_maps_texture, poses_texture,
    //                                       tid);
    //   }
    //   *((float*)((char*)cost_map + pitch_height * image_idx + row_index) + col_index) = cost;
    // }
  }
  *((curandState*)((char*)rand_state_map + row_index) + col_index) = *rand_state;
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
                                cudaTextureObject_t poses_texture) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (threadIdx.x % 2 == 0) {
    row = (row << 1);
  } else {
    row = (row << 1) + 1;
  }

  if (row < 0 || row >= depth_map.GetHeight() || 
      col < 0 || col >= depth_map.GetWidth()) {
    return;
  }

  SweepFromCheckerBoard(rand_state_map.GetPtr(), depth_map.GetPtr(),
                        normal_map.GetPtr(), conf_map.GetPtr(),
                        row, col, depth_map.GetWidth(), 
                        depth_map.GetHeight(), depth_map.GetPitch(), 
                        num_src_image, options, weight_computer, 
                        ref_image_texture, ref_sum_image_texture, 
                        ref_squared_sum_image_texture, src_images_texture, 
                        src_depth_maps_texture, poses_texture);
}

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
                              cudaTextureObject_t poses_texture) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (threadIdx.x % 2 == 0) {
    row = (row << 1) + 1;
  } else {
    row = (row << 1);
  }

  if (row < 0 || row >= depth_map.GetHeight() || 
      col < 0 || col >= depth_map.GetWidth()) {
    return;
  }
  
  SweepFromCheckerBoard(rand_state_map.GetPtr(), depth_map.GetPtr(),
                        normal_map.GetPtr(), conf_map.GetPtr(),
                        row, col, depth_map.GetWidth(), 
                        depth_map.GetHeight(), depth_map.GetPitch(), 
                        num_src_image, options, weight_computer, 
                        ref_image_texture, ref_sum_image_texture, 
                        ref_squared_sum_image_texture, src_images_texture, 
                        src_depth_maps_texture, poses_texture);
}

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
                                    cudaTextureObject_t poses_texture) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (threadIdx.x % 2 == 0) {
    row = (row << 1);
  } else {
    row = (row << 1) + 1;
  }

  if (row < 0 || row >= depth_map.GetHeight() || 
      col < 0 || col >= depth_map.GetWidth()) {
    return;
  }

  SweepFromCheckerBoardOpt(rand_state_map.GetPtr(), depth_map.GetPtr(),
                          normal_map.GetPtr(), conf_map.GetPtr(), 
                          grad_map.GetPtr(), cost_map.GetPtr(), 
                          view_sel_map.GetPtr(), /*planarity_map.GetPtr(),*/ 
                          (options.propagate_depth || options.prior_depth_ncc) ? prior_depth_map.GetPtr() : NULL,
                          (options.propagate_depth || options.prior_depth_ncc) ? prior_normal_map.GetPtr() : NULL,
                          (options.propagate_depth || options.prior_depth_ncc) ? prior_wgt_map.GetPtr() : NULL,
                          /*selected_images_list.GetPtr(), */row, col, depth_map.GetWidth(), depth_map.GetHeight(), 
                          depth_map.GetPitch(), num_src_image,
                          options, weight_computer, ref_image_texture,
                          ref_sum_image_texture, ref_squared_sum_image_texture, 
                          src_images_texture, /*ref_semantic_texture, src_semantics_texture, */
                          src_depth_maps_texture, poses_texture);
}

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
                                  cudaTextureObject_t poses_texture) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (threadIdx.x % 2 == 0) {
    row = (row << 1) + 1;
  } else {
    row = (row << 1);
  }

  if (row < 0 || row >= depth_map.GetHeight() || 
      col < 0 || col >= depth_map.GetWidth()) {
    return;
  }
  
  SweepFromCheckerBoardOpt(rand_state_map.GetPtr(), depth_map.GetPtr(),
                          normal_map.GetPtr(), conf_map.GetPtr(), 
                          grad_map.GetPtr(), cost_map.GetPtr(), 
                          view_sel_map.GetPtr(), /*planarity_map.GetPtr(),*/ 
                          (options.propagate_depth || options.prior_depth_ncc) ? prior_depth_map.GetPtr() : NULL,
                          (options.propagate_depth || options.prior_depth_ncc) ? prior_normal_map.GetPtr() : NULL,
                          (options.propagate_depth || options.prior_depth_ncc) ? prior_wgt_map.GetPtr() : NULL,
                          /*selected_images_list.GetPtr(), */row, col, depth_map.GetWidth(), depth_map.GetHeight(), 
                          depth_map.GetPitch(), num_src_image, 
                          options, weight_computer, ref_image_texture,
                          ref_sum_image_texture, ref_squared_sum_image_texture, 
                          src_images_texture, /*ref_semantic_texture, src_semantics_texture,*/
                          src_depth_maps_texture, poses_texture);
}

__device__ void LocalOptimization(curandState* rand_state_map,
                                  float* depth_map,
                                  float* normal_map,
                                  float* conf_map,
                                  float* cost_map,
                                  uint32_t* view_sel_map,
                                  // uint32_t* selected_images_list,
                                  const int row, const int col,
                                  const int width, const int height, 
                                  const int pitch, 
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
                                  cudaTextureObject_t poses_texture) {
  int row_index = pitch * row;
  int pitch_height = pitch * height;
  int row_index1 = pitch_height + row_index;
  int row_index2 = (pitch_height << 1) + row_index;
  int col_index = col;

  curandState* rand_state = (curandState*)((char*)rand_state_map + row_index) + col_index;

  float conf = *((float*)((char*)conf_map + row_index) + col_index);
  float depth = *((float*)((char*)depth_map + row_index) + col_index);
  float normal[3];
  normal[0] = *((float*)((char*)normal_map + row_index) + col_index);
  normal[1] = *((float*)((char*)normal_map + row_index1) + col_index);
  normal[2] = *((float*)((char*)normal_map + row_index2) + col_index);
  uint32_t ref_view_sel = *((uint32_t*)((char*)view_sel_map + row_index) + col_index);
  float ndepth, nnormal[3];

  const int tid = options.thread_index;
  const float random_angle1 = DEG2RAD(4.0);
  const float random_angle2 = DEG2RAD(2.0);
  const int window_radius = options.window_radius;
  const int window_step = options.window_step;

  int k_idx = (tid << 2);
  const float ref_invK[8] = {ref_inv_K[k_idx], ref_inv_K[k_idx + 1], ref_inv_K[k_idx + 2], ref_inv_K[k_idx + 3],
                             ref_K[k_idx], ref_K[k_idx + 1], ref_K[k_idx + 2], ref_K[k_idx + 3]};
  const float normalized_point[3] = {
    ref_invK[0] * col + ref_invK[1], ref_invK[2] * row + ref_invK[3], 1.0f
  };

  const float center_color = tex2D<float>(ref_image_texture, col, row);
  const float ref_color_sum = tex2D<float>(ref_sum_image_texture, col, row);
  const float ref_color_squared_sum = tex2D<float>(ref_squared_sum_image_texture, col, row);
  const float ref_color_var = ref_color_squared_sum - ref_color_sum * ref_color_sum;

  extern __shared__ float shared_addr[];
  const int row0 = blockIdx.y * blockDim.y * 2;
  const int col0 = blockIdx.x * blockDim.x;
  const int col_size = blockDim.x + options.window_radius * 2;
  float *ref_colors = shared_addr + (options.window_radius + (row - row0)) * col_size
                                  + (options.window_radius + (col - col0));
  for (int dr = -options.window_radius; dr <= options.window_radius; dr += options.window_step) {
    for (int dc = -options.window_radius; dc <= options.window_radius; dc += options.window_step) {
      const float ref_color = tex2D<float>(ref_image_texture, col + dc, row + dr);
      ref_colors[dr * col_size + dc] = ref_color;
    }
  }

  __shared__ float shared_pose_addr[43 * MAX_NUM_SRC_IMAGE];
  float *shared_poses_texture = shared_pose_addr;
  for (int image_idx = 0; image_idx < num_src_image; ++image_idx) {
    int pose_index = image_idx * 43;
    for (int i = 0; i < 43; ++i) {
      shared_pose_addr[i + pose_index] = tex2D<float>(poses_texture, i, image_idx);
    }
  }

  int num_valid_samp = 0;

  float4 hypothesis, m_hypothesis, up_hypothesis, low_hypothesis;
  m_hypothesis = make_float4(normal[0], normal[1], normal[2], depth);
  up_hypothesis = low_hypothesis = m_hypothesis;
  num_valid_samp++;

  for (int i = 0; i < MAX_CIRCULAR_NEIGHBOR; ++i) {
    int u = neighbor_offs[i][1] + col;
    int v = neighbor_offs[i][0] + row;
    if (u < 0 || u >= width || v < 0 || v >= height) {
      continue;
    }
    int nrow_index = pitch * v;
    int nrow_index1 = pitch_height + nrow_index;
    int nrow_index2 = (pitch_height << 1) + nrow_index;
    int ncol_index = u;
    
    ndepth = *((float*)((char*)depth_map + nrow_index) + ncol_index);
    nnormal[0] = *((float*)((char*)normal_map + nrow_index) + ncol_index);
    nnormal[1] = *((float*)((char*)normal_map + nrow_index1) + ncol_index);
    nnormal[2] = *((float*)((char*)normal_map + nrow_index2) + ncol_index);
    
    num_valid_samp++;
    m_hypothesis.w += ndepth;
    m_hypothesis.x += nnormal[0];
    m_hypothesis.y += nnormal[1];
    m_hypothesis.z += nnormal[2];
    if (ndepth <= low_hypothesis.w) {
      low_hypothesis.w = ndepth;
      low_hypothesis.x = nnormal[2];
      low_hypothesis.y = nnormal[1];
      low_hypothesis.z = nnormal[2];
    }
    if (ndepth > up_hypothesis.w){
      up_hypothesis.w = ndepth;
      up_hypothesis.x = nnormal[2];
      up_hypothesis.y = nnormal[1];
      up_hypothesis.z = nnormal[2];
    }
  }

  float min_depth = low_hypothesis.w;
  float max_depth = up_hypothesis.w;
  // float mdepth = m_hypothesis.w / num_valid_samp;
  float mnormal[3] = {m_hypothesis.x, m_hypothesis.y, m_hypothesis.z};
  float inv_norm = rsqrtf(mnormal[0] * mnormal[0] + mnormal[1] * mnormal[1] + mnormal[2] * mnormal[2]);
  mnormal[0] *= inv_norm; mnormal[1] *= inv_norm; mnormal[2] *= inv_norm;

  int min_cost_idx = -1;
  float M[5][MAX_NUM_SRC_IMAGE];
  for (int iter = 0; iter < 5; ++iter) {
    // ndepth = PerturbDepth(random_depth_ratio, depth, rand_state);
    hypothesis.w = GenerateRandomDepth(min_depth, max_depth, rand_state);
    PerturbNormal(mnormal, &hypothesis.x, random_angle1, random_angle2, rand_state);

    float *ptr = M[iter];
    float mconf(0.0), m_weight(0.0);
    for (int image_idx = 0; image_idx < num_src_image; ++image_idx) {
      if (!isSet(ref_view_sel, image_idx)) {
        continue;
      }
      float cost = ComputeNCCCost(image_idx, row, col, hypothesis, width, height, window_radius, window_step,
                                  center_color, ref_color_sum, ref_color_squared_sum, ref_color_var, 
                                  weight_computer, ref_colors, src_images_texture, shared_poses_texture, ref_invK);
      if (options.geom_consistency) {
        cost += options.geom_consistency_regularizer *
                ComputeGeomConsistencyCostShared(row, col, hypothesis.w, normalized_point, image_idx,
                                          options.geom_consistency_max_cost, src_depth_maps_texture, 
                                          shared_poses_texture, &ref_invK[4]);
      }
      ptr[image_idx] = cost;
      // if (isSet(ref_view_sel, image_idx)) {
      mconf += cost;
      m_weight += 1;
      // }
    }
    if (m_weight <= 0) {
      return ;
    }
    mconf /= m_weight;
    if (mconf < conf) {
      conf = mconf;
      depth = hypothesis.w;
      normal[0] = hypothesis.x;
      normal[1] = hypothesis.y;
      normal[2] = hypothesis.z;
      min_cost_idx = iter;
    }
  }

  if (min_cost_idx != -1) {
    *((float*)((char*)conf_map + row_index) + col_index) = conf;
    *((float*)((char*)depth_map + row_index) + col_index) = depth;
    *((float*)((char*)normal_map + row_index) + col_index) = normal[0];
    *((float*)((char*)normal_map + row_index1) + col_index) = normal[1];
    *((float*)((char*)normal_map + row_index2) + col_index) = normal[2];
    float *ptr = M[min_cost_idx];
    hypothesis = make_float4(normal[0], normal[1], normal[2], depth);
    for (int image_idx = 0; image_idx < num_src_image; ++image_idx) {
      if (isSet(ref_view_sel, image_idx)) {
        *((float*)((char*)cost_map + pitch_height * image_idx + row_index) + col_index) = ptr[image_idx];
      } else {
        float cost = ComputeNCCCost(image_idx, row, col, hypothesis, width, height, window_radius, window_step,
                                    center_color, ref_color_sum, ref_color_squared_sum, ref_color_var, 
                                    weight_computer, ref_colors, src_images_texture, shared_poses_texture, ref_invK);
        if (options.geom_consistency) {
          cost += options.geom_consistency_regularizer *
                ComputeGeomConsistencyCostShared(row, col, hypothesis.w, normalized_point, image_idx,
                                          options.geom_consistency_max_cost, src_depth_maps_texture, 
                                          shared_poses_texture, ref_invK);
        }
        *((float*)((char*)cost_map + pitch_height * image_idx + row_index) + col_index) = cost;
      }
    }
  }
  *((curandState*)((char*)rand_state_map + row_index) + col_index) = *rand_state;
}

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
                                  cudaTextureObject_t poses_texture) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (threadIdx.x % 2 == 0) {
    row = (row << 1);
  } else {
    row = (row << 1) + 1;
  }

  if (row < 0 || row >= depth_map.GetHeight() || 
      col < 0 || col >= depth_map.GetWidth()) {
    return;
  }

  LocalOptimization(rand_state_map.GetPtr(), depth_map.GetPtr(),
                    normal_map.GetPtr(), conf_map.GetPtr(), 
                    cost_map.GetPtr(), view_sel_map.GetPtr(), 
                    /*selected_images_list.GetPtr(),*/ row, col, 
                    depth_map.GetWidth(), depth_map.GetHeight(), 
                    depth_map.GetPitch(), num_src_image,
                    options, weight_computer, ref_image_texture,
                    ref_sum_image_texture, ref_squared_sum_image_texture,
                    src_images_texture,/* ref_semantic_texture, src_semantics_texture, */
                    src_depth_maps_texture, poses_texture);
}

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
                                  cudaTextureObject_t poses_texture) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (threadIdx.x % 2 == 0) {
    row = (row << 1) + 1;
  } else {
    row = (row << 1);
  }

  if (row < 0 || row >= depth_map.GetHeight() || 
      col < 0 || col >= depth_map.GetWidth()) {
    return;
  }

  LocalOptimization(rand_state_map.GetPtr(), depth_map.GetPtr(),
                    normal_map.GetPtr(), conf_map.GetPtr(), 
                    cost_map.GetPtr(), view_sel_map.GetPtr(), 
                    /*selected_images_list.GetPtr(),*/ row, col, 
                    depth_map.GetWidth(), depth_map.GetHeight(), 
                    depth_map.GetPitch(), num_src_image,
                    options, weight_computer, ref_image_texture,
                    ref_sum_image_texture, ref_squared_sum_image_texture,
                    src_images_texture, /*ref_semantic_texture, src_semantics_texture,*/
                    src_depth_maps_texture, poses_texture);
}

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
    GpuMat<float> filtered_depth_map, GpuMat<float> filtered_normal_map) {
    const int row = blockDim.y * blockIdx.y + threadIdx.y;
    const int col = blockDim.x * blockIdx.x + threadIdx.x;

    const int width = depth_map.GetWidth();
    const int height = depth_map.GetHeight();
    if (row < 1 || row >= height - 1 || col < 1 || col >= width - 1) {
      return;
    }
    const float spatial_norm(1.0f / 4.5f);
    const float color_norm(1.0f / 0.1f);
    const float center_color = tex2D<float>(ref_image_texture, col, row);

    float m_weight(0.0f), m_depth(0.0f), m_normal[3] = {0.0, 0.0, 0.0};
    for (int dr = -1; dr <= 1; ++dr) {
      for (int dc = -1; dc <= 1; ++dc) {
        int ir = row + dr;
        int ic = col + dc;
        const float depth = depth_map.Get(ir, ic);
        if (depth <= 0) {
          continue;
        }
        float normal[3];
        normal_map.GetSlice(ir, ic, normal);

        const float d2 = dr * dr + dc * dc;
        const float ncolor = tex2D<float>(ref_image_texture, ic, ir);
        const float diff_color = center_color - ncolor;
        const float w = __expf(-d2 * spatial_norm - diff_color * diff_color * color_norm);

        m_depth += w * depth;
        m_normal[0] += w * normal[0];
        m_normal[1] += w * normal[1];
        m_normal[2] += w * normal[2];
        m_weight += w;
      }
    }

    m_depth /= m_weight;
    const float inv_norm = rsqrtf(m_normal[0] * m_normal[0] + 
                            m_normal[1] * m_normal[1] +
                            m_normal[2] * m_normal[2]);
    m_normal[0] *= inv_norm;
    m_normal[1] *= inv_norm;
    m_normal[2] *= inv_norm;

    filtered_depth_map.Set(row, col, m_depth);
    filtered_normal_map.SetSlice(row, col, m_normal);
}

__global__ void MaskDepthMapKernel(GpuMat<float> depth_map, cudaTextureObject_t ref_mask_texture) {
    const int row = blockDim.y * blockIdx.y + threadIdx.y;
    const int col = blockDim.x * blockIdx.x + threadIdx.x;

    const int width = depth_map.GetWidth();
    const int height = depth_map.GetHeight();
    if (row < 0 || row >= height - 1 || col < 0 || col >= width - 1) {
      return;
    }
    const uint8_t mask = tex2D<uint8_t>(ref_mask_texture, col, row);
    if (mask <= 0) {
      depth_map.Set(row, col, 0);
    }
}

__global__ void ComputeGeomGradientMap(GpuMat<float> depth_map, GpuMat<float> grad_map) {
  const int row = blockDim.y * blockIdx.y + threadIdx.y;
  const int col = blockDim.x * blockIdx.x + threadIdx.x;
  const int width = grad_map.GetWidth();
  const int height = grad_map.GetHeight();
  if (col < width && row < height) {
    float depth = depth_map.Get(row, col);
    if (depth <= 0) return;
    float conv_x = 0, conv_y = 0;
    for (int dr = -1; dr <= 1; ++dr) {
      for (int dc = -1; dc <= 1; ++dc) {
        int nrow = row + dr;
        int ncol = col + dc;
        if (nrow < 0 || nrow >= height || ncol < 0 || ncol >= width) {
          continue;
        }
        const float ndepth = depth_map.Get(nrow, ncol);
        const float diff_depth = fabs(ndepth - depth) / depth;
        conv_x += diff_depth * filter_kernel_x[dr + 1][dc + 1];
        conv_y += diff_depth * filter_kernel_y[dr + 1][dc + 1];
      }
    }
    const float mag = sqrtf(conv_x * conv_x + conv_y * conv_y);
    grad_map.Set(row, col, mag);
  }
}

__forceinline__ __device__ void MedianFilter(float* depth_map, float* normal_map, float *grad_map, 
    const int row, const int col, const int width, const int height, const int pitch,
    const float diff_thres) {
    const int patch_size = 17;
    const float grad_thres = 1.4f * diff_thres;
    int index = 0;

    float depths[patch_size], nx[patch_size], ny[patch_size], nz[patch_size];

    int row_index = pitch * row;
    int pitch_height = pitch * height;
    int row_index1 = pitch_height + row_index;
    int row_index2 = (pitch_height << 1) + row_index;
    int col_index = col;
    int nrow, ncol, nrow_index, nrow_index1, nrow_index2, ncol_index;

    float grad = *((float*)((char*)grad_map + row_index) + col_index);
    float cdepth = *((float*)((char*)depth_map + row_index) + col_index);
    float inv_cdepth = 1.0f / cdepth;
    depths[index] = cdepth;

    nx[index] = *((float*)((char*)normal_map + row_index) + col_index);
    ny[index] = *((float*)((char*)normal_map + row_index1) + col_index);
    nz[index] = *((float*)((char*)normal_map + row_index2) + col_index);
    index++;
    for (int i = 0; i < 2; ++i) {      
      nrow = row - ((i << 1) + 1);
      nrow_index = pitch * nrow;
      nrow_index1 = pitch_height + nrow_index;
      nrow_index2 = (pitch_height << 1) + nrow_index;
      ncol_index = col;
      if (nrow >= 0) {
        depths[index] = *((float*)((char*)depth_map + nrow_index) + ncol_index);
        float diff_depth = fabs(depths[index] - cdepth) * inv_cdepth;
        if (grad < grad_thres || (grad >= grad_thres && diff_depth < diff_thres)) {
          nx[index] = *((float*)((char*)normal_map + nrow_index) + ncol_index);
          ny[index] = *((float*)((char*)normal_map + nrow_index1) + ncol_index);
          nz[index] = *((float*)((char*)normal_map + nrow_index2) + ncol_index);
          index++;
        }
      }

      nrow = row + ((i << 1) + 1);
      nrow_index = pitch * nrow;
      nrow_index1 = pitch_height + nrow_index;
      nrow_index2 = (pitch_height << 1) + nrow_index;
      ncol_index = col;
      if (nrow < height) {
        depths[index] = *((float*)((char*)depth_map + nrow_index) + ncol_index);
        float diff_depth = fabs(depths[index] - cdepth) * inv_cdepth;
        if (grad < grad_thres || (grad >= grad_thres && diff_depth < diff_thres)) {
          nx[index] = *((float*)((char*)normal_map + nrow_index) + ncol_index);
          ny[index] = *((float*)((char*)normal_map + nrow_index1) + ncol_index);
          nz[index] = *((float*)((char*)normal_map + nrow_index2) + ncol_index);
          index++;
        }
      }
      
      ncol = col - ((i << 1) + 1);
      nrow_index = pitch * row;
      nrow_index1 = pitch_height + nrow_index;
      nrow_index2 = (pitch_height << 1) + nrow_index;
      ncol_index = ncol;
      if (ncol >= 0) {
        depths[index] = *((float*)((char*)depth_map + nrow_index) + ncol_index);
        float diff_depth = fabs(depths[index] - cdepth) * inv_cdepth;
        if (grad < grad_thres || (grad >= grad_thres && diff_depth < diff_thres)) {
          nx[index] = *((float*)((char*)normal_map + nrow_index) + ncol_index);
          ny[index] = *((float*)((char*)normal_map + nrow_index1) + ncol_index);
          nz[index] = *((float*)((char*)normal_map + nrow_index2) + ncol_index);
          index++;
        }
      }

      ncol = col + ((i << 1) + 1);
      nrow_index = pitch * row;
      nrow_index1 = pitch_height + nrow_index;
      nrow_index2 = (pitch_height << 1) + nrow_index;
      ncol_index = ncol;
      if (ncol < width) {
        depths[index] = *((float*)((char*)depth_map + nrow_index) + ncol_index);
        float diff_depth = fabs(depths[index] - cdepth) * inv_cdepth;
        if (grad < grad_thres || (grad >= grad_thres && diff_depth < diff_thres)) {
          nx[index] = *((float*)((char*)normal_map + nrow_index) + ncol_index);
          ny[index] = *((float*)((char*)normal_map + nrow_index1) + ncol_index);
          nz[index] = *((float*)((char*)normal_map + nrow_index2) + ncol_index);
          index++;
        }
      }
    }

    int dirs[8][2] = {{-2, 1}, {-1, 2}, {1, 2}, {2, 1}, {2, -1}, {1, -2}, {-1, -2}, {-2, -1}};
    for (int i = 0; i < 8; ++i) {
      int nrow = row + dirs[i][0];
      int ncol = col + dirs[i][1];
      if (nrow >= 0 && nrow < height && ncol >= 0 && ncol < width) {
        nrow_index = pitch * nrow;
        nrow_index1 = pitch_height + nrow_index;
        nrow_index2 = (pitch_height << 1) + nrow_index;
        ncol_index = ncol;
        depths[index] = *((float*)((char*)depth_map + nrow_index) + ncol_index);
        float diff_depth = fabs(depths[index] - cdepth) * inv_cdepth;
        if (grad < grad_thres || (grad >= grad_thres && diff_depth < diff_thres)) {
          nx[index] = *((float*)((char*)normal_map + nrow_index) + ncol_index);
          ny[index] = *((float*)((char*)normal_map + nrow_index1) + ncol_index);
          nz[index] = *((float*)((char*)normal_map + nrow_index2) + ncol_index);
          index++;
        }
      }
    }

    BubbleSort(depths, index);
    BubbleSort(nx, index);
    BubbleSort(ny, index);
    BubbleSort(nz, index);
    int nth = (index >> 1);
    float inv_norm = rsqrtf(nx[nth] * nx[nth] + ny[nth] * ny[nth] + nz[nth] * nz[nth]);

    *((float*)((char*)depth_map + row_index) + col_index) = depths[nth];
    *((float*)((char*)normal_map + row_index) + col_index) = nx[nth] * inv_norm;
    *((float*)((char*)normal_map + row_index1) + col_index) = ny[nth] * inv_norm;
    *((float*)((char*)normal_map + row_index2) + col_index) = nz[nth] * inv_norm;
}

__global__ void BlackPixelFilter(GpuMat<float> depth_map, GpuMat<float> normal_map, 
  GpuMat<float> grad_map, const float diff_thres) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (threadIdx.x % 2 == 0) {
    row = (row << 1);
  } else {
    row = (row << 1) + 1;
  }

  if (row < 0 || row >= depth_map.GetHeight() || 
      col < 0 || col >= depth_map.GetWidth()) {
    return;
  }
  MedianFilter(depth_map.GetPtr(), normal_map.GetPtr(), grad_map.GetPtr(), row, col,
               depth_map.GetWidth(), depth_map.GetHeight(), depth_map.GetPitch(), diff_thres);
}

__global__ void RedPixelFilter(GpuMat<float> depth_map, GpuMat<float> normal_map, 
  GpuMat<float> grad_map, const float diff_thres) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (threadIdx.x % 2 == 0) {
    row = (row << 1) + 1;
  } else {
    row = (row << 1);
  }

  if (row < 0 || row >= depth_map.GetHeight() || 
      col < 0 || col >= depth_map.GetWidth()) {
    return;
  }
  MedianFilter(depth_map.GetPtr(), normal_map.GetPtr(), grad_map.GetPtr(), row, col,
               depth_map.GetWidth(), depth_map.GetHeight(), depth_map.GetPitch(), diff_thres);
}

__global__ void FilterDepthMap(GpuMat<float> depth_map,
                               GpuMat<float> normal_map,
                               GpuMat<float> conf_map,
                               const float depth_diff_thres,
                               const float conf_thres,
                               const int thread_index,
                               const bool red_or_black) {
  int row = blockDim.y * blockIdx.y + threadIdx.y;
  int col = blockDim.x * blockIdx.x + threadIdx.x;
  if (threadIdx.x % 2 == 0) {
    row = (row << 1) + (red_or_black ? 1 : 0);
  } else {
    row = (row << 1) + (red_or_black ? 0 : 1);
  }

  const int width = depth_map.GetWidth();
  const int height = depth_map.GetHeight();
  if (col < width && row < height) {
    // const float conf = conf_map.Get(row, col);
    // if (conf > conf_thres) {
    //   depth_map.Set(row, col, 0.0f);
    //   normal_map.Set(row, col, 0, 0.0f);
    //   normal_map.Set(row, col, 1, 0.0f);
    //   normal_map.Set(row, col, 2, 0.0f);
    //   conf_map.Set(row, col, 0.0f);
    //   return;
    // }

    float depth = depth_map.Get(row, col);
    if (depth <= 0) {
      return;
    }
    float normal[3];
    normal_map.GetSlice(row, col, normal);

    float conf = conf_map.Get(row, col);

    float point[3];
    ComputePointAtDepth(row, col, depth, point, thread_index);
    
    int num_good_neighbor(0);
    for (int i = 0; i < MAX_CIRCULAR_NEIGHBOR; ++i) {
      int u = neighbor_offs[i][1] + col;
      int v = neighbor_offs[i][0] + row;
      if (u < 0 || u >= width || v < 0 || v >= height) {
        continue;
      }
      float ndepth = depth_map.Get(v, u);
      float nnormal[3];
      normal_map.GetSlice(v, u, nnormal);
      float nconf = conf_map.Get(v, u);
      float npoint[3];
      ComputePointAtDepth(v, u, ndepth, npoint, thread_index);
      float dist1 = fabs((point[0] - npoint[0]) * normal[0]
                      + (point[1] - npoint[1]) * normal[1]
                      + (point[2] - npoint[2]) * normal[2]);
      float dist2 = fabs((point[0] - npoint[0]) * nnormal[0]
                      + (point[1] - npoint[1]) * nnormal[1]
                      + (point[2] - npoint[2]) * nnormal[2]);
      if ((conf < conf_thres && dist1 / depth < depth_diff_thres) ||
          (nconf < conf_thres && dist2 / ndepth < depth_diff_thres)) {
        num_good_neighbor++;
      }
    }
    if (num_good_neighbor < 2) {
      depth_map.Set(row, col, 0.0f);
      normal_map.Set(row, col, 0, 0.0f);
      normal_map.Set(row, col, 1, 0.0f);
      normal_map.Set(row, col, 2, 0.0f);
      conf_map.Set(row, col, 0.0f);
    }
  }
}

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
                                  const int thread_index) {
  const int row = blockDim.y * blockIdx.y + threadIdx.y;
  const int col = blockDim.x * blockIdx.x + threadIdx.x;
  if (col < depth_map.GetWidth() && row < depth_map.GetHeight()) {

    // const float zeros[3] = {0,0,0};
    const float depth = depth_map.Get(row, col);
    // float normal[3];
    // normal_map.GetSlice(row, col, normal);
    // const float conf = conf_map.Get(row, col);

    if (depth < 1e-6){
      depth_map.Set(row, col, 0);
      // normal_map.SetSlice(row, col, zeros);
      // conf_map.Set(row, col, 0);
      return;
    }

    int num_good_src_depth(0);
    for (size_t image_idx = 0; image_idx < num_src_image; ++image_idx) {
      if (ComputeGeomConsistencyCost(row, col, depth, image_idx,
          geom_consistency_max_cost, src_depth_maps_texture, poses_texture, 
          thread_index) <= filter_geom_consistency_max_cost){
        num_good_src_depth++;
      }
    }

    if (num_good_src_depth < filter_min_num_consistent
    //  && conf > fitler_conf_threshold
     ){
      depth_map.Set(row, col, 0);
      // normal_map.SetSlice(row, col, zeros);
      // conf_map.Set(row, col, 0);
      return;
    }
#if 0
    float forward_point[3];
    ComputePointAtDepth(row, col, depth, forward_point, thread_index);

    const int delta_col[8] = {-1, 1, 0, 0, -1, 1, -1, 1};
    const int delta_row[8] = {0, 0, -1, 1, -1, -1, 1, 1};
    int num_good_neighbor_depth(0);
    int num_neighbor(0);
    for (int i = 0; i < 8; i++){
      const int around_col = col + delta_col[i];
      const int around_row = row + delta_row[i];

      if (around_col < 0 || around_col >= depth_map.GetWidth() 
        || around_row < 0 || around_row >= depth_map.GetHeight()){
        continue;
      }

      float neighbor_depth = depth_map.Get(around_row, around_col);
      if (neighbor_depth < 1e-6){
        continue;
      }
      num_neighbor++;
      float normal_around[3];
      normal_map.GetSlice(around_row, around_col, normal_around);

      float forward_point_around[3];
      ComputePointAtDepth(around_row, around_col, neighbor_depth, 
                          forward_point_around, thread_index);
      float dist1 = fabs((forward_point[0] - forward_point_around[0]) * normal[0]
                    + (forward_point[1] - forward_point_around[1]) * normal[1]
                    + (forward_point[2] - forward_point_around[2]) * normal[2]);
      float dist2 = fabs((forward_point[0] - forward_point_around[0]) * normal_around[0]
                    + (forward_point[1] - forward_point_around[1]) * normal_around[1]
                    + (forward_point[2] - forward_point_around[2]) * normal_around[2]);
      if (dist1 / depth < filter_neighbor_depth_error && 
          dist2 / depth < filter_neighbor_depth_error ){
          ++ num_good_neighbor_depth;
      }
    }

    if(num_good_neighbor_depth < 4 && conf > fitler_conf_threshold){
      depth_map.Set(row, col, 0);
      normal_map.SetSlice(row, col, zeros);
      conf_map.Set(row, col, 0);
      return;
    }
#endif
  }
}

__global__ void DeduplicDepthMapsKernel(const int num_src_image,
                                  GpuMat<float> depth_map,
                                  // GpuMat<float> normal_map,
                                  float geom_consistency_max_cost,
                                  float filter_geom_consistency_max_cost,
                                  float filter_neighbor_depth_error,
                                  int filter_min_num_consistent,
                                  cudaTextureObject_t src_depth_maps_texture,
                                  cudaTextureObject_t poses_texture,
                                  const int thread_index) {
  const int row = blockDim.y * blockIdx.y + threadIdx.y;
  const int col = blockDim.x * blockIdx.x + threadIdx.x;
  if (col < depth_map.GetWidth() && row < depth_map.GetHeight()) {

    // const float zeros[3] = {0,0,0};
    const float depth = depth_map.Get(row, col);
    // float normal[3];
    // normal_map.GetSlice(row, col, normal);

    if (depth < 1e-6){
      depth_map.Set(row, col, 0);
      // normal_map.SetSlice(row, col, zeros);
      return;
    }

    for (size_t image_idx = 0; image_idx < num_src_image; ++image_idx) {
      if (ComputeGeomConsistencyCost(row, col, depth, image_idx,
          geom_consistency_max_cost, src_depth_maps_texture, poses_texture, 
          thread_index) <= filter_geom_consistency_max_cost){
        depth_map.Set(row, col, 0);
        // normal_map.SetSlice(row, col, zeros);
        return;
      }
    }
  }
}

__forceinline__ __device__ float ComputeCrossConsistencyCost(const float row,
                                                   const float col,
                                                   const float depth,
                                                   const int image_idx,
                                                   const int pose_idx,
                                                   const float max_cost,
                                                   cudaTextureObject_t src_depth_maps_texture,
                                                   cudaTextureObject_t poses_texture,
                                                   const int thread_index) {
  // Extract projection matrices for source image.
  // printf("ComputeCross: %f, %f, %f, %d, %d \n", row, col, depth, image_idx, pose_idx);
  float P[12];
  for (int i = 0; i < 12; ++i) {
    // printf("%d, %d", i, pose_idx);
    P[i] = tex2D<float>(poses_texture, i + 19, pose_idx);
    // printf("%f  ", P[i]);
  }
  float inv_P[12];
  for (int i = 0; i < 12; ++i) {
    inv_P[i] = tex2D<float>(poses_texture, i + 31, pose_idx);
  }

  // Project point in reference image to world.
  float forward_point[3];
  ComputePointAtDepth(row, col, depth, forward_point, thread_index);

  // Project world point to source image.
  const float inv_forward_z =
      1.0f / (P[8] * forward_point[0] + P[9] * forward_point[1] +
              P[10] * forward_point[2] + P[11]);
  float src_col =
      inv_forward_z * (P[0] * forward_point[0] + P[1] * forward_point[1] +
                       P[2] * forward_point[2] + P[3]);
  float src_row =
      inv_forward_z * (P[4] * forward_point[0] + P[5] * forward_point[1] +
                       P[6] * forward_point[2] + P[7]);

  // Extract depth in source image.
  const float src_depth = tex2DLayered<float>(src_depth_maps_texture, src_col + 0.5f,
                                       src_row + 0.5f, image_idx);

  // Projection outside of source image.
  if (src_depth == 0.0f) {
    return max_cost;
  }

  // Project point in source image to world.
  src_col *= src_depth;
  src_row *= src_depth;
  const float backward_point_x =
      inv_P[0] * src_col + inv_P[1] * src_row + inv_P[2] * src_depth + inv_P[3];
  const float backward_point_y =
      inv_P[4] * src_col + inv_P[5] * src_row + inv_P[6] * src_depth + inv_P[7];
  const float backward_point_z = inv_P[8] * src_col + inv_P[9] * src_row +
                                 inv_P[10] * src_depth + inv_P[11];
  const float inv_backward_point_z = 1.0f / backward_point_z;

  // Project world point back to reference image.
  const float backward_col =
      inv_backward_point_z *
      (ref_K[thread_index * 4 + 0] * backward_point_x + ref_K[thread_index * 4 + 1] * backward_point_z);
  const float backward_row =
      inv_backward_point_z *
      (ref_K[thread_index * 4 + 2] * backward_point_y + ref_K[thread_index * 4 + 3] * backward_point_z);

  // Return truncated reprojection error between original observation and
  // the forward-backward projected observation.
  const float diff_col = col - backward_col;
  const float diff_row = row - backward_row;
  return min(max_cost, sqrt(diff_col * diff_col + diff_row * diff_row));
}

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
                                  const int thread_index) {
  const int row = blockDim.y * blockIdx.y + threadIdx.y;
  const int col = blockDim.x * blockIdx.x + threadIdx.x;
  if (col < ref_width && row < ref_height) {

    const float depth = tex2DLayered<float>(depth_maps_texture, col, row, images_idxs.list[0]);
    if (depth < 1e-6 || col < 3 || col > ref_width - 3 || 
        row < 3 || row > ref_height - 3){
      refine_depth_map.Set(row, col, 0.0f);
      return;
    }

    int num_good_src_depth(0);
    float depths[MAX_NUM_EXTEND_SRC_IMAGE];
    depths[num_good_src_depth] = depth;

    float ref_normals_x;
    float ref_normals_y;
    float ref_normals_z;
    const unsigned int normal_code = tex2DLayered<unsigned int>(normal_maps_texture, col, row, images_idxs.list[0]);
    {
      uint32_t iq1 = (normal_code >> 16);
      uint32_t iq2 = (normal_code & 0x0000ffff);
      float q1 = iq1 / decode_factor - 1;
      float q2 = iq2 / decode_factor - 1;
      float s = q1 * q1 + q2 * q2;
      ref_normals_x = 1.0 - 2.0 * s;
      ref_normals_y = 2 * q1 * sqrtf(1 - s);
      ref_normals_z = 2 * q2 * sqrtf(1 - s);
    }

    float max_view_ray_cos_rad = cosf(DEG2RAD(100));
    const float normal[3] = {ref_normals_x, ref_normals_y, ref_normals_z};
    float view_ray[3] = 
      {ref_inv_K[thread_index * 4 + 0] * col + ref_inv_K[thread_index * 4 + 1],
      ref_inv_K[thread_index * 4 + 2] * row + ref_inv_K[thread_index * 4 + 3], 1.0f};
    const float norm_view = sqrtf(DotProduct3(view_ray, view_ray));
    view_ray[0] /= norm_view;
    view_ray[1] /= norm_view;
    view_ray[2] /= norm_view;
    if (DotProduct3(normal, view_ray) >= max_view_ray_cos_rad){
      refine_depth_map.Set(row, col, 0.0f);
      return;
    }
    // Project point in reference image to world.
    float forward_point[3];
    ComputePointAtDepth(row, col, depth, forward_point, thread_index);

    float sqr_depth_diff_threshold = depth_diff_threshold * depth_diff_threshold;
    float max_cos_rad = cosf(DEG2RAD(max_normal_error));

    for (int i = 1; i <= num_src_image; ++i) {
      int image_idx = images_idxs.list[i];
      int pose_idx = i - 1;

      float P[12];
      for (int j = 0; j < 12; ++j) {
        P[j] = tex2D<float>(poses_texture, j + 19, pose_idx);
      }
      float inv_P[12];
      for (int j = 0; j < 12; ++j) {
        inv_P[j] = tex2D<float>(poses_texture, j + 31, pose_idx);
      }

      // Project world point to source image.
      const float inv_forward_z =
          1.0f / (P[8] * forward_point[0] + P[9] * forward_point[1] +
                  P[10] * forward_point[2] + P[11]);
      float src_col =
          inv_forward_z * (P[0] * forward_point[0] + P[1] * forward_point[1] +
                          P[2] * forward_point[2] + P[3]);
      float src_row =
          inv_forward_z * (P[4] * forward_point[0] + P[5] * forward_point[1] +
                          P[6] * forward_point[2] + P[7]);

      // Extract depth in source image.
      const float src_depth = tex2DLayered<float>(depth_maps_texture, 
                              src_col + 0.5f, src_row + 0.5f, image_idx);
      const unsigned int src_normal_code = tex2DLayered<unsigned int>(
          normal_maps_texture, src_col + 0.5f, src_row + 0.5f, image_idx);

      // Projection outside of source image.
      if (src_depth < 1e-4) {
        continue;
      }

      // float ori_src_col = src_col;
      // float ori_src_row = src_row;

      // Project point in source image to world.
      src_col *= src_depth;
      src_row *= src_depth;
      const float backward_point_x =
          inv_P[0] * src_col + inv_P[1] * src_row + inv_P[2] * src_depth + inv_P[3];
      const float backward_point_y =
          inv_P[4] * src_col + inv_P[5] * src_row + inv_P[6] * src_depth + inv_P[7];
      const float backward_point_z = inv_P[8] * src_col + inv_P[9] * src_row +
                                    inv_P[10] * src_depth + inv_P[11];
      const float inv_backward_point_z = 1.0f / backward_point_z;

      // Project world point back to reference image.
      const float backward_col = inv_backward_point_z *
          (ref_K[thread_index * 4 + 0] * backward_point_x + ref_K[thread_index * 4 + 1] * backward_point_z);
      const float backward_row = inv_backward_point_z *
          (ref_K[thread_index * 4 + 2] * backward_point_y + ref_K[thread_index * 4 + 3] * backward_point_z);

      // Return truncated reprojection error between original observation and
      // the forward-backward projected observation.
      const float diff_col = col - backward_col;
      const float diff_row = row - backward_row;
      float reproj_error = sqrtf(diff_col * diff_col + diff_row * diff_row);

      if (reproj_error > filter_geom_consistency_max_cost || 
          (depth - backward_point_z) * (depth - backward_point_z) 
          > depth * depth * sqr_depth_diff_threshold) {
        continue;
      }

      float src_normal[3], rot_src_normal[3];
      {
        uint32_t iq1 = (src_normal_code >> 16);
        uint32_t iq2 = (src_normal_code & 0x0000ffff);
        float q1 = iq1 / decode_factor - 1;
        float q2 = iq2 / decode_factor - 1;
        float s = q1 * q1 + q2 * q2;
        src_normal[0] = 1.0 - 2.0 * s;
        src_normal[1] = 2 * q1 * sqrtf(1 - s);
        src_normal[2] = 2 * q2 * sqrtf(1 - s);
      }      
      float R[9];
      for (int j = 0; j < 9; ++j){
        R[j] = tex2D<float>(poses_texture, j + 4, pose_idx);
      }
      rot_src_normal[0] = R[0] * src_normal[0] + 
        R[3] * src_normal[1] + R[6] * src_normal[2];
      rot_src_normal[1] = R[1] * src_normal[0] + 
        R[4] * src_normal[1] + R[7] * src_normal[2];
      rot_src_normal[2] = R[2] * src_normal[0] + 
        R[5] * src_normal[1] + R[8] * src_normal[2];

      float ori_cos_phi = ref_normals_x * rot_src_normal[0] +
                          ref_normals_y * rot_src_normal[1] +
                          ref_normals_z * rot_src_normal[2];
      if (ori_cos_phi < max_cos_rad) {
        continue;
      }
#if 0
      // reproject back points to src image
      const float reprj_depth = tex2DLayered<float>(depth_maps_texture, 
         backward_col+0.5, backward_row+0.5, images_idxs.list[0]);
      float reprj_forward_point[3];
      ComputePointAtDepth(backward_row, backward_col, reprj_depth, reprj_forward_point, thread_index);
      const float reprj_inv_forward_z =
          1.0f / (P[8] * reprj_forward_point[0] + P[9] * reprj_forward_point[1] +
                  P[10] * reprj_forward_point[2] + P[11]);
      float reprj_src_col =
          reprj_inv_forward_z * (P[0] * reprj_forward_point[0] + P[1] * reprj_forward_point[1] +
                          P[2] * reprj_forward_point[2] + P[3]);
      float reprj_src_row =
          reprj_inv_forward_z * (P[4] * reprj_forward_point[0] + P[5] * reprj_forward_point[1] +
                          P[6] * reprj_forward_point[2] + P[7]);
      const float reprj_diff_col = ori_src_col - reprj_src_col;
      const float reprj_diff_row = ori_src_row - reprj_src_row;
      float rereproj_error = sqrtf(reprj_diff_col * reprj_diff_col + reprj_diff_row * reprj_diff_row);
      if (rereproj_error > filter_geom_consistency_max_cost){
        continue;
      }

      float reprj_normals_x;
      float reprj_normals_y;
      float reprj_normals_z;
      const unsigned int reprj_normal_code = tex2DLayered<unsigned int>(
        normal_maps_texture, backward_col+0.5, backward_row+0.5, images_idxs.list[0]);
      {
        uint32_t iq1 = (reprj_normal_code >> 16);
        uint32_t iq2 = (reprj_normal_code & 0x0000ffff);
        float q1 = iq1 / decode_factor - 1;
        float q2 = iq2 / decode_factor - 1;
        float s = q1 * q1 + q2 * q2;
        reprj_normals_x = 1.0 - 2.0 * s;
        reprj_normals_y = 2 * q1 * sqrtf(1 - s);
        reprj_normals_z = 2 * q2 * sqrtf(1 - s);
      }
      float reprj_ori_cos_phi = reprj_normals_x * rot_src_normal[0] +
                          reprj_normals_y * rot_src_normal[1] +
                          reprj_normals_z * rot_src_normal[2];
      if (reprj_ori_cos_phi < max_cos_rad) {
        continue;
      }
#endif

      num_good_src_depth++;
      depths[num_good_src_depth] = backward_point_z;
    }
    
    int nth = num_good_src_depth * 0.5;
    if (num_good_src_depth >= filter_min_num_consistent){
      BubbleSort(depths, num_good_src_depth);
      refine_depth_map.Set(row, col, depths[nth]);
      
    } else {
      refine_depth_map.Set(row, col, 0.0f);
    }
  }
}

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
                                     const int thread_index) {
  const int row = blockDim.y * blockIdx.y + threadIdx.y;
  const int col = blockDim.x * blockIdx.x + threadIdx.x;
  if (col < ref_width && row < ref_height) {

    const float depth = tex2DLayered<float>(depth_maps_texture, col, row, images_idxs.list[0].id);
    if (depth < 1e-6 || col < 3 || col > ref_width - 3 || row < 3 || row > ref_height - 3){
      visbility_maps.Set(row, col, images_idxs.list[0].id,  0x00000000);
      return;
    }

    uint32_t visbility_value = visbility_maps.Get(row, col, images_idxs.list[0].id);
    if (!bool(visbility_value >> (sizeof(uint32_t) - 1))){
      visbility_maps.Set(row, col, images_idxs.list[0].id,  0x00000000);
      return;
    }

    uint8_t num_vis_map(0);
    int img_idx[MAX_NUM_EXTEND_SRC_IMAGE];
    img_idx[0] = images_idxs.list[0].id;

    int num_good_src_depth(0);
    float delt_depths[MAX_NUM_EXTEND_SRC_IMAGE];
    // delt_depths[num_good_src_depth] = depth;

    // Project point in reference image to world.
    float forward_point[3];
    ComputePointAtDepth(row, col, depth, forward_point, thread_index);

    float sqr_depth_diff_threshold = depth_diff_threshold * depth_diff_threshold;
    for (int i = 1; i <= num_src_image; ++i) {
      int image_idx = images_idxs.list[i].id;
      int pose_idx = i - 1;

      float P[12];
      for (int j = 0; j < 12; ++j) {
        P[j] = tex2D<float>(poses_texture, j + 19, pose_idx);
      }
      float inv_P[12];
      for (int j = 0; j < 12; ++j) {
        inv_P[j] = tex2D<float>(poses_texture, j + 31, pose_idx);
      }

      // Project world point to source image.
      const float inv_forward_z =
          1.0f / (P[8] * forward_point[0] + P[9] * forward_point[1] +
                  P[10] * forward_point[2] + P[11]);
      float src_col =
          inv_forward_z * (P[0] * forward_point[0] + P[1] * forward_point[1] +
                          P[2] * forward_point[2] + P[3]);
      float src_row =
          inv_forward_z * (P[4] * forward_point[0] + P[5] * forward_point[1] +
                          P[6] * forward_point[2] + P[7]);

      // Extract depth in source image.
      const float src_depth = tex2DLayered<float>(depth_maps_texture, 
                              src_col + 0.5f, src_row + 0.5f, image_idx);

      // Projection outside of source image.
      if (src_depth == 0.0f) {
        continue;
      }

      float ori_src_col = src_col;
      float ori_src_row = src_row;

      // Project point in source image to world.
      src_col *= src_depth;
      src_row *= src_depth;
      const float backward_point_x =
          inv_P[0] * src_col + inv_P[1] * src_row + inv_P[2] * src_depth + inv_P[3];
      const float backward_point_y =
          inv_P[4] * src_col + inv_P[5] * src_row + inv_P[6] * src_depth + inv_P[7];
      const float backward_point_z = inv_P[8] * src_col + inv_P[9] * src_row +
                                    inv_P[10] * src_depth + inv_P[11];
      const float inv_backward_point_z = 1.0f / backward_point_z;

      // Project world point back to reference image.
      const float backward_col = inv_backward_point_z *
          (ref_K[thread_index * 4 + 0] * backward_point_x + ref_K[thread_index * 4 + 1] * backward_point_z);
      const float backward_row = inv_backward_point_z *
          (ref_K[thread_index * 4 + 2] * backward_point_y + ref_K[thread_index * 4 + 3] * backward_point_z);

      // Return truncated reprojection error between original observation and
      // the forward-backward projected observation.
      const float diff_col = col - backward_col;
      const float diff_row = row - backward_row;
      float reproj_error = sqrtf(diff_col * diff_col + diff_row * diff_row);

      if (reproj_error > filter_geom_consistency_max_cost ||
          (depth - backward_point_z) * (depth - backward_point_z) 
          > depth * depth * sqr_depth_diff_threshold){
        continue;
      }

      num_good_src_depth++;
      delt_depths[num_good_src_depth - 1] = fabs(backward_point_z - depth);
      img_idx[num_good_src_depth] = i - 1;

      uint32_t src1_visbility_value = visbility_maps.Get(ori_src_row + 0.5f, ori_src_col + 0.5f, image_idx);
      if (src1_visbility_value & 0x80000000 && images_idxs.list[i].flag){
        num_vis_map++;
      }
    }

    if (num_good_src_depth < filter_min_num_consistent || num_vis_map > 0){
      visbility_maps.Set(row, col, images_idxs.list[0].id, 0x00000000);
      return;
    }

    uint32_t bit_value = visbility_maps.Get(row, col, img_idx[0]);
    // float mean_delt_depth = 0;
    for (int  i = 1; i <= num_good_src_depth; i++){
      // mean_delt_depth += delt_depths[i];
      bit_value = bit_value | (uint32_t)(1 << (MAX_NUM_CROSS_SRC - 1 - img_idx[i]));
    }
    // float mean_error = mean_delt_depth / num_good_src_depth;
    // mean_delt_depth = mean_error / delt_depths[0];
    // delt_depth_map.Set(row, col, mean_delt_depth);
    visbility_maps.Set(row, col, img_idx[0], bit_value);

    int nth = (num_good_src_depth - 1) * 0.5;
    BubbleSort(delt_depths, num_good_src_depth - 1);
    delt_depth_map.Set(row, col, delt_depths[nth] / (depth_diff_threshold * depth));
  }
}

__host__ void SetRefK(const float * ref_K_host, int thread_index) {
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(ref_K, ref_K_host, 
        sizeof(float) * 4, sizeof(float) * 4 * thread_index,
        cudaMemcpyHostToDevice));
}

__host__ void SetRefInvK(const float * ref_inv_K_host, int thread_index) {
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(ref_inv_K, ref_inv_K_host, 
        sizeof(float) * 4, sizeof(float) * 4 * thread_index,
        cudaMemcpyHostToDevice));
}

__global__ void FillGpuMat(const int max_width, const int max_height,
  GpuMat<uint32_t> visbility_maps) {
  const int row = blockDim.y * blockIdx.y + threadIdx.y;
  const int col = blockDim.x * blockIdx.x + threadIdx.x;

  if (col < max_width && row < max_height){
    int num_depth = visbility_maps.GetDepth();
    for (int i = 0; i < num_depth; i++){
      visbility_maps.Set(row, col, i, 0x80000000);
    }
  }
}

__global__ void GetDeduplicDepthMapsLayer(
  const GpuMat<uint32_t> visbility_maps,
  const int image_id,
  const int width,
  const int height,
  GpuMat<float > depth_map,
  cudaTextureObject_t depth_maps_texture){
  const int row = blockDim.y * blockIdx.y + threadIdx.y;
  const int col = blockDim.x * blockIdx.x + threadIdx.x;
  if (col < width && row < height){
    const uint32_t vis = visbility_maps.Get(row, col, image_id);
    if (vis & 0x80000000){
      const float depth =  tex2DLayered<float>(
        depth_maps_texture, col, row, image_id);
      depth_map.Set(row, col, depth);
    }
  }
}

__global__ void GetVisbilityMapsLayer(
  const GpuMat<uint32_t> visbility_maps,
  const int image_cross_id,
  const int width,
  const int height,
  GpuMat<uint32_t> visbility_map){
  const int row = blockDim.y * blockIdx.y + threadIdx.y;
  const int col = blockDim.x * blockIdx.x + threadIdx.x;
  if (col < width && row < height){
    const uint32_t vis = visbility_maps.Get(row, col, image_cross_id);
    visbility_map.Set(row, col, vis);
  }
}

} // namespace mvs
} // namespace sensemap