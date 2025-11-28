//Copyright (c) 2019, SenseTime Group.
//All rights reserved.

#ifndef SENSEMAP_MVS_UTILS_H_
#define SENSEMAP_MVS_UTILS_H_

#define _USE_MATH_DEFINES

#include <map>
#include <vector>
#include <cuda_runtime.h>

#include "util/types.h"
#include "util/logging.h"
#include "base/reconstruction.h"

#define WEIGHTED_DEFAULT 0
#define WEIGHTED_ACMH    1
#define VIEW_WEIGHTED_METHOD WEIGHTED_ACMH

#define DENSE_SMOOTHNESS_NA 0
#define DENSE_SMOOTHNESS_PLANE 1
#define DENSE_SMOOTHNESS DENSE_SMOOTHNESS_NA
// #define DENSE_SMOOTHNESS DENSE_SMOOTHNESS_PLANE

#define DENSE_SHARPNESS_NA 0
#define DENSE_SHARPNESS_GRAD 1
#define DENSE_SHARPNESS DENSE_SHARPNESS_NA

// #ifndef THREADS_PER_GPU
//     #define THREADS_PER_GPU 1
// #endif
#define MAX_THREADS_PER_GPU 3
// #define ACMMPM_SHMEM_CUDA
#define COLMAP_BLOCK_SWEEP

#define FUSED_PC_MEMORY

#define MAX_NUM_CROSS_SRC 31
#define PLY_INIT_DEPTH_LEVEL 2

// #define DEBUG_MODEL_PM

namespace sensemap {
namespace mvs {
namespace utility {

struct NeighborEstimate {
    float w = 1.0f;
    float depth = -1.f;
    float normal[3];
    float x[3];
};

struct NeighborData {
    int r, c;
    float depth = -1.f;
    float normal[3];
    float conf = FLT_MAX;
};

struct Planef {
    float m_fD;
    float m_vN[3];
};

struct List{
  int id;
  bool flag;
};

struct BilateralWeightComputer {
  __host__ __device__ BilateralWeightComputer(const float sigma_spatial,
                                              const float sigma_color)
      : spatial_normalization_(1.0f / (2.0f * sigma_spatial * sigma_spatial)),
        color_normalization_(1.0f / (2.0f * sigma_color * sigma_color)) {}

  __host__ __device__ inline float Compute(const float row_diff, 
                                           const float col_diff,
                                           const float color1, 
                                           const float color2) const {
    const float spatial_dist_squared =
        row_diff * row_diff + col_diff * col_diff;
    const float color_dist = color1 - color2;
    return expf(-spatial_dist_squared * spatial_normalization_ -
               color_dist * color_dist * color_normalization_);
  }

 private:
  const float spatial_normalization_;
  const float color_normalization_;
};

__device__
inline void Normal2Dir(const float d[3], float& x, float& y) {
    y = atan2f(sqrtf(d[0] * d[0] + d[1] * d[1]), d[2]);
    x = atan2f(d[1], d[0]);
}

__device__
inline void Dir2Normal(const float x, const float y, float d[3]) {
    float siny = __sinf(y);
    d[0] = __cosf(x) * siny;
    d[1] = __sinf(x) * siny;
    d[2] = __cosf(y);
}

// encode/decode NCC score and refinement level in one float
__device__ inline float EncodeScoreScale(
    float score, unsigned inv_scale_range = 0) {
    return score * 0.1f + (float)inv_scale_range;
}

__device__ inline unsigned DecodeScoreScale(float& score) {
    const unsigned inv_scale_range = (unsigned)score;
    score = (score - (float)inv_scale_range) * 10.f;
    return inv_scale_range;
}

__device__ inline float EncodeCurvatureGradient(float curvature, 
                                                         float grad) {
    // return curvature * 0.1 + int((grad + 1) * 1000);
    return grad * 0.01 + int((curvature / M_PI * 180) * 10);
}

__device__ inline float DecodeCurvatureGradient(float& curvature) {
    // int encode_grad = int(curvature);
    // curvature = (curvature - encode_grad) * 10;
    // return encode_grad / 1000.0f - 1;
    int encode_angle = int(curvature);
    float grad = (curvature - encode_angle) * 100;
    curvature = encode_angle * M_PI / 1800;
    return grad;
}

template<typename T>
__device__ inline void Swap(T& a, T& b) {
    T c(a);
    a = b;
    b = c;
}

template<typename T>
__device__ void QuickSort(T *vals, const int low, const int high, 
                                   bool ascend = true) {
    int i = low;
    int j = high;
    int pivot = vals[(i + j) / 2];

    while (i <= j) {
        if (ascend) {
            while (vals[i] < pivot) i++;
        } else {
            while (vals[i] > pivot) i++;
        }
        if (ascend) {
            while (vals[j] > pivot) j--;
        } else {
            while (vals[j] < pivot) j--;
        }
        if (i <= j) {
            Swap(vals[i], vals[j]);
            i++;
            j--;
        }
    }
    if (j > low)
        QuickSort(vals, low, j);
    if (i < high)
        QuickSort(vals, i, high);
}

template<typename KEY_TYPE, typename VAL_TYPE>
__device__ void QuickSortByValue(KEY_TYPE *keys, VAL_TYPE* vals, 
                                          const int low, const int high, 
                                          bool ascend = true) {
    int i = low;
    int j = high;
    int pivot = vals[(i + j) / 2];

    while (i <= j) {
        if (ascend) {
            while (vals[i] < pivot) i++;
        } else {
            while (vals[i] > pivot) i++;
        }
        if (ascend) {
            while (vals[j] > pivot) j--;
        } else {
            while (vals[j] < pivot) j--;
        }
        if (i <= j) {
            Swap(keys[i], keys[j]);
            Swap(vals[i], vals[j]);
            i++;
            j--;
        }
    }
    if (j > low)
        QuickSortByValue(keys, vals, low, j);
    if (i < high)
        QuickSortByValue(keys, vals, i, high);
}

template<typename T>
__device__ void BubbleSort(T *vals, int length, bool ascend = true) {
    int i, j;
    for (i = 0; i < length - 1; ++i) {
        for (j = 0; j < length - 1 - i; ++j) {
            if (ascend) {
                if (vals[j] > vals[j + 1]) {
                    Swap(vals[j], vals[j + 1]);
                }
            } else {
                if (vals[j] < vals[j + 1]) {
                    Swap(vals[j], vals[j + 1]);
                }
            }
        }
    }
}

template<typename KEY_TYPE, typename VAL_TYPE>
__device__ void BubbleSortByValue(KEY_TYPE *keys, VAL_TYPE *vals, 
                                           int length, bool ascend = true) {
    int i, j;
    for (i = 0; i < length - 1; ++i) {
        for (j = 0; j < length - 1 - i; ++j) {
            if (ascend) {
                if (vals[j] > vals[j + 1]) {
                    Swap(vals[j], vals[j + 1]);
                    Swap(keys[j], keys[j + 1]);
                }
            } else {
                if (vals[j] < vals[j + 1]) {
                    Swap(vals[j], vals[j + 1]);
                    Swap(keys[j], keys[j + 1]);
                }
            }
        }
    }
}

__device__ inline void Mat33DotVec3(const float mat[9], 
                                             const float vec[3],
                                             float result[3]) {
  result[0] = mat[0] * vec[0] + mat[1] * vec[1] + mat[2] * vec[2];
  result[1] = mat[3] * vec[0] + mat[4] * vec[1] + mat[5] * vec[2];
  result[2] = mat[6] * vec[0] + mat[7] * vec[1] + mat[8] * vec[2];
}

__device__ inline void Mat33DotVec3Homogeneous(const float mat[9],
                                                        const float vec[2],
                                                        float result[2]) {
  const float inv_z = 1.0f / (mat[6] * vec[0] + mat[7] * vec[1] + mat[8]);
  result[0] = inv_z * (mat[0] * vec[0] + mat[1] * vec[1] + mat[2]);
  result[1] = inv_z * (mat[3] * vec[0] + mat[4] * vec[1] + mat[5]);
}

__device__ inline float DotProduct3(const float vec1[3], 
                                             const float vec2[3]) {
  return vec1[0] * vec2[0] + vec1[1] * vec2[1] + vec1[2] * vec2[2];
}

__device__ inline void CrossProduct3(const float u[3], const float v[3], float uv[3]) {
    uv[0] = u[1] * v[2] - u[2] * v[1];
    uv[1] = u[2] * v[0] - u[0] * v[2];
    uv[2] = u[0] * v[1] - u[1] * v[0];
}


void ComputeRelativePose(const float R1[9], const float T1[3],
                         const float R2[9], const float T2[3], float R[9],
                         float T[3]);

void ComputeProjectionCenter(const float R[9], const float T[3], float C[3]);

void ComposeProjectionMatrix(const float K[9], const float R[9],
                             const float T[3], float P[12]);

void ComposeInverseProjectionMatrix(const float K[9], const float R[9],
                                    const float T[3], float inv_P[12]);

void RotatePose(const float RR[9], float R[9], float T[3]);

float Footprint(const float K[9], const float R[9], const float t[3],
                const float X[3]);

// given an array of values and their bound, approximate the area covered, 
// in percentage.
template<typename TYPE, int n, int s, bool bCentered>
inline TYPE ComputeCoveredArea(const TYPE* values, size_t size, 
                               const TYPE* bound, int stride=n) {
    // ASSERT(size > 0);
    typedef Eigen::Matrix<TYPE,1,n,Eigen::RowMajor> Vector;
    typedef Eigen::Map<const Vector,Eigen::Aligned> MapVector;
    typedef Eigen::Matrix<TYPE,Eigen::Dynamic,n,Eigen::RowMajor> Matrix;
    typedef Eigen::Map<const Matrix,Eigen::Aligned,Eigen::OuterStride<> > 
        MapMatrix;
    typedef Eigen::Matrix<unsigned,s + 1,s + 1,Eigen::RowMajor> MatrixSurface;
    const MapMatrix points(values, size, n, Eigen::OuterStride<>(stride));
    const Vector norm = MapVector(bound);
    const Vector offset(Vector::Constant(bCentered ? TYPE(0.5) : TYPE(0)));
    MatrixSurface surface;
    surface.setZero();
    for (size_t i=0; i<size; ++i) {
        const Vector point((points.row(i).cwiseQuotient(norm)+offset)*TYPE(s));
        // ASSERT((point(0)>=0 && point(0)<s) && (point(1)>=0 && point(1)<s));
        surface((int)floor(point(0)), (int)floor(point(1))) = 1;
    }
    return TYPE(surface.sum())/((s + 1)*(s + 1));
} // ComputeCoveredArea

} // namespace utility
} // namespace mvs
} // namespace sensemap

#endif