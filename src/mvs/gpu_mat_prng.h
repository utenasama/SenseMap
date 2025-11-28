//Copyright (c) 2019, SenseTime Group.
//All rights reserved.

#ifndef SENSEMAP_UTIL_GPU_MAT_PRNG_H_
#define SENSEMAP_UTIL_GPU_MAT_PRNG_H_

#include "util/gpu_mat.h"

namespace sensemap {
namespace mvs {

class GpuMatPRNG : public GpuMat<curandState> {
 public:
  GpuMatPRNG(const int width, const int height);
  GpuMatPRNG(const int width, const int height, unsigned long long seed);

 private:
  void InitRandomState();
};

}  // namespace mvs
}  // namespace sensemap

#endif  // SENSEMAP_UTIL_GPU_MAT_PRNG_H_
