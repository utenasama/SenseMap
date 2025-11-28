//Copyright (c) 2019, SenseTime Group.
//All rights reserved.

#ifndef SENSEMAP_SRC_UTIL_CUDA_H_
#define SENSEMAP_SRC_UTIL_CUDA_H_

#include <cuda_runtime.h>

namespace sensemap {

int GetNumCudaDevices();

void SetBestCudaDevice(const int gpu_index);

void GetDeviceProp(const int gpu_index, cudaDeviceProp &device);

int ConvertSMVer2Cores(int major, int minor);

}  // namespace sensemap

#endif  // SENSEMAP_SRC_UTIL_CUDA_H_
