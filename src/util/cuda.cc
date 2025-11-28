//Copyright (c) 2019, SenseTime Group.
//All rights reserved.
#include "util/cuda.h"

#include <algorithm>
#include <iostream>
#include <vector>

#include "util/logging.h"
#include "util/cudacc.h"

namespace sensemap {
namespace {

// Check whether the first Cuda device is better than the second.
bool CompareCudaDevice(const cudaDeviceProp& d1, const cudaDeviceProp& d2) {
    bool result = (d1.major > d2.major) ||
                  ((d1.major == d2.major) && (d1.minor > d2.minor)) ||
                  ((d1.major == d2.major) && (d1.minor == d2.minor) &&
                  (d1.multiProcessorCount > d2.multiProcessorCount));
    return result;
}

}  // namespace

int GetNumCudaDevices() {
    int num_cuda_devices;
    cudaError_t code = cudaGetDeviceCount(&num_cuda_devices);
    // std::cout << "code = " << code << std::endl;
    if (code != cudaSuccess) {
        std::cout << cudaGetErrorString(code) << std::endl;
    }
    return num_cuda_devices;
}

void SetBestCudaDevice(const int gpu_index) {
    const int num_cuda_devices = GetNumCudaDevices();
    CHECK_GT(num_cuda_devices, 0) << "No CUDA devices available";

    int selected_gpu_index = -1;
    if (gpu_index >= 0) {
      	selected_gpu_index = gpu_index;
    } 
    else {
      	std::vector<cudaDeviceProp> all_devices(num_cuda_devices);
      	for (int device_id = 0; device_id < num_cuda_devices; ++device_id) {
        	cudaGetDeviceProperties(&all_devices[device_id], device_id);
      	}
      	std::sort(all_devices.begin(), all_devices.end(), CompareCudaDevice);
      	CUDA_SAFE_CALL(cudaChooseDevice(&selected_gpu_index, 
		  								all_devices.data()));
    }

    CHECK_GE(selected_gpu_index, 0);
    CHECK_LT(selected_gpu_index, num_cuda_devices) 
                      << "Invalid CUDA GPU selected";

    cudaDeviceProp device;
    cudaGetDeviceProperties(&device, selected_gpu_index);
    CUDA_SAFE_CALL(cudaSetDevice(selected_gpu_index));
}

void GetDeviceProp(const int gpu_index, cudaDeviceProp &device) {
    CUDA_SAFE_CALL(cudaSetDevice(gpu_index));
    CUDA_SAFE_CALL(cudaGetDeviceProperties(&device, gpu_index));
    printf("GPU Device %d: \"%s\"\n", gpu_index, device.name);
}

// Beginning of GPU Architecture definitions
int ConvertSMVer2Cores(int major, int minor) {
// Defines for GPU Architecture types (using the SM version to determine
  // the # of cores per SM
  typedef struct {
    int SM;  // 0xMm (hexidecimal notation), M = SM Major version,
    // and m = SM minor version
    int Cores;
  } sSMtoCores;

  sSMtoCores nGpuArchCoresPerSM[] = {
      {0x30, 192},
      {0x32, 192},
      {0x35, 192},
      {0x37, 192},
      {0x50, 128},
      {0x52, 128},
      {0x53, 128},
      {0x60,  64},
      {0x61, 128},
      {0x62, 128},
      {0x70,  64},
      {0x72,  64},
      {0x75,  64},
      {0x80,  64},
      {0x86, 128},
      {0x87, 128},
      {-1, -1}};

  int index = 0;

  while (nGpuArchCoresPerSM[index].SM != -1) {
    if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor)) {
      return nGpuArchCoresPerSM[index].Cores;
    }

    index++;
  }

  // If we don't find the values, we default use the previous one
  // to run properly
  printf(
      "MapSMtoCores for SM %d.%d is undefined."
      "  Default to use %d Cores/SM\n",
      major, minor, nGpuArchCoresPerSM[index - 1].Cores);
  return nGpuArchCoresPerSM[index - 1].Cores;
}

}  // namespace sensemap
