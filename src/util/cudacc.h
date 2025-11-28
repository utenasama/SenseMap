//Copyright (c) 2019, SenseTime Group.
//All rights reserved.

#ifndef SENSEMAP_SRC_UTIL_CUDACC_H_
#define SENSEMAP_SRC_UTIL_CUDACC_H_

#include <string>
#include <cuda_runtime.h>

#define CUDA_SAFE_CALL(error) CudaSafeCall(error, __FILE__, __LINE__)
#define CUDA_CHECK() CudaCheck(__FILE__, __LINE__)
#define CUDA_SYNC_AND_CHECK() CudaSyncAndCheck(__FILE__, __LINE__)

namespace sensemap {

class CudaTimer {
public:
      CudaTimer();
      ~CudaTimer();

      void Print(const std::string& message);

private:
      cudaEvent_t start_;
      cudaEvent_t stop_;
      float elapsed_time_;
};

void CudaSafeCall(const cudaError_t error, const std::string& file,
                  const int line);

void CudaCheck(const char* file, const int line);
void CudaSyncAndCheck(const char* file, const int line);

}  // namespace sensemap

#endif  // SENSEMAP_SRC_UTIL_CUDACC_H_
