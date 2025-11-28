//Copyright (c) 2019, SenseTime Group.
//All rights reserved.

#ifndef SENSEMAP_UTIL_CUDA_TRANSPOSE_H_
#define SENSEMAP_UTIL_CUDA_TRANSPOSE_H_

#include <cuda_runtime.h>

namespace sensemap {

// Transpose the input matrix.
template <typename T>
void CudaTranspose(const T* input, T* output, const int width, const int height,
                   const int pitch_input, const int pitch_output);

////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////

#ifdef __CUDACC__

// TILE_DIM_TRANSPOSE must divide by BLOCK_ROWS. Do not change these values.
#define TILE_DIM_TRANSPOSE 32
#define BLOCK_ROWS_TRANSPOSE 8

namespace internal {

template <typename T>
__global__ void CudaTransposeKernel(T* output_data, const T* input_data,
                                    const int width, const int height,
                                    const int input_pitch,
                                    const int output_pitch) {
  int x_index = blockIdx.x * TILE_DIM_TRANSPOSE + threadIdx.x;
  int y_index = blockIdx.y * TILE_DIM_TRANSPOSE + threadIdx.y;

  __shared__ T tile[TILE_DIM_TRANSPOSE][TILE_DIM_TRANSPOSE + 1];
  const int tile_x =
      min(threadIdx.x, width - 1 - blockIdx.x * TILE_DIM_TRANSPOSE);
  const int tile_y =
      min(threadIdx.y, height - 1 - blockIdx.y * TILE_DIM_TRANSPOSE);

  for (int i = 0; i < TILE_DIM_TRANSPOSE; i += BLOCK_ROWS_TRANSPOSE) {
    const int x = min(x_index, width - 1);
    const int y = min(y_index, height - i - 1);
    const int index = y * input_pitch + x + i * input_pitch;
    tile[tile_y + i][tile_x] = input_data[index];
  }

  __syncthreads();

  x_index = blockIdx.y * TILE_DIM_TRANSPOSE + threadIdx.x;
  if (x_index < height) {
    y_index = blockIdx.x * TILE_DIM_TRANSPOSE + threadIdx.y;
    const int index = x_index + y_index * output_pitch;
    for (int i = 0; i < TILE_DIM_TRANSPOSE; i += BLOCK_ROWS_TRANSPOSE) {
      if (y_index + i < width) {
        output_data[index + i * output_pitch] =
            tile[threadIdx.x][threadIdx.y + i];
      }
    }
  }
}

}  // namespace internal

template <typename T>
void CudaTranspose(const T* input, T* output, const int width, const int height,
                   const int pitch_input, const int pitch_output) {
  dim3 block_dim(TILE_DIM_TRANSPOSE, BLOCK_ROWS_TRANSPOSE, 1);
  dim3 grid_dim;
  grid_dim.x = (width - 1) / TILE_DIM_TRANSPOSE + 1;
  grid_dim.y = (height - 1) / TILE_DIM_TRANSPOSE + 1;

  internal::CudaTransposeKernel<<<grid_dim, block_dim>>>(
      output, input, width, height, pitch_input / sizeof(T),
      pitch_output / sizeof(T));
}

#undef TILE_DIM_TRANSPOSE
#undef BLOCK_ROWS_TRANSPOSE

#endif  // __CUDACC__

}  // namespace sensemap

#endif  // SENSEMAP_UTIL_CUDA_TRANSPOSE_H_
