//Copyright (c) 2020, SenseTime Group.
//All rights reserved.

#include <cuda_runtime.h>

#include "util/gpu_mat.h"
#include "mvs/cuda_array_wrapper.h"
#include "mvs/utils.h"
#include "mvs/kernel_functions.h"

#include "point_cloud_filter_cuda.h"

namespace sensemap {
namespace mvs {

void ComputeDistanceMap(const Bitmap& bitmap, Mat<float>& dist_map,
                        const float min_grad_thres) {
    
    int width = bitmap.Width();
    int height = bitmap.Height();

    // Upload to device.
    std::unique_ptr<CudaArrayWrapper<uint8_t> > ref_image_device;
    ref_image_device.reset(new CudaArrayWrapper<uint8_t>(width, height, 1));

    auto ref_image_array = bitmap.ConvertToRowMajorArray();
    ref_image_device->CopyToDevice(ref_image_array.data());

    // Create texture.
    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = ref_image_device->GetPtr();

    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeBorder;
    texDesc.addressMode[1] = cudaAddressModeBorder;
    texDesc.addressMode[2] = cudaAddressModeBorder;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.readMode = cudaReadModeNormalizedFloat;
    texDesc.normalizedCoords = 0;

    cudaTextureObject_t ref_image_texture = 0;
    CUDA_SAFE_CALL(
        cudaCreateTextureObject(&ref_image_texture, &resDesc, &texDesc, NULL));

    std::unique_ptr<GpuMat<float> > grad_map_device;
    grad_map_device.reset(new GpuMat<float>(width, height));
    grad_map_device->FillWithScalar(0);

    dim3 elem_wise_block_size;
    dim3 elem_wise_grid_size;
    elem_wise_block_size.x = THREADS_PER_BLOCK;
    elem_wise_block_size.y = THREADS_PER_BLOCK;
    elem_wise_block_size.z = 1;
    elem_wise_grid_size.x = (width - 1) / THREADS_PER_BLOCK + 1;
    elem_wise_grid_size.y = (height - 1) / THREADS_PER_BLOCK + 1;
    elem_wise_grid_size.z = 1;

    ComputeGradientMap<<<elem_wise_grid_size, elem_wise_block_size>>>(*grad_map_device, ref_image_texture);
    CUDA_SYNC_AND_CHECK();

    std::unique_ptr<GpuMat<float> > dist_map_device;
    dist_map_device.reset(new GpuMat<float>(width, height));
    InitializeDistMap<<<elem_wise_grid_size, elem_wise_block_size>>>(
      *grad_map_device, *dist_map_device, min_grad_thres);
    CUDA_SYNC_AND_CHECK();

    elem_wise_block_size.x = THREADS_PER_BLOCK;
    elem_wise_block_size.y = 1;
    elem_wise_block_size.z = 1;
    elem_wise_grid_size.x = (width - 1) / THREADS_PER_BLOCK + 1;
    elem_wise_grid_size.y = 1;
    elem_wise_grid_size.z = 1;
    ComputeRowDistMap<<<elem_wise_grid_size, elem_wise_block_size>>>(*dist_map_device);
    CUDA_SYNC_AND_CHECK();

    elem_wise_grid_size.x = (height - 1) / THREADS_PER_BLOCK + 1;
    elem_wise_grid_size.y = 1;
    elem_wise_grid_size.z = 1;
    ComputeColDistMap<<<elem_wise_grid_size, elem_wise_block_size>>>(*dist_map_device);
    CUDA_SYNC_AND_CHECK();

    dist_map = dist_map_device->CopyToMat();

    CUDA_SAFE_CALL(cudaDestroyTextureObject(ref_image_texture));
}

} // namespace mvs
} // namespace sensemap