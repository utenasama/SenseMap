//Copyright (c) 2020, SenseTime Group.
//All rights reserved.

#ifndef SENSEMAP_MVS_POINT_CLOUD_FILTER_CUDA_H_
#define SENSEMAP_MVS_POINT_CLOUD_FILTER_CUDA_H_

// #include "util/gpu_mat.h"
#include "util/mat.h"
#include "util/bitmap.h"

namespace sensemap {
namespace mvs {

void ComputeDistanceMap(const Bitmap& bitmap, Mat<float>& grad_map,
                        const float min_grad_thres);

} // namespace mvs
} // namespace sensemap

#endif