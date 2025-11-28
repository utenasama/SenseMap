//Copyright (c) 2022, SenseTime Group.
//All rights reserved.

#ifndef SENSEMAP_UTIL_COMMON_H_
#define SENSEMAP_UTIL_COMMON_H_

#include <vector>
#include <Eigen/Dense>

namespace sensemap
{

Eigen::Matrix3f ComputePovitMatrix(const std::vector<Eigen::Vector3f> &points);

float ComputeAvergeSapcing(std::vector<float>& point_spacings, 
                           const std::vector<Eigen::Vector3f> &points_sparse,
                           const unsigned int nb_neighbors = 6);

}  // namespace sensemap

#endif  // SENSEMAP_UTIL_COMMON_H_