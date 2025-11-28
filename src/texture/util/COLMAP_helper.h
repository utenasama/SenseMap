//Copyright (c) 2019, SenseTime Group.
//All rights reserved.

#ifndef SENSEMAP_TEXTURE_UTIL_COLMAP_HELPER_H_
#define SENSEMAP_TEXTURE_UTIL_COLMAP_HELPER_H_

#include <opencv2/opencv.hpp>
#include <Eigen/Eigen>

#include "../../util/endian.h"

namespace sensemap {
namespace texture {


void ReadDepthFromCOLMAP(const std::string &path, cv::Mat &depth_map);


bool ReadIntrinsicMatrixFromCOLMAPBinary(const std::string &filename,
                                         std::map<uint32_t, Eigen::Matrix3d> &intrinsic_map);


bool ReadExtrinsicMatrixFromCOLMAPBinary(const std::string &filename,
                                         std::vector<Eigen::Matrix4d> &poses,
                                         std::vector<std::string> &names,
                                         std::vector<uint32_t> &camera_ids);


} // namespace sensemap
} // namespace texture


#endif //SENSEMAP_TEXTURE_UTIL_COLMAP_HELPER_H_

