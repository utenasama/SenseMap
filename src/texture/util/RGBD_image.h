//Copyright (c) 2019, SenseTime Group.
//All rights reserved.

#ifndef SENSEMAP_TEXTURE_UTIL_RGBD_IMAGE_H_
#define SENSEMAP_TEXTURE_UTIL_RGBD_IMAGE_H_

#include <opencv2/opencv.hpp>
#include <utility>

namespace sensemap {
namespace texture {

class RGBDImage {
public:
    RGBDImage() {};

    RGBDImage(cv::Mat &color, cv::Mat &depth) :
            color_(std::move(color)), depth_(std::move(depth)) {};

    ~RGBDImage() {
        color_.release();
        depth_.release();
    };

public:
    cv::Mat color_;
    cv::Mat depth_;
};

} // namespace sensemap
} // namespace texture
#endif //SENSEMAP_TEXTURE_UTIL_RGBD_IMAGE_H_
