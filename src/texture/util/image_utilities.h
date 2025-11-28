//Copyright (c) 2019, SenseTime Group.
//All rights reserved.

#ifndef SENSEMAP_TEXTURE_UTIL_IMAGE_UTILITIES_H_
#define SENSEMAP_TEXTURE_UTIL_IMAGE_UTILITIES_H_

#include <opencv2/opencv.hpp>

namespace sensemap {
namespace texture {

class ImageUtilities {
public:
    ImageUtilities() {};
public:
    static bool TestImageBoundary(
            const cv::Mat &image,
            double u, double v,
            double inner_margin = 0.0);

    static std::pair<bool, double> FloatValue(
            const cv::Mat &image, double u, double v);
};

} // namespace sensemap
} // namespace texture
#endif //SENSEMAP_TEXTURE_UTIL_IMAGE_UTILITIES_H_
