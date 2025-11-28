//Copyright (c) 2019, SenseTime Group.
//All rights reserved.

#include "image_utilities.h"

namespace sensemap {
namespace texture {


bool ImageUtilities::TestImageBoundary(
        const cv::Mat &image, double u, double v, double inner_margin) {
    return (u >= inner_margin && u < image.cols - inner_margin &&
            v >= inner_margin && v < image.rows - inner_margin);
}

std::pair<bool, double>
ImageUtilities::FloatValue(const cv::Mat &image, double u, double v) {
    if ((u < 0.0 || u > (double) (image.cols - 1) ||
         v < 0.0 || v > (double) (image.rows - 1))) {
        return std::make_pair(false, 0.0);
    }
    int ui = std::max(std::min((int) u, image.cols - 2), 0);
    int vi = std::max(std::min((int) v, image.rows - 2), 0);
    double pu = u - ui;
    double pv = v - vi;
    float value[4] = {
            image.at<float>(vi, ui),
            image.at<float>(vi + 1, ui),
            image.at<float>(vi, ui + 1),
            image.at<float>(vi + 1, ui + 1)
    };
    return std::make_pair(true,
                          (value[0] * (1 - pv) + value[1] * pv) * (1 - pu) +
                          (value[2] * (1 - pv) + value[3] * pv) * pu);
}


} // namespace sensemap
} // namespace texture
