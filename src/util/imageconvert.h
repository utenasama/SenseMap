//
// Created by sensetime on 2020/11/25.
//

#ifndef SENSEMAP_IMAGECONVERT_H
#define SENSEMAP_IMAGECONVERT_H

#include <opencv2/opencv.hpp>
#include "util/bitmap.h"
#include "mat.h"


namespace sensemap {

void Mat2FreeImage(cv::Mat& src, Bitmap* dst);

void FreeImage2Mat(Bitmap* src, cv::Mat& dst);

void CvMat2Mat(cv::Mat &src, MatXu &dst);

void CvMat2Mat(cv::Mat &src, MatXf &dst);

void Mat2CvMat(MatXu &src, cv::Mat &dst);

void Mat2CvMat(MatXf &src, cv::Mat &dst);
}

#endif //SENSEMAP_IMAGECONVERT_H
