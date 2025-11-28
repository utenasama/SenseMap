//
// Created by sensetime on 2020/11/25.
//

#include "imageconvert.h"

namespace sensemap {
void Mat2FreeImage(cv::Mat& src, Bitmap* dst) {
    const int height = src.rows;
    const int width = src.cols;

    if (src.channels() == 1) {
        for (int y = 0; y < height; ++y) {
            uint8_t* line = FreeImage_GetScanLine(dst->Data(), height - 1 - y);
            uchar* ptr = src.ptr(y);
            for(int x = 0; x < width; ++x, ++ptr) {
                line[x] = ptr[0];
            }
        }
    } else if (src.channels() == 3) {
        for (int y = 0; y < height; ++y) {
            uint8_t* line = FreeImage_GetScanLine(dst->Data(), height - 1 - y);
            cv::Vec3b* ptr = src.ptr<cv::Vec3b>(y);
            for(int x = 0; x < width; ++x, ++ptr) {
                line[3 * x + FI_RGBA_RED] = (*ptr)[2];
                line[3 * x + FI_RGBA_GREEN] = (*ptr)[1];
                line[3 * x + FI_RGBA_BLUE] = (*ptr)[0];
            }
        }
    }
}

void FreeImage2Mat(Bitmap* src, cv::Mat& dst) {
    const int height = src->Height();
    const int width = src->Width();

    if (src->IsGrey()) {
        dst = cv::Mat(height, width, CV_8UC1);
        for (int y = 0; y < height; ++y) {
            uint8_t* line = FreeImage_GetScanLine(src->Data(), height - 1 - y);
            uchar* ptr = dst.ptr(y);
            for (int x = 0; x < width; ++x, ++ptr) {
                *ptr = line[x];
            }
        }
    } else if (src->IsRGB()) {
        dst = cv::Mat(height, width, CV_8UC3);
        for (int y = 0; y < height; ++y) {
            uint8_t* line = FreeImage_GetScanLine(src->Data(), height - 1 - y);
            cv::Vec3b* ptr = dst.ptr<cv::Vec3b>(y);
            for (int x = 0; x < width; ++x, ++ptr) {
                (*ptr)[2] = line[3 * x + FI_RGBA_RED];
                (*ptr)[1] = line[3 * x + FI_RGBA_GREEN];
                (*ptr)[0] = line[3 * x + FI_RGBA_BLUE];
            }
        }
    }
}

void CvMat2Mat(cv::Mat &src, MatXu &dst) {
    int depth = -1;
    if (src.type() == CV_8UC1) depth = 1;
    else if (src.type() == CV_8UC3) depth = 3;
    else return;
    if (dst.GetDepth() != depth || dst.GetHeight() != src.rows || dst.GetWidth() != src.cols) {
        dst = MatXu(src.cols, src.rows, depth);
    }
    memcpy(dst.GetPtr(), src.data,  src.cols * src.rows * depth);
}

void CvMat2Mat(cv::Mat &src, MatXf &dst) {
    int depth = -1;
    if (src.type() == CV_32FC1) depth = 1;
    else if (src.type() == CV_32FC3) depth = 3;
    else return;
    if (dst.GetDepth() != depth || dst.GetHeight() != src.rows || dst.GetWidth() != src.cols) {
        dst = MatXf(src.cols, src.rows, depth);
    }
    memcpy(dst.GetPtr(), src.data,  sizeof(float) * src.cols * src.rows * depth);
}

void Mat2CvMat(MatXu &src, cv::Mat &dst) {
    int type = -1;
    if (src.GetDepth() == 1) type = CV_8UC1;
    else if (src.GetDepth() == 3) type = CV_8UC3;
    else return;

    if (dst.type() != type || dst.rows != src.GetHeight() || dst.cols != src.GetWidth()) {
        dst = cv::Mat(src.GetHeight(), src.GetWidth(), type);
    }
    memcpy(dst.data, src.GetPtr(),  src.GetWidth() * src.GetHeight() * src.GetDepth());
}

void Mat2CvMat(MatXf &src, cv::Mat &dst) {
    int type = -1;
    if (src.GetDepth() == 1) type = CV_32FC1;
    else if (src.GetDepth() == 3) type = CV_32FC3;
    else return;

    if (dst.type() != type || dst.rows != src.GetHeight() || dst.cols != src.GetWidth()) {
        dst = cv::Mat(src.GetHeight(), src.GetWidth(), type);
    }
    memcpy(dst.data, src.GetPtr(), sizeof(float) * src.GetWidth() * src.GetHeight() * src.GetDepth());
}
}