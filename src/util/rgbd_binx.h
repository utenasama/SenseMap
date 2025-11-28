//Copyright (c) 2020, SenseTime Group.
//All rights reserved.
#ifndef SENSETIME_UTIL_RGBD_BINX_H_
#define SENSETIME_UTIL_RGBD_BINX_H_

#include <iostream>
#include <Eigen/Dense>
#include <opencv2/imgproc.hpp>
#include "base/camera.h"
#include "util/bitmap.h"
#include "util/mat.h"
#include "util/misc.h"
#include "util/imageconvert.h"
#include "util/rgbd_helper.h"
#include "zlib.h"

namespace sensemap {

enum class RGBDBinxDataType
{
    /*
    Version: 0.0.1
    */
    MAGIC_NUMBER = 0x77498864,                              // no meaning, must appear at the beginning of the stream

    COLOR_INFO_MIN_            = 0x00000000,
    COLOR_INFO_TIMESTAMP       = COLOR_INFO_MIN_ | 0x0000,  // float64, seconds
    COLOR_INFO_INTRINSIC_STR   = COLOR_INFO_MIN_ | 0x0100,  // CameraModel + CameraParams delimited by ","
    COLOR_INFO_MAX_            = COLOR_INFO_MIN_ | 0xFFFF,

    DEPTH_INFO_MIN_            = 0x00010000,
    DEPTH_INFO_TIMESTAMP       = DEPTH_INFO_MIN_ | 0x0000,  // float64, seconds
    DEPTH_INFO_INTRINSIC_STR   = DEPTH_INFO_MIN_ | 0x0100,  // CameraModel + CameraParams delimited by ","
    DEPTH_INFO_RT_MM           = DEPTH_INFO_MIN_ | 0x0200,  // float64(qw) + float64(qx) + float64(qy) + float64(qz) + float64(tx) + float64(ty) + float64(tz), depth -> color transform, unit mm
    DEPTH_INFO_MAX_            = DEPTH_INFO_MIN_ | 0xFFFF,

    FRAME_INFO_MIN_            = 0x0FFF0000,
    FRAME_INFO_TIMESTAMP       = FRAME_INFO_MIN_ | 0x0000,  // float64, seconds
    FRAME_INFO_ORIENTATION     = FRAME_INFO_MIN_ | 0x0001,  // uint16, same as EXIF
    FRAME_INFO_GRAVITY         = FRAME_INFO_MIN_ | 0x0100,  // float64(x) + float64(y) + float64(z)
    FRAME_INFO_MAX_            = FRAME_INFO_MIN_ | 0xFFFF,

    COLOR_DATA_MIN_            = 0x10000000,
    COLOR_DATA_GRAY            = COLOR_DATA_MIN_ | 0x0000,  // int32(width) + int32(height) + width*height*uint8, row-majored
    COLOR_DATA_BGR             = COLOR_DATA_MIN_ | 0x0001,  // int32(width) + int32(height) + width*height*uint8*3, row-majored
    COLOR_DATA_YUV_I420        = COLOR_DATA_MIN_ | 0x0100,  // int32(width) + int32(height) + width*height*uint8*3/2, row-majored, YYYYUUVV
    COLOR_DATA_YUV_YV12        = COLOR_DATA_MIN_ | 0x0101,  // int32(width) + int32(height) + width*height*uint8*3/2, row-majored, YYYYVVUU
    COLOR_DATA_YUV_NV12        = COLOR_DATA_MIN_ | 0x0102,  // int32(width) + int32(height) + width*height*uint8*3/2, row-majored, YYYYVVUU
    COLOR_DATA_YUV_NV21        = COLOR_DATA_MIN_ | 0x0103,  // int32(width) + int32(height) + width*height*uint8*3/2, row-majored, YYYYVVUU
    COLOR_DATA_MAX_            = COLOR_DATA_MIN_ | 0xFFFF,

    DEPTH_DATA_MIN_            = 0x10010000,
    DEPTH_DATA_U16_MM          = DEPTH_DATA_MIN_ | 0x0000,  // int32(width) + int32(height) + width*height*uint16, row-majored, unit mm
    DEPTH_DATA_F32_M           = DEPTH_DATA_MIN_ | 0x0001,  // int32(width) + int32(height) + width*height*float32, row-majored, unit m
    DEPTH_DATA_MAX_            = DEPTH_DATA_MIN_ | 0xFFFF,
};

class RGBDBinxHelper {
protected:
    template<typename T>
    static bool IsMagicNumber(T code) {
        return (code == (T)RGBDBinxDataType::MAGIC_NUMBER);
    }
    
    template<typename T>
    static bool IsColorInfo(T code) {
        return (code >= (T)RGBDBinxDataType::COLOR_INFO_MIN_) &&
               (code <= (T)RGBDBinxDataType::COLOR_INFO_MAX_);
    }
    
    template<typename T>
    static bool IsDepthInfo(T code) {
        return (code >= (T)RGBDBinxDataType::DEPTH_INFO_MIN_) &&
               (code <= (T)RGBDBinxDataType::DEPTH_INFO_MAX_);
    }
    
    template<typename T>
    static bool IsFrameInfo(T code) {
        return (code >= (T)RGBDBinxDataType::FRAME_INFO_MIN_) &&
               (code <= (T)RGBDBinxDataType::FRAME_INFO_MAX_);
    }
    
    template<typename T>
    static bool IsColorData(T code) {
        return (code >= (T)RGBDBinxDataType::COLOR_DATA_MIN_) &&
               (code <= (T)RGBDBinxDataType::COLOR_DATA_MAX_);
    }
    
    template<typename T>
    static bool IsDepthData(T code) {
        return (code >= (T)RGBDBinxDataType::DEPTH_DATA_MIN_) &&
               (code <= (T)RGBDBinxDataType::DEPTH_DATA_MAX_);
    }
    
    template<typename T>
    static bool IsColorGray(T code) {
        return (code == (T)RGBDBinxDataType::COLOR_DATA_GRAY);
    }
    
    template<typename T>
    static bool IsColorYUV(T code) {
        return (code == (T)RGBDBinxDataType::COLOR_DATA_YUV_I420) ||
               (code == (T)RGBDBinxDataType::COLOR_DATA_YUV_YV12) ||
               (code == (T)RGBDBinxDataType::COLOR_DATA_YUV_NV12) ||
               (code == (T)RGBDBinxDataType::COLOR_DATA_YUV_NV21);
    }

protected:
    template<typename T>
    static bool Read(gzFile fp, T & value) {
        if (gzread(fp, &value, sizeof(value)) != sizeof(value)) return false;
        return true;
    }

    static bool ReadAndSkip(gzFile fp) {
        int32_t size;
        if (!Read(fp, size)) return false;

        if (size > 0) {
            gzseek(fp, size, SEEK_CUR);
        }
        return true;
    }

    static bool ReadColorInfo(gzFile fp, int32_t code, RGBDData & data) {
        int32_t size;
        switch (code)
        {
        case (int32_t)RGBDBinxDataType::COLOR_INFO_TIMESTAMP: {
            double timestamp;
            if (!Read(fp, size)) return false;
            if (size != sizeof(timestamp)) return false;
            if (!Read(fp, timestamp)) return false;
            data.color_timestamp = timestamp;
        }
            break;

        case (int32_t)RGBDBinxDataType::COLOR_INFO_INTRINSIC_STR: {
            if (!Read(fp, size)) return false;
            std::vector<char> buffer(size + 1, 0);
            if (gzread(fp, buffer.data(), size) != size) return false;

            std::string params(buffer.data());
            if (!IntrinsicStringToCamera(params, data.color_camera)) return false;
        }
            break;
        
        default:
            std::cerr << "Unknown color info " << code << std::endl;
            return false;
        }

        return true;
    }

    static bool ReadDepthInfo(gzFile fp, int32_t code, RGBDData & data) {
        int32_t size;
        switch (code)
        {
        case (int32_t)RGBDBinxDataType::DEPTH_INFO_TIMESTAMP: {
            double timestamp;
            if (!Read(fp, size)) return false;
            if (size != sizeof(timestamp)) return false;
            if (!Read(fp, timestamp)) return false;
            data.depth_timestamp = timestamp;
        }
            break;

        case (int32_t)RGBDBinxDataType::DEPTH_INFO_INTRINSIC_STR: {
            if (!Read(fp, size)) return false;
            std::vector<char> buffer(size + 1, 0);
            if (gzread(fp, buffer.data(), size) != size) return false;

            std::string params(buffer.data());
            if (!IntrinsicStringToCamera(params, data.depth_camera)) return false;
        }
            break;

        case (int32_t)RGBDBinxDataType::DEPTH_INFO_RT_MM: {
            double RT[7];
            if (!Read(fp, size)) return false;
            if (size != sizeof(RT)) return false;
            if (!Read(fp, RT)) return false;
            Eigen::Quaternion<double> q(RT[0], RT[1], RT[2], RT[3]);
            data.depth_RT.setIdentity();
            data.depth_RT.block<3, 3>(0, 0) = q.toRotationMatrix();
            data.depth_RT(0, 3) = RT[4];
            data.depth_RT(1, 3) = RT[5];
            data.depth_RT(2, 3) = RT[6];
        }
            break;
        
        default:
            std::cerr << "Unknown depth info " << code << std::endl;
            return false;
        }

        return true;
    }

    static bool ReadFrameInfo(gzFile fp, int32_t code, RGBDData & data) {
        int32_t size;
        switch (code)
        {
        case (int32_t)RGBDBinxDataType::FRAME_INFO_TIMESTAMP: {
            double timestamp;
            if (!Read(fp, size)) return false;
            if (size != sizeof(timestamp)) return false;
            if (!Read(fp, timestamp)) return false;
            data.timestamp = timestamp;
        }
            break;

        case (int32_t)RGBDBinxDataType::FRAME_INFO_ORIENTATION: {
            uint16_t orientation;
            if (!Read(fp, size)) return false;
            if (size != sizeof(orientation)) return false;
            if (!Read(fp, orientation)) return false;
            data.orientation = orientation;
        }
            break;

        case (int32_t)RGBDBinxDataType::FRAME_INFO_GRAVITY: {
            double g[3];
            if (!Read(fp, size)) return false;
            if (size != sizeof(g)) return false;
            if (!Read(fp, g)) return false;
            data.gravity(0) = g[0];
            data.gravity(1) = g[1];
            data.gravity(2) = g[2];
        }
            break;
        
        default:
            std::cerr << "Unknown depth info " << code << std::endl;
            return false;
        }

        return true;
    }

    static bool ReadColorData(gzFile fp, int32_t code, const RGBDReadOption & option, RGBDData & data) {
        bool is_gray = IsColorGray(code);
        bool is_yuv = IsColorYUV(code);

        cv::Mat mat;
        int32_t width, height, size;
        if (gzread(fp, &size, sizeof(size)) != sizeof(size)) return false;
        if (gzread(fp, &width, sizeof(width)) != sizeof(width)) return false;
        if (gzread(fp, &height, sizeof(height)) != sizeof(height)) return false;
        if (is_gray) {
            mat = cv::Mat(height, width, CV_8UC1);
        } else if (is_yuv) {
            mat = cv::Mat(height * 3 / 2, width, CV_8UC1);
        } else if (code == (int32_t)RGBDBinxDataType::COLOR_DATA_BGR) {
            mat = cv::Mat(height, width, CV_8UC3);
        } else {
            std::cerr << "Unknown color data type " << code << std::endl;
            return false;
        }

        const int32_t total = mat.total() * mat.elemSize();
        if (size != total + sizeof(width) + sizeof(height)) return false;
        if (option.with_color) {
            if (gzread(fp, mat.data, total) != total) return false;

            data.color.Allocate(width, height, option.color_as_rgb);
            if (option.color_as_rgb) {
                if (is_gray) {
                    cv::cvtColor(mat, mat, cv::COLOR_GRAY2BGR);
                } else if (is_yuv) {
                    if (code == (int32_t)RGBDBinxDataType::COLOR_DATA_YUV_I420) {
                        cv::cvtColor(mat, mat, cv::COLOR_YUV2BGR_I420);
                    } else if (code == (int32_t)RGBDBinxDataType::COLOR_DATA_YUV_YV12) {
                        cv::cvtColor(mat, mat, cv::COLOR_YUV2BGR_YV12);
                    } else if (code == (int32_t)RGBDBinxDataType::COLOR_DATA_YUV_NV12) {
                        cv::cvtColor(mat, mat, cv::COLOR_YUV2BGR_NV12);
                    } else if (code == (int32_t)RGBDBinxDataType::COLOR_DATA_YUV_NV21) {
                        cv::cvtColor(mat, mat, cv::COLOR_YUV2BGR_NV21);
                    } else {
                        std::cerr << "YUV type " << code << " not implemented. " << std::endl;
                        return false;
                    }
                }
            } else {
                if (is_yuv) {
                    if (code == (int32_t)RGBDBinxDataType::COLOR_DATA_YUV_I420) {
                        cv::cvtColor(mat, mat, cv::COLOR_YUV2GRAY_I420);
                    } else if (code == (int32_t)RGBDBinxDataType::COLOR_DATA_YUV_YV12) {
                        cv::cvtColor(mat, mat, cv::COLOR_YUV2GRAY_YV12);
                    } else if (code == (int32_t)RGBDBinxDataType::COLOR_DATA_YUV_NV12) {
                        cv::cvtColor(mat, mat, cv::COLOR_YUV2GRAY_NV12);
                    } else if (code == (int32_t)RGBDBinxDataType::COLOR_DATA_YUV_NV21) {
                        cv::cvtColor(mat, mat, cv::COLOR_YUV2GRAY_NV21);
                    } else {
                        std::cerr << "YUV type " << code << " not implemented. " << std::endl;
                        return false;
                    }
                } else if (code == (int32_t)RGBDBinxDataType::COLOR_DATA_BGR) {
                    cv::cvtColor(mat, mat, cv::COLOR_BGR2GRAY);
                }
            }
            
            Mat2FreeImage(mat, &data.color);
        } else {
            gzseek(fp, total, SEEK_CUR);
            data.color.Allocate(width, height, option.color_as_rgb);
        }

        return true;
    }

    static bool ReadDepthData(gzFile fp, int32_t code, const RGBDReadOption & option, RGBDData & data) {
        cv::Mat mat;
        int32_t width, height, size;
        if (gzread(fp, &size, sizeof(size)) != sizeof(size)) return false;
        if (gzread(fp, &width, sizeof(width)) != sizeof(width)) return false;
        if (gzread(fp, &height, sizeof(height)) != sizeof(height)) return false;
        if (code == (int32_t)RGBDBinxDataType::DEPTH_DATA_U16_MM) {
            mat = cv::Mat(height, width, CV_16UC1);
        } else if (code == (int32_t)RGBDBinxDataType::DEPTH_DATA_F32_M) {
            mat = cv::Mat(height, width, CV_32FC1);
        } else {
            std::cerr << "Unknown depth data type " << code << std::endl;
            return false;
        }

        const int32_t total = mat.total() * mat.elemSize();
        if (size != total + sizeof(width) + sizeof(height)) return false;
        if (option.with_depth) {
            if (gzread(fp, mat.data, total) != total) return false;

            if (code == (int32_t)RGBDBinxDataType::DEPTH_DATA_U16_MM) {
                mat.convertTo(mat, CV_32FC1, 0.001);
            }

            CvMat2Mat(mat, data.depth);
        } else {
            gzseek(fp, total, SEEK_CUR);
            data.depth = MatXf(width, height, 1);
        }

        return true;
    }

public:
    static bool CheckMagicNumber(gzFile fp) {
        int32_t code;
        if (!Read(fp, code)) return false;
        if (!IsMagicNumber(code)) return false;
        if (!ReadAndSkip(fp)) return false;

        return true;
    }

    static bool ReadNextCode(gzFile fp, int32_t & code) {
        return Read(fp, code);
    }

    static bool ReadNext(gzFile fp, int32_t code, const RGBDReadOption & option, RGBDData & data) {
        if (IsMagicNumber(code)) {
            if (!ReadAndSkip(fp)) {
                return false;
            }
        } else if (IsColorInfo(code)) {
            if (!ReadColorInfo(fp, code, data)) {
                return false;
            }
        } else if (IsDepthInfo(code)) {
            if (!ReadDepthInfo(fp, code, data)) {
                return false;
            }
        } else if (IsFrameInfo(code)) {
            if (!ReadFrameInfo(fp, code, data)) {
                return false;
            }
        } else if (IsColorData(code)) {
            if (!ReadColorData(fp, code, option, data)) {
                return false;
            }
        } else if (IsDepthData(code)) {
            if (!ReadDepthData(fp, code, option, data)) {
                return false;
            }
        } else {
            std::cerr << "Unknown data type " << code << std::endl;
            if (!ReadAndSkip(fp)) {
                return false;
            }
        }

        return true;
    }

public:
    static bool WriteMagicNumber(gzFile fp) {
        int32_t code = (int32_t)RGBDBinxDataType::MAGIC_NUMBER;
        int32_t size = 0;
        gzwrite(fp, &code, sizeof(code));
        gzwrite(fp, &size, sizeof(size));

        return true;
    }

    static bool WriteColorData(gzFile fp, const Bitmap & color) {
        int32_t width = color.Width();
        int32_t height = color.Height();
        int32_t code = color.IsRGB() ? (int32_t)RGBDBinxDataType::COLOR_DATA_BGR : 
                                       (int32_t)RGBDBinxDataType::COLOR_DATA_GRAY;
        int32_t size = color.IsRGB() ? width * height * 3 :
                                       width * height;
                size += sizeof(width) + sizeof (height);
        gzwrite(fp, &code, sizeof(code));
        gzwrite(fp, &size, sizeof(size));
        gzwrite(fp, &width, sizeof(width));
        gzwrite(fp, &height, sizeof(height));
        for (int32_t i = 0; i < height; i++) {
            const auto line = color.GetScanline(i);
            if (color.IsRGB()) {
                gzwrite(fp, line, width * 3);
            } else {
                gzwrite(fp, line, width);
            }
        }
        return true;
    }

    static bool WriteColorData(gzFile fp, const cv::Mat & color, cv::ColorConversionCodes color_type_hint) {
        if (color.type() == CV_8UC3) {
            int32_t code = (int32_t)RGBDBinxDataType::COLOR_DATA_BGR;
            int32_t width = color.cols;
            int32_t height = color.rows;
            int32_t size = sizeof(width) + sizeof(height) + color.rows * color.cols * 3;
            gzwrite(fp, &code, sizeof(code));
            gzwrite(fp, &size, sizeof(size));
            gzwrite(fp, &width, sizeof(width));
            gzwrite(fp, &height, sizeof(height));
            for (int32_t i = 0; i < color.rows; i++) {
                const auto line = color.ptr(i);
                gzwrite(fp, line, color.cols * 3);
            }
        } else if (color.type() == CV_8UC1) {
            int32_t code;
            if (color_type_hint == cv::COLOR_GRAY2BGR) {
                code = (int32_t)RGBDBinxDataType::COLOR_DATA_GRAY;
            } else if (color_type_hint == cv::COLOR_YUV2BGR_I420) {
                code = (int32_t)RGBDBinxDataType::COLOR_DATA_YUV_I420;
            } else if (color_type_hint == cv::COLOR_YUV2BGR_YV12) {
                code = (int32_t)RGBDBinxDataType::COLOR_DATA_YUV_YV12;
            } else if (color_type_hint == cv::COLOR_YUV2BGR_NV12) {
                code = (int32_t)RGBDBinxDataType::COLOR_DATA_YUV_NV12;
            } else if (color_type_hint == cv::COLOR_YUV2BGR_NV21) {
                code = (int32_t)RGBDBinxDataType::COLOR_DATA_YUV_NV21;
            } else {
                std::cerr << "Invalid color type hint. " << std::endl;
                return false;
            }
            bool is_gray = IsColorGray(code);
            bool is_yuv = IsColorYUV(code);

            int32_t width = color.cols;
            int32_t height = is_yuv ? color.rows / 3 * 2 : color.rows;
            int32_t size = sizeof(width) + sizeof(height) + color.rows * color.cols;
            gzwrite(fp, &code, sizeof(code));
            gzwrite(fp, &size, sizeof(size));
            gzwrite(fp, &width, sizeof(width));
            gzwrite(fp, &height, sizeof(height));
            for (int32_t i = 0; i < color.rows; i++) {
                const auto line = color.ptr(i);
                gzwrite(fp, line, color.cols);
            }
        } else {
            std::cerr << "Color format not supported yet. " << std::endl;
            return false;
        }

        return true;
    }

    static bool WriteColorTimestamp(gzFile fp, double timestamp) {
        int32_t code = (int32_t)RGBDBinxDataType::COLOR_INFO_TIMESTAMP;
        gzwrite(fp, &code, sizeof(code));

        int32_t size = sizeof(timestamp);
        gzwrite(fp, &size, sizeof(size));
        gzwrite(fp, &timestamp, sizeof(timestamp));

        return true;
    }
    
    static bool WriteColorIntrinsicString(gzFile fp, const Camera & camera) {
        int32_t code = (int32_t)RGBDBinxDataType::COLOR_INFO_INTRINSIC_STR;
        gzwrite(fp, &code, sizeof(code));

        std::string str;
        if (!CameraToIntrinsicString(camera, str)) return false;

        int32_t size = str.length();
        gzwrite(fp, &size, sizeof(size));
        gzwrite(fp, str.c_str(), size);

        return true;
    }

    static bool WriteDepthData(gzFile fp, const MatXf & depth) {
        int32_t code = (int32_t)RGBDBinxDataType::DEPTH_DATA_F32_M;
        gzwrite(fp, &code, sizeof(code));

        int32_t width = depth.GetWidth();
        int32_t height = depth.GetHeight();
        int32_t size = sizeof(width) + sizeof(height) + width * height * sizeof(float);
        gzwrite(fp, &size, sizeof(size));
        gzwrite(fp, &width, sizeof(width));
        gzwrite(fp, &height, sizeof(height));
        gzwrite(fp, depth.GetPtr(), width * height * sizeof(float));

        return true;
    }

    static bool WriteDepthData(gzFile fp, const cv::Mat & depth) {
        int32_t code;
        int32_t unit_size = 0;
        if (depth.type() == CV_32FC1) {
            code = (int32_t)RGBDBinxDataType::DEPTH_DATA_F32_M;
            unit_size = sizeof(float);
        } else if (depth.type() == CV_16UC1) {
            code = (int32_t)RGBDBinxDataType::DEPTH_DATA_U16_MM;
            unit_size = sizeof(uint16_t);
        } else {
            std::cerr << "Depth mat format not supported. " << std::endl;
            return false;
        }
        gzwrite(fp, &code, sizeof(code));

        int32_t width = depth.cols;
        int32_t height = depth.rows;
        int32_t size = sizeof(width) + sizeof(height) + width * height * unit_size;
        gzwrite(fp, &size, sizeof(size));
        gzwrite(fp, &width, sizeof(width));
        gzwrite(fp, &height, sizeof(height));
        for (int32_t i = 0; i < height; i++) {
            const auto line = depth.ptr(i);
            gzwrite(fp, line, width * unit_size);
        }

        return true;
    }

    static bool WriteDepthTimestamp(gzFile fp, double timestamp) {
        int32_t code = (int32_t)RGBDBinxDataType::DEPTH_INFO_TIMESTAMP;
        gzwrite(fp, &code, sizeof(code));

        int32_t size = sizeof(timestamp);
        gzwrite(fp, &size, sizeof(size));
        gzwrite(fp, &timestamp, sizeof(timestamp));

        return true;
    }

    static bool WriteDepthIntrinsicString(gzFile fp, const Camera & camera) {
        int32_t code = (int32_t)RGBDBinxDataType::DEPTH_INFO_INTRINSIC_STR;
        gzwrite(fp, &code, sizeof(code));

        std::string str;
        if (!CameraToIntrinsicString(camera, str)) return false;

        int32_t size = str.length();
        gzwrite(fp, &size, sizeof(size));
        gzwrite(fp, str.c_str(), size);

        return true;
    }

    static bool WriteDepthRT(gzFile fp, const Eigen::Matrix4d & RT) {
        int32_t code = (int32_t)RGBDBinxDataType::DEPTH_INFO_RT_MM;
        gzwrite(fp, &code, sizeof(code));

        Eigen::Matrix3d R = RT.block<3, 3>(0, 0);
        Eigen::Vector3d t = RT.block<3, 1>(0, 3);
        Eigen::Quaternion<double> q(R);
        double RTv[] = { q.w(), q.x(), q.y(), q.z(), t.x(), t.y(), t.z() };
        int32_t size = sizeof(RTv);
        gzwrite(fp, &size, sizeof(size));
        gzwrite(fp, &RTv, sizeof(RTv));

        return true;
    }

    static bool WriteFrameTimestamp(gzFile fp, double timestamp) {
        int32_t code = (int32_t)RGBDBinxDataType::FRAME_INFO_TIMESTAMP;
        gzwrite(fp, &code, sizeof(code));

        int32_t size = sizeof(timestamp);
        gzwrite(fp, &size, sizeof(size));
        gzwrite(fp, &timestamp, sizeof(timestamp));

        return true;
    }

    static bool WriteFrameOrientation(gzFile fp, uint16_t orientation) {
        int32_t code = (int32_t)RGBDBinxDataType::FRAME_INFO_ORIENTATION;
        gzwrite(fp, &code, sizeof(code));

        int32_t size = sizeof(orientation);
        gzwrite(fp, &size, sizeof(size));
        gzwrite(fp, &orientation, sizeof(orientation));

        return true;
    }

    static bool WriteFrameGravity(gzFile fp, const Eigen::Vector3d & gravity) {
        int32_t code = (int32_t)RGBDBinxDataType::FRAME_INFO_GRAVITY;
        gzwrite(fp, &code, sizeof(code));

        double g[3] { gravity[0], gravity[1], gravity[2] };
        int32_t size = sizeof(g);
        gzwrite(fp, &size, sizeof(size));
        gzwrite(fp, g, sizeof(g));

        return true;
    }
};

}
#endif