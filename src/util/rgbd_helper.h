//Copyright (c) 2020, SenseTime Group.
//All rights reserved.

#ifndef SENSETIME_UTIL_RGBD_HELPER_H_
#define SENSETIME_UTIL_RGBD_HELPER_H_

#include <stdio.h>
#include <dirent.h>

#include <string>
#include <vector>
#include <sstream>
#include <memory>

#include <Eigen/Core>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "util/bitmap.h"
#include "util/mat.h"

#include "base/camera.h"

#include <map>

#define MAX_VALID_DEPTH_IN_MM 10000
#define MAX_VALID_DEPTH_IN_M  (MAX_VALID_DEPTH_IN_MM * 0.001)

namespace sensemap {

typedef struct Frm {
    std::string name; // save pose name
    Eigen::Vector3d t;
    Eigen::Matrix3d R;
}Frm;

class CalibBinReader {

public:
    FILE* fp;

    template<typename type>
    void Read(type& data, const int N = 1) {
        fread(&data, sizeof(type), N, fp);
    }

public:
    bool ReadCalib(std::string fileName);
    void CloseCalib();

    std::string ToParamString();

    virtual bool GetRT(Eigen::Matrix4f &RT) = 0;
    virtual bool GetRGB_K(Eigen::Vector4f& K, int& rgb_w, int& rgb_h) = 0;
    virtual bool GetRGB_K(Eigen::Matrix3f& K, int& rgb_w, int& rgb_h) = 0;

    virtual bool GetToF_K(Eigen::Vector4f& K, int& tof_w, int& tof_h) = 0;
    virtual bool GetToF_K(Eigen::Matrix3f& K, int& tof_w, int& tof_h) = 0;
};

class CalibOPPOBinReader : public CalibBinReader {
protected:
    /// R17 Pro
    const static int RGB1_START = 18;
    const static int TOF_START = 6324;
    const static int RT_START = 7909;

    /// 516
    // const static int RGB1_START = 55;
    // const static int TOF_START = 1572;
    // const static int RT_START = 3074;
public:
    virtual bool GetRT(Eigen::Matrix4f &RT);
    virtual bool GetRGB_K(Eigen::Vector4f& K, int& rgb_w, int& rgb_h);
    virtual bool GetRGB_K(Eigen::Matrix3f& K, int& rgb_w, int& rgb_h);

    virtual bool GetToF_K(Eigen::Vector4f& K, int& tof_w, int& tof_h);
    virtual bool GetToF_K(Eigen::Matrix3f& K, int& tof_w, int& tof_h);
};

class CalibSamsungBinReader : public CalibBinReader {
public:
    virtual bool GetRT(Eigen::Matrix4f &RT);
    virtual bool GetRGB_K(Eigen::Vector4f& K, int& rgb_w, int& rgb_h);
    virtual bool GetRGB_K(Eigen::Matrix3f& K, int& rgb_w, int& rgb_h);

    virtual bool GetToF_K(Eigen::Vector4f& K, int& tof_w, int& tof_h);
    virtual bool GetToF_K(Eigen::Matrix3f& K, int& tof_w, int& tof_h);
};

class CalibTxtReader{
public:
    CalibTxtReader();
    CalibTxtReader(std::string path);


    std::string calib_path;

    std::string RT_name = "RT.txt";
    std::string KRGB_name = "K_rgb.txt";
    std::string KTOF_name = "K_tof.txt";

    bool GetRT(Eigen::Matrix4f &RT);
    bool GetRGB_K(Eigen::Matrix3f& K, int& rgb_w, int& rgb_h);
    bool GetToF_K(Eigen::Matrix3f& K, int& tof_w, int& tof_h);
};

class CalibIPadTxtReader : public CalibBinReader {
    const static int IPAD_RGB_WIDTH = 1920;
    const static int IPAD_RGB_HEIGHT = 1440;
    const static int IPAD_TOF_WIDTH = 256;
    const static int IPAD_TOF_HEIGHT = 192;

public:
    virtual bool GetRT(Eigen::Matrix4f &RT);
    virtual bool GetRGB_K(Eigen::Vector4f& K, int& rgb_w, int& rgb_h);
    virtual bool GetRGB_K(Eigen::Matrix3f& K, int& rgb_w, int& rgb_h);

    virtual bool GetToF_K(Eigen::Vector4f& K, int& tof_w, int& tof_h);
    virtual bool GetToF_K(Eigen::Matrix3f& K, int& tof_w, int& tof_h);
};

class CalibOPPOBinWriter : public CalibOPPOBinReader {
public:
    bool Write(
        const std::string & file,
        const Eigen::Matrix3f & K_tof, int w_tof, int h_tof,
        const Eigen::Matrix3f & K_rgb, int w_rgb, int h_rgb,
        const Eigen::Matrix4f & RT
    ) {
        FILE * fp = fopen(file.c_str(), "wb");
        if (!fp) return false;

        fseek(fp, RGB1_START, SEEK_SET);
        fwrite(&h_rgb, sizeof(int), 1, fp);
        fwrite(&w_rgb, sizeof(int), 1, fp);
        fwrite(&K_rgb(0, 0), sizeof(float), 1, fp);
        fwrite(&K_rgb(1, 1), sizeof(float), 1, fp);
        fwrite(&K_rgb(0, 2), sizeof(float), 1, fp);
        fwrite(&K_rgb(1, 2), sizeof(float), 1, fp);

        fseek(fp, TOF_START, SEEK_SET);
        fwrite(&h_tof, sizeof(int), 1, fp);
        fwrite(&w_tof, sizeof(int), 1, fp);
        fwrite(&K_tof(0, 0), sizeof(float), 1, fp);
        fwrite(&K_tof(1, 1), sizeof(float), 1, fp);
        fwrite(&K_tof(0, 2), sizeof(float), 1, fp);
        fwrite(&K_tof(1, 2), sizeof(float), 1, fp);

        Eigen::Vector3f eular = RT.block<3, 3>(0, 0).eulerAngles(0, 1, 2);
        Eigen::Vector3f t = RT.block<3, 1>(0, 3);
        fseek(fp, RT_START, SEEK_SET);
        fwrite(eular.data(), sizeof(float), 3, fp);
        fwrite(t.data(), sizeof(float), 3, fp);

        fclose(fp);
        return true;
    };
};

std::shared_ptr<CalibBinReader> GetCalibBinReaderFromName(const std::string & name);

// deprecated
// use RGBDData::ReadRGBDCameraParams instead
// std::vector<float> RGBDParamsToVector(const std::string& rgbd_params);

struct RGBDData {
    Bitmap color;
    double color_timestamp = 0.0;
    Camera color_camera;

    MatXf depth;
    double depth_timestamp = 0.0;
    Camera depth_camera;
    Eigen::Matrix4d depth_RT = Eigen::Matrix4d::Zero();
    
    Eigen::Vector3d gravity = Eigen::Vector3d::Zero();
    double timestamp = 0.0;
    uint16_t orientation = 0;

    bool HasRGBCalibration();
    bool HasRGBDCalibration();
    bool ReadRGBDCameraParams(const std::string & rgbd_camera_params);
};

struct RGBDReadOption {
    bool with_color = true;
    bool color_as_rgb = true;
    bool with_depth = true;

    static RGBDReadOption NoDepth(bool color_as_rgb = true) {
        RGBDReadOption option;
        option.color_as_rgb = color_as_rgb;
        option.with_depth = false;
        return option;
    }

    static RGBDReadOption NoColor() {
        RGBDReadOption option;
        option.with_color = false;
        return option;
    }

    static RGBDReadOption NoColorNoDepth() {
        RGBDReadOption option;
        option.with_color = false;
        option.with_depth = false;
        return option;
    }
};

struct RGBDWriteData : public RGBDData {
    cv::ColorConversionCodes color_type_hint = cv::COLOR_GRAY2BGR;
    cv::Mat color_mat;
    cv::Mat depth_mat;
};

static bool ScaleCandidateComparer(const std::pair<double, double> & a, const std::pair<double, double> & b) {
    return a.first < b.first;
}

double GetBestScaleByStatitics(const std::vector<std::pair<double, double>> & scale_candidates, double threshold = 0.02);

bool IntrinsicStringToCamera(const std::string & str, Camera & camera);

bool CameraToIntrinsicString(const Camera & camera, std::string & str);

bool IsFileRGBD(const std::string & image_path);

bool WriteRGBDData(const std::string & image_path, const RGBDWriteData & data);

bool ExtractRGBDData(const std::string & image_path, const RGBDReadOption & option, RGBDData & data);

bool ExtractRGBDData(const std::string & image_path, RGBDData & data);

bool ExtractRGBDData(const std::string & image_path, Bitmap & bitmap, bool as_rgb = true);

bool ExtractRGBDData(const std::string & image_path, Bitmap & bitmap, MatXf & depthmap, bool as_rgb = true);

bool WriteRGBDBinData(const std::string & image_path, const RGBDWriteData & data);

bool ExtractRGBDBinData(const std::string & image_path, const RGBDReadOption & option, RGBDData & data);

bool WriteRGBDBinxData(const std::string & image_path, const RGBDWriteData & data);

bool ExtractRGBDBinxData(const std::string & image_path, const RGBDReadOption & option, RGBDData & data);

cv::Mat ImageConvertBGR2NV12(const cv::Mat &bgr);

void ResizeDepthMap(MatXf &resized_depthmap,
                    const MatXf &depthmap);

void FastWarpDepthMap(MatXf &warped_depthmap,
                      const MatXf &depthmap, 
                      const Eigen::Matrix3f &rgb_K,
					  const Eigen::Matrix3f &depth_K,
					  const Eigen::Matrix4f &RT);

void WarpDepthMap2RGB(MatXf &warped_depthmap,
                      const MatXf &depthmap, 
                      const Eigen::Matrix3f &rgb_K,
					  const Eigen::Matrix3f &depth_K,
					  const Eigen::Matrix4f &RT);

void MeshWarpDepthMap(MatXf &warped_depthmap,
                      const MatXf &depthmap, 
                      const Eigen::Matrix3f &rgb_K,
					  const Eigen::Matrix3f &depth_K,
					  const Eigen::Matrix4f &RT);

void UniversalWarpDepthMap(MatXf &warped_depthmap,
                           const MatXf &depthmap, 
                           const Camera &rgb_camera,
                           const Camera &depth_camera,
                           const Eigen::Matrix4f &RT);

void AddWeightedDepthMap(MatXf &warped_depthmap, 
                         const MatXf & depthmap,
                         MatXf & depth_weight);

} // namespace sensemap

#endif