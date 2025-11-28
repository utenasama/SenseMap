//Copyright (c) 2020, SenseTime Group.
//All rights reserved.

#include "rgbd_helper.h"
#include "base/camera_models.h"
#include "util/imageconvert.h"
#include "util/string.h"
#include "util/depth_mesh.h"
#include "util/rgbd_binx.h"

#include <iostream>
#include <fstream>
#include <Eigen/Geometry>
#include <boost/filesystem.hpp>

namespace sensemap {

bool CalibBinReader::ReadCalib(std::string fileName){
    fp = fopen(fileName.c_str(), "rb");
    if (fp == NULL) {
        fprintf(stderr, "%s fopen error!\n", fileName.c_str());
        return false;
    } else {
        printf("open %s OK\n", fileName.c_str());
    }
    return true;
}

void CalibBinReader::CloseCalib(){
    if (fp) {
        fclose(fp);
        fp = NULL;
    }
}

std::string CalibBinReader::ToParamString() {
    if (fp == NULL) return "";
    
    Eigen::Matrix4f RT;
    Eigen::Matrix3f rgb_K, depth_K;
    int rgb_width, rgb_height;
    int depth_width, depth_height;

    GetRT(RT);
    GetRGB_K(rgb_K, rgb_width, rgb_height);
    GetToF_K(depth_K, depth_width, depth_height);

    std::string strparams;
    strparams.append(std::to_string(rgb_width));
    strparams.append(",");
    strparams.append(std::to_string(rgb_height));
    for (int i = 0; i < 9; ++i) {
        strparams.append(",");
        strparams.append(std::to_string(rgb_K(i / 3, i % 3)));
    }
    for (int i = 0; i < 9; ++i) {
        strparams.append(",");
        strparams.append(std::to_string(depth_K(i / 3, i % 3)));
    }
    for (int i = 0; i < 16; ++i) {
        strparams.append(",");
        strparams.append(std::to_string(RT(i / 4, i % 4)));
    }
    return strparams;
}

bool CalibSamsungBinReader::GetRT(Eigen::Matrix4f &RT){
    RT.setIdentity();
    fseek(fp, 4, SEEK_SET);
    Eigen::Vector3d eular;
    Read<double>(eular(0));
    Read<double>(eular(1));
    Read<double>(eular(2));

    Eigen::Matrix3d R;
    Eigen::Vector3d t;

    Read<double>(R(0, 0));
    Read<double>(R(0, 1));
    Read<double>(R(0, 2));
    Read<double>(R(1, 0));
    Read<double>(R(1, 1));
    Read<double>(R(1, 2));
    Read<double>(R(2, 0));
    Read<double>(R(2, 1));
    Read<double>(R(2, 2));

    // R = Eigen::AngleAxisd(eular(0), Eigen::Vector3d::UnitX())
    //     * Eigen::AngleAxisd(eular(1), Eigen::Vector3d::UnitY())
    //     * Eigen::AngleAxisd(eular(2), Eigen::Vector3d::UnitZ());
    R.transposeInPlace();

    fseek(fp, 100, SEEK_SET);
    Read<double>(t(0));
    Read<double>(t(1));
    Read<double>(t(2));
    t = -R * t;

    RT.block<3,3>(0,0) = R.cast<float>();
    RT.block<3,1>(0,3) = t.cast<float>();
    return true;
}

bool CalibSamsungBinReader::GetRGB_K(Eigen::Vector4f& K, int& rgb_w, int& rgb_h){
    fseek(fp, 564, SEEK_SET);
    Read<int>(rgb_w);
    Read<int>(rgb_h);
    K.setIdentity();

    fseek(fp, 124, SEEK_SET);
    double fx, fy, cx, cy;
    double value;

    Read<double>(fx);
    Read<double>(value);
    Read<double>(cx);
    Read<double>(value);

    Read<double>(fy);
    Read<double>(cy);

    K(0) = fx;
    K(1) = fy;
    K(2) = cx;
    K(3) = cy;
    return true;
}

bool CalibSamsungBinReader::GetRGB_K(Eigen::Matrix3f& K, int& rgb_w, int& rgb_h){
    Eigen::Vector4f K_vec;
    GetRGB_K(K_vec, rgb_w, rgb_h);
    K.setIdentity();
    K(0,0) = K_vec(0);
    K(1,1) = K_vec(1);
    K(0,2) = K_vec(2);
    K(1,2) = K_vec(3);

    return true;
}

bool CalibSamsungBinReader::GetToF_K(Eigen::Vector4f& K, int& tof_w, int& tof_h){
    fseek(fp, 572, SEEK_SET);
    Read<int>(tof_w);
    Read<int>(tof_h);
    K.setIdentity();

    fseek(fp, 196, SEEK_SET);
    double fx, fy, cx, cy;
    double value;

    Read<double>(fx);
    Read<double>(value);
    Read<double>(cx);
    Read<double>(value);

    Read<double>(fy);
    Read<double>(cy);

    K(0) = fx;
    K(1) = fy;
    K(2) = cx;
    K(3) = cy;
    return true;
}

bool CalibSamsungBinReader::GetToF_K(Eigen::Matrix3f& K, int& tof_w, int& tof_h){
    Eigen::Vector4f K_vec;
    GetToF_K(K_vec, tof_w, tof_h);
    K.setIdentity();
    K(0,0) = K_vec(0);
    K(1,1) = K_vec(1);
    K(0,2) = K_vec(2);
    K(1,2) = K_vec(3);
    return true;
}

bool CalibOPPOBinReader::GetRT(Eigen::Matrix4f &RT){
    RT.setIdentity();
    fseek(fp, RT_START, SEEK_SET);
    Eigen::Vector3f eular;
    Read<float>(eular(0));
    Read<float>(eular(1));
    Read<float>(eular(2));

    Eigen::Matrix3f R;
    Eigen::Vector3f t;

    R = Eigen::AngleAxisf(eular(0), Eigen::Vector3f::UnitX())
        * Eigen::AngleAxisf(eular(1), Eigen::Vector3f::UnitY())
        * Eigen::AngleAxisf(eular(2), Eigen::Vector3f::UnitZ());
    R.transposeInPlace();

    Read<float>(t(0));
    Read<float>(t(1));
    Read<float>(t(2));
    t = -R * t;

    RT.block<3,3>(0,0) = R;
    RT.block<3,1>(0,3) = t;
    return true;
}

bool CalibOPPOBinReader::GetRGB_K(Eigen::Vector4f& K, int& rgb_w, int& rgb_h){
    fseek(fp, RGB1_START, SEEK_SET);
    Read<int>(rgb_h);
    Read<int>(rgb_w);
    K.setIdentity();
    Read<float>(K(0));
    Read<float>(K(1));
    Read<float>(K(2));
    Read<float>(K(3));
    return true;
}

bool CalibOPPOBinReader::GetRGB_K(Eigen::Matrix3f& K, int& rgb_w, int& rgb_h){
    Eigen::Vector4f K_vec;
    GetRGB_K(K_vec, rgb_w, rgb_h);
    K.setIdentity();
    K(0,0) = K_vec(0);
    K(1,1) = K_vec(1);
    K(0,2) = K_vec(2);
    K(1,2) = K_vec(3);

    return true;
}

bool CalibOPPOBinReader::GetToF_K(Eigen::Vector4f& K, int& tof_w, int& tof_h){
    fseek(fp, TOF_START, SEEK_SET);
    Read<int>(tof_h);
    Read<int>(tof_w);
    K.setIdentity();
    Read<float>(K(0));
    Read<float>(K(1));
    Read<float>(K(2));
    Read<float>(K(3));
    return true;
}

bool CalibOPPOBinReader::GetToF_K(Eigen::Matrix3f& K, int& tof_w, int& tof_h){
    Eigen::Vector4f K_vec;
    GetToF_K(K_vec, tof_w, tof_h);
    K.setIdentity();
    K(0,0) = K_vec(0);
    K(1,1) = K_vec(1);
    K(0,2) = K_vec(2);
    K(1,2) = K_vec(3);
    return true;
}

bool CalibIPadTxtReader::GetRT(Eigen::Matrix4f &RT) {
    RT.setIdentity();
    return true;
}

bool CalibIPadTxtReader::GetRGB_K(Eigen::Vector4f& K, int& rgb_w, int& rgb_h) {
    Eigen::Matrix3f k;
    GetRGB_K(k, rgb_w, rgb_h);
    K(0) = k(0, 0);
    K(1) = k(1, 1);
    K(2) = k(0, 2);
    K(3) = k(1, 2);
    return true;
}

bool CalibIPadTxtReader::GetRGB_K(Eigen::Matrix3f& K, int& rgb_w, int& rgb_h) {
    rgb_w = IPAD_RGB_WIDTH;
    rgb_h = IPAD_RGB_HEIGHT;
    fseek(fp, 0, SEEK_SET);
    fread(K.data(), sizeof(float), 9, fp);
    K.transposeInPlace();
    return true;
}

bool CalibIPadTxtReader::GetToF_K(Eigen::Vector4f& K, int& tof_w, int& tof_h) {
    Eigen::Matrix3f k;
    GetToF_K(k, tof_w, tof_h);
    K(0) = k(0, 0);
    K(1) = k(1, 1);
    K(2) = k(0, 2);
    K(3) = k(1, 2);
    return true;
}

bool CalibIPadTxtReader::GetToF_K(Eigen::Matrix3f& K, int& tof_w, int& tof_h) {
    tof_w = IPAD_TOF_WIDTH;
    tof_h = IPAD_TOF_HEIGHT;
    fseek(fp, 0, SEEK_SET);
    fread(K.data(), sizeof(float), 9, fp);
    K.transposeInPlace();
    K.row(0) *= 1.0 * IPAD_TOF_WIDTH / IPAD_RGB_WIDTH;
    K.row(1) *= 1.0 * IPAD_TOF_HEIGHT / IPAD_RGB_HEIGHT;
    return true;
}


CalibTxtReader::CalibTxtReader(){}
CalibTxtReader::CalibTxtReader(std::string path){
    calib_path = path;
    char e = calib_path[calib_path.size()-1];
//    printf("%c\n", e);
    if(e!='/') calib_path += '/';
}


bool CalibTxtReader::GetRT(Eigen::Matrix4f &RT){
    std::ifstream file;
    file.open(calib_path  + "RT.txt");
    if (!file.is_open()) return false;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++)
            file >> RT(i, j);
    }
    return true;
}
bool CalibTxtReader::GetRGB_K(Eigen::Matrix3f& K, int& rgb_w, int& rgb_h){
    std::ifstream file(calib_path  + "K_rgb.txt");
    if (!file.is_open()) return false;
    file >> rgb_w >> rgb_h;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++)
            file >> K(i, j);
    }
    file.close();
    return true;
}
bool CalibTxtReader::GetToF_K(Eigen::Matrix3f& K, int& tof_w, int& tof_h){
    std::ifstream file(calib_path  + "K_tof.txt");
    if (!file.is_open()) return false;
    file >> tof_w >> tof_h;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++)
            file >> K(i, j);
    }
    file.close();
    return true;
}

std::shared_ptr<CalibBinReader> GetCalibBinReaderFromName(const std::string & name) {
    if (boost::filesystem::path(name).extension().string() == ".bin") {
        return std::make_shared<sensemap::CalibOPPOBinReader>();
    } else if (boost::filesystem::path(name).extension().string() == ".txt") {
        return std::make_shared<sensemap::CalibIPadTxtReader>();
    } else {
        std::cerr << "Unknown rgbd_parmas_file format." << std::endl;
        std::abort();
    }
}

std::vector<float> RGBDParamsToVector(const std::string& rgbd_params) {
    std::vector<std::string> strparams = StringSplit(rgbd_params, ",");
    std::vector<float> params;
    for (auto strparam : strparams) {
        params.push_back(std::atof(strparam.c_str()));
    }
    return params;
}

bool RGBDData::HasRGBCalibration(
    
)  {
    return color_camera.ModelId() != kInvalidCameraModelId;
}

bool RGBDData::HasRGBDCalibration(
    
)  {
    return color_camera.ModelId() != kInvalidCameraModelId && 
           depth_camera.ModelId() != kInvalidCameraModelId && 
           !depth_RT.isZero();
}

bool RGBDData::ReadRGBDCameraParams(
    const std::string & rgbd_camera_params_string
) {
    if (!rgbd_camera_params_string.empty()) {
        std::vector<float> rgbd_camera_params =
            RGBDParamsToVector(rgbd_camera_params_string);
        int rgb_width = rgbd_camera_params[0];
        int rgb_height = rgbd_camera_params[1];
        Eigen::Matrix3f rgb_K, depth_K;
        memcpy(rgb_K.data(), rgbd_camera_params.data() + 2, 9 * sizeof(float));
        rgb_K.transposeInPlace();
        memcpy(depth_K.data(), rgbd_camera_params.data() + 11, 9 * sizeof(float));
        depth_K.transposeInPlace();

        Eigen::Matrix4f RT;
        memcpy(RT.data(), rgbd_camera_params.data() + 20, 16 * sizeof(float));
        RT.transposeInPlace();

        this->color_camera.SetModelIdFromName("PINHOLE");
        if (this->color.Width() > 0) {
            float width_scale = this->color.Width() * 1.0f / rgb_width;
            float height_scale = this->color.Height() * 1.0f / rgb_height;
            rgb_K.row(0) *= width_scale;
            rgb_K.row(1) *= height_scale;
            this->color_camera.SetWidth(this->color.Width());
            this->color_camera.SetHeight(this->color.Height());
        } else {
            this->color_camera.SetWidth(rgb_width);
            this->color_camera.SetHeight(rgb_height);
        }
        this->color_camera.SetFocalLengthX(rgb_K(0, 0));
        this->color_camera.SetFocalLengthY(rgb_K(1, 1));
        this->color_camera.SetPrincipalPointX(rgb_K(0, 2));
        this->color_camera.SetPrincipalPointY(rgb_K(1, 2));

        this->depth_camera.SetModelIdFromName("PINHOLE");
        this->depth_camera.SetWidth(this->depth.GetWidth());
        this->depth_camera.SetHeight(this->depth.GetHeight());
        this->depth_camera.SetFocalLengthX(depth_K(0, 0));
        this->depth_camera.SetFocalLengthY(depth_K(1, 1));
        this->depth_camera.SetPrincipalPointX(depth_K(0, 2));
        this->depth_camera.SetPrincipalPointY(depth_K(1, 2));

        this->depth_RT = RT.cast<double>();
    }

    return true;
}

double GetBestScaleByStatitics(const std::vector<std::pair<double, double>> & scale_candidates0, double threshold) {
    if (scale_candidates0.size() < 10) return -1;

    std::vector<std::pair<double, double>> scale_candidates(scale_candidates0.begin(), scale_candidates0.end());
    std::sort(scale_candidates.begin(), scale_candidates.end(), ScaleCandidateComparer);

    // median filter
    {
        const size_t scale_count_old = scale_candidates.size();

        std::vector<std::pair<double, double>> new_scale_candidates;
        new_scale_candidates.insert(new_scale_candidates.end(), 
            scale_candidates.begin() + scale_count_old * 0.1,
            scale_candidates.begin() + scale_count_old * 0.9);
        std::swap(new_scale_candidates, scale_candidates);

        std::cout << "scale candidates filter: " << scale_count_old << " => " << scale_candidates.size() << std::endl;
    }

    // statistic
    double total_weight_count = 0.0;
    double best_weight_count = 0.0;
    double best_scale = -1;
    #pragma omp parallel for schedule(dynamic, 16) reduction(+: total_weight_count)
    for(int i = 0; i < scale_candidates.size(); i++)
    {
        const double inv_candidate = 1.0 / std::fabs(scale_candidates[i].first);
        total_weight_count += scale_candidates[i].second;

        double scale_sum           = 0.0;
        double scale_weights       = 0.0;
        double scale_total_weights = 0.0;
        for (int j = i; j >= 0; j--) {
            const double diff = std::fabs(scale_candidates[i].first - scale_candidates[j].first) * inv_candidate;
            if (diff <= threshold * 0.5) {
                scale_sum           += scale_candidates[j].second * scale_candidates[j].first;
                scale_weights       += scale_candidates[j].second;
                scale_total_weights += scale_candidates[j].second;
            } else if (diff < threshold) {
                scale_total_weights += scale_candidates[j].second;
            } else {
                break;
            }
        }
        for (int j = i + 1; j < scale_candidates.size(); j++) {
            const double diff = std::fabs(scale_candidates[i].first - scale_candidates[j].first) * inv_candidate;
            if (diff <= threshold * 0.5) {
                scale_sum           += scale_candidates[j].second * scale_candidates[j].first;
                scale_weights       += scale_candidates[j].second;
                scale_total_weights += scale_candidates[j].second;
            } else if (diff < threshold) {
                scale_total_weights += scale_candidates[j].second;
            } else {
                break;
            }
        }

        if (scale_total_weights > best_weight_count)
        {
            #pragma omp critical
            {
                // dual check is necessary
                if (scale_total_weights > best_weight_count)
                {
                    best_weight_count = scale_total_weights;
                    best_scale = scale_sum / scale_weights;
                }
            }
        }
    }
    double scale = best_scale;
    printf("The %.2f%% scales in inliner\n", best_weight_count / total_weight_count * 100.f);
    std::cout << "scale:" << 1.0 / scale << std::endl;

    return 1.0 / scale;
}

bool IntrinsicStringToCamera(const std::string & str, Camera & camera) {
    camera = Camera();
    auto vec = CSVToVector<std::string>(str);
    if (vec.size() == 0) {
        return false;
    }

    std::string params = "";
    for (int i = 1; i < vec.size(); i++) {
        if (i != 1) params += ",";
        params += vec[i];
    }

    camera.SetModelIdFromName(vec[0]);
    camera.SetParamsFromString(params);

    return true;
}

bool CameraToIntrinsicString(const Camera & camera, std::string & str) {
    if (camera.ModelId() == kInvalidCameraModelId) return false;

    std::stringstream ss;
    ss << camera.ModelName();
    ss << std::setprecision(9);
    for (double p : camera.Params()) {
        ss << ", " << p;
    }

    str = ss.str();
    return true;
}

bool IsFileRGBD(const std::string & image_path) {
    boost::filesystem::path path = image_path;
    // return (path.extension().string() == ".bin") || 
    //        (path.extension().string() == ".binx");
    return (path.extension().string() == ".binx");
}

bool WriteRGBDData(
    const std::string & image_path, 
    const RGBDWriteData & data
) {
    boost::filesystem::path path = image_path;
    if (path.extension().string() == ".bin") {
        return WriteRGBDBinData(image_path, data);
    } else if (path.extension().string() == ".binx") {
        return WriteRGBDBinxData(image_path, data);
    } else {
        return false;
    }
}

bool ExtractRGBDData(
    const std::string & image_path,
    const RGBDReadOption & option,
    RGBDData & data
) {
    boost::filesystem::path path = image_path;
    if (path.extension().string() == ".bin") {
        if (!ExtractRGBDBinData(image_path, option, data)) return false;
    } else if (path.extension().string() == ".binx") {
        if (!ExtractRGBDBinxData(image_path, option, data)) return false;
    } else {
        return false;
    }

    for (int y = 0; y < data.depth.GetHeight(); y++) {
        for (int x = 0; x < data.depth.GetWidth(); x++) {
            if (!(data.depth.Get(y, x) <= MAX_VALID_DEPTH_IN_M &&
                  data.depth.Get(y, x) >= 0)
            ) {
                data.depth.Set(y, x, 0.0f);
            }
        }
    }

    return true;
}

bool ExtractRGBDData(
    const std::string & image_path, 
    RGBDData & data
) {
    RGBDReadOption option;
    return ExtractRGBDData(image_path, option, data);
}

bool ExtractRGBDData(
    const std::string & image_path, 
    Bitmap & bitmap,
    bool as_rgb
) {
    RGBDReadOption option;
    option.with_depth = false;
    option.color_as_rgb = as_rgb;

    RGBDData data;
    if (!ExtractRGBDData(image_path, option, data)) return false;

    bitmap = std::move(data.color);
    return true;
}

bool ExtractRGBDData(
    const std::string & image_path, 
    Bitmap & bitmap, 
    MatXf & depthmap, 
    bool as_rgb
) {
    RGBDReadOption option;
    option.color_as_rgb = as_rgb;

    RGBDData data;
    if (!ExtractRGBDData(image_path, option, data)) return false;

    bitmap = std::move(data.color);
    depthmap = std::move(data.depth);
    return true; 
}

bool WriteRGBDBinxData(
    const std::string & image_path, 
    const RGBDWriteData & data
) {
    auto fp = gzopen(image_path.c_str(), "wb");
    if (!fp) {
        std::cerr << "Failed to open " << image_path << std::endl;
        return false;
    }

    std::unique_ptr<gzFile_s, decltype(gzclose) *> fp_guard(fp, gzclose);

    // Write magic number
    RGBDBinxHelper::WriteMagicNumber(fp);

    // Write color data
    if (data.color.Width() > 0 && data.color.Height() > 0) {
        if (!RGBDBinxHelper::WriteColorData(fp, data.color)) {
            return false;
        }
    } else if (data.color_mat.rows > 0 && data.color_mat.cols > 0) {
        if (!RGBDBinxHelper::WriteColorData(fp, data.color_mat, data.color_type_hint)) {
            return false;
        }
    } else {
        std::cerr << "At least color data is required." << std::endl;
        return false;
    }

    // Write color info
    if (data.color_timestamp > 0) {
        if (!RGBDBinxHelper::WriteColorTimestamp(fp, data.color_timestamp)) {
            return false;
        }
    }
    if (!data.color_camera.ModelId() != kInvalidCameraModelId) {
        if (!RGBDBinxHelper::WriteColorIntrinsicString(fp, data.color_camera)) {
            return false;
        }
    }

    // Write depth
    if (data.depth.GetWidth() > 0 && data.depth.GetHeight() > 0) {
        if (!RGBDBinxHelper::WriteDepthData(fp, data.depth)) {
            return false;
        }
    } else if (data.depth_mat.rows > 0 && data.depth_mat.cols > 0) {
        if (!RGBDBinxHelper::WriteDepthData(fp, data.depth_mat)) {
            return false;
        }
    }

    // Write depth info
    if (data.depth_timestamp > 0) {
        if (!RGBDBinxHelper::WriteDepthTimestamp(fp, data.depth_timestamp)){
            return false;
        }
    }
    if (data.depth_camera.ModelId() != kInvalidCameraModelId) {
        if (!RGBDBinxHelper::WriteDepthIntrinsicString(fp, data.depth_camera)){
            return false;
        }
    }
    if (!data.depth_RT.isZero()) {
        if (!RGBDBinxHelper::WriteDepthRT(fp, data.depth_RT)){
            return false;
        }
    }

    // Write frame info
    if (data.timestamp > 0) {
        if (!RGBDBinxHelper::WriteFrameTimestamp(fp, data.timestamp)){
            return false;
        }
    }
    if (data.orientation > 0) {
        if (!RGBDBinxHelper::WriteFrameOrientation(fp, data.orientation)){
            return false;
        }
    }
    if (!data.gravity.isZero()) {
        if (!RGBDBinxHelper::WriteFrameGravity(fp, data.gravity)){
            return false;
        }
    }

    return true;
}

bool ExtractRGBDBinxData(
    const std::string & image_path, 
    const RGBDReadOption & option, 
    RGBDData & data
) {
    auto fp = gzopen(image_path.c_str(), "rb");
    if (!fp) {
        std::cerr << "Failed to open " << image_path << std::endl;
        return false;
    }

    std::unique_ptr<gzFile_s, decltype(gzclose) *> fp_guard(fp, gzclose);

    int code;
    if (!RGBDBinxHelper::CheckMagicNumber(fp)) {
        return false;
    }

    while (RGBDBinxHelper::ReadNextCode(fp, code)) {
        if (!RGBDBinxHelper::ReadNext(fp, code, option, data)) {
            std::cerr << "Binx file broken" << std::endl;
            return false;
        }
    }

    if (data.color_camera.ModelId() != kInvalidCameraModelId && data.color.Width() > 0) {
        data.color_camera.SetWidth(data.color.Width());
        data.color_camera.SetHeight(data.color.Height());
    }

    if (data.depth_camera.ModelId() != kInvalidCameraModelId && data.depth.GetWidth() > 0) {
        data.depth_camera.SetWidth(data.depth.GetWidth());
        data.depth_camera.SetHeight(data.depth.GetHeight());
    }

    return true;
}

bool ExtractRGBDBinData(
    const std::string & image_path,
    const RGBDReadOption & option,
    RGBDData & rgbd_data
) {
    std::ifstream file(image_path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "fail to open data file" << std::endl;
        return false;
    }

    file.seekg (0, file.end);
    int length = file.tellg();
    file.seekg (0, file.beg);

    /// rgb
    int width, height;
    int format, stride;
    int orientation;
    double timestamp;
    file.read((char *) &width, sizeof(width));
    file.read((char *) &height, sizeof(height));
    file.read((char *) &format, sizeof(format));
    file.read((char *) &stride, sizeof(stride));
    file.read((char *) &orientation, sizeof(orientation));
    file.read((char *) &timestamp, sizeof(timestamp));
    rgbd_data.orientation = orientation;
    rgbd_data.color_timestamp = timestamp;

    if (stride == 0 || stride >= width && stride < width * 3) {
        // YUV
        const int color_size = width * height * 3 / 2;
        if (option.with_color) {
            std::vector<unsigned char> data(color_size);
            file.read((char *)data.data(), sizeof(char) * color_size);

            cv::Mat yuv(height * 3 / 2, width, CV_8UC1, data.data());
            cv::Mat rgb;
            if (option.color_as_rgb) {
                cv::cvtColor(yuv, rgb, cv::COLOR_YUV2RGB_NV21);
            } else {
                cv::cvtColor(yuv, rgb, cv::COLOR_YUV2GRAY_NV21);
            }

            rgbd_data.color.Allocate(width, height, option.color_as_rgb);
            Mat2FreeImage(rgb, &rgbd_data.color);
        } else {
            rgbd_data.color.Allocate(width, height, option.color_as_rgb);
            file.seekg(color_size, file.cur);
        }
    } else if (stride >= width * 3) {
        // BGR
        const int color_size = width * height * 3;
        if (option.with_color) {
            std::vector<unsigned char> data(color_size);
            file.read((char *)data.data(), sizeof(char) * color_size);

            cv::Mat rgb(height, width, CV_8UC3, data.data());
            if (!option.color_as_rgb) {
                cv::cvtColor(rgb, rgb, cv::COLOR_BGR2GRAY);
            }

            rgbd_data.color.Allocate(width, height, option.color_as_rgb);
            Mat2FreeImage(rgb, &rgbd_data.color);
        } else {
            rgbd_data.color.Allocate(width, height, option.color_as_rgb);
            file.seekg(color_size, file.cur);
        }
    } else {
        std::cerr << "unknown color format" << std::endl;
        return false;
    }

    /// depth
    file.read((char *) &width, sizeof(width));
    file.read((char *) &height, sizeof(height));
    file.read((char *) &format, sizeof(format));
    file.read((char *) &stride, sizeof(stride));
    file.read((char *) &orientation, sizeof(orientation));
    file.read((char *) &timestamp, sizeof(timestamp));
    rgbd_data.depth_timestamp = timestamp;

    if (stride >= width * sizeof(float)) {
        // iPad f32 depth
        const int depth_size = sizeof(float) * width * height;
        if (option.with_depth) {
            std::vector<float> vdepthmap(width * height);
            file.read((char *)vdepthmap.data(), depth_size);
            rgbd_data.depth = MatXf(width, height, 1);
            for (int i = 0; i < height; ++i) {
                for (int j = 0; j < width; ++j) {
                    float depth = vdepthmap.at(i * width + j);
                    rgbd_data.depth.Set(i, j, depth);
                }
            }
        } else {
            rgbd_data.depth = MatXf(width, height, 1);
            file.seekg(depth_size, file.cur);
        }
    } else if (stride >= width * sizeof(ushort) || stride == 0) {
        // oppo u16 depth
        const int depth_size = sizeof(ushort) * width * height;
        if (option.with_depth) {
            std::vector<ushort> vdepthmap(width * height);
            file.read((char *)vdepthmap.data(), depth_size);
            rgbd_data.depth = MatXf(width, height, 1);
            for (int i = 0; i < height; ++i) {
                for (int j = 0; j < width; ++j) {
                    unsigned short depth = vdepthmap.at(i * width + j);
                    rgbd_data.depth.Set(i, j, depth / 1000.0);
                }
            }
        } else {
            rgbd_data.depth = MatXf(width, height, 1);
            file.seekg(depth_size, file.cur);
        }
    } else {
        std::cerr << "unknown depth format" << std::endl;
        return false;
    }

    // For OPPOR17 Pro.
    {
        float matrix[16];
        /// model pose
        file.read((char *) matrix, sizeof(float) * 16);

        /// imu data
        int imu_size;
        file.read((char *) &imu_size, sizeof(imu_size));

        for (int i = 0; i < imu_size; ++ i)
        {
            // index [0-2] is raw data, index [3-5] is bias
            double acc_data[6];
            file.read((char *)acc_data, sizeof(double) * 6);

            // index [0-2] is raw data, index [3-5] is bias
            double gyr_data[6];
            file.read((char *)gyr_data, sizeof(double) * 6);

            double timestamp;
            file.read((char *)&timestamp, sizeof(timestamp));
        }

        double r_data[4], g_data[3];
        file.read((char *)r_data, sizeof(double) * 4);
        file.read((char *)g_data, sizeof(double) * 3);


        static Eigen::Quaterniond Calib_QIC = Eigen::Quaterniond{0.0, 0.707107, -0.707107, 0.0};
        Calib_QIC.normalize();
        rgbd_data.gravity<<g_data[0], g_data[1], g_data[2];
        rgbd_data.gravity.applyOnTheLeft(Calib_QIC.toRotationMatrix().transpose());
        rgbd_data.gravity.normalize();

        double timestamp;
        file.read((char *)&timestamp, sizeof(double));
        rgbd_data.timestamp = timestamp;
    }
    return true;
}

cv::Mat ImageConvertBGR2NV12(const cv::Mat &rgb) {
    cv::Mat yuv_yv12;
    cv::Mat yuv_nv12 = cv::Mat(rgb.rows * 3 / 2, rgb.cols, CV_8UC1);
    cv::cvtColor(rgb, yuv_yv12, cv::COLOR_BGR2YUV_YV12);
    for (int y = 0; y < rgb.rows; y++) {
        for (int x = 0; x < rgb.cols; x++) {
            yuv_nv12.at<uchar>(y, x) = yuv_yv12.at<uchar>(y, x);
        }
    }
    for (int y = 0; y < rgb.rows / 2; y++) {
        for (int x = 0; x < rgb.cols; x++) {
            if (x % 2 == 1) {
                yuv_nv12.at<uchar>(rgb.rows + y, x) =
                        yuv_yv12.at<uchar>(rgb.rows + y / 2, x / 2);
            }
            else {
                yuv_nv12.at<uchar>(rgb.rows + y, x) =
                        yuv_yv12.at<uchar>(rgb.rows + rgb.rows / 4 + y / 2, x / 2);
            }
        }
    }

    return yuv_nv12;
}

bool WriteRGBDBinData(
    const std::string & image_path, 
    const RGBDWriteData & data
) {
    std::cerr << "Bin format writing no longer supported. " << std::endl;
    return false;
    // std::ofstream file(image_path, std::ios::binary);
    // if (!file.is_open()) {
    //     std::cout << "fail to open data file" << std::endl;
    //     return false;
    // }

    // /// rgb
    // if (color.type() == CV_8UC1) {
    //     int height = color.rows * 2 / 3;
    //     int width = color.cols;
    //     int format = 0, stride = std::min(width * sizeof(char), (size_t)color.step);
    //     int orientation = orientation_u16;

    //     file.write((const char *) &width, sizeof(width));
    //     file.write((const char *) &height, sizeof(height));
    //     file.write((const char *) &format, sizeof(format));
    //     file.write((const char *) &stride, sizeof(stride));
    //     file.write((const char *) &orientation, sizeof(orientation));
    //     file.write((const char *) &rgb_timestamp, sizeof(rgb_timestamp));
        
    //     for (int i = 0; i < color.rows; i++) {
    //         file.write((const char *)color.ptr(i), sizeof(char) * width);
    //     }
    // } else if (color.type() == CV_8UC3) {
    //     int height = color.rows;
    //     int width = color.cols;
    //     int format = 0, stride = std::min(width * sizeof(char) * 3, (size_t)color.step);
    //     int orientation = orientation_u16;

    //     file.write((const char *) &width, sizeof(width));
    //     file.write((const char *) &height, sizeof(height));
    //     file.write((const char *) &format, sizeof(format));
    //     file.write((const char *) &stride, sizeof(stride));
    //     file.write((const char *) &orientation, sizeof(orientation));
    //     file.write((const char *) &rgb_timestamp, sizeof(rgb_timestamp));
        
    //     for (int i = 0; i < color.rows; i++) {
    //         file.write((const char *)color.ptr(i), sizeof(char) * 3 * width);
    //     }
    // } else {
    //     std::cout << "unknown color format" << std::endl;
    //     return false;
    // }

    // /// depth
    // if (depth.type() == CV_16UC1) {
    //     int height = depth.rows;
    //     int width = depth.cols;
    //     int format = 0, stride = std::min(width * sizeof(ushort), (size_t)depth.step);
    //     int orientation = orientation_u16;

    //     file.write((const char *) &width, sizeof(width));
    //     file.write((const char *) &height, sizeof(height));
    //     file.write((const char *) &format, sizeof(format));
    //     file.write((const char *) &stride, sizeof(stride));
    //     file.write((const char *) &orientation, sizeof(orientation));
    //     file.write((const char *) &depth_timestamp, sizeof(depth_timestamp));

    //     for (int i = 0; i < depth.rows; i++) {
    //         file.write((const char *)depth.ptr(i), sizeof(ushort) * width);
    //     }
    // } else if (depth.type() == CV_32FC1) {
    //     int height = depth.rows;
    //     int width = depth.cols;
    //     int format = 0, stride = std::min(width * sizeof(float), (size_t)depth.step);
    //     int orientation = orientation_u16;

    //     file.write((const char *) &width, sizeof(width));
    //     file.write((const char *) &height, sizeof(height));
    //     file.write((const char *) &format, sizeof(format));
    //     file.write((const char *) &stride, sizeof(stride));
    //     file.write((const char *) &orientation, sizeof(orientation));
    //     file.write((const char *) &depth_timestamp, sizeof(depth_timestamp));

    //     for (int i = 0; i < depth.rows; i++) {
    //         file.write((const char *)depth.ptr(i), sizeof(float) * width);
    //     }
    // } else {
    //     std::cout << "unknown depth format" << std::endl;
    //     return false;
    // }

    // // For OPPOR17 Pro.
    // {
    //     float matrix[16] = { 0 }; // not implemented
    //     /// model pose
    //     file.write((const char *) matrix, sizeof(float) * 16);
        
    //     /// imu data
    //     int imu_size = 0;   // not implemented
    //     file.write((const char *) &imu_size, sizeof(imu_size));

    //     double r_data[4] = { 0 }, g_data[3] = { 0 };
    //     file.write((const char *)r_data, sizeof(double) * 4);
    //     file.write((const char *)gravity.data(), sizeof(double) * 3);

    //     file.write((const char *)&rgb_timestamp, sizeof(double));
    // }

    // return true;
}

void ResizeDepthMap(MatXf &resized_depthmap,
                    const MatXf &depthmap) {
    const int raw_width = depthmap.GetWidth();
    const int raw_height = depthmap.GetHeight();
    const int resized_width = resized_depthmap.GetWidth();
    const int resized_height = resized_depthmap.GetHeight();

    const float width_scale = raw_width * 1.0 / resized_width;
    const float height_scale = raw_height * 1.0 / resized_height;

    resized_depthmap.Fill(0.0f);
    for (int r = 0; r < resized_height; ++r) {
        float yf = r * height_scale;
        int y0 = std::floor(yf);
        int y1 = std::ceil(yf);
        if (y0 < 0 || y1 >= raw_height) continue;

        for (int c = 0; c < resized_width; ++c) {
            float xf = c * width_scale;
            int x0 = std::floor(xf);
            int x1 = std::ceil(xf);
            if (x0 < 0 || x1 >= raw_width) continue;

            float depth = 0.0f;
            float depth_weight = 0.0f;
            float depth00 = depthmap.Get(y0, x0);
            float depth01 = depthmap.Get(y0, x1);
            float depth10 = depthmap.Get(y1, x0);
            float depth11 = depthmap.Get(y1, x1);
            if (depth00 > 0.0f) {
                float w  = 1.0f - (xf - x0);
                      w *= 1.0f - (yf - y0);
                depth += w * depth00;
                depth_weight += w;
            }
            if (depth01 > 0.0f) {
                float w  = 1.0f - (xf - x0);
                      w *= 1.0f - (y1 - yf);
                depth += w * depth01;
                depth_weight += w;
            }
            if (depth10 > 0.0f) {
                float w  = 1.0f - (x1 - xf);
                      w *= 1.0f - (yf - y0);
                depth += w * depth10;
                depth_weight += w;
            }
            if (depth11 > 0.0f) {
                float w  = 1.0f - (x1 - xf);
                      w *= 1.0f - (y1 - yf);
                depth += w * depth11;
                depth_weight += w;
            }

            if (depth_weight == 0.0f) continue;
            depth /= depth_weight;

            resized_depthmap.Set(r, c, depth);
        }
    }
}

void FastWarpDepthMap(MatXf &warped_depthmap,
                      const MatXf &depthmap, 
                      const Eigen::Matrix3f &rgb_K,
					  const Eigen::Matrix3f &depth_K,
					  const Eigen::Matrix4f &RT) {
    const int raw_width = depthmap.GetWidth();
    const int raw_height = depthmap.GetHeight();
    const int warped_width = warped_depthmap.GetWidth();
    const int warped_height = warped_depthmap.GetHeight();
    const int temp_width = warped_width * 1.5;
    const int temp_height = warped_height * 1.5;

    const float width_scale = raw_width * 1.0 / temp_width;
    const float height_scale = raw_height * 1.0 / temp_height;

    Eigen::Matrix3f warp_depth_K = depth_K;
    warp_depth_K.row(0) /= width_scale;
    warp_depth_K.row(1) /= height_scale;

	Eigen::Matrix3f warp_R = rgb_K * RT.block<3, 3>(0, 0) * warp_depth_K.inverse();
    Eigen::Vector3f warp_T = rgb_K * RT.block<3, 1>(0, 3) / 1000.0;

    warped_depthmap.Fill(0.0f);
    for (int r = 0; r < temp_height; ++r) {
        float yf = r * height_scale;
        int y0 = std::floor(yf);
        int y1 = std::ceil(yf);
        if (y0 < 0 || y1 >= raw_height) continue;

        for (int c = 0; c < temp_width; ++c) {
            float xf = c * width_scale;
            int x0 = std::floor(xf);
            int x1 = std::ceil(xf);
            if (x0 < 0 || x1 >= raw_width) continue;

            float depth = 0.0f;
            float depth_weight = 0.0f;
            float depth00 = depthmap.Get(y0, x0);
            float depth01 = depthmap.Get(y0, x1);
            float depth10 = depthmap.Get(y1, x0);
            float depth11 = depthmap.Get(y1, x1);
            if (depth00 > 0.0f) {
                float w  = 1.0f - (xf - x0);
                      w *= 1.0f - (yf - y0);
                depth += w * depth00;
                depth_weight += w;
            }
            if (depth01 > 0.0f) {
                float w  = 1.0f - (xf - x0);
                      w *= 1.0f - (y1 - yf);
                depth += w * depth01;
                depth_weight += w;
            }
            if (depth10 > 0.0f) {
                float w  = 1.0f - (x1 - xf);
                      w *= 1.0f - (yf - y0);
                depth += w * depth10;
                depth_weight += w;
            }
            if (depth11 > 0.0f) {
                float w  = 1.0f - (x1 - xf);
                      w *= 1.0f - (y1 - yf);
                depth += w * depth11;
                depth_weight += w;
            }

            if (depth_weight == 0.0f) continue;
            depth /= depth_weight;

            Eigen::Vector3f proj = 
                warp_R * Eigen::Vector3f(c, r, 1) * depth + warp_T;
            int x = proj[0] / proj[2] + 0.5;
            int y = proj[1] / proj[2] + 0.5;
            if (x < 0 || x >= warped_width || y < 0 || y >= warped_height) {
                continue;
            }
            float warp_depth = warped_depthmap.Get(y, x);
            if (warp_depth == 0 || warp_depth > proj[2]) {
                warped_depthmap.Set(y, x, proj[2]);
            }
        }
    }
}

void UniversalWarpDepthMap(MatXf &warped_depthmap,
                           const MatXf &depthmap, 
                           const Camera &rgb_camera,
                           const Camera &depth_camera,
                           const Eigen::Matrix4f &RT) {
    if ((rgb_camera.ModelName() == "PINHOLE" || rgb_camera.ModelName() == "SIMPLE_PINHOLE") &&
        (depth_camera.ModelName() == "PINHOLE" || depth_camera.ModelName() == "SIMPLE_PINHOLE")
    ) {
        Eigen::Matrix3d color_K = rgb_camera.CalibrationMatrix();
        Eigen::Matrix3d depth_K = depth_camera.CalibrationMatrix();
        FastWarpDepthMap(warped_depthmap, depthmap, color_K.cast<float>(), depth_K.cast<float>(), RT);
    }

    const int raw_width = depthmap.GetWidth();
    const int raw_height = depthmap.GetHeight();
    const int warped_width = warped_depthmap.GetWidth();
    const int warped_height = warped_depthmap.GetHeight();
    const int temp_width = warped_width * 1.5;
    const int temp_height = warped_height * 1.5;

    const float width_scale = raw_width * 1.0 / temp_width;
    const float height_scale = raw_height * 1.0 / temp_height;

	Eigen::Matrix3d warp_R = RT.block<3, 3>(0, 0).cast<double>();
    Eigen::Vector3d warp_T = RT.block<3, 1>(0, 3).cast<double>() / 1000.0;

    warped_depthmap.Fill(0.0f);
    for (int r = 0; r < temp_height; ++r) {
        float yf = r * height_scale;
        int y0 = std::floor(yf);
        int y1 = std::ceil(yf);
        if (y0 < 0 || y1 >= raw_height) continue;

        for (int c = 0; c < temp_width; ++c) {
            float xf = c * width_scale;
            int x0 = std::floor(xf);
            int x1 = std::ceil(xf);
            if (x0 < 0 || x1 >= raw_width) continue;

            float depth = 0.0f;
            float depth_weight = 0.0f;
            float depth00 = depthmap.Get(y0, x0);
            float depth01 = depthmap.Get(y0, x1);
            float depth10 = depthmap.Get(y1, x0);
            float depth11 = depthmap.Get(y1, x1);
            if (depth00 > 0.0f) {
                float w  = 1.0f - (xf - x0);
                      w *= 1.0f - (yf - y0);
                depth += w * depth00;
                depth_weight += w;
            }
            if (depth01 > 0.0f) {
                float w  = 1.0f - (xf - x0);
                      w *= 1.0f - (y1 - yf);
                depth += w * depth01;
                depth_weight += w;
            }
            if (depth10 > 0.0f) {
                float w  = 1.0f - (x1 - xf);
                      w *= 1.0f - (yf - y0);
                depth += w * depth10;
                depth_weight += w;
            }
            if (depth11 > 0.0f) {
                float w  = 1.0f - (x1 - xf);
                      w *= 1.0f - (y1 - yf);
                depth += w * depth11;
                depth_weight += w;
            }

            if (depth_weight == 0.0f) continue;
            depth /= depth_weight;

            Eigen::Vector2d depth_pt_image_2d(xf, yf);
            Eigen::Vector3d depth_pt_world_3d = depth_camera.ImageToWorld(depth_pt_image_2d).homogeneous();
            Eigen::Vector3d rgb_pt_world_3d = warp_R * depth_pt_world_3d * depth + warp_T;
            Eigen::Vector2d rgb_pt_image_2d = rgb_camera.WorldToImage(rgb_pt_world_3d.hnormalized());

            int x = rgb_pt_image_2d.x() + 0.5;
            int y = rgb_pt_image_2d.y() + 0.5;
            if (x < 0 || x >= warped_width || y < 0 || y >= warped_height) {
                continue;
            }
            float warp_depth = warped_depthmap.Get(y, x);
            if (warp_depth == 0 || warp_depth > rgb_pt_world_3d[2]) {
                warped_depthmap.Set(y, x, rgb_pt_world_3d[2]);
            }
        }
    }
}

void WarpDepthMap2RGB(MatXf &warped_depthmap,
                      const MatXf &depthmap, 
                      const Eigen::Matrix3f &rgb_K,
					  const Eigen::Matrix3f &depth_K,
					  const Eigen::Matrix4f &RT) {
    const int warped_width = warped_depthmap.GetWidth();
    const int warped_height = warped_depthmap.GetHeight();

    Eigen::Matrix3f warp_R = rgb_K * RT.block<3, 3>(0, 0) * depth_K.inverse();
    Eigen::Vector3f warp_T = rgb_K * RT.block<3, 1>(0, 3) / 1000.0;

    warped_depthmap.Fill(0.0f);
    for (int r = 0; r < depthmap.GetHeight(); r++) {
        for (int c = 0; c < depthmap.GetWidth(); c++) {
            float depth = depthmap.Get(r, c);
            if (depth <= 0.0f) {
                continue;
            }

            Eigen::Vector3f proj = warp_R * Eigen::Vector3f(c, r, 1) * depth + warp_T;
            int x = proj[0] / proj[2] + 0.5;
            int y = proj[1] / proj[2] + 0.5;
            if (x < 0 || x >= warped_width || y < 0 || y >= warped_height) {
                continue;
            }

            warped_depthmap.Set(y, x, proj[2]);
        }
    }
}

void MeshWarpDepthMap(MatXf &warped_depthmap,
                      const MatXf &depthmap, 
                      const Eigen::Matrix3f &rgb_K,
					  const Eigen::Matrix3f &depth_K,
					  const Eigen::Matrix4f &RT) {
    DepthMesh mesh;
    GenerateMesh(mesh, depthmap.GetPtr(), depthmap.GetWidth(), depthmap.GetWidth(), depthmap.GetHeight(),
                  depth_K, Eigen::Matrix3f::Identity(), Eigen::Vector3f::Zero(), 1, 0.05, VertexFormat::POSITION, std::numeric_limits<float>::epsilon(), std::numeric_limits<float>::max());
    //WriteTriangleMeshObj(saveMeshName, mesh);
    Eigen::Matrix3f R = RT.block<3, 3>(0, 0);
    Eigen::Vector3f t = RT.block<3, 1>(0, 3)*0.001;
    //const std::string saveImageName = saveMeshName + ".png";
    cv::Mat warped_depth_map;
    const int warped_width = warped_depthmap.GetWidth();
    const int warped_height = warped_depthmap.GetHeight();
    warped_depth_map.create(warped_height, warped_width, CV_32FC1);
    WarpFrameBuffer(warped_depth_map, rgb_K, R, t, &mesh);

    for (int i = 0; i < warped_height; i++) {
        for (int j = 0; j < warped_width; j++) {
            warped_depthmap.Set(i, j, warped_depth_map.at<float>(i, j));
        }
    }
}

void AddWeightedDepthMap(MatXf &warped_depthmap, const MatXf & depthmap,MatXf & depth_weight)
{
    CHECK_EQ(warped_depthmap.GetHeight(),depthmap.GetHeight());
    CHECK_EQ(warped_depthmap.GetWidth(),depthmap.GetWidth());
    for(int y =0;y<depthmap.GetHeight();y++)
    {
        for(int x = 0;x<depthmap.GetWidth();x++)
        {
            if (depthmap.Get(y,x) <= 0.0)
                continue;
            float w = depth_weight.Get(y,x)/(depth_weight.Get(y,x)+1.f);
            float depth = warped_depthmap.Get(y,x)*w+(1-w)*depthmap.Get(y,x);
            depth_weight.Set(y,x,depth_weight.Get(y,x)+1);
            warped_depthmap.Set(y,x,depth);
        }

    }
}

} // namespace