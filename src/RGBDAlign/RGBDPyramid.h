//
// Created by sensetime on 2020/10/29.
//

#ifndef SLAM_RGBDPYRAMID_H
#define SLAM_RGBDPYRAMID_H
#include <Eigen/Eigen>
#include <opencv2/opencv.hpp>
#include <memory>
#include "RGBDAlignUtility.h"

struct XYZImage{

    typedef std::shared_ptr<XYZImage> Ptr;

    XYZImage();
    XYZImage(int w, int h);

    std::vector<Eigen::Vector3f> pts_;
    int width_;
    int height_;
    int sample_step_;
    int count_;

    Eigen::Vector4f K_;

    Eigen::Vector3f operator() (int x, int y) const;
    Eigen::Vector3f& operator() (int x, int y);

    Eigen::Vector3f operator[] (int idx) const;
    Eigen::Vector3f& operator[] (int idx);

};

 void CreateXYZImage(std::shared_ptr<XYZImage> &xyz, cv::Mat depth, Eigen::Vector4f K, int step);


class RGBDPyramid {
public:
    typedef std::shared_ptr<RGBDPyramid> Ptr;

    RGBDPyramid();

    RGBDPyramid(int levels, float scale, Eigen::Vector4f K, cv::Mat gray, cv::Mat depth,
                cv::Mat conf = cv::Mat(), cv::Mat highlight = cv::Mat(), int id = -1);

    ~RGBDPyramid();

    int id_;

    int levels_;

    std::vector<Eigen::Vector4f> pyramid_K_;

    std::vector<cv::Mat> pyramid_gray_;//CV_32FC1
    std::vector<cv::Mat> pyramid_depth_;//CV_32FC1
//    std::vector<XYZImage> pyramid_XYZ_;

    std::vector<cv::Mat> pyramid_gray_dx_;
    std::vector<cv::Mat> pyramid_gray_dy_;

    std::vector<cv::Mat> pyramid_depth_dx_;
    std::vector<cv::Mat> pyramid_depth_dy_;

    std::vector<cv::Mat> pyramid_conf_;
    std::vector<cv::Mat> pyramid_highlight_;

    void UpdateDepth(cv::Mat depth);

    void CreateImagePyramid();

    void AddLevel(int level);
    //    void CreateXYZImage();

    void UpdateIntrinsic(Eigen::Vector4f K);

public:
    cv::Mat orig_gray_; //CV_8UC1
    cv::Mat orig_depth_; //CV_32FC1
    cv::Mat orig_conf_;
    cv::Mat orig_highlight_;
    cv::Mat orig_mask_prob_;

    Eigen::Vector4f orig_K_;
    std::map<int, std::shared_ptr<XYZImage>> xyzs_;

    float scale_ = 2.0;

    void PreProcess();
    cv::Mat DownSampleImage(cv::Mat &input, bool with_gaussian_filter = false, float scale = 2.0);
    cv::Mat DownSampleImage2(cv::Mat &input, bool with_gaussian_filter = false);

    std::shared_ptr<XYZImage> GetXYZ(int level, int sample_step = -1);

    static float gaussian_kernel[9];
    static float sobel_dx_kernel[9];
    static float sobel_dy_kernel[9];

    static cv::Mat cv_gaussian_kernel;
    static cv::Mat cv_dx_kernel;
    static cv::Mat cv_dy_kernel;

    static double sobel_scale;

    void GaussianFilter2D(const cv::Mat & src_mat, cv::Mat & dst_mat);

    void SobelDxDyFilter2D(const cv::Mat & src_mat, cv::Mat & dst_dx_mat, cv::Mat & dst_dy_mat);



};

void SaveDepthAsObj(std::string filename, std::shared_ptr<RGBDPyramid> pyr, Eigen::Matrix4f pose);

#endif //SLAM_RGBDPYRAMID_H
