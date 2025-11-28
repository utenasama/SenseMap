//
// Created by sensetime on 2020/10/30.
//

#ifndef SLAM_RGBDREGISTRATION_H
#define SLAM_RGBDREGISTRATION_H
#include "RGBDPyramid.h"
#include <memory>
#include "RGBDAlignUtility.h"

class RGBDRegistration {
public:

    struct MatchedInfo{
        int corres_num_;
        float residuals_;
        bool BetterThan(const MatchedInfo &info2);
    };

    RGBDRegistration();
    RGBDRegistration(int levels, std::vector<int> sample_steps, double max_depth_diff, double max_color_diff);
    ~RGBDRegistration();

    std::shared_ptr<RGBDPyramid> src_= nullptr;
    std::shared_ptr<RGBDPyramid> dst_= nullptr;

    int levels_;
    std::vector<int> sample_steps_;

    double max_depth_diff_;
    double max_color_diff_;

    bool enable_color_ = true;
    bool enable_bright_ = false;

    double bright_gain_ = 1.0;
    double bright_bias_ = 0.0;

    double bright_gain_bk_ = 1.0;
    double bright_bias_bk_ = 0.0;

    double depth_weight_ = 0.968;

    void SetInput(const cv::Mat &src_gray, const cv::Mat &src_depth, const cv::Mat &dst_gray, const cv::Mat &dst_depth,
                  const Eigen::Vector4f K);

    void SetInput(const cv::Mat &src_gray, const cv::Mat &src_depth, const Eigen::Vector4f src_K,
                  const cv::Mat &dst_gray, const cv::Mat &dst_depth, const Eigen::Vector4f dst_K);

    int SelectInitPose(std::vector<Eigen::Matrix4d> candidates);

    void ComputeJacobian(int level, Eigen::Matrix4d init,
                         int &corres_num, float &residual, Eigen::Matrix6d &JtJ, Eigen::Vector6d &Jtr);

    void ComputeJacobianWithBright(int level, Eigen::Matrix4d init,
                         int &corres_num, float &residual, Eigen::Matrix7d &JtJ, Eigen::Vector7d &Jtr);


    void DoSingleIteration(int level, Eigen::Matrix4d init,
                           int &corres_num, float &residual, Eigen::Matrix6d &JtJ, Eigen::Vector6d &Jtr);


    void DoSingleIteration_omp(int level, Eigen::Matrix4d init,
                           int &corres_num, float &residual, Eigen::Matrix6d &JtJ, Eigen::Vector6d &Jtr);


    void DoSingleIteration_bright(int level, Eigen::Matrix4d init,
                                  int &corres_num, float &residual, Eigen::Matrix7d &JtJ, Eigen::Vector7d &Jtr);

    void ComputeErrorOnly(int level, Eigen::Matrix4d &initial, int &corres_num, float &residual);



    std::shared_ptr<RGBDPyramid> GetSrcPyr();
    std::shared_ptr<RGBDPyramid> GetDstPyr();

    void ResetState();

    void RollBackBrightPara();

    void UpdateBrightPara(float delta);

    void ComputeColorState();

    float ComputeIlluminateDiff(cv::Mat gray1, cv::Mat mask1, cv::Mat gray2, cv::Mat mask2, int step);


};





#endif //SLAM_RGBDREGISTRATION_H
