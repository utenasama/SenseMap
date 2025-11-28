//
// Created by sensetime on 2020/10/30.
//

#include "RGBDRegistration.h"
#include <chrono>

RGBDRegistration::RGBDRegistration(){}

RGBDRegistration::RGBDRegistration(int levels, std::vector<int> sample_steps, double max_depth_diff, double max_color_diff):
    levels_(levels), sample_steps_(sample_steps), max_depth_diff_(max_depth_diff), max_color_diff_(max_color_diff){}

RGBDRegistration::~RGBDRegistration(){}

bool RGBDRegistration::MatchedInfo::BetterThan(const MatchedInfo &info2){
    if(residuals_<info2.residuals_ && corres_num_ > info2.corres_num_) return true;
    if(residuals_>=info2.residuals_ && corres_num_<= info2.corres_num_) return false;

    float ratio = corres_num_/float(info2.corres_num_);
    float ratio_th = 1.2;
    if(ratio<ratio_th && ratio > 1/ratio_th){
        return residuals_/corres_num_ < info2.residuals_/info2.corres_num_;
    }
    return corres_num_ > info2.corres_num_;
};


void RGBDRegistration::SetInput(const cv::Mat &src_gray, const cv::Mat &src_depth,
                                const cv::Mat &dst_gray, const cv::Mat &dst_depth,
                                const Eigen::Vector4f K){
    src_ = std::make_shared<RGBDPyramid>(levels_, 2.0, K, src_gray, src_depth,
                                         cv::Mat(), cv::Mat(), -1);

    dst_ = std::make_shared<RGBDPyramid>(levels_, 2.0, K, dst_gray, dst_depth,
                                         cv::Mat(), cv::Mat(), -1);
}

void RGBDRegistration::SetInput(const cv::Mat &src_gray, const cv::Mat &src_depth, const Eigen::Vector4f src_K,
                                const cv::Mat &dst_gray, const cv::Mat &dst_depth, const Eigen::Vector4f dst_K){
    src_ = std::make_shared<RGBDPyramid>(levels_, 2.0, src_K, src_gray, src_depth,
                                         cv::Mat(), cv::Mat(), -1);

    dst_ = std::make_shared<RGBDPyramid>(levels_, 2.0, dst_K, dst_gray, dst_depth,
                                         cv::Mat(), cv::Mat(), -1);
}



int RGBDRegistration::SelectInitPose(std::vector<Eigen::Matrix4d> candidates) {
    std::vector<MatchedInfo> matched_infos(candidates.size());

    int cmp_level = 1;
    if(cmp_level>levels_) cmp_level = levels_-1;
    for (int i = 0; i < candidates.size(); i++) {
        ComputeErrorOnly(cmp_level, candidates[i], matched_infos[i].corres_num_, matched_infos[i].residuals_);
        std::cout << "matched info for: " << i << " is " << matched_infos[i].corres_num_ << " "
                  << matched_infos[i].residuals_ << std::endl;
    }
    int select_id = 0;
    for(int i=1; i<matched_infos.size(); i++){
        bool keep = matched_infos[select_id].BetterThan(matched_infos[i]);
        if(!keep) select_id = i;
    }
    return select_id;
}

std::shared_ptr<RGBDPyramid> RGBDRegistration::GetSrcPyr(){
    return src_;
}
std::shared_ptr<RGBDPyramid> RGBDRegistration::GetDstPyr(){
    return dst_;
}

void RGBDRegistration::ComputeJacobian(int level, Eigen::Matrix4d init,
                     int &corres_num, float &residual, Eigen::Matrix6d &JtJ, Eigen::Vector6d &Jtr){

    DoSingleIteration_omp(level, init, corres_num, residual, JtJ, Jtr);

}

void RGBDRegistration::ComputeJacobianWithBright(int level, Eigen::Matrix4d init,
                               int &corres_num, float &residual, Eigen::Matrix7d &JtJ, Eigen::Vector7d &Jtr){
    DoSingleIteration_bright(level, init, corres_num, residual, JtJ, Jtr);
}


void RGBDRegistration::DoSingleIteration(int level, Eigen::Matrix4d init,
                       int &corres_num, float &residual, Eigen::Matrix6d &JtJ, Eigen::Vector6d &Jtr){


    residual = 0;
    corres_num = 0;
    JtJ = Eigen::Matrix6d::Zero();
    Jtr = Eigen::Vector6d::Zero();

    int width = src_->pyramid_depth_[level].cols;
    int height = src_->pyramid_depth_[level].rows;

    const auto &K_vec = src_->pyramid_K_[level];
    const double sobel_scale = RGBDPyramid::sobel_scale;

    Eigen::Matrix3d K = Eigen::Matrix3d::Identity();
    K(0, 0) = dst_->pyramid_K_[level][0];
    K(1, 1) = dst_->pyramid_K_[level][1];
    K(0, 2) = dst_->pyramid_K_[level][2];
    K(1, 2) = dst_->pyramid_K_[level][3];

    const auto &src_gray = src_->pyramid_gray_[level];
    const auto &src_depth = src_->pyramid_depth_[level];


    double sqrt_lamba_dep = sqrt(depth_weight_);
    double sqrt_lambda_img = sqrt(1.0 - depth_weight_);

    const double fx = K_vec[0];
    const double fy = K_vec[1];

    auto dst = dst_;

    const Eigen::Matrix3d R = init.block<3, 3>(0, 0);
    const Eigen::Vector3d T = init.block<3, 1>(0, 3);

    const auto &dst_gray = dst->pyramid_gray_[level];
    const auto &dst_depth = dst->pyramid_depth_[level];

    const auto &dst_gray_dx = dst->pyramid_gray_dx_[level];
    const auto &dst_gray_dy = dst->pyramid_gray_dy_[level];

    const auto &dst_depth_dx = dst->pyramid_depth_dx_[level];
    const auto &dst_depth_dy = dst->pyramid_depth_dy_[level];

    const auto &XYZ = *(src_->GetXYZ(level, sample_steps_[level]));

    for (int y = 0; y < height; y += sample_steps_[level]) {
        for (int x = 0; x < width; x += sample_steps_[level]) {
            const int index = y * width + x;

            if (std::isnan(XYZ(x, y)[2])) continue;

            Eigen::Vector3d pt = XYZ(x, y).cast<double>();

            Eigen::Vector3d trans_pt = R * pt + T;

            Eigen::Vector3d proj_pt = K * trans_pt;
            double inv_proj_z = 1.0 / trans_pt[2];

            int u_t = (int) (proj_pt[0] / proj_pt[2] + 0.5);
            int v_t = (int) (proj_pt[1] / proj_pt[2] + 0.5);
//            PRINT_D("PROJECT U V is: %d %d, image size %d %d\n", u_t, v_t, width, height);

            if (u_t < 0 || u_t > width - 1 || v_t < 0 || v_t > height - 1) {
                continue;
            }
            float d_t = dst_depth.at<float>(v_t, u_t);
            if (d_t < FLT_EPSILON || std::isnan(d_t)) {
                continue;
            }

            double pixel_scale = 1.0;
            double depth_residual = dst_depth.at<float>(v_t, u_t) - trans_pt[2];
            const double lamda = 1.0;
            if (std::fabs(depth_residual) > 10 * max_depth_diff_) {
                continue;
            } else if (std::fabs(depth_residual) > max_depth_diff_) {
                pixel_scale = exp(lamda * (1 - std::fabs(depth_residual) /
                                               max_depth_diff_));
            }

            double photo_residual = (dst_gray.at<float>(v_t, u_t) -
                                     src_gray.at<float>(y, x));

            double dIdx = sobel_scale * dst_gray_dx.at<float>(v_t, u_t);
            double dIdy = sobel_scale * dst_gray_dy.at<float>(v_t, u_t);
            double dDdx = sobel_scale * dst_depth_dx.at<float>(v_t, u_t);
            double dDdy = sobel_scale * dst_depth_dy.at<float>(v_t, u_t);


            if (std::fabs(photo_residual) > max_color_diff_) {
                dIdx = dIdy = 0;
            }

            photo_residual *= sqrt_lambda_img;
            depth_residual *= sqrt_lamba_dep;

            photo_residual *= pixel_scale;
            depth_residual *= pixel_scale;

            if (std::isnan(dDdx)) dDdx = 0;
            if (std::isnan(dDdy)) dDdy = 0;

            double invz = 1. / trans_pt(2);
            double c0 = dIdx * fx * invz;
            double c1 = dIdy * fy * invz;
            double c2 = -(c0 * trans_pt(0) + c1 * trans_pt(1)) * invz;
            double d0 = dDdx * fx * invz;
            double d1 = dDdy * fy * invz;
            double d2 = -(d0 * trans_pt(0) + d1 * trans_pt(1)) * invz;

            Eigen::Vector6d photo_Jt;
            Eigen::Vector6d depth_Jt;


            Eigen::Matrix<double, 3, 6> j_pt;
            j_pt << 0, trans_pt[2], -trans_pt[1], 1, 0, 0,
                    -trans_pt[2], 0, trans_pt[0], 0, 1, 0,
                    trans_pt[1], -trans_pt[0], 0, 0, 0, 1;

            Eigen::Vector3d j_c(c0, c1, c2);
            Eigen::Vector3d j_d(d0, d1, d2);

            photo_Jt = (j_c.transpose() * j_pt).transpose();
            photo_Jt *= (pixel_scale * sqrt_lambda_img);

            depth_Jt = (j_d.transpose() * j_pt - j_pt.row(2)).transpose();
            depth_Jt *= (pixel_scale * sqrt_lamba_dep);

            JtJ.noalias() += photo_Jt * photo_Jt.transpose();
            JtJ.noalias() += depth_Jt * depth_Jt.transpose();

            Jtr.noalias() += photo_residual * photo_Jt;
            Jtr.noalias() += depth_residual * depth_Jt;

            residual += photo_residual * photo_residual +
                         depth_residual * depth_residual;

            corres_num++;
        }
    }
}


void RGBDRegistration::DoSingleIteration_omp(int level, Eigen::Matrix4d init,
                           int &corres_num, float &residual, Eigen::Matrix6d &JtJ, Eigen::Vector6d &Jtr){
    // std::cout << "DoSingleIteration_omp" << std::endl;


    residual = 0;
    corres_num = 0;
    JtJ = Eigen::Matrix6d::Zero();
    Jtr = Eigen::Vector6d::Zero();

    int width = src_->pyramid_depth_[level].cols;
    int height = src_->pyramid_depth_[level].rows;

    const auto &K_vec = src_->pyramid_K_[level];
    Eigen::Matrix3d K = Eigen::Matrix3d::Identity();
    K(0, 0) = dst_->pyramid_K_[level][0];
    K(1, 1) = dst_->pyramid_K_[level][1];
    K(0, 2) = dst_->pyramid_K_[level][2];
    K(1, 2) = dst_->pyramid_K_[level][3];

    const Eigen::Matrix3d R = init.block<3, 3>(0, 0);
    const Eigen::Vector3d T = init.block<3, 1>(0, 3);

    const auto &src_gray = src_->pyramid_gray_[level];
    const auto &src_depth = src_->pyramid_depth_[level];
    const auto &dst_gray = dst_->pyramid_gray_[level];
    const auto &dst_depth = dst_->pyramid_depth_[level];

    const auto &src_conf = src_->pyramid_conf_[level];
    const auto &dst_conf = dst_->pyramid_conf_[level];

    const auto &dst_gray_dx = dst_->pyramid_gray_dx_[level];
    const auto &dst_gray_dy = dst_->pyramid_gray_dy_[level];

    const auto &dst_depth_dx = dst_->pyramid_depth_dx_[level];
    const auto &dst_depth_dy = dst_->pyramid_depth_dy_[level];

    const auto &src_highlight = src_->pyramid_highlight_[level];
    const auto &dst_highlight = dst_->pyramid_highlight_[level];

    const double sqrt_lamba_dep = sqrt(depth_weight_);
    const double sqrt_lambda_img = sqrt(1.0 - depth_weight_);
    const double fx = K_vec[0];
    const double fy = K_vec[1];

    const double sobel_scale = RGBDPyramid::sobel_scale;

    double residual_odd = 0;
    double residual_even = 0;

    int corres_num_odd = 0;
    int corres_num_even = 0;

    Eigen::Matrix6d JTJ_odd = Eigen::Matrix6d::Zero();
    Eigen::Matrix6d JTJ_even = Eigen::Matrix6d::Zero();

    Eigen::Vector6d JTr_odd = Eigen::Vector6d::Zero();
    Eigen::Vector6d JTr_even = Eigen::Vector6d::Zero();

    double ph_odd = 0;
    double ph_even = 0;
    double depth_odd = 0;
    double depth_even = 0;

    cv::Mat photo_error_map = cv::Mat::zeros(src_gray.size(), CV_32FC1);
    const auto &XYZ = *(src_->GetXYZ(level, sample_steps_[level]));

    auto st = std::chrono::steady_clock::now();
#ifdef MY_USE_OMP
#pragma omp parallel for schedule(dynamic, 1) num_threads(2)
#endif
    for (int y = 0; y < height; y += sample_steps_[level]) {
        // need accumulate
#ifdef MY_USE_OMP
        const int thread_index = omp_get_thread_num();
#else
        const int thread_index = (y/sample_steps_[level])%2;

#endif
        bool isEven = ((thread_index & 1) == 0);
        double &tmp_residual = isEven? residual_even : residual_odd;
        int &cur_corres_num = isEven? corres_num_even : corres_num_odd;

        double &ph_error = isEven? ph_even : ph_odd;
        double &depth_error = isEven? depth_even : depth_odd;

        double *curJTJPtr = isEven? JTJ_even.data() : JTJ_odd.data();
        double *curJTrPtr = isEven? JTr_even.data() : JTr_odd.data();

        decltype(JTJ_even) &curJtJ =  isEven? JTJ_even : JTJ_odd;
        decltype(JTr_even) &curJtr =  isEven? JTr_even : JTr_odd;

        // tmp local variables
        Eigen::Vector6d photo_Jt = Eigen::Vector6d::Zero();
        Eigen::Vector6d depth_Jt = Eigen::Vector6d::Zero();
        double *photo_Jt_ptr = photo_Jt.data();
        double *depth_Jt_ptr = depth_Jt.data();

        Eigen::Vector3d pt;
        Eigen::Vector3d world_pt;
        Eigen::Vector3d trans_pt;
        Eigen::Vector3d proj_pt;
        Eigen::Matrix<double, 3, 6> j_pt;
        double *j_pt_ptr = j_pt.data();

        for (int x = 0; x < width; x += sample_steps_[level]) {
            const int index = y * width + x;
            if (std::isnan(XYZ[index][2])) continue;
            if (std::isnan(src_conf.at<float>(y,x))) continue;
            pt[0] = XYZ[index][0]; pt[1] = XYZ[index][1]; pt[2] = XYZ[index][2];
            trans_pt[2] = R(2, 0) * pt[0] + R(2, 1) * pt[1]+R(2,2)*pt[2]+T[2];
            trans_pt[1] = R(1, 0) * pt[0] + R(1, 1) * pt[1] + R(1, 2) * pt[2] + T[1];
            trans_pt[0] = R(0, 0) * pt[0] + R(0, 1) * pt[1] + R(0, 2) * pt[2] + T[0];

            proj_pt[2] = trans_pt[2];
            proj_pt[1] = K(1, 0) * trans_pt[0] + K(1, 1) * trans_pt[1] + K(1, 2) * trans_pt[2];
            proj_pt[0] = K(0, 0) * trans_pt[0] + K(0, 1) * trans_pt[1] + K(0, 2) * trans_pt[2];

            double inv_proj_z = 1.0 / trans_pt[2];


            int u_t = (int) (proj_pt[0] / proj_pt[2] + 0.5);
            int v_t = (int) (proj_pt[1] / proj_pt[2] + 0.5);


//            PRINT_D("PROJECT U V is: %d %d, image size %d %d\n", u_t, v_t, width, height);


            if ((u_t < 0 || u_t > width - 1) || (v_t < 0 || v_t > height - 1)) {
                continue;
            }

            if (std::isnan(dst_conf.at<float>(v_t,u_t))) continue;

            float conf = std::min(src_conf.at<float>(y,x), dst_conf.at<float>(v_t,u_t));
            if(conf<0.3) conf= 0.3;

            float d_t = dst_depth.at<float>(v_t, u_t);
            if (d_t < FLT_EPSILON || std::isnan(d_t)) continue;

            double pixel_scale = 1.0;

            double depth_residual =dst_depth.at<float>(v_t, u_t) - trans_pt[2];
            const double lamda = 1.0;

            if(std::fabs(depth_residual) > 10 * max_depth_diff_){
                continue;
            }

            else if(std::fabs(depth_residual) > max_depth_diff_){
                pixel_scale = exp( lamda * (1 - std::fabs(depth_residual)/max_depth_diff_) );
            }

//            printf("depth diff  %lf   %lf and scale %lf\n",depth_residual, depth_residual/max_depth_diff_, pixel_scale );

            cur_corres_num++;

            double photo_residual = (dst_gray.at<float>(v_t, u_t) - src_gray.at<float>(y, x));
            double dIdx = sobel_scale * dst_gray_dx.at<float>(v_t, u_t);
            double dIdy = sobel_scale * dst_gray_dy.at<float>(v_t, u_t);
            double dDdx = sobel_scale * dst_depth_dx.at<float>(v_t, u_t);
            double dDdy = sobel_scale * dst_depth_dy.at<float>(v_t, u_t);

            photo_error_map.at<float>(v_t, u_t) = photo_residual;
            if(std::fabs(photo_residual) > max_color_diff_){
                dIdx = dIdy = 0;
                photo_error_map.at<float>(v_t, u_t) = 0;

            }

            if(!enable_color_){
                dIdx = dIdy = 0;
            }
            else if(src_highlight.at<uchar>(y,x)!=0 || dst_highlight.at<uchar>(v_t, u_t)!=0){
                dIdx = dIdy = 0;
            }

            double tmp_p = pixel_scale * sqrt_lambda_img;
            double tmp_d = pixel_scale * sqrt_lamba_dep;

            photo_residual *= tmp_p;
            depth_residual *= tmp_d;

            if (std::isnan(dDdx)) dDdx = 0;
            if (std::isnan(dDdy)) dDdy = 0;

            double invz = 1. / trans_pt(2);
            double c0 = dIdx * fx * invz;
            double c1 = dIdy * fy * invz;
            double c2 = -(c0 * trans_pt(0) + c1 * trans_pt(1)) * invz;
            double d0 = dDdx * fx * invz;
            double d1 = dDdy * fy * invz;
            double d2 = -(d0 * trans_pt(0) + d1 * trans_pt(1)) * invz;

            photo_Jt_ptr[0] = tmp_p * (-trans_pt(2) * c1 + trans_pt(1) * c2);
            photo_Jt_ptr[1] = tmp_p * (trans_pt(2) * c0 - trans_pt(0) * c2);
            photo_Jt_ptr[2] = tmp_p * (-trans_pt(1) * c0 + trans_pt(0) * c1);
            photo_Jt_ptr[3] = tmp_p * (c0);
            photo_Jt_ptr[4] = tmp_p * (c1);
            photo_Jt_ptr[5] = tmp_p * (c2);


            d2 -= 1.0;
            depth_Jt_ptr[0] = tmp_d * (-trans_pt(2) * d1 + trans_pt(1) * d2);
            depth_Jt_ptr[1] = tmp_d * (trans_pt(2) * d0 - trans_pt(0) * d2);
            depth_Jt_ptr[2] = tmp_d * (-trans_pt(1) * d0 + trans_pt(0) * d1);
            depth_Jt_ptr[3] = tmp_d * d0;
            depth_Jt_ptr[4] = tmp_d * d1;
            depth_Jt_ptr[5] = tmp_d * d2;

            curJTJPtr[0] += photo_Jt_ptr[0] * photo_Jt_ptr[0] + depth_Jt_ptr[0] * depth_Jt_ptr[0];
            curJTJPtr[1] += photo_Jt_ptr[0] * photo_Jt_ptr[1] + depth_Jt_ptr[0] * depth_Jt_ptr[1];
            curJTJPtr[2] += photo_Jt_ptr[0] * photo_Jt_ptr[2] + depth_Jt_ptr[0] * depth_Jt_ptr[2];
            curJTJPtr[3] += photo_Jt_ptr[0] * photo_Jt_ptr[3] + depth_Jt_ptr[0] * depth_Jt_ptr[3];
            curJTJPtr[4] += photo_Jt_ptr[0] * photo_Jt_ptr[4] + depth_Jt_ptr[0] * depth_Jt_ptr[4];
            curJTJPtr[5] += photo_Jt_ptr[0] * photo_Jt_ptr[5] + depth_Jt_ptr[0] * depth_Jt_ptr[5];

//            curJTJPtr[6] += photo_Jt_ptr[1] * photo_Jt_ptr[0] + depth_Jt_ptr[1] * depth_Jt_ptr[0];
            curJTJPtr[7] += photo_Jt_ptr[1] * photo_Jt_ptr[1] + depth_Jt_ptr[1] * depth_Jt_ptr[1];
            curJTJPtr[8] += photo_Jt_ptr[1] * photo_Jt_ptr[2] + depth_Jt_ptr[1] * depth_Jt_ptr[2];
            curJTJPtr[9] += photo_Jt_ptr[1] * photo_Jt_ptr[3] + depth_Jt_ptr[1] * depth_Jt_ptr[3];
            curJTJPtr[10] += photo_Jt_ptr[1] * photo_Jt_ptr[4] + depth_Jt_ptr[1] * depth_Jt_ptr[4];
            curJTJPtr[11] += photo_Jt_ptr[1] * photo_Jt_ptr[5] + depth_Jt_ptr[1] * depth_Jt_ptr[5];

//            curJTJPtr[12] += photo_Jt_ptr[2] * photo_Jt_ptr[0] + depth_Jt_ptr[2] * depth_Jt_ptr[0];
//            curJTJPtr[13] += photo_Jt_ptr[2] * photo_Jt_ptr[1] + depth_Jt_ptr[2] * depth_Jt_ptr[1];
            curJTJPtr[14] += photo_Jt_ptr[2] * photo_Jt_ptr[2] + depth_Jt_ptr[2] * depth_Jt_ptr[2];
            curJTJPtr[15] += photo_Jt_ptr[2] * photo_Jt_ptr[3] + depth_Jt_ptr[2] * depth_Jt_ptr[3];
            curJTJPtr[16] += photo_Jt_ptr[2] * photo_Jt_ptr[4] + depth_Jt_ptr[2] * depth_Jt_ptr[4];
            curJTJPtr[17] += photo_Jt_ptr[2] * photo_Jt_ptr[5] + depth_Jt_ptr[2] * depth_Jt_ptr[5];

//            curJTJPtr[18] += photo_Jt_ptr[3] * photo_Jt_ptr[0] + depth_Jt_ptr[3] * depth_Jt_ptr[0];
//            curJTJPtr[19] += photo_Jt_ptr[3] * photo_Jt_ptr[1] + depth_Jt_ptr[3] * depth_Jt_ptr[1];
//            curJTJPtr[20] += photo_Jt_ptr[3] * photo_Jt_ptr[2] + depth_Jt_ptr[3] * depth_Jt_ptr[2];
            curJTJPtr[21] += photo_Jt_ptr[3] * photo_Jt_ptr[3] + depth_Jt_ptr[3] * depth_Jt_ptr[3];
            curJTJPtr[22] += photo_Jt_ptr[3] * photo_Jt_ptr[4] + depth_Jt_ptr[3] * depth_Jt_ptr[4];
            curJTJPtr[23] += photo_Jt_ptr[3] * photo_Jt_ptr[5] + depth_Jt_ptr[3] * depth_Jt_ptr[5];

//            curJTJPtr[24] += photo_Jt_ptr[4] * photo_Jt_ptr[0] + depth_Jt_ptr[4] * depth_Jt_ptr[0];
//            curJTJPtr[25] += photo_Jt_ptr[4] * photo_Jt_ptr[1] + depth_Jt_ptr[4] * depth_Jt_ptr[1];
//            curJTJPtr[26] += photo_Jt_ptr[4] * photo_Jt_ptr[2] + depth_Jt_ptr[4] * depth_Jt_ptr[2];
//            curJTJPtr[27] += photo_Jt_ptr[4] * photo_Jt_ptr[3] + depth_Jt_ptr[4] * depth_Jt_ptr[3];
            curJTJPtr[28] += photo_Jt_ptr[4] * photo_Jt_ptr[4] + depth_Jt_ptr[4] * depth_Jt_ptr[4];
            curJTJPtr[29] += photo_Jt_ptr[4] * photo_Jt_ptr[5] + depth_Jt_ptr[4] * depth_Jt_ptr[5];

//            curJTJPtr[30] += photo_Jt_ptr[5] * photo_Jt_ptr[0] + depth_Jt_ptr[5] * depth_Jt_ptr[0];
//            curJTJPtr[31] += photo_Jt_ptr[5] * photo_Jt_ptr[1] + depth_Jt_ptr[5] * depth_Jt_ptr[1];
//            curJTJPtr[32] += photo_Jt_ptr[5] * photo_Jt_ptr[2] + depth_Jt_ptr[5] * depth_Jt_ptr[2];
//            curJTJPtr[33] += photo_Jt_ptr[5] * photo_Jt_ptr[3] + depth_Jt_ptr[5] * depth_Jt_ptr[3];
//            curJTJPtr[34] += photo_Jt_ptr[5] * photo_Jt_ptr[4] + depth_Jt_ptr[5] * depth_Jt_ptr[4];
            curJTJPtr[35] += photo_Jt_ptr[5] * photo_Jt_ptr[5] + depth_Jt_ptr[5] * depth_Jt_ptr[5];

            curJTrPtr[0] += photo_residual * photo_Jt_ptr[0] + depth_residual * depth_Jt_ptr[0];
            curJTrPtr[1] += photo_residual * photo_Jt_ptr[1] + depth_residual * depth_Jt_ptr[1];
            curJTrPtr[2] += photo_residual * photo_Jt_ptr[2] + depth_residual * depth_Jt_ptr[2];
            curJTrPtr[3] += photo_residual * photo_Jt_ptr[3] + depth_residual * depth_Jt_ptr[3];
            curJTrPtr[4] += photo_residual * photo_Jt_ptr[4] + depth_residual * depth_Jt_ptr[4];
            curJTrPtr[5] += photo_residual * photo_Jt_ptr[5] + depth_residual * depth_Jt_ptr[5];

            tmp_residual += photo_residual * photo_residual + depth_residual * depth_residual;

            ph_error += photo_residual * photo_residual;
            depth_error += depth_residual * depth_residual;
        }

    }

    double total_ph_error = ph_even + ph_odd;
    double total_depth_error = depth_even + depth_odd;
    residual = residual_even + residual_odd;
    corres_num = corres_num_even + corres_num_odd;

    for (int c = 0; c < 6; c++) {
        for (int r = 0; r < c; r++) {
            JTJ_even(r,c) = JTJ_even(c,r);
            JTJ_odd(r,c) = JTJ_odd(c,r);
        }
    }

//    printf("depth error and photo error is: %f %f %f\n",total_depth_error,  total_ph_error, residual);


    JtJ += (JTJ_even + JTJ_odd);
    Jtr += (JTr_even + JTr_odd);

//    if(level == 0){
//        cv::imshow("photo_error_map", 5 * photo_error_map);
//    }

//    LOGD("depth error and photo error is: %lf %lf || %lf %lf %d\n", sqrt(total_depth_error/(depth_weight_*corres_num)),
//                                                          sqrt(total_ph_error/((1-depth_weight_)*corres_num)), total_depth_error, total_ph_error,
//            corres_num );
}


void RGBDRegistration::DoSingleIteration_bright(int level, Eigen::Matrix4d init,
                              int &corres_num, float &residual, Eigen::Matrix7d &JtJ, Eigen::Vector7d &Jtr){

    residual = 0;
    corres_num = 0;
    JtJ = Eigen::Matrix7d::Zero();
    Jtr = Eigen::Vector7d::Zero();

    int width = src_->pyramid_depth_[level].cols;
    int height = src_->pyramid_depth_[level].rows;

    const auto &K_vec = src_->pyramid_K_[level];
    const double sobel_scale = RGBDPyramid::sobel_scale;

    Eigen::Matrix3d K = Eigen::Matrix3d::Identity();
    K(0, 0) = dst_->pyramid_K_[level][0];
    K(1, 1) = dst_->pyramid_K_[level][1];
    K(0, 2) = dst_->pyramid_K_[level][2];
    K(1, 2) = dst_->pyramid_K_[level][3];

    const Eigen::Matrix3d R = init.block<3, 3>(0, 0);
    const Eigen::Vector3d T = init.block<3, 1>(0, 3);

    const auto &src_gray = src_->pyramid_gray_[level];
    const auto &src_depth = src_->pyramid_depth_[level];
    const auto &dst_gray = dst_->pyramid_gray_[level];
    const auto &dst_depth = dst_->pyramid_depth_[level];

    const auto &src_conf = src_->pyramid_conf_[level];
    const auto &dst_conf = dst_->pyramid_conf_[level];

    const auto &dst_gray_dx = dst_->pyramid_gray_dx_[level];
    const auto &dst_gray_dy = dst_->pyramid_gray_dy_[level];

    const auto &dst_depth_dx = dst_->pyramid_depth_dx_[level];
    const auto &dst_depth_dy = dst_->pyramid_depth_dy_[level];

    const auto &src_highlight = src_->pyramid_highlight_[level];
    const auto &dst_highlight = dst_->pyramid_highlight_[level];

    const double sqrt_lamba_dep = sqrt(depth_weight_);
    const double sqrt_lambda_img = sqrt(1.0 - depth_weight_);

    const double fx = K_vec[0];
    const double fy = K_vec[1];

    double residual_odd = 0;
    double residual_even = 0;

    int corres_num_odd = 0;
    int corres_num_even = 0;

    Eigen::Matrix7d JTJ_odd = Eigen::Matrix7d::Zero();
    Eigen::Matrix7d JTJ_even = Eigen::Matrix7d::Zero();

    Eigen::Vector7d JTr_odd = Eigen::Vector7d::Zero();
    Eigen::Vector7d JTr_even = Eigen::Vector7d::Zero();

    double ph_odd = 0;
    double ph_even = 0;
    double depth_odd = 0;
    double depth_even = 0;

    Eigen::Matrix6d JTJ_6 = Eigen::Matrix6d::Zero();
    Eigen::Vector6d JTr_6 = Eigen::Vector6d::Zero();
    const auto &XYZ = *(src_->GetXYZ(level, sample_steps_[level]));

    auto st = std::chrono::steady_clock::now();
#ifdef MY_USE_OMP
#pragma omp parallel for schedule(dynamic, 1) num_threads(2)
#endif
    for (int y = 0; y < height; y += sample_steps_[level]) {
        // need accumulate
#ifdef MY_USE_OMP
        const int index = omp_get_thread_num();
#else
        const int thread_index = (y/sample_steps_[level])%2;
#endif

        bool isEven = ((thread_index & 1) == 0);
        double &tmp_residual = isEven? residual_even : residual_odd;
        int &cur_corres_num = isEven? corres_num_even : corres_num_odd;

        double &ph_error = isEven? ph_even : ph_odd;
        double &depth_error = isEven? depth_even : depth_odd;

        double *curJTJPtr = isEven? JTJ_even.data() : JTJ_odd.data();
        double *curJTrPtr = isEven? JTr_even.data() : JTr_odd.data();
        Eigen::Matrix7d &curJtJ =  isEven? JTJ_even : JTJ_odd;
        Eigen::Vector7d &curJtr =  isEven? JTr_even : JTr_odd;


        // tmp local variables
        Eigen::Vector7d photo_Jt = Eigen::Vector7d::Zero();
        Eigen::Vector7d depth_Jt = Eigen::Vector7d::Zero();
        double *photo_Jt_ptr = photo_Jt.data();
        double *depth_Jt_ptr = depth_Jt.data();

        Eigen::Vector3d pt;
        Eigen::Vector3d trans_pt;
        Eigen::Vector3d proj_pt;
        Eigen::Matrix<double, 3, 6> j_pt;
        double color_th = max_color_diff_;

        for (int x = 0; x < width; x += sample_steps_[level]) {
            const int index = y * width + x;
            if (std::isnan(XYZ[index][2])) continue;

            if (std::isnan(src_conf.at<float>(y,x))) continue;

            pt = XYZ[index].cast<double>();
            trans_pt = R * pt + T;
            proj_pt = K * trans_pt;
            double inv_proj_z = 1.0 / trans_pt[2];

            int u_t = (int) (proj_pt[0] / proj_pt[2] + 0.5);
            int v_t = (int) (proj_pt[1] / proj_pt[2] + 0.5);

            if ((u_t < 0 || u_t > width - 1) || (v_t < 0 || v_t > height - 1)) {
                continue;
            }

            if (std::isnan(dst_conf.at<float>(v_t,u_t))) continue;

            float conf = std::min(src_conf.at<float>(y,x), dst_conf.at<float>(v_t,u_t));

            float d_t = dst_depth.at<float>(v_t, u_t);
            if (d_t < FLT_EPSILON || std::isnan(d_t)) continue;

            double pixel_scale = 1.0;

            double depth_residual =dst_depth.at<float>(v_t, u_t) - trans_pt[2];
            const double lamda = 1.0;

            if(std::fabs(depth_residual) > 10 * max_depth_diff_){
                continue;
            }
            else if(std::fabs(depth_residual) > max_depth_diff_){
                pixel_scale = exp( lamda * (1 - std::fabs(depth_residual)/max_depth_diff_) );
            }

//            printf("depth diff  %lf   %lf and scale %lf\n",depth_residual, depth_residual/max_depth_diff_, pixel_scale );

            cur_corres_num++;

            double photo_residual = (dst_gray.at<float>(v_t, u_t) - src_gray.at<float>(y, x) + bright_bias_);


            double dIdx = sobel_scale * dst_gray_dx.at<float>(v_t, u_t);
            double dIdy = sobel_scale * dst_gray_dy.at<float>(v_t, u_t);
            double dDdx = sobel_scale * dst_depth_dx.at<float>(v_t, u_t);
            double dDdy = sobel_scale * dst_depth_dy.at<float>(v_t, u_t);

            if(std::fabs(photo_residual) > max_color_diff_){
                dIdx = dIdy = 0;
            }

            if(!enable_color_){
                dIdx = dIdy = 0;
            }
            else if(src_highlight.at<uchar>(y,x)!=0 || dst_highlight.at<uchar>(v_t, u_t)!=0){
                dIdx = dIdy = 0;
            }

            double tmp_p = pixel_scale * sqrt_lambda_img;
            double tmp_d = pixel_scale * sqrt_lamba_dep;

            photo_residual *= tmp_p;
            depth_residual *= tmp_d;
            if (std::isnan(dDdx)) dDdx = 0;
            if (std::isnan(dDdy)) dDdy = 0;

            double invz = 1. / trans_pt(2);
            double c0 = dIdx * fx * invz;
            double c1 = dIdy * fy * invz;
            double c2 = -(c0 * trans_pt(0) + c1 * trans_pt(1)) * invz;
            double d0 = dDdx * fx * invz;
            double d1 = dDdy * fy * invz;
            double d2 = -(d0 * trans_pt(0) + d1 * trans_pt(1)) * invz;

            photo_Jt_ptr[0] = tmp_p * (-trans_pt(2) * c1 + trans_pt(1) * c2);
            photo_Jt_ptr[1] = tmp_p * (trans_pt(2) * c0 - trans_pt(0) * c2);
            photo_Jt_ptr[2] = tmp_p * (-trans_pt(1) * c0 + trans_pt(0) * c1);
            photo_Jt_ptr[3] = tmp_p * (c0);
            photo_Jt_ptr[4] = tmp_p * (c1);
            photo_Jt_ptr[5] = tmp_p * (c2);
            photo_Jt_ptr[6] = tmp_p * (std::fabs(photo_residual) < color_th);

            d2 -= 1.0;
            depth_Jt_ptr[0] = tmp_d * (-trans_pt(2) * d1 + trans_pt(1) * d2);
            depth_Jt_ptr[1] = tmp_d * (trans_pt(2) * d0 - trans_pt(0) * d2);
            depth_Jt_ptr[2] = tmp_d * (-trans_pt(1) * d0 + trans_pt(0) * d1);
            depth_Jt_ptr[3] = tmp_d * d0;
            depth_Jt_ptr[4] = tmp_d * d1;
            depth_Jt_ptr[5] = tmp_d * d2;
            depth_Jt_ptr[6] = 0;


            curJTJPtr[0] += photo_Jt_ptr[0] * photo_Jt_ptr[0] + depth_Jt_ptr[0] * depth_Jt_ptr[0];
            curJTJPtr[1] += photo_Jt_ptr[0] * photo_Jt_ptr[1] + depth_Jt_ptr[0] * depth_Jt_ptr[1];
            curJTJPtr[2] += photo_Jt_ptr[0] * photo_Jt_ptr[2] + depth_Jt_ptr[0] * depth_Jt_ptr[2];
            curJTJPtr[3] += photo_Jt_ptr[0] * photo_Jt_ptr[3] + depth_Jt_ptr[0] * depth_Jt_ptr[3];
            curJTJPtr[4] += photo_Jt_ptr[0] * photo_Jt_ptr[4] + depth_Jt_ptr[0] * depth_Jt_ptr[4];
            curJTJPtr[5] += photo_Jt_ptr[0] * photo_Jt_ptr[5] + depth_Jt_ptr[0] * depth_Jt_ptr[5];
            curJTJPtr[6] += photo_Jt_ptr[0] * photo_Jt_ptr[6];

//            curJTJPtr[7] += photo_Jt_ptr[1] * photo_Jt_ptr[0] + depth_Jt_ptr[1] * depth_Jt_ptr[0];
            curJTJPtr[8] += photo_Jt_ptr[1] * photo_Jt_ptr[1] + depth_Jt_ptr[1] * depth_Jt_ptr[1];
            curJTJPtr[9] += photo_Jt_ptr[1] * photo_Jt_ptr[2] + depth_Jt_ptr[1] * depth_Jt_ptr[2];
            curJTJPtr[10] += photo_Jt_ptr[1] * photo_Jt_ptr[3] + depth_Jt_ptr[1] * depth_Jt_ptr[3];
            curJTJPtr[11] += photo_Jt_ptr[1] * photo_Jt_ptr[4] + depth_Jt_ptr[1] * depth_Jt_ptr[4];
            curJTJPtr[12] += photo_Jt_ptr[1] * photo_Jt_ptr[5] + depth_Jt_ptr[1] * depth_Jt_ptr[5];
            curJTJPtr[13] += photo_Jt_ptr[1] * photo_Jt_ptr[6];


//            curJTJPtr[14] += photo_Jt_ptr[2] * photo_Jt_ptr[0] + depth_Jt_ptr[2] * depth_Jt_ptr[0];
//            curJTJPtr[15] += photo_Jt_ptr[2] * photo_Jt_ptr[1] + depth_Jt_ptr[2] * depth_Jt_ptr[1];
            curJTJPtr[16] += photo_Jt_ptr[2] * photo_Jt_ptr[2] + depth_Jt_ptr[2] * depth_Jt_ptr[2];
            curJTJPtr[17] += photo_Jt_ptr[2] * photo_Jt_ptr[3] + depth_Jt_ptr[2] * depth_Jt_ptr[3];
            curJTJPtr[18] += photo_Jt_ptr[2] * photo_Jt_ptr[4] + depth_Jt_ptr[2] * depth_Jt_ptr[4];
            curJTJPtr[19] += photo_Jt_ptr[2] * photo_Jt_ptr[5] + depth_Jt_ptr[2] * depth_Jt_ptr[5];
            curJTJPtr[20] += photo_Jt_ptr[2] * photo_Jt_ptr[6];

//            curJTJPtr[21] += photo_Jt_ptr[3] * photo_Jt_ptr[0] + depth_Jt_ptr[3] * depth_Jt_ptr[0];
//            curJTJPtr[22] += photo_Jt_ptr[3] * photo_Jt_ptr[1] + depth_Jt_ptr[3] * depth_Jt_ptr[1];
//            curJTJPtr[23] += photo_Jt_ptr[3] * photo_Jt_ptr[2] + depth_Jt_ptr[3] * depth_Jt_ptr[2];
            curJTJPtr[24] += photo_Jt_ptr[3] * photo_Jt_ptr[3] + depth_Jt_ptr[3] * depth_Jt_ptr[3];
            curJTJPtr[25] += photo_Jt_ptr[3] * photo_Jt_ptr[4] + depth_Jt_ptr[3] * depth_Jt_ptr[4];
            curJTJPtr[26] += photo_Jt_ptr[3] * photo_Jt_ptr[5] + depth_Jt_ptr[3] * depth_Jt_ptr[5];
            curJTJPtr[27] += photo_Jt_ptr[3] * photo_Jt_ptr[6];

//            curJTJPtr[28] += photo_Jt_ptr[4] * photo_Jt_ptr[0] + depth_Jt_ptr[4] * depth_Jt_ptr[0];
//            curJTJPtr[29] += photo_Jt_ptr[4] * photo_Jt_ptr[1] + depth_Jt_ptr[4] * depth_Jt_ptr[1];
//            curJTJPtr[30] += photo_Jt_ptr[4] * photo_Jt_ptr[2] + depth_Jt_ptr[4] * depth_Jt_ptr[2];
//            curJTJPtr[31] += photo_Jt_ptr[4] * photo_Jt_ptr[3] + depth_Jt_ptr[4] * depth_Jt_ptr[3];
            curJTJPtr[32] += photo_Jt_ptr[4] * photo_Jt_ptr[4] + depth_Jt_ptr[4] * depth_Jt_ptr[4];
            curJTJPtr[33] += photo_Jt_ptr[4] * photo_Jt_ptr[5] + depth_Jt_ptr[4] * depth_Jt_ptr[5];
            curJTJPtr[34] += photo_Jt_ptr[4] * photo_Jt_ptr[6];

//            curJTJPtr[35] += photo_Jt_ptr[5] * photo_Jt_ptr[0] + depth_Jt_ptr[5] * depth_Jt_ptr[0];
//            curJTJPtr[36] += photo_Jt_ptr[5] * photo_Jt_ptr[1] + depth_Jt_ptr[5] * depth_Jt_ptr[1];
//            curJTJPtr[37] += photo_Jt_ptr[5] * photo_Jt_ptr[2] + depth_Jt_ptr[5] * depth_Jt_ptr[2];
//            curJTJPtr[38] += photo_Jt_ptr[5] * photo_Jt_ptr[3] + depth_Jt_ptr[5] * depth_Jt_ptr[3];
//            curJTJPtr[39] += photo_Jt_ptr[5] * photo_Jt_ptr[4] + depth_Jt_ptr[5] * depth_Jt_ptr[4];
            curJTJPtr[40] += photo_Jt_ptr[5] * photo_Jt_ptr[5] + depth_Jt_ptr[5] * depth_Jt_ptr[5];
            curJTJPtr[41] += photo_Jt_ptr[5] * photo_Jt_ptr[6];

//            curJTJPtr[42] += photo_Jt_ptr[6] * photo_Jt_ptr[0] + depth_Jt_ptr[6] * depth_Jt_ptr[0];
//            curJTJPtr[43] += photo_Jt_ptr[6] * photo_Jt_ptr[1] + depth_Jt_ptr[6] * depth_Jt_ptr[1];
//            curJTJPtr[44] += photo_Jt_ptr[6] * photo_Jt_ptr[2] + depth_Jt_ptr[6] * depth_Jt_ptr[2];
//            curJTJPtr[45] += photo_Jt_ptr[6] * photo_Jt_ptr[3] + depth_Jt_ptr[6] * depth_Jt_ptr[3];
//            curJTJPtr[46] += photo_Jt_ptr[6] * photo_Jt_ptr[4] + depth_Jt_ptr[6] * depth_Jt_ptr[4];
//            curJTJPtr[47] += photo_Jt_ptr[6] * photo_Jt_ptr[5] + depth_Jt_ptr[6] * depth_Jt_ptr[5];
            curJTJPtr[48] += photo_Jt_ptr[6] * photo_Jt_ptr[6];

            curJTrPtr[0] += photo_residual * photo_Jt_ptr[0] + depth_residual * depth_Jt_ptr[0];
            curJTrPtr[1] += photo_residual * photo_Jt_ptr[1] + depth_residual * depth_Jt_ptr[1];
            curJTrPtr[2] += photo_residual * photo_Jt_ptr[2] + depth_residual * depth_Jt_ptr[2];
            curJTrPtr[3] += photo_residual * photo_Jt_ptr[3] + depth_residual * depth_Jt_ptr[3];
            curJTrPtr[4] += photo_residual * photo_Jt_ptr[4] + depth_residual * depth_Jt_ptr[4];
            curJTrPtr[5] += photo_residual * photo_Jt_ptr[5] + depth_residual * depth_Jt_ptr[5];
            curJTrPtr[6] += photo_residual * photo_Jt_ptr[6];

            tmp_residual += photo_residual * photo_residual + depth_residual * depth_residual;
//            ph_error += photo_residual * photo_residual;
//            depth_error += depth_residual * depth_residual;
        }

    }

    double total_ph_error = ph_even + ph_odd;
    double total_depth_error = depth_even + depth_odd;
    residual = residual_even + residual_odd;
    corres_num = corres_num_even + corres_num_odd;

    for (int c = 0; c < 7; c++) {
        for (int r = 0; r < c; r++) {
            JTJ_even(r,c) = JTJ_even(c,r);
            JTJ_odd(r,c) = JTJ_odd(c,r);
        }
    }

    JtJ += (JTJ_even + JTJ_odd);
    Jtr += (JTr_even + JTr_odd);

}


void RGBDRegistration::ComputeErrorOnly(int level, Eigen::Matrix4d &initial, int &corres_num, float &residual){
    corres_num = 0;
    residual = 0;

    int width = src_->pyramid_depth_[level].cols;
    int height = src_->pyramid_depth_[level].rows;

    const auto &K_vec = src_->pyramid_K_[level];
    const double fx = K_vec[0];
    const double fy = K_vec[1];
    Eigen::Matrix3d K = Eigen::Matrix3d::Identity();
    K(0, 0) = dst_->pyramid_K_[level][0];
    K(1, 1) = dst_->pyramid_K_[level][1];
    K(0, 2) = dst_->pyramid_K_[level][2];
    K(1, 2) = dst_->pyramid_K_[level][3];


    const Eigen::Matrix3d R = initial.block<3, 3>(0, 0);
    const Eigen::Vector3d T = initial.block<3, 1>(0, 3);

    const auto &src_gray = src_->pyramid_gray_[level];
    const auto &src_depth = src_->pyramid_depth_[level];
    const auto &dst_gray = dst_->pyramid_gray_[level];
    const auto &dst_depth = dst_->pyramid_depth_[level];

    const double sqrt_lamba_dep = sqrt(depth_weight_);
    const double sqrt_lambda_img = sqrt(1.0 - depth_weight_);

    double residual_odd = 0;
    double residual_even = 0;

    int corres_num_odd = 0;
    int corres_num_even = 0;

    double ph_odd = 0;
    double ph_even = 0;
    double depth_odd = 0;
    double depth_even = 0;

    const auto &XYZ = *(src_->GetXYZ(level, sample_steps_[level]));

#ifdef MY_USE_OMP
#pragma omp parallel for schedule(dynamic, 1) num_threads(2)
#endif
    for (int y = 0; y < height; y += sample_steps_[level]) {
        // need accumulate
#ifdef MY_USE_OMP
        const int index = omp_get_thread_num();
#else
        const int thread_index = (y/sample_steps_[level])%2;

#endif
        bool isEven = ((thread_index & 1) == 0);
        double &tmp_residual = isEven? residual_even : residual_odd;
        int &cur_corres_num = isEven? corres_num_even : corres_num_odd;

        double &ph_error = isEven? ph_even : ph_odd;
        double &depth_error = isEven? depth_even : depth_odd;

        Eigen::Vector3d pt;
        Eigen::Vector3d world_pt;
        Eigen::Vector3d trans_pt;
        Eigen::Vector3d proj_pt;
        Eigen::Matrix<double, 3, 6> j_pt;
        double *j_pt_ptr = j_pt.data();

        for (int x = 0; x < width; x += sample_steps_[level]) {
            const int index = y * width + x;
            if (std::isnan(XYZ[index][2])) continue;

            pt[0] = XYZ[index][0]; pt[1] = XYZ[index][1]; pt[2] = XYZ[index][2];
            trans_pt[2] = R(2, 0) * pt[0] + R(2, 1) * pt[1]+R(2,2)*pt[2]+T[2];
            trans_pt[1] = R(1, 0) * pt[0] + R(1, 1) * pt[1] + R(1, 2) * pt[2] + T[1];
            trans_pt[0] = R(0, 0) * pt[0] + R(0, 1) * pt[1] + R(0, 2) * pt[2] + T[0];

            proj_pt[2] = trans_pt[2];
            proj_pt[1] = K(1, 0) * trans_pt[0] + K(1, 1) * trans_pt[1] + K(1, 2) * trans_pt[2];
            proj_pt[0] = K(0, 0) * trans_pt[0] + K(0, 1) * trans_pt[1] + K(0, 2) * trans_pt[2];

            double inv_proj_z = 1.0 / trans_pt[2];

            int u_t = (int) (proj_pt[0] / proj_pt[2] + 0.5);
            int v_t = (int) (proj_pt[1] / proj_pt[2] + 0.5);

            if ((u_t < 0 || u_t > width - 1) || (v_t < 0 || v_t > height - 1)) {
                continue;
            }

            float d_t = dst_depth.at<float>(v_t, u_t);
            if (d_t < FLT_EPSILON || std::isnan(d_t)) continue;

            double pixel_scale = 1.0;

            double depth_residual =dst_depth.at<float>(v_t, u_t) - trans_pt[2];
            const double lamda = 1.0;

            if(std::fabs(depth_residual) > 10 * max_depth_diff_){
                continue;
            }
            else if(std::fabs(depth_residual) > max_depth_diff_){
                pixel_scale = exp( lamda * (1 - std::fabs(depth_residual)/max_depth_diff_) );
            }
//            printf("depth diff  %lf   %lf and scale %lf\n",depth_residual, depth_residual/max_depth_diff_, pixel_scale );
            cur_corres_num++;
            double photo_residual = (dst_gray.at<float>(v_t, u_t) - src_gray.at<float>(y, x));

            double tmp_p = pixel_scale * sqrt_lambda_img;
            double tmp_d = pixel_scale * sqrt_lamba_dep;

            photo_residual *= tmp_p;
            depth_residual *= tmp_d;
            tmp_residual += photo_residual * photo_residual + depth_residual * depth_residual;
            ph_error += photo_residual * photo_residual;
            depth_error += depth_residual * depth_residual;
        }

    }
    double total_ph_error = ph_even + ph_odd;
    double total_depth_error = depth_even + depth_odd;
    residual = residual_even + residual_odd;
    corres_num = corres_num_even + corres_num_odd;
}


void RGBDRegistration::ComputeColorState(){
    auto src_gray = src_->pyramid_gray_[0];
    auto src_mask = src_->pyramid_depth_[0];

    auto dst_gray = dst_->pyramid_gray_[0];
    auto dst_mask = dst_->pyramid_depth_[0];

    float illuminate_diff = ComputeIlluminateDiff(src_gray, src_mask, dst_gray, src_mask, 3);
    if(illuminate_diff > 10/255.0f){
        enable_color_ = false;
        printf("[6025][rgbdregistration] close color!");
    }
    else if(illuminate_diff > 5/255.0f) {
        enable_bright_ = true;
        printf("[6025][rgbdregistration] enable bright estimate!");
    }
}


float RGBDRegistration::ComputeIlluminateDiff(cv::Mat gray1, cv::Mat mask1, cv::Mat gray2, cv::Mat mask2, int step){
    if(mask1.empty()){
        mask1 = cv::Mat::ones(gray1.size(), CV_8UC1);
    }
    if(mask2.empty()){
        mask2 = cv::Mat::ones(gray1.size(), CV_8UC1);
    }

    float avg1 = 0, avg2 = 0;
    int cnt1 = 0, cnt2 = 0;
    for(int r = 0; r<gray1.rows; r+=step){
        for(int c = 0; c<gray1.cols; c+=step){
            if(std::isnan(mask1.at<float>(r,c))) continue;
            avg1+=gray1.at<float>(r,c);
            cnt1++;
        }
    }
    avg1/=cnt1;

    for(int r = 0; r<gray2.rows; r+=step){
        for(int c = 0; c<gray2.cols; c+=step){
            if(std::isnan(mask2.at<float>(r,c))) continue;
            avg2+=gray2.at<float>(r,c);
            cnt2++;
        }
    }
    avg2/=cnt2;

    return std::fabs(avg1 - avg2);
}


void RGBDRegistration::ResetState(){
    bright_gain_ = 1.0;
    bright_bias_ = 0.0;

    bright_gain_bk_ = 1.0;
    bright_bias_bk_ = 0.0;

    enable_color_ = true;
    enable_bright_ = false;
}

void RGBDRegistration::UpdateBrightPara(float delta){
    bright_bias_bk_ = bright_bias_;
    bright_bias_+=delta;
}


void RGBDRegistration::RollBackBrightPara(){
     bright_bias_ = bright_bias_bk_;
}






