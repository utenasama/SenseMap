//
// Created by sensetime on 2020/10/29.
//

#include "RGBDPyramid.h"
#include "RGBDAlignUtility.h"

float RGBDPyramid::gaussian_kernel[9] = {0.0625, 0.125, 0.0625, 0.125, 0.25, 0.125, 0.0625, 0.125, 0.0625};;
float RGBDPyramid::sobel_dx_kernel[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
float RGBDPyramid::sobel_dy_kernel[9] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};
cv::Mat RGBDPyramid::cv_gaussian_kernel(3, 3, CV_32FC1, gaussian_kernel);
cv::Mat RGBDPyramid::cv_dx_kernel(3, 3, CV_32FC1, sobel_dx_kernel);
cv::Mat RGBDPyramid::cv_dy_kernel(3, 3, CV_32FC1, sobel_dy_kernel);
double RGBDPyramid::sobel_scale = 0.125;


XYZImage::XYZImage(){}
XYZImage::XYZImage(int w, int h):
width_(w), height_(h){
    Eigen::Vector3f invalid_pt = Eigen::Vector3f(0, 0, std::numeric_limits<float>::quiet_NaN());
    pts_.resize(width_ * height_, invalid_pt);
}


Eigen::Vector3f XYZImage::operator()(int x, int y) const{
    return pts_[y * width_ + x];
}

Eigen::Vector3f &XYZImage::operator()(int x, int y) {
    return pts_[y * width_ + x];
}

Eigen::Vector3f XYZImage::operator[] (int idx) const{
    return pts_[idx];
}
Eigen::Vector3f& XYZImage::operator[] (int idx){
    return pts_[idx];
}

void CreateXYZImage(std::shared_ptr<XYZImage> &xyz, cv::Mat depth, Eigen::Vector4f K, int step) {

    int width = depth.cols;
    int height = depth.rows;
    if(step <= 0) step = 1;
    if(xyz==nullptr) xyz = std::make_shared<XYZImage>(width, height);

    double inv_fx = 1 / (double) K[0];
    double inv_fy = 1 / (double) K[1];
    double cx = K[2];
    double cy = K[3];

    int count = 0;

    for (int y = 0; y < height; y += step) {
        for (int x = 0; x < width; x += step) {
            const float d = depth.at<float>(y, x);
            if (std::isnan(d)) {
                continue;
            }
            count++;
            if(!std::isnan( (*xyz)(x,y)[2])) continue;
            float tx = d * (x - cx) * inv_fx;
            float ty = d * (y - cy) * inv_fy;
            (*xyz)(x,y) = Eigen::Vector3f(tx, ty, d);
        }
    }
    xyz->count_ = count;
    xyz->sample_step_ = step;
    xyz->K_ =  K;
    return;
}

RGBDPyramid::RGBDPyramid(){}

RGBDPyramid::RGBDPyramid(int levels, float scale, Eigen::Vector4f K, cv::Mat gray,
                         cv::Mat depth, cv::Mat conf, cv::Mat highlight, int id) :
        levels_(levels), scale_(scale), orig_K_(K), id_(id) {
    orig_gray_ = gray.clone();
    orig_depth_ = depth.clone();
    if (!conf.empty()) orig_conf_ = conf.clone();
    else orig_conf_ = cv::Mat::ones(orig_depth_.size(), CV_32FC1);
    if (!highlight.empty()) orig_highlight_ = highlight.clone();
    else orig_highlight_ = cv::Mat::zeros(orig_depth_.size(), CV_32FC1);
    CreateImagePyramid();
}


RGBDPyramid::~RGBDPyramid(){}

void RGBDPyramid::GaussianFilter2D(const cv::Mat & src_mat, cv::Mat & dst_mat) {
    const int width = src_mat.cols;
    const int height = src_mat.rows;

    if (width < 4 || height < 2 || src_mat.type() != CV_32FC1) {
        cv::filter2D(src_mat, dst_mat, CV_32FC1, cv_gaussian_kernel);
        return;
    }
    if (dst_mat.cols != width || dst_mat.rows != height || dst_mat.type() != CV_32FC1) {
        dst_mat = cv::Mat(height, width, CV_32FC1);
    }

    // 3x3 Gaussian Filter
    // 0.0625, 0.1250, 0.0625,
    // 0.1250, 0.2500, 0.1250,
    // 0.0625, 0.1250, 0.0625

    // y = 0
    {
        const int y = 0;
        int x = 0;

#if __ARM_NEON
        float left_1 = src_mat.at<float>(y,     1);
        float left_2 = src_mat.at<float>(y + 1, 1);
        float32x4_t s11_0123 = vld1q_f32(&src_mat.at<float>(y,     x));
        float32x4_t s21_0123 = vld1q_f32(&src_mat.at<float>(y + 1, x));
        for ( ; x + 4 <= width; x += 4) {
            // load all source data
            float32x4_t s11_next_0123;
            float32x4_t s21_next_0123;
            if (x + 8 <= width) {
                s11_next_0123 = vld1q_f32(&src_mat.at<float>(y,     x + 4));
                s21_next_0123 = vld1q_f32(&src_mat.at<float>(y + 1, x + 4));
            }
            else if (x + 4 == width) {
                // mirror value
                s11_next_0123 = vld1q_lane_f32(&src_mat.at<float>(y    , x + 2), s11_next_0123, 0);
                s21_next_0123 = vld1q_lane_f32(&src_mat.at<float>(y + 1, x + 2), s21_next_0123, 0);
            }
            else {
                // when (width % 4 != 0)
                s11_next_0123 = vld1q_lane_f32(&src_mat.at<float>(y    , x + 4), s11_next_0123, 0);
                s21_next_0123 = vld1q_lane_f32(&src_mat.at<float>(y + 1, x + 4), s21_next_0123, 0);
            }
            float32x4_t left_1_at3 = vsetq_lane_f32(left_1, left_1_at3, 3);
            float32x4_t left_2_at3 = vsetq_lane_f32(left_2, left_2_at3, 3);
            float32x4_t s12_0123 = vextq_f32(s11_0123, s11_next_0123, 1);
            float32x4_t s22_0123 = vextq_f32(s21_0123, s21_next_0123, 1);
            float32x4_t s10_0123 = vextq_f32(left_1_at3, s11_0123, 3);
            float32x4_t s20_0123 = vextq_f32(left_2_at3, s21_0123, 3);

            // calculate result
            float32x4_t c_0_2500_x4 = vdupq_n_f32(0.2500f);
            float32x4_t c_0_1250_x4 = vdupq_n_f32(0.1250f);
            float32x4_t
            result = vmulq_f32(        s11_0123, c_0_2500_x4);
            result = vmlaq_f32(result, s21_0123, c_0_2500_x4);
            result = vmlaq_f32(result, s12_0123, c_0_1250_x4);
            result = vmlaq_f32(result, s22_0123, c_0_1250_x4);
            result = vmlaq_f32(result, s10_0123, c_0_1250_x4);
            result = vmlaq_f32(result, s20_0123, c_0_1250_x4);

            // store result
            vst1q_f32(&dst_mat.at<float>(y, x), result);

            // update sliding window
            left_1 = vgetq_lane_f32(s11_0123, 3);
            left_2 = vgetq_lane_f32(s21_0123, 3);
            s11_0123 = s11_next_0123;
            s21_0123 = s21_next_0123;
        }
#endif

        for ( ; x < width; x++) {
            float s00, s01, s02, s10, s11 ,s12, s20, s21, s22;
            if (x == 0) {
                s00 = src_mat.at<float>(y + 1, x + 1);
                s01 = src_mat.at<float>(y + 1, x);
                s02 = src_mat.at<float>(y + 1, x + 1);
                s10 = src_mat.at<float>(y, x + 1);
                s11 = src_mat.at<float>(y, x);
                s12 = src_mat.at<float>(y, x + 1);
                s20 = src_mat.at<float>(y + 1, x + 1);
                s21 = src_mat.at<float>(y + 1, x);
                s22 = src_mat.at<float>(y + 1, x + 1);
            }
            else if (x == width - 1) {
                s00 = src_mat.at<float>(y + 1, x - 1);
                s01 = src_mat.at<float>(y + 1, x);
                s02 = src_mat.at<float>(y + 1, x - 1);
                s10 = src_mat.at<float>(y, x - 1);
                s11 = src_mat.at<float>(y, x);
                s12 = src_mat.at<float>(y, x - 1);
                s20 = src_mat.at<float>(y + 1, x - 1);
                s21 = src_mat.at<float>(y + 1, x);
                s22 = src_mat.at<float>(y + 1, x - 1);
            }
            else {
                s00 = src_mat.at<float>(y + 1, x - 1);
                s01 = src_mat.at<float>(y + 1, x);
                s02 = src_mat.at<float>(y + 1, x + 1);
                s10 = src_mat.at<float>(y, x - 1);
                s11 = src_mat.at<float>(y, x);
                s12 = src_mat.at<float>(y, x + 1);
                s20 = src_mat.at<float>(y + 1, x - 1);
                s21 = src_mat.at<float>(y + 1, x);
                s22 = src_mat.at<float>(y + 1, x + 1);
            }

            dst_mat.at<float>(y, x) =
                    s00 * 0.0625f + s01 * 0.1250f + s02 * 0.0625f +
                    s10 * 0.1250f + s11 * 0.2500f + s12 * 0.1250f +
                    s20 * 0.0625f + s21 * 0.1250f + s22 * 0.0625f;
        }
    }

    // y \in [1, height - 2]
    for (int y = 1; y < height - 1; y++) {
        int x = 0;

#if __ARM_NEON
        float left_0 = src_mat.at<float>(y - 1, 1);
        float left_1 = src_mat.at<float>(y,     1);
        float left_2 = src_mat.at<float>(y + 1, 1);
        float32x4_t s01_0123 = vld1q_f32(&src_mat.at<float>(y - 1, x));
        float32x4_t s11_0123 = vld1q_f32(&src_mat.at<float>(y,     x));
        float32x4_t s21_0123 = vld1q_f32(&src_mat.at<float>(y + 1, x));
        for ( ; x + 4 <= width; x += 4) {
            // load all source data
            float32x4_t s01_next_0123;
            float32x4_t s11_next_0123;
            float32x4_t s21_next_0123;
            if (x + 8 <= width) {
                s01_next_0123 = vld1q_f32(&src_mat.at<float>(y - 1, x + 4));
                s11_next_0123 = vld1q_f32(&src_mat.at<float>(y,     x + 4));
                s21_next_0123 = vld1q_f32(&src_mat.at<float>(y + 1, x + 4));
            }
            else if (x + 4 == width) {
                // mirror value
                s01_next_0123 = vld1q_lane_f32(&src_mat.at<float>(y - 1, x + 2), s01_next_0123, 0);
                s11_next_0123 = vld1q_lane_f32(&src_mat.at<float>(y    , x + 2), s11_next_0123, 0);
                s21_next_0123 = vld1q_lane_f32(&src_mat.at<float>(y + 1, x + 2), s21_next_0123, 0);
            }
            else {
                // when (width % 4 != 0)
                s01_next_0123 = vld1q_lane_f32(&src_mat.at<float>(y - 1, x + 4), s01_next_0123, 0);
                s11_next_0123 = vld1q_lane_f32(&src_mat.at<float>(y    , x + 4), s11_next_0123, 0);
                s21_next_0123 = vld1q_lane_f32(&src_mat.at<float>(y + 1, x + 4), s21_next_0123, 0);
            }
            float32x4_t left_0_at3 = vsetq_lane_f32(left_0, left_0_at3, 3);
            float32x4_t left_1_at3 = vsetq_lane_f32(left_1, left_1_at3, 3);
            float32x4_t left_2_at3 = vsetq_lane_f32(left_2, left_2_at3, 3);
            float32x4_t s02_0123 = vextq_f32(s01_0123, s01_next_0123, 1);
            float32x4_t s12_0123 = vextq_f32(s11_0123, s11_next_0123, 1);
            float32x4_t s22_0123 = vextq_f32(s21_0123, s21_next_0123, 1);
            float32x4_t s00_0123 = vextq_f32(left_0_at3, s01_0123, 3);
            float32x4_t s10_0123 = vextq_f32(left_1_at3, s11_0123, 3);
            float32x4_t s20_0123 = vextq_f32(left_2_at3, s21_0123, 3);

            // calculate result
            float32x4_t c_0_1250_x4 = vdupq_n_f32(0.1250f);
            float32x4_t c_0_2500_x4 = vdupq_n_f32(0.2500f);
            float32x4_t c_0_0625_x4 = vdupq_n_f32(0.0625f);
            float32x4_t
            result = vmulq_f32(        s01_0123, c_0_1250_x4);
            result = vmlaq_f32(result, s11_0123, c_0_2500_x4);
            result = vmlaq_f32(result, s21_0123, c_0_1250_x4);
            result = vmlaq_f32(result, s02_0123, c_0_0625_x4);
            result = vmlaq_f32(result, s12_0123, c_0_1250_x4);
            result = vmlaq_f32(result, s22_0123, c_0_0625_x4);
            result = vmlaq_f32(result, s00_0123, c_0_0625_x4);
            result = vmlaq_f32(result, s10_0123, c_0_1250_x4);
            result = vmlaq_f32(result, s20_0123, c_0_0625_x4);

            // store result
            vst1q_f32(&dst_mat.at<float>(y, x), result);

            // update sliding window
            left_0 = vgetq_lane_f32(s01_0123, 3);
            left_1 = vgetq_lane_f32(s11_0123, 3);
            left_2 = vgetq_lane_f32(s21_0123, 3);
            s01_0123 = s01_next_0123;
            s11_0123 = s11_next_0123;
            s21_0123 = s21_next_0123;
        }
#endif

        // tail data and no-SIMD equivacence
        for ( ; x < width; x++) {
            float s00, s01, s02, s10, s11 ,s12, s20, s21, s22;
            if (x == 0) {
                s00 = src_mat.at<float>(y - 1, x + 1);
                s01 = src_mat.at<float>(y - 1, x);
                s02 = src_mat.at<float>(y - 1, x + 1);
                s10 = src_mat.at<float>(y, x + 1);
                s11 = src_mat.at<float>(y, x);
                s12 = src_mat.at<float>(y, x + 1);
                s20 = src_mat.at<float>(y + 1, x + 1);
                s21 = src_mat.at<float>(y + 1, x);
                s22 = src_mat.at<float>(y + 1, x + 1);
            }
            else if (x == width - 1) {
                s00 = src_mat.at<float>(y - 1, x - 1);
                s01 = src_mat.at<float>(y - 1, x);
                s02 = src_mat.at<float>(y - 1, x - 1);
                s10 = src_mat.at<float>(y, x - 1);
                s11 = src_mat.at<float>(y, x);
                s12 = src_mat.at<float>(y, x - 1);
                s20 = src_mat.at<float>(y + 1, x - 1);
                s21 = src_mat.at<float>(y + 1, x);
                s22 = src_mat.at<float>(y + 1, x - 1);
            }
            else {
                s00 = src_mat.at<float>(y - 1, x - 1);
                s01 = src_mat.at<float>(y - 1, x);
                s02 = src_mat.at<float>(y - 1, x + 1);
                s10 = src_mat.at<float>(y, x - 1);
                s11 = src_mat.at<float>(y, x);
                s12 = src_mat.at<float>(y, x + 1);
                s20 = src_mat.at<float>(y + 1, x - 1);
                s21 = src_mat.at<float>(y + 1, x);
                s22 = src_mat.at<float>(y + 1, x + 1);
            }

            dst_mat.at<float>(y, x) =
                    s00 * 0.0625f + s01 * 0.1250f + s02 * 0.0625f +
                    s10 * 0.1250f + s11 * 0.2500f + s12 * 0.1250f +
                    s20 * 0.0625f + s21 * 0.1250f + s22 * 0.0625f;
        }
    }

    // y = height - 1
    {
        const int y = height - 1;
        int x = 0;

#if __ARM_NEON
        float left_0 = src_mat.at<float>(y - 1, 1);
        float left_1 = src_mat.at<float>(y,     1);
        float32x4_t s01_0123 = vld1q_f32(&src_mat.at<float>(y - 1, x));
        float32x4_t s11_0123 = vld1q_f32(&src_mat.at<float>(y,     x));
        for ( ; x + 4 <= width; x += 4) {
            float32x4_t s01_next_0123;
            float32x4_t s11_next_0123;
            // load all source data
            if (x + 8 <= width) {
                s01_next_0123 = vld1q_f32(&src_mat.at<float>(y - 1, x + 4));
                s11_next_0123 = vld1q_f32(&src_mat.at<float>(y,     x + 4));
            }
            else if (x + 4 == width) {
                // mirror value
                s01_next_0123 = vld1q_lane_f32(&src_mat.at<float>(y - 1, x + 2), s01_next_0123, 0);
                s11_next_0123 = vld1q_lane_f32(&src_mat.at<float>(y    , x + 2), s11_next_0123, 0);
            }
            else {
                // when (width % 4 != 0)
                s01_next_0123 = vld1q_lane_f32(&src_mat.at<float>(y - 1, x + 4), s01_next_0123, 0);
                s11_next_0123 = vld1q_lane_f32(&src_mat.at<float>(y    , x + 4), s11_next_0123, 0);
            }
            float32x4_t left_0_at3 = vsetq_lane_f32(left_0, left_0_at3, 3);
            float32x4_t left_1_at3 = vsetq_lane_f32(left_1, left_1_at3, 3);
            float32x4_t s02_0123 = vextq_f32(s01_0123, s01_next_0123, 1);
            float32x4_t s12_0123 = vextq_f32(s11_0123, s11_next_0123, 1);
            float32x4_t s00_0123 = vextq_f32(left_0_at3, s01_0123, 3);
            float32x4_t s10_0123 = vextq_f32(left_1_at3, s11_0123, 3);

            // calculate result
            float32x4_t c_0_2500_x4 = vdupq_n_f32(0.2500f);
            float32x4_t c_0_1250_x4 = vdupq_n_f32(0.1250f);
            float32x4_t
            result = vmulq_f32(        s01_0123, c_0_2500_x4);
            result = vmlaq_f32(result, s11_0123, c_0_2500_x4);
            result = vmlaq_f32(result, s02_0123, c_0_1250_x4);
            result = vmlaq_f32(result, s12_0123, c_0_1250_x4);
            result = vmlaq_f32(result, s00_0123, c_0_1250_x4);
            result = vmlaq_f32(result, s10_0123, c_0_1250_x4);

            // store result
            vst1q_f32(&dst_mat.at<float>(y, x), result);

            // update sliding window
            left_0 = vgetq_lane_f32(s01_0123, 3);
            left_1 = vgetq_lane_f32(s11_0123, 3);
            s01_0123 = s01_next_0123;
            s11_0123 = s11_next_0123;
        }
#endif

        for ( ; x < width; x++) {
            float s00, s01, s02, s10, s11 ,s12, s20, s21, s22;
            if (x == 0) {
                s00 = src_mat.at<float>(y - 1, x + 1);
                s01 = src_mat.at<float>(y - 1, x);
                s02 = src_mat.at<float>(y - 1, x + 1);
                s10 = src_mat.at<float>(y, x + 1);
                s11 = src_mat.at<float>(y, x);
                s12 = src_mat.at<float>(y, x + 1);
                s20 = src_mat.at<float>(y - 1, x + 1);
                s21 = src_mat.at<float>(y - 1, x);
                s22 = src_mat.at<float>(y - 1, x + 1);
            }
            else if (x == width - 1) {
                s00 = src_mat.at<float>(y - 1, x - 1);
                s01 = src_mat.at<float>(y - 1, x);
                s02 = src_mat.at<float>(y - 1, x - 1);
                s10 = src_mat.at<float>(y, x - 1);
                s11 = src_mat.at<float>(y, x);
                s12 = src_mat.at<float>(y, x - 1);
                s20 = src_mat.at<float>(y - 1, x - 1);
                s21 = src_mat.at<float>(y - 1, x);
                s22 = src_mat.at<float>(y - 1, x - 1);
            }
            else {
                s00 = src_mat.at<float>(y - 1, x - 1);
                s01 = src_mat.at<float>(y - 1, x);
                s02 = src_mat.at<float>(y - 1, x + 1);
                s10 = src_mat.at<float>(y, x - 1);
                s11 = src_mat.at<float>(y, x);
                s12 = src_mat.at<float>(y, x + 1);
                s20 = src_mat.at<float>(y - 1, x - 1);
                s21 = src_mat.at<float>(y - 1, x);
                s22 = src_mat.at<float>(y - 1, x + 1);
            }

            dst_mat.at<float>(y, x) =
                    s00 * 0.0625f + s01 * 0.1250f + s02 * 0.0625f +
                    s10 * 0.1250f + s11 * 0.2500f + s12 * 0.1250f +
                    s20 * 0.0625f + s21 * 0.1250f + s22 * 0.0625f;
        }
    }

    // // Check result
    // cv::Mat check_mat;
    // cv::filter2D(src_mat, check_mat, CV_32FC1, cv_gaussian_kernel);
    // check_mat = cv::abs(dst_mat - check_mat);
    // float error = *std::max_element(check_mat.begin<float>(), check_mat.end<float>());
    // if (error > 1E-6) std::abort();
}

void RGBDPyramid::SobelDxDyFilter2D(const cv::Mat & src_mat, cv::Mat & dst_dx_mat, cv::Mat & dst_dy_mat) {
    const int width = src_mat.cols;
    const int height = src_mat.rows;
    if (width < 4 || height < 2 || src_mat.type() != CV_32FC1) {
        cv::filter2D(src_mat, dst_dx_mat, CV_32FC1, cv_dx_kernel);
        cv::filter2D(src_mat, dst_dy_mat, CV_32FC1, cv_dy_kernel);
        return;
    }
    if (dst_dx_mat.cols != width || dst_dx_mat.rows != height || dst_dx_mat.type() != CV_32FC1) {
        dst_dx_mat = cv::Mat(height, width, CV_32FC1);
    }
    if (dst_dy_mat.cols != width || dst_dy_mat.rows != height || dst_dy_mat.type() != CV_32FC1) {
        dst_dy_mat = cv::Mat(height, width, CV_32FC1);
    }

    // 3x3 Sobel-dx Filter
    // -1, 0, 1,
    // -2, 0, 2,
    // -1, 0, 1

    // 3x3 Sobel-dy Filter
    // -1, -2, -1
    // 0, 0, 0,
    // 1, 2, 1

    // y = 0
    {
        const int y = 0;
        int x = 0;

#if __ARM_NEON
        float32x4_t left_1_at3 = vld1q_lane_f32(&src_mat.at<float>(y,     1), left_1_at3, 3);
        float32x4_t left_2_at3 = vld1q_lane_f32(&src_mat.at<float>(y + 1, 1), left_2_at3, 3);
        float32x4_t s10_0123 = vextq_f32(left_1_at3, vld1q_f32(&src_mat.at<float>(y,     x)), 3);
        float32x4_t s20_0123 = vextq_f32(left_2_at3, vld1q_f32(&src_mat.at<float>(y + 1, x)), 3);
        for ( ; x + 4 <= width; x += 4) {
            // load all source data
            float32x4_t s10_next_0123;
            float32x4_t s20_next_0123;
            if (x + 7 <= width) {
                s10_next_0123 = vld1q_f32(&src_mat.at<float>(y,     x + 3));
                s20_next_0123 = vld1q_f32(&src_mat.at<float>(y + 1, x + 3));
            }
            else if (x + 4 == width) {
                // mirror value
                s10_next_0123 = vld1q_lane_f32(&src_mat.at<float>(y    , x + 3), s10_next_0123, 0);
                s20_next_0123 = vld1q_lane_f32(&src_mat.at<float>(y + 1, x + 3), s20_next_0123, 0);
                s10_next_0123 = vld1q_lane_f32(&src_mat.at<float>(y    , x + 2), s10_next_0123, 1);
                s20_next_0123 = vld1q_lane_f32(&src_mat.at<float>(y + 1, x + 2), s20_next_0123, 1);
            }
            else {
                // when (width % 4 != 0)
                s10_next_0123 = vld1q_lane_f32(&src_mat.at<float>(y    , x + 3), s10_next_0123, 0);
                s20_next_0123 = vld1q_lane_f32(&src_mat.at<float>(y + 1, x + 3), s20_next_0123, 0);
                s10_next_0123 = vld1q_lane_f32(&src_mat.at<float>(y    , x + 4), s10_next_0123, 1);
                s20_next_0123 = vld1q_lane_f32(&src_mat.at<float>(y + 1, x + 4), s20_next_0123, 1);
            }
            float32x4_t s12_0123 = vextq_f32(s10_0123, s10_next_0123, 2);
            float32x4_t s22_0123 = vextq_f32(s20_0123, s20_next_0123, 2);

            // calculate result
            float32x4_t c_2_x4 = vdupq_n_f32(2.0f);
            float32x4_t c_n2_x4 = vdupq_n_f32(-2.0f);
            float32x4_t result_dy = vdupq_n_f32(0.0f);
            float32x4_t
            result_dx = vmulq_f32(           s10_0123, c_n2_x4);
            result_dx = vmlaq_f32(result_dx, s20_0123, c_n2_x4);
            result_dx = vmlaq_f32(result_dx, s12_0123, c_2_x4);
            result_dx = vmlaq_f32(result_dx, s22_0123, c_2_x4);

            // store result
            vst1q_f32(&dst_dy_mat.at<float>(y, x), result_dy);
            vst1q_f32(&dst_dx_mat.at<float>(y, x), result_dx);

            // update sliding window
            s10_0123 = s10_next_0123;
            s20_0123 = s20_next_0123;
        }
#endif

        // tail data and no-SIMD equivacence
        for ( ; x < width; x++) {
            float s00, s02, s10, s12, s20, s22;
            if (x == 0) {
                s00 = src_mat.at<float>(y + 1, x + 1);
                s02 = src_mat.at<float>(y + 1, x + 1);
                s10 = src_mat.at<float>(y, x + 1);
                s12 = src_mat.at<float>(y, x + 1);
                s20 = src_mat.at<float>(y + 1, x + 1);
                s22 = src_mat.at<float>(y + 1, x + 1);
            }
            else if (x == width - 1) {
                s00 = src_mat.at<float>(y + 1, x - 1);
                s02 = src_mat.at<float>(y + 1, x - 1);
                s10 = src_mat.at<float>(y, x - 1);
                s12 = src_mat.at<float>(y, x - 1);
                s20 = src_mat.at<float>(y + 1, x - 1);
                s22 = src_mat.at<float>(y + 1, x - 1);
            }
            else {
                s00 = src_mat.at<float>(y + 1, x - 1);
                s02 = src_mat.at<float>(y + 1, x + 1);
                s10 = src_mat.at<float>(y, x - 1);
                s12 = src_mat.at<float>(y, x + 1);
                s20 = src_mat.at<float>(y + 1, x - 1);
                s22 = src_mat.at<float>(y + 1, x + 1);
            }

            dst_dy_mat.at<float>(y, x) = 0.0f;
            dst_dx_mat.at<float>(y, x) =
                    s00 * -1.0f + s02 * 1.0f +
                    s10 * -2.0f + s12 * 2.0f +
                    s20 * -1.0f + s22 * 1.0f;
        }
    }

    // y \in [1, height - 2]
    for (int y = 1; y < height - 1; y++) {
        int x = 0;

#if __ARM_NEON
        float left_0 = src_mat.at<float>(y - 1, 1);
        float left_1 = src_mat.at<float>(y,     1);
        float left_2 = src_mat.at<float>(y + 1, 1);
        float32x4_t s01_0123 = vld1q_f32(&src_mat.at<float>(y - 1, x));
        float32x4_t s11_0123 = vld1q_f32(&src_mat.at<float>(y,     x));
        float32x4_t s21_0123 = vld1q_f32(&src_mat.at<float>(y + 1, x));
        for ( ; x + 4 <= width; x += 4) {
            // load all source data
            float32x4_t s01_next_0123;
            float32x4_t s11_next_0123;
            float32x4_t s21_next_0123;
            if (x + 8 <= width) {
                s01_next_0123 = vld1q_f32(&src_mat.at<float>(y - 1, x + 4));
                s11_next_0123 = vld1q_f32(&src_mat.at<float>(y,     x + 4));
                s21_next_0123 = vld1q_f32(&src_mat.at<float>(y + 1, x + 4));
            }
            else if (x + 4 == width) {
                // mirror value
                s01_next_0123 = vld1q_lane_f32(&src_mat.at<float>(y - 1, x + 2), s01_next_0123, 0);
                s11_next_0123 = vld1q_lane_f32(&src_mat.at<float>(y    , x + 2), s11_next_0123, 0);
                s21_next_0123 = vld1q_lane_f32(&src_mat.at<float>(y + 1, x + 2), s21_next_0123, 0);
            }
            else {
                // when (width % 4 != 0)
                s01_next_0123 = vld1q_lane_f32(&src_mat.at<float>(y - 1, x + 4), s01_next_0123, 0);
                s11_next_0123 = vld1q_lane_f32(&src_mat.at<float>(y    , x + 4), s11_next_0123, 0);
                s21_next_0123 = vld1q_lane_f32(&src_mat.at<float>(y + 1, x + 4), s21_next_0123, 0);
            }
            float32x4_t left_0_at3 = vsetq_lane_f32(left_0, left_0_at3, 3);
            float32x4_t left_1_at3 = vsetq_lane_f32(left_1, left_1_at3, 3);
            float32x4_t left_2_at3 = vsetq_lane_f32(left_2, left_2_at3, 3);
            float32x4_t s02_0123 = vextq_f32(s01_0123, s01_next_0123, 1);
            float32x4_t s12_0123 = vextq_f32(s11_0123, s11_next_0123, 1);
            float32x4_t s22_0123 = vextq_f32(s21_0123, s21_next_0123, 1);
            float32x4_t s00_0123 = vextq_f32(left_0_at3, s01_0123, 3);
            float32x4_t s10_0123 = vextq_f32(left_1_at3, s11_0123, 3);
            float32x4_t s20_0123 = vextq_f32(left_2_at3, s21_0123, 3);

            // calculate result
            float32x4_t c_n2_x4 = vdupq_n_f32(-2.0f);
            float32x4_t c_2_x4 = vdupq_n_f32(2.0f);
            float32x4_t result_dy_a = vaddq_f32(s02_0123, s00_0123);
            float32x4_t result_dy_b = vaddq_f32(s22_0123, s20_0123);
            float32x4_t
            result_dy = vsubq_f32(result_dy_b, result_dy_a);
            result_dy = vmlaq_f32(result_dy, s01_0123, c_n2_x4);
            result_dy = vmlaq_f32(result_dy, s21_0123, c_2_x4);
            float32x4_t result_dx_a = vaddq_f32(s00_0123, s20_0123);
            float32x4_t result_dx_b = vaddq_f32(s02_0123, s22_0123);
            float32x4_t
            result_dx = vsubq_f32(result_dx_b, result_dx_a);
            result_dx = vmlaq_f32(result_dx, s10_0123, c_n2_x4);
            result_dx = vmlaq_f32(result_dx, s12_0123, c_2_x4);

            // store result
            vst1q_f32(&dst_dy_mat.at<float>(y, x), result_dy);
            vst1q_f32(&dst_dx_mat.at<float>(y, x), result_dx);

            // update sliding window
            left_0 = vgetq_lane_f32(s01_0123, 3);
            left_1 = vgetq_lane_f32(s11_0123, 3);
            left_2 = vgetq_lane_f32(s21_0123, 3);
            s01_0123 = s01_next_0123;
            s11_0123 = s11_next_0123;
            s21_0123 = s21_next_0123;
        }
#endif

        // tail data and no-SIMD equivacence
        for ( ; x < width; x++) {
            float s00, s01, s02, s10, s11 ,s12, s20, s21, s22;
            if (x == 0) {
                s00 = src_mat.at<float>(y - 1, x + 1);
                s01 = src_mat.at<float>(y - 1, x);
                s02 = src_mat.at<float>(y - 1, x + 1);
                s10 = src_mat.at<float>(y, x + 1);
                s11 = src_mat.at<float>(y, x);
                s12 = src_mat.at<float>(y, x + 1);
                s20 = src_mat.at<float>(y + 1, x + 1);
                s21 = src_mat.at<float>(y + 1, x);
                s22 = src_mat.at<float>(y + 1, x + 1);
            }
            else if (x == width - 1) {
                s00 = src_mat.at<float>(y - 1, x - 1);
                s01 = src_mat.at<float>(y - 1, x);
                s02 = src_mat.at<float>(y - 1, x - 1);
                s10 = src_mat.at<float>(y, x - 1);
                s11 = src_mat.at<float>(y, x);
                s12 = src_mat.at<float>(y, x - 1);
                s20 = src_mat.at<float>(y + 1, x - 1);
                s21 = src_mat.at<float>(y + 1, x);
                s22 = src_mat.at<float>(y + 1, x - 1);
            }
            else {
                s00 = src_mat.at<float>(y - 1, x - 1);
                s01 = src_mat.at<float>(y - 1, x);
                s02 = src_mat.at<float>(y - 1, x + 1);
                s10 = src_mat.at<float>(y, x - 1);
                s11 = src_mat.at<float>(y, x);
                s12 = src_mat.at<float>(y, x + 1);
                s20 = src_mat.at<float>(y + 1, x - 1);
                s21 = src_mat.at<float>(y + 1, x);
                s22 = src_mat.at<float>(y + 1, x + 1);
            }

            dst_dy_mat.at<float>(y, x) =
                    s00 * -1.0f + s01 * -2.0f + s02 * -1.0f +
                    s20 * 1.0f  + s21 * 2.0f  + s22 * 1.0f;
            dst_dx_mat.at<float>(y, x) =
                    s00 * -1.0f + s02 * 1.0f +
                    s10 * -2.0f + s12 * 2.0f +
                    s20 * -1.0f + s22 * 1.0f;
        }
    }

    // y = height - 1
    {
        const int y = height - 1;
        int x = 0;

#if __ARM_NEON
        float32x4_t left_0_at3 = vld1q_lane_f32(&src_mat.at<float>(y - 1, 1), left_0_at3, 3);
        float32x4_t left_1_at3 = vld1q_lane_f32(&src_mat.at<float>(y,     1), left_1_at3, 3);
        float32x4_t s00_0123 = vextq_f32(left_0_at3, vld1q_f32(&src_mat.at<float>(y - 1, x)), 3);
        float32x4_t s10_0123 = vextq_f32(left_1_at3, vld1q_f32(&src_mat.at<float>(y,     x)), 3);
        for ( ; x + 4 <= width; x += 4) {
            // load all source data
            float32x4_t s00_next_0123;
            float32x4_t s10_next_0123;
            if (x + 7 <= width) {
                s00_next_0123 = vld1q_f32(&src_mat.at<float>(y - 1, x + 3));
                s10_next_0123 = vld1q_f32(&src_mat.at<float>(y,     x + 3));
            }
            else if (x + 4 == width) {
                // mirror value
                s00_next_0123 = vld1q_lane_f32(&src_mat.at<float>(y - 1, x + 3), s00_next_0123, 0);
                s10_next_0123 = vld1q_lane_f32(&src_mat.at<float>(y    , x + 3), s10_next_0123, 0);
                s00_next_0123 = vld1q_lane_f32(&src_mat.at<float>(y - 1, x + 2), s00_next_0123, 1);
                s10_next_0123 = vld1q_lane_f32(&src_mat.at<float>(y    , x + 2), s10_next_0123, 1);
            }
            else {
                // when (width % 4 != 0)
                s00_next_0123 = vld1q_lane_f32(&src_mat.at<float>(y - 1, x + 3), s00_next_0123, 0);
                s10_next_0123 = vld1q_lane_f32(&src_mat.at<float>(y    , x + 3), s10_next_0123, 0);
                s00_next_0123 = vld1q_lane_f32(&src_mat.at<float>(y - 1, x + 4), s00_next_0123, 1);
                s10_next_0123 = vld1q_lane_f32(&src_mat.at<float>(y    , x + 4), s10_next_0123, 1);
            }
            float32x4_t s02_0123 = vextq_f32(s00_0123, s00_next_0123, 2);
            float32x4_t s12_0123 = vextq_f32(s10_0123, s10_next_0123, 2);

            // calculate result
            float32x4_t c_n2_x4 = vdupq_n_f32(-2.0f);
            float32x4_t c_2_x4 = vdupq_n_f32(2.0f);
            float32x4_t result_dy = vdupq_n_f32(0.0f);
            float32x4_t
            result_dx = vmulq_f32(           s00_0123, c_n2_x4);
            result_dx = vmlaq_f32(result_dx, s10_0123, c_n2_x4);
            result_dx = vmlaq_f32(result_dx, s02_0123, c_2_x4);
            result_dx = vmlaq_f32(result_dx, s12_0123, c_2_x4);

            // store result
            vst1q_f32(&dst_dy_mat.at<float>(y, x), result_dy);
            vst1q_f32(&dst_dx_mat.at<float>(y, x), result_dx);

            // update sliding window
            s00_0123 = s00_next_0123;
            s10_0123 = s10_next_0123;
        }
#endif

        // tail data and no-SIMD equivacence
        for ( ; x < width; x++) {
            float s00, s02, s10, s12, s20, s22;
            if (x == 0) {
                s00 = src_mat.at<float>(y - 1, x + 1);
                s02 = src_mat.at<float>(y - 1, x + 1);
                s10 = src_mat.at<float>(y, x + 1);
                s12 = src_mat.at<float>(y, x + 1);
                s20 = src_mat.at<float>(y - 1, x + 1);
                s22 = src_mat.at<float>(y - 1, x + 1);
            }
            else if (x == width - 1) {
                s00 = src_mat.at<float>(y - 1, x - 1);
                s02 = src_mat.at<float>(y - 1, x - 1);
                s10 = src_mat.at<float>(y, x - 1);
                s12 = src_mat.at<float>(y, x - 1);
                s20 = src_mat.at<float>(y - 1, x - 1);
                s22 = src_mat.at<float>(y - 1, x - 1);
            }
            else {
                s00 = src_mat.at<float>(y - 1, x - 1);
                s02 = src_mat.at<float>(y - 1, x + 1);
                s10 = src_mat.at<float>(y, x - 1);
                s12 = src_mat.at<float>(y, x + 1);
                s20 = src_mat.at<float>(y - 1, x - 1);
                s22 = src_mat.at<float>(y - 1, x + 1);
            }

            dst_dy_mat.at<float>(y, x) = 0.0f;
            dst_dx_mat.at<float>(y, x) =
                    s00 * -1.0f + s02 * 1.0f +
                    s10 * -2.0f + s12 * 2.0f +
                    s20 * -1.0f + s22 * 1.0f;
        }
    }

    // // Check result
    // cv::Mat check_dx_mat, check_dy_mat;
    // cv::filter2D(src_mat, check_dx_mat, CV_32FC1, cv_dx_kernel);
    // cv::filter2D(src_mat, check_dy_mat, CV_32FC1, cv_dy_kernel);
    // check_dx_mat = cv::abs(dst_dx_mat - check_dx_mat);
    // check_dy_mat = cv::abs(dst_dy_mat - check_dy_mat);
    // float error = *std::max_element(check_dx_mat.begin<float>(), check_dx_mat.end<float>());
    // if (error > 1E-6) std::abort();
    // error = *std::max_element(check_dy_mat.begin<float>(), check_dy_mat.end<float>());
    // if (error > 1E-6) std::abort();
}

cv::Mat RGBDPyramid::DownSampleImage(cv::Mat &input, bool with_gaussian_filter, float scale) {
//    cv::Mat kernel(3, 3, CV_32FC1, gaussian_kernel);
    cv::Mat ret;

    int new_width = input.cols/scale;
    int new_height = input.rows/scale;

    cv::Size s(new_width, new_height);

    if (!with_gaussian_filter) {
        cv::resize(input, ret, s);
    } else {
        cv::Mat filter_input;
//        cv::filter2D(input, filter_input, CV_32FC1, cv_gaussian_kernel);
        GaussianFilter2D(input, filter_input);
        cv::resize(filter_input, ret, s);
    }
    return ret;
}

cv::Mat RGBDPyramid::DownSampleImage2(cv::Mat &input, bool with_gaussian_filter) {
    int width = floor(input.cols / 2.0);
    int height = floor(input.rows / 2.0);
    cv::Mat output = cv::Mat(height, width, CV_32FC1);
    cv::Mat img;
    if(!with_gaussian_filter){
        img = input;
    }
    else {
//        cv::filter2D(input, img, CV_32FC1, cv_gaussian_kernel);
        GaussianFilter2D(input, img);
    }
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float p1 = img.at<float>(y * 2, x * 2);
            float p2 = img.at<float>(y * 2, x * 2 + 1);
            float p3 = img.at<float>(y * 2 + 1, x * 2);
            float p4 = img.at<float>(y * 2 + 1, x * 2 + 1);
            output.at<float>(y, x) = (p1 + p2 + p3 + p4) / 4.0f;
        }
    }
    return output;

}

void RGBDPyramid::PreProcess() {
//    LOGD("[RGBDPyramid] PreProcess start %d\n", levels_);
//    cv::Mat kernel(3, 3, CV_32FC1, gaussian_kernel);

    cv::Mat floatImage;
    cv::Mat filter_gray, filter_depth;
//    cv::Mat depth_with_nan = orig_depth_;
    cv::Mat depth_with_nan = orig_depth_.clone();
    for (int r = 0; r < depth_with_nan.rows; r++) {
        for (int c = 0; c < depth_with_nan.cols; c++) {
            if (depth_with_nan.at<float>(r, c) < FLT_EPSILON) {
                depth_with_nan.at<float>(r, c) = std::numeric_limits<float>::quiet_NaN();
            }
        }
    }

    orig_gray_.convertTo(floatImage, CV_32FC1, 1.0 / 255.0f);
//    cv::filter2D(floatImage, filter_gray, CV_32FC1, cv_gaussian_kernel);
    GaussianFilter2D(floatImage, filter_gray);
    pyramid_gray_.emplace_back(filter_gray);
//    cv::filter2D(depth_with_nan, filter_depth, CV_32FC1, cv_gaussian_kernel);
    GaussianFilter2D(depth_with_nan, filter_depth);
    pyramid_depth_.emplace_back(filter_depth);
    pyramid_K_.push_back(orig_K_);
    pyramid_conf_.emplace_back(orig_conf_);
    pyramid_highlight_.emplace_back(orig_highlight_);
    //    LOGD("[RGBDPyramid] PreProcess complete\n");

}

void RGBDPyramid::UpdateDepth(cv::Mat depth){
    orig_depth_ = depth;
    cv::Mat depth_with_nan = orig_depth_;

//    cv::Mat depth_with_nan = orig_depth_.clone();
//    for (int r = 0; r < depth_with_nan.rows; r++) {
//        for (int c = 0; c < depth_with_nan.cols; c++) {
//            if (depth_with_nan.at<float>(r, c) < FLT_EPSILON) {
//                depth_with_nan.at<float>(r, c) = std::numeric_limits<float>::quiet_NaN();
//            }
//        }
//    }

    cv::Mat filter_depth;
//    cv::filter2D(depth_with_nan, filter_depth, CV_32FC1, cv_gaussian_kernel);
    GaussianFilter2D(depth_with_nan, filter_depth);
    pyramid_depth_[0] = filter_depth;


    for (int i = 1; i < levels_; i++) {
        if (i == 1) {
            cv::Mat sample_depth = DownSampleImage(pyramid_depth_[i - 1], false, scale_);
            pyramid_depth_[i] = sample_depth;
        } else {
            cv::Mat sample_depth = DownSampleImage(pyramid_depth_[i - 1], false, scale_);
            pyramid_depth_[i] = sample_depth;
        }
    }

    //compute dx dy for pyramid depth
    for (int i = 0; i < levels_; i++) {
        cv::Mat cv_dx, cv_dy;
//        cv::filter2D(pyramid_depth_[i], cv_dx, CV_32FC1, cv_dx_kernel);
//        cv::filter2D(pyramid_depth_[i], cv_dy, CV_32FC1, cv_dy_kernel);
        SobelDxDyFilter2D(pyramid_depth_[i], cv_dx, cv_dy);
        pyramid_depth_dx_[i] = cv_dx;
        pyramid_depth_dy_[i] = cv_dy;
    }

    return;
}

void RGBDPyramid::CreateImagePyramid() {
    PreProcess();
    for (int i = 1; i < levels_; i++) {
        pyramid_K_.push_back(pyramid_K_[i - 1] / scale_);
    }
    for (int i = 1; i < levels_; i++) {
        if (i == 1) {
            cv::Mat gray = DownSampleImage(pyramid_gray_[i - 1], false, scale_);
            pyramid_gray_.emplace_back(gray);
        } else {
            cv::Mat gray = DownSampleImage(pyramid_gray_[i - 1], true, scale_);
            pyramid_gray_.emplace_back(gray);
        }
    }
    for (int i = 1; i < levels_; i++) {
        if (i == 1) {
            cv::Mat depth = DownSampleImage(pyramid_depth_[i - 1], false, scale_);
            pyramid_depth_.emplace_back(depth);
        } else {
            cv::Mat depth = DownSampleImage(pyramid_depth_[i - 1], false, scale_);
            pyramid_depth_.emplace_back(depth);
        }
    }
    for (int i = 1; i < levels_; i++) {
        cv::Mat conf = DownSampleImage(pyramid_conf_[i - 1], false, scale_);
        pyramid_conf_.emplace_back(conf);
    }

    for (int i = 1; i < levels_; i++) {
        cv::Mat highlight = DownSampleImage(pyramid_highlight_[i - 1], false, scale_);
        cv::Mat highlight2 = highlight > 60;
        pyramid_highlight_.emplace_back(highlight2);
    }
    //compute dx dy for pyramid gray
    for (int i = 0; i < levels_; i++) {
        cv::Mat cv_dx, cv_dy;
//        cv::filter2D(pyramid_gray_[i], cv_dx, CV_32FC1, cv_dx_kernel);
//        cv::filter2D(pyramid_gray_[i], cv_dy, CV_32FC1, cv_dy_kernel);
        SobelDxDyFilter2D(pyramid_gray_[i], cv_dx, cv_dy);
        pyramid_gray_dx_.emplace_back(cv_dx);
        pyramid_gray_dy_.emplace_back(cv_dy);
    }
    //compute dx dy for pyramid depth
    for (int i = 0; i < levels_; i++) {
        cv::Mat cv_dx, cv_dy;
//        cv::filter2D(pyramid_depth_[i], cv_dx, CV_32FC1, cv_dx_kernel);
//        cv::filter2D(pyramid_depth_[i], cv_dy, CV_32FC1, cv_dy_kernel);
        SobelDxDyFilter2D(pyramid_depth_[i], cv_dx, cv_dy);
        pyramid_depth_dx_.emplace_back(cv_dx);
        pyramid_depth_dy_.emplace_back(cv_dy);
    }
}

void RGBDPyramid::AddLevel(int level){
    if(level<=levels_) return;

    for (int i = levels_; i < level; i++) {
        pyramid_K_.push_back(pyramid_K_[i - 1] / scale_);
    }

    for (int i = levels_; i < level; i++) {
        if (i == 1) {
            cv::Mat gray = DownSampleImage(pyramid_gray_[i - 1], false, scale_);
            pyramid_gray_.emplace_back(gray);
        } else {
            cv::Mat gray = DownSampleImage(pyramid_gray_[i - 1], true, scale_);
            pyramid_gray_.emplace_back(gray);
        }
    }
    for (int i = levels_; i < level; i++) {
        if (i == 1) {
            cv::Mat depth = DownSampleImage(pyramid_depth_[i - 1], false, scale_);
            pyramid_depth_.emplace_back(depth);
        } else {
            cv::Mat depth = DownSampleImage(pyramid_depth_[i - 1], false, scale_);
            pyramid_depth_.emplace_back(depth);
        }
    }
    for (int i = levels_; i < level; i++) {
        cv::Mat conf = DownSampleImage(pyramid_conf_[i - 1], false, scale_);
        pyramid_conf_.emplace_back(conf);
    }

    for (int i = levels_; i < level; i++) {
        cv::Mat highlight = DownSampleImage(pyramid_highlight_[i - 1], false, scale_);
        cv::Mat highlight2 = highlight > 60;
        pyramid_highlight_.emplace_back(highlight2);
    }


    //compute dx dy for pyramid gray
    for (int i = levels_; i < level; i++) {
        cv::Mat cv_dx, cv_dy;
//        cv::filter2D(pyramid_gray_[i], cv_dx, CV_32FC1, cv_dx_kernel);
//        cv::filter2D(pyramid_gray_[i], cv_dy, CV_32FC1, cv_dy_kernel);
        SobelDxDyFilter2D(pyramid_gray_[i], cv_dx, cv_dy);
        pyramid_gray_dx_.emplace_back(cv_dx);
        pyramid_gray_dy_.emplace_back(cv_dy);
    }

    //compute dx dy for pyramid depth
    for (int i = levels_; i < level; i++) {
        cv::Mat cv_dx, cv_dy;
//        cv::filter2D(pyramid_depth_[i], cv_dx, CV_32FC1, cv_dx_kernel);
//        cv::filter2D(pyramid_depth_[i], cv_dy, CV_32FC1, cv_dy_kernel);
        SobelDxDyFilter2D(pyramid_depth_[i], cv_dx, cv_dy);
        pyramid_depth_dx_.emplace_back(cv_dx);
        pyramid_depth_dy_.emplace_back(cv_dy);
    }

    levels_ = level;
}

void RGBDPyramid::UpdateIntrinsic(Eigen::Vector4f K){
    pyramid_K_.clear();
    orig_K_ = K;
    pyramid_K_.push_back(orig_K_);
    for (int i = 1; i < levels_; i++) {
        pyramid_K_.push_back(pyramid_K_[i - 1] / scale_);
    }
}

std::shared_ptr<XYZImage> RGBDPyramid::GetXYZ(int level, int sample_step){
    if(level>levels_) return nullptr;
    if(xyzs_.count(level)){
        auto xyz = xyzs_[level];
        if(sample_step <=0 || sample_step % xyz->sample_step_==0) return xyz;
    }
    std::shared_ptr<XYZImage> xyz = nullptr;
    CreateXYZImage(xyz, pyramid_depth_[level], pyramid_K_[level], sample_step);
    xyzs_[level] = xyz;
    return xyz;
}

void SaveDepthAsObj(std::string filename, std::shared_ptr<RGBDPyramid> pyr, Eigen::Matrix4f pose){
    auto depth = pyr->orig_depth_;
    auto gray = pyr->orig_gray_;
    auto K_vec = pyr->orig_K_;
    Eigen::Matrix3f K;
    K<<K_vec[0], 0, K_vec[2], 0, K_vec[1], K_vec[3], 0, 0, 1;
    SaveDepthAsObj(filename, depth, gray, K, pose);
}