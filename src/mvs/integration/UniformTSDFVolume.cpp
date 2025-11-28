#include "UniformTSDFVolume.h"

namespace sensemap {

namespace tsdf {

void TSDFVolume<>::Integrate(
        const cv::Mat &depth,
        const Eigen::Matrix3f &intrinsic,
        const Eigen::Matrix4f &extrinsic,
        bool use_sdf_weight, bool de_integrate,
        float base_weight)
{
    const float fx = intrinsic(0, 0);
    const float fy = intrinsic(1, 1);
    const float cx = intrinsic(0, 2);
    const float cy = intrinsic(1, 2);
    const float half_voxel_length = voxel_length_ * 0.5f;
    const float sdf_trunc_inv = 1.0f / sdf_trunc_;
    const Eigen::Matrix4f extrinsic_scaled = extrinsic * voxel_length_;

    for (int x = 0; x < resolution_; x++) {
        for (int y = 0; y < resolution_; y++) {
            int idx_shift = x * resolution_ * resolution_ + y * resolution_;
            tsdf_t   *p_tsdf   = tsdf_.data()   + idx_shift;
            weight_t *p_weight = weight_.data() + idx_shift;
            Eigen::Vector4f voxel_pt_camera = extrinsic * Eigen::Vector4f(
                    half_voxel_length + voxel_length_ * x + origin_(0),
                    half_voxel_length + voxel_length_ * y + origin_(1),
                    half_voxel_length + origin_(2),
                    1.0f);
            for (int z = 0; z < resolution_; z++,
                    voxel_pt_camera(0) += extrinsic_scaled(0, 2),
                    voxel_pt_camera(1) += extrinsic_scaled(1, 2),
                    voxel_pt_camera(2) += extrinsic_scaled(2, 2),
                    p_tsdf++, p_weight++) {
                if (voxel_pt_camera(2) > 0) {

                    float u_f = voxel_pt_camera(0) * fx /
                            voxel_pt_camera(2) + cx;
                    float v_f = voxel_pt_camera(1) * fy /
                            voxel_pt_camera(2) + cy;

                    int v0 = std::floor(v_f);
                    int v1 = std::ceil (v_f);
                    int u0 = std::floor(u_f);
                    int u1 = std::ceil (u_f);
                    if (u0 < 0 || v0 < 0 || u1 >= depth.cols || v1 >= depth.rows) continue;

                    float xx[2];
                    float yy[2];
                    for (int j = 0; j < 2; j++) {
                        xx[j] = (u0 + j - cx) / fx;
                    }
                    for (int i = 0; i < 2; i++) {
                        yy[i] = (v0 + i - cy) / fy;
                    }

                    // bilinear interpolate depth
                    float bilinear_sigma = 0.0f;
                    float bilinear_depth = 0.0f;
                    float bilinear_weight = 0.0f;
                    {
                        const float v_weight = 1.0f - (v_f - v0);
                        const float u_weight = 1.0f - (u_f - u0);
                        const float weight = v_weight * u_weight;

                        const float val = depth.at<float>(v0, u0);
                        if (val > 0.0f) {
                            bilinear_sigma += weight * std::sqrt(yy[0] * yy[0] + xx[0] * xx[0] + 1.0f);
                            bilinear_depth += weight * val;
                            bilinear_weight += weight;
                        }
                    }
                    {
                        const float v_weight = 1.0f - (v_f - v0);
                        const float u_weight = 1.0f - (u1 - u_f);
                        const float weight = v_weight * u_weight;

                        const float val = depth.at<float>(v0, u1);
                        if (val > 0.0f) {
                            bilinear_sigma += weight * std::sqrt(yy[0] * yy[0] + xx[1] * xx[1] + + 1.0f);
                            bilinear_depth += weight * val;
                            bilinear_weight += weight;
                        }
                    }
                    {
                        const float v_weight = 1.0f - (v1 - v_f);
                        const float u_weight = 1.0f - (u_f - u0);
                        const float weight = v_weight * u_weight;

                        const float val = depth.at<float>(v1, u0);
                        if (val > 0.0f) {
                            bilinear_sigma += weight * std::sqrt(yy[1] * yy[1] + xx[0] * xx[0] + 1.0f);
                            bilinear_depth += weight * val;
                            bilinear_weight += weight;
                        }
                    }
                    {
                        const float v_weight = 1.0f - (v1 - v_f);
                        const float u_weight = 1.0f - (u1 - u_f);
                        const float weight = v_weight * u_weight;

                        const float val = depth.at<float>(v1, u1);
                        if (val > 0.0f) {
                            bilinear_sigma += weight * std::sqrt(yy[1] * yy[1] + xx[1] * xx[1] + 1.0f);
                            bilinear_depth += weight * val;
                            bilinear_weight += weight;
                        }
                    }
                    if (bilinear_weight <= 0.0f) continue;
                    bilinear_sigma /= bilinear_weight;
                    bilinear_depth /= bilinear_weight;

                    if (bilinear_depth > min_depth_ && bilinear_depth < max_depth_) {
                        float sdf = (bilinear_depth - voxel_pt_camera(2)) * bilinear_sigma;

                        if (sdf > -sdf_trunc_) {
                            /// integrate
                            float tsdf = std::min(1.0f, sdf * sdf_trunc_inv);
                            float weight = use_sdf_weight ? 
                                            std::max(0.1f, 1.0f - std::fabs(tsdf)) : 
                                            1.0f;
                            weight *= base_weight;
                            if (de_integrate) {
                                weight *= -1.0f;
                            }
                            
                            const float old_weight = *p_weight;
                            const float inv_weight = 1.0f / (old_weight + weight);
                            *p_weight = old_weight + weight;

                            const float old_tsdf = *p_tsdf;
                            *p_tsdf = (old_tsdf * old_weight + tsdf * weight) * inv_weight;

                            if (de_integrate && *p_weight < std::numeric_limits<float>::epsilon()) {
                                *p_tsdf = TSDF_DEFAULT_TSDF;
                                *p_weight = TSDF_DEFAULT_WEIGHT;
                            }
                        }
                    }
                }
            }
        }
    }
}

void ColoredTSDFVolume<>::Integrate(
        const cv::Mat &depth, const cv::Mat &color, 
        const Eigen::Matrix3f &intrinsic,
        const Eigen::Matrix4f &extrinsic,
        bool use_sdf_weight, bool de_integrate,
        float base_weight)
{
    const float fx = intrinsic(0, 0);
    const float fy = intrinsic(1, 1);
    const float cx = intrinsic(0, 2);
    const float cy = intrinsic(1, 2);
    const float half_voxel_length = voxel_length_ * 0.5f;
    const float sdf_trunc_inv = 1.0f / sdf_trunc_;
    const Eigen::Matrix4f extrinsic_scaled = extrinsic * voxel_length_;
    const float width_scale = 1.0f * color.cols / depth.cols;
    const float height_scale = 1.0f * color.rows / depth.rows;

    for (int x = 0; x < resolution_; x++) {
        for (int y = 0; y < resolution_; y++) {
            int idx_shift = x * resolution_ * resolution_ + y * resolution_;
            tsdf_t   *p_tsdf   = tsdf_.data()   + idx_shift;
            weight_t *p_weight = weight_.data() + idx_shift;
            color_t  *p_color  = color_.data()  + idx_shift;
            Eigen::Vector4f voxel_pt_camera = extrinsic * Eigen::Vector4f(
                    half_voxel_length + voxel_length_ * x + origin_(0),
                    half_voxel_length + voxel_length_ * y + origin_(1),
                    half_voxel_length + origin_(2),
                    1.0f);
            for (int z = 0; z < resolution_; z++,
                    voxel_pt_camera(0) += extrinsic_scaled(0, 2),
                    voxel_pt_camera(1) += extrinsic_scaled(1, 2),
                    voxel_pt_camera(2) += extrinsic_scaled(2, 2),
                    p_tsdf++, p_weight++, p_color++) {
                if (voxel_pt_camera(2) > 0) {

                    float u_f = voxel_pt_camera(0) * fx /
                            voxel_pt_camera(2) + cx;
                    float v_f = voxel_pt_camera(1) * fy /
                            voxel_pt_camera(2) + cy;

                    float rgb_u_f = u_f * width_scale;
                    float rgb_v_f = v_f * height_scale;

                    int v0 = std::floor(v_f);
                    int v1 = std::ceil (v_f);
                    int u0 = std::floor(u_f);
                    int u1 = std::ceil (u_f);
                    int rgb_u = (int)(rgb_u_f + 0.5f);
                    int rgb_v = (int)(rgb_v_f + 0.5f);
                    if (u0 < 0 || v0 < 0 || u1 >= depth.cols || v1 >= depth.rows) continue;
                    if (rgb_u < 0 || rgb_v < 0 || rgb_u >= color.cols || rgb_v >= color.rows) continue;

                    float xx[2];
                    float yy[2];
                    for (int j = 0; j < 2; j++) {
                        xx[j] = (u0 + j - cx) / fx;
                    }
                    for (int i = 0; i < 2; i++) {
                        yy[i] = (v0 + i - cy) / fy;
                    }

                    // bilinear interpolate depth
                    float bilinear_sigma = 0.0f;
                    float bilinear_depth = 0.0f;
                    float bilinear_weight = 0.0f;
                    {
                        const float v_weight = 1.0f - (v_f - v0);
                        const float u_weight = 1.0f - (u_f - u0);
                        const float weight = v_weight * u_weight;

                        const float val = depth.at<float>(v0, u0);
                        if (val > 0.0f) {
                            bilinear_sigma += weight * std::sqrt(yy[0] * yy[0] + xx[0] * xx[0] + 1.0f);
                            bilinear_depth += weight * val;
                            bilinear_weight += weight;
                        }
                    }
                    {
                        const float v_weight = 1.0f - (v_f - v0);
                        const float u_weight = 1.0f - (u1 - u_f);
                        const float weight = v_weight * u_weight;

                        const float val = depth.at<float>(v0, u1);
                        if (val > 0.0f) {
                            bilinear_sigma += weight * std::sqrt(yy[0] * yy[0] + xx[1] * xx[1] + + 1.0f);
                            bilinear_depth += weight * val;
                            bilinear_weight += weight;
                        }
                    }
                    {
                        const float v_weight = 1.0f - (v1 - v_f);
                        const float u_weight = 1.0f - (u_f - u0);
                        const float weight = v_weight * u_weight;

                        const float val = depth.at<float>(v1, u0);
                        if (val > 0.0f) {
                            bilinear_sigma += weight * std::sqrt(yy[1] * yy[1] + xx[0] * xx[0] + 1.0f);
                            bilinear_depth += weight * val;
                            bilinear_weight += weight;
                        }
                    }
                    {
                        const float v_weight = 1.0f - (v1 - v_f);
                        const float u_weight = 1.0f - (u1 - u_f);
                        const float weight = v_weight * u_weight;

                        const float val = depth.at<float>(v1, u1);
                        if (val > 0.0f) {
                            bilinear_sigma += weight * std::sqrt(yy[1] * yy[1] + xx[1] * xx[1] + 1.0f);
                            bilinear_depth += weight * val;
                            bilinear_weight += weight;
                        }
                    }
                    if (bilinear_weight <= 0.0f) continue;
                    bilinear_sigma /= bilinear_weight;
                    bilinear_depth /= bilinear_weight;

                    if (bilinear_depth > min_depth_ && bilinear_depth < max_depth_) {
                        float sdf = (bilinear_depth - voxel_pt_camera(2)) * bilinear_sigma;

                        if (sdf > -sdf_trunc_) {
                            /// integrate
                            float tsdf = std::min(1.0f, sdf * sdf_trunc_inv);
                            float weight = use_sdf_weight ? 
                                            std::max(0.1f, 1.0f - std::fabs(tsdf)) : 
                                            1.0f;
                            weight *= base_weight;
                            if (de_integrate) {
                                weight *= -1.0f;
                            }
                            
                            const float old_weight = *p_weight;
                            const float inv_weight = 1.0f / (old_weight + weight);
                            *p_weight = old_weight + weight;

                            const float old_tsdf = *p_tsdf;
                            *p_tsdf = (old_tsdf * old_weight + tsdf * weight) * inv_weight;

                            const cv::Vec3b &rgb = color.at<cv::Vec3b>(rgb_v, rgb_u);
                            Eigen::Vector3f rgb_old = *p_color;
                            float b = rgb_old[0];
                            float g = rgb_old[1];
                            float r = rgb_old[2];
                            b = (b * (old_weight) + rgb[0] * weight) * inv_weight;
                            g = (g * (old_weight) + rgb[1] * weight) * inv_weight;
                            r = (r * (old_weight) + rgb[2] * weight) * inv_weight;
                            b = std::min(b, 255.0f);
                            g = std::min(g, 255.0f);
                            r = std::min(r, 255.0f);
                            b = std::max(b, 0.0f);
                            g = std::max(g, 0.0f);
                            r = std::max(r, 0.0f);
                            *p_color = Eigen::Vector3f(r, g, b);

                            if (de_integrate && *p_weight < std::numeric_limits<float>::epsilon()) {
                                *p_tsdf = TSDF_DEFAULT_TSDF;
                                *p_weight = TSDF_DEFAULT_WEIGHT;
                                *p_color = TSDF_DEFAULT_COLOR;
                            }
                        }
                    }
                }
            }
        }
    }
}

}

}