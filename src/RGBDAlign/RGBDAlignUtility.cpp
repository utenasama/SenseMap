#include <fstream>
#include "RGBDAlignUtility.h"
#include "RGBDPyramid.h"

bool RGBDAlignUtility::ComputeJacAndResRight(
    const Eigen::Matrix4d &src_pose, const Eigen::Matrix4d &dst_pose, 
    const Eigen::Matrix4d &delta_pose, const Eigen::Matrix6d &information, 
    Eigen::Vector6d &out_residual, 
    Eigen::Matrix<double, 6, 6, Eigen::RowMajor> &out_jac_src,
    Eigen::Matrix<double, 6, 6, Eigen::RowMajor> &out_jac_dst)
{
    // Todo
    return false;
}

bool RGBDAlignUtility::ComputeJacAndResLeft(
    const Eigen::Matrix4d &src_pose, const Eigen::Matrix4d &dst_pose, 
    const Eigen::Matrix4d &delta_pose, const Eigen::Matrix6d &information, 
    Eigen::Vector6d &out_residual, 
    Eigen::Matrix<double, 6, 6, Eigen::RowMajor> &out_jac_src,
    Eigen::Matrix<double, 6, 6, Eigen::RowMajor> &out_jac_dst)
{

    auto sqrt_info = Eigen::LLT<Eigen::Matrix<double, 6, 6>>(information).matrixL().transpose();

    Eigen::Quaterniond obs_Q(delta_pose.block<3,3>(0,0));
    Eigen::Vector3d obs_T(delta_pose.block<3,1>(0,3));

    Eigen::Matrix4d x_inv = delta_pose.inverse();

    Eigen::Matrix4d dst_inv = dst_pose.inverse();
    Eigen::Vector3d dst_t = dst_pose.block<3, 1>(0, 3);
    Eigen::Matrix4d x_inv_mul_dst_inv = x_inv * dst_inv;
    Eigen::Quaterniond x_inv_mul_dst_inv_q(x_inv_mul_dst_inv.block<3, 3>(0, 0));

    Eigen::Vector3d src_t = src_pose.block<3, 1>(0, 3);
    Eigen::Quaterniond src_q(src_pose.block<3, 3>(0, 0));
    
    Eigen::Matrix4d zeta_pose = x_inv * dst_inv * src_pose;
    Eigen::Quaterniond zeta_Q(zeta_pose.block<3,3>(0,0));
    Eigen::Vector3d zeta_T = zeta_pose.block<3,1>(0,3);

    out_residual << zeta_T[0], zeta_T[1], zeta_T[2], 2.0 * zeta_Q.x(), 2.0 * zeta_Q.y(), 2.0 * zeta_Q.z();
    out_residual.applyOnTheLeft(sqrt_info);

    // src jac

    out_jac_src.block<3,3>(0,0) = x_inv_mul_dst_inv.block<3, 3>(0, 0);  /// zeta_t / src_t 
    out_jac_src.block<3,3>(0,3).setZero();  /// zeta_t / src_q

    out_jac_src.block<3,3>(3,0).setZero();     /// zeta_q / src_t
    out_jac_src.block<3, 3>(3, 3) = 2.0 * (Qleft(x_inv_mul_dst_inv_q) * Qright(src_q)).bottomRightCorner<3, 3>(); /// zeta_q/src_q

    // dst jac
    out_jac_dst.block<3,3>(0,0) = -x_inv_mul_dst_inv.block<3, 3>(0, 0);/// zeta_t / dst_t

    out_jac_dst.block<3, 3>(0, 3) = -x_inv_mul_dst_inv.block<3, 3>(0, 0) * 
                                skewSymmetric(dst_t - src_t); /// zeta_t / dst_q

    out_jac_dst.block<3,3>(3,0).setZero(); /// zeta_q / dst_t
    out_jac_dst.block<3,3>(3,3) = -out_jac_src.block<3, 3>(3, 3); /// zeta_q / dst_q

    out_jac_src.applyOnTheLeft(sqrt_info);
    out_jac_dst.applyOnTheLeft(sqrt_info);
    // out_jac_dst.setZero();

    return true;
}


void SaveDepthAsObj(std::string fileName, cv::Mat depth, cv::Mat color, Eigen::Matrix3f K, Eigen::Matrix4f trans){
    float TOF_DEPTH_FACTOR = 0.001;
    if (depth.type() == CV_16UC1) {
        depth.convertTo(depth, CV_32FC1, TOF_DEPTH_FACTOR);
        float max_depth = 10;
        float min_depth = 0.1;
        for (int r = 0; r < depth.rows; r+=3) {
            for (int c = 0; c < depth.cols; c+=3) {
                if (depth.at<float>(r, c) < min_depth ||
                    depth.at<float>(r, c) > max_depth) {
                    depth.at<float>(r, c) = std::numeric_limits<float>::quiet_NaN();
                }
            }
        }
    }


    std::ofstream objfile(fileName);
    for(int i=0;i<depth.rows;i+=3){
        for(int j=0;j<depth.cols;j+=3){
            float value = depth.at<float>(i,j);
            if(std::isnan(value)) continue;
            if(value<FLT_EPSILON) continue;
            float x,y;
            x = value * (j - K(0,2))/K(0,0);
            y = value * (i - K(1,2))/K(1,1);

            Eigen::Vector4f pt(x,y,value,1);
            Eigen::Vector4f trans_pt = trans*pt;
            cv::Vec3b c;
            if(color.channels()==3) c = color.at<cv::Vec3b>(i,j);
            else c[0]=c[1]=c[2]=  color.at<uchar>(i,j);
            objfile<<"v "<<trans_pt.head<3>().transpose()<<" ";
            objfile<<(int)c[0]<<" "<<(int)c[1]<<" "<<(int)c[2];
            objfile<<std::endl;
        }
    }
}




///////////////////////////////////////  ICP_Check  ///////////////////////////////////////////////////////

ICP_Check::ICP_Check(){}
ICP_Check::ICP_Check(float d_th, float c_th, bool color, bool bright, float bright_bias):
        depth_th_(d_th), color_th_(c_th), enable_color_(color), enable_bright_(bright), bright_bias_(bright_bias){
}

float ICP_Check::CheckAlignResult(RGBDPyramid::Ptr src, RGBDPyramid::Ptr dst, Eigen::Matrix4d delta_pose){

    if(!enable_bright_) bright_bias_ = 0;

    auto &dst_depth = dst->pyramid_depth_[0];
    auto &src_conf = src->pyramid_conf_[0];
    auto &dst_conf = dst->pyramid_conf_[0];
    auto &src_gray = src->pyramid_gray_[0];
    auto &dst_gray = dst->pyramid_gray_[0];

    const auto &src_highlight = src->pyramid_highlight_[0];
    const auto &dst_highlight = dst->pyramid_highlight_[0];


    Eigen::Matrix3f K = Eigen::Matrix3f::Identity();
    K(0,0) = dst->pyramid_K_[0][0];
    K(1,1) = dst->pyramid_K_[0][1];
    K(0,2) = dst->pyramid_K_[0][2];
    K(1,2) = dst->pyramid_K_[0][3];


    const Eigen::Matrix3f R = delta_pose.block<3,3>(0,0).cast<float>();
    const Eigen::Vector3f T = delta_pose.block<3,1>(0,3).cast<float>();

    int height = dst_depth.rows;
    int width = dst_depth.cols;
    const auto &src_xyz = *(src->GetXYZ(0));

    show_result_ = cv::Mat::zeros(height, width, CV_8UC3);
    diff_result_ = cv::Mat::zeros(height, width, CV_32FC1);

    int valid_num = 0;
    int inlier_num = 0;
    int total_num = 0;
    int vertical_num = 0;
    int depth_outlier = 0, color_outlier = 0;

    for (int y = 0; y < height; y += src_xyz.sample_step_) {
        for (int x = 0; x < width; x += src_xyz.sample_step_) {
            const int index = y * width + x;
            if (std::isnan(src_xyz[index][2])) continue;
            if (std::isnan(src_conf.at<float>(y,x) )) continue;
            total_num++;

            Eigen::Vector3f pt = src_xyz[index];
            Eigen::Vector3f trans_pt = R * pt + T;
            Eigen::Vector3f proj_pt = K * trans_pt;
            double inv_proj_z = 1.0 / trans_pt[2];
            int u_t = (int) (proj_pt[0] / proj_pt[2] + 0.5);
            int v_t = (int) (proj_pt[1] / proj_pt[2] + 0.5);

            if (u_t < 0 || u_t > width - 1) continue;
            if (v_t < 0 || v_t > height - 1) continue;
            if (std::isnan(dst_conf.at<float>(v_t,u_t) )) continue;

            float d_t = dst_depth.at<float>(v_t, u_t);
            if (std::isnan(d_t) || d_t < FLT_EPSILON) continue;

            float conf = std::min(src_conf.at<float>(y,x), dst_conf.at<float>(v_t,u_t));
            if(conf<0.3) conf = 0.3;

            valid_num++;
            show_result_.at<cv::Vec3b>(v_t, u_t) = cv::Vec3b(255,0,0);
            float scale = 1.0/conf;
//            if(src_conf.at<float>(y,x)<0.5) scale*=2;

            diff_result_.at<float>(v_t, u_t) = std::fabs(d_t - proj_pt[2]);

            if (std::fabs(d_t - proj_pt[2]) > scale * depth_th_) {
                ++depth_outlier;
                continue;
            }
            if (enable_color_ &&
                std::fabs(src_gray.at<float>(y, x) - dst_gray.at<float>(v_t, u_t) - bright_bias_) >
                color_th_ / 255.0f
                && (src_highlight.at<uchar>(y, x) == 0 && dst_highlight.at<uchar>(v_t, u_t) == 0)
                    ) {
                ++color_outlier;
                show_result_.at<cv::Vec3b>(v_t, u_t) = cv::Vec3b(0, 255, 255);
                continue;
            }

            show_result_.at<cv::Vec3b>(v_t, u_t) = cv::Vec3b(255,255,0);
            if(src_highlight.at<uchar>(y,x)!=0 || dst_highlight.at<uchar>(v_t, u_t)!=0){
                show_result_.at<cv::Vec3b>(v_t, u_t) = cv::Vec3b(255,0,255);
            }
            inlier_num++;
        }
    }

    float inlier_ratio = inlier_num/(float)valid_num;
    printf("inlier_ratio: %lf  %d  %d\n", inlier_ratio, inlier_num, valid_num);
    return inlier_ratio;
}

float ICP_Check::CheckAlignResult(XYZImage::Ptr src, XYZImage::Ptr dst, Eigen::Matrix4d delta_pose){

    Eigen::Matrix3f K = Eigen::Matrix3f::Identity();
    K<<dst->K_(0), 0, dst->K_(2), 0, dst->K_(1), dst->K_(3), 0, 0, 1;

    const Eigen::Matrix3f R = delta_pose.block<3,3>(0,0).cast<float>();
    const Eigen::Vector3f T = delta_pose.block<3,1>(0,3).cast<float>();

    const auto &src_xyz = *(src);

    int step = src->sample_step_;

    show_result_ = cv::Mat::zeros(dst->height_, dst->width_, CV_8UC3);
    diff_result_ = cv::Mat::zeros(dst->height_, dst->width_, CV_32FC1);

    int valid_num = 0;
    int inlier_num = 0;
    int total_num = 0;
    int vertical_num = 0;
    int depth_outlier = 0, color_outlier = 0;

    for (int y = 0; y < src->height_; y += step) {
        for (int x = 0; x < src->width_; x += step) {
            if (std::isnan(src_xyz(x,y)[2] || src_xyz(x,y)[2]<FLT_EPSILON)) continue;
            total_num++;
            Eigen::Vector3f trans_pt = R * src_xyz(x,y) + T;
            Eigen::Vector3f proj_pt = K * trans_pt;
            double inv_proj_z = 1.0 / trans_pt[2];
            int u_t = (int) (proj_pt[0] / proj_pt[2] + 0.5);
            int v_t = (int) (proj_pt[1] / proj_pt[2] + 0.5);

            if (u_t < 0 || u_t > dst->width_ - 1) continue;
            if (v_t < 0 || v_t > dst->height_ - 1) continue;

            Eigen::Vector3f &d_pt = (*dst)(u_t, v_t);
            if (std::isnan(d_pt[2]) || d_pt[2]<FLT_EPSILON) continue;


            valid_num++;
            show_result_.at<cv::Vec3b>(v_t, u_t) = cv::Vec3b(255,0,0);
            float dist = (d_pt - trans_pt).norm();
            diff_result_.at<float>(v_t, u_t) = dist;


//            printf("diff_result_: %d  %d  %lf\n", x, y, dist);


            if (dist > 1.0 * depth_th_) {
                ++depth_outlier;
                continue;
            }

            show_result_.at<cv::Vec3b>(v_t, u_t) = cv::Vec3b(255,255,0);
            inlier_num++;
        }
    }

    float inlier_ratio = inlier_num/(float)valid_num;
    printf("inlier_ratio: %lf  %d  %d\n", inlier_ratio, inlier_num, valid_num);
    return inlier_ratio;
}


///////////////////////////////////////  ICP_Info  ///////////////////////////////////////////////////////

ICP_Info::ICP_Info(){}
ICP_Info::ICP_Info(float d_th){
    depth_th_ = d_th;
}
Eigen::Matrix6d ICP_Info::GetInfo(){
    return info_;
}

int ICP_Info::ComputeInfo(std::shared_ptr<XYZImage> src, std::shared_ptr<XYZImage> dst, Eigen::Matrix4d delta_pose) {

    Eigen::Matrix3f R = delta_pose.block<3,3>(0,0).cast<float>();
    Eigen::Vector3f T = delta_pose.block<3,1>(0,3).cast<float>();

    Eigen::Matrix<float, 3, 6> G;
    G.block<3,3>(0,3) = Eigen::Matrix3f::Identity();

    info_.setZero();
    Eigen::Matrix<float, 6, 6> float_info_; float_info_.setZero();

    Eigen::Matrix3f dst_K;
    dst_K<<dst->K_(0), 0, dst->K_(2), 0, dst->K_(1), dst->K_(3), 0, 0, 1;
    int cnt = 0;

    int step = src->sample_step_;

    for (int r = 0; r < src->height_; r+=step) {
        for (int c = 0; c < src->width_; c+=step) {
            if (std::isnan((*src)(c,r)[2]) || (*src)(c,r)[2]<FLT_EPSILON) continue;
            Eigen::Vector3f pc = R * (*src)(c,r) + T;
            Eigen::Vector3f proj_pt = dst_K * pc;

            int x = proj_pt[0] / proj_pt[2];
            int y = proj_pt[1] / proj_pt[2];

            if (x < 0 || x >=  dst->width_) continue;
            if (y < 0 || y >=  dst->height_) continue;

            const int umin = std::max(x - search_radius_, 0);
            const int vmin = std::max(y - search_radius_, 0);
            const int umax = std::min(x + search_radius_,  dst->width_ - 1);
            const int vmax = std::min(y + search_radius_, dst->height_ - 1);

            bool found = false;
            for (int v1 = vmin; v1 <= vmax; ++v1) {
                for (int u1 = umin; u1 <= umax; ++u1) {
                    auto &dst_pt = (*dst)(u1,v1);
                    if (std::isnan(dst_pt[2]) || dst_pt[2]<FLT_EPSILON) continue;
                    float dist = (pc -dst_pt).norm();

                    if (dist < depth_th_) {
                        G.block<3, 3>(0, 0) = -skewSymmetric((*src)(c,r));
                        float_info_ += G.transpose() * G;
                        found = true;
                        cnt++;
                        break;
                    }
                }
                if (found) break;
            }
        }
    }
    info_ = float_info_.cast<double>();
//    info_ *= 100;
    return cnt;


}

int ICP_Info::ComputeInfo(std::shared_ptr<RGBDPyramid> src, std::shared_ptr<RGBDPyramid> dst, Eigen::Matrix4d delta_pose){

    auto src_depth = src->pyramid_depth_[0];
    auto dst_depth = dst->pyramid_depth_[0];

    Eigen::Vector3d init_pt(0, 0, std::numeric_limits<float>::quiet_NaN());
    auto &src_xyz = *(src->GetXYZ(0));
    int step = src_xyz.sample_step_;

    const float fx = dst->pyramid_K_[0][0];
    const float fy = dst->pyramid_K_[0][1];
    const float cx = dst->pyramid_K_[0][2];
    const float cy = dst->pyramid_K_[0][3];

    Eigen::Matrix3f K;     K<<fx, 0, cx, 0, fy, cy, 0, 0, 1;

    if(pt_scale_mat_.empty()){
        pt_scale_mat_ = cv::Mat::zeros(src_depth.size(),CV_32FC1);
        for(int r = 0; r<pt_scale_mat_.rows; r++){
            for(int c=0; c<pt_scale_mat_.cols; c++){
                Eigen::Vector3f s((c-cx)/fx, (r-cy)/fy, 1);
                pt_scale_mat_.at<float>(r,c) = s.norm();
            }
        }
    }

    Eigen::Matrix3f R = delta_pose.block<3,3>(0,0).cast<float>();
    Eigen::Vector3f T = delta_pose.block<3,1>(0,3).cast<float>();

    Eigen::Matrix<float, 3, 6> G;
    G.block<3,3>(0,3) = Eigen::Matrix3f::Identity();

    info_.setZero();
    Eigen::Matrix<float, 6, 6> float_info_; float_info_.setZero();

    int cnt = 0;

    for (int r = 0; r < src_depth.rows; r+=step) {
        for (int c = 0; c < src_depth.cols; c+=step) {
            if (std::isnan(src_xyz(c,r)[2]) || src_xyz(c,r)[2]<FLT_EPSILON) continue;
            Eigen::Vector3f pc = R * src_xyz(c,r) + T;
            Eigen::Vector3f proj_pt = K * pc;

            int x = proj_pt[0] / proj_pt[2];
            int y = proj_pt[1] / proj_pt[2];

            if (x < 0 || x >=  dst_depth.cols) continue;
            if (y < 0 || y >=  dst_depth.rows) continue;

            const int umin = std::max(x - search_radius_, 0);
            const int vmin = std::max(y - search_radius_, 0);
            const int umax = std::min(x + search_radius_,  dst_depth.cols - 1);
            const int vmax = std::min(y + search_radius_, dst_depth.rows - 1);

            bool found = false;
            for (int v1 = vmin; v1 <= vmax; ++v1) {
                for (int u1 = umin; u1 <= umax; ++u1) {
                    auto &dst_d = dst_depth.at<float>(v1,u1);
                    if (std::isnan(dst_d)|| dst_d<FLT_EPSILON) continue;
                    float dist = std::fabs(dst_d - pc[2]) * pt_scale_mat_.at<float>(v1, u1);

                    if (dist < depth_th_) {
                        G.block<3, 3>(0, 0) = -skewSymmetric(src_xyz(c,r));
                        float_info_ += G.transpose() * G;
                        found = true;
                        cnt++;
                        break;
                    }
                }
                if (found) break;
            }
        }
    }
    info_ = float_info_.cast<double>();
//    info_ *= 100;
    return cnt;

}

int  ICP_Info::ComputeInfo2(std::shared_ptr<RGBDPyramid> src, std::shared_ptr<RGBDPyramid> dst, Eigen::Matrix4d delta_pose){
    auto depth0 = src->pyramid_depth_[0];
    auto depth1 = dst->pyramid_depth_[0];

    Eigen::Vector3d init_pt(0, 0, std::numeric_limits<float>::quiet_NaN());

    std::vector<std::vector<Eigen::Vector3d>> pt_mat0(depth0.rows, std::vector<Eigen::Vector3d>(depth0.cols, init_pt));
    std::vector<std::vector<Eigen::Vector3d>> pt_mat1(depth1.rows, std::vector<Eigen::Vector3d>(depth1.cols, init_pt));

    const float fx = dst->pyramid_K_[0][0];
    const float fy = dst->pyramid_K_[0][1];
    const float cx = dst->pyramid_K_[0][2];
    const float cy = dst->pyramid_K_[0][3];

    Eigen::Matrix3d K;     K<<fx, 0, cx, 0, fy, cy, 0, 0, 1;

    Eigen::Matrix3d R = delta_pose.block<3,3>(0,0);
    Eigen::Vector3d T = delta_pose.block<3,1>(0,3);

    Eigen::Matrix<double, 3, 6> G;
    G.block<3,3>(0,3) = Eigen::Matrix3d::Identity();

    for(int r = 0; r<depth0.rows; r++){
        for(int c = 0; c<depth0.cols; c++){
            auto d = depth0.at<float>(r,c);
            if(std::isnan(d) || d<FLT_EPSILON) continue;
            pt_mat0[r][c][2] = d;
            pt_mat0[r][c][0] =  d * (c-cx)/fx;
            pt_mat0[r][c][1] =  d * (r-cy)/fy;
        }
    }

    for(int r = 0; r<depth1.rows; r++){
        for(int c = 0; c<depth1.cols; c++){
            auto d = depth1.at<float>(r,c);
            if(std::isnan(d) || d<FLT_EPSILON) continue;
            pt_mat1[r][c][2] = d;
            pt_mat1[r][c][0] =  d * (c-cx)/fx;
            pt_mat1[r][c][1] =  d * (r-cy)/fy;
        }
    }

    info_.setZero();

    int cnt = 0;

    for (int r = 0; r < depth0.rows; r+=3) {
        for (int c = 0; c < depth0.cols; c+=3) {
            if (std::isnan(pt_mat0[r][c][2]) || pt_mat0[r][c][2]<FLT_EPSILON) continue;
            Eigen::Vector3d pc = R * pt_mat0[r][c] + T;
            Eigen::Vector3d proj_pt = K * pc;

            int x = proj_pt[0] / proj_pt[2];
            int y = proj_pt[1] / proj_pt[2];

            if (x < 0 || x >=  depth1.cols) continue;
            if (y < 0 || y >=  depth1.rows) continue;

            const int umin = std::max(x - search_radius_, 0);
            const int vmin = std::max(y - search_radius_, 0);
            const int umax = std::min(x + search_radius_,  depth1.cols - 1);
            const int vmax = std::min(y + search_radius_, depth1.rows - 1);

            bool found = false;
            for (int v1 = vmin; v1 <= vmax; ++v1) {
                for (int u1 = umin; u1 <= umax; ++u1) {
                    if (std::isnan(pt_mat1[v1][u1][2]) || pt_mat1[v1][u1][2]<FLT_EPSILON) continue;
                    float dist = (pt_mat1[v1][u1] - pc).norm();

                    if (dist < depth_th_) {
                        G.block<3, 3>(0, 0) = -skewSymmetric(pt_mat0[r][c]);
                        info_ += G.transpose() * G;
                        found = true;
                        cnt++;
                        break;
                    }
                }
                if (found) break;
            }
        }
    }

    std::cout<<"ComputeInfo2: "<<std::endl;
    std::cout<<info_<<std::endl;

//    info_ *= 100;
    return cnt;

}
