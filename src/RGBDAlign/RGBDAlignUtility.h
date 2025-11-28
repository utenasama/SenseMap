#ifndef SENSESLAM_RGBDALIGN_UTILITY_H
#define SENSESLAM_RGBDALIGN_UTILITY_H

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <memory>
#include "util/types.h"

namespace Eigen {

/// Extending Eigen namespace by adding frequently used matrix type
typedef Eigen::Matrix<double, 6, 6> Matrix6d;
typedef Eigen::Matrix<double, 6, 1> Vector6d;
typedef Eigen::Matrix<double, 7, 7> Matrix7d;
typedef Eigen::Matrix<double, 7, 1> Vector7d;
}

template <typename Derived>
static Eigen::Matrix<typename Derived::Scalar, 3, 3>
skewSymmetric(const Eigen::MatrixBase<Derived> &q) {
    Eigen::Matrix<typename Derived::Scalar, 3, 3> ans;
    ans << typename Derived::Scalar(0), -q(2), q(1),
            q(2), typename Derived::Scalar(0), -q(0),
            -q(1), q(0), typename Derived::Scalar(0);
    return ans;
}


template <typename Derived>
static Eigen::Quaternion<typename Derived::Scalar>
positify(const Eigen::QuaternionBase<Derived> &q) {
    // printf("a: %f %f %f %f", q.w(), q.x(), q.y(), q.z());
    // Eigen::Quaternion<typename Derived::Scalar> p(-q.w(), -q.x(), -q.y(), -q.z());
    // printf("b: %f %f %f %f", p.w(), p.x(), p.y(), p.z());
    // return q.template w() >= (typename Derived::Scalar)(0.0) ?
    // q : Eigen::Quaternion<typename Derived::Scalar>
    // (-q.w(), -q.x(), -q.y(), -q.z());
    return q;
}
template <typename Derived>
static Eigen::Matrix<typename Derived::Scalar, 4, 4>
Qleft(const Eigen::QuaternionBase<Derived> &q) {
    Eigen::Quaternion<typename Derived::Scalar> qq = positify(q);
    Eigen::Matrix<typename Derived::Scalar, 4, 4> ans;
    ans(0, 0) = qq.w(), ans.template block<1, 3>(0, 1) = -qq.vec().transpose();
    ans.template block<3, 1>(1, 0) = qq.vec(), ans.template block<3, 3>(1, 1) =
                                                       qq.w() *
                                                       Eigen::Matrix<typename Derived::Scalar, 3, 3>::Identity() +
                                                       skewSymmetric(qq.vec());
    return ans;
}
template <typename Derived>
static Eigen::Matrix<typename Derived::Scalar, 4, 4>
Qright(const Eigen::QuaternionBase<Derived> &p) {
    Eigen::Quaternion<typename Derived::Scalar> pp = positify(p);
    Eigen::Matrix<typename Derived::Scalar, 4, 4> ans;
    ans(0, 0) = pp.w(), ans.template block<1, 3>(0, 1) = -pp.vec().transpose();
    ans.template block<3, 1>(1, 0) = pp.vec(), ans.template block<3, 3>(1, 1) =
                                                       pp.w() *
                                                       Eigen::Matrix<typename Derived::Scalar, 3, 3>::Identity() -
                                                       skewSymmetric(pp.vec());
    return ans;
}



class RGBDAlignUtility {
public:
    // 右乘
    bool ComputeJacAndResRight(
        const Eigen::Matrix4d &src_pose, const Eigen::Matrix4d &dst_pose, 
        const Eigen::Matrix4d &delta_pose, const Eigen::Matrix6d &information, 
        Eigen::Vector6d &out_residual, 
        Eigen::Matrix<double, 6, 6, Eigen::RowMajor> &out_jac_src,
        Eigen::Matrix<double, 6, 6, Eigen::RowMajor> &out_jac_dst);

    // 左乘
    bool ComputeJacAndResLeft(
        const Eigen::Matrix4d &src_pose, const Eigen::Matrix4d &dst_pose, 
        const Eigen::Matrix4d &delta_pose, const Eigen::Matrix6d &information, 
        Eigen::Vector6d &out_residual, 
        Eigen::Matrix<double, 6, 6, Eigen::RowMajor> &out_jac_src,
        Eigen::Matrix<double, 6, 6, Eigen::RowMajor> &out_jac_dst);

};


template<int s>
class ICP_Solver{
public:
    ICP_Solver(){}
    ICP_Solver(Eigen::Matrix4d init, int corres_num, float residual){
        roll_back_ = false;
        corres_num_ = corres_num;
        residual_ = residual;
        odometry_ = init;
    }

    void UpdateState(Eigen::Matrix4d init, int corres_num, float residual){
        roll_back_ = false;
        if(corres_num_ * 0.7 > corres_num){
            roll_back_ = true;
            return;
        }

//        if(corres_num_ > corres_num && residual_ < residual){
//            roll_back_ = true;
//            return;
//        }

        corres_num_ = corres_num;
        residual_ = residual;
        odometry_ = init;

    }

    bool roll_back_ = false;

    Eigen::Matrix<double, s, 1> x_;

    Eigen::Matrix4d RollBackSolve(){
        x_ *= 0.3;
        Eigen::Matrix4d ret = Eigen::Matrix4d::Identity();
        ret.block<3, 3>(0, 0) =
                (Eigen::AngleAxisd(x_(2), Eigen::Vector3d::UnitZ()) *
                 Eigen::AngleAxisd(x_(1), Eigen::Vector3d::UnitY()) *
                 Eigen::AngleAxisd(x_(0), Eigen::Vector3d::UnitX())).matrix();
        ret.block<3, 1>(0, 3) = x_.block(3, 0, 3, 1);
        return ret * odometry_;
    }

    std::tuple<bool, Eigen::Matrix4d> Solve(Eigen::Matrix<double, s, s> A, Eigen::Matrix<double, s, 1> b){

        Eigen::Matrix4d ret = Eigen::Matrix4d::Identity();
        bool solution_exist = false;
        double det = A.determinant();
        if (fabs(det) < DBL_EPSILON || std::isnan(det) || std::isinf(det)) {
            printf("[SolveLinearSystem] det is %lf\n", det);
            return std::make_tuple(solution_exist, ret);
        }

        solution_exist = true;
        x_ = A.ldlt().solve(b);
        if(s==7){
            if(std::fabs(x_[6]) > 3/255.0f) x_[6] = x_[6] > 0 ? +5/255.0f: -5/255.0f;
        }

        ret.block<3, 3>(0, 0) =
                (Eigen::AngleAxisd(x_(2), Eigen::Vector3d::UnitZ()) *
                 Eigen::AngleAxisd(x_(1), Eigen::Vector3d::UnitY()) *
                 Eigen::AngleAxisd(x_(0), Eigen::Vector3d::UnitX())).matrix();
        ret.block<3, 1>(0, 3) = x_.block(3, 0, 3, 1);

        return std::make_tuple(true,  ret * odometry_);
    }

    Eigen::Matrix4d odometry_;

    int corres_num_=0;
    float residual_=0;
};


void SaveDepthAsObj(std::string fileName, cv::Mat depth, cv::Mat color, Eigen::Matrix3f K, Eigen::Matrix4f trans);


class RGBDPyramid;
class XYZImage;

class ICP_Check{
public:
    ICP_Check();
    ICP_Check(float d_th, float c_th, bool color, bool bright = false, float bright_bias = 0.0);
    float CheckAlignResult(std::shared_ptr<RGBDPyramid> src, std::shared_ptr<RGBDPyramid> dst, Eigen::Matrix4d delta_pose);
    float CheckAlignResult(std::shared_ptr<XYZImage> src, std::shared_ptr<XYZImage> dst, Eigen::Matrix4d delta_pose);

    cv::Mat show_result_;
    cv::Mat diff_result_;
    float depth_th_;
    float color_th_;
    bool enable_color_;
    bool enable_bright_;
    double bright_gain_ = 1.0;
    double bright_bias_ = 0.0;

};


class ICP_Info{
public:
    ICP_Info();
    ICP_Info(float d_th);

    int ComputeInfo(std::shared_ptr<RGBDPyramid> src, std::shared_ptr<RGBDPyramid> dst, Eigen::Matrix4d delta_pose);

    int ComputeInfo(std::shared_ptr<XYZImage> src, std::shared_ptr<XYZImage> dst, Eigen::Matrix4d delta_pose);

    int ComputeInfo2(std::shared_ptr<RGBDPyramid> src, std::shared_ptr<RGBDPyramid> dst, Eigen::Matrix4d delta_pose);

    Eigen::Matrix6d GetInfo();
    cv::Mat pt_scale_mat_ = cv::Mat();
    Eigen::Matrix6d info_;
    float depth_th_;
    int search_radius_ = 3;
    int step_ = 3;

};

#endif