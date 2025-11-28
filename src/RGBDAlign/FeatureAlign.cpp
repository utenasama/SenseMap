//
// Created by sensetime on 2020/11/25.
//

#include "FeatureAlign.h"

FeatureAlign::FeatureAlign(){}

FeatureAlign::FeatureAlign(float max_dist):max_dist_(max_dist){}

void FeatureAlign::SetMaxDist(float v){
    max_dist_ = v;
}


void FeatureAlign::SetInput(std::vector<Eigen::Vector3d> src_feas, Eigen::Matrix3d src_K,
              std::vector<Eigen::Vector3d> dst_feas, Eigen::Matrix3d dst_K) {

    src_feas_ = src_feas;
    dst_feas_ = dst_feas;

    src_K_ = src_K;
    dst_K_ = dst_K;

    src_pts_.resize(src_feas.size());
    dst_pts_.resize(dst_feas.size());

    for(int i=0; i<src_feas.size(); i++){
        auto &fea = src_feas_[i];
        src_pts_[i] = fea[2] *
                Eigen::Vector3d((fea[0] - src_K(0, 2)) / src_K(0, 0), (fea[1] - src_K(1, 2)) / src_K(1, 1), 1.0);
    }
    auto &K = dst_K;
    for(int i=0; i<dst_feas.size(); i++){
        auto &fea = dst_feas_[i];
        dst_pts_[i] = fea[2] *
                Eigen::Vector3d((fea[0] - dst_K(0, 2)) / dst_K(0, 0), (fea[1] - dst_K(1, 2)) / dst_K(1, 1), 1.0);
    }

}


void FeatureAlign::ComputeJacobian(Eigen::Matrix4d init, int &corres_num, float &residual,
                     Eigen::Matrix6d &JtJ, Eigen::Vector6d &Jtr){

    JtJ.setZero();
    Jtr.setZero();
    corres_num = 0;
    residual = 0;
    double max_dist2 = max_dist_ * max_dist_;
    for(int i = 0; i<src_pts_.size(); i++){
        Eigen::Vector3d r_pt = init.block<3,3>(0,0) * src_pts_[i];
        Eigen::Vector3d re = r_pt + init.block<3,1>(0,3)- dst_pts_[i];

        double re2 = re.squaredNorm();
        if(re2 > max_dist2) continue;

        Eigen::Matrix<double, 3, 6> J;
        J.block<3,3>(0,0) = -skewSymmetric(r_pt);
        J.block<3,3>(0,3).setIdentity();
        JtJ += J.transpose() * J;
        Jtr += J.transpose() * re;
        residual += re2;
        corres_num++;
    }
}
