//
// Created by sensetime on 2020/11/25.
//

#ifndef SENSEMAP_FEATUREALIGN_H
#define SENSEMAP_FEATUREALIGN_H
#include "RGBDAlignUtility.h"

class FeatureAlign {
public:
    FeatureAlign();
    FeatureAlign(float max_dist);

    void SetInput(std::vector<Eigen::Vector3d> src_feas, Eigen::Matrix3d src_K,
                  std::vector<Eigen::Vector3d> dst_feas, Eigen::Matrix3d dst_K); //x, y, depth


    void ComputeJacobian(Eigen::Matrix4d init, int &corres_num, float &residual,
                         Eigen::Matrix6d &JtJ, Eigen::Vector6d &Jtr);

    void SetMaxDist(float v);

private:

    std::vector<Eigen::Vector3d> src_pts_;
    std::vector<Eigen::Vector3d> dst_pts_;

    std::vector<Eigen::Vector3d> src_feas_;
    std::vector<Eigen::Vector3d> dst_feas_;

    Eigen::Matrix3d src_K_;
    Eigen::Matrix3d dst_K_;

    float max_dist_;

};


#endif //SENSEMAP_FEATUREALIGN_H
