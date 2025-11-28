//Copyright (c) 2021, SenseTime Group.
//All rights reserved.

#ifndef SENSEMAP_ESTIMATORS_SCALE_GRAVITY_H_
#define SENSEMAP_ESTIMATORS_SCALE_GRAVITY_H_


#include <Eigen/Core>
#include "base/reconstruction.h"



namespace sensemap{

class ScaleGravityEstimator{


public:
 ScaleGravityEstimator(const std::shared_ptr<Reconstruction> reconstruction) : reconstruction_(reconstruction) {}

 bool Estimate(
     std::vector<std::unordered_map<image_t, std::pair<Eigen::Vector4d, Eigen::Vector3d>>> pose_group_from_measurement);


 double GetScale() { return scale_; }
 Eigen::Vector3d GetGravity() { return gravity_; }
 Eigen::Matrix<double,3,4> GetTransform() {return transform_;}

private:
    std::shared_ptr<Reconstruction> reconstruction_;
    double scale_;
    Eigen::Vector3d gravity_;
    Eigen::Matrix<double,3,4> transform_;

};



}

#endif
