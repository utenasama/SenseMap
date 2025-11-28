//Copyright (c) 2019, SenseTime Group.
//All rights reserved.
#ifndef SENSEMAP_ESTIMATORS_MOTION_AVERAGE_ROTATION_AVERAGE_H_
#define SENSEMAP_ESTIMATORS_MOTION_AVERAGE_ROTATION_AVERAGE_H_

#include <vector>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include "util/alignment.h"
namespace sensemap{

struct RotationAveragerOptions {
    //stop threshold of geodesic mean iteration
    double stop_threshold_geodesic_mean=1.0/180.0*M_PI;
    int max_iteration_geodesci_mean=10;
};


//This class computes the average of a group of rotations. Currently, only the
//Geodesic mean is implemented based on this paper:
//    Venu Madhav Govindu, Lie-Algebraic Averaging For Globally Consistent
//     Motion Estimation, CVPR 2004 
//
class RotationAverager{
public:
    RotationAverager(RotationAveragerOptions& options);
    Eigen::Matrix3d GeodesicMeanL2(
                const std::vector<Eigen::Matrix3d>& rotations);
private:
    RotationAveragerOptions options_;

};

//The estimator of rotation average used in RANSAC 
class RotationAverageEstimator{
public:
	static const int kMinNumSamples = 1;

  	typedef Eigen::Matrix3d X_t;
  	typedef Eigen::Matrix3d Y_t;
  	typedef Eigen::Matrix3d M_t; 

	std::vector<M_t> Estimate(const std::vector<X_t>& src,
							  const std::vector<Y_t>& dst);

	void Residuals(const std::vector<X_t>& src,
				   const std::vector<Y_t>& dst, const M_t& absolute_rotation,
				   std::vector<double>* residuals); 

};

}//namespace sensemap

#endif //SENSEMAP_ESTIMATORS_MOTION_AVERAGE_ROTATION_AVERAGE_H_