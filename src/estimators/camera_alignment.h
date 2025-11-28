//Copyright (c) 2019, SenseTime Group.
//All rights reserved.
#ifndef SENSEMAP_ESTIMATORS_CAMERA_ALIGNMENT_H_
#define SENSEMAP_ESTIMATORS_CAMERA_ALIGNMENT_H_
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "util/types.h"
#include "base/reconstruction.h"

namespace sensemap{

//The estimator of the alignment(similarity transform) between two camera 
//sets.  The residuals are the reprojection errors of the aligned points. 

class CameraAlignmentEstimator{
public:
	static const int kMinNumSamples = 6;

  	typedef Eigen::Vector3d X_t;
  	typedef Eigen::Vector3d Y_t;
  	typedef Eigen::Matrix3x4d M_t; 

	std::vector<M_t> Estimate(const std::vector<X_t>& src,
							  const std::vector<Y_t>& dst);

	void Residuals(const std::vector<X_t>& src,
				   const std::vector<Y_t>& dst, const M_t& alignment12,
				   std::vector<double>* residuals); 
};

}
#endif //SENSEMAP_ESTIMATORS_MAPPOINT_ALIGNMENT_H_