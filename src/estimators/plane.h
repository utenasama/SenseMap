//Copyright (c) 2019, SenseTime Group.
//All rights reserved.
#ifndef SENSEMAP_ESTIMATORS_PLANE_H_
#define SENSEMAP_ESTIMATORS_PLANE_H_

#include <vector>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include "util/alignment.h"

namespace sensemap{

//The estimator of the plane from a set of 3D points. 

class PlaneEstimator{
public:
	static const int kMinNumSamples = 3;

  	typedef Eigen::Vector3d X_t;
  	typedef Eigen::Vector3d Y_t;
  	typedef Eigen::Vector4d M_t; 

	
	std::vector<M_t> Estimate(const std::vector<X_t>& src,
							  const std::vector<Y_t>& dst);

	void Residuals(const std::vector<X_t>& src,
				   const std::vector<Y_t>& dst, const M_t& plane,
				   std::vector<double>* residuals); 
	
};

}




















#endif //SENSEMAP_ESTIMATORS_PLANE_H_