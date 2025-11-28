//Copyright (c) 2019, SenseTime Group.
//All rights reserved.
#ifndef SENSEMAP_ESTIMATORS_SCALE_SELECTION_H_
#define SENSEMAP_ESTIMATORS_SCALE_SELECTION_H_
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "util/types.h"

namespace sensemap{

//The estimator of the best scale factor amount all the scales
//The residuals are error of the best scale to other scales. 

class ScaleSelectionEstimator{
public:
	static const int kMinNumSamples = 2;

	typedef double X_t;
  	typedef double Y_t;
  	typedef double M_t; 

	std::vector<M_t> Estimate(const std::vector<X_t>& src,
							  const std::vector<Y_t>& dst);

	void Residuals(const std::vector<X_t>& src,
				   const std::vector<Y_t>& dst, const M_t& alignment12,
				   std::vector<double>* residuals); 
};

}
#endif //SENSEMAP_ESTIMATORS_SCALE_SELECTION_H_