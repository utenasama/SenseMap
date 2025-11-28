//Copyright (c) 2019, SenseTime Group.
//All rights reserved.
#ifndef SENSEMAP_ESTIMATORS_MAPPOINT_ALIGNMENT_H_
#define SENSEMAP_ESTIMATORS_MAPPOINT_ALIGNMENT_H_
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "util/types.h"
#include "base/reconstruction.h"

namespace sensemap{

//The estimator of the alignment(similarity transform) between two mapppoint 
//sets.  The residuals are the reprojection errors of the aligned points. 

class MapPointAlignmentEstimator{
public:
	static const int kMinNumSamples = 3;

  	typedef mappoint_t X_t;
  	typedef mappoint_t Y_t;
  	typedef Eigen::Matrix3x4d M_t; 

	void SetReconstruction(const Reconstruction* reconstruction_src, 
						   const Reconstruction* reconstruction_dst);
	std::vector<M_t> Estimate(const std::vector<X_t>& src,
							  const std::vector<Y_t>& dst);

	void Residuals(const std::vector<X_t>& src,
				   const std::vector<Y_t>& dst, const M_t& alignment12,
				   std::vector<double>* residuals); 

private:

	const Reconstruction* reconstruction_src_=nullptr;
	const Reconstruction* reconstruction_dst_=nullptr;
};


}
#endif //SENSEMAP_ESTIMATORS_MAPPOINT_ALIGNMENT_H_