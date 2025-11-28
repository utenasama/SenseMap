//Copyright (c) 2019, SenseTime Group.
//All rights reserved.

#include "camera_alignment.h"
#include "base/similarity_transform.h"
#include "base/projection.h"

namespace sensemap{

std::vector<typename CameraAlignmentEstimator::M_t> 
		CameraAlignmentEstimator::Estimate(const std::vector<X_t>& src,
						  				   const std::vector<Y_t>& dst){	
	CHECK(src.size()==dst.size());
	CHECK_GE(src.size(), 3);
    CHECK_GE(dst.size(), 3);	

	SimilarityTransform3 tform12;
	tform12.Estimate(src,dst);

	return {tform12.Matrix().topRows<3>()};
}

void CameraAlignmentEstimator::Residuals(
				 const std::vector<X_t>& points1,
                 const std::vector<Y_t>& points2, const M_t& alignment12,
                 std::vector<double>* residuals){

	CHECK_EQ(points1.size(), points2.size());


	for(size_t i=0; i<points1.size(); ++i){

        //reproject points1 into reconstruction_dst using alignment12  
		const Eigen::Vector3d xyz12 = alignment12*points1[i].homogeneous();  

        double error = (points2[i] - xyz12).squaredNorm();
		(*residuals)[i]=error;
	}
}

}//namespace senemap

