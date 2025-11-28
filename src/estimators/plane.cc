//Copyright (c) 2019, SenseTime Group.
//All rights reserved.

#include "plane.h"
#include "util/logging.h"

namespace sensemap{

std::vector<PlaneEstimator::M_t> PlaneEstimator::Estimate(
                                const std::vector<X_t>& src,
							    const std::vector<Y_t>& dst){
    
    CHECK(src.size()==dst.size());
	CHECK_GE(src.size(), 3);
    CHECK_GE(dst.size(), 3);
    
    Eigen::Vector3d plane_center(0,0,0);
    Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> point_mat;
    
    point_mat.resize(src.size(),Eigen::NoChange_t::NoChange);

    for(size_t i = 0; i<src.size(); ++i){
        
        plane_center = plane_center + src[i];
        
        point_mat.row(i) = src[i].transpose();
    }
    plane_center  = plane_center/static_cast<double>(src.size());
  
    for(int i = 0 ;i< src.size(); ++i){
        point_mat.row(i) = point_mat.row(i) - plane_center.transpose();
    }

    Eigen::JacobiSVD<Eigen::Matrix<double, Eigen::Dynamic, 3>> svd(
      point_mat, Eigen::ComputeFullV);

    Eigen::Vector3d normal = svd.matrixV().col(2);


    Eigen::Vector4d primary_plane;
    primary_plane(0) = normal(0);
    primary_plane(1) = normal(1);
    primary_plane(2) = normal(2);
    primary_plane(3) = -normal.dot(plane_center);

    return{primary_plane};
}

void PlaneEstimator::Residuals(const std::vector<X_t>& src,
			                   const std::vector<Y_t>& dst, 
                               const M_t& plane,
			                   std::vector<double>* residuals){

    CHECK(src.size()==dst.size());
    
    residuals->resize(src.size());
    
    Eigen::Vector3d normal = plane.block<3,1>(0,0);
    double d = plane(3);

    for(size_t i = 0; i<src.size(); ++i){
        (*residuals)[i] = 
            (normal.dot(src[i])+d)*(normal.dot(src[i])+d)/normal.squaredNorm();
    }  
}


}//namespace senemap