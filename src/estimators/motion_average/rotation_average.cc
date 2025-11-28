//Copyright (c) 2019, SenseTime Group.
//All rights reserved.
#include "rotation_average.h"
#include "util/logging.h"

namespace sensemap{

RotationAverager::RotationAverager(RotationAveragerOptions& options)
    :options_(options){
}

Eigen::Matrix3d RotationAverager::GeodesicMeanL2(
                const std::vector<Eigen::Matrix3d>& rotations){
    
    Eigen::Matrix3d R=rotations[0];
    int loop=0;
    while(true){
        Eigen::Vector3d mean_r=Eigen::Vector3d::Zero();
    
        for(size_t i=0;i<rotations.size();i++){
            Eigen::AngleAxisd angle_axis(R.transpose()*rotations[i]);
            Eigen::Vector3d r=angle_axis.axis()*angle_axis.angle();

            mean_r+=r;
        }
        mean_r/=static_cast<double>(rotations.size());

        if(mean_r.norm()<options_.stop_threshold_geodesic_mean){
            return R;
        }

        double angle_inc=mean_r.norm();
        Eigen::Vector3d axis_inc=mean_r.normalized();

        Eigen::AngleAxisd angle_axis_inc(angle_inc,axis_inc);
        Eigen::Matrix3d R_inc=angle_axis_inc.toRotationMatrix();

        R=R*R_inc;
        if(loop>=options_.max_iteration_geodesci_mean){
            return R;
        }
    }
}


std::vector<RotationAverageEstimator::M_t> 
RotationAverageEstimator::Estimate(const std::vector<X_t>& src,
							  	 const std::vector<Y_t>& dst){
	CHECK(src.size()>0);
	std::vector<M_t> model(1);

	if(src.size()==1){
		model[0]=src[0];
	}
	else{
		RotationAveragerOptions options;
		RotationAverager rotation_averager(options);
		model[0]=rotation_averager.GeodesicMeanL2(src);
	}
	return model;
}

void RotationAverageEstimator::Residuals(const std::vector<X_t>& src,
				   					   const std::vector<Y_t>& dst, 
								       const M_t& absolute_rotation,
				   				       std::vector<double>* residuals){

    for(size_t i=0; i<src.size();++i){
		Eigen::AngleAxisd angle_axis(absolute_rotation.transpose()*src[i]);
		(*residuals)[i]=angle_axis.angle()*angle_axis.angle();
	}									   
} 

}//namespace sensemap
