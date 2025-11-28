//Copyright (c) 2019, SenseTime Group.
//All rights reserved.
#include "translation_average.h"
#include "util/logging.h"

namespace sensemap{

TranslationAverager::TranslationAverager(TranslationAveragerOptions& options)
    :options_(options){}

Eigen::Vector3d TranslationAverager::Triangulate(
                            const std::vector<Eigen::Vector3d>& t_relative,
                            const std::vector<Eigen::Vector3d>& t_anchor){
    
    CHECK(t_relative.size()>=2);
    CHECK(t_relative.size()==t_anchor.size());

    Eigen::Matrix<double,Eigen::Dynamic, Eigen::Dynamic,Eigen::RowMajor> A;
    A.resize(t_relative.size()*2,3);                            
    std::vector<double> lambda(t_relative.size(),1.0);
    Eigen::Matrix<double,Eigen::Dynamic, Eigen::Dynamic,Eigen::RowMajor> b;
    b.resize(t_relative.size()*2,1);

    Eigen::Vector3d t;
    
    //iteratively solve the weighted linear equation
    for(int iter=0; iter<options_.max_iteration_triangulation; ++iter){

        //set up the linear equations
        for(size_t i=0;i<t_relative.size();++i){
            Eigen::Matrix3d skew_t_relative;
            skew_t_relative <<0, t_relative[i](2), -t_relative[i](1), 
                              -t_relative[i](2), 0, t_relative[i](0),  
                              t_relative[i](1), -t_relative[i](0), 0;
            
            A.row(i*2)=skew_t_relative.row(0)*lambda[i];
            A.row(i*2+1)=skew_t_relative.row(1)*lambda[i];

            Eigen::Vector3d trel_x_tanc= skew_t_relative* t_anchor[i];
            b.block<2,1>(2*i,0)=trel_x_tanc.block<2,1>(0,0)*lambda[i];      
        }

        t=A.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b);

        for(size_t i=0; i<t_anchor.size(); ++i){
            if((t-t_anchor[i]).norm()==0){
                lambda[i]=0.0;
            }
            else{
                lambda[i]=1.0/(t-t_anchor[i]).norm(); 
            }
        }
    }

    return t;
}

 Eigen::Vector3d TranslationAverager::Mean(
        const std::vector<Eigen::Vector3d>& translations){
    
    Eigen::Vector3d mean=Eigen::Vector3d::Zero();        
    for(size_t i=0; i<translations.size(); ++i){
        mean+=translations[i];
    }

    mean/=static_cast<double>(translations.size());
    return mean;
 }


std::vector<TranslationTriangulationEstimator::M_t> 
TranslationTriangulationEstimator::Estimate(const std::vector<X_t>& t_relative,
							  	            const std::vector<Y_t>& t_anchor){
    CHECK(t_relative.size()>=2);
    CHECK(t_relative.size()==t_anchor.size());
    TranslationAveragerOptions options;
    TranslationAverager translation_averager(options);

    X_t t_absolute=translation_averager.Triangulate(t_relative, t_anchor);

    std::vector<X_t> model(1);
    model[0]=t_absolute;
    return model;
}

void TranslationTriangulationEstimator::Residuals(
                                          const std::vector<X_t>& t_relative,
				                          const std::vector<Y_t>& t_anchor, 
                                          const M_t& t_absolute,
				                          std::vector<double>* residuals){

    CHECK(t_relative.size()==t_anchor.size());

    for(size_t i=0; i<t_relative.size(); ++i){
        X_t t_relative_est=t_absolute-t_anchor[i];

        if(t_relative[i].norm()==0&&t_relative_est.norm()==0){
            (*residuals)[i]=0.0;
        }
        else if(t_relative_est.norm()==0){
            (*residuals)[i]=1.0;
        }
        else{
            (*residuals)[i]=
              (t_relative[i].cross(t_relative_est.normalized())).squaredNorm();
            
            CHECK_LE((*residuals)[i], 1.0);
        }
    }                                          
}


std::vector<TranslationAverageEstimator::M_t> 
TranslationAverageEstimator::Estimate(const std::vector<X_t>& src,
							  	    const std::vector<Y_t>& dst){
    CHECK(src.size()>=1);
    CHECK(src.size()==dst.size());
    TranslationAveragerOptions options;
    TranslationAverager translation_averager(options);

    X_t t_estimated=translation_averager.Mean(src);

    std::vector<X_t> model(1);
    model[0]=t_estimated;
    return model;
}

void TranslationAverageEstimator::Residuals(const std::vector<X_t>& src,
				   const std::vector<Y_t>& dst, const M_t& t_esimated,
				   std::vector<double>* residuals){

    for(size_t i=0; i<src.size(); ++i){
        if(src[i].squaredNorm()==0&&t_esimated.squaredNorm()==0){
            (*residuals)[i]=0.0;        
        }
        else if(src[i].squaredNorm()==0){
            (*residuals)[i]=1.0;    
        }
        else{
            (*residuals)[i]=
                    (t_esimated-src[i]).squaredNorm()/src[i].squaredNorm(); 
        }    
    }
}


}//namespace sensemap

