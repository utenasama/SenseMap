//Copyright (c) 2019, SenseTime Group.
//All rights reserved.
#include "similarity_average.h"
#include "util/logging.h"
#include "optim/ransac/loransac.h"
#include "rotation_average.h"
#include "translation_average.h"

namespace sensemap{

size_t SimilarityAverageEstimator::SimilarityFromCrossMotion(
            std::vector<std::pair<Eigen::Matrix3d, Eigen::Matrix3d>> r_A_B,
            std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> t_A_B,
            Eigen::Matrix3x4d& transform){

    CHECK_GE(r_A_B.size(),3);
    CHECK(r_A_B.size()==t_A_B.size());

    //esimate the scale first
    double estimated_scale;
    std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> centers_A_B;

    for(size_t i=0; i< r_A_B.size(); ++i){
        Eigen::Vector3d center_A=-r_A_B[i].first.transpose()*t_A_B[i].first;
        Eigen::Vector3d center_B=-r_A_B[i].second.transpose()*t_A_B[i].second;
        centers_A_B.emplace_back(center_A,center_B);
    }
    std::vector<double> scales;
    size_t pair_count=0;
    for(size_t i=0; i< centers_A_B.size(); ++i){
        for(size_t j=i+1; j<centers_A_B.size(); ++j){
            Eigen::Vector3d diff_B=centers_A_B[j].second-centers_A_B[i].second;
            Eigen::Vector3d diff_A=centers_A_B[j].first-centers_A_B[i].first;
            if(diff_A.norm()!=0){
                scales.push_back(diff_B.norm()/diff_A.norm());       
                pair_count++;
            }
        }
    }
    if(pair_count<3){
        return 0;
    }
    std::sort(scales.begin(),scales.end());
    estimated_scale=scales[pair_count/2];


    //then the rotation
    std::vector<Eigen::Matrix3d> rotations;
    for(size_t i=0; i< r_A_B.size(); ++i){
        rotations.push_back(r_A_B[i].second.transpose()*r_A_B[i].first);
    }

    RANSACOptions ransac_options_rotation;
    ransac_options_rotation.max_error=5.0/180*M_PI;
    ransac_options_rotation.min_inlier_ratio = 0.2;
    ransac_options_rotation.max_num_trials=2000;

    LORANSAC<RotationAverageEstimator, RotationAverageEstimator> 
            ransac(ransac_options_rotation);

    const auto report = ransac.Estimate(rotations,
                                        rotations);
    if(!report.success){
        return 0;
    }
    if(report.support.num_inliers < 3){
        return 0;
    }
    Eigen::Matrix3d estimated_rotation=report.model;

    //the translation in the last
    std::vector<Eigen::Vector3d> translations;
    for(size_t i=0; i<r_A_B.size(); ++i){
        Eigen::Vector3d trans=estimated_scale*r_A_B[i].second.transpose()*
                              t_A_B[i].first-r_A_B[i].second.transpose()*  
                              t_A_B[i].second;

        translations.push_back(trans);
    }

    RANSACOptions ransac_options_trans;
    ransac_options_trans.max_error=0.1;
    ransac_options_trans.min_inlier_ratio = 0.2;
    ransac_options_trans.max_num_trials=2000;

    LORANSAC<TranslationAverageEstimator, TranslationAverageEstimator> 
            ransac_trans(ransac_options_rotation);

    const auto report_trans = ransac_trans.Estimate(translations,translations);
    
    if(!report_trans.success){
        return 0;
    }
    if(report_trans.support.num_inliers < 3){
        return 0;
    }

    double inlier_ratio = static_cast<double>(report_trans.support.num_inliers)/
                          static_cast<double>(translations.size());

    if(inlier_ratio<0.5){
        return 0;
    }


    Eigen::Vector3d estimated_translation=report_trans.model;

    transform.block<3,3>(0,0)=estimated_rotation*estimated_scale;
    transform.block<3,1>(0,3)=estimated_translation;
    /*
    std::cout<<"Similarity estimation inlier num, rotation: "
             <<report.support.num_inliers<<" translation "
             <<report_trans.support.num_inliers<<std::endl;
    */
    return report_trans.support.num_inliers;
    
}

}//namespace sensemap