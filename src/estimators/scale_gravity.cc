//Copyright (c) 2021, SenseTime Group.
//All rights reserved.

#include "scale_gravity.h"
#include "base/pose.h"
#include "similarity_transform.h"
#include "base/similarity_transform.h"
#include "optim/ransac/loransac.h"


namespace sensemap{

bool ScaleGravityEstimator::Estimate(
    std::vector<std::unordered_map<image_t, std::pair<Eigen::Vector4d, Eigen::Vector3d>>> pose_groups_measured) {
    
    Eigen::Vector3d reconstruction_center = Eigen::Vector3d(0,0,0);
    
    std::vector<image_t> image_ids = reconstruction_->RegisterImageIds();     
    for(const auto image_id: image_ids){
        reconstruction_center = reconstruction_center + reconstruction_->Image(image_id).ProjectionCenter();    
    }
    reconstruction_center = reconstruction_center / image_ids.size();
    
    
    std::vector<int> successfully_registered_group_indices;


    std::cout<<"Scale estimation ..."<<std::endl;
    //================ scale estimation ================
    std::vector<double> scale_group;
    std::vector<double> residual_group;
    for (size_t group_idx = 0; group_idx < pose_groups_measured.size(); ++group_idx) {
        const std::unordered_map<image_t, std::pair<Eigen::Vector4d, Eigen::Vector3d>>& poses_measured =
            pose_groups_measured[group_idx];

        std::vector<Eigen::Vector3d> centers_measured;
        std::vector<Eigen::Vector3d> centers_sfm;
        for (const auto& pose : poses_measured) {
            if (reconstruction_->ExistsImage(pose.first) && reconstruction_->IsImageRegistered(pose.first)) {
                Eigen::Vector3d center_measured = ProjectionCenterFromPose(pose.second.first, pose.second.second);
                Eigen::Vector3d center_sfm = reconstruction_->Image(pose.first).ProjectionCenter();

                centers_measured.push_back(center_measured);
                centers_sfm.push_back(center_sfm);
            }
        }

        std::cout<<"centers measured count: "<<centers_measured.size()<<std::endl;
        if (centers_measured.size() < 3) {
            continue;
        }

        RANSACOptions ransac_options;
        ransac_options.max_error = 0.1;
        ransac_options.min_inlier_ratio = 0.2;

        LORANSAC<SimilarityTransformEstimator<3, true>, SimilarityTransformEstimator<3, true>> ransac(ransac_options);

        const auto report = ransac.Estimate(centers_sfm, centers_measured);

        std::cout<<"inlier count for similarity transform: "<<report.support.num_inliers<<std::endl;

        Eigen::Matrix3x4d transform;
        if (report.success &&
            static_cast<double>(report.support.num_inliers) / static_cast<double>(centers_sfm.size()) > 0.75) {
            transform = report.model;


            SimilarityTransformEstimator<3, true> similarity_estimator;
            std::vector<Eigen::Vector3d> inlier_sfm_locations;
            std::vector<Eigen::Vector3d> inlier_slam_locations;

            for (size_t i = 0; i < report.inlier_mask.size(); ++i) {
                if (report.inlier_mask[i]) {
                    inlier_sfm_locations.push_back(centers_sfm[i]);
                    inlier_slam_locations.push_back(centers_measured[i]);
                }
            }
            std::vector<Eigen::Matrix<double, 3, 4>> refined_transforms =
                similarity_estimator.Estimate(inlier_sfm_locations, inlier_slam_locations);

            std::vector<double> residuals;
            residuals.resize(centers_sfm.size());
            similarity_estimator.Residuals(centers_sfm, centers_measured, refined_transforms[0], &residuals);

            double average_residual = 0.0;
            for (const auto& residual : residuals) {
                average_residual += residual;
            }
            average_residual /= residuals.size();
            std::cout << "average residuals: " << sqrt(average_residual) << std::endl;

            std::vector<double> inlier_residuals;
            residuals.resize(centers_sfm.size());
            similarity_estimator.Residuals(inlier_sfm_locations, inlier_slam_locations, refined_transforms[0], &inlier_residuals);

            average_residual = 0.0;
            for (const auto& residual : inlier_residuals) {
                average_residual += residual;
            }
            average_residual /= inlier_residuals.size();
            std::cout << "average residuals for inliers: " << sqrt(average_residual) << std::endl;


            SimilarityTransform3 sim3(refined_transforms[0]);
            scale_group.push_back(sim3.Scale());
            successfully_registered_group_indices.push_back(group_idx);
        }
    }

    std::cout<<"scale group size: "<<scale_group.size()<<std::endl;

    if(!scale_group.size()>0){
        std::cout<<"too few scale estimation"<<std::endl;
        return false;
    }

    int nth = scale_group.size() / 2;
    std::nth_element(scale_group.begin(), scale_group.begin() + nth, scale_group.end());
    double median_scale = scale_group[nth];

    std::cout<<"median_scale: "<<median_scale<<std::endl;


    std::vector<double> normalized_scale_group = scale_group;
    for(size_t i = 0; i<normalized_scale_group.size(); ++i){
        normalized_scale_group[i] = normalized_scale_group[i] / median_scale;
    }

    int max_inlier_count_i = 0;
    double best_normalized_scale;
    for(size_t i = 0; i<normalized_scale_group.size(); ++i){
        
        double normalized_scale_i = normalized_scale_group[i];
        int inlier_count_i = 0;
        for(size_t j = 0; j<normalized_scale_group.size(); ++j){
            double normalized_scale_j = normalized_scale_group[j];
            if(abs(normalized_scale_i - normalized_scale_j) < 0.1){
                inlier_count_i ++;
            }
        }

        if(inlier_count_i > max_inlier_count_i){
            max_inlier_count_i = inlier_count_i;
            best_normalized_scale = normalized_scale_i;
        }
    }

    if(max_inlier_count_i < 2){
        std::cout<<"estimated scales have large divergency"<<std::endl;
    }

    double mean_scale = 0.0;
    std::vector<double> valid_scale_group;
    for(size_t i = 0; i<normalized_scale_group.size(); ++i){
        if(abs(best_normalized_scale - normalized_scale_group[i]) < 0.1){
            valid_scale_group.push_back(scale_group[i]);
            mean_scale += scale_group[i];
        }
    }
    mean_scale /= valid_scale_group.size();

    scale_ = mean_scale;

    std::cout<<"final scale is: "<<scale_<<std::endl;
    std::cout<<"Scale estimation done"<<std::endl;

    std::cout<<"Estimate gravity ..."<<std::endl;

    //================ gravity estimation ================
    std::vector<Eigen::Vector3d> gravity_directions_in_sfm;
    Eigen::Vector3d world_gravity(0,1,0);
    
    for(size_t i = 0; i<successfully_registered_group_indices.size(); ++i){
        int group_idx = successfully_registered_group_indices[i];
        const std::unordered_map<image_t, std::pair<Eigen::Vector4d, Eigen::Vector3d>>& poses_measured =
            pose_groups_measured[group_idx];

        for (const auto& pose : poses_measured) {
            if (reconstruction_->ExistsImage(pose.first) && reconstruction_->IsImageRegistered(pose.first)) {

                Eigen::Matrix3d rotation_measured = QuaternionToRotationMatrix(pose.second.first);
                Eigen::Matrix3d rotation_sfm = QuaternionToRotationMatrix(reconstruction_->Image(pose.first).Qvec()); 

                Eigen::Vector3d gravity_sample = rotation_sfm.transpose() * rotation_measured * world_gravity;
                gravity_directions_in_sfm.push_back(gravity_sample);
            }
        }
    }

    std::cout<<"gravity sample count: "<<gravity_directions_in_sfm.size()<<std::endl;

    int max_gravity_inlier_count = 0;
    int best_gravity_sample_index = -1;

    for(size_t i = 0; i < gravity_directions_in_sfm.size(); ++i){
        Eigen::Vector3d gravity_sample_i = gravity_directions_in_sfm[i];
        int inlier_count_i = 0;        


        for(size_t j = 0; j<gravity_directions_in_sfm.size(); ++j){
            Eigen::Vector3d gravity_sample_j = gravity_directions_in_sfm[j];
            if(gravity_sample_i.dot(gravity_sample_j)>0.999){
                inlier_count_i ++;
            }
        }

        if(inlier_count_i > max_gravity_inlier_count){
            max_gravity_inlier_count = inlier_count_i;
            best_gravity_sample_index = i;
        }
    }

    std::cout<<"max gravity inlier count: "<<max_gravity_inlier_count<<std::endl;

    if (max_gravity_inlier_count < 20 ||
        static_cast<double>(max_gravity_inlier_count) / static_cast<double>(gravity_directions_in_sfm.size()) < 0.5) {
        std::cout << "too many gravity outlier" << std::endl;
        return false;
    }

    std::vector<Eigen::Vector3d> inlier_gravity_directions_in_sfm;
    Eigen::Vector3d mean_gravity_direction = Eigen::Vector3d(0.0,0.0,0.0);
    for(size_t i = 0; i < gravity_directions_in_sfm.size(); ++i){
        if(gravity_directions_in_sfm[i].dot(gravity_directions_in_sfm[best_gravity_sample_index])>0.999){
            inlier_gravity_directions_in_sfm.push_back(gravity_directions_in_sfm[i]);
            mean_gravity_direction += gravity_directions_in_sfm[i];
        }
    }
    mean_gravity_direction = mean_gravity_direction / inlier_gravity_directions_in_sfm.size();

    gravity_ = mean_gravity_direction.normalized();



    Eigen::Vector3d any_vector = Eigen::Vector3d(0,0,1);
    Eigen::Vector3d x_axis = gravity_.cross(any_vector);
    Eigen::Vector3d y_axis = gravity_.cross(x_axis);

    Eigen::Matrix3d R; 
    R.block<3, 1>(0, 0) = x_axis;
    R.block<3, 1>(0, 1) = y_axis;
    R.block<3, 1>(0, 2) = gravity_;

    R.transposeInPlace();

    Eigen::Vector3d T = -scale_ * R * reconstruction_center;
    transform_.block<3,3>(0,0) = scale_ * R;
    transform_.block<3,1>(0,3) = T;

    std::cout<<"Gravity estimation done"<<std::endl;

    return true;
}



}

