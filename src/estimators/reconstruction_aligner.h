//Copyright (c) 2019, SenseTime Group.
//All rights reserved.

#ifndef SENSEMAP_ESTIMATORS_RECONSTRUCTION_ALIGNER_H_
#define SENSEMAP_ESTIMATORS_RECONSTRUCTION_ALIGNER_H_

#include <vector>
#include "util/types.h"
#include "base/reconstruction.h"

namespace sensemap{

struct ReconstructionAlignerOptions{
    
    //Minimal number of inlier matches between connected images across clusters
    int min_inlier_matches_connected_images=15;

    //Minimal similarity between connected images across clusters
    int min_similarity_connected_images=0.6;
    
    //Max transitivity for finding common points
    int max_transitivity_common_points=5;

    //The minimal number of common points to estimate the relaitve similarity
    //transform with a neighbor cluster
    int min_num_common_points = 40;
    int min_num_common_points_inlier = 30;
    
    //The minimal number of inlier 2D-3D correspondence for registering an image
    int abs_pose_min_num_inliers = 30;
    //maximal reprojection error in absolute pose estimation
    double abs_pose_max_error = 12.0;
    //minimal inlier ratio in absolute pose estimation
    double abs_pose_min_inlier_ratio = 0.25;

    // minimal inlier connected images for registering image with two-view 
    // geometry 
    int min_num_inlier_connected_images = 2;

    // minimal number of cross registered images for estimating the relative
    // similarity transform
    int min_num_cross_registered_image= 3;

    //number of threads for absolute pose estimation
    int num_threads = 1;

    bool save_strong_pairs = true;

    size_t candidate_strong_pairs_num = 50;

    bool load_strong_pairs = false;
};


class ReconstructionAligner{

public:
    ReconstructionAligner(const ReconstructionAlignerOptions& options)
        :options_(options){
        // view_graph_=nullptr;
        full_correspondence_graph_=nullptr;
    }

    void FindCommonPoints(const Reconstruction& reconstruction_src,
                          const Reconstruction& reconstruction_dst,  
                          std::vector<mappoint_t>& common_points_src, 
                          std::vector<mappoint_t>& common_points_dst);

    void FindCommonPointsByCorrespondence(
                          const Reconstruction& reconstruction_src,
                          const Reconstruction& reconstruction_dst,  
                          std::vector<std::pair<std::pair<image_t, image_t>, point2D_t> >& loaded_image_pairs,
                          size_t& loaded_max_num_image_pair,
                          std::vector<mappoint_t>& common_points_src, 
                          std::vector<mappoint_t>& common_points_dst,
                          const bool random_sample);

    //compute similarity transform between two reconstructins using 3D point 
    //correspondences
    size_t RelativeTransformFrom3DCorrespondences(
                                      const Reconstruction& reconstruction_src,
                                      const Reconstruction& reconstruction_dst,
                                      Eigen::Matrix3x4d& transform,
                                      std::vector<std::pair<std::pair<image_t, image_t>, point2D_t> >& loaded_image_pairs,
                                      size_t& loaded_max_num_image_pair,
                                      bool &save_image_pairs, int write_inliers=0);


    void SetGraphs(const CorrespondenceGraph* correspondence_graph);


private:
 
    //const ViewGraph* view_graph_;
    ReconstructionAlignerOptions options_;
    const CorrespondenceGraph* full_correspondence_graph_;
   
};

}//namespace sensemap

#endif //SENSEMAP_ESTIMATORS_RECONSTRUCTION_ALIGNER_H_