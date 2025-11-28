//Copyright (c) 2019, SenseTime Group.
//All rights reserved.
#include "reconstruction_aligner.h"
#include "base/pose.h"
#include "optim/ransac/loransac.h"

#include "estimators/absolute_pose.h"
#include "estimators/motion_average/rotation_average.h"
#include "estimators/motion_average/translation_average.h"
#include "estimators/motion_average/similarity_average.h"
#include "estimators/mappoint_alignment.h"
#include <fstream>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace sensemap{

void ReconstructionAligner::FindCommonPoints(
                                const Reconstruction& reconstruction_src,
                                const Reconstruction& reconstruction_dst,  
                                std::vector<mappoint_t>& common_points_src, 
                                std::vector<mappoint_t>& common_points_dst){
    
    CHECK_NOTNULL(full_correspondence_graph_);
    const EIGEN_STL_UMAP(mappoint_t, class MapPoint) & mappoints_src=
                                                reconstruction_src.MapPoints();
    const EIGEN_STL_UMAP(mappoint_t, class MapPoint) & mappoints_dst=
                                                reconstruction_dst.MapPoints();
    const std::vector<image_t> & ids_dst_vec = 
                                         reconstruction_dst.RegisterImageIds();
    std::unordered_set<image_t> image_ids_dst;
    for(const auto id: ids_dst_vec){
        image_ids_dst.insert(id);
    }

    std::unordered_map<mappoint_t,std::unordered_set<mappoint_t>> common_points;

    
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    // for(const auto& it:mappoints_src){    
    for(size_t i = 0; i < mappoints_src.size(); i++){
        EIGEN_STL_UMAP(mappoint_t, class MapPoint)::const_iterator mappoints_iterator = mappoints_src.begin();
        advance(mappoints_iterator, i);
        const MapPoint& mappoint_src = mappoints_iterator->second;
        const mappoint_t mappoint_id_src = mappoints_iterator->first;
        std::vector<TrackElement> queue = mappoint_src.Track().Elements();
        const int max_transitivity = 1; //options_.max_transitivity_common_points;
        
        //find common points in the dst cluster using the transitive 
        //correspondences
        for (const TrackElement queue_elem : queue){
            std::vector<CorrespondenceGraph::Correspondence> corrs;
                // full_correspondence_graph_->FindTransitiveCorrespondences(
                //                                 queue_elem.image_id,
                //                                 queue_elem.point2D_idx,
                //                                 max_transitivity);
            full_correspondence_graph_->FindTransitiveCorrespondences(queue_elem.image_id, queue_elem.point2D_idx, max_transitivity, &corrs);

            for (const auto corr : corrs) {
                if(!image_ids_dst.count(corr.image_id)){
                    continue;
                }
                const Image &image = reconstruction_dst.Image(corr.image_id);
                CHECK(image.IsRegistered());

                const Point2D &point2D = image.Point2D(corr.point2D_idx);
                if(!point2D.HasMapPoint()) {
                    continue;
                }

                //If we find a common point in the dst cluster
                mappoint_t mappoint_id_dst = point2D.MapPointId();
#ifdef _OPENMP
#pragma omp critical(common_points)
#endif
                {
                    if (common_points[mappoint_id_src].count(mappoint_id_dst) == 0){
                        common_points[mappoint_id_src].insert(mappoint_id_dst);   
                    }
                }
            }

        }
    }

    std::cout << "Common Points = "<< common_points.size() << std::endl;
    // output the common points 
    // for(auto& common_point:common_points){
    //     CHECK(reconstruction_src.ExistsMapPoint(common_point.first));
    //     for(auto& point_id2:common_point.second){
    //         CHECK(reconstruction_dst.ExistsMapPoint(point_id2));
    //         common_points_src.emplace_back(common_point.first);
    //         common_points_dst.emplace_back(point_id2);
    //     } 
    // }

    // FIXME:
    // If the number of common points reach the max numebr threshold
    // just sample from the current common points
    const size_t common_point_threshold = 8000;
    int common_point_counter = 0;
    for(auto& common_point:common_points){
        CHECK(reconstruction_src.ExistsMapPoint(common_point.first));
        for(auto& point_id2:common_point.second){
            CHECK(reconstruction_dst.ExistsMapPoint(point_id2));
            if(common_point_counter < common_point_threshold){
                common_points_src.emplace_back(common_point.first);
                common_points_dst.emplace_back(point_id2);
                common_point_counter++;
            }
        } 
    }

    if(common_points.size() > common_point_threshold){
        std::cout << "Down Sample the common points from number "<<  common_points.size() 
            << " to " << common_point_threshold << std::endl;
    }
}

bool CorrCompare(std::pair<std::pair<image_t, image_t>, point2D_t>& p1,
                 std::pair<std::pair<image_t, image_t>, point2D_t>& p2) {
    return p1.second > p2.second;
}

void ReconstructionAligner::FindCommonPointsByCorrespondence(
                                const Reconstruction& reconstruction_src,
                                const Reconstruction& reconstruction_dst, 
                                std::vector<std::pair<std::pair<image_t, image_t>, point2D_t> >& loaded_image_pairs, 
                                size_t& loaded_max_num_image_pair,
                                std::vector<mappoint_t>& common_points_src, 
                                std::vector<mappoint_t>& common_points_dst,
                                const bool random_sample = true){

    CHECK_NOTNULL(full_correspondence_graph_);

    const auto& images1 = reconstruction_src.Images();
    const auto& images2 = reconstruction_dst.Images();   

    auto strong_loops = full_correspondence_graph_->GetStrongLoopPairs();
    auto normal_loops = full_correspondence_graph_->GetLoopPairsInfo();
    auto normal_pairs = full_correspondence_graph_->GetNormalPairs();

    std::vector<image_pair_t> candidate_pairs;
    for(auto pair: strong_loops){
        candidate_pairs.push_back(pair);
    }
    for(auto pair_info: normal_loops){
        auto pair = pair_info.first;
        candidate_pairs.push_back(pair);
    }
    for(auto pair: normal_pairs){
        candidate_pairs.push_back(pair);
    }
    if(candidate_pairs.size()==0){
        auto normal_image_pairs = full_correspondence_graph_->ImagePairs();
        for(auto pair: normal_image_pairs){
            candidate_pairs.push_back(pair.first);
        }
    }
    std::cout<<"[InitializeMotionAverage] candidate image_pairs size: "<<candidate_pairs.size()<<std::endl;

    // auto normal_image_pairs = full_correspondence_graph_->ImagePairs();
    // for(auto pair: normal_image_pairs){
    //     candidate_pairs.push_back(pair.first);
    // }
    // std::cout<<"candidate_pairs.size() "<<candidate_pairs.size()<<std::endl;
    
    //TODO: speed up common points finding
    // Find common image ids.
    std::vector<std::pair<std::pair<image_t, image_t>, point2D_t> > image_pairs;
    std::vector<std::pair<std::pair<image_t, image_t>, point2D_t> > normal_image_pairs;
    std::set<image_t> src_images,dst_images;
    std::set<image_t> all_src_images,all_dst_images;
    size_t max_num_image_pair;
    if (options_.load_strong_pairs) {
        image_pairs = loaded_image_pairs;
        max_num_image_pair = loaded_max_num_image_pair;

    } else {
        for(auto image_pair_id: candidate_pairs){
            image_t image_id1,image_id2;
            sensemap::utility::PairIdToImagePair(image_pair_id,&image_id1,&image_id2);

            // bool strong_loop = false;
        
            class Image image1,image2;
            if(reconstruction_src.ExistsImage(image_id1)){
                image1 = reconstruction_src.Image(image_id1);
            }else if(reconstruction_dst.ExistsImage(image_id1)){
                image1 = reconstruction_dst.Image(image_id1);
            }else{
                // std::cout<<"image "<<image_id1<<" not in whole reconstructions"<<std::endl;
                continue;
            }
            if(reconstruction_src.ExistsImage(image_id2)){
                image2 = reconstruction_src.Image(image_id2);
            }else if(reconstruction_dst.ExistsImage(image_id2)){
                image2 = reconstruction_dst.Image(image_id2);
            }else{
                // std::cout<<"image "<<image_id2<<" not in whole reconstructions"<<std::endl;
                continue;
            }

            // if (image1.LabelId() != image2.LabelId() ) {
            //     strong_loop = true;
            // }
            point2D_t num_corr = full_correspondence_graph_->NumCorrespondencesBetweenImages(image_id1, image_id2);

            // if(strong_loop){
            if(reconstruction_src.ExistsImage(image_id1) && reconstruction_dst.ExistsImage(image_id2)){
                image_pairs.emplace_back(std::make_pair(std::make_pair(image_id1, image_id2),num_corr));
                src_images.emplace(image_id1);
                dst_images.emplace(image_id2);
            }else if(reconstruction_dst.ExistsImage(image_id1) && reconstruction_src.ExistsImage(image_id2)){
                image_pairs.emplace_back(std::make_pair(std::make_pair(image_id2, image_id1),num_corr));
                src_images.emplace(image_id2);
                dst_images.emplace(image_id1);
            }
            // }
        }

        // TODO: check whether add the loop pair in one sequence into motion average
        // std::cout<<"image_pairs.size() "<<image_pairs.size()<<std::endl;

        if (image_pairs.size() == 0) {
            return;

        }

        std::sort(image_pairs.begin(),image_pairs.end(),CorrCompare);
        std::sort(normal_image_pairs.begin(),normal_image_pairs.end(),CorrCompare);

        max_num_image_pair = image_pairs.size();  
    }
    


    if(options_.save_strong_pairs&& !options_.load_strong_pairs){
        std::cout << "saving strong pairs" << std ::endl;
        loaded_image_pairs.clear();
        if(image_pairs.size() > options_.candidate_strong_pairs_num){
            loaded_image_pairs.insert(loaded_image_pairs.end(), image_pairs.begin(), image_pairs.begin() + options_.candidate_strong_pairs_num);
        } else{
            loaded_image_pairs = image_pairs;
            loaded_image_pairs.insert(loaded_image_pairs.end(), normal_image_pairs.begin(), normal_image_pairs.begin() + (options_.candidate_strong_pairs_num - max_num_image_pair));
        }
        loaded_max_num_image_pair = max_num_image_pair;
    }

    std::cout << "image_pairs.size() " << image_pairs.size() << " max_num_image_pair " << max_num_image_pair
              << std::endl;

    std::unordered_set<mappoint_t> mappoint_set1;
    std::unordered_set<mappoint_t> mappoint_set2;

    std::unordered_map<mappoint_t, std::unordered_set<mappoint_t>> common_points;

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for (size_t pair_id = 0; pair_id < max_num_image_pair; pair_id++) {
        auto image_pair = image_pairs[pair_id];
        image_t image_id1 = image_pair.first.first;
        image_t image_id2 = image_pair.first.second;
        if(!images1.count(image_id1)){
            continue;
        }
        if(!images2.count(image_id2)){
            continue;
        }

        const class Image& image1 = images1.at(image_id1);
        const class Image& image2 = images2.at(image_id2);
        const FeatureMatches& feature_matches =
            full_correspondence_graph_->FindCorrespondencesBetweenImages(image_id1, image_id2);
        for (const auto& feature_match : feature_matches) {
            point2D_t point2D_idx1 = feature_match.point2D_idx1;
            point2D_t point2D_idx2 = feature_match.point2D_idx2;
            const class Point2D& point2D1 = image1.Point2D(point2D_idx1);
            const class Point2D& point2D2 = image2.Point2D(point2D_idx2);
            if (!point2D1.HasMapPoint() || !point2D2.HasMapPoint()) {
                continue;
            }
            mappoint_t mappoint_id1 = point2D1.MapPointId();
            mappoint_t mappoint_id2 = point2D2.MapPointId();
            const class MapPoint& mappoint1 = reconstruction_src.MapPoint(mappoint_id1);
            const class MapPoint& mappoint2 = reconstruction_dst.MapPoint(mappoint_id2);

#ifdef _OPENMP
#pragma omp critical(common_points)
#endif
            if (mappoint1.Track().Length() >= 3 &&
                mappoint2.Track().Length() >= 3 &&
                mappoint_set1.count(mappoint_id1) == 0 &&
                mappoint_set2.count(mappoint_id2) == 0) {
                    
                mappoint_set1.insert(mappoint_id1);
                mappoint_set2.insert(mappoint_id2);

                common_points[mappoint_id1].insert(mappoint_id2);   
                // std::cout<<"common point set "<<image_id1<<"-"<<mappoint_id1<<" "<<image_id2<<"-"<<mappoint_id2<<std::endl;
            }
        }
    }

    if (random_sample) {
        // std::cout << "common_points.size() " << common_points.size() << std::endl;

        // If the number of common points reach the max numebr threshold
        // just sample from the current common points
        const size_t common_point_threshold = std::min(80000, (int)common_points.size());
        // int common_point_counter = 0;
        
        if (common_point_threshold > 80000) {
            std::cout << "Prune common points from " << common_points.size() << " to 80000" << std::endl;
        }

        std::vector<std::pair<mappoint_t, std::unordered_set<mappoint_t>>> unique_common_points;
        unique_common_points.reserve(common_points.size());
        for (auto& common_point : common_points) {
            unique_common_points.emplace_back(common_point.first, common_point.second);
        }

        std::srand(time(0));
        for (int i = 0; i < common_point_threshold; ++i) {
            int index = i + std::rand() % (common_points.size() - i);
            auto common_point = unique_common_points[index];
            for (auto& point_id2 : common_point.second) {
                CHECK(reconstruction_dst.ExistsMapPoint(point_id2));

                common_points_src.emplace_back(common_point.first);
                common_points_dst.emplace_back(point_id2);
            }
            std::swap(unique_common_points[index], unique_common_points[i]);
        }

        std::cout << "Collect " << common_points_src.size() << " mappoint pairs" << std::endl;
    } else {
        for (auto& common_point : common_points) {
            CHECK(reconstruction_src.ExistsMapPoint(common_point.first));
            for (auto& point_id2 : common_point.second) {
                CHECK(reconstruction_dst.ExistsMapPoint(point_id2));

                common_points_src.emplace_back(common_point.first);
                common_points_dst.emplace_back(point_id2);
            }
        }
    }
    // if (common_points.size() > common_point_threshold) {
    //     std::cout << "Down Sample the common points from number " << common_points.size() << " to "
    //               << common_point_threshold << std::endl;
    // }
}


size_t ReconstructionAligner::RelativeTransformFrom3DCorrespondences(
                                      const Reconstruction& reconstruction_src,
                                      const Reconstruction& reconstruction_dst,
                                      Eigen::Matrix3x4d& transform,
                                      std::vector<std::pair<std::pair<image_t, image_t>, point2D_t> >& loaded_image_pairs,
                                      size_t& loaded_max_num_image_pair,
                                      bool& save_image_pairs, int write_inliers){

    std::vector<mappoint_t> common_points_src;
    std::vector<mappoint_t> common_points_dst;
    Reconstruction sub_reconstruction_src_tmp;
    Reconstruction sub_reconstruction_dst_tmp;
    std::cout<<"Find common 3D points in two reconstructions ..."<<std::endl;

    FindCommonPointsByCorrespondence(reconstruction_src, reconstruction_dst, loaded_image_pairs,
                     loaded_max_num_image_pair, common_points_src, common_points_dst);
    std::cout<<"Find "<<common_points_src.size()<<" common points"<<std::endl;
    if(common_points_src.size()){
        save_image_pairs = true;
        std::cout << "save_image_pairs inside " << save_image_pairs<<" " <<loaded_image_pairs.size()<< std::endl;
    }
    if(common_points_src.size()<options_.min_num_common_points){
        return 0; // -- If we can not find enough common map point, return 0
    }

    RANSACOptions ransac_options;
    ransac_options.max_error = 8.0f;
    // ransac_options.max_error = reconstruction_src.baseline_distance * 0.1;
    ransac_options.min_inlier_ratio = 0.2f;
    ransac_options.max_num_trials = 50000;

    LORANSAC<MapPointAlignmentEstimator, MapPointAlignmentEstimator> 
                                                        ransac(ransac_options);
    
    ransac.estimator.SetReconstruction(&reconstruction_src,&reconstruction_dst);
    ransac.local_estimator.SetReconstruction(&reconstruction_src,
                                             &reconstruction_dst);

    const auto report = ransac.Estimate(common_points_src, common_points_dst);

    if(!report.success){
        return 0;
    }
    else{
        transform=report.model;
    }

#if 0
    if (report.support.num_inliers >= options_.min_num_common_points_inlier) {
        std::string filename = StringPrintf("points-%d-%d", reconstruction_src.NumImages(), reconstruction_dst.NumImages());
        std::ofstream file1(filename + "1.obj", std::ofstream::out);
        std::ofstream file2(filename + "2.obj", std::ofstream::out);
        std::ofstream file_align1(filename + "-align1.obj", std::ofstream::out);
        std::ofstream file_align2(filename + "-align2.obj", std::ofstream::out);
        for (size_t i = 0; i < common_points_src.size(); ++i) {
            mappoint_t id1 = common_points_src.at(i);
            mappoint_t id2 = common_points_dst.at(i);
            MapPoint mappoint1 = reconstruction_src.MapPoint(id1);
            MapPoint mappoint2 = reconstruction_dst.MapPoint(id2);
            Eigen::Vector3d X1 = mappoint1.XYZ();
            Eigen::Vector3d X2 = mappoint2.XYZ();
            Eigen::Vector3i C1 = mappoint1.Color().cast<int>();
            Eigen::Vector3i C2 = mappoint2.Color().cast<int>();
            file1 << "v " << X1.x() << " " << X1.y() << " " << X1.z() << " "
                 << C1.x() << " " << C1.y() << " " << C1.z() << std::endl;
            file2 << "v " << X2.x() << " " << X2.y() << " " << X2.z() << " "
                 << C2.x() << " " << C2.y() << " " << C2.z() << std::endl;

            Eigen::Vector3d X12 = report.model * X1.homogeneous();
            file_align1 << "v " << X12.x() << " " << X12.y() << " " << X12.z() << " "
                 << C1.x() << " " << C1.y() << " " << C1.z() << std::endl;
            file_align2 << "v " << X2.x() << " " << X2.y() << " " << X2.z() << " "
                 << C2.x() << " " << C2.y() << " " << C2.z() << std::endl;
        }
        file1.close();
        file2.close();
        file_align1.close();
        file_align2.close();
    }
#endif

    if(report.support.num_inliers < options_.min_num_common_points_inlier){
        return 0;
    }
    else{
        return report.support.num_inliers;
    }
}

void ReconstructionAligner::SetGraphs(
                    const CorrespondenceGraph* correspondence_graph){
    // view_graph_=view_graph;
    full_correspondence_graph_=correspondence_graph;
    
}

}//namespace sensemap


