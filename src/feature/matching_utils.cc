//Copyright (c) 2019, SenseTime Group.
//All rights reserved.
#include <mutex>
#include <map>
#include <iomanip>
#include <functional>
#include "feature/matching_utils.h"
#include "feature/utils.h"
#include "util/timer.h"
#include "util/misc.h"
#include "estimators/utils.h"

#if defined(_OPENMP)
#include <omp.h>
#endif

#ifdef CUDA_ENABLED
#include "SiftGPU/SiftGPU.h"
#endif

#define FEATURE_MATCH_SEARCH_RANGE 30

namespace sensemap{

bool PairMatchingOptions::Check() const {
  if (use_gpu) {
    CHECK_OPTION_GT(CSVToVector<int>(gpu_index).size(), 0);
  }
  CHECK_OPTION_GT(max_ratio, 0.0);
  CHECK_OPTION_GT(max_distance, 0.0);
  CHECK_OPTION_GT(max_error, 0.0);
  CHECK_OPTION_GE(min_num_trials, 0);
  CHECK_OPTION_GT(max_num_trials, 0);
  CHECK_OPTION_LE(min_num_trials, max_num_trials);
  CHECK_OPTION_GE(min_inlier_ratio, 0);
  CHECK_OPTION_LE(min_inlier_ratio, 1);
  CHECK_OPTION_GE(min_num_inliers, 0);
  return true;
}

std::vector<image_t> GetOrderedImageIds(FeatureDataContainer* container, 
    const std::vector<image_t> &image_ids) {
  
    std::vector<Image> ordered_images;
    ordered_images.reserve(image_ids.size());
    for (const auto image_id : image_ids){
        ordered_images.push_back(container->GetImage(image_id));
    }

    std::sort(ordered_images.begin(), ordered_images.end(),
              [](const Image &image1, const Image &image2) {
                  return image1.Name() < image2.Name();
              });

    std::vector<image_t> ordered_image_ids;
    ordered_image_ids.reserve(image_ids.size());
    for (const auto &image : ordered_images){
        ordered_image_ids.push_back(image.ImageId());
    }

    return ordered_image_ids;
}



void IndexImagesInVisualIndex(const int num_threads, const int num_checks,
							  const int max_num_features,
							  const std::vector<image_t>& image_ids,
							  FeatureDataContainer* container,
							  retrieval::VisualIndex<>* visual_index) {
  	retrieval::VisualIndex<>::IndexOptions index_options;
  	index_options.num_threads = num_threads;
  	index_options.num_checks = num_checks;

    std::cout << std::endl;
  	for (size_t i = 0; i < image_ids.size(); ++i) {	
		Timer timer;
		timer.Start();
		
		auto keypoints = container->GetKeypoints(image_ids[i]);
		auto descriptors = container->GetCompressedDescriptors(image_ids[i]);
		if (max_num_features > 0 && descriptors.rows() > max_num_features) {
	  		ExtractTopScaleFeatures(&keypoints, &descriptors, max_num_features);
		}
        if(keypoints.size()>0){
		    visual_index->Add(index_options, image_ids[i], keypoints, descriptors);
        }
        std::cout << "\r";
        std::cout << "Indexing image [" << i + 1 << " / " << image_ids.size()
                  << "] in " << std::setfill(' ') << std::fixed 
                  << std::setprecision(3)
                  << timer.ElapsedSeconds() << "s" << std::flush;
  	}
    std::cout << std::endl;

  	// Compute the TF-IDF weights, etc.
  	visual_index->Prepare();
}

void FindNearestNeighborsInVisualIndex(
	const int num_threads, const int num_images, const int num_neighbors,
	const int num_checks, const int num_images_after_verification,
	const int max_num_features, const std::vector<image_t>& image_ids,
	FeatureDataContainer* container,
	retrieval::VisualIndex<>* visual_index, 
    std::vector<std::vector<image_t>>& neighbors,
    const bool hybrid_status,
	const float max_score_factor) {

    std::unordered_map<image_t, int> image_orders;
    for(size_t i = 0; i<image_ids.size(); ++i){
        image_orders.emplace(image_ids[i],i);
    }
	
	struct Retrieval{
		image_t image_id = kInvalidImageId;
		std::vector<retrieval::ImageScore> image_scores;
	};

	// Create a thread pool to retrieve the nearest neighbors.
	ThreadPool retrieval_thread_pool(num_threads);
	JobQueue<Retrieval> retrieval_queue(num_threads);

	// The retrieval thread kernel function. Note that the descriptors should be
	// extracted outside of this function sequentially to avoid any concurrent
	// access to the database causing race conditions.
	retrieval::VisualIndex<>::QueryOptions query_options;

    // -- Double the number of image when hybrid matching is enabled
	query_options.max_num_images = hybrid_status ? num_images*3 : num_images; 
	query_options.num_neighbors = num_neighbors;
	query_options.num_checks = num_checks;
	query_options.num_images_after_verification = num_images_after_verification;
	
	auto QueryFunc = [&](const image_t image_id) {
		auto keypoints = container->GetKeypoints(image_id);
		auto descriptors = container->GetCompressedDescriptors(image_id);
		if (max_num_features > 0 && descriptors.rows() > max_num_features){
			ExtractTopScaleFeatures(&keypoints, &descriptors, max_num_features);
		}

		Retrieval retrieval;
		retrieval.image_id = image_id;
		visual_index->Query(query_options, keypoints, descriptors,
							&retrieval.image_scores);

		CHECK(retrieval_queue.Push(std::move(retrieval)));
	};

	// Initially, make all retrieval threads busy
	size_t image_idx = 0;
	const size_t init_num_tasks =
		std::min(image_ids.size(), 2 * retrieval_thread_pool.NumThreads());
	for (; image_idx < init_num_tasks; ++image_idx){
		retrieval_thread_pool.AddTask(QueryFunc, image_ids[image_idx]);
	}
    
	// Pop the finished retrieval results.
    neighbors.clear();
    neighbors.resize(image_ids.size());
    std::cout << std::endl;
    for (size_t i = 0; i < image_ids.size(); ++i){
		
		Timer timer;
		timer.Start();
	
		// Push the next image to the retrieval queue.
		if (image_idx < image_ids.size()){
			retrieval_thread_pool.AddTask(QueryFunc, image_ids[image_idx]);
			image_idx += 1;
		}

		// Pop the next results from the retrieval queue.
		const auto retrieval = retrieval_queue.Pop();
		CHECK(retrieval.IsValid());

		const auto &image_id = retrieval.Data().image_id;
		const auto &image_scores = retrieval.Data().image_scores;

        size_t image_counter = 0;

		// Get the neighbors from the scores.
        CHECK(image_orders.find(image_id)!= image_orders.end());
        const auto image_order = image_orders.at(image_id);
        CHECK_LT(image_order, image_ids.size());
		neighbors[image_order].reserve(num_images);

        std::vector<float> scores;
        scores.reserve(num_images);
        
        for (const auto image_score : image_scores){
                // Check the image pair has the same label or not
                /*
                if (hybrid_status && 
                    container->GetImage(image_id).LabelId() ==
                    container->GetImage(image_score.image_id).LabelId()){ 
                    continue;      
                }*/
                if(image_score.image_id == image_id){
                    continue;
                }

			    neighbors[image_order].push_back(image_score.image_id);
                scores.push_back(image_score.score);   
		}

        float max_score = *std::max_element(scores.begin(),scores.end());
        
        std::vector<image_t> neighbors_kept;
        for(size_t j = 0; j < neighbors[image_order].size(); ++j){
            if(scores[j] > max_score * max_score_factor){

                if(!hybrid_status||( hybrid_status && 
                    container->GetImage(image_id).LabelId() !=
                    container->GetImage(neighbors[image_order][j]).LabelId())){    

                    neighbors_kept.push_back(neighbors[image_order][j]); 
                    image_counter++;
                }
            }
            if(image_counter >= num_images){
                break;
            }
        }
        neighbors[image_order] = neighbors_kept;
        std::cout<<"\r"<<std::endl;
        std::cout<<" Neighbor count: "<<neighbors[image_order].size()<<" ";
        std::cout << "Finding nearest neighbors for image [" << image_order + 1
                  << " / " << image_ids.size() << "] in " 
                  << std::setfill(' ') << std::fixed << std::setprecision(3) 
                  << timer.ElapsedSeconds() << "s" << std::flush;
	}
    std::cout << std::endl;
}

void FindNearestNeighborsInVladVisualIndex(const int num_images, const std::vector<image_t>& image_ids,
                                           FeatureDataContainer* container,
                                           VladVisualIndex* vlad_visual_index,
                                           std::vector<std::vector<image_t>>& neighbors, const bool hybrid_status,
                                           const float max_score_factor) {
    
    std::unordered_map<image_t, int> image_orders;
    for (size_t i = 0; i < image_ids.size(); ++i) {
        image_orders.emplace(image_ids[i], i);
    }

    struct Retrieval {
        image_t image_id = kInvalidImageId;
        std::vector<retrieval::ImageScore> image_scores;
    };

  
    VladVisualIndex::QueryOptions query_options;

    // -- Double the number of image when hybrid matching is enabled
    query_options.max_num_images = hybrid_status ? num_images * 3 : num_images;

    // Start the query
    neighbors.clear();
    neighbors.resize(image_ids.size());

    Timer timer;
    timer.Start();
   
    for (size_t i = 0; i < image_ids.size(); ++i) {
        image_t image_id = image_ids[i];
        std::vector<retrieval::ImageScore> image_scores;
        const VladVisualIndex::VLAD& current_vlad = container->GetVladVector(image_id); 
        vlad_visual_index->Query(query_options,current_vlad, &image_scores);

        size_t image_counter = 0;
        // Get the neighbors from the scores.
        CHECK(image_orders.find(image_id) != image_orders.end());
        const auto image_order = image_orders.at(image_id);
        CHECK_LT(image_order, image_ids.size());
        neighbors[image_order].reserve(num_images);

        std::vector<float> scores;
        scores.reserve(num_images);

        for (const auto image_score : image_scores) {
            if (image_score.image_id == image_id) {
                continue;
            }

            neighbors[image_order].push_back(image_score.image_id);
            scores.push_back(image_score.score);
        }

        // float max_score = *std::max_element(scores.begin(), scores.end());

        std::vector<image_t> neighbors_kept;
        for (size_t j = 0; j < neighbors[image_order].size(); ++j) {
            // if (scores[j] > max_score * max_score_factor) {
            if (!hybrid_status || (hybrid_status && container->GetImage(image_id).LabelId() !=
                container->GetImage(neighbors[image_order][j]).LabelId())) {
                neighbors_kept.push_back(neighbors[image_order][j]);
                image_counter++;
            }
            // }
            if (image_counter >= num_images) {
                break;
            }
        }
        neighbors[image_order] = neighbors_kept;

        std::cout << "Neighbor count: " << neighbors[image_order].size() << " ";
        std::cout << "Finding nearest neighbors for image [" << image_order + 1 << " / " << image_ids.size() << "] in "
                  << std::setfill(' ') << std::fixed << std::setprecision(3) << timer.ElapsedSeconds() << "s"
                  << std::endl << std::flush;
    }

    std::cout<< "Query in "<<timer.ElapsedSeconds()<<"s"<<std::endl;
}

bool Consecutive(std::vector<int> ids, int max_perspective_idx){
    
    std::stable_sort(ids.begin(),ids.end());
    bool consecutive = true;
    if(ids.size()<=1){
        return true;
    }
    for(size_t i = 1; i<ids.size(); ++i){
        if(ids[i]!=max_perspective_idx && ids[i]-ids[i-1]>1){
            consecutive = false;
            break;
        }
        else if(ids[i] == max_perspective_idx && 
                ids[i]-ids[i-1]>1 && ids[0] != 0){
            consecutive = false;
            break;
        }            
    }
    return consecutive;
}




bool MatchFromRepetitivePattern(const FeatureMatches& matches,
								const PanoramaIndexs& panorama_indices1,
								const PanoramaIndexs& panorama_indices2,
                                const PieceIndexs& piece_indices1,
								const PieceIndexs& piece_indices2,
                                double& covered_area_ratio,
                                const double min_covered_sub_image_ratio,
                                const int perspective_image_count1,
                                const int perspective_image_count2,
                                const int min_matched_feature_per_piece,
                                bool piecewise){
                                    
    std::unordered_set<int> sub_image_ids1,sub_image_ids2; 
    std::unordered_map<int,int> match_count_per_sub_image1;
    std::unordered_map<int,int> match_count_per_sub_image2;
    if (panorama_indices1.empty() || panorama_indices2.empty()) return false;
    if (piece_indices1.empty() || piece_indices2.empty()) return false;

    for(const auto match: matches){
        CHECK_LT(match.point2D_idx1,panorama_indices1.size());
        CHECK_LT(match.point2D_idx2,panorama_indices2.size());
        int sub_image_id1;
        if(piece_indices1.size()==0){
            sub_image_id1 = panorama_indices1[match.point2D_idx1].sub_image_id;
        }else{
            sub_image_id1 =piecewise?
                            (panorama_indices1[match.point2D_idx1].sub_image_id * perspective_image_count1 /2 +
                             piece_indices1[match.point2D_idx1].piece_id) :
                            panorama_indices1[match.point2D_idx1].sub_image_id;
        }
        int sub_image_id2;
        if(piece_indices2.size()==0){
            sub_image_id2 = panorama_indices2[match.point2D_idx2].sub_image_id;
        }else{
            sub_image_id2 =piecewise?
                            (panorama_indices2[match.point2D_idx2].sub_image_id * perspective_image_count2 /2 +
                             piece_indices2[match.point2D_idx2].piece_id) :
                            panorama_indices2[match.point2D_idx2].sub_image_id;
        }

        if(sub_image_ids1.count(sub_image_id1) == 0){
            sub_image_ids1.insert(sub_image_id1);
        }
        if(sub_image_ids2.count(sub_image_id2) == 0){
            sub_image_ids2.insert(sub_image_id2);
        }
        if(match_count_per_sub_image1.find(sub_image_id1) == 
           match_count_per_sub_image1.end()){
            match_count_per_sub_image1.emplace(sub_image_id1,1);
        }
        else{
            match_count_per_sub_image1.at(sub_image_id1) += 1;
        }

        if(match_count_per_sub_image2.find(sub_image_id2) == 
           match_count_per_sub_image2.end()){
            match_count_per_sub_image2.emplace(sub_image_id2,1);
        }
        else{
            match_count_per_sub_image2.at(sub_image_id2) += 1;
        }
    }


    std::vector<int> sub_image_ids1_v, sub_image_ids2_v;
    for(const auto id: sub_image_ids1){
        if(match_count_per_sub_image1.at(id)>min_matched_feature_per_piece){
            sub_image_ids1_v.push_back(id);
        }
    } 

    for(const auto id: sub_image_ids2){
        if(match_count_per_sub_image2.at(id)>min_matched_feature_per_piece){
            sub_image_ids2_v.push_back(id);
        }
    }

    std::cout<<"sub image ids1: "<<std::endl;
    for(size_t i = 0; i<sub_image_ids1_v.size(); ++i){
        std::cout<<sub_image_ids1_v[i]<<" ";
    }
    std::cout<<std::endl;

    std::cout<<"sub image ids2: "<<std::endl;
    for(size_t i = 0; i<sub_image_ids2_v.size(); ++i){
        std::cout<<sub_image_ids2_v[i]<<" ";
    }
    std::cout<<std::endl;
    
    double sub_image_ratio1 = static_cast<double>(sub_image_ids1_v.size())/
                              static_cast<double>(perspective_image_count1);
    double sub_image_ratio2 = static_cast<double>(sub_image_ids2_v.size())/
                              static_cast<double>(perspective_image_count2);

    covered_area_ratio = std::min(sub_image_ratio1,sub_image_ratio2);

    if((sub_image_ratio1<=min_covered_sub_image_ratio&&
        Consecutive(sub_image_ids1_v,perspective_image_count1-1))||
       (sub_image_ratio2<=min_covered_sub_image_ratio&&
        Consecutive(sub_image_ids2_v,perspective_image_count2-1))||
        sub_image_ratio1<min_covered_sub_image_ratio/2||
        sub_image_ratio2<min_covered_sub_image_ratio/2){
        return true;
    }
    else{
        return false;
    }
}

//compute the distance matrix of the descriptors in two images
Eigen::MatrixXi ComputeSiftDistanceMatrix(
    const FeatureKeypoints* keypoints1, const FeatureKeypoints* keypoints2,
    const FeatureDescriptors& descriptors1,
    const FeatureDescriptors& descriptors2,
    const std::function<bool(float, float, float, float)>& guided_filter) {
  
    if (guided_filter != nullptr) {
        CHECK_NOTNULL(keypoints1);
        CHECK_NOTNULL(keypoints2);
        CHECK_EQ(keypoints1->size(), descriptors1.rows());
        CHECK_EQ(keypoints2->size(), descriptors2.rows());
    }

    const Eigen::Matrix<int, Eigen::Dynamic, 32> descriptors1_int =
          descriptors1.cast<int>();
    const Eigen::Matrix<int, Eigen::Dynamic, 32> descriptors2_int =
          descriptors2.cast<int>();

    Eigen::Matrix<int, 1, 32> mean_descriptor = Eigen::Matrix<int, 1, 32>::Ones() * 128;


    Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> dists(
        descriptors1.rows(), descriptors2.rows());

    for (FeatureDescriptors::Index i1 = 0; i1 < descriptors1.rows(); ++i1) {
        for (FeatureDescriptors::Index i2 = 0; i2 < descriptors2.rows(); ++i2) {
            if (guided_filter != nullptr &&
                guided_filter((*keypoints1)[i1].x, (*keypoints1)[i1].y,
                              (*keypoints2)[i2].x, (*keypoints2)[i2].y)) {
                dists(i1, i2) = -1.0;
            } 
            else {
                dists(i1, i2) = 
                    (descriptors1_int.row(i1)-mean_descriptor).dot(descriptors2_int.row(i2)-mean_descriptor);
                    //256 * 12 - (descriptors1_int.row(i1) - descriptors2_int.row(i2)).norm();
            }
        }
    }

  return dists;
}


//compute the distance matrix of the descriptors in two images
Eigen::MatrixXi ComputeSiftDistanceMatrix(
    const FeatureKeypoints* keypoints1,
    const FeatureKeypoints* keypoints2,
    const FeatureDescriptors& descriptors1,
    const FeatureDescriptors& descriptors2,
    const std::vector<Eigen::Matrix3f>& H_list,
    const int search_range,
    const std::function<bool(float, float, float, float)>& guided_filter_F,
    const std::vector<std::function<bool(float, float, float, float)> >& guided_filter_H_list) {
  
    float x1, y1, x2, y2, dx, dy, wx1, wy1;

    if (guided_filter_F != nullptr) {
        CHECK_NOTNULL(keypoints1);
        CHECK_NOTNULL(keypoints2);
        CHECK_EQ(keypoints1->size(), descriptors1.rows());
        CHECK_EQ(keypoints2->size(), descriptors2.rows());
    }

    const Eigen::Matrix<int, Eigen::Dynamic, 128> descriptors1_int =
          descriptors1.cast<int>();
    const Eigen::Matrix<int, Eigen::Dynamic, 128> descriptors2_int =
          descriptors2.cast<int>();

    Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> dists(
        descriptors1.rows(), descriptors2.rows());

    for (FeatureDescriptors::Index i1 = 0; i1 < descriptors1.rows(); ++i1) {
        x1 = (*keypoints1)[i1].x;
        y1 = (*keypoints1)[i1].y;

        std::vector<float> wxs1, wys1;
        for (const auto & H : H_list) {
            float w = H(2, 0) * x1 + H(2, 1) * y1 + H(2, 2);
            wx1 = (H(0, 0) * x1 + H(0, 1) * y1 + H(0, 2)) / w;
            wy1 = (H(1, 0) * x1 + H(1, 1) * y1 + H(1, 2)) / w;
            wxs1.push_back(wx1);
            wys1.push_back(wy1);
        }
        for (FeatureDescriptors::Index i2 = 0; i2 < descriptors2.rows(); 
            ++i2) {
            x2 = (*keypoints2)[i2].x;
            y2 = (*keypoints2)[i2].y;

            bool valid_F = (guided_filter_F != nullptr &&
                    guided_filter_F((*keypoints1)[i1].x, (*keypoints1)[i1].y,
                                (*keypoints2)[i2].x, (*keypoints2)[i2].y));
            if (valid_F) {
                dists(i1, i2) = -1.0f;
                continue;
            }
            for (size_t k = 0; k < H_list.size(); ++k) {
                dx = wxs1[k] - x2;
                dy = wys1[k] - y2;
                bool not_match = (dx < -search_range || dx > search_range ||
                                  dy < -search_range || dy > search_range); 
                if (not_match || 
                    (guided_filter_H_list[k] != nullptr &&
                    guided_filter_H_list[k]((*keypoints1)[i1].x, 
                                            (*keypoints1)[i1].y,
                                            (*keypoints2)[i2].x, 
                                            (*keypoints2)[i2].y))
                ) {
                    dists(i1, i2) = -1.0;
                } else {
                    dists(i1, i2) = 
                        descriptors1_int.row(i1).dot(descriptors2_int.row(i2));
                    break;
                }
            }
        }
    }

  return dists;
}

// compute the distance of a single descriptor pair
int ComputeSiftDistance(
    const FeatureKeypoint* keypoint1, const FeatureKeypoint* keypoint2,
    const FeatureDescriptor& descriptor1,
    const FeatureDescriptor& descriptor2,
    const std::function<bool(float, float, float, float)>& guided_filter) {

    const Eigen::Matrix<int, 1, 128> descriptor1_int =
          descriptor1.cast<int>();
    const Eigen::Matrix<int, 1, 128> descriptor2_int =
          descriptor2.cast<int>();

    int dist;    
    if (guided_filter != nullptr &&
        guided_filter(keypoint1->x, keypoint1->y,
                      keypoint2->x, keypoint2->y)) {
        dist = 0;
    } 
    else {
        dist = descriptor1_int.dot(descriptor2_int);
    } 
    return dist;
}


size_t FindBestMatchesOneWay(const Eigen::MatrixXi& dists,
                             const float max_ratio, const float max_distance,
                             std::vector<int>* matches) {
  // SIFT descriptor vectors are normalized to length 512.
    const float kDistNorm = 1.0f / (128.0f * 128.0f);

    size_t num_matches = 0;
    matches->resize(dists.rows(), -1);

    for (Eigen::MatrixXi::Index i1 = 0; i1 < dists.rows(); ++i1) {
        int best_i2 = -1;
        int best_dist = 0;
        int second_best_dist = 0;
        for (Eigen::MatrixXi::Index i2 = 0; i2 < dists.cols(); ++i2) {
            const int dist = dists(i1, i2);
            if (dist < 0) {
                continue;
            }
            if (dist > best_dist) {
                best_i2 = i2;
                second_best_dist = best_dist;
                best_dist = dist;
            } 
            else if (dist > second_best_dist) {
                second_best_dist = dist;
            }
        }
        
        // Check if any match found.
        if (best_i2 == -1) {
            continue;
        }

        const float best_dist_normed =
            std::acos(std::min(kDistNorm * best_dist, 1.0f));

        // Check if match distance passes threshold.
        if (best_dist_normed > max_distance) {
            continue;
        }
        
        const float second_best_dist_normed =
            std::acos(std::min(kDistNorm * second_best_dist, 1.0f));

        // Check if match passes ratio test. Keep this comparison >= in order to
        // ensure that the case of best == second_best is detected.
        if (best_dist_normed >= max_ratio * second_best_dist_normed) {
            continue;
        }

        num_matches += 1;
        (*matches)[i1] = best_i2;
    }
    return num_matches;
}

void FindBestMatches(const Eigen::MatrixXi& dists, const float max_ratio,
                     const float max_distance, const bool cross_check,
                     FeatureMatches* matches) {
    matches->clear();

    std::vector<int> matches12;
    const size_t num_matches12 =
          FindBestMatchesOneWay(dists, max_ratio, max_distance, &matches12);

    if (cross_check) {
        std::vector<int> matches21;
        const size_t num_matches21 = FindBestMatchesOneWay(
            dists.transpose(), max_ratio, max_distance, &matches21);
        matches->reserve(std::min(num_matches12, num_matches21));
        for (size_t i1 = 0; i1 < matches12.size(); ++i1) {
            if (matches12[i1] != -1 && matches21[matches12[i1]] != -1 &&
                matches21[matches12[i1]] == static_cast<int>(i1)) {
                FeatureMatch match;
                match.point2D_idx1 = i1;
                match.point2D_idx2 = matches12[i1];
                matches->push_back(match);
            }
        }
    } 
    else {
        matches->reserve(num_matches12);
        for (size_t i1 = 0; i1 < matches12.size(); ++i1) {
            if (matches12[i1] != -1) {
                FeatureMatch match;
                match.point2D_idx1 = i1;
                match.point2D_idx2 = matches12[i1];
                matches->push_back(match);
            }
        }
    }
}

void PrintElapsedTime(const Timer& timer) {
  std::cout << StringPrintf(" in %.3fs", timer.ElapsedSeconds()) << std::endl;
}

void MatchWarpSiftFeaturesCPU(const PairMatchingOptions& match_options,
						      const FeatureKeypoints& keypoints1,
						      const FeatureKeypoints& keypoints2,
                              const FeatureDescriptors& descriptors1,
                              const FeatureDescriptors& descriptors2,
                              TwoViewGeometry* two_view_geometry) {

    const int search_range = FEATURE_MATCH_SEARCH_RANGE;
    const float max_residual = 
        match_options.max_error * match_options.max_error;

    // const Eigen::Matrix3f H = two_view_geometry->H.cast<float>();
    const Eigen::Matrix3f F = two_view_geometry->F.cast<float>();
    std::vector<Eigen::Matrix3f> H_list;
    for (const auto & H : two_view_geometry->H_list) {
        H_list.emplace_back(H.cast<float>());
    }

    std::function<bool(float, float, float, float)> guided_filter_F;
    guided_filter_F = [&](const float x1, const float y1, 
                          const float x2, const float y2) {
        const Eigen::Vector3f p1(x1, y1, 1.0f);
        const Eigen::Vector3f p2(x2, y2, 1.0f);
        const Eigen::Vector3f Fx1 = F * p1;
        const Eigen::Vector3f Ftx2 = F.transpose() * p2;
        const float x2tFx1 = p2.transpose() * Fx1;
        return x2tFx1 * x2tFx1 /
                (Fx1(0) * Fx1(0) + Fx1(1) * Fx1(1) + Ftx2(0) * Ftx2(0) +
                Ftx2(1) * Ftx2(1)) >
                max_residual;
    };

    std::vector<std::function<bool(float, float, float, float)> > guided_filter_H_list;
    for (const auto & H : H_list) {
        std::function<bool(float, float, float, float)> guided_filter_H;
        guided_filter_H = [&](const float x1, const float y1, 
                            const float x2, const float y2) {
            const Eigen::Vector3f p1(x1, y1, 1.0f);
            const Eigen::Vector2f p2(x2, y2);
            return ((H * p1).hnormalized() - p2).squaredNorm() > 
                    max_residual;
        };
        guided_filter_H_list.emplace_back(guided_filter_H);
    }


    CHECK(guided_filter_F);
    // CHECK(guided_filter_H);

    const Eigen::MatrixXi dists = ComputeSiftDistanceMatrix(
        &keypoints1, &keypoints2, descriptors1, descriptors2, 
        H_list, search_range,
        guided_filter_F, guided_filter_H_list);

    FindBestMatches(dists, match_options.max_ratio, match_options.max_distance,
                    match_options.cross_check,
                    &two_view_geometry->inlier_matches);
}

size_t FindBestMatchesOneWayFLANN(
    const Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&
        indices,
    const Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&
        distances,
    const float max_ratio,
    const float max_distance,
    std::vector<int>* matches) {
  // SIFT descriptor vectors are normalized to length 512.
  const float kDistNorm = 1.0f / (512.0f * 512.0f);

  size_t num_matches = 0;
  matches->resize(indices.rows(), -1);

  for (int d1_idx = 0; d1_idx < indices.rows(); ++d1_idx) {
    int best_i2 = -1;
    int best_dist = 0;
    int second_best_dist = 0;
    for (int n_idx = 0; n_idx < indices.cols(); ++n_idx) {
      const int d2_idx = indices(d1_idx, n_idx);
      const int dist = distances(d1_idx, n_idx);
      if (dist > best_dist) {
        best_i2 = d2_idx;
        second_best_dist = best_dist;
        best_dist = dist;
      } else if (dist > second_best_dist) {
        second_best_dist = dist;
      }
    }

    // Check if any match found.
    if (best_i2 == -1) {
      continue;
    }

    const float best_dist_normed =
        std::acos(std::min(kDistNorm * best_dist, 1.0f));

    // Check if match distance passes threshold.
    if (best_dist_normed > max_distance) {
      continue;
    }

    const float second_best_dist_normed =
        std::acos(std::min(kDistNorm * second_best_dist, 1.0f));

    // Check if match passes ratio test. Keep this comparison >= in order to
    // ensure that the case of best == second_best is detected.
    if (best_dist_normed >= max_ratio * second_best_dist_normed) {
      continue;
    }

    num_matches += 1;
    (*matches)[d1_idx] = best_i2;
  }

  return num_matches;
}

void FindNearestNeighborsFLANN(
    const FeatureDescriptors& query,
    const FeatureDescriptors& database,
    Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>*
        indices,
    Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>*
        distances) {
  if (query.rows() == 0 || database.rows() == 0) {
    return;
  }

  const size_t kNumNearestNeighbors = 2;
  const size_t kNumTreesInForest = 4;

  const size_t num_nearest_neighbors =
      std::min(kNumNearestNeighbors, static_cast<size_t>(database.rows()));

  indices->resize(query.rows(), num_nearest_neighbors);
  distances->resize(query.rows(), num_nearest_neighbors);
  const flann::Matrix<uint8_t> query_matrix(
      const_cast<uint8_t*>(query.data()), query.rows(), 128);
  const flann::Matrix<uint8_t> database_matrix(
      const_cast<uint8_t*>(database.data()), database.rows(), 128);

  flann::Matrix<int> indices_matrix(
      indices->data(), query.rows(), num_nearest_neighbors);
  std::vector<float> distances_vector(query.rows() * num_nearest_neighbors);
  flann::Matrix<float> distances_matrix(
      distances_vector.data(), query.rows(), num_nearest_neighbors);
  flann::Index<flann::L2<uint8_t>> index(
      database_matrix, flann::KDTreeIndexParams(kNumTreesInForest));
  index.buildIndex();
  index.knnSearch(query_matrix,
                  indices_matrix,
                  distances_matrix,
                  num_nearest_neighbors,
                  flann::SearchParams(128));

  for (Eigen::Index query_index = 0; query_index < indices->rows();
       ++query_index) {
    for (Eigen::Index k = 0; k < indices->cols(); ++k) {
      const Eigen::Index database_index = indices->coeff(query_index, k);
      distances->coeffRef(query_index, k) =
          query.row(query_index)
              .cast<int>()
              .dot(database.row(database_index).cast<int>());
    }
  }
}

void FindBestMatchesFLANN(
    const Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&
        indices_1to2,
    const Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&
        distances_1to2,
    const Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&
        indices_2to1,
    const Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&
        distances_2to1,
    const float max_ratio,
    const float max_distance,
    const bool cross_check,
    FeatureMatches* matches) {
  matches->clear();

  std::vector<int> matches12;
  const size_t num_matches12 = FindBestMatchesOneWayFLANN(
      indices_1to2, distances_1to2, max_ratio, max_distance, &matches12);

  if (cross_check && indices_2to1.rows()) {
    std::vector<int> matches21;
    const size_t num_matches21 = FindBestMatchesOneWayFLANN(
        indices_2to1, distances_2to1, max_ratio, max_distance, &matches21);
    matches->reserve(std::min(num_matches12, num_matches21));
    for (size_t i1 = 0; i1 < matches12.size(); ++i1) {
      if (matches12[i1] != -1 && matches21[matches12[i1]] != -1 &&
          matches21[matches12[i1]] == static_cast<int>(i1)) {
        FeatureMatch match;
        match.point2D_idx1 = i1;
        match.point2D_idx2 = matches12[i1];
        matches->push_back(match);
      }
    }
  } else {
    matches->reserve(num_matches12);
    for (size_t i1 = 0; i1 < matches12.size(); ++i1) {
      if (matches12[i1] != -1) {
        FeatureMatch match;
        match.point2D_idx1 = i1;
        match.point2D_idx2 = matches12[i1];
        matches->push_back(match);
      }
    }
  }
}

void MatchSiftFeaturesCPUFLANN(const PairMatchingOptions& match_options,
                               const FeatureDescriptors& descriptors1,
                               const FeatureDescriptors& descriptors2,
                               FeatureMatches* matches) {
  CHECK(match_options.Check());
  CHECK_NOTNULL(matches);

  Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      indices_1to2;
  Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      distances_1to2;
  Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      indices_2to1;
  Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      distances_2to1;

  FindNearestNeighborsFLANN(
      descriptors1, descriptors2, &indices_1to2, &distances_1to2);
  if (match_options.cross_check) {
    FindNearestNeighborsFLANN(
        descriptors2, descriptors1, &indices_2to1, &distances_2to1);
  }

  FindBestMatchesFLANN(indices_1to2,
                       distances_1to2,
                       indices_2to1,
                       distances_2to1,
                       match_options.max_ratio,
                       match_options.max_distance,
                       match_options.cross_check,
                       matches);
}

void MatchSiftFeaturesCPU(const PairMatchingOptions& match_options,
                          const FeatureDescriptors& descriptors1,
                          const FeatureDescriptors& descriptors2,
                          FeatureMatches* matches) {
    CHECK(match_options.Check());
    CHECK_NOTNULL(matches);
    Timer timer;
    timer.Start();
    //std::cout << StringPrintf("Matching image ")<< std::flush;
    
    // const Eigen::MatrixXi dists = ComputeSiftDistanceMatrix(
    //      nullptr, nullptr, descriptors1, descriptors2, nullptr);
    // FindBestMatches(dists, match_options.max_ratio, match_options.max_distance,
    //                 match_options.cross_check, matches);

    MatchSiftFeaturesCPUFLANN(match_options, descriptors1, descriptors2, matches);

    //PrintElapsedTime(timer);                        

}

void MatchGuidedSiftFeaturesCPU(const PairMatchingOptions& match_options,
                                const FeatureKeypoints& keypoints1,
                                const FeatureKeypoints& keypoints2,
                                const FeatureDescriptors& descriptors1,
                                const FeatureDescriptors& descriptors2,
                                TwoViewGeometry* two_view_geometry) {
    CHECK(match_options.Check());
    CHECK_NOTNULL(two_view_geometry);

    const float max_residual = 
        match_options.max_error * match_options.max_error;

    const Eigen::Matrix3f F = two_view_geometry->F.cast<float>();
    const Eigen::Matrix3f H = two_view_geometry->H.cast<float>();

    std::function<bool(float, float, float, float)> guided_filter;
    if (two_view_geometry->config == TwoViewGeometry::CALIBRATED ||
        two_view_geometry->config == TwoViewGeometry::UNCALIBRATED) {
        guided_filter = [&](const float x1, const float y1, const float x2,
                        const float y2) {
            const Eigen::Vector3f p1(x1, y1, 1.0f);
            const Eigen::Vector3f p2(x2, y2, 1.0f);
            const Eigen::Vector3f Fx1 = F * p1;
            const Eigen::Vector3f Ftx2 = F.transpose() * p2;
            const float x2tFx1 = p2.transpose() * Fx1;
            return x2tFx1 * x2tFx1 /
                 (Fx1(0) * Fx1(0) + Fx1(1) * Fx1(1) + Ftx2(0) * Ftx2(0) +
                  Ftx2(1) * Ftx2(1)) >
                  max_residual;
        };
    } 
    else if (two_view_geometry->config == TwoViewGeometry::PLANAR ||
             two_view_geometry->config == TwoViewGeometry::PANORAMIC ||
             two_view_geometry->config ==
                 TwoViewGeometry::PLANAR_OR_PANORAMIC) {
        guided_filter = [&](const float x1, const float y1, const float x2,
                        const float y2) {
            const Eigen::Vector3f p1(x1, y1, 1.0f);
            const Eigen::Vector2f p2(x2, y2);
            return ((H * p1).hnormalized() - p2).squaredNorm() > max_residual;
        };
    } 
    else {
        return;
    }

    CHECK(guided_filter);

    const Eigen::MatrixXi dists = ComputeSiftDistanceMatrix(
        &keypoints1, &keypoints2, descriptors1, descriptors2, guided_filter);

    FindBestMatches(dists, match_options.max_ratio, match_options.max_distance,
                    match_options.cross_check,
                    &two_view_geometry->inlier_matches);
}


//compute the distance matrix of the descriptors in two images
Eigen::MatrixXi ComputeBinaryDistanceMatrix(
    const BinaryFeatureDescriptors& descriptors1,
    const BinaryFeatureDescriptors& descriptors2) {
     
    Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> dists(
        descriptors1.rows(), descriptors2.rows());

    int max_distance = descriptors1.cols() * sizeof(uint64_t) * 8; 
    int section_num = descriptors1.cols();

    for (FeatureDescriptors::Index i1 = 0; i1 < descriptors1.rows(); ++i1) {
        
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif        
        for (FeatureDescriptors::Index i2 = 0; i2 < descriptors2.rows(); ++i2) {
                
            int distance = 0;
            for(int col = 0; col < section_num; ++col){
                uint64_t xor_temp = descriptors1(i1,col) ^ descriptors2(i2,col);
                distance += _mm_popcnt_u64(xor_temp);
            }
            dists(i1, i2) = max_distance - distance;
        }
    }

  return dists;
}


size_t FindBestMatchesOneWayBinary(const Eigen::MatrixXi& dists,
                             const float max_ratio, const float max_distance,
                             std::vector<int>* matches,
                             float full_distance = 256.0) {
     
    size_t num_matches = 0;
    matches->resize(dists.rows(), -1);

    for (Eigen::MatrixXi::Index i1 = 0; i1 < dists.rows(); ++i1) {
        int best_i2 = -1;
        int best_dist = 0;
        int second_best_dist = 0;
        for (Eigen::MatrixXi::Index i2 = 0; i2 < dists.cols(); ++i2) {
            const int dist = dists(i1, i2);
            if (dist < 0) {
                continue;
            }
            if (dist > best_dist) {
                best_i2 = i2;
                second_best_dist = best_dist;
                best_dist = dist;
            } 
            else if (dist > second_best_dist) {
                second_best_dist = dist;
            }
        }

        // Check if any match found.
        if (best_i2 == -1) {
            continue;
        }

        const float best_dist_normed = full_distance - static_cast<float>(best_dist);
            
        // Check if match distance passes threshold.
        if (best_dist_normed > max_distance) {
            continue;
        }

        const float second_best_dist_normed = full_distance - static_cast<float>(second_best_dist);
        
        // Check if match passes ratio test. Keep this comparison >= in order to
        // ensure that the case of best == second_best is detected.
        if (best_dist_normed >= max_ratio * second_best_dist_normed) {
            continue;
        }

        num_matches += 1;
        (*matches)[i1] = best_i2;
    }
    return num_matches;
}

void FindBestMatchesBinary(const Eigen::MatrixXi& dists, const float max_ratio,
                     const float max_distance, const bool cross_check,
                     FeatureMatches* matches,
                     float full_distance = 256.0) {
    matches->clear();

    std::vector<int> matches12;
    const size_t num_matches12 =
          FindBestMatchesOneWayBinary(dists, max_ratio, max_distance, &matches12, full_distance);

    if (cross_check) {
        std::vector<int> matches21;
        const size_t num_matches21 = FindBestMatchesOneWayBinary(
            dists.transpose(), max_ratio, max_distance, &matches21, full_distance);
        matches->reserve(std::min(num_matches12, num_matches21));
        for (size_t i1 = 0; i1 < matches12.size(); ++i1) {
            if (matches12[i1] != -1 && matches21[matches12[i1]] != -1 &&
                matches21[matches12[i1]] == static_cast<int>(i1)) {
                FeatureMatch match;
                match.point2D_idx1 = i1;
                match.point2D_idx2 = matches12[i1];
                matches->push_back(match);
            }
        }
    } 
    else {
        matches->reserve(num_matches12);
        for (size_t i1 = 0; i1 < matches12.size(); ++i1) {
            if (matches12[i1] != -1) {
                FeatureMatch match;
                match.point2D_idx1 = i1;
                match.point2D_idx2 = matches12[i1];
                matches->push_back(match);
            }
        }
    }
}


void MatchBinaryFeaturesCPU(const PairMatchingOptions& match_options,
                          const BinaryFeatureDescriptors& descriptors1,
                          const BinaryFeatureDescriptors& descriptors2,
                          FeatureMatches* matches) {

    CHECK(match_options.Check());
    CHECK_NOTNULL(matches);
    Timer timer;
    timer.Start();

    const Eigen::MatrixXi dists = ComputeBinaryDistanceMatrix(descriptors1, descriptors2);
    float full_distance = descriptors1.cols() * sizeof(uint64_t)* 8;
    FindBestMatchesBinary(dists, match_options.max_ratio, match_options.max_distance,
                    match_options.cross_check, matches,full_distance);

}



#ifdef CUDA_ENABLED
//==============================================================================
/*GPU Feature Matching*/
//==============================================================================
static std::map<int, std::unique_ptr<std::mutex>> sift_matching_mutexes;

void WarnIfMaxNumMatchesReachedGPU(const SiftMatchGPU& sift_match_gpu,
                                   const CompressedFeatureDescriptors& descriptors) {
    if (sift_match_gpu.GetMaxSift() < descriptors.rows()) {
        std::cout << StringPrintf(
                     "WARNING: Clamping features from %d to %d - consider "
                     "increasing the maximum number of matches.",
                     descriptors.rows(), sift_match_gpu.GetMaxSift())
                << std::endl;
  }
}
void WarnIfMaxNumMatchesReachedGPU(const SiftMatchGPU& sift_match_gpu,
                                   const int num_descriptor) {
    if (sift_match_gpu.GetMaxSift() < num_descriptor) {
        std::cout << StringPrintf(
                     "WARNING: Clamping features from %d to %d - consider "
                     "increasing the maximum number of matches.",
                     num_descriptor, sift_match_gpu.GetMaxSift())
                << std::endl;
  }
}

bool CreateSiftGPUMatcher(const PairMatchingOptions& match_options,
                          SiftMatchGPU* sift_match_gpu) {
    CHECK(match_options.Check());
    CHECK_NOTNULL(sift_match_gpu);

    // SiftGPU uses many global static state variables and the initialization 
    // must be thread-safe in order to work correctly. This is enforced here.
    static std::mutex mutex;
    std::unique_lock<std::mutex> lock(mutex);

    const std::vector<int> gpu_indices =
      CSVToVector<int>(match_options.gpu_index);
    CHECK_EQ(gpu_indices.size(), 1) << "SiftGPU can only run on one GPU";

    SiftGPU sift_gpu;
    sift_gpu.SetVerbose(1);

    *sift_match_gpu = SiftMatchGPU(match_options.max_num_matches);

    if (gpu_indices[0] >= 0) {
        sift_match_gpu->SetLanguage(SiftMatchGPU::SIFTMATCH_CUDA_DEVICE0 +
                                gpu_indices[0]);
    } 
    else {
        sift_match_gpu->SetLanguage(SiftMatchGPU::SIFTMATCH_CUDA);
    }

    if (sift_match_gpu->VerifyContextGL() == 0) {
        return false;
    }

    if (!sift_match_gpu->Allocate(match_options.max_num_matches,
                                match_options.cross_check)) {
        std::cout << StringPrintf(
                     "ERROR: Not enough GPU memory to match %d features. "
                     "Reduce the maximum number of matches.",
                     match_options.max_num_matches)
              << std::endl;
        return false;
    }
    
    sift_match_gpu->gpu_index = gpu_indices[0];
    if (sift_matching_mutexes.count(gpu_indices[0]) == 0) {
        sift_matching_mutexes.emplace(
            gpu_indices[0], std::unique_ptr<std::mutex>(new std::mutex()));
    }

    return true;
}

void MatchWarpSiftFeaturesGPU(const PairMatchingOptions& match_options,
							  const FeatureKeypoints* keypoints1,
							  const FeatureKeypoints* keypoints2,
							  const CompressedFeatureDescriptors* descriptors1,
							  const CompressedFeatureDescriptors* descriptors2,
							  SiftMatchGPU* sift_match_gpu,
							  TwoViewGeometry* two_view_geometry) {
    // static_assert(offsetof(FeatureKeypoint, x) == 0 * sizeof(float),
    //         "Invalid keypoint format");
    // static_assert(offsetof(FeatureKeypoint, y) == 1 * sizeof(float),
    //             "Invalid keypoint format");
    // static_assert(sizeof(FeatureKeypoint) == 6 * sizeof(float),
    //             "Invalid keypoint format");

    // static_assert(sizeof(float)==sizeof(int)&&sizeof(float)==sizeof(point2D_t),
    //               "float and int has different size");

    // CHECK(match_options.Check());
    // CHECK_NOTNULL(sift_match_gpu);
    // CHECK_NOTNULL(two_view_geometry);

    // std::unique_lock<std::mutex> lock(
    //     *sift_matching_mutexes[sift_match_gpu->gpu_index]);

    // const size_t kFeatureShapeNumElems = 4;

    // if (descriptors1 != nullptr) {
    //     CHECK_NOTNULL(keypoints1);
    //     CHECK_EQ(descriptors1->rows(), keypoints1->size());
    //     CHECK_EQ(descriptors1->cols(), 128);
    //     WarnIfMaxNumMatchesReachedGPU(*sift_match_gpu, *descriptors1);
    //     const size_t kIndex = 0;
    //     sift_match_gpu->SetDescriptors(kIndex, descriptors1->rows(),
    //                                 descriptors1->data());
    //     sift_match_gpu->SetFeautreLocation(
    //         kIndex, reinterpret_cast<const float*>(keypoints1->data()),
    //         kFeatureShapeNumElems);
    // }

    // if (descriptors2 != nullptr) {
    //     CHECK_NOTNULL(keypoints2);
    //     CHECK_EQ(descriptors2->rows(), keypoints2->size());
    //     CHECK_EQ(descriptors2->cols(), 128);
    //     WarnIfMaxNumMatchesReachedGPU(*sift_match_gpu, *descriptors2);
    //     const size_t kIndex = 1;
    //     sift_match_gpu->SetDescriptors(kIndex, descriptors2->rows(),
    //                                 descriptors2->data());
    //     sift_match_gpu->SetFeautreLocation(
    //         kIndex, reinterpret_cast<const float*>(keypoints2->data()),
    //         kFeatureShapeNumElems);
    // }

    // Eigen::Matrix<float, 3, 3, Eigen::RowMajor> F;
    // Eigen::Matrix<float, 3, 3, Eigen::RowMajor> H;
    // float* F_ptr = nullptr;
    // float* Hs_ptr = nullptr;

    // F = two_view_geometry->F.cast<float>();
    // F_ptr = F.data();
    
    // size_t lengthHomo = two_view_geometry->H_list.size();
    // Hs_ptr = new float[lengthHomo * 9];
    // for (size_t k = 0; k < lengthHomo; ++k) {
    //     H = two_view_geometry->H_list[k].cast<float>();
    //     memcpy(Hs_ptr + k * 9, H.data(), 9 * sizeof(float));
    // }

    // CHECK(F_ptr != nullptr || Hs_ptr != nullptr);

    // two_view_geometry->inlier_matches.resize(
    //     static_cast<size_t>(match_options.max_num_matches));

    // const int num_matches = sift_match_gpu->GetGuidedSiftMatch(
    //     match_options.max_num_matches,
    //     reinterpret_cast<uint32_t(*)[2]>(
    //         two_view_geometry->inlier_matches.data()),
    //     Hs_ptr, F_ptr, NULL, lengthHomo, static_cast<float>(match_options.max_distance),
    //     static_cast<float>(match_options.max_ratio),
    //     static_cast<float>(match_options.max_error * match_options.max_error),
    //     static_cast<float>(match_options.max_error * match_options.max_error),
    //     match_options.cross_check);

    // if (num_matches < 0) {
    //     std::cerr << "ERROR: Feature matching failed. This is probably caused "
    //              "by insufficient GPU memory. Consider reducing the maximum "
    //              "number of features."
    //           << std::endl;
    //     two_view_geometry->inlier_matches.clear();
    // } 
    // else {
    //     CHECK_LE(num_matches, two_view_geometry->inlier_matches.size());
    //     two_view_geometry->inlier_matches.resize(num_matches);
    // }
}

void MatchSiftFeaturesGPU(const PairMatchingOptions& match_options,
                          const CompressedFeatureDescriptors* descriptors1,
                          const CompressedFeatureDescriptors* descriptors2,
                          SiftMatchGPU* sift_match_gpu,
                          FeatureMatches* matches) {
    CHECK(match_options.Check());
    CHECK_NOTNULL(sift_match_gpu);
    CHECK_NOTNULL(matches);

    Timer timer;
    timer.Start();
    //std::cout << StringPrintf("Matching image ")<< std::flush;

    std::unique_lock<std::mutex> lock(
        *sift_matching_mutexes[sift_match_gpu->gpu_index]);

    if (descriptors1 != nullptr) {
        CHECK(descriptors1->cols() == 32 || descriptors1->cols() == 64 || descriptors1->cols() == 128);
        WarnIfMaxNumMatchesReachedGPU(*sift_match_gpu, *descriptors1);
        sift_match_gpu->SetDescriptors(0, descriptors1->rows(),
                                    (unsigned char*)(descriptors1->data()),-1, descriptors1->cols());
    }

    if (descriptors2 != nullptr) {
        CHECK(descriptors2->cols() == 32 || descriptors2->cols() == 64 || descriptors2->cols() == 128);
        WarnIfMaxNumMatchesReachedGPU(*sift_match_gpu, *descriptors2);
        sift_match_gpu->SetDescriptors(1, descriptors2->rows(),
                                       (unsigned char*)(descriptors2->data()),-1, descriptors2->cols());
    }
    
    CHECK(descriptors1 != nullptr || descriptors2 != nullptr);
    int feature_dim = (descriptors1 != nullptr) ? descriptors1->cols(): descriptors2->cols();

    matches->resize(static_cast<size_t>(match_options.max_num_matches));

    const int num_matches = sift_match_gpu->GetSiftMatch(
        match_options.max_num_matches,
        reinterpret_cast<uint32_t(*)[2]>(matches->data()),
        static_cast<float>(match_options.max_distance),
        static_cast<float>(match_options.max_ratio), match_options.cross_check, feature_dim);

    if (num_matches < 0) {
        std::cerr << "ERROR: Feature matching failed. This is probably caused  "
                 "by insufficient GPU memory. Consider reducing the maximum "
                 "number of features and/or matches."
              << std::endl;
        matches->clear();
    } 
    else {
        CHECK_LE(num_matches, matches->size());
        matches->resize(num_matches);
    }

    //PrintElapsedTime(timer); 
}

void MatchSiftFeaturesGPU(const PairMatchingOptions& match_options,
                          const CompressedFeatureDescriptors* descriptors1,
						  const int start_row1, const int num_descriptor1,
                          const CompressedFeatureDescriptors* descriptors2,
						  const int start_row2, const int num_descriptor2,
                          SiftMatchGPU* sift_match_gpu,
                          FeatureMatches* matches) {
    CHECK(match_options.Check());
    CHECK_NOTNULL(sift_match_gpu);
    CHECK_NOTNULL(matches);

    Timer timer;
    timer.Start();

    std::unique_lock<std::mutex> lock(
        *sift_matching_mutexes[sift_match_gpu->gpu_index]);

    if (descriptors1 != nullptr) {
        CHECK(descriptors1->cols() == 32 || descriptors1->cols() == 64 || descriptors1->cols() == 128);
        WarnIfMaxNumMatchesReachedGPU(*sift_match_gpu, num_descriptor1);
        sift_match_gpu->SetDescriptors(0, num_descriptor1,
                                       (unsigned char*)(descriptors1->data()) + start_row1 * descriptors1->cols(), -1,
                                       descriptors1->cols());
    }

    if (descriptors2 != nullptr) {
        CHECK(descriptors2->cols() == 32 || descriptors2->cols() == 64 || descriptors2->cols() == 128);
        WarnIfMaxNumMatchesReachedGPU(*sift_match_gpu, num_descriptor2);
        sift_match_gpu->SetDescriptors(1, num_descriptor2,
                                       (unsigned char*)(descriptors2->data()) + start_row2 * descriptors2->cols(), -1,
                                       descriptors2->cols());
    }

    CHECK(descriptors1 != nullptr || descriptors2 != nullptr);
    int feature_dim = (descriptors1 != nullptr) ? descriptors1->cols(): descriptors2->cols();

    matches->resize(static_cast<size_t>(match_options.max_num_matches));

    const int num_matches = sift_match_gpu->GetSiftMatch(
        match_options.max_num_matches,
        reinterpret_cast<uint32_t(*)[2]>(matches->data()),
        static_cast<float>(match_options.max_distance),
        static_cast<float>(match_options.max_ratio), match_options.cross_check,feature_dim);

    if (num_matches < 0) {
        std::cerr << "ERROR: Feature matching failed. This is probably caused  "
                 "by insufficient GPU memory. Consider reducing the maximum "
                 "number of features and/or matches."
              << std::endl;
        matches->clear();
    } 
    else {
        CHECK_LE(num_matches, matches->size());
        matches->resize(num_matches);
    }
}

void MatchGuidedSiftFeaturesGPU(const PairMatchingOptions& match_options,
                                const FeatureKeypoints* keypoints1,
                                const FeatureKeypoints* keypoints2,
                                const CompressedFeatureDescriptors* descriptors1,
                                const CompressedFeatureDescriptors* descriptors2,
                                SiftMatchGPU* sift_match_gpu,
                                TwoViewGeometry* two_view_geometry) {
    Timer timer;
    timer.Start();
    //std::cout << StringPrintf("guided matching image ")<< std::flush;                                

    static_assert(offsetof(FeatureKeypoint, x) == 0 * sizeof(float),
                "Invalid keypoint format");
    static_assert(offsetof(FeatureKeypoint, y) == 1 * sizeof(float),
                "Invalid keypoint format");
    static_assert(sizeof(FeatureKeypoint) == 6 * sizeof(float),
                "Invalid keypoint format");
    
    static_assert(sizeof(float)==sizeof(int)&&sizeof(float)==sizeof(point2D_t),
                  "float and int has different size");

    CHECK(match_options.Check());
    CHECK_NOTNULL(sift_match_gpu);
    CHECK_NOTNULL(two_view_geometry);

    std::unique_lock<std::mutex> lock(
        *sift_matching_mutexes[sift_match_gpu->gpu_index]);

    const size_t kFeatureShapeNumElems = 4;

    if (descriptors1 != nullptr) {
        CHECK_NOTNULL(keypoints1);
        CHECK_EQ(descriptors1->rows(), keypoints1->size());
        CHECK(descriptors1->cols() == 32 || descriptors1->cols() == 64 || descriptors1->cols() == 128);
        WarnIfMaxNumMatchesReachedGPU(*sift_match_gpu, *descriptors1);
        const size_t kIndex = 0;
        sift_match_gpu->SetDescriptors(kIndex, descriptors1->rows(), (unsigned char*)descriptors1->data(), -1,
                                       descriptors1->cols());
        sift_match_gpu->SetFeautreLocation(kIndex, reinterpret_cast<const float*>(keypoints1->data()),
                                           kFeatureShapeNumElems);
    }

    if (descriptors2 != nullptr) {
        CHECK_NOTNULL(keypoints2);
        CHECK_EQ(descriptors2->rows(), keypoints2->size());
        CHECK(descriptors2->cols() == 32 || descriptors2->cols() == 64 || descriptors2->cols() == 128);
        WarnIfMaxNumMatchesReachedGPU(*sift_match_gpu, *descriptors2);
        const size_t kIndex = 1;
        sift_match_gpu->SetDescriptors(kIndex, descriptors2->rows(), (unsigned char*)descriptors2->data(), -1,
                                       descriptors2->cols());
        sift_match_gpu->SetFeautreLocation(kIndex, reinterpret_cast<const float*>(keypoints2->data()),
                                           kFeatureShapeNumElems);
    }

    CHECK(descriptors1 != nullptr || descriptors2 != nullptr);
    int feature_dim = (descriptors1 != nullptr) ? descriptors1->cols(): descriptors2->cols();

    Eigen::Matrix<float, 3, 3, Eigen::RowMajor> F;
    Eigen::Matrix<float, 3, 3, Eigen::RowMajor> H;
    float* F_ptr = nullptr;
    float* H_ptr = nullptr;
    if (two_view_geometry->config == TwoViewGeometry::CALIBRATED ||
         two_view_geometry->config == TwoViewGeometry::UNCALIBRATED) {
        F = two_view_geometry->F.cast<float>();
        F_ptr = F.data();
    } 
    else if (two_view_geometry->config == TwoViewGeometry::PLANAR ||
             two_view_geometry->config == TwoViewGeometry::PANORAMIC ||
             two_view_geometry->config ==
                 TwoViewGeometry::PLANAR_OR_PANORAMIC) {
        H = two_view_geometry->H.cast<float>();
        H_ptr = H.data();
    } 
    else {
        return;
    }

    CHECK(F_ptr != nullptr || H_ptr != nullptr);

    two_view_geometry->inlier_matches.resize(
        static_cast<size_t>(match_options.max_num_matches));

    const int num_matches = sift_match_gpu->GetGuidedSiftMatch(
        match_options.max_num_matches,
        reinterpret_cast<uint32_t(*)[2]>(
            two_view_geometry->inlier_matches.data()),
        H_ptr, F_ptr, NULL, 1, static_cast<float>(match_options.max_distance),
        static_cast<float>(match_options.max_ratio),
        static_cast<float>(match_options.max_error * match_options.max_error),
        static_cast<float>(match_options.max_error * match_options.max_error),
        match_options.cross_check, feature_dim);

    if (num_matches < 0) {
        std::cerr << "ERROR: Feature matching failed. This is probably caused "
                 "by insufficient GPU memory. Consider reducing the maximum "
                 "number of features."
              << std::endl;
        two_view_geometry->inlier_matches.clear();
    } 
    else {
        CHECK_LE(num_matches, two_view_geometry->inlier_matches.size());
        two_view_geometry->inlier_matches.resize(num_matches);
    }

    //PrintElapsedTime(timer); 
}


void MatchGuidedSiftFeaturesGPUSpherical(
                        const Camera& camera1,
                        const Camera& camera2,
                        const PairMatchingOptions& match_options,
                        const FeatureKeypoints* keypoints1,
                        const FeatureKeypoints* keypoints2,
                        const CompressedFeatureDescriptors* descriptors1,
                        const CompressedFeatureDescriptors* descriptors2,
                        SiftMatchGPU* sift_match_gpu,
                        TwoViewGeometry* two_view_geometry) {

    static_assert(offsetof(FeatureKeypoint, x) == 0 * sizeof(float), "Invalid keypoint format");
    static_assert(offsetof(FeatureKeypoint, y) == 1 * sizeof(float), "Invalid keypoint format");
    static_assert(sizeof(FeatureKeypoint) == 6 * sizeof(float), "Invalid keypoint format");

    static_assert(sizeof(float) == sizeof(int) && sizeof(float) == sizeof(point2D_t),
                  "float and int has different size");

    CHECK(match_options.Check());
    CHECK_NOTNULL(sift_match_gpu);
    CHECK_NOTNULL(two_view_geometry);

    std::unique_lock<std::mutex> lock(*sift_matching_mutexes[sift_match_gpu->gpu_index]);

    const size_t kFeatureShapeNumElems = 4;

    if (descriptors1 != nullptr) {
        CHECK_NOTNULL(keypoints1);
        CHECK_EQ(descriptors1->rows(), keypoints1->size());
        CHECK(descriptors1->cols() == 32 || descriptors1->cols() == 64 || descriptors1->cols() == 128);
        WarnIfMaxNumMatchesReachedGPU(*sift_match_gpu, *descriptors1);
        FeatureKeypoints keypoints1_perspective = *keypoints1;

        for (size_t i = 0; i < keypoints1_perspective.size(); ++i) {
            Eigen::Vector2d kp_spherical(keypoints1_perspective[i].x, keypoints1_perspective[i].y);
            Eigen::Vector2d kp_perspective = camera1.ImageToWorld(kp_spherical);
            keypoints1_perspective[i].x = kp_perspective(0);
            keypoints1_perspective[i].y = kp_perspective(1);
        }

        const size_t kIndex = 0;
        sift_match_gpu->SetDescriptors(kIndex, descriptors1->rows(), (unsigned char*)descriptors1->data(), -1,
                                       descriptors1->cols());
        sift_match_gpu->SetFeautreLocation(kIndex, reinterpret_cast<const float*>(keypoints1_perspective.data()),
                                           kFeatureShapeNumElems);
    }

    if (descriptors2 != nullptr) {
        CHECK_NOTNULL(keypoints2);
        CHECK_EQ(descriptors2->rows(), keypoints2->size());
        CHECK(descriptors2->cols() == 32 || descriptors2->cols() == 64 || descriptors2->cols() == 128);
        WarnIfMaxNumMatchesReachedGPU(*sift_match_gpu, *descriptors2);
        FeatureKeypoints keypoints2_perspective = *keypoints2;

        for (size_t i = 0; i < keypoints2_perspective.size(); ++i) {
            Eigen::Vector2d kp_spherical(keypoints2_perspective[i].x, keypoints2_perspective[i].y);
            Eigen::Vector2d kp_perspective = camera2.ImageToWorld(kp_spherical);
            keypoints2_perspective[i].x = kp_perspective(0);
            keypoints2_perspective[i].y = kp_perspective(1);
        }

        const size_t kIndex = 1;
        sift_match_gpu->SetDescriptors(kIndex, descriptors2->rows(), (unsigned char*)descriptors2->data(), -1,
                                       descriptors2->cols());
        sift_match_gpu->SetFeautreLocation(kIndex, reinterpret_cast<const float*>(keypoints2_perspective.data()),
                                           kFeatureShapeNumElems);
    }

    CHECK(descriptors1 != nullptr || descriptors2 != nullptr);
    int feature_dim = (descriptors1 != nullptr) ? descriptors1->cols(): descriptors2->cols();

    Eigen::Matrix<float, 3, 3, Eigen::RowMajor> F, E;
    Eigen::Matrix<float, 3, 3, Eigen::RowMajor> H;
    float* F_ptr = nullptr;
    float* E_ptr = nullptr;
    float* H_ptr = nullptr;
    if (two_view_geometry->config == TwoViewGeometry::CALIBRATED ||
        two_view_geometry->config == TwoViewGeometry::UNCALIBRATED) {
        E = two_view_geometry->E.cast<float>();
        E_ptr = E.data();
    } else {
        return;
    }

    CHECK(F_ptr != nullptr || E_ptr != nullptr || H_ptr != nullptr);

    two_view_geometry->inlier_matches.resize(static_cast<size_t>(match_options.max_num_matches));

    const int num_matches = sift_match_gpu->GetGuidedSiftMatch(
        match_options.max_num_matches, reinterpret_cast<uint32_t(*)[2]>(two_view_geometry->inlier_matches.data()),
        H_ptr, NULL, E_ptr, 1, static_cast<float>(match_options.max_distance),
        static_cast<float>(match_options.guided_match_max_ratio),
        static_cast<float>(match_options.max_error * match_options.max_error),
        static_cast<float>(match_options.max_angular_error * match_options.max_angular_error),
        match_options.cross_check, feature_dim);

    if (num_matches < 0) {
        std::cerr << "ERROR: Feature matching failed. This is probably caused "
                     "by insufficient GPU memory. Consider reducing the maximum "
                     "number of features."
                  << std::endl;
        two_view_geometry->inlier_matches.clear();
    } else {
        CHECK_LE(num_matches, two_view_geometry->inlier_matches.size());
        two_view_geometry->inlier_matches.resize(num_matches);
    }
}

void MatchGuidedSiftFeaturesGPURig(
						const Camera& camera1,
						const Camera& camera2,
						const PairMatchingOptions& match_options,
						const FeatureKeypoints* keypoints1,
						const FeatureKeypoints* keypoints2,
						const CompressedFeatureDescriptors* descriptors1,
						const CompressedFeatureDescriptors* descriptors2,
						const std::vector<uint32_t>& local_camera_indices1,
						const std::vector<uint32_t>& local_camera_indices2,
						SiftMatchGPU* sift_match_gpu,
						TwoViewGeometry* two_view_geometry) {
    static_assert(offsetof(FeatureKeypoint, x) == 0 * sizeof(float),
                "Invalid keypoint format");
    static_assert(offsetof(FeatureKeypoint, y) == 1 * sizeof(float),
                "Invalid keypoint format");
    static_assert(sizeof(FeatureKeypoint) == 6 * sizeof(float),
                "Invalid keypoint format");

    static_assert(sizeof(float)==sizeof(int)&&sizeof(float)==sizeof(point2D_t),
                  "float and int has different size");

    CHECK(match_options.Check());
    CHECK_NOTNULL(sift_match_gpu);
    CHECK_NOTNULL(two_view_geometry);

    std::unique_lock<std::mutex> lock(
        *sift_matching_mutexes[sift_match_gpu->gpu_index]);

    const size_t kFeatureShapeNumElems = 4;

    if (descriptors1 != nullptr) {
        CHECK_NOTNULL(keypoints1);
        CHECK_EQ(descriptors1->rows(), keypoints1->size());
        CHECK(descriptors1->cols() == 32 || descriptors1->cols() == 64 || descriptors1->cols() == 128);
        WarnIfMaxNumMatchesReachedGPU(*sift_match_gpu, *descriptors1);
        FeatureKeypoints keypoints1_norm = *keypoints1;

        for(size_t i = 0; i < keypoints1_norm.size(); ++i){
            Eigen::Vector2d kp(keypoints1_norm[i].x, keypoints1_norm[i].y);
            Eigen::Vector2d kp_norm;
            if (camera1.NumLocalCameras() > 1) {
                const uint32_t local_idx = local_camera_indices1[i];
                kp_norm = camera1.LocalImageToWorld(local_idx, kp);
            } else {
                kp_norm = camera1.ImageToWorld(kp);
            }

            keypoints1_norm[i].x = kp_norm(0);
            keypoints1_norm[i].y = kp_norm(1);
        }

        const size_t kIndex = 0;
        sift_match_gpu->SetDescriptors(kIndex, descriptors1->rows(),
                                       (unsigned char*)descriptors1->data(), -1, descriptors1->cols());
        sift_match_gpu->SetFeautreLocation(
            kIndex, reinterpret_cast<const float*>(keypoints1_norm.data()),
            kFeatureShapeNumElems);
     }

    if (descriptors2 != nullptr) {
        CHECK_NOTNULL(keypoints2);
        CHECK_EQ(descriptors2->rows(), keypoints2->size());
        CHECK(descriptors2->cols() == 32 || descriptors2->cols() == 64 || descriptors2->cols() == 128);
        WarnIfMaxNumMatchesReachedGPU(*sift_match_gpu, *descriptors2);
        FeatureKeypoints keypoints2_norm = *keypoints2;

        for(size_t i = 0; i < keypoints2_norm.size(); ++i){
            Eigen::Vector2d kp(keypoints2_norm[i].x, keypoints2_norm[i].y);
            Eigen::Vector2d kp_norm;
            if (camera2.NumLocalCameras() > 1) {
                const uint32_t local_idx = local_camera_indices2[i];
                kp_norm = camera2.LocalImageToWorld(local_idx, kp);
            } else {
                kp_norm = camera2.ImageToWorld(kp);
            }
            keypoints2_norm[i].x = kp_norm(0);
            keypoints2_norm[i].y = kp_norm(1);
        }

        const size_t kIndex = 1;
        sift_match_gpu->SetDescriptors(kIndex, descriptors2->rows(),
                                       (unsigned char*)descriptors2->data(), -1, descriptors2->cols());
        sift_match_gpu->SetFeautreLocation(
            kIndex, reinterpret_cast<const float*>(keypoints2_norm.data()),
            kFeatureShapeNumElems);
    }

    CHECK(descriptors1 != nullptr || descriptors2 != nullptr);
    int feature_dim = (descriptors1 != nullptr) ? descriptors1->cols(): descriptors2->cols();

    Eigen::Matrix<float, 3, 3, Eigen::RowMajor> F, E;
    Eigen::Matrix<float, 3, 3, Eigen::RowMajor> H;
    float* F_ptr = nullptr;
    float* E_ptr = nullptr;
    float* H_ptr = nullptr;
    if (two_view_geometry->config == TwoViewGeometry::CALIBRATED ||
         two_view_geometry->config == TwoViewGeometry::UNCALIBRATED) {
        E = two_view_geometry->E.cast<float>();
        E_ptr = E.data();
    } 
    else {
        return;
    }

    CHECK(F_ptr != nullptr || E_ptr != nullptr || H_ptr != nullptr);

    two_view_geometry->inlier_matches.resize(
        static_cast<size_t>(match_options.max_num_matches));

    const int num_matches = sift_match_gpu->GetGuidedSiftMatch(
        match_options.max_num_matches,
        reinterpret_cast<uint32_t(*)[2]>(
            two_view_geometry->inlier_matches.data()),
        H_ptr, NULL, E_ptr, 1, static_cast<float>(match_options.max_distance),
        static_cast<float>(match_options.max_ratio),
        static_cast<float>(match_options.max_error * match_options.max_error),
        static_cast<float>(match_options.max_angular_error * match_options.max_angular_error),
        match_options.cross_check, feature_dim);

    if (num_matches < 0) {
        std::cerr << "ERROR: Feature matching failed. This is probably caused "
                 "by insufficient GPU memory. Consider reducing the maximum "
                 "number of features."
              << std::endl;
        two_view_geometry->inlier_matches.clear();
    } 
    else {
        CHECK_LE(num_matches, two_view_geometry->inlier_matches.size());
        two_view_geometry->inlier_matches.resize(num_matches);
    }
}

#endif //CUDA_ENABLE

}//namespace sensemap