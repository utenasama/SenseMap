//Copyright (c) 2019, SenseTime Group.
//All rights reserved.
#ifndef SENSEMAP_MATCHING_UTILS_H_
#define SENSEMAP_MATCHING_UTILS_H_

#include <string>

#include "feature/sift.h"

#include "estimators/two_view_geometry.h"
#include "container/feature_data_container.h"
#include "retrieval/visual_index.h"
#include "container/match_data_container.h"



#ifdef CUDA_ENABLED
	class SiftMatchGPU;
#endif
namespace sensemap{

struct PairMatchingOptions {
	// Number of threads for feature matching and geometric verification.
	int num_threads = 1;

	// Whether to use the GPU for feature matching.
	bool use_gpu = true;

	// Index of the GPU used for feature matching. For multi-GPU matching,
	// you should separate multiple GPU indices by comma, e.g., "0,1,2,3".
	std::string gpu_index = "-1";

	// Maximum distance ratio between first and second best match.
	double max_ratio = 0.8;

	// Maximum distance ratio between first and second best match in 
	// guided match
	double guided_match_max_ratio = 0.8;

	// Maximum distance to best match.
	double max_distance = 0.7;

	// Whether to enable cross checking in matching.
	bool cross_check = true;

	// Maximum number of matches.
	int max_num_matches = 32768;

	// Maximum epipolar error in pixels for geometric verification.
	double max_error = 4.0;

	// Maximum angular error for geometric verification, compatible with 
	// spherical camera model. 0.4 in angular error is equivalent to 6.98
	// in pixel error when the focal length is 1000. 
	double max_angular_error = 0.40;


	// Confidence threshold for geometric verification.
	double confidence = 0.999;

	// Minimum/maximum number of RANSAC iterations. Note that this option
	// overrules the min_inlier_ratio option.
	int min_num_trials = 100;
	int max_num_trials = 10000;

	// A priori assumed minimum inlier ratio, which determines the maximum
	// number of iterations.
	double min_inlier_ratio = 0.25;

	// Minimum number of inliers for an image pair to be considered as
	// geometrically verified.
	int min_num_inliers = 15;

	// Whether to attempt to estimate multiple geometric models per image pair.
	bool multiple_models = false;

	// Whether to perform guided matching, if geometric verification succeeds.
	bool guided_matching = false;

	// Whether to perform subimage matching, it is valid only if camera-rig mode.
	bool sub_matching = false;

	// Whether to perform self matching, it is valid only if camera-rig mode.
	bool self_matching = false;

	bool is_sphere = false;

	// Sort the scene graph output
	bool sort_graph = true;
	// Whether to perform guided matching by plane homography.
	bool guided_matching_single_homography = false;

	// Whether to perform guided matching by multi-plane homography.
	bool guided_matching_multi_homography = false;


	// When the matched keypoints covers only a few sub perspective images of the
	// panorama, these matches are possibly caused by local repetitive patterns.
	double min_covered_sub_image_ratio = 0.5;
	int perspective_image_count = 8;
	bool convert_to_perspective_image = false;
	int min_matched_feature_per_piece = 5;

	double covered_sub_image_ratio_strong_loop = 0.8;
	int strong_loop_check_neighbor_count_src = 1;
	int strong_loop_check_neighbor_count_dst = 10;
	int strong_loop_transitivity = 1;
	int transitive_strong_loop_neighbor_count = 5;

	bool Check() const;
};

std::vector<image_t> GetOrderedImageIds(FeatureDataContainer* container, 
									    const std::vector<image_t> &image_ids);

void IndexImagesInVisualIndex(const int num_threads, const int num_checks,
							  const int max_num_features,
							  const std::vector<image_t>& image_ids,
							  FeatureDataContainer* container,
							  retrieval::VisualIndex<>* visual_index);


void FindNearestNeighborsInVisualIndex(
	const int num_threads, const int num_images, const int num_neighbors,
	const int num_checks, const int num_images_after_verification,
	const int max_num_features, const std::vector<image_t>& image_ids,
	FeatureDataContainer* container,
	retrieval::VisualIndex<>* visual_index, 
	std::vector<std::vector<image_t>>& neighbors,
	const bool hybrid_status = false,
	const float max_score_factor = 0.0f);

void FindNearestNeighborsInVladVisualIndex(const int num_images, const std::vector<image_t>& image_ids,
                                           FeatureDataContainer* container, VladVisualIndex* vlad_visual_index,
                                           std::vector<std::vector<image_t>>& neighbors,
                                           const bool hybrid_status = false, const float max_score_factor = 0.0f);

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
								bool piecewise = false);


void MatchWarpSiftFeaturesCPU(const PairMatchingOptions& match_options,
						  	  const FeatureKeypoints& keypoints1,
						  	  const FeatureKeypoints& keypoints2,
                          	  const FeatureDescriptors& descriptors1,
                          	  const FeatureDescriptors& descriptors2,
                          	  TwoViewGeometry* two_view_geometry);

void MatchSiftFeaturesCPU(const PairMatchingOptions& match_options,
                          const FeatureDescriptors& descriptors1,
                          const FeatureDescriptors& descriptors2,
                          FeatureMatches* matches);
void MatchGuidedSiftFeaturesCPU(const PairMatchingOptions& match_options,
                                const FeatureKeypoints& keypoints1,
                                const FeatureKeypoints& keypoints2,
                                const FeatureDescriptors& descriptors1,
                                const FeatureDescriptors& descriptors2,
                                TwoViewGeometry* two_view_geometry);

void MatchBinaryFeaturesCPU(const PairMatchingOptions& match_options,
                          const BinaryFeatureDescriptors& descriptors1,
                          const BinaryFeatureDescriptors& descriptors2,
                          FeatureMatches* matches);


#ifdef CUDA_ENABLED
bool CreateSiftGPUMatcher(const PairMatchingOptions& match_options,
                          SiftMatchGPU* sift_match_gpu);

void MatchWarpSiftFeaturesGPU(const PairMatchingOptions& match_options,
							  const FeatureKeypoints* keypoints1,
							  const FeatureKeypoints* keypoints2,
							  const CompressedFeatureDescriptors* descriptors1,
							  const CompressedFeatureDescriptors* descriptors2,
							  SiftMatchGPU* sift_match_gpu,
							  TwoViewGeometry* two_view_geometry);

// Match the given SIFT features on the GPU.
void MatchSiftFeaturesGPU(const PairMatchingOptions& match_options,
                          const CompressedFeatureDescriptors* descriptors1,
                          const CompressedFeatureDescriptors* descriptors2,
                          SiftMatchGPU* sift_match_gpu,
                          FeatureMatches* matches);
void MatchSiftFeaturesGPU(const PairMatchingOptions& match_options,
                          const CompressedFeatureDescriptors* descriptors1,
						  const int start_row1, const int num_descriptor1,
                          const CompressedFeatureDescriptors* descriptors2,
						  const int start_row2, const int num_descriptor2,
                          SiftMatchGPU* sift_match_gpu,
                          FeatureMatches* matches);

void MatchGuidedSiftFeaturesGPU(const PairMatchingOptions& match_options,
                                const FeatureKeypoints* keypoints1,
                                const FeatureKeypoints* keypoints2,
                                const CompressedFeatureDescriptors* descriptors1,
                                const CompressedFeatureDescriptors* descriptors2,
                                SiftMatchGPU* sift_match_gpu,
                                TwoViewGeometry* two_view_geometry);

void MatchGuidedSiftFeaturesGPUSpherical(
						const Camera& camera1,
						const Camera& camera2,
						const PairMatchingOptions& match_options,
						const FeatureKeypoints* keypoints1,
						const FeatureKeypoints* keypoints2,
						const CompressedFeatureDescriptors* descriptors1,
						const CompressedFeatureDescriptors* descriptors2,
						SiftMatchGPU* sift_match_gpu,
						TwoViewGeometry* two_view_geometry);

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
						TwoViewGeometry* two_view_geometry);
#endif
}//namespace sensemap

#endif //SENSEMAP_MATCHING_UTILS_H