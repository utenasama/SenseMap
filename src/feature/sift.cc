//Copyright (c) 2019, SenseTime Group.
//All rights reserved.

#include "sift.h"

#include <array>
#include <fstream>
#include <memory>
#include <map>
#include <mutex>
#include "VLFeat/covdet.h"
#include "VLFeat/sift.h"
#include "feature/utils.h"
#include "util/logging.h"
#include "util/math.h"
#include "util/misc.h"

// AprilTag
// #include <ethz_apriltag2/include/apriltags/TagDetection.h>
// #include <ethz_apriltag2/include/apriltags/TagDetector.h>
// #include <ethz_apriltag2/include/apriltags/TagFamily.h>
// #include <ethz_apriltag2/include/apriltags/Tag16h5.h>
// #include <ethz_apriltag2/include/apriltags/Tag25h7.h>
// #include <ethz_apriltag2/include/apriltags/Tag25h9.h>
// #include <ethz_apriltag2/include/apriltags/Tag36h9.h>
// #include <ethz_apriltag2/include/apriltags/Tag36h11.h>




#include <unordered_map>
#ifdef CUDA_ENABLED
#include "SiftGPU/SiftGPU.h"
#include <GL/gl.h>
#endif
namespace sensemap {

namespace {

// Mutexes that ensure that only one thread extracts/matches on the same GPU
// at the same time, since SiftGPU internally uses static variables.
static std::map<int, std::unique_ptr<std::mutex>> sift_extraction_mutexes;
static std::map<int, std::unique_ptr<std::mutex>> sift_matching_mutexes;

// VLFeat uses a different convention to store its descriptors. This transforms
// the VLFeat format into the original SIFT format that is also used by SiftGPU.
FeatureDescriptors TransformVLFeatToUBCFeatureDescriptors(
		const FeatureDescriptors& vlfeat_descriptors) {
	FeatureDescriptors ubc_descriptors(vlfeat_descriptors.rows(),
	                                   vlfeat_descriptors.cols());
	const std::array<int, 8> q{{0, 7, 6, 5, 4, 3, 2, 1}};
	for (FeatureDescriptors::Index n = 0; n < vlfeat_descriptors.rows(); ++n) {
		for (int i = 0; i < 4; ++i) {
			for (int j = 0; j < 4; ++j) {
				for (int k = 0; k < 8; ++k) {
					ubc_descriptors(n, 8 * (j + 4 * i) + q[k]) =
							vlfeat_descriptors(n, 8 * (j + 4 * i) + k);
				}
			}
		}
	}
	return ubc_descriptors;
}

void WarnDarknessAdaptivityNotAvailable() {
	std::cout << "WARNING: Darkness adaptivity only available for GLSL SiftGPU."
	          << std::endl;
}

void BitmapToMat(const Bitmap& bitmap, cv::Mat& mat){
    if(bitmap.Channels()==3){
        mat.create(bitmap.Height(),bitmap.Width(),CV_8UC3);
    }
    else{
        mat.create(bitmap.Height(),bitmap.Width(),CV_8UC1);
    }

    for(int r = 0; r < bitmap.Height(); ++r){
        for(int c = 0; c < bitmap.Width(); ++c){
            BitmapColor<uint8_t> color;
            bitmap.GetPixel(c,r,&color);
            cv::Vec3b color_in_mat;
            color_in_mat(0) = color.b;
            color_in_mat(1) = color.g;
            color_in_mat(2) = color.r;
            if(bitmap.Channels()==3){
                mat.at<cv::Vec3b>(r,c) = color_in_mat;
            }
            else{
                mat.at<uint8_t>(r,c) = color_in_mat(2);
            }
        }
    }
}

}  // namespace

bool SiftExtractionOptions::Check() const {
	if (use_gpu) {
		CHECK_OPTION_GT(CSVToVector<int>(gpu_index).size(), 0);
	}
	CHECK_OPTION_GT(max_image_size, 0);
	CHECK_OPTION_GT(max_num_features, 0);
	CHECK_OPTION_GT(octave_resolution, 0);
	CHECK_OPTION_GT(peak_threshold, 0.0);
	CHECK_OPTION_GT(edge_threshold, 0.0);
	CHECK_OPTION_GT(max_num_orientations, 0);
	if (domain_size_pooling) {
		CHECK_OPTION_GT(dsp_min_scale, 0);
		CHECK_OPTION_GE(dsp_max_scale, dsp_min_scale);
		CHECK_OPTION_GT(dsp_num_scales, 0);
	}
	return true;
}


bool ExtractSiftFeaturesCPU(const SiftExtractionOptions& options,
                            const std::vector<Bitmap>& bitmaps, 
                            apriltag_detector_t* tag_detector,
                            FeatureKeypoints* keypoints,
                            FeatureDescriptors* descriptors,
							PanoramaIndexs* panoramaidxs,
                            AprilTagDetections* detections) {
	//CHECK(options.Check());
	
	CHECK_NOTNULL(keypoints);

	size_t total_num_keypoints = 0;

	for (size_t local_camera_id = 0; local_camera_id < bitmaps.size();
		 ++local_camera_id){

		const Bitmap &bitmap = bitmaps[local_camera_id];

		//CHECK(options.Check());
		CHECK(bitmap.IsGrey());
		CHECK_NOTNULL(keypoints);

		CHECK(!options.estimate_affine_shape);
		CHECK(!options.domain_size_pooling);

		if (options.darkness_adaptivity){
			WarnDarknessAdaptivityNotAvailable();
		}

		// Setup SIFT extractor.
		std::unique_ptr<VlSiftFilt, void (*)(VlSiftFilt *)> sift(
			vl_sift_new(bitmap.Width(), bitmap.Height(), options.num_octaves,
						options.octave_resolution, options.first_octave),
			&vl_sift_delete);
		if (!sift){
			return false;
		}

		vl_sift_set_peak_thresh(sift.get(), options.peak_threshold);
		vl_sift_set_edge_thresh(sift.get(), options.edge_threshold);

		// Iterate through octaves.
		std::vector<size_t> level_num_features;
		std::vector<FeatureKeypoints> level_keypoints;
		std::vector<FeatureDescriptors> level_descriptors;
		bool first_octave = true;
		while (true){
			if (first_octave){
				const std::vector<uint8_t> data_uint8 = bitmap.ConvertToRowMajorArray();
				std::vector<float> data_float(data_uint8.size());
				for (size_t i = 0; i < data_uint8.size(); ++i){
					data_float[i] = static_cast<float>(data_uint8[i]) / 255.0f;
				}
				if (vl_sift_process_first_octave(sift.get(), data_float.data())){
					break;
				}
				first_octave = false;
			}
			else{
				if (vl_sift_process_next_octave(sift.get())){
					break;
				}
			}

			// Detect keypoints.
			vl_sift_detect(sift.get());

			// Extract detected keypoints.
			const VlSiftKeypoint *vl_keypoints = vl_sift_get_keypoints(sift.get());
			const int num_keypoints = vl_sift_get_nkeypoints(sift.get());
			if (num_keypoints == 0){
				continue;
			}

			// Extract features with different orientations per DOG level.
			size_t level_idx = 0;
			int prev_level = -1;
			for (int i = 0; i < num_keypoints; ++i){
				if (vl_keypoints[i].is != prev_level){
					if (i > 0){
						// Resize containers of previous DOG level.
						level_keypoints.back().resize(level_idx);
						if (descriptors != nullptr){
							level_descriptors.back().conservativeResize(level_idx, 128);
						}
					}

					// Add containers for new DOG level.
					level_idx = 0;
					level_num_features.push_back(0);
					level_keypoints.emplace_back(options.max_num_orientations *
												 num_keypoints);
					if (descriptors != nullptr){
						level_descriptors.emplace_back(
							options.max_num_orientations * num_keypoints, 128);
					}
				}

				level_num_features.back() += 1;
				prev_level = vl_keypoints[i].is;

				// Extract feature orientations.
				double angles[4];
				int num_orientations;
				if (options.upright){
					num_orientations = 1;
					angles[0] = 0.0;
				}
				else{
					num_orientations = vl_sift_calc_keypoint_orientations(
						sift.get(), angles, &vl_keypoints[i]);
				}

				// Note that this is different from SiftGPU, which selects the top
				// global maxima as orientations while this selects the first two
				// local maxima. It is not clear which procedure is better.
				const int num_used_orientations =
					std::min(num_orientations, options.max_num_orientations);

				for (int o = 0; o < num_used_orientations; ++o){
					level_keypoints.back()[level_idx] =
						FeatureKeypoint(vl_keypoints[i].x + 0.5f, vl_keypoints[i].y + 0.5f,
										vl_keypoints[i].sigma, angles[o]);
					if (descriptors != nullptr){
						Eigen::MatrixXf desc(1, 128);
						vl_sift_calc_keypoint_descriptor(sift.get(), desc.data(),
														 &vl_keypoints[i], angles[o]);
						if (options.normalization ==
							SiftExtractionOptions::Normalization::L2){
							desc = L2NormalizeFeatureDescriptors(desc);
						}
						else if (options.normalization ==
								 SiftExtractionOptions::Normalization::L1_ROOT){
							desc = L1RootNormalizeFeatureDescriptors(desc);
						}
						else{
							LOG(FATAL) << "Normalization type not supported";
						}

						level_descriptors.back().row(level_idx) =
							FeatureDescriptorsToUnsignedByte(desc);
					}

					level_idx += 1;
				}
			}

			// Resize containers for last DOG level in octave.
			level_keypoints.back().resize(level_idx);
			if (descriptors != nullptr){
				level_descriptors.back().conservativeResize(level_idx, 128);
			}
		}

		// Determine how many DOG levels to keep to satisfy max_num_features option.
		int first_level_to_keep = 0;
		int num_features = 0;
		int num_features_with_orientations = 0;
		for (int i = level_keypoints.size() - 1; i >= 0; --i){
			num_features += level_num_features[i];
			num_features_with_orientations += level_keypoints[i].size();
			if (num_features > options.max_num_features){
				first_level_to_keep = i;
				break;
			}
		}

		// Extract the features to be kept.
		{
			size_t k = 0;
			keypoints->resize(num_features_with_orientations + total_num_keypoints);
			panoramaidxs->resize(num_features_with_orientations + total_num_keypoints);
			for (size_t i = first_level_to_keep; i < level_keypoints.size(); ++i){
				for (size_t j = 0; j < level_keypoints[i].size(); ++j){
					(*keypoints)[k + total_num_keypoints] = level_keypoints[i][j];
					(*panoramaidxs)[k + total_num_keypoints].sub_image_id = local_camera_id;
					(*panoramaidxs)[k + total_num_keypoints].sub_x = level_keypoints[i][j].x;
					(*panoramaidxs)[k + total_num_keypoints].sub_y = level_keypoints[i][j].y;
					k += 1;
				}
			}
		}

		// Compute the descriptors for the detected keypoints.
		if (descriptors != nullptr){
			size_t k = 0;
			descriptors->conservativeResize(num_features_with_orientations + total_num_keypoints, 128);
			for (size_t i = first_level_to_keep; i < level_keypoints.size(); ++i){
				for (size_t j = 0; j < level_keypoints[i].size(); ++j){
					descriptors->row(k + total_num_keypoints) = level_descriptors[i].row(j);
					k += 1;
				}
			}
		}
		total_num_keypoints += num_features_with_orientations;
	}

	if(descriptors !=nullptr){
		*descriptors = TransformVLFeatToUBCFeatureDescriptors(*descriptors);
	}



	return true;
}

bool ExtractCovariantSiftFeaturesCPU(const SiftExtractionOptions& options,
                            const std::vector<Bitmap>& bitmaps, 
                            FeatureKeypoints* keypoints,
                            FeatureDescriptors* descriptors,
							PanoramaIndexs* panoramaidxs,
                            bool undistort,
							LargeFovImage* large_fov_image) {
	//CHECK(options.Check());
	
	CHECK_NOTNULL(keypoints);

	size_t total_num_keypoints = 0;

	keypoints->clear();
	panoramaidxs->clear();
	for (size_t local_camera_id = 0; local_camera_id < bitmaps.size();
		 ++local_camera_id){

		const Bitmap &bitmap = bitmaps[local_camera_id];

		//CHECK(options.Check());
		CHECK(bitmap.IsGrey());
		CHECK_NOTNULL(keypoints);

		if (options.darkness_adaptivity){
			WarnDarknessAdaptivityNotAvailable();
		}

  		// Setup covariant SIFT detector.
		std::unique_ptr<VlCovDet, void (*)(VlCovDet*)> covdet(
			vl_covdet_new(VL_COVDET_METHOD_DOG), &vl_covdet_delete);
		if (!covdet) {
			return false;
		}

		const int kMaxOctaveResolution = 1000;
		CHECK_LE(options.octave_resolution, kMaxOctaveResolution);

		vl_covdet_set_first_octave(covdet.get(), options.first_octave);
		vl_covdet_set_octave_resolution(covdet.get(), options.octave_resolution);
		vl_covdet_set_peak_threshold(covdet.get(), options.peak_threshold);
		vl_covdet_set_edge_threshold(covdet.get(), options.edge_threshold);

		{
			const std::vector<uint8_t> data_uint8 = bitmap.ConvertToRowMajorArray();
			std::vector<float> data_float(data_uint8.size());
			for (size_t i = 0; i < data_uint8.size(); ++i) {
				data_float[i] = static_cast<float>(data_uint8[i]) / 255.0f;
			}
			vl_covdet_put_image(covdet.get(), data_float.data(), bitmap.Width(),
								bitmap.Height());
		}

		vl_covdet_detect(covdet.get(), options.max_num_features);

		if (!options.upright) {
			if (options.estimate_affine_shape) {
				vl_covdet_extract_affine_shape(covdet.get());
			} else {
				vl_covdet_extract_orientations(covdet.get());
			}
		}

		const int num_features = vl_covdet_get_num_features(covdet.get());
		VlCovDetFeature* features = vl_covdet_get_features(covdet.get());

		// Sort features according to detected octave and scale.
		std::sort(
			features, features + num_features,
			[](const VlCovDetFeature& feature1, const VlCovDetFeature& feature2) {
			if (feature1.o == feature2.o) {
				return feature1.s > feature2.s;
			} else {
				return feature1.o > feature2.o;
			}
		});

  		const size_t max_num_features = static_cast<size_t>(options.max_num_features);

		// Copy detected keypoints and clamp when maximum number of features reached.
		int prev_octave_scale_idx = std::numeric_limits<int>::max();
		int valid_num_features = 0;
		for (int i = 0; i < num_features; ++i) {
			++valid_num_features;
			const double u_in = features[i].frame.x + 0.5;
			const double v_in = features[i].frame.y + 0.5;
			double u, v;
			if (undistort && large_fov_image!=NULL) {
				large_fov_image->ConvertPerspectiveCoordToOriginal(u_in, v_in, local_camera_id, u, v);
			} else {
				u = u_in;
				v = v_in;
			}

			FeatureKeypoint keypoint;
			keypoint.x = u;
			keypoint.y = v;
			keypoint.a11 = features[i].frame.a11;
			keypoint.a12 = features[i].frame.a12;
			keypoint.a21 = features[i].frame.a21;
			keypoint.a22 = features[i].frame.a22;
			keypoints->push_back(keypoint);

			PanoramaIndex index;
			index.sub_image_id = local_camera_id;
			index.sub_x = u_in;
			index.sub_y = v_in;
			panoramaidxs->push_back(index);

			const int octave_scale_idx =
				features[i].o * kMaxOctaveResolution + features[i].s;
			CHECK_LE(octave_scale_idx, prev_octave_scale_idx);

			if (octave_scale_idx != prev_octave_scale_idx &&
				keypoints->size() >= max_num_features) {
				break;
			}

			prev_octave_scale_idx = octave_scale_idx;
		}

		// Compute the descriptors for the detected keypoints.
		if (descriptors != nullptr) {
			descriptors->conservativeResize(valid_num_features + total_num_keypoints, 128);

			const size_t kPatchResolution = 15;
			const size_t kPatchSide = 2 * kPatchResolution + 1;
			const double kPatchRelativeExtent = 7.5;
			const double kPatchRelativeSmoothing = 1;
			const double kPatchStep = kPatchRelativeExtent / kPatchResolution;
			const double kSigma =
				kPatchRelativeExtent / (3.0 * (4 + 1) / 2) / kPatchStep;

			std::vector<float> patch(kPatchSide * kPatchSide);
			std::vector<float> patchXY(2 * kPatchSide * kPatchSide);

			float dsp_min_scale = 1;
			float dsp_scale_step = 0;
			int dsp_num_scales = 1;
			if (options.domain_size_pooling) {
				dsp_min_scale = options.dsp_min_scale;
				dsp_scale_step = (options.dsp_max_scale - options.dsp_min_scale) /
								options.dsp_num_scales;
				dsp_num_scales = options.dsp_num_scales;
			}

			Eigen::Matrix<float, Eigen::Dynamic, 128, Eigen::RowMajor>
				scaled_descriptors(dsp_num_scales, 128);

			std::unique_ptr<VlSiftFilt, void (*)(VlSiftFilt*)> sift(
				vl_sift_new(16, 16, 1, 3, 0), &vl_sift_delete);
			if (!sift) {
				return false;
			}

			vl_sift_set_magnif(sift.get(), 3.0);

			for (size_t i = 0; i < valid_num_features; ++i) {
				for (int s = 0; s < dsp_num_scales; ++s) {
					const double dsp_scale = dsp_min_scale + s * dsp_scale_step;

					VlFrameOrientedEllipse scaled_frame = features[i].frame;
					scaled_frame.a11 *= dsp_scale;
					scaled_frame.a12 *= dsp_scale;
					scaled_frame.a21 *= dsp_scale;
					scaled_frame.a22 *= dsp_scale;

					vl_covdet_extract_patch_for_frame(
						covdet.get(), patch.data(), kPatchResolution, kPatchRelativeExtent,
						kPatchRelativeSmoothing, scaled_frame);

					vl_imgradient_polar_f(patchXY.data(), patchXY.data() + 1, 2,
										2 * kPatchSide, patch.data(), kPatchSide,
										kPatchSide, kPatchSide);

					vl_sift_calc_raw_descriptor(sift.get(), patchXY.data(),
												scaled_descriptors.row(s).data(),
												kPatchSide, kPatchSide, kPatchResolution,
												kPatchResolution, kSigma, 0);
				}

				Eigen::Matrix<float, 1, 128> descriptor;
				if (options.domain_size_pooling) {
					descriptor = scaled_descriptors.colwise().mean();
				} else {
					descriptor = scaled_descriptors;
				}

				if (options.normalization == SiftExtractionOptions::Normalization::L2) {
					descriptor = L2NormalizeFeatureDescriptors(descriptor);
				} else if (options.normalization ==
							SiftExtractionOptions::Normalization::L1_ROOT) {
					descriptor = L1RootNormalizeFeatureDescriptors(descriptor);
				} else {
					LOG(FATAL) << "Normalization type not supported";
				}

				descriptors->row(i + total_num_keypoints) = FeatureDescriptorsToUnsignedByte(descriptor);
			}
		}
		total_num_keypoints += valid_num_features;
	}

	if(descriptors !=nullptr){
		*descriptors = TransformVLFeatToUBCFeatureDescriptors(*descriptors);
		CHECK_EQ(descriptors->rows(), keypoints->size());
	}

	return true;
}

bool ExtractCovariantSiftFeaturesCPU(const SiftExtractionOptions& options,
                            const std::vector<Bitmap>& bitmaps, 
                            FeatureKeypoints* keypoints,
                            FeatureDescriptors* descriptors,
							PanoramaIndexs* panoramaidxs,
							PieceIndexs* pieceidxs, 
							bool undistort,
							PiecewiseImage* piecewise_image,
							const int num_local_camera) {
	//CHECK(options.Check());
	
	CHECK_NOTNULL(keypoints);

	size_t total_num_keypoints = 0;

	keypoints->clear();
	panoramaidxs->clear();
	pieceidxs->clear();
	int piece_num = bitmaps.size() / num_local_camera;
	for (size_t local_camera_id = 0; local_camera_id < bitmaps.size();
		 ++local_camera_id){

		const Bitmap &bitmap = bitmaps[local_camera_id];

		//CHECK(options.Check());
		CHECK(bitmap.IsGrey());
		CHECK_NOTNULL(keypoints);

		if (options.darkness_adaptivity){
			WarnDarknessAdaptivityNotAvailable();
		}

  		// Setup covariant SIFT detector.
		std::unique_ptr<VlCovDet, void (*)(VlCovDet*)> covdet(
			vl_covdet_new(VL_COVDET_METHOD_DOG), &vl_covdet_delete);
		if (!covdet) {
			return false;
		}

		const int kMaxOctaveResolution = 1000;
		CHECK_LE(options.octave_resolution, kMaxOctaveResolution);

		vl_covdet_set_first_octave(covdet.get(), options.first_octave);
		vl_covdet_set_octave_resolution(covdet.get(), options.octave_resolution);
		vl_covdet_set_peak_threshold(covdet.get(), options.peak_threshold);
		vl_covdet_set_edge_threshold(covdet.get(), options.edge_threshold);

		{
			const std::vector<uint8_t> data_uint8 = bitmap.ConvertToRowMajorArray();
			std::vector<float> data_float(data_uint8.size());
			for (size_t i = 0; i < data_uint8.size(); ++i) {
				data_float[i] = static_cast<float>(data_uint8[i]) / 255.0f;
			}
			vl_covdet_put_image(covdet.get(), data_float.data(), bitmap.Width(),
								bitmap.Height());
		}

		vl_covdet_detect(covdet.get(), options.max_num_features);

		if (!options.upright) {
			if (options.estimate_affine_shape) {
				vl_covdet_extract_affine_shape(covdet.get());
			} else {
				vl_covdet_extract_orientations(covdet.get());
			}
		}

		const int num_features = vl_covdet_get_num_features(covdet.get());
		VlCovDetFeature* features = vl_covdet_get_features(covdet.get());

		// Sort features according to detected octave and scale.
		std::sort(
			features, features + num_features,
			[](const VlCovDetFeature& feature1, const VlCovDetFeature& feature2) {
			if (feature1.o == feature2.o) {
				return feature1.s > feature2.s;
			} else {
				return feature1.o > feature2.o;
			}
		});

  		const size_t max_num_features = static_cast<size_t>(options.max_num_features);

		// Copy detected keypoints and clamp when maximum number of features reached.
		int prev_octave_scale_idx = std::numeric_limits<int>::max();
		int valid_num_features = 0;
		for (int i = 0; i < num_features; ++i) {
			++valid_num_features;
			const double u_in = features[i].frame.x + 0.5;
			const double v_in = features[i].frame.y + 0.5;
			double u, v;
			if (undistort && piecewise_image != NULL) {
				piecewise_image->ConvertSplitedPerspectiveCoordToOriginal(u_in, v_in, local_camera_id / piece_num, local_camera_id % piece_num, u, v);
			} else {
				u = u_in;
				v = v_in;
			}

			FeatureKeypoint keypoint;
			keypoint.x = u;
			keypoint.y = v;
			keypoint.a11 = features[i].frame.a11;
			keypoint.a12 = features[i].frame.a12;
			keypoint.a21 = features[i].frame.a21;
			keypoint.a22 = features[i].frame.a22;
			keypoints->push_back(keypoint);

			PanoramaIndex index;
			index.sub_image_id = local_camera_id / piece_num;
			panoramaidxs->push_back(index);

			PieceIndex piece_index;
			piece_index.piece_id = local_camera_id % piece_num;
			piece_index.piece_x = u_in;
			piece_index.piece_y = v_in;
			pieceidxs->push_back(piece_index);

			const int octave_scale_idx =
				features[i].o * kMaxOctaveResolution + features[i].s;
			CHECK_LE(octave_scale_idx, prev_octave_scale_idx);

			if (octave_scale_idx != prev_octave_scale_idx &&
				keypoints->size() >= max_num_features) {
				break;
			}

			prev_octave_scale_idx = octave_scale_idx;
		}

		// Compute the descriptors for the detected keypoints.
		if (descriptors != nullptr) {
			descriptors->conservativeResize(valid_num_features + total_num_keypoints, 128);

			const size_t kPatchResolution = 15;
			const size_t kPatchSide = 2 * kPatchResolution + 1;
			const double kPatchRelativeExtent = 7.5;
			const double kPatchRelativeSmoothing = 1;
			const double kPatchStep = kPatchRelativeExtent / kPatchResolution;
			const double kSigma =
				kPatchRelativeExtent / (3.0 * (4 + 1) / 2) / kPatchStep;

			std::vector<float> patch(kPatchSide * kPatchSide);
			std::vector<float> patchXY(2 * kPatchSide * kPatchSide);

			float dsp_min_scale = 1;
			float dsp_scale_step = 0;
			int dsp_num_scales = 1;
			if (options.domain_size_pooling) {
				dsp_min_scale = options.dsp_min_scale;
				dsp_scale_step = (options.dsp_max_scale - options.dsp_min_scale) /
								options.dsp_num_scales;
				dsp_num_scales = options.dsp_num_scales;
			}

			Eigen::Matrix<float, Eigen::Dynamic, 128, Eigen::RowMajor>
				scaled_descriptors(dsp_num_scales, 128);

			std::unique_ptr<VlSiftFilt, void (*)(VlSiftFilt*)> sift(
				vl_sift_new(16, 16, 1, 3, 0), &vl_sift_delete);
			if (!sift) {
				return false;
			}

			vl_sift_set_magnif(sift.get(), 3.0);

			for (size_t i = 0; i < valid_num_features; ++i) {
				for (int s = 0; s < dsp_num_scales; ++s) {
					const double dsp_scale = dsp_min_scale + s * dsp_scale_step;

					VlFrameOrientedEllipse scaled_frame = features[i].frame;
					scaled_frame.a11 *= dsp_scale;
					scaled_frame.a12 *= dsp_scale;
					scaled_frame.a21 *= dsp_scale;
					scaled_frame.a22 *= dsp_scale;

					vl_covdet_extract_patch_for_frame(
						covdet.get(), patch.data(), kPatchResolution, kPatchRelativeExtent,
						kPatchRelativeSmoothing, scaled_frame);

					vl_imgradient_polar_f(patchXY.data(), patchXY.data() + 1, 2,
										2 * kPatchSide, patch.data(), kPatchSide,
										kPatchSide, kPatchSide);

					vl_sift_calc_raw_descriptor(sift.get(), patchXY.data(),
												scaled_descriptors.row(s).data(),
												kPatchSide, kPatchSide, kPatchResolution,
												kPatchResolution, kSigma, 0);
				}

				Eigen::Matrix<float, 1, 128> descriptor;
				if (options.domain_size_pooling) {
					descriptor = scaled_descriptors.colwise().mean();
				} else {
					descriptor = scaled_descriptors;
				}

				if (options.normalization == SiftExtractionOptions::Normalization::L2) {
					descriptor = L2NormalizeFeatureDescriptors(descriptor);
				} else if (options.normalization ==
							SiftExtractionOptions::Normalization::L1_ROOT) {
					descriptor = L1RootNormalizeFeatureDescriptors(descriptor);
				} else {
					LOG(FATAL) << "Normalization type not supported";
				}

				descriptors->row(i + total_num_keypoints) = FeatureDescriptorsToUnsignedByte(descriptor);
			}
		}
		total_num_keypoints += valid_num_features;
	}

	if(descriptors !=nullptr){
		*descriptors = TransformVLFeatToUBCFeatureDescriptors(*descriptors);
		CHECK_EQ(descriptors->rows(), keypoints->size());
	}

	return true;
}

bool ExtractSiftFeaturesCPUPanorama(const SiftExtractionOptions& options,
                            const Bitmap& bitmap,
                            apriltag_detector_t* tag_detector,
                            FeatureKeypoints* keypoints,
                            FeatureDescriptors* descriptors,
                            PanoramaIndexs* panoramaidxs,
                            AprilTagDetections* detections){
	Panorama panorama;
	std::vector<Bitmap> perspective_images;
    panorama.PerspectiveParamsProcess(options.perspective_image_width,
                                      options.perspective_image_height,
                                      options.perspective_image_count,
                                      options.fov_w,
                                      bitmap.Height(),
                                      bitmap.Width());
    panorama.PanoramaToPerspectives(&bitmap, perspective_images);


	size_t feature_count = 0;

	for(size_t i = 0; i<perspective_images.size(); ++i){
		FeatureKeypoints sub_keypoints;
		FeatureDescriptors sub_descriptors;
        AprilTagDetections sub_detections;

		std::vector<Bitmap> bitmap_v;
		bitmap_v.push_back(perspective_images[i]);
		PanoramaIndexs perspective_indices;

		if(!ExtractSiftFeaturesCPU(options,bitmap_v, tag_detector,
							   &sub_keypoints,&sub_descriptors,&perspective_indices,&sub_detections)){
			return false;					   
		}

		keypoints->resize(feature_count + sub_keypoints.size());
		descriptors->conservativeResize(feature_count + 
										sub_keypoints.size(), 128);
		panoramaidxs->resize(feature_count + sub_keypoints.size());
		
		for(size_t j = 0; j<sub_keypoints.size(); ++j){
			double u, v;
			double u_in = sub_keypoints[j].x;
			double v_in = sub_keypoints[j].y;

			panorama.ConvertPerspectiveCoordToPanorama(i,u_in,v_in,u,v);

			(*keypoints)[j+feature_count] = sub_keypoints[j];
			(*keypoints)[j+feature_count].x = u;
			(*keypoints)[j+feature_count].y = v;
			(*panoramaidxs)[j+feature_count].sub_image_id = i;
			(*panoramaidxs)[j+feature_count].sub_x = u_in;
			(*panoramaidxs)[j+feature_count].sub_y = v_in;
			
			descriptors->row( j+feature_count) = sub_descriptors.row(j);
		}


        // Convert sub detection result to panorama detection result
        for (auto & sub_detection : sub_detections){
            AprilTagDetection cur_detection;
			// local_camera_id
            cur_detection.local_camera_id = sub_detection.local_camera_id;
            // id
            cur_detection.id = sub_detection.id;
            double u, v, u_in, v_in;
            // cxy
            u_in = sub_detection.cxy.first;
            v_in = sub_detection.cxy.second;
            panorama.ConvertPerspectiveCoordToPanorama(i,u_in,v_in,u,v);
            cur_detection.cxy = {(float)u, (float)v};
            // p[4]
            for(size_t j = 0; j < 4; ++j){
                u_in = sub_detection.p[j].first;
                v_in = sub_detection.p[j].second;
                panorama.ConvertPerspectiveCoordToPanorama(i,u_in,v_in,u,v);
                cur_detection.p[j] = {(float)u, (float)v};
            }
            detections->emplace_back(cur_detection);
        }

		feature_count += sub_keypoints.size();
	}

	return true;
}
//#define CUDA_ENABLED

#ifdef CUDA_ENABLED
bool CreateSiftGPUExtractor(const SiftExtractionOptions& options,
                            SiftGPU* sift_gpu) {
    CHECK(options.Check());
    CHECK_NOTNULL(sift_gpu);

    // SiftGPU uses many global static state variables and the initialization must
    // be thread-safe in order to work correctly. This is enforced here.
    static std::mutex mutex;
    std::unique_lock<std::mutex> lock(mutex);
	


    std::vector<int> gpu_indices = CSVToVector<int>(options.gpu_index);
    CHECK_EQ(gpu_indices.size(), 1) << "SiftGPU can only run on one GPU";

    std::vector<std::string> sift_gpu_args;

    sift_gpu_args.push_back("./sift_gpu");


  // Use CUDA version by default if darkness adaptivity is disabled.
  	if (!options.darkness_adaptivity && gpu_indices[0] < 0) {
    	gpu_indices[0] = 0;
  	}
 	if (gpu_indices[0] >= 0) {
    	sift_gpu_args.push_back("-cuda");
    	sift_gpu_args.push_back(std::to_string(gpu_indices[0]));
  	}


    // Darkness adaptivity (hidden feature). Significantly improves
    // distribution of features. Only available in GLSL version.
    if (options.darkness_adaptivity) {
        if (gpu_indices[0] >= 0) {
            WarnDarknessAdaptivityNotAvailable();
        }
        sift_gpu_args.push_back("-da");
    }

    // No verbose logging.
    sift_gpu_args.push_back("-v");
    sift_gpu_args.push_back("0");

    // Fixed maximum image dimension.
    sift_gpu_args.push_back("-maxd");
    sift_gpu_args.push_back(std::to_string(options.max_image_size));

    // Keep the highest level features.
    sift_gpu_args.push_back("-tc2");
    sift_gpu_args.push_back(std::to_string(options.max_num_features));

    // First octave level.
    sift_gpu_args.push_back("-fo");
    sift_gpu_args.push_back(std::to_string(options.first_octave));

    // Number of octave levels.
    sift_gpu_args.push_back("-d");
    sift_gpu_args.push_back(std::to_string(options.octave_resolution));

    // Peak threshold.
    sift_gpu_args.push_back("-t");
    sift_gpu_args.push_back(std::to_string(options.peak_threshold));

    // Edge threshold.
    sift_gpu_args.push_back("-e");
    sift_gpu_args.push_back(std::to_string(options.edge_threshold));

    if (options.upright) {
        // Fix the orientation to 0 for upright features.
        sift_gpu_args.push_back("-ofix");
        // Maximum number of orientations.
        sift_gpu_args.push_back("-mo");
        sift_gpu_args.push_back("1");
    } else {
        // Maximum number of orientations.
        sift_gpu_args.push_back("-mo");
        sift_gpu_args.push_back(std::to_string(options.max_num_orientations));
    }

    std::vector<const char*> sift_gpu_args_cstr;
    sift_gpu_args_cstr.reserve(sift_gpu_args.size());
    for (const auto& arg : sift_gpu_args) {
        sift_gpu_args_cstr.push_back(arg.c_str());
    }

    sift_gpu->ParseParam(sift_gpu_args_cstr.size(), sift_gpu_args_cstr.data());

    sift_gpu->gpu_index = gpu_indices[0];
    if (sift_extraction_mutexes.count(gpu_indices[0]) == 0) {
        sift_extraction_mutexes.emplace(
                gpu_indices[0], std::unique_ptr<std::mutex>(new std::mutex()));
    }

    return sift_gpu->VerifyContextGL() == SiftGPU::SIFTGPU_FULL_SUPPORTED;
}


bool ExtractSiftFeaturesGPU(const SiftExtractionOptions& options,
                            const std::vector<Bitmap>& bitmaps, SiftGPU* sift_gpu,
                            FeatureKeypoints* keypoints,
                            FeatureDescriptors* descriptors,
							PanoramaIndexs* panoramaidxs,
							bool undistort,
							LargeFovImage* large_fov_image) {

	//std::cout<<"Extract sift features for large fov image or perspective image"<<std::endl;
	std::unique_lock<std::mutex> lock(
			*sift_extraction_mutexes[sift_gpu->gpu_index]);

	size_t total_num_keypoints = 0;
	
	keypoints->clear();
	panoramaidxs->clear();
	
	for (size_t local_camera_id = 0; local_camera_id < bitmaps.size();
		 local_camera_id++){
		const Bitmap &bitmap = bitmaps[local_camera_id];
	
		CHECK(options.Check());
		CHECK(bitmap.IsGrey());
		CHECK_NOTNULL(keypoints);
		CHECK_NOTNULL(descriptors);
		CHECK_EQ(options.max_image_size, sift_gpu->GetMaxDimension());

		CHECK(!options.estimate_affine_shape);
		CHECK(!options.domain_size_pooling);

		// set dog threshold for each calling of the extraction
		sift_gpu->SetDogThreshold(options.peak_threshold);

		// Note, that this produces slightly different results than using SiftGPU
		// directly for RGB->GRAY conversion, since it uses different weights.
		const std::vector<uint8_t> bitmap_raw_bits = bitmap.ConvertToRawBits();
		const int code =
			sift_gpu->RunSIFT(bitmap.ScanWidth(), bitmap.Height(),
							  bitmap_raw_bits.data(), GL_LUMINANCE, GL_UNSIGNED_BYTE);

		const int kSuccessCode = 1;
		if (code != kSuccessCode){
			return false;
		}

		const size_t num_features = static_cast<size_t>(sift_gpu->GetFeatureNum());

		std::vector<SiftKeypoint> keypoints_data(num_features);

		// Eigen's default is ColMajor, but SiftGPU stores result as RowMajor.
		Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
			descriptors_float(num_features, 128);

		// Download the extracted keypoints and descriptors.
		sift_gpu->GetFeatureVector(keypoints_data.data(), descriptors_float.data());

		keypoints->resize(num_features + total_num_keypoints);
		panoramaidxs->resize(num_features + total_num_keypoints);

		int valid_num_features = 0;
		for (size_t i = 0; i < num_features; ++i){
	
			valid_num_features ++;
			double u, v;
			if(undistort && large_fov_image!=NULL){
				double u_in = keypoints_data[i].x;
				double v_in = keypoints_data[i].y;
				large_fov_image->ConvertPerspectiveCoordToOriginal(u_in, v_in, local_camera_id, u, v);
			}
			else{
				u = keypoints_data[i].x;
				v = keypoints_data[i].y;
			}
			(*keypoints)[i + total_num_keypoints] = FeatureKeypoint(u, v,
																	keypoints_data[i].s, keypoints_data[i].o);
			(*panoramaidxs)[i + total_num_keypoints].sub_image_id = local_camera_id;
			(*panoramaidxs)[i + total_num_keypoints].sub_x = keypoints_data[i].x;
			(*panoramaidxs)[i + total_num_keypoints].sub_y = keypoints_data[i].y;
		}
		//std::cout<<"valid num features: "<<valid_num_features<<std::endl;
		// Save and normalize the descriptors.
		if (options.normalization == SiftExtractionOptions::Normalization::L2){
			descriptors_float = L2NormalizeFeatureDescriptors(descriptors_float);
		}
		else if (options.normalization ==
				 SiftExtractionOptions::Normalization::L1_ROOT){
			descriptors_float = L1RootNormalizeFeatureDescriptors(descriptors_float);
		}
		else{
			LOG(FATAL) << "Normalization type not supported";
		}

		FeatureDescriptors descriptors_local_image = FeatureDescriptorsToUnsignedByte(descriptors_float);

		//BinaryFeatureDescriptors binary_descriptors_local_image;
		//ConvertFloatDescriptorsToBinary(descriptors_float,dim_index_pairs,binary_descriptors_local_image);
		
		descriptors->conservativeResize(num_features + total_num_keypoints, 128);
		//binary_descriptors->conservativeResize(num_features + total_num_keypoints, binary_descriptors_local_image.cols());
		for (size_t j = 0; j < num_features; ++j){
			descriptors->row(j + total_num_keypoints) = descriptors_local_image.row(j);
			//binary_descriptors->row(j + total_num_keypoints) = binary_descriptors_local_image.row(j);
		}
		total_num_keypoints += num_features;
	}	
	return true;
}


bool ExtractSiftFeaturesGPU(const SiftExtractionOptions& options,
                            const std::vector<Bitmap>& bitmaps, SiftGPU* sift_gpu,
                            FeatureKeypoints* keypoints,
                            FeatureDescriptors* descriptors,
							PanoramaIndexs* panoramaidxs,
							PieceIndexs* pieceidxs, 
							bool undistort,
							PiecewiseImage* piecewise_image,
							const int num_local_camera) {
								
	//std::cout<<"Extract sift features for piecewise images"<<std::endl;
	std::unique_lock<std::mutex> lock(
			*sift_extraction_mutexes[sift_gpu->gpu_index]);

	size_t total_num_keypoints = 0;
	
	keypoints->clear();
	panoramaidxs->clear();

	int piece_num = bitmaps.size() / num_local_camera;
	
	for (size_t local_camera_id = 0; local_camera_id < bitmaps.size();
		 local_camera_id++){
		const Bitmap &bitmap = bitmaps[local_camera_id];
	
		CHECK(options.Check());
		CHECK(bitmap.IsGrey());
		CHECK_NOTNULL(keypoints);
		CHECK_NOTNULL(descriptors);
		CHECK_EQ(options.max_image_size, sift_gpu->GetMaxDimension());

		CHECK(!options.estimate_affine_shape);
		CHECK(!options.domain_size_pooling);

		// set dog threshold for each calling of the extraction
		sift_gpu->SetDogThreshold(options.peak_threshold);

		// Note, that this produces slightly different results than using SiftGPU
		// directly for RGB->GRAY conversion, since it uses different weights.
		const std::vector<uint8_t> bitmap_raw_bits = bitmap.ConvertToRawBits();
		const int code =
			sift_gpu->RunSIFT(bitmap.ScanWidth(), bitmap.Height(),
							  bitmap_raw_bits.data(), GL_LUMINANCE, GL_UNSIGNED_BYTE);

		const int kSuccessCode = 1;
		if (code != kSuccessCode){
			return false;
		}

		const size_t num_features = static_cast<size_t>(sift_gpu->GetFeatureNum());

		std::vector<SiftKeypoint> keypoints_data(num_features);

		// Eigen's default is ColMajor, but SiftGPU stores result as RowMajor.
		Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
			descriptors_float(num_features, 128);

		// Download the extracted keypoints and descriptors.
		sift_gpu->GetFeatureVector(keypoints_data.data(), descriptors_float.data());

		keypoints->resize(num_features + total_num_keypoints);
		panoramaidxs->resize(num_features + total_num_keypoints);
		pieceidxs->resize(num_features + total_num_keypoints);

		int valid_num_features = 0;
		for (size_t i = 0; i < num_features; ++i){
	
			valid_num_features ++;
			double u, v;
			if(undistort && piecewise_image!=NULL){
				double u_in = keypoints_data[i].x;
				double v_in = keypoints_data[i].y;
				piecewise_image->ConvertSplitedPerspectiveCoordToOriginal(u_in, v_in, local_camera_id / piece_num, local_camera_id % piece_num, u, v);
			
			}
			else{
				u = keypoints_data[i].x;
				v = keypoints_data[i].y;
			}
			(*keypoints)[i + total_num_keypoints] = FeatureKeypoint(u, v,
																	keypoints_data[i].s, keypoints_data[i].o);
			(*panoramaidxs)[i + total_num_keypoints].sub_image_id = local_camera_id / piece_num;
			(*pieceidxs)[i + total_num_keypoints].piece_id = local_camera_id % piece_num;
			(*pieceidxs)[i + total_num_keypoints].piece_x = keypoints_data[i].x;
			(*pieceidxs)[i + total_num_keypoints].piece_y = keypoints_data[i].y;
		}
		//std::cout<<"valid num features: "<<valid_num_features<<std::endl;
		// Save and normalize the descriptors.
		if (options.normalization == SiftExtractionOptions::Normalization::L2){
			descriptors_float = L2NormalizeFeatureDescriptors(descriptors_float);
		}
		else if (options.normalization ==
				 SiftExtractionOptions::Normalization::L1_ROOT){
			descriptors_float = L1RootNormalizeFeatureDescriptors(descriptors_float);
		}
		else{
			LOG(FATAL) << "Normalization type not supported";
		}

		FeatureDescriptors descriptors_local_image = FeatureDescriptorsToUnsignedByte(descriptors_float);
		
		//BinaryFeatureDescriptors binary_descriptors_local_image;
		//ConvertFloatDescriptorsToBinary(descriptors_float,dim_index_pairs,binary_descriptors_local_image);
		
		descriptors->conservativeResize(num_features + total_num_keypoints, 128);
		//binary_descriptors->conservativeResize(num_features + total_num_keypoints, binary_descriptors_local_image.cols());
		for (size_t j = 0; j < num_features; ++j){
			descriptors->row(j + total_num_keypoints) = descriptors_local_image.row(j);
			//binary_descriptors->row(j + total_num_keypoints) = binary_descriptors_local_image.row(j);
		}
		total_num_keypoints += num_features;
	}	
	return true;
}

//#define SHOW_PANORAMA_FEATURE

bool ExtractSiftFeaturesGPUPanorama(const SiftExtractionOptions& options,
                            const Bitmap& bitmap, 
							const std::vector<Bitmap>& perspective_images,
							Panorama* panorama,
							SiftGPU* sift_gpu,
                            FeatureKeypoints* keypoints,
                            FeatureDescriptors* descriptors,
							PanoramaIndexs* panoramaidxs){

	std::cout<<"Extract sift features panorama"<<std::endl;


#ifdef SHOW_PANORAMA_FEATURE
	cv::Mat full_image_mat;
#endif

	size_t feature_count = 0;

	for(size_t i = 0; i<perspective_images.size(); ++i){
		FeatureKeypoints sub_keypoints;
		FeatureDescriptors sub_descriptors;

#ifdef SHOW_PANORAMA_FEATURE		
		std::stringstream ss;
        ss << i;
		std::string filename = "/home/fengyouji/sensemap_test/"+ss.str()+".png";
		std::string fullfilename ="/home/fengyouji/sensemap_test/full_"+ss.str()+".png";

		cv::Mat image_mat;
		BitmapToMat(perspective_images[i], image_mat);
		//cv::imwrite(filename,image_mat);
#endif		

		std::vector<Bitmap> bitmap_v;
		bitmap_v.push_back(perspective_images[i]);
		PanoramaIndexs perspective_index;
		if(!ExtractSiftFeaturesGPU(options,bitmap_v, sift_gpu,
							   &sub_keypoints,&sub_descriptors, &perspective_index)){
			return false;
		}

		keypoints->resize(feature_count + sub_keypoints.size());
		descriptors->conservativeResize(feature_count + 
										sub_keypoints.size(), 128);

		panoramaidxs->resize(feature_count + sub_keypoints.size());
#ifdef SHOW_PANORAMA_FEATURE		
		std::vector<cv::KeyPoint> keypoints_show;
		for(auto keypoint : sub_keypoints) {
			keypoints_show.emplace_back(keypoint.x, keypoint.y,
			                            keypoint.ComputeScale(),
			                            keypoint.ComputeOrientation());
		}
		// cv::drawKeypoints(image_mat, keypoints_show, image_mat, 
		// 			  cv::Scalar::all(-1),
		//               cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
		cv::imwrite(filename,image_mat);
		std::vector<cv::KeyPoint> full_keypoints_show;
		std::vector<std::vector<cv::Point2f>> full_tag_detection_show;
#endif

		for(size_t j = 0; j<sub_keypoints.size(); ++j){
			double u, v;
			double u_in = sub_keypoints[j].x;
			double v_in = sub_keypoints[j].y;

			panorama->ConvertPerspectiveCoordToPanorama(i,u_in,v_in,u,v);

			(*keypoints)[j+feature_count] = sub_keypoints[j];
			(*keypoints)[j+feature_count].x = u;
			(*keypoints)[j+feature_count].y = v;
			(*panoramaidxs)[j+feature_count].sub_image_id = i;
			(*panoramaidxs)[j+feature_count].sub_x = u_in;
			(*panoramaidxs)[j+feature_count].sub_y = v_in;

			descriptors->row( j+feature_count) = sub_descriptors.row(j);
			
#ifdef SHOW_PANORAMA_FEATURE			
			full_keypoints_show.emplace_back(u, v,
			                        	 sub_keypoints[j].ComputeScale(),
				                         sub_keypoints[j].ComputeOrientation());
#endif			
		}

		feature_count += sub_keypoints.size();

#ifdef SHOW_PANORAMA_FEATURE	
		BitmapToMat(bitmap, full_image_mat);
		cv::drawKeypoints(full_image_mat, full_keypoints_show, full_image_mat, 
					  cv::Scalar::all(-1), 
					  cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	
		cv::imwrite(fullfilename, full_image_mat);
#endif
	}


	return true;
}


#endif

} // namespace sensemap
