//Copyright (c) 2019, SenseTime Group.
//All rights reserved.

#ifndef SENSEMAP_FEATURE_SIFT_H_
#define SENSEMAP_FEATURE_SIFT_H_

#include "feature/types.h"
#include "util/bitmap.h"
#include "util/panorama.h"
#include "util/large_fov_image.h"
#include "util/piecewise_image.h"

extern "C" {
#include "apriltag3/apriltag.h"
#include "apriltag3/tag36h11.h"
#include "apriltag3/tag25h9.h"
#include "apriltag3/tag16h5.h"
#include "apriltag3/tagCircle21h7.h"
#include "apriltag3/tagCircle49h12.h"
#include "apriltag3/tagCustom48h12.h"
#include "apriltag3/tagStandard41h12.h"
#include "apriltag3/tagStandard52h13.h"
}

#ifdef CUDA_ENABLED
class SiftGPU;
#endif
namespace sensemap {

struct SiftExtractionOptions {
	// Number of threads for feature extraction.
	int num_threads = -1;

	// Whether to use the GPU for feature extraction.
	bool use_gpu = true;

	// Index of the GPU used for feature extraction. For multi-GPU extraction,
	// you should separate multiple GPU indices by comma, e.g., "0,1,2,3".
	std::string gpu_index = "-1";

	// Maximum image size, otherwise image will be down-scaled.
	int max_image_size = 3200;

	// Maximum number of features to detect, keeping larger-scale features.
	int max_num_features = 16384;

	// First octave in the pyramid, i.e. -1 upsamples the image by one level.
	int first_octave = -1;

	// Number of octaves.
	int num_octaves = 4;

	// Number of levels per octave.
	int octave_resolution = 3;

	// Peak threshold for detection.
	double peak_threshold = 0.02 / octave_resolution;

	// Edge threshold for detection.
	double edge_threshold = 10.0;

	// Estimate affine shape of SIFT features in the form of oriented ellipses as
	// opposed to original SIFT which estimates oriented disks.
	bool estimate_affine_shape = false;

	// Maximum number of orientations per keypoint if not estimate_affine_shape.
	int max_num_orientations = 2;

	// Fix the orientation to 0 for upright features.
	bool upright = false;

	// Whether to adapt the feature detection depending on the image darkness.
	// Note that this feature is only available in the OpenGL SiftGPU version.
	bool darkness_adaptivity = false;

	// The max and min number of features in adaptive feature extraction. 
	// Feature extraction will be re-implemented until the two numbers are  
	// fulfilled. 
	int min_num_features_customized = 1024;
	int max_num_features_customized = 2048;

	// number of perspective images converted from panorama
	int perspective_image_count = 8;
	
	// size of perspective image converted from panorama
	int perspective_image_width = 600;
	int perspective_image_height = 600;

	// size of panorama image size
	int panorama_image_width = 5760;
	int panorama_image_height = 2880;

	// whether converte to perspective image first
	bool convert_to_perspective_image = false;

	// Use panorama param config or input param
	bool use_panorama_config = false;

	// fov of perspective image converted from panorama
	int fov_w = 90;

	// Panorama param config
	std::vector<PanoramaParam> panorama_config_params;

	// AprilTag
	bool detect_apriltag = false;
    apriltag_family_t *apriltag_family = NULL;

	// Binary feature
	std::string binary_descriptor_pattern = "";

	std::string pca_matrix_path = "";
	
	// Domain-size pooling parameters. Domain-size pooling computes an average
	// SIFT descriptor across multiple scales around the detected scale. This was
	// proposed in "Domain-Size Pooling in Local Descriptors and Network
	// Architectures", J. Dong and S. Soatto, CVPR 2015. This has been shown to
	// outperform other SIFT variants and learned descriptors in "Comparative
	// Evaluation of Hand-Crafted and Learned Local Features", Schönberger,
	// Hardmeier, Sattler, Pollefeys, CVPR 2016.
	bool domain_size_pooling = false;
	double dsp_min_scale = 1.0 / 6.0;
	double dsp_max_scale = 3.0;
	int dsp_num_scales = 10;

	enum class Normalization {
		// L1-normalizes each descriptor followed by element-wise square rooting.
		// This normalization is usually better than standard L2-normalization.
		// See "Three things everyone should know to improve object retrieval",
		// Relja Arandjelovic and Andrew Zisserman, CVPR 2012.
				L1_ROOT,
		// Each vector is L2-normalized.
				L2,
	};
	Normalization normalization = Normalization::L1_ROOT;

	bool Check() const;
};


// Extract SIFT features for the given image on the CPU. Only extract
// descriptors if the given input is not NULL.
bool ExtractSiftFeaturesCPU(const SiftExtractionOptions& options,
                            const std::vector<Bitmap>& bitmap, 
                            apriltag_detector_t* tag_detector,
                            FeatureKeypoints* keypoints,
                            FeatureDescriptors* descriptors,
							PanoramaIndexs* panoramaidxs,
                            AprilTagDetections* detections);

bool ExtractCovariantSiftFeaturesCPU(const SiftExtractionOptions& options,
                            const std::vector<Bitmap>& bitmaps,
                            FeatureKeypoints* keypoints,
                            FeatureDescriptors* descriptors,
							PanoramaIndexs* panoramaidxs,
							bool undistort = false,
							LargeFovImage* large_fov_image = NULL);

bool ExtractCovariantSiftFeaturesCPU(const SiftExtractionOptions& options,
                            const std::vector<Bitmap>& bitmap, 
                            FeatureKeypoints* keypoints,
                            FeatureDescriptors* descriptors,
							PanoramaIndexs* panoramaidxs,
							PieceIndexs* pieceidxs, 
							bool undistort,
							PiecewiseImage* piecewise_image,
							const int num_local_camera);

bool ExtractSiftFeaturesCPUPanorama(const SiftExtractionOptions& options,
                            const Bitmap& bitmap,
                            apriltag_detector_t* tag_detector,
                            FeatureKeypoints* keypoints,
                            FeatureDescriptors* descriptors,
                            PanoramaIndexs* panoramaidxs,
                            AprilTagDetections* detections);


#ifdef CUDA_ENABLED
// Create a SiftGPU feature extractor. The same SiftGPU instance can be used to
// extract features for multiple images. Note a OpenGL context must be made
// current in the thread of the caller. If the gpu_index is not -1, the CUDA
// version of SiftGPU is used, which produces slightly different results
// than the OpenGL implementation.
bool CreateSiftGPUExtractor(const SiftExtractionOptions& options,
                            SiftGPU* sift_gpu);

// Extract SIFT features for the given image on the GPU.
// SiftGPU must already be initialized using `CreateSiftGPU`.
bool ExtractSiftFeaturesGPU(const SiftExtractionOptions& options,
                            const std::vector<Bitmap>& bitmaps, SiftGPU* sift_gpu,
                            FeatureKeypoints* keypoints,
                            FeatureDescriptors* descriptors,
							PanoramaIndexs* panoramaidxs,
							bool undistort = false,
							LargeFovImage* large_fov_image = NULL);

bool ExtractSiftFeaturesGPU(const SiftExtractionOptions& options,
                            const std::vector<Bitmap>& bitmaps, SiftGPU* sift_gpu,
                            FeatureKeypoints* keypoints,
                            FeatureDescriptors* descriptors,
							PanoramaIndexs* panoramaidxs,
							PieceIndexs* pieceidxs, 
							bool undistort,
							PiecewiseImage* piecewise_image,
							const int num_local_camera);

bool ExtractSiftFeaturesGPUPanorama(const SiftExtractionOptions& options,
                            const Bitmap& bitmap, 
							const std::vector<Bitmap>& perspective_images,
							Panorama* panorama,
							SiftGPU* sift_gpu,
                            FeatureKeypoints* keypoints,
                            FeatureDescriptors* descriptors,
							PanoramaIndexs* panoramaidxs);


#endif

} // namespace sensemap

#endif //SENSEMAP_FEATURE_SIFT_H
