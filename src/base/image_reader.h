//Copyright (c) 2019, SenseTime Group.
//All rights reserved.

#ifndef SENSEMAP_BASE_IMAGE_READER_H_
#define SENSEMAP_BASE_IMAGE_READER_H_

#include <unordered_set>
#include <map>
#include <unordered_map>

#include "util/gps_reader.h"
#include "util/bitmap.h"
#include "util/mat.h"
#include "base/camera.h"
#include "base/image.h"

namespace sensemap {

struct ImageReaderOptions {
	// Root path to folder which contains the images.
	std::string image_path = "";

	// name witch contains current image
	std::string child_path = "";

	// Optional root path to folder which contains image masks. For a given image,
	// the corresponding mask must have the same sub-path below this root as the
	// image has below image_path. The filename must be equal, aside from the
	// added extension .png. For example, for an image image_path/abc/012.jpg, the
	// mask would be mask_path/abc/012.jpg.png. No features will be extracted in
	// regions where the mask image is black (pixel intensity value 0 in
	// grayscale).
	std::string mask_path = "";

	// Optional list of images to read. The list must contain the relative path
	// of the images with respect to the image_path.
	std::vector<std::string> image_list;

	// Using the image_list to generate image label list which intend to use image from 
	// different folder to generate image clusters
	std::map<std::string, label_t> image_label_list;  

	// Image name id map
	std::unordered_map<std::string, image_t> image_name_id_map;

	// Assign camera for specific images
	std::unordered_map<std::string, Camera> image_name_camera_map;
	std::unordered_map<std::string, Camera> subpath_camera_map;

	// Name of the camera model.
	std::string camera_model = "SIMPLE_RADIAL";

	

	// Whether to use the same camera for all images.
	bool single_camera = false;

	// Whether to use the same camera for all images in the same sub-folder.
	bool single_camera_per_folder = false;

	// Whether to fix camera parameters in BA.
	bool fixed_camera = false;

	// Whether to explicitly use an existing camera for all images. Note that in
	// this case the specified camera model and parameters are ignored.
	camera_t existing_camera_id = kInvalidCameraId;

	// Manual specification of camera parameters. If empty, camera parameters
	// will be extracted from EXIF, i.e. principal point and focal length.
	std::string camera_params = "";

	// If camera parameters are not specified manually and the image does not
	// have focal length EXIF information, the focal length is set to the
	// value `default_focal_length_factor * max(width, height)`.
	double default_focal_length_factor = 1.2;

	// Optional path to an image file specifying a mask for all images. No
	// features will be extracted in regions where the mask is black (pixel
	// intensity value 0 in grayscale).
	std::string camera_mask_path = "";

	std::string gps_origin = "";
	//////////////////////////////////////////////////////////////////////////
	// Extension to RGBD camera
	/////////////////////////////////////////////////////////////////////////

	// Whether to load RGBD data.
	bool with_depth = false;

	// Performs an image selection operation for images in image_path after sort. 
	// When this is an empty string, no operation is done. 
	// The basic syntax is:
	//   start_index,end_index,interval
	// And either start_index, end_index, interval can be omitted to format variants such as:
	//   start_index,end_index,
	//   start_index,,interval
	//   ,,interval
	// start_index: Start image index (included). When not given, 0 is assumed. 
	// end_index: End image index (not included). When not given, images.size() is assumed. 
	//            This value is safely truncated when greater than images.size().
	// interval: Image interval. When not given, 1 is assumed. 
	std::string image_selection = "";

	//////////////////////////////////////////////////////////////////////////
	// Extension to multi camera system
	/////////////////////////////////////////////////////////////////////////
	
	// Number of local cameras, will be larger than one when this is a 
	// multi-camera system
	int num_local_cameras;
	std::unordered_map<std::string, int> num_local_cameras_devices;
	// Manual specification of camera parameters. Params may be different for
	// different multi-camera system
	
	std::unordered_map<std::string,std::string> local_camera_params;
	std::unordered_map<std::string,std::string> local_camera_extrinsics;
	std::unordered_map<std::string,std::string> local_camera_models;


	bool read_image_info_first = false; 
	int bitmap_read_num_threads = -1;
	
	bool Check() const;
};

// Recursively iterate over the images in a directory.
class ImageReader {
public:
	enum class Status {
		FAILURE,
		SUCCESS,
		IMAGE_EXISTS,
		BITMAP_ERROR,
		CAMERA_SINGLE_DIM_ERROR,
		CAMERA_EXIST_DIM_ERROR,
		CAMERA_PARAM_ERROR
	};
	explicit ImageReader(const ImageReaderOptions& options, image_t image_index = 0, 
						 camera_t camera_index = 0, label_t label_index = 1);
    Status Next(Camera* camera, Image* image, std::vector<Bitmap>* bitmap, std::vector<Bitmap>* mask,
                std::vector<std::string>* bitmap_paths, bool info_only = false);
    size_t NextIndex() const;
    size_t NumImages() const;
    size_t InitialIndex() const;
    size_t GeoImageIndex() const;
    std::vector<std::string> ImageList() const;

       private:
	// Image reader options.
	ImageReaderOptions options_;
	// Index of previously processed image.
	image_t image_index_;
	// Index of initial processed image
	image_t inital_image_index_;
	// Index of geodesic image.
	image_t geo_image_index_;
	// Index of previously processed camera.
	camera_t camera_index_;
	// Previously processed camera.
	Camera prev_camera_;
	// Names of image sub-folders.
	std::string prev_image_folder_;
	std::unordered_set<std::string> image_folders_;
	camera_t dewarp_camera_index_;
	camera_t warp_camera_index_;
	std::unordered_map<camera_t, Camera> cameras_map_;

	std::shared_ptr<GeodeticConverter> geo_converter;
};

////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////

}  // namespace sensemap

#endif //SENSEMAP_BASE_IMAGE_READER_H
