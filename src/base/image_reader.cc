//Copyright (c) 2019, SenseTime Group.
//All rights reserved.

#include <fstream>

#define OPENCV_TRAITS_ENABLE_DEPRECATED
#include <opencv2/opencv.hpp>
#include <boost/algorithm/string.hpp>

#include "base/image_reader.h"
#include "base/camera_models.h"
#include "base/pose.h"

#include "util/misc.h"
#include "util/rgbd_helper.h"

namespace sensemap {

bool ImageReaderSelection(
	const std::string & image_selection, 
	std::vector<std::string> & image_list, 
	int num_local_cameras
) {
	std::vector<std::string> params;
  	boost::split(params, image_selection, boost::is_any_of(","), boost::token_compress_off);
	for (auto & param : params) {
		StringTrim(&param);
	}
	if (params.size() != 3) return false;

	int start_index = params[0].empty() ? 0                 : std::atoi(params[0].c_str());
	int end_index   = params[1].empty() ? image_list.size() : std::atoi(params[1].c_str());
	int interval    = params[2].empty() ? 1                 : std::atoi(params[2].c_str());
	if (start_index < 0) return false;
	if (interval <= 0) return false;

	std::vector<std::string> selected_image_list;
	if (num_local_cameras > 1) {
		std::vector<std::vector<std::string> > image_list_for_local_cameras;
		image_list_for_local_cameras.resize(num_local_cameras);

		for (auto & image_path : image_list) {
			std::string local_camera = image_path.substr(image_path.rfind("cam"), 4);
			int local_camera_index = std::stoi(local_camera.substr(3));

			CHECK_LT(local_camera_index, num_local_cameras);
			image_list_for_local_cameras[local_camera_index].emplace_back(std::move(image_path));
		}

		for (auto & local_image_list : image_list_for_local_cameras) {
			for (int i = start_index; i < std::min(end_index, (int)local_image_list.size()); i += interval) {
				selected_image_list.emplace_back(std::move(local_image_list[i]));
			}
		}
	} else {
		for (int i = start_index; i < std::min(end_index, (int)image_list.size()); i += interval) {
			selected_image_list.emplace_back(std::move(image_list[i]));
		}
	}
	image_list.swap(selected_image_list);

	return true;
}

bool ImageReaderOptions::Check() const {
	CHECK_OPTION_GT(default_focal_length_factor, 0.0);
	CHECK_OPTION(ExistsCameraModelWithName(camera_model));
	const int model_id = CameraModelNameToId(camera_model);
	if (!camera_params.empty()&& num_local_cameras == 1) {
		CHECK_OPTION(
				CameraModelVerifyParams(model_id, CSVToVector<double>(camera_params)));
	}
	return true;
}

ImageReader::ImageReader(const ImageReaderOptions& options, image_t image_index, 
		camera_t camera_index, label_t label_index)
		: options_(options), image_index_(image_index), geo_image_index_(-1),
		  camera_index_(camera_index) {
	// Ensure trailing slash, so that we can build the correct image name.
	options_.image_path =
			EnsureTrailingSlash(StringReplace(options_.image_path, "\\", "/"));

	// Get a list of all files in the image path, sorted by image name.
	if (options_.image_list.empty()) {
		std::string image_path = options_.image_path;
		if (!options_.child_path.empty()){
			options_.child_path =
				EnsureTrailingSlash(StringReplace(options_.child_path, "\\", "/"));
			image_path = JoinPaths(image_path, options_.child_path);
		}
		options_.image_list = GetRecursiveFileList(image_path);
		std::sort(options_.image_list.begin(), options_.image_list.end());
		if (!options_.image_selection.empty()) {
			CHECK(ImageReaderSelection(options_.image_selection, options_.image_list, options.num_local_cameras)) <<
				"invalid image_selection syntax: " << options_.image_selection;
		}
	} else {
		std::string image_path = options_.image_path;
		if (!options_.child_path.empty()){
			options_.child_path =
				EnsureTrailingSlash(StringReplace(options_.child_path, "\\", "/"));
			image_path = JoinPaths(image_path, options_.child_path);
		}
		for (auto& image_name : options_.image_list) {
			image_name = JoinPaths(image_path, image_name);
		}
		std::sort(options_.image_list.begin(), options_.image_list.end());
	}
    std::cout << "Image candidate number = " << options_.image_list.size() << std::endl;
    // Generate label image list using the given image list, start from 1
	// label_t label_index= 1;
	for (auto current_image_list : options_.image_list){
		std::string current_list = current_image_list.substr(0, current_image_list.rfind ("/"));
        if (options_.image_label_list.find(current_list) == options_.image_label_list.end()){
            options_.image_label_list.insert(std::pair<std::string, int>(current_list, label_index));
            label_index++;
        }
	}

	// Rerank the image list if this is a multi-camera system
	if(options.num_local_cameras > 1){
		std::vector<std::vector<std::string> > image_list_for_local_cameras;
		image_list_for_local_cameras.resize(options.num_local_cameras);

		for(auto image_path: options_.image_list){
			std::string local_camera = image_path.substr(image_path.rfind("cam"),4);
			camera_t local_camera_index = std::stoi(local_camera.substr(3));

			CHECK_LT(local_camera_index,options.num_local_cameras);
			image_list_for_local_cameras[local_camera_index].push_back(image_path);
		}

		size_t image_num = image_list_for_local_cameras[0].size();
		for(size_t i = 0; i< image_list_for_local_cameras.size(); ++i){
			CHECK_EQ(image_list_for_local_cameras[i].size(),image_num)
				<<"image numbers for local cameras are different";
		}

		options_.image_list.clear();

		for(size_t j = 0; j<image_num; ++j){
			for(size_t i = 0; i<options.num_local_cameras; ++i){
				options_.image_list.push_back(image_list_for_local_cameras[i][j]);
			}
		}
	}

	// Set the manually specified camera parameters.
	prev_camera_.SetCameraId(kInvalidCameraId);
	prev_camera_.SetModelIdFromName(options_.camera_model);
	prev_camera_.SetNumLocalCameras(options_.num_local_cameras);
	if (!options_.camera_params.empty()&& options_.num_local_cameras ==1) {
		prev_camera_.SetParamsFromString(options_.camera_params);
		prev_camera_.SetPriorFocalLength(true);
	}
	
	if(options_.num_local_cameras > 1){
		CHECK(!options_.local_camera_models.empty());
		prev_camera_.SetModelIdFromName(options_.local_camera_models.begin()->second);

		CHECK(!options_.local_camera_extrinsics.empty());
		prev_camera_.SetLocalCameraExtrinsicParamsFromString(
			options_.local_camera_extrinsics.begin()->second);

        for(local_camera_t i = 0; i<prev_camera_.NumLocalCameras(); ++i){

			Eigen::Vector4d qvec1;
			Eigen::Vector3d tvec1;
			Eigen::Matrix3d R1;
			prev_camera_.GetLocalCameraExtrinsic(i, qvec1, tvec1);
			R1 = QuaternionToRotationMatrix(qvec1);
			Eigen::Vector3d C1 = -R1.transpose() * tvec1;
			std::cout << "init prev local camera " << i << " tvec:" << std::endl;
			std::cout << tvec1.transpose() << std::endl;
			std::cout << "init prev local camera " << i << " C:" << std::endl;
			std::cout << C1.transpose() << std::endl;
		}

        CHECK(!options_.local_camera_params.empty());
        prev_camera_.SetLocalCameraIntrinsicParamsFromString(options_.local_camera_params.begin()->second);
        prev_camera_.SetPriorFocalLength(true);
    }

	dewarp_camera_index_ = kInvalidCameraId;
	warp_camera_index_ = kInvalidCameraId;
	// Update the initial image index
    inital_image_index_ = image_index_;
}

std::vector<std::string> ImageReader::ImageList() const {
	return options_.image_list;
}


bool OmnidirectionToPerspective(const Bitmap* img_in, Bitmap* img_out,
								const Camera& camera,
								const local_camera_t local_camera_id,
                                const int width, const int height){
    

    // Resize the output image using given image size
    if(img_in->Channels()==3){
        img_out->Allocate(width,height,true);
    }
    else{
        img_out->Allocate(width,height,false);
    }    

    // Apply remap to update the output image	
    for(int y = 0; y<height; ++y){
        for(int x = 0; x<width; ++x){
            BitmapColor<float> color;

			// backword mapping
			Eigen::Vector2d point_in;
			point_in(0) = static_cast<double>(x-width/2)/800.0;
			point_in(1) = static_cast<double>(y-height/2)/800.0;
			Eigen::Vector2d point_out = camera.WorldToLocalImage(local_camera_id,
																 point_in);
			if(img_in->InterpolateBilinear(point_out(0),point_out(1),&color)){
				BitmapColor<uint8_t> color_uint;
            	color_uint.r = color.r >255.0?255:static_cast<uint8_t>(color.r);
            	color_uint.g = color.g >255.0?255:static_cast<uint8_t>(color.g);
            	color_uint.b = color.b >255.0?255:static_cast<uint8_t>(color.b);
            	img_out->SetPixel(x,y,color_uint);
			}

			// forward mapping
			// Eigen::Vector2d point_in;
			// point_in(0) = static_cast<double>(x);
			// point_in(1) = static_cast<double>(y);
			// Eigen::Vector2d point_out = camera.LocalImageToWorld(local_camera_id,point_in);

			// int x_out = point_out(0) * 800 + width/2;
			// int y_out = point_out(1) * 800 + height/2;

			// if(img_in->InterpolateBilinear(point_in(0),point_in(1),&color)){
			// 	BitmapColor<uint8_t> color_uint;
            // 	color_uint.r = color.r >255.0?255:static_cast<uint8_t>(color.r);
            // 	color_uint.g = color.g >255.0?255:static_cast<uint8_t>(color.g);
            // 	color_uint.b = color.b >255.0?255:static_cast<uint8_t>(color.b);
            // 	img_out->SetPixel(x_out,y_out,color_uint);
			// }

        }
    }
    return true;
}


ImageReader::Status ImageReader::Next(Camera *camera, Image *image,
                                      std::vector<Bitmap> *bitmap, 
									  std::vector<Bitmap> *mask,
									  std::vector<std::string> *bitmap_paths, 
									  bool info_only) {
	CHECK_NOTNULL(camera);
	CHECK_NOTNULL(image);
	CHECK_NOTNULL(bitmap);

	std::vector<std::string> image_paths;

    image_index_ += 1;
	CHECK_LE(image_index_ - inital_image_index_, options_.image_list.size()/options_.num_local_cameras);

	const std::string image_path0 = options_.image_list.at((image_index_ - inital_image_index_ - 1)*
		options_.num_local_cameras + 0);

	////////////////////////////////////////////////////////////////////////////
	// Set the image name. If this is a multi camera system, the image name is
	// selected as the name of the first image
	////////////////////////////////////////////////////////////////////////////

	image->SetName(image_path0);
	image->SetName(StringReplace(image->Name(), "\\", "/"));
	image->SetName(
			image->Name().substr(
					options_.image_path.size(),
					image->Name().size() - options_.image_path.size()));
	
	// If the image id has already been specified
	if(!options_.image_name_id_map.empty()){
		if(options_.image_name_id_map.count(image->Name())){
			image->SetImageId(options_.image_name_id_map[image->Name()]);
		}else{
		    std::cout << "Image name not exist ... " << image->Name() << std::endl;
			return ImageReader::Status::FAILURE;
		}
	}else{
    	image->SetImageId(image_index_);
	}
	
	const std::string image_folder = GetParentDir(image->Name());

	//////////////////////////////////////////////////////////////////////////////
	// Load Image label from image or file
	//////////////////////////////////////////////////////////////////////////////

	std::string current_image_dir = image_path0.substr(0, image_path0.rfind ("/"));
	auto iter = options_.image_label_list.find(current_image_dir);
	CHECK(iter != options_.image_label_list.end()) << "Image do not contain label!!!";
	label_t current_label = iter->second;
	image->SetLabelId(current_label);

	image_paths.push_back(image_path0);
	for(int i = 1; i<options_.num_local_cameras; ++i){
		const std::string image_path = options_.image_list.at(
							(image_index_ - inital_image_index_- 1)*options_.num_local_cameras+i);
		image_paths.push_back(image_path);
	}
	*bitmap_paths = image_paths;

	std::vector<std::string> local_image_names(options_.num_local_cameras);
	for (int i = 0; i < options_.num_local_cameras; ++i) {
		auto image_path = image_paths.at(i);
		auto image_name = StringReplace(image_path, "\\", "/");
		auto local_image_name = image_name.substr(options_.image_path.size(), image_name.size() - options_.image_path.size());
		local_image_names[i] = local_image_name;	
	}
	image->SetLocalNames(local_image_names);

	if(info_only && image_index_ > inital_image_index_ + 1){
		if (options_.single_camera_per_folder && image_folders_.count(image_folder) == 0) {
            camera_index_++;
            prev_camera_.SetCameraId(camera_index_);

            for (const auto& device_local_params : options_.local_camera_params) {
                if (image_folder.find(device_local_params.first) != std::string::npos) {
                    prev_camera_.SetModelIdFromName(options_.local_camera_models.at(device_local_params.first));
                    prev_camera_.SetLocalCameraIntrinsicParamsFromString(
                        options_.local_camera_params.at(device_local_params.first));
                    prev_camera_.SetLocalCameraExtrinsicParamsFromString(
                        options_.local_camera_extrinsics.at(device_local_params.first));
					break;
				}
			}
        }

        image_folders_.insert(image_folder);
		prev_image_folder_ = image_folder;

		// prev_camera_.SetLocalPoseDiff();
        image->SetCameraId(prev_camera_.CameraId());
		*camera = prev_camera_;

		return Status::SUCCESS;
	}



	std::vector<double> focal_lengths;
	size_t image_width  = 0;
	size_t image_height = 0;

	std::vector<Eigen::Vector4d> local_qvecs_prior;
	std::vector<Eigen::Vector3d> local_tvecs_prior;

	for(int camera_id = 0; camera_id < options_.num_local_cameras; ++camera_id){

		const std::string image_path = image_paths[camera_id];
		
		////////////////////////////////////////////////////////////////////////
		// Read image.
		////////////////////////////////////////////////////////////////////////

		if (IsFileRGBD(image_path)) {
			if (!ExtractRGBDData(image_path, (*bitmap)[camera_id], false)) {
				return Status::BITMAP_ERROR;
			}
		} else {
			if (!(*bitmap)[camera_id].Read(image_path, false)) {
				return Status::BITMAP_ERROR;
			}
		}
		
		if(camera_id>0){
			CHECK_EQ(image_width,(*bitmap)[camera_id].Width());
			CHECK_EQ(image_height,(*bitmap)[camera_id].Height());
		}
		else{
			image_width = (*bitmap)[camera_id].Width();
			image_height = (*bitmap)[camera_id].Height();
		}

		////////////////////////////////////////////////////////////////////////
		// Read mask.
		////////////////////////////////////////////////////////////////////////

		if (mask && !options_.mask_path.empty()) {
			std::vector<std::string> name_parts = StringSplit(image->LocalName(camera_id), ".");
			const std::string mask_path =
				JoinPaths(options_.mask_path, 
				           image->LocalName(camera_id).substr(0, image->LocalName(camera_id).size() - name_parts.back().size() - 1) +
				          ".png");
			if (ExistsFile(mask_path) && 
				!(*mask)[camera_id].Read(mask_path, false)) {
				// NOTE: Maybe introduce a separate error type MASK_ERROR?
				return Status::BITMAP_ERROR;
			}
		}

		////////////////////////////////////////////////////////////////////////
		// Check image dimensions.
		////////////////////////////////////////////////////////////////////////

		if (prev_camera_.CameraId() != kInvalidCameraId &&
	    	((options_.single_camera && !options_.single_camera_per_folder) ||
	     	 (options_.single_camera_per_folder &&
	      	  image_folder == prev_image_folder_)) &&
	    	 (prev_camera_.Width() != image_width ||
	     	  prev_camera_.Height() != image_height)) {
			
			return Status::CAMERA_SINGLE_DIM_ERROR;
		}

		////////////////////////////////////////////////////////////////////////
		// Extract camera model and focal length
		////////////////////////////////////////////////////////////////////////

		if ((options_.num_local_cameras==1&&options_.camera_params.empty())||
			(options_.num_local_cameras>1&&
			 options_.local_camera_params.empty())){
			// Extract focal length.
			double focal_length = 0.0;
			if ((*bitmap)[camera_id].ExifFocalLength(&focal_length)) {
				prev_camera_.SetPriorFocalLength(true);
			} else {
				focal_length = (image_width + image_height) * 0.5f;
				prev_camera_.SetPriorFocalLength(false);
			}
			focal_lengths.push_back(focal_length);
		}

		////////////////////////////////////////////////////////////////////////
		// Extract GPS data.
		////////////////////////////////////////////////////////////////////////
		double latitude, longitude, altitude;
		if (!(*bitmap)[camera_id].ExifLatitude(&latitude) ||
			!(*bitmap)[camera_id].ExifLongitude(&longitude) ||
			!(*bitmap)[camera_id].ExifAltitude(&altitude)) {
			if (camera_id == 0) {
				image->TvecPrior().setConstant(std::numeric_limits<double>::quiet_NaN());
			}
			local_tvecs_prior.push_back(Eigen::Vector3d().setConstant(std::numeric_limits<double>::quiet_NaN()));
		} else {
			std::cout << StringPrintf("Extract GPS(%d): %f %f %f\n", image_index_, latitude,
				longitude, altitude);
			// Eigen::Vector3d location = GPSReader::gpsToLocation(latitude, longitude, altitude);
			// image->TvecPrior() = location;

			if (!geo_converter){
				if(!options_.gps_origin.empty()){
					std::vector<double> gps_origin =  CSVToVector<double>(options_.gps_origin);
					if (gps_origin.size() == 3){
						geo_converter.reset(new GeodeticConverter(gps_origin[0], gps_origin[1], gps_origin[2]));
					} else {
						geo_image_index_ = image_index_;
						geo_converter.reset(new GeodeticConverter(latitude, longitude, altitude));
					}
				} else if (geo_image_index_ == -1) {
					geo_image_index_ = image_index_;
					geo_converter.reset(new GeodeticConverter(latitude, longitude, altitude));
				}
			}
			
			if (geo_converter) {
				double n, e, d;
				geo_converter->LLAToNed(latitude, longitude, altitude, &n, &e, &d);
				if (camera_id == 0) {
					image->TvecPrior(0) = n;
					image->TvecPrior(1) = e;
					image->TvecPrior(2) = d;
				}
				local_tvecs_prior.emplace_back(n, e, d);
			}

			if (camera_id == 0) {
				int8_t rtk_flg = -1;
				double rtk_std_lon;
				double rtk_std_lat;
				double rtk_std_hgt;
				if ((*bitmap)[camera_id].ExifRtkFlag(&rtk_flg) && 
					(*bitmap)[camera_id].ExifRktStdLon(&rtk_std_lon) &&
					(*bitmap)[camera_id].ExifRktStdLat(&rtk_std_lat) &&
					(*bitmap)[camera_id].ExifRktStdHgt(&rtk_std_hgt)){
					image->RtkFlag() = rtk_flg;
					image->SetRtkStd(rtk_std_lon, rtk_std_lat, rtk_std_hgt);
					std::cout << StringPrintf("Extract Std(%d): %d %f %f %f\n", image_index_,
						rtk_flg, rtk_std_lon, rtk_std_lat, rtk_std_hgt);
				} else if ((*bitmap)[camera_id].ExifRtkFlag(&rtk_flg)) {
					image->RtkFlag() = rtk_flg;
					std::cout << StringPrintf("Extract Rtk Flag(%d): %d\n", image_index_, rtk_flg);
				} else {
					image->RtkFlag() = 0;
				}
				double relative_altitude;
				if ((*bitmap)[camera_id].ExifRltAltitude(&relative_altitude)) {
					image->SetRelativeAltitude(relative_altitude);
					std::cout << StringPrintf("Extract RelativeAltitude(%d): %f\n", image_index_, relative_altitude);
				}
			}
		}
		double pitch, roll, yaw;
		if (!(*bitmap)[camera_id].ExifPitchDegree(&pitch) ||
			!(*bitmap)[camera_id].ExifRollDegree(&roll) ||
			!(*bitmap)[camera_id].ExifYawDegree(&yaw)) {
			if (camera_id == 0) {
				image->QvecPrior().setConstant(std::numeric_limits<double>::quiet_NaN());
			}
			local_qvecs_prior.push_back(Eigen::Vector4d().setConstant(std::numeric_limits<double>::quiet_NaN()));
		} else {
			std::cout << StringPrintf("Extract RTK(%d): %f %f %f\n", image_index_, yaw,
				pitch, roll);
			Eigen::AngleAxisf pitchAngle = 
				Eigen::AngleAxisf(pitch / 180 * M_PI, Eigen::Vector3f::UnitY());
			Eigen::AngleAxisf rollAngle = 
				Eigen::AngleAxisf(roll / 180 * M_PI, Eigen::Vector3f::UnitX());
			Eigen::AngleAxisf yawAngle = 
				Eigen::AngleAxisf(yaw / 180 * M_PI, Eigen::Vector3f::UnitZ());

			Eigen::Matrix3f rot;
			rot << 0.0,0.0,1.0,1.0,0.0,0.0,0.0,1.0,0.0;
			Eigen::Matrix3f R = yawAngle.matrix() * pitchAngle.matrix() * rollAngle.matrix() * rot;

			Eigen::Quaternionf q(R.inverse());
			if (camera_id == 0) {
				image->QvecPrior() = Eigen::Vector4d(q.w(), q.x(), q.y(), q.z());
			}
			local_qvecs_prior.emplace_back(q.w(), q.x(), q.y(), q.z());
		}
	}

	image->SetLocalQvecsPrior(local_qvecs_prior);
	image->SetLocalTvecsPrior(local_tvecs_prior);

	Camera prior_camera;
	CHECK_EQ(prior_camera.ModelId(), kInvalidCameraModelId);
	if (IsFileRGBD(image->Name())) {
		RGBDData data;
		if (ExtractRGBDData(JoinPaths(options_.image_path, image->Name()), RGBDReadOption::NoColorNoDepth(), data)) {
			if (data.color_camera.ModelName() == "PINHOLE" || 
				data.color_camera.ModelName() == "SIMPLE_PINHOLE"
			) {
				// If the RGBD camera is PINHOLE,
				// enable upgrading to other camera models for better optimization
				prior_camera.InitializeWithName(options_.camera_model,
												focal_lengths[0],
												image_width, image_height);

				double color_focal_length;
				double color_focal_length_x;
				double color_focal_length_y;
				if (data.color_camera.FocalLengthIdxs().size() == 1) {
					color_focal_length = data.color_camera.FocalLength();
					color_focal_length_x = color_focal_length;
					color_focal_length_y = color_focal_length;
				} else {
					color_focal_length_x = data.color_camera.FocalLengthX();
					color_focal_length_y = data.color_camera.FocalLengthY();
					color_focal_length = 0.5 * (color_focal_length_x + color_focal_length_y);
				}

				if (prior_camera.FocalLengthIdxs().size() == 1) {
					prior_camera.SetFocalLength(color_focal_length);
				} else {
					prior_camera.SetFocalLengthX(color_focal_length_x);
					prior_camera.SetFocalLengthY(color_focal_length_y);
				}

				prior_camera.SetPrincipalPointX(data.color_camera.PrincipalPointX());
				prior_camera.SetPrincipalPointY(data.color_camera.PrincipalPointY());
			} else {
				prior_camera = data.color_camera;
				prior_camera.SetWidth(image_width);
				prior_camera.SetHeight(image_height);
			}
		}
	}
	if (!options_.image_name_camera_map.empty() && options_.image_name_camera_map.count(image->Name())) {
		prior_camera = options_.image_name_camera_map[image->Name()];
	}
	if (!options_.subpath_camera_map.empty()) {
		for (auto & pair : options_.subpath_camera_map) {
			if (IsInsideSubpath(image->Name(), pair.first)) {
				prior_camera = pair.second;
				break;
			}
		}
	}

	if (prior_camera.ModelId() != kInvalidCameraModelId) {
		// If the camera has already been specified
		if (prev_camera_.CameraId() == kInvalidCameraId ||
		    prior_camera.ModelId() != prev_camera_.ModelId() || 
			prior_camera.Width() != prev_camera_.Width() || 
			prior_camera.Height() != prev_camera_.Height() || 
			prior_camera.ParamsToString() != prev_camera_.ParamsToString()
		) {
			prev_camera_ = prior_camera;

			camera_index_++;
			prev_camera_.SetCameraId(camera_index_);
		}
		prev_camera_.SetPriorFocalLength(true);
	} else if (prev_camera_.CameraId() == kInvalidCameraId ||
	    (!options_.single_camera && !options_.single_camera_per_folder &&
	    static_cast<camera_t>(options_.existing_camera_id) ==
	    kInvalidCameraId) ||
	    (options_.single_camera_per_folder &&
	    image_folders_.count(image_folder) == 0)) {

		dewarp_camera_index_ = kInvalidCameraId;
		warp_camera_index_ = kInvalidCameraId;
		
		if(prev_camera_.ModelName().compare("SPHERICAL")==0){
			prev_camera_.SetPrincipalPointX(static_cast<size_t>(image_width));
			prev_camera_.SetPrincipalPointY(static_cast<size_t>(image_height));
		}

		if ((options_.num_local_cameras==1&&options_.camera_params.empty())||
		    (options_.num_local_cameras>1&&
			 options_.local_camera_params.empty())) {

			CHECK(focal_lengths.size() >= 1);
			std::cout << "focal lengths: " << std::endl;
			for (auto focal_length : focal_lengths){
				std::cout << focal_length << " ";
			}
			std::cout << std::endl;

			if (options_.num_local_cameras == 1){
				std::vector<double> intrinsic;
				if ((*bitmap)[0].ExifIntrinsic(intrinsic)) {
					prev_camera_.InitializeWithName(options_.camera_model,
												focal_lengths[0],
												image_width, image_height);
                    prev_camera_.SetPriorFocalLength(true);
                    if (prev_camera_.NumParams() == intrinsic.size()) {
						memcpy(prev_camera_.ParamsData(), intrinsic.data(), intrinsic.size() * sizeof(double));
						std::cout << "camera param: " << prev_camera_.ParamsToString() << std::endl;
					} else if (prev_camera_.NumParams() < intrinsic.size()) {
						prev_camera_.SetFocalLengthX(intrinsic[0]);
						prev_camera_.SetFocalLengthY(intrinsic[1]);
						prev_camera_.SetPrincipalPointX(intrinsic[2]);
						prev_camera_.SetPrincipalPointY(intrinsic[3]);
					}
				} else {
					prev_camera_.InitializeWithId(prev_camera_.ModelId(),
												focal_lengths[0],
												image_width, image_height);
				}
				if (!prev_camera_.VerifyParams()){
					return Status::CAMERA_PARAM_ERROR;
				}
			}
			else{
				prev_camera_.InitializeLocalCameraIntricsWithId(
					prev_camera_.ModelId(),
					focal_lengths,
					image_width, image_height);

				if (!prev_camera_.VerifyLocalParams()){
					return Status::CAMERA_PARAM_ERROR;
				}
			}
		}
		
		prev_camera_.SetWidth(image_width);
		prev_camera_.SetHeight(image_height);
	
		camera_index_++;
		prev_camera_.SetCameraId(camera_index_);
		
		for (const auto& device_local_params : options_.local_camera_params) {
            if (image_folder.find(device_local_params.first) != std::string::npos) {
                prev_camera_.SetModelIdFromName(options_.local_camera_models.at(device_local_params.first));
                prev_camera_.SetLocalCameraIntrinsicParamsFromString(
                    options_.local_camera_params.at(device_local_params.first));
                prev_camera_.SetLocalCameraExtrinsicParamsFromString(
                    options_.local_camera_extrinsics.at(device_local_params.first));
				break;
			}
		}

		if ((*bitmap)[0].HasDewarp()) {
			dewarp_camera_index_ = camera_index_;
			cameras_map_[dewarp_camera_index_] = prev_camera_;
		} else {
			warp_camera_index_ = camera_index_;
			cameras_map_[warp_camera_index_] = prev_camera_;
		}
	} else if ((options_.single_camera || options_.single_camera_per_folder) &&
		options_.camera_params.empty()) {
		if ((*bitmap)[0].HasDewarp()) {
			if (dewarp_camera_index_ != kInvalidCameraId) {
				prev_camera_ = cameras_map_.at(dewarp_camera_index_);
			} else {
				prev_camera_.InitializeWithName(options_.camera_model,
												focal_lengths[0],
												image_width, image_height);
				prev_camera_.SetWidth(image_width);
				prev_camera_.SetHeight(image_height);
				
				camera_index_++;
				prev_camera_.SetCameraId(camera_index_);
				dewarp_camera_index_ = camera_index_;
				cameras_map_[dewarp_camera_index_] = prev_camera_;
			}
			if (!prev_camera_.VerifyParams()){
				return Status::CAMERA_PARAM_ERROR;
			}
		} else {
			if (warp_camera_index_ != kInvalidCameraId) {
				prev_camera_ = cameras_map_.at(warp_camera_index_);
			} else {
				std::vector<double> intrinsic;
				if ((*bitmap)[0].ExifIntrinsic(intrinsic)) {
					prev_camera_.InitializeWithName(options_.camera_model,
												focal_lengths[0],
												image_width, image_height);
                                        prev_camera_.SetPriorFocalLength(true);
                                        if (prev_camera_.NumParams() == intrinsic.size()) {
						memcpy(prev_camera_.ParamsData(), intrinsic.data(), intrinsic.size() * sizeof(double));
						std::cout << "camera param: " << prev_camera_.ParamsToString() << std::endl;
					}
				} else {
					prev_camera_.InitializeWithName(options_.camera_model,
													focal_lengths[0],
													image_width, image_height);
				}
				prev_camera_.SetWidth(image_width);
				prev_camera_.SetHeight(image_height);
			
				camera_index_++;
				prev_camera_.SetCameraId(camera_index_);
				warp_camera_index_ = camera_index_;
				cameras_map_[warp_camera_index_] = prev_camera_;
			}
		}
	}

	image->SetCameraId(prev_camera_.CameraId());

	*camera = prev_camera_;
	if (options_.fixed_camera) {
		camera->SetCameraConstant(true);
	}

	// camera->SetLocalPoseDiff();

	image_folders_.insert(image_folder);
	prev_image_folder_ = image_folder;

	return Status::SUCCESS;
}

size_t ImageReader::NextIndex() const { return image_index_; }

size_t ImageReader::NumImages() const { return options_.image_list.size()/options_.num_local_cameras; }

size_t ImageReader::InitialIndex() const { return inital_image_index_; }

size_t ImageReader::GeoImageIndex() const { return geo_image_index_; }

}  // namespace sensemap
