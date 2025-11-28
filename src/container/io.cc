//Copyright (c) 2019, SenseTime Group.
//All rights reserved.

#include "io.h"

#include <fstream>

#include "util/bitmap.h"
#include "util/misc.h"
#include "util/rgbd_helper.h"

namespace sensemap {

void ReadCamerasText(const std::string& path, CameraPtrUmap *camera_data_ump,
					 bool camera_rig = false) {
	camera_data_ump->clear();

	std::ifstream file(path);
	CHECK(file.is_open()) << path;

	std::string line;
	std::string item;

	while (std::getline(file, line)) {
		StringTrim(&line);

		if (line.empty() || line[0] == '#') {
			continue;
		}

		std::stringstream line_stream(line);

		class Camera camera;

		// ID
		std::getline(line_stream, item, ' ');
		camera.SetCameraId(std::stoul(item));

		// MODEL
		std::getline(line_stream, item, ' ');
		camera.SetModelIdFromName(item);

		// WIDTH
		std::getline(line_stream, item, ' ');
		camera.SetWidth(std::stoll(item));

		// HEIGHT
		std::getline(line_stream, item, ' ');
		camera.SetHeight(std::stoll(item));

		// PARAMS
		camera.Params().clear();
		while (!line_stream.eof()) {
			std::getline(line_stream, item, ' ');
			camera.Params().push_back(std::stold(item));
		}
		camera.SetPriorFocalLength(true);
		CHECK(camera.VerifyParams());

		camera.SetNumLocalCameras(1);

		if (camera_rig) {
			// for camera-rig
			// Num Local Camera
			std::getline(line_stream, item, ' ');
			local_camera_t num_local_camera = std::stoi(item);
			camera.SetNumLocalCameras(num_local_camera);

			// Intrinsics
			std::getline(line_stream, item, ' ');
			size_t local_params_size = std::stoi(item);
			camera.LocalParams().resize(local_params_size);
			std::vector<double>& local_params = camera.LocalParams();
			for (size_t i = 0; i < local_params_size; ++i) {
				std::getline(line_stream, item, ' ');
				local_params[i] = std::stod(item);
			}
			
			// Local qvecs
			size_t local_qvecs_size,local_tvecs_size;
			std::getline(line_stream, item, ' ');
			local_qvecs_size = std::stoi(item);
			camera.LocalQvecs().resize(local_qvecs_size);
			std::vector<double>& local_qvecs = camera.LocalQvecs();
			for (size_t i = 0; i < local_qvecs_size; ++i) {
				std::getline(line_stream, item, ' ');
				local_qvecs[i] = std::stod(item);
			}

			// Local tvecs
			std::getline(line_stream, item, ' ');
			local_tvecs_size = std::stoi(item);
			camera.LocalTvecs().resize(local_tvecs_size);
			std::vector<double>& local_tvecs = camera.LocalTvecs();
			for (size_t i = 0; i < local_tvecs_size; ++i) {
				std::getline(line_stream, item, ' ');
				local_tvecs[i] = std::stod(item);
			}

			CHECK(camera.VerifyLocalParams());
			camera.SetPriorFocalLength(true);
		}

		camera_data_ump->emplace(camera.CameraId(), std::make_shared<Camera>(camera));
	}
	file.close();
}

void WriteCamerasText(const std::string& path,
                      const CameraPtrUmap& camera_data_ump) {
	std::ofstream file(path, std::ios::trunc);
	CHECK(file.is_open()) << path;

	file << "# Camera list with one line of data per camera:" << std::endl;
	file << "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]" << std::endl;
	file << "# Number of cameras: " << camera_data_ump.size() << std::endl;

	for (const auto& camera : camera_data_ump) {
		std::ostringstream line;

		line << camera.first << " ";
		line << camera.second->ModelName() << " ";
		line << camera.second->Width() << " ";
		line << camera.second->Height() << " ";

		for (const double param : camera.second->Params()) {
			line << param << " ";
		}

		// for camera-rig
		if (camera.second->NumLocalCameras() > 1) {
			line << camera.second->NumLocalCameras() << " ";
			// local intrinsics
			line << camera.second->LocalParams().size() << " ";
			for (const double local_param: camera.second->LocalParams()){
				line << local_param << " ";
			}
			// local qvecs
			line << camera.second->LocalQvecs().size() << " ";
			for (const double local_qvec: camera.second->LocalQvecs()){
				line << local_qvec << " ";
			}
			// local tvecs
			line << camera.second->LocalTvecs().size() << " ";
			for (const double local_tvec : camera.second->LocalTvecs()){
				line << local_tvec << " ";
			}
		}

		std::string line_string = line.str();
		line_string = line_string.substr(0, line_string.size() - 1);

		file << line_string << std::endl;
	}
	file.close();
}

void ReadFeatureDataText(const std::string &path, std::string* image_path,
                         FeatureDataPtrUmap *feature_data_ump,
                         FeatureNameUmap* feature_data_name) {
	std::ifstream file(path.c_str());
	CHECK(file.is_open()) << path;

	std::string line;
	std::string item;

	while (std::getline(file, line)) {
		StringTrim(&line);

		if (line.empty() || line[0] == '#') {
			continue;
		} else{
			break;
		}
	}

	std::stringstream header_line_stream(line);
	//image_path
	std::getline(header_line_stream >> std::ws, item, ' ');
	*image_path = item;

	std::getline(header_line_stream >> std::ws, item, ' ');
	const image_t num_images = std::stoul(item);

	feature_data_ump->clear();
	feature_data_ump->reserve(num_images);
	for (size_t i = 0; i < num_images; ++i) {

		FeatureDataPtr feature_data_ptr = std::make_shared<FeatureData>();
		std::getline(file, line);
		std::stringstream image_line_stream(line);

		//image
		std::getline(image_line_stream >> std::ws, item, ' ');
		feature_data_ptr->image.SetImageId(std::stol(item));
		std::getline(image_line_stream >> std::ws, item, ' ');
		feature_data_ptr->image.SetCameraId(std::stol(item));
		std::getline(image_line_stream >> std::ws, item, ' ');
		feature_data_ptr->image.SetName(item);
		std::getline(image_line_stream >> std::ws, item, ' ');
		feature_data_ptr->image.SetLabelId(std::stol(item));
		std::getline(image_line_stream >> std::ws, item, ' ');
		const point2D_t num_features = std::stoul(item);
		std::getline(image_line_stream >> std::ws, item, ' ');
		const size_t dim = std::stoul(item);
		CHECK_EQ(dim, 128) << "SIFT features must have 128 dimensions";

		feature_data_ptr->keypoints.resize(num_features);
		feature_data_ptr->descriptors.resize(num_features, dim);

		for (size_t i = 0; i < num_features; ++i) {
			std::getline(file, line);
			std::stringstream feature_line_stream(line);

			//keypoint
			std::getline(feature_line_stream >> std::ws, item, ' ');
			const float x = std::stold(item);
			std::getline(feature_line_stream >> std::ws, item, ' ');
			const float y = std::stold(item);
			std::getline(feature_line_stream >> std::ws, item, ' ');
			const float scale = std::stold(item);
			std::getline(feature_line_stream >> std::ws, item, ' ');
			const float orientation = std::stold(item);
			feature_data_ptr->keypoints[i] =
					FeatureKeypoint(x, y, scale, orientation);

			// Descriptor
			for (size_t j = 0; j < dim; ++j) {
				std::getline(feature_line_stream >> std::ws, item, ' ');
				const float value = std::stod(item);
				CHECK_GE(value, 0);
				CHECK_LE(value, 255);
				feature_data_ptr->descriptors(i, j) =
						TruncateCast<float, uint8_t>(value);
			}
		}
		feature_data_ump->emplace(feature_data_ptr->image.ImageId(),
		                          feature_data_ptr);
        feature_data_name->emplace(feature_data_ptr->image.Name(),
                                   feature_data_ptr->image.ImageId());
		std::cout << "\rLoad Feature Data [ " << i+1 << " / " << num_images << "]" << std::flush; 
	}
	std::cout << "\n";
	file.close();
}

void ReadLocalCamerasText(const std::string& path, CameraPtrUmap *camera_data_ump) {
	std::ifstream file(path);
	CHECK(file.is_open()) << path;

	std::string line;
	std::string item;

	while (std::getline(file, line)) {
		StringTrim(&line);

		if (line.empty() || line[0] == '#') {
			continue;
		}

		std::stringstream line_stream(line);

		// ID
		std::getline(line_stream, item, ' ');
		camera_t camera_id = std::stoul(item);
		auto& camera = camera_data_ump->at(camera_id);

		// for camera-rig
		// Num Local Camera
		std::getline(line_stream, item, ' ');
		local_camera_t num_local_camera = std::stoi(item);
		camera->SetNumLocalCameras(num_local_camera);

		if (num_local_camera > 1) {
			// Intrinsics
			std::getline(line_stream, item, ' ');
			size_t local_params_size = std::stoi(item);
			camera->LocalParams().resize(local_params_size);
			std::vector<double>& local_params = camera->LocalParams();
			for (size_t i = 0; i < local_params_size; ++i) {
				std::getline(line_stream, item, ' ');
				local_params[i] = std::stod(item);
			}
			
			// Local qvecs
			size_t local_qvecs_size,local_tvecs_size;
			std::getline(line_stream, item, ' ');
			local_qvecs_size = std::stoi(item);
			camera->LocalQvecs().resize(local_qvecs_size);
			std::vector<double>& local_qvecs = camera->LocalQvecs();
			for (size_t i = 0; i < local_qvecs_size; ++i) {
				std::getline(line_stream, item, ' ');
				local_qvecs[i] = std::stod(item);
			}

			// Local tvecs
			std::getline(line_stream, item, ' ');
			local_tvecs_size = std::stoi(item);
			camera->LocalTvecs().resize(local_tvecs_size);
			std::vector<double>& local_tvecs = camera->LocalTvecs();
			for (size_t i = 0; i < local_tvecs_size; ++i) {
				std::getline(line_stream, item, ' ');
				local_tvecs[i] = std::stod(item);
			}

			CHECK(camera->VerifyLocalParams());
			camera->SetPriorFocalLength(true);
		}
	}
	file.close();
}

void WriteLocalCamerasText(const std::string& path, const CameraPtrUmap& camera_data_ump) {
	std::ofstream file(path, std::ios::trunc);
	CHECK(file.is_open()) << path;

	for (const auto& camera : camera_data_ump) {
		std::ostringstream line;

		line << camera.first << " ";
		line << camera.second->NumLocalCameras() << " ";

		// for camera-rig
		if (camera.second->NumLocalCameras() > 1) {
			// local intrinsics
			line << camera.second->LocalParams().size() << " ";
			for (const double local_param: camera.second->LocalParams()){
				line << local_param << " ";
			}
			// local qvecs
			line << camera.second->LocalQvecs().size() << " ";
			for (const double local_qvec: camera.second->LocalQvecs()){
				line << local_qvec << " ";
			}
			// local tvecs
			line << camera.second->LocalTvecs().size() << " ";
			for (const double local_tvec : camera.second->LocalTvecs()){
				line << local_tvec << " ";
			}
		}

		std::string line_string = line.str();
		line_string = line_string.substr(0, line_string.size() - 1);

		file << line_string << std::endl;
	}
	file.close();
}

void ReadCamerasBinary(const std::string& path, 
					   CameraPtrUmap *camera_data_ump,
					   bool camera_rig = false) {
	camera_data_ump->clear();

    std::ifstream file(path, std::ios::binary);
    CHECK(file.is_open()) << path;

    const size_t num_cameras = ReadBinaryLittleEndian<uint64_t>(&file);
    for (size_t i = 0; i < num_cameras; ++i) {
        class Camera camera;
        camera.SetCameraId(ReadBinaryLittleEndian<camera_t>(&file));
        camera.SetModelId(ReadBinaryLittleEndian<int>(&file));
        camera.SetWidth(ReadBinaryLittleEndian<uint64_t>(&file));
        camera.SetHeight(ReadBinaryLittleEndian<uint64_t>(&file));
        ReadBinaryLittleEndian<double>(&file, &camera.Params());
		CHECK(camera.VerifyParams());

		camera.SetNumLocalCameras(1);

		if (camera_rig) {
			local_camera_t num_local_camera = 
				ReadBinaryLittleEndian<local_camera_t>(&file);
			// for camera-rig
			camera.SetNumLocalCameras(num_local_camera);

			// Intrinsics
			size_t local_params_size = ReadBinaryLittleEndian<size_t>(&file);
			camera.LocalParams().resize(local_params_size);
			ReadBinaryLittleEndian<double>(&file,&camera.LocalParams());
			
			// Local qvecs
			size_t local_qvecs_size,local_tvecs_size;
			local_qvecs_size = ReadBinaryLittleEndian<size_t>(&file);
			camera.LocalQvecs().resize(local_qvecs_size);
			ReadBinaryLittleEndian<double>(&file, &camera.LocalQvecs());

			// Local tvecs
			local_tvecs_size = ReadBinaryLittleEndian<size_t>(&file);
			camera.LocalTvecs().resize(local_tvecs_size);
			ReadBinaryLittleEndian<double>(&file, &camera.LocalTvecs());

			CHECK(camera.VerifyLocalParams());
		}
		// camera.SetPriorFocalLength(true);
		//camera.SetCameraConstant(true);/////for test only
		camera_data_ump->emplace(camera.CameraId(), std::make_shared<Camera>(camera));
    }
    file.close();
}

void WriteCamerasBinary(const std::string& path, const CameraPtrUmap& camera_data_ump) {
    std::ofstream file(path, std::ios::binary);
    CHECK(file.is_open()) << path;

    WriteBinaryLittleEndian<uint64_t>(&file, camera_data_ump.size());

    for (const auto& camera_data : camera_data_ump) {
		auto camera = camera_data.second;
        WriteBinaryLittleEndian<camera_t>(&file, camera_data.first);
        WriteBinaryLittleEndian<int>(&file, camera->ModelId());
        WriteBinaryLittleEndian<uint64_t>(&file, camera->Width());
        WriteBinaryLittleEndian<uint64_t>(&file, camera->Height());
        for (const double param : camera->Params()) {
        	WriteBinaryLittleEndian<double>(&file, param);
        }

		// // for camera-rig
		// if (camera->NumLocalCameras() > 1) {
		// 	WriteBinaryLittleEndian<local_camera_t>(&file,camera->NumLocalCameras());
		// 	// local intrinsics
		// 	WriteBinaryLittleEndian<size_t>(&file,camera->LocalParams().size());
		// 	for (const double local_param: camera->LocalParams()){
		// 		WriteBinaryLittleEndian<double>(&file, local_param);
		// 	}
		// 	// local qvecs
		// 	WriteBinaryLittleEndian<size_t>(&file,camera->LocalQvecs().size());
		// 	for (const double local_qvec: camera->LocalQvecs()){
		// 		WriteBinaryLittleEndian<double>(&file, local_qvec);
		// 	}
		// 	// local tvecs
		// 	WriteBinaryLittleEndian<size_t>(&file, camera->LocalTvecs().size());
		// 	for (const double local_tvec : camera->LocalTvecs()){
		// 		WriteBinaryLittleEndian<double>(&file, local_tvec);
		// 	}
		// }
    }
    file.close();
}

void ReadLocalCamerasBinary(const std::string& path, CameraPtrUmap *camera_data_ump) {
    std::ifstream file(path, std::ios::binary);
    CHECK(file.is_open()) << path;

    const size_t num_cameras = ReadBinaryLittleEndian<uint64_t>(&file);
    for (size_t i = 0; i < num_cameras; ++i) {
        camera_t camera_id = ReadBinaryLittleEndian<camera_t>(&file);
		local_camera_t num_local_camera = ReadBinaryLittleEndian<local_camera_t>(&file);
		auto& camera = camera_data_ump->at(camera_id);

		if (num_local_camera > 1) {
			// for camera-rig
			camera->SetNumLocalCameras(num_local_camera);

			// Intrinsics
			size_t local_params_size = ReadBinaryLittleEndian<size_t>(&file);
			camera->LocalParams().resize(local_params_size);
			ReadBinaryLittleEndian<double>(&file, &camera->LocalParams());
			
			// Local qvecs
			size_t local_qvecs_size,local_tvecs_size;
			local_qvecs_size = ReadBinaryLittleEndian<size_t>(&file);
			camera->LocalQvecs().resize(local_qvecs_size);
			ReadBinaryLittleEndian<double>(&file, &camera->LocalQvecs());

			// Local tvecs
			local_tvecs_size = ReadBinaryLittleEndian<size_t>(&file);
			camera->LocalTvecs().resize(local_tvecs_size);
			ReadBinaryLittleEndian<double>(&file, &camera->LocalTvecs());

			CHECK(camera->VerifyLocalParams());
		}
		camera->SetPriorFocalLength(true);
    }
    file.close();
}

void WriteLocalCamerasBinary(const std::string& path, const CameraPtrUmap& camera_data_ump) {
	std::ofstream file(path, std::ios::binary);
    CHECK(file.is_open()) << path;

    WriteBinaryLittleEndian<uint64_t>(&file, camera_data_ump.size());

    for (const auto& camera_data : camera_data_ump) {
		const auto camera = camera_data.second;
        WriteBinaryLittleEndian<camera_t>(&file, camera_data.first);
		WriteBinaryLittleEndian<local_camera_t>(&file,camera->NumLocalCameras());
		if (camera->NumLocalCameras() > 1) {
			// for camera-rig
			// local intrinsics
			WriteBinaryLittleEndian<size_t>(&file,camera->LocalParams().size());
			for (const double local_param: camera->LocalParams()){
				WriteBinaryLittleEndian<double>(&file, local_param);
			}
			// local qvecs
			WriteBinaryLittleEndian<size_t>(&file,camera->LocalQvecs().size());
			for (const double local_qvec: camera->LocalQvecs()){
				WriteBinaryLittleEndian<double>(&file, local_qvec);
			}
			// local tvecs
			WriteBinaryLittleEndian<size_t>(&file, camera->LocalTvecs().size());
			for (const double local_tvec : camera->LocalTvecs()){
				WriteBinaryLittleEndian<double>(&file, local_tvec);
			}
		}
    }
    file.close();
}

void ReadLocalImagesBinary(const std::string& path, FeatureDataPtrUmap *image_data_ump) {
	std::ifstream file(path, std::ios::binary);
    CHECK(file.is_open()) << path;

    const size_t num_images = ReadBinaryLittleEndian<uint64_t>(&file);
    for (size_t i = 0; i < num_images; ++i) {
        image_t image_id = ReadBinaryLittleEndian<image_t>(&file);
		auto& image = image_data_ump->at(image_id)->image;

		local_camera_t num_local_tvec_priors = ReadBinaryLittleEndian<local_camera_t>(&file);
		auto& local_tvecs_prior = image.LocalTvecsPrior();
		for (local_camera_t local_image_id = 0; local_image_id < num_local_tvec_priors; ++local_image_id) {
			Eigen::Vector3d tvec;
			tvec.x() = ReadBinaryLittleEndian<double>(&file);
			tvec.y() = ReadBinaryLittleEndian<double>(&file);
			tvec.z() = ReadBinaryLittleEndian<double>(&file);
			local_tvecs_prior.push_back(tvec);
		}

		local_camera_t num_local_qvec_priors = ReadBinaryLittleEndian<local_camera_t>(&file);
		auto& local_qvecs_prior = image.LocalQvecsPrior();
		for (local_camera_t local_image_id = 0; local_image_id < num_local_qvec_priors; ++local_image_id) {
			Eigen::Vector4d qvec;
			qvec.w() = ReadBinaryLittleEndian<double>(&file);
			qvec.x() = ReadBinaryLittleEndian<double>(&file);
			qvec.y() = ReadBinaryLittleEndian<double>(&file);
			qvec.z() = ReadBinaryLittleEndian<double>(&file);
			local_qvecs_prior.push_back(qvec);
		}

		local_camera_t num_local_image_names = ReadBinaryLittleEndian<local_camera_t>(&file);
		std::vector<std::string> local_image_names(num_local_image_names, "");
		for (local_camera_t local_image_id = 0; local_image_id < num_local_image_names; ++local_image_id) {
			char image_name_char;
			std::string image_name;
			do {
				file.read(&image_name_char, 1);
				if (image_name_char != '\0') {
					image_name += image_name_char;
				}
			} while (image_name_char != '\0');
			local_image_names[local_image_id] = image_name;
		}
		image.SetLocalNames(local_image_names);
    }
    file.close();
}

void WriteLocalImagesBinary(const std::string& path, const FeatureDataPtrUmap& image_data_ump) {
	std::ofstream file(path, std::ios::binary);
    CHECK(file.is_open()) << path;

    WriteBinaryLittleEndian<uint64_t>(&file, image_data_ump.size());

    for (const auto& image_data : image_data_ump) {
		const auto image = image_data.second->image;
        WriteBinaryLittleEndian<image_t>(&file, image_data.first);

		const auto & local_tvec_priors = image.LocalTvecsPrior();

		WriteBinaryLittleEndian<local_camera_t>(&file, local_tvec_priors.size());
		for (size_t local_image_id = 0; local_image_id < local_tvec_priors.size(); ++local_image_id) {
			const Eigen::Vector3d& tvec = local_tvec_priors[local_image_id];
			WriteBinaryLittleEndian<double>(&file, tvec.x());
			WriteBinaryLittleEndian<double>(&file, tvec.y());
			WriteBinaryLittleEndian<double>(&file, tvec.z());
		}

		const auto & local_qvec_priors = image.LocalQvecsPrior();

		WriteBinaryLittleEndian<local_camera_t>(&file, local_qvec_priors.size());
		for (size_t local_image_id = 0; local_image_id < local_qvec_priors.size(); ++local_image_id) {
			const Eigen::Vector4d& qvec = local_qvec_priors[local_image_id];
			WriteBinaryLittleEndian<double>(&file, qvec.w());
			WriteBinaryLittleEndian<double>(&file, qvec.x());
			WriteBinaryLittleEndian<double>(&file, qvec.y());
			WriteBinaryLittleEndian<double>(&file, qvec.z());
		}

		WriteBinaryLittleEndian<local_camera_t>(&file, image_data.second->bitmap_paths.size());
		for (size_t local_image_id = 0; local_image_id < image_data.second->bitmap_paths.size(); ++local_image_id) {
			const std::string image_name = image.LocalName(local_image_id) + '\0';
			file.write(image_name.c_str(), image_name.size());
		}
    }
    file.close();
}

void WriteFeatureDataText(const std::string &path,
                          const std::string& image_path,
                          const FeatureDataPtrUmap &feature_data_ump) {
	std::ofstream file(path, std::ios::trunc);
	CHECK(file.is_open()) << path;

	file << "# IMAGE_PATH NUM_IMAGE" << std::endl;
	file << "## Image list with NUM_FEATURES lines of feature data per image:"
	     << std::endl;
	file << "# IMAGE_ID, CAMERA_ID, NAME, LABEL, NUM_FEATURES DIM " << std::endl;
	file << "# X Y SCALE ORIENTATION D_1 D_2 D_3 ... D_DIM" << std::endl;
	file << "# ..." << std::endl;
	file << "# X Y SCALE ORIENTATION D_1 D_2 D_3 ... D_DIM" << std::endl;

	file << image_path << " " << feature_data_ump.size() <<std::endl;
	std::cout <<"image size : " << feature_data_ump.size() <<std::endl;

	// Order the feature data ump
	std::vector< std::pair<image_t, FeatureDataPtr> > image_neighbors_sorted(feature_data_ump.begin(), feature_data_ump.end());
	sort(image_neighbors_sorted.begin(), image_neighbors_sorted.end(),
		[](const std::pair<image_t, FeatureDataPtr>& lhs, const std::pair<image_t, FeatureDataPtr>& rhs){
			return lhs.first < rhs.first;
		});

	int feature_counter = 0;
	for (const auto& feature_data : image_neighbors_sorted) {
		std::ostringstream line;
		std::string line_string;
		const auto &data_ptr = feature_data.second;
		line << feature_data.first << " ";
		line << data_ptr->image.CameraId() << " ";
		line << data_ptr->image.Name() << " ";
		line << data_ptr->image.LabelId() << " ";
		line << data_ptr->keypoints.size() << " " ;
		line << data_ptr->descriptors.cols();
		file << line.str() << std::endl;
		for(auto i = 0; i <data_ptr->keypoints.size(); ++i) {
			line.str("");
			line.clear();
			line << data_ptr->keypoints[i].x << " ";
			line << data_ptr->keypoints[i].y << " ";
			line << data_ptr->keypoints[i].ComputeScale() << " ";
			line << data_ptr->keypoints[i].ComputeOrientation() << " ";

			for (int j = 0; j < data_ptr->descriptors.cols(); ++j) {
				line << static_cast<float>(data_ptr->descriptors(i, j)) << " ";
			}
			line_string = line.str();
			line_string = line_string.substr(0, line_string.size() - 1);
			file << line_string << std::endl;
		}
		std::cout << "\rWrite Feature Data [ " << feature_counter+1 << " / " << image_neighbors_sorted.size() << "]" << std::flush;
		feature_counter++; 
	}
	std::cout << "\n";
	file.close();
}

void WriteFeatureDataBinary(const std::string &path,
                          const std::string& image_path,
                          const FeatureDataPtrUmap &feature_data_ump, bool write_descriptors) {
	std::ofstream file(path, std::ios::binary);
	CHECK(file.is_open()) << path;

	const std::string image_path_m = image_path + '\0';
	file.write(image_path_m.c_str(), image_path_m.size());

	image_t image_size = feature_data_ump.size();
	file.write((char*)&image_size, sizeof(image_t));


	// Order the feature data ump
	std::vector< std::pair<image_t, FeatureDataPtr> > image_neighbors_sorted(feature_data_ump.begin(), feature_data_ump.end());
	sort(image_neighbors_sorted.begin(), image_neighbors_sorted.end(),
		[](const std::pair<image_t, FeatureDataPtr>& lhs, const std::pair<image_t, FeatureDataPtr>& rhs){
			return lhs.first < rhs.first;
		});
	int feature_counter = 0;
	for (const auto& feature_data : image_neighbors_sorted) {
		const auto &data_ptr = feature_data.second;
		image_t image_id = feature_data.first;
		file.write((char*)&image_id, sizeof(image_t));

		camera_t camera_id = data_ptr->image.CameraId();
		file.write((char*)&camera_id, sizeof(camera_t));

		const std::string image_name = data_ptr->image.Name() + '\0';
		file.write(image_name.c_str(), image_name.size());

		label_t image_label = data_ptr->image.LabelId();
		file.write((char*)&image_label, sizeof(label_t));

		point2D_t keypoint_number = data_ptr->keypoints.size();
		file.write((char*)&keypoint_number, sizeof(point2D_t));

		size_t dim;
		if(!write_descriptors){
			dim = 64;
		}
		else if(keypoint_number > 0){ 
			dim = data_ptr->compressed_descriptors.cols();
		}
		else{
			dim = 64;
		}
			
		file.write((char*)&dim, sizeof(size_t));
		for(auto i = 0; i <data_ptr->keypoints.size(); ++i) {
			float x, y, scale, orientation;
			x = data_ptr->keypoints[i].x;
			y = data_ptr->keypoints[i].y;
			scale = data_ptr->keypoints[i].ComputeScale();
			orientation = data_ptr->keypoints[i].ComputeOrientation();

			file.write((char*)&x, sizeof(float));
			file.write((char*)&y, sizeof(float));
			file.write((char*)&scale, sizeof(float));
			file.write((char*)&orientation, sizeof(float));

			if(write_descriptors){
				// for (int j = 0; j < data_ptr->compressed_descriptors.cols(); ++j) {
				// 	auto des = data_ptr->compressed_descriptors(i, j);
				// 	file.write((char*)&des, sizeof(int8_t));
				// }
            	file.write((char*)(data_ptr->compressed_descriptors.row(i).data()),
                           sizeof(int8_t) * data_ptr->compressed_descriptors.cols());
            }
			else{
				for (int j = 0; j< dim; ++j){
					int8_t des = 0;
					file.write((char*)&des, sizeof(int8_t));
				}
			}
		}
		std::cout << "\rWrite Feature Data [ " << feature_counter+1 << " / " << image_neighbors_sorted.size() << "]" << std::flush;
		feature_counter++; 
	}
	std::cout << "\n";
	file.close();
}


void WriteBinaryDescriptorsIO(const std::string &path,
                          const FeatureDataPtrUmap &feature_data_ump) {
	std::ofstream file(path, std::ios::binary);
	CHECK(file.is_open()) << path;

	
	image_t image_size = feature_data_ump.size();
	file.write((char*)&image_size, sizeof(image_t));

	int feature_counter = 0;
	for (const auto& feature_data : feature_data_ump) {
        const auto& data_ptr = feature_data.second;

        image_t image_id = feature_data.first;
        file.write((char*)&image_id, sizeof(image_t));

		// keypoint number
        point2D_t num_features = data_ptr->keypoints.size();
        file.write((char*)&num_features, sizeof(point2D_t));	 
	 
		// length
		int length = data_ptr->binary_descriptors.cols();
		file.write((char*)&length, sizeof(int));

		for(auto i = 0; i <num_features; ++i) {
			for (int j = 0; j < data_ptr->binary_descriptors.cols(); ++j) {
				auto des = data_ptr->binary_descriptors(i, j);
				file.write((char*)&des, sizeof(uint64_t));
			}
		}
		std::cout << "\rWrite Feature Data [ " << feature_counter+1 << " / " << image_size << "]" << std::flush;
		feature_counter++;
	}

	std::cout << "\n";
	file.close();
}



void ReadFeatureDataBinary(const std::string &path, std::string* image_path,
                         FeatureDataPtrUmap *feature_data_ump,
                         FeatureNameUmap* feature_data_name) {
	std::ifstream file(path.c_str(), std::ios::binary);
	CHECK(file.is_open()) << path;

	//image_path
	char image_path_char;
	do {
		file.read(&image_path_char, 1);
		if (image_path_char != '\0') {
			*image_path += image_path_char;
		}
	} while (image_path_char != '\0');

	image_t num_images;
	file.read(reinterpret_cast<char*>(&num_images), sizeof(image_t));

	feature_data_ump->clear();
	feature_data_ump->reserve(num_images);
	for (size_t i = 0; i < num_images; ++i) {
		FeatureDataPtr feature_data_ptr = std::make_shared<FeatureData>();
		
		//image
		image_t image_id;
		file.read(reinterpret_cast<char*>(&image_id), sizeof(image_t));
		feature_data_ptr->image.SetImageId(image_id);
		
		camera_t camera_id;
		file.read(reinterpret_cast<char*>(&camera_id), sizeof(camera_t));
		feature_data_ptr->image.SetCameraId(camera_id);

		char image_name_char;
		std::string image_name;
		do {
			file.read(&image_name_char, 1);
			if (image_name_char != '\0') {
				image_name += image_name_char;
			}
		} while (image_name_char != '\0');
		feature_data_ptr->image.SetName(image_name);
		
		label_t label_id;
		file.read(reinterpret_cast<char*>(&label_id), sizeof(label_t));
		feature_data_ptr->image.SetLabelId(label_id);
		
		point2D_t num_features;
		file.read(reinterpret_cast<char*>(&num_features), sizeof(point2D_t));

		size_t dim;
		file.read(reinterpret_cast<char*>(&dim), sizeof(size_t));
		
		
		feature_data_ptr->keypoints.resize(num_features);
		
		feature_data_ptr->compressed_descriptors.resize(num_features, dim);
		
		for (size_t i = 0; i < num_features; ++i) {

			//keypoint
			float x, y, scale, orientation;
			file.read(reinterpret_cast<char*>(&x), sizeof(float));
			file.read(reinterpret_cast<char*>(&y), sizeof(float));
			file.read(reinterpret_cast<char*>(&scale), sizeof(float));
			file.read(reinterpret_cast<char*>(&orientation), sizeof(float));
			feature_data_ptr->keypoints[i] =
					FeatureKeypoint(x, y, scale, orientation);
			
			// Descriptor
			file.read(reinterpret_cast<char*>(feature_data_ptr->compressed_descriptors.row(i).data()),sizeof(int8_t)*dim);
			// for (size_t j = 0; j < dim; ++j) {
			// 	int8_t value;
			// 	file.read(reinterpret_cast<char*>(&value), sizeof(int8_t));
			// 	feature_data_ptr->compressed_descriptors(i, j) = value;
			// }
		}
		feature_data_ump->emplace(feature_data_ptr->image.ImageId(),
		                          feature_data_ptr);
        feature_data_name->emplace(feature_data_ptr->image.Name(),
                                   feature_data_ptr->image.ImageId());
		std::cout << "\rLoad Feature Data [ " << i+1 << " / " << num_images << "]" << std::flush; 
	}
	std::cout << "\n";
	file.close();
}

void ReadFeatureBinaryDataWithoutDescriptor(const std::string &path, std::string* image_path,
                         FeatureDataPtrUmap *feature_data_ump,
                         FeatureNameUmap* feature_data_name){
	std::ifstream file(path.c_str(), std::ios::binary);
	CHECK(file.is_open()) << path;

	//image_path
	char image_path_char;
	do {
		file.read(&image_path_char, 1);
		if (image_path_char != '\0') {
			*image_path += image_path_char;
		}
	} while (image_path_char != '\0');

	image_t num_images;
	file.read(reinterpret_cast<char*>(&num_images), sizeof(image_t));

	feature_data_ump->clear();
	feature_data_ump->reserve(num_images);
	for (size_t i = 0; i < num_images; ++i) {
		FeatureDataPtr feature_data_ptr = std::make_shared<FeatureData>();
		
		//image
		image_t image_id;
		file.read(reinterpret_cast<char*>(&image_id), sizeof(image_t));
		feature_data_ptr->image.SetImageId(image_id);
		
		camera_t camera_id;
		file.read(reinterpret_cast<char*>(&camera_id), sizeof(camera_t));
		feature_data_ptr->image.SetCameraId(camera_id);

		char image_name_char;
		std::string image_name;
		do {
			file.read(&image_name_char, 1);
			if (image_name_char != '\0') {
				image_name += image_name_char;
			}
		} while (image_name_char != '\0');
		feature_data_ptr->image.SetName(image_name);
		
		
		label_t label_id;
		file.read(reinterpret_cast<char*>(&label_id), sizeof(label_t));
		feature_data_ptr->image.SetLabelId(label_id);
		
		point2D_t num_features;
		file.read(reinterpret_cast<char*>(&num_features), sizeof(point2D_t));

		size_t dim = ReadBinaryLittleEndian<size_t>(&file);

		feature_data_ptr->keypoints.resize(num_features);
		//feature_data_ptr->descriptors.resize(num_features, dim);
		std::vector<uint8_t> descriptor(dim);
		for (size_t i = 0; i < num_features; ++i) {

			//keypoint
			float x, y, scale, orientation;
			file.read(reinterpret_cast<char*>(&x), sizeof(float));
			file.read(reinterpret_cast<char*>(&y), sizeof(float));
			file.read(reinterpret_cast<char*>(&scale), sizeof(float));
			file.read(reinterpret_cast<char*>(&orientation), sizeof(float));
			feature_data_ptr->keypoints[i] =
					FeatureKeypoint(x, y, scale, orientation);

			// Descriptor			
			file.read(reinterpret_cast<char*>(descriptor.data()),sizeof(int8_t)*dim);

		}
		feature_data_ump->emplace(feature_data_ptr->image.ImageId(),
		                          feature_data_ptr);
        feature_data_name->emplace(feature_data_ptr->image.Name(),
                                   feature_data_ptr->image.ImageId());
		std::cout << "\rLoad Feature Data [ " << i+1 << " / " << num_images << "]" << std::flush; 
	}
	std::cout << "\n";
	file.close();						 
}



void ReadBinaryDescriptorsIO(const std::string &path,
                         FeatureDataPtrUmap *feature_data_ump) {
	std::ifstream file(path.c_str(), std::ios::binary);
	CHECK(file.is_open()) << path;


	image_t num_images;
	file.read(reinterpret_cast<char*>(&num_images), sizeof(image_t));
	
    for (size_t idx = 0; idx < num_images; ++idx) {
        // image
        image_t image_id;
        file.read(reinterpret_cast<char*>(&image_id), sizeof(image_t));
	
		auto& feature_data_ptr = feature_data_ump->at(image_id);

        // keypoint number
        point2D_t num_features;
        file.read(reinterpret_cast<char*>(&num_features), sizeof(point2D_t));
	
		// length of binary descriptors
		int length;
		file.read(reinterpret_cast<char*>(&length), sizeof(int));

		feature_data_ptr->binary_descriptors.resize(num_features, length);

		for (size_t i = 0; i < num_features; ++i) {
			for (size_t j = 0; j < length; ++j) {
				uint64_t value;
				file.read(reinterpret_cast<char*>(&value), sizeof(uint64_t));
				feature_data_ptr->binary_descriptors(i, j) = value;
			}
		}
		std::cout << "\rLoad Binary Descriptor Data [ " << idx+1 << " / " << num_images << "]" << std::flush; 
	}

	std::cout << "\n";
	file.close();
}


// Sub Panorama Data
void ReadSubPanoramaDataText(const std::string& path, FeatureDataPtrUmap* feature_data_ump) {
    std::ifstream file(path.c_str());
    CHECK(file.is_open()) << path;

    std::string line;
    std::string item;

    while (std::getline(file, line)) {
        StringTrim(&line);

        if (line.empty() || line[0] == '#') {
            continue;
        } else {
            break;
        }
    }

    // Get image number
    const image_t num_images = std::stoul(line);

    for (size_t i = 0; i < num_images; ++i) {
        std::getline(file, line);
        std::stringstream image_line_stream(line);

        // Image id
        std::getline(image_line_stream >> std::ws, item, ' ');
        image_t image_id = std::stol(item);

        auto& feature_data_ptr = feature_data_ump->at(image_id);

        // Feature number
        std::getline(image_line_stream >> std::ws, item, ' ');
        const point2D_t num_features = std::stoul(item);

        feature_data_ptr->panoramaidxs.resize(num_features);

        for (size_t i = 0; i < num_features; ++i) {
            std::getline(file, line);
            std::stringstream feature_line_stream(line);

            // for panorama
            std::getline(feature_line_stream >> std::ws, item, ' ');
            const int sub_image_id = std::stold(item);
            std::getline(feature_line_stream >> std::ws, item, ' ');
            const float sub_x = std::stold(item);
            std::getline(feature_line_stream >> std::ws, item, ' ');
            const float sub_y = std::stold(item);
            feature_data_ptr->panoramaidxs[i].sub_image_id = sub_image_id;
            feature_data_ptr->panoramaidxs[i].sub_x = sub_x;
            feature_data_ptr->panoramaidxs[i].sub_y = sub_y;
        }
    }
    file.close();
}

void WriteSubPanoramaDataText(const std::string& path, const FeatureDataPtrUmap& feature_data_ump) {
    std::ofstream file(path, std::ios::trunc);
    CHECK(file.is_open()) << path;

    file << "# NUM_IMAGE" << std::endl;
    file << "## Image list with NUM_FEATURES lines of sub panorama data per image:" << std::endl;
    file << "# IMAGE_ID, NUM_FEATURES" << std::endl;
    file << "# Sub_image_id Sub_X Sub_Y" << std::endl;
    file << "# ..." << std::endl;
    file << "# Sub_image_id Sub_X Sub_Y" << std::endl;

    std::cout << "image size : " << feature_data_ump.size() << std::endl;
    file << feature_data_ump.size() << std::endl;

    for (const auto& feature_data : feature_data_ump) {
        std::ostringstream line;
        std::string line_string;
        const auto& data_ptr = feature_data.second;
        line << feature_data.first << " ";
        line << data_ptr->keypoints.size();
        file << line.str() << std::endl;
        for (auto i = 0; i < data_ptr->keypoints.size(); ++i) {
            line.str("");
            line.clear();

            // for panorama
            line << data_ptr->panoramaidxs[i].sub_image_id << " ";
            line << data_ptr->panoramaidxs[i].sub_x << " ";
            line << data_ptr->panoramaidxs[i].sub_y << " ";

            line_string = line.str();
            line_string = line_string.substr(0, line_string.size() - 1);
            file << line_string << std::endl;
        }
    }
    file.close();
}

void ReadSubPanoramaDataBinary(const std::string& path, FeatureDataPtrUmap* feature_data_ump) {
    std::ifstream file(path.c_str(), std::ios::binary);
    CHECK(file.is_open()) << path;

    image_t num_images;
    file.read(reinterpret_cast<char*>(&num_images), sizeof(image_t));
	
    for (size_t i = 0; i < num_images; ++i) {
        // image
        image_t image_id;
        file.read(reinterpret_cast<char*>(&image_id), sizeof(image_t));
        auto& feature_data_ptr = feature_data_ump->at(image_id);

        // keypoint number
        point2D_t num_features;
        file.read(reinterpret_cast<char*>(&num_features), sizeof(point2D_t));

        feature_data_ptr->panoramaidxs.resize(num_features);

        for (size_t i = 0; i < num_features; ++i) {
            // for panorama
            int sub_image_id;
            float sub_x, sub_y;
            file.read(reinterpret_cast<char*>(&sub_image_id), sizeof(int));
			file.read(reinterpret_cast<char*>(&sub_x), sizeof(float));
            file.read(reinterpret_cast<char*>(&sub_y), sizeof(float));

            feature_data_ptr->panoramaidxs[i].sub_image_id = sub_image_id;
            feature_data_ptr->panoramaidxs[i].sub_x = sub_x;
            feature_data_ptr->panoramaidxs[i].sub_y = sub_y;
        }
    }
    file.close();
}

void WriteSubPanoramaDataBinary(const std::string& path, const FeatureDataPtrUmap& feature_data_ump) {
    std::ofstream file(path, std::ios::binary);
    CHECK(file.is_open()) << path;

    image_t num_images = feature_data_ump.size();
    file.write((char*)&num_images, sizeof(image_t));

    for (const auto& feature_data : feature_data_ump) {
        const auto& data_ptr = feature_data.second;

        // image
        image_t image_id = feature_data.first;
        file.write((char*)&image_id, sizeof(image_t));

        // keypoint number
        point2D_t num_features = data_ptr->keypoints.size();
        file.write((char*)&num_features, sizeof(point2D_t));
		
		CHECK_EQ(data_ptr->panoramaidxs.size(), num_features) << "Panorama number not equal to feature number";

        for (size_t i = 0; i < num_features; ++i) {
            // for panorama
            int sub_image_id;
            float sub_x, sub_y;
            sub_image_id = data_ptr->panoramaidxs[i].sub_image_id;
            sub_x = data_ptr->panoramaidxs[i].sub_x;
            sub_y = data_ptr->panoramaidxs[i].sub_y;

            file.write((char*)&sub_image_id, sizeof(int));
			file.write((char*)&sub_x, sizeof(float));
            file.write((char*)&sub_y, sizeof(float));

        }
    }
    file.close();
}


void ReadGlobalFeaturesDataBinary(const std::string& path, FeatureDataPtrUmap* feature_data_ump){

	std::ifstream file(path.c_str(), std::ios::binary);
    CHECK(file.is_open()) << path;

    image_t num_images;
    file.read(reinterpret_cast<char*>(&num_images), sizeof(image_t));

	for(size_t i = 0; i < num_images; ++i){
		
		image_t image_id;
        file.read(reinterpret_cast<char*>(&image_id), sizeof(image_t));

        auto& feature_data_ptr = feature_data_ump->at(image_id);		

		size_t dim;
		file.read(reinterpret_cast<char*>(&dim), sizeof(size_t));
		
		feature_data_ptr->vlad_vector.resize(dim, 1);
		for(size_t j = 0; j< dim; ++j){
			float desc;
			file.read(reinterpret_cast<char*>(&desc), sizeof(float));
			
			feature_data_ptr->vlad_vector(j) = desc;
		}
	}
	file.close();
}

void WriteGlobalFeaturesDataBinary(const std::string& path, const FeatureDataPtrUmap& feature_data_ump){
	std::ofstream file(path, std::ios::binary);
	CHECK(file.is_open()) << path;

	image_t num_images = feature_data_ump.size();
    file.write((char*)&num_images, sizeof(image_t));

	for(const auto& feature_data : feature_data_ump){

		image_t image_id = feature_data.first;
		file.write((char*)&image_id, sizeof(image_t));

		size_t dim = feature_data.second->vlad_vector.rows(); 

		file.write((char*)&dim, sizeof(size_t));

		for(size_t i = 0; i<dim; ++i){
			auto desc = feature_data.second->vlad_vector(i);	
			file.write((char*)&desc, sizeof(float));
		}
	}

	file.close();
}



void ReadPieceIndicesDataText(const std::string& path, FeatureDataPtrUmap* feature_data_ump) {
    std::ifstream file(path.c_str());
    CHECK(file.is_open()) << path;

    std::string line;
    std::string item;

    while (std::getline(file, line)) {
        StringTrim(&line);

        if (line.empty() || line[0] == '#') {
            continue;
        } else {
            break;
        }
    }

    // Get image number
    const image_t num_images = std::stoul(line);

    for (size_t i = 0; i < num_images; ++i) {
        std::getline(file, line);
        std::stringstream image_line_stream(line);

        // Image id
        std::getline(image_line_stream >> std::ws, item, ' ');
        image_t image_id = std::stol(item);

        auto& feature_data_ptr = feature_data_ump->at(image_id);

        // Feature number
        std::getline(image_line_stream >> std::ws, item, ' ');
        const point2D_t num_features = std::stoul(item);

        feature_data_ptr->pieceidxs.resize(num_features);

        for (size_t i = 0; i < num_features; ++i) {
            std::getline(file, line);
            std::stringstream feature_line_stream(line);

            // for panorama
            std::getline(feature_line_stream >> std::ws, item, ' ');
            const int sub_image_id = std::stold(item);
			std::getline(feature_line_stream >> std::ws, item, ' ');
            const int piece_id = std::stold(item);
            std::getline(feature_line_stream >> std::ws, item, ' ');
            const float piece_x = std::stold(item);
            std::getline(feature_line_stream >> std::ws, item, ' ');
            const float piece_y = std::stold(item);
			feature_data_ptr->pieceidxs[i].piece_id = piece_id;
            feature_data_ptr->pieceidxs[i].piece_x = piece_x;
            feature_data_ptr->pieceidxs[i].piece_y = piece_y;
        }
    }
    file.close();
}

void WritePieceIndicesDataText(const std::string& path, const FeatureDataPtrUmap& feature_data_ump) {
    std::ofstream file(path, std::ios::trunc);
    CHECK(file.is_open()) << path;

    file << "# NUM_IMAGE" << std::endl;
    file << "## Image list with NUM_FEATURES lines of piece indices data per image:" << std::endl;
    file << "# IMAGE_ID, NUM_FEATURES" << std::endl;
    file << "# piece_id piece_x piece_y" << std::endl;
    file << "# ..." << std::endl;
    file << "# piece_id piece_x piece_y" << std::endl;

    int piece_image_num = 0;
    for (const auto& feature_data : feature_data_ump) {
        const auto &data_ptr = feature_data.second;
        point2D_t num_features = data_ptr->keypoints.size();
        if (data_ptr->pieceidxs.size() == num_features){
            piece_image_num++;
		}
    }

    std::cout << "image size : " << piece_image_num << std::endl;
    file << piece_image_num << std::endl;

    for (const auto& feature_data : feature_data_ump) {
        std::ostringstream line;
        std::string line_string;
        const auto& data_ptr = feature_data.second;
		point2D_t num_features = data_ptr->keypoints.size();
		if (data_ptr->pieceidxs.size() != num_features){
			continue;
		}

        line << feature_data.first << " ";
        line << data_ptr->keypoints.size();
        file << line.str() << std::endl;
        for (auto i = 0; i < data_ptr->keypoints.size(); ++i) {
            line.str("");
            line.clear();

            // for panorama
            line << data_ptr->pieceidxs[i].piece_id << " ";
            line << data_ptr->pieceidxs[i].piece_x << " ";
            line << data_ptr->pieceidxs[i].piece_y << " ";

            line_string = line.str();
            line_string = line_string.substr(0, line_string.size() - 1);
            file << line_string << std::endl;
        }
    }
    file.close();
}



void ReadPieceIndicesDataBinary(const std::string& path, FeatureDataPtrUmap* feature_data_ump) {
    std::ifstream file(path.c_str(), std::ios::binary);
    CHECK(file.is_open()) << path;

    image_t num_images;
    file.read(reinterpret_cast<char*>(&num_images), sizeof(image_t));

    for (size_t i = 0; i < num_images; ++i) {
        // image
        image_t image_id;
        file.read(reinterpret_cast<char*>(&image_id), sizeof(image_t));
        auto& feature_data_ptr = feature_data_ump->at(image_id);

        // keypoint number
        point2D_t num_features;
        file.read(reinterpret_cast<char*>(&num_features), sizeof(point2D_t));

        feature_data_ptr->pieceidxs.resize(num_features);

        for (size_t i = 0; i < num_features; ++i) {
            int piece_id;
            float piece_x, piece_y;
            file.read(reinterpret_cast<char*>(&piece_id), sizeof(int));
			file.read(reinterpret_cast<char*>(&piece_x), sizeof(float));
            file.read(reinterpret_cast<char*>(&piece_y), sizeof(float));

            feature_data_ptr->pieceidxs[i].piece_id = piece_id;
            feature_data_ptr->pieceidxs[i].piece_x = piece_x;
            feature_data_ptr->pieceidxs[i].piece_y = piece_y;
        }
    }
    file.close();
}



void WritePieceIndicesDataBinary(const std::string& path, const FeatureDataPtrUmap& feature_data_ump){
	
	std::ofstream file(path, std::ios::binary);
    CHECK(file.is_open()) << path;

    image_t num_images = feature_data_ump.size();

    int piece_image_num = 0;
    for (const auto& feature_data : feature_data_ump) {
        const auto &data_ptr = feature_data.second;
        point2D_t num_features = data_ptr->keypoints.size();
        if (data_ptr->pieceidxs.size() == num_features){
            piece_image_num++;
		}
    }

    file.write((char*)&piece_image_num, sizeof(piece_image_num));

    for (const auto& feature_data : feature_data_ump) {
        const auto& data_ptr = feature_data.second;
        point2D_t num_features = data_ptr->keypoints.size();
		if (data_ptr->pieceidxs.size() != num_features){
			continue;
		}

        // image
        image_t image_id = feature_data.first;
        file.write((char*)&image_id, sizeof(image_t));

        // keypoint number
        file.write((char*)&num_features, sizeof(point2D_t));
		
		// CHECK_EQ(data_ptr->pieceidxs.size(), num_features) << "Pieceindices number not equal to feature number";

        for (size_t i = 0; i < num_features; ++i) {
            // for panorama
            int piece_id;
			// int piece_id;
            float sub_x, sub_y;
            piece_id = data_ptr->pieceidxs[i].piece_id;
            sub_x = data_ptr->pieceidxs[i].piece_x;
            sub_y = data_ptr->pieceidxs[i].piece_y;

            file.write((char*)&piece_id, sizeof(int));
            // file.write((char*)&piece_id, sizeof(int));
			file.write((char*)&sub_x, sizeof(float));
            file.write((char*)&sub_y, sizeof(float));
        }
    }
    file.close();
}

void ReadAprilTagDataText(const std::string& path, FeatureDataPtrUmap* feature_data_ump){
    //TODO: Not implement yet
    std::cout << "Error: function not implemented yet ... " << std::endl;
	exit(-1);
}
void WriteAprilTagDataText(const std::string& path, const FeatureDataPtrUmap& feature_data_ump){
    //TODO: Not implement yet
    std::cout << "Error: function not implemented yet ... " << std::endl;
	exit(-1);
}

void ReadAprilTagDataBinary(const std::string& path, FeatureDataPtrUmap* feature_data_ump){
    std::ifstream file(path.c_str(), std::ios::binary);
    CHECK(file.is_open()) << path;

    image_t num_images;
    file.read(reinterpret_cast<char*>(&num_images), sizeof(image_t));

    for (size_t i = 0; i < num_images; ++i) {
        // image
        image_t image_id;
        file.read(reinterpret_cast<char*>(&image_id), sizeof(image_t));
		
		FeatureDataPtr feature_data_ptr;

		bool insert_feature_data = false;
		if (!feature_data_ump->count(image_id)) {
			feature_data_ptr = std::make_shared<FeatureData>();
			feature_data_ptr->image.SetImageId(image_id);
			insert_feature_data = true;
		} else {
			feature_data_ptr = feature_data_ump->at(image_id);
		}

        // detection number
        int detection_num;
        file.read(reinterpret_cast<char*>(&detection_num), sizeof(int));

        feature_data_ptr->detections.resize(detection_num);

        for (size_t i = 0; i < detection_num; ++i) {
            // for april tag detection
            int local_camera_id;
            file.read(reinterpret_cast<char*>(&local_camera_id), sizeof(int));
            feature_data_ptr->detections[i].local_camera_id = local_camera_id;

			int id;
            file.read(reinterpret_cast<char*>(&id), sizeof(int));
            feature_data_ptr->detections[i].id = id;

            float u, v;
            file.read(reinterpret_cast<char*>(&u), sizeof(float));
            file.read(reinterpret_cast<char*>(&v), sizeof(float));
            feature_data_ptr->detections[i].cxy = {u ,v};

            for (auto& p : feature_data_ptr->detections[i].p){
                file.read(reinterpret_cast<char*>(&u), sizeof(float));
                file.read(reinterpret_cast<char*>(&v), sizeof(float));
                p = {u ,v};
            }
        }

		if (insert_feature_data) {
			// std::cout << "Emplace image id = " << feature_data_ptr->image.ImageId() << std::endl;
			feature_data_ump->emplace(feature_data_ptr->image.ImageId(),
		                          feature_data_ptr);
		}
    }

	std::cout << "feature map size = " << feature_data_ump->size() << std::endl;
    file.close();
}

void WriteAprilTagDataBinary(const std::string& path, const FeatureDataPtrUmap& feature_data_ump){
    std::ofstream file(path, std::ios::binary);
    CHECK(file.is_open()) << path;

    image_t num_images = feature_data_ump.size();
    file.write((char*)&num_images, sizeof(image_t));

    for (const auto& feature_data : feature_data_ump) {
        const auto& data_ptr = feature_data.second;

        // image
        image_t image_id = feature_data.first;
        file.write((char*)&image_id, sizeof(image_t));

        // detection number
        int detection_num = data_ptr->detections.size();
        file.write((char*)&detection_num, sizeof(int));

        for (size_t i = 0; i < detection_num; ++i) {
            // for april tag detection
			int local_camera_id = data_ptr->detections[i].local_camera_id;
            file.write((char*)&local_camera_id, sizeof(int));
			
            int id = data_ptr->detections[i].id;
            file.write((char*)&id, sizeof(int));

            float u, v;
            u = data_ptr->detections[i].cxy.first;
            v = data_ptr->detections[i].cxy.second;
            file.write((char*)&u, sizeof(float));
            file.write((char*)&v, sizeof(float));

            for(const auto& p : data_ptr->detections[i].p){
                u = p.first;
                v = p.second;
                file.write((char*)&u, sizeof(float));
                file.write((char*)&v, sizeof(float));
            }

        }
    }
    file.close();
}

void ReadGPSDataBinary(const std::string& path, FeatureDataPtrUmap* feature_data_ump) {
	std::ifstream file(path.c_str(), std::ios::binary);
    CHECK(file.is_open()) << path;

	std::string version_name = "\0";

	char name_char;
	do {
		file.read(&name_char, 1);
		if (name_char != '\0') {
			version_name += name_char;
		}
	} while (name_char != '\0');
	bool old_version = false;
	if (version_name.find("version") == std::string::npos) {
		old_version = true;
		file.seekg(0);
	}
	std::cout << version_name.c_str() << " " << version_name.length() << " " << old_version << std::endl;

	int gps_size = 0;
	file.read((char*)&gps_size, sizeof(int));

	for (size_t i = 0; i < gps_size; ++i) {
		image_t image_id;
		file.read((char*)&image_id, sizeof(image_t));

		FeatureDataPtr& data_ptr = feature_data_ump->at(image_id);

		int8_t rtk_flag;
		file.read((char*)&rtk_flag, sizeof(int8_t));
		data_ptr->image.RtkFlag() = rtk_flag;

		bool has_qvec_prior;
		file.read((char*)&has_qvec_prior, sizeof(bool));
		if (has_qvec_prior) {
			double qw, qx, qy, qz;
			file.read((char*)&qw, sizeof(double));
			file.read((char*)&qx, sizeof(double));
			file.read((char*)&qy, sizeof(double));
			file.read((char*)&qz, sizeof(double));
			data_ptr->image.QvecPrior(0) = qw;	
			data_ptr->image.QvecPrior(1) = qx;	
			data_ptr->image.QvecPrior(2) = qy;	
			data_ptr->image.QvecPrior(3) = qz;
		}

		double tx, ty, tz;
		file.read((char*)&tx, sizeof(double));
		file.read((char*)&ty, sizeof(double));
		file.read((char*)&tz, sizeof(double));
		data_ptr->image.TvecPrior(0) = tx;
		data_ptr->image.TvecPrior(1) = ty;
		data_ptr->image.TvecPrior(2) = tz;

		bool has_std_prior;
		file.read((char*)&has_std_prior, sizeof(bool));
		if (has_std_prior) {
			double std_lon;
			double std_lat;
			double std_hgt;
			file.read((char*)&std_lon, sizeof(double));
			file.read((char*)&std_lat, sizeof(double));
			file.read((char*)&std_hgt, sizeof(double));
			data_ptr->image.SetRtkStd(std_lon, std_lat, std_hgt);
		}

		if (!old_version) {
			bool has_relative_altitude;
			file.read((char*)&has_relative_altitude, sizeof(bool));
			if (has_relative_altitude) {
				double relative_altitude;
				file.read((char*)&relative_altitude, sizeof(double));
				data_ptr->image.SetRelativeAltitude(relative_altitude);
			}
		}
	}
	file.close();
}

void WriteGPSDataBinary(const std::string& path, 
						const FeatureDataPtrUmap& feature_data_ump) {
	
	int gps_size = 0;
	for (const auto& feature_data : feature_data_ump) {
		const auto &data_ptr = feature_data.second;
		if (data_ptr->image.HasTvecPrior()) {
			gps_size++;
		}
	}
	if (gps_size == 0){
		std::cout << "Warning: GPSData is empty() ... continue" << std::endl;
		return;
	}

	std::ofstream file(path, std::ios::binary);
	CHECK(file.is_open()) << path;

	std::string version_name("version1.1");
	version_name += '\0';
	std::cout << version_name.c_str() << " " << version_name.length() << std::endl;
	file.write(version_name.c_str(), version_name.length());

	file.write((char*)&gps_size, sizeof(int));

	for (const auto& feature_data : feature_data_ump) {
		const auto &data_ptr = feature_data.second;
		if (!data_ptr->image.HasTvecPrior()) {
			continue;
		}
		image_t image_id = feature_data.first;
		file.write((char*)&image_id, sizeof(image_t));

		int8_t rtk_flag = data_ptr->image.RtkFlag();
		file.write((char*)&rtk_flag, sizeof(int8_t));

		bool has_qvec_prior = data_ptr->image.HasQvecPrior();
		file.write((char*)&has_qvec_prior, sizeof(bool));
		if (has_qvec_prior) {
			Eigen::Vector4d qvec = data_ptr->image.QvecPrior();
			file.write((char*)&qvec[0], sizeof(double));
			file.write((char*)&qvec[1], sizeof(double));
			file.write((char*)&qvec[2], sizeof(double));			
			file.write((char*)&qvec[3], sizeof(double));
		}

		Eigen::Vector3d tvec = data_ptr->image.TvecPrior();
		file.write((char*)&tvec.x(), sizeof(double));
		file.write((char*)&tvec.y(), sizeof(double));
		file.write((char*)&tvec.z(), sizeof(double));

		bool has_std_prior = data_ptr->image.HasRtkStd();
		file.write((char*)&has_std_prior, sizeof(bool));
		if (has_std_prior) {
			double std_lon = data_ptr->image.RtkStdLon();
			double std_lat = data_ptr->image.RtkStdLat();
			double std_hgt = data_ptr->image.RtkStdHgt();
			file.write((char*)&std_lon, sizeof(double));
			file.write((char*)&std_lat, sizeof(double));
			file.write((char*)&std_hgt, sizeof(double));
		}

		bool has_relative_altitude = data_ptr->image.HasRelativeAltitude(); 
		file.write((char*)&has_relative_altitude, sizeof(bool));
		if (has_relative_altitude) {
			double relative_altitude = data_ptr->image.RelativeAltitude();
			file.write((char*)&relative_altitude, sizeof(double));
		}
	}
	file.close();
}

} // namespace sensemap
