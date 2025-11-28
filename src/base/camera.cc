//Copyright (c) 2019, SenseTime Group.
//All rights reserved.

#include "base/camera.h"

#include <iomanip>

#include "base/camera_models.h"
#include "util/logging.h"
#include "util/misc.h"
#include "base/pose.h"

namespace sensemap {

Camera::Camera()
		: camera_id_(kInvalidCameraId),
		  model_id_(kInvalidCameraModelId),
		  width_(0),
		  height_(0),
		  prior_focal_length_(false),
		  fixed_(false),
		  num_local_cameras_(1),
		  init_local_qvec_diff(0.0, 0.0, 0.0, 0.0),
		  init_local_tvec_diff(0.0, 0.0, 0.0) {}

std::string Camera::ModelName() const { return CameraModelIdToName(model_id_); }

void Camera::SetModelId(const int model_id) {
	CHECK(ExistsCameraModelWithId(model_id));
	model_id_ = model_id;
	params_.resize(CameraModelNumParams(model_id_), 0);
}

void Camera::SetModelIdFromName(const std::string& model_name) {
	CHECK(ExistsCameraModelWithName(model_name));
	model_id_ = CameraModelNameToId(model_name);
	params_.resize(CameraModelNumParams(model_id_), 0);
}

const std::vector<size_t>& Camera::FocalLengthIdxs() const {
	return CameraModelFocalLengthIdxs(model_id_);
}

const std::vector<size_t>& Camera::PrincipalPointIdxs() const {
	return CameraModelPrincipalPointIdxs(model_id_);
}

const std::vector<size_t>& Camera::ExtraParamsIdxs() const {
	return CameraModelExtraParamsIdxs(model_id_);
}

Eigen::Matrix3d Camera::CalibrationMatrix() const {
	Eigen::Matrix3d K = Eigen::Matrix3d::Identity();

	const std::vector<size_t>& idxs = FocalLengthIdxs();
	if (idxs.size() == 1) {
		K(0, 0) = params_[idxs[0]];
		K(1, 1) = params_[idxs[0]];
	} else if (idxs.size() == 2) {
		K(0, 0) = params_[idxs[0]];
		K(1, 1) = params_[idxs[1]];
	} else {
		LOG(FATAL)
				<< "Camera model must either have 1 or 2 focal length parameters.";
	}

	K(0, 2) = PrincipalPointX();
	K(1, 2) = PrincipalPointY();

	return K;
}

std::string Camera::ParamsInfo() const {
	return CameraModelParamsInfo(model_id_);
}

double Camera::MeanFocalLength() const {
	const auto& focal_length_idxs = FocalLengthIdxs();
	double focal_length = 0;
	for (const auto idx : focal_length_idxs) {
		focal_length += params_[idx];
	}
	return focal_length / focal_length_idxs.size();
}

double Camera::FocalLength() const {
	const std::vector<size_t>& idxs = FocalLengthIdxs();
	CHECK_EQ(idxs.size(), 1);
	return params_[idxs[0]];
}

double Camera::FocalLengthX() const {
	const std::vector<size_t>& idxs = FocalLengthIdxs();
	CHECK_EQ(idxs.size(), 2);
	return params_[idxs[0]];
}

double Camera::FocalLengthY() const {
	const std::vector<size_t>& idxs = FocalLengthIdxs();
	CHECK_EQ(idxs.size(), 2);
	return params_[idxs[1]];
}

void Camera::SetFocalLength(const double focal_length) {
	const std::vector<size_t>& idxs = FocalLengthIdxs();
	for (const auto idx : idxs) {
		params_[idx] = focal_length;
	}
}

void Camera::SetFocalLengthX(const double focal_length_x) {
	const std::vector<size_t>& idxs = FocalLengthIdxs();
	CHECK_EQ(idxs.size(), 2);
	params_[idxs[0]] = focal_length_x;
}

void Camera::SetFocalLengthY(const double focal_length_y) {
	const std::vector<size_t>& idxs = FocalLengthIdxs();
	CHECK_EQ(idxs.size(), 2);
	params_[idxs[1]] = focal_length_y;
}

double Camera::PrincipalPointX() const {
	const std::vector<size_t>& idxs = PrincipalPointIdxs();
	CHECK_EQ(idxs.size(), 2);
	return params_[idxs[0]];
}

double Camera::PrincipalPointY() const {
	const std::vector<size_t>& idxs = PrincipalPointIdxs();
	CHECK_EQ(idxs.size(), 2);
	return params_[idxs[1]];
}

void Camera::SetPrincipalPointX(const double ppx) {
	const std::vector<size_t>& idxs = PrincipalPointIdxs();
	CHECK_EQ(idxs.size(), 2);
	params_[idxs[0]] = ppx;
}

void Camera::SetPrincipalPointY(const double ppy) {
	const std::vector<size_t>& idxs = PrincipalPointIdxs();
	CHECK_EQ(idxs.size(), 2);
	params_[idxs[1]] = ppy;
}

std::string Camera::ParamsToString() const { return VectorToCSV(params_); }

bool Camera::SetParamsFromString(const std::string& string) {
	params_ = CSVToVector<double>(string);
	return VerifyParams();
}

bool Camera::VerifyParams() const {
	return CameraModelVerifyParams(model_id_, params_);
}

bool Camera::HasBogusParams(const double min_focal_length_ratio,
                            const double max_focal_length_ratio,
                            const double max_extra_param) const {
	if(NumLocalCameras()>1){
		return false;
	}
	return CameraModelHasBogusParams(model_id_, params_, width_, height_,
	                                 min_focal_length_ratio,
	                                 max_focal_length_ratio, max_extra_param);
}

void Camera::InitializeWithId(const int model_id, const double focal_length,
                              const size_t width, const size_t height) {
	CHECK(ExistsCameraModelWithId(model_id));
	model_id_ = model_id;
	width_ = width;
	height_ = height;
	params_ = CameraModelInitializeParams(model_id, focal_length, width, height);
}

void Camera::InitializeWithName(const std::string& model_name,
                                const double focal_length, const size_t width,
                                const size_t height) {
	InitializeWithId(CameraModelNameToId(model_name), focal_length, width,
	                 height);
}

Eigen::Vector2d Camera::ImageToWorld(const Eigen::Vector2d& image_point) const {
	Eigen::Vector2d world_point;
	CameraModelImageToWorld(model_id_, params_, image_point(0), image_point(1),
	                        &world_point(0), &world_point(1));
	return world_point;
}

double Camera::ImageToWorldThreshold(const double threshold) const {
	return CameraModelImageToWorldThreshold(model_id_, params_, threshold);
}

Eigen::Vector2d Camera::WorldToImage(const Eigen::Vector2d& world_point) const {
	Eigen::Vector2d image_point;
	CameraModelWorldToImage(model_id_, params_, world_point(0), world_point(1),
	                        &image_point(0), &image_point(1));
	return image_point;
}

Eigen::Vector3d Camera::ImageToBearing(const Eigen::Vector2d& image_point) const{
	Eigen::Vector3d bearing;
	CameraModelImageToBearing(model_id_,params_,image_point(0),image_point(1),
							  &bearing(0),&bearing(1),&bearing(2));
	return bearing;
}

Eigen::Vector3d Camera::WorldToBearing(const Eigen::Vector3d& world_point) const{
	Eigen::Vector3d bearing;
	CameraModelWorldToBearing(model_id_,params_,world_point(0),world_point(1),
							  world_point(2),
							  &bearing(0),&bearing(1),&bearing(2));
	return bearing;
}



void Camera::Rescale(const double scale) {
	CHECK_GT(scale, 0.0);
	const double scale_x =
			std::round(scale * width_) / static_cast<double>(width_);
	const double scale_y =
			std::round(scale * height_) / static_cast<double>(height_);
	width_ = static_cast<size_t>(std::round(scale * width_));
	height_ = static_cast<size_t>(std::round(scale * height_));
	SetPrincipalPointX(scale_x * PrincipalPointX());
	SetPrincipalPointY(scale_y * PrincipalPointY());
	if (FocalLengthIdxs().size() == 1) {
		SetFocalLength((scale_x + scale_y) / 2.0 * FocalLength());
	} else if (FocalLengthIdxs().size() == 2) {
		SetFocalLengthX(scale_x * FocalLengthX());
		SetFocalLengthY(scale_y * FocalLengthY());
	} else {
		LOG(FATAL)
				<< "Camera model must either have 1 or 2 focal length parameters.";
	}
}

void Camera::Rescale(const size_t width, const size_t height) {
	const double scale_x =
			static_cast<double>(width) / static_cast<double>(width_);
	const double scale_y =
			static_cast<double>(height) / static_cast<double>(height_);
	width_ = width;
	height_ = height;
	SetPrincipalPointX(scale_x * PrincipalPointX());
	SetPrincipalPointY(scale_y * PrincipalPointY());
	if (FocalLengthIdxs().size() == 1) {
		SetFocalLength((scale_x + scale_y) / 2.0 * FocalLength());
	} else if (FocalLengthIdxs().size() == 2) {
		SetFocalLengthX(scale_x * FocalLengthX());
		SetFocalLengthY(scale_y * FocalLengthY());
	} else {
		LOG(FATAL)
				<< "Camera model must either have 1 or 2 focal length parameters.";
	}
}

////////////////////////////////////////////////////////////////////////////
//  Extra functions in multi-camera system (MCS)
////////////////////////////////////////////////////////////////////////////

void Camera::SetNumLocalCameras(int num){
	num_local_cameras_ = num;
	local_params_.resize(num_local_cameras_*CameraModelNumParams(model_id_));
}

void Camera::SetLocalCameraExtrinsics(
						 const std::vector<Eigen::Vector4d>& local_qvecs,
						 const std::vector<Eigen::Vector3d>& local_tvecs){

	CHECK_EQ(local_qvecs.size(),num_local_cameras_);
	CHECK_EQ(local_qvecs.size(),num_local_cameras_);

	local_qvecs_.clear();
	local_tvecs_.clear();

	for(const auto& qvec: local_qvecs){
		for(size_t i = 0; i<4; ++i){
			local_qvecs_.push_back(qvec(i));
		}
	}						 

	for(const auto& tvec: local_tvecs){
		for(size_t i = 0; i<3; ++i){
			local_tvecs_.push_back(tvec(i));
		}
	}
}


void Camera::RescaleLocalCameraExtrinsics(const double scale){
	CHECK_GE(scale, 0);

	for(auto& tvec: local_tvecs_){
		tvec *= scale;
	}
}

void Camera::GetLocalCameraExtrinsic(local_camera_t camera_id, 
							 Eigen::Vector4d& qvec, 
							 Eigen::Vector3d& tvec) const{

    CHECK_LT(camera_id, num_local_cameras_);								 
	CHECK_EQ(local_qvecs_.size(),num_local_cameras_*4);
	CHECK_EQ(local_tvecs_.size(),num_local_cameras_*3);

	for(size_t i = 0; i<4; ++i){
		qvec(i) = local_qvecs_[camera_id*4+i];
	}
	for(size_t i = 0; i<3; ++i){
		tvec(i) = local_tvecs_[camera_id*3+i];
	}
}

void Camera::GetLocalCameraIntrisic(local_camera_t camera_id,
									std::vector<double>& params) const{

  CHECK_LT(camera_id, num_local_cameras_);
	
	size_t local_param_size = local_params_.size()/num_local_cameras_;

	CHECK(local_param_size>0); 
	params.resize(local_param_size);

	for(size_t i = 0; i<local_param_size; ++i){
		params[i] = local_params_[camera_id*local_param_size + i];
	}
}

void Camera::InitializeLocalCameraIntricsWithId(int model_id, 
									  const std::vector<double>& focal_lengths,
									  const size_t width,
									  const size_t height){
    
	CHECK_EQ(focal_lengths.size(),num_local_cameras_);

	model_id_ = model_id;
	width_ = width;
	height_ = height;
	
	local_params_.clear();

	for(size_t i = 0; i<focal_lengths.size(); ++i){
		std::vector<double> local_param = 
			CameraModelInitializeParams(model_id, focal_lengths[i], 
										width, height);						

		for(const auto& local_param_elem:local_param){
			local_params_.push_back(local_param_elem);
		}
	}
}


bool Camera::SetLocalCameraIntrinsicParamsFromString(
							const std::string& param_strings){
    std::cout<<"local params string: "<<param_strings<<std::endl;
	std::vector<double> local_params = CSVToVector<double>(param_strings);
	std::cout<<"total local params size: "<<local_params.size()<<std::endl;
	size_t local_param_size = local_params.size()/num_local_cameras_;	
	CHECK(local_param_size > 0);
	local_params_.clear();

	for (local_camera_t camera_id = 0; camera_id<num_local_cameras_; ++camera_id){
		std::vector<double> local_param;
		for(size_t i = 0 ;i<local_param_size; ++i){
			local_param.push_back(local_params[camera_id*local_param_size+i]);
		}

		CHECK(CameraModelVerifyParams(model_id_, local_param))<<local_param.size();
		
		for (size_t i = 0; i < local_param.size(); ++i){
			local_params_.push_back(local_param[i]);
		}
	}

	return true;
}

bool Camera::SetLocalCameraExtrinsicParamsFromString(
								const std::string& param_strings){
    std::cout<<"local extrinsics string: "<<param_strings<<std::endl;
	std::vector<double> local_extrinsics = CSVToVector<double>(param_strings);
	std::cout<<"total local extrinsics size: "<<local_extrinsics.size()<<std::endl;
	CHECK_EQ(local_extrinsics.size(), 7*num_local_cameras_)
			<<"Camera extrinsics invalid";
	
	local_qvecs_.clear();
	local_tvecs_.clear();

	for(local_camera_t camera_id = 0; camera_id <num_local_cameras_; ++camera_id){
		Eigen::Vector4d qvec;
		for(size_t i = 0; i<4; ++i){
			qvec(i) = local_extrinsics[camera_id*7 + i];
		}

		Eigen::Vector3d tvec;
		for(size_t i = 0; i<3; ++i){
			tvec(i) = local_extrinsics[camera_id*7 + i +4];
		}

		// Eigen::Matrix3d R = QuaternionToRotationMatrix(qvec);
		// Eigen::Vector3d t = -R.transpose() * tvec;
	 	// qvec = RotationMatrixToQuaternion(R.transpose());
		// tvec = t;
		
		for(size_t i = 0; i<4; ++i){
			local_qvecs_.push_back(qvec(i));
		}
	  for(size_t i = 0; i<3; ++i){
			local_tvecs_.push_back(tvec(i));
		}

	}
	return true;
}

Eigen::Vector2d Camera::LocalImageToWorld(local_camera_t camera_id, 
								  const Eigen::Vector2d& image_point) const{
	
	Eigen::Vector2d world_point;
	std::vector<double> local_param;
	
	GetLocalCameraIntrisic(camera_id,local_param);

	CameraModelImageToWorld(model_id_, local_param, image_point(0), 
							image_point(1),&world_point(0), &world_point(1));
	return world_point;
}


Eigen::Vector3d Camera::LocalImageToBearing(local_camera_t camera_id,
		const Eigen::Vector2d& image_point) const{
	
	Eigen::Vector3d bearing;
	std::vector<double> local_param;
	GetLocalCameraIntrisic(camera_id,local_param);

	CameraModelImageToBearing(model_id_,local_param,image_point(0),image_point(1),
							  &bearing(0),&bearing(1),&bearing(2));
	return bearing;
}



Eigen::Vector2d Camera::WorldToLocalImage(local_camera_t camera_id,
								  const Eigen::Vector2d& world_point) const{
	
	Eigen::Vector2d image_point;
	std::vector<double> local_param;
	
	GetLocalCameraIntrisic(camera_id,local_param);

	CameraModelWorldToImage(model_id_, local_param, world_point(0),
							world_point(1),&image_point(0), &image_point(1));

	return image_point;
}

Eigen::Vector2d Camera::BearingToLocalImage(local_camera_t camera_id,
								  const Eigen::Vector3d& bearing_point) const{
	
	Eigen::Vector2d image_point;
	std::vector<double> local_param;
	
	GetLocalCameraIntrisic(camera_id,local_param);

	CameraModelBearingToImage(model_id_, local_param, bearing_point(0),
							bearing_point(1),bearing_point(2),&image_point(0), &image_point(1));

	return image_point;
}

Eigen::Vector2d Camera::BearingToImage(const Eigen::Vector3d& bearing_point) const{

	Eigen::Vector2d image_point;	
	CameraModelBearingToImage(model_id_, params_, bearing_point(0),
							bearing_point(1),bearing_point(2),&image_point(0), &image_point(1));

	return image_point;
}


bool Camera::VerifyLocalParams() const {
	std::cout<<"num local cameras: "<< num_local_cameras_<<std::endl;
	for(uint camera_id = 0; camera_id < num_local_cameras_; ++camera_id){
		std::vector<double> local_param;
		GetLocalCameraIntrisic(camera_id,local_param);
		if(!CameraModelVerifyParams(model_id_, local_param)){
			return false;
		}
	}
	return true;
}


double Camera::LocalImageToWorldThreshold(local_camera_t camera_id, 
																					const double threshold) const {
	
	std::vector<double> local_param;
	GetLocalCameraIntrisic(camera_id,local_param);

	return CameraModelImageToWorldThreshold(model_id_, local_param, threshold);
}


double Camera::LocalMeanFocalLength(local_camera_t camera_id) const {
	
	std::vector<double> local_param;
	GetLocalCameraIntrisic(camera_id,local_param);

	const auto& focal_length_idxs = FocalLengthIdxs();
	double focal_length = 0;
	for (const auto idx : focal_length_idxs) {
		focal_length += local_param[idx];
	}
	return focal_length / focal_length_idxs.size();
}

void Camera::SetDisturb(){
	Eigen::Vector4d qvec;
	qvec << 1, 0, 0, 0;
	SetDisturb(qvec, Eigen::Vector3d::Zero());
}

void Camera::SetDisturb(Eigen::Vector4d qvec, const Eigen::Vector3d tvec){
	qvec = NormalizeQuaternion(qvec);
	SetPriorDisturb(qvec, tvec);
	SetQvecDisturb(qvec);
	SetTvecDisturb(tvec);
	has_disturb_ = true;
}

void Camera::SetPriorDisturb(const Eigen::Vector4d& qvec, const Eigen::Vector3d& tvec){
	qvec_prior_disturb_ = qvec; 
	tvec_prior_disturb_ = tvec;
}

void Camera::SetTrLidar2Camera(const Eigen::Matrix3x4d& Tr_lidar_to_cam) { 
    // Tr_lidar_to_cam_ = Tr_lidar_to_cam; 
	lidar_qvecs_ = RotationMatrixToQuaternion(Tr_lidar_to_cam.leftCols<3>());
	lidar_tvecs_ = Tr_lidar_to_cam.rightCols<1>();
}

Eigen::Matrix3x4d Camera::GetLidar2Camera() {
	Eigen::Matrix3x4d Tr_lidar_to_cam;
	Tr_lidar_to_cam.block<3,3>(0,0) = QuaternionToRotationMatrix(lidar_qvecs_);
	Tr_lidar_to_cam.block<3,1>(0,3) = lidar_tvecs_;
	return Tr_lidar_to_cam;
};

} // namespace sensemap