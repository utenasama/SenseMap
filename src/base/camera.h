//Copyright (c) 2019, SenseTime Group.
//All rights reserved.

#ifndef SENSEMAP_BASE_CAMERA_H_
#define SENSEMAP_BASE_CAMERA_H_

#include <vector>
#include <iostream>

#include "util/types.h"
#include "base/pose.h"

namespace sensemap {

// Camera class that holds the intrinsic parameters. Cameras may be shared
// between multiple images, e.g., if the same "physical" camera took multiple
// pictures with the exact same lens and intrinsics (focal length, etc.).
// This class has a specific distortion model defined by a camera model class.
class Camera {
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	Camera();

	// Access the unique identifier of the camera.
	inline camera_t CameraId() const;
	inline void SetCameraId(const camera_t camera_id);

	// Access the camera model.
	inline int ModelId() const;
	std::string ModelName() const;
	void SetModelId(const int model_id);
	void SetModelIdFromName(const std::string& model_name);

	// Access dimensions of the camera sensor.
	inline size_t Width() const;
	inline size_t Height() const;
	inline void SetWidth(const size_t width);
	inline void SetHeight(const size_t height);

	// Access focal length parameters.
	double MeanFocalLength() const;
	double LocalMeanFocalLength(local_camera_t camera_id) const;
	
	double FocalLength() const;
	double FocalLengthX() const;
	double FocalLengthY() const;
	void SetFocalLength(const double focal_length);
	void SetFocalLengthX(const double focal_length_x);
	void SetFocalLengthY(const double focal_length_y);

	// Check if camera has prior focal length.
	inline bool HasPriorFocalLength() const;
	inline void SetPriorFocalLength(const bool prior);

	// Access principal point parameters. Only works if there are two
	// principal point parameters.
	double PrincipalPointX() const;
	double PrincipalPointY() const;
	void SetPrincipalPointX(const double ppx);
	void SetPrincipalPointY(const double ppy);

	// Get the indices of the parameter groups in the parameter vector.
	const std::vector<size_t>& FocalLengthIdxs() const;
	const std::vector<size_t>& PrincipalPointIdxs() const;
	const std::vector<size_t>& ExtraParamsIdxs() const;

	// Get intrinsic calibration matrix composed from focal length and principal
	// point parameters, excluding distortion parameters.
	Eigen::Matrix3d CalibrationMatrix() const;

	// Get human-readable information about the parameter vector ordering.
	std::string ParamsInfo() const;

	// Access the raw parameter vector.
	inline size_t NumParams() const;
	inline const std::vector<double>& Params() const;
	inline std::vector<double>& Params();
	inline double Params(const size_t idx) const;
	inline double& Params(const size_t idx);
	inline const double* ParamsData() const;
	inline double* ParamsData();
	inline void SetParams(const std::vector<double>& params);
	inline void SetCameraConstant(bool fixed);
	inline bool IsCameraConstant() const;

	// Concatenate parameters as comma-separated list.
	std::string ParamsToString() const;

	// Set camera parameters from comma-separated list.
	bool SetParamsFromString(const std::string& string);

	// Check whether parameters are valid, i.e. the parameter vector has
	// the correct dimensions that match the specified camera model.
	bool VerifyParams() const;

	// Check whether camera has bogus parameters.
	bool HasBogusParams(const double min_focal_length_ratio,
	                    const double max_focal_length_ratio,
	                    const double max_extra_param) const;

	// Initialize parameters for given camera model and focal length, and set
	// the principal point to be the image center.
	void InitializeWithId(const int model_id, const double focal_length,
	                      const size_t width, const size_t height);
	void InitializeWithName(const std::string& model_name,
	                        const double focal_length, const size_t width,
	                        const size_t height);

	// Project point in image plane to world / infinity.
	Eigen::Vector2d ImageToWorld(const Eigen::Vector2d& image_point) const;

	// Convert pixel threshold in image plane to world space.
	double ImageToWorldThreshold(const double threshold) const;

	// Project point from world / infinity to image plane.
	Eigen::Vector2d WorldToImage(const Eigen::Vector2d& world_point) const;

	// Project point in image plane to bearing for spherical camera
	Eigen::Vector3d ImageToBearing(const Eigen::Vector2d& image_point) const;

	// Project point in image plane to bearing for spherical camera
	Eigen::Vector3d WorldToBearing(const Eigen::Vector3d& world_point) const;


	// Rescale camera dimensions and accordingly the focal length and
	// and the principal point.
	void Rescale(const double scale);
	void Rescale(const size_t width, const size_t height);


	////////////////////////////////////////////////////////////////////////////
	//  Extra functions in multi-camera system (MCS)
    ////////////////////////////////////////////////////////////////////////////

	// The cameras in the MCS will be referred to as local cameras 

	// Set the number of local cameras in the MCS
	void SetNumLocalCameras(const int num);
	inline int NumLocalCameras() const;

	// Set the intrinsic parmas from a string
	bool SetLocalCameraIntrinsicParamsFromString(
								const std::string& param_strings);
	// Set the extrinsic parmas from a string
	bool SetLocalCameraExtrinsicParamsFromString(
								const std::string& param_strings);

	// Initialize parameters for given camera model and focal length, and set
	// the principal point to be the image center.
	void InitializeLocalCameraIntricsWithId(int model_id, 
									  const std::vector<double>& focal_lengths,
									  const size_t width,
									  const size_t height);

	// Set the extrinsic params of each local camera, 
	// in the coordinate frame of the MCS. 
	void SetLocalCameraExtrinsics(
						 const std::vector<Eigen::Vector4d>& local_qvecs,
						 const std::vector<Eigen::Vector3d>& local_tves);
	void RescaleLocalCameraExtrinsics(double scale);

	// Obtain the params of a specified local camera
	void GetLocalCameraExtrinsic(local_camera_t camera_id, 
								  Eigen::Vector4d& qvec, 
								  Eigen::Vector3d& tvec) const;
	void GetLocalCameraIntrisic(local_camera_t camera_id,
							    std::vector<double>& params) const;

	// Access the raw parameter vector
	inline double* LocalIntrinsicParamsData();
	inline double* LocalQvecsData();
	inline double* LocalTvecsData();

	inline void SetLocalPoseDiff();
	inline double* InitLocalQvecsData();
	inline double* InitLocalTvecsData();

	// Project point in local image plane to world / infinity.
	Eigen::Vector2d LocalImageToWorld(local_camera_t camera_id, 
									  const Eigen::Vector2d& image_point) const;


	// Project point from world / infinity to image plane.
	Eigen::Vector2d WorldToLocalImage(local_camera_t camera_id,
									  const Eigen::Vector2d& world_point) const;

	Eigen::Vector2d BearingToLocalImage(local_camera_t camera_id,
									  const Eigen::Vector3d& bearing_point) const;

	// Project point in image plane to bearing for spherical camera
	Eigen::Vector3d LocalImageToBearing(local_camera_t camera_id, 
										const Eigen::Vector2d& image_point) const;


	Eigen::Vector2d BearingToImage(const Eigen::Vector3d& bearing_point) const;

	bool VerifyLocalParams() const;

	inline const std::vector<double>& LocalParams() const;
	inline std::vector<double>& LocalParams();

	inline const std::vector<double>& LocalQvecs() const;
	inline std::vector<double>& LocalQvecs();

	inline const std::vector<double>& LocalTvecs() const;
	inline std::vector<double>& LocalTvecs();

	// Convert pixel threshold in image plane to world space.
	double LocalImageToWorldThreshold(local_camera_t camera_id,
									  const double threshold) const;

    inline bool IsFromRIG() const;
    inline void SetFromRIG(bool is_rig);

	////////////////////////////////////////////////////////////////////////////////
	//  Extri members in Novatel-system
    ////////////////////////////////////////////////////////////////////////////////
	void SetDisturb();
	void SetDisturb(Eigen::Vector4d qvec, const Eigen::Vector3d tvec);
	void SetPriorDisturb(const Eigen::Vector4d& qvec, const Eigen::Vector3d& tvec);
	inline bool HasDisturb() const;

	inline const Eigen::Vector4d& QvecDisturb() const;
 	inline Eigen::Vector4d& QvecDisturb();
	inline void SetQvecDisturb(const Eigen::Vector4d& qvec);

	inline const Eigen::Vector3d& TvecDisturb() const;
    inline Eigen::Vector3d& TvecDisturb();
	inline void SetTvecDisturb(const Eigen::Vector3d& tvec);

	inline const Eigen::Vector4d& QvecPriorDisturb() const;
 	inline Eigen::Vector4d& QvecPriorDisturb();

	inline const Eigen::Vector3d& TvecPriorDisturb() const;
    inline Eigen::Vector3d& TvecPriorDisturb();


	////////////////////////////////////////////////////////////////////////////
	//  Extra functions in lidar-camera system (LCS)
    ////////////////////////////////////////////////////////////////////////////
    
	inline void SetLidarFlag(const bool has_lidar);
	inline bool HasLidar() const;
    void SetTrLidar2Camera(const Eigen::Matrix3x4d& Tr_lidar_to_cam);
    Eigen::Matrix3x4d GetLidar2Camera();
	inline const Eigen::Vector4d& LidarQvecs() const;
	inline Eigen::Vector4d& LidarQvecs();
	inline const Eigen::Vector3d& LidarTvecs() const;
	inline Eigen::Vector3d& LidarTvecs();
	
private:
	// The unique identifier of the camera. If the identifier is not specified
	// it is set to `kInvalidCameraId`.
	camera_t camera_id_;

	// The identifier of the camera model. If the camera model is not specified
	// the identifier is `kInvalidCameraModelId`.
	int model_id_;

	// The dimensions of the image, 0 if not initialized.
	size_t width_;
	size_t height_;

	// The focal length, principal point, and extra parameters. If the camera
	// model is not specified, this vector is empty.
	std::vector<double> params_;

	// Whether there is a safe prior for the focal length,
	// e.g. manually provided or extracted from EXIF
	bool prior_focal_length_;

	// Whether camera params keep constant in BA
	bool fixed_;


	////////////////////////////////////////////////////////////////////////////////
	//  Extra members in multi-camera system
    ////////////////////////////////////////////////////////////////////////////////

	local_camera_t num_local_cameras_;

	// individual camera poses in the frame of the multi-camera system (MCS), 
	// transform points in the body (MCS) frame to the individual camera frame  
	std::vector<double> local_qvecs_;
	std::vector<double> local_tvecs_;

	//individual camera params and models in MCS
	std::vector<double> local_params_;

	////////////////////////////////////////////////////////////////////////////////
	//  Extri members in Novatel-system
    ////////////////////////////////////////////////////////////////////////////////
	Eigen::Vector4d qvec_disturb_;
    Eigen::Vector3d tvec_disturb_;
	Eigen::Vector4d qvec_prior_disturb_;
    Eigen::Vector3d tvec_prior_disturb_;
	bool has_disturb_ = false;

	Eigen::Vector4d init_local_qvec_diff;
	Eigen::Vector3d init_local_tvec_diff;


	////////////////////////////////////////////////////////////////////////////////
	//  Extra members in lidar-camera system
    ////////////////////////////////////////////////////////////////////////////////

    // Whether the lidar extra is estimated. Avoid duplicated setparams. 
	bool has_lidar_ = false;
	
	// transformation from Lidar coordinates into the camera coordinate system
	Eigen::Vector4d lidar_qvecs_;
	Eigen::Vector3d lidar_tvecs_;
	
	bool is_from_rig_ = false;
};



////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////

camera_t Camera::CameraId() const { return camera_id_; }

void Camera::SetCameraId(const camera_t camera_id) { camera_id_ = camera_id; }

int Camera::ModelId() const { return model_id_; }

size_t Camera::Width() const { return width_; }

size_t Camera::Height() const { return height_; }

void Camera::SetWidth(const size_t width) { width_ = width; }

void Camera::SetHeight(const size_t height) { height_ = height; }

bool Camera::HasPriorFocalLength() const { return prior_focal_length_; }

void Camera::SetPriorFocalLength(const bool prior) {
	prior_focal_length_ = prior;
}

size_t Camera::NumParams() const { return params_.size(); }

const std::vector<double>& Camera::Params() const { return params_; }

std::vector<double>& Camera::Params() { return params_; }

double Camera::Params(const size_t idx) const { return params_[idx]; }

double& Camera::Params(const size_t idx) { return params_[idx]; }

const double* Camera::ParamsData() const { return params_.data(); }

double* Camera::ParamsData() { return params_.data(); }

void Camera::SetParams(const std::vector<double>& params) { params_ = params; }

void Camera::SetCameraConstant(bool fixed) { fixed_ = fixed; }

bool Camera::IsCameraConstant() const { return fixed_; }

double* Camera::LocalIntrinsicParamsData() {return local_params_.data();}

double* Camera::LocalQvecsData() {return local_qvecs_.data();}

double* Camera::LocalTvecsData() {return local_tvecs_.data();}

double* Camera::InitLocalQvecsData() { return init_local_qvec_diff.data(); }

double* Camera::InitLocalTvecsData() { return init_local_tvec_diff.data(); }

void Camera::SetLocalPoseDiff() {
	if (this->NumLocalCameras() == 2) {
		std::cout << "SetlocalPoseDiff" << std::endl;
		Eigen::Vector4d local_qvec0(this->LocalQvecsData());
		Eigen::Vector3d local_tvec0(this->LocalTvecsData());
		Eigen::Vector4d local_qvec1(this->LocalQvecsData());
		Eigen::Vector3d local_tvec1(this->LocalTvecsData());
		Eigen::Matrix3d R0 = QuaternionToRotationMatrix(local_qvec0);
		Eigen::Matrix3d R1 = QuaternionToRotationMatrix(local_qvec1);
		Eigen::Matrix3d R_diff = R0.transpose() * R1;

		init_local_qvec_diff = RotationMatrixToQuaternion(R_diff);
		init_local_tvec_diff =  -R1.transpose() * local_tvec1 + R0.transpose() * local_tvec0;
		std::cout << "init_local_qvec_diff: " << init_local_qvec_diff << std::endl;
		std::cout << "init_local_tvec_diff: " << init_local_tvec_diff << std::endl;
	}
}

int Camera::NumLocalCameras() const {return num_local_cameras_;}

const std::vector<double>& Camera::LocalParams() const { return local_params_; }
std::vector<double>& Camera::LocalParams() { return local_params_; }

const std::vector<double>& Camera::LocalQvecs() const { return local_qvecs_; }
std::vector<double>& Camera::LocalQvecs() { return local_qvecs_; }

const std::vector<double>& Camera::LocalTvecs() const { return local_tvecs_; }
std::vector<double>& Camera::LocalTvecs() { return local_tvecs_; }

bool Camera::HasDisturb() const {return has_disturb_; }

const Eigen::Vector4d& Camera::QvecDisturb() const {return qvec_disturb_; }
Eigen::Vector4d& Camera::QvecDisturb() {return qvec_disturb_; }
void Camera::SetQvecDisturb(const Eigen::Vector4d& qvec) { qvec_disturb_ = qvec; }

const Eigen::Vector3d& Camera::TvecDisturb() const {return tvec_disturb_; }
Eigen::Vector3d& Camera::TvecDisturb() {return tvec_disturb_; }
void Camera::SetTvecDisturb(const Eigen::Vector3d& tvec) { tvec_disturb_ = tvec; }

const Eigen::Vector4d& Camera::QvecPriorDisturb() const {return qvec_prior_disturb_; }
Eigen::Vector4d& Camera::QvecPriorDisturb() {return qvec_prior_disturb_; }

const Eigen::Vector3d& Camera::TvecPriorDisturb() const {return tvec_prior_disturb_; }
Eigen::Vector3d& Camera::TvecPriorDisturb() {return tvec_prior_disturb_; }

void Camera::SetLidarFlag(const bool has_lidar) { has_lidar_ = has_lidar; }
bool Camera::HasLidar() const { return has_lidar_; }

const Eigen::Vector4d& Camera::LidarQvecs() const { return lidar_qvecs_; }
Eigen::Vector4d& Camera::LidarQvecs() { return lidar_qvecs_; }

const Eigen::Vector3d& Camera::LidarTvecs() const { return lidar_tvecs_; }
Eigen::Vector3d& Camera::LidarTvecs() { return lidar_tvecs_; }

bool Camera::IsFromRIG() const { return  is_from_rig_;}
void Camera::SetFromRIG(bool is_rig) { is_from_rig_ = is_rig;}

} // namespace sensemap

#endif