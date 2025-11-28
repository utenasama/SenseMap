//Copyright (c) 2019, SenseTime Group.
//All rights reserved.

#ifndef SENSEMAP_BASE_CAMERA_MODELS_H_
#define SENSEMAP_BASE_CAMERA_MODELS_H_

#include <cfloat>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Dense>

#include <ceres/ceres.h>

namespace sensemap {

// This file defines several different camera models and arbitrary new camera
// models can be added by the following steps:
//
//  1. Add a new struct in this file which implements all the necessary methods.
//  2. Define an unique model_name and model_id for the camera model.
//  3. Add camera model to `CAMERA_MODEL_CASES` macro in this file.
//  4. Add new template specialization of test case for camera model to
//     `camera_models_test.cc`.
//
// A camera model can have three different types of camera parameters: focal
// length, principal point, extra parameters (distortion parameters). The
// parameter array is split into different groups, so that we can enable or
// disable the refinement of the individual groups during bundle adjustment. It
// is up to the camera model to access the parameters correctly (it is free to
// do so in an arbitrary manner) - the parameters are not accessed from outside.
//
// A camera model must have the following methods:
//
//  - `WorldToImage`: transform normalized camera coordinates to image
//    coordinates (the inverse of `ImageToWorld`). Assumes that the world
//    coordinates are given as (u, v, 1).
//  - `ImageToWorld`: transform image coordinates to normalized camera
//    coordinates (the inverse of `WorldToImage`). Produces world coordinates
//    as (u, v, 1).
//  - `ImageToWorldThreshold`: transform a threshold given in pixels to
//    normalized units (e.g. useful for reprojection error thresholds).
//
// Whenever you specify the camera parameters in a list, they must appear
// exactly in the order as they are accessed in the defined model struct.
//
// The camera models follow the convention that the upper left image corner has
// the coordinate (0, 0), the lower right corner (width, height), i.e. that
// the upper left pixel center has coordinate (0.5, 0.5) and the lower right
// pixel center has the coordinate (width - 0.5, height - 0.5).

static const int kInvalidCameraModelId = -1;

#ifndef CAMERA_MODEL_DEFINITIONS
#define CAMERA_MODEL_DEFINITIONS(model_id_value, model_name_value,           \
                             num_params_value, residual_num)                 \
static const int kModelId = model_id_value;                                  \
static const size_t kNumParams = num_params_value;                           \
static const int model_id;                                                   \
static const int kNumResidual = residual_num;                                               \
static const std::string model_name;                                         \
static const size_t num_params;                                              \
static const std::string params_info;                                        \
static const std::vector<size_t> focal_length_idxs;                          \
static const std::vector<size_t> principal_point_idxs;                       \
static const std::vector<size_t> extra_params_idxs;                          \
                                                                           \
static inline int InitializeModelId() { return model_id_value; };            \
static inline std::string InitializeModelName() {                            \
return model_name_value;                                                   \
};                                                                           \
static inline size_t InitializeNumParams() { return num_params_value; };     \
static inline std::string InitializeParamsInfo();                            \
static inline std::vector<size_t> InitializeFocalLengthIdxs();               \
static inline std::vector<size_t> InitializePrincipalPointIdxs();            \
static inline std::vector<size_t> InitializeExtraParamsIdxs();               \
static inline std::vector<double> InitializeParams(                          \
  const double focal_length, const size_t width, const size_t height);     \
                                                                           \
template <typename T>                                                        \
static void WorldToImage(const T* params, const T u, const T v, T* x, T* y); \
template <typename T>                                                        \
static void ImageToWorld(const T* params, const T x, const T y, T* u, T* v); \
template <typename T>                                                        \
static void Distortion(const T* extra_params, const T u, const T v, T* du,   \
                     T* dv);                                                \
template <typename T>                                                        \
static void WorldToBearing(const T* params, const T u, const T v, const T w, \
                           T* x, T* y, T* z);                                \
template <typename T>                                                        \
static void ImageToBearing(const T* params, const T u, const T v,           \
                           T* x, T* y, T* z);                               \
template <typename T>                                                       \
static void BearingToImage(const T* params, const T x, const T y, const T z, \
                           T* u, T* v);                             
#endif

#ifndef CAMERA_MODEL_CASES
#define CAMERA_MODEL_CASES                          \
CAMERA_MODEL_CASE(SimplePinholeCameraModel)       \
CAMERA_MODEL_CASE(PinholeCameraModel)             \
CAMERA_MODEL_CASE(SimpleRadialCameraModel)        \
CAMERA_MODEL_CASE(SimpleRadialFisheyeCameraModel) \
CAMERA_MODEL_CASE(RadialCameraModel)              \
CAMERA_MODEL_CASE(RadialFisheyeCameraModel)       \
CAMERA_MODEL_CASE(OpenCVCameraModel)              \
CAMERA_MODEL_CASE(OpenCVFisheyeCameraModel)       \
CAMERA_MODEL_CASE(FullOpenCVCameraModel)          \
CAMERA_MODEL_CASE(FOVCameraModel)                 \
CAMERA_MODEL_CASE(ThinPrismFisheyeCameraModel)    \
CAMERA_MODEL_CASE(SphericalCameraModel)           \
CAMERA_MODEL_CASE(OcamOmnidirectionalCameraModel) \
CAMERA_MODEL_CASE(UnifiedCameraModel)             \
CAMERA_MODEL_CASE(PartialOpenCVCameraModel)
#endif

#ifndef CAMERA_MODEL_SWITCH_CASES
#define CAMERA_MODEL_SWITCH_CASES         \
CAMERA_MODEL_CASES                      \
default:                                \
CAMERA_MODEL_DOES_NOT_EXIST_EXCEPTION \
break;
#endif

#define CAMERA_MODEL_DOES_NOT_EXIST_EXCEPTION \
throw std::domain_error("Camera model does not exist");

// The "Curiously Recurring Template Pattern" (CRTP) is used here, so that we
// can reuse some shared functionality between all camera models -
// defined in the BaseCameraModel.
template <typename CameraModel>
struct BaseCameraModel {
  template <typename T>
  static inline bool HasBogusParams(const std::vector<T>& params,
                                    const size_t width, const size_t height,
                                    const T min_focal_length_ratio,
                                    const T max_focal_length_ratio,
                                    const T max_extra_param);

  template <typename T>
  static inline bool HasBogusFocalLength(const std::vector<T>& params,
                                         const size_t width,
                                         const size_t height,
                                         const T min_focal_length_ratio,
                                         const T max_focal_length_ratio);

  template <typename T>
  static inline bool HasBogusPrincipalPoint(const std::vector<T>& params,
                                            const size_t width,
                                            const size_t height);

  template <typename T>
  static inline bool HasBogusExtraParams(const std::vector<T>& params,
                                         const T max_extra_param);

  template <typename T>
  static inline T ImageToWorldThreshold(const T* params, const T threshold);

  template <typename T>
  static inline void IterativeUndistortion(const T* params, T* u, T* v);
};

// Simple Pinhole camera model.
//
// No Distortion is assumed. Only focal length and principal point is modeled.
//
// Parameter list is expected in the following order:
//
//   f, cx, cy
//
// See https://en.wikipedia.org/wiki/Pinhole_camera_model
struct SimplePinholeCameraModel
    : public BaseCameraModel<SimplePinholeCameraModel> {
  CAMERA_MODEL_DEFINITIONS(0, "SIMPLE_PINHOLE", 3, 2)
};

// Pinhole camera model.
//
// No Distortion is assumed. Only focal length and principal point is modeled.
//
// Parameter list is expected in the following order:
//
//    fx, fy, cx, cy
//
// See https://en.wikipedia.org/wiki/Pinhole_camera_model
struct PinholeCameraModel : public BaseCameraModel<PinholeCameraModel> {
  CAMERA_MODEL_DEFINITIONS(1, "PINHOLE", 4, 2)
};

// Simple camera model with one focal length and one radial distortion
// parameter.
//
// This model is similar to the camera model that VisualSfM uses with the
// difference that the distortion here is applied to the projections and
// not to the measurements.
//
// Parameter list is expected in the following order:
//
//    f, cx, cy, k
//
struct SimpleRadialCameraModel
    : public BaseCameraModel<SimpleRadialCameraModel> {
  CAMERA_MODEL_DEFINITIONS(2, "SIMPLE_RADIAL", 4, 2)
};

// Simple camera model with one focal length and two radial distortion
// parameters.
//
// This model is equivalent to the camera model that Bundler uses
// (except for an inverse z-axis in the camera coordinate system).
//
// Parameter list is expected in the following order:
//
//    f, cx, cy, k1, k2
//
struct RadialCameraModel : public BaseCameraModel<RadialCameraModel> {
  CAMERA_MODEL_DEFINITIONS(3, "RADIAL", 5, 2)
};

// OpenCV camera model.
//
// Based on the pinhole camera model. Additionally models radial and
// tangential distortion (up to 2nd degree of coefficients). Not suitable for
// large radial distortions of fish-eye cameras.
//
// Parameter list is expected in the following order:
//
//    fx, fy, cx, cy, k1, k2, p1, p2
//
// See
// http://docs.opencv.org/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
struct OpenCVCameraModel : public BaseCameraModel<OpenCVCameraModel> {
  CAMERA_MODEL_DEFINITIONS(4, "OPENCV", 8, 2)
};

// OpenCV fish-eye camera model.
//
// Based on the pinhole camera model. Additionally models radial and
// tangential Distortion (up to 2nd degree of coefficients). Suitable for
// large radial distortions of fish-eye cameras.
//
// Parameter list is expected in the following order:
//
//    fx, fy, cx, cy, k1, k2, k3, k4
//
// See
// http://docs.opencv.org/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
struct OpenCVFisheyeCameraModel
    : public BaseCameraModel<OpenCVFisheyeCameraModel> {
  CAMERA_MODEL_DEFINITIONS(5, "OPENCV_FISHEYE", 8, 2)
};

// Full OpenCV camera model.
//
// Based on the pinhole camera model. Additionally models radial and
// tangential Distortion.
//
// Parameter list is expected in the following order:
//
//    fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6
//
// See
// http://docs.opencv.org/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
struct FullOpenCVCameraModel : public BaseCameraModel<FullOpenCVCameraModel> {
  CAMERA_MODEL_DEFINITIONS(6, "FULL_OPENCV", 12, 2)
};

// FOV camera model.
//
// Based on the pinhole camera model. Additionally models radial distortion.
// This model is for example used by Project Tango for its equidistant
// calibration type.
//
// Parameter list is expected in the following order:
//
//    fx, fy, cx, cy, omega
//
// See:
// Frederic Devernay, Olivier Faugeras. Straight lines have to be straight:
// Automatic calibration and removal of distortion from scenes of structured
// environments. Machine vision and applications, 2001.
struct FOVCameraModel : public BaseCameraModel<FOVCameraModel> {
  CAMERA_MODEL_DEFINITIONS(7, "FOV", 5, 2)

  template <typename T>
  static void Undistortion(const T* extra_params, const T u, const T v, T* du,
                           T* dv);
};

// Simple camera model with one focal length and one radial distortion
// parameter, suitable for fish-eye cameras.
//
// This model is equivalent to the OpenCVFisheyeCameraModel but has only one
// radial distortion coefficient.
//
// Parameter list is expected in the following order:
//
//    f, cx, cy, k
//
struct SimpleRadialFisheyeCameraModel
    : public BaseCameraModel<SimpleRadialFisheyeCameraModel> {
  CAMERA_MODEL_DEFINITIONS(8, "SIMPLE_RADIAL_FISHEYE", 4, 2)
};

// Simple camera model with one focal length and two radial distortion
// parameters, suitable for fish-eye cameras.
//
// This model is equivalent to the OpenCVFisheyeCameraModel but has only two
// radial distortion coefficients.
//
// Parameter list is expected in the following order:
//
//    f, cx, cy, k1, k2
//
struct RadialFisheyeCameraModel
    : public BaseCameraModel<RadialFisheyeCameraModel> {
  CAMERA_MODEL_DEFINITIONS(9, "RADIAL_FISHEYE", 5, 2)
};

// Camera model with radial and tangential distortion coefficients and
// additional coefficients accounting for thin-prism distortion.
//
// This camera model is described in
//
//    "Camera Calibration with Distortion Models and Accuracy Evaluation",
//    J Weng et al., TPAMI, 1992.
//
// Parameter list is expected in the following order:
//
//    fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, sx1, sy1
//
struct ThinPrismFisheyeCameraModel
    : public BaseCameraModel<ThinPrismFisheyeCameraModel> {
  CAMERA_MODEL_DEFINITIONS(10, "THIN_PRISM_FISHEYE", 12, 2)
};

// Camera model with panorama image
//
// This camera model is described in
//
//    "Gnomonic Projection, http://mathworld.wolfram.com/GnomonicProjection.html"
//
// Paramater list is expected in the following order:
//
//    f, image_width, image_height, 
//      camera_1_qw, camera_1_qx, camera_1_qy, camera_1_qz, camera_1_tx, camera_1_ty, camera_1_tz,
//      camera_2_qw, camera_2_qx, camera_2_qy, camera_2_qz, camera_2_tx, camera_2_ty, camera_2_tz
//
struct SphericalCameraModel
    : public BaseCameraModel<SphericalCameraModel> {
  CAMERA_MODEL_DEFINITIONS(11, "SPHERICAL", 17, 3)      
};

// This camera model is described in the Ocam calibration tool box

// Parameters are as follows: 
// f, a synthesized focal length
// cx, cy, the image center
// ocam_model.invpol used by world2cam have 12 coefficients of the inverse
// mapping polynomial 
// ocam_model.ss used by cam2world have 5 coefficients of the mapping polynomial
// c, d, e, the affine parameter for non-square pixel

struct OcamOmnidirectionalCameraModel
    : public BaseCameraModel<OcamOmnidirectionalCameraModel> {
  CAMERA_MODEL_DEFINITIONS(12, "OCAM_OMNIDIRECTIONAL", 23, 2)      
};


// This camera model is described in the kalibr tool box
// parameters are as follows:
// xi, the mirror parameter
// fx, fy, focal-length
// cx, cy, principle point
// [k1 k2 r1 r2] : radial-tangential (radtan) distortion

struct UnifiedCameraModel
    : public BaseCameraModel<UnifiedCameraModel> {
  CAMERA_MODEL_DEFINITIONS(13, "UNIFIED", 9, 2)      
};

// Partial OpenCV camera model.
//
// Based on the pinhole camera model. Additionally models radial and
// tangential Distortion.
//
// Parameter list is expected in the following order:
//
//    fx, fy, cx, cy, k1, k2, p1, p2, k3
//
struct PartialOpenCVCameraModel : public BaseCameraModel<PartialOpenCVCameraModel> {
  CAMERA_MODEL_DEFINITIONS(14, "PARTIAL_OPENCV", 9, 2)
};

// Check whether camera model with given name or identifier exists.
bool ExistsCameraModelWithName(const std::string& model_name);
bool ExistsCameraModelWithId(const int model_id);

// Convert camera name to unique camera model identifier.
//
// @param name         Unique name of camera model.
//
// @return             Unique identifier of camera model.
int CameraModelNameToId(const std::string& model_name);

// Convert camera model identifier to unique camera model name.
//
// @param model_id     Unique identifier of camera model.
//
// @return             Unique name of camera model.
std::string CameraModelIdToName(const int model_id);

// Initialize camera parameters using given image properties.
//
// Initializes all focal length parameters to the same given focal length and
// sets the principal point to the image center.
//
// @param model_id      Unique identifier of camera model.
// @param focal_length  Focal length, equal for all focal length parameters.
// @param width         Sensor width of the camera.
// @param height        Sensor height of the camera.
std::vector<double> CameraModelInitializeParams(const int model_id,
                                                const double focal_length,
                                                const size_t width,
                                                const size_t height);

// Get human-readable information about the parameter vector order.
//
// @param model_id     Unique identifier of camera model.
std::string CameraModelParamsInfo(const int model_id);

// Get the indices of the parameter groups in the parameter vector.
//
// @param model_id     Unique identifier of camera model.
const std::vector<size_t>& CameraModelFocalLengthIdxs(const int model_id);
const std::vector<size_t>& CameraModelPrincipalPointIdxs(const int model_id);
const std::vector<size_t>& CameraModelExtraParamsIdxs(const int model_id);

// Get the total number of parameters of a camera model.
size_t CameraModelNumParams(const int model_id);

// Check whether parameters are valid, i.e. the parameter vector has
// the correct dimensions that match the specified camera model.
//
// @param model_id      Unique identifier of camera model.
// @param params        Array of camera parameters.
bool CameraModelVerifyParams(const int model_id,
                             const std::vector<double>& params);

// Check whether camera has bogus parameters.
//
// @param model_id                Unique identifier of camera model.
// @param params                  Array of camera parameters.
// @param width                   Sensor width of the camera.
// @param height                  Sensor height of the camera.
// @param min_focal_length_ratio  Minimum ratio of focal length over
//                                maximum sensor dimension.
// @param min_focal_length_ratio  Maximum ratio of focal length over
//                                maximum sensor dimension.
// @param max_extra_param         Maximum magnitude of each extra parameter.
bool CameraModelHasBogusParams(const int model_id,
                               const std::vector<double>& params,
                               const size_t width, const size_t height,
                               const double min_focal_length_ratio,
                               const double max_focal_length_ratio,
                               const double max_extra_param);

// Transform world coordinates in camera coordinate system to image coordinates.
//
// This is the inverse of `CameraModelImageToWorld`.
//
// @param model_id     Unique model_id of camera model as defined in
//                     `CAMERA_MODEL_NAME_TO_CODE`.
// @param params       Array of camera parameters.
// @param u, v         Coordinates in camera system as (u, v, 1).
// @param x, y         Output image coordinates in pixels.
inline void CameraModelWorldToImage(const int model_id,
                                    const std::vector<double>& params,
                                    const double u, const double v, double* x,
                                    double* y);

// Transform image coordinates to world coordinates in camera coordinate system.
//
// This is the inverse of `CameraModelWorldToImage`.
//
// @param model_id      Unique identifier of camera model.
// @param params        Array of camera parameters.
// @param x, y          Image coordinates in pixels.
// @param v, u          Output Coordinates in camera system as (u, v, 1).
inline void CameraModelImageToWorld(const int model_id,
                                    const std::vector<double>& params,
                                    const double x, const double y, double* u,
                                    double* v);

// Convert pixel threshold in image plane to world space by dividing
// the threshold through the mean focal length.
//
// @param model_id      Unique identifier of camera model.
// @param params        Array of camera parameters.
// @param threshold     Image space threshold in pixels.
//
// @ return             World space threshold.
inline double CameraModelImageToWorldThreshold(
    const int model_id, const std::vector<double>& params,
    const double threshold);


// Transform image coordinates to coordinates on the bearing. For perspective
// camera the bearing is a plane, for spherical camera the bearing is a
// shperical surface.
inline void CameraModelImageToBearing(const int model_id,
                             const std::vector<double>& params, const double u,
                             const double v, double* x, double* y, double* z);

// Transform world coordinates onto the bearing.
inline void CameraModelWorldToBearing(const int model_id,
                             const std::vector<double>& params, const double u,
                             const double v, const double w, 
                             double* x, double* y, double* z); 

inline void CameraModelBearingToImage(const int model_id,
                             const std::vector<double>& params, const double u,
                             const double v, const double w, 
                             double* x, double* y);  

////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// BaseCameraModel

template <typename CameraModel>
template <typename T>
bool BaseCameraModel<CameraModel>::HasBogusParams(
    const std::vector<T>& params, const size_t width, const size_t height,
    const T min_focal_length_ratio, const T max_focal_length_ratio,
    const T max_extra_param) {
    if (HasBogusPrincipalPoint(params, width, height)) {
        return true;
    }

    if (HasBogusFocalLength(params, width, height, min_focal_length_ratio,
                            max_focal_length_ratio)) {
        return true;
    }

    if (HasBogusExtraParams(params, max_extra_param)) {
        return true;
    }

    return false;
}

template <typename CameraModel>
template <typename T>
bool BaseCameraModel<CameraModel>::HasBogusFocalLength(
    const std::vector<T>& params, const size_t width, const size_t height,
    const T min_focal_length_ratio, const T max_focal_length_ratio) {
    const size_t max_size = std::max(width, height);

    for (const auto& idx : CameraModel::focal_length_idxs) {
        const T focal_length_ratio = params[idx] / max_size;
        if (focal_length_ratio < min_focal_length_ratio ||
            focal_length_ratio > max_focal_length_ratio) {
            return true;
        }
    }

    return false;
}

template <typename CameraModel>
template <typename T>
bool BaseCameraModel<CameraModel>::HasBogusPrincipalPoint(
    const std::vector<T>& params, const size_t width, const size_t height) {
    const T cx = params[CameraModel::principal_point_idxs[0]];
    const T cy = params[CameraModel::principal_point_idxs[1]];
    return cx < 0 || cx > width || cy < 0 || cy > height;
}

template <typename CameraModel>
template <typename T>
bool BaseCameraModel<CameraModel>::HasBogusExtraParams(
    const std::vector<T>& params, const T max_extra_param) {
    for (const auto& idx : CameraModel::extra_params_idxs) {
        if (std::abs(params[idx]) >= max_extra_param) {
            return true;
        }
    }

    return false;
}

template <typename CameraModel>
template <typename T>
T BaseCameraModel<CameraModel>::ImageToWorldThreshold(const T* params,
                                                      const T threshold) {
    T mean_focal_length = 0;
    for (const auto& idx : CameraModel::focal_length_idxs) {
        mean_focal_length += params[idx];
    }
    mean_focal_length /= CameraModel::focal_length_idxs.size();
    return threshold / mean_focal_length;
}

template <typename CameraModel>
template <typename T>
void BaseCameraModel<CameraModel>::IterativeUndistortion(const T* params, T* u,
                                                         T* v) {
    // Parameters for Newton iteration using numerical differentiation with
    // central differences, 100 iterations should be enough even for complex
    // camera models with higher order terms.
    
    const size_t kNumIterations = 100;
    const double kMaxStepNorm = 1e-10;
    const double kRelStepSize = 1e-6;

    Eigen::Matrix<T,2,2> J;
    const Eigen::Matrix<T,2,1> x0(*u, *v);    
    Eigen::Matrix<T,2,1> x(*u, *v);
    Eigen::Matrix<T,2,1> dx;
    Eigen::Matrix<T,2,1> dx_0b;
    Eigen::Matrix<T,2,1> dx_0f;
    Eigen::Matrix<T,2,1> dx_1b;
    Eigen::Matrix<T,2,1> dx_1f;

    for (size_t i = 0; i < kNumIterations; ++i) {
        // const T step0 = ceres::max(std::numeric_limits<double>::epsilon(),
        //                               ceres::abs(kRelStepSize * x(0)));
        // const T step1 = cers::max(std::numeric_limits<double>::epsilon(),
        //                               ceres::abs(kRelStepSize * x(1)));

        const T step0 = (T(std::numeric_limits<double>::epsilon()) > ceres::abs(kRelStepSize * x(0)))? 
            T(std::numeric_limits<double>::epsilon()): ceres::abs(kRelStepSize * x(0));
        const T step1 = (T(std::numeric_limits<double>::epsilon()) > ceres::abs(kRelStepSize * x(1)))? 
            T(std::numeric_limits<double>::epsilon()): ceres::abs(kRelStepSize * x(1));

        CameraModel::Distortion(params, x(0), x(1), &dx(0),  &dx(1));
        CameraModel::Distortion(params, x(0) - step0, x(1), &dx_0b(0), &dx_0b(1));
        CameraModel::Distortion(params, x(0) + step0, x(1), &dx_0f(0), &dx_0f(1));
        CameraModel::Distortion(params, x(0), x(1) - step1, &dx_1b(0), &dx_1b(1));
        CameraModel::Distortion(params, x(0), x(1) + step1, &dx_1f(0), &dx_1f(1));
        J(0, 0) = 1.0 + (dx_0f(0) - dx_0b(0)) / (2.0 * step0);
        J(0, 1) = (dx_1f(0) - dx_1b(0)) / (2.0 * step1);
        J(1, 0) = (dx_0f(1) - dx_0b(1)) / (2.0 * step0);
        J(1, 1) = 1.0 + (dx_1f(1) - dx_1b(1)) / (2.0 * step1);
        const Eigen::Matrix<T,2,1> step_x = J.inverse() * (x + dx - x0);
        x -= step_x;
        if (step_x.squaredNorm() < kMaxStepNorm) {
            break;
        }
    }

    // const size_t kNumIterations = 100;
    // const T kMaxStepNorm = T(1e-10);
    // const T kRelStepSize = T(1e-6);
    

    // Eigen::Matrix<T,2,2> J;
    // const Eigen::Matrix<T,2,1> x0 (*u,*v);
    
    // Eigen::Matrix<T,2,1> x(*u, *v);
    // Eigen::Matrix<T,2,1> dx;
    // Eigen::Matrix<T,2,1> dx_0b;
    // Eigen::Matrix<T,2,1> dx_0f;
    // Eigen::Matrix<T,2,1> dx_1b;
    // Eigen::Matrix<T,2,1> dx_1f;

    // for (size_t i = 0; i < kNumIterations; ++i) {
    //     const T step0 = //ceres::max(T(std::numeric_limits<double>::epsilon()),
    //                     ceres::abs(kRelStepSize * x(0))>T(std::numeric_limits<double>::epsilon())?ceres::abs(kRelStepSize * x(0)):T(std::numeric_limits<double>::epsilon());
        
    //     const T step1 = //ceres::max(T(std::numeric_limits<double>::epsilon()),
    //                     ceres::abs(kRelStepSize * x(1))>T(std::numeric_limits<double>::epsilon())?ceres::abs(kRelStepSize * x(0)):T(std::numeric_limits<double>::epsilon());

    //     CameraModel::Distortion(params, x(0), x(1), &dx(0),  &dx(1));
    //     CameraModel::Distortion(params, x(0) - step0, x(1), &dx_0b(0), &dx_0b(1));
    //     CameraModel::Distortion(params, x(0) + step0, x(1), &dx_0f(0), &dx_0f(1));
    //     CameraModel::Distortion(params, x(0), x(1) - step1, &dx_1b(0), &dx_1b(1));
    //     CameraModel::Distortion(params, x(0), x(1) + step1, &dx_1f(0), &dx_1f(1));
    //     J(0, 0) = T(1) + (dx_0f(0) - dx_0b(0)) / (T(2) * step0);
    //     J(0, 1) = (dx_1f(0) - dx_1b(0)) / (T(2) * step1);
    //     J(1, 0) = (dx_0f(1) - dx_0b(1)) / (T(2) * step0);
    //     J(1, 1) = T(1) + (dx_1f(1) - dx_1b(1)) / (T(2) * step1);
    //     const Eigen::Matrix<T,2,1> step_x = J.inverse() * (x + dx - x0);
    //     x -= step_x;
    //     if (step_x.squaredNorm() < kMaxStepNorm) {
    //         break;
    //     }
    // }

    *u = T(x(0));
    *v = T(x(1));
}

////////////////////////////////////////////////////////////////////////////////
// SimplePinholeCameraModel

std::string SimplePinholeCameraModel::InitializeParamsInfo() {
    return "f, cx, cy";
}

std::vector<size_t> SimplePinholeCameraModel::InitializeFocalLengthIdxs() {
    return {0};
}

std::vector<size_t> SimplePinholeCameraModel::InitializePrincipalPointIdxs() {
    return {1, 2};
}

std::vector<size_t> SimplePinholeCameraModel::InitializeExtraParamsIdxs() {
    return {};
}

std::vector<double> SimplePinholeCameraModel::InitializeParams(
    const double focal_length, const size_t width, const size_t height) {
    return {focal_length, width / 2.0, height / 2.0};
}

template <typename T>
void SimplePinholeCameraModel::WorldToImage(const T* params, const T u,
                                            const T v, T* x, T* y) {
    const T f = params[0];
    const T c1 = params[1];
    const T c2 = params[2];

    // No Distortion

    // Transform to image coordinates
    *x = f * u + c1;
    *y = f * v + c2;
}

template <typename T>
void SimplePinholeCameraModel::ImageToWorld(const T* params, const T x,
                                            const T y, T* u, T* v) {
    const T f = params[0];
    const T c1 = params[1];
    const T c2 = params[2];

    *u = (x - c1) / f;
    *v = (y - c2) / f;
}

template <typename T>
void SimplePinholeCameraModel::WorldToBearing(const T* params, const T u,
                                              const T v, const T w, 
                                              T* x, T* y, T*z) {
    *x = u/w;
    *y = v/w;
    *z = w/w;
}

template <typename T>
void SimplePinholeCameraModel::ImageToBearing(const T* params, const T x,
                                            const T y, T* u, T* v, T* w) {
    const T f = params[0];
    const T c1 = params[1];
    const T c2 = params[2];

    *u = (x - c1) / f;
    *v = (y - c2) / f;
    *w = T(1.0);
}

template <typename T>
void SimplePinholeCameraModel::BearingToImage(const T* params, const T x,
                                              const T y, const T z, 
                                              T* u, T* v) {
}


////////////////////////////////////////////////////////////////////////////////
// PinholeCameraModel

std::string PinholeCameraModel::InitializeParamsInfo() {
    return "fx, fy, cx, cy";
}

std::vector<size_t> PinholeCameraModel::InitializeFocalLengthIdxs() {
    return {0, 1};
}

std::vector<size_t> PinholeCameraModel::InitializePrincipalPointIdxs() {
    return {2, 3};
}

std::vector<size_t> PinholeCameraModel::InitializeExtraParamsIdxs() {
    return {};
}

std::vector<double> PinholeCameraModel::InitializeParams(
    const double focal_length, const size_t width, const size_t height) {
    return {focal_length, focal_length, width / 2.0, height / 2.0};
}

template <typename T>
void PinholeCameraModel::WorldToImage(const T* params, const T u, const T v,
                                      T* x, T* y) {
    const T f1 = params[0];
    const T f2 = params[1];
    const T c1 = params[2];
    const T c2 = params[3];

    // No Distortion

    // Transform to image coordinates
    *x = f1 * u + c1;
    *y = f2 * v + c2;
}

template <typename T>
void PinholeCameraModel::ImageToWorld(const T* params, const T x, const T y,
                                      T* u, T* v) {
    const T f1 = params[0];
    const T f2 = params[1];
    const T c1 = params[2];
    const T c2 = params[3];

    *u = (x - c1) / f1;
    *v = (y - c2) / f2;
}


template <typename T>
void PinholeCameraModel::WorldToBearing(const T* params, const T u,
                                              const T v, const T w, 
                                              T* x, T* y, T*z) {
    *x = u/w;
    *y = v/w;
    *z = w/w;
}

template <typename T>
void PinholeCameraModel::ImageToBearing(const T* params, const T x,
                                        const T y, T* u, T* v, T* w) {
    const T f1 = params[0];
    const T f2 = params[1];
    const T c1 = params[2];
    const T c2 = params[3];

    *u = (x - c1) / f1;
    *v = (y - c2) / f2;
    *w = T(1.0);
}

template <typename T>
void PinholeCameraModel::BearingToImage(const T* params, const T x,
                                              const T y, const T z, 
                                              T* u, T* v) {
}


////////////////////////////////////////////////////////////////////////////////
// SimpleRadialCameraModel

std::string SimpleRadialCameraModel::InitializeParamsInfo() {
    return "f, cx, cy, k";
}

std::vector<size_t> SimpleRadialCameraModel::InitializeFocalLengthIdxs() {
    return {0};
}

std::vector<size_t> SimpleRadialCameraModel::InitializePrincipalPointIdxs() {
    return {1, 2};
}

std::vector<size_t> SimpleRadialCameraModel::InitializeExtraParamsIdxs() {
    return {3};
}

std::vector<double> SimpleRadialCameraModel::InitializeParams(
    const double focal_length, const size_t width, const size_t height) {
    return {focal_length, width / 2.0, height / 2.0, 0};
}

template <typename T>
void SimpleRadialCameraModel::WorldToImage(const T* params, const T u,
                                           const T v, T* x, T* y) {
    const T f = params[0];
    const T c1 = params[1];
    const T c2 = params[2];

    // Distortion
    T du, dv;
    Distortion(&params[3], u, v, &du, &dv);
    *x = u + du;
    *y = v + dv;

    // Transform to image coordinates
    *x = f * *x + c1;
    *y = f * *y + c2;
}

template <typename T>
void SimpleRadialCameraModel::ImageToWorld(const T* params, const T x,
                                           const T y, T* u, T* v) {
    const T f = params[0];
    const T c1 = params[1];
    const T c2 = params[2];

    // Lift points to normalized plane
    *u = (x - c1) / f;
    *v = (y - c2) / f;

    IterativeUndistortion(&params[3], u, v);
}

template <typename T>
void SimpleRadialCameraModel::Distortion(const T* extra_params, const T u,
                                         const T v, T* du, T* dv) {
    const T k = extra_params[0];

    const T u2 = u * u;
    const T v2 = v * v;
    const T r2 = u2 + v2;
    const T radial = k * r2;
    *du = u * radial;
    *dv = v * radial;
}


template <typename T>
void SimpleRadialCameraModel::WorldToBearing(const T* params, const T u,
                                              const T v, const T w, 
                                              T* x, T* y, T*z) {
    *x = u/w;
    *y = v/w;
    *z = w/w;
}

template <typename T>
void SimpleRadialCameraModel::ImageToBearing(const T* params, const T x,
                                        const T y, T* u, T* v, T* w) {
    const T f = params[0];
    const T c1 = params[1];
    const T c2 = params[2];

    // Lift points to normalized plane
    *u = (x - c1) / f;
    *v = (y - c2) / f;

    IterativeUndistortion(&params[3], u, v);
    *w = T(1.0);
}

template <typename T>
void SimpleRadialCameraModel::BearingToImage(const T* params, const T x,
                                              const T y, const T z, 
                                              T* u, T* v) {
}


////////////////////////////////////////////////////////////////////////////////
// RadialCameraModel

std::string RadialCameraModel::InitializeParamsInfo() {
    return "f, cx, cy, k1, k2";
}

std::vector<size_t> RadialCameraModel::InitializeFocalLengthIdxs() {
    return {0};
}

std::vector<size_t> RadialCameraModel::InitializePrincipalPointIdxs() {
    return {1, 2};
}

std::vector<size_t> RadialCameraModel::InitializeExtraParamsIdxs() {
    return {3, 4};
}

std::vector<double> RadialCameraModel::InitializeParams(
    const double focal_length, const size_t width, const size_t height) {
    return {focal_length, width / 2.0, height / 2.0, 0, 0};
}

template <typename T>
void RadialCameraModel::WorldToImage(const T* params, const T u, const T v,
                                     T* x, T* y) {
    const T f = params[0];
    const T c1 = params[1];
    const T c2 = params[2];

    // Distortion
    T du, dv;
    Distortion(&params[3], u, v, &du, &dv);
    *x = u + du;
    *y = v + dv;

    // Transform to image coordinates
    *x = f * *x + c1;
    *y = f * *y + c2;
}

template <typename T>
void RadialCameraModel::ImageToWorld(const T* params, const T x, const T y,
                                     T* u, T* v) {
    const T f = params[0];
    const T c1 = params[1];
    const T c2 = params[2];

    // Lift points to normalized plane
    *u = (x - c1) / f;
    *v = (y - c2) / f;

    IterativeUndistortion(&params[3], u, v);
}

template <typename T>
void RadialCameraModel::Distortion(const T* extra_params, const T u, const T v,
                                   T* du, T* dv) {
    const T k1 = extra_params[0];
    const T k2 = extra_params[1];

    const T u2 = u * u;
    const T v2 = v * v;
    const T r2 = u2 + v2;
    const T radial = k1 * r2 + k2 * r2 * r2;
    *du = u * radial;
    *dv = v * radial;
}

template <typename T>
void RadialCameraModel::WorldToBearing(const T* params, const T u,
                                              const T v, const T w, 
                                              T* x, T* y, T*z) {
    *x = u/w;
    *y = v/w;
    *z = w/w;
}

template <typename T>
void RadialCameraModel::ImageToBearing(const T* params, const T x,
                                        const T y, T* u, T* v, T* w) {
    const T f = params[0];
    const T c1 = params[1];
    const T c2 = params[2];

    // Lift points to normalized plane
    *u = (x - c1) / f;
    *v = (y - c2) / f;

    IterativeUndistortion(&params[3], u, v);                                        
    *w = T(1.0);
}

template <typename T>
void RadialCameraModel::BearingToImage(const T* params, const T x,
                                              const T y, const T z, 
                                              T* u, T* v) {
}

////////////////////////////////////////////////////////////////////////////////
// OpenCVCameraModel

std::string OpenCVCameraModel::InitializeParamsInfo() {
    return "fx, fy, cx, cy, k1, k2, p1, p2";
}

std::vector<size_t> OpenCVCameraModel::InitializeFocalLengthIdxs() {
    return {0, 1};
}

std::vector<size_t> OpenCVCameraModel::InitializePrincipalPointIdxs() {
    return {2, 3};
}

std::vector<size_t> OpenCVCameraModel::InitializeExtraParamsIdxs() {
    return {4, 5, 6, 7};
}

std::vector<double> OpenCVCameraModel::InitializeParams(
    const double focal_length, const size_t width, const size_t height) {
    return {focal_length, focal_length, width / 2.0, height / 2.0, 0, 0, 0, 0};
}

template <typename T>
void OpenCVCameraModel::WorldToImage(const T* params, const T u, const T v,
                                     T* x, T* y) {
    const T f1 = params[0];
    const T f2 = params[1];
    const T c1 = params[2];
    const T c2 = params[3];

    // Distortion
    T du, dv;
    Distortion(&params[4], u, v, &du, &dv);
    *x = u + du;
    *y = v + dv;

    // Transform to image coordinates
    *x = f1 * *x + c1;
    *y = f2 * *y + c2;
}

template <typename T>
void OpenCVCameraModel::ImageToWorld(const T* params, const T x, const T y,
                                     T* u, T* v) {
    const T f1 = params[0];
    const T f2 = params[1];
    const T c1 = params[2];
    const T c2 = params[3];

    // Lift points to normalized plane
    *u = (x - c1) / f1;
    *v = (y - c2) / f2;

    IterativeUndistortion(&params[4], u, v);
}

template <typename T>
void OpenCVCameraModel::Distortion(const T* extra_params, const T u, const T v,
                                   T* du, T* dv) {
    const T k1 = extra_params[0];
    const T k2 = extra_params[1];
    const T p1 = extra_params[2];
    const T p2 = extra_params[3];

    const T u2 = u * u;
    const T uv = u * v;
    const T v2 = v * v;
    const T r2 = u2 + v2;
    const T radial = k1 * r2 + k2 * r2 * r2;
    *du = u * radial + T(2) * p1 * uv + p2 * (r2 + T(2) * u2);
    *dv = v * radial + T(2) * p2 * uv + p1 * (r2 + T(2) * v2);
}

template <typename T>
void OpenCVCameraModel::WorldToBearing(const T* params, const T u,
                                              const T v, const T w, 
                                              T* x, T* y, T*z) {
    *x = u/w;
    *y = v/w;
    *z = w/w;
}

template <typename T>
void OpenCVCameraModel::ImageToBearing(const T* params, const T x,
                                        const T y, T* u, T* v, T* w) {
    const T f1 = params[0];
    const T f2 = params[1];
    const T c1 = params[2];
    const T c2 = params[3];

    // Lift points to normalized plane
    *u = (x - c1) / f1;
    *v = (y - c2) / f2;

    IterativeUndistortion(&params[4], u, v);
    *w = T(1.0);
}

template <typename T>
void OpenCVCameraModel::BearingToImage(const T* params, const T x,
                                              const T y, const T z, 
                                              T* u, T* v) {
}


////////////////////////////////////////////////////////////////////////////////
// OpenCVFisheyeCameraModel

std::string OpenCVFisheyeCameraModel::InitializeParamsInfo() {
    return "fx, fy, cx, cy, k1, k2, k3, k4";
}

std::vector<size_t> OpenCVFisheyeCameraModel::InitializeFocalLengthIdxs() {
    return {0, 1};
}

std::vector<size_t> OpenCVFisheyeCameraModel::InitializePrincipalPointIdxs() {
    return {2, 3};
}

std::vector<size_t> OpenCVFisheyeCameraModel::InitializeExtraParamsIdxs() {
    return {4, 5, 6, 7};
}

std::vector<double> OpenCVFisheyeCameraModel::InitializeParams(
    const double focal_length, const size_t width, const size_t height) {
    return {focal_length, focal_length, width / 2.0, height / 2.0, 0, 0, 0, 0};
}

template <typename T>
void OpenCVFisheyeCameraModel::WorldToImage(const T* params, const T u,
                                            const T v, T* x, T* y) {
    const T f1 = params[0];
    const T f2 = params[1];
    const T c1 = params[2];
    const T c2 = params[3];

    // Distortion
    T du, dv;
    Distortion(&params[4], u, v, &du, &dv);
    *x = u + du;
    *y = v + dv;

    // Transform to image coordinates
    *x = f1 * *x + c1;
    *y = f2 * *y + c2;
}

template <typename T>
void OpenCVFisheyeCameraModel::ImageToWorld(const T* params, const T x,
                                            const T y, T* u, T* v) {
    const T f1 = params[0];
    const T f2 = params[1];
    const T c1 = params[2];
    const T c2 = params[3];

    // Lift points to normalized plane
    *u = (x - c1) / f1;
    *v = (y - c2) / f2;

    IterativeUndistortion(&params[4], u, v);
}

template <typename T>
void OpenCVFisheyeCameraModel::Distortion(const T* extra_params, const T u,
                                          const T v, T* du, T* dv) {
    const T k1 = extra_params[0];
    const T k2 = extra_params[1];
    const T k3 = extra_params[2];
    const T k4 = extra_params[3];

    const T r = ceres::sqrt(u * u + v * v);

    if (r > T(std::numeric_limits<double>::epsilon())) {
        const T theta = ceres::atan(r);
        const T theta2 = theta * theta;
        const T theta4 = theta2 * theta2;
        const T theta6 = theta4 * theta2;
        const T theta8 = theta4 * theta4;
        const T thetad =
            theta * (T(1) + k1 * theta2 + k2 * theta4 + k3 * theta6 + k4 * theta8);
        *du = u * thetad / r - u;
        *dv = v * thetad / r - v;
    } else {
        *du = T(0);
        *dv = T(0);
    }
}

template <typename T>
void OpenCVFisheyeCameraModel::WorldToBearing(const T* params, const T u,
                                              const T v, const T w, 
                                              T* x, T* y, T*z) {
    *x = u/w;
    *y = v/w;
    *z = w/w;
}

template <typename T>
void OpenCVFisheyeCameraModel::ImageToBearing(const T* params, const T x,
                                        const T y, T* u, T* v, T* w) {
    
    const T f1 = params[0];
    const T f2 = params[1];
    const T c1 = params[2];
    const T c2 = params[3];

    // Lift points to normalized plane
    *u = (x - c1) / f1;
    *v = (y - c2) / f2;

    IterativeUndistortion(&params[4], u, v);
    *w = T(1.0);


    T norm = ceres::sqrt((*u)*(*u)+(*v)*(*v)+(*w)*(*w));

    if(norm > T(0)){
        *u = *u /norm;
        *v = *v /norm;
        *w = *w /norm;
    }
    else{
        *u = T(0);
        *v = T(0);
        *w = T(0);
    }
}

template <typename T>
void OpenCVFisheyeCameraModel::BearingToImage(const T* params, const T u,
                                              const T v, const T w, 
                                              T* x, T* y) {

    const T f1 = params[0];
    const T f2 = params[1];
    const T c1 = params[2];
    const T c2 = params[3];

    const T u_n = u / w;
    const T v_n = v / w;

    // Distortion
    T du, dv;
    Distortion(&params[4], u_n, v_n, &du, &dv);
    *x = u_n + du;
    *y = v_n + dv;

    // Transform to image coordinates
    *x = f1 * *x + c1;
    *y = f2 * *y + c2;

}

////////////////////////////////////////////////////////////////////////////////
// PartialOpenCVCameraModel

std::string PartialOpenCVCameraModel::InitializeParamsInfo() {
    return "fx, fy, cx, cy, k1, k2, p1, p2, k3";
}

std::vector<size_t> PartialOpenCVCameraModel::InitializeFocalLengthIdxs() {
    return {0, 1};
}

std::vector<size_t> PartialOpenCVCameraModel::InitializePrincipalPointIdxs() {
    return {2, 3};
}

std::vector<size_t> PartialOpenCVCameraModel::InitializeExtraParamsIdxs() {
    return {4, 5, 6, 7, 8};
}

std::vector<double> PartialOpenCVCameraModel::InitializeParams(
    const double focal_length, const size_t width, const size_t height) {
    return {focal_length,
            focal_length,
            width / 2.0,
            height / 2.0,
            0,
            0,
            0,
            0,
            0};
}

template <typename T>
void PartialOpenCVCameraModel::WorldToImage(const T* params, const T u, const T v,
                                         T* x, T* y) {
    const T f1 = params[0];
    const T f2 = params[1];
    const T c1 = params[2];
    const T c2 = params[3];

    // Distortion
    T du, dv;
    Distortion(&params[4], u, v, &du, &dv);
    *x = u + du;
    *y = v + dv;

    // Transform to image coordinates
    *x = f1 * *x + c1;
    *y = f2 * *y + c2;
}

template <typename T>
void PartialOpenCVCameraModel::ImageToWorld(const T* params, const T x, const T y,
                                         T* u, T* v) {
    const T f1 = params[0];
    const T f2 = params[1];
    const T c1 = params[2];
    const T c2 = params[3];

    // Lift points to normalized plane
    *u = (x - c1) / f1;
    *v = (y - c2) / f2;

    IterativeUndistortion(&params[4], u, v);
}

template <typename T>
void PartialOpenCVCameraModel::Distortion(const T* extra_params, const T u,
                                       const T v, T* du, T* dv) {
    const T k1 = extra_params[0];
    const T k2 = extra_params[1];
    const T p1 = extra_params[2];
    const T p2 = extra_params[3];
    const T k3 = extra_params[4];

    const T u2 = u * u;
    const T uv = u * v;
    const T v2 = v * v;
    const T r2 = u2 + v2;
    const T r4 = r2 * r2;
    const T r6 = r4 * r2;
    const T radial = (T(1) + k1 * r2 + k2 * r4 + k3 * r6);
    *du = u * radial + T(2) * p1 * uv + p2 * (r2 + T(2) * u2) - u;
    *dv = v * radial + T(2) * p2 * uv + p1 * (r2 + T(2) * v2) - v;
}

template <typename T>
void PartialOpenCVCameraModel::WorldToBearing(const T* params, const T u,
                                              const T v, const T w, 
                                              T* x, T* y, T*z) {
    *x = u/w;
    *y = v/w;
    *z = w/w;
}

template <typename T>
void PartialOpenCVCameraModel::ImageToBearing(const T* params, const T x,
                                        const T y, T* u, T* v, T* w) {
    
    const T f1 = params[0];
    const T f2 = params[1];
    const T c1 = params[2];
    const T c2 = params[3];

    // Lift points to normalized plane
    *u = (x - c1) / f1;
    *v = (y - c2) / f2;

    IterativeUndistortion(&params[4], u, v);
    * w = T(1.0);
}

template <typename T>
void PartialOpenCVCameraModel::BearingToImage(const T* params, const T x,
                                              const T y, const T z, 
                                              T* u, T* v) {
}

////////////////////////////////////////////////////////////////////////////////
// FullOpenCVCameraModel

std::string FullOpenCVCameraModel::InitializeParamsInfo() {
    return "fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6";
}

std::vector<size_t> FullOpenCVCameraModel::InitializeFocalLengthIdxs() {
    return {0, 1};
}

std::vector<size_t> FullOpenCVCameraModel::InitializePrincipalPointIdxs() {
    return {2, 3};
}

std::vector<size_t> FullOpenCVCameraModel::InitializeExtraParamsIdxs() {
    return {4, 5, 6, 7, 8, 9, 10, 11};
}

std::vector<double> FullOpenCVCameraModel::InitializeParams(
    const double focal_length, const size_t width, const size_t height) {
    return {focal_length,
            focal_length,
            width / 2.0,
            height / 2.0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0};
}

template <typename T>
void FullOpenCVCameraModel::WorldToImage(const T* params, const T u, const T v,
                                         T* x, T* y) {
    const T f1 = params[0];
    const T f2 = params[1];
    const T c1 = params[2];
    const T c2 = params[3];

    // Distortion
    T du, dv;
    Distortion(&params[4], u, v, &du, &dv);
    *x = u + du;
    *y = v + dv;

    // Transform to image coordinates
    *x = f1 * *x + c1;
    *y = f2 * *y + c2;
}

template <typename T>
void FullOpenCVCameraModel::ImageToWorld(const T* params, const T x, const T y,
                                         T* u, T* v) {
    const T f1 = params[0];
    const T f2 = params[1];
    const T c1 = params[2];
    const T c2 = params[3];

    // Lift points to normalized plane
    *u = (x - c1) / f1;
    *v = (y - c2) / f2;

    IterativeUndistortion(&params[4], u, v);
}

template <typename T>
void FullOpenCVCameraModel::Distortion(const T* extra_params, const T u,
                                       const T v, T* du, T* dv) {
    const T k1 = extra_params[0];
    const T k2 = extra_params[1];
    const T p1 = extra_params[2];
    const T p2 = extra_params[3];
    const T k3 = extra_params[4];
    const T k4 = extra_params[5];
    const T k5 = extra_params[6];
    const T k6 = extra_params[7];

    const T u2 = u * u;
    const T uv = u * v;
    const T v2 = v * v;
    const T r2 = u2 + v2;
    const T r4 = r2 * r2;
    const T r6 = r4 * r2;
    const T radial = (T(1) + k1 * r2 + k2 * r4 + k3 * r6) /
                     (T(1) + k4 * r2 + k5 * r4 + k6 * r6);
    *du = u * radial + T(2) * p1 * uv + p2 * (r2 + T(2) * u2) - u;
    *dv = v * radial + T(2) * p2 * uv + p1 * (r2 + T(2) * v2) - v;
}

template <typename T>
void FullOpenCVCameraModel::WorldToBearing(const T* params, const T u,
                                              const T v, const T w, 
                                              T* x, T* y, T*z) {
    *x = u/w;
    *y = v/w;
    *z = w/w;
}

template <typename T>
void FullOpenCVCameraModel::ImageToBearing(const T* params, const T x,
                                        const T y, T* u, T* v, T* w) {
    
    const T f1 = params[0];
    const T f2 = params[1];
    const T c1 = params[2];
    const T c2 = params[3];

    // Lift points to normalized plane
    *u = (x - c1) / f1;
    *v = (y - c2) / f2;

    IterativeUndistortion(&params[4], u, v);
    * w = T(1.0);
}

template <typename T>
void FullOpenCVCameraModel::BearingToImage(const T* params, const T x,
                                              const T y, const T z, 
                                              T* u, T* v) {
}

////////////////////////////////////////////////////////////////////////////////
// FOVCameraModel

std::string FOVCameraModel::InitializeParamsInfo() {
    return "fx, fy, cx, cy, omega";
}

std::vector<size_t> FOVCameraModel::InitializeFocalLengthIdxs() {
    return {0, 1};
}

std::vector<size_t> FOVCameraModel::InitializePrincipalPointIdxs() {
    return {2, 3};
}

std::vector<size_t> FOVCameraModel::InitializeExtraParamsIdxs() { return {4}; }

std::vector<double> FOVCameraModel::InitializeParams(const double focal_length,
                                                     const size_t width,
                                                     const size_t height) {
    return {focal_length, focal_length, width / 2.0, height / 2.0, 1e-2};
}

template <typename T>
void FOVCameraModel::WorldToImage(const T* params, const T u, const T v, T* x,
                                  T* y) {
    const T f1 = params[0];
    const T f2 = params[1];
    const T c1 = params[2];
    const T c2 = params[3];

    // Distortion
    Distortion(&params[4], u, v, x, y);

    // Transform to image coordinates
    *x = f1 * *x + c1;
    *y = f2 * *y + c2;
}

template <typename T>
void FOVCameraModel::ImageToWorld(const T* params, const T x, const T y, T* u,
                                  T* v) {
    const T f1 = params[0];
    const T f2 = params[1];
    const T c1 = params[2];
    const T c2 = params[3];

    // Lift points to normalized plane
    const T uu = (x - c1) / f1;
    const T vv = (y - c2) / f2;

    // Undistortion
    Undistortion(&params[4], uu, vv, u, v);
}

template <typename T>
void FOVCameraModel::WorldToBearing(const T* params, const T u,
                                    const T v, const T w, 
                                    T* x, T* y, T*z) {
    *x = u/w;
    *y = v/w;
    *z = w/w;
}

template <typename T>
void FOVCameraModel::ImageToBearing(const T* params, const T x,
                                    const T y, T* u, T* v, T* w) {
    
    const T f1 = params[0];
    const T f2 = params[1];
    const T c1 = params[2];
    const T c2 = params[3];

    // Lift points to normalized plane
    const T uu = (x - c1) / f1;
    const T vv = (y - c2) / f2;

    // Undistortion
    Undistortion(&params[4], uu, vv, u, v);
    *w = T(1.0);
}

template <typename T>
void FOVCameraModel::Distortion(const T* extra_params, const T u, const T v,
                                T* du, T* dv) {
    const T omega = extra_params[0];

    // Chosen arbitrarily.
    const T kEpsilon = T(1e-4);

    const T radius2 = u * u + v * v;
    const T omega2 = omega * omega;

    T factor;
    if (omega2 < kEpsilon) {
        // Derivation of this case with Matlab:
        // syms radius omega;
        // factor(radius) = atan(radius * 2 * tan(omega / 2)) / ...
        //                  (radius * omega);
        // simplify(taylor(factor, omega, 'order', 3))
        factor = (omega2 * radius2) / T(3) - omega2 / T(12) + T(1);
    } else if (radius2 < kEpsilon) {
        // Derivation of this case with Matlab:
        // syms radius omega;
        // factor(radius) = atan(radius * 2 * tan(omega / 2)) / ...
        //                  (radius * omega);
        // simplify(taylor(factor, radius, 'order', 3))
        const T tan_half_omega = ceres::tan(omega / T(2));
        factor = (T(-2) * tan_half_omega *
                  (T(4) * radius2 * tan_half_omega * tan_half_omega - T(3))) /
                 (T(3) * omega);
    } else {
        const T radius = ceres::sqrt(radius2);
        const T numerator = ceres::atan(radius * T(2) * ceres::tan(omega / T(2)));
        factor = numerator / (radius * omega);
    }

    *du = u * factor;
    *dv = v * factor;
}

template <typename T>
void FOVCameraModel::Undistortion(const T* extra_params, const T u, const T v,
                                  T* du, T* dv) {
    T omega = extra_params[0];

    // Chosen arbitrarily.
    const T kEpsilon = T(1e-4);

    const T radius2 = u * u + v * v;
    const T omega2 = omega * omega;

    T factor;
    if (omega2 < kEpsilon) {
        // Derivation of this case with Matlab:
        // syms radius omega;
        // factor(radius) = tan(radius * omega) / ...
        //                  (radius * 2*tan(omega/2));
        // simplify(taylor(factor, omega, 'order', 3))
        factor = (omega2 * radius2) / T(3) - omega2 / T(12) + T(1);
    } else if (radius2 < kEpsilon) {
        // Derivation of this case with Matlab:
        // syms radius omega;
        // factor(radius) = tan(radius * omega) / ...
        //                  (radius * 2*tan(omega/2));
        // simplify(taylor(factor, radius, 'order', 3))
        factor = (omega * (omega * omega * radius2 + T(3))) /
                 (T(6) * ceres::tan(omega / T(2)));
    } else {
        const T radius = ceres::sqrt(radius2);
        const T numerator = ceres::tan(radius * omega);
        factor = numerator / (radius * T(2) * ceres::tan(omega / T(2)));
    }

    *du = u * factor;
    *dv = v * factor;
}

template <typename T>
void FOVCameraModel::BearingToImage(const T* params, const T x,
                                              const T y, const T z, 
                                              T* u, T* v) {
}


////////////////////////////////////////////////////////////////////////////////
// SimpleRadialFisheyeCameraModel

std::string SimpleRadialFisheyeCameraModel::InitializeParamsInfo() {
    return "f, cx, cy, k";
}

std::vector<size_t>
SimpleRadialFisheyeCameraModel::InitializeFocalLengthIdxs() {
    return {0};
}

std::vector<size_t>
SimpleRadialFisheyeCameraModel::InitializePrincipalPointIdxs() {
    return {1, 2};
}

std::vector<size_t>
SimpleRadialFisheyeCameraModel::InitializeExtraParamsIdxs() {
    return {3};
}

std::vector<double> SimpleRadialFisheyeCameraModel::InitializeParams(
    const double focal_length, const size_t width, const size_t height) {
    return {focal_length, width / 2.0, height / 2.0, 0};
}

template <typename T>
void SimpleRadialFisheyeCameraModel::WorldToImage(const T* params, const T u,
                                                  const T v, T* x, T* y) {
    const T f = params[0];
    const T c1 = params[1];
    const T c2 = params[2];

    // Distortion
    T du, dv;
    Distortion(&params[3], u, v, &du, &dv);
    *x = u + du;
    *y = v + dv;

    // Transform to image coordinates
    *x = f * *x + c1;
    *y = f * *y + c2;
}

template <typename T>
void SimpleRadialFisheyeCameraModel::ImageToWorld(const T* params, const T x,
                                                  const T y, T* u, T* v) {
    const T f = params[0];
    const T c1 = params[1];
    const T c2 = params[2];

    // Lift points to normalized plane
    *u = (x - c1) / f;
    *v = (y - c2) / f;

    IterativeUndistortion(&params[3], u, v);
}

template <typename T>
void SimpleRadialFisheyeCameraModel::WorldToBearing(const T* params, const T u,
                                                    const T v, const T w, 
                                                    T* x, T* y, T*z) {
    *x = u/w;
    *y = v/w;
    *z = w/w;
}

template <typename T>
void SimpleRadialFisheyeCameraModel::ImageToBearing(const T* params, const T x,
                                                const T y, T* u, T* v, T* w) {
    
    const T f = params[0];
    const T c1 = params[1];
    const T c2 = params[2];

    // Lift points to normalized plane
    *u = (x - c1) / f;
    *v = (y - c2) / f;

    IterativeUndistortion(&params[3], u, v);
    *w = T(1.0);
}


template <typename T>
void SimpleRadialFisheyeCameraModel::Distortion(const T* extra_params,
                                                const T u, const T v, T* du,
                                                T* dv) {
    const T k = extra_params[0];

    const T r = ceres::sqrt(u * u + v * v);

    if (r > T(std::numeric_limits<double>::epsilon())) {
        const T theta = ceres::atan(r);
        const T theta2 = theta * theta;
        const T thetad = theta * (T(1) + k * theta2);
        *du = u * thetad / r - u;
        *dv = v * thetad / r - v;
    } else {
        *du = T(0);
        *dv = T(0);
    }
}

template <typename T>
void SimpleRadialFisheyeCameraModel::BearingToImage(const T* params, const T x,
                                              const T y, const T z, 
                                              T* u, T* v) {
}

////////////////////////////////////////////////////////////////////////////////
// RadialFisheyeCameraModel

std::string RadialFisheyeCameraModel::InitializeParamsInfo() {
    return "f, cx, cy, k1, k2";
}

std::vector<size_t> RadialFisheyeCameraModel::InitializeFocalLengthIdxs() {
    return {0};
}

std::vector<size_t> RadialFisheyeCameraModel::InitializePrincipalPointIdxs() {
    return {1, 2};
}

std::vector<size_t> RadialFisheyeCameraModel::InitializeExtraParamsIdxs() {
    return {3, 4};
}

std::vector<double> RadialFisheyeCameraModel::InitializeParams(
    const double focal_length, const size_t width, const size_t height) {
    return {focal_length, width / 2.0, height / 2.0, 0, 0};
}

template <typename T>
void RadialFisheyeCameraModel::WorldToImage(const T* params, const T u,
                                            const T v, T* x, T* y) {
    const T f = params[0];
    const T c1 = params[1];
    const T c2 = params[2];

    // Distortion
    T du, dv;
    Distortion(&params[3], u, v, &du, &dv);
    *x = u + du;
    *y = v + dv;

    // Transform to image coordinates
    *x = f * *x + c1;
    *y = f * *y + c2;
}

template <typename T>
void RadialFisheyeCameraModel::ImageToWorld(const T* params, const T x,
                                            const T y, T* u, T* v) {
    const T f = params[0];
    const T c1 = params[1];
    const T c2 = params[2];

    // Lift points to normalized plane
    *u = (x - c1) / f;
    *v = (y - c2) / f;

    IterativeUndistortion(&params[3], u, v);
}

template <typename T>
void RadialFisheyeCameraModel::WorldToBearing(const T* params, const T u,
                                                    const T v, const T w, 
                                                    T* x, T* y, T*z) {
    *x = u/w;
    *y = v/w;
    *z = w/w;
}

template <typename T>
void RadialFisheyeCameraModel::ImageToBearing(const T* params, const T x,
                                                const T y, T* u, T* v, T* w) {
    
    const T f = params[0];
    const T c1 = params[1];
    const T c2 = params[2];

    // Lift points to normalized plane
    *u = (x - c1) / f;
    *v = (y - c2) / f;

    IterativeUndistortion(&params[3], u, v);
    *w = T(1.0);
}


template <typename T>
void RadialFisheyeCameraModel::Distortion(const T* extra_params, const T u,
                                          const T v, T* du, T* dv) {
    const T k1 = extra_params[0];
    const T k2 = extra_params[1];

    const T r = ceres::sqrt(u * u + v * v);

    if (r > T(std::numeric_limits<double>::epsilon())) {
        const T theta = ceres::atan(r);
        const T theta2 = theta * theta;
        const T theta4 = theta2 * theta2;
        const T thetad =
            theta * (T(1) + k1 * theta2 + k2 * theta4);
        *du = u * thetad / r - u;
        *dv = v * thetad / r - v;
    } else {
        *du = T(0);
        *dv = T(0);
    }
}

template <typename T>
void RadialFisheyeCameraModel::BearingToImage(const T* params, const T x,
                                              const T y, const T z, 
                                              T* u, T* v) {
}

////////////////////////////////////////////////////////////////////////////////
// ThinPrismFisheyeCameraModel

std::string ThinPrismFisheyeCameraModel::InitializeParamsInfo() {
    return "fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, sx1, sy1";
}

std::vector<size_t> ThinPrismFisheyeCameraModel::InitializeFocalLengthIdxs() {
    return {0, 1};
}

std::vector<size_t>
ThinPrismFisheyeCameraModel::InitializePrincipalPointIdxs() {
    return {2, 3};
}

std::vector<size_t> ThinPrismFisheyeCameraModel::InitializeExtraParamsIdxs() {
    return {4, 5, 6, 7, 8, 9, 10, 11};
}

std::vector<double> ThinPrismFisheyeCameraModel::InitializeParams(
    const double focal_length, const size_t width, const size_t height) {
    return {focal_length,
            focal_length,
            width / 2.0,
            height / 2.0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0};
}

template <typename T>
void ThinPrismFisheyeCameraModel::WorldToImage(const T* params, const T u,
                                               const T v, T* x, T* y) {
    const T f1 = params[0];
    const T f2 = params[1];
    const T c1 = params[2];
    const T c2 = params[3];

    const T r = ceres::sqrt(u * u + v * v);

    T uu, vv;
    if (r > T(std::numeric_limits<double>::epsilon())) {
        const T theta = ceres::atan(r);
        uu = theta * u / r;
        vv = theta * v / r;
    } else {
        uu = u;
        vv = v;
    }

    // Distortion
    T du, dv;
    Distortion(&params[4], uu, vv, &du, &dv);
    *x = uu + du;
    *y = vv + dv;

    // Transform to image coordinates
    *x = f1 * *x + c1;
    *y = f2 * *y + c2;
}

template <typename T>
void ThinPrismFisheyeCameraModel::ImageToWorld(const T* params, const T x,
                                               const T y, T* u, T* v) {
    const T f1 = params[0];
    const T f2 = params[1];
    const T c1 = params[2];
    const T c2 = params[3];

    // Lift points to normalized plane
    *u = (x - c1) / f1;
    *v = (y - c2) / f2;

    IterativeUndistortion(&params[4], u, v);

    const T theta = ceres::sqrt(*u * *u + *v * *v);
    const T theta_cos_theta = theta * ceres::cos(theta);
    if (theta_cos_theta > T(std::numeric_limits<double>::epsilon())) {
        const T scale = ceres::sin(theta) / theta_cos_theta;
        *u *= scale;
        *v *= scale;
    }
}

template <typename T>
void ThinPrismFisheyeCameraModel::WorldToBearing(const T* params, const T u,
                                                    const T v, const T w, 
                                                    T* x, T* y, T*z) {
    *x = u/w;
    *y = v/w;
    *z = w/w;
}

template <typename T>
void ThinPrismFisheyeCameraModel::ImageToBearing(const T* params, const T x,
                                                const T y, T* u, T* v, T* w) {
    
    const T f1 = params[0];
    const T f2 = params[1];
    const T c1 = params[2];
    const T c2 = params[3];

    // Lift points to normalized plane
    *u = (x - c1) / f1;
    *v = (y - c2) / f2;

    IterativeUndistortion(&params[4], u, v);

    const T theta = ceres::sqrt(*u * *u + *v * *v);
    const T theta_cos_theta = theta * ceres::cos(theta);
    if (theta_cos_theta > T(std::numeric_limits<double>::epsilon())) {
        const T scale = ceres::sin(theta) / theta_cos_theta;
        *u *= scale;
        *v *= scale;
    }
    *w = T(1.0);
}


template <typename T>
void ThinPrismFisheyeCameraModel::Distortion(const T* extra_params, const T u,
                                             const T v, T* du, T* dv) {
    const T k1 = extra_params[0];
    const T k2 = extra_params[1];
    const T p1 = extra_params[2];
    const T p2 = extra_params[3];
    const T k3 = extra_params[4];
    const T k4 = extra_params[5];
    const T sx1 = extra_params[6];
    const T sy1 = extra_params[7];

    const T u2 = u * u;
    const T uv = u * v;
    const T v2 = v * v;
    const T r2 = u2 + v2;
    const T r4 = r2 * r2;
    const T r6 = r4 * r2;
    const T r8 = r6 * r2;
    const T radial = k1 * r2 + k2 * r4 + k3 * r6 + k4 * r8;
    *du = u * radial + T(2) * p1 * uv + p2 * (r2 + T(2) * u2) + sx1 * r2;
    *dv = v * radial + T(2) * p2 * uv + p1 * (r2 + T(2) * v2) + sy1 * r2;
}

template <typename T>
void ThinPrismFisheyeCameraModel::BearingToImage(const T* params, const T x,
                                              const T y, const T z, 
                                              T* u, T* v) {
}

////////////////////////////////////////////////////////////////////////////////
//SphericalCameraModel

std::string SphericalCameraModel::InitializeParamsInfo() {
    return "f, cx, cy, ox1, oy1, oz1, ox2, oy2, oz2";
}

std::vector<size_t> SphericalCameraModel::InitializeFocalLengthIdxs() {
    return {0};
}

std::vector<size_t> SphericalCameraModel::InitializePrincipalPointIdxs() {
    return {1, 2};
}

std::vector<size_t> SphericalCameraModel::InitializeExtraParamsIdxs() {
    return {3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
}

std::vector<double> SphericalCameraModel::InitializeParams(
    const double focal_length, const size_t width, const size_t height) {
    
    return {focal_length, static_cast<double>(width), 
            static_cast<double>(height), 
            1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0};
}

template <typename T>
void SphericalCameraModel::WorldToImage(const T* params, const T u,
                                        const T v, T* x, T* y) {
    
    CHECK(0)<<"WorldToImage for Spherical Camera undefined";
    
    const T f = params[0];
    const T width = params[1];
    const T height = params[2];

    // Transform to image coordinates
    
    T bearing_x = u;
    T bearing_y = v;
    T bearing_z = T(1.0);

    T norm = ceres::sqrt(u*u+v*v+T(1));

    bearing_x = bearing_x /norm;
    bearing_y = bearing_y /norm;
    bearing_z = bearing_z /norm;

    T lat = -ceres::asin(bearing_y);
    T lon = ceres::atan2(bearing_x, bearing_z);

    if(lon >= M_PI) {
        lon = lon-2*M_PI;
    }

    *x = width * (0.5 + lon / (2 * M_PI));
    *y = height * (0.5 - lat / M_PI);  
}

template <typename T>
void SphericalCameraModel::ImageToWorld(const T* params, const T x,
                                        const T y, T* u, T* v) {

    const T f = params[0];
    const T width = params[1];
    const T height = params[2];

    Eigen::Vector3d bearing;

    const double lon = (x / width - 0.5) * (2 * M_PI);
    const double lat = -(y / height - 0.5) * M_PI;
    // convert to equirectangular coordinates
    bearing(0) = std::cos(lat) * std::sin(lon);
    bearing(1) = -std::sin(lat);
    bearing(2) = std::cos(lat) * std::cos(lon);  

    if(bearing(2)!=0){
        *u = bearing(0)/bearing(2);
        *v = bearing(1)/bearing(2);
    }
    else{
        *u = bearing(0);
        *v = bearing(1);
    }
}

template <typename T>                                                       
void SphericalCameraModel::WorldToBearing(const T* params, const T u, const T v,
                                          const T w, T* x, T* y, T* z){

    const T f = params[0];
    const T width = params[1];
    const T height = params[2];


    T bearing_x = u;
    T bearing_y = v;
    T bearing_z = w;

    T norm = ceres::sqrt(u*u+v*v+w*w);

    if(norm > 0){
        *x = bearing_x /norm;
        *y = bearing_y /norm;
        *z = bearing_z /norm;
    }
    else{
        *x = 0;
        *y = 0;
        *z = 0;
    }
} 

template <typename T>                                                       
void SphericalCameraModel::ImageToBearing(const T* params, const T u, const T v,            
                                          T* x, T* y, T* z){

    const T f = params[0];
    const T width = params[1];
    const T height = params[2];

    const T lon = (u / width - 0.5) * (2 * M_PI);
    const T lat = -(v / height - 0.5) * M_PI;
 
    // convert to equirectangular coordinates
    *x = ceres::cos(lat) * ceres::sin(lon);
    *y = -ceres::sin(lat);
    *z = ceres::cos(lat) * ceres::cos(lon);                                          
}

template <typename T>
void SphericalCameraModel::BearingToImage(const T* params, const T x,
                                              const T y, const T z, 
                                              T* u, T* v) {
}

////////////////////////////////////////////////////////////////////////////////
// OcamOmnidirectional camera model

std::string OcamOmnidirectionalCameraModel::InitializeParamsInfo() {
    return "f, cx, cy, invp1, invp2, invp3, invp4, invp5, invp6, invp7, \
            invp8, invp9,invp10, invp11, invp12, p1, p2, p3, p4, p5, \
            c, d, e";
}

std::vector<size_t> OcamOmnidirectionalCameraModel::InitializeFocalLengthIdxs() {
    return {0};
}

std::vector<size_t> OcamOmnidirectionalCameraModel::InitializePrincipalPointIdxs() {
    return {1, 2};
}

std::vector<size_t> OcamOmnidirectionalCameraModel::InitializeExtraParamsIdxs() {
    return {3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22};
}

std::vector<double> OcamOmnidirectionalCameraModel::InitializeParams(
    const double focal_length, const size_t width, const size_t height) {
    
    return {focal_length, static_cast<double>(width)/2, static_cast<double>(height)/2,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            0};
}

template <typename T>
void OcamOmnidirectionalCameraModel::WorldToImage(const T* params, const T u,
                                                  const T v, T* x, T* y){
    T pol[12];
    for(int i = 0; i<12; ++i){
        pol[i] = params[3+i];            
    }

    T cx = params[1];
    T cy = params[2];

    T c = params[20];
    T d = params[21];
    T e = params[22];

    T w = T(-1.0);
    
    T norm = ceres::sqrt(u*u+v*v);
    if(norm == T(0)){
        norm = T(std::numeric_limits<double>::epsilon());
    } 

    T theta = ceres::atan(w/norm);

    T poly_val = T(0);
    for(int i = 0; i< 12; ++i){
        poly_val += pol[i]*ceres::pow(theta,T(i));
    }

    T m = u/norm*poly_val;
    T n = v/norm*poly_val;
    *x = m*c + n*d + cx;
    *y = m*e + n   + cy;
}

template <typename T>
void OcamOmnidirectionalCameraModel::ImageToWorld(const T* params, const T x,
                                        const T y, T* u, T* v) {
    
    T pol[5];
    for(int i = 0; i<5; ++i){
        pol[i] = params[15+i];
    }

    T cx = params[1];
    T cy = params[2];

    T c = params[20];
    T d = params[21];
    T e = params[22];

    Eigen::Matrix2d A;
    A << c,d,e,1;
    Eigen::Vector2d m;
    m << x-cx,y-cy;

    Eigen::Vector2d m_lens = A.inverse()*m;

    double r = m_lens.norm();

    double poly_val = 0;
    for(int i = 0; i< 5; ++i){
        poly_val += pol[i]*pow(r,i);
    }
    
    *u = -m_lens(0)/poly_val;
    *v = -m_lens(1)/poly_val;
}

template <typename T>                                                       
void OcamOmnidirectionalCameraModel::WorldToBearing(const T* params, const T u, 
                                                    const T v,const T w, 
                                                    T* x, T* y, T* z){

    T bearing_x = u;
    T bearing_y = v;
    T bearing_z = w;

    T norm = ceres::sqrt(u*u+v*v+w*w);

    if(norm > 0){
        *x = bearing_x /norm;
        *y = bearing_y /norm;
        *z = bearing_z /norm;
    }
    else{
        *x = 0;
        *y = 0;
        *z = 0;
    }

} 

template <typename T>                                                       
void OcamOmnidirectionalCameraModel::ImageToBearing(const T* params, const T u, const T v,            
                                          T* x, T* y, T* z){

    T pol[5];
    for(int i = 0; i<5; ++i){
        pol[i] = params[15+i];
    }

    T cx = params[1];
    T cy = params[2];

    T c = params[20];
    T d = params[21];
    T e = params[22];

    Eigen::Matrix2d A;
    A << c,d,e,1;
    Eigen::Vector2d m;
    m << u-cx,v-cy;

    Eigen::Vector2d m_lens = A.inverse()*m;

    double r = m_lens.norm();

    T poly_val = 0;
    for(int i = 0; i< 5; ++i){
        poly_val += pol[i]*T(pow(r,i));
    }
    
    Eigen::Vector3d ray;
    ray<<-m_lens(0),-m_lens(1),poly_val;
    ray.normalize();
    if(ray(2)<0){
        ray = -ray;
    }

    *x = T(ray(0));
    *y = T(ray(1));
    *z = T(ray(2));
}

template <typename T>
void OcamOmnidirectionalCameraModel::BearingToImage(const T* params, const T x,
                                              const T y, const T z, 
                                              T* u, T* v) {
}


////////////////////////////////////////////////////////////////////////////////
// UnifiedCameraModel

std::string UnifiedCameraModel::InitializeParamsInfo() {
    return "xi,fx, fy, cx, cy, k1, k2, p1, p2";
}

std::vector<size_t> UnifiedCameraModel::InitializeFocalLengthIdxs() {
    return {1, 2};
}

std::vector<size_t> UnifiedCameraModel::InitializePrincipalPointIdxs() {
    return {3, 4};
}

std::vector<size_t> UnifiedCameraModel::InitializeExtraParamsIdxs() {
    return {0, 5, 6, 7, 8};
}

std::vector<double> UnifiedCameraModel::InitializeParams(
    const double focal_length, const size_t width, const size_t height) {
    return {0,focal_length, focal_length, width / 2.0, height / 2.0, 0, 0, 0, 0};
}

template <typename T>
void UnifiedCameraModel::WorldToImage(const T* params, const T u, const T v,
                                     T* x, T* y) {
    
    const T xi = params[0];
    const T f1 = params[1];
    const T f2 = params[2];
    const T c1 = params[3];
    const T c2 = params[4];
    
    // projection onto image plane
    const T d= ceres::sqrt(u*u+v*v+T(1.0));
    const T mx = u/((xi*d)+T(1.0));
    const T my = v/((xi*d)+T(1.0));

    // Distortion
    T du, dv;
    T u_d, v_d;
    Distortion(&params[5], mx, my, &du, &dv);
    u_d = mx + du;
    v_d = my + dv;

    // Transform to image coordinates
    *x = f1 * u_d + c1;
    *y = f2 * v_d + c2;
}

template <typename T>
void UnifiedCameraModel::ImageToWorld(const T* params, const T x, const T y,
                                     T* u, T* v) {
    const T xi = params[0];
    const T f1 = params[1];
    const T f2 = params[2];
    const T c1 = params[3];
    const T c2 = params[4];

    T mx,my;

    // Lift points to normalized plane
    mx = (x - c1) / f1;
    my = (y - c2) / f2;

    IterativeUndistortion(&params[5], &mx, &my);

    const T r_square = mx*mx + my*my;

    T sx,sy,sz;

    sx = mx;
    sy = my;
    sz = 1- xi*(T(1.0)+r_square)/(xi+ceres::sqrt(T(1.0)+(T(1.0)-xi*xi)*r_square));

    *u = sx/sz;
    *v = sy/sz;
}

template <typename T>
void UnifiedCameraModel::Distortion(const T* extra_params, const T u, const T v,
                                   T* du, T* dv) {
    const T k1 = extra_params[0];
    const T k2 = extra_params[1];
    const T p1 = extra_params[2];
    const T p2 = extra_params[3];

    const T u2 = u * u;
    const T uv = u * v;
    const T v2 = v * v;
    const T r2 = u2 + v2;
    const T radial = k1 * r2 + k2 * r2 * r2;
    *du = u * radial + T(2) * p1 * uv + p2 * (r2 + T(2) * u2);
    *dv = v * radial + T(2) * p2 * uv + p1 * (r2 + T(2) * v2);
}

template <typename T>
void UnifiedCameraModel::WorldToBearing(const T* params, const T u,
                                              const T v, const T w, 
                                              T* x, T* y, T*z) {
    T bearing_x = u;
    T bearing_y = v;
    T bearing_z = w;

    T norm = ceres::sqrt(u*u+v*v+w*w);

    if(norm > 0){
        *x = bearing_x /norm;
        *y = bearing_y /norm;
        *z = bearing_z /norm;
    }
    else{
        *x = 0;
        *y = 0;
        *z = 0;
    }
}

template <typename T>
void UnifiedCameraModel::ImageToBearing(const T* params, const T x,
                                        const T y, T* u, T* v, T* w) {
   
    const T xi = params[0];
    const T f1 = params[1];
    const T f2 = params[2];
    const T c1 = params[3];
    const T c2 = params[4];

    T mx,my;

    // Lift points to normalized plane
    mx = (x - c1) / f1;
    my = (y - c2) / f2;

    IterativeUndistortion(&params[5], &mx, &my);

    const T r_square = mx*mx + my*my;

    T sx,sy,sz;

    sx = mx;
    sy = my;
    sz = T(1.0)- xi*(T(1.0)+r_square)/(xi+ceres::sqrt(T(1.0)+(T(1.0)-xi*xi)*r_square));

    T norm = ceres::sqrt(sx*sx+sy*sy+sz*sz);

    *u = sx / norm;
    *v = sy / norm;
    *w = sz / norm;
}


template <typename T>
void UnifiedCameraModel::BearingToImage(const T* params, const T u,
                                              const T v, const T w, 
                                              T* x, T* y) {
    
    const T xi = params[0];
    const T f1 = params[1];
    const T f2 = params[2];
    const T c1 = params[3];
    const T c2 = params[4];
    
    // projection onto image plane
    const T d= ceres::sqrt(u*u+v*v+w*w);
    const T mx = u/((xi*d)+w);
    const T my = v/((xi*d)+w);

    // Distortion
    T du, dv;
    T u_d, v_d;
    Distortion(&params[5], mx, my, &du, &dv);
    u_d = mx + du;
    v_d = my + dv;

    // Transform to image coordinates
    *x = f1 * u_d + c1;
    *y = f2 * v_d + c2;
}


////////////////////////////////////////////////////////////////////////////////

void CameraModelWorldToImage(const int model_id,
                             const std::vector<double>& params, const double u,
                             const double v, double* x, double* y) {
    switch (model_id) {
#define CAMERA_MODEL_CASE(CameraModel)                    \
case CameraModel::kModelId:                             \
CameraModel::WorldToImage(params.data(), u, v, x, y); \
break;

        CAMERA_MODEL_SWITCH_CASES

#undef CAMERA_MODEL_CASE
    }
}

void CameraModelImageToWorld(const int model_id,
                             const std::vector<double>& params, const double x,
                             const double y, double* u, double* v) {
    switch (model_id) {
#define CAMERA_MODEL_CASE(CameraModel)                    \
case CameraModel::kModelId:                             \
CameraModel::ImageToWorld(params.data(), x, y, u, v); \
break;

        CAMERA_MODEL_SWITCH_CASES

#undef CAMERA_MODEL_CASE
    }
}

double CameraModelImageToWorldThreshold(const int model_id,
                                        const std::vector<double>& params,
                                        const double threshold) {
    switch (model_id) {
#define CAMERA_MODEL_CASE(CameraModel)                                   \
case CameraModel::kModelId:                                            \
return CameraModel::ImageToWorldThreshold(params.data(), threshold); \
break;

        CAMERA_MODEL_SWITCH_CASES

#undef CAMERA_MODEL_CASE
    }

    return -1;
}

void CameraModelImageToBearing(const int model_id,
                             const std::vector<double>& params, const double u,
                             const double v, double* x, double* y, double* z) {
    switch (model_id) {
#define CAMERA_MODEL_CASE(CameraModel)                    \
case CameraModel::kModelId:                             \
CameraModel::ImageToBearing(params.data(), u, v, x, y,z); \
break;

        CAMERA_MODEL_SWITCH_CASES

#undef CAMERA_MODEL_CASE
    }
    
}

void CameraModelWorldToBearing(const int model_id,
                             const std::vector<double>& params, const double u,
                             const double v, const double w, 
                             double* x, double* y, double* z) { 
    switch (model_id) {
#define CAMERA_MODEL_CASE(CameraModel)                    \
case CameraModel::kModelId:                             \
CameraModel::WorldToBearing(params.data(), u, v, w, x, y,z); \
break;

        CAMERA_MODEL_SWITCH_CASES

#undef CAMERA_MODEL_CASE
    } 
}


void CameraModelBearingToImage(const int model_id,
                             const std::vector<double>& params, const double u,
                             const double v, const double w, 
                             double* x, double* y) { 
    switch (model_id) {
#define CAMERA_MODEL_CASE(CameraModel)                    \
case CameraModel::kModelId:                             \
CameraModel::BearingToImage(params.data(), u, v, w, x, y); \
break;

        CAMERA_MODEL_SWITCH_CASES

#undef CAMERA_MODEL_CASE
    } 
}


void ComputeDistortionValueAndJacobian(const double* params,
                                       const double* coord,
                                       double* value,
                                       double** jacobian);





} // namespace sensemap

#endif //SENSEMAP_BASE_CAMERA_MODELS_H
