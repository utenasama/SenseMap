//Copyright (c) 2019, SenseTime Group.
//All rights reserved.

#ifndef SENSEMAP_BASE_IMAGE_H_
#define SENSEMAP_BASE_IMAGE_H_

#include <string>
#include <vector>
#include <memory>

#include <Eigen/Core>

#include "base/camera.h"
#include "base/point2d.h"
#include "util/mat.h"
#include "util/logging.h"
#include "util/math.h"
#include "util/types.h"
#include "feature/types.h"
#include "camera.h"
#include "point2d.h"
#include "visibility_pyramid.h"
#include "RGBDAlign/ICPLink.h"

namespace sensemap {

// Class that holds information about an image. An image is the product of one
// camera shot at a certain location (parameterized as the pose). An image may
// share a camera with multiple other images, if its intrinsics are the same.
class Image {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Image();
  
    // Setup / tear down the image and necessary internal data structures before
    // and after being used in reconstruction.
    void SetUp(const Camera& camera);
    void TearDown();

    // Access the unique identifier of the image.
    inline image_t ImageId() const;
    inline void SetImageId(const image_t image_id);

    // Access the name of the image.
    inline const std::string& Name() const;
    inline std::string& Name();
    inline void SetName(const std::string& name);

    inline std::vector<std::string> LocalNames() const;
    inline std::vector<std::string>& LocalNames();
    inline void SetLocalNames(const std::vector<std::string>& local_names);
    inline bool HasLocalName(const size_t local_camera_id) const;
    inline const std::string& LocalName(const size_t local_camera_id) const;
    inline std::string& LocalName(const size_t local_camera_id);

    // Access the label of the image.
    inline label_t LabelId() const;
    inline void SetLabelId(const label_t label_id);
    // Check whether the image has lable or not
    inline bool HasLabel() const;

    // Access the unique identifier of the camera. Note that multiple images
    // might share the same camera.
    inline camera_t CameraId() const;
    inline void SetCameraId(const camera_t camera_id);
    // Check whether identifier of camera has been set.
    inline bool HasCamera() const;

    // Check if image is registered.
    inline bool IsRegistered() const;
    inline void SetRegistered(const bool registered);

    inline bool HasPose() const;
    inline void SetPoseFlag(const bool has_pose);

    inline bool IsKeyFrame() const;
    inline void SetKeyFrame(const bool keyframe);

    // Get the number of image points.
    inline point2D_t NumPoints2D() const;

    // Get the number of triangulations, i.e. the number of points that
    // are part of a Map Point track.
    inline point2D_t NumMapPoints() const;

    // Get the number of observations, i.e. the number of image points that
    // have at least one correspondence to another image.
    inline point2D_t NumObservations() const;
    inline void SetNumObservations(const point2D_t num_observations);

    // Get the number of correspondences for all image points.
    inline point2D_t NumCorrespondences() const;
    inline void SetNumCorrespondences(const point2D_t num_observations);

    // Get the number of observations that see a triangulated point, i.e. the
    // number of image points that have at least one correspondence to a
    // triangulated point in another image.
    inline point2D_t NumVisibleMapPoints() const;

    // Get the score of triangulated observations. In contrast to
    // `NumVisibleMapPoints`, this score also captures the distribution
    // of triangulated observations in the image. This is useful to select
    // the next best image in incremental reconstruction, because a more
    // uniform distribution of observations results in more robust registration.
    inline size_t MapPointVisibilityScore() const;

    // Access quaternion vector as (qw, qx, qy, qz) specifying the rotation of the
    // pose which is defined as the transformation from world to image space.
    inline const Eigen::Vector4d& Qvec() const;
    inline Eigen::Vector4d& Qvec();
    inline double Qvec(const size_t idx) const;
    inline double& Qvec(const size_t idx);
    inline void SetQvec(const Eigen::Vector4d& qvec);

    // Quaternion prior, e.g. given by EXIF gyroscope tag.
    inline const Eigen::Vector4d& QvecPrior() const;
    inline Eigen::Vector4d& QvecPrior();
    inline double QvecPrior(const size_t idx) const;
    inline double& QvecPrior(const size_t idx);
    inline bool HasQvecPrior() const;
    inline void SetQvecPrior(const Eigen::Vector4d& qvec);
    inline std::vector<Eigen::Vector4d> LocalQvecsPrior() const;
    inline std::vector<Eigen::Vector4d>& LocalQvecsPrior();
    inline void SetLocalQvecsPrior(const std::vector<Eigen::Vector4d> & local_qvecs_prior);

    inline bool PriorInlier() const;
    inline void SetPriorInlier(const bool prior_inlier);
    inline bool PriorQvecGood() const;
    inline void SetPriorQvecGood(const bool prior_inlier);

    // Access quaternion vector as (tx, ty, tz) specifying the translation of the
    // pose which is defined as the transformation from world to image space.
    inline const Eigen::Vector3d& Tvec() const;
    inline Eigen::Vector3d& Tvec();
    inline double Tvec(const size_t idx) const;
    inline double& Tvec(const size_t idx);
    inline void SetTvec(const Eigen::Vector3d& tvec);

    // Quaternion prior, e.g. given by EXIF GPS tag.
    inline const Eigen::Vector3d& TvecPrior() const;
    inline Eigen::Vector3d& TvecPrior();
    inline double TvecPrior(const size_t idx) const;
    inline double& TvecPrior(const size_t idx);
    inline bool HasTvecPrior() const;
    inline void SetTvecPrior(const Eigen::Vector3d& tvec);
    inline std::vector<Eigen::Vector3d> LocalTvecsPrior() const;
    inline std::vector<Eigen::Vector3d>& LocalTvecsPrior();
    inline void SetLocalTvecsPrior(const std::vector<Eigen::Vector3d> & local_tvecs_prior);
    
    // 
    inline const int8_t& RtkFlag() const;
    inline int8_t& RtkFlag(); 

    inline void SetRtkStd(const double std_lon, 
                       const double std_lat, 
                       const double std_hgt);
    inline double RtkStdLon() const;
    inline double RtkStdLat() const;
    inline double RtkStdHgt() const;
    inline bool HasRtkStd() const;

    inline void SetRelativeAltitude(const double relative_altitude);
    inline bool HasRelativeAltitude();
    inline double RelativeAltitude();

    inline void SetOrientStd(const double std_orient);
    inline double OrientStd() const;
    inline bool HasOrientStd() const;
    
    // Avoid duplicated computation of depth info
    inline bool DepthFlag() const;
    inline void SetDepthFlag(const bool depth_flag);

    // Function for get sim3 pose
    inline Eigen::Vector7d& Sim3pose();

    // Access the coordinates of image points.
    inline const class Point2D& Point2D(const point2D_t point2D_idx) const;
    inline class Point2D& Point2D(const point2D_t point2D_idx);
    inline const std::vector<class Point2D>& Points2D() const;
    void SetPoints2D(const std::vector<Eigen::Vector2d>& points);
    void SetPoints2D(const std::vector<class Point2D>& points);
    void SetPoints2D(const FeatureKeypoints& keypoints);

    // Add Points2D to current image
    std::vector<point2D_t> AddPoints2D(const std::vector<class Point2D>& points);
    std::vector<point2D_t> AddPoints2D(const std::vector<class Point2D>& points, const std::vector<uint32_t>& indices);

    // Set the point as triangulated, i.e. it is part of a Map Point track.
    void SetMapPointForPoint2D(const point2D_t point2D_idx,
                                const mappoint_t mappoint_id);

    // Set the point as not triangulated, i.e. it is not part of a Map Point track.
    void ResetMapPointForPoint2D(const point2D_t point2D_idx);

    // Check whether an image point has a correspondence to an image point in
    // another image that has a Map Point.
    inline bool IsMapPointVisible(const point2D_t point2D_idx) const;

    // Check whether one of the image points is part of the Map Point track.
    bool HasMapPoint(const mappoint_t mappoint_id) const;

    // Indicate that another image has a point that is triangulated and has
    // a correspondence to this image point. Note that this must only be called
    // after calling `SetUp`.
    void IncrementCorrespondenceHasMapPoint(const point2D_t point2D_idx);

    // Indicate that another image has a point that is not triangulated any more
    // and has a correspondence to this image point. This assumes that
    // `IncrementCorrespondenceHasMapPoint` was called for the same image point
    // and correspondence before. Note that this must only be called
    // after calling `SetUp`.
    void DecrementCorrespondenceHasMapPoint(const point2D_t point2D_idx);

    // Normalize the quaternion vector.
    void NormalizeQvec();
    void NormalizePriorQvec();

    // Compose the projection matrix from world to image space.
    Eigen::Matrix3x4d ProjectionMatrix() const;

    // Compose the inverse projection matrix from image to world space
    Eigen::Matrix3x4d InverseProjectionMatrix() const;

    // Compose rotation matrix from quaternion vector.
    Eigen::Matrix3d RotationMatrix() const;

    // Extract the projection center in world space.
    Eigen::Vector3d ProjectionCenter() const;

    // Extract the viewing direction of the image.
    Eigen::Vector3d ViewingDirection() const;

    // The number of levels in the Map Point multi-resolution visibility pyramid.
    static const int kNumMapPointVisibilityPyramidLevels;


    ///////////////////////////////////////////////////////////////////////////
	//  Extra functions in multi-camera system
    ///////////////////////////////////////////////////////////////////////////

    void SetLocalImageIndices( const std::vector<uint32_t>& indices);
    inline const std::vector<uint32_t>& LocalImageIndices() const;

    std::vector<ICPLink> icp_links_;

    //gravity
    Eigen::Vector3d gravity_ = Eigen::Vector3d(0,0,0);

    // timestamp
    long long timestamp_ = 0;

    uint64_t create_time_ = 0;


private:
    // Identifier of the image, if not specified `kInvalidImageId`.
    image_t image_id_;

    // The name of the image, i.e. the relative path.
    std::string name_;

    // Identifier of the image lable, if not specified `kInvalidLableId`
    label_t label_id_;

    // The identifier of the associated camera. Note that multiple images might
    // share the same camera. If not specified `kInvalidCameraId`.
    camera_t camera_id_;

    // Whether the image is successfully registered in the reconstruction.
    bool registered_;

    // Whether the camera pose is estimated. Avoid duplicated computation 
    // in AddKeyFrame. 
    bool has_pose_;

    // Whether to be keyframe.
    bool keyframe_;

    // The number of Map Points the image observes, i.e. the sum of its `points2D`
    // where `mappoint_id != kInvalidMapPointId`.
    point2D_t num_MapPoints_;

    // The number of image points that have at least one correspondence to
    // another image.
    point2D_t num_observations_;

    // The sum of correspondences per image point.
    point2D_t num_correspondences_;

    // The number of 2D points, which have at least one corresponding 2D point in
    // another image that is part of a Map Point track, i.e. the sum of `points2D`
    // where `num_tris > 0`.
    point2D_t num_visible_MapPoints_;

    // The pose of the image, defined as the transformation from world to image.
    Eigen::Vector4d qvec_;
    Eigen::Vector3d tvec_;

    // The pose prior of the image, e.g. extracted from EXIF tags.
    Eigen::Vector4d qvec_prior_;
    Eigen::Vector3d tvec_prior_;
    bool prior_inlier_;

    bool prior_qvec_good_ = false;

    // The rtk status and of the image
    int8_t rtk_flag_ = -1;
    bool has_rtk_std_ = false;
    double std_longitude_;
    double std_latitude_;
    double std_Altitude_;
    double relative_altitude_ = std::numeric_limits<double>::max();

    ////////////////////////////////////////////////////////////////////////////////
	//  Extra priors in multi-camera system
    ////////////////////////////////////////////////////////////////////////////////
    std::vector<Eigen::Vector4d> local_qvecs_prior_;
    std::vector<Eigen::Vector3d> local_tvecs_prior_;

    bool has_orient_std_ = false;
    double std_orientation_ = std::numeric_limits<double>::max();

    // The depth computation flag
    bool depth_flag_ = false;

    // All image points, including points that are not part of a Map Point track.
    std::vector<class Point2D> points2D_;

    // Per image point, the number of correspondences that have a Map Point.
    std::vector<image_t> num_correspondences_have_mappoint_;

    // Data structure to compute the distribution of triangulated correspondences
    // in the image. Note that this structure is only usable after `SetUp`.
    VisibilityPyramid mappoint_visibility_pyramid_;

    // The SIM3 pose only used for pose graph optimization
    Eigen::Vector7d sim3_pose_;

    ///////////////////////////////////////////////////////////////////////////
	//  Extra members in multi-camera system
    ///////////////////////////////////////////////////////////////////////////
    
    // local camera index for each 2D point
    std::vector<uint32_t> local_image_indices_;

    std::vector<std::string> local_image_names_;
};


////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////

image_t Image::ImageId() const { return image_id_; }

void Image::SetImageId(const image_t image_id) { image_id_ = image_id; }

const std::string& Image::Name() const { return name_; }

std::string& Image::Name() { return name_; }

void Image::SetName(const std::string& name) { name_ = name; }

std::vector<std::string> Image::LocalNames() const { return local_image_names_; }

std::vector<std::string>& Image::LocalNames() { return local_image_names_; }

void Image::SetLocalNames(const std::vector<std::string>& local_names) {
    local_image_names_ = local_names;
}

inline bool Image::HasLocalName(const size_t local_camera_id) const {
    if (local_camera_id >= local_image_names_.size()) return false;
    return true;
}

const std::string& Image::LocalName(const size_t local_camera_id) const {
    return local_image_names_.at(local_camera_id);
}

std::string& Image::LocalName(const size_t local_camera_id) {
    return local_image_names_.at(local_camera_id);
}

inline camera_t Image::CameraId() const { return camera_id_; }

label_t Image::LabelId() const { return label_id_; }

void Image::SetLabelId(const label_t label_id) { label_id_ = label_id; }

inline bool Image::HasLabel() const { return label_id_ != kInvalidLabelId; }

inline void Image::SetCameraId(const camera_t camera_id) {
    CHECK_NE(camera_id, kInvalidCameraId);
    camera_id_ = camera_id;
}

inline bool Image::HasCamera() const { return camera_id_ != kInvalidCameraId; }

bool Image::IsRegistered() const { return registered_; }

void Image::SetRegistered(const bool registered) { registered_ = registered; }

bool Image::HasPose() const { return has_pose_; }

void Image::SetPoseFlag(const bool has_pose) { has_pose_ = has_pose; }

bool Image::IsKeyFrame() const { return keyframe_; }

void Image::SetKeyFrame(const bool keyframe) { keyframe_ = keyframe; }

point2D_t Image::NumPoints2D() const {
    return static_cast<point2D_t>(points2D_.size());
}

point2D_t Image::NumMapPoints() const { return num_MapPoints_; }

point2D_t Image::NumObservations() const { return num_observations_; }

void Image::SetNumObservations(const point2D_t num_observations) {
    num_observations_ = num_observations;
}

point2D_t Image::NumCorrespondences() const { return num_correspondences_; }

void Image::SetNumCorrespondences(const point2D_t num_correspondences) {
    num_correspondences_ = num_correspondences;
}

point2D_t Image::NumVisibleMapPoints() const { return num_visible_MapPoints_; }

size_t Image::MapPointVisibilityScore() const { return mappoint_visibility_pyramid_.Score(); }

const Eigen::Vector4d& Image::Qvec() const { return qvec_; }

Eigen::Vector4d& Image::Qvec() { return qvec_; }

inline double Image::Qvec(const size_t idx) const { return qvec_(idx); }

inline double& Image::Qvec(const size_t idx) { return qvec_(idx); }

void Image::SetQvec(const Eigen::Vector4d& qvec) { qvec_ = qvec; }

const Eigen::Vector4d& Image::QvecPrior() const { return qvec_prior_; }

Eigen::Vector4d& Image::QvecPrior() { return qvec_prior_; }

inline double Image::QvecPrior(const size_t idx) const {
    return qvec_prior_(idx);
}

inline double& Image::QvecPrior(const size_t idx) { return qvec_prior_(idx); }

inline bool Image::HasQvecPrior() const { return !IsNaN(qvec_prior_.sum()); }

void Image::SetQvecPrior(const Eigen::Vector4d& qvec) { qvec_prior_ = qvec; }

inline std::vector<Eigen::Vector4d> Image::LocalQvecsPrior() const { return local_qvecs_prior_; }

inline std::vector<Eigen::Vector4d>& Image::LocalQvecsPrior() { return local_qvecs_prior_; }

void Image::SetLocalQvecsPrior(const std::vector<Eigen::Vector4d> & local_qvecs_prior) {
    local_qvecs_prior_ = local_qvecs_prior;
}

inline bool Image::PriorInlier() const { return prior_inlier_; }

inline void Image::SetPriorInlier(const bool prior_inlier) { prior_inlier_ = prior_inlier; }

inline bool Image::PriorQvecGood() const { return prior_qvec_good_; }

inline void Image::SetPriorQvecGood(const bool prior_qvec_good) { prior_qvec_good_ = prior_qvec_good; }

const Eigen::Vector3d& Image::Tvec() const { return tvec_; }

Eigen::Vector3d& Image::Tvec() { return tvec_; }

inline double Image::Tvec(const size_t idx) const { return tvec_(idx); }

inline double& Image::Tvec(const size_t idx) { return tvec_(idx); }

void Image::SetTvec(const Eigen::Vector3d& tvec) { tvec_ = tvec; }

const Eigen::Vector3d& Image::TvecPrior() const { return tvec_prior_; }

Eigen::Vector3d& Image::TvecPrior() { return tvec_prior_; }

inline double Image::TvecPrior(const size_t idx) const {
    return tvec_prior_(idx);
}

inline double& Image::TvecPrior(const size_t idx) { return tvec_prior_(idx); }

inline bool Image::HasTvecPrior() const { return !IsNaN(tvec_prior_.sum()); }

void Image::SetTvecPrior(const Eigen::Vector3d& tvec) { tvec_prior_ = tvec; }

inline std::vector<Eigen::Vector3d> Image::LocalTvecsPrior() const { return local_tvecs_prior_; }

inline std::vector<Eigen::Vector3d>& Image::LocalTvecsPrior() { return local_tvecs_prior_; }

void Image::SetLocalTvecsPrior(const std::vector<Eigen::Vector3d> & local_tvecs_prior) {
    local_tvecs_prior_ = local_tvecs_prior;
}

inline const int8_t& Image::RtkFlag() const { return rtk_flag_; }

inline int8_t& Image::RtkFlag() { return rtk_flag_; }

inline void Image::SetRtkStd(const double std_lon, 
                          const double std_lat, 
                          const double std_hgt){
    std_longitude_ = std_lon;
    std_latitude_ = std_lat;
    std_Altitude_ = std_hgt;
    has_rtk_std_ = true;
};

inline double Image::RtkStdLon()const { return std_longitude_; };

inline double Image::RtkStdLat()const { return std_latitude_; };

inline double Image::RtkStdHgt()const { return std_Altitude_; };

inline bool Image::HasRtkStd() const { return has_rtk_std_; };

inline void Image::SetRelativeAltitude(const double relative_altitude) {
    relative_altitude_ = relative_altitude;
}

inline bool Image::HasRelativeAltitude() {
    return relative_altitude_ != std::numeric_limits<double>::max();
}

inline double Image::RelativeAltitude() { return relative_altitude_; }

inline void Image::SetOrientStd(const double std_orient){
    std_orientation_ = std_orient;
    has_orient_std_ = true;
};

inline double Image::OrientStd() const { return std_orientation_; };

inline bool Image::HasOrientStd() const { return has_orient_std_; };

inline bool Image::DepthFlag() const { return depth_flag_; };

inline void Image::SetDepthFlag(const bool depth_flag) { depth_flag_ = depth_flag; };

const class Point2D& Image::Point2D(const point2D_t point2D_idx) const {
    return points2D_.at(point2D_idx);
}

class Point2D& Image::Point2D(const point2D_t point2D_idx) {
    return points2D_.at(point2D_idx);
}

const std::vector<class Point2D>& Image::Points2D() const { return points2D_; }

inline const std::vector<uint32_t>& Image::LocalImageIndices() const{
    return local_image_indices_;}

bool Image::IsMapPointVisible(const point2D_t point2D_idx) const {
    return num_correspondences_have_mappoint_.at(point2D_idx) > 0;
}

Eigen::Vector7d& Image::Sim3pose() { return sim3_pose_; }

} // namespace sensemap

#endif // SENSEMAP_BASE_IMAGE_H_