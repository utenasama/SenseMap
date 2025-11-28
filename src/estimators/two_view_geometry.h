//Copyright (c) 2019, SenseTime Group.
//All rights reserved.

#ifndef SENSEMAP_ESTIMATORS_TWO_VIEW_GEOMETRY_H_
#define SENSEMAP_ESTIMATORS_TWO_VIEW_GEOMETRY_H_

#include "util/types.h"
#include "util/logging.h"
#include "feature/types.h"
#include "base/camera.h"
#include "optim/ransac/ransac.h"
#include <unordered_map>
namespace sensemap {

// Two-view geometry estimator.
struct TwoViewGeometry {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // The configuration of the estimated two-view geometry.
    enum ConfigurationType {
        UNDEFINED = 0,
        // Degenerate configuration (e.g., no overlap or not enough inliers).
        DEGENERATE = 1,
        // Essential matrix.
        CALIBRATED = 2,
        // Fundamental matrix.
        UNCALIBRATED = 3,
        // Homography, planar scene with baseline.
        PLANAR = 4,
        // Homography, pure rotation without baseline.
        PANORAMIC = 5,
        // Homography, planar or panoramic.
        PLANAR_OR_PANORAMIC = 6,
        // Watermark, pure 2D translation in image borders.
        WATERMARK = 7,
        // Multi-model configuration, i.e. the inlier matches result from
        // multiple individual, non-degenerate configurations.
        MULTIPLE = 8,
    };

    // Estimation options.
    struct Options {
        // Minimum number of inliers for non-degenerate two-view geometry.
        size_t min_num_inliers = 15;

        // Minimum number of inliers for estimating relative pose, 
        // the pose is deemed invalid if the number of inliers is below the 
        // threshold
        size_t min_num_inliers_relative_pose = 80;

        // 
        size_t max_num_matches = 32768;

        // In case both cameras are calibrated, the calibration is verified by
        // estimating an essential and fundamental matrix and comparing their
        // fractions of number of inliers. If the essential matrix produces
        // a similar number of inliers (`min_E_F_inlier_ratio * F_num_inliers`),
        // the calibration is assumed to be correct.
        double min_E_F_inlier_ratio = 0.95;

        // In case an epipolar geometry can be verified, it is checked whether
        // the geometry describes a planar scene or panoramic view (pure 
        // rotation)described by a homography. This is a degenerate case, since
        // epipolar geometry is only defined for a moving camera. If the inlier
        // ratio of a homography comes close to the inlier ratio of the epipolar
        // geometry, a planar or panoramic configuration is assumed.
        double max_H_inlier_ratio = 0.8;

        // In case of valid two-view geometry, it is checked whether the 
        // geometry describes a pure translation in the border region of the 
        // image. If more than a certain ratio of inlier points conform with a
        // pure image translation, a watermark is assumed.
        double watermark_min_inlier_ratio = 0.7;

        // Watermark matches have to be in the border region of the image. The
        // border region is defined as the area around the image borders and
        // is defined as a fraction of the image diagonal.
        double watermark_border_size = 0.1;

        // Whether to enable watermark detection. A watermark causes a pure
        // translation in the image space with inliers in the border region.
        bool detect_watermark = true;

        // Whether to ignore watermark models in multiple model estimation.
        bool multiple_ignore_watermark = true;

        // Maximal number of homography models estiamted for ENFT feature 
        // tracking
        size_t max_H_num_ENFT = 3;

        // Minimal number of inliers for each homograhy in ENFT  
        size_t min_num_inliers_per_H_ENFT = 20;

        // max pixel error for a correspondence
        double max_error = 4.0;
        // max angle error for a correspondence
        double max_angle_error = 0.4;

        bool loose_constraint = false;

        bool is_sphere = false;

        // Options used to robustly estimate the geometry.
        RANSACOptions ransac_options;

        void Check() const {
            CHECK_GE(min_num_inliers, 0);
            CHECK_GE(min_E_F_inlier_ratio, 0);
            CHECK_LE(min_E_F_inlier_ratio, 1);
            CHECK_GE(max_H_inlier_ratio, 0);
            CHECK_LE(max_H_inlier_ratio, 1);
            CHECK_GE(watermark_min_inlier_ratio, 0);
            CHECK_LE(watermark_min_inlier_ratio, 1);
            CHECK_GE(watermark_border_size, 0);
            CHECK_LE(watermark_border_size, 1);
            ransac_options.Check();
        }
    };

    TwoViewGeometry()
        : config(ConfigurationType::UNDEFINED),
            E(Eigen::Matrix3d::Zero()),
            F(Eigen::Matrix3d::Zero()),
            H(Eigen::Matrix3d::Zero()),
            qvec(Eigen::Vector4d::Zero()),
            tvec(Eigen::Vector3d::Zero()),
            tri_angle(0),
            confidence(0) {}

    // Invert the two-view geometry in-place.
    void Invert();


    // Estimate multiple homographies for ENFT feature tracking
    // @param points1           Feature points in the first image
    // @param points2           Feature points in the second image
    // @param h_list            The estimated homographies    
    // @param options           Two-view geometry estimation options
    void EstimateMultipleHomographies(
                                const std::vector<Eigen::Vector2d>& points1,
                                const std::vector<Eigen::Vector2d>& points2,
                                const FeatureMatches& matches,
                                const Options& options,
                                std::vector<Eigen::Matrix3d>& h_list); 

    // Estimate two-view geometry from calibrated or uncalibrated image pair,
    // depending on whether a prior focal length is given or not.
    //
    // @param camera1         Camera of first image.
    // @param points1         Feature points in first image.
    // @param camera2         Camera of second image.
    // @param points2         Feature points in second image.
    // @param matches         Feature matches between first and second image.
    // @param options         Two-view geometry estimation options.
    void Estimate(const Camera& camera1,
                  const std::vector<Eigen::Vector2d>& points1,
                  const Camera& camera2,
                  const std::vector<Eigen::Vector2d>& points2,
                  const FeatureMatches& matches, const Options& options);

    // Recursively estimate multiple configurations by removing the previous set
    // of inliers from the matches until not enough inliers are found. Inlier
    // matches are concatenated and the configuration type is `MULTIPLE` if
    // multiple models could be estimated. This is useful to estimate the 
    // two-view geometry for images with large distortion or multiple rigidly
    // moving objects in the scene.
    //
    // Note that in case the model type is `MULTIPLE`, only the `inlier_matches`
    // field will be initialized.
    //
    // @param camera1         Camera of first image.
    // @param points1         Feature points in first image.
    // @param camera2         Camera of second image.
    // @param points2         Feature points in second image.
    // @param matches         Feature matches between first and second image.
    // @param options         Two-view geometry estimation options.
    void EstimateMultiple(const Camera& camera1,
                          const std::vector<Eigen::Vector2d>& points1,
                          const Camera& camera2,
                          const std::vector<Eigen::Vector2d>& points2,
                          const FeatureMatches& matches, 
                          const Options& options);

    // Estimate two-view geometry and its relative pose from a calibrated or an
    // uncalibrated image pair.
    //
    // @param camera1         Camera of first image.
    // @param points1         Feature points in first image.
    // @param camera2         Camera of second image.
    // @param points2         Feature points in second image.
    // @param matches         Feature matches between first and second image.
    // @param options         Two-view geometry estimation options.
    bool EstimateRelativePose(const Camera& camera1,
                              const std::vector<Eigen::Vector2d>& points1,
                              const Camera& camera2,
                              const std::vector<Eigen::Vector2d>& points2,
                              const Options& options,
                              std::vector<double>* angles=nullptr,
                              bool verbose = false);

    // Estimate relative pose for a pair of shperical cameras
    bool EstimateRelativePoseSphericalCamera(const Camera& camera1,
                              const std::vector<Eigen::Vector2d>& points1,
                              const Camera& camera2,
                              const std::vector<Eigen::Vector2d>& points2,
                              std::vector<double>* angles=nullptr,
                              bool verbose = false);

    // Estimate relative pose for a pair of shperical cameras in the form of a two-camera rig
    bool EstimateRelativePoseSphericalCameraRig(const Camera& camera1,
                              const std::vector<Eigen::Vector2d>& points1,
                              const Camera& camera2,
                              const std::vector<Eigen::Vector2d>& points2,
                              const std::vector<uint32_t>& local_camera_indices1,
                              const std::vector<uint32_t>& local_camera_indices2,
                              std::vector<double>* angles=nullptr,
                              bool verbose = false);
    

    // Estimate two-view geometry from calibrated image pair.
    //
    // @param camera1         Camera of first image.
    // @param points1         Feature points in first image.
    // @param camera2         Camera of second image.
    // @param points2         Feature points in second image.
    // @param matches         Feature matches between first and second image.
    // @param options         Two-view geometry estimation options.
    void EstimateCalibrated(const Camera& camera1,
                            const std::vector<Eigen::Vector2d>& points1,
                            const Camera& camera2,
                            const std::vector<Eigen::Vector2d>& points2,
                            const FeatureMatches& matches,
                            const Options& options);

    // Estimate two-view geometry from spherical image pair.
    //
    // @param camera1         Camera of first image.
    // @param points1         Feature points in first image.
    // @param camera2         Camera of second image.
    // @param points2         Feature points in second image.
    // @param matches         Feature matches between first and second image.
    // @param options         Two-view geometry estimation options.
    void EstimateSpherical(const Camera& camera1,
                            const std::vector<Eigen::Vector2d>& points1,
                            const Camera& camera2,
                            const std::vector<Eigen::Vector2d>& points2,
                            const FeatureMatches& matches,
                            const Options& options);


    // Estimate two-view geometry from uncalibrated image pair.
    //
    // @param camera1         Camera of first image.
    // @param points1         Feature points in first image.
    // @param camera2         Camera of second image.
    // @param points2         Feature points in second image.
    // @param matches         Feature matches between first and second image.
    // @param options         Two-view geometry estimation options.
    void EstimateUncalibrated(const Camera& camera1,
                                const std::vector<Eigen::Vector2d>& points1,
                                const Camera& camera2,
                                const std::vector<Eigen::Vector2d>& points2,
                                const FeatureMatches& matches,
                                const Options& options);

    // Detect if inlier matches are caused by a watermark.
    // A watermark causes a pure translation in the border are of the image.
    static bool DetectWatermark(const Camera& camera1,
                                const std::vector<Eigen::Vector2d>& points1,
                                const Camera& camera2,
                                const std::vector<Eigen::Vector2d>& points2,
                                const size_t num_inliers,
                                const std::vector<char>& inlier_mask,
                                const Options& options);
    


    ////////////////////////////////////////////////////////////////////////////

    // Estimate the two view geometry for non-homogeneous camera.

    void EstimatePespectiveAndSpherical(const Camera& camera1, const std::vector<Eigen::Vector2d>& points1,
                                        const Camera& camera2, const std::vector<Eigen::Vector2d>& points2,
                                        const FeatureMatches& matches, const Options& options);



    // Estimate the two view geometry for camera-rig
    
    
    void EstimateRig(const Camera& camera1,
                  const std::vector<Eigen::Vector2d>& points1,
                  const Camera& camera2,
                  const std::vector<Eigen::Vector2d>& points2,
                  const FeatureMatches& matches,
                  const std::vector<uint32_t>& local_camera_indices1,
                  const std::vector<uint32_t>& local_camera_indices2, 
                  const Options& options);
    

    // @param camera1         Camera of first image.
    // @param points1         Feature points in first image.
    // @param camera2         Camera of second image.
    // @param points2         Feature points in second image.
    // @param matches         Feature matches between first and second image.
    // @param local_image_indices1    Local image indices of points in the 
    //                                first image.
    // @param local_image_indices2    Local image indices of points in the 
    //                                second image.

    // @param options         Two-view geometry estimation options.
    void EstimateCalibratedRig(const Camera& camera1,
                            const std::vector<Eigen::Vector2d>& points1,
                            const Camera& camera2,
                            const std::vector<Eigen::Vector2d>& points2,
                            const FeatureMatches& matches,
                            const std::vector<uint32_t>& local_camera_indices1,
                            const std::vector<uint32_t>& local_camera_indices2,
                            const Options& options);

    // Estimate the two view geometry for non-homogeneous camera.

    void EstimateOneAndRig(const Camera& camera1, const std::vector<Eigen::Vector2d>& points1,
                            const Camera& camera2, const std::vector<Eigen::Vector2d>& points2,
                            const FeatureMatches& matches,
                            const std::vector<uint32_t>& local_camera_indices1,
                            const std::vector<uint32_t>& local_camera_indices2, 
                            const Options& options);

    bool EstimateRelativePoseOneAndRig(const Camera& camera1, const std::vector<Eigen::Vector2d>& points1,
                                       const Camera& camera2, const std::vector<Eigen::Vector2d>& points2,
                                       const std::vector<uint32_t>& local_camera_indices1,
                                       const std::vector<uint32_t>& local_camera_indices2, 
                                       std::vector<double>* angles, bool verbose);
    
    // Estimate relative pose from a calibrated or an uncalibrated image pair,
    // using the estimated E or H
    //
    // @param camera1         Camera of first image.
    // @param points1         Feature points in first image.
    // @param camera2         Camera of second image.
    // @param points2         Feature points in second image.
    // @param options         Two-view geometry estimation options.
    // @param local_image_indices1    Local image indices of points in the 
    //                                first image.
    // @param local_image_indices2    Local image indices of points in the 
    //                                second image.
    bool EstimateRelativePoseRigGV(const Camera& camera1,
                            const std::vector<Eigen::Vector2d>& points1,
                            const Camera& camera2,
                            const std::vector<Eigen::Vector2d>& points2,
                            const std::vector<uint32_t>& local_camera_indices1,
                            const std::vector<uint32_t>& local_camera_indices2,
                            const Options& options,
                            std::vector<double>* angles=nullptr,
                            bool verbose = false);

    

    // One of `ConfigurationType`.
    int config;
    // Essential matrix.
    Eigen::Matrix3d E;
    // Fundamental matrix.
    Eigen::Matrix3d F;
    // Homography matrix.
    Eigen::Matrix3d H;

    std::vector<Eigen::Matrix3d> Es;

    // Multi homography for ENFT.
    std::vector<Eigen::Matrix3d> H_list;

    // Relative pose.
    Eigen::Vector4d qvec;
    Eigen::Vector3d tvec;

    // Relative Pose Confidence.
    double confidence = 0.0;

    // Inlier matches of the configuration.
    FeatureMatches inlier_matches;

    // Median triangulation angle.
    double tri_angle;

    bool relative_pose_valid;

    
    ////////////////////////////////////////////////////////////////////////////

    // Relations for camera rig
    std::unordered_map<int,Eigen::Matrix3d> local_E;

    std::unordered_map<int,std::vector<Eigen::Matrix3d> > local_multiple_Es;

    FeatureMatches local_inlier_matches;
    int best_local_config;
    int best_local_camera_pair;

    Eigen::Vector4d qvec_rig;
    Eigen::Vector3d tvec_rig;

    std::unordered_map<int,Eigen::Matrix3d> valid_local_R;
	std::unordered_map<int, Eigen::Vector3d> valid_local_t;

};

}

#endif