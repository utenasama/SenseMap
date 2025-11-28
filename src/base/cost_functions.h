// Copyright (c) 2019, SenseTime Group.
// All rights reserved.

#ifndef SENSEMAP_BASE_COST_FUNCTIONS_H_
#define SENSEMAP_BASE_COST_FUNCTIONS_H_

#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include <Eigen/Core>

#include "base/similarity_transform.h"
#include "ceres/autodiff_cost_function.h"
#include "base/camera_models.h"
#include "util/jacobian.h"
#include "base/pose.h"

const double rgbd_factor = 10.0;

namespace sensemap {

// Standard bundle adjustment cost function for variable
// camera pose and calibration and point parameters.
template <typename CameraModel>
class BundleAdjustmentCostFunction {
public:
    explicit BundleAdjustmentCostFunction(const Eigen::Vector2d& point2D, const double weight = 1.0)
        : observed_x_(point2D(0)), observed_y_(point2D(1)), weight_(weight) {}

    static ceres::CostFunction* Create(const Eigen::Vector2d& point2D, const double weight = 1.0) {
        return (new ceres::AutoDiffCostFunction<BundleAdjustmentCostFunction<CameraModel>, CameraModel::kNumResidual, 4,
                                                3, 3, CameraModel::kNumParams>(
            new BundleAdjustmentCostFunction(point2D, weight)));
    }

    template <typename T>
    bool operator()(const T* const qvec, const T* const tvec, const T* const point3D, const T* const camera_params,
                    T* residuals) const {
        // Rotate and translate.
        T projection[3];
        ceres::UnitQuaternionRotatePoint(qvec, point3D, projection);
        projection[0] += tvec[0];
        projection[1] += tvec[1];
        projection[2] += tvec[2];

        // Project to image plane.
        projection[0] /= projection[2];
        projection[1] /= projection[2];

        // Distort and transform to pixel space.
        CameraModel::WorldToImage(camera_params, projection[0], projection[1], &residuals[0], &residuals[1]);

        // Re-projection error.
        residuals[0] -= T(observed_x_);
        residuals[1] -= T(observed_y_);

        residuals[0] *= T(weight_);
        residuals[1] *= T(weight_);

        return true;
    }

private:
    const double observed_x_;
    const double observed_y_;
    const double weight_;
};

template <typename CameraModel>
class BundleAdjustmentConstantMapPointCostFunction {
public:
    explicit BundleAdjustmentConstantMapPointCostFunction(
        const Eigen::Vector2d& point2D, const Eigen::Vector3d& point3D, 
        const double weight = 1.0)
        : observed_x_(point2D(0)), observed_y_(point2D(1)), 
          mappoint_x_(point3D(0)), mappoint_y_(point3D(1)), mappoint_z_(point3D(2)),
          weight_(weight) {}

    static ceres::CostFunction* Create(const Eigen::Vector2d& point2D,
        const Eigen::Vector3d& point3D, const double weight = 1.0) {
        return (new ceres::AutoDiffCostFunction<BundleAdjustmentConstantMapPointCostFunction<CameraModel>, CameraModel::kNumResidual, 4, 3, CameraModel::kNumParams>(
            new BundleAdjustmentConstantMapPointCostFunction(point2D, point3D, weight)));
    }

    template <typename T>
    bool operator()(const T* const qvec, const T* const tvec, 
                    const T* const camera_params, T* residuals) const {
        // Rotate and translate.
        T point3D[3] = {T(mappoint_x_), T(mappoint_y_), T(mappoint_z_)};
        T projection[3];
        ceres::UnitQuaternionRotatePoint(qvec, point3D, projection);
        projection[0] += tvec[0];
        projection[1] += tvec[1];
        projection[2] += tvec[2];

        // Project to image plane.
        projection[0] /= projection[2];
        projection[1] /= projection[2];

        // Distort and transform to pixel space.
        CameraModel::WorldToImage(camera_params, projection[0], projection[1], &residuals[0], &residuals[1]);

        // Re-projection error.
        residuals[0] -= T(observed_x_);
        residuals[1] -= T(observed_y_);

        residuals[0] *= T(weight_);
        residuals[1] *= T(weight_);

        return true;
    }

private:
    const double observed_x_;
    const double observed_y_;
    const double mappoint_x_;
    const double mappoint_y_;
    const double mappoint_z_;
    const double weight_;
};

// Bundle adjustment cost function for variable
// camera calibration and point parameters, and fixed camera pose.
template <typename CameraModel>
class BundleAdjustmentConstantPoseCostFunction {
public:
    BundleAdjustmentConstantPoseCostFunction(const Eigen::Vector4d& qvec, const Eigen::Vector3d& tvec,
                                             const Eigen::Vector2d& point2D, const double weight = 1.0)
        : qw_(qvec(0)),
          qx_(qvec(1)),
          qy_(qvec(2)),
          qz_(qvec(3)),
          tx_(tvec(0)),
          ty_(tvec(1)),
          tz_(tvec(2)),
          observed_x_(point2D(0)),
          observed_y_(point2D(1)),
          weight_(weight) {}

    static ceres::CostFunction* Create(const Eigen::Vector4d& qvec, const Eigen::Vector3d& tvec,
                                       const Eigen::Vector2d& point2D, const double weight = 1.0) {
        return (new ceres::AutoDiffCostFunction<BundleAdjustmentConstantPoseCostFunction<CameraModel>,
                                                CameraModel::kNumResidual, 3, CameraModel::kNumParams>(
            new BundleAdjustmentConstantPoseCostFunction(qvec, tvec, point2D, weight)));
    }

    template <typename T>
    bool operator()(const T* const point3D, const T* const camera_params, T* residuals) const {
        const T qvec[4] = {T(qw_), T(qx_), T(qy_), T(qz_)};

        // Rotate and translate.
        T projection[3];
        ceres::UnitQuaternionRotatePoint(qvec, point3D, projection);
        projection[0] += T(tx_);
        projection[1] += T(ty_);
        projection[2] += T(tz_);

        // Project to image plane.
        projection[0] /= projection[2];
        projection[1] /= projection[2];

        // Distort and transform to pixel space.
        CameraModel::WorldToImage(camera_params, projection[0], projection[1], &residuals[0], &residuals[1]);

        // Re-projection error.
        residuals[0] -= T(observed_x_);
        residuals[1] -= T(observed_y_);

        residuals[0] *= T(weight_);
        residuals[1] *= T(weight_);

        return true;
    }

private:
    const double qw_;
    const double qx_;
    const double qy_;
    const double qz_;
    const double tx_;
    const double ty_;
    const double tz_;
    const double observed_x_;
    const double observed_y_;
    const double weight_;
};

template <typename CameraModel>
class BundleAdjustmentConstantPoseAndMapPointCostFunction {
public:
    BundleAdjustmentConstantPoseAndMapPointCostFunction(
        const Eigen::Vector4d& qvec, const Eigen::Vector3d& tvec,
        const Eigen::Vector2d& point2D, const Eigen::Vector3d& point3D,
        const double weight = 1.0)
        : qw_(qvec(0)),
          qx_(qvec(1)),
          qy_(qvec(2)),
          qz_(qvec(3)),
          tx_(tvec(0)),
          ty_(tvec(1)),
          tz_(tvec(2)),
          observed_x_(point2D(0)),
          observed_y_(point2D(1)),
          mappoint_x_(point3D(0)),
          mappoint_y_(point3D(1)),
          mappoint_z_(point3D(2)),
          weight_(weight) {}

    static ceres::CostFunction* Create(const Eigen::Vector4d& qvec, 
        const Eigen::Vector3d& tvec, const Eigen::Vector2d& point2D, const Eigen::Vector3d& point3D, const double weight = 1.0) {
        return (new ceres::AutoDiffCostFunction<BundleAdjustmentConstantPoseAndMapPointCostFunction<CameraModel>,
                                                CameraModel::kNumResidual, 
                                                CameraModel::kNumParams>(
            new BundleAdjustmentConstantPoseAndMapPointCostFunction(qvec, tvec, point2D, point3D, weight)));
    }

    template <typename T>
    bool operator()(const T* const camera_params, T* residuals) const {
        const T qvec[4] = {T(qw_), T(qx_), T(qy_), T(qz_)};
        const T point3D[3] = {T(mappoint_x_), T(mappoint_y_), T(mappoint_z_)};

        // Rotate and translate.
        T projection[3];
        ceres::UnitQuaternionRotatePoint(qvec, point3D, projection);
        projection[0] += T(tx_);
        projection[1] += T(ty_);
        projection[2] += T(tz_);

        // Project to image plane.
        projection[0] /= projection[2];
        projection[1] /= projection[2];

        // Distort and transform to pixel space.
        CameraModel::WorldToImage(camera_params, projection[0], projection[1], &residuals[0], &residuals[1]);

        // Re-projection error.
        residuals[0] -= T(observed_x_);
        residuals[1] -= T(observed_y_);

        residuals[0] *= T(weight_);
        residuals[1] *= T(weight_);

        return true;
    }

private:
    const double qw_;
    const double qx_;
    const double qy_;
    const double qz_;
    const double tx_;
    const double ty_;
    const double tz_;
    const double observed_x_;
    const double observed_y_;
    const double mappoint_x_;
    const double mappoint_y_;
    const double mappoint_z_;
    const double weight_;
};

// Depth re-projection cost function for variable
// camera pose and point parameters.
class BundleAdjustmentDepthCostFunction {
public:
    explicit BundleAdjustmentDepthCostFunction(const Eigen::Vector3d& point3D, const double weight)
        : observed_z_(point3D(2)),
          weight_(weight) {}

    static ceres::CostFunction* Create(const Eigen::Vector3d& point3D, const double weight) {
        return (new ceres::AutoDiffCostFunction<BundleAdjustmentDepthCostFunction,
                                                1, 4, 3, 3>(
            new BundleAdjustmentDepthCostFunction(point3D, weight)));
    }

    template <typename T>
    bool operator()(const T* const qvec, const T* const tvec, const T* const point3D, T* residuals) const {
        // Rotate and translate.
        T projection[3];
        ceres::UnitQuaternionRotatePoint(qvec, point3D, projection);
        projection[2] += tvec[2];

        // Re-projection error.
        residuals[0] = T(projection[2] - observed_z_);

        residuals[0] *= T(weight_);

        return true;
    }

private:
    const double weight_;
    const double observed_z_;
};

// Depth re-projection cost function for variable
// point parameters, and fixed camera pose.
class BundleAdjustmentConstantPoseDepthCostFunction {
public:
    explicit BundleAdjustmentConstantPoseDepthCostFunction(const Eigen::Vector4d& qvec, const Eigen::Vector3d& tvec,
                                                                    const Eigen::Vector3d& point3D, const double weight)
        : qw_(qvec(0)),
          qx_(qvec(1)),
          qy_(qvec(2)),
          qz_(qvec(3)),
          tz_(tvec(2)),
          observed_z_(point3D(2)),
          weight_(weight) {}

    static ceres::CostFunction* Create(const Eigen::Vector4d& qvec, const Eigen::Vector3d& tvec,
                                       const Eigen::Vector3d& point3D, const double weight) {
        return (new ceres::AutoDiffCostFunction<BundleAdjustmentConstantPoseDepthCostFunction,
                                                1, 3>(
            new BundleAdjustmentConstantPoseDepthCostFunction(qvec, tvec, point3D, weight)));
    }

    template <typename T>
    bool operator()(const T* const point3D, T* residuals) const {
        const T qvec[4] = {T(qw_), T(qx_), T(qy_), T(qz_)};

        // Rotate and translate.
        T projection[3];
        ceres::UnitQuaternionRotatePoint(qvec, point3D, projection);
        projection[2] += T(tz_);

        // Re-projection error.
        residuals[0] = T(projection[2] - observed_z_);

        residuals[0] *= T(weight_);

        return true;
    }

private:
    const double weight_;
    const double qw_;
    const double qx_;
    const double qy_;
    const double qz_;
    const double tz_;
    const double observed_z_;
};

// 3D to 3D correspondence error term.
class BundleAdjustmentConstantPose3DErrorCostFunction {
public:
    explicit BundleAdjustmentConstantPose3DErrorCostFunction(const Eigen::Vector4d& qvec, const Eigen::Vector3d& tvec,
                                                             const Eigen::Vector3d& point3D, const double info)
        : qw_(qvec(0)),
          qx_(qvec(1)),
          qy_(qvec(2)),
          qz_(qvec(3)),
          tx_(tvec(0)),
          ty_(tvec(1)),
          tz_(tvec(2)),
          observed_x_(point3D(0)),
          observed_y_(point3D(1)),
          observed_z_(point3D(2)),
          info_(info) {}

    static ceres::CostFunction* Create(const Eigen::Vector4d& qvec, const Eigen::Vector3d& tvec,
                                       const Eigen::Vector3d& point3D, const double info) {
        return (new ceres::AutoDiffCostFunction<BundleAdjustmentConstantPose3DErrorCostFunction, 3, 3>(
            new BundleAdjustmentConstantPose3DErrorCostFunction(qvec, tvec, point3D, info)));
    }

    template <typename T>
    bool operator()(const T* const point3D, T* residuals) const {
        const T qvec[4] = {T(qw_), T(qx_), T(qy_), T(qz_)};

        // Rotate and translate.
        T projection[3];
        ceres::UnitQuaternionRotatePoint(qvec, point3D, projection);
        projection[0] += T(tx_);
        projection[1] += T(ty_);
        projection[2] += T(tz_);

        residuals[0] = T(rgbd_factor * info_) * T(projection[0] - observed_x_);
        residuals[1] = T(rgbd_factor * info_) * T(projection[1] - observed_y_);
        residuals[2] = T(rgbd_factor * info_) * T(projection[2] - observed_z_);

        return true;
    }

private:
    const double info_;
    const double qw_;
    const double qx_;
    const double qy_;
    const double qz_;
    const double tx_;
    const double ty_;
    const double tz_;
    const double observed_x_;
    const double observed_y_;
    const double observed_z_;
};

// 3D to 3D correspondence error term for variable
// point parameters, and fixed camera pose.
class BundleAdjustment3DErrorCostFunction {
public:
    explicit BundleAdjustment3DErrorCostFunction(const Eigen::Vector3d& point3D, const double info)
        : observed_x_(point3D(0)), observed_y_(point3D(1)), observed_z_(point3D(2)), info_(info) {}

    static ceres::CostFunction* Create(const Eigen::Vector3d& point3D, const double info) {
        return (new ceres::AutoDiffCostFunction<BundleAdjustment3DErrorCostFunction, 3, 4, 3, 3>(
            new BundleAdjustment3DErrorCostFunction(point3D, info)));
    }

    template <typename T>
    bool operator()(const T* const qvec, const T* const tvec, const T* const point3D, T* residuals) const {
        // Rotate and translate.
        T projection[3];
        ceres::UnitQuaternionRotatePoint(qvec, point3D, projection);
        projection[0] += tvec[0];
        projection[1] += tvec[1];
        projection[2] += tvec[2];

        residuals[0] = T(rgbd_factor * info_) * T(projection[0] - observed_x_);
        residuals[1] = T(rgbd_factor * info_) * T(projection[1] - observed_y_);
        residuals[2] = T(rgbd_factor * info_) * T(projection[2] - observed_z_);

        return true;
    }

private:
    const double info_;
    const double observed_x_;
    const double observed_y_;
    const double observed_z_;
};

// Specialization for spherical camera, which is splitted into two cameras

template <typename CameraModel>
class SphericalBundleAdjustmentCostFunction {
public:
    explicit SphericalBundleAdjustmentCostFunction(const Eigen::Vector3d& bearing, const double f, const double weight = 1.0)
        : observed_x_(bearing(0)), observed_y_(bearing(1)), observed_z_(bearing(2)), f_(f), weight_(weight) {}

    static ceres::CostFunction* Create(const Eigen::Vector3d& bearing, const double f, const double weight = 1.0) {
        return (new ceres::AutoDiffCostFunction<SphericalBundleAdjustmentCostFunction<CameraModel>,
                                                CameraModel::kNumResidual, 4, 3, 3,
                                                4, 3>(new SphericalBundleAdjustmentCostFunction(bearing, f, weight)));
    }

    template <typename T>
    bool operator()(const T* const qvec, const T* const tvec, const T* const point3D,
                    const T* const local_qvec2,
                    const T* const local_tvec2, T* residuals) const {
        // Rotate and translate.
        T projection[3];
        ceres::UnitQuaternionRotatePoint(qvec, point3D, projection);
        projection[0] += tvec[0];
        projection[1] += tvec[1];
        projection[2] += tvec[2];


        const T point3D_c[3] = {projection[0], projection[1], projection[2]};
        if (!(observed_z_ >= 0)) {
            ceres::UnitQuaternionRotatePoint(local_qvec2, point3D_c, projection);
            projection[0] += local_tvec2[0];
            projection[1] += local_tvec2[1];
            projection[2] += local_tvec2[2];
        }

        T radius_projection = T(
            ceres::sqrt(projection[0] * projection[0] + projection[1] * projection[1] + projection[2] * projection[2]));
        if (radius_projection > T(0)) {
            projection[0] /= radius_projection;
            projection[1] /= radius_projection;
            projection[2] /= radius_projection;
        }

        residuals[0] = f_ * (projection[0] - observed_x_);
        residuals[1] = f_ * (projection[1] - observed_y_);
        residuals[2] = f_ * (projection[2] - observed_z_);

        residuals[0] *= weight_;
        residuals[1] *= weight_;
        residuals[2] *= weight_;

        return true;
    }

private:
    const double observed_x_;
    const double observed_y_;
    const double observed_z_;
    const double f_;
    const double weight_;
};

template <typename CameraModel>
class SphericalBundleAdjustmentConstantPoseCostFunction {
public:
    SphericalBundleAdjustmentConstantPoseCostFunction(const Eigen::Vector4d& qvec, const Eigen::Vector3d& tvec,
                                                      const Eigen::Vector3d& bearing, const double f,
                                                      const double weight = 1.0)
        : qw_(qvec(0)),
          qx_(qvec(1)),
          qy_(qvec(2)),
          qz_(qvec(3)),
          tx_(tvec(0)),
          ty_(tvec(1)),
          tz_(tvec(2)),
          observed_x_(bearing(0)),
          observed_y_(bearing(1)),
          observed_z_(bearing(2)),
          f_(f),
          weight_(weight) {}

    static ceres::CostFunction* Create(const Eigen::Vector4d& qvec, const Eigen::Vector3d& tvec,
                                       const Eigen::Vector3d& bearing, const double f, const double weight = 1.0) {
        return (new ceres::AutoDiffCostFunction<SphericalBundleAdjustmentConstantPoseCostFunction<CameraModel>,
                                                CameraModel::kNumResidual, 3, 4, 3>(
            new SphericalBundleAdjustmentConstantPoseCostFunction(qvec, tvec, bearing, f, weight)));
    }

    template <typename T>
    bool operator()(const T* const point3D, const T* const local_qvec2, const T* const local_tvec2,
                    T* residuals) const {
        const T qvec[4] = {T(qw_), T(qx_), T(qy_), T(qz_)};

        // Rotate and translate.
        T projection[3];
        ceres::UnitQuaternionRotatePoint(qvec, point3D, projection);
        projection[0] += T(tx_);
        projection[1] += T(ty_);
        projection[2] += T(tz_);


        const T point3D_c[3] = {projection[0], projection[1], projection[2]};

        if (!(observed_z_ >= 0)) {
            ceres::UnitQuaternionRotatePoint(local_qvec2, point3D_c, projection);
            projection[0] += local_tvec2[0];
            projection[1] += local_tvec2[1];
            projection[2] += local_tvec2[2];
        }

        T radius_projection = T(
            ceres::sqrt(projection[0] * projection[0] + projection[1] * projection[1] + projection[2] * projection[2]));
        if (radius_projection > T(0)) {
            projection[0] /= radius_projection;
            projection[1] /= radius_projection;
            projection[2] /= radius_projection;
        }

        residuals[0] = f_ * (projection[0] - observed_x_);
        residuals[1] = f_ * (projection[1] - observed_y_);
        residuals[2] = f_ * (projection[2] - observed_z_);

        residuals[0] *= weight_;
        residuals[1] *= weight_;
        residuals[2] *= weight_;

        return true;
    }

private:
    const double qw_;
    const double qx_;
    const double qy_;
    const double qz_;
    const double tx_;
    const double ty_;
    const double tz_;
    const double observed_x_;
    const double observed_y_;
    const double observed_z_;
    const double f_;
    const double weight_;
};

// Analytical derivative cost function
template <typename CameraModel>
class AnalyticalSphericalBACostFunction
    : public ceres::SizedCostFunction<CameraModel::kNumResidual, 4, 3, 3, 4, 3> {

public:

    AnalyticalSphericalBACostFunction(const Eigen::Vector3d& bearing, const double f)
    : observed_x_(bearing(0)), observed_y_(bearing(1)), observed_z_(bearing(2)), f_(f){}


    virtual bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const{

        Eigen::Map<const Eigen::Vector4d> qvec(parameters[0]);
        Eigen::Map<const Eigen::Vector3d> tvec(parameters[1]);
        Eigen::Map<const Eigen::Vector3d> point3D(parameters[2]);
        Eigen::Map<const Eigen::Vector4d> local_qvec2(parameters[3]);
        Eigen::Map<const Eigen::Vector3d> local_tvec2(parameters[4]);

        Eigen::Matrix3d R = QuaternionToRotationMatrix(qvec);
        Eigen::Matrix3d local_R2 = QuaternionToRotationMatrix(local_qvec2);

        double projection[3];
        ceres::UnitQuaternionRotatePoint(qvec.data(), point3D.data(), projection);
        projection[0] += tvec[0];
        projection[1] += tvec[1];
        projection[2] += tvec[2];

        const double point3D_c[3] = {projection[0], projection[1], projection[2]};
        if (!(observed_z_ >= 0)) {
            ceres::UnitQuaternionRotatePoint(local_qvec2.data(), point3D_c, projection);
            projection[0] += local_tvec2[0];
            projection[1] += local_tvec2[1];
            projection[2] += local_tvec2[2];
        }

        double norm = 
            ceres::sqrt(projection[0] * projection[0] + projection[1] * projection[1] + projection[2] * projection[2]);
        
        double normalized_projection[3];
        if (norm > 0) {
            normalized_projection[0] = projection[0] / norm;
            normalized_projection[1] = projection[1] / norm;
            normalized_projection[2] = projection[2] / norm;
        }
        else{
            normalized_projection[0] = 0;
            normalized_projection[1] = 0;
            normalized_projection[2] = 0;
        }

        residuals[0] = f_ * (normalized_projection[0] - observed_x_);
        residuals[1] = f_ * (normalized_projection[1] - observed_y_);
        residuals[2] = f_ * (normalized_projection[2] - observed_z_);

        if(jacobians && norm > 0){
            Eigen::Matrix3d point_normalization_jacobian;
            Point3DNormalizationJacobian(projection,norm,point_normalization_jacobian);

            if(jacobians[0]){
                Eigen::Map<Eigen::Matrix<double,3,4,Eigen::RowMajor>> jacobian_q(jacobians[0]);
                QuaternionJacobian(qvec.data(),point3D.data(),jacobian_q);
                if(observed_z_ >=0) {
                    jacobian_q = f_* point_normalization_jacobian * jacobian_q;  
                }
                else{
                    jacobian_q = f_* point_normalization_jacobian *  local_R2 * jacobian_q;
                }
            }

            if(jacobians[1]){
                Eigen::Map<Eigen::Matrix<double,3,3,Eigen::RowMajor>> jacobian_t(jacobians[1]);
                if(observed_z_ >=0){
                    jacobian_t = f_ * point_normalization_jacobian;
                }
                else{
                    jacobian_t = f_ * point_normalization_jacobian *  local_R2;
                }
            }

            if(jacobians[2]){
                Eigen::Map<Eigen::Matrix<double,3,3,Eigen::RowMajor>> jacobian_point3d(jacobians[2]);
                if(observed_z_ >=0){
                    jacobian_point3d = f_ * point_normalization_jacobian * R; 
                }
                else{
                    jacobian_point3d = f_ * point_normalization_jacobian *  local_R2 * R;
                }
            }

            if(jacobians[3]){
                if(observed_z_>=0){
                    memset(jacobians[3],0,12*sizeof(double));
                }
                else{
                    Eigen::Map<Eigen::Matrix<double,3,4,Eigen::RowMajor>> jacobian_local_q2(jacobians[3]);
                    QuaternionJacobian(local_qvec2.data(),point3D_c,jacobian_local_q2);  
                    jacobian_local_q2 = f_ * point_normalization_jacobian * jacobian_local_q2;
                }    
            }
            
            if(jacobians[4]){
                if(observed_z_>=0){
                    memset(jacobians[4],0,9*sizeof(double));
                }
                else{
                    Eigen::Map<Eigen::Matrix<double,3,3,Eigen::RowMajor>> jacobian_local_t2(jacobians[4]);
                    jacobian_local_t2 = f_ * point_normalization_jacobian;
                }
            }

        }
        else if(jacobians){
            if(jacobians[0]){
                memset(jacobians[0],0,12*sizeof(double)); 
            }

            if(jacobians[1]){
                memset(jacobians[1],0,9*sizeof(double)); 
            }

            if(jacobians[2]){
                memset(jacobians[2],0,9*sizeof(double)); 
            }

            if(jacobians[3]){
                memset(jacobians[3],0,12*sizeof(double)); 
            }

            if(jacobians[4]){
                memset(jacobians[4],0,9*sizeof(double)); 
            }
        }
        return true;
    }

    const double observed_x_;
    const double observed_y_;
    const double observed_z_;
    const double f_;
};





// Analytical derivative cost function
template <typename CameraModel>
class AnalyticalSphericalBAConstantPoseCostFunction
    : public ceres::SizedCostFunction<CameraModel::kNumResidual, 3, 4, 3> {

public:  
    AnalyticalSphericalBAConstantPoseCostFunction(const Eigen::Vector4d& qvec, const Eigen::Vector3d& tvec,
                                              const Eigen::Vector3d& bearing, const double f)
        : qw_(qvec(0)),
          qx_(qvec(1)),
          qy_(qvec(2)),
          qz_(qvec(3)),
          tx_(tvec(0)),
          ty_(tvec(1)),
          tz_(tvec(2)),
          observed_x_(bearing(0)),
          observed_y_(bearing(1)),
          observed_z_(bearing(2)),
          f_(f) {}

    virtual bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const{

        Eigen::Vector4d qvec(qw_, qx_, qy_, qz_);
        Eigen::Vector3d tvec(tx_,ty_,tz_);
        Eigen::Map<const Eigen::Vector3d> point3D(parameters[0]);
        Eigen::Map<const Eigen::Vector4d> local_qvec2(parameters[1]);
        Eigen::Map<const Eigen::Vector3d> local_tvec2(parameters[2]);


        Eigen::Matrix3d R = QuaternionToRotationMatrix(qvec);
        Eigen::Matrix3d local_R2 = QuaternionToRotationMatrix(local_qvec2);

        double projection[3];
        ceres::UnitQuaternionRotatePoint(qvec.data(), point3D.data(), projection);
        projection[0] += tvec[0];
        projection[1] += tvec[1];
        projection[2] += tvec[2];

        const double point3D_c[3] = {projection[0], projection[1], projection[2]};
        if (!(observed_z_ >= 0)){
            ceres::UnitQuaternionRotatePoint(local_qvec2.data(), point3D_c, projection);
            projection[0] += local_tvec2[0];
            projection[1] += local_tvec2[1];
            projection[2] += local_tvec2[2];
        }

        double norm = 
            ceres::sqrt(projection[0] * projection[0] + projection[1] * projection[1] + projection[2] * projection[2]);
        
        double normalized_projection[3];
        if (norm > 0) {
            normalized_projection[0] = projection[0] / norm;
            normalized_projection[1] = projection[1] / norm;
            normalized_projection[2] = projection[2] / norm;
        }
        else{
            normalized_projection[0] = 0;
            normalized_projection[1] = 0;
            normalized_projection[2] = 0;
        }

        residuals[0] = f_ * (normalized_projection[0] - observed_x_);
        residuals[1] = f_ * (normalized_projection[1] - observed_y_);
        residuals[2] = f_ * (normalized_projection[2] - observed_z_);



        if(jacobians && norm > 0){
            Eigen::Matrix3d point_normalization_jacobian;
            Point3DNormalizationJacobian(projection,norm,point_normalization_jacobian);
            if(jacobians[0]){
                Eigen::Map<Eigen::Matrix<double,3,3,Eigen::RowMajor>> jacobian_point3d(jacobians[0]);
                if(observed_z_ >=0){
                    jacobian_point3d = f_ * point_normalization_jacobian * R; 
                }
                else{
                    jacobian_point3d = f_ * point_normalization_jacobian * local_R2 * R;
                }
            }

            if(jacobians[1]){
                if(observed_z_>=0){
                    memset(jacobians[1],0,12*sizeof(double));
                }
                else{
                    Eigen::Map<Eigen::Matrix<double,3,4,Eigen::RowMajor>> jacobian_local_q2(jacobians[1]);
                    QuaternionJacobian(local_qvec2.data(),point3D_c,jacobian_local_q2);     
                    jacobian_local_q2 = f_ * point_normalization_jacobian * jacobian_local_q2;
                }    
            }
            
            if(jacobians[2]){
                if(observed_z_>=0){
                    memset(jacobians[2],0,9*sizeof(double));
                }
                else{
                    Eigen::Map<Eigen::Matrix<double,3,3,Eigen::RowMajor>> jacobian_local_t2(jacobians[2]);
                    jacobian_local_t2 = f_ * point_normalization_jacobian; 
                }
            }
        }
        else if(jacobians){

            if(jacobians[0]){
                memset(jacobians[0],0,9*sizeof(double));
            }

            if(jacobians[1]){
                memset(jacobians[1],0,12*sizeof(double));
            }

            if(jacobians[2]){
                memset(jacobians[2],0,9*sizeof(double));
            }
        }

        return true;
    }


    const double qw_;
    const double qx_;
    const double qy_;
    const double qz_;
    const double tx_;
    const double ty_;
    const double tz_;
    const double observed_x_;
    const double observed_y_;
    const double observed_z_;
    const double f_;
};


// Cost function for large fov camera model
template <typename CameraModel>
class LargeFovBundleAdjustmentCostFunction {
public:
    explicit LargeFovBundleAdjustmentCostFunction(const Eigen::Vector3d& bearing, const double f,
                                                  const double weight = 1.0)
        : observed_x_(bearing(0)), observed_y_(bearing(1)), observed_z_(bearing(2)), f_(f), weight_(weight) {}

    static ceres::CostFunction* Create(const Eigen::Vector3d& bearing, const double f, const double weight = 1.0) {
        return (new ceres::AutoDiffCostFunction<LargeFovBundleAdjustmentCostFunction<CameraModel>,
                                                CameraModel::kNumResidual + 1, 4, 3, 3>(
            new LargeFovBundleAdjustmentCostFunction(bearing, f, weight)));
    }

    template <typename T>
    bool operator()(const T* const qvec, const T* const tvec, const T* const point3D, T* residuals) const {
        // Rotate and translate.
        T projection[3];
        ceres::UnitQuaternionRotatePoint(qvec, point3D, projection);
        projection[0] += tvec[0];
        projection[1] += tvec[1];
        projection[2] += tvec[2];

        T radius_projection = T(
            ceres::sqrt(projection[0] * projection[0] + projection[1] * projection[1] + projection[2] * projection[2]));
        if (radius_projection > T(0)) {
            projection[0] /= radius_projection;
            projection[1] /= radius_projection;
            projection[2] /= radius_projection;
        }

        residuals[0] = f_ * (projection[0] - observed_x_);
        residuals[1] = f_ * (projection[1] - observed_y_);
        residuals[2] = f_ * (projection[2] - observed_z_);

        residuals[0] *= weight_;
        residuals[1] *= weight_;
        residuals[2] *= weight_;

        return true;
    }

private:
    const double observed_x_;
    const double observed_y_;
    const double observed_z_;
    const double f_;
    const double weight_;
};

// Bundle adjustment cost function for variable
// camera calibration and point parameters, and fixed camera pose.
template <typename CameraModel>
class LargeFovBundleAdjustmentConstantPoseCostFunction {
public:
    LargeFovBundleAdjustmentConstantPoseCostFunction(const Eigen::Vector4d& qvec, const Eigen::Vector3d& tvec,
                                                     const Eigen::Vector3d& bearing, const double f,
                                                     const double weight = 1.0)
        : qw_(qvec(0)),
          qx_(qvec(1)),
          qy_(qvec(2)),
          qz_(qvec(3)),
          tx_(tvec(0)),
          ty_(tvec(1)),
          tz_(tvec(2)),
          observed_x_(bearing(0)),
          observed_y_(bearing(1)),
          observed_z_(bearing(2)),
          f_(f),
          weight_(weight) {}

    static ceres::CostFunction* Create(const Eigen::Vector4d& qvec, const Eigen::Vector3d& tvec,
                                       const Eigen::Vector3d& bearing, const double f, const double weight = 1.0) {
        return (new ceres::AutoDiffCostFunction<LargeFovBundleAdjustmentConstantPoseCostFunction<CameraModel>,
                                                CameraModel::kNumResidual + 1, 3>(
            new LargeFovBundleAdjustmentConstantPoseCostFunction(qvec, tvec, bearing, f, weight)));
    }

    template <typename T>
    bool operator()(const T* const point3D, T* residuals) const {
        const T qvec[4] = {T(qw_), T(qx_), T(qy_), T(qz_)};

        // Rotate and translate.
        T projection[3];
        ceres::UnitQuaternionRotatePoint(qvec, point3D, projection);
        projection[0] += T(tx_);
        projection[1] += T(ty_);
        projection[2] += T(tz_);

        T radius_projection = T(
            ceres::sqrt(projection[0] * projection[0] + projection[1] * projection[1] + projection[2] * projection[2]));
        if (radius_projection > T(0)) {
            projection[0] /= radius_projection;
            projection[1] /= radius_projection;
            projection[2] /= radius_projection;
        }

        residuals[0] = T(f_) * (projection[0] - T(observed_x_));
        residuals[1] = T(f_) * (projection[1] - T(observed_y_));
        residuals[2] = T(f_) * (projection[2] - T(observed_z_));

        residuals[0] *= T(weight_);
        residuals[1] *= T(weight_);
        residuals[2] *= T(weight_);

        return true;
    }

private:
    const double qw_;
    const double qx_;
    const double qy_;
    const double qz_;
    const double tx_;
    const double ty_;
    const double tz_;
    const double observed_x_;
    const double observed_y_;
    const double observed_z_;
    const double f_;
    const double weight_;
};

// Rig bundle adjustment cost function for variable camera pose and calibration
// and point parameters. Different from the standard bundle adjustment function,
// this cost function is suitable for camera rigs with consistent relative poses
// of the cameras within the rig. The cost function first projects points into
// the local system of the camera rig and then into the local system of the
// camera within the rig.
template <typename CameraModel>
class RigBundleAdjustmentCostFunction {
public:
    explicit RigBundleAdjustmentCostFunction(const Eigen::Vector2d& point2D,
        const double factor = 1.0)
        : observed_x_(point2D(0)), observed_y_(point2D(1)), factor_(factor) {}

    static ceres::CostFunction* Create(const Eigen::Vector2d& point2D,
                                       const double factor = 1.0) {
        return (new ceres::AutoDiffCostFunction<RigBundleAdjustmentCostFunction<CameraModel>, CameraModel::kNumResidual,
                                                4, 3, 4, 3, 3, CameraModel::kNumParams>(
            new RigBundleAdjustmentCostFunction(point2D, factor)));
    }

    template <typename T>
    bool operator()(const T* const rig_qvec, const T* const rig_tvec, const T* const rel_qvec, const T* const rel_tvec,
                    const T* const point3D, const T* const camera_params, T* residuals) const {
        // Concatenate rotations.
        T qvec[4];
        ceres::QuaternionProduct(rel_qvec, rig_qvec, qvec);

        // Concatenate translations.
        T tvec[3];
        ceres::UnitQuaternionRotatePoint(rel_qvec, rig_tvec, tvec);
        tvec[0] += rel_tvec[0];
        tvec[1] += rel_tvec[1];
        tvec[2] += rel_tvec[2];

        // Rotate and translate.
        T projection[3];
        ceres::UnitQuaternionRotatePoint(qvec, point3D, projection);
        projection[0] += tvec[0];
        projection[1] += tvec[1];
        projection[2] += tvec[2];

        if (projection[2] < T(std::numeric_limits<double>::epsilon())) {
            projection[2] = T(std::numeric_limits<double>::epsilon());
        }

        // Project to image plane.
        projection[0] /= projection[2];
        projection[1] /= projection[2];
        CameraModel::WorldToImage(camera_params, projection[0], projection[1], &residuals[0], &residuals[1]);

        // Re-projection error.
        // residuals[0] -= T(observed_x_);
        // residuals[1] -= T(observed_y_);
        residuals[0] = T(factor_) * (residuals[0] - T(observed_x_));
        residuals[1] = T(factor_) * (residuals[1] - T(observed_y_));

        return true;
    }

private:
    const double observed_x_;
    const double observed_y_;
    const double factor_;
};

// Rig bundle adjustment cost function for fixed camera pose and variable
// calibration and point parameters. Different from the standard bundle
// adjustment function,this cost function is suitable for camera rigs with
// consistent relative pose of the cameras within the rig. The cost function
// first projects points into the local system of the camera rig and then into
// the local system of the camera within the rig.

template <typename CameraModel>
class RigBundleAdjustmentConstantPoseCostFunction {
public:
    explicit RigBundleAdjustmentConstantPoseCostFunction(
        const Eigen::Vector4d& qvec, const Eigen::Vector3d& tvec,
        const Eigen::Vector2d& point2D, const double factor = 1.0)
        : qw_(qvec(0)),
          qx_(qvec(1)),
          qy_(qvec(2)),
          qz_(qvec(3)),
          tx_(tvec(0)),
          ty_(tvec(1)),
          tz_(tvec(2)),
          observed_x_(point2D(0)),
          observed_y_(point2D(1)),
          factor_(factor) {}

    static ceres::CostFunction* Create(const Eigen::Vector4d& qvec, 
                                       const Eigen::Vector3d& tvec, 
                                       const Eigen::Vector2d& point2D,
                                       const double factor = 1.0) {
        return (new ceres::AutoDiffCostFunction<RigBundleAdjustmentConstantPoseCostFunction<CameraModel>,
                                                CameraModel::kNumResidual, 4, 3, 3, CameraModel::kNumParams>(
            new RigBundleAdjustmentConstantPoseCostFunction(qvec, tvec, point2D, factor)));
    }

    template <typename T>
    bool operator()(const T* const rel_qvec, const T* const rel_tvec, const T* const point3D,
                    const T* const camera_params, T* residuals) const {
        const T rig_qvec[4] = {T(qw_), T(qx_), T(qy_), T(qz_)};
        const T rig_tvec[3] = {T(tx_), T(ty_), T(tz_)};

        // Concatenate rotations.
        T qvec[4];
        ceres::QuaternionProduct(rel_qvec, rig_qvec, qvec);

        // Concatenate translations.
        T tvec[3];
        ceres::UnitQuaternionRotatePoint(rel_qvec, rig_tvec, tvec);
        tvec[0] += rel_tvec[0];
        tvec[1] += rel_tvec[1];
        tvec[2] += rel_tvec[2];

        // Rotate and translate.
        T projection[3];
        ceres::UnitQuaternionRotatePoint(qvec, point3D, projection);
        projection[0] += tvec[0];
        projection[1] += tvec[1];
        projection[2] += tvec[2];

        if (projection[2] < T(std::numeric_limits<double>::epsilon())) {
            projection[2] = T(std::numeric_limits<double>::epsilon());
        }

        // Project to image plane.
        projection[0] /= projection[2];
        projection[1] /= projection[2];
        CameraModel::WorldToImage(camera_params, projection[0], projection[1], &residuals[0], &residuals[1]);

        // Re-projection error.
        // residuals[0] -= T(observed_x_);
        // residuals[1] -= T(observed_y_);
        residuals[0] = T(factor_) * (residuals[0] - T(observed_x_));
        residuals[1] = T(factor_) * (residuals[1] - T(observed_y_));

        return true;
    }

private:
    const double qw_;
    const double qx_;
    const double qy_;
    const double qz_;
    const double tx_;
    const double ty_;
    const double tz_;
    const double observed_x_;
    const double observed_y_;
    const double factor_;
};

// Large Fov camera rig
template <typename CameraModel>
class LargeFovRigBundleAdjustmentCostFunction {
public:
    explicit LargeFovRigBundleAdjustmentCostFunction(const Eigen::Vector3d& bearing, const double f, const double weight = 1.0)
        : observed_x_(bearing(0)), observed_y_(bearing(1)), observed_z_(bearing(2)), f_(f), weight_(weight) {}

    static ceres::CostFunction* Create(const Eigen::Vector3d& bearing, const double f, const double weight = 1.0) {
        return (new ceres::AutoDiffCostFunction<LargeFovRigBundleAdjustmentCostFunction<CameraModel>,
                                                CameraModel::kNumResidual+1, 4, 3, 4, 3, 3>(
            new LargeFovRigBundleAdjustmentCostFunction(bearing, f, weight)));
    }

    template <typename T>
    bool operator()(const T* const rig_qvec, const T* const rig_tvec, const T* const rel_qvec, const T* const rel_tvec,
                    const T* const point3D, T* residuals) const {
        // Concatenate rotations.
        T qvec[4];
        ceres::QuaternionProduct(rel_qvec, rig_qvec, qvec);

        // Concatenate translations.
        T tvec[3];
        ceres::UnitQuaternionRotatePoint(rel_qvec, rig_tvec, tvec);
        tvec[0] += rel_tvec[0];
        tvec[1] += rel_tvec[1];
        tvec[2] += rel_tvec[2];

        // Rotate and translate.
        T projection[3];
        ceres::UnitQuaternionRotatePoint(qvec, point3D, projection);
        projection[0] += tvec[0];
        projection[1] += tvec[1];
        projection[2] += tvec[2];

        T radius_projection = T(
            ceres::sqrt(projection[0] * projection[0] + projection[1] * projection[1] + projection[2] * projection[2]));
        if (radius_projection > T(0)) {
            projection[0] /= radius_projection;
            projection[1] /= radius_projection;
            projection[2] /= radius_projection;
        }

        residuals[0] = f_ * (projection[0] - observed_x_);
        residuals[1] = f_ * (projection[1] - observed_y_);
        residuals[2] = f_ * (projection[2] - observed_z_);

        residuals[0] *= T(weight_);
        residuals[1] *= T(weight_);
        residuals[2] *= T(weight_);

        return true;
    }

private:
    const double observed_x_;
    const double observed_y_;
    const double observed_z_;
    const double f_;
    const double weight_;
};

// Large Fov camera rig with constant pose

template <typename CameraModel>
class LargeFovRigBundleAdjustmentConstantPoseCostFunction {
public:
    explicit LargeFovRigBundleAdjustmentConstantPoseCostFunction(const Eigen::Vector4d& qvec,
                                                                 const Eigen::Vector3d& tvec,
                                                                 const Eigen::Vector3d& bearing, const double f)
        : qw_(qvec(0)),
          qx_(qvec(1)),
          qy_(qvec(2)),
          qz_(qvec(3)),
          tx_(tvec(0)),
          ty_(tvec(1)),
          tz_(tvec(2)),
          observed_x_(bearing(0)),
          observed_y_(bearing(1)),
          observed_z_(bearing(2)),
          f_(f) {}

    static ceres::CostFunction* Create(const Eigen::Vector4d& qvec, const Eigen::Vector3d& tvec,
                                       const Eigen::Vector3d& bearing, const double f) {
        return (new ceres::AutoDiffCostFunction<LargeFovRigBundleAdjustmentConstantPoseCostFunction<CameraModel>,
                                                CameraModel::kNumResidual + 1, 4, 3, 3>(
            new LargeFovRigBundleAdjustmentConstantPoseCostFunction(qvec, tvec, bearing, f)));
    }

    template <typename T>
    bool operator()(const T* const rel_qvec, const T* const rel_tvec, const T* const point3D, T* residuals) const {
        const T rig_qvec[4] = {T(qw_), T(qx_), T(qy_), T(qz_)};
        const T rig_tvec[3] = {T(tx_), T(ty_), T(tz_)};

        // Concatenate rotations.
        T qvec[4];
        ceres::QuaternionProduct(rel_qvec, rig_qvec, qvec);

        // Concatenate translations.
        T tvec[3];
        ceres::UnitQuaternionRotatePoint(rel_qvec, rig_tvec, tvec);
        tvec[0] += rel_tvec[0];
        tvec[1] += rel_tvec[1];
        tvec[2] += rel_tvec[2];

        // Rotate and translate.
        T projection[3];
        ceres::UnitQuaternionRotatePoint(qvec, point3D, projection);
        projection[0] += tvec[0];
        projection[1] += tvec[1];
        projection[2] += tvec[2];

        T radius_projection = T(
            ceres::sqrt(projection[0] * projection[0] + projection[1] * projection[1] + projection[2] * projection[2]));
        if (radius_projection > T(0)) {
            projection[0] /= radius_projection;
            projection[1] /= radius_projection;
            projection[2] /= radius_projection;
        }

        residuals[0] = f_ * (projection[0] - observed_x_);
        residuals[1] = f_ * (projection[1] - observed_y_);
        residuals[2] = f_ * (projection[2] - observed_z_);

        return true;
    }

private:
    const double qw_;
    const double qx_;
    const double qy_;
    const double qz_;
    const double tx_;
    const double ty_;
    const double tz_;
    const double observed_x_;
    const double observed_y_;
    const double observed_z_;
    const double f_;
};

class RigLocalRelativeBundleAdjustmentPoseFunction {
public:
    explicit RigLocalRelativeBundleAdjustmentPoseFunction(const Eigen::Vector3d& weight)
        : weight_x_(weight[0]),
          weight_y_(weight[1]),
          weight_z_(weight[2]) {}

    static ceres::CostFunction* Create(const Eigen::Vector3d weight) {
        return (new ceres::AutoDiffCostFunction<RigLocalRelativeBundleAdjustmentPoseFunction, 3, 4, 3, 4, 3>(
            new RigLocalRelativeBundleAdjustmentPoseFunction(weight)));
    }

    template <typename T>
    bool operator()(const T* const qvec0, const T* const tvec0, const T* const qvec1, const T* const tvec1, T* residuals) const {
        const T qvec0_inverse[4] = {T(qvec0[0]), T(-qvec0[1]), T(-qvec0[2]), T(-qvec0[3])};
        T C0[3];
        ceres::UnitQuaternionRotatePoint(qvec0_inverse, tvec0, C0);
        // C0[0] = -C0[0];
        // C0[1] = -C0[1];
        // C0[2] = -C0[2];

        const T qvec1_inverse[4] = {T(qvec1[0]), T(-qvec1[1]), T(-qvec1[2]), T(-qvec1[3])};
        T C1[3];
        ceres::UnitQuaternionRotatePoint(qvec1_inverse, tvec1, C1);
        // C1[0] = -C1[0];
        // C1[1] = -C1[1];
        // C1[2] = -C1[2];

        residuals[0] = T(weight_x_) * (C1[0] - C0[0]);
        residuals[1] = T(weight_y_) * (C1[1] - C0[1]);
        residuals[2] = T(weight_z_) * (C1[2] - C0[2]);
        return true;
    }

private:
    const double weight_x_;
    const double weight_y_;
    const double weight_z_;
};

// struct only cost function

template <typename CameraModel>
class LargeFovStructBundleAdjustmentCostFunction {
public:
 explicit LargeFovStructBundleAdjustmentCostFunction(const Eigen::Vector4d& qvec, const Eigen::Vector3d& tvec,
                                                     const Eigen::Vector3d& bearing, const double f)
     : qw_(qvec(0)),
       qx_(qvec(1)),
       qy_(qvec(2)),
       qz_(qvec(3)),
       tx_(tvec(0)),
       ty_(tvec(1)),
       tz_(tvec(2)),
       observed_x_(bearing(0)),
       observed_y_(bearing(1)),
       observed_z_(bearing(2)),
       f_(f) {}

 static ceres::CostFunction* Create(const Eigen::Vector4d& qvec, const Eigen::Vector3d& tvec,
                                    const Eigen::Vector3d& bearing, const double f) {
     return (new ceres::AutoDiffCostFunction<LargeFovStructBundleAdjustmentCostFunction<CameraModel>,
                                             CameraModel::kNumResidual + 1, 3>(
         new LargeFovStructBundleAdjustmentCostFunction(qvec, tvec, bearing, f)));
    }

    template <typename T>
    bool operator()(const T* const point3D, T* residuals) const {
        const T qvec[4] = {T(qw_), T(qx_), T(qy_), T(qz_)};
        const T tvec[3] = {T(tx_), T(ty_), T(tz_)};


        // Rotate and translate.
        T projection[3];
        ceres::UnitQuaternionRotatePoint(qvec, point3D, projection);
        projection[0] += tvec[0];
        projection[1] += tvec[1];
        projection[2] += tvec[2];

        T radius_projection = T(
            ceres::sqrt(projection[0] * projection[0] + projection[1] * projection[1] + projection[2] * projection[2]));
        if (radius_projection > T(0)) {
            projection[0] /= radius_projection;
            projection[1] /= radius_projection;
            projection[2] /= radius_projection;
        }

        residuals[0] = f_ * (projection[0] - observed_x_);
        residuals[1] = f_ * (projection[1] - observed_y_);
        residuals[2] = f_ * (projection[2] - observed_z_);

        return true;
    }

private:
    const double qw_;
    const double qx_;
    const double qy_;
    const double qz_;
    const double tx_;
    const double ty_;
    const double tz_;
    const double observed_x_;
    const double observed_y_;
    const double observed_z_;
    const double f_;
};


template <typename CameraModel>
class StructBundleAdjustmentCostFunction {
public:
    explicit StructBundleAdjustmentCostFunction(
        const Eigen::Vector4d& qvec, const Eigen::Vector3d& tvec, const double* camera_params, 
        const size_t camera_params_count,
        const Eigen::Vector2d& point2D)
        : qw_(qvec(0)),
          qx_(qvec(1)),
          qy_(qvec(2)),
          qz_(qvec(3)),
          tx_(tvec(0)),
          ty_(tvec(1)),
          tz_(tvec(2)),
          camera_params_(camera_params),
          camera_params_count_(camera_params_count),
          observed_x_(point2D(0)),
          observed_y_(point2D(1)){}

    static ceres::CostFunction* Create(const Eigen::Vector4d& qvec, 
                                       const Eigen::Vector3d& tvec, 
                                       const double* camera_params, 
                                       const size_t camera_params_count,
                                       const Eigen::Vector2d& point2D
                                       ) {
        return (new ceres::AutoDiffCostFunction<StructBundleAdjustmentCostFunction<CameraModel>,
                                                CameraModel::kNumResidual, 3>(
            new StructBundleAdjustmentCostFunction(qvec, tvec, camera_params, camera_params_count, point2D)));
    }

    template <typename T>
    bool operator()(const T* const point3D, T* residuals) const {
        const T qvec[4] = {T(qw_), T(qx_), T(qy_), T(qz_)};
        const T tvec[3] = {T(tx_), T(ty_), T(tz_)};

        // Rotate and translate.
        T projection[3];
        ceres::UnitQuaternionRotatePoint(qvec, point3D, projection);
        projection[0] += tvec[0];
        projection[1] += tvec[1];
        projection[2] += tvec[2];

        if (projection[2] < T(std::numeric_limits<double>::epsilon())) {
            projection[2] = T(std::numeric_limits<double>::epsilon());
        }

        T camera_params[camera_params_count_];
        for(size_t i = 0; i< camera_params_count_; ++i){
            camera_params[i] = T(camera_params_[i]);
        }

        // Project to image plane.
        projection[0] /= projection[2];
        projection[1] /= projection[2];
        CameraModel::WorldToImage(camera_params, projection[0], projection[1], &residuals[0], &residuals[1]);

        residuals[0] = residuals[0] - observed_x_;
        residuals[1] = residuals[1] - observed_y_;

        return true;
    }

private:
    const double qw_;
    const double qx_;
    const double qy_;
    const double qz_;
    const double tx_;
    const double ty_;
    const double tz_;
    const double observed_x_;
    const double observed_y_;
    const double* camera_params_;
    const size_t camera_params_count_;
};



template <typename CameraModel>
class SphericalStructBundleAdjustmentCostFunction {
public:
    SphericalStructBundleAdjustmentCostFunction(const Eigen::Vector4d& qvec, const Eigen::Vector3d& tvec,
                                                const Eigen::Vector3d& bearing, const double f)
        : qw_(qvec(0)),
          qx_(qvec(1)),
          qy_(qvec(2)),
          qz_(qvec(3)),
          tx_(tvec(0)),
          ty_(tvec(1)),
          tz_(tvec(2)),
          observed_x_(bearing(0)),
          observed_y_(bearing(1)),
          observed_z_(bearing(2)),
          f_(f){}

    static ceres::CostFunction* Create(const Eigen::Vector4d& qvec, const Eigen::Vector3d& tvec,
                                       const Eigen::Vector3d& bearing, const double f) {
        return (new ceres::AutoDiffCostFunction<SphericalStructBundleAdjustmentCostFunction<CameraModel>,
                                                CameraModel::kNumResidual, 3>(
            new SphericalStructBundleAdjustmentCostFunction(qvec, tvec, bearing, f)));
    }

    template <typename T>
    bool operator()(const T* const point3D, T* residuals) const {
        const T qvec[4] = {T(qw_), T(qx_), T(qy_), T(qz_)};

        // Rotate and translate.
        T projection[3];
        ceres::UnitQuaternionRotatePoint(qvec, point3D, projection);
        projection[0] += T(tx_);
        projection[1] += T(ty_);
        projection[2] += T(tz_);

        T radius_projection = T(
            ceres::sqrt(projection[0] * projection[0] + projection[1] * projection[1] + projection[2] * projection[2]));
        if (radius_projection > T(0)) {
            projection[0] /= radius_projection;
            projection[1] /= radius_projection;
            projection[2] /= radius_projection;
        }

        residuals[0] = f_ * (projection[0] - observed_x_);
        residuals[1] = f_ * (projection[1] - observed_y_);
        residuals[2] = f_ * (projection[2] - observed_z_);
        return true;
    }

private:
    const double qw_;
    const double qx_;
    const double qy_;
    const double qz_;
    const double tx_;
    const double ty_;
    const double tz_;
    const double observed_x_;
    const double observed_y_;
    const double observed_z_;
    const double f_;
};

class ComputeUndistortionFunction : public ceres::SizedCostFunction<2, 4, 2> {
public:
   
    virtual bool Evaluate(double const* const* parameters, double* residuals, double ** jacobians) const{
        if(!jacobians){
            ComputeDistortionValueAndJacobian(parameters[0], parameters[1], residuals, NULL);
        }
        else{
            ComputeDistortionValueAndJacobian(parameters[0], parameters[1], residuals, jacobians);
        }
        return true;
    }
};



// Large Fov camera rig auto diff 
template <typename CameraModel>
class LargeFovRigBundleAdjustmentCostFunctionAuto {
public:
    explicit LargeFovRigBundleAdjustmentCostFunctionAuto(const Eigen::Vector2d& point2D, const double f)
        : observed_x_(point2D(0)), observed_y_(point2D(1)), f_(f) {
            compute_undistortion.reset(
            new ceres::CostFunctionToFunctor<2, 4, 2>(new ComputeUndistortionFunction()));
        }

    static ceres::CostFunction* Create(const Eigen::Vector2d& point2D, const double f) {
        return (new ceres::AutoDiffCostFunction<LargeFovRigBundleAdjustmentCostFunctionAuto<CameraModel>,
                                                CameraModel::kNumResidual+1, 4, 3, 4, 3, 3, 8>(
            new LargeFovRigBundleAdjustmentCostFunctionAuto(point2D, f)));
    }

    template <typename T>
    bool operator()(const T* const rig_qvec, const T* const rig_tvec, const T* const rel_qvec, const T* const rel_tvec,
                    const T* const point3D, const T* const camera_params, T* residuals) const {
        // Concatenate rotations.
        T qvec[4];
        ceres::QuaternionProduct(rel_qvec, rig_qvec, qvec);

        // Concatenate translations.
        T tvec[3];
        ceres::UnitQuaternionRotatePoint(rel_qvec, rig_tvec, tvec);
        tvec[0] += rel_tvec[0];
        tvec[1] += rel_tvec[1];
        tvec[2] += rel_tvec[2];

        // Rotate and translate.
        T projection[3];
        ceres::UnitQuaternionRotatePoint(qvec, point3D, projection);
        projection[0] += tvec[0];
        projection[1] += tvec[1];
        projection[2] += tvec[2];

        T radius_projection = T(
            ceres::sqrt(projection[0] * projection[0] + projection[1] * projection[1] + projection[2] * projection[2]));
        if (radius_projection > T(0)) {
            projection[0] /= radius_projection;
            projection[1] /= radius_projection;
            projection[2] /= radius_projection;
        }

        T u, v, w;
        CameraModel::ImageToBearing(camera_params,T(observed_x_),T(observed_y_),&u,&v,&w);        


        residuals[0] = f_ * (projection[0] - u);
        residuals[1] = f_ * (projection[1] - v);
        residuals[2] = f_ * (projection[2] - w);

        return true;
    }

private:
    const double observed_x_;
    const double observed_y_;
    const double f_;
    std::unique_ptr<ceres::CostFunctionToFunctor<2, 4, 2> > compute_undistortion;

};

// Large Fov camera rig with constant pose

template <typename CameraModel>
class LargeFovRigBundleAdjustmentConstantPoseCostFunctionAuto {
public:
    explicit LargeFovRigBundleAdjustmentConstantPoseCostFunctionAuto(const Eigen::Vector4d& qvec,
                                                                 const Eigen::Vector3d& tvec,
                                                                 const Eigen::Vector2d& point2D, const double f)
        : qw_(qvec(0)),
          qx_(qvec(1)),
          qy_(qvec(2)),
          qz_(qvec(3)),
          tx_(tvec(0)),
          ty_(tvec(1)),
          tz_(tvec(2)),
          observed_x_(point2D(0)),
          observed_y_(point2D(1)),
          f_(f) {
            compute_undistortion.reset(
            new ceres::CostFunctionToFunctor<2, 4, 2>(new ComputeUndistortionFunction()));
          }

    static ceres::CostFunction* Create(const Eigen::Vector4d& qvec, const Eigen::Vector3d& tvec,
                                       const Eigen::Vector2d& point2D, const double f) {
        return (new ceres::AutoDiffCostFunction<LargeFovRigBundleAdjustmentConstantPoseCostFunctionAuto<CameraModel>,
                                                CameraModel::kNumResidual + 1, 4, 3, 3, 8>(
            new LargeFovRigBundleAdjustmentConstantPoseCostFunctionAuto(qvec, tvec, point2D, f)));
    }

    template <typename T>
    bool operator()(const T* const rel_qvec, const T* const rel_tvec, const T* const point3D, 
                    const T* const camera_params, T* residuals) const {
        const T rig_qvec[4] = {T(qw_), T(qx_), T(qy_), T(qz_)};
        const T rig_tvec[3] = {T(tx_), T(ty_), T(tz_)};

        // Concatenate rotations.
        T qvec[4];
        ceres::QuaternionProduct(rel_qvec, rig_qvec, qvec);

        // Concatenate translations.
        T tvec[3];
        ceres::UnitQuaternionRotatePoint(rel_qvec, rig_tvec, tvec);
        tvec[0] += rel_tvec[0];
        tvec[1] += rel_tvec[1];
        tvec[2] += rel_tvec[2];

        // Rotate and translate.
        T projection[3];
        ceres::UnitQuaternionRotatePoint(qvec, point3D, projection);
        projection[0] += tvec[0];
        projection[1] += tvec[1];
        projection[2] += tvec[2];

        T radius_projection = T(
            ceres::sqrt(projection[0] * projection[0] + projection[1] * projection[1] + projection[2] * projection[2]));
        if (radius_projection > T(0)) {
            projection[0] /= radius_projection;
            projection[1] /= radius_projection;
            projection[2] /= radius_projection;
        }

        T u, v, w;
        CameraModel::ImageToBearing(camera_params,T(observed_x_),T(observed_y_), &u, &v, &w);  

        // const T f1 = camera_params[0];
        // const T f2 = camera_params[1];
        // const T c1 = camera_params[2];
        // const T c2 = camera_params[3];

        // // Lift points to normalized plane
        // T x = (observed_x_ - c1) / f1;
        // T y = (observed_y_ - c2) / f2;
        
        // T params[4] = {camera_params[4],camera_params[5],camera_params[6],camera_params[7]};
        // T coord[2] =  {x, y};
        
        // T undistort_coord[2];
        // (*compute_undistortion)(params,coord,undistort_coord);

        // T u = undistort_coord[0];
        // T v = undistort_coord[1];
        // T w = T(1.0);
        // T norm = ceres::sqrt(u*u + v*v + w*w);

        // if(norm > T(0)){
        //     u = u /norm;
        //     v = v /norm;
        //     w = w /norm;
        // }
        // else{
        //     u = T(0);
        //     v = T(0);
        //     w = T(0);
        // }

        residuals[0] = f_ * (projection[0] - u);
        residuals[1] = f_ * (projection[1] - v);
        residuals[2] = f_ * (projection[2] - w);

        return true;
    }

private:
    const double qw_;
    const double qx_;
    const double qy_;
    const double qz_;
    const double tx_;
    const double ty_;
    const double tz_;
    const double observed_x_;
    const double observed_y_;
    const double f_;
    std::unique_ptr<ceres::CostFunctionToFunctor<2, 4, 2> > compute_undistortion;
};

class RigScaleCostFunction {
    public:
    explicit RigScaleCostFunction(const double f,
        const Eigen::Vector4d& qvec, const Eigen::Vector3d& tvec,
        const Eigen::Vector4d& local_qvec, const Eigen::Vector3d& local_tvec,
        const Eigen::Vector3d& point3D, const Eigen::Vector3d& observed)
        : f_(f),
          qw_(qvec(0)),
          qx_(qvec(1)),
          qy_(qvec(2)),
          qz_(qvec(3)),
          tx_(tvec(0)),
          ty_(tvec(1)),
          tz_(tvec(2)),
          local_qw_(local_qvec(0)),
          local_qx_(local_qvec(1)),
          local_qy_(local_qvec(2)),
          local_qz_(local_qvec(3)),
          local_tx_(local_tvec(0)),
          local_ty_(local_tvec(1)),
          local_tz_(local_tvec(2)),
          observed_x_(observed(0)),
          observed_y_(observed(1)),
          observed_z_(observed(2)),
          point3D_x_(point3D(0)),
          point3D_y_(point3D(1)),
          point3D_z_(point3D(2)) {}

    static ceres::CostFunction* Create(const double f,
                                       const Eigen::Vector4d& qvec, 
                                       const Eigen::Vector3d& tvec,
                                       const Eigen::Vector4d& local_qvec, 
                                       const Eigen::Vector3d& local_tvec,
                                       const Eigen::Vector3d& point3D,
                                       const Eigen::Vector3d& observed) {
        return (new ceres::AutoDiffCostFunction<RigScaleCostFunction, 3, 1>(
            new RigScaleCostFunction(f, qvec, tvec, local_qvec, local_tvec, point3D, observed)));
    }

    template <typename T>
    bool operator()(const T* const scale, T* residuals) const {
        const T rig_qvec[4] = {T(qw_), T(qx_), T(qy_), T(qz_)};
        const T rig_tvec[3] = {T(tx_), T(ty_), T(tz_)};
        const T local_qvec[4] = {T(local_qw_), T(local_qx_), T(local_qy_), T(local_qz_)};
        const T local_tvec[3] = {T(local_tx_), T(local_ty_), T(local_tz_)};

        // Concatenate rotations.
        T qvec[4];
        ceres::QuaternionProduct(local_qvec, rig_qvec, qvec);

        // Concatenate translations.
        T tvec[3];
        ceres::UnitQuaternionRotatePoint(local_qvec, rig_tvec, tvec);
        tvec[0] = tvec[0] * *scale + local_tvec[0];
        tvec[1] = tvec[1] * *scale + local_tvec[1];
        tvec[2] = tvec[2] * *scale + local_tvec[2];

        T point3D[3] = {T(point3D_x_), T(point3D_y_), T(point3D_z_)};
        point3D[0] *= *scale;
        point3D[1] *= *scale;
        point3D[2] *= *scale;

        // Rotate and translate.
        T projection[3];
        ceres::UnitQuaternionRotatePoint(qvec, point3D, projection);
        projection[0] += tvec[0];
        projection[1] += tvec[1];
        projection[2] += tvec[2];

        T radius_projection = T(
            ceres::sqrt(projection[0] * projection[0] + projection[1] * projection[1] + projection[2] * projection[2]));
        if (radius_projection > T(0)) {
            projection[0] /= radius_projection;
            projection[1] /= radius_projection;
            projection[2] /= radius_projection;
        }

        residuals[0] = f_ * (projection[0] - observed_x_);
        residuals[1] = f_ * (projection[1] - observed_y_);
        residuals[2] = f_ * (projection[2] - observed_z_);
        return true;
    }

private:
    const double f_;
    const double qw_;
    const double qx_;
    const double qy_;
    const double qz_;
    const double tx_;
    const double ty_;
    const double tz_;
    const double local_qw_;
    const double local_qx_;
    const double local_qy_;
    const double local_qz_;
    const double local_tx_;
    const double local_ty_;
    const double local_tz_;
    const double observed_x_;
    const double observed_y_;
    const double observed_z_;
    const double point3D_x_;
    const double point3D_y_;
    const double point3D_z_;
};



// Cost function for refining two-view geometry based on the Sampson-Error.
//
// First pose is assumed to be located at the origin with 0 rotation. Second
// pose is assumed to be on the unit sphere around the first pose, i.e. the
// pose of the second camera is parameterized by a 3D rotation and a
// 3D translation with unit norm. `tvec` is therefore over-parameterized as is
// and should be down-projected using `HomogeneousVectorParameterization`.
class RelativePoseCostFunction {
public:
    RelativePoseCostFunction(const Eigen::Vector2d& x1, const Eigen::Vector2d& x2)
        : x1_(x1(0)), y1_(x1(1)), x2_(x2(0)), y2_(x2(1)) {}

    static ceres::CostFunction* Create(const Eigen::Vector2d& x1, const Eigen::Vector2d& x2) {
        return (
            new ceres::AutoDiffCostFunction<RelativePoseCostFunction, 1, 4, 3>(new RelativePoseCostFunction(x1, x2)));
    }

    template <typename T>
    bool operator()(const T* const qvec, const T* const tvec, T* residuals) const {
        Eigen::Matrix<T, 3, 3, Eigen::RowMajor> R;
        ceres::QuaternionToRotation(qvec, R.data());

        // Matrix representation of the cross product t x R.
        Eigen::Matrix<T, 3, 3> t_x;
        t_x << T(0), -tvec[2], tvec[1], tvec[2], T(0), -tvec[0], -tvec[1], tvec[0], T(0);

        // Essential matrix.
        const Eigen::Matrix<T, 3, 3> E = t_x * R;

        // Homogeneous image coordinates.
        const Eigen::Matrix<T, 3, 1> x1_h(T(x1_), T(y1_), T(1));
        const Eigen::Matrix<T, 3, 1> x2_h(T(x2_), T(y2_), T(1));

        // Squared sampson error.
        const Eigen::Matrix<T, 3, 1> Ex1 = E * x1_h;
        const Eigen::Matrix<T, 3, 1> Etx2 = E.transpose() * x2_h;
        const T x2tEx1 = x2_h.transpose() * Ex1;
        residuals[0] = x2tEx1 * x2tEx1 / (Ex1(0) * Ex1(0) + Ex1(1) * Ex1(1) + Etx2(0) * Etx2(0) + Etx2(1) * Etx2(1));

        return true;
    }

private:
    const double x1_;
    const double y1_;
    const double x2_;
    const double y2_;
};


class ICPRelativePoseCostFunction{
public:
    ICPRelativePoseCostFunction(Eigen::Matrix4d X, Eigen::Matrix<double, 6, 6> info, double conf = 1.0):
    X_(X), info_(info), conf_(conf)
    {
        info_ *= conf_;
        sqrt_info_ = Eigen::LLT<Eigen::Matrix<double, 6, 6>>(info_).matrixL().transpose();
        Eigen::Quaterniond q(X_.block<3,3>(0,0).transpose());
        q.normalize();
        inv_X_Q_<<q.w(), q.x(), q.y(), q.z();
        inv_X_T_ = - X_.block<3,3>(0,0).transpose() * X_.block<3,1>(0,3);
    }

    static ceres::CostFunction* Create(Eigen::Matrix4d X, Eigen::Matrix<double, 6, 6> info, double conf = 1.0) {
        return new ceres::AutoDiffCostFunction<ICPRelativePoseCostFunction, 6, 4, 3, 4, 3>(
                new ICPRelativePoseCostFunction(X, info, conf));
    }


    //residuals = X.inverse() * dst * src.inverse()
    template <typename T>
    bool operator()(const T* const src_q, const T* const src_t, const T* const dst_q, const T* const dst_t,
                    T* residuals) const {
        Eigen::Map<Eigen::Matrix<T,6,1>> residuals_vec(residuals);

        Eigen::Matrix<T,4,1> inv_X_Q_mul_dst_q, zeta_q;
        Eigen::Matrix<T,4,1> inv_X_Q = inv_X_Q_.template cast<T>();
        T inv_src_q[4] = {T(src_q[0]), -T(src_q[1]), -T(src_q[2]), -T(src_q[3])};

        ceres::QuaternionProduct(inv_X_Q.data(), dst_q, inv_X_Q_mul_dst_q.data());
        ceres::QuaternionProduct(inv_X_Q_mul_dst_q.data(), inv_src_q, zeta_q.data());

        residuals_vec.template head<3>()= T(2.0) * zeta_q.template tail<3>();

        Eigen::Matrix<T,3,1> inv_X_Q_mul_dst_t, zeta_q_mul_src_t;

        ceres::QuaternionRotatePoint(inv_X_Q.data(), dst_t, inv_X_Q_mul_dst_t.data());
        ceres::QuaternionRotatePoint(zeta_q.data(), src_t, zeta_q_mul_src_t.data());
        Eigen::Matrix<T,3,1> inv_X_T = inv_X_T_.template cast<T>();

        residuals_vec.template tail<3>() = inv_X_T + inv_X_Q_mul_dst_t - zeta_q_mul_src_t;

        residuals_vec.applyOnTheLeft(sqrt_info_.template cast<T>());
        return true;
    }

    Eigen::Matrix4d X_;
    Eigen::Vector4d inv_X_Q_;
    Eigen::Vector3d inv_X_T_;
    Eigen::Matrix<double, 6, 6> info_;
    Eigen::Matrix<double, 6, 6> sqrt_info_;
    double conf_;
};


class GravityCostFunction{
public:
    GravityCostFunction(Eigen::Vector3d world_G, Eigen::Vector3d cur_G, Eigen::Matrix3d info, double relax_th,
                        double weight = 1.0) :
            world_G_(world_G), cur_G_(cur_G), info_(info), relax_th_(relax_th), weight_(weight){

        sqrt_info_ = Eigen::LLT<Eigen::Matrix3d>(weight_ * info_).matrixL().transpose();
    }

    static ceres::CostFunction *
    Create(Eigen::Vector3d world_G, Eigen::Vector3d cur_G, Eigen::Matrix3d info, double relax_th, double weight = 1.0) {
        return new ceres::AutoDiffCostFunction<GravityCostFunction, 3, 4>(
                new GravityCostFunction(world_G, cur_G, info, relax_th, weight));
    }


    //residuals = cur*world_g - cur_g;
    template <typename T>
    bool operator()(const T* const cur_q, T* residuals) const {
        Eigen::Map<Eigen::Matrix<T,3,1>> residuals_vec(residuals);
        T world_G[3] = {T(world_G_[0]), T(world_G_[1]), T(world_G_[2])};
        Eigen::Matrix<T,3,1> R_WG;
        ceres::QuaternionRotatePoint(cur_q, world_G, R_WG.data());
        residuals_vec = R_WG - cur_G_.template cast<T>();

        if (relax_th_ > 0) {
            if(residuals_vec.norm()<T(relax_th_)) residuals_vec.setZero();
        }
        residuals_vec.applyOnTheLeft(sqrt_info_.template cast<T>());
        return true;
    }


    Eigen::Vector3d world_G_;
    Eigen::Vector3d cur_G_;

    Eigen::Matrix3d info_;
    Eigen::Matrix3d sqrt_info_;
    double weight_;
    double relax_th_;

};


class TimeDomainSmoothingCostFunction {
public:
    TimeDomainSmoothingCostFunction(double prev_time, double next_time, double weight = 1.0)
    : weight_(weight),
      inv_prev_time_(1.0 / next_time),
      inv_next_time_(1.0 / prev_time),
      prev_weight_(next_time / (prev_time + next_time)),
      next_weight_(prev_time / (prev_time + next_time))
    {
    }

    static ceres::CostFunction *
    Create(double prev_weight, double next_weight, double weight = 1.0) {
        return new ceres::AutoDiffCostFunction<TimeDomainSmoothingCostFunction, 3, 4, 3, 4, 3, 4, 3>(
            new TimeDomainSmoothingCostFunction(prev_weight, next_weight, weight));
    }

    template <typename T>
    bool operator() (
        const T* const curr_q,
        const T* const curr_t,
        const T* const prev_q,
        const T* const prev_t,
        const T* const next_q,
        const T* const next_t,
        T* residuals
    ) const {
        const T prev_q_inverse[4] = { prev_q[0], -prev_q[1], -prev_q[2], -prev_q[3]};
        const T curr_q_inverse[4] = { curr_q[0], -curr_q[1], -curr_q[2], -curr_q[3]};
        const T next_q_inverse[4] = { next_q[0], -next_q[1], -next_q[2], -next_q[3]};
        T prev_p[3];
        T curr_p[3];
        T next_p[3];
        ceres::QuaternionRotatePoint(prev_q_inverse, prev_t, prev_p);
        ceres::QuaternionRotatePoint(curr_q_inverse, curr_t, curr_p);
        ceres::QuaternionRotatePoint(next_q_inverse, next_t, next_p);
        const T prev_velocity[3] = {
            (curr_p[0] - prev_p[0]) * inv_prev_time_,
            (curr_p[1] - prev_p[1]) * inv_prev_time_,
            (curr_p[2] - prev_p[2]) * inv_prev_time_
        };
        const T next_velocity[3] = {
            (next_p[0] - curr_p[0]) * inv_next_time_,
            (next_p[1] - curr_p[1]) * inv_next_time_,
            (next_p[2] - curr_p[2]) * inv_next_time_
        };

        residuals[0] = T(weight_) * (next_velocity[0] - prev_velocity[0]);
        residuals[1] = T(weight_) * (next_velocity[1] - prev_velocity[1]);
        residuals[2] = T(weight_) * (next_velocity[2] - prev_velocity[2]);

        return true;
    }

    const double inv_prev_time_;
    const double inv_next_time_;
    const double prev_weight_;
    const double next_weight_;
    const double weight_;
};


class PriorRelativePoseCostFunction {
public:
    PriorRelativePoseCostFunction(const Eigen::Vector4d& relative_q, const Eigen::Vector3d& relative_t,
                                  const double weight = 1.0)
        : relative_qw_(relative_q(0)),
          relative_qx_(relative_q(1)),
          relative_qy_(relative_q(2)),
          relative_qz_(relative_q(3)),
          relative_tx_(relative_t(0)),
          relative_ty_(relative_t(1)),
          relative_tz_(relative_t(2)),
          weight_(weight) {}

    static ceres::CostFunction* Create(const Eigen::Vector4d& relative_q, const Eigen::Vector3d& relative_t,
                                       const double weight = 1.0) {
        return new ceres::AutoDiffCostFunction<PriorRelativePoseCostFunction, 6, 4, 3, 4, 3>(
            new PriorRelativePoseCostFunction(relative_q, relative_t, weight));
    }

    template <typename T>
    bool operator()(const T* const qvec1, const T* const tvec1, const T* const qvec2, const T* const tvec2,
                    T* residuals) const {
        T q12_estimate[4];
        T q1_inverse[4] = {T(qvec1[0]), T(-qvec1[1]), T(-qvec1[2]), T(-qvec1[3])};

        ceres::QuaternionProduct(qvec2, q1_inverse, q12_estimate);

        T t12_estimate[3];
        ceres::QuaternionRotatePoint(q12_estimate, tvec1, t12_estimate);

        t12_estimate[0] = tvec2[0] - t12_estimate[0];
        t12_estimate[1] = tvec2[1] - t12_estimate[1];
        t12_estimate[2] = tvec2[2] - t12_estimate[2];

        T relative_q_diff[4];
        T q12_estimate_inverse[4] = {q12_estimate[0], -q12_estimate[1], -q12_estimate[2], -q12_estimate[3]};
        T q12_prior[4] = {T(relative_qw_), T(relative_qx_), T(relative_qy_), T(relative_qz_)};

        ceres::QuaternionProduct(q12_prior, q12_estimate_inverse, relative_q_diff);

        residuals[0] = T(weight_) * T(2.0) * relative_q_diff[1];
        residuals[1] = T(weight_) * T(2.0) * relative_q_diff[2];
        residuals[2] = T(weight_) * T(2.0) * relative_q_diff[3];
        residuals[3] = T(weight_) * (t12_estimate[0] - relative_tx_);
        residuals[4] = T(weight_) * (t12_estimate[1] - relative_ty_);
        residuals[5] = T(weight_) * (t12_estimate[2] - relative_tz_);

        return true;
    }

private:
    const double relative_qw_;
    const double relative_qx_;
    const double relative_qy_;
    const double relative_qz_;
    const double relative_tx_;
    const double relative_ty_;
    const double relative_tz_;
    const double weight_;
};

class PriorRelativeTranslationCostFunction {
public:
    PriorRelativeTranslationCostFunction(const Eigen::Vector4d& relative_q, const Eigen::Vector3d& relative_t,
                                         const double weight = 1.0)
        : relative_qw_(relative_q(0)),
          relative_qx_(relative_q(1)),
          relative_qy_(relative_q(2)),
          relative_qz_(relative_q(3)),
          relative_tx_(relative_t(0)),
          relative_ty_(relative_t(1)),
          relative_tz_(relative_t(2)),
          weight_(weight),
          relative_dist_(
              sqrt(relative_tx_ * relative_tx_ + relative_ty_ * relative_ty_ + relative_tz_ * relative_tz_)) {}

    static ceres::CostFunction* Create(const Eigen::Vector4d& relative_q, const Eigen::Vector3d& relative_t,
                                       const double weight = 1.0) {
        return new ceres::AutoDiffCostFunction<PriorRelativeTranslationCostFunction, 3, 4, 3, 4, 3>(
            new PriorRelativeTranslationCostFunction(relative_q, relative_t, weight));
    }

    template <typename T>
    bool operator()(const T* const qvec1, const T* const tvec1, const T* const qvec2, const T* const tvec2,
                    T* residuals) const {
        T q12_estimate[4];
        T q1_inverse[4] = {T(qvec1[0]), T(-qvec1[1]), T(-qvec1[2]), T(-qvec1[3])};

        ceres::QuaternionProduct(qvec2, q1_inverse, q12_estimate);

        T t12_estimate[3];
        ceres::QuaternionRotatePoint(q12_estimate, tvec1, t12_estimate);

        t12_estimate[0] = tvec2[0] - t12_estimate[0];
        t12_estimate[1] = tvec2[1] - t12_estimate[1];
        t12_estimate[2] = tvec2[2] - t12_estimate[2];

        T dist_estimate = ceres::sqrt(t12_estimate[0] * t12_estimate[0] + t12_estimate[1] * t12_estimate[1] +
                                      t12_estimate[2] * t12_estimate[2]);

        residuals[0] = T(weight_) * (t12_estimate[0] - relative_tx_);
        residuals[1] = T(weight_) * (t12_estimate[1] - relative_ty_);
        residuals[2] = T(weight_) * (t12_estimate[2] - relative_tz_);

        return true;
    }

private:
    const double relative_qw_;
    const double relative_qx_;
    const double relative_qy_;
    const double relative_qz_;
    const double relative_tx_;
    const double relative_ty_;
    const double relative_tz_;
    const double weight_;
    const double relative_dist_;
};

class PriorRelativeDistanceCostFunction {
public:
    PriorRelativeDistanceCostFunction(const Eigen::Vector4d& relative_q, const Eigen::Vector3d& relative_t,
                                      const double weight = 1.0)
        : relative_qw_(relative_q(0)),
          relative_qx_(relative_q(1)),
          relative_qy_(relative_q(2)),
          relative_qz_(relative_q(3)),
          relative_tx_(relative_t(0)),
          relative_ty_(relative_t(1)),
          relative_tz_(relative_t(2)),
          weight_(weight),
          relative_dist_(
              sqrt(relative_tx_ * relative_tx_ + relative_ty_ * relative_ty_ + relative_tz_ * relative_tz_)) {}

    static ceres::CostFunction* Create(const Eigen::Vector4d& relative_q, const Eigen::Vector3d& relative_t,
                                       const double weight = 1.0) {
        return new ceres::AutoDiffCostFunction<PriorRelativeDistanceCostFunction, 1, 4, 3, 4, 3>(
            new PriorRelativeDistanceCostFunction(relative_q, relative_t, weight));
    }

    template <typename T>
    bool operator()(const T* const qvec1, const T* const tvec1, const T* const qvec2, const T* const tvec2,
                    T* residuals) const {
        T q12_estimate[4];
        T q1_inverse[4] = {T(qvec1[0]), T(-qvec1[1]), T(-qvec1[2]), T(-qvec1[3])};

        ceres::QuaternionProduct(qvec2, q1_inverse, q12_estimate);

        T t12_estimate[3];
        ceres::QuaternionRotatePoint(q12_estimate, tvec1, t12_estimate);

        t12_estimate[0] = tvec2[0] - t12_estimate[0];
        t12_estimate[1] = tvec2[1] - t12_estimate[1];
        t12_estimate[2] = tvec2[2] - t12_estimate[2];

        T dist_estimate = ceres::sqrt(t12_estimate[0] * t12_estimate[0] + t12_estimate[1] * t12_estimate[1] +
                                      t12_estimate[2] * t12_estimate[2]);

        residuals[0] = T(weight_) * (dist_estimate - relative_dist_);
        return true;
    }

private:
    const double relative_qw_;
    const double relative_qx_;
    const double relative_qy_;
    const double relative_qz_;
    const double relative_tx_;
    const double relative_ty_;
    const double relative_tz_;
    const double weight_;
    const double relative_dist_;
};


// prior absolute location cost function, used if gps prior is available

class PriorAbsoluteLocationCostFunction {
public:
    PriorAbsoluteLocationCostFunction(const Eigen::Vector3d& prior_c, const double weight_x = 1.0,
                                      const double weight_y = 1.0, const double weight_z = 1.0)
        : prior_x_(prior_c(0)),
          prior_y_(prior_c(1)),
          prior_z_(prior_c(2)),
          weight_x_(weight_x),
          weight_y_(weight_y),
          weight_z_(weight_z)
        {}

    static ceres::CostFunction* Create(const Eigen::Vector3d& prior_c, const double weight_x = 1.0,
                                       const double weight_y = 1.0, const double weight_z = 1.0) {
        return new ceres::AutoDiffCostFunction<PriorAbsoluteLocationCostFunction, 3, 4, 3>(
            new PriorAbsoluteLocationCostFunction(prior_c, weight_x, weight_y, weight_z));
    }

    template <typename T>
    bool operator()(const T* const qvec, const T* const tvec, T* residuals) const {

        T q_inverse[4] = {qvec[0], -qvec[1], -qvec[2], -qvec[3]};
        
        T C[3];
        ceres::QuaternionRotatePoint(q_inverse, tvec, C);

        C[0] = - C[0];
        C[1] = - C[1];
        C[2] = - C[2];

        residuals[0] = T(weight_x_) * (C[0] - prior_x_);
        residuals[1] = T(weight_y_) * (C[1] - prior_y_);
        residuals[2] = T(weight_z_) * (C[2] - prior_z_);
        return true;
    }
private:
    const double prior_x_;
    const double prior_y_;
    const double prior_z_;
    const double weight_x_;
    const double weight_y_;
    const double weight_z_;
};

// prior absolute location cost function, used if gps prior is available but the altitude is not accurate

class PriorAbsoluteLocationOnPlaneCostFunction {
public:
    PriorAbsoluteLocationOnPlaneCostFunction(const Eigen::Vector3d& prior_c, const Eigen::Vector4d& plane,
                                             const double weight = 1.0)
        : prior_x_(prior_c(0)),
          prior_y_(prior_c(1)),
          prior_z_(prior_c(2)),
          a_(plane(0)),
          b_(plane(1)),
          c_(plane(2)),
          d_(plane(3)),
          weight_(weight) {}

    static ceres::CostFunction* Create(const Eigen::Vector3d& prior_c, const Eigen::Vector4d& plane,
                                       const double weight = 1.0) {
        return new ceres::AutoDiffCostFunction<PriorAbsoluteLocationOnPlaneCostFunction, 3, 4, 3>(
            new PriorAbsoluteLocationOnPlaneCostFunction(prior_c, plane, weight));
    }

    template <typename T>
    bool operator()(const T* const qvec, const T* const tvec, T* residuals) const {
        T q_inverse[4] = {qvec[0], -qvec[1], -qvec[2], -qvec[3]};

        T C[3];
        ceres::QuaternionRotatePoint(q_inverse, tvec, C);

        C[0] = -C[0];
        C[1] = -C[1];
        C[2] = -C[2];

        T Projection_C[3];

        T h = C[0] * a_ + C[1] * b_ + C[2] * c_ + d_;

        Projection_C[0] = C[0] - h * a_;
        Projection_C[1] = C[1] - h * b_;
        Projection_C[2] = C[2] - h * c_;

        residuals[0] = T(weight_) * (Projection_C[0] - prior_x_);
        residuals[1] = T(weight_) * (Projection_C[1] - prior_y_);
        residuals[2] = T(weight_) * (Projection_C[2] - prior_z_);
        return true;
    }

private:
    const double prior_x_;
    const double prior_y_;
    const double prior_z_;
    const double a_;
    const double b_;
    const double c_;
    const double d_;
    const double weight_;
};

class PoseGraphSE3ErrorTerm {
public:
    PoseGraphSE3ErrorTerm(const Eigen::Quaterniond& relative_q_measured, const Eigen::Vector3d& relative_t_measured, double weight = 1.0)
        : relative_q_measured_(relative_q_measured), relative_t_measured_(relative_t_measured), weight_(weight){};

    template <typename T>
    bool operator()(const T* const t1_ptr, const T* const q1_ptr, const T* const t2_ptr, const T* const q2_ptr,
                    T* residuals_ptr) const {
        Eigen::Map<const Eigen::Matrix<T, 3, 1> > t1(t1_ptr);
        Eigen::Quaternion<T> q1(q1_ptr[0], q1_ptr[1], q1_ptr[2], q1_ptr[3]);
        Eigen::Map<const Eigen::Matrix<T, 3, 1> > t2(t2_ptr);
        Eigen::Quaternion<T> q2(q2_ptr[0], q2_ptr[1], q2_ptr[2], q2_ptr[3]);

        // Compute the relative transform between the two frames.
        Eigen::Quaternion<T> q21_estimated = q1 * q2.conjugate();

        // Compute the displacement between the two frames.
        Eigen::Matrix<T, 3, 1> t21_estimated = t1 - q21_estimated * t2;

        Eigen::Quaternion<T> delta_q = relative_q_measured_.template cast<T>() * q21_estimated.conjugate();

        // Compute the residuals.
        // [position]     = [delta_t]
        // [orientation]  = [2 * delta_q(0:2)]
        Eigen::Map<Eigen::Matrix<T, 6, 1> > residuals(residuals_ptr);
        residuals.template block<3, 1>(0, 0) = T(weight_) * (t21_estimated - relative_t_measured_.template cast<T>());
        residuals.template block<3, 1>(3, 0) = T(weight_) * T(2.0) * delta_q.vec();

        return true;
    }

    static ceres::CostFunction* Create(const Eigen::Quaterniond& relative_q_measured,
                                       const Eigen::Vector3d& relative_t_measured,double weight = 1.0) {
        return new ceres::AutoDiffCostFunction<PoseGraphSE3ErrorTerm, 6, 3, 4, 3, 4>(
            new PoseGraphSE3ErrorTerm(relative_q_measured, relative_t_measured, weight));
    }

private:
    const Eigen::Quaterniond relative_q_measured_;
    const Eigen::Vector3d relative_t_measured_;
    double weight_;
};

class PoseGraphSIM3ErrorTerm {
public:
    PoseGraphSIM3ErrorTerm(const Eigen::Vector7d& pose, double weight = 1.0) : pose_(pose), weight_(weight) {}

    template <typename T>
    bool operator()(const T* const vec_spose0, const T* const vec_spose1, T* residuals) const {
        // S_ij = S_i^-1 * S_j
        // S_ij^-1 = S_j^-1 * Si
        // --> S_ij * S_ij^-1 should be Identity
        // --> S_ij * S_j^-1 * S_i
        T cur_spose[7];
        cur_spose[0] = T(pose_[0]);
        cur_spose[1] = T(pose_[1]);
        cur_spose[2] = T(pose_[2]);
        cur_spose[3] = T(pose_[3]);
        cur_spose[4] = T(pose_[4]);
        cur_spose[5] = T(pose_[5]);
        cur_spose[6] = T(pose_[6]);

        T spose0[7];
        spose0[0] = T(vec_spose0[0]);
        spose0[1] = T(vec_spose0[1]);
        spose0[2] = T(vec_spose0[2]);
        spose0[3] = T(vec_spose0[3]);
        spose0[4] = T(vec_spose0[4]);
        spose0[5] = T(vec_spose0[5]);
        spose0[6] = T(vec_spose0[6]);

        T spose1[7];
        spose1[0] = T(vec_spose1[0]);
        spose1[1] = T(vec_spose1[1]);
        spose1[2] = T(vec_spose1[2]);
        spose1[3] = T(vec_spose1[3]);
        spose1[4] = T(vec_spose1[4]);
        spose1[5] = T(vec_spose1[5]);
        spose1[6] = T(vec_spose1[6]);

        TSimilarityTransform3<T> Sij(cur_spose);
        TSimilarityTransform3<T> Si(spose0);
        TSimilarityTransform3<T> Sj(spose1);
        TSimilarityTransform3<T> Serr = Sij * Sj * Si.Inverse();
        Serr.GetLog(residuals);

        for(size_t i = 0; i<7; ++i){
            residuals[i] = residuals[i] * weight_;
        }

        return true;
    }

    static ceres::CostFunction* Create(Eigen::Vector7d& pose,double weight = 1.0) {
        return (new ceres::AutoDiffCostFunction<PoseGraphSIM3ErrorTerm, 7, 7, 7>(new PoseGraphSIM3ErrorTerm(pose,weight)));
    }

private:
    Eigen::Vector7d pose_;
    double weight_;
};

// cost function for plane constrain
class PlaneConstrainCostFunction {
public:
    PlaneConstrainCostFunction(const Eigen::Vector4d& plane, const double& baseline_distance, const int num_observation,
                               const double weight)
        : plane_(plane), baseline_distance_(baseline_distance), num_observation_(num_observation), weight_(weight) {}

    static ceres::CostFunction* Create(const Eigen::Vector4d& plane, const double& baseline_distance,
                                       const int num_observation, const double weight) {
        return (new ceres::AutoDiffCostFunction<PlaneConstrainCostFunction, 1, 4, 3>(
            new PlaneConstrainCostFunction(plane, baseline_distance, num_observation, weight)));
    }

    template <typename T>
    bool operator()(const T* const qvec, const T* const tvec, T* residuals) const {
        const T inverse_qvec[4] = {qvec[0], -qvec[1], -qvec[2], -qvec[3]};
        T center[3];

        const T minus_tvec[3] = {-tvec[0], -tvec[1], -tvec[2]};

        ceres::UnitQuaternionRotatePoint(inverse_qvec, minus_tvec, center);

        double normal = ceres::sqrt(plane_(0) * plane_(0) + plane_(1) * plane_(1) + plane_(2) * plane_(2));

        residuals[0] = T(num_observation_) *
                       (T(plane_(0)) * center[0] + T(plane_(1)) * center[1] + T(plane_(2)) * center[2] + T(plane_(3))) /
                       T(normal) / T(baseline_distance_) * weight_;
        return true;
    }

private:
    Eigen::Vector4d plane_;
    double baseline_distance_;
    int num_observation_;
    double weight_;
};

// cost function for relative distance between landmarks
class LandMarkDistanceCostFunction {
public:
    LandMarkDistanceCostFunction(const double distance, const int num_mappoints, const double weight)
        : distance_(distance), num_mappoints_(num_mappoints), weight_(weight) {}

    static ceres::CostFunction* Create(const double distance, const int num_mappoints, const double weight) {
        return (new ceres::AutoDiffCostFunction<LandMarkDistanceCostFunction, 1, 3, 3>(
            new LandMarkDistanceCostFunction(distance, num_mappoints, weight)));
    }

    template <typename T>
    bool operator()(const T* const point3D1, const T* const point3D2, T* residuals) const {
        const T estimated_distance = ceres::sqrt((point3D1[0] - point3D2[0]) * (point3D1[0] - point3D2[0]) +
                                                 (point3D1[1] - point3D2[1]) * (point3D1[1] - point3D2[1]) +
                                                 (point3D1[2] - point3D2[2]) * (point3D1[2] - point3D2[2]));
        residuals[0] = (estimated_distance - T(distance_)) * T(num_mappoints_) * T(weight_);

        return true;
    }

private:
    double distance_;
    int num_mappoints_;
    double weight_;
};

// prior absolute pose cost function
class PriorAbsolutePoseCostFunction {
public:
    PriorAbsolutePoseCostFunction(const Eigen::Vector4d& prior_q,
                                  const Eigen::Vector3d& prior_c,
                                  const double weight_q = 1.0,
                                  const double weight_x = 1.0,
                                  const double weight_y = 1.0,
                                  const double weight_z = 1.0)
        : prior_qw_(prior_q(0)),
          prior_qx_(prior_q(1)),
          prior_qy_(prior_q(2)),
          prior_qz_(prior_q(3)),
          prior_x_(prior_c(0)),
          prior_y_(prior_c(1)),
          prior_z_(prior_c(2)),
          weight_q_(weight_q),
          weight_x_(weight_x),
          weight_y_(weight_y),
          weight_z_(weight_z)
        {}

    static ceres::CostFunction* Create(const Eigen::Vector4d& prior_q,
                                       const Eigen::Vector3d& prior_c,
                                       const double weight_q = 1.0,
                                       const double weight_x = 1.0,
                                       const double weight_y = 1.0,
                                       const double weight_z = 1.0) {
        return new ceres::AutoDiffCostFunction<PriorAbsolutePoseCostFunction, 6, 4, 3>(
            new PriorAbsolutePoseCostFunction(prior_q, prior_c, weight_q, weight_x, weight_y, weight_z));
    }

    template <typename T>
    bool operator()(const T* const qvec, const T* const tvec, T* residuals) const {
        T q_prior[4] = {T(prior_qw_), T(prior_qx_), T(prior_qy_), T(prior_qz_)};
        T q_inverse[4] = {qvec[0], -qvec[1], -qvec[2], -qvec[3]};

        T relative_q_diff[4];
        ceres::QuaternionProduct(q_inverse, q_prior, relative_q_diff);

        T C[3];
        ceres::QuaternionRotatePoint(q_inverse, tvec, C);

        C[0] = - C[0];
        C[1] = - C[1];
        C[2] = - C[2];

        residuals[0] = T(weight_q_) * T(2.0) * relative_q_diff[1];
        residuals[1] = T(weight_q_) * T(2.0) * relative_q_diff[2];
        residuals[2] = T(weight_q_) * T(2.0) * relative_q_diff[3];
        residuals[3] = T(weight_x_) * (C[0] - prior_x_);
        residuals[4] = T(weight_y_) * (C[1] - prior_y_);
        residuals[5] = T(weight_z_) * (C[2] - prior_z_);

        return true;
    }
private:
    const double prior_qw_;
    const double prior_qx_;
    const double prior_qy_;
    const double prior_qz_;
    const double prior_x_;
    const double prior_y_;
    const double prior_z_;
    const double weight_q_;
    const double weight_x_;
    const double weight_y_;
    const double weight_z_;
};

// prior absolute pose cost function
class PriorAbsoluteLocalPoseCostFunction {
public:
    PriorAbsoluteLocalPoseCostFunction(const Eigen::Vector4d& prior_q,
                                        const Eigen::Vector3d& prior_c,
                                        const Eigen::Vector4d& local_qvec,
                                        const Eigen::Vector3d& local_tvec,
                                        const double weight_q = 1.0,
                                        const double weight_x = 1.0,
                                        const double weight_y = 1.0,
                                        const double weight_z = 1.0)
        : prior_qw_(prior_q(0)),
          prior_qx_(prior_q(1)),
          prior_qy_(prior_q(2)),
          prior_qz_(prior_q(3)),
          prior_x_(prior_c(0)),
          prior_y_(prior_c(1)),
          prior_z_(prior_c(2)),
          local_qw_(local_qvec(0)),
          local_qx_(local_qvec(1)),
          local_qy_(local_qvec(2)),
          local_qz_(local_qvec(3)),
          local_x_(local_tvec(0)),
          local_y_(local_tvec(1)),
          local_z_(local_tvec(2)),
          weight_q_(weight_q),
          weight_x_(weight_x),
          weight_y_(weight_y),
          weight_z_(weight_z)
        {}

    static ceres::CostFunction* Create(const Eigen::Vector4d& prior_q,
                                       const Eigen::Vector3d& prior_c,
                                       const Eigen::Vector4d& local_qvec,
                                       const Eigen::Vector3d& local_tvec,
                                       const double weight_q = 1.0,
                                       const double weight_x = 1.0,
                                       const double weight_y = 1.0,
                                       const double weight_z = 1.0) {
        return new ceres::AutoDiffCostFunction<PriorAbsoluteLocalPoseCostFunction, 6, 4, 3>(
            new PriorAbsoluteLocalPoseCostFunction(prior_q, prior_c, local_qvec, local_tvec, weight_q, weight_x, weight_y, weight_z));
    }

    template <typename T>
    bool operator()(const T* const rig_qvec, const T* const rig_tvec, T* residuals) const {
        T q_prior[4] = {T(prior_qw_), T(prior_qx_), T(prior_qy_), T(prior_qz_)};

        // Concatenate rotations.
        T qvec[4];
        T rel_qvec[4] = {T(local_qw_), T(local_qx_), T(local_qy_), T(local_qz_)};
        ceres::QuaternionProduct(rel_qvec, rig_qvec, qvec);

        // Concatenate translations.
        T tvec[3];
        T rel_tvec[3] = {T(local_x_), T(local_y_), T(local_z_)};
        ceres::UnitQuaternionRotatePoint(rel_qvec, rig_tvec, tvec);
        tvec[0] += rel_tvec[0];
        tvec[1] += rel_tvec[1];
        tvec[2] += rel_tvec[2];

        T q_inverse[4] = {qvec[0], -qvec[1], -qvec[2], -qvec[3]};

        T relative_q_diff[4];
        ceres::QuaternionProduct(q_inverse, q_prior, relative_q_diff);

        T C[3];
        ceres::QuaternionRotatePoint(q_inverse, tvec, C);

        C[0] = - C[0];
        C[1] = - C[1];
        C[2] = - C[2];

        residuals[0] = T(weight_q_) * T(2.0) * relative_q_diff[1];
        residuals[1] = T(weight_q_) * T(2.0) * relative_q_diff[2];
        residuals[2] = T(weight_q_) * T(2.0) * relative_q_diff[3];
        residuals[3] = T(weight_x_) * (C[0] - prior_x_);
        residuals[4] = T(weight_y_) * (C[1] - prior_y_);
        residuals[5] = T(weight_z_) * (C[2] - prior_z_);

        return true;
    }
private:
    const double prior_qw_;
    const double prior_qx_;
    const double prior_qy_;
    const double prior_qz_;
    const double prior_x_;
    const double prior_y_;
    const double prior_z_;
    const double local_qw_;
    const double local_qx_;
    const double local_qy_;
    const double local_qz_;
    const double local_x_;
    const double local_y_;
    const double local_z_;
    const double weight_q_;
    const double weight_x_;
    const double weight_y_;
    const double weight_z_;
};

// prior absolute pose cost function
class PriorAbsoluteDistanceCostFunction {
public:
    PriorAbsoluteDistanceCostFunction(const Eigen::Vector3d& prior_c,
                                      const double weight = 1.0)
        : prior_x_(prior_c(0)),
          prior_y_(prior_c(1)),
          prior_z_(prior_c(2)),
          weight_(weight)
        {}

    static ceres::CostFunction* Create(const Eigen::Vector3d& prior_c,
                                       const double weight = 1.0) {
        return new ceres::AutoDiffCostFunction<PriorAbsoluteDistanceCostFunction, 1, 4, 3>(
            new PriorAbsoluteDistanceCostFunction(prior_c, weight));
    }

    template <typename T>
    bool operator()(const T* const qvec, const T* const tvec, T* residuals) const {
        T q_inverse[4] = {T(qvec[0]), T(-qvec[1]), T(-qvec[2]), T(-qvec[3])};

        T C[3];
        ceres::QuaternionRotatePoint(q_inverse, tvec, C);

        C[0] = - C[0];
        C[1] = - C[1];
        C[2] = - C[2];

        T dx = C[0] - prior_x_;
        T dy = C[1] - prior_y_;
        T dz = C[2] - prior_z_;
        residuals[0] = T(weight_) * ceres::sqrt(dx * dx + dy * dy + dz * dz);
        return true;
    }
private:
    const double prior_x_;
    const double prior_y_;
    const double prior_z_;
    const double weight_;
};

// Computes the error between a translation direction and the direction formed
// from two positions such that (c_j - c_i) - scalar * t_ij is minimized.
class PairwiseTranslationCostFunction {
public:
    PairwiseTranslationCostFunction(const Eigen::Vector3d& translation_direction,
                                    const double weight = 1.0) 
        : translation_direction_(translation_direction), 
          weight_(weight) {
        // CHECK_GT(weight_, 0);
    }

    // The error is given by the position error described above.
    template <typename T>
    bool operator()(const T* position1, const T* position2, T* residuals) const {
        const T kNormTolerance = T(1e-12);

        T translation[3];
        translation[0] = position2[0] - position1[0];
        translation[1] = position2[1] - position1[1];
        translation[2] = position2[2] - position1[2];
        T norm =
                sqrt(translation[0] * translation[0] + translation[1] * translation[1] +
                     translation[2] * translation[2]);

        // If the norm is very small then the positions are very close together. In
        // this case, avoid dividing by a tiny number which will cause the weight of
        // the residual term to potentially skyrocket.
        if (T(norm) < kNormTolerance) {
            norm = T(1.0);
        }

        residuals[0] = weight_ * (translation[0] / norm - translation_direction_[0]);
        residuals[1] = weight_ * (translation[1] / norm - translation_direction_[1]);
        residuals[2] = weight_ * (translation[2] / norm - translation_direction_[2]);
        return true;
    }

    static ceres::CostFunction* Create(
            const Eigen::Vector3d& translation_direction, const double weight = 1.0){
        return (new ceres::AutoDiffCostFunction<PairwiseTranslationCostFunction, 3, 3, 3>(
                new PairwiseTranslationCostFunction(translation_direction, weight)));
    }

    const Eigen::Vector3d translation_direction_;
    const double weight_;
};

// Computes the error between a translation direction and the direction formed
// from two positions such that (c_j - c_i) - scalar * t_ij is minimized.
class RigPairwiseTranslationCostFunction {
public:
    RigPairwiseTranslationCostFunction(const Eigen::Vector3d& translation_direction,
                                       const Eigen::Vector3d& local_position,
                                       const double weight = 1.0) 
        : translation_direction_(translation_direction), 
          local_position_(local_position),
          weight_(weight) {
        // CHECK_GT(weight_, 0);
    }

    // The error is given by the position error described above.
    template <typename T>
    bool operator()(const T* position1, const T* position2, T* residuals) const {
        const T kNormTolerance = T(1e-12);

        T translation[3];
        translation[0] = position2[0] - position1[0] - T(local_position_[0]);
        translation[1] = position2[1] - position1[1] - T(local_position_[1]);
        translation[2] = position2[2] - position1[2] - T(local_position_[2]);
        T norm =
                sqrt(translation[0] * translation[0] + translation[1] * translation[1] +
                     translation[2] * translation[2]);

        // If the norm is very small then the positions are very close together. In
        // this case, avoid dividing by a tiny number which will cause the weight of
        // the residual term to potentially skyrocket.
        if (T(norm) < kNormTolerance) {
            norm = T(1.0);
        }

        residuals[0] = weight_ * (translation[0] / norm - translation_direction_[0]);
        residuals[1] = weight_ * (translation[1] / norm - translation_direction_[1]);
        residuals[2] = weight_ * (translation[2] / norm - translation_direction_[2]);
        return true;
    }

    static ceres::CostFunction* Create(
            const Eigen::Vector3d& translation_direction, const Eigen::Vector3d& local_position, const double weight = 1.0){
        return (new ceres::AutoDiffCostFunction<RigPairwiseTranslationCostFunction, 3, 3, 3>(
                new RigPairwiseTranslationCostFunction(translation_direction, local_position, weight)));
    }

    const Eigen::Vector3d translation_direction_;
    const Eigen::Vector3d local_position_;
    const double weight_;
};

// prior absolute location cost function, used if gps prior is available

class PriorAbsoluteLocationGlobalSfMCostFunction {
public:
    PriorAbsoluteLocationGlobalSfMCostFunction(const Eigen::Vector3d& prior_c,  const double weight_x = 1.0,
                                               const double weight_y = 1.0, const double weight_z = 1.0, 
                                               const Eigen::Matrix3x4d sRT = Eigen::Matrix3x4d::Identity())
            : prior_x_(prior_c(0)),
              prior_y_(prior_c(1)),
              prior_z_(prior_c(2)),
              weight_x_(weight_x),
              weight_y_(weight_y),
              weight_z_(weight_z),
              t00(sRT(0, 0)),
              t01(sRT(0, 1)),
              t02(sRT(0, 2)),
              t03(sRT(0, 3)),
              t10(sRT(1, 0)),
              t11(sRT(1, 1)),
              t12(sRT(1, 2)),
              t13(sRT(1, 3)),
              t20(sRT(2, 0)),
              t21(sRT(2, 1)),
              t22(sRT(2, 2)),
              t23(sRT(2, 3))
    {}

    static ceres::CostFunction* Create(const Eigen::Vector3d& prior_c, const double weight_x = 1.0,
                                       const double weight_y = 1.0, const double weight_z = 1.0,
                                       const Eigen::Matrix3x4d sRT = Eigen::Matrix3x4d::Identity()) {
        return new ceres::AutoDiffCostFunction<PriorAbsoluteLocationGlobalSfMCostFunction, 3, 3>(
                new PriorAbsoluteLocationGlobalSfMCostFunction(prior_c, weight_x, weight_y, weight_z, sRT));
    }

    template <typename T>
    bool operator()( const T* const tvec, T* residuals) const {

        T tvec_t[3];
        tvec_t[0] = T(t00) * tvec[0] + T(t01) * tvec[1] + T(t02) * tvec[2] + T(t03);
        tvec_t[1] = T(t10) * tvec[0] + T(t11) * tvec[1] + T(t12) * tvec[2] + T(t13);
        tvec_t[2] = T(t20) * tvec[0] + T(t21) * tvec[1] + T(t22) * tvec[2] + T(t23);

        residuals[0] = T(weight_x_) * (tvec_t[0] - prior_x_);
        residuals[1] = T(weight_y_) * (tvec_t[1] - prior_y_);
        residuals[2] = T(weight_z_) * (tvec_t[2] - prior_z_);
        return true;
    }
private:
    const double t00;
    const double t01;
    const double t02;
    const double t03;
    const double t10;
    const double t11;
    const double t12;
    const double t13;
    const double t20;
    const double t21;
    const double t22;
    const double t23;
    const double prior_x_;
    const double prior_y_;
    const double prior_z_;
    const double weight_x_;
    const double weight_y_;
    const double weight_z_;

};


// Standard bundle adjustment cost function for variable
// camera pose and calibration and point parameters.
template <typename CameraModel>
class BundleAdjustmentNovatelCostFunction {
public:
    explicit BundleAdjustmentNovatelCostFunction(
        const Eigen::Vector4d& prior_q, const Eigen::Vector3d& prior_t,
        const Eigen::Vector2d& point2D, const double weight = 1.0)
        : prior_qw_(prior_q(0)),
          prior_qx_(prior_q(1)),
          prior_qy_(prior_q(2)),
          prior_qz_(prior_q(3)),
          prior_tx_(prior_t(0)),
          prior_ty_(prior_t(1)),
          prior_tz_(prior_t(2)),
          observed_x_(point2D(0)),
          observed_y_(point2D(1)),
          weight_(weight) {}

    static ceres::CostFunction* Create(const Eigen::Vector4d& prior_q, const Eigen::Vector3d& prior_t,
        const Eigen::Vector2d& point2D, const double weight = 1.0) {
        return (new ceres::AutoDiffCostFunction<BundleAdjustmentNovatelCostFunction<CameraModel>, CameraModel::kNumResidual, 4,
                                                3, 3, CameraModel::kNumParams>(
            new BundleAdjustmentNovatelCostFunction(prior_q, prior_t, point2D, weight)));
    }

    template <typename T>
    bool operator()(const T* const delt_qvec, const T* const delt_tvec, const T* const point3D, const T* const camera_params,
                    T* residuals) const {
        T q_prior[4] = {T(prior_qw_), T(prior_qx_), T(prior_qy_), T(prior_qz_)};
        T t_prior[3] = {T(prior_tx_), T(prior_ty_), T(prior_tz_)};

        T qvec[4];
        ceres::QuaternionProduct(delt_qvec, q_prior, qvec);

        T tvec[3];
        ceres::QuaternionRotatePoint(delt_qvec, t_prior, tvec);

        tvec[0] = delt_tvec[0] + tvec[0];
        tvec[1] = delt_tvec[1] + tvec[1];
        tvec[2] = delt_tvec[2] + tvec[2];

        // Rotate and translate.
        T projection[3];
        ceres::UnitQuaternionRotatePoint(qvec, point3D, projection);
        projection[0] += tvec[0];
        projection[1] += tvec[1];
        projection[2] += tvec[2];

        // Project to image plane.
        projection[0] /= projection[2];
        projection[1] /= projection[2];

        // Distort and transform to pixel space.
        CameraModel::WorldToImage(camera_params, projection[0], projection[1], &residuals[0], &residuals[1]);

        // Re-projection error.
        residuals[0] -= T(observed_x_);
        residuals[1] -= T(observed_y_);

        residuals[0] *= T(weight_);
        residuals[1] *= T(weight_);

        return true;
    }

private:
    const double prior_qw_;
    const double prior_qx_;
    const double prior_qy_;
    const double prior_qz_;
    const double prior_tx_;
    const double prior_ty_;
    const double prior_tz_;
    const double observed_x_;
    const double observed_y_;
    const double weight_;
};

class ColorCorrectionCostFunction {
public :
    ColorCorrectionCostFunction(const double his_i, const double his_j)
    : his_i_(his_i), his_j_(his_j) {}

    static ceres::CostFunction* Create(const double his_i, const double his_j) {
        return (new ceres::AutoDiffCostFunction<ColorCorrectionCostFunction, 1, 1, 1, 1, 1>(
                new ColorCorrectionCostFunction(his_i, his_j)));
    }

    template <typename T>
    bool operator()(const T* const si, const T* const sj, const T* const oi, const T* const oj, T* residuals) const{
        residuals[0] = ((si[0] * T(his_i_) + oi[0]) - (sj[0] * T(his_j_) + oj[0])) / (si[0] + sj[0]);
        return true;
    }

private:
    const double his_i_;
    const double his_j_;
};

class BundleAdjustmentLidarEdgeConstantRefPoseCostFunction {
public:
    BundleAdjustmentLidarEdgeConstantRefPoseCostFunction(
                                        const Eigen::Vector4d qvec,
                                        const Eigen::Vector3d tvec,
                                        Eigen::Vector3d curr_point_, 
                                        Eigen::Vector3d last_point_a_, 
                                        Eigen::Vector3d last_point_b_, 
                                        double s_, 
                                        const double weight = 1.0)
        : qw_(qvec(0)), qx_(qvec(1)), qy_(qvec(2)), qz_(qvec(3)), tx_(tvec(0)),
        ty_(tvec(1)), tz_(tvec(2)), curr_point(curr_point_), 
        last_point_a(last_point_a_), last_point_b(last_point_b_), s(s_),weight_(weight) {}

    static ceres::CostFunction *Create(const Eigen::Vector4d qvec,
                                        const Eigen::Vector3d tvec,
                                        const Eigen::Vector3d curr_point_, 
                                        const Eigen::Vector3d last_point_a_,
                                                        const Eigen::Vector3d last_point_b_, 
                                        const double s_, 
                                        const double weight = 1.0)
    {
        return (new ceres::AutoDiffCostFunction<
                BundleAdjustmentLidarEdgeConstantRefPoseCostFunction, 3, 4, 3>(
            new BundleAdjustmentLidarEdgeConstantRefPoseCostFunction(qvec, tvec, 
                curr_point_, last_point_a_, last_point_b_, s_,weight)));
    }

    template <typename T>
    bool operator()(const T *q2, const T *t2, T *residual) const
    {
        Eigen::Matrix<T, 3, 1> cp{T(curr_point.x()), T(curr_point.y()), T(curr_point.z())};
        Eigen::Matrix<T, 3, 1> spa{T(last_point_a.x()), T(last_point_a.y()), T(last_point_a.z())};
        Eigen::Matrix<T, 3, 1> spb{T(last_point_b.x()), T(last_point_b.y()), T(last_point_b.z())};

        Eigen::Quaternion<T> inv_q_ref{T(qw_), T(-qx_), T(-qy_), T(-qz_)};
        Eigen::Matrix<T, 3, 1> t_ref{T(tx_), T(ty_), T(tz_)};
        Eigen::Quaternion<T> inv_q_src{q2[0], -q2[1], -q2[2], -q2[3]};
        Eigen::Matrix<T, 3, 1> t_src{t2[0], t2[1], t2[2]};

        Eigen::Matrix<T, 3, 1> lp, lpa, lpb;
        lp = inv_q_ref * (cp - t_ref);
        lpa = inv_q_src * (spa - t_src);
        lpb = inv_q_src * (spb - t_src);

        Eigen::Matrix<T, 3, 1> nu = (lp - lpa).cross(lp - lpb);
        Eigen::Matrix<T, 3, 1> de = lpa - lpb;

        // T norm = (de.norm() * cp.norm());
        T norm = de.norm();
        residual[0] = T(weight_) * nu.x() / norm;
        residual[1] = T(weight_) * nu.y() / norm;
        residual[2] = T(weight_) * nu.z() / norm;

        return true;
    }

private:
    // Eigen::Matrix4d Tr_lidar2camera_4d;
    const double qw_;
    const double qx_;
    const double qy_;
    const double qz_;
    const double tx_;
    const double ty_;
    const double tz_;
    Eigen::Vector3d curr_point, last_point_a, last_point_b;
    double s;
    const double weight_;
};

class BundleAdjustmentLidarEdgeConstantSrcPoseCostFunction {
public:
    BundleAdjustmentLidarEdgeConstantSrcPoseCostFunction(
                                        const Eigen::Vector4d qvec,
                                        const Eigen::Vector3d tvec,
                                        Eigen::Vector3d curr_point_, 
                                        Eigen::Vector3d last_point_a_, 
                                        Eigen::Vector3d last_point_b_, 
                                        double s_,
                                        const double weight = 1.0)
        : qw_(qvec(0)), qx_(qvec(1)), qy_(qvec(2)), qz_(qvec(3)), tx_(tvec(0)),
        ty_(tvec(1)), tz_(tvec(2)), curr_point(curr_point_), 
        last_point_a(last_point_a_), last_point_b(last_point_b_), s(s_),weight_(weight) {}

    static ceres::CostFunction *Create(const Eigen::Vector4d qvec,
                                        const Eigen::Vector3d tvec,
                                        const Eigen::Vector3d curr_point_, 
                                        const Eigen::Vector3d last_point_a_,
                                        const Eigen::Vector3d last_point_b_, 
                                        const double s_,
                                        const double weight = 1.0)
    {
        return (new ceres::AutoDiffCostFunction<
                BundleAdjustmentLidarEdgeConstantSrcPoseCostFunction, 3, 4, 3>(
            new BundleAdjustmentLidarEdgeConstantSrcPoseCostFunction(qvec, tvec, 
                curr_point_, last_point_a_, last_point_b_, s_, weight)));
    }

    template <typename T>
    bool operator()(const T *q1, const T *t1, T *residual) const
    {
        Eigen::Matrix<T, 3, 1> cp{T(curr_point.x()), T(curr_point.y()), T(curr_point.z())};
        Eigen::Matrix<T, 3, 1> spa{T(last_point_a.x()), T(last_point_a.y()), T(last_point_a.z())};
        Eigen::Matrix<T, 3, 1> spb{T(last_point_b.x()), T(last_point_b.y()), T(last_point_b.z())};

        Eigen::Quaternion<T> inv_q_ref{q1[0], -q1[1], -q1[2], -q1[3]};
        Eigen::Matrix<T, 3, 1> t_ref{t1[0], t1[1], t1[2]};
        Eigen::Quaternion<T> inv_q_src{T(qw_), T(-qx_), T(-qy_), T(-qz_)};
        Eigen::Matrix<T, 3, 1> t_src{T(tx_), T(ty_), T(tz_)};

        Eigen::Matrix<T, 3, 1> lp, lpa, lpb;
        lp = inv_q_ref * (cp - t_ref);
        lpa = inv_q_src * (spa - t_src);
        lpb = inv_q_src * (spb - t_src);

        Eigen::Matrix<T, 3, 1> nu = (lp - lpa).cross(lp - lpb);
        Eigen::Matrix<T, 3, 1> de = lpa - lpb;
        
        // T norm = (de.norm() * cp.norm());
        T norm = de.norm();
        residual[0] = T(weight_) * nu.x() / norm;
        residual[1] = T(weight_) * nu.y() / norm;
        residual[2] = T(weight_) * nu.z() / norm;

        return true;
    }

private:
    // Eigen::Matrix4d Tr_lidar2camera_4d;
    const double qw_;
    const double qx_;
    const double qy_;
    const double qz_;
    const double tx_;
    const double ty_;
    const double tz_;
    Eigen::Vector3d curr_point, last_point_a, last_point_b;
    double s;
    const double weight_;
};

class BundleAdjustmentLidarEdgeCostFunction {
public:
    BundleAdjustmentLidarEdgeCostFunction(/* Eigen::Matrix4d Tr_lidar2camera_4d_, */ 
                                    Eigen::Vector3d curr_point_, 
                                    Eigen::Vector3d last_point_a_, 
                                    Eigen::Vector3d last_point_b_, 
                                    double s_, 
                                    const double weight = 1.0)
    : curr_point(curr_point_), last_point_a(last_point_a_), 
      last_point_b(last_point_b_), s(s_),weight_(weight) {}

    static ceres::CostFunction *Create(const Eigen::Vector3d curr_point_, 
                                        const Eigen::Vector3d last_point_a_,
                                        const Eigen::Vector3d last_point_b_, 
                                        const double s_, 
                                        const double weight = 1.0)
    {
        return (new ceres::AutoDiffCostFunction<
                BundleAdjustmentLidarEdgeCostFunction, 3, 4, 3, 4, 3>(
            new BundleAdjustmentLidarEdgeCostFunction(curr_point_, last_point_a_, 
                                                last_point_b_, s_,weight)));
    }

    template <typename T>
    bool operator()(const T *q1, const T *t1, const T *q2, const T *t2, T *residual) const
    {
        Eigen::Matrix<T, 3, 1> cp{T(curr_point.x()), T(curr_point.y()), T(curr_point.z())};
        Eigen::Matrix<T, 3, 1> spa{T(last_point_a.x()), T(last_point_a.y()), T(last_point_a.z())};
        Eigen::Matrix<T, 3, 1> spb{T(last_point_b.x()), T(last_point_b.y()), T(last_point_b.z())};

        Eigen::Quaternion<T> inv_q_ref{q1[0], -q1[1], -q1[2], -q1[3]};
        Eigen::Matrix<T, 3, 1> t_ref{t1[0], t1[1], t1[2]};
        Eigen::Quaternion<T> inv_q_src{q2[0], -q2[1], -q2[2], -q2[3]};
        Eigen::Matrix<T, 3, 1> t_src{t2[0], t2[1], t2[2]};

        Eigen::Matrix<T, 3, 1> lp, lpa, lpb;
        lp = inv_q_ref * (cp - t_ref);
        lpa = inv_q_src * (spa - t_src);
        lpb = inv_q_src * (spb - t_src);
        // lp = q_ref.inverse() * cp - q_ref.inverse() * t_ref;
        // lpa = q_src.inverse() * spa - q_src.inverse() * t_src;
        // lpb = q_src.inverse() * spb - q_src.inverse() * t_src;

        Eigen::Matrix<T, 3, 1> nu = (lp - lpa).cross(lp - lpb);
        Eigen::Matrix<T, 3, 1> de = lpa - lpb;

        // T norm = (de.norm() * cp.norm());
        T norm = de.norm();
        residual[0] = T(weight_) * nu.x() / norm;
        residual[1] = T(weight_) * nu.y() / norm;
        residual[2] = T(weight_) * nu.z() / norm;

        return true;
    }

 private:
  // Eigen::Matrix4d Tr_lidar2camera_4d;
    Eigen::Vector3d curr_point, last_point_a, last_point_b;
    double s;
    const double weight_;
};

class BundleAdjustmentLidarPlaneConstantRefPoseCostFunction {
public:
    BundleAdjustmentLidarPlaneConstantRefPoseCostFunction(
                                        const Eigen::Vector4d qvec,
                                        const Eigen::Vector3d tvec,
                                        Eigen::Vector3d curr_point_, 
                                        Eigen::Vector3d last_point_j_, 
                                        Eigen::Vector3d last_point_l_, 
                                        Eigen::Vector3d last_point_m_, 
                                        double s_, 
                                        const double weight = 1.0)
        : qw_(qvec(0)), qx_(qvec(1)), qy_(qvec(2)), qz_(qvec(3)), tx_(tvec(0)),
        ty_(tvec(1)), tz_(tvec(2)), curr_point(curr_point_), 
        last_point_j(last_point_j_),  last_point_l(last_point_l_), 
        last_point_m(last_point_m_), s(s_),weight_(weight) {}

    static ceres::CostFunction *Create(const Eigen::Vector4d qvec,
                                        const Eigen::Vector3d tvec,
                                        const Eigen::Vector3d curr_point_, 
                                        const Eigen::Vector3d last_point_j_,
                                        const Eigen::Vector3d last_point_l_, 
                                        const Eigen::Vector3d last_point_m_,
                                        const double s_, 
                                        const double weight = 1.0)
    {
        return (new ceres::AutoDiffCostFunction<
                BundleAdjustmentLidarPlaneConstantRefPoseCostFunction, 1, 4, 3>(
            new BundleAdjustmentLidarPlaneConstantRefPoseCostFunction(qvec, tvec, 
                curr_point_, last_point_j_, last_point_l_, last_point_m_, s_, weight)));
    }

    template <typename T>
    bool operator()( const T *q2, const T *t2, T *residual) const
    {
        Eigen::Matrix<T, 3, 1> cp{T(curr_point.x()), T(curr_point.y()), T(curr_point.z())};
        Eigen::Matrix<T, 3, 1> spj{T(last_point_j.x()), T(last_point_j.y()), T(last_point_j.z())};
        Eigen::Matrix<T, 3, 1> spl{T(last_point_l.x()), T(last_point_l.y()), T(last_point_l.z())};
        Eigen::Matrix<T, 3, 1> spm{T(last_point_m.x()), T(last_point_m.y()), T(last_point_m.z())};

        Eigen::Quaternion<T> inv_q_ref{T(qw_), T(-qx_), T(-qy_), T(-qz_)};
        Eigen::Matrix<T, 3, 1> t_ref{T(tx_), T(ty_), T(tz_)};
        Eigen::Quaternion<T> inv_q_src{q2[0], -q2[1], -q2[2], -q2[3]};
        Eigen::Matrix<T, 3, 1> t_src{t2[0], t2[1], t2[2]};

        Eigen::Matrix<T, 3, 1> lp, lpj, lpl, lpm;
        lp = inv_q_ref * (cp - t_ref);
        lpj = inv_q_src * (spj - t_src);
        lpl = inv_q_src * (spl - t_src);
        lpm = inv_q_src * (spm - t_src);

        Eigen::Matrix<T, 3, 1>  ljm_norm;
        ljm_norm = (lpj - lpl).cross(lpj - lpm);
        ljm_norm.normalize();

        residual[0] = T(weight_) * (lp - lpj).dot(ljm_norm);
        // residual[0] = T(weight_) * (lp - lpj).dot(ljm_norm) / cp.norm();

        return true;
    }

private:
    const double qw_;
    const double qx_;
    const double qy_;
    const double qz_;
    const double tx_;
    const double ty_;
    const double tz_;
    Eigen::Vector3d curr_point, last_point_j, last_point_l, last_point_m;
    double s;
    const double weight_;
};

class BundleAdjustmentLidarPlaneConstantSrcPoseCostFunction {
public:
    BundleAdjustmentLidarPlaneConstantSrcPoseCostFunction(
                                            const Eigen::Vector4d qvec,
                                            const Eigen::Vector3d tvec,
                                            Eigen::Vector3d curr_point_, 
                                            Eigen::Vector3d last_point_j_, 
                                            Eigen::Vector3d last_point_l_, 
                                            Eigen::Vector3d last_point_m_, 
                                            double s_, 
                                            const double weight = 1.0)
        : qw_(qvec(0)), qx_(qvec(1)), qy_(qvec(2)), qz_(qvec(3)), tx_(tvec(0)),
        ty_(tvec(1)), tz_(tvec(2)), curr_point(curr_point_), 
        last_point_j(last_point_j_), last_point_l(last_point_l_), 
        last_point_m(last_point_m_), s(s_),weight_(weight) {}

    static ceres::CostFunction *Create(const Eigen::Vector4d qvec,
                                        const Eigen::Vector3d tvec,
                                        const Eigen::Vector3d curr_point_, 
                                        const Eigen::Vector3d last_point_j_,
                                        const Eigen::Vector3d last_point_l_, 
                                        const Eigen::Vector3d last_point_m_,
                                        const double s_, 
                                        const double weight = 1.0)
    {
        return (new ceres::AutoDiffCostFunction<
                BundleAdjustmentLidarPlaneConstantSrcPoseCostFunction, 1, 4, 3>(
            new BundleAdjustmentLidarPlaneConstantSrcPoseCostFunction(qvec, tvec, 
                curr_point_, last_point_j_, last_point_l_, last_point_m_, s_, weight)));
    }

    template <typename T>
    bool operator()(const T *q1, const T *t1,  T *residual) const
    {
        Eigen::Matrix<T, 3, 1> cp{T(curr_point.x()), T(curr_point.y()), T(curr_point.z())};
        Eigen::Matrix<T, 3, 1> spj{T(last_point_j.x()), T(last_point_j.y()), T(last_point_j.z())};
        Eigen::Matrix<T, 3, 1> spl{T(last_point_l.x()), T(last_point_l.y()), T(last_point_l.z())};
        Eigen::Matrix<T, 3, 1> spm{T(last_point_m.x()), T(last_point_m.y()), T(last_point_m.z())};
        // Eigen::Matrix<T, 3, 1> ljm{T(ljm_norm.x()), T(ljm_norm.y()), T(ljm_norm.z())};

        Eigen::Quaternion<T> inv_q_ref{q1[0], -q1[1], -q1[2], -q1[3]};
        Eigen::Matrix<T, 3, 1> t_ref{t1[0], t1[1], t1[2]};
        Eigen::Quaternion<T> inv_q_src{T(qw_), T(-qx_), T(-qy_), T(-qz_)};
        Eigen::Matrix<T, 3, 1> t_src{T(tx_), T(ty_), T(tz_)};

        // Eigen::Quaternion<T> q_ref_l2c{q_l1[0], q_l1[1], q_l1[2], q_l1[3]};
        // Eigen::Matrix<T, 3, 1> t_ref_l2c{t_l1[0], t_l1[1], t_l1[2]};    
        // Eigen::Quaternion<T> q_src_l2c{q_l2[0], q_l2[1], q_l2[2], q_l2[3]};
        // Eigen::Matrix<T, 3, 1> t_src_l2c{t_l2[0], t_l2[1], t_l2[2]};    

        Eigen::Matrix<T, 3, 1> lp, lpj, lpl, lpm;
        lp = inv_q_ref * (cp - t_ref);
        lpj = inv_q_src * (spj - t_src);
        lpl = inv_q_src * (spl - t_src);
        lpm = inv_q_src * (spm - t_src);

        Eigen::Matrix<T, 3, 1>  ljm_norm;
        ljm_norm = (lpj - lpl).cross(lpj - lpm);
        ljm_norm.normalize();

        residual[0] = T(weight_) * (lp - lpj).dot(ljm_norm);
    //    residual[0] = T(weight_) * (lp - lpj).dot(ljm_norm) / cp.norm();

        return true;
    }

private:
    const double qw_;
    const double qx_;
    const double qy_;
    const double qz_;
    const double tx_;
    const double ty_;
    const double tz_;
    Eigen::Vector3d curr_point, last_point_j, last_point_l, last_point_m;
    double s;
    const double weight_;
};

class BundleAdjustmentLidarPlaneCostFunction {
public:
    BundleAdjustmentLidarPlaneCostFunction(Eigen::Vector3d curr_point_, 
                                        Eigen::Vector3d last_point_j_, 
                                        Eigen::Vector3d last_point_l_, 
                                        Eigen::Vector3d last_point_m_, 
                                        double s_,
                                        const double weight = 1.0)
        : curr_point(curr_point_), last_point_j(last_point_j_), 
          last_point_l(last_point_l_), last_point_m(last_point_m_), s(s_),weight_(weight) {}

    static ceres::CostFunction *Create(const Eigen::Vector3d curr_point_, 
                                    const Eigen::Vector3d last_point_j_,
                                    const Eigen::Vector3d last_point_l_, 
                                    const Eigen::Vector3d last_point_m_,
                                    const double s_, 
                                    const double weight = 1.0)
    {
        return (new ceres::AutoDiffCostFunction<
                BundleAdjustmentLidarPlaneCostFunction, 1, 4, 3, 4, 3>(
            new BundleAdjustmentLidarPlaneCostFunction(curr_point_, last_point_j_, 
                                            last_point_l_, last_point_m_, s_,weight)));
    }

    template <typename T>
    bool operator()(const T *q1, const T *t1, const T *q2, const T *t2, T *residual) const
    {
        Eigen::Matrix<T, 3, 1> cp{T(curr_point.x()), T(curr_point.y()), T(curr_point.z())};
        Eigen::Matrix<T, 3, 1> spj{T(last_point_j.x()), T(last_point_j.y()), T(last_point_j.z())};
        Eigen::Matrix<T, 3, 1> spl{T(last_point_l.x()), T(last_point_l.y()), T(last_point_l.z())};
        Eigen::Matrix<T, 3, 1> spm{T(last_point_m.x()), T(last_point_m.y()), T(last_point_m.z())};
        // Eigen::Matrix<T, 3, 1> ljm{T(ljm_norm.x()), T(ljm_norm.y()), T(ljm_norm.z())};

        Eigen::Quaternion<T> inv_q_ref{q1[0], -q1[1], -q1[2], -q1[3]};
        Eigen::Matrix<T, 3, 1> t_ref{t1[0], t1[1], t1[2]};
        Eigen::Quaternion<T> inv_q_src{q2[0], -q2[1], -q2[2], -q2[3]};
        Eigen::Matrix<T, 3, 1> t_src{t2[0], t2[1], t2[2]};

        // Eigen::Quaternion<T> q_ref_l2c{q_l1[0], q_l1[1], q_l1[2], q_l1[3]};
        // Eigen::Matrix<T, 3, 1> t_ref_l2c{t_l1[0], t_l1[1], t_l1[2]};    
        // Eigen::Quaternion<T> q_src_l2c{q_l2[0], q_l2[1], q_l2[2], q_l2[3]};
        // Eigen::Matrix<T, 3, 1> t_src_l2c{t_l2[0], t_l2[1], t_l2[2]};    

        Eigen::Matrix<T, 3, 1> lp, lpj, lpl, lpm;
        lp = inv_q_ref * (cp - t_ref);
        lpj = inv_q_src * (spj - t_src);
        lpl = inv_q_src * (spl - t_src);
        lpm = inv_q_src * (spm - t_src);

        Eigen::Matrix<T, 3, 1>  ljm_norm;
        ljm_norm = (lpj - lpl).cross(lpj - lpm);
        ljm_norm.normalize();

        residual[0] = T(weight_) * (lp - lpj).dot(ljm_norm);
        // residual[0] = T(weight_) * (lp - lpj).dot(ljm_norm) / cp.norm();

        return true;
    }

private:
    Eigen::Vector3d curr_point, last_point_j, last_point_l, last_point_m;
    double s;
    const double weight_;
};

class BundleAdjustmentLidarPointConstantRefPoseCostFunction {
public:
    BundleAdjustmentLidarPointConstantRefPoseCostFunction(
                                        const Eigen::Vector4d qvec,
                                        const Eigen::Vector3d tvec,
                                        Eigen::Vector3d curr_point_, 
                                        Eigen::Vector3d last_point_a_, 
                                        double s_, 
                                        const double weight = 1.0)
        : qw_(qvec(0)), qx_(qvec(1)), qy_(qvec(2)), qz_(qvec(3)), tx_(tvec(0)),
        ty_(tvec(1)), tz_(tvec(2)), curr_point(curr_point_), 
        last_point_a(last_point_a_), s(s_),weight_(weight) {}

    static ceres::CostFunction *Create(const Eigen::Vector4d qvec,
                                        const Eigen::Vector3d tvec,
                                        const Eigen::Vector3d curr_point_, 
                                        const Eigen::Vector3d last_point_a_,
                                        const double s_, 
                                        const double weight = 1.0)
    {
        return (new ceres::AutoDiffCostFunction<
                BundleAdjustmentLidarPointConstantRefPoseCostFunction, 3, 4, 3>(
            new BundleAdjustmentLidarPointConstantRefPoseCostFunction(qvec, tvec, 
                curr_point_, last_point_a_, s_, weight)));
    }

    template <typename T>
    bool operator()(const T *q2, const T *t2, T *residual) const
    {
        Eigen::Matrix<T, 3, 1> cp{T(curr_point.x()), T(curr_point.y()), T(curr_point.z())};
        Eigen::Matrix<T, 3, 1> spa{T(last_point_a.x()), T(last_point_a.y()), T(last_point_a.z())};

        Eigen::Quaternion<T> inv_q_ref{T(qw_), T(-qx_), T(-qy_), T(-qz_)};
        Eigen::Matrix<T, 3, 1> t_ref{T(tx_), T(ty_), T(tz_)};
        Eigen::Quaternion<T> inv_q_src{q2[0], -q2[1], -q2[2], -q2[3]};
        Eigen::Matrix<T, 3, 1> t_src{t2[0], t2[1], t2[2]};

        Eigen::Matrix<T, 3, 1> lp, lpa;
        lp = inv_q_ref * (cp - t_ref);
        lpa = inv_q_src * (spa - t_src);

        Eigen::Matrix<T, 3, 1> nu = lp - lpa;

        T norm = cp.norm();
        residual[0] = T(weight_) * nu.x() / norm;
        residual[1] = T(weight_) * nu.y() / norm;
        residual[2] = T(weight_) * nu.z() / norm;

        return true;
    }

private:
    // Eigen::Matrix4d Tr_lidar2camera_4d;
    const double qw_;
    const double qx_;
    const double qy_;
    const double qz_;
    const double tx_;
    const double ty_;
    const double tz_;
    Eigen::Vector3d curr_point, last_point_a;
    double s;
    const double weight_;
};

class BundleAdjustmentPointEdgeConstantSrcPoseCostFunction {
public:
    BundleAdjustmentPointEdgeConstantSrcPoseCostFunction(
                                        const Eigen::Vector4d qvec,
                                        const Eigen::Vector3d tvec,
                                        Eigen::Vector3d curr_point_, 
                                        Eigen::Vector3d last_point_a_, 
                                        double s_,
                                        const double weight = 1.0)
        : qw_(qvec(0)), qx_(qvec(1)), qy_(qvec(2)), qz_(qvec(3)), tx_(tvec(0)),
        ty_(tvec(1)), tz_(tvec(2)), curr_point(curr_point_), 
        last_point_a(last_point_a_), s(s_),weight_(weight) {}

    static ceres::CostFunction *Create(const Eigen::Vector4d qvec,
                                        const Eigen::Vector3d tvec,
                                        const Eigen::Vector3d curr_point_, 
                                        const Eigen::Vector3d last_point_a_,
                                        const double s_,
                                        const double weight = 1.0)
    {
        return (new ceres::AutoDiffCostFunction<
                BundleAdjustmentPointEdgeConstantSrcPoseCostFunction, 3, 4, 3>(
            new BundleAdjustmentPointEdgeConstantSrcPoseCostFunction(qvec, tvec, 
                curr_point_, last_point_a_, s_, weight)));
    }

    template <typename T>
    bool operator()(const T *q1, const T *t1, T *residual) const
    {
        Eigen::Matrix<T, 3, 1> cp{T(curr_point.x()), T(curr_point.y()), T(curr_point.z())};
        Eigen::Matrix<T, 3, 1> spa{T(last_point_a.x()), T(last_point_a.y()), T(last_point_a.z())};

        Eigen::Quaternion<T> inv_q_ref{q1[0], -q1[1], -q1[2], -q1[3]};
        Eigen::Matrix<T, 3, 1> t_ref{t1[0], t1[1], t1[2]};
        Eigen::Quaternion<T> inv_q_src{T(qw_), T(-qx_), T(-qy_), T(-qz_)};
        Eigen::Matrix<T, 3, 1> t_src{T(tx_), T(ty_), T(tz_)};

        Eigen::Matrix<T, 3, 1> lp, lpa;
        lp = inv_q_ref * (cp - t_ref);
        lpa = inv_q_src * (spa - t_src);

        Eigen::Matrix<T, 3, 1> nu = lp - lpa;
        
        T norm = cp.norm();
        residual[0] = T(weight_) * nu.x() / norm;
        residual[1] = T(weight_) * nu.y() / norm;
        residual[2] = T(weight_) * nu.z() / norm;

        return true;
    }

private:
    const double qw_;
    const double qx_;
    const double qy_;
    const double qz_;
    const double tx_;
    const double ty_;
    const double tz_;
    Eigen::Vector3d curr_point, last_point_a;
    double s;
    const double weight_;
};

class BundleAdjustmentLidarPointCostFunction {
public:
    BundleAdjustmentLidarPointCostFunction(
                                    Eigen::Vector3d curr_point_, 
                                    Eigen::Vector3d last_point_a_, 
                                    double s_, 
                                    const double weight = 1.0)
    : curr_point(curr_point_), last_point_a(last_point_a_), s(s_),weight_(weight) {}

    static ceres::CostFunction *Create(const Eigen::Vector3d curr_point_, 
                                        const Eigen::Vector3d last_point_a_,
                                        const double s_, 
                                        const double weight = 1.0)
    {
        return (new ceres::AutoDiffCostFunction<
                BundleAdjustmentLidarPointCostFunction, 3, 4, 3, 4, 3>(
            new BundleAdjustmentLidarPointCostFunction(curr_point_, last_point_a_, s_, weight)));
    }

    template <typename T>
    bool operator()(const T *q1, const T *t1, const T *q2, const T *t2, T *residual) const
    {
        Eigen::Matrix<T, 3, 1> cp{T(curr_point.x()), T(curr_point.y()), T(curr_point.z())};
        Eigen::Matrix<T, 3, 1> spa{T(last_point_a.x()), T(last_point_a.y()), T(last_point_a.z())};

        Eigen::Quaternion<T> inv_q_ref{q1[0], -q1[1], -q1[2], -q1[3]};
        Eigen::Matrix<T, 3, 1> t_ref{t1[0], t1[1], t1[2]};
        Eigen::Quaternion<T> inv_q_src{q2[0], -q2[1], -q2[2], -q2[3]};
        Eigen::Matrix<T, 3, 1> t_src{t2[0], t2[1], t2[2]};

        Eigen::Matrix<T, 3, 1> lp, lpa;
        lp = inv_q_ref * (cp - t_ref);
        lpa = inv_q_src * (spa - t_src);

        Eigen::Matrix<T, 3, 1> nu = lp - lpa;

        T norm =  cp.norm();
        residual[0] = T(weight_) * nu.x() / norm;
        residual[1] = T(weight_) * nu.y() / norm;
        residual[2] = T(weight_) * nu.z() / norm;

        return true;
    }

 private:
    Eigen::Vector3d curr_point, last_point_a;
    double s;
    const double weight_;
};


class BundleAdjustmentAbsolatePoseCostFunction {
  public:
    BundleAdjustmentAbsolatePoseCostFunction(const Eigen::Quaterniond& absolute_q_measured,
                         const Eigen::Vector3d& absolute_t_measured, const double weight = 1.0)
      : absolute_q_measured_(absolute_q_measured),
        absolute_t_measured_(absolute_t_measured),weight_(weight) {};

  template <typename T>
  bool operator()(const T* const q_ptr, const T* const t_ptr,
                  T* residuals_ptr) const {
    Eigen::Map<const Eigen::Matrix<T, 3, 1> > t21_estimated(t_ptr);
    Eigen::Quaternion<T> q21_estimated(q_ptr[0], q_ptr[1], q_ptr[2], q_ptr[3]);

    Eigen::Quaternion<T> delta_q =
      absolute_q_measured_.template cast<T>() * q21_estimated.conjugate();

    // Compute the residuals.
    // [position]     = [delta_t]
    // [orientation]  = [2 * delta_q(0:2)]
    Eigen::Map<Eigen::Matrix<T, 6, 1> > residuals(residuals_ptr);
    residuals.template block<3, 1>(0, 0) =  T(weight_) * (
      absolute_t_measured_.template cast<T>() - t21_estimated);
    residuals.template block<3, 1>(3, 0) = T(2.0 * weight_) * delta_q.vec();

    return true;
  }

  static ceres::CostFunction* Create(const Eigen::Quaterniond& absolute_q_measured,
                                     const Eigen::Vector3d& absolute_t_measured, 
                                     const double weight = 1.0) {
    return new ceres::AutoDiffCostFunction<BundleAdjustmentAbsolatePoseCostFunction, 6, 4, 3>(
      new BundleAdjustmentAbsolatePoseCostFunction(absolute_q_measured, absolute_t_measured, weight)
    );
  }

  private:
    const Eigen::Quaterniond absolute_q_measured_;
    const Eigen::Vector3d absolute_t_measured_;
    const double weight_;
};


class LidarCameraPoseCostFunction {
public:
    LidarCameraPoseCostFunction(const Eigen::Vector4d lidar_to_cam_qvec,
                                const Eigen::Vector3d lidar_to_cam_tvec,
                                const double weight_q = 1.0,
                                const double weight_x = 1.0,
                                const double weight_y = 1.0,
                                const double weight_z = 1.0)
        : lidar_to_cam_qvec_(lidar_to_cam_qvec),
          lidar_to_cam_tvec_(lidar_to_cam_tvec),
          weight_q_(weight_q),
          weight_x_(weight_x),
          weight_y_(weight_y),
          weight_z_(weight_z)
        {}

    static ceres::CostFunction* Create(const Eigen::Vector4d lidar_to_cam_qvec,
                                        const Eigen::Vector3d lidar_to_cam_tvec,
                                        const double weight_q = 1.0,
                                        const double weight_x = 1.0,
                                        const double weight_y = 1.0,
                                        const double weight_z = 1.0) {
        return new ceres::AutoDiffCostFunction<LidarCameraPoseCostFunction, 6, 4, 3, 4, 3>(
            new LidarCameraPoseCostFunction(lidar_to_cam_qvec, lidar_to_cam_tvec, weight_q, weight_x, weight_y, weight_z));
    }

    template <typename T>
    bool operator()(const T* const lidar_qvec, const T* const lidar_tvec, const T* const cam_qvec, const T* const cam_tvec, T* residuals) const {
        T l2c_qvec[4] = {T(lidar_to_cam_qvec_[0]), T(lidar_to_cam_qvec_[1]), T(lidar_to_cam_qvec_[2]), T(lidar_to_cam_qvec_[3])};
        T l2c_tvec[3] = {T(lidar_to_cam_tvec_[0]), T(lidar_to_cam_tvec_[1]), T(lidar_to_cam_tvec_[2])};

        T qvec[4];
        ceres::QuaternionProduct(l2c_qvec, lidar_qvec, qvec);
        T tvec[3];
        ceres::QuaternionRotatePoint(l2c_qvec, lidar_tvec, tvec);
        tvec[0] += l2c_tvec[0];
        tvec[1] += l2c_tvec[1];
        tvec[2] += l2c_tvec[2];

        T q_inverse[4] = {qvec[0], -qvec[1], -qvec[2], -qvec[3]};
        T relative_q_diff[4];
        ceres::QuaternionProduct(q_inverse, cam_qvec, relative_q_diff);

        T C[3];
        ceres::QuaternionRotatePoint(q_inverse, tvec, C);
        C[0] = - C[0];
        C[1] = - C[1];
        C[2] = - C[2];

        T cam_q_inverse[4] = {cam_qvec[0], -cam_qvec[1], -cam_qvec[2], -cam_qvec[3]};
        T cam_C[3];
        ceres::QuaternionRotatePoint(cam_q_inverse, cam_tvec, cam_C);
        cam_C[0] = - cam_C[0];
        cam_C[1] = - cam_C[1];
        cam_C[2] = - cam_C[2];

        // T C_diff[3] = { C[0] - cam_C[0], C[1] - cam_C[1], C[2] - cam_C[2] };
        // T norm = ceres::sqrt(C_diff[0] * C_diff[0] + C_diff[1] * C_diff[1] + C_diff[2] * C_diff[2]);
        // norm = T(2.0) * (ceres::sqrt(T(1.0) + T(3.0) * norm) - T(1.0));

        residuals[0] = T(weight_q_) * T(2.0) * relative_q_diff[1];
        residuals[1] = T(weight_q_) * T(2.0) * relative_q_diff[2];
        residuals[2] = T(weight_q_) * T(2.0) * relative_q_diff[3];
        residuals[3] = T(weight_x_) * (C[0] - cam_C[0]);
        residuals[4] = T(weight_y_) * (C[1] - cam_C[1]);
        residuals[5] = T(weight_z_) * (C[2] - cam_C[2]);

        return true;
    }
private:
    Eigen::Vector4d lidar_to_cam_qvec_;
    Eigen::Vector3d lidar_to_cam_tvec_;
    const double weight_q_;
    const double weight_x_;
    const double weight_y_;
    const double weight_z_;
};

class LidarCameraConstPoseCostFunction {
public:
    LidarCameraConstPoseCostFunction(const Eigen::Vector4d relative_qvec,
                                    const Eigen::Vector3d relative_tvec,
                                    const Eigen::Vector4d prior_qvec,
                                    const Eigen::Vector3d prior_tvec,
                                    const double weight_q = 1.0,
                                    const double weight_x = 1.0,
                                    const double weight_y = 1.0,
                                    const double weight_z = 1.0)
        : relative_qvec_(relative_qvec),
          relative_tvec_(relative_tvec),
          prior_qvec_(prior_qvec),
          prior_tvec_(prior_tvec),
          weight_q_(weight_q),
          weight_x_(weight_x),
          weight_y_(weight_y),
          weight_z_(weight_z)
        {}

    static ceres::CostFunction* Create(const Eigen::Vector4d relative_qvec,
                                        const Eigen::Vector3d relative_tvec,
                                        const Eigen::Vector4d prior_qvec,
                                        const Eigen::Vector3d prior_tvec,
                                        const double weight_q = 1.0,
                                        const double weight_x = 1.0,
                                        const double weight_y = 1.0,
                                        const double weight_z = 1.0) {
        return new ceres::AutoDiffCostFunction<LidarCameraConstPoseCostFunction, 6, 4, 3>(
            new LidarCameraConstPoseCostFunction(relative_qvec, relative_tvec, prior_qvec, prior_tvec, weight_q, weight_x, weight_y, weight_z));
    }

    template <typename T>
    bool operator()(const T* const qvec, const T* const tvec, T* residuals) const {
        T relative_qvec[4] = {T(relative_qvec_[0]), T(relative_qvec_[1]), T(relative_qvec_[2]), T(relative_qvec_[3])};
        T relative_tvec[3] = {T(relative_tvec_[0]), T(relative_tvec_[1]), T(relative_tvec_[2])};

        T prior_qvec[4] = {T(prior_qvec_[0]), T(prior_qvec_[1]), T(prior_qvec_[2]), T(prior_qvec_[3])};
        T prior_tvec[3] = {T(prior_tvec_[0]), T(prior_tvec_[1]), T(prior_tvec_[2])};

        T qvec_t[4];
        ceres::QuaternionProduct(relative_qvec, prior_qvec, qvec_t);
        T tvec_t[3];
        ceres::QuaternionRotatePoint(relative_qvec, prior_tvec, tvec_t);
        tvec_t[0] += relative_tvec[0];
        tvec_t[1] += relative_tvec[1];
        tvec_t[2] += relative_tvec[2];

        T q_inverse[4] = {qvec[0], -qvec[1], -qvec[2], -qvec[3]};
        T relative_q_diff[4];
        ceres::QuaternionProduct(q_inverse, qvec_t, relative_q_diff);

        T C[3];
        ceres::QuaternionRotatePoint(q_inverse, tvec, C);
        C[0] = - C[0];
        C[1] = - C[1];
        C[2] = - C[2];

        T q_inverse_t[4] = {qvec_t[0], -qvec_t[1], -qvec_t[2], -qvec_t[3]};
        T C_t[3];
        ceres::QuaternionRotatePoint(q_inverse_t, tvec_t, C_t);
        C_t[0] = - C_t[0];
        C_t[1] = - C_t[1];
        C_t[2] = - C_t[2];

        residuals[0] = T(weight_q_) * T(2.0) * relative_q_diff[1];
        residuals[1] = T(weight_q_) * T(2.0) * relative_q_diff[2];
        residuals[2] = T(weight_q_) * T(2.0) * relative_q_diff[3];
        residuals[3] = T(weight_x_) * (C[0] - C_t[0]);
        residuals[4] = T(weight_y_) * (C[1] - C_t[1]);
        residuals[5] = T(weight_z_) * (C[2] - C_t[2]);

        return true;
    }
private:
    Eigen::Vector4d relative_qvec_;
    Eigen::Vector3d relative_tvec_;
    Eigen::Vector4d prior_qvec_;
    Eigen::Vector3d prior_tvec_;
    const double weight_q_;
    const double weight_x_;
    const double weight_y_;
    const double weight_z_;
};

class LidarAbsolutePoseCostFunction {
public:
    LidarAbsolutePoseCostFunction(const Eigen::Vector3d & point, 
                                  const Eigen::Vector3d & m_var,
                                  const Eigen::Vector3d & m_pivot,
                                  const Eigen::Matrix3d & m_cov,
                                  const double weight = 1.0)
        : point_(point), 
          m_var_(m_var),
          m_pivot_(m_pivot),
          m_cov_(m_cov),
          weight_(weight) {}
    
    static ceres::CostFunction* Create(const Eigen::Vector3d & point, 
                                       const Eigen::Vector3d & m_var,
                                       const Eigen::Vector3d & m_pivot, 
                                       const Eigen::Matrix3d & m_cov,
                                       const double weight = 1.0) {
        return new ceres::AutoDiffCostFunction<LidarAbsolutePoseCostFunction, 1, 4, 3>(
            new LidarAbsolutePoseCostFunction(point, m_var, m_pivot, m_cov, weight)
        );
    }

    template <typename T>
    bool operator()(const T* const qvec, const T* const tvec, T* residuals) const {
        const T qvec_inv[4] = {qvec[0], -qvec[1], -qvec[2], -qvec[3]};
        T tvec_t[3];
        ceres::QuaternionRotatePoint(qvec_inv, tvec, tvec_t);
        T point[3] = {T(point_[0]), T(point_[1]), T(point_[2])};
        T point_t[3];
        ceres::QuaternionRotatePoint(qvec_inv, point, point_t);
        point_t[0] -= tvec_t[0];
        point_t[1] -= tvec_t[1];
        point_t[2] -= tvec_t[2];

        T res[3] = { point_t[0] - T(m_var_[0]), point_t[1] - T(m_var_[1]), point_t[2] - T(m_var_[2]) };
        // T norm = ceres::sqrt(res[0] * res[0] + res[1] * res[1] + res[2] * res[2]);
        // res[0] /= norm;
        // res[1] /= norm;
        // res[2] /= norm;
        // residuals[0] = res[0] * T(m_pivot_[0]) + 
        //                res[1] * T(m_pivot_[1]) + 
        //                res[2] * T(m_pivot_[2]);
        T proj_len = res[0] * T(m_pivot_[0]) + 
                     res[1] * T(m_pivot_[1]) +
                     res[2] * T(m_pivot_[2]);
        res[0] = m_pivot_[0] * proj_len;
        res[1] = m_pivot_[1] * proj_len;
        res[2] = m_pivot_[2] * proj_len;
        T y = 0.5 * (T(m_cov_(0, 0)) * res[0] * res[0] + 
                    T(m_cov_(1, 1)) * res[1] * res[1] + 
                    T(m_cov_(2, 2)) * res[2] * res[2] +
                    T(m_cov_(0, 1) + m_cov_(1, 0)) * res[0] * res[1] +
                    T(m_cov_(1, 2) + m_cov_(2, 1)) * res[1] * res[2] +
                    T(m_cov_(0, 2) + m_cov_(2, 0)) * res[0] * res[2]);
        if (y < T(0.0)) residuals[0] = T(1.0);
        else residuals[0] = T(weight_) * (T(1.0) - ceres::exp(-y));
        return true;
    }
private:
    Eigen::Vector3d point_;
    Eigen::Vector3d m_var_;
    Eigen::Vector3d m_pivot_;
    Eigen::Matrix3d m_cov_;
    double weight_;
};


class LidarAbsolutePoseArrayCostFunction {
public:
    LidarAbsolutePoseArrayCostFunction(const Eigen::Vector3d & point, 
                                       const Eigen::Matrix<double, Eigen::Dynamic, 1> & m_vars,
                                       const Eigen::Matrix<double, Eigen::Dynamic, 1> & m_pivots,
                                       const Eigen::Matrix<double, Eigen::Dynamic, 1> & m_covs,
                                       const int m_num_voxel,
                                       const double weight = 1.0)
        : point_(point), 
          m_vars_(m_vars),
          m_pivots_(m_pivots),
          m_covs_(m_covs),
          m_num_voxel_(m_num_voxel),
          weight_(weight) {}
    
    static ceres::CostFunction* Create(const Eigen::Vector3d & point, 
                                       const Eigen::Matrix<double, Eigen::Dynamic, 1> & m_vars,
                                       const Eigen::Matrix<double, Eigen::Dynamic, 1> & m_pivots,
                                       const Eigen::Matrix<double, Eigen::Dynamic, 1> & m_covs,
                                       const int m_num_voxel,
                                       const double weight = 1.0) {
        return new ceres::AutoDiffCostFunction<LidarAbsolutePoseArrayCostFunction, 1, 4, 3>(
            new LidarAbsolutePoseArrayCostFunction(point, m_vars, m_pivots, m_covs, m_num_voxel, weight)
        );
    }

    template <typename T>
    bool operator()(const T* const qvec, const T* const tvec, T* residuals) const {
        const T qvec_inv[4] = {qvec[0], -qvec[1], -qvec[2], -qvec[3]};
        T tvec_t[3];
        ceres::QuaternionRotatePoint(qvec_inv, tvec, tvec_t);
        T point[3] = {T(point_[0]), T(point_[1]), T(point_[2])};
        T point_t[3];
        ceres::QuaternionRotatePoint(qvec_inv, point, point_t);
        point_t[0] -= tvec_t[0];
        point_t[1] -= tvec_t[1];
        point_t[2] -= tvec_t[2];

        T min_y = T(10000.0);
        if (T(m_num_voxel_) > T(0)) {
            T m_var[3] = {T(m_vars_(0, 0)), T(m_vars_(1, 0)), T(m_vars_(2, 0))};
            T m_pivot[3] = {T(m_pivots_(0, 0)), T(m_pivots_(1, 0)), T(m_pivots_(2, 0))};
            T m_cov[9] = {T(m_covs_(0, 0)), T(m_covs_(1, 0)), T(m_covs_(2, 0)),
                          T(m_covs_(3, 0)), T(m_covs_(4, 0)), T(m_covs_(5, 0)),
                          T(m_covs_(6, 0)), T(m_covs_(7, 0)), T(m_covs_(8, 0))};

            T res[3] = { point_t[0] - m_var[0], point_t[1] - m_var[1], point_t[2] - m_var[2] };
            T proj_len = res[0] * m_pivot[0] + res[1] * m_pivot[1] + res[2] * m_pivot[2];
            // if (proj_len <= 10-9) {
            //     std::cout << "m_var:" << m_var[0] << " " << m_var[1] << " " << m_var[2] << std::endl;
            //     std::cout << "m_pivot:" << m_pivot[0] << " " << m_pivot[1] << " " << m_pivot[2] << std::endl;
            //     // print("m_cov: %f %f %f", m_cov[0], m_cov[1], m_cov[2]);
            //     // print("%f %f %f", m_cov[3], m_cov[4], m_cov[5]);
            //     // print("%f %f %f", m_cov[6], m_cov[7], m_cov[8]);
            // }
            
            res[0] = m_pivot[0] * proj_len;
            res[1] = m_pivot[1] * proj_len;
            res[2] = m_pivot[2] * proj_len;
            T y = 0.5 * (m_cov[0] * res[0] * res[0] + 
                        m_cov[4] * res[1] * res[1] + 
                        m_cov[8] * res[2] * res[2] +
                        (m_cov[1] + m_cov[3]) * res[0] * res[1] +
                        (m_cov[5] + m_cov[7]) * res[1] * res[2] +
                        (m_cov[2] + m_cov[6]) * res[0] * res[2]);
            min_y = (min_y > y) ? y : min_y;
        }
        if (T(m_num_voxel_) > T(1)) {
            T m_var[3] = {T(m_vars_(3, 0)), T(m_vars_(4, 0)), T(m_vars_(5, 0))};
            T m_pivot[3] = {T(m_pivots_(3, 0)), T(m_pivots_(4, 0)), T(m_pivots_(5, 0))};
            T m_cov[9] = {T(m_covs_(9, 0)), T(m_covs_(10, 0)), T(m_covs_(11, 0)),
                          T(m_covs_(12, 0)), T(m_covs_(13, 0)), T(m_covs_(14, 0)),
                          T(m_covs_(15, 0)), T(m_covs_(16, 0)), T(m_covs_(17, 0))};

            T res[3] = { point_t[0] - m_var[0], point_t[1] - m_var[1], point_t[2] - m_var[2] };
            T proj_len = res[0] * m_pivot[0] + res[1] * m_pivot[1] + res[2] * m_pivot[2];
            
            res[0] = m_pivot[0] * proj_len;
            res[1] = m_pivot[1] * proj_len;
            res[2] = m_pivot[2] * proj_len;
            T y = 0.5 * (m_cov[0] * res[0] * res[0] + 
                        m_cov[4] * res[1] * res[1] + 
                        m_cov[8] * res[2] * res[2] +
                        (m_cov[1] + m_cov[3]) * res[0] * res[1] +
                        (m_cov[5] + m_cov[7]) * res[1] * res[2] +
                        (m_cov[2] + m_cov[6]) * res[0] * res[2]);
            min_y = (min_y > y) ? y : min_y;
        }
        if (T(m_num_voxel_) > T(2)) {
            T m_var[3] = {T(m_vars_(6, 0)), T(m_vars_(7, 0)), T(m_vars_(8, 0))};
            T m_pivot[3] = {T(m_pivots_(6, 0)), T(m_pivots_(7, 0)), T(m_pivots_(8, 0))};
            T m_cov[9] = {T(m_covs_(18, 0)), T(m_covs_(19, 0)), T(m_covs_(20, 0)),
                          T(m_covs_(21, 0)), T(m_covs_(22, 0)), T(m_covs_(23, 0)),
                          T(m_covs_(24, 0)), T(m_covs_(25, 0)), T(m_covs_(26, 0))};

            T res[3] = { point_t[0] - m_var[0], point_t[1] - m_var[1], point_t[2] - m_var[2] };
            T proj_len = res[0] * m_pivot[0] + res[1] * m_pivot[1] + res[2] * m_pivot[2];
            
            res[0] = m_pivot[0] * proj_len;
            res[1] = m_pivot[1] * proj_len;
            res[2] = m_pivot[2] * proj_len;
            T y = 0.5 * (m_cov[0] * res[0] * res[0] + 
                        m_cov[4] * res[1] * res[1] + 
                        m_cov[8] * res[2] * res[2] +
                        (m_cov[1] + m_cov[3]) * res[0] * res[1] +
                        (m_cov[5] + m_cov[7]) * res[1] * res[2] +
                        (m_cov[2] + m_cov[6]) * res[0] * res[2]);
            min_y = (min_y > y) ? y : min_y;
        }
        if (T(m_num_voxel_) > T(3)) {
            T m_var[3] = {T(m_vars_(9, 0)), T(m_vars_(10, 0)), T(m_vars_(11, 0))};
            T m_pivot[3] = {T(m_pivots_(9, 0)), T(m_pivots_(10, 0)), T(m_pivots_(11, 0))};
            T m_cov[9] = {T(m_covs_(27, 0)), T(m_covs_(28, 0)), T(m_covs_(29, 0)),
                          T(m_covs_(30, 0)), T(m_covs_(31, 0)), T(m_covs_(32, 0)),
                          T(m_covs_(33, 0)), T(m_covs_(34, 0)), T(m_covs_(35, 0))};

            T res[3] = { point_t[0] - m_var[0], point_t[1] - m_var[1], point_t[2] - m_var[2] };
            T proj_len = res[0] * m_pivot[0] + res[1] * m_pivot[1] + res[2] * m_pivot[2];
            
            res[0] = m_pivot[0] * proj_len;
            res[1] = m_pivot[1] * proj_len;
            res[2] = m_pivot[2] * proj_len;
            T y = 0.5 * (m_cov[0] * res[0] * res[0] + 
                        m_cov[4] * res[1] * res[1] + 
                        m_cov[8] * res[2] * res[2] +
                        (m_cov[1] + m_cov[3]) * res[0] * res[1] +
                        (m_cov[5] + m_cov[7]) * res[1] * res[2] +
                        (m_cov[2] + m_cov[6]) * res[0] * res[2]);
            min_y = (min_y > y) ? y : min_y;
        }
        if (T(m_num_voxel_) > T(4)) {
            T m_var[3] = {T(m_vars_(12, 0)), T(m_vars_(13, 0)), T(m_vars_(14, 0))};
            T m_pivot[3] = {T(m_pivots_(12, 0)), T(m_pivots_(13, 0)), T(m_pivots_(14, 0))};
            T m_cov[9] = {T(m_covs_(36, 0)), T(m_covs_(37, 0)), T(m_covs_(38, 0)),
                          T(m_covs_(39, 0)), T(m_covs_(40, 0)), T(m_covs_(41, 0)),
                          T(m_covs_(42, 0)), T(m_covs_(43, 0)), T(m_covs_(44, 0))};

            T res[3] = { point_t[0] - m_var[0], point_t[1] - m_var[1], point_t[2] - m_var[2] };
            T proj_len = res[0] * m_pivot[0] + res[1] * m_pivot[1] + res[2] * m_pivot[2];
            
            res[0] = m_pivot[0] * proj_len;
            res[1] = m_pivot[1] * proj_len;
            res[2] = m_pivot[2] * proj_len;
            T y = 0.5 * (m_cov[0] * res[0] * res[0] + 
                        m_cov[4] * res[1] * res[1] + 
                        m_cov[8] * res[2] * res[2] +
                        (m_cov[1] + m_cov[3]) * res[0] * res[1] +
                        (m_cov[5] + m_cov[7]) * res[1] * res[2] +
                        (m_cov[2] + m_cov[6]) * res[0] * res[2]);
            min_y = (min_y > y) ? y : min_y;
        }
        if (T(m_num_voxel_) > T(5)) {
            T m_var[3] = {T(m_vars_(15, 0)), T(m_vars_(16, 0)), T(m_vars_(17, 0))};
            T m_pivot[3] = {T(m_pivots_(15, 0)), T(m_pivots_(16, 0)), T(m_pivots_(17, 0))};
            T m_cov[9] = {T(m_covs_(45, 0)), T(m_covs_(46, 0)), T(m_covs_(47, 0)),
                          T(m_covs_(48, 0)), T(m_covs_(49, 0)), T(m_covs_(50, 0)),
                          T(m_covs_(51, 0)), T(m_covs_(52, 0)), T(m_covs_(53, 0))};

            T res[3] = { point_t[0] - m_var[0], point_t[1] - m_var[1], point_t[2] - m_var[2] };
            T proj_len = res[0] * m_pivot[0] + res[1] * m_pivot[1] + res[2] * m_pivot[2];
            
            res[0] = m_pivot[0] * proj_len;
            res[1] = m_pivot[1] * proj_len;
            res[2] = m_pivot[2] * proj_len;
            T y = 0.5 * (m_cov[0] * res[0] * res[0] + 
                        m_cov[4] * res[1] * res[1] + 
                        m_cov[8] * res[2] * res[2] +
                        (m_cov[1] + m_cov[3]) * res[0] * res[1] +
                        (m_cov[5] + m_cov[7]) * res[1] * res[2] +
                        (m_cov[2] + m_cov[6]) * res[0] * res[2]);
            min_y = (min_y > y) ? y : min_y;
        }
        if (T(m_num_voxel_) > T(6)) {
            T m_var[3] = {T(m_vars_(18, 0)), T(m_vars_(19, 0)), T(m_vars_(20, 0))};
            T m_pivot[3] = {T(m_pivots_(18, 0)), T(m_pivots_(19, 0)), T(m_pivots_(20, 0))};
            T m_cov[9] = {T(m_covs_(54, 0)), T(m_covs_(55, 0)), T(m_covs_(56, 0)),
                          T(m_covs_(57, 0)), T(m_covs_(58, 0)), T(m_covs_(59, 0)),
                          T(m_covs_(60, 0)), T(m_covs_(61, 0)), T(m_covs_(62, 0))};

            T res[3] = { point_t[0] - m_var[0], point_t[1] - m_var[1], point_t[2] - m_var[2] };
            T proj_len = res[0] * m_pivot[0] + res[1] * m_pivot[1] + res[2] * m_pivot[2];
            
            res[0] = m_pivot[0] * proj_len;
            res[1] = m_pivot[1] * proj_len;
            res[2] = m_pivot[2] * proj_len;
            T y = 0.5 * (m_cov[0] * res[0] * res[0] + 
                        m_cov[4] * res[1] * res[1] + 
                        m_cov[8] * res[2] * res[2] +
                        (m_cov[1] + m_cov[3]) * res[0] * res[1] +
                        (m_cov[5] + m_cov[7]) * res[1] * res[2] +
                        (m_cov[2] + m_cov[6]) * res[0] * res[2]);
            min_y = (min_y > y) ? y : min_y;
        }
        if (T(m_num_voxel_) > T(7)) {
            T m_var[3] = {T(m_vars_(21, 0)), T(m_vars_(22, 0)), T(m_vars_(23, 0))};
            T m_pivot[3] = {T(m_pivots_(21, 0)), T(m_pivots_(22, 0)), T(m_pivots_(23, 0))};
            T m_cov[9] = {T(m_covs_(63, 0)), T(m_covs_(64, 0)), T(m_covs_(65, 0)),
                          T(m_covs_(66, 0)), T(m_covs_(67, 0)), T(m_covs_(68, 0)),
                          T(m_covs_(69, 0)), T(m_covs_(70, 0)), T(m_covs_(71, 0))};

            T res[3] = { point_t[0] - m_var[0], point_t[1] - m_var[1], point_t[2] - m_var[2] };
            T proj_len = res[0] * m_pivot[0] + res[1] * m_pivot[1] + res[2] * m_pivot[2];
            
            res[0] = m_pivot[0] * proj_len;
            res[1] = m_pivot[1] * proj_len;
            res[2] = m_pivot[2] * proj_len;
            T y = 0.5 * (m_cov[0] * res[0] * res[0] + 
                        m_cov[4] * res[1] * res[1] + 
                        m_cov[8] * res[2] * res[2] +
                        (m_cov[1] + m_cov[3]) * res[0] * res[1] +
                        (m_cov[5] + m_cov[7]) * res[1] * res[2] +
                        (m_cov[2] + m_cov[6]) * res[0] * res[2]);
            min_y = (min_y > y) ? y : min_y;
        }
        if (min_y < T(0.0)) residuals[0] = T(1.0);
        else residuals[0] = T(weight_) * (T(1.0) - ceres::exp(-min_y));
        return true;
    }
private:
    Eigen::Vector3d point_;
    Eigen::Matrix<double, Eigen::Dynamic, 1> m_vars_;
    Eigen::Matrix<double, Eigen::Dynamic, 1> m_pivots_;
    Eigen::Matrix<double, Eigen::Dynamic, 1> m_covs_;
    int m_num_voxel_;
    double weight_;
};

class LidarAbsoluteDistanceCostFunction {
public:
    LidarAbsoluteDistanceCostFunction(const Eigen::Vector3d & m_var, 
                                      const Eigen::Vector3d & m_pivot,
                                      const Eigen::Matrix3d & m_cov,
                                      const double weight = 1.0)
        : m_var_(m_var),
          m_pivot_(m_pivot),
          m_cov_(m_cov),
          weight_(weight) {}
    
    static ceres::CostFunction* Create(const Eigen::Vector3d & m_var, 
                                       const Eigen::Vector3d & m_pivot,
                                       const Eigen::Matrix3d & m_cov,
                                       const double weight = 1.0) {
        return new ceres::AutoDiffCostFunction<LidarAbsoluteDistanceCostFunction, 1, 3>(
            new LidarAbsoluteDistanceCostFunction(m_var, m_pivot, m_cov, weight)
        );
    }

    template <typename T>
    bool operator()(const T* const point, T* residuals) const {
        T res[3] = { point[0] - T(m_var_[0]), point[1] - T(m_var_[1]), point[2] - T(m_var_[2]) };
        // T norm = ceres::sqrt(res[0] * res[0] + res[1] * res[1] + res[2] * res[2]);
        // res[0] /= norm;
        // res[1] /= norm;
        // res[2] /= norm;
        // residuals[0] = res[0] * T(m_pivot_[0]) + 
        //                res[1] * T(m_pivot_[1]) + 
        //                res[2] * T(m_pivot_[2]);
        // residuals[0] = weight_ * residuals[0];
        T proj_len = res[0] * T(m_pivot_[0]) + 
                     res[1] * T(m_pivot_[1]) +
                     res[2] * T(m_pivot_[2]);
        res[0] = m_pivot_[0] * proj_len;
        res[1] = m_pivot_[1] * proj_len;
        res[2] = m_pivot_[2] * proj_len;
        T y = 0.5 * (T(m_cov_(0, 0)) * res[0] * res[0] + 
                    T(m_cov_(1, 1)) * res[1] * res[1] + 
                    T(m_cov_(2, 2)) * res[2] * res[2] +
                    T(m_cov_(0, 1) + m_cov_(1, 0)) * res[0] * res[1] +
                    T(m_cov_(1, 2) + m_cov_(2, 1)) * res[1] * res[2] +
                    T(m_cov_(0, 2) + m_cov_(2, 0)) * res[0] * res[2]);
        residuals[0] = weight_ * (T(1.0) - ceres::exp(-y));
        return true;
    }
private:
    Eigen::Vector3d m_var_;
    Eigen::Vector3d m_pivot_;
    Eigen::Matrix3d m_cov_;
    double weight_;
};

class LidarRelativeDistanceCostFunction {
public:
    LidarRelativeDistanceCostFunction(const double distance,
                                      const double weight = 1.0)
        : distance_(distance),
          weight_(weight) {}
    
    static ceres::CostFunction* Create(const double distance,
                                       const double weight = 1.0) {
        return new ceres::AutoDiffCostFunction<LidarRelativeDistanceCostFunction, 1, 4, 3, 4, 3>(
            new LidarRelativeDistanceCostFunction(distance, weight)
        );
    }

    template <typename T>
    bool operator()(const T* const qvec1, const T* const tvec1, const T* const qvec2, const T* const tvec2, T* residuals) const {
        T qvec_inverse1[4] = {qvec1[0], -qvec1[1], -qvec1[2], -qvec1[2]};
        T C1[3];
        ceres::QuaternionRotatePoint(qvec_inverse1, tvec1, C1);
        C1[0] = -C1[0];
        C1[1] = -C1[1];
        C1[2] = -C1[2];

        T qvec_inverse2[4] = {qvec2[0], -qvec2[1], -qvec2[2], -qvec2[2]};
        T C2[3];
        ceres::QuaternionRotatePoint(qvec_inverse2, tvec2, C2);
        C2[0] = -C2[0];
        C2[1] = -C2[1];
        C2[2] = -C2[2];

        T C_diff[3] = {C1[0] - C2[0], C1[1] - C2[1], C1[2] - C2[2]};
        T dist = ceres::sqrt(C_diff[0] * C_diff[0] + C_diff[1] * C_diff[1] + C_diff[2] * C_diff[2]);

        residuals[0] = T(weight_) * (T(distance_) - dist);
        return true;
    }
private:
    double distance_;
    double weight_;
};

class LidarPointToPointCostFunction {
public:
    LidarPointToPointCostFunction(const Eigen::Vector3d & query_point,
                                  const Eigen::Vector3d & nearest_point,
                                  const double weight = 1.0)
        : query_point_(query_point), 
          nearest_point_(nearest_point),
          weight_(weight) {}
    
    static ceres::CostFunction* Create(const Eigen::Vector3d & query_point,
                                       const Eigen::Vector3d & nearest_point,
                                       const double weight = 1.0) {
        return new ceres::AutoDiffCostFunction<LidarPointToPointCostFunction, 1, 4, 3>(
            new LidarPointToPointCostFunction(query_point, nearest_point, weight)
        );
    }

    template <typename T>
    bool operator()(const T* const qvec, const T* const tvec, T* residuals) const {
        const T qvec_inv[4] = {qvec[0], -qvec[1], -qvec[2], -qvec[3]};
        T tvec_t[3];
        ceres::QuaternionRotatePoint(qvec_inv, tvec, tvec_t);
        T point[3] = {T(query_point_[0]), T(query_point_[1]), T(query_point_[2])};
        T point_t[3];
        ceres::QuaternionRotatePoint(qvec_inv, point, point_t);
        point_t[0] -= tvec_t[0];
        point_t[1] -= tvec_t[1];
        point_t[2] -= tvec_t[2];

        T diff[3] = { point_t[0] - T(nearest_point_[0]), 
                      point_t[1] - T(nearest_point_[1]), 
                      point_t[2] - T(nearest_point_[2]) };
        // residuals[0] = T(weight_) * ceres::sqrt(diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2]);
        T error = T(1.0) - exp(T(-12.0) * (diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2]));
        residuals[0] = T(weight_) * error;
        return true;
    }
private:
    Eigen::Vector3d query_point_;
    Eigen::Vector3d nearest_point_;
    double weight_;
};

class LidarVisionAlignmentCostFunction {
public:
    LidarVisionAlignmentCostFunction(const Eigen::Vector4d& lidar_qvec,
                                     const Eigen::Vector3d& lidar_tvec,
                                     const Eigen::Vector4d& qvec,
                                     const Eigen::Vector3d& tvec)
    : lqvec_(lidar_qvec),
      ltvec_(lidar_tvec),
      qvec_(qvec),
      tvec_(tvec) {}

    static ceres::CostFunction* Create(const Eigen::Vector4d& lidar_qvec,
                                       const Eigen::Vector3d& lidar_tvec,
                                       const Eigen::Vector4d& qvec,
                                       const Eigen::Vector3d& tvec) {
        return new ceres::AutoDiffCostFunction<LidarVisionAlignmentCostFunction, 6, 1, 4, 3>(
            new LidarVisionAlignmentCostFunction(lidar_qvec, lidar_tvec, qvec, tvec)
        );
    }
    
    template <typename T>
    bool operator()(const T* const s, const T* const rel_qvec, const T* const rel_tvec, T* residuals) const {
        T qvec[4] = {T(qvec_[0]), T(qvec_[1]), T(qvec_[2]), T(qvec_[2])};
        T tvec[3] = {T(tvec_[0]), T(tvec_[1]), T(tvec_[2])};
        // T t_qvec[4];
        // ceres::QuaternionProduct(rel_qvec, qvec, t_qvec);
        // T t_tvec[3];
        // ceres::QuaternionRotatePoint(rel_qvec, tvec, t_tvec);
        // t_tvec[0] += rel_tvec[0];
        // t_tvec[1] += rel_tvec[1];
        // t_tvec[2] += rel_tvec[2];

        T rel_qvec_inv[4] = {rel_qvec[0], -rel_qvec[1], -rel_qvec[2], -rel_qvec[3]};
        T t_qvec[4];
        ceres::QuaternionProduct(qvec, rel_qvec_inv, t_qvec);
        T t_qvec_inv[4] = {t_qvec[0], -t_qvec[1], -t_qvec[2], -t_qvec[3]};
        T t_tvec[3];
        ceres::QuaternionRotatePoint(t_qvec, rel_tvec, t_tvec);
        t_tvec[0] = -t_tvec[0] + tvec[0] * *s;
        t_tvec[1] = -t_tvec[1] + tvec[1] * *s;
        t_tvec[2] = -t_tvec[2] + tvec[2] * *s;

        T relative_q_diff[4];
        T lidar_qvec[4] = {T(lqvec_[0]), T(lqvec_[1]), T(lqvec_[2]), T(lqvec_[3])};
        T lidar_qvec_inv[4] = {T(lqvec_[0]), T(-lqvec_[1]), T(-lqvec_[2]), T(-lqvec_[3])};

        ceres::QuaternionProduct(lidar_qvec, t_qvec_inv, relative_q_diff);

        T lidar_tvec[3] = { T(ltvec_[0]), T(ltvec_[1]), T(ltvec_[2]) };
        T lidar_C[3];
        ceres::QuaternionRotatePoint(lidar_qvec_inv, lidar_tvec, lidar_C);
        lidar_C[0] = -lidar_C[0];
        lidar_C[1] = -lidar_C[1];
        lidar_C[2] = -lidar_C[2];

        T C[3];
        ceres::QuaternionRotatePoint(t_qvec_inv, t_tvec, C);
        C[0] = -C[0];
        C[1] = -C[1];
        C[2] = -C[2];

        residuals[0] = T(2.0) * relative_q_diff[1];
        residuals[1] = T(2.0) * relative_q_diff[2];
        residuals[2] = T(2.0) * relative_q_diff[3];
        residuals[3] = (C[0] - lidar_C[0]);
        residuals[4] = (C[1] - lidar_C[1]);
        residuals[5] = (C[2] - lidar_C[2]);

        return true;
    }
private:
    Eigen::Vector4d lqvec_;
    Eigen::Vector3d ltvec_;
    Eigen::Vector4d qvec_;
    Eigen::Vector3d tvec_;
};

}  // namespace sensemap

#endif  // SENSEMAP_BASE_COST_FUNCTIONS_H_
