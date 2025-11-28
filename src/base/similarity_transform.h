//Copyright (c) 2019, SenseTime Group.
//All rights reserved.
#ifndef SENSEMAP_BASE_SIMILARITY_TRANSFORM_H_
#define SENSEMAP_BASE_SIMILARITY_TRANSFORM_H_

#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include "util/alignment.h"
#include "util/types.h"
#include "base/reconstruction.h"
#include "ceres/rotation.h"
#include "projection.h"

namespace sensemap {

class Cluster;

// 3D similarity transformation with 7 degrees of freedom.
class SimilarityTransform3 {
public:
  	EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  	SimilarityTransform3();

  	explicit SimilarityTransform3(const Eigen::Matrix3x4d& matrix);

  	explicit SimilarityTransform3(
	  	const Eigen::Transform<double, 3, Eigen::Affine>& transform);

  	SimilarityTransform3(const double scale, const Eigen::Vector4d& qvec,
					   const Eigen::Vector3d& tvec);

  	void Estimate(const std::vector<Eigen::Vector3d>& src,
				  const std::vector<Eigen::Vector3d>& dst);

  	SimilarityTransform3 Inverse() const;

  	void TransformPoint(Eigen::Vector3d* xyz) const;
  	void TransformPose(Eigen::Vector4d* qvec, Eigen::Vector3d* tvec) const;

  	Eigen::Matrix4d Matrix() const;
  	double Scale() const;
  	Eigen::Vector4d Rotation() const;
  	Eigen::Vector3d Translation() const;

private:
  	Eigen::Transform<double, 3, Eigen::Affine> transform_;

};

// Template SimilarityTransform3 used for optimization
template <typename T>
class TSimilarityTransform3
{
    typedef Eigen::Quaternion<T> TQuat;
    typedef Eigen::Matrix<T, 3, 1> TVec3;
    typedef Eigen::Matrix<T, 3, 3> TMat3;

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    TSimilarityTransform3()
    {
        r_ = TQuat(T(1), T(0), T(0), T(0));
        t_.fill(T(0));
        s_ = T(1);
    }

    TSimilarityTransform3(const TQuat &r, const TVec3 &t, T s)
            : r_(r), t_(t), s_(s)
    {
        r_.normalize();
    }

    TSimilarityTransform3(const T *spose)
    {
        const T &a0 = spose[0];
        const T &a1 = spose[1];
        const T &a2 = spose[2];
        const T theta_squared = a0 * a0 + a1 * a1 + a2 * a2;

        T q0, q1, q2, q3;
        if (theta_squared > T(1e-5))
        {
            const T theta = sqrt(theta_squared);
            const T half_theta = theta * T(0.5);
            const T k = sin(half_theta) / theta;
            q0 = cos(half_theta);
            q1 = a0 * k;
            q2 = a1 * k;
            q3 = a2 * k;
        }
        else
        {
            const T k(0.5);
            q0 = T(1.0);
            q1 = a0 * k;
            q2 = a1 * k;
            q3 = a2 * k;
        }
        r_ = TQuat(q0, q1, q2, q3);
        t_[0] = spose[3];
        t_[1] = spose[4];
        t_[2] = spose[5];
        s_ = spose[6];
    }

    void GetLog(T *u) const
    {
        T eps(1e-5);
        T lambda = log(s_);
        T lambda2 = lambda * lambda;
        T lambda_abs = (lambda < T(0)) ? -lambda : lambda;

        T w[3], q[4];
        q[0] = r_.w();
        q[1] = r_.x();
        q[2] = r_.y();
        q[3] = r_.z();
        ceres::QuaternionToAngleAxis(q, w);
        const T &w0 = w[0];
        const T &w1 = w[1];
        const T &w2 = w[2];
        TMat3 W; // -- Construct skew matrix
        W << T(0), -w2, w1, w2, T(0), -w0, -w1, w0, T(0);
        TMat3 W2 = W * W;
        T a2 = w0 * w0 + w1 * w1 + w2 * w2;
        // The derivative of sqrt(0) (i.e., sqrt'(0)) is infinite.
        // Be sure to avoid it.
        T a = (a2 < eps * eps) ? T(0) : sqrt(a2);
        T sa = sin(a);
        T ca = cos(a);
        TMat3 I = TMat3::Identity();
        TMat3 R = r_.toRotationMatrix();

        T c0, c1, c2;

        if (a < eps)
        {
            R = I + W + T(0.5) * W2;
            if (lambda_abs < eps)
                c0 = T(1);
            else
                c0 = (s_ - T(1)) / lambda;
            c1 = (T(3) * s_ * ca - lambda * s_ * ca - a * s_ * sa) / T(6);
            c2 = s_ * a / T(6) - lambda * s_ * ca / T(24);
        }
        else
        {
            R = I + W * sa / a + W2 * (T(1) - ca) / a2;
            if (lambda_abs < eps)
            {
                c0 = T(1);
                c1 = (T(2) * s_ * sa - a * s_ * ca + lambda * s_ * sa) / (T(2) * a);
                c2 = s_ / a2 - s_ * sa / (T(2) * a) - (T(2) * s_ * ca + lambda * s_ * ca) / (T(2) * a2);
            }
            else
            {
                c0 = (s_ - T(1)) / lambda;
                c1 = (a * (T(1) - s_ * ca) + s_ * lambda * sa) / (a * (lambda2 + a2));
                c2 = (s_ - T(1)) / (lambda * a2) - (s_ * sa) / (a * (lambda2 + a2)) -
                     (lambda * (s_ * ca - T(1))) / (a2 * (lambda2 + a2));
            }
        }
        TMat3 V = c0 * I + c1 * W + c2 * W2;
        TVec3 v = V.lu().solve(t_);

        for (int i = 0; i < 3; ++i)
        {
            u[i] = v[i];
            u[i + 3] = w[i];
        }
        u[6] = lambda;
    }

    TSimilarityTransform3 Inverse() const
    {
        return TSimilarityTransform3(r_.conjugate(), r_.conjugate() * ((-1. / s_) * t_), 1. / s_);
    }

    TSimilarityTransform3 operator*(const TSimilarityTransform3 &u) const
    {
        TSimilarityTransform3 result;
        result.r_ = r_ * u.r_;
        result.t_ = s_ * (r_ * u.t_) + t_;
        result.s_ = s_ * u.s_;
        return result;
    }

protected:
    TQuat r_;
    TVec3 t_;
    T s_;
};

struct PointCloudAlignmentEstimator {
    static const int kMinNumSamples = 3;
    
    typedef mappoint_t X_t;
    typedef mappoint_t Y_t;
    typedef Eigen::Matrix3x4d M_t;

    void SetMaxReprojError(const double max_reproj_error) {
        max_squared_reproj_error_ = max_reproj_error * max_reproj_error;
    }

    void SetReconstructions(const Reconstruction* reconstruction1,
                            const Reconstruction* reconstruction2) {
        CHECK_NOTNULL(reconstruction1);
        CHECK_NOTNULL(reconstruction2);
        reconstruction1_ = reconstruction1;
        reconstruction2_ = reconstruction2;
    }

    std::vector<M_t> Estimate(const std::vector<X_t>& mappoint_ids1,
                              const std::vector<Y_t>& mappoint_ids2) const {
        CHECK_GE(mappoint_ids1.size(), 3);
        CHECK_GE(mappoint_ids2.size(), 3);

        std::vector<Eigen::Vector3d> points3D1(mappoint_ids1.size());
        std::vector<Eigen::Vector3d> points3D2(mappoint_ids2.size());
        for (size_t i = 0; i < mappoint_ids1.size(); ++i) {
            points3D1[i] = reconstruction1_->MapPoint(mappoint_ids1[i]).XYZ();
            points3D2[i] = reconstruction2_->MapPoint(mappoint_ids2[i]).XYZ();
        }

        SimilarityTransform3 tform12;
        tform12.Estimate(points3D1, points3D2);

        return {tform12.Matrix().topRows<3>()};
    }

    void Residuals(const std::vector<X_t>& mappoint_ids1,
                   const std::vector<Y_t>& mappoint_ids2,
                   const M_t& alignment12,
                   std::vector<double>* residuals) const {
        CHECK_EQ(mappoint_ids1.size(), mappoint_ids2.size());

        const Eigen::Matrix3x4d alignment21 =
            SimilarityTransform3(alignment12).Inverse().Matrix().topRows<3>();

        residuals->resize(mappoint_ids1.size());

        for (size_t i = 0; i < mappoint_ids1.size(); ++i) {
            const class MapPoint& mappoint1 = reconstruction1_->MapPoint(mappoint_ids1[i]);
            const class MapPoint& mappoint2 = reconstruction2_->MapPoint(mappoint_ids2[i]);

            const auto& point3D1 = mappoint1.XYZ();
            const auto& point3D2 = mappoint2.XYZ();

            std::vector<double> reproj_errors1, reproj_errors2;
            // Project mappoint1 to image2
            for (const auto & track_el : mappoint2.Track().Elements()) {
                const class Image& image2 = reconstruction2_->Image(track_el.image_id);
                const class Point2D& point2D2 = image2.Point2D(track_el.point2D_idx);

                const class Camera& camera2 = reconstruction2_->Camera(image2.CameraId());
                Eigen::Matrix3x4d proj_matrix2 = image2.ProjectionMatrix();

                const Eigen::Vector3d xyz12 = alignment12 * point3D1.homogeneous();
                double reproj_error2 = 
                    CalculateSquaredReprojectionError(point2D2.XY(), xyz12, proj_matrix2, camera2);
                reproj_errors2.push_back(reproj_error2);
            }

            // Project mappoint2 to image1
            for (const auto & track_el : mappoint1.Track().Elements()) {
                const class Image& image1 = reconstruction1_->Image(track_el.image_id);
                const class Point2D& point2D1 = image1.Point2D(track_el.point2D_idx);

                const class Camera& camera1 = reconstruction1_->Camera(image1.CameraId());
                Eigen::Matrix3x4d proj_matrix1 = image1.ProjectionMatrix();

                const Eigen::Vector3d xyz21 = alignment21 * point3D2.homogeneous();
                double reproj_error1 = 
                    CalculateSquaredReprojectionError(point2D1.XY(), xyz21, proj_matrix1, camera1);
                reproj_errors1.push_back(reproj_error1);
            }

            int nth1 = reproj_errors1.size() / 2;
            std::nth_element(reproj_errors1.begin(), reproj_errors1.begin() + nth1, reproj_errors1.end());
            int nth2 = reproj_errors2.size() / 2;
            std::nth_element(reproj_errors2.begin(), reproj_errors2.begin() + nth2, reproj_errors2.end());

            double mid_reproj_error1 = reproj_errors1.size() > 0 ? 
                reproj_errors1[nth1] : max_squared_reproj_error_ + 1.0;
            double mid_reproj_error2 = reproj_errors2.size() > 0 ? 
                reproj_errors2[nth2] : max_squared_reproj_error_ + 1.0;

            if (mid_reproj_error1 > max_squared_reproj_error_ ||
                mid_reproj_error2 > max_squared_reproj_error_) {
                (*residuals)[i] = 1.0;
            } else {
                (*residuals)[i] = std::max(mid_reproj_error1, mid_reproj_error2) / 
                    max_squared_reproj_error_;
            }
        }
    }

private:
    double max_squared_reproj_error_ = 0.0;
    const Reconstruction* reconstruction1_ = nullptr;
    const Reconstruction* reconstruction2_ = nullptr;

};


// Robustly compute alignment between reconstructions by finding images that
// are registered in both reconstructions. The alignment is then estimated
// robustly inside RANSAC from corresponding projection centers. An alignment
// is verified by reprojecting common 3D point observations.
// The min_inlier_observations threshold determines how many observations
// in a common image must reproject within the given threshold.
bool ComputeAlignmentBetweenReconstructions(
    const Reconstruction& src_reconstruction,
    const Reconstruction& ref_reconstruction,
    const double min_inlier_observations,
    const double max_reproj_error,
    Eigen::Matrix3x4d* alignment);

// Compute alignment between reconstructions by finding map points that observed
// by botn reconstructions. The alignment is then estimated robustly inside 
// RANSAC from correspondence map points.
bool ComputeAlignmentBetweenReconstructions(
    const Reconstruction& src_reconstruction,
    const Reconstruction& ref_reconstruction,
    const CorrespondenceGraph& correspondence_graph,
    const double min_inlier_observations,
    const double max_reproj_error,
    Eigen::Matrix3x4d* alignment);

}  // namespace colmap

// EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION_CUSTOM(colmap::SimilarityTransform3)

#endif  // SENSEMAP_BASE_SIMILARITY_TRANSFORM_H_
