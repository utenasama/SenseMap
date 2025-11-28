// Copyright (c) 2019, SenseTime Group.
// All rights reserved.

#include "util/ceres_types.h"
#include "util/logging.h"
#include "util/misc.h"
#include "base/polynomial.h"
#include "base/pose.h"
#include "base/matrix.h"
#include "base/camera_models.h"
#include "base/cost_functions.h"
#include "estimators/utils.h"
#include "optim/bundle_adjustment.h"

#include "absolute_pose.h"

namespace sensemap {

namespace {

typedef LORANSAC<P3PEstimator, EPNPEstimator> AbsolutePoseRANSAC;
typedef LORANSAC<P3PEstimatorSpherical, EPNPEstimatorSpherical> AbsolutePoseSphericalRANSAC;
Eigen::Vector3d LiftImagePoint(const Eigen::Vector2d& point) {
    return point.homogeneous() / std::sqrt(point.squaredNorm() + 1);
}

void EstimateAbsolutePoseKernel(const Camera& camera, const double focal_length_factor,
                                const std::vector<Eigen::Vector2d>& points2D,
                                const std::vector<Eigen::Vector3d>& points3D, const RANSACOptions& options,
                                AbsolutePoseRANSAC::Report* report) {
    // Scale the focal length by the given factor.
    Camera scaled_camera = camera;
    const std::vector<size_t>& focal_length_idxs = camera.FocalLengthIdxs();
    for (const size_t idx : focal_length_idxs) {
        scaled_camera.Params(idx) *= focal_length_factor;
    }

    // Normalize image coordinates with current camera hypothesis.
    std::vector<Eigen::Vector2d> points2D_N(points2D.size());
    for (size_t i = 0; i < points2D.size(); ++i) {
        points2D_N[i] = scaled_camera.ImageToWorld(points2D[i]);
    }
    // Estimate pose for given focal length.
    auto custom_options = options;
    custom_options.max_error = scaled_camera.ImageToWorldThreshold(options.max_error);
    AbsolutePoseRANSAC ransac(custom_options);

    // *report = ransac.Estimate(points2D_N, points3D);
    *report = ransac.EstimateMultiple(points2D_N, points3D);
}

void EstimateAbsolutePoseSphericalKernel(const Camera& camera, const double focal_length_factor,
                                         const std::vector<Eigen::Vector2d>& points2D,
                                         const std::vector<Eigen::Vector3d>& points3D, const RANSACOptions& options,
                                         AbsolutePoseSphericalRANSAC::Report* report) {
    // Scale the focal length by the given factor.
    Camera scaled_camera = camera;
    const std::vector<size_t>& focal_length_idxs = camera.FocalLengthIdxs();
    for (const size_t idx : focal_length_idxs) {
        scaled_camera.Params(idx) *= focal_length_factor;
    }

    // Normalize image coordinates with current camera hypothesis.
    std::vector<Eigen::Vector3d> points2D_bearing(points2D.size());
    for (size_t i = 0; i < points2D.size(); ++i) {
        points2D_bearing[i] = scaled_camera.ImageToBearing(points2D[i]);
    }
    // Estimate pose for given focal length.
    auto custom_options = options;
    custom_options.max_error = scaled_camera.ImageToWorldThreshold(options.max_error);
    AbsolutePoseSphericalRANSAC ransac(custom_options);

    *report = ransac.EstimateMultiple(points2D_bearing, points3D);
}

}  // namespace

std::vector<P3PEstimator::M_t> P3PEstimator::Estimate(const std::vector<X_t>& points2D,
                                                      const std::vector<Y_t>& points3D) {
    CHECK_EQ(points2D.size(), 3);
    CHECK_EQ(points3D.size(), 3);

    Eigen::Matrix3d points3D_world;
    points3D_world.col(0) = points3D[0];
    points3D_world.col(1) = points3D[1];
    points3D_world.col(2) = points3D[2];

    const Eigen::Vector3d u = LiftImagePoint(points2D[0]);
    const Eigen::Vector3d v = LiftImagePoint(points2D[1]);
    const Eigen::Vector3d w = LiftImagePoint(points2D[2]);

    // Angles between 2D points.
    const double cos_uv = u.transpose() * v;
    const double cos_uw = u.transpose() * w;
    const double cos_vw = v.transpose() * w;

    // Distances between 2D points.
    const double dist_AB_2 = (points3D[0] - points3D[1]).squaredNorm();
    const double dist_AC_2 = (points3D[0] - points3D[2]).squaredNorm();
    const double dist_BC_2 = (points3D[1] - points3D[2]).squaredNorm();

    const double dist_AB = std::sqrt(dist_AB_2);

    const double a = dist_BC_2 / dist_AB_2;
    const double b = dist_AC_2 / dist_AB_2;

    // Helper variables for calculation of coefficients.
    const double a2 = a * a;
    const double b2 = b * b;
    const double p = 2 * cos_vw;
    const double q = 2 * cos_uw;
    const double r = 2 * cos_uv;
    const double p2 = p * p;
    const double p3 = p2 * p;
    const double q2 = q * q;
    const double r2 = r * r;
    const double r3 = r2 * r;
    const double r4 = r3 * r;
    const double r5 = r4 * r;

    // Build polynomial coefficients: a4*x^4 + a3*x^3 + a2*x^2 + a1*x + a0 = 0.
    Eigen::Matrix<double, 5, 1> coeffs;
    coeffs(0) = -2 * b + b2 + a2 + 1 + a * b * (2 - r2) - 2 * a;
    coeffs(1) = -2 * q * a2 - r * p * b2 + 4 * q * a + (2 * q + p * r) * b + (r2 * q - 2 * q + r * p) * a * b - 2 * q;
    coeffs(2) = (2 + q2) * a2 + (p2 + r2 - 2) * b2 - (4 + 2 * q2) * a - (p * q * r + p2) * b -
                (p * q * r + r2) * a * b + q2 + 2;
    coeffs(3) = -2 * q * a2 - r * p * b2 + 4 * q * a + (p * r + q * p2 - 2 * q) * b + (r * p + 2 * q) * a * b - 2 * q;
    coeffs(4) = a2 + b2 - 2 * a + (2 - p2) * b - 2 * a * b + 1;

    Eigen::VectorXd roots_real;
    Eigen::VectorXd roots_imag;
    if (!FindPolynomialRootsCompanionMatrix(coeffs, &roots_real, &roots_imag)) {
        return std::vector<P3PEstimator::M_t>({});
    }

    std::vector<M_t> models;
    models.reserve(roots_real.size());

    for (Eigen::VectorXd::Index i = 0; i < roots_real.size(); ++i) {
        const double kMaxRootImag = 1e-10;
        if (std::abs(roots_imag(i)) > kMaxRootImag) {
            continue;
        }

        const double x = roots_real(i);
        if (x < 0) {
            continue;
        }

        const double x2 = x * x;
        const double x3 = x2 * x;

        // Build polynomial coefficients: b1*y + b0 = 0.
        const double bb1 = (p2 - p * q * r + r2) * a + (p2 - r2) * b - p2 + p * q * r - r2;
        const double b1 = b * bb1 * bb1;
        const double b0 =
            ((1 - a - b) * x2 + (a - 1) * q * x - a + b + 1) *
            (r3 * (a2 + b2 - 2 * a - 2 * b + (2 - r2) * a * b + 1) * x3 +
             r2 *
                 (p + p * a2 - 2 * r * q * a * b + 2 * r * q * b - 2 * r * q - 2 * p * a - 2 * p * b + p * r2 * b +
                  4 * r * q * a + q * r3 * a * b - 2 * r * q * a2 + 2 * p * a * b + p * b2 - r2 * p * b2) *
                 x2 +
             (r5 * (b2 - a * b) - r4 * p * q * b + r3 * (q2 - 4 * a - 2 * q2 * a + q2 * a2 + 2 * a2 - 2 * b2 + 2) +
              r2 * (4 * p * q * a - 2 * p * q * a * b + 2 * p * q * b - 2 * p * q - 2 * p * q * a2) +
              r * (p2 * b2 - 2 * p2 * b + 2 * p2 * a * b - 2 * p2 * a + p2 + p2 * a2)) *
                 x +
             (2 * p * r2 - 2 * r3 * q + p3 - 2 * p2 * q * r + p * q2 * r2) * a2 + (p3 - 2 * p * r2) * b2 +
             (4 * q * r3 - 4 * p * r2 - 2 * p3 + 4 * p2 * q * r - 2 * p * q2 * r2) * a +
             (-2 * q * r3 + p * r4 + 2 * p2 * q * r - 2 * p3) * b + (2 * p3 + 2 * q * r3 - 2 * p2 * q * r) * a * b +
             p * q2 * r2 - 2 * p2 * q * r + 2 * p * r2 + p3 - 2 * r3 * q);

        // Solve for y.
        const double y = b0 / b1;
        const double y2 = y * y;

        const double nu = x2 + y2 - 2 * x * y * cos_uv;

        const double dist_PC = dist_AB / std::sqrt(nu);
        const double dist_PB = y * dist_PC;
        const double dist_PA = x * dist_PC;

        Eigen::Matrix3d points3D_camera;
        points3D_camera.col(0) = u * dist_PA;  // A'
        points3D_camera.col(1) = v * dist_PB;  // B'
        points3D_camera.col(2) = w * dist_PC;  // C'

        // Find transformation from the world to the camera system.
        const Eigen::Matrix4d transform = Eigen::umeyama(points3D_world, points3D_camera, false);
        models.push_back(transform.topLeftCorner<3, 4>());
    }

    return models;
}

void P3PEstimator::Residuals(const std::vector<X_t>& points2D, const std::vector<Y_t>& points3D, const M_t& proj_matrix,
                             std::vector<double>* residuals) {
    ComputeSquaredReprojectionError(points2D, points3D, proj_matrix, residuals);
}

std::vector<P3PEstimatorSpherical::M_t> P3PEstimatorSpherical::Estimate(const std::vector<X_t>& points2D_bearing,
                                                                        const std::vector<Y_t>& points3D) {
    CHECK_EQ(points2D_bearing.size(), 3);
    CHECK_EQ(points3D.size(), 3);

    Eigen::Matrix3d points3D_world;
    points3D_world.col(0) = points3D[0];
    points3D_world.col(1) = points3D[1];
    points3D_world.col(2) = points3D[2];

    const Eigen::Vector3d u = points2D_bearing[0];
    const Eigen::Vector3d v = points2D_bearing[1];
    const Eigen::Vector3d w = points2D_bearing[2];

    // Angles between 2D points.
    const double cos_uv = u.transpose() * v;
    const double cos_uw = u.transpose() * w;
    const double cos_vw = v.transpose() * w;

    // Distances between 2D points.
    const double dist_AB_2 = (points3D[0] - points3D[1]).squaredNorm();
    const double dist_AC_2 = (points3D[0] - points3D[2]).squaredNorm();
    const double dist_BC_2 = (points3D[1] - points3D[2]).squaredNorm();

    const double dist_AB = std::sqrt(dist_AB_2);

    const double a = dist_BC_2 / dist_AB_2;
    const double b = dist_AC_2 / dist_AB_2;

    // Helper variables for calculation of coefficients.
    const double a2 = a * a;
    const double b2 = b * b;
    const double p = 2 * cos_vw;
    const double q = 2 * cos_uw;
    const double r = 2 * cos_uv;
    const double p2 = p * p;
    const double p3 = p2 * p;
    const double q2 = q * q;
    const double r2 = r * r;
    const double r3 = r2 * r;
    const double r4 = r3 * r;
    const double r5 = r4 * r;

    // Build polynomial coefficients: a4*x^4 + a3*x^3 + a2*x^2 + a1*x + a0 = 0.
    Eigen::Matrix<double, 5, 1> coeffs;
    coeffs(0) = -2 * b + b2 + a2 + 1 + a * b * (2 - r2) - 2 * a;
    coeffs(1) = -2 * q * a2 - r * p * b2 + 4 * q * a + (2 * q + p * r) * b + (r2 * q - 2 * q + r * p) * a * b - 2 * q;
    coeffs(2) = (2 + q2) * a2 + (p2 + r2 - 2) * b2 - (4 + 2 * q2) * a - (p * q * r + p2) * b -
                (p * q * r + r2) * a * b + q2 + 2;
    coeffs(3) = -2 * q * a2 - r * p * b2 + 4 * q * a + (p * r + q * p2 - 2 * q) * b + (r * p + 2 * q) * a * b - 2 * q;
    coeffs(4) = a2 + b2 - 2 * a + (2 - p2) * b - 2 * a * b + 1;

    Eigen::VectorXd roots_real;
    Eigen::VectorXd roots_imag;
    if (!FindPolynomialRootsCompanionMatrix(coeffs, &roots_real, &roots_imag)) {
        return std::vector<P3PEstimator::M_t>({});
    }

    std::vector<M_t> models;
    models.reserve(roots_real.size());

    for (Eigen::VectorXd::Index i = 0; i < roots_real.size(); ++i) {
        const double kMaxRootImag = 1e-10;
        if (std::abs(roots_imag(i)) > kMaxRootImag) {
            continue;
        }

        const double x = roots_real(i);
        if (x < 0) {
            continue;
        }

        const double x2 = x * x;
        const double x3 = x2 * x;

        // Build polynomial coefficients: b1*y + b0 = 0.
        const double bb1 = (p2 - p * q * r + r2) * a + (p2 - r2) * b - p2 + p * q * r - r2;
        const double b1 = b * bb1 * bb1;
        const double b0 =
            ((1 - a - b) * x2 + (a - 1) * q * x - a + b + 1) *
            (r3 * (a2 + b2 - 2 * a - 2 * b + (2 - r2) * a * b + 1) * x3 +
             r2 *
                 (p + p * a2 - 2 * r * q * a * b + 2 * r * q * b - 2 * r * q - 2 * p * a - 2 * p * b + p * r2 * b +
                  4 * r * q * a + q * r3 * a * b - 2 * r * q * a2 + 2 * p * a * b + p * b2 - r2 * p * b2) *
                 x2 +
             (r5 * (b2 - a * b) - r4 * p * q * b + r3 * (q2 - 4 * a - 2 * q2 * a + q2 * a2 + 2 * a2 - 2 * b2 + 2) +
              r2 * (4 * p * q * a - 2 * p * q * a * b + 2 * p * q * b - 2 * p * q - 2 * p * q * a2) +
              r * (p2 * b2 - 2 * p2 * b + 2 * p2 * a * b - 2 * p2 * a + p2 + p2 * a2)) *
                 x +
             (2 * p * r2 - 2 * r3 * q + p3 - 2 * p2 * q * r + p * q2 * r2) * a2 + (p3 - 2 * p * r2) * b2 +
             (4 * q * r3 - 4 * p * r2 - 2 * p3 + 4 * p2 * q * r - 2 * p * q2 * r2) * a +
             (-2 * q * r3 + p * r4 + 2 * p2 * q * r - 2 * p3) * b + (2 * p3 + 2 * q * r3 - 2 * p2 * q * r) * a * b +
             p * q2 * r2 - 2 * p2 * q * r + 2 * p * r2 + p3 - 2 * r3 * q);

        // Solve for y.
        const double y = b0 / b1;
        const double y2 = y * y;

        const double nu = x2 + y2 - 2 * x * y * cos_uv;

        const double dist_PC = dist_AB / std::sqrt(nu);
        const double dist_PB = y * dist_PC;
        const double dist_PA = x * dist_PC;

        Eigen::Matrix3d points3D_camera;
        points3D_camera.col(0) = u * dist_PA;  // A'
        points3D_camera.col(1) = v * dist_PB;  // B'
        points3D_camera.col(2) = w * dist_PC;  // C'

        // Find transformation from the world to the camera system.
        const Eigen::Matrix4d transform = Eigen::umeyama(points3D_world, points3D_camera, false);
        models.push_back(transform.topLeftCorner<3, 4>());
    }

    return models;
}

void P3PEstimatorSpherical::Residuals(const std::vector<X_t>& points2D_bearing, const std::vector<Y_t>& points3D,
                                      const M_t& proj_matrix, std::vector<double>* residuals) {
    ComputeSquaredReprojectionErrorSpherical(points2D_bearing, points3D, proj_matrix, residuals);
}

std::vector<EPNPEstimator::M_t> EPNPEstimator::Estimate(const std::vector<X_t>& points2D,
                                                        const std::vector<Y_t>& points3D) {
    CHECK_GE(points2D.size(), 4);
    CHECK_EQ(points2D.size(), points3D.size());

    EPNPEstimator epnp;
    M_t proj_matrix;
    if (!epnp.ComputePose(points2D, points3D, &proj_matrix)) {
        return std::vector<EPNPEstimator::M_t>({});
    }

    return std::vector<EPNPEstimator::M_t>({proj_matrix});
}

void EPNPEstimator::Residuals(const std::vector<X_t>& points2D, const std::vector<Y_t>& points3D,
                              const M_t& proj_matrix, std::vector<double>* residuals) {
    ComputeSquaredReprojectionError(points2D, points3D, proj_matrix, residuals);
}

bool EPNPEstimator::ComputePose(const std::vector<Eigen::Vector2d>& points2D,
                                const std::vector<Eigen::Vector3d>& points3D, Eigen::Matrix3x4d* proj_matrix) {
    points2D_ = &points2D;
    points3D_ = &points3D;

    ChooseControlPoints();

    if (!ComputeBarycentricCoordinates()) {
        return false;
    }

    const Eigen::Matrix<double, Eigen::Dynamic, 12> M = ComputeM();
    const Eigen::Matrix<double, 12, 12> MtM = M.transpose() * M;

    Eigen::JacobiSVD<Eigen::Matrix<double, 12, 12>> svd(MtM, Eigen::ComputeFullV | Eigen::ComputeFullU);
    const Eigen::Matrix<double, 12, 12> Ut = svd.matrixU().transpose();

    const Eigen::Matrix<double, 6, 10> L6x10 = ComputeL6x10(Ut);
    const Eigen::Matrix<double, 6, 1> rho = ComputeRho();

    Eigen::Vector4d betas[4];
    std::array<double, 4> reproj_errors;
    std::array<Eigen::Matrix3d, 4> Rs;
    std::array<Eigen::Vector3d, 4> ts;

    FindBetasApprox1(L6x10, rho, &betas[1]);
    RunGaussNewton(L6x10, rho, &betas[1]);
    reproj_errors[1] = ComputeRT(Ut, betas[1], &Rs[1], &ts[1]);

    FindBetasApprox2(L6x10, rho, &betas[2]);
    RunGaussNewton(L6x10, rho, &betas[2]);
    reproj_errors[2] = ComputeRT(Ut, betas[2], &Rs[2], &ts[2]);

    FindBetasApprox3(L6x10, rho, &betas[3]);
    RunGaussNewton(L6x10, rho, &betas[3]);
    reproj_errors[3] = ComputeRT(Ut, betas[3], &Rs[3], &ts[3]);

    int best_idx = 1;
    if (reproj_errors[2] < reproj_errors[1]) {
        best_idx = 2;
    }
    if (reproj_errors[3] < reproj_errors[best_idx]) {
        best_idx = 3;
    }

    proj_matrix->leftCols<3>() = Rs[best_idx];
    proj_matrix->rightCols<1>() = ts[best_idx];

    return true;
}

void EPNPEstimator::ChooseControlPoints() {
    // Take C0 as the reference points centroid:
    cws_[0].setZero();
    for (size_t i = 0; i < points3D_->size(); ++i) {
        cws_[0] += (*points3D_)[i];
    }
    cws_[0] /= points3D_->size();

    Eigen::Matrix<double, Eigen::Dynamic, 3> PW0(points3D_->size(), 3);
    for (size_t i = 0; i < points3D_->size(); ++i) {
        PW0.row(i) = (*points3D_)[i] - cws_[0];
    }

    const Eigen::Matrix3d PW0tPW0 = PW0.transpose() * PW0;
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(PW0tPW0, Eigen::ComputeFullV | Eigen::ComputeFullU);
    const Eigen::Vector3d D = svd.singularValues();
    const Eigen::Matrix3d Ut = svd.matrixU().transpose();

    for (int i = 1; i < 4; ++i) {
        const double k = std::sqrt(D(i - 1) / points3D_->size());
        cws_[i] = cws_[0] + k * Ut.row(i - 1).transpose();
    }
}

bool EPNPEstimator::ComputeBarycentricCoordinates() {
    Eigen::Matrix3d CC;
    for (int i = 0; i < 3; ++i) {
        for (int j = 1; j < 4; ++j) {
            CC(i, j - 1) = cws_[j][i] - cws_[0][i];
        }
    }

    if (CC.colPivHouseholderQr().rank() < 3) {
        return false;
    }

    const Eigen::Matrix3d CC_inv = CC.inverse();

    alphas_.resize(points2D_->size());
    for (size_t i = 0; i < points3D_->size(); ++i) {
        for (int j = 0; j < 3; ++j) {
            alphas_[i][1 + j] = CC_inv(j, 0) * ((*points3D_)[i][0] - cws_[0][0]) +
                                CC_inv(j, 1) * ((*points3D_)[i][1] - cws_[0][1]) +
                                CC_inv(j, 2) * ((*points3D_)[i][2] - cws_[0][2]);
        }
        alphas_[i][0] = 1.0 - alphas_[i][1] - alphas_[i][2] - alphas_[i][3];
    }

    return true;
}

Eigen::Matrix<double, Eigen::Dynamic, 12> EPNPEstimator::ComputeM() {
    Eigen::Matrix<double, Eigen::Dynamic, 12> M(2 * points2D_->size(), 12);
    for (size_t i = 0; i < points3D_->size(); ++i) {
        for (size_t j = 0; j < 4; ++j) {
            M(2 * i, 3 * j) = alphas_[i][j];
            M(2 * i, 3 * j + 1) = 0.0;
            M(2 * i, 3 * j + 2) = -alphas_[i][j] * (*points2D_)[i].x();

            M(2 * i + 1, 3 * j) = 0.0;
            M(2 * i + 1, 3 * j + 1) = alphas_[i][j];
            M(2 * i + 1, 3 * j + 2) = -alphas_[i][j] * (*points2D_)[i].y();
        }
    }
    return M;
}

Eigen::Matrix<double, 6, 10> EPNPEstimator::ComputeL6x10(const Eigen::Matrix<double, 12, 12>& Ut) {
    Eigen::Matrix<double, 6, 10> L6x10;

    std::array<std::array<Eigen::Vector3d, 6>, 4> dv;
    for (int i = 0; i < 4; ++i) {
        int a = 0, b = 1;
        for (int j = 0; j < 6; ++j) {
            dv[i][j][0] = Ut(11 - i, 3 * a) - Ut(11 - i, 3 * b);
            dv[i][j][1] = Ut(11 - i, 3 * a + 1) - Ut(11 - i, 3 * b + 1);
            dv[i][j][2] = Ut(11 - i, 3 * a + 2) - Ut(11 - i, 3 * b + 2);

            b += 1;
            if (b > 3) {
                a += 1;
                b = a + 1;
            }
        }
    }

    for (int i = 0; i < 6; ++i) {
        L6x10(i, 0) = dv[0][i].transpose() * dv[0][i];
        L6x10(i, 1) = 2.0 * dv[0][i].transpose() * dv[1][i];
        L6x10(i, 2) = dv[1][i].transpose() * dv[1][i];
        L6x10(i, 3) = 2.0 * dv[0][i].transpose() * dv[2][i];
        L6x10(i, 4) = 2.0 * dv[1][i].transpose() * dv[2][i];
        L6x10(i, 5) = dv[2][i].transpose() * dv[2][i];
        L6x10(i, 6) = 2.0 * dv[0][i].transpose() * dv[3][i];
        L6x10(i, 7) = 2.0 * dv[1][i].transpose() * dv[3][i];
        L6x10(i, 8) = 2.0 * dv[2][i].transpose() * dv[3][i];
        L6x10(i, 9) = dv[3][i].transpose() * dv[3][i];
    }

    return L6x10;
}

Eigen::Matrix<double, 6, 1> EPNPEstimator::ComputeRho() {
    Eigen::Matrix<double, 6, 1> rho;
    rho[0] = (cws_[0] - cws_[1]).squaredNorm();
    rho[1] = (cws_[0] - cws_[2]).squaredNorm();
    rho[2] = (cws_[0] - cws_[3]).squaredNorm();
    rho[3] = (cws_[1] - cws_[2]).squaredNorm();
    rho[4] = (cws_[1] - cws_[3]).squaredNorm();
    rho[5] = (cws_[2] - cws_[3]).squaredNorm();
    return rho;
}

// betas10        = [B11 B12 B22 B13 B23 B33 B14 B24 B34 B44]
// betas_approx_1 = [B11 B12     B13         B14]

void EPNPEstimator::FindBetasApprox1(const Eigen::Matrix<double, 6, 10>& L6x10, const Eigen::Matrix<double, 6, 1>& rho,
                                     Eigen::Vector4d* betas) {
    Eigen::Matrix<double, 6, 4> L_6x4;
    for (int i = 0; i < 6; ++i) {
        L_6x4(i, 0) = L6x10(i, 0);
        L_6x4(i, 1) = L6x10(i, 1);
        L_6x4(i, 2) = L6x10(i, 3);
        L_6x4(i, 3) = L6x10(i, 6);
    }

    Eigen::JacobiSVD<Eigen::Matrix<double, 6, 4>> svd(L_6x4, Eigen::ComputeFullV | Eigen::ComputeFullU);
    Eigen::Matrix<double, 6, 1> Rho_temp = rho;
    const Eigen::Matrix<double, 4, 1> b4 = svd.solve(Rho_temp);

    if (b4[0] < 0) {
        (*betas)[0] = std::sqrt(-b4[0]);
        (*betas)[1] = -b4[1] / (*betas)[0];
        (*betas)[2] = -b4[2] / (*betas)[0];
        (*betas)[3] = -b4[3] / (*betas)[0];
    } else {
        (*betas)[0] = std::sqrt(b4[0]);
        (*betas)[1] = b4[1] / (*betas)[0];
        (*betas)[2] = b4[2] / (*betas)[0];
        (*betas)[3] = b4[3] / (*betas)[0];
    }
}

// betas10        = [B11 B12 B22 B13 B23 B33 B14 B24 B34 B44]
// betas_approx_2 = [B11 B12 B22                            ]

void EPNPEstimator::FindBetasApprox2(const Eigen::Matrix<double, 6, 10>& L6x10, const Eigen::Matrix<double, 6, 1>& rho,
                                     Eigen::Vector4d* betas) {
    Eigen::Matrix<double, 6, 3> L_6x3(6, 3);

    for (int i = 0; i < 6; ++i) {
        L_6x3(i, 0) = L6x10(i, 0);
        L_6x3(i, 1) = L6x10(i, 1);
        L_6x3(i, 2) = L6x10(i, 2);
    }

    Eigen::JacobiSVD<Eigen::Matrix<double, 6, 3>> svd(L_6x3, Eigen::ComputeFullV | Eigen::ComputeFullU);
    Eigen::Matrix<double, 6, 1> Rho_temp = rho;
    const Eigen::Matrix<double, 3, 1> b3 = svd.solve(Rho_temp);

    if (b3[0] < 0) {
        (*betas)[0] = std::sqrt(-b3[0]);
        (*betas)[1] = (b3[2] < 0) ? std::sqrt(-b3[2]) : 0.0;
    } else {
        (*betas)[0] = std::sqrt(b3[0]);
        (*betas)[1] = (b3[2] > 0) ? std::sqrt(b3[2]) : 0.0;
    }

    if (b3[1] < 0) {
        (*betas)[0] = -(*betas)[0];
    }

    (*betas)[2] = 0.0;
    (*betas)[3] = 0.0;
}

// betas10        = [B11 B12 B22 B13 B23 B33 B14 B24 B34 B44]
// betas_approx_3 = [B11 B12 B22 B13 B23                    ]

void EPNPEstimator::FindBetasApprox3(const Eigen::Matrix<double, 6, 10>& L6x10, const Eigen::Matrix<double, 6, 1>& rho,
                                     Eigen::Vector4d* betas) {
    Eigen::JacobiSVD<Eigen::Matrix<double, 6, 5>> svd(L6x10.leftCols<5>(), Eigen::ComputeFullV | Eigen::ComputeFullU);
    Eigen::Matrix<double, 6, 1> Rho_temp = rho;
    const Eigen::Matrix<double, 5, 1> b5 = svd.solve(Rho_temp);

    if (b5[0] < 0) {
        (*betas)[0] = std::sqrt(-b5[0]);
        (*betas)[1] = (b5[2] < 0) ? std::sqrt(-b5[2]) : 0.0;
    } else {
        (*betas)[0] = std::sqrt(b5[0]);
        (*betas)[1] = (b5[2] > 0) ? std::sqrt(b5[2]) : 0.0;
    }
    if (b5[1] < 0) {
        (*betas)[0] = -(*betas)[0];
    }
    (*betas)[2] = b5[3] / (*betas)[0];
    (*betas)[3] = 0.0;
}

void EPNPEstimator::RunGaussNewton(const Eigen::Matrix<double, 6, 10>& L6x10, const Eigen::Matrix<double, 6, 1>& rho,
                                   Eigen::Vector4d* betas) {
    Eigen::Matrix<double, 6, 4> A;
    Eigen::Matrix<double, 6, 1> b;

    const int kNumIterations = 5;
    for (int k = 0; k < kNumIterations; ++k) {
        for (int i = 0; i < 6; ++i) {
            A(i, 0) = 2 * L6x10(i, 0) * (*betas)[0] + L6x10(i, 1) * (*betas)[1] + L6x10(i, 3) * (*betas)[2] +
                      L6x10(i, 6) * (*betas)[3];
            A(i, 1) = L6x10(i, 1) * (*betas)[0] + 2 * L6x10(i, 2) * (*betas)[1] + L6x10(i, 4) * (*betas)[2] +
                      L6x10(i, 7) * (*betas)[3];
            A(i, 2) = L6x10(i, 3) * (*betas)[0] + L6x10(i, 4) * (*betas)[1] + 2 * L6x10(i, 5) * (*betas)[2] +
                      L6x10(i, 8) * (*betas)[3];
            A(i, 3) = L6x10(i, 6) * (*betas)[0] + L6x10(i, 7) * (*betas)[1] + L6x10(i, 8) * (*betas)[2] +
                      2 * L6x10(i, 9) * (*betas)[3];

            b(i) = rho[i] - (L6x10(i, 0) * (*betas)[0] * (*betas)[0] + L6x10(i, 1) * (*betas)[0] * (*betas)[1] +
                             L6x10(i, 2) * (*betas)[1] * (*betas)[1] + L6x10(i, 3) * (*betas)[0] * (*betas)[2] +
                             L6x10(i, 4) * (*betas)[1] * (*betas)[2] + L6x10(i, 5) * (*betas)[2] * (*betas)[2] +
                             L6x10(i, 6) * (*betas)[0] * (*betas)[3] + L6x10(i, 7) * (*betas)[1] * (*betas)[3] +
                             L6x10(i, 8) * (*betas)[2] * (*betas)[3] + L6x10(i, 9) * (*betas)[3] * (*betas)[3]);
        }

        const Eigen::Vector4d x = A.colPivHouseholderQr().solve(b);

        (*betas) += x;
    }
}

double EPNPEstimator::ComputeRT(const Eigen::Matrix<double, 12, 12>& Ut, const Eigen::Vector4d& betas,
                                Eigen::Matrix3d* R, Eigen::Vector3d* t) {
    ComputeCcs(betas, Ut);
    ComputePcs();

    SolveForSign();

    EstimateRT(R, t);

    return ComputeTotalReprojectionError(*R, *t);
}

void EPNPEstimator::ComputeCcs(const Eigen::Vector4d& betas, const Eigen::Matrix<double, 12, 12>& Ut) {
    for (int i = 0; i < 4; ++i) {
        ccs_[i][0] = ccs_[i][1] = ccs_[i][2] = 0.0;
    }

    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            for (int k = 0; k < 3; ++k) {
                ccs_[j][k] += betas[i] * Ut(11 - i, 3 * j + k);
            }
        }
    }
}

void EPNPEstimator::ComputePcs() {
    pcs_.resize(points2D_->size());
    for (size_t i = 0; i < points3D_->size(); ++i) {
        for (int j = 0; j < 3; ++j) {
            pcs_[i][j] = alphas_[i][0] * ccs_[0][j] + alphas_[i][1] * ccs_[1][j] + alphas_[i][2] * ccs_[2][j] +
                         alphas_[i][3] * ccs_[3][j];
        }
    }
}

void EPNPEstimator::SolveForSign() {
    if (pcs_[0][2] < 0.0 || pcs_[0][2] > 0.0) {
        for (int i = 0; i < 4; ++i) {
            ccs_[i] = -ccs_[i];
        }
        for (size_t i = 0; i < points3D_->size(); ++i) {
            pcs_[i] = -pcs_[i];
        }
    }
}

void EPNPEstimator::EstimateRT(Eigen::Matrix3d* R, Eigen::Vector3d* t) {
    Eigen::Vector3d pc0 = Eigen::Vector3d::Zero();
    Eigen::Vector3d pw0 = Eigen::Vector3d::Zero();

    for (size_t i = 0; i < points3D_->size(); ++i) {
        pc0 += pcs_[i];
        pw0 += (*points3D_)[i];
    }
    pc0 /= points3D_->size();
    pw0 /= points3D_->size();

    Eigen::Matrix3d abt = Eigen::Matrix3d::Zero();
    for (size_t i = 0; i < points3D_->size(); ++i) {
        for (int j = 0; j < 3; ++j) {
            abt(j, 0) += (pcs_[i][j] - pc0[j]) * ((*points3D_)[i][0] - pw0[0]);
            abt(j, 1) += (pcs_[i][j] - pc0[j]) * ((*points3D_)[i][1] - pw0[1]);
            abt(j, 2) += (pcs_[i][j] - pc0[j]) * ((*points3D_)[i][2] - pw0[2]);
        }
    }

    Eigen::JacobiSVD<Eigen::Matrix3d> svd(abt, Eigen::ComputeFullV | Eigen::ComputeFullU);
    const Eigen::Matrix3d abt_U = svd.matrixU();
    const Eigen::Matrix3d abt_V = svd.matrixV();

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            (*R)(i, j) = abt_U.row(i) * abt_V.row(j).transpose();
        }
    }

    if (R->determinant() < 0) {
        Eigen::Matrix3d Abt_v_prime = abt_V;
        Abt_v_prime.col(2) = -abt_V.col(2);
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                (*R)(i, j) = abt_U.row(i) * Abt_v_prime.row(j).transpose();
            }
        }
    }

    *t = pc0 - *R * pw0;
}

double EPNPEstimator::ComputeTotalReprojectionError(const Eigen::Matrix3d& R, const Eigen::Vector3d& t) {
    Eigen::Matrix3x4d proj_matrix;
    proj_matrix.leftCols<3>() = R;
    proj_matrix.rightCols<1>() = t;

    std::vector<double> residuals;
    ComputeSquaredReprojectionError(*points2D_, *points3D_, proj_matrix, &residuals);

    double reproj_error = 0.0;
    for (const double residual : residuals) {
        reproj_error += std::sqrt(residual);
    }

    return reproj_error;
}

std::vector<EPNPEstimatorSpherical::M_t> EPNPEstimatorSpherical::Estimate(const std::vector<X_t>& points2D_bearing,
                                                                          const std::vector<Y_t>& points3D) {
    CHECK_GE(points2D_bearing.size(), 4);
    CHECK_EQ(points2D_bearing.size(), points3D.size());

    EPNPEstimatorSpherical epnp;
    std::vector<Eigen::Vector2d> points2D;

    points2D.resize(points2D_bearing.size());
    for (size_t i = 0; i < points2D.size(); ++i) {
        points2D[i] = points2D_bearing[i].hnormalized();
    }
    epnp.points2D_bearing_ = &points2D_bearing;
    M_t proj_matrix;
    if (!epnp.ComputePose(points2D, points3D, &proj_matrix)) {
        return std::vector<EPNPEstimatorSpherical::M_t>({});
    }

    return std::vector<EPNPEstimatorSpherical::M_t>({proj_matrix});
}

void EPNPEstimatorSpherical::Residuals(const std::vector<X_t>& points2D_bearing, const std::vector<Y_t>& points3D,
                                       const M_t& proj_matrix, std::vector<double>* residuals) {
    ComputeSquaredReprojectionErrorSpherical(points2D_bearing, points3D, proj_matrix, residuals);
}

bool EPNPEstimatorSpherical::ComputePose(const std::vector<Eigen::Vector2d>& points2D,
                                         const std::vector<Eigen::Vector3d>& points3D, Eigen::Matrix3x4d* proj_matrix) {
    points2D_ = &points2D;
    points3D_ = &points3D;

    ChooseControlPoints();

    if (!ComputeBarycentricCoordinates()) {
        return false;
    }

    const Eigen::Matrix<double, Eigen::Dynamic, 12> M = ComputeM();
    const Eigen::Matrix<double, 12, 12> MtM = M.transpose() * M;

    Eigen::JacobiSVD<Eigen::Matrix<double, 12, 12>> svd(MtM, Eigen::ComputeFullV | Eigen::ComputeFullU);
    const Eigen::Matrix<double, 12, 12> Ut = svd.matrixU().transpose();

    const Eigen::Matrix<double, 6, 10> L6x10 = ComputeL6x10(Ut);
    const Eigen::Matrix<double, 6, 1> rho = ComputeRho();

    Eigen::Vector4d betas[4];
    std::array<double, 4> reproj_errors;
    std::array<Eigen::Matrix3d, 4> Rs;
    std::array<Eigen::Vector3d, 4> ts;

    FindBetasApprox1(L6x10, rho, &betas[1]);
    RunGaussNewton(L6x10, rho, &betas[1]);
    reproj_errors[1] = ComputeRT(Ut, betas[1], &Rs[1], &ts[1]);

    FindBetasApprox2(L6x10, rho, &betas[2]);
    RunGaussNewton(L6x10, rho, &betas[2]);
    reproj_errors[2] = ComputeRT(Ut, betas[2], &Rs[2], &ts[2]);

    FindBetasApprox3(L6x10, rho, &betas[3]);
    RunGaussNewton(L6x10, rho, &betas[3]);
    reproj_errors[3] = ComputeRT(Ut, betas[3], &Rs[3], &ts[3]);

    int best_idx = 1;
    if (reproj_errors[2] < reproj_errors[1]) {
        best_idx = 2;
    }
    if (reproj_errors[3] < reproj_errors[best_idx]) {
        best_idx = 3;
    }

    proj_matrix->leftCols<3>() = Rs[best_idx];
    proj_matrix->rightCols<1>() = ts[best_idx];

    return true;
}

void EPNPEstimatorSpherical::ChooseControlPoints() {
    // Take C0 as the reference points centroid:
    cws_[0].setZero();
    for (size_t i = 0; i < points3D_->size(); ++i) {
        cws_[0] += (*points3D_)[i];
    }
    cws_[0] /= points3D_->size();

    Eigen::Matrix<double, Eigen::Dynamic, 3> PW0(points3D_->size(), 3);
    for (size_t i = 0; i < points3D_->size(); ++i) {
        PW0.row(i) = (*points3D_)[i] - cws_[0];
    }

    const Eigen::Matrix3d PW0tPW0 = PW0.transpose() * PW0;
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(PW0tPW0, Eigen::ComputeFullV | Eigen::ComputeFullU);
    const Eigen::Vector3d D = svd.singularValues();
    const Eigen::Matrix3d Ut = svd.matrixU().transpose();

    for (int i = 1; i < 4; ++i) {
        const double k = std::sqrt(D(i - 1) / points3D_->size());
        cws_[i] = cws_[0] + k * Ut.row(i - 1).transpose();
    }
}

bool EPNPEstimatorSpherical::ComputeBarycentricCoordinates() {
    Eigen::Matrix3d CC;
    for (int i = 0; i < 3; ++i) {
        for (int j = 1; j < 4; ++j) {
            CC(i, j - 1) = cws_[j][i] - cws_[0][i];
        }
    }

    if (CC.colPivHouseholderQr().rank() < 3) {
        return false;
    }

    const Eigen::Matrix3d CC_inv = CC.inverse();

    alphas_.resize(points2D_->size());
    for (size_t i = 0; i < points3D_->size(); ++i) {
        for (int j = 0; j < 3; ++j) {
            alphas_[i][1 + j] = CC_inv(j, 0) * ((*points3D_)[i][0] - cws_[0][0]) +
                                CC_inv(j, 1) * ((*points3D_)[i][1] - cws_[0][1]) +
                                CC_inv(j, 2) * ((*points3D_)[i][2] - cws_[0][2]);
        }
        alphas_[i][0] = 1.0 - alphas_[i][1] - alphas_[i][2] - alphas_[i][3];
    }

    return true;
}

Eigen::Matrix<double, Eigen::Dynamic, 12> EPNPEstimatorSpherical::ComputeM() {
    Eigen::Matrix<double, Eigen::Dynamic, 12> M(2 * points2D_->size(), 12);
    for (size_t i = 0; i < points3D_->size(); ++i) {
        for (size_t j = 0; j < 4; ++j) {
            M(2 * i, 3 * j) = alphas_[i][j];
            M(2 * i, 3 * j + 1) = 0.0;
            M(2 * i, 3 * j + 2) = -alphas_[i][j] * (*points2D_)[i].x();

            M(2 * i + 1, 3 * j) = 0.0;
            M(2 * i + 1, 3 * j + 1) = alphas_[i][j];
            M(2 * i + 1, 3 * j + 2) = -alphas_[i][j] * (*points2D_)[i].y();
        }
    }
    return M;
}

Eigen::Matrix<double, 6, 10> EPNPEstimatorSpherical::ComputeL6x10(const Eigen::Matrix<double, 12, 12>& Ut) {
    Eigen::Matrix<double, 6, 10> L6x10;

    std::array<std::array<Eigen::Vector3d, 6>, 4> dv;
    for (int i = 0; i < 4; ++i) {
        int a = 0, b = 1;
        for (int j = 0; j < 6; ++j) {
            dv[i][j][0] = Ut(11 - i, 3 * a) - Ut(11 - i, 3 * b);
            dv[i][j][1] = Ut(11 - i, 3 * a + 1) - Ut(11 - i, 3 * b + 1);
            dv[i][j][2] = Ut(11 - i, 3 * a + 2) - Ut(11 - i, 3 * b + 2);

            b += 1;
            if (b > 3) {
                a += 1;
                b = a + 1;
            }
        }
    }

    for (int i = 0; i < 6; ++i) {
        L6x10(i, 0) = dv[0][i].transpose() * dv[0][i];
        L6x10(i, 1) = 2.0 * dv[0][i].transpose() * dv[1][i];
        L6x10(i, 2) = dv[1][i].transpose() * dv[1][i];
        L6x10(i, 3) = 2.0 * dv[0][i].transpose() * dv[2][i];
        L6x10(i, 4) = 2.0 * dv[1][i].transpose() * dv[2][i];
        L6x10(i, 5) = dv[2][i].transpose() * dv[2][i];
        L6x10(i, 6) = 2.0 * dv[0][i].transpose() * dv[3][i];
        L6x10(i, 7) = 2.0 * dv[1][i].transpose() * dv[3][i];
        L6x10(i, 8) = 2.0 * dv[2][i].transpose() * dv[3][i];
        L6x10(i, 9) = dv[3][i].transpose() * dv[3][i];
    }

    return L6x10;
}

Eigen::Matrix<double, 6, 1> EPNPEstimatorSpherical::ComputeRho() {
    Eigen::Matrix<double, 6, 1> rho;
    rho[0] = (cws_[0] - cws_[1]).squaredNorm();
    rho[1] = (cws_[0] - cws_[2]).squaredNorm();
    rho[2] = (cws_[0] - cws_[3]).squaredNorm();
    rho[3] = (cws_[1] - cws_[2]).squaredNorm();
    rho[4] = (cws_[1] - cws_[3]).squaredNorm();
    rho[5] = (cws_[2] - cws_[3]).squaredNorm();
    return rho;
}

// betas10        = [B11 B12 B22 B13 B23 B33 B14 B24 B34 B44]
// betas_approx_1 = [B11 B12     B13         B14]

void EPNPEstimatorSpherical::FindBetasApprox1(const Eigen::Matrix<double, 6, 10>& L6x10,
                                              const Eigen::Matrix<double, 6, 1>& rho, Eigen::Vector4d* betas) {
    Eigen::Matrix<double, 6, 4> L_6x4;
    for (int i = 0; i < 6; ++i) {
        L_6x4(i, 0) = L6x10(i, 0);
        L_6x4(i, 1) = L6x10(i, 1);
        L_6x4(i, 2) = L6x10(i, 3);
        L_6x4(i, 3) = L6x10(i, 6);
    }

    Eigen::JacobiSVD<Eigen::Matrix<double, 6, 4>> svd(L_6x4, Eigen::ComputeFullV | Eigen::ComputeFullU);
    Eigen::Matrix<double, 6, 1> Rho_temp = rho;
    const Eigen::Matrix<double, 4, 1> b4 = svd.solve(Rho_temp);

    if (b4[0] < 0) {
        (*betas)[0] = std::sqrt(-b4[0]);
        (*betas)[1] = -b4[1] / (*betas)[0];
        (*betas)[2] = -b4[2] / (*betas)[0];
        (*betas)[3] = -b4[3] / (*betas)[0];
    } else {
        (*betas)[0] = std::sqrt(b4[0]);
        (*betas)[1] = b4[1] / (*betas)[0];
        (*betas)[2] = b4[2] / (*betas)[0];
        (*betas)[3] = b4[3] / (*betas)[0];
    }
}

// betas10        = [B11 B12 B22 B13 B23 B33 B14 B24 B34 B44]
// betas_approx_2 = [B11 B12 B22                            ]

void EPNPEstimatorSpherical::FindBetasApprox2(const Eigen::Matrix<double, 6, 10>& L6x10,
                                              const Eigen::Matrix<double, 6, 1>& rho, Eigen::Vector4d* betas) {
    Eigen::Matrix<double, 6, 3> L_6x3(6, 3);

    for (int i = 0; i < 6; ++i) {
        L_6x3(i, 0) = L6x10(i, 0);
        L_6x3(i, 1) = L6x10(i, 1);
        L_6x3(i, 2) = L6x10(i, 2);
    }

    Eigen::JacobiSVD<Eigen::Matrix<double, 6, 3>> svd(L_6x3, Eigen::ComputeFullV | Eigen::ComputeFullU);
    Eigen::Matrix<double, 6, 1> Rho_temp = rho;
    const Eigen::Matrix<double, 3, 1> b3 = svd.solve(Rho_temp);

    if (b3[0] < 0) {
        (*betas)[0] = std::sqrt(-b3[0]);
        (*betas)[1] = (b3[2] < 0) ? std::sqrt(-b3[2]) : 0.0;
    } else {
        (*betas)[0] = std::sqrt(b3[0]);
        (*betas)[1] = (b3[2] > 0) ? std::sqrt(b3[2]) : 0.0;
    }

    if (b3[1] < 0) {
        (*betas)[0] = -(*betas)[0];
    }

    (*betas)[2] = 0.0;
    (*betas)[3] = 0.0;
}

// betas10        = [B11 B12 B22 B13 B23 B33 B14 B24 B34 B44]
// betas_approx_3 = [B11 B12 B22 B13 B23                    ]

void EPNPEstimatorSpherical::FindBetasApprox3(const Eigen::Matrix<double, 6, 10>& L6x10,
                                              const Eigen::Matrix<double, 6, 1>& rho, Eigen::Vector4d* betas) {
    Eigen::JacobiSVD<Eigen::Matrix<double, 6, 5>> svd(L6x10.leftCols<5>(), Eigen::ComputeFullV | Eigen::ComputeFullU);
    Eigen::Matrix<double, 6, 1> Rho_temp = rho;
    const Eigen::Matrix<double, 5, 1> b5 = svd.solve(Rho_temp);

    if (b5[0] < 0) {
        (*betas)[0] = std::sqrt(-b5[0]);
        (*betas)[1] = (b5[2] < 0) ? std::sqrt(-b5[2]) : 0.0;
    } else {
        (*betas)[0] = std::sqrt(b5[0]);
        (*betas)[1] = (b5[2] > 0) ? std::sqrt(b5[2]) : 0.0;
    }
    if (b5[1] < 0) {
        (*betas)[0] = -(*betas)[0];
    }
    (*betas)[2] = b5[3] / (*betas)[0];
    (*betas)[3] = 0.0;
}

void EPNPEstimatorSpherical::RunGaussNewton(const Eigen::Matrix<double, 6, 10>& L6x10,
                                            const Eigen::Matrix<double, 6, 1>& rho, Eigen::Vector4d* betas) {
    Eigen::Matrix<double, 6, 4> A;
    Eigen::Matrix<double, 6, 1> b;

    const int kNumIterations = 5;
    for (int k = 0; k < kNumIterations; ++k) {
        for (int i = 0; i < 6; ++i) {
            A(i, 0) = 2 * L6x10(i, 0) * (*betas)[0] + L6x10(i, 1) * (*betas)[1] + L6x10(i, 3) * (*betas)[2] +
                      L6x10(i, 6) * (*betas)[3];
            A(i, 1) = L6x10(i, 1) * (*betas)[0] + 2 * L6x10(i, 2) * (*betas)[1] + L6x10(i, 4) * (*betas)[2] +
                      L6x10(i, 7) * (*betas)[3];
            A(i, 2) = L6x10(i, 3) * (*betas)[0] + L6x10(i, 4) * (*betas)[1] + 2 * L6x10(i, 5) * (*betas)[2] +
                      L6x10(i, 8) * (*betas)[3];
            A(i, 3) = L6x10(i, 6) * (*betas)[0] + L6x10(i, 7) * (*betas)[1] + L6x10(i, 8) * (*betas)[2] +
                      2 * L6x10(i, 9) * (*betas)[3];

            b(i) = rho[i] - (L6x10(i, 0) * (*betas)[0] * (*betas)[0] + L6x10(i, 1) * (*betas)[0] * (*betas)[1] +
                             L6x10(i, 2) * (*betas)[1] * (*betas)[1] + L6x10(i, 3) * (*betas)[0] * (*betas)[2] +
                             L6x10(i, 4) * (*betas)[1] * (*betas)[2] + L6x10(i, 5) * (*betas)[2] * (*betas)[2] +
                             L6x10(i, 6) * (*betas)[0] * (*betas)[3] + L6x10(i, 7) * (*betas)[1] * (*betas)[3] +
                             L6x10(i, 8) * (*betas)[2] * (*betas)[3] + L6x10(i, 9) * (*betas)[3] * (*betas)[3]);
        }

        const Eigen::Vector4d x = A.colPivHouseholderQr().solve(b);

        (*betas) += x;
    }
}

double EPNPEstimatorSpherical::ComputeRT(const Eigen::Matrix<double, 12, 12>& Ut, const Eigen::Vector4d& betas,
                                         Eigen::Matrix3d* R, Eigen::Vector3d* t) {
    ComputeCcs(betas, Ut);
    ComputePcs();

    SolveForSign();

    EstimateRT(R, t);

    return ComputeTotalReprojectionError(*R, *t);
}

void EPNPEstimatorSpherical::ComputeCcs(const Eigen::Vector4d& betas, const Eigen::Matrix<double, 12, 12>& Ut) {
    for (int i = 0; i < 4; ++i) {
        ccs_[i][0] = ccs_[i][1] = ccs_[i][2] = 0.0;
    }

    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            for (int k = 0; k < 3; ++k) {
                ccs_[j][k] += betas[i] * Ut(11 - i, 3 * j + k);
            }
        }
    }
}

void EPNPEstimatorSpherical::ComputePcs() {
    pcs_.resize(points2D_->size());
    for (size_t i = 0; i < points3D_->size(); ++i) {
        for (int j = 0; j < 3; ++j) {
            pcs_[i][j] = alphas_[i][0] * ccs_[0][j] + alphas_[i][1] * ccs_[1][j] + alphas_[i][2] * ccs_[2][j] +
                         alphas_[i][3] * ccs_[3][j];
        }
    }
}

void EPNPEstimatorSpherical::SolveForSign() {
    if (pcs_[0][2] < 0.0 || pcs_[0][2] > 0.0) {
        for (int i = 0; i < 4; ++i) {
            ccs_[i] = -ccs_[i];
        }
        for (size_t i = 0; i < points3D_->size(); ++i) {
            pcs_[i] = -pcs_[i];
        }
    }
}

void EPNPEstimatorSpherical::EstimateRT(Eigen::Matrix3d* R, Eigen::Vector3d* t) {
    Eigen::Vector3d pc0 = Eigen::Vector3d::Zero();
    Eigen::Vector3d pw0 = Eigen::Vector3d::Zero();

    for (size_t i = 0; i < points3D_->size(); ++i) {
        pc0 += pcs_[i];
        pw0 += (*points3D_)[i];
    }
    pc0 /= points3D_->size();
    pw0 /= points3D_->size();

    Eigen::Matrix3d abt = Eigen::Matrix3d::Zero();
    for (size_t i = 0; i < points3D_->size(); ++i) {
        for (int j = 0; j < 3; ++j) {
            abt(j, 0) += (pcs_[i][j] - pc0[j]) * ((*points3D_)[i][0] - pw0[0]);
            abt(j, 1) += (pcs_[i][j] - pc0[j]) * ((*points3D_)[i][1] - pw0[1]);
            abt(j, 2) += (pcs_[i][j] - pc0[j]) * ((*points3D_)[i][2] - pw0[2]);
        }
    }

    Eigen::JacobiSVD<Eigen::Matrix3d> svd(abt, Eigen::ComputeFullV | Eigen::ComputeFullU);
    const Eigen::Matrix3d abt_U = svd.matrixU();
    const Eigen::Matrix3d abt_V = svd.matrixV();

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            (*R)(i, j) = abt_U.row(i) * abt_V.row(j).transpose();
        }
    }

    if (R->determinant() < 0) {
        Eigen::Matrix3d Abt_v_prime = abt_V;
        Abt_v_prime.col(2) = -abt_V.col(2);
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                (*R)(i, j) = abt_U.row(i) * Abt_v_prime.row(j).transpose();
            }
        }
    }

    *t = pc0 - *R * pw0;
}

double EPNPEstimatorSpherical::ComputeTotalReprojectionError(const Eigen::Matrix3d& R, const Eigen::Vector3d& t) {
    Eigen::Matrix3x4d proj_matrix;
    proj_matrix.leftCols<3>() = R;
    proj_matrix.rightCols<1>() = t;

    std::vector<double> residuals;
    ComputeSquaredReprojectionErrorSpherical(*points2D_bearing_, *points3D_, proj_matrix, &residuals);

    double reproj_error = 0.0;
    for (const double residual : residuals) {
        reproj_error += std::sqrt(residual);
    }

    return reproj_error;
}

bool EstimateAbsolutePose(const AbsolutePoseEstimationOptions& options, const std::vector<Eigen::Vector2d>& points2D,
                          const std::vector<Eigen::Vector3d>& points3D, Eigen::Vector4d* qvec, Eigen::Vector3d* tvec,
                          Camera* camera, size_t* num_inliers, std::vector<char>* inlier_mask) {
    options.Check();

    std::vector<double> focal_length_factors;
    if (options.estimate_focal_length) {
        // Generate focal length factors using a quadratic function,
        // such that more samples are drawn for small focal lengths
        focal_length_factors.reserve(options.num_focal_length_samples + 1);
        const double fstep = 1.0 / options.num_focal_length_samples;
        const double fscale = options.max_focal_length_ratio - options.min_focal_length_ratio;
        for (double f = 0; f <= 1.0; f += fstep) {
            focal_length_factors.push_back(options.min_focal_length_ratio + fscale * f * f);
        }
    } else {
        focal_length_factors.reserve(1);
        focal_length_factors.push_back(1);
    }

    std::vector<std::future<void>> futures;
    futures.resize(focal_length_factors.size());

    double focal_length_factor = 0;
    Eigen::Matrix3x4d proj_matrix;

    if (camera->ModelName().compare("SPHERICAL") == 0||
        camera->ModelName().compare("OPENCV_FISHEYE") == 0) {
        std::vector<typename AbsolutePoseSphericalRANSAC::Report,
                    Eigen::aligned_allocator<typename AbsolutePoseSphericalRANSAC::Report>>
            reports;
        reports.resize(focal_length_factors.size());

        ThreadPool thread_pool(std::min(options.num_threads, static_cast<int>(focal_length_factors.size())));

        for (size_t i = 0; i < focal_length_factors.size(); ++i) {
            futures[i] = thread_pool.AddTask(EstimateAbsolutePoseSphericalKernel, *camera, focal_length_factors[i],
                                             points2D, points3D, options.ransac_options, &reports[i]);
        }

        *num_inliers = 0;
        inlier_mask->clear();

        // Find best model among all focal lengths.
        for (size_t i = 0; i < focal_length_factors.size(); ++i) {
            futures[i].get();
            const auto report = reports[i];
            if (report.success && report.support.num_inliers > *num_inliers) {
                *num_inliers = report.support.num_inliers;
                proj_matrix = report.model;
                *inlier_mask = report.inlier_mask;
                focal_length_factor = focal_length_factors[i];
            }
        }
    } else {
        std::vector<typename AbsolutePoseRANSAC::Report, Eigen::aligned_allocator<typename AbsolutePoseRANSAC::Report>>
            reports;
        reports.resize(focal_length_factors.size());

        ThreadPool thread_pool(std::min(options.num_threads, static_cast<int>(focal_length_factors.size())));

        for (size_t i = 0; i < focal_length_factors.size(); ++i) {
            futures[i] = thread_pool.AddTask(EstimateAbsolutePoseKernel, *camera, focal_length_factors[i], points2D,
                                             points3D, options.ransac_options, &reports[i]);
        }

        *num_inliers = 0;
        inlier_mask->clear();

        // Find best model among all focal lengths.
        for (size_t i = 0; i < focal_length_factors.size(); ++i) {
            futures[i].get();
            const auto report = reports[i];
            if (report.success && report.support.num_inliers > *num_inliers) {
                *num_inliers = report.support.num_inliers;
                proj_matrix = report.model;
                *inlier_mask = report.inlier_mask;
                focal_length_factor = focal_length_factors[i];
            }
        }
    }

    if (*num_inliers == 0) {
        return false;
    }

    // Scale output camera with best estimated focal length.
    if (options.estimate_focal_length && *num_inliers > 0) {
        const std::vector<size_t>& focal_length_idxs = camera->FocalLengthIdxs();
        for (const size_t idx : focal_length_idxs) {
            camera->Params(idx) *= focal_length_factor;
        }
    }

    // Extract pose parameters.
    *qvec = RotationMatrixToQuaternion(proj_matrix.leftCols<3>());
    *tvec = proj_matrix.rightCols<1>();

    if (IsNaN(*qvec) || IsNaN(*tvec)) {
        return false;
    }

    return true;
}

bool EstimateAbsolutePoses(const AbsolutePoseEstimationOptions& options, const std::vector<Eigen::Vector2d>& points2D,
                           const std::vector<Eigen::Vector3d>& points3D, const Camera* camera,
                           std::vector<double>& estimated_focal_length_factors, std::vector<Eigen::Vector4d>& qvecs,
                           std::vector<Eigen::Vector3d>& tvecs) {
    options.Check();

    std::vector<double> focal_length_factors;
    if (options.estimate_focal_length) {
        // Generate focal length factors using a quadratic function,
        // such that more samples are drawn for small focal lengths
        focal_length_factors.reserve(options.num_focal_length_samples + 1);
        const double fstep = 1.0 / options.num_focal_length_samples;
        const double fscale = options.max_focal_length_ratio - options.min_focal_length_ratio;
        for (double f = 0; f <= 1.0; f += fstep) {
            focal_length_factors.push_back(options.min_focal_length_ratio + fscale * f * f);
        }
    } else {
        focal_length_factors.reserve(1);
        focal_length_factors.push_back(1);
    }

    std::vector<std::future<void>> futures;
    futures.resize(focal_length_factors.size());
    std::vector<typename AbsolutePoseRANSAC::Report, Eigen::aligned_allocator<typename AbsolutePoseRANSAC::Report>>
        reports;
    reports.resize(focal_length_factors.size());

    ThreadPool thread_pool(std::min(options.num_threads, static_cast<int>(focal_length_factors.size())));

    for (size_t i = 0; i < focal_length_factors.size(); ++i) {
        futures[i] = thread_pool.AddTask(EstimateAbsolutePoseKernel, *camera, focal_length_factors[i], points2D,
                                         points3D, options.ransac_options, &reports[i]);
    }

    struct IndexReport {
        size_t idx;
        size_t num_inliers;
        Eigen::Matrix3x4d proj_matrix;
    };
    std::vector<IndexReport> index_reports;
    for (size_t i = 0; i < reports.size(); ++i) {
        futures[i].get();
        for (size_t j = 0; j < reports[i].multiple_models.size(); ++j) {
            IndexReport index_report;
            index_report.idx = i;
            index_report.proj_matrix = reports[i].multiple_models[j];
            index_report.num_inliers = reports[i].multiple_supports[j].num_inliers;
            index_reports.emplace_back(index_report);
        }
    }

    std::sort(index_reports.begin(), index_reports.end(),
              [&](const IndexReport& index_report1, const IndexReport& index_report2) {
                  return index_report1.num_inliers > index_report2.num_inliers;
              });

    estimated_focal_length_factors.clear();
    qvecs.clear();
    tvecs.clear();

    for (size_t i = 0; i < index_reports.size(); ++i) {
        size_t idx = index_reports[i].idx;
        if (index_reports[i].num_inliers > 3) {
            const Eigen::Matrix3x4d proj_matrix = index_reports[i].proj_matrix;
            // Extract pose parameters.
            Eigen::Vector4d qvec = RotationMatrixToQuaternion(proj_matrix.leftCols<3>());
            Eigen::Vector3d tvec = proj_matrix.rightCols<1>();
            if (IsNaN(qvec) || IsNaN(tvec)) {
                continue;
            }
            estimated_focal_length_factors.push_back(focal_length_factors[idx]);
            qvecs.push_back(qvec);
            tvecs.push_back(tvec);
        }
    }

    if (qvecs.size() == 0) {
        return false;
    }

    return true;
}

bool RefineAbsolutePose(const AbsolutePoseRefinementOptions& options, const std::vector<char>& inlier_mask,
                        const std::vector<Eigen::Vector2d>& points2D, const std::vector<Eigen::Vector3d>& points3D,
                        const uint64_t num_reg_images, std::vector<uint64_t> mappoints_create_time, 
                        const std::vector<double>& mappoint_weights,
                        Eigen::Vector4d* qvec, Eigen::Vector3d* tvec, Camera* camera) {
    CHECK_EQ(inlier_mask.size(), points2D.size());
    CHECK_EQ(points2D.size(), points3D.size());
    options.Check();

    ceres::LossFunction* loss_function = new ceres::CauchyLoss(options.loss_function_scale);

    const bool with_depth = 
        options.point_depths.size() == options.point_depths_weights.size() &&
        options.point_depths.size() == points2D.size();
    double* camera_params_data = camera->ParamsData();
    double* qvec_data = qvec->data();
    double* tvec_data = tvec->data();


    double* local_qvec2_data;
    double* local_tvec2_data;

    std::vector<Eigen::Vector3d> points3D_copy = points3D;

    ceres::Problem problem;

    for (size_t i = 0; i < points2D.size(); ++i) {
        // Skip outlier observations
        if (!inlier_mask[i]) {
            continue;
        }

        double weight = mappoint_weights.empty() ? 1.0 : mappoint_weights[i];
        double loop_weight = 1.0;
        if (mappoints_create_time.size() > 0 && mappoints_create_time[i] > 0) {
            int time_diff = std::abs((int)num_reg_images - (int)mappoints_create_time[i]);
            if (time_diff > 200) loop_weight = 20.0;
            // else {
            //     loop_weight = std::max(1.0, time_diff / 10.0);
            // }
            else if (time_diff > 100) {
                loop_weight = 0.19 * time_diff - 18;
            }
        } else {
            std::cout << StringPrintf("RefinePose has no timestamp!") << std::endl;
        }

        ceres::CostFunction* cost_function = nullptr;

        if (camera->ModelName().compare("SPHERICAL") == 0) {
            Eigen::Vector3d bearing = camera->ImageToBearing(points2D[i]);
            double f = camera->FocalLength();
            cost_function = SphericalBundleAdjustmentCostFunction<SphericalCameraModel>::Create(bearing,f, weight * loop_weight);
            local_qvec2_data = camera_params_data + 10;
            local_tvec2_data = camera_params_data + 14;

            problem.AddResidualBlock(cost_function, loss_function, qvec_data, tvec_data, points3D_copy[i].data(),
                                    local_qvec2_data,
                                    local_tvec2_data);

            problem.SetParameterBlockConstant(points3D_copy[i].data());
        } else {
            switch (camera->ModelId()) {
#define CAMERA_MODEL_CASE(CameraModel)                                                  \
    case CameraModel::kModelId:                                                         \
        cost_function = BundleAdjustmentCostFunction<CameraModel>::Create(points2D[i], weight * loop_weight); \
        break;
                CAMERA_MODEL_SWITCH_CASES
#undef CAMERA_MODEL_CASE
            }
            problem.AddResidualBlock(cost_function, loss_function, qvec_data, tvec_data, points3D_copy[i].data(),
                                     camera_params_data);
            problem.SetParameterBlockConstant(points3D_copy[i].data());
        }

        if (with_depth && options.point_depths[i] > 0.0 && options.point_depths_weights[i] > 0.0) {
            cost_function = BundleAdjustmentDepthCostFunction::Create(
                Eigen::Vector3d::Constant(options.point_depths[i]),
                options.point_depths_weights[i] * loop_weight);
            problem.AddResidualBlock(cost_function, loss_function, qvec_data, tvec_data, points3D_copy[i].data());
            problem.SetParameterBlockConstant(points3D_copy[i].data());
        }
    }

    if (problem.NumResiduals() > 0) {
        // Quaternion parameterization.
        *qvec = NormalizeQuaternion(*qvec);
        ceres::LocalParameterization* quaternion_parameterization = new ceres::QuaternionParameterization;
#if CERES_VERSION_MAJOR >= 2 && CERES_VERSION_MINOR >= 1
        problem.SetManifold(qvec_data, quaternion_parameterization);
#else
        problem.SetParameterization(qvec_data, quaternion_parameterization);
#endif

        if (camera->ModelName().compare("SPHERICAL") == 0) {
            problem.SetParameterBlockConstant(local_qvec2_data);
            problem.SetParameterBlockConstant(local_tvec2_data);
        } else {
            // Camera parameterization.
            if ((!options.refine_focal_length && !options.refine_extra_params) || camera->IsCameraConstant()) {
                problem.SetParameterBlockConstant(camera->ParamsData());
            } else {
                // Always set the principal point as fixed.
                std::vector<int> camera_params_const;
                const std::vector<size_t>& principal_point_idxs = camera->PrincipalPointIdxs();
                camera_params_const.insert(camera_params_const.end(), principal_point_idxs.begin(),
                                           principal_point_idxs.end());

                if (!options.refine_focal_length) {
                    const std::vector<size_t>& focal_length_idxs = camera->FocalLengthIdxs();
                    camera_params_const.insert(camera_params_const.end(), focal_length_idxs.begin(),
                                               focal_length_idxs.end());
                }

                if (!options.refine_extra_params) {
                    const std::vector<size_t>& extra_params_idxs = camera->ExtraParamsIdxs();
                    camera_params_const.insert(camera_params_const.end(), extra_params_idxs.begin(),
                                               extra_params_idxs.end());
                }

                for (size_t idx : camera->FocalLengthIdxs()) {
                    double est_focal = camera->ParamsData()[idx];
                    problem.SetParameterLowerBound(camera->ParamsData(), idx,
                                                   options.lower_bound_focal_length_factor * est_focal);
                    problem.SetParameterUpperBound(camera->ParamsData(), idx,
                                                   options.upper_bound_focal_length_factor * est_focal);
                }
                std::string model_name = camera->ModelName();
                if ("SIMPLE_RADIAL" == model_name || "RADIAL" == model_name) {
                    for (size_t idx : camera->ExtraParamsIdxs()) {
                        problem.SetParameterLowerBound(camera->ParamsData(), idx, -1.0);
                        problem.SetParameterUpperBound(camera->ParamsData(), idx, 1.0);
                    }
                }

                if (camera_params_const.size() == camera->NumParams()) {
                    problem.SetParameterBlockConstant(camera->ParamsData());
                } else {
                    ceres::SubsetParameterization* camera_params_parameterization =
                        new ceres::SubsetParameterization(static_cast<int>(camera->NumParams()), camera_params_const);
#if CERES_VERSION_MAJOR >= 2 && CERES_VERSION_MINOR >= 1
                    problem.SetManifold(camera->ParamsData(), camera_params_parameterization);
#else
                    problem.SetParameterization(camera->ParamsData(), camera_params_parameterization);
#endif
                }
            }
        }
    }

    ceres::Solver::Options solver_options;
    solver_options.gradient_tolerance = options.gradient_tolerance;
    solver_options.max_num_iterations = options.max_num_iterations;
    solver_options.linear_solver_type = ceres::DENSE_QR;
#if CERES_VERSION_MAJOR >= 2 && CERES_VERSION_MINOR >= 2 && !defined(CERES_NO_CUDA)
    solver_options.dense_linear_algebra_library_type = ceres::CUDA;
#endif

    // The overhead of creating threads is too large.
    solver_options.num_threads = std::min(16,GetEffectiveNumThreads(-1));
#if CERES_VERSION_MAJOR < 2
    solver_options.num_linear_solver_threads = 1;
#endif  // CERES_VERSION_MAJOR

    ceres::Solver::Summary summary;
    ceres::Solve(solver_options, &problem, &summary);

    if (solver_options.minimizer_progress_to_stdout) {
        std::cout << std::endl;
    }

    if (options.print_summary) {
        PrintHeading2("Pose refinement report");
        PrintSolverSummary(summary);
    }

    return summary.IsSolutionUsable();
}

bool RefineAbsolutePoseRig(const AbsolutePoseRefinementOptions& options, const std::vector<char>& inlier_mask,
                           const std::vector<Eigen::Vector2d>& points2D, const std::vector<Eigen::Vector3d>& points3D,
                           const uint64_t num_reg_images, std::vector<uint64_t> mappoints_create_time, 
                           const std::vector<double>& mappoint_weights, const std::vector<int>& local_camera_indices, 
                           Eigen::Vector4d* qvec, Eigen::Vector3d* tvec, Camera* camera) {
    CHECK_EQ(inlier_mask.size(), points2D.size());
    CHECK_EQ(points2D.size(), points3D.size());
    options.Check();

    ceres::LossFunction* loss_function = new ceres::CauchyLoss(options.loss_function_scale);

    double* camera_params_data = camera->ParamsData();
    double* qvec_data = qvec->data();
    double* tvec_data = tvec->data();

    std::vector<double*> local_qvec_data;
    std::vector<double*> local_tvec_data;
    std::vector<double*> local_camera_params_data;

    int local_param_size = camera->LocalParams().size() / camera->NumLocalCameras();

    for (size_t i = 0; i < camera->NumLocalCameras(); ++i) {
        local_qvec_data.push_back(camera->LocalQvecsData() + 4 * i);
        local_tvec_data.push_back(camera->LocalTvecsData() + 3 * i);
        local_camera_params_data.push_back(camera->LocalIntrinsicParamsData() + i * local_param_size);
    }

    std::vector<Eigen::Vector3d> points3D_copy = points3D;

    ceres::Problem problem;

    for (size_t i = 0; i < points2D.size(); ++i) {
        // Skip outlier observations
        if (!inlier_mask[i]) {
            continue;
        }
        int local_camera_id = local_camera_indices[i];
        ceres::CostFunction* cost_function = nullptr;

        double weight = mappoint_weights.empty() ? 1.0 : mappoint_weights[i];
        double loop_weight = 1.0;
        if (mappoints_create_time.size() > 0 && mappoints_create_time[i] > 0) {
            int time_diff = std::abs((int)num_reg_images - (int)mappoints_create_time[i]);
            if (time_diff > 200) loop_weight = 20.0;
            // else {
            //     loop_weight = std::max(1.0, time_diff / 10.0);
            // }
            else if (time_diff > 100) {
                loop_weight = 0.19 * time_diff - 18;
            }
        } else {
            std::cout << StringPrintf("RefinePoseRig has no timestamp!") << std::endl;
        }

        if (camera->ModelName().compare("UNIFIED") == 0 || camera->ModelName().compare("OPENCV_FISHEYE") == 0) {

            Eigen::Vector3d bearing = camera->LocalImageToBearing(local_camera_id, points2D[i]);
            double f = camera->LocalMeanFocalLength(local_camera_id);
            cost_function = LargeFovRigBundleAdjustmentCostFunction<OpenCVFisheyeCameraModel>::Create(bearing, f, weight * loop_weight);

            problem.AddResidualBlock(cost_function, loss_function, qvec_data, tvec_data, local_qvec_data[local_camera_id],
                                     local_tvec_data[local_camera_id], points3D_copy[i].data()); 

        }    
        else{

            switch (camera->ModelId()) {
#define CAMERA_MODEL_CASE(CameraModel)                                                     \
                case CameraModel::kModelId:                                                            \
                    cost_function = RigBundleAdjustmentCostFunction<CameraModel>::Create(points2D[i], weight * loop_weight); \
                    break;

            CAMERA_MODEL_SWITCH_CASES

#undef CAMERA_MODEL_CASE
            }

            problem.AddResidualBlock(cost_function, loss_function, qvec_data, tvec_data, local_qvec_data[local_camera_id],
                                    local_tvec_data[local_camera_id], points3D_copy[i].data(),
                                    local_camera_params_data[local_camera_id]);
            problem.SetParameterBlockConstant(local_camera_params_data[local_camera_id]);
        }

        // The points and the

        problem.SetParameterBlockConstant(points3D_copy[i].data());
        problem.SetParameterBlockConstant(local_qvec_data[local_camera_id]);
        problem.SetParameterBlockConstant(local_tvec_data[local_camera_id]);
    }

    if (problem.NumResiduals() > 0) {
        // Quaternion parameterization.
        *qvec = NormalizeQuaternion(*qvec);
        ceres::LocalParameterization* quaternion_parameterization = new ceres::QuaternionParameterization;
#if CERES_VERSION_MAJOR >= 2 && CERES_VERSION_MINOR >= 1
        problem.SetManifold(qvec_data, quaternion_parameterization);
#else
        problem.SetParameterization(qvec_data, quaternion_parameterization);
#endif
    }

    ceres::Solver::Options solver_options;
    solver_options.gradient_tolerance = options.gradient_tolerance;
    solver_options.max_num_iterations = options.max_num_iterations;
    solver_options.linear_solver_type = ceres::DENSE_QR;
#if CERES_VERSION_MAJOR >= 2 && CERES_VERSION_MINOR >= 2 && !defined(CERES_NO_CUDA)
    solver_options.dense_linear_algebra_library_type = ceres::CUDA;
#endif

    // The overhead of creating threads is too large.
    solver_options.num_threads = 4;
#if CERES_VERSION_MAJOR < 2
    solver_options.num_linear_solver_threads = 1;
#endif  // CERES_VERSION_MAJOR

    ceres::Solver::Summary summary;
    ceres::Solve(solver_options, &problem, &summary);

    if (solver_options.minimizer_progress_to_stdout) {
        std::cout << std::endl;
    }

    if (options.print_summary) {
        PrintHeading2("Pose refinement report");
        PrintSolverSummary(summary);
    }

    return summary.IsSolutionUsable();
}

}  // namespace sensemap
