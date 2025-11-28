// Copyright (c) 2019, SenseTime Group.
// All rights reserved.

#include <unsupported/Eigen/NonLinearOptimization>
#include <unsupported/Eigen/NumericalDiff>

#include "estimators/optimization_functor.h"

#include "estimators/generalized_relative_pose.h"

#include "base/essential_matrix.h"
#include "base/pose.h"
#include "base/projection.h"
#include "base/triangulation.h"
#include "util/logging.h"
#include "util/random.h"

namespace sensemap {
namespace {

void ComposePlueckerData(const Eigen::Matrix3x4d& rel_tform, const Eigen::Vector2d& point2D,
                         Eigen::Vector3d* proj_center, Eigen::Vector6d* pluecker) {
    const Eigen::Matrix3x4d inv_proj_matrix = InvertProjectionMatrix(rel_tform);
    const Eigen::Vector3d bearing = inv_proj_matrix.leftCols<3>() * point2D.homogeneous();
    const Eigen::Vector3d bearing_normalized = bearing.normalized();
    *proj_center = inv_proj_matrix.rightCols<1>();
    *pluecker << bearing_normalized, proj_center->cross(bearing_normalized);
}

Eigen::Matrix3d CayleyToRotationMatrix(const Eigen::Vector3d& cayley) {
    const double cayley0_sqr = cayley[0] * cayley[0];
    const double cayley1_sqr = cayley[1] * cayley[1];
    const double cayley2_sqr = cayley[2] * cayley[2];
    const double cayley01 = cayley[0] * cayley[1];
    const double cayley12 = cayley[1] * cayley[2];
    const double cayley02 = cayley[0] * cayley[2];

    const double scale = 1 + cayley0_sqr + cayley1_sqr + cayley2_sqr;
    const double inv_scale = 1.0 / scale;

    Eigen::Matrix3d R;

    R(0, 0) = inv_scale * (1 + cayley0_sqr - cayley1_sqr - cayley2_sqr);
    R(0, 1) = inv_scale * (2 * (cayley01 - cayley[2]));
    R(0, 2) = inv_scale * (2 * (cayley02 + cayley[1]));
    R(1, 0) = inv_scale * (2 * (cayley01 + cayley[2]));
    R(1, 1) = inv_scale * (1 - cayley0_sqr + cayley1_sqr - cayley2_sqr);
    R(1, 2) = inv_scale * (2 * (cayley12 - cayley[0]));
    R(2, 0) = inv_scale * (2 * (cayley02 - cayley[1]));
    R(2, 1) = inv_scale * (2 * (cayley12 + cayley[0]));
    R(2, 2) = inv_scale * (1 - cayley0_sqr - cayley1_sqr + cayley2_sqr);

    return R;
}

Eigen::Matrix3d CayleyToRotationReduced( const Eigen::Vector3d & cayley) {
  Eigen::Matrix3d R;

  R(0,0) = 1 + pow(cayley[0], 2) - pow(cayley[1], 2) - pow(cayley[2], 2);
  R(0,1) = 2 * (cayley[0] * cayley[1] - cayley[2]);
  R(0,2) = 2 * (cayley[0] * cayley[2] + cayley[1]);
  R(1,0) = 2 * (cayley[0] * cayley[1] + cayley[2]);
  R(1,1) = 1 - pow(cayley[0], 2) + pow(cayley[1], 2) - pow(cayley[2], 2);
  R(1,2) = 2 * (cayley[1] * cayley[2] - cayley[0]);
  R(2,0) = 2 * (cayley[0] * cayley[2] - cayley[1]);
  R(2,1) = 2 * (cayley[1] * cayley[2] + cayley[0]);
  R(2,2) = 1 - pow(cayley[0], 2) - pow(cayley[1], 2) + pow(cayley[2], 2);

  return R;
}

Eigen::Vector3d RotationMatrixToCaley(const Eigen::Matrix3d& R) {
    const Eigen::Matrix3d C1 = R - Eigen::Matrix3d::Identity();
    const Eigen::Matrix3d C2 = R + Eigen::Matrix3d::Identity();
    const Eigen::Matrix3d C = C1 * C2.inverse();
    return Eigen::Vector3d(-C(1, 2), C(0, 2), -C(0, 1));
}

Eigen::Vector3d ComputeRotationBetweenPoints(const std::vector<Eigen::Vector6d>& plueckers1,
                                             const std::vector<Eigen::Vector6d>& plueckers2) {
    CHECK_EQ(plueckers1.size(), plueckers2.size());

    // Compute the center of all observed points.
    Eigen::Vector3d points_center1 = Eigen::Vector3d::Zero();
    Eigen::Vector3d points_center2 = Eigen::Vector3d::Zero();
    for (size_t i = 0; i < plueckers1.size(); i++) {
        points_center1 += plueckers1[i].head<3>();
        points_center2 += plueckers2[i].head<3>();
    }
    points_center1 = points_center1 / plueckers1.size();
    points_center2 = points_center2 / plueckers1.size();

    Eigen::Matrix3d Hcross = Eigen::Matrix3d::Zero();
    for (size_t i = 0; i < plueckers1.size(); i++) {
        const Eigen::Vector3d f1 = plueckers1[i].head<3>() - points_center1;
        const Eigen::Vector3d f2 = plueckers2[i].head<3>() - points_center2;
        Hcross += f2 * f1.transpose();
    }

    const Eigen::JacobiSVD<Eigen::Matrix3d> svd(Hcross, Eigen::ComputeFullU | Eigen::ComputeFullV);
    const Eigen::Matrix3d V = svd.matrixV();
    const Eigen::Matrix3d U = svd.matrixU();

    Eigen::Matrix3d R = V * U.transpose();
    if (R.determinant() < 0) {
        Eigen::Matrix3d V_prime;
        V_prime.col(0) = V.col(0);
        V_prime.col(1) = V.col(1);
        V_prime.col(2) = -V.col(2);
        R = V_prime * U.transpose();
    }

    return RotationMatrixToCaley(R);
}

Eigen::Matrix4d ComposeG(const Eigen::Matrix3d& xxF, const Eigen::Matrix3d& yyF, const Eigen::Matrix3d& zzF,
                         const Eigen::Matrix3d& xyF, const Eigen::Matrix3d& yzF, const Eigen::Matrix3d& zxF,
                         const Eigen::Matrix<double, 3, 9>& x1P, const Eigen::Matrix<double, 3, 9>& y1P,
                         const Eigen::Matrix<double, 3, 9>& z1P, const Eigen::Matrix<double, 3, 9>& x2P,
                         const Eigen::Matrix<double, 3, 9>& y2P, const Eigen::Matrix<double, 3, 9>& z2P,
                         const Eigen::Matrix<double, 9, 9>& m11P, const Eigen::Matrix<double, 9, 9>& m12P,
                         const Eigen::Matrix<double, 9, 9>& m22P, const Eigen::Vector3d& rotation) {
    const Eigen::Matrix3d R = CayleyToRotationMatrix(rotation);

    Eigen::Matrix<double, 1, 9> R_rows;
    R_rows << R.row(0), R.row(1), R.row(2);

    Eigen::Matrix<double, 9, 1> R_cols;
    R_cols << R.col(0), R.col(1), R.col(2);

    const Eigen::Vector3d xxFr1t = xxF * R.row(1).transpose();
    const Eigen::Vector3d yyFr0t = yyF * R.row(0).transpose();
    const Eigen::Vector3d zzFr0t = zzF * R.row(0).transpose();
    const Eigen::Vector3d yzFr0t = yzF * R.row(0).transpose();
    const Eigen::Vector3d xyFr1t = xyF * R.row(1).transpose();
    const Eigen::Vector3d xyFr2t = xyF * R.row(2).transpose();
    const Eigen::Vector3d zxFr1t = zxF * R.row(1).transpose();
    const Eigen::Vector3d zxFr2t = zxF * R.row(2).transpose();

    const Eigen::Vector3d x1PC = x1P * R_cols;
    const Eigen::Vector3d y1PC = y1P * R_cols;
    const Eigen::Vector3d z1PC = z1P * R_cols;

    const Eigen::Vector3d x2PR = x2P * R_rows.transpose();
    const Eigen::Vector3d y2PR = y2P * R_rows.transpose();
    const Eigen::Vector3d z2PR = z2P * R_rows.transpose();

    Eigen::Matrix4d G;

    G(0, 0) = R.row(2) * yyF * R.row(2).transpose();
    G(0, 0) += -2.0 * R.row(2) * yzF * R.row(1).transpose();
    G(0, 0) += R.row(1) * zzF * R.row(1).transpose();

    G(0, 1) = R.row(2) * yzFr0t;
    G(0, 1) += -1.0 * R.row(2) * xyFr2t;
    G(0, 1) += -1.0 * R.row(1) * zzFr0t;
    G(0, 1) += R.row(1) * zxFr2t;

    G(0, 2) = R.row(2) * xyFr1t;
    G(0, 2) += -1.0 * R.row(2) * yyFr0t;
    G(0, 2) += -1.0 * R.row(1) * zxFr1t;
    G(0, 2) += R.row(1) * yzFr0t;

    G(1, 1) = R.row(0) * zzFr0t;
    G(1, 1) += -2.0 * R.row(0) * zxFr2t;
    G(1, 1) += R.row(2) * xxF * R.row(2).transpose();

    G(1, 2) = R.row(0) * zxFr1t;
    G(1, 2) += -1.0 * R.row(0) * yzFr0t;
    G(1, 2) += -1.0 * R.row(2) * xxFr1t;
    G(1, 2) += R.row(0) * xyFr2t;

    G(2, 2) = R.row(1) * xxFr1t;
    G(2, 2) += -2.0 * R.row(0) * xyFr1t;
    G(2, 2) += R.row(0) * yyFr0t;

    G(1, 0) = G(0, 1);
    G(2, 0) = G(0, 2);
    G(2, 1) = G(1, 2);

    G(0, 3) = R.row(2) * y1PC;
    G(0, 3) += R.row(2) * y2PR;
    G(0, 3) += -1.0 * R.row(1) * z1PC;
    G(0, 3) += -1.0 * R.row(1) * z2PR;

    G(1, 3) = R.row(0) * z1PC;
    G(1, 3) += R.row(0) * z2PR;
    G(1, 3) += -1.0 * R.row(2) * x1PC;
    G(1, 3) += -1.0 * R.row(2) * x2PR;

    G(2, 3) = R.row(1) * x1PC;
    G(2, 3) += R.row(1) * x2PR;
    G(2, 3) += -1.0 * R.row(0) * y1PC;
    G(2, 3) += -1.0 * R.row(0) * y2PR;

    G(3, 3) = -1.0 * R_cols.transpose() * m11P * R_cols;
    G(3, 3) += -1.0 * R_rows * m22P * R_rows.transpose();
    G(3, 3) += -2.0 * R_rows * m12P * R_cols;

    G(3, 0) = G(0, 3);
    G(3, 1) = G(1, 3);
    G(3, 2) = G(2, 3);

    return G;
}

Eigen::Vector4d ComputeEigenValue(const Eigen::Matrix3d& xxF, const Eigen::Matrix3d& yyF, const Eigen::Matrix3d& zzF,
                                  const Eigen::Matrix3d& xyF, const Eigen::Matrix3d& yzF, const Eigen::Matrix3d& zxF,
                                  const Eigen::Matrix<double, 3, 9>& x1P, const Eigen::Matrix<double, 3, 9>& y1P,
                                  const Eigen::Matrix<double, 3, 9>& z1P, const Eigen::Matrix<double, 3, 9>& x2P,
                                  const Eigen::Matrix<double, 3, 9>& y2P, const Eigen::Matrix<double, 3, 9>& z2P,
                                  const Eigen::Matrix<double, 9, 9>& m11P, const Eigen::Matrix<double, 9, 9>& m12P,
                                  const Eigen::Matrix<double, 9, 9>& m22P, const Eigen::Vector3d& rotation) {
    const Eigen::Matrix4d G =
        ComposeG(xxF, yyF, zzF, xyF, yzF, zxF, x1P, y1P, z1P, x2P, y2P, z2P, m11P, m12P, m22P, rotation);

    // Compute the roots in closed-form.
    // const double G00_2 = G(0,0) * G(0,0);
    const double G01_2 = G(0, 1) * G(0, 1);
    const double G02_2 = G(0, 2) * G(0, 2);
    const double G03_2 = G(0, 3) * G(0, 3);
    // const double G11_2 = G(1,1) * G(1,1);
    const double G12_2 = G(1, 2) * G(1, 2);
    const double G13_2 = G(1, 3) * G(1, 3);
    // const double G22_2 = G(2,2) * G(2,2);
    const double G23_2 = G(2, 3) * G(2, 3);
    // const double G33_2 = G(3,3) * G(3,3);

    const double B = -G(3, 3) - G(2, 2) - G(1, 1) - G(0, 0);
    const double C = -G23_2 + G(2, 2) * G(3, 3) - G13_2 - G12_2 + G(1, 1) * G(3, 3) + G(1, 1) * G(2, 2) - G03_2 -
                     G02_2 - G01_2 + G(0, 0) * G(3, 3) + G(0, 0) * G(2, 2) + G(0, 0) * G(1, 1);
    const double D = G13_2 * G(2, 2) - 2.0 * G(1, 2) * G(1, 3) * G(2, 3) + G12_2 * G(3, 3) + G(1, 1) * G23_2 -
                     G(1, 1) * G(2, 2) * G(3, 3) + G03_2 * G(2, 2) + G03_2 * G(1, 1) -
                     2.0 * G(0, 2) * G(0, 3) * G(2, 3) + G02_2 * G(3, 3) + G02_2 * G(1, 1) -
                     2.0 * G(0, 1) * G(0, 3) * G(1, 3) - 2.0 * G(0, 1) * G(0, 2) * G(1, 2) + G01_2 * G(3, 3) +
                     G01_2 * G(2, 2) + G(0, 0) * G23_2 - G(0, 0) * G(2, 2) * G(3, 3) + G(0, 0) * G13_2 +
                     G(0, 0) * G12_2 - G(0, 0) * G(1, 1) * G(3, 3) - G(0, 0) * G(1, 1) * G(2, 2);
    const double E = G03_2 * G12_2 - G03_2 * G(1, 1) * G(2, 2) - 2.0 * G(0, 2) * G(0, 3) * G(1, 2) * G(1, 3) +
                     2.0 * G(0, 2) * G(0, 3) * G(1, 1) * G(2, 3) + G02_2 * G13_2 - G02_2 * G(1, 1) * G(3, 3) +
                     2.0 * G(0, 1) * G(0, 3) * G(1, 3) * G(2, 2) - 2.0 * G(0, 1) * G(0, 3) * G(1, 2) * G(2, 3) -
                     2.0 * G(0, 1) * G(0, 2) * G(1, 3) * G(2, 3) + 2.0 * G(0, 1) * G(0, 2) * G(1, 2) * G(3, 3) +
                     G01_2 * G23_2 - G01_2 * G(2, 2) * G(3, 3) - G(0, 0) * G13_2 * G(2, 2) +
                     2.0 * G(0, 0) * G(1, 2) * G(1, 3) * G(2, 3) - G(0, 0) * G12_2 * G(3, 3) -
                     G(0, 0) * G(1, 1) * G23_2 + G(0, 0) * G(1, 1) * G(2, 2) * G(3, 3);

    const double B_pw2 = B * B;
    const double B_pw3 = B_pw2 * B;
    const double B_pw4 = B_pw3 * B;
    const double alpha = -0.375 * B_pw2 + C;
    const double beta = B_pw3 / 8.0 - B * C / 2.0 + D;
    const double gamma = -0.01171875 * B_pw4 + B_pw2 * C / 16.0 - B * D / 4.0 + E;
    const double alpha_pw2 = alpha * alpha;
    const double alpha_pw3 = alpha_pw2 * alpha;
    const double p = -alpha_pw2 / 12.0 - gamma;
    const double q = -alpha_pw3 / 108.0 + alpha * gamma / 3.0 - beta * beta / 8.0;
    const double helper1 = -p * p * p / 27.0;
    const double theta2 = pow(helper1, (1.0 / 3.0));
    const double theta1 = std::sqrt(theta2) * std::cos((1.0 / 3.0) * std::acos((-q / 2.0) / std::sqrt(helper1)));
    const double y = -(5.0 / 6.0) * alpha - ((1.0 / 3.0) * p * theta1 - theta1 * theta2) / theta2;
    const double w = std::sqrt(alpha + 2.0 * y);

    Eigen::Vector4d roots;
    roots(0) = -B / 4.0 + 0.5 * w + 0.5 * std::sqrt(-3.0 * alpha - 2.0 * y - 2.0 * beta / w);
    roots(1) = -B / 4.0 + 0.5 * w - 0.5 * std::sqrt(-3.0 * alpha - 2.0 * y - 2.0 * beta / w);
    roots(2) = -B / 4.0 - 0.5 * w + 0.5 * std::sqrt(-3.0 * alpha - 2.0 * y + 2.0 * beta / w);
    roots(3) = -B / 4.0 - 0.5 * w - 0.5 * std::sqrt(-3.0 * alpha - 2.0 * y + 2.0 * beta / w);
    return roots;
}

double ComputeCost(const Eigen::Matrix3d& xxF, const Eigen::Matrix3d& yyF, const Eigen::Matrix3d& zzF,
                   const Eigen::Matrix3d& xyF, const Eigen::Matrix3d& yzF, const Eigen::Matrix3d& zxF,
                   const Eigen::Matrix<double, 3, 9>& x1P, const Eigen::Matrix<double, 3, 9>& y1P,
                   const Eigen::Matrix<double, 3, 9>& z1P, const Eigen::Matrix<double, 3, 9>& x2P,
                   const Eigen::Matrix<double, 3, 9>& y2P, const Eigen::Matrix<double, 3, 9>& z2P,
                   const Eigen::Matrix<double, 9, 9>& m11P, const Eigen::Matrix<double, 9, 9>& m12P,
                   const Eigen::Matrix<double, 9, 9>& m22P, const Eigen::Vector3d& rotation, const int step) {
    CHECK_GE(step, 0);
    CHECK_LE(step, 1);

    const Eigen::Vector4d roots =
        ComputeEigenValue(xxF, yyF, zzF, xyF, yzF, zxF, x1P, y1P, z1P, x2P, y2P, z2P, m11P, m12P, m22P, rotation);

    if (step == 0) {
        return roots[2];
    } else if (step == 1) {
        return roots[3];
    }

    return 0;
}

Eigen::Vector3d ComputeJacobian(const Eigen::Matrix3d& xxF, const Eigen::Matrix3d& yyF, const Eigen::Matrix3d& zzF,
                                const Eigen::Matrix3d& xyF, const Eigen::Matrix3d& yzF, const Eigen::Matrix3d& zxF,
                                const Eigen::Matrix<double, 3, 9>& x1P, const Eigen::Matrix<double, 3, 9>& y1P,
                                const Eigen::Matrix<double, 3, 9>& z1P, const Eigen::Matrix<double, 3, 9>& x2P,
                                const Eigen::Matrix<double, 3, 9>& y2P, const Eigen::Matrix<double, 3, 9>& z2P,
                                const Eigen::Matrix<double, 9, 9>& m11P, const Eigen::Matrix<double, 9, 9>& m12P,
                                const Eigen::Matrix<double, 9, 9>& m22P, const Eigen::Vector3d& rotation,
                                const double current_cost, const int step) {
    Eigen::Vector3d jacobian;
    const double kEpsilon = 0.00000001;
    for (int j = 0; j < 3; j++) {
        Eigen::Vector3d cayley_j = rotation;
        cayley_j[j] += kEpsilon;
        const double cost_j =
            ComputeCost(xxF, yyF, zzF, xyF, yzF, zxF, x1P, y1P, z1P, x2P, y2P, z2P, m11P, m12P, m22P, cayley_j, step);
        jacobian(j) = cost_j - current_cost;
    }
    return jacobian;
}

Eigen::Matrix3d ComposeMwithJacobians(
    const Eigen::Matrix3d & xxF,
    const Eigen::Matrix3d & yyF,
    const Eigen::Matrix3d & zzF,
    const Eigen::Matrix3d & xyF,
    const Eigen::Matrix3d & yzF,
    const Eigen::Matrix3d & zxF,
    const Eigen::Vector3d & cayley,
    Eigen::Matrix3d & M_jac1,
    Eigen::Matrix3d & M_jac2,
    Eigen::Matrix3d & M_jac3 )
{
    typedef Eigen::Matrix3d rotation_t;
  rotation_t R = CayleyToRotationReduced(cayley);
  rotation_t R_jac1;
  rotation_t R_jac2;
  rotation_t R_jac3;

  R_jac1(0,0) = 2*cayley[0];
  R_jac1(0,1) = 2*cayley[1];
  R_jac1(0,2) = 2*cayley[2];
  R_jac1(1,0) = 2*cayley[1];
  R_jac1(1,1) = -2*cayley[0];
  R_jac1(1,2) = -2;
  R_jac1(2,0) = 2*cayley[2];
  R_jac1(2,1) = 2;
  R_jac1(2,2) = -2*cayley[0];
  R_jac2(0,0) = -2*cayley[1];
  R_jac2(0,1) = 2*cayley[0];
  R_jac2(0,2) = 2;
  R_jac2(1,0) = 2*cayley[0];
  R_jac2(1,1) = 2*cayley[1];
  R_jac2(1,2) = 2*cayley[2];
  R_jac2(2,0) = -2;
  R_jac2(2,1) = 2*cayley[2];
  R_jac2(2,2) = -2*cayley[1];
  R_jac3(0,0) = -2*cayley[2];
  R_jac3(0,1) = -2;
  R_jac3(0,2) = 2*cayley[0];
  R_jac3(1,0) = 2;
  R_jac3(1,1) = -2*cayley[2];
  R_jac3(1,2) = 2*cayley[1];
  R_jac3(2,0) = 2*cayley[0];
  R_jac3(2,1) = 2*cayley[1];
  R_jac3(2,2) = 2*cayley[2];

  //Fill the matrix M using the precomputed summation terms. Plus Jacobian.
  Eigen::Matrix3d M;
  double temp;
  temp =      R.row(2)*yyF*R.row(2).transpose();
  M(0,0)  = temp;
  temp = -2.0*R.row(2)*yzF*R.row(1).transpose();
  M(0,0) += temp;
  temp =      R.row(1)*zzF*R.row(1).transpose();
  M(0,0) += temp;
  temp = 2.0*R_jac1.row(2)*yyF*R.row(2).transpose();
  M_jac1(0,0)  = temp;
  temp = -2.0*R_jac1.row(2)*yzF*R.row(1).transpose();
  M_jac1(0,0) += temp;
  temp = -2.0*R.row(2)*yzF*R_jac1.row(1).transpose();
  M_jac1(0,0) += temp;
  temp = 2.0*R_jac1.row(1)*zzF*R.row(1).transpose();
  M_jac1(0,0) += temp;
  temp = 2.0*R_jac2.row(2)*yyF*R.row(2).transpose();
  M_jac2(0,0)  = temp;
  temp = -2.0*R_jac2.row(2)*yzF*R.row(1).transpose();
  M_jac2(0,0) += temp;
  temp = -2.0*R.row(2)*yzF*R_jac2.row(1).transpose();
  M_jac2(0,0) += temp;
  temp = 2.0*R_jac2.row(1)*zzF*R.row(1).transpose();
  M_jac2(0,0) += temp;
  temp = 2.0*R_jac3.row(2)*yyF*R.row(2).transpose();
  M_jac3(0,0)  = temp;
  temp = -2.0*R_jac3.row(2)*yzF*R.row(1).transpose();
  M_jac3(0,0) += temp;
  temp = -2.0*R.row(2)*yzF*R_jac3.row(1).transpose();
  M_jac3(0,0) += temp;
  temp = 2.0*R_jac3.row(1)*zzF*R.row(1).transpose();
  M_jac3(0,0) += temp;

  temp =      R.row(2)*yzF*R.row(0).transpose();
  M(0,1)  = temp;
  temp = -1.0*R.row(2)*xyF*R.row(2).transpose();
  M(0,1) += temp;
  temp = -1.0*R.row(1)*zzF*R.row(0).transpose();
  M(0,1) += temp;
  temp =      R.row(1)*zxF*R.row(2).transpose();
  M(0,1) += temp;
  temp = R_jac1.row(2)*yzF*R.row(0).transpose();
  M_jac1(0,1)  = temp;
  temp = R.row(2)*yzF*R_jac1.row(0).transpose();
  M_jac1(0,1) += temp;
  temp = -2.0*R_jac1.row(2)*xyF*R.row(2).transpose();
  M_jac1(0,1) += temp;
  temp = -R_jac1.row(1)*zzF*R.row(0).transpose();
  M_jac1(0,1) += temp;
  temp = -R.row(1)*zzF*R_jac1.row(0).transpose();
  M_jac1(0,1) += temp;
  temp = R_jac1.row(1)*zxF*R.row(2).transpose();
  M_jac1(0,1) += temp;
  temp = R.row(1)*zxF*R_jac1.row(2).transpose();
  M_jac1(0,1) += temp;
  temp = R_jac2.row(2)*yzF*R.row(0).transpose();
  M_jac2(0,1)  = temp;
  temp = R.row(2)*yzF*R_jac2.row(0).transpose();
  M_jac2(0,1) += temp;
  temp = -2.0*R_jac2.row(2)*xyF*R.row(2).transpose();
  M_jac2(0,1) += temp;
  temp = -R_jac2.row(1)*zzF*R.row(0).transpose();
  M_jac2(0,1) += temp;
  temp = -R.row(1)*zzF*R_jac2.row(0).transpose();
  M_jac2(0,1) += temp;
  temp = R_jac2.row(1)*zxF*R.row(2).transpose();
  M_jac2(0,1) += temp;
  temp = R.row(1)*zxF*R_jac2.row(2).transpose();
  M_jac2(0,1) += temp;
  temp = R_jac3.row(2)*yzF*R.row(0).transpose();
  M_jac3(0,1)  = temp;
  temp = R.row(2)*yzF*R_jac3.row(0).transpose();
  M_jac3(0,1) += temp;
  temp = -2.0*R_jac3.row(2)*xyF*R.row(2).transpose();
  M_jac3(0,1) += temp;
  temp = -R_jac3.row(1)*zzF*R.row(0).transpose();
  M_jac3(0,1) += temp;
  temp = -R.row(1)*zzF*R_jac3.row(0).transpose();
  M_jac3(0,1) += temp;
  temp = R_jac3.row(1)*zxF*R.row(2).transpose();
  M_jac3(0,1) += temp;
  temp = R.row(1)*zxF*R_jac3.row(2).transpose();
  M_jac3(0,1) += temp;

  temp =      R.row(2)*xyF*R.row(1).transpose();
  M(0,2)  = temp;
  temp = -1.0*R.row(2)*yyF*R.row(0).transpose();
  M(0,2) += temp;
  temp = -1.0*R.row(1)*zxF*R.row(1).transpose();
  M(0,2) += temp;
  temp =      R.row(1)*yzF*R.row(0).transpose();
  M(0,2) += temp;
  temp = R_jac1.row(2)*xyF*R.row(1).transpose();
  M_jac1(0,2)  = temp;
  temp = R.row(2)*xyF*R_jac1.row(1).transpose();
  M_jac1(0,2) += temp;
  temp = -R_jac1.row(2)*yyF*R.row(0).transpose();
  M_jac1(0,2) += temp;
  temp = -R.row(2)*yyF*R_jac1.row(0).transpose();
  M_jac1(0,2) += temp;
  temp = -2.0*R_jac1.row(1)*zxF*R.row(1).transpose();
  M_jac1(0,2) += temp;
  temp = R_jac1.row(1)*yzF*R.row(0).transpose();
  M_jac1(0,2) += temp;
  temp = R.row(1)*yzF*R_jac1.row(0).transpose();
  M_jac1(0,2) += temp;
  temp = R_jac2.row(2)*xyF*R.row(1).transpose();
  M_jac2(0,2)  = temp;
  temp = R.row(2)*xyF*R_jac2.row(1).transpose();
  M_jac2(0,2) += temp;
  temp = -R_jac2.row(2)*yyF*R.row(0).transpose();
  M_jac2(0,2) += temp;
  temp = -R.row(2)*yyF*R_jac2.row(0).transpose();
  M_jac2(0,2) += temp;
  temp = -2.0*R_jac2.row(1)*zxF*R.row(1).transpose();
  M_jac2(0,2) += temp;
  temp = R_jac2.row(1)*yzF*R.row(0).transpose();
  M_jac2(0,2) += temp;
  temp = R.row(1)*yzF*R_jac2.row(0).transpose();
  M_jac2(0,2) += temp;
  temp = R_jac3.row(2)*xyF*R.row(1).transpose();
  M_jac3(0,2)  = temp;
  temp = R.row(2)*xyF*R_jac3.row(1).transpose();
  M_jac3(0,2) += temp;
  temp = -R_jac3.row(2)*yyF*R.row(0).transpose();
  M_jac3(0,2) += temp;
  temp = -R.row(2)*yyF*R_jac3.row(0).transpose();
  M_jac3(0,2) += temp;
  temp = -2.0*R_jac3.row(1)*zxF*R.row(1).transpose();
  M_jac3(0,2) += temp;
  temp = R_jac3.row(1)*yzF*R.row(0).transpose();
  M_jac3(0,2) += temp;
  temp = R.row(1)*yzF*R_jac3.row(0).transpose();
  M_jac3(0,2) += temp;

  temp =      R.row(0)*zzF*R.row(0).transpose();
  M(1,1)  = temp;
  temp = -2.0*R.row(0)*zxF*R.row(2).transpose();
  M(1,1) += temp;
  temp =      R.row(2)*xxF*R.row(2).transpose();
  M(1,1) += temp;
  temp = 2.0*R_jac1.row(0)*zzF*R.row(0).transpose();
  M_jac1(1,1)  = temp;
  temp = -2.0*R_jac1.row(0)*zxF*R.row(2).transpose();
  M_jac1(1,1) += temp;
  temp = -2.0*R.row(0)*zxF*R_jac1.row(2).transpose();
  M_jac1(1,1) += temp;
  temp = 2.0*R_jac1.row(2)*xxF*R.row(2).transpose();
  M_jac1(1,1) += temp;
  temp = 2.0*R_jac2.row(0)*zzF*R.row(0).transpose();
  M_jac2(1,1)  = temp;
  temp = -2.0*R_jac2.row(0)*zxF*R.row(2).transpose();
  M_jac2(1,1) += temp;
  temp = -2.0*R.row(0)*zxF*R_jac2.row(2).transpose();
  M_jac2(1,1) += temp;
  temp = 2.0*R_jac2.row(2)*xxF*R.row(2).transpose();
  M_jac2(1,1) += temp;
  temp = 2.0*R_jac3.row(0)*zzF*R.row(0).transpose();
  M_jac3(1,1)  = temp;
  temp = -2.0*R_jac3.row(0)*zxF*R.row(2).transpose();
  M_jac3(1,1) += temp;
  temp = -2.0*R.row(0)*zxF*R_jac3.row(2).transpose();
  M_jac3(1,1) += temp;
  temp = 2.0*R_jac3.row(2)*xxF*R.row(2).transpose();
  M_jac3(1,1) += temp;

  temp =      R.row(0)*zxF*R.row(1).transpose();
  M(1,2)  = temp;
  temp = -1.0*R.row(0)*yzF*R.row(0).transpose();
  M(1,2) += temp;
  temp = -1.0*R.row(2)*xxF*R.row(1).transpose();
  M(1,2) += temp;
  temp =      R.row(2)*xyF*R.row(0).transpose();
  M(1,2) += temp;
  temp = R_jac1.row(0)*zxF*R.row(1).transpose();
  M_jac1(1,2)  = temp;
  temp = R.row(0)*zxF*R_jac1.row(1).transpose();
  M_jac1(1,2) += temp;
  temp = -2.0*R_jac1.row(0)*yzF*R.row(0).transpose();
  M_jac1(1,2) += temp;
  temp = -R_jac1.row(2)*xxF*R.row(1).transpose();
  M_jac1(1,2) += temp;
  temp = -R.row(2)*xxF*R_jac1.row(1).transpose();
  M_jac1(1,2) += temp;
  temp = R_jac1.row(2)*xyF*R.row(0).transpose();
  M_jac1(1,2) += temp;
  temp = R.row(2)*xyF*R_jac1.row(0).transpose();
  M_jac1(1,2) += temp;
  temp = R_jac2.row(0)*zxF*R.row(1).transpose();
  M_jac2(1,2)  = temp;
  temp = R.row(0)*zxF*R_jac2.row(1).transpose();
  M_jac2(1,2) += temp;
  temp = -2.0*R_jac2.row(0)*yzF*R.row(0).transpose();
  M_jac2(1,2) += temp;
  temp = -R_jac2.row(2)*xxF*R.row(1).transpose();
  M_jac2(1,2) += temp;
  temp = -R.row(2)*xxF*R_jac2.row(1).transpose();
  M_jac2(1,2) += temp;
  temp = R_jac2.row(2)*xyF*R.row(0).transpose();
  M_jac2(1,2) += temp;
  temp = R.row(2)*xyF*R_jac2.row(0).transpose();
  M_jac2(1,2) += temp;
  temp = R_jac3.row(0)*zxF*R.row(1).transpose();
  M_jac3(1,2)  = temp;
  temp = R.row(0)*zxF*R_jac3.row(1).transpose();
  M_jac3(1,2) += temp;
  temp = -2.0*R_jac3.row(0)*yzF*R.row(0).transpose();
  M_jac3(1,2) += temp;
  temp = -R_jac3.row(2)*xxF*R.row(1).transpose();
  M_jac3(1,2) += temp;
  temp = -R.row(2)*xxF*R_jac3.row(1).transpose();
  M_jac3(1,2) += temp;
  temp = R_jac3.row(2)*xyF*R.row(0).transpose();
  M_jac3(1,2) += temp;
  temp = R.row(2)*xyF*R_jac3.row(0).transpose();
  M_jac3(1,2) += temp;

  temp =      R.row(1)*xxF*R.row(1).transpose();
  M(2,2)  = temp;
  temp = -2.0*R.row(0)*xyF*R.row(1).transpose();
  M(2,2) += temp;
  temp =      R.row(0)*yyF*R.row(0).transpose();
  M(2,2) += temp;
  temp = 2.0*R_jac1.row(1)*xxF*R.row(1).transpose();
  M_jac1(2,2)  = temp;
  temp = -2.0*R_jac1.row(0)*xyF*R.row(1).transpose();
  M_jac1(2,2) += temp;
  temp = -2.0*R.row(0)*xyF*R_jac1.row(1).transpose();
  M_jac1(2,2) += temp;
  temp = 2.0*R_jac1.row(0)*yyF*R.row(0).transpose();
  M_jac1(2,2) += temp;
  temp = 2.0*R_jac2.row(1)*xxF*R.row(1).transpose();
  M_jac2(2,2)  = temp;
  temp = -2.0*R_jac2.row(0)*xyF*R.row(1).transpose();
  M_jac2(2,2) += temp;
  temp = -2.0*R.row(0)*xyF*R_jac2.row(1).transpose();
  M_jac2(2,2) += temp;
  temp = 2.0*R_jac2.row(0)*yyF*R.row(0).transpose();
  M_jac2(2,2) += temp;
  temp = 2.0*R_jac3.row(1)*xxF*R.row(1).transpose();
  M_jac3(2,2)  = temp;
  temp = -2.0*R_jac3.row(0)*xyF*R.row(1).transpose();
  M_jac3(2,2) += temp;
  temp = -2.0*R.row(0)*xyF*R_jac3.row(1).transpose();
  M_jac3(2,2) += temp;
  temp = 2.0*R_jac3.row(0)*yyF*R.row(0).transpose();
  M_jac3(2,2) += temp;

  M(1,0) = M(0,1);
  M(2,0) = M(0,2);
  M(2,1) = M(1,2);
  M_jac1(1,0) = M_jac1(0,1);
  M_jac1(2,0) = M_jac1(0,2);
  M_jac1(2,1) = M_jac1(1,2);
  M_jac2(1,0) = M_jac2(0,1);
  M_jac2(2,0) = M_jac2(0,2);
  M_jac2(2,1) = M_jac2(1,2);
  M_jac3(1,0) = M_jac3(0,1);
  M_jac3(2,0) = M_jac3(0,2);
  M_jac3(2,1) = M_jac3(1,2);

  return M;
}

double GetSmallestEVwithJacobian(
    const Eigen::Matrix3d & xxF,
    const Eigen::Matrix3d & yyF,
    const Eigen::Matrix3d & zzF,
    const Eigen::Matrix3d & xyF,
    const Eigen::Matrix3d & yzF,
    const Eigen::Matrix3d & zxF,
    const Eigen::Vector3d & cayley,
    Eigen::Matrix<double,1,3> & jacobian)
{
  Eigen::Matrix3d M_jac1 = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d M_jac2 = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d M_jac3 = Eigen::Matrix3d::Zero();

  Eigen::Matrix3d M = ComposeMwithJacobians(
      xxF,yyF,zzF,xyF,yzF,zxF,cayley,M_jac1,M_jac2,M_jac3);

  //Retrieve the smallest Eigenvalue by the following closed form solution.
  //Plus Jacobian.
  double b = -M(0,0)-M(1,1)-M(2,2);
  double b_jac1 = -M_jac1(0,0)-M_jac1(1,1)-M_jac1(2,2);
  double b_jac2 = -M_jac2(0,0)-M_jac2(1,1)-M_jac2(2,2);
  double b_jac3 = -M_jac3(0,0)-M_jac3(1,1)-M_jac3(2,2);
  double c =
      -pow(M(0,2),2)-pow(M(1,2),2)-pow(M(0,1),2)+
      M(0,0)*M(1,1)+M(0,0)*M(2,2)+M(1,1)*M(2,2);
  double c_jac1 =
      -2.0*M(0,2)*M_jac1(0,2)-2.0*M(1,2)*M_jac1(1,2)-2.0*M(0,1)*M_jac1(0,1)
      +M_jac1(0,0)*M(1,1)+M(0,0)*M_jac1(1,1)+M_jac1(0,0)*M(2,2)
      +M(0,0)*M_jac1(2,2)+M_jac1(1,1)*M(2,2)+M(1,1)*M_jac1(2,2);
  double c_jac2 =
      -2.0*M(0,2)*M_jac2(0,2)-2.0*M(1,2)*M_jac2(1,2)-2.0*M(0,1)*M_jac2(0,1)
      +M_jac2(0,0)*M(1,1)+M(0,0)*M_jac2(1,1)+M_jac2(0,0)*M(2,2)
      +M(0,0)*M_jac2(2,2)+M_jac2(1,1)*M(2,2)+M(1,1)*M_jac2(2,2);
  double c_jac3 =
      -2.0*M(0,2)*M_jac3(0,2)-2.0*M(1,2)*M_jac3(1,2)-2.0*M(0,1)*M_jac3(0,1)
      +M_jac3(0,0)*M(1,1)+M(0,0)*M_jac3(1,1)+M_jac3(0,0)*M(2,2)
      +M(0,0)*M_jac3(2,2)+M_jac3(1,1)*M(2,2)+M(1,1)*M_jac3(2,2);
  double d =
      M(1,1)*pow(M(0,2),2)+M(0,0)*pow(M(1,2),2)+M(2,2)*pow(M(0,1),2)-
      M(0,0)*M(1,1)*M(2,2)-2*M(0,1)*M(1,2)*M(0,2);
  double d_jac1 =
      M_jac1(1,1)*pow(M(0,2),2)+M(1,1)*2*M(0,2)*M_jac1(0,2)
      +M_jac1(0,0)*pow(M(1,2),2)+M(0,0)*2.0*M(1,2)*M_jac1(1,2)
      +M_jac1(2,2)*pow(M(0,1),2)+M(2,2)*2.0*M(0,1)*M_jac1(0,1)
      -M_jac1(0,0)*M(1,1)*M(2,2)-M(0,0)*M_jac1(1,1)*M(2,2)
      -M(0,0)*M(1,1)*M_jac1(2,2)-2.0*(M_jac1(0,1)*M(1,2)*M(0,2)
      +M(0,1)*M_jac1(1,2)*M(0,2)+M(0,1)*M(1,2)*M_jac1(0,2));
  double d_jac2 =
      M_jac2(1,1)*pow(M(0,2),2)+M(1,1)*2*M(0,2)*M_jac2(0,2)
      +M_jac2(0,0)*pow(M(1,2),2)+M(0,0)*2.0*M(1,2)*M_jac2(1,2)
      +M_jac2(2,2)*pow(M(0,1),2)+M(2,2)*2.0*M(0,1)*M_jac2(0,1)
      -M_jac2(0,0)*M(1,1)*M(2,2)-M(0,0)*M_jac2(1,1)*M(2,2)
      -M(0,0)*M(1,1)*M_jac2(2,2)-2.0*(M_jac2(0,1)*M(1,2)*M(0,2)
      +M(0,1)*M_jac2(1,2)*M(0,2)+M(0,1)*M(1,2)*M_jac2(0,2));
  double d_jac3 =
      M_jac3(1,1)*pow(M(0,2),2)+M(1,1)*2*M(0,2)*M_jac3(0,2)
      +M_jac3(0,0)*pow(M(1,2),2)+M(0,0)*2.0*M(1,2)*M_jac3(1,2)
      +M_jac3(2,2)*pow(M(0,1),2)+M(2,2)*2.0*M(0,1)*M_jac3(0,1)
      -M_jac3(0,0)*M(1,1)*M(2,2)-M(0,0)*M_jac3(1,1)*M(2,2)
      -M(0,0)*M(1,1)*M_jac3(2,2)-2.0*(M_jac3(0,1)*M(1,2)*M(0,2)
      +M(0,1)*M_jac3(1,2)*M(0,2)+M(0,1)*M(1,2)*M_jac3(0,2));

  double s = 2*pow(b,3)-9*b*c+27*d;
  double t = 4*pow((pow(b,2)-3*c),3);
  double s_jac1 = 2.0*3.0*pow(b,2)*b_jac1-9.0*b_jac1*c-9.0*b*c_jac1+27.0*d_jac1;
  double s_jac2 = 2.0*3.0*pow(b,2)*b_jac2-9.0*b_jac2*c-9.0*b*c_jac2+27.0*d_jac2;
  double s_jac3 = 2.0*3.0*pow(b,2)*b_jac3-9.0*b_jac3*c-9.0*b*c_jac3+27.0*d_jac3;
  double t_jac1 = 4.0*3.0*pow((pow(b,2)-3.0*c),2)*(2.0*b*b_jac1-3.0*c_jac1);
  double t_jac2 = 4.0*3.0*pow((pow(b,2)-3.0*c),2)*(2.0*b*b_jac2-3.0*c_jac2);
  double t_jac3 = 4.0*3.0*pow((pow(b,2)-3.0*c),2)*(2.0*b*b_jac3-3.0*c_jac3);

  double alpha = acos(s/sqrt(t));
  double alpha_jac1 =
      -1.0/sqrt(1.0-(pow(s,2)/t)) *
      (s_jac1*sqrt(t)-s*0.5*pow(t,-0.5)*t_jac1)/t;
  double alpha_jac2 =
      -1.0/sqrt(1.0-(pow(s,2)/t)) *
      (s_jac2*sqrt(t)-s*0.5*pow(t,-0.5)*t_jac2)/t;
  double alpha_jac3 =
      -1.0/sqrt(1.0-(pow(s,2)/t)) *
      (s_jac3*sqrt(t)-s*0.5*pow(t,-0.5)*t_jac3)/t;
  double beta = alpha/3;
  double beta_jac1 = alpha_jac1/3.0;
  double beta_jac2 = alpha_jac2/3.0;
  double beta_jac3 = alpha_jac3/3.0;
  double y = cos(beta);
  double y_jac1 = -sin(beta)*beta_jac1;
  double y_jac2 = -sin(beta)*beta_jac2;
  double y_jac3 = -sin(beta)*beta_jac3;

  double r = 0.5*sqrt(t);
  double r_jac1 = 0.25*pow(t,-0.5)*t_jac1;
  double r_jac2 = 0.25*pow(t,-0.5)*t_jac2;
  double r_jac3 = 0.25*pow(t,-0.5)*t_jac3;
  double w = pow(r,(1.0/3.0));
  double w_jac1 = (1.0/3.0)*pow(r,-2.0/3.0)*r_jac1;
  double w_jac2 = (1.0/3.0)*pow(r,-2.0/3.0)*r_jac2;
  double w_jac3 = (1.0/3.0)*pow(r,-2.0/3.0)*r_jac3;

  double k = w*y;
  double k_jac1 = w_jac1*y+w*y_jac1;
  double k_jac2 = w_jac2*y+w*y_jac2;
  double k_jac3 = w_jac3*y+w*y_jac3;
  double smallestEV = (-b-2*k)/3;
  double smallestEV_jac1 = (-b_jac1-2.0*k_jac1)/3.0;
  double smallestEV_jac2 = (-b_jac2-2.0*k_jac2)/3.0;
  double smallestEV_jac3 = (-b_jac3-2.0*k_jac3)/3.0;

  jacobian(0,0) = smallestEV_jac1;
  jacobian(0,1) = smallestEV_jac2;
  jacobian(0,2) = smallestEV_jac3;
  return smallestEV;
}


Eigen::Matrix3d ComposeM(
  const Eigen::Matrix3d & xxF,
  const Eigen::Matrix3d & yyF,
  const Eigen::Matrix3d & zzF,
  const Eigen::Matrix3d & xyF,
  const Eigen::Matrix3d & yzF,
  const Eigen::Matrix3d & zxF,
  const Vector3d & cayley)
{
  Eigen::Matrix3d M;
  Eigen::Matrix3d R = CayleyToRotationReduced(cayley);

  //Fill the matrix M using the precomputed summation terms
  double temp;
  temp =      R.row(2)*yyF*R.row(2).transpose();
  M(0,0)  = temp;
  temp = -2.0*R.row(2)*yzF*R.row(1).transpose();
  M(0,0) += temp;
  temp =      R.row(1)*zzF*R.row(1).transpose();
  M(0,0) += temp;

  temp =      R.row(2)*yzF*R.row(0).transpose();
  M(0,1)  = temp;
  temp = -1.0*R.row(2)*xyF*R.row(2).transpose();
  M(0,1) += temp;
  temp = -1.0*R.row(1)*zzF*R.row(0).transpose();
  M(0,1) += temp;
  temp =      R.row(1)*zxF*R.row(2).transpose();
  M(0,1) += temp;

  temp =      R.row(2)*xyF*R.row(1).transpose();
  M(0,2)  = temp;
  temp = -1.0*R.row(2)*yyF*R.row(0).transpose();
  M(0,2) += temp;
  temp = -1.0*R.row(1)*zxF*R.row(1).transpose();
  M(0,2) += temp;
  temp =      R.row(1)*yzF*R.row(0).transpose();
  M(0,2) += temp;

  temp =      R.row(0)*zzF*R.row(0).transpose();
  M(1,1)  = temp;
  temp = -2.0*R.row(0)*zxF*R.row(2).transpose();
  M(1,1) += temp;
  temp =      R.row(2)*xxF*R.row(2).transpose();
  M(1,1) += temp;

  temp =      R.row(0)*zxF*R.row(1).transpose();
  M(1,2)  = temp;
  temp = -1.0*R.row(0)*yzF*R.row(0).transpose();
  M(1,2) += temp;
  temp = -1.0*R.row(2)*xxF*R.row(1).transpose();
  M(1,2) += temp;
  temp =      R.row(2)*xyF*R.row(0).transpose();
  M(1,2) += temp;

  temp =      R.row(1)*xxF*R.row(1).transpose();
  M(2,2)  = temp;
  temp = -2.0*R.row(0)*xyF*R.row(1).transpose();
  M(2,2) += temp;
  temp =      R.row(0)*yyF*R.row(0).transpose();
  M(2,2) += temp;

  M(1,0) = M(0,1);
  M(2,0) = M(0,2);
  M(2,1) = M(1,2);

  return M;
}

}  // namespace

struct myPair {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Eigen::Vector3d first;
  double second;
};

bool operator<( const myPair & p1, const myPair & p2 ) {
  if( p1.second > p2.second )
    return true;
  return false;
}

struct Eigensolver_step : OptimizationFunctor<double>
{
  const Eigen::Matrix3d & _xxF;
  const Eigen::Matrix3d & _yyF;
  const Eigen::Matrix3d & _zzF;
  const Eigen::Matrix3d & _xyF;
  const Eigen::Matrix3d & _yzF;
  const Eigen::Matrix3d & _zxF;

  Eigensolver_step(
    const Eigen::Matrix3d & xxF,
    const Eigen::Matrix3d & yyF,
    const Eigen::Matrix3d & zzF,
    const Eigen::Matrix3d & xyF,
    const Eigen::Matrix3d & yzF,
    const Eigen::Matrix3d & zxF ) :
    OptimizationFunctor<double>(3,3),
    _xxF(xxF),_yyF(yyF),_zzF(zzF),_xyF(xyF),_yzF(yzF),_zxF(zxF) {}

  int operator()(const Eigen::VectorXd &x, Eigen::VectorXd &fvec) const
  {
    Eigen::Vector3d cayley = x;
    Eigen::Matrix<double,1,3> jacobian;
    GetSmallestEVwithJacobian(
        _xxF,_yyF,_zzF,_xyF,_yzF,_zxF,cayley,jacobian);

    fvec[0] = jacobian(0,0);
    fvec[1] = jacobian(0,1);
    fvec[2] = jacobian(0,2);
    return 0;
  }
};

std::vector<GR6PEstimator::M_t> GR6PEstimator::Estimate(const std::vector<X_t>& points1,
                                                        const std::vector<Y_t>& points2) {
    CHECK_GE(points1.size(), 6);
    CHECK_EQ(points1.size(), points2.size());

    double inv_norm = 0.0;
    std::vector<Eigen::Vector3d> proj_centers1(points1.size());
    std::vector<Eigen::Vector3d> proj_centers2(points1.size());
    std::vector<Eigen::Vector6d> plueckers1(points1.size());
    std::vector<Eigen::Vector6d> plueckers2(points1.size());
    for (size_t i = 0; i < points1.size(); ++i) {
        ComposePlueckerData(points1[i].rel_tform, points1[i].xy, &proj_centers1[i], &plueckers1[i]);
        ComposePlueckerData(points2[i].rel_tform, points2[i].xy, &proj_centers2[i], &plueckers2[i]);
        inv_norm += points1[i].weight * points1[i].weight;
    }
    inv_norm = 1.0 / std::sqrt(inv_norm);

    Eigen::Matrix3d xxF = Eigen::Matrix3d::Zero();
    Eigen::Matrix3d yyF = Eigen::Matrix3d::Zero();
    Eigen::Matrix3d zzF = Eigen::Matrix3d::Zero();
    Eigen::Matrix3d xyF = Eigen::Matrix3d::Zero();
    Eigen::Matrix3d yzF = Eigen::Matrix3d::Zero();
    Eigen::Matrix3d zxF = Eigen::Matrix3d::Zero();

    Eigen::Matrix<double, 3, 9> x1P = Eigen::Matrix<double, 3, 9>::Zero();
    Eigen::Matrix<double, 3, 9> y1P = Eigen::Matrix<double, 3, 9>::Zero();
    Eigen::Matrix<double, 3, 9> z1P = Eigen::Matrix<double, 3, 9>::Zero();
    Eigen::Matrix<double, 3, 9> x2P = Eigen::Matrix<double, 3, 9>::Zero();
    Eigen::Matrix<double, 3, 9> y2P = Eigen::Matrix<double, 3, 9>::Zero();
    Eigen::Matrix<double, 3, 9> z2P = Eigen::Matrix<double, 3, 9>::Zero();

    Eigen::Matrix<double, 9, 9> m11P = Eigen::Matrix<double, 9, 9>::Zero();
    Eigen::Matrix<double, 9, 9> m12P = Eigen::Matrix<double, 9, 9>::Zero();
    Eigen::Matrix<double, 9, 9> m22P = Eigen::Matrix<double, 9, 9>::Zero();

    for (size_t i = 0; i < points1.size(); ++i) {
        const double weight = points1[i].weight * inv_norm;

        const Eigen::Vector3d f1 = plueckers1[i].head<3>();
        const Eigen::Vector3d f2 = plueckers2[i].head<3>();
        const Eigen::Vector3d t1 = proj_centers1[i];
        const Eigen::Vector3d t2 = proj_centers2[i];

        const Eigen::Matrix3d F = f2 * f2.transpose();
        xxF += weight * f1[0] * f1[0] * F;
        yyF += weight * f1[1] * f1[1] * F;
        zzF += weight * f1[2] * f1[2] * F;
        xyF += weight * f1[0] * f1[1] * F;
        yzF += weight * f1[1] * f1[2] * F;
        zxF += weight * f1[2] * f1[0] * F;

        Eigen::Matrix<double, 9, 1> ff1;
        double temp = f2[1] * t2[2] - f2[2] * t2[1];
        ff1(0) = f1[0] * temp;
        ff1(1) = f1[1] * temp;
        ff1(2) = f1[2] * temp;
        temp = f2[2] * t2[0] - f2[0] * t2[2];
        ff1(3) = f1[0] * temp;
        ff1(4) = f1[1] * temp;
        ff1(5) = f1[2] * temp;
        temp = f2[0] * t2[1] - f2[1] * t2[0];
        ff1(6) = f1[0] * temp;
        ff1(7) = f1[1] * temp;
        ff1(8) = f1[2] * temp;

        x1P += weight * f1[0] * f2 * ff1.transpose();
        y1P += weight * f1[1] * f2 * ff1.transpose();
        z1P += weight * f1[2] * f2 * ff1.transpose();

        Eigen::Matrix<double, 9, 1> ff2;
        temp = f1[1] * t1[2] - f1[2] * t1[1];
        ff2(0) = f2[0] * temp;
        ff2(1) = f2[1] * temp;
        ff2(2) = f2[2] * temp;
        temp = f1[2] * t1[0] - f1[0] * t1[2];
        ff2(3) = f2[0] * temp;
        ff2(4) = f2[1] * temp;
        ff2(5) = f2[2] * temp;
        temp = f1[0] * t1[1] - f1[1] * t1[0];
        ff2(6) = f2[0] * temp;
        ff2(7) = f2[1] * temp;
        ff2(8) = f2[2] * temp;

        x2P += weight * f1[0] * f2 * ff2.transpose();
        y2P += weight * f1[1] * f2 * ff2.transpose();
        z2P += weight * f1[2] * f2 * ff2.transpose();

        m11P -= weight * ff1 * ff1.transpose();
        m22P -= weight * ff2 * ff2.transpose();
        m12P -= weight * ff2 * ff1.transpose();
    }

    const Eigen::Vector3d initial_rotation = ComputeRotationBetweenPoints(plueckers1, plueckers2);

    const double kMinLambda = 0.00001;
    const double kMaxLambda = 0.08;
    const double kLambdaModifier = 2.0;
    const int kMaxNumIterations = 50;
    const bool kDisableIncrements = true;

    double perturbation_amplitude = 0.3;
    int num_random_trials = 0;

    Eigen::Vector3d rotation;
    while (num_random_trials < 5) {
        if (num_random_trials > 2) {
            perturbation_amplitude = 0.6;
        }

        if (num_random_trials == 0) {
            rotation = initial_rotation;
        } else {
            const Eigen::Vector3d perturbation(RandomReal<double>(-perturbation_amplitude, perturbation_amplitude),
                                               RandomReal<double>(-perturbation_amplitude, perturbation_amplitude),
                                               RandomReal<double>(-perturbation_amplitude, perturbation_amplitude));
            rotation = initial_rotation + perturbation;
        }

        double lambda = 0.01;
        int num_iterations = 0;
        double smallest_eigen_value =
            ComputeCost(xxF, yyF, zzF, xyF, yzF, zxF, x1P, y1P, z1P, x2P, y2P, z2P, m11P, m12P, m22P, rotation, 1);

        for (int iter = 0; iter < kMaxNumIterations; ++iter) {
            const Eigen::Vector3d jacobian = ComputeJacobian(xxF, yyF, zzF, xyF, yzF, zxF, x1P, y1P, z1P, x2P, y2P, z2P,
                                                             m11P, m12P, m22P, rotation, smallest_eigen_value, 1);

            const Eigen::Vector3d normalized_jacobian = jacobian.normalized();

            Eigen::Vector3d sampling_point = rotation - lambda * normalized_jacobian;
            double sampling_eigen_value = ComputeCost(xxF, yyF, zzF, xyF, yzF, zxF, x1P, y1P, z1P, x2P, y2P, z2P, m11P,
                                                      m12P, m22P, sampling_point, 1);

            if (num_iterations == 0 || !kDisableIncrements) {
                while (sampling_eigen_value < smallest_eigen_value) {
                    smallest_eigen_value = sampling_eigen_value;
                    if (lambda * kLambdaModifier > kMaxLambda) {
                        break;
                    }
                    lambda *= kLambdaModifier;
                    sampling_point = rotation - lambda * normalized_jacobian;
                    sampling_eigen_value = ComputeCost(xxF, yyF, zzF, xyF, yzF, zxF, x1P, y1P, z1P, x2P, y2P, z2P, m11P,
                                                       m12P, m22P, sampling_point, 1);
                }
            }

            while (sampling_eigen_value > smallest_eigen_value) {
                lambda /= kLambdaModifier;
                sampling_point = rotation - lambda * normalized_jacobian;
                sampling_eigen_value = ComputeCost(xxF, yyF, zzF, xyF, yzF, zxF, x1P, y1P, z1P, x2P, y2P, z2P, m11P,
                                                   m12P, m22P, sampling_point, 1);
            }

            rotation = sampling_point;
            smallest_eigen_value = sampling_eigen_value;

            if (lambda < kMinLambda) {
                break;
            }
        }

        if (rotation.norm() < 0.01) {
            const double eigen_value2 =
                ComputeCost(xxF, yyF, zzF, xyF, yzF, zxF, x1P, y1P, z1P, x2P, y2P, z2P, m11P, m12P, m22P, rotation, 0);
            if (eigen_value2 > 0.001) {
                num_random_trials += 1;
            } else {
                break;
            }
        } else {
            break;
        }
    }

#if 0
    const int n = 3;
    Eigen::VectorXd x(n);

    x = rotation;
    Eigensolver_step functor(xxF, yyF, zzF, xyF, yzF, zxF);
    Eigen::NumericalDiff<Eigensolver_step> numDiff(functor);
    Eigen::LevenbergMarquardt<Eigen::NumericalDiff<Eigensolver_step> > lm(numDiff);

    lm.resetParameters();
    lm.parameters.ftol = 0.00005;
    lm.parameters.xtol = 1.E1*NumTraits<double>::epsilon();
    lm.parameters.maxfev = 100;
    lm.minimize(x);

    rotation = x;

    const Eigen::Matrix3d R = CayleyToRotationMatrix(rotation).transpose();

    const Eigen::Matrix4d G =
        ComposeG(xxF, yyF, zzF, xyF, yzF, zxF, x1P, y1P, z1P, x2P, y2P, z2P, m11P, m12P, m22P, rotation);

    const Eigen::EigenSolver<Eigen::Matrix4d> eigen_solver_G(G, true);
    const Eigen::Matrix4cd V = eigen_solver_G.eigenvectors();
    const Eigen::Matrix3x4d VV = V.real().colwise().hnormalized();

    std::vector<M_t> models(4);
    for (int i = 0; i < 4; ++i) {
        models[i].leftCols<3>() = R;
        models[i].rightCols<1>() = -R * VV.col(i);
    }

    return models;
#else
    const int n = 3;
    Eigen::VectorXd x(n);

    x = rotation;
    Eigensolver_step functor(xxF, yyF, zzF, xyF, yzF, zxF);
    Eigen::NumericalDiff<Eigensolver_step> numDiff(functor);
    Eigen::LevenbergMarquardt<Eigen::NumericalDiff<Eigensolver_step> > lm(numDiff);

    lm.resetParameters();
    lm.parameters.ftol = 0.00005;
    lm.parameters.xtol = 1.E1*NumTraits<double>::epsilon();
    lm.parameters.maxfev = 100;
    lm.minimize(x);

    rotation = x;

    std::vector<Eigen::Vector3d> rotations;
    rotations.push_back(rotation);

    Eigen::Matrix3d R = CayleyToRotationMatrix(rotation).transpose();
    R /= R.col(2).norm();
    Eigen::AngleAxisd angle_axis(R);
    Eigen::AngleAxisd ne_angle_axis(M_PI - angle_axis.angle(), angle_axis.axis());
    R = ne_angle_axis.matrix();
    rotations.push_back(RotationMatrixToCaley(R));

    const Eigen::Matrix3x4d inv_proj_matrix1 = InvertProjectionMatrix(points1[0].rel_tform);
    const Eigen::Vector3d bearing1 = inv_proj_matrix1.leftCols<3>() * points1[0].xy.homogeneous();
    const Eigen::Vector3d bearing_normalized1 = bearing1.normalized();
    const Eigen::Matrix3x4d inv_proj_matrix2 = InvertProjectionMatrix(points2[0].rel_tform);
    const Eigen::Vector3d bearing2 = inv_proj_matrix2.leftCols<3>() * points2[0].xy.homogeneous();
    const Eigen::Vector3d bearing_normalized2 = bearing2.normalized();
    Eigen::Vector3d optical_flow = bearing_normalized1 - R.transpose() * bearing_normalized2;

    std::vector<M_t> models(rotations.size());
    for (size_t i = 0; i < rotations.size(); ++i) {
        Eigen::Vector3d rotation = rotations.at(i);

        Eigen::Matrix3d M = ComposeM(xxF, yyF, zzF, xyF, yzF, zxF, rotation);
        Eigen::EigenSolver< Eigen::Matrix3d > eigen_solver_G(M, true);
        Eigen::Matrix<std::complex<double>,3,1> D_complex = eigen_solver_G.eigenvalues();
        Eigen::Matrix<std::complex<double>,3,3> V_complex = eigen_solver_G.eigenvectors();
        Eigen::Vector3d D;
        Eigen::Matrix3d V;
        
        std::vector< myPair > pairs;
        for(size_t i = 0; i < 3; i++) {
            myPair newPair;
            newPair.second = D_complex[i].real();
            for(size_t j = 0; j < 3; j++)
            newPair.first(j,0) = V_complex(j,i).real();
            pairs.push_back(newPair);
        }
        std::sort(pairs.begin(),pairs.end());
        for(size_t i = 0; i < 3; i++) {
            D[i] = pairs[i].second;
            V.col(i) = pairs[i].first;
        }

        const Eigen::Matrix3d R = CayleyToRotationMatrix(rotation).transpose();

        double translationMagnitude = sqrt(pow(D[0],2) + pow(D[1],2));
        // Eigen::Vector3d t = translationMagnitude * V.col(2);

        models[i].leftCols<3>() = R;
        models[i].rightCols<1>() = -R * translationMagnitude * V.col(2);

        if (models[i].rightCols<1>().dot(optical_flow) < 0.0) {
            models[i].rightCols<1>() = -models[i].rightCols<1>();
        }
    }
    return models;
#endif
}

void GR6PEstimator::Residuals(const std::vector<X_t>& points1, const std::vector<Y_t>& points2, const M_t& proj_matrix,
                              std::vector<double>* residuals) {
    CHECK_EQ(points1.size(), points2.size());

    residuals->resize(points1.size(), 0);

    Eigen::Matrix4d proj_matrix_homogeneous;
    proj_matrix_homogeneous.topRows<3>() = proj_matrix;
    proj_matrix_homogeneous.bottomRows<1>() = Eigen::Vector4d(0, 0, 0, 1);

    for (size_t i = 0; i < points1.size(); ++i) {
        const Eigen::Matrix3x4d& proj_matrix1 = points1[i].rel_tform;
        const Eigen::Matrix3x4d& proj_matrix2 = points2[i].rel_tform * proj_matrix_homogeneous;
        const Eigen::Matrix3d R12 = proj_matrix2.leftCols<3>() * proj_matrix1.leftCols<3>().transpose();
        const Eigen::Vector3d t12 = proj_matrix2.rightCols<1>() - R12 * proj_matrix1.rightCols<1>();
        const Eigen::Matrix3d E = EssentialMatrixFromPose(R12, t12);
        const Eigen::Vector3d Ex1 = E * points1[i].xy.homogeneous();
        const Eigen::Vector3d Etx2 = E.transpose() * points2[i].xy.homogeneous();
        const double x2tEx1 = points2[i].xy.homogeneous().transpose() * Ex1;
        (*residuals)[i] = x2tEx1 * x2tEx1 / (Ex1(0) * Ex1(0) + Ex1(1) * Ex1(1) + Etx2(0) * Etx2(0) + Etx2(1) * Etx2(1));
    
        // Eigen::Vector3d point1_h = (points1[i].xy.homogeneous()).normalized();
		// Eigen::Vector3d point2_h = (points2[i].xy.homogeneous()).normalized();
			
		// Eigen::Vector3d Ex1 = (E*point1_h).normalized();
		// double x2tEx1 = point2_h.transpose()*Ex1;
		// double theta1 = RadToDeg(std::asin(x2tEx1));
		// theta1 = theta1*theta1;

		// Eigen::Vector3d Etx2 = (E.transpose()*point2_h).normalized();
		// double x1tEtx2 = point1_h.transpose()*Etx2;
		// double theta2 = RadToDeg(std::asin(x1tEtx2));
		// theta2 = theta2 * theta2;

		// (*residuals)[i] = std::max(theta1,theta2);
    }
}

}  // namespace sensemap
