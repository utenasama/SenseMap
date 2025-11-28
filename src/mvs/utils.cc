#include <Eigen/Dense>
#include <fstream>

#include "util/logging.h"
#include "util/endian.h"
#include "base/triangulation.h"

#include "mvs/utils.h"

namespace sensemap {
namespace mvs {
namespace utility {

void ComputeRelativePose(const float R1[9], const float T1[3],
                         const float R2[9], const float T2[3], 
                         float R[9], float T[3]) {
  const Eigen::Map<const Eigen::RowMatrix3f> R1_m(R1);
  const Eigen::Map<const Eigen::RowMatrix3f> R2_m(R2);
  const Eigen::Map<const Eigen::Vector3f> T1_m(T1);
  const Eigen::Map<const Eigen::Vector3f> T2_m(T2);
  Eigen::Map<Eigen::RowMatrix3f> R_m(R);
  Eigen::Map<Eigen::Vector3f> T_m(T);

  R_m = R2_m * R1_m.transpose();
  T_m = T2_m - R_m * T1_m;
}

void ComputeProjectionCenter(const float R[9], const float T[3], float C[3]) {
  const Eigen::Map<const Eigen::RowMatrix3f> R_m(R);
  const Eigen::Map<const Eigen::Vector3f> T_m(T);
  Eigen::Map<Eigen::Vector3f> C_m(C);
  C_m = -R_m.transpose() * T_m;
}

void ComposeProjectionMatrix(const float K[9], const float R[9],
                             const float T[3], float P[12]) {
  Eigen::Map<Eigen::RowMatrix3x4f> P_m(P);
  P_m.leftCols<3>() =
      Eigen::Map<const Eigen::RowMatrix3f>(R);
  P_m.rightCols<1>() = Eigen::Map<const Eigen::Vector3f>(T);
  P_m = Eigen::Map<const Eigen::RowMatrix3f>(K) * P_m;
}

void ComposeInverseProjectionMatrix(const float K[9], const float R[9],
                                    const float T[3], float inv_P[12]) {
  Eigen::RowMatrix4f P;
  ComposeProjectionMatrix(K, R, T, P.data());
  P.row(3) = Eigen::Vector4f(0, 0, 0, 1);
  const Eigen::Matrix4f inv_P_temp = P.inverse();
  Eigen::Map<Eigen::RowMatrix3x4f> inv_P_m(inv_P);
  inv_P_m = inv_P_temp.topRows<3>();
}

void RotatePose(const float RR[9], float R[9], float T[3]) {
  Eigen::Map<Eigen::RowMatrix3f> R_m(R);
  Eigen::Map<Eigen::Vector3f> T_m(T);
  const Eigen::Map<const Eigen::RowMatrix3f> RR_m(RR);
  R_m = Eigen::RowMatrix3f(RR_m * R_m);
  T_m = Eigen::Vector3f(RR_m * T_m);
}

float Footprint(const float K[9], const float R[9], const float t[3],
                const float X[3]) {
    const float point_depth = R[6] * X[0] + R[7] * X[1] + R[8] * X[2] + t[2];
    // const float m_focal_length = (K[0] + K[4]) * 0.5f;
    const float m_focal_length = K[0];
    return m_focal_length / point_depth;
}

} // namespace utility
} // namespace mvs
} // namespace sensemap