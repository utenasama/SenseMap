//Copyright (c) 2019, SenseTime Group.
//All rights reserved.

#include "utils.h"

namespace sensemap{

namespace globalmotion{

Eigen::Matrix3d VectorToRotationMatrix(Eigen::Vector3d rotation_vec){
    double angle=rotation_vec.norm();
    if(angle==0){
        Eigen::Matrix3d rotation_mat=Eigen::Matrix3d::Identity();
        return rotation_mat;
    }

    Eigen::Vector3d axis=rotation_vec.normalized(); 
    Eigen::AngleAxisd angle_axis(angle, axis);
    
    Eigen::Matrix3d rotation_mat=angle_axis.toRotationMatrix();
    return rotation_mat;
}

Eigen::Vector3d RotationMatrixToVector(Eigen::Matrix3d rotation_mat){
    Eigen::AngleAxisd angle_axis(rotation_mat);
    Eigen::Vector3d rotation_vec=angle_axis.angle()*angle_axis.axis();
    return rotation_vec;
}




Eigen::Vector3d MultiplyRotations(const Eigen::Vector3d& rotation1,
                                  const Eigen::Vector3d& rotation2) {
    //cautious for the negative rotation***********************************
    Eigen::Matrix3d rotation1_mat, rotation2_mat;
    
    rotation1_mat = VectorToRotationMatrix(rotation1);
    rotation2_mat = VectorToRotationMatrix(rotation2);

    //ceres::AngleAxisToRotationMatrix(rotation1.data(), rotation1_mat.data());
    //ceres::AngleAxisToRotationMatrix(rotation2.data(), rotation2_mat.data());

    const Eigen::Matrix3d rotation = rotation1_mat * rotation2_mat;
    Eigen::Vector3d rotation_vec = RotationMatrixToVector(rotation);
    //ceres::RotationMatrixToAngleAxis(rotation.data(), rotation_aa.data());
    
    return rotation_vec;
}

}//namespace globalmotion

void ConvertSIM3tosim3(Eigen::Vector7d& spose, const Eigen::Quaterniond& qvec, const Eigen::Vector3d& tvec, const double& scale){
    Eigen::Vector3d r;
    const double& q1 = qvec.x();
    const double& q2 = qvec.y();
    const double& q3 = qvec.z();
    const double sin_squared_theta = q1 * q1 + q2 * q2 + q3 * q3;
    if (sin_squared_theta > 1e-5) {
        const double sin_theta = sqrt(sin_squared_theta);
        const double& cos_theta = qvec.w();
        const double two_theta = 2.0 * ((cos_theta < 0.0)
                                        ? atan2(-sin_theta, -cos_theta)
                                        : atan2(sin_theta, cos_theta));
        const double k = two_theta / sin_theta;
        r << q1 * k, q2 * k, q3 * k;
    } else {
        const double k = 2.0;
        r << q1 * k, q2 * k, q3 * k;
    }
    spose[0] = r[0];
    spose[1] = r[1];
    spose[2] = r[2];
    spose[3] = tvec[0];
    spose[4] = tvec[1];
    spose[5] = tvec[2];
    spose[6] = scale;
}

void Convertsim3toSIM3(const Eigen::Vector7d& spose, Eigen::Quaterniond& qvec, Eigen::Vector3d& tvec, double& scale){
    const double& a0 = spose[0];
    const double& a1 = spose[1];
    const double& a2 = spose[2];
    const double theta_squared = a0 * a0 + a1 * a1 + a2 * a2;

    double q0, q1, q2, q3;
    if (theta_squared > 1e-5) {
        const double theta = sqrt(theta_squared);
        const double half_theta = theta * 0.5;
        const double k = sin(half_theta) / theta;
        q0 = cos(half_theta);
        q1 = a0 * k;
        q2 = a1 * k;
        q3 = a2 * k;
    } else {
        const double k = 0.5;
        q0 = 1.0;
        q1 = a0 * k;
        q2 = a1 * k;
        q3 = a2 * k;
    }
    qvec = Eigen::Quaterniond(q0, q1, q2, q3);
    tvec[0] = spose[3];
    tvec[1] = spose[4];
    tvec[2] = spose[5];
    scale = spose[6];
}
}//namespace sensemap