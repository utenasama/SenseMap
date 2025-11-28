//Copyright (c) 2019, SenseTime Group.
//All rights reserved.

#include "jacobian.h"

namespace sensemap{

void Point3DNormalizationJacobian(const double* point, const double normal, Eigen::Matrix3d& jacobian){
    double normal3_inverse = 1.0 / (normal* normal * normal);
    double xy = point[0] * point[1];
    double xz = point[0] * point[2];
    double yz = point[1] * point[2];
    double x2 = point[0] * point[0];
    double y2 = point[1] * point[1];
    double z2 = point[2] * point[2];

    double xy_by_normal3 = xy*normal3_inverse;
    double xz_by_normal3 = xz*normal3_inverse;
    double yz_by_normal3 = yz*normal3_inverse;


    jacobian<<(y2+z2)*normal3_inverse, -xy_by_normal3, -xz_by_normal3,
              -xy_by_normal3, (x2+z2)*normal3_inverse, -yz_by_normal3,
              -xz_by_normal3, -yz_by_normal3, (x2+y2)*normal3_inverse;

}

void QuaternionJacobian(const double* q, const double* point, Eigen::Map<Eigen::Matrix<double,3,4,Eigen::RowMajor>>& jacobian){
    double qw = q[0], qx = q[1], qy = q[2], qz= q[3];
    double x = point[0], y = point[1], z = point[2];

    double qw_x = qw*x*2.0;
    double qw_y = qw*y*2.0;
    double qw_z = qw*z*2.0;
    
    double qx_x = qx*x*2.0;
    double qx_y = qx*y*2.0;
    double qx_z = qx*z*2.0;
    
    double qy_x = qy*x*2.0;
    double qy_y = qy*y*2.0;
    double qy_z = qy*z*2.0;

    double qz_x = qz*x*2.0;
    double qz_y = qz*y*2.0;
    double qz_z = qz*z*2.0;
    
    jacobian<<qy_z - qz_y, qy_y + qz_z, qw_z + qx_y -2*qy_x, -qw_y + qx_z -2*qz_x,
              -qx_z + qz_x, -qw_z -2*qx_y + qy_x, qx_x + qz_z, qw_x + qy_z - 2*qz_y,
              qx_y - qy_x, qw_y - 2*qx_z +qz_x, -qw_x -2*qy_z + qz_y, qx_x + qy_y; 

    // jacobian<<qy*z - qz*y, qy*y + qz*z, qw*z + qx*y -2*qy*x, -qw*y + qx*z -2*qz*x,
    //           -qx*z + qz*x, -qw*z -2*qx*y + qy*x, qx*x + qz*z, qw*x + qy*z - 2*qz*y,
    //           qx*y - qy*x, qw*y - 2*qx*z +qz*x, -qw*x -2*qy*z + qz*y, qx*x + qy*y; 

    // jacobian = jacobian * 2.0;

}    

}