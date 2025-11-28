//Copyright (c) 2019, SenseTime Group.
//All rights reserved.
#ifndef SENSEMAP_UTIL_JACOBIAN_H_
#define SENSEMAP_UTIL_JACOBIAN_H_
#include <Eigen/Core>

namespace sensemap{

void Point3DNormalizationJacobian(const double* point, const double normal, Eigen::Matrix3d& jacobian);
void QuaternionJacobian(const double* q, const double* point, Eigen::Map<Eigen::Matrix<double,3,4,Eigen::RowMajor>>& jacobian);    

}

#endif