//Copyright (c) 2019, SenseTime Group.
//All rights reserved.
#ifndef SENSEMAP_OPTIM_UTILS_H_
#define SENSEMAP_OPTIM_UTILS_H_

#include <Eigen/Core>

#include <Eigen/Geometry>
#include "util/types.h"

namespace sensemap{

namespace globalmotion{

template <class Collection>
const typename Collection::value_type::second_type& FindOrDie(
                const Collection& collection,
                const typename Collection::value_type::first_type& key){
    
    typename Collection::const_iterator it = collection.find(key);
    CHECK(it != collection.end()) << "Map key not found: " << key;
    return it->second;
}

Eigen::Matrix3d VectorToRotationMatrix(Eigen::Vector3d rotation_vec);

Eigen::Vector3d RotationMatrixToVector(Eigen::Matrix3d rotation_mat);


Eigen::Vector3d MultiplyRotations(const Eigen::Vector3d& rotation1,
                                  const Eigen::Vector3d& rotation2);

}//namespace globalmotion

void ConvertSIM3tosim3(Eigen::Vector7d& spose, const Eigen::Quaterniond& qvec, const Eigen::Vector3d& tvec, const double& scale);

void Convertsim3toSIM3(const Eigen::Vector7d& spose, Eigen::Quaterniond& qvec, Eigen::Vector3d& tvec, double& scale);

}//namespace sensemap
#endif //SENSEMAP_OPTIM_UTILS_H_