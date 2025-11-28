//Copyright (c) 2019, SenseTime Group.
//All rights reserved.
#ifndef SENSEMAP_ESTIMATORS_MOTION_AVERAGE_SIMILARITY_AVERAGE_H_
#define SENSEMAP_ESTIMATORS_MOTION_AVERAGE_SIMILARITY_AVERAGE_H_

#include <vector>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include "util/types.h"

namespace sensemap{

class SimilarityAverageEstimator{
public:
    size_t SimilarityFromCrossMotion(
            std::vector<std::pair<Eigen::Matrix3d, Eigen::Matrix3d>> r_A_B,
            std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> t_A_B,
            Eigen::Matrix3x4d& transform);

};

}//namespace sensemap


#endif //SENSEMAP_ESTIMATORS_MOTION_AVERAGE_SIMILARITY_AVERAGE_H_