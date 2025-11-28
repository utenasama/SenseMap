#include "util/types.h"
#include <Eigen/Dense>

namespace sensemap {
struct ICPLink{
    ///X * current_point = reference_point
    ICPLink():ref_id_(kInvalidImageId){};
    ICPLink(image_t id, int cnt, Eigen::Matrix4d p, Eigen::Matrix<double, 6, 6> info):
            ref_id_(id), count_(cnt), X_(p), infomation_(info){};
    image_t ref_id_ = kInvalidImageId;
    int count_ = 0;
    float conf_ = 1.0f;
    Eigen::Matrix4d X_;
    Eigen::Matrix<double, 6, 6> infomation_;
    Eigen::Matrix4d icp_pose_ = Eigen::Matrix4d::Identity();
};
}