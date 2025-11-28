#include "util/logging.h"
#include "plane_estimator.h"

namespace sensemap {

std::vector<PlaneEstimator::M_t> PlaneEstimator::Estimate(
    const std::vector<X_t>& points) {
    CHECK_EQ(points.size(), 3);
    X_t v1 = points[1] - points[0];
    X_t v2 = points[2] - points[0];
    X_t normal = v1.cross(v2).normalized();

    M_t model;
    model[0] = normal[0];
    model[1] = normal[1];
    model[2] = normal[2];
    model[3] = -(normal[0] * points[0][0] + 
                 normal[1] * points[0][1] + 
                 normal[2] * points[0][2]);
    
    return std::vector<M_t>{ model };
}

void PlaneEstimator::Residuals(
    const std::vector<PlaneEstimator::X_t>& points, 
    const PlaneEstimator::M_t& H,
    std::vector<double>* residuals) {
    residuals->resize(points.size());
    for (size_t i = 0; i < points.size(); ++i) {
        auto point = points[i];
        double err = H[0] * point[0] + H[1] * point[1] + H[2] * point[2] + H[3];
        (*residuals)[i] = err * err;
    }
}

std::vector<PlaneLocalEstimator::M_t> PlaneLocalEstimator::Estimate(
    const std::vector<X_t>& points) {

    Eigen::Matrix4d M = Eigen::Matrix4d::Zero();
    for (size_t i = 0; i < points.size(); ++i) {
        Eigen::Vector4d hpoint = points[i].homogeneous();
        for (int j = 0; j < 4; ++j) {
            M(j, 0) += hpoint[j] * hpoint[0];
            M(j, 1) += hpoint[j] * hpoint[1];
            M(j, 2) += hpoint[j] * hpoint[2];
            M(j, 3) += hpoint[j] * hpoint[3];
        }
    }

    Eigen::JacobiSVD<Eigen::Matrix4d> svd(M, Eigen::ComputeFullU | Eigen::ComputeFullV);
    M_t model = svd.matrixV().col(3);
    double norm = model.head<3>().norm();
    model /= norm;
    
    return std::vector<M_t>{ model };
}

void PlaneLocalEstimator::Residuals(
    const std::vector<PlaneLocalEstimator::X_t>& points, 
    const PlaneLocalEstimator::M_t& H,
    std::vector<double>* residuals) {
    residuals->resize(points.size());
    for (size_t i = 0; i < points.size(); ++i) {
        auto point = points[i];
        double err = H[0] * point[0] + H[1] * point[1] + H[2] * point[2] + H[3];
        (*residuals)[i] = err * err;
    }
}
  
std::vector<WeightedPlaneEstimator::M_t> WeightedPlaneEstimator::Estimate(
    const std::vector<X_t>& points) {
    CHECK_EQ(points.size(), 3);
    Eigen::Vector3d v1 = (points[1] - points[0]).head<3>();
    Eigen::Vector3d v2 = (points[2] - points[0]).head<3>();
    Eigen::Vector3d normal = v1.cross(v2).normalized();

    M_t model;
    model[0] = normal[0];
    model[1] = normal[1];
    model[2] = normal[2];
    model[3] = -(normal[0] * points[0][0] + 
                 normal[1] * points[0][1] + 
                 normal[2] * points[0][2]);
    
    return std::vector<M_t>{ model };
}

void WeightedPlaneEstimator::Residuals(
    const std::vector<WeightedPlaneEstimator::X_t>& points, 
    const WeightedPlaneEstimator::M_t& H,
    std::vector<double>* residuals) {
    residuals->resize(points.size());
    for (size_t i = 0; i < points.size(); ++i) {
        auto point = points[i];
        double err = H[0] * point[0] + H[1] * point[1] + H[2] * point[2] + H[3];
        double angle_regu = 1.1 - H[0] * point[3] + H[1] * point[4] + H[2] * point[5];
        (*residuals)[i] = angle_regu * err * err;
    }
}

std::vector<WeightedPlaneLocalEstimator::M_t> WeightedPlaneLocalEstimator::Estimate(
    const std::vector<X_t>& points) {

    Eigen::Matrix4d M = Eigen::Matrix4d::Zero();
    for (size_t i = 0; i < points.size(); ++i) {
        Eigen::Vector4d hpoint = Eigen::Vector4d(points[i][0], points[i][1], points[i][2], 1.0);
        for (int j = 0; j < 4; ++j) {
            M(j, 0) += hpoint[j] * hpoint[0];
            M(j, 1) += hpoint[j] * hpoint[1];
            M(j, 2) += hpoint[j] * hpoint[2];
            M(j, 3) += hpoint[j] * hpoint[3];
        }
    }

    Eigen::JacobiSVD<Eigen::Matrix4d> svd(M, Eigen::ComputeFullU | Eigen::ComputeFullV);
    M_t model = svd.matrixV().col(3);
    double norm = model.head<3>().norm();
    model /= norm;
    
    return std::vector<M_t>{ model };
}

void WeightedPlaneLocalEstimator::Residuals(
    const std::vector<WeightedPlaneLocalEstimator::X_t>& points, 
    const WeightedPlaneLocalEstimator::M_t& H,
    std::vector<double>* residuals) {
    residuals->resize(points.size());
    for (size_t i = 0; i < points.size(); ++i) {
        auto point = points[i];
        double err = H[0] * point[0] + H[1] * point[1] + H[2] * point[2] + H[3];
        double angle_regu = 1.1 - H[0] * point[3] + H[1] * point[4] + H[2] * point[5];
        (*residuals)[i] = angle_regu * err * err;
    }
}

} // namespace sensemap