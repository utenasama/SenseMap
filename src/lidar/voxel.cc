#include "util/logging.h"
#include "util/misc.h"
#include "voxel.h"

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/StdVector>
#include <Eigen/SVD>

namespace sensemap {

const static double gauss_norm_factor = 1.0 / std::sqrt(8 * M_PI * M_PI * M_PI);
const static double eps = std::numeric_limits<double>::epsilon();

Voxel::Voxel(const Option & option) 
    : option_(option),
      num_elem_(0),
      m_var_({0.0, 0.0, 0.0}),
      m_cov_(Eigen::Matrix3d::Zero()),
      m_inv_cov_(Eigen::Matrix3d::Zero()),
      min_evec_(Eigen::Vector3d::Zero()),
      m_det_(0.0),
      min_eval_(0.0),
      max_eval_(0.0),
      mid_eval_(0.0),
      determined_(false),
      is_scatter_(false),
      is_plane_(false),
      is_line_(false) {}

void Voxel::Init(const std::vector<Eigen::Vector3d> & points) {
    num_elem_ = points.size();
    m_var_.setZero();
    for (const auto & point : points) {
        m_var_ += point;
    }
    m_var_ /= num_elem_;

    std::vector<Eigen::Vector3d> m_points(num_elem_);
    for (size_t i = 0; i < num_elem_; ++i) {
        m_points[i] = points[i] - m_var_;
    }

    Eigen::Matrix3d M = Eigen::Matrix3d::Zero();
    for (int i = 0; i < 3; ++i) {
        for (size_t k = 0; k < num_elem_; ++k) {
            M(i, 0) += m_points[k][i] * m_points[k][0];
            M(i, 1) += m_points[k][i] * m_points[k][1];
            M(i, 2) += m_points[k][i] * m_points[k][2];
        }
    }
    m_cov_ = M / num_elem_;

    ComputeFeature();
}

void Voxel::Add(const Eigen::Vector3d & x) {
    double u0 = m_var_[0];
    double u1 = m_var_[1];
    double u2 = m_var_[2];

    double inv_new_size = 1.0 / (num_elem_ + 1);
    double ex2 = ((m_cov_(0, 0) + u0 * u0) * num_elem_ + x[0] * x[0]) * inv_new_size;
    double exy = ((m_cov_(0, 1) + u0 * u1) * num_elem_ + x[0] * x[1]) * inv_new_size;
    double exz = ((m_cov_(0, 2) + u0 * u2) * num_elem_ + x[0] * x[2]) * inv_new_size;
    double ey2 = ((m_cov_(1, 1) + u1 * u1) * num_elem_ + x[1] * x[1]) * inv_new_size;
    double eyz = ((m_cov_(1, 2) + u1 * u2) * num_elem_ + x[1] * x[2]) * inv_new_size;
    double ez2 = ((m_cov_(2, 2) + u2 * u2) * num_elem_ + x[2] * x[2]) * inv_new_size;

    m_var_ = (m_var_ * num_elem_ + x) * inv_new_size;

    double cx2 = m_var_[0] * m_var_[0];
    double cxy = m_var_[0] * m_var_[1];
    double cxz = m_var_[0] * m_var_[2];
    double cy2 = m_var_[1] * m_var_[1];
    double cyz = m_var_[1] * m_var_[2];
    double cz2 = m_var_[2] * m_var_[2];

    m_cov_(0, 0) = ex2 - cx2;
    m_cov_(0, 1) = exy - cxy;
    m_cov_(0, 2) = exz - cxz;
    m_cov_(1, 0) = m_cov_(0, 1);
    m_cov_(1, 1) = ey2 - cy2;
    m_cov_(1, 2) = eyz - cyz;
    m_cov_(2, 0) = m_cov_(0, 2);
    m_cov_(2, 1) = m_cov_(1, 2);
    m_cov_(2, 2) = ez2 - cz2;

    num_elem_++;
}

void Voxel::Sub(const Eigen::Vector3d & x) {
    CHECK_GT(num_elem_, 0);
    if (num_elem_ == 1) {
        Reset();
    } else {
        double u0 = m_var_[0];
        double u1 = m_var_[1];
        double u2 = m_var_[2];

        double inv_new_size = 1.0 / (num_elem_ - 1);
        double ex2 = ((m_cov_(0, 0) + u0 * u0) * num_elem_ - x[0] * x[0]) * inv_new_size;
        double exy = ((m_cov_(0, 1) + u0 * u1) * num_elem_ - x[0] * x[1]) * inv_new_size;
        double exz = ((m_cov_(0, 2) + u0 * u2) * num_elem_ - x[0] * x[2]) * inv_new_size;
        double ey2 = ((m_cov_(1, 1) + u1 * u1) * num_elem_ - x[1] * x[1]) * inv_new_size;
        double eyz = ((m_cov_(1, 2) + u1 * u2) * num_elem_ - x[1] * x[2]) * inv_new_size;
        double ez2 = ((m_cov_(2, 2) + u2 * u2) * num_elem_ - x[2] * x[2]) * inv_new_size;

        m_var_ = (m_var_ * num_elem_ - x) * inv_new_size;

        double cx2 = m_var_[0] * m_var_[0];
        double cxy = m_var_[0] * m_var_[1];
        double cxz = m_var_[0] * m_var_[2];
        double cy2 = m_var_[1] * m_var_[1];
        double cyz = m_var_[1] * m_var_[2];
        double cz2 = m_var_[2] * m_var_[2];

        m_cov_(0, 0) = ex2 - cx2;
        m_cov_(0, 1) = exy - cxy;
        m_cov_(0, 2) = exz - cxz;
        m_cov_(1, 0) = m_cov_(0, 1);
        m_cov_(1, 1) = ey2 - cy2;
        m_cov_(1, 2) = eyz - cyz;
        m_cov_(2, 0) = m_cov_(0, 2);
        m_cov_(2, 1) = m_cov_(1, 2);
        m_cov_(2, 2) = ez2 - cz2;

        num_elem_--;
    }
}

void Voxel::Update(const Eigen::Vector3d & old_val, const Eigen::Vector3d & new_val) {
    CHECK_GT(num_elem_, 0);
    double inv_size = 1.0 / num_elem_;
    double u0 = m_var_[0];
    double u1 = m_var_[1];
    double u2 = m_var_[2];

    double x0 = old_val[0], x1 = old_val[1], x2 = old_val[2];
    double y0 = new_val[0], y1 = new_val[1], y2 = new_val[2];
    double ex2 = m_cov_(0, 0) + u0 * u0 + (y0 * y0 - x0 * x0) * inv_size;
    double exy = m_cov_(0, 1) + u0 * u1 + (y0 * y1 - x0 * x1) * inv_size;
    double exz = m_cov_(0, 2) + u0 * u2 + (y0 * y2 - x0 * x2) * inv_size;
    double ey2 = m_cov_(1, 1) + u1 * u1 + (y1 * y1 - x1 * x1) * inv_size;
    double eyz = m_cov_(1, 2) + u1 * u2 + (y1 * y2 - x1 * x2) * inv_size;
    double ez2 = m_cov_(2, 2) + u2 * u2 + (y2 * y2 - x2 * x2) * inv_size;

    m_var_ += (new_val - old_val) * inv_size;
    double cx2 = m_var_[0] * m_var_[0];
    double cxy = m_var_[0] * m_var_[1];
    double cxz = m_var_[0] * m_var_[2];
    double cy2 = m_var_[1] * m_var_[1];
    double cyz = m_var_[1] * m_var_[2];
    double cz2 = m_var_[2] * m_var_[2];

    m_cov_(0, 0) = ex2 - cx2;
    m_cov_(0, 1) = exy - cxy;
    m_cov_(0, 2) = exz - cxz;
    m_cov_(1, 0) = m_cov_(0, 1);
    m_cov_(1, 1) = ey2 - cy2;
    m_cov_(1, 2) = eyz - cyz;
    m_cov_(2, 0) = m_cov_(0, 2);
    m_cov_(2, 1) = m_cov_(1, 2);
    m_cov_(2, 2) = ez2 - cz2;
}

void Voxel::ComputeFeature() {
    if (num_elem_ <= 2) {
        is_scatter_ = false;
        determined_ = false;
        return ;
    }
    m_det_ = m_cov_.determinant();
    if (std::abs(m_det_) < std::numeric_limits<double>::epsilon()) {
        is_scatter_ = false;
        determined_ = false;
        return;
    }
    determined_ = true;
#if 1
    m_inv_cov_ = m_cov_.inverse();

    Eigen::EigenSolver<Eigen::Matrix3d> es(m_cov_);
    Eigen::Matrix3cd evecs = es.eigenvectors();
    Eigen::Vector3cd evals = es.eigenvalues();
    Eigen::Vector3d evalsReal;
    evalsReal = evals.real();
    Eigen::Matrix3f::Index evalsMin, evalsMax;
    evalsReal.rowwise().sum().minCoeff(&evalsMin);
    evalsReal.rowwise().sum().maxCoeff(&evalsMax);
    int evalsMid = 3 - evalsMin - evalsMax;
    Eigen::Vector3d evecMin = evecs.real().col(evalsMin);
    Eigen::Vector3d evecMid = evecs.real().col(evalsMid);
    Eigen::Vector3d evecMax = evecs.real().col(evalsMax);

    min_evec_ = evecMin.normalized();

    min_eval_ = evalsReal(evalsMin);
    max_eval_ = evalsReal(evalsMax);
    mid_eval_ = evalsReal(evalsMid);
#else
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(m_cov_, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();
    Eigen::Vector3d S = svd.singularValues();
    
    Eigen::Vector3d evecMin = V.col(2);
    Eigen::Vector3d evecMid = V.col(1);
    Eigen::Vector3d evecMax = V.col(0);

    min_eval_ = S[2];
    max_eval_ = S[0];
    mid_eval_ = S[1];
    
    S(2) *= 0.1;

    m_inv_cov_ = (U * S.asDiagonal() * V.transpose()).inverse();
#endif

    min_max_eigen_ratio_ = (min_eval_ + eps) / (max_eval_ + eps);
    min_mid_eigen_ratio_ = (min_eval_ + eps) / (mid_eval_ + eps);
    mid_max_eigen_ratio_ = (mid_eval_ + eps) / (max_eval_ + eps);

    // if (min_max_eigen_ratio > 0.5 && min_mid_eigen_ratio > 0.2) {
    //     is_scatter_ = true;
    // }

    // bool plane_feature = false, line_feature = false;
    // if (min_eval_ < option_.plane_min_eigen_eval && 
    //     min_max_eigen_ratio < option_.plane_min_max_eigen_ratio && 
    //     min_mid_eigen_ratio < option_.plane_min_mid_eigen_ratio && 
    //     mid_max_eigen_ratio > option_.plane_mid_max_eigen_ratio) {
    //     plane_feature = true;
    // } else if (min_max_eigen_ratio < option_.line_min_max_eigen_ratio && 
    //            min_mid_eigen_ratio > option_.line_min_mid_eigen_ratio) {
    //     line_feature = true;
    // }
    // is_plane_ = plane_feature;
    // is_line_ = line_feature;

    // if (option_.verbose) {
    //     if (is_line_ || is_plane_) {
    //         std::cout << "+----------------------------------------------------------------+" << std::endl;
    //         std::cout << StringPrintf("min eigen value: %f, eigen vector %f %f %f\n", min_eval_, evecMin(0), evecMin(1), evecMin(2));
    //         std::cout << StringPrintf("mid eigen value: %f, eigen vector %f %f %f\n", mid_eval_, evecMid(0), evecMid(1), evecMid(2));
    //         std::cout << StringPrintf("max eigen value: %f, eigen vector %f %f %f\n", max_eval_, evecMax(0), evecMax(1), evecMax(2));
            
    //         std::cout << StringPrintf("min_max_eigen_ratio: %f\n", min_max_eigen_ratio);
    //         std::cout << StringPrintf("min_mid_eigen_ratio: %f\n", min_mid_eigen_ratio);
    //         std::cout << StringPrintf("mid_max_eigen_ratio: %f\n", mid_max_eigen_ratio);
    //         std::cout << StringPrintf("is_plane_: %d\n", (int)is_plane_);
    //         std::cout << StringPrintf("is_line_: %d\n", (int)is_line_);
    //     }
    // }
}

void Voxel::SetFeature(const bool is_feature) {
    is_plane_ = is_feature;
}

double Voxel::Error(const Eigen::Vector3d & x) {
    Eigen::Vector3d ray = x - m_var_;
    double proj_len = ray.dot(min_evec_);
    Eigen::Vector3d residual_vec = min_evec_ * proj_len;
    // Eigen::Vector3d residual_vec = x - m_var_;
    double y = 0.5 * residual_vec.transpose() * m_inv_cov_ * residual_vec;
    double error = 1.0 - std::exp(-y);
    return error;
}

void Voxel::Reset() {
    num_elem_ = 0;
    m_var_.setZero();
    m_cov_.setZero();
    m_inv_cov_.setZero();
    m_det_ = 0.0;
    min_eval_ = 0.0;
    max_eval_ = 0.0;
    mid_eval_ = 0.0;
    is_plane_ = false;
    is_line_ = false;
    determined_ = false;
    is_scatter_ = false;
    min_evec_.setZero();
}

}