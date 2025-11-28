//Copyright (c) 2023, SenseTime Group.
//All rights reserved.

#ifndef SENSEMAP_LIDAR_VOXEL_H_
#define SENSEMAP_LIDAR_VOXEL_H_

#define GAUSSIAN_FEATURE 0
#define PLANE_FEATURE 1

#define FEATURE_TYPE GAUSSIAN_FEATURE

#include "util/types.h"

namespace sensemap {

class Voxel {
public:
enum FeatureType {
    NONE = 0,
    PLANE = 1,
    LINE = 2
};

struct Option {
    bool verbose = false;
    double plane_min_eigen_eval = 0.03;
    double plane_min_max_eigen_ratio = 0.03;
    double plane_min_mid_eigen_ratio = 0.03;
    double plane_mid_max_eigen_ratio = 0.6;
    double line_min_max_eigen_ratio = 0.01;
    double line_min_mid_eigen_ratio = 0.9;
};

public:
    Voxel(const Option & option);

    inline Eigen::Vector3d GetEx();

    inline Eigen::Matrix3d GetCov();

    inline Eigen::Matrix3d GetInvCov();

    inline Eigen::Vector3d GetPivot();

    inline double GetMinEigen();

    inline bool IsDetermined();

    inline bool IsScatter();

    inline bool IsFeature();

    inline FeatureType FeatureType();

    void Init(const std::vector<Eigen::Vector3d> & points);

    void Add(const Eigen::Vector3d & x);

    void Sub(const Eigen::Vector3d & x);

    void Update(const Eigen::Vector3d & old_val, const Eigen::Vector3d & new_val);

    void ComputeFeature();

    void SetFeature(const bool is_feature);

    double Error(const Eigen::Vector3d & x);

    void Reset();

    double min_max_eigen_ratio_;
    double min_mid_eigen_ratio_;
    double mid_max_eigen_ratio_;

    double min_eval_;
    double max_eval_;
    double mid_eval_;
private:
    Option option_;

    size_t num_elem_;

#if FEATURE_TYPE == GAUSSIAN_FEATURE
    // mean value of multi-variable gaussian.
    Eigen::Vector3d m_var_;
    // covariance of multi-variable gaussian.
    Eigen::Matrix3d m_cov_;

    Eigen::Matrix3d m_inv_cov_;
    double m_det_;

    Eigen::Vector3d min_evec_;

    bool determined_;
    bool is_scatter_;
    bool is_plane_;
    bool is_line_;
#else
    Eigen::Vector4d m_plane_;
#endif
};

Eigen::Vector3d Voxel::GetEx() { return m_var_; }

Eigen::Matrix3d Voxel::GetCov() { return m_cov_; }

Eigen::Matrix3d Voxel::GetInvCov() { return m_inv_cov_; }

Eigen::Vector3d Voxel::GetPivot() { return min_evec_; }

double Voxel::GetMinEigen() { return min_eval_; }

bool Voxel::IsDetermined() { return determined_; }

bool Voxel::IsScatter() { return is_scatter_; }

bool Voxel::IsFeature() { return (is_plane_ || is_line_); }

enum Voxel::FeatureType Voxel::FeatureType() {
    if (is_plane_) return FeatureType::PLANE;
    else if (is_line_) return FeatureType::LINE;
    else return FeatureType::NONE;
}

}

#endif