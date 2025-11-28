// Copyright (c) 2019, SenseTime Group.
// All rights reserved.

#include <set>
#include <unordered_set>

#include "base/camera.h"
#include "base/essential_matrix.h"
#include "base/homography_matrix.h"
#include "base/pose.h"
#include "base/projection.h"
#include "base/triangulation.h"
#include "estimators/essential_matrix.h"
#include "estimators/fundamental_matrix.h"
#include "estimators/homography_matrix.h"
#include "estimators/translation_transform.h"
#include "optim/ransac/loransac.h"

#include "estimators/two_view_geometry.h"
#include "estimators/motion_average/translation_average.h"
#include "estimators/motion_average/rotation_average.h"
#include "base/reconstruction.h"

#include <boost/filesystem/path.hpp>
#include <boost/filesystem.hpp>

#include "estimators/generalized_relative_pose.h"


namespace sensemap {

namespace {

FeatureMatches ExtractInlierMatches(const FeatureMatches& matches, const size_t num_inliers,
                                    const std::vector<char>& inlier_mask) {
    FeatureMatches inlier_matches(num_inliers);
    size_t j = 0;
    for (size_t i = 0; i < matches.size(); ++i) {
        if (inlier_mask[i]) {
            inlier_matches[j] = matches[i];
            j += 1;
        }
    }
    return inlier_matches;
}

FeatureMatches ExtractOutlierMatches(const FeatureMatches& matches, const FeatureMatches& inlier_matches) {
    CHECK_GE(matches.size(), inlier_matches.size());

    std::set<std::pair<point2D_t, point2D_t>> inlier_matches_set;
    for (const auto& match : inlier_matches) {
        inlier_matches_set.emplace(match.point2D_idx1, match.point2D_idx2);
    }

    FeatureMatches outlier_matches;
    outlier_matches.reserve(matches.size() - inlier_matches.size());

    for (const auto& match : matches) {
        if (inlier_matches_set.count(std::make_pair(match.point2D_idx1, match.point2D_idx2)) == 0) {
            outlier_matches.push_back(match);
        }
    }

    return outlier_matches;
}

inline bool IsImagePointInBoundingBox(const Eigen::Vector2d& point, const double minx, const double maxx,
                                      const double miny, const double maxy) {
    return point.x() >= minx && point.x() <= maxx && point.y() >= miny && point.y() <= maxy;
}

}  // namespace

void TwoViewGeometry::Invert() {
    F.transposeInPlace();
    E.transposeInPlace();
    H = H.inverse().eval();

    const Eigen::Vector4d orig_qvec = qvec;
    const Eigen::Vector3d orig_tvec = tvec;
    InvertPose(orig_qvec, orig_tvec, &qvec, &tvec);

    for (auto& match : inlier_matches) {
        std::swap(match.point2D_idx1, match.point2D_idx2);
    }
}

void TwoViewGeometry::EstimateMultipleHomographies(const std::vector<Eigen::Vector2d>& points1,
                                                   const std::vector<Eigen::Vector2d>& points2,
                                                   const FeatureMatches& matches, const Options& options,
                                                   std::vector<Eigen::Matrix3d>& h_list) {
    FeatureMatches remaining_matches = matches;

    LORANSAC<HomographyMatrixEstimator, HomographyMatrixEstimator> H_ransac(options.ransac_options);

    for (size_t imodel = 0; imodel < options.max_H_num_ENFT; ++imodel) {
        std::vector<Eigen::Vector2d> matched_points1(remaining_matches.size());
        std::vector<Eigen::Vector2d> matched_points2(remaining_matches.size());
        for (size_t i = 0; i < remaining_matches.size(); ++i) {
            const point2D_t idx1 = remaining_matches[i].point2D_idx1;
            const point2D_t idx2 = remaining_matches[i].point2D_idx2;
            matched_points1[i] = points1[idx1];
            matched_points2[i] = points2[idx2];
        }

        const auto H_report = H_ransac.Estimate(matched_points1, matched_points2);

        if (!H_report.success || H_report.support.num_inliers < options.min_num_inliers_per_H_ENFT) {
            break;
        }

        h_list.push_back(H_report.model);

        FeatureMatches inlier_matches_ =
            ExtractInlierMatches(remaining_matches, H_report.support.num_inliers, H_report.inlier_mask);

        remaining_matches = ExtractOutlierMatches(remaining_matches, inlier_matches_);
    }
}

void TwoViewGeometry::Estimate(const Camera& camera1, const std::vector<Eigen::Vector2d>& points1,
                               const Camera& camera2, const std::vector<Eigen::Vector2d>& points2,
                               const FeatureMatches& matches, const Options& options) {
    if (camera1.ModelName().compare("SPHERICAL") == 0 || camera2.ModelName().compare("SPHERICAL") == 0) {
        if(camera1.ModelName().compare("SPHERICAL") == 0 && camera2.ModelName().compare("SPHERICAL") == 0)
            EstimateSpherical(camera1, points1, camera2, points2, matches, options);
        else
            EstimatePespectiveAndSpherical(camera1, points1, camera2, points2, matches, options);
    } else if (camera1.HasPriorFocalLength() && camera2.HasPriorFocalLength()) {
        EstimateCalibrated(camera1, points1, camera2, points2, matches, options);
    } else {
        EstimateUncalibrated(camera1, points1, camera2, points2, matches, options);
    }
}

void TwoViewGeometry::EstimateMultiple(const Camera& camera1, const std::vector<Eigen::Vector2d>& points1,
                                       const Camera& camera2, const std::vector<Eigen::Vector2d>& points2,
                                       const FeatureMatches& matches, const Options& options) {
    FeatureMatches remaining_matches = matches;
    std::vector<TwoViewGeometry> two_view_geometries;
    while (true) {
        TwoViewGeometry two_view_geometry;
        two_view_geometry.Estimate(camera1, points1, camera2, points2, remaining_matches, options);
        if (two_view_geometry.config == ConfigurationType::DEGENERATE) {
            break;
        }

        if (options.multiple_ignore_watermark) {
            if (two_view_geometry.config != ConfigurationType::WATERMARK) {
                two_view_geometries.push_back(two_view_geometry);
            }
        } else {
            two_view_geometries.push_back(two_view_geometry);
        }

        remaining_matches = ExtractOutlierMatches(remaining_matches, two_view_geometry.inlier_matches);
    }

    if (two_view_geometries.empty()) {
        config = ConfigurationType::DEGENERATE;
    } else if (two_view_geometries.size() == 1) {
        *this = two_view_geometries[0];
    } else {
        config = ConfigurationType::MULTIPLE;

        for (const auto& two_view_geometry : two_view_geometries) {
            inlier_matches.insert(inlier_matches.end(), two_view_geometry.inlier_matches.begin(),
                                  two_view_geometry.inlier_matches.end());
        }
    }
}

bool TwoViewGeometry::EstimateRelativePose(const Camera& camera1, const std::vector<Eigen::Vector2d>& points1,
                                           const Camera& camera2, const std::vector<Eigen::Vector2d>& points2,
                                           const Options& options, std::vector<double>* angles, bool verbose) {
    if (inlier_matches.size() < options.min_num_inliers_relative_pose) {
        relative_pose_valid = false;
        return false;
    }

    if (camera1.ModelName().compare("SPHERICAL") == 0 || camera2.ModelName().compare("SPHERICAL") == 0) {
        relative_pose_valid = EstimateRelativePoseSphericalCamera(camera1, points1, camera2, points2, angles, verbose);
        return relative_pose_valid;
    }

    // We need a valid epopolar geometry to estimate the relative pose.
    if (config != CALIBRATED && config != UNCALIBRATED && config != PLANAR && config != PANORAMIC &&
        config != PLANAR_OR_PANORAMIC) {
        relative_pose_valid = false;
        return false;
    }

    // Extract normalized inlier points.
    std::vector<Eigen::Vector2d> inlier_points1_normalized;
    inlier_points1_normalized.reserve(inlier_matches.size());
    std::vector<Eigen::Vector2d> inlier_points2_normalized;
    inlier_points2_normalized.reserve(inlier_matches.size());
    for (const auto& match : inlier_matches) {
        const point2D_t idx1 = match.point2D_idx1;
        const point2D_t idx2 = match.point2D_idx2;
        inlier_points1_normalized.push_back(camera1.ImageToWorld(points1[idx1]));
        inlier_points2_normalized.push_back(camera2.ImageToWorld(points2[idx2]));
    }

    Eigen::Matrix3d R;
    std::vector<Eigen::Vector3d> points3D;

    // if (config == UNCALIBRATED) {
    //     // x2Fx1.
    //     Eigen::Matrix3d K2;
    //     K2 << camera2.Params(0), 0, camera2.Params(1), 0, camera2.Params(0), camera2.Params(2), 0, 0, 1;
    //     Eigen::Matrix3d K1;
    //     K1 << camera1.Params(0), 0, camera1.Params(1), 0, camera1.Params(0), camera1.Params(2), 0, 0, 1;
    //     E = K2.transpose() * F * K1;
    // }

    double consistency_error = 0.0;
    std::vector<int> bin_counters;
    bin_counters.resize(19, 0);
    std::vector<double> bin_errors;
    bin_errors.resize(19, 0.0);

    const double angle_interval = DEG2RAD(10);

    if (config == CALIBRATED || config == UNCALIBRATED) {
        // Try to recover relative pose for calibrated and uncalibrated
        // configurations. In the uncalibrated case, this most likely leads to a
        // ill-defined reconstruction, but sometimes it succeeds anyways after
        // e.g. subsequent bundle-adjustment etc.

        PoseFromEssentialMatrix(E, inlier_points1_normalized, inlier_points2_normalized, &R, &tvec, &points3D);
        qvec = RotationMatrixToQuaternion(R);

        Eigen::Vector3d C = -(R.transpose() * tvec);
        if (C.norm() == 0) {
            relative_pose_valid = false;
            return false;
        }

        for (size_t E_idx = 0; E_idx < Es.size(); ++E_idx) {
            auto& sample_E = Es[E_idx];

            Eigen::Matrix3d R1;
            Eigen::Matrix3d R2;
            Eigen::Vector3d t;
            DecomposeEssentialMatrix(sample_E, &R1, &R2, &t);

            const std::array<Eigen::Matrix3d, 4> R_cmbs{{R1, R2, R1, R2}};
            const std::array<Eigen::Vector3d, 4> t_cmbs{{t, t, -t, -t}};

            // bool b_consistent = false;
            double min_R_diff = M_PI;
            double min_t_diff = M_PI;

            for (size_t idx = 0; idx < R_cmbs.size(); ++idx) {
                Eigen::Vector3d c_cmb = -(R_cmbs[idx].transpose() * t_cmbs[idx]);
                if (c_cmb.norm() == 0) {
                    continue;
                }

                Eigen::Matrix3d R_diff = (R.transpose()) * R_cmbs[idx];
                Eigen::AngleAxisd angle_axis(R_diff);
                double R_angle = angle_axis.angle();
                min_R_diff = std::min(R_angle, min_R_diff);

                double cos_t_angle = c_cmb.dot(C) / (c_cmb.norm() * C.norm());
                cos_t_angle = std::max(-1.0, std::min(1.0, cos_t_angle));
                double t_angle = std::acos(cos_t_angle);
                min_t_diff = std::min(t_angle, min_t_diff);

                // if (RadToDeg(R_angle) < 20 && RadToDeg(t_angle) < 20) {
                //     b_consistent = true;
                //     break;
                // }
            }
            double angle_diff = std::max(min_t_diff, min_R_diff);
            int bin_id = std::min((int)(angle_diff / angle_interval), 17);
            bin_counters[bin_id]++;
            bin_errors[bin_id] += angle_diff;

            // if (!b_consistent) {
            //     if (verbose) {
            //         std::cout << "Pose from best E and good E not consistent" << std::endl << std::endl;
            //     }
            //     relative_pose_valid = false;
            //     return false;
            // }
        }
    } else if (config == PLANAR || config == PANORAMIC || config == PLANAR_OR_PANORAMIC) {
        if (verbose) {
            std::cout << "Pose from H" << std::endl;
        }

        std::vector<Eigen::Matrix3d> R_Hs;
        std::vector<Eigen::Vector3d> t_Hs;
        Eigen::Vector3d n;
        MultiplePoseFromHomographyMatrix(H, camera1.CalibrationMatrix(), camera2.CalibrationMatrix(),
                                         inlier_points1_normalized, inlier_points2_normalized, &R, &tvec, &n, &points3D,
                                         R_Hs, t_Hs);

        qvec = RotationMatrixToQuaternion(R);
        Eigen::Vector3d C;
        C = -(R.transpose() * tvec);
        qvec = RotationMatrixToQuaternion(R);

        if (C.norm() == 0) {
            relative_pose_valid = false;
            return false;
        }
        
        for (size_t pose_idx = 0; pose_idx < R_Hs.size(); ++pose_idx) {
            Eigen::Vector3d c_Hs = -(R_Hs[pose_idx].transpose() * t_Hs[pose_idx]);
            if (c_Hs.norm() == 0) {
                // relative_pose_valid = false;
                // return false;
                continue;
            }

            Eigen::Matrix3d R_diff = (R.transpose()) * R_Hs[pose_idx];
            Eigen::AngleAxisd angle_axis(R_diff);
            double R_angle = angle_axis.angle();

            double cos_t_angle = c_Hs.dot(C) / (c_Hs.norm() * C.norm());
            cos_t_angle = std::max(-1.0, std::min(1.0, cos_t_angle));
            double t_angle = std::acos(cos_t_angle);
            // if (RadToDeg(R_angle) > 20 || RadToDeg(t_angle) > 20) {
            //     if (verbose) {
            //         std::cout << "Pose from H not consistent" << std::endl << std::endl;
            //     }
            //     relative_pose_valid = false;
            //     return false;
            // }
            // consistency_error += R_angle + t_angle;
            // num_Hs++;
            double angle_diff = std::max(t_angle, R_angle);
            int bin_id = std::min((int)(angle_diff / angle_interval), 17);
            bin_counters[bin_id]++;
            bin_errors[bin_id] += angle_diff;
        }
    } else {
        consistency_error = M_PI;
        relative_pose_valid = false;
        return false;
    }

    size_t num_checkes = 0;
    for (int i = 0; i < 18; ++i) {
        if (bin_counters[i] > 0) {
            consistency_error += bin_counters[i] * (bin_errors[i] / bin_counters[i]);
            num_checkes += bin_counters[i];
        }
    }
    if (num_checkes > 0) {
        consistency_error /= num_checkes;
    }

    confidence = 1.0 - std::min(M_PI, 2 * consistency_error) / M_PI;

    if (points3D.empty()) {
        tri_angle = 0;
    } else {
        std::vector<double> tri_angles =
            CalculateTriangulationAngles(Eigen::Vector3d::Zero(), -R.transpose() * tvec, points3D);
        tri_angle = Median(tri_angles);
        if (angles != nullptr) {
            *angles = tri_angles;
        }
    }

    if (config == PLANAR_OR_PANORAMIC) {
        if (tvec.norm() == 0) {
            config = PANORAMIC;
            tri_angle = 0;
        } else {
            config = PLANAR;
        }
    }
    relative_pose_valid = true;
    return true;
}

bool TwoViewGeometry::EstimateRelativePoseSphericalCamera(const Camera& camera1,
                                                          const std::vector<Eigen::Vector2d>& points1,
                                                          const Camera& camera2,
                                                          const std::vector<Eigen::Vector2d>& points2,
                                                          std::vector<double>* angles, bool verbose) {
    // We need a valid epopolar geometry to estimate the relative pose.
    if (config != CALIBRATED && config != UNCALIBRATED && config != PLANAR && config != PANORAMIC &&
        config != PLANAR_OR_PANORAMIC) {
        relative_pose_valid = false;
        return false;
    }

    // Extract normalized inlier points.
    std::vector<Eigen::Vector3d> inlier_points1_normalized;
    inlier_points1_normalized.reserve(inlier_matches.size());
    std::vector<Eigen::Vector3d> inlier_points2_normalized;
    inlier_points2_normalized.reserve(inlier_matches.size());
    for (const auto& match : inlier_matches) {
        const point2D_t idx1 = match.point2D_idx1;
        const point2D_t idx2 = match.point2D_idx2;
        inlier_points1_normalized.push_back(camera1.ImageToBearing(points1[idx1]));
        inlier_points2_normalized.push_back(camera2.ImageToBearing(points2[idx2]));
    }

    Eigen::Matrix3d R;
    std::vector<Eigen::Vector3d> points3D;

    if (config == CALIBRATED || config == UNCALIBRATED) {
        PoseFromEssentialMatrixSphericalCamera(E, inlier_points1_normalized, inlier_points2_normalized, &R, &tvec,
                                               &points3D);
        qvec = RotationMatrixToQuaternion(R);

        Eigen::Vector3d C = -(R.transpose() * tvec);
        if (C.norm() == 0) {
            relative_pose_valid = false;
            return false;
        }
        // for (size_t E_idx = 0; E_idx < Es.size(); ++E_idx) {
        //     auto& sample_E = Es[E_idx];

        //     Eigen::Matrix3d R1;
        //     Eigen::Matrix3d R2;
        //     Eigen::Vector3d t;
        //     DecomposeEssentialMatrix(sample_E, &R1, &R2, &t);

        //     const std::array<Eigen::Matrix3d, 4> R_cmbs{{R1, R2, R1, R2}};
        //     const std::array<Eigen::Vector3d, 4> t_cmbs{{t, t, -t, -t}};

        //     bool b_consistent = false;

        //     for (size_t idx = 0; idx < R_cmbs.size(); ++idx) {
        //         Eigen::Vector3d c_cmb = -(R_cmbs[idx].transpose() * t_cmbs[idx]);

        //         Eigen::Matrix3d R_diff = (R.transpose()) * R_cmbs[idx];
        //         Eigen::AngleAxisd angle_axis(R_diff);
        //         double R_angle = angle_axis.angle();

        //         if (c_cmb.norm() == 0) {
        //             continue;
        //         }

        //         double cos_t_angle = c_cmb.dot(C) / (c_cmb.norm() * C.norm());
        //         cos_t_angle = std::max(-1.0, std::min(1.0, cos_t_angle));

        //         double t_angle = std::acos(cos_t_angle);

        //         if (RadToDeg(R_angle) < 15 && RadToDeg(t_angle) < 15) {
        //             b_consistent = true;
        //             break;
        //         }
        //     }
        //     if (!b_consistent) {
        //         if (verbose) {
        //             std::cout << "Pose from best E and good E not consistent" << std::endl << std::endl;
        //             ;
        //         }
        //         relative_pose_valid = false;
        //         return false;
        //     }
        // }
    } else {
        relative_pose_valid = false;
        return false;
    }

    if (points3D.empty()) {
        tri_angle = 0;
    } else {
        std::vector<double> tri_angles =
            CalculateTriangulationAngles(Eigen::Vector3d::Zero(), -R.transpose() * tvec, points3D);
        tri_angle = Median(tri_angles);
        if (angles != nullptr) {
            *angles = tri_angles;
        }
    }

    if (config == PLANAR_OR_PANORAMIC) {
        if (tvec.norm() == 0) {
            config = PANORAMIC;
            tri_angle = 0;
        } else {
            config = PLANAR;
        }
    }
    relative_pose_valid = true;
    return true;
}

bool TwoViewGeometry::EstimateRelativePoseSphericalCameraRig(const Camera& camera1,
                                                          const std::vector<Eigen::Vector2d>& points1,
                                                          const Camera& camera2,
                                                          const std::vector<Eigen::Vector2d>& points2,
                                                          const std::vector<uint32_t>& local_camera_indices1,
                                                          const std::vector<uint32_t>& local_camera_indices2,
                                                          std::vector<double>* angles, bool verbose) {
    // We need a valid epopolar geometry to estimate the relative pose.
    if (config != CALIBRATED && config != UNCALIBRATED && config != PLANAR && config != PANORAMIC &&
        config != PLANAR_OR_PANORAMIC) {
        relative_pose_valid = false;
        return false;
    }

    // Extract normalized inlier points.
    std::vector<Eigen::Vector3d> inlier_points1_normalized;
    inlier_points1_normalized.reserve(inlier_matches.size());
    std::vector<Eigen::Vector3d> inlier_points2_normalized;
    inlier_points2_normalized.reserve(inlier_matches.size());
    for (const auto& match : inlier_matches) {
        const point2D_t idx1 = match.point2D_idx1;
        const point2D_t idx2 = match.point2D_idx2;

        Eigen::Vector3d point1_normalized = camera1.LocalImageToBearing(local_camera_indices1[idx1], points1[idx1]);
        Eigen::Vector3d point2_normalized = camera2.LocalImageToBearing(local_camera_indices2[idx2], points2[idx2]);

        if(local_camera_indices1[idx1] == 1){

            point1_normalized[0] = - point1_normalized[0];
            point1_normalized[2] = - point1_normalized[2];
        }
        if(local_camera_indices2[idx2] == 1){

            point2_normalized[0] = - point2_normalized[0];
            point2_normalized[2] = - point2_normalized[2];
        }

        inlier_points1_normalized.push_back(point1_normalized);
        inlier_points2_normalized.push_back(point2_normalized);
    }

    Eigen::Matrix3d R;
    std::vector<Eigen::Vector3d> points3D;

    if (config == CALIBRATED ) {
        PoseFromEssentialMatrixSphericalCamera(E, inlier_points1_normalized, inlier_points2_normalized, &R, &tvec,
                                               &points3D);
        qvec = RotationMatrixToQuaternion(R);

        Eigen::Vector3d C = -(R.transpose() * tvec);
        if (C.norm() == 0) {
            relative_pose_valid = false;
            return false;
        }
        // std::cout<<"Es count: "<<Es.size()<<std::endl;
        // for (size_t E_idx = 0; E_idx < Es.size(); ++E_idx) {
        //     auto& sample_E = Es[E_idx];

        //     Eigen::Matrix3d R1;
        //     Eigen::Matrix3d R2;
        //     Eigen::Vector3d t;
        //     DecomposeEssentialMatrix(sample_E, &R1, &R2, &t);

        //     const std::array<Eigen::Matrix3d, 4> R_cmbs{{R1, R2, R1, R2}};
        //     const std::array<Eigen::Vector3d, 4> t_cmbs{{t, t, -t, -t}};

        //     bool b_consistent = false;

        //     for (size_t idx = 0; idx < R_cmbs.size(); ++idx) {
        //         Eigen::Vector3d c_cmb = -(R_cmbs[idx].transpose() * t_cmbs[idx]);

        //         Eigen::Matrix3d R_diff = (R.transpose()) * R_cmbs[idx];
        //         Eigen::AngleAxisd angle_axis(R_diff);
        //         double R_angle = angle_axis.angle();

        //         if (c_cmb.norm() == 0) {
        //             continue;
        //         }

        //         double cos_t_angle = c_cmb.dot(C) / (c_cmb.norm() * C.norm());
        //         cos_t_angle = std::max(-1.0, std::min(1.0, cos_t_angle));

        //         double t_angle = std::acos(cos_t_angle);

        //         if (RadToDeg(R_angle) < 15 && RadToDeg(t_angle) < 15) {
        //             b_consistent = true;
        //             break;
        //         }
        //     }
        //     if (!b_consistent) {
        //         if (verbose) {
        //             std::cout << "Pose from best E and good E not consistent" << std::endl << std::endl;
        //             ;
        //         }
        //         relative_pose_valid = false;
        //         return false;
        //     }
        // }
    } else {
        relative_pose_valid = false;
        return false;
    }

    if (points3D.empty()) {
        tri_angle = 0;
    } else {
        std::vector<double> tri_angles =
            CalculateTriangulationAngles(Eigen::Vector3d::Zero(), -R.transpose() * tvec, points3D);
        tri_angle = Median(tri_angles);
        if (angles != nullptr) {
            *angles = tri_angles;
        }
    }

    if (config == PLANAR_OR_PANORAMIC) {
        if (tvec.norm() == 0) {
            config = PANORAMIC;
            tri_angle = 0;
        } else {
            config = PLANAR;
        }
    }
    qvec_rig = qvec;
    tvec_rig = tvec;

    relative_pose_valid = true;
    return true;
}



void TwoViewGeometry::EstimateSpherical(const Camera& camera1, const std::vector<Eigen::Vector2d>& points1,
                                        const Camera& camera2, const std::vector<Eigen::Vector2d>& points2,
                                        const FeatureMatches& matches, const Options& options) {
    options.Check();

    if (matches.size() < options.min_num_inliers) {
        config = ConfigurationType::DEGENERATE;
        return;
    }

    // Extract corresponding points.
    std::vector<Eigen::Vector3d> matched_points1_normalized(matches.size());
    std::vector<Eigen::Vector3d> matched_points2_normalized(matches.size());
    for (size_t i = 0; i < matches.size(); ++i) {
        const point2D_t idx1 = matches[i].point2D_idx1;
        const point2D_t idx2 = matches[i].point2D_idx2;
        matched_points1_normalized[i] = camera1.ImageToWorld(points1[idx1]).homogeneous();
        matched_points2_normalized[i] = camera2.ImageToWorld(points2[idx2]).homogeneous();
    }

    // Estimate epipolar models.

    auto E_ransac_options = options.ransac_options;
    E_ransac_options.max_error = options.ransac_options.max_error;

    LORANSAC<SphericalEssentialMatrixFivePointEstimator, SphericalEssentialMatrixFivePointEstimator> E_ransac(
        E_ransac_options);
    const auto E_report = E_ransac.EstimateMultiple(matched_points1_normalized, matched_points2_normalized);
    E = E_report.model;
    Es = E_report.multiple_models;

    const std::vector<char>* best_inlier_mask = nullptr;
    size_t num_inliers = 0;
    num_inliers = E_report.support.num_inliers;
    best_inlier_mask = &E_report.inlier_mask;

    config = ConfigurationType::CALIBRATED;

    if (best_inlier_mask != nullptr) {
        inlier_matches = ExtractInlierMatches(matches, num_inliers, *best_inlier_mask);
    }
}

void TwoViewGeometry::EstimateCalibrated(const Camera& camera1, const std::vector<Eigen::Vector2d>& points1,
                                         const Camera& camera2, const std::vector<Eigen::Vector2d>& points2,
                                         const FeatureMatches& matches, const Options& options) {
    options.Check();

    if (matches.size() < options.min_num_inliers) {
        config = ConfigurationType::DEGENERATE;
        return;
    }

    // Extract corresponding points.
    std::vector<Eigen::Vector2d> matched_points1(matches.size());
    std::vector<Eigen::Vector2d> matched_points2(matches.size());
    std::vector<Eigen::Vector2d> matched_points1_normalized(matches.size());
    std::vector<Eigen::Vector2d> matched_points2_normalized(matches.size());
    for (size_t i = 0; i < matches.size(); ++i) {
        const point2D_t idx1 = matches[i].point2D_idx1;
        const point2D_t idx2 = matches[i].point2D_idx2;
        matched_points1[i] = points1[idx1];
        matched_points2[i] = points2[idx2];
        matched_points1_normalized[i] = camera1.ImageToWorld(points1[idx1]);
        matched_points2_normalized[i] = camera2.ImageToWorld(points2[idx2]);
    }

    // Estimate epipolar models.

    auto E_ransac_options = options.ransac_options;
    E_ransac_options.max_error = (camera1.ImageToWorldThreshold(options.ransac_options.max_error) +
                                  camera2.ImageToWorldThreshold(options.ransac_options.max_error)) /
                                 2;

    LORANSAC<EssentialMatrixFivePointEstimator, EssentialMatrixFivePointEstimator> E_ransac(E_ransac_options);
    const auto E_report = E_ransac.EstimateMultiple(matched_points1_normalized, matched_points2_normalized);
    E = E_report.model;
    Es = E_report.multiple_models;

    LORANSAC<FundamentalMatrixSevenPointEstimator, FundamentalMatrixEightPointEstimator> F_ransac(
        options.ransac_options);
    const auto F_report = F_ransac.Estimate(matched_points1, matched_points2);
    F = F_report.model;

    // Estimate planar or panoramic model.

    LORANSAC<HomographyMatrixEstimator, HomographyMatrixEstimator> H_ransac(options.ransac_options);
    const auto H_report = H_ransac.Estimate(matched_points1, matched_points2);
    H = H_report.model;

    if ((!E_report.success && !F_report.success && !H_report.success) ||
        (E_report.support.num_inliers < options.min_num_inliers &&
         F_report.support.num_inliers < options.min_num_inliers &&
         H_report.support.num_inliers < options.min_num_inliers)) {
        config = ConfigurationType::DEGENERATE;
        return;
    }

    // Determine inlier ratios of different models.

    const double E_F_inlier_ratio = static_cast<double>(E_report.support.num_inliers) / F_report.support.num_inliers;
    const double H_F_inlier_ratio = static_cast<double>(H_report.support.num_inliers) / F_report.support.num_inliers;
    const double H_E_inlier_ratio = static_cast<double>(H_report.support.num_inliers) / E_report.support.num_inliers;
    
    // std::cout<<"E_F_ratio: "<<E_F_inlier_ratio<<std::endl;
    // std::cout<<"H_F_ratio: "<<H_F_inlier_ratio<<std::endl;
    // std::cout<<"H_E_ratio: "<<H_E_inlier_ratio<<std::endl;

    const std::vector<char>* best_inlier_mask = nullptr;
    size_t num_inliers = 0;

    if (E_report.success && E_F_inlier_ratio > options.min_E_F_inlier_ratio &&
        E_report.support.num_inliers >= options.min_num_inliers) {
        // Calibrated configuration.

        // Always use the model with maximum matches.
        if (E_report.support.num_inliers >= F_report.support.num_inliers) {
            num_inliers = E_report.support.num_inliers;
            best_inlier_mask = &E_report.inlier_mask;
        } else {
            num_inliers = F_report.support.num_inliers;
            best_inlier_mask = &F_report.inlier_mask;
        }

        if (H_E_inlier_ratio > options.max_H_inlier_ratio && !options.loose_constraint) {
            config = PLANAR_OR_PANORAMIC;
            if (H_report.support.num_inliers > num_inliers) {
                num_inliers = H_report.support.num_inliers;
                best_inlier_mask = &H_report.inlier_mask;
            }
        } else {
            config = ConfigurationType::CALIBRATED;
        }
    } else if (F_report.success && F_report.support.num_inliers >= options.min_num_inliers) {
        // Uncalibrated configuration.

        num_inliers = F_report.support.num_inliers;
        best_inlier_mask = &F_report.inlier_mask;

        if (H_F_inlier_ratio > options.max_H_inlier_ratio && !options.loose_constraint) {
            config = ConfigurationType::PLANAR_OR_PANORAMIC;
            if (H_report.support.num_inliers > num_inliers) {
                num_inliers = H_report.support.num_inliers;
                best_inlier_mask = &H_report.inlier_mask;
            }
        } else {
            config = ConfigurationType::UNCALIBRATED;
        }
    } else if (H_report.success && H_report.support.num_inliers >= options.min_num_inliers) {
        num_inliers = H_report.support.num_inliers;
        best_inlier_mask = &H_report.inlier_mask;
        config = ConfigurationType::PLANAR_OR_PANORAMIC;
    } else {
        config = ConfigurationType::DEGENERATE;
        return;
    }

    if (best_inlier_mask != nullptr) {
        inlier_matches = ExtractInlierMatches(matches, num_inliers, *best_inlier_mask);
        if (options.detect_watermark && DetectWatermark(camera1, matched_points1, camera2, matched_points2, num_inliers,
                                                        *best_inlier_mask, options)) {
            config = ConfigurationType::WATERMARK;
        }
    }
}

void TwoViewGeometry::EstimateUncalibrated(const Camera& camera1, const std::vector<Eigen::Vector2d>& points1,
                                           const Camera& camera2, const std::vector<Eigen::Vector2d>& points2,
                                           const FeatureMatches& matches, const Options& options) {
    options.Check();

    if (matches.size() < options.min_num_inliers) {
        config = ConfigurationType::DEGENERATE;
        return;
    }

    // Extract corresponding points.
    std::vector<Eigen::Vector2d> matched_points1(matches.size());
    std::vector<Eigen::Vector2d> matched_points2(matches.size());
    for (size_t i = 0; i < matches.size(); ++i) {
        matched_points1[i] = points1[matches[i].point2D_idx1];
        matched_points2[i] = points2[matches[i].point2D_idx2];
    }

    // Estimate epipolar model.

    LORANSAC<FundamentalMatrixSevenPointEstimator, FundamentalMatrixEightPointEstimator> F_ransac(
        options.ransac_options);
    const auto F_report = F_ransac.Estimate(matched_points1, matched_points2);
    F = F_report.model;

    // Estimate planar or panoramic model.

    LORANSAC<HomographyMatrixEstimator, HomographyMatrixEstimator> H_ransac(options.ransac_options);
    const auto H_report = H_ransac.Estimate(matched_points1, matched_points2);
    H = H_report.model;

    if ((!F_report.success && !H_report.success) || (F_report.support.num_inliers < options.min_num_inliers &&
                                                     H_report.support.num_inliers < options.min_num_inliers)) {
        config = ConfigurationType::DEGENERATE;
        return;
    }

    // Determine inlier ratios of different models.

    const double H_F_inlier_ratio = static_cast<double>(H_report.support.num_inliers) / F_report.support.num_inliers;

    if (H_F_inlier_ratio > options.max_H_inlier_ratio) {
        config = ConfigurationType::PLANAR_OR_PANORAMIC;
    } else {
        config = ConfigurationType::UNCALIBRATED;
    }

    inlier_matches = ExtractInlierMatches(matches, F_report.support.num_inliers, F_report.inlier_mask);

    if (options.detect_watermark && DetectWatermark(camera1, matched_points1, camera2, matched_points2,
                                                    F_report.support.num_inliers, F_report.inlier_mask, options)) {
        config = ConfigurationType::WATERMARK;
    }
}

bool TwoViewGeometry::DetectWatermark(const Camera& camera1, const std::vector<Eigen::Vector2d>& points1,
                                      const Camera& camera2, const std::vector<Eigen::Vector2d>& points2,
                                      const size_t num_inliers, const std::vector<char>& inlier_mask,
                                      const Options& options) {
    options.Check();

    // Check if inlier points in border region and extract inlier matches.

    const double diagonal1 = std::sqrt(camera1.Width() * camera1.Width() + camera1.Height() * camera1.Height());
    const double diagonal2 = std::sqrt(camera2.Width() * camera2.Width() + camera2.Height() * camera2.Height());
    const double minx1 = options.watermark_border_size * diagonal1;
    const double miny1 = minx1;
    const double maxx1 = camera1.Width() - minx1;
    const double maxy1 = camera1.Height() - miny1;
    const double minx2 = options.watermark_border_size * diagonal2;
    const double miny2 = minx2;
    const double maxx2 = camera2.Width() - minx2;
    const double maxy2 = camera2.Height() - miny2;

    std::vector<Eigen::Vector2d> inlier_points1(num_inliers);
    std::vector<Eigen::Vector2d> inlier_points2(num_inliers);

    size_t num_matches_in_border = 0;

    size_t j = 0;
    for (size_t i = 0; i < inlier_mask.size(); ++i) {
        if (inlier_mask[i]) {
            const auto& point1 = points1[i];
            const auto& point2 = points2[i];

            inlier_points1[j] = point1;
            inlier_points2[j] = point2;
            j += 1;

            if (!IsImagePointInBoundingBox(point1, minx1, maxx1, miny1, maxy1) &&
                !IsImagePointInBoundingBox(point2, minx2, maxx2, miny2, maxy2)) {
                num_matches_in_border += 1;
            }
        }
    }

    const double matches_in_border_ratio = static_cast<double>(num_matches_in_border) / num_inliers;

    if (matches_in_border_ratio < options.watermark_min_inlier_ratio) {
        return false;
    }

    // Check if matches follow a translational model.

    RANSACOptions ransac_options = options.ransac_options;
    ransac_options.min_inlier_ratio = options.watermark_min_inlier_ratio;

    LORANSAC<TranslationTransformEstimator<2>, TranslationTransformEstimator<2>> ransac(ransac_options);
    const auto report = ransac.Estimate(inlier_points1, inlier_points2);

    const double inlier_ratio = static_cast<double>(report.support.num_inliers) / num_inliers;

    return inlier_ratio >= options.watermark_min_inlier_ratio;
}

void TwoViewGeometry::EstimatePespectiveAndSpherical(const Camera& camera1, const std::vector<Eigen::Vector2d>& points1,
                                                    const Camera& camera2, const std::vector<Eigen::Vector2d>& points2,
                                                    const FeatureMatches& matches, const Options& options) {

    options.Check();

    if (matches.size() < options.min_num_inliers) {
        config = ConfigurationType::DEGENERATE;
        return;
    }

    // Extract corresponding points.
    std::vector<Eigen::Vector3d> matched_points1_normalized(matches.size());
    std::vector<Eigen::Vector3d> matched_points2_normalized(matches.size());
#if 1
    for (size_t i = 0; i < matches.size(); ++i) {
        const point2D_t idx1 = matches[i].point2D_idx1;
        const point2D_t idx2 = matches[i].point2D_idx2;
        matched_points1_normalized[i] = camera1.ImageToBearing(points1[idx1]);
        matched_points2_normalized[i] = camera2.ImageToBearing(points2[idx2]);
    }
#else
        for (size_t i = 0; i < matches.size(); ++i) {
        const point2D_t idx1 = matches[i].point2D_idx1;
        const point2D_t idx2 = matches[i].point2D_idx2;
        matched_points1_normalized[i] = camera1.ImageToWorld(points1[idx1]).homogeneous();
        matched_points2_normalized[i] = camera2.ImageToWorld(points2[idx2]).homogeneous();
    }

    if(camera1.ModelName() == "SPHERICAL"){
        for(auto &point : matched_points2_normalized){
            // const float f = camera2.FocalLength();
            const float f = camera2.MeanFocalLength();
            const float width = camera2.Width();
            const float height = camera2.Height();

            Eigen::Vector3d bearing;

            const double lon = (point[0] / width - 0.5) * (2 * M_PI);
            const double lat = -(point[1] / height - 0.5) * M_PI;
            // convert to equirectangular coordinates
            bearing(0) = std::cos(lat) * std::sin(lon);
            bearing(1) = -std::sin(lat);
            bearing(2) = std::cos(lat) * std::cos(lon);

            if(bearing(2)!=0){
                point[0] = bearing(0)/bearing(2);
                point[1] = bearing(1)/bearing(2);
            }
            else{
                point[0] = bearing(0);
                point[1] = bearing(1);
            }
        }
    } else {
        for(auto &point : matched_points1_normalized){
            // const float f = camera1.FocalLength();
            const float f = camera1.MeanFocalLength();
            const float width = camera1.Width();
            const float height = camera1.Height();

            Eigen::Vector3d bearing;

            const double lon = (point[0] / width - 0.5) * (2 * M_PI);
            const double lat = -(point[1] / height - 0.5) * M_PI;
            // convert to equirectangular coordinates
            bearing(0) = std::cos(lat) * std::sin(lon);
            bearing(1) = -std::sin(lat);
            bearing(2) = std::cos(lat) * std::cos(lon);

            if(bearing(2)!=0){
                point[0] = bearing(0)/bearing(2);
                point[1] = bearing(1)/bearing(2);
            }
            else{
                point[0] = bearing(0);
                point[1] = bearing(1);
            }
        }
    }
#endif

    // Estimate epipolar models.

    auto E_ransac_options = options.ransac_options;
    // E_ransac_options.max_error = options.max_angle_error;

    LORANSAC<SphericalEssentialMatrixFivePointEstimator, SphericalEssentialMatrixFivePointEstimator> E_ransac(
            E_ransac_options);
    const auto E_report = E_ransac.EstimateMultiple(matched_points1_normalized, matched_points2_normalized);
    E = E_report.model;
    Es = E_report.multiple_models;

    const std::vector<char>* best_inlier_mask = nullptr;
    size_t num_inliers = 0;
    num_inliers = E_report.support.num_inliers;
    best_inlier_mask = &E_report.inlier_mask;

    config = ConfigurationType::CALIBRATED;

    if (best_inlier_mask != nullptr) {
        inlier_matches = ExtractInlierMatches(matches, num_inliers, *best_inlier_mask);
    }

    std::cout<<"estimatemix :"<<matches.size()<<" "<<num_inliers<<std::endl;
}


void TwoViewGeometry::EstimateRig(const Camera& camera1, const std::vector<Eigen::Vector2d>& points1,
                                  const Camera& camera2, const std::vector<Eigen::Vector2d>& points2,
                                  const FeatureMatches& matches, const std::vector<uint32_t>& local_camera_indices1,
                                  const std::vector<uint32_t>& local_camera_indices2, const Options& options) {
    CHECK(camera1.HasPriorFocalLength() && camera2.HasPriorFocalLength())
        << "Two view geometry for camera rig requires prior intrinsic params";
    EstimateCalibratedRig(camera1, points1, camera2, points2, matches, local_camera_indices1, local_camera_indices2,
                          options);
}

void TwoViewGeometry::EstimateCalibratedRig(const Camera& camera1, const std::vector<Eigen::Vector2d>& points1,
                                            const Camera& camera2, const std::vector<Eigen::Vector2d>& points2,
                                            const FeatureMatches& matches,
                                            const std::vector<uint32_t>& local_camera_indices1,
                                            const std::vector<uint32_t>& local_camera_indices2,
                                            const Options& options) {

    // if this is actually a spherical camera
    if(camera1.NumLocalCameras() == 2 && camera2.NumLocalCameras() == 2/* && options.is_sphere*/) {
        std::vector<Eigen::Vector3d> matched_points1_normalized(matches.size());
        std::vector<Eigen::Vector3d> matched_points2_normalized(matches.size());
        for (size_t i = 0; i < matches.size(); ++i) {
            const point2D_t idx1 = matches[i].point2D_idx1;
            const point2D_t idx2 = matches[i].point2D_idx2;

            const uint32_t local_camera_id1 = local_camera_indices1[idx1];
            Eigen::Vector3d local_bearing1 = camera1.LocalImageToBearing(local_camera_id1, points1[idx1]);
            Eigen::Vector4d local_qvec1;
            Eigen::Vector3d local_tvec1;
            camera1.GetLocalCameraExtrinsic(local_camera_id1, local_qvec1, local_tvec1);
            Eigen::Vector4d local_qvec_t1 = InvertQuaternion(local_qvec1);
            Eigen::Vector3d bearing1 = QuaternionRotatePoint(local_qvec_t1, (local_bearing1 - local_tvec1));
            matched_points1_normalized[i] = bearing1;

            const uint32_t local_camera_id2 = local_camera_indices2[idx2];
            Eigen::Vector3d local_bearing2 = camera2.LocalImageToBearing(local_camera_id2, points2[idx2]);
            Eigen::Vector4d local_qvec2;
            Eigen::Vector3d local_tvec2;
            camera2.GetLocalCameraExtrinsic(local_camera_id2, local_qvec2, local_tvec2);
            Eigen::Vector4d local_qvec_t2 = InvertQuaternion(local_qvec2);
            Eigen::Vector3d bearing2 = QuaternionRotatePoint(local_qvec_t2, (local_bearing2 - local_tvec2));
            matched_points2_normalized[i] = bearing2;
            
            // Eigen::Vector2d point1_normalized = camera1.LocalImageToWorld(local_camera_indices1[idx1], points1[idx1]);
            // Eigen::Vector2d point2_normalized = camera2.LocalImageToWorld(local_camera_indices2[idx2], points2[idx2]);

            // if(local_camera_indices1[idx1] == 1){
            //     point1_normalized[1] = -point1_normalized[1];
            // }
            // if(local_camera_indices2[idx2] == 1){
            //     point2_normalized[1] = -point2_normalized[1];
            // }

            // matched_points1_normalized[i] = point1_normalized.homogeneous();
            // matched_points2_normalized[i] = point2_normalized.homogeneous();
        }

        // Estimate epipolar models.

        auto E_ransac_options = options.ransac_options;
        LORANSAC<SphericalEssentialMatrixFivePointEstimator, SphericalEssentialMatrixFivePointEstimator> E_ransac(
            E_ransac_options);
        const auto E_report = E_ransac.EstimateMultiple(matched_points1_normalized, matched_points2_normalized);
        E = E_report.model;
        Es = E_report.multiple_models;

        const std::vector<char>* best_inlier_mask = nullptr;
        size_t num_inliers = 0;
        num_inliers = E_report.support.num_inliers;
        best_inlier_mask = &E_report.inlier_mask;

        config = ConfigurationType::CALIBRATED;

        if (best_inlier_mask != nullptr) {
            inlier_matches = ExtractInlierMatches(matches, num_inliers, *best_inlier_mask);
        }
        return;
    }

    const size_t max_num_local_cameras = std::max(camera1.NumLocalCameras(), camera2.NumLocalCameras());

    std::vector<Eigen::Matrix3x4d> local_transforms;
    local_transforms.resize(camera1.NumLocalCameras());

    for (size_t i = 0; i < camera1.NumLocalCameras(); ++i) {
        Eigen::Vector4d local_qvec;
        Eigen::Vector3d local_tvec;
        camera1.GetLocalCameraExtrinsic(i, local_qvec, local_tvec);
        Eigen::Matrix3x4d local_transform = ComposeProjectionMatrix(local_qvec, local_tvec);
        local_transforms[i] = local_transform;
    }

    std::vector<GR6PEstimator::X_t> points1_normalized;
    std::vector<GR6PEstimator::Y_t> points2_normalized;
    std::unordered_map<int, float> local_matche_weights;
   
    for (size_t i = 0; i < matches.size(); ++i ) {
        const FeatureMatch& match = matches[i];
        const point2D_t idx1 = match.point2D_idx1;
        const point2D_t idx2 = match.point2D_idx2;
        const size_t local_camera_id1 = local_camera_indices1[idx1];
        const size_t local_camera_id2 = local_camera_indices2[idx2];

        Eigen::Vector2d point1_normalized = camera1.LocalImageToWorld(local_camera_id1, points1[idx1]);
        Eigen::Vector2d point2_normalized = camera2.LocalImageToWorld(local_camera_id2, points2[idx2]);

        points1_normalized.emplace_back();
        points1_normalized.back().rel_tform = local_transforms[local_camera_id1];
        points1_normalized.back().xy = point1_normalized;

        points2_normalized.emplace_back();
        points2_normalized.back().rel_tform = local_transforms[local_camera_id2];
        points2_normalized.back().xy = point2_normalized;

        const size_t hash_idx = local_camera_id1 * max_num_local_cameras + local_camera_id2;
        if (local_matche_weights.find(hash_idx) == local_matche_weights.end()) {
            local_matche_weights[hash_idx] = 1.0f;
        } else {
            local_matche_weights[hash_idx] += 1.0f;
        }
    }

    const size_t match_size = matches.size();
    for (auto & local_weight : local_matche_weights) {
        // local_weight.second = (local_weight.second * local_weight.second) / match_size / match_size;
        local_weight.second /= matches.size();
    }

    for (size_t i = 0; i < matches.size(); ++i ) {
        const FeatureMatch& match = matches[i];
        const point2D_t idx1 = match.point2D_idx1;
        const point2D_t idx2 = match.point2D_idx2;
        const size_t local_camera_id1 = local_camera_indices1[idx1];
        const size_t local_camera_id2 = local_camera_indices2[idx2];
        const double weight = local_matche_weights[local_camera_id1 * max_num_local_cameras + local_camera_id2];
        points1_normalized[i].weight = weight;
        points2_normalized[i].weight = weight;
    }

    RANSACOptions ransac_options = options.ransac_options;
    // ransac_options.max_num_trials = 2000;
    ransac_options.max_error = 5e-4;
    LORANSAC<GR6PEstimator, GR6PEstimator> ransac(ransac_options);
    
    const auto report = ransac.EstimateMultiple(points1_normalized, points2_normalized);

    if (!report.success) {
        inlier_matches.clear();
        config = ConfigurationType::DEGENERATE;
        std::cout<<"GR6P failed"<<std::endl;
        return;
    }

    size_t num_inliers = 0;
    num_inliers = report.support.num_inliers;
    std::vector<char> inlier_mask = report.inlier_mask;
    config = ConfigurationType::CALIBRATED;

    if(!inlier_mask.empty()) {
        inlier_matches = ExtractInlierMatches(matches, num_inliers, inlier_mask);
    }
    else{
        inlier_matches.clear();
    }
}

void TwoViewGeometry::EstimateOneAndRig(const Camera& camera1, 
                                        const std::vector<Eigen::Vector2d>& points1,
                                        const Camera& camera2, 
                                        const std::vector<Eigen::Vector2d>& points2,
                                        const FeatureMatches& matches,
                                        const std::vector<uint32_t>& local_camera_indices1,
                                        const std::vector<uint32_t>& local_camera_indices2, 
                                        const Options& options) {
    std::vector<Eigen::Vector2d> points1_normalized;
    std::vector<Eigen::Vector2d> points2_normalized;
    points1_normalized.resize(matches.size());
    points2_normalized.resize(matches.size());
    for (size_t i = 0; i < matches.size(); ++i) {
        const FeatureMatch& match = matches[i];
        const point2D_t idx1 = match.point2D_idx1;
        const point2D_t idx2 = match.point2D_idx2;
        if (camera1.NumLocalCameras() > 1) {
            Eigen::Vector2d point1 = camera1.LocalImageToWorld(local_camera_indices1[idx1], points1[idx1]);
            if(local_camera_indices1[idx1] == 1){
                point1[1] = -point1[1];
            }
            points1_normalized[i] = point1;
        } else {
            points1_normalized[i] = camera1.ImageToWorld(points1[idx1]);
        }
        if (camera2.NumLocalCameras() > 1) {
            Eigen::Vector2d point2 = camera2.LocalImageToWorld(local_camera_indices2[idx2], points2[idx2]);
            if(local_camera_indices2[idx2] == 1){
                point2[1] = -point2[1];
            }
            points2_normalized[i] = point2;
        } else {
            points2_normalized[i] = camera2.ImageToWorld(points2[idx2]);
        }
    }
    
    auto E_ransac_options = options.ransac_options;
    LORANSAC<EssentialMatrixFivePointEstimator, EssentialMatrixFivePointEstimator> E_ransac(
        E_ransac_options);
    const auto E_report = E_ransac.EstimateMultiple(points1_normalized, points2_normalized);
    E = E_report.model;
    Es = E_report.multiple_models;

    const std::vector<char>* best_inlier_mask = nullptr;
    size_t num_inliers = 0;
    num_inliers = E_report.support.num_inliers;
    best_inlier_mask = &E_report.inlier_mask;

    config = ConfigurationType::CALIBRATED;

    if (best_inlier_mask != nullptr) {
        inlier_matches = ExtractInlierMatches(matches, num_inliers, *best_inlier_mask);
    }
}


bool TwoViewGeometry::EstimateRelativePoseOneAndRig(const Camera& camera1, const std::vector<Eigen::Vector2d>& points1,
                                                    const Camera& camera2, const std::vector<Eigen::Vector2d>& points2,
                                                    const std::vector<uint32_t>& local_camera_indices1,
                                                    const std::vector<uint32_t>& local_camera_indices2, 
                                                    std::vector<double>* angles, bool verbose) {
    // We need a valid epopolar geometry to estimate the relative pose.
    if (config != CALIBRATED && config != UNCALIBRATED && config != PLANAR && config != PANORAMIC &&
        config != PLANAR_OR_PANORAMIC) {
        relative_pose_valid = false;
        return false;
    }

    // Extract normalized inlier points.
    std::vector<Eigen::Vector2d> inlier_points1_normalized;
    inlier_points1_normalized.reserve(inlier_matches.size());
    std::vector<Eigen::Vector2d> inlier_points2_normalized;
    inlier_points2_normalized.reserve(inlier_matches.size());
    for (const auto& match : inlier_matches) {
        const point2D_t idx1 = match.point2D_idx1;
        const point2D_t idx2 = match.point2D_idx2;

        Eigen::Vector2d point1_normalized;
        Eigen::Vector2d point2_normalized;
        if (camera1.NumLocalCameras() > 1) {
            Eigen::Vector2d point1 = camera1.LocalImageToWorld(local_camera_indices1[idx1], points1[idx1]);
            if(local_camera_indices1[idx1] == 1){
                point1[1] = -point1[1];
            }
            point1_normalized = point1;
        } else {
            point1_normalized = camera1.ImageToWorld(points1[idx1]);
        }
        if (camera2.NumLocalCameras() > 1) {
            Eigen::Vector2d point2 = camera2.LocalImageToWorld(local_camera_indices2[idx2], points2[idx2]);
            if(local_camera_indices2[idx2] == 1){
                point2[1] = -point2[1];
            }
            point2_normalized = point2;
        } else {
            point2_normalized = camera2.ImageToWorld(points2[idx2]);
        }

        inlier_points1_normalized.push_back(point1_normalized);
        inlier_points2_normalized.push_back(point2_normalized);
    }

    Eigen::Matrix3d R;
    std::vector<Eigen::Vector3d> points3D;

    if (config == CALIBRATED ) {
        PoseFromEssentialMatrix(E, inlier_points1_normalized, inlier_points2_normalized, &R, &tvec,
                                &points3D);
        qvec = RotationMatrixToQuaternion(R);

        Eigen::Vector3d C = -(R.transpose() * tvec);
        if (C.norm() == 0) {
            relative_pose_valid = false;
            return false;
        }
        // std::cout<<"Es count: "<<Es.size()<<std::endl;
        // for (size_t E_idx = 0; E_idx < Es.size(); ++E_idx) {
        //     auto& sample_E = Es[E_idx];

        //     Eigen::Matrix3d R1;
        //     Eigen::Matrix3d R2;
        //     Eigen::Vector3d t;
        //     DecomposeEssentialMatrix(sample_E, &R1, &R2, &t);

        //     const std::array<Eigen::Matrix3d, 4> R_cmbs{{R1, R2, R1, R2}};
        //     const std::array<Eigen::Vector3d, 4> t_cmbs{{t, t, -t, -t}};

        //     bool b_consistent = false;

        //     for (size_t idx = 0; idx < R_cmbs.size(); ++idx) {
        //         Eigen::Vector3d c_cmb = -(R_cmbs[idx].transpose() * t_cmbs[idx]);

        //         Eigen::Matrix3d R_diff = (R.transpose()) * R_cmbs[idx];
        //         Eigen::AngleAxisd angle_axis(R_diff);
        //         double R_angle = angle_axis.angle();

        //         if (c_cmb.norm() == 0) {
        //             continue;
        //         }

        //         double cos_t_angle = c_cmb.dot(C) / (c_cmb.norm() * C.norm());
        //         cos_t_angle = std::max(-1.0, std::min(1.0, cos_t_angle));

        //         double t_angle = std::acos(cos_t_angle);

        //         if (RadToDeg(R_angle) < 15 && RadToDeg(t_angle) < 15) {
        //             b_consistent = true;
        //             break;
        //         }
        //     }
        //     if (!b_consistent) {
        //         if (verbose) {
        //             std::cout << "Pose from best E and good E not consistent" << std::endl << std::endl;
        //             ;
        //         }
        //         relative_pose_valid = false;
        //         return false;
        //     }
        // }
    } else {
        relative_pose_valid = false;
        return false;
    }

    if (points3D.empty()) {
        tri_angle = 0;
    } else {
        std::vector<double> tri_angles =
            CalculateTriangulationAngles(Eigen::Vector3d::Zero(), -R.transpose() * tvec, points3D);
        tri_angle = Median(tri_angles);
        if (angles != nullptr) {
            *angles = tri_angles;
        }
    }

    if (config == PLANAR_OR_PANORAMIC) {
        if (tvec.norm() == 0) {
            config = PANORAMIC;
            tri_angle = 0;
        } else {
            config = PLANAR;
        }
    }
    qvec_rig = qvec;
    tvec_rig = tvec;

    relative_pose_valid = true;
    return true;
}

bool TwoViewGeometry::EstimateRelativePoseRigGV(const Camera& camera1, const std::vector<Eigen::Vector2d>& points1,
                                                const Camera& camera2, const std::vector<Eigen::Vector2d>& points2,
                                                const std::vector<uint32_t>& local_camera_indices1,
                                                const std::vector<uint32_t>& local_camera_indices2,
                                                const Options& options, std::vector<double>* angles, bool verbose) {
    // CHECK_EQ(camera1.NumLocalCameras(), camera2.NumLocalCameras());

    // this is actually a spherical camera
    if(camera1.NumLocalCameras() == 2 && camera2.NumLocalCameras() == 2/* && options.is_sphere*/) {
        bool success = EstimateRelativePoseSphericalCameraRig(camera1, points1,
                                                              camera2, points2,
                                                              local_camera_indices1,
                                                              local_camera_indices2,
                                                              angles, verbose);

        return success;
    }

    std::vector<Eigen::Matrix3x4d> local_transforms;
    local_transforms.resize(camera1.NumLocalCameras());

    for (size_t i = 0; i < camera1.NumLocalCameras(); ++i) {
        Eigen::Vector4d local_qvec;
        Eigen::Vector3d local_tvec;
        camera1.GetLocalCameraExtrinsic(i, local_qvec, local_tvec);
        Eigen::Matrix3x4d local_transform = ComposeProjectionMatrix(local_qvec, local_tvec);
        local_transforms[i] = local_transform;
    }

    const size_t max_num_local_cameras = std::max(camera1.NumLocalCameras(), camera2.NumLocalCameras());

    std::vector<GR6PEstimator::X_t> points1_normalized;
    std::vector<GR6PEstimator::Y_t> points2_normalized;
    std::unordered_map<int, float> local_matche_weights;

    for (size_t i = 0; i < inlier_matches.size(); ++i ) {
        const FeatureMatch& match = inlier_matches[i];
        const point2D_t idx1 = match.point2D_idx1;
        const point2D_t idx2 = match.point2D_idx2;
        const size_t local_camera_id1 = local_camera_indices1[idx1];
        const size_t local_camera_id2 = local_camera_indices2[idx2];

        Eigen::Vector2d point1_normalized = camera1.LocalImageToWorld(local_camera_id1, points1[idx1]);
        Eigen::Vector2d point2_normalized = camera2.LocalImageToWorld(local_camera_id2, points2[idx2]);

        points1_normalized.emplace_back();
        points1_normalized.back().rel_tform = local_transforms[local_camera_id1];
        points1_normalized.back().xy = point1_normalized;

        points2_normalized.emplace_back();
        points2_normalized.back().rel_tform = local_transforms[local_camera_id2];
        points2_normalized.back().xy = point2_normalized;

        const size_t hash_idx = local_camera_id1 * max_num_local_cameras + local_camera_id2;
        if (local_matche_weights.find(hash_idx) == local_matche_weights.end()) {
            local_matche_weights[hash_idx] = 1.0f;
        } else {
            local_matche_weights[hash_idx] += 1.0f;
        }
    }

    const size_t match_size = inlier_matches.size();
    for (auto & local_weight : local_matche_weights) {
        // local_weight.second = (local_weight.second * local_weight.second) / match_size / match_size;
        local_weight.second /= inlier_matches.size();
    }

    for (size_t i = 0; i < inlier_matches.size(); ++i ) {
        const FeatureMatch& match = inlier_matches[i];
        const point2D_t idx1 = match.point2D_idx1;
        const point2D_t idx2 = match.point2D_idx2;
        const size_t local_camera_id1 = local_camera_indices1[idx1];
        const size_t local_camera_id2 = local_camera_indices2[idx2];
        const double weight = local_matche_weights[local_camera_id1 * max_num_local_cameras + local_camera_id2];
        points1_normalized[i].weight = weight;
        points2_normalized[i].weight = weight;
    }

    RANSACOptions ransac_options = options.ransac_options;
    ransac_options.max_error = 5e-4;
    // ransac_options.max_num_trials = 2000;

    LORANSAC<GR6PEstimator, GR6PEstimator> ransac(ransac_options);
    auto report = ransac.EstimateMultiple(points1_normalized, points2_normalized);

    if (!report.success) {
        std::cout<<"GR6P failed"<<std::endl;
        return false;
    }

    Eigen::Matrix3x4d relative_pose = report.model;
    std::vector<Eigen::Matrix3x4d> relative_poses = report.multiple_models;

    Eigen::Matrix3d R_rig = report.model.block<3, 3>(0, 0);
    qvec_rig = RotationMatrixToQuaternion(report.model.block<3, 3>(0, 0));
    tvec_rig = report.model.col(3);
    qvec = qvec_rig;
    tvec = tvec_rig;

    Eigen::Vector3d C_rig = -(R_rig.transpose() * tvec_rig);
    if (C_rig.norm() == 0) {
        relative_pose_valid = false;
        std::cout << "relative pose invalid! " << std::endl;
        std::cout << R_rig << std::endl << tvec_rig.transpose() << std::endl;
        return false;
    }
    // std::cout << "relative pose count: " << relative_poses.size() << std::endl;
    // for (size_t pose_idx = 0; pose_idx < relative_poses.size(); ++pose_idx) {
    //     auto& sample_pose = relative_poses[pose_idx];

    //     Eigen::Matrix3d R_sample = sample_pose.block<3, 3>(0, 0);
    //     Eigen::Vector3d t_sample = sample_pose.col(3);

    //     Eigen::Vector3d C_sample = -(R_sample.transpose() * t_sample);

    //     Eigen::Matrix3d R_diff = R_rig.transpose() * R_sample;

    //     Eigen::AngleAxisd angle_axis(R_diff);
    //     double R_angle = angle_axis.angle();

    //     if (C_sample.norm() == 0) {
    //         relative_pose_valid = false;
    //         return false;
    //     }

    //     double cos_t_angle = C_sample.dot(C_rig) / (C_sample.norm() * C_rig.norm());
    //     cos_t_angle = std::max(-1.0, std::min(1.0, cos_t_angle));

    //     double t_angle = std::acos(cos_t_angle);

    //     if (RadToDeg(R_angle) > 20 || RadToDeg(t_angle) > 20) {
    //         if (verbose) {
    //             std::cout << "Pose from best E and good E not consistent" << std::endl << std::endl;
    //         }
    //         relative_pose_valid = false;
    //         return false;
    //     }
    // }

	angles->clear();

    CHECK_EQ(points1_normalized.size(), inlier_matches.size());
    CHECK_EQ(points1_normalized.size(), report.inlier_mask.size());

    size_t temp_idx = 0;
    for (auto match : inlier_matches) {
        if (!report.inlier_mask[temp_idx++]) {
            continue;
        }

        const point2D_t idx1 = match.point2D_idx1;
        const point2D_t idx2 = match.point2D_idx2;

        Eigen::Vector2d point1_N = camera1.LocalImageToWorld(local_camera_indices1[idx1], points1[idx1]);
        Eigen::Vector2d point2_N = camera2.LocalImageToWorld(local_camera_indices2[idx2], points2[idx2]);

        Eigen::Vector4d local_qvec1, local_qvec2;
        Eigen::Vector3d local_tvec1, local_tvec2;

        camera1.GetLocalCameraExtrinsic(local_camera_indices1[idx1], local_qvec1, local_tvec1);
        camera2.GetLocalCameraExtrinsic(local_camera_indices2[idx2], local_qvec2, local_tvec2);

        const Eigen::Matrix3x4d local_camera_proj_matrix1 =
            ComposeProjectionMatrix(QuaternionToRotationMatrix(local_qvec1), local_tvec1);

        const Eigen::Vector3d local_camera_proj_center1 =
            -QuaternionToRotationMatrix(local_qvec1).transpose() * local_tvec1;

        const Eigen::Matrix3d local_camera_R2 =
            QuaternionToRotationMatrix(local_qvec2) * QuaternionToRotationMatrix(qvec_rig);

        const Eigen::Vector3d local_camera_T2 = QuaternionToRotationMatrix(local_qvec2) * tvec_rig + local_tvec2;

        const Eigen::Matrix3x4d local_camera_proj_matrix2 = ComposeProjectionMatrix(local_camera_R2, local_camera_T2);

        const Eigen::Vector3d local_camera_proj_center2 = -local_camera_R2.transpose() * local_camera_T2;

        const Eigen::Vector3d& xyz =
            TriangulatePoint(local_camera_proj_matrix1, local_camera_proj_matrix2, point1_N, point2_N);

        const double angle = CalculateTriangulationAngle(local_camera_proj_center1, local_camera_proj_center2, xyz);

        double error1 = CalculateSquaredReprojectionErrorRig(points1[idx1],xyz,
                local_camera_proj_matrix1,local_camera_indices1[idx1],camera1);
        double error2 = CalculateSquaredReprojectionErrorRig(points2[idx2],xyz,
                local_camera_proj_matrix2,local_camera_indices2[idx2],camera2);    

        if(error1 < options.max_error * options.max_error &&
           error2 < options.max_error * options.max_error){
		    angles->push_back(angle);
        }
	}
	if(angles->size()>0){
		tri_angle = Median(*angles);
	}

    if (config == PLANAR_OR_PANORAMIC) {
        if (tvec.norm() == 0) {
            config = PANORAMIC;
            tri_angle = 0;
        } else {
            config = PLANAR;
        }
    }

    relative_pose_valid = true;

    return true;
}

}  // namespace sensemap
