#include <iomanip>

#include "util/ceres_types.h"
#include "util/misc.h"
#include "base/cost_functions.h"
#include "base/pose.h"

#include "absolute_pose.h"

namespace sensemap {

namespace {
void PrintSolverSummary(const ceres::Solver::Summary &summary) {
    std::cout << std::right << std::setw(16) << "Residuals : ";
    std::cout << std::left << summary.num_residuals_reduced << std::endl;

    std::cout << std::right << std::setw(16) << "Parameters : ";
    std::cout << std::left << summary.num_effective_parameters_reduced << std::endl;

    std::cout << std::right << std::setw(16) << "Iterations : ";
    std::cout << std::left << summary.num_successful_steps + summary.num_unsuccessful_steps << std::endl;

    std::cout << std::right << std::setw(16) << "Time : ";
    std::cout << std::left << summary.total_time_in_seconds << " [s]" << std::endl;

    std::cout << std::right << std::setw(16) << "Initial cost : ";
    std::cout << std::right << std::setprecision(6) << std::sqrt(summary.initial_cost / summary.num_residuals_reduced)
              << std::endl;

    std::cout << std::right << std::setw(16) << "Final cost : ";
    std::cout << std::right << std::setprecision(6) << std::sqrt(summary.final_cost / summary.num_residuals_reduced)
              << std::endl;

    std::cout << std::right << std::setw(16) << "Termination : ";

    std::string termination = "";

    switch (summary.termination_type) {
        case ceres::CONVERGENCE:
            termination = "Convergence";
            break;
        case ceres::NO_CONVERGENCE:
            termination = "No convergence";
            break;
        case ceres::FAILURE:
            termination = "Failure";
            break;
        case ceres::USER_SUCCESS:
            termination = "User success";
            break;
        case ceres::USER_FAILURE:
            termination = "User failure";
            break;
        default:
            termination = "Unknown";
            break;
    }

    std::cout << std::right << termination << std::endl;
    std::cout << std::endl;
}

}

bool RefineAbsolutePose(const LidarAbsolutePoseRefinementOptions & options,
                        Reconstruction *reconstruction,
                        const image_t image_id,
                        const std::vector<lidar::OctoTree::Point> & points, 
                        VoxelMap * voxel_map, 
                        Eigen::Vector4d * qvec, 
                        Eigen::Vector3d * tvec) {
    ceres::LossFunction* loss_function = nullptr;

    class Image & image = reconstruction->Image(image_id);
    class Camera & camera = reconstruction->Camera(image.CameraId());

    Eigen::Matrix3x4d inv_proj_matrix;
    inv_proj_matrix.block<3, 3>(0, 0) = QuaternionToRotationMatrix(*qvec).transpose();
    inv_proj_matrix.block<3, 1>(0, 3) = -inv_proj_matrix.block<3, 3>(0, 0) * *tvec;

    Eigen::Matrix4d world2cam = Eigen::Matrix4d::Identity();
    world2cam.topRows(3) = image.ProjectionMatrix();
    Eigen::Matrix4d lidar2cam = Eigen::Matrix4d::Identity();
    lidar2cam.topRows(3) = reconstruction->lidar_to_cam_matrix;
    Eigen::Matrix4d world2lidar = lidar2cam.inverse() * world2cam;
    Eigen::Vector4d prior_qvec = RotationMatrixToQuaternion(world2lidar.block<3, 3>(0, 0));
    Eigen::Vector3d prior_tvec = world2lidar.block<3, 1>(0, 3);

    size_t num_lidar_point = 0;
    for (size_t i = 0; i < points.size(); ++i) {
        Eigen::Vector3d point(&points[i].x);
        Eigen::Vector3d query = inv_proj_matrix * point.homogeneous();

        lidar::OctoTree::Point p;
        p.x = query[0];
        p.y = query[1];
        p.z = query[2];

        lidar::OctoTree * loc = voxel_map->LocateOctree(p, 3);
        if (!loc) continue;

        Voxel * voxel = loc->voxel_;
        if (!voxel->IsDetermined() || voxel->IsScatter()) continue;
        num_lidar_point++;
    }

    size_t num_map_point = 0;
    for (point2D_t point2D_idx = 0; point2D_idx < image.NumPoints2D(); ++point2D_idx) {
        const Point2D& point2D = image.Point2D(point2D_idx);
        if (!point2D.HasMapPoint()) {
            continue;
        }
        num_map_point++;
    }

    double map_factor = (double)num_lidar_point / (double)(num_lidar_point + num_map_point);
    double lidar_factor = 1.0 - map_factor;
    std::cout << "lidar_factor: " << lidar_factor << std::endl;
    std::cout << "map_factor: " << map_factor << std::endl;
    std::cout << "num_map_point: " << num_map_point << std::endl;
    
    uint32_t m_num_visible_point = 0;
    const auto register_image_ids = reconstruction->RegisterImageIds();
    for (auto register_image_id : register_image_ids) {
        m_num_visible_point += reconstruction->Image(register_image_id).NumVisibleMapPoints();
    }
    m_num_visible_point /= register_image_ids.size();

    const size_t num_visible_point = image.NumVisibleMapPoints();
    const double factor = (double)num_visible_point / (double)m_num_visible_point;
    std::cout << "num visible point: " << num_visible_point << "/" <<  m_num_visible_point << std::endl;
    std::cout << "LiDAR/visual ratio: " << factor << std::endl;

    const std::vector<uint32_t> &local_image_indices = image.LocalImageIndices();
    std::vector<double *> local_qvec_data;
    std::vector<double *> local_tvec_data;
    std::vector<double *> local_camera_params_data;
    // This is a camera-rig
    int local_param_size = camera.LocalParams().size() / camera.NumLocalCameras();
    for (size_t i = 0; i < camera.NumLocalCameras(); ++i) {
        local_qvec_data.push_back(camera.LocalQvecsData() + 4 * i);
        local_tvec_data.push_back(camera.LocalTvecsData() + 3 * i);
        local_camera_params_data.push_back(camera.LocalIntrinsicParamsData() + i * local_param_size);
    }
    std::unordered_set<uint32_t> local_camera_in_the_image;

    for (int iter = 0; iter < options.num_iteration_pose_estimation; ++iter) {
        Eigen::Matrix3x4d inv_proj_matrix;
        inv_proj_matrix.block<3, 3>(0, 0) = QuaternionToRotationMatrix(*qvec).transpose();
        inv_proj_matrix.block<3, 1>(0, 3) = -inv_proj_matrix.block<3, 3>(0, 0) * *tvec;

        std::cout << "prior qvec: " << prior_qvec.transpose() << std::endl;
        std::cout << "qvec:       " << qvec->transpose() << std::endl;
        Eigen::Vector4d qvec_diff = ConcatenateQuaternions(InvertQuaternion(*qvec), prior_qvec);
        std::cout << "qvec_diff: " << qvec_diff.transpose() << std::endl;
        Eigen::AngleAxisd angle_axis(QuaternionToRotationMatrix(qvec_diff));
        double R_angle = angle_axis.angle();
        std::cout << "angle diff: " << RAD2DEG(R_angle) << std::endl;

        double dist = (prior_tvec - *tvec).norm();
        std::cout << "dist: " << dist << std::endl;

        ceres::Problem problem;
        ceres::CostFunction* cost_function = nullptr;

        size_t num_surf_observation = 0, num_corner_observation = 0;
        size_t num_feature = 0, num_feature_time = 0;
        for (size_t i = 0; i < points.size(); ++i) {
            Eigen::Vector3d point(&points[i].x);
            Eigen::Vector3d query = inv_proj_matrix * point.homogeneous();

            lidar::OctoTree::Point p;
            p.x = query[0];
            p.y = query[1];
            p.z = query[2];

            lidar::OctoTree * loc = voxel_map->LocateOctree(p, 3);
            if (!loc) continue;

            Voxel * voxel = loc->voxel_;
            if (voxel->IsScatter()) continue;

            if (!voxel->IsDetermined()) {
                voxel->ComputeFeature();
            }

            if (!voxel->IsDetermined()) continue;

            double weight = 1.0;
            if (voxel->IsFeature()) {
                num_feature++;
                if (loc->lifetime_ != 0) {
                    // int time_diff = std::abs((int)points[i].lifetime - (int)loc->lifetime_);
                    // if (time_diff > 200) weight = 20.0;
                    // else if (time_diff > 100) {
                    //     weight = 0.19 * time_diff - 18;
                    // }
                    num_feature_time++;
                }
            }

            // int time_diff = std::abs((int)points[i].lifetime - (int)loc->lastest_time_);
            // if (time_diff > 200) weight = 5.0;
            // else if (time_diff > 100) {
            //     weight = 0.04 * time_diff - 3;
            // }
            double time_diff = std::abs((double)points[i].lifetime - (double)loc->create_time_) / (double)1e9;
            // weight = 20.0 - 10.0 / (1 + 19.0 * std::exp(-0.2 * time_diff));
            weight = 40.0 - 20.0 / (1 + std::exp(-0.1 * time_diff));

            double exp_weight = 1.0;//1.0 - std::exp(-points[i].intensity * points[i].intensity / 30);
            double regu_weight = factor * weight * exp_weight;
            // double regu_weight = lidar_factor * weight * exp_weight;

            Eigen::Vector3d m_var = voxel->GetEx();
            Eigen::Matrix3d m_inv_cov = voxel->GetInvCov();
            Eigen::Vector3d m_pivot = voxel->GetPivot();

            cost_function = LidarAbsolutePoseCostFunction::Create(point, m_var, m_pivot, m_inv_cov, regu_weight);
            problem.AddResidualBlock(cost_function, loss_function, qvec->data(), tvec->data());
            num_surf_observation++;
        }

        std::cout << "num_observation: " << num_surf_observation << std::endl;
        // std::cout << "num_corner_observation: " << num_corner_observation << std::endl;
        std::cout << "num_feature: " << num_feature << std::endl;
        std::cout << "num_feature_time: " << num_feature_time << std::endl;

        if (problem.NumResiduals() > 0) {
            // Quaternion parameterization.
            ceres::LocalParameterization* quaternion_parameterization = new ceres::QuaternionParameterization;
    #if CERES_VERSION_MAJOR >= 2 && CERES_VERSION_MINOR >= 1
            problem.SetManifold(qvec->data(), quaternion_parameterization);
    #else
            problem.SetParameterization(qvec->data(), quaternion_parameterization);
    #endif
        }

        // add image observation
        if (reconstruction->NumRegisterLidarSweep() > 100) {
            double* qvec_data = image.Qvec().data();
            double* tvec_data = image.Tvec().data();

            std::unordered_set<mappoint_t> mappoint_ids;
            for (point2D_t point2D_idx = 0; point2D_idx < image.NumPoints2D(); ++point2D_idx) {
                const Point2D& point2D = image.Point2D(point2D_idx);
                if (!point2D.HasMapPoint()) {
                    continue;
                }
                
                uint32_t local_camera_id = local_image_indices[point2D_idx];

                class MapPoint & mappoint = reconstruction->MapPoint(point2D.MapPointId());
                mappoint_ids.insert(point2D.MapPointId());

                if (camera.NumLocalCameras() > 1) {
                    switch (camera.ModelId()) {
    #define CAMERA_MODEL_CASE(CameraModel)                                                      \
                        case CameraModel::kModelId:                                                             \
                        cost_function = RigBundleAdjustmentCostFunction<CameraModel>::Create(point2D.XY(), 1.0); \
                        break;
                        CAMERA_MODEL_SWITCH_CASES
    #undef CAMERA_MODEL_CASE
                    }
                    problem.AddResidualBlock(cost_function, loss_function, qvec_data, tvec_data,
                                                local_qvec_data[local_camera_id], local_tvec_data[local_camera_id],
                                                mappoint.XYZ().data(), local_camera_params_data[local_camera_id]);
                    local_camera_in_the_image.insert(local_camera_id);
                } else {
                    switch (camera.ModelId()) {
    #define CAMERA_MODEL_CASE(CameraModel)                                                           \
                        case CameraModel::kModelId:                                                                  \
                        cost_function = BundleAdjustmentCostFunction<CameraModel>::Create(point2D.XY(), 1.0); \
                        break;
                        CAMERA_MODEL_SWITCH_CASES
    #undef CAMERA_MODEL_CASE
                    }
                    problem.AddResidualBlock(cost_function, loss_function, qvec_data, tvec_data,
                                            mappoint.XYZ().data(), camera.ParamsData());
                }
            }

            Eigen::Vector4d lidar_to_cam_qvec = RotationMatrixToQuaternion(reconstruction->lidar_to_cam_matrix.block<3, 3>(0, 0));
            Eigen::Vector3d lidar_to_cam_tvec = reconstruction->lidar_to_cam_matrix.block<3, 1>(0, 3);
            // add image/lidar pair
            const double qvec_weight = 5 * m_num_visible_point;
            const double pos_weight = 100.0 * m_num_visible_point;
            cost_function = LidarCameraPoseCostFunction::Create(lidar_to_cam_qvec, lidar_to_cam_tvec, qvec_weight, pos_weight, pos_weight, pos_weight);
            problem.AddResidualBlock(cost_function, loss_function, qvec->data(), tvec->data(), qvec_data, tvec_data);

            // problem.SetParameterBlockConstant(qvec_data);
            // problem.SetParameterBlockConstant(tvec_data);

            // Quaternion parameterization.
            ceres::LocalParameterization* quaternion_parameterization = new ceres::QuaternionParameterization;
    #if CERES_VERSION_MAJOR >= 2 && CERES_VERSION_MINOR >= 1
            problem.SetManifold(qvec_data, quaternion_parameterization);
    #else
            problem.SetParameterization(qvec_data, quaternion_parameterization);
    #endif
            for (auto local_image_id : local_camera_in_the_image) {
                problem.SetParameterBlockConstant(local_qvec_data[local_image_id]);
                problem.SetParameterBlockConstant(local_tvec_data[local_image_id]);
                problem.SetParameterBlockConstant(local_camera_params_data[local_image_id]);
            }
            // for (auto mappoint_id : mappoint_ids) {
            //     class MapPoint & mappoint = reconstruction->MapPoint(mappoint_id);
            //     problem.SetParameterBlockConstant(mappoint.XYZ().data());
            // }
        }

        ceres::Solver::Options solver_options;
        solver_options.gradient_tolerance = options.gradient_tolerance;
        solver_options.max_num_iterations = options.max_num_iterations;
        solver_options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;

        // The overhead of creating threads is too large.
        solver_options.num_threads = 4;
    #if CERES_VERSION_MAJOR < 2
        solver_options.num_linear_solver_threads = 1;
    #endif  // CERES_VERSION_MAJOR

        ceres::Solver::Summary summary;
        ceres::Solve(solver_options, &problem, &summary);

        if (solver_options.minimizer_progress_to_stdout) {
            std::cout << std::endl;
        }

        if (options.print_summary) {
            PrintHeading2("Lidar Pose refinement report");
            PrintSolverSummary(summary);
        }

        double init_cost = std::sqrt(summary.initial_cost / summary.num_residuals_reduced);
        double final_cost = std::sqrt(summary.final_cost / summary.num_residuals_reduced);
        if (std::fabs(init_cost - final_cost) < options.error_tolerance) {
            break;
        }
        if (!summary.IsSolutionUsable()) {
            return false;
        }
    }
    return true;
}
}