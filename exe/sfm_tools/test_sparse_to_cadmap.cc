// Copyright (c) 2019, SenseTime Group.
// All rights reserved.

#include <boost/filesystem/path.hpp>
#include <opencv2/opencv.hpp>

#include "base/common.h"
#include "base/reconstruction.h"
#include "util/misc.h"
#include "util/timer.h"

#include "estimators/plane_estimator.h"
#include "optim/ransac/loransac.h"
#include "base/version.h"

using namespace sensemap;

Eigen::Matrix3d ToRotationMatrix(const Eigen::Vector3d vec1, const Eigen::Vector3d vec2) {
    auto n_vec1 = vec1.normalized();
    auto n_vec2 = vec2.normalized();
    double cosin = n_vec1.dot(n_vec2);
    Eigen::Vector3d axis;
    if (cosin > 0) {
        axis = vec1.cross(vec2);
    } else {
        axis = vec1.cross(-vec2);
        cosin = -cosin;
    }

    double angle = std::acos(cosin);
    double sine = std::sin(angle);

    std::cout << "Angle(Y axis) diff: " << angle / M_PI * 180.0 << std::endl;

    Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
    Eigen::Matrix3d NNT;
    NNT(0, 0) = axis(0) * axis(0);
    NNT(0, 1) = axis(0) * axis(1);
    NNT(0, 2) = axis(0) * axis(2);
    NNT(1, 0) = axis(1) * axis(0);
    NNT(1, 1) = axis(1) * axis(1);
    NNT(1, 2) = axis(1) * axis(2);
    NNT(2, 0) = axis(2) * axis(0);
    NNT(2, 1) = axis(2) * axis(1);
    NNT(2, 2) = axis(2) * axis(2);

    Eigen::Matrix3d skew;
    skew << 0.0, -axis(2), axis(1), axis(2), 0.0, -axis(0), -axis(1), axis(0), 0.0;
    Eigen::Matrix3d rotation = I * cosin + NNT * (1 - cosin) + skew * sine;
    return rotation;
}

bool EstimateReconstructionPlane(std::shared_ptr<Reconstruction> reconstruction, Eigen::Vector4d& plane,
                                 double dist_ratio_point_to_plane, double& mean_sequential_neighbor_distance) {
    // Calculate Distance Between images
    size_t distance_count = 0;

    std::vector<image_t> image_ids = reconstruction->RegisterImageIds();

    for (const image_t image_id : image_ids) {
        auto image = reconstruction->Image(image_id);

        image_t image_neighbor_id = image_id + 1;
        if (reconstruction->ExistsImage(image_neighbor_id)) {
            Eigen::Vector3d baseline =
                image.ProjectionCenter() - reconstruction->Image(image_neighbor_id).ProjectionCenter();
            mean_sequential_neighbor_distance += baseline.norm();
            distance_count++;
        }
    }

    if (distance_count > 0) {
        mean_sequential_neighbor_distance /= distance_count;
    }

    std::cout << "Sequential neighbor distance count: " << distance_count << std::endl;
    std::cout << "Mean sequential neighbor distance: " << mean_sequential_neighbor_distance << std::endl;

    double mean_distance = mean_sequential_neighbor_distance;

    std::vector<Eigen::Vector3d> camera_poses;

    for (const image_t image_id : image_ids) {
        auto image = reconstruction->Image(image_id);
        camera_poses.emplace_back(image.ProjectionCenter());
    }

    // Estimate Plane Using Camera Pose
    RANSACOptions options;
    options.max_error = mean_distance * dist_ratio_point_to_plane;
    LORANSAC<PlaneEstimator, PlaneLocalEstimator> estimator(options);

    auto report = estimator.Estimate(camera_poses);
    if (!report.success) {
        return false;
    }

    plane = report.model;

    return true;
}

Eigen::Matrix3x4d ComputeProjectionMatrix(const std::vector<Eigen::Vector3d>& points, const int max_grid_res,
                                          const double boarder_size) {
    if (points.size() == 0 || max_grid_res <= 0) {
        return Eigen::Matrix3x4d::Identity();
    }

    Eigen::Vector4d bbox;
    bbox[0] = bbox[2] = points.at(0)[0];
    bbox[1] = bbox[3] = points.at(0)[2];
    for (auto point : points) {
        // std::cout << "point = " << point << std::endl;

        bbox[0] = std::min(point[0], bbox[0]);
        bbox[1] = std::min(point[2], bbox[1]);
        bbox[2] = std::max(point[0], bbox[2]);
        bbox[3] = std::max(point[2], bbox[3]);
    }

    bbox[0] -= boarder_size;
    bbox[1] -= boarder_size;
    bbox[2] += boarder_size;
    bbox[3] += boarder_size;

    // std::cout << "bbox = " << bbox << std::endl;

    int grid_res_x, grid_res_y;
    const double scale = (bbox[3] - bbox[1]) / (bbox[2] - bbox[0]);
    if (scale > 1) {
        grid_res_y = (max_grid_res - 1);
        grid_res_x = (max_grid_res - 1) / scale;
    } else {
        grid_res_y = (max_grid_res - 1) * scale;
        grid_res_x = (max_grid_res - 1);
    }

    int xpadding = 0, ypadding = 0;
    if (grid_res_x > grid_res_y) {
        ypadding = (grid_res_x - grid_res_y) / 2;
    } else {
        xpadding = (grid_res_y - grid_res_x) / 2;
    }

    const double gridX_size = (bbox[2] - bbox[0]) / grid_res_x;
    const double gridY_size = (bbox[3] - bbox[1]) / grid_res_y;

    const double gxy = 1.0 / std::max(gridX_size, gridY_size);

    Eigen::Matrix3x4d M;
    M(0, 0) = gxy;
    M(0, 1) = M(0, 2) = 0.0;
    M(0, 3) = -bbox[0] * gxy + xpadding;
    M(1, 1) = gxy;
    M(1, 0) = M(1, 2) = 0.0;
    M(1, 3) = 0.0;
    M(2, 2) = gxy;
    M(2, 0) = M(2, 1) = 0.0;
    M(2, 3) = -bbox[1] * gxy + ypadding;
    return M;
}

void ProjectCADMap(const std::string& output_path, Eigen::Matrix3x4d project_matrix,
                   const std::vector<Eigen::Vector3d>& poses, const std::vector<Eigen::Vector3d>& mappoints,
                   const std::vector<Eigen::Vector3ub>& mappoint_colors, const int point_size, const int cadmap_size) {
    std::string cad_map_path = output_path + "/sparse_points.jpg";
    std::string cad_map_with_pose_path = output_path + "/sparse_camera_track.jpg";

    cv::Mat cadmap = cv::Mat::zeros(cadmap_size, cadmap_size, CV_8UC3);
    for (int y = 0; y < cadmap_size; y++) {
        for (int x = 0; x < cadmap_size; x++) {
            for (int t = 0; t < 3; t++) cadmap.at<cv::Vec3b>(y, x)[t] = 255;
        }
    }

    for (int i = 0; i < mappoints.size(); i++) {
        Eigen::Vector3d xyz = mappoints[i];
        Eigen::Vector3ub rgb = mappoint_colors[i];

        Eigen::Vector3d proj = project_matrix * xyz.homogeneous();
        int c = proj[0];
        int r = proj[2];

        if (c < 0 || c >= cadmap_size || r < 0 || r >= cadmap_size) {
            continue;
        }

        int min_r = std::max(r - point_size, 0);
        int min_c = std::max(c - point_size, 0);
        int max_r = std::min(r + point_size, cadmap_size - 1);
        int max_c = std::min(c + point_size, cadmap_size - 1);
        for (int v = min_r; v <= max_r; ++v) {
            for (int u = min_c; u <= max_c; ++u) {
                cadmap.at<cv::Vec3b>(v, u)[0] = rgb[2];
                cadmap.at<cv::Vec3b>(v, u)[1] = rgb[1];
                cadmap.at<cv::Vec3b>(v, u)[2] = rgb[0];
            }
        }
    }

    for (int i = 0; i < poses.size(); i++) {
        Eigen::Vector3d xyz = poses[i];
        Eigen::Vector3ub rgb = Eigen::Vector3ub(255, 0, 0);
        int pose_point_size = std::max(1, cadmap_size / 2000);
        // std::cout << "pose_point_size = " << pose_point_size << std::endl;

        Eigen::Vector3d proj = project_matrix * xyz.homogeneous();
        int c = proj[0];
        int r = proj[2];

        if (c < 0 || c >= cadmap_size || r < 0 || r >= cadmap_size) {
            continue;
        }

        int min_r = std::max(r - pose_point_size, 0);
        int min_c = std::max(c - pose_point_size, 0);
        int max_r = std::min(r + pose_point_size, cadmap_size - 1);
        int max_c = std::min(c + pose_point_size, cadmap_size - 1);
        for (int v = min_r; v <= max_r; ++v) {
            for (int u = min_c; u <= max_c; ++u) {
                cadmap.at<cv::Vec3b>(v, u)[0] = rgb[2];
                cadmap.at<cv::Vec3b>(v, u)[1] = rgb[1];
                cadmap.at<cv::Vec3b>(v, u)[2] = rgb[0];
            }
        }
    }

    cv::imwrite(cad_map_path, cadmap);
}

int main(int argc, char* argv[]) {
    PrintHeading("SenseMap.  Copyright(c) 2019, SenseTime Group.");
    PrintHeading(std::string("Version: Sparse_To_CAD_map-") + __VERSION__);

    if (argc != 3) {
        std::cout << "Usage test_sparse_to_cadmap 1.reconstruction_path 2.output_path\n";
        exit(-1);
    }

    std::string reconstruction_path = argv[1];
    std::string output_path = argv[2];

    int cadmap_size = 8000;
    int point_size = 5;
    double dist_ratio_point_to_plane = 1.5;

    // Load Reconstruction
    Timer timer;
    std::shared_ptr<Reconstruction> reconstruction = std::make_shared<Reconstruction>();
    timer.Start();
    std::cout << "Model path : " << reconstruction_path << std::endl;
    reconstruction->ReadReconstruction(reconstruction_path);
    std::cout << std::endl;
    timer.PrintMinutes();
    std::cout << std::endl;

    Eigen::Vector4d plane;
    double mean_sequential_neighbor_distance = 0;
    if (!EstimateReconstructionPlane(reconstruction, plane, dist_ratio_point_to_plane,
                                     mean_sequential_neighbor_distance)) {
        std::cout << "ERROR: Estimate Plane Failed !" << std::endl;
        exit(-1);
    }

    std::cout << "Plane = " << plane << std::endl;

    Eigen::Vector3d target_plane = Eigen::Vector3d(0, 1, 0);
    Eigen::Vector3d normal_plane = Eigen::Vector3d(plane[0], plane[1], plane[2]);
    Eigen::Matrix3d rotm = ToRotationMatrix(target_plane, normal_plane);

    // Convert Point3d and Camera Pose Using Rotation Matrix
    std::vector<Eigen::Vector3d> poses;
    std::vector<Eigen::Vector3d> mappoints;
    std::vector<Eigen::Vector3ub> mappoint_colors;

    std::vector<image_t> image_ids = reconstruction->RegisterImageIds();
    for (const image_t image_id : image_ids) {
        auto image = reconstruction->Image(image_id);
        Eigen::Vector3d pose = rotm * image.ProjectionCenter();
        pose[0] = -pose[0];
        poses.emplace_back(pose);
    }

    std::unordered_set<mappoint_t> mappoint_ids = reconstruction->MapPointIds();
    std::cout << "Original Mappoint Number = " << mappoint_ids.size() << std::endl;
    int erase_counter = 0;
    for (const mappoint_t mappoint_id : mappoint_ids) {
        auto mappoint = reconstruction->MapPoint(mappoint_id);

        if (mappoint.Track().Length() < 5) {
            erase_counter++;
            continue;
        }

        if (mappoint.Error() > 2) {
            erase_counter++;
            continue;
        }

        Eigen::Vector3d xyz = mappoint.XYZ();
        double eval = normal_plane[0] * xyz[0] + normal_plane[1] * xyz[1] + normal_plane[2] * xyz[2];
        if (eval < 0) {
            erase_counter++;
            continue;
        }

        Eigen::Vector3d pos = rotm * mappoint.XYZ();
        pos[0] = -pos[0];
        mappoints.emplace_back(pos);
        mappoint_colors.emplace_back(mappoint.Color());
    }

    std::cout << "erase_counter = " << erase_counter << std::endl;

    std::cout << "mappoints.size() = " << mappoints.size() << std::endl;
    // Calculate BDBox and Project Matrix for Point Cloud
    Eigen::Matrix3x4d project_matrix =
        ComputeProjectionMatrix(poses, cadmap_size, mean_sequential_neighbor_distance * 10);

    // Project Mappoints Using Estimated Normal
    ProjectCADMap(output_path, project_matrix, poses, mappoints, mappoint_colors, point_size, cadmap_size);
}