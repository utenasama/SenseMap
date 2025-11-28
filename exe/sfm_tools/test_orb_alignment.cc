// Copyright (c) 2019, SenseTime Group.
// All rights reserved.

#include <boost/filesystem/path.hpp>
#include <opencv2/opencv.hpp>

#include "base/camera.h"
#include "base/image.h"
#include "base/image_reader.h"
#include "base/reconstruction_manager.h"
#include "graph/correspondence_graph.h"
#include "util/misc.h"

#include "container/feature_data_container.h"
#include "feature/extraction.h"
#include "feature/feature_matcher.h"
#include "feature/utils.h"

#include "estimators/absolute_pose.h"
#include "estimators/camera_alignment.h"
#include "estimators/mappoint_alignment.h"
#include "estimators/motion_average/rotation_average.h"
#include "estimators/motion_average/similarity_average.h"
#include "estimators/motion_average/translation_average.h"
#include "estimators/reconstruction_aligner.h"
#include "optim/ransac/loransac.h"

using namespace sensemap;

std::string image_path;
std::string workspace_path;
std::string vocab_path;

void WritePLY(const std::vector<Eigen::Vector3d> &points_1, const std::vector<Eigen::Vector3d> &points_2,
              const std::string &path) {
    std::ofstream file(path, std::ios::trunc);
    CHECK(file.is_open()) << path;

    // Calculate pose number and mappoint number
    const int pose_num_1 = points_1.size();
    const int pose_num_2 = points_1.size();

    file << "ply" << std::endl;
    file << "format ascii 1.0" << std::endl;
    file << "element vertex " << pose_num_1 + pose_num_2 << std::endl;
    file << "property float x" << std::endl;
    file << "property float y" << std::endl;
    file << "property float z" << std::endl;
    file << "property uchar red" << std::endl;
    file << "property uchar green" << std::endl;
    file << "property uchar blue" << std::endl;
    file << "end_header" << std::endl;

    // Output mappoints_1
    for (const auto point : points_1) {
        std::ostringstream line;
        line << point[0] << " ";
        line << point[1] << " ";
        line << point[2] << " ";
        line << 214 << " ";
        line << 213 << " ";
        line << 183;
        file << line.str() << std::endl;
    }

    // Output mappoint_2
    for (const auto point : points_2) {
        std::ostringstream line;
        line << point[0] << " ";
        line << point[1] << " ";
        line << point[2] << " ";
        line << 244 << " ";
        line << 96 << " ";
        line << 108;
        file << line.str() << std::endl;
    }

    file.close();
}

void WritePLY(Reconstruction &reconstruction_1, Reconstruction &reconstruction_2, const std::string &path) {
    std::ofstream file(path, std::ios::trunc);
    CHECK(file.is_open()) << path;

    // Calculate pose number and mappoint number
    const int pose_num_1 = reconstruction_1.NumImages();
    // const int mappoint_num_1 = reconstruction_1.NumMapPoints();
    int mappoint_num_1 = 0;

    const int pose_num_2 = reconstruction_2.NumImages();
    // const int mappoint_num_2 = reconstruction_2.NumMapPoints();
    int mappoint_num_2 = 0;

    int counter = 0;
    for (const auto mappoint_id : reconstruction_1.MapPointIds()) {
        if (counter % 30 == 0) {
            mappoint_num_1++;
        }
        counter++;
    }

    for (const auto mappoint_id : reconstruction_2.MapPointIds()) {
        auto cur_mappoint = reconstruction_2.MapPoint(mappoint_id);
        if (std::abs(cur_mappoint.X()) < 100 && std::abs(cur_mappoint.Y()) < 100 && std::abs(cur_mappoint.Z()) < 100) {
            mappoint_num_2++;
        }
    }

    file << "ply" << std::endl;
    file << "format ascii 1.0" << std::endl;
    // file << "element vertex " << pose_num_1 + mappoint_num_1 + pose_num_2 + mappoint_num_2 << std::endl;
    file << "element vertex " << pose_num_1 + pose_num_2 << std::endl;
    file << "property float x" << std::endl;
    file << "property float y" << std::endl;
    file << "property float z" << std::endl;
    file << "property uchar red" << std::endl;
    file << "property uchar green" << std::endl;
    file << "property uchar blue" << std::endl;
    file << "end_header" << std::endl;

    // // Output mappoints_1
    // counter = 0;
    // for (const auto mappoint_id : reconstruction_1.MapPointIds()) {
    //     if (counter%30 == 0){
    //         auto cur_mappoint = reconstruction_1.MapPoint(mappoint_id);

    //         std::ostringstream line;
    //         line << cur_mappoint.X() << " ";
    //         line << cur_mappoint.Y() << " ";
    //         line << cur_mappoint.Z() << " ";
    //         line << 214 << " ";
    //         line << 213 << " ";
    //         line << 183;
    //         file << line.str() << std::endl;
    //     }
    //     counter++;
    // }

    // // Output mappoint_2
    //  for (const auto mappoint_id : reconstruction_2.MapPointIds()) {
    //     auto cur_mappoint = reconstruction_2.MapPoint(mappoint_id);
    //     if (std::abs(cur_mappoint.X()) < 100 && std::abs(cur_mappoint.Y()) < 100 && std::abs(cur_mappoint.Z()) <
    //     100){
    //         std::ostringstream line;
    //         line << cur_mappoint.X() << " ";
    //         line << cur_mappoint.Y() << " ";
    //         line << cur_mappoint.Z() << " ";
    //         line << 244 << " ";
    //         line << 96 << " ";
    //         line << 108;
    //         file << line.str() << std::endl;
    //     }
    // }

    // Output camera pose
    for (const auto keyframe_id : reconstruction_1.RegisterImageIds()) {
        auto cur_image = reconstruction_1.Image(keyframe_id);
        // Convert to -R^T t
        auto tvec = cur_image.Tvec();
        auto rot = cur_image.RotationMatrix();

        Eigen::Vector4d qvec;

        // Inverse
        tvec = -rot.transpose() * tvec;

        std::ostringstream line;
        // line << keyframe_pose.second.translation[0] << " ";
        // line << keyframe_pose.second.translation[1] << " ";
        // line << keyframe_pose.second.translation[2] << " ";

        line << tvec[0] << " ";
        line << tvec[1] << " ";
        line << tvec[2] << " ";
        line << 25 << " ";
        line << 202 << " ";
        line << 173;
        file << line.str() << std::endl;
    }

    for (const auto keyframe_id : reconstruction_2.RegisterImageIds()) {
        auto cur_image = reconstruction_2.Image(keyframe_id);
        // Convert to -R^T t
        auto tvec = cur_image.Tvec();
        auto rot = cur_image.RotationMatrix();

        Eigen::Vector4d qvec;

        // Inverse
        tvec = -rot.transpose() * tvec;

        std::ostringstream line;
        // line << keyframe_pose.second.translation[0] << " ";
        // line << keyframe_pose.second.translation[1] << " ";
        // line << keyframe_pose.second.translation[2] << " ";

        line << tvec[0] << " ";
        line << tvec[1] << " ";
        line << tvec[2] << " ";

        line << 190 << " ";
        line << 237 << " ";
        line << 199;

        file << line.str() << std::endl;
    }
    file.close();
}

int main(int argc, char *argv[]) {
    workspace_path = std::string(argv[1]);
    double max_error = std::stod(argv[2]);

    std::string sfm_reconstruction_path = workspace_path + "/slam+sfm";
    std::string slam_reconstruction_path = workspace_path + "/slam";
    std::string sift_reconstruction_path = workspace_path + "/sift_model/0";
    std::string orb_reconstruction_path = workspace_path + "/orb_model/0";

    std::shared_ptr<Reconstruction> sfm_reconstruction = std::make_shared<Reconstruction>();
    std::shared_ptr<Reconstruction> slam_reconstruction = std::make_shared<Reconstruction>();
    std::shared_ptr<Reconstruction> sift_reconstruction = std::make_shared<Reconstruction>();
    std::shared_ptr<Reconstruction> orb_reconstruction = std::make_shared<Reconstruction>();

    std::cout << "Load the Reconstruction ... " << std::endl;
    sfm_reconstruction->ReadReconstruction(sfm_reconstruction_path);
    slam_reconstruction->ReadReconstruction(slam_reconstruction_path);
    sift_reconstruction->ReadReconstruction(sift_reconstruction_path);
    orb_reconstruction->ReadReconstruction(orb_reconstruction_path);
    std::cout << "Finished image load ... " << std::endl;


    // for all the image in the slam reconstruction
    auto slam_image_ids = slam_reconstruction->RegisterImageIds();

    // Get all the image name from sfm reconstruction
    auto sfm_image_names = sfm_reconstruction->GetImageNames();

    std::vector<Eigen::Vector3d> sfm_poses, slam_poses;
    for (const auto slam_image_id : slam_image_ids) {
        auto slam_image_name = slam_reconstruction->Image(slam_image_id).Name();
        auto sfm_image_name = "mobile_images/" + slam_image_name + ".jpg";

        if (!sfm_image_names.count(sfm_image_name)) {
            continue;
        }

        auto sfm_image_id = sfm_image_names[sfm_image_name];

        auto sfm_image = sfm_reconstruction->Image(sfm_image_id);
        auto slam_image = slam_reconstruction->Image(slam_image_id);

        // Get pose
        Eigen::Vector3d sfm_pose = sfm_image.ProjectionCenter();
        Eigen::Vector3d slam_pose = slam_image.ProjectionCenter();

        sfm_poses.emplace_back(sfm_pose);
        slam_poses.emplace_back(slam_pose);
    }

    std::cout << "Find " << sfm_poses.size() << " pairs poses ..." << std::endl;
   
    RANSACOptions ransac_options;
    ransac_options.max_error = max_error;
    ransac_options.min_inlier_ratio = 0.2;
    LORANSAC<CameraAlignmentEstimator, CameraAlignmentEstimator> ransac(ransac_options);

    const auto report = ransac.Estimate(sfm_poses, slam_poses);

    if (!report.success) {
        std::cout << "Estimate fail ..." << std::endl;
    }

    std::cout << "Inlier number = " << report.support.num_inliers << std::endl;

    std::cout << report.model << std::endl;

    Eigen::Matrix3x4d result_model;
    // result_model << 0.959065, 0.00461365, -0.0884243,   10.7426,
    //                 0.0881718,  0.0384059,   0.958331,   2.063,
    //                 0.00811656,  -0.962367,  0.0378208,  -1.37861;
    result_model = report.model;

    // Calculate average residual
    std::vector<Eigen::Vector3d> converted_sfm_poses;
    double average_inlier_error = 0;
    double average_error = 0;
    for (size_t i = 0; i < sfm_poses.size(); i++) {
        auto cur_sfm_pose = sfm_poses[i];
        auto cur_slam_pose = slam_poses[i];

        // reproject points1 into reconstruction_dst using alignment12
        const Eigen::Vector3d xyz12 = result_model * cur_sfm_pose.homogeneous();

        converted_sfm_poses.emplace_back(xyz12);

        double cur_error = (cur_slam_pose - xyz12).squaredNorm();
        if (report.inlier_mask[i]) {
            average_inlier_error += cur_error;
        }
        average_error += cur_error;
    }

    std::cout << "All point number = " << sfm_poses.size() << std::endl;

    std::cout << "Average inlier error = " << average_inlier_error / report.support.num_inliers << std::endl;
    std::cout << "Average error = " << average_error / sfm_poses.size() << std::endl;

    orb_reconstruction->TransformReconstruction(report.model);
    sift_reconstruction->TransformReconstruction(report.model);

    // convert the sfm pose
    WritePLY(converted_sfm_poses, slam_poses, workspace_path + "/out_1.ply");

    WritePLY(*orb_reconstruction, *slam_reconstruction, workspace_path + "/out_2.ply");

    orb_reconstruction->WriteBinary(workspace_path + "/orb_model/1");
    sift_reconstruction->WriteBinary(workspace_path + "/sift_model/1");
}