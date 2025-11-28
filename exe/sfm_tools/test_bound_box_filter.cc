// Copyright (c) 2021, SenseTime Group.
// All rights reserved.

#include <boost/filesystem/path.hpp>
#include <opencv2/opencv.hpp>

#include "base/camera.h"
#include "base/image.h"
#include "base/image_reader.h"
#include "base/reconstruction_manager.h"
#include "base/common.h"
#include "util/misc.h"
#include "util/obj.h"
#include "util/ply.h"
#include "util/roi_box.h"

#include "../Configurator_yaml.h"

std::string configuration_file_path;

Eigen::Matrix3f ComputePovitMatrix(const std::vector<Eigen::Vector3f> &points){
    Eigen::Matrix3f pivot;
    Eigen::Vector3f centroid(Eigen::Vector3f::Zero());
    for (const auto &point : points) {
        centroid += point;
    }
    std::size_t point_num = points.size();
    centroid /= point_num;

    Eigen::Matrix3f M = Eigen::Matrix3f::Zero();
    for (const auto &point : points) {
        Eigen::Vector3f V = point - centroid;
        M += V * V.transpose();
    }

    Eigen::JacobiSVD<Eigen::Matrix3f> svd(M, Eigen::ComputeFullU | Eigen::ComputeFullV);
    pivot = svd.matrixU().transpose();
    return pivot;
}

using namespace sensemap;
int main(int argc, char* argv[]) {

    PrintHeading("SenseMap.  Copyright(c) 2019, SenseTime Group.");

    const std::string rec_path = std::string(argv[1]);
    CHECK(!rec_path.empty()) << "Usage: ./test_bound_box_filter ./rect_path (./box_file_path)";

    std::string bbox_path;
    if (argc == 3){
        bbox_path = std::string(argv[2]);
    }

    // const std::string rec_path = JoinPaths(workspace_path, "0");
    if (!(boost::filesystem::exists(JoinPaths(rec_path, "cameras.bin")) &&
        boost::filesystem::exists(JoinPaths(rec_path, "images.bin")) &&
        boost::filesystem::exists(JoinPaths(rec_path, "points3D.bin")))){
        std::cout << "reconstruction files empty" << std::endl;
        return 0;
    }
    Reconstruction reconstruction;
    reconstruction.ReadBinary(rec_path);

    Box box;
    if (!bbox_path.empty() && ExistsFile(bbox_path)){
        ReadBoundBoxText(bbox_path, box);
    }else {
        std::cout << "2 " << std::endl;
        const auto& image_ids = reconstruction.RegisterImageIds();
        std::vector<Eigen::Vector3f> image_poses;
        image_poses.reserve(image_ids.size());

        for (auto id : image_ids){
            const auto& image = reconstruction.Image(id);
            const auto& center = image.ProjectionCenter();

            Eigen::Vector3f image_pose((float)center.x(), (float)center.y(), 1.f);
            image_poses.push_back(image_pose);        
        }

        Eigen::Matrix3f pivot = ComputePovitMatrix(image_poses);

        box.rot = pivot;
        Eigen::Vector3f trans_pose_0 = pivot * image_poses.at(0);
        box.x_min = trans_pose_0.x();
        box.y_min = trans_pose_0.y();
        box.x_max = trans_pose_0.x();
        box.y_max = trans_pose_0.y();
        for (int i = 0; i < image_poses.size(); i++){
            Eigen::Vector3f trans_pose = pivot * image_poses.at(i);
            if (box.x_min > trans_pose.x()){
                box.x_min = trans_pose.x();
            }
            if (box.x_max < trans_pose.x()){
                box.x_max = trans_pose.x();
            }
            if (box.y_min > trans_pose.y()){
                box.y_min = trans_pose.y();
            }
            if (box.y_max < trans_pose.y()){
                box.y_max = trans_pose.y();
            }
        }
        
        // WriteBoundBoxText(JoinPaths(rec_path, ROI_BOX_NAME), box);
    }

    std::cout << "rot: \n" << box.rot << "\nx_min, x_max, y_min, y_max: " 
            << box.x_min << ", " << box.x_max << ", "
            << box.y_min << ", " << box.y_max
            << "\nGPS Bound Box Done" << std::endl;

    Eigen::Vector3f bb_min(box.x_min, box.y_min, box.z_min);
    Eigen::Vector3f bb_max(box.x_max, box.y_max, box.y_max);
    size_t num_filtered = reconstruction.FilterMapPointsWithBoundBox(bb_min, bb_max, box.rot);
    std::cout << "FilterMapPointsWithBoundBox: " << num_filtered << std::endl;

    std::string reconstruction_dir_filtered = rec_path+"/0_bb_filtered";
    if (!boost::filesystem::exists(reconstruction_dir_filtered)) {
        boost::filesystem::create_directories(reconstruction_dir_filtered);
    }

    reconstruction.WriteReconstruction(reconstruction_dir_filtered);
    std::cout << "Save Rectruction " << reconstruction_dir_filtered << std::endl;
    return 0;
}