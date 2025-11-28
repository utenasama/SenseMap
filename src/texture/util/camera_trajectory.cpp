//Copyright (c) 2019, SenseTime Group.
//All rights reserved.

#include "camera_trajectory.h"
#include <fstream>
#include <iostream>

namespace sensemap {
namespace texture {


bool ReadIntrinsicMatrix(const std::string &filename,
                         Eigen::Matrix3d &intrinsic) {
    FILE *file = fopen(filename.c_str(), "r");
    double input;
    fscanf(file, "%lf", &input);
    intrinsic(0, 0) = input;
    fscanf(file, "%lf", &input);
    intrinsic(0, 1) = input;
    fscanf(file, "%lf", &input);
    intrinsic(0, 2) = input;
    fscanf(file, "%lf", &input);
    intrinsic(1, 0) = input;
    fscanf(file, "%lf", &input);
    intrinsic(1, 1) = input;
    fscanf(file, "%lf", &input);
    intrinsic(1, 2) = input;
    fscanf(file, "%lf", &input);
    intrinsic(2, 0) = input;
    fscanf(file, "%lf", &input);
    intrinsic(2, 1) = input;
    fscanf(file, "%lf", &input);
    intrinsic(2, 2) = input;
    fclose(file);
    return true;
}


bool ReadExtrinsicMatrix(const std::string &filename, Eigen::Matrix4d &pose,
                         bool camera_to_world) {
    FILE *file = fopen(filename.c_str(), "r");
    double input;
    fscanf(file, "%lf", &input);
    pose(0, 0) = input;
    fscanf(file, "%lf", &input);
    pose(0, 1) = input;
    fscanf(file, "%lf", &input);
    pose(0, 2) = input;
    fscanf(file, "%lf", &input);
    pose(0, 3) = input;
    fscanf(file, "%lf", &input);
    pose(1, 0) = input;
    fscanf(file, "%lf", &input);
    pose(1, 1) = input;
    fscanf(file, "%lf", &input);
    pose(1, 2) = input;
    fscanf(file, "%lf", &input);
    pose(1, 3) = input;
    fscanf(file, "%lf", &input);
    pose(2, 0) = input;
    fscanf(file, "%lf", &input);
    pose(2, 1) = input;
    fscanf(file, "%lf", &input);
    pose(2, 2) = input;
    fscanf(file, "%lf", &input);
    pose(2, 3) = input;
    fscanf(file, "%lf", &input);
    pose(3, 0) = input;
    fscanf(file, "%lf", &input);
    pose(3, 1) = input;
    fscanf(file, "%lf", &input);
    pose(3, 2) = input;
    fscanf(file, "%lf", &input);
    pose(3, 3) = input;
    fclose(file);

    if (!camera_to_world)
        pose = pose.inverse().eval();

    return true;
}


bool WriteExtrinsicToTxtFiles(const std::string &filename_without_suffix,
                              const CameraTrajectory &camera,
                              bool camera_to_world) {
    for (int i = 0; i < camera.parameters_.size(); ++i) {
        FILE *txt_file = fopen((filename_without_suffix +
                                std::to_string(i) + ".txt").c_str(), "w");
        Eigen::Matrix4d extrinsic;
        if (camera_to_world)
            extrinsic = camera.parameters_[i]->extrinsic_;
        else
            extrinsic = camera.parameters_[i]->extrinsic_.inverse();
        fprintf(txt_file, "%lf %lf %lf %lf\n",
                extrinsic(0, 0), extrinsic(0, 1), extrinsic(0, 2),
                extrinsic(0, 3));
        fprintf(txt_file, "%lf %lf %lf %lf\n",
                extrinsic(1, 0), extrinsic(1, 1), extrinsic(1, 2),
                extrinsic(1, 3));
        fprintf(txt_file, "%lf %lf %lf %lf\n",
                extrinsic(2, 0), extrinsic(2, 1), extrinsic(2, 2),
                extrinsic(2, 3));
        fprintf(txt_file, "%lf %lf %lf %lf\n",
                extrinsic(3, 0), extrinsic(3, 1), extrinsic(3, 2),
                extrinsic(3, 3));
        fclose(txt_file);
    }
    return false;
}

} // namespace sensemap
} // namespace texture

