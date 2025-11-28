// Copyright (c) 2019, SenseTime Group.
// All rights reserved.

#include "util/types.h"
#include "util/logging.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>

using namespace sensemap;
//using namespace cv;

int main(int argc, char* argv[]) {
    CHECK(argc == 5);
    std::string current_to_cad_trans_file = argv[1];
    std::string ref_to_cad_trans_file = argv[2];
    std::string ref_to_target_trans_file = argv[3];
    std::string current_to_target_trans_file = argv[4];

    Eigen::Matrix3x4d current_to_cad_trans;
    Eigen::Matrix3x4d ref_to_cad_trans;
    Eigen::Matrix3x4d ref_to_target_trans;
    Eigen::Matrix3x4d current_to_target_trans;

    cv::Mat saved_trans;

    cv::FileStorage fs(current_to_cad_trans_file, cv::FileStorage::READ);
    fs["transMatrix"] >> saved_trans;
    fs.release();
    cv::cv2eigen(saved_trans, current_to_cad_trans);

    fs.open(ref_to_cad_trans_file, cv::FileStorage::READ);
    fs["transMatrix"] >> saved_trans;
    fs.release();
    cv::cv2eigen(saved_trans, ref_to_cad_trans);

    fs.open(ref_to_target_trans_file, cv::FileStorage::READ);
    fs["transMatrix"] >> saved_trans;
    fs.release();
    cv::cv2eigen(saved_trans, ref_to_target_trans);

    Eigen::Matrix3d R = ref_to_cad_trans.block<3, 3>(0, 0);
    Eigen::Vector3d t = ref_to_cad_trans.block<3, 1>(0, 3);

    Eigen::Matrix3x4d cad_to_ref_trans;
    cad_to_ref_trans.block<3, 3>(0, 0) = R.transpose();
    cad_to_ref_trans.block<3, 1>(0, 3) = -R.transpose() * t;

    Eigen::Matrix4d current_to_cad_trans_homo = Eigen::Matrix4d::Identity();
    current_to_cad_trans_homo.block<3, 4>(0, 0) = current_to_cad_trans;

    Eigen::Matrix4d cad_to_ref_trans_homo = Eigen::Matrix4d::Identity();
    cad_to_ref_trans_homo.block<3, 4>(0, 0) = cad_to_ref_trans;

    Eigen::Matrix4d ref_to_target_trans_homo = Eigen::Matrix4d::Identity();
    ref_to_target_trans_homo.block<3, 4>(0, 0) = ref_to_target_trans;

    Eigen::Matrix4d current_to_target_trans_homo =
        ref_to_target_trans_homo * (cad_to_ref_trans_homo * current_to_cad_trans_homo);

    current_to_target_trans = current_to_target_trans_homo.block<3, 4>(0, 0);

    cv::Mat result_trans;
    cv::eigen2cv(current_to_target_trans, result_trans);

    cv::FileStorage result_fs(current_to_target_trans_file, cv::FileStorage::WRITE);
    result_fs << "transMatrix" << result_trans;
    result_fs.release();

    return 0;
}
