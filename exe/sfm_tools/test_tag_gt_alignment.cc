// Copyright (c) 2019, SenseTime Group.
// All rights reserved.

#include <boost/filesystem/path.hpp>
#include <opencv2/opencv.hpp>

#include "base/camera.h"
#include "base/image.h"
#include "base/image_reader.h"
#include "util/misc.h"
#include "util/alignment.h"
#include "base/similarity_transform.h"

#include "estimators/absolute_pose.h"
#include "estimators/camera_alignment.h"
#include "estimators/mappoint_alignment.h"
#include "estimators/motion_average/rotation_average.h"
#include "estimators/motion_average/similarity_average.h"
#include "estimators/motion_average/translation_average.h"
#include "estimators/reconstruction_aligner.h"
#include "optim/ransac/loransac.h"

using namespace sensemap;

template <typename _Tp, int _rows, int _cols, int _options, int _maxRows, int _maxCols>
void eigen2cv(const Eigen::Matrix<_Tp, _rows, _cols, _options, _maxRows, _maxCols> &src, cv::Mat &dst) {
    if (!(src.Flags & Eigen::RowMajorBit)) {
        cv::Mat _src(src.cols(), src.rows(), cv::DataType<_Tp>::type, (void *)src.data(), src.stride() * sizeof(_Tp));
        cv::transpose(_src, dst);
    } else {
        cv::Mat _src(src.rows(), src.cols(), cv::DataType<_Tp>::type, (void *)src.data(), src.stride() * sizeof(_Tp));
        _src.copyTo(dst);
    }
}


void WritePLY(const std::vector<Eigen::Vector3d> &points_1,
              const std::string &path) {
    std::ofstream file(path, std::ios::trunc);
    CHECK(file.is_open()) << path;

    // Calculate pose number and mappoint number
    const int pose_num_1 = points_1.size();

    file << "ply" << std::endl;
    file << "format ascii 1.0" << std::endl;
    file << "element vertex " << pose_num_1 << std::endl;
    file << "property float x" << std::endl;
    file << "property float y" << std::endl;
    file << "property float z" << std::endl;
    file << "property uchar red" << std::endl;
    file << "property uchar green" << std::endl;
    file << "property uchar blue" << std::endl;
    file << "end_header" << std::endl;

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

    file.close();
}

void WriteLinesPLY(const std::string& path, std::vector<std::vector<Eigen::Vector3d>> lines_3d) {
    std::fstream text_file(path, std::ios::out);
    CHECK(text_file.is_open()) << path;

    text_file << "ply" << std::endl;
    text_file << "format binary_little_endian 1.0" << std::endl;
    text_file << "comment object: lines" << std::endl;
    text_file << "element vertex " << (lines_3d.size() * 2) << std::endl;

    text_file << "property double x" << std::endl;
    text_file << "property double y" << std::endl;
    text_file << "property double z" << std::endl;

    text_file << "property uchar red" << std::endl;
    text_file << "property uchar green" << std::endl;
    text_file << "property uchar blue" << std::endl;

    text_file << "element edge " << lines_3d.size() << std::endl;
    text_file << "property int vertex1" << std::endl;
    text_file << "property int vertex2" << std::endl;

    text_file << "end_header" << std::endl;
    text_file.close();

    std::fstream binary_file(path, std::ios::out | std::ios::binary | std::ios::app);
    CHECK(binary_file.is_open()) << path;

    for (const auto& line : lines_3d) {
        // int r_color = rng.uniform(0,255);
        // int g_color = rng.uniform(0,255);
        // int b_color = rng.uniform(0,255);

        WriteBinaryLittleEndian<double>(&binary_file, line[0][0]);
        WriteBinaryLittleEndian<double>(&binary_file, line[0][1]);
        WriteBinaryLittleEndian<double>(&binary_file, line[0][2]);

        WriteBinaryLittleEndian<uint8_t>(&binary_file, 255);
        WriteBinaryLittleEndian<uint8_t>(&binary_file, 255);
        WriteBinaryLittleEndian<uint8_t>(&binary_file, 255);

        WriteBinaryLittleEndian<double>(&binary_file, line[1][0]);
        WriteBinaryLittleEndian<double>(&binary_file, line[1][1]);
        WriteBinaryLittleEndian<double>(&binary_file, line[1][2]);

        WriteBinaryLittleEndian<uint8_t>(&binary_file, 255);
        WriteBinaryLittleEndian<uint8_t>(&binary_file, 255);
        WriteBinaryLittleEndian<uint8_t>(&binary_file, 255);
    }

    int i = 0;
    while (i < lines_3d.size()) {
        WriteBinaryLittleEndian<int>(&binary_file, i * 2);
        WriteBinaryLittleEndian<int>(&binary_file, i * 2 + 1);
        i++;
    }

    binary_file.close();
}

int main(int argc, char *argv[]) {
    std::string target_pose_path = std::string(argv[1]);
    std::string original_pose_path = std::string(argv[2]);
    std::string output_pose_path = std::string(argv[3]);
    double max_error = std::stod(argv[4]);

    EIGEN_STL_UMAP(std::string, Eigen::Vector3d) target_pose_map, original_pose_map;

    std::ifstream target_pose_file(target_pose_path);
    CHECK(target_pose_file.is_open()) << target_pose_path;

    std::string line;
    std::string item;

    while (std::getline(target_pose_file, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }

        std::stringstream line_stream(line);
        std::string pose_index;
        std::getline(line_stream, item, ' ');
        pose_index = item;

        double x, y, z;
        std::getline(line_stream, item, ' ');
        x = std::stold(item);
        std::getline(line_stream, item, ' ');
        y = std::stold(item);
        std::getline(line_stream, item, ' ');
        z = std::stold(item);
        
        std::cout << pose_index << " " << x << " " << y << " " << z << std::endl;
        target_pose_map[pose_index] = Eigen::Vector3d(x, y, z);
    }

    std::ifstream original_pose_file(original_pose_path);
    CHECK(original_pose_file.is_open()) << original_pose_path;

    while (std::getline(original_pose_file, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }

        std::stringstream line_stream(line);
        std::string pose_index;
        std::getline(line_stream, item, ' ');
        pose_index = item;

        auto split_elems = StringSplit(pose_index, "_");
        if (std::stoi(split_elems[1]) == 1) {
            pose_index = split_elems[0] + "_1";
        }
        if (std::stoi(split_elems[1]) == 2) {
            pose_index = split_elems[0] + "_4";
        }
        if (std::stoi(split_elems[1]) == 3) {
            pose_index = split_elems[0] + "_3";
        }
        if (std::stoi(split_elems[1]) == 4) {
            pose_index = split_elems[0] + "_2";
        }


        double x, y, z;
        std::getline(line_stream, item, ' ');
        x = std::stold(item);
        std::getline(line_stream, item, ' ');
        y = std::stold(item);
        std::getline(line_stream, item, ' ');
        z = std::stold(item);
        
        std::cout << pose_index << " " << x << " " << y << " " << z << std::endl;
        original_pose_map[pose_index] = Eigen::Vector3d(x, y, z);
    }



    std::vector<Eigen::Vector3d> target_pose, original_pose;
    std::vector<std::string> pose_name;
    for (auto cur_target_pose : target_pose_map) {
        if (original_pose_map.count(cur_target_pose.first)) {
            std::cout << "cur_target_pose.first = " << cur_target_pose.first << std::endl;
            std::cout << "cur_target_pose.second = " << cur_target_pose.second << std::endl;
            std::cout << "original_pose_map[cur_target_pose.first] = " << original_pose_map[cur_target_pose.first] << std::endl;
            target_pose.emplace_back(cur_target_pose.second);
            original_pose.emplace_back(original_pose_map[cur_target_pose.first]);
            pose_name.emplace_back(cur_target_pose.first);
        }
    }





    std::cout << "Find " << original_pose.size() << " pairs poses ..." << std::endl;
   
    RANSACOptions ransac_options;
    ransac_options.max_error = max_error;
    ransac_options.min_inlier_ratio = 0.2;
    LORANSAC<CameraAlignmentEstimator, CameraAlignmentEstimator> ransac(ransac_options);

    const auto report = ransac.Estimate(original_pose, target_pose);

    // Save Src and dst
    WritePLY(original_pose, "./original.ply");
    WritePLY(target_pose, "./target.ply");

    if (!report.success) {
        std::cout << "Estimate fail ..." << std::endl;
    }

    std::cout << "Inlier number = " << report.support.num_inliers << std::endl;


    SimilarityTransform3 tform12;
	tform12.Estimate(original_pose, target_pose);
    Eigen::Matrix3x4d transform_matrix;

    // transform_matrix = tform12.Matrix().topRows<3>();
    transform_matrix = report.model;


    std::cout << transform_matrix << std::endl;

    // Calculate average residual
    std::vector<Eigen::Vector3d> converted_sfm_poses;
    double average_inlier_error = 0;
    double average_error = 0;
    for (size_t i = 0; i < original_pose.size(); i++) {
        auto cur_sfm_pose = original_pose[i];
        auto cur_slam_pose = target_pose[i];

        // reproject points1 into reconstruction_dst using alignment12
        const Eigen::Vector3d xyz12 = transform_matrix * cur_sfm_pose.homogeneous();

        converted_sfm_poses.emplace_back(xyz12);

        double cur_error = (cur_slam_pose - xyz12).norm();
        std::cout <<"error " <<cur_error <<std::endl;
        if (report.inlier_mask[i]) {
            average_inlier_error += cur_error;
        }
        average_error += cur_error;
    }

    WritePLY(converted_sfm_poses, "./original_align.ply");

    double distance = 0;
    for (size_t i = 1; i < original_pose.size(); i++) {
        distance += (original_pose[0] - original_pose[1]).norm();
    }
    std::cout << "Distance = " << distance <<std::endl;
    std::cout <<"Error Percentage = " << average_error / distance <<std::endl;



    std::cout << "All point number = " << original_pose.size() << std::endl;

    std::cout << "Average inlier error = " << average_inlier_error / report.support.num_inliers << std::endl;
    std::cout << "Average error = " << average_error / original_pose.size() << std::endl;

    const Eigen::Transform<double, 3, Eigen::Affine> transform(transform_matrix);
    std::cout << "scale = " << transform.matrix().block<1, 3>(0, 0).norm() << std::endl;

    std::ofstream output_file(output_pose_path+"/out.txt", std::ios::trunc);
    CHECK(output_file.is_open()) << output_pose_path;

    int tmp_counter = 0;
    for (auto pose : original_pose) {
        pose = transform * pose;
        std::ostringstream line;
        line << pose_name[tmp_counter] << " ";
        line << pose[0] << " ";
        line << pose[1] << " ";
        line << pose[2] << " 0 0 0 1";
        
        tmp_counter++;
        std::string line_string = line.str();

        output_file << line_string << std::endl;
    }

    cv::Mat result_trans;
    eigen2cv(transform_matrix, result_trans);
    cv::FileStorage fsm(output_pose_path + "/trans.yaml", cv::FileStorage::WRITE);
    fsm << "transMatrix" << result_trans;
    fsm.release();
}