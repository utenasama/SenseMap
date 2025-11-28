//Copyright (c) 2019, SenseTime Group.
//All rights reserved.

#include <fstream>
#include "util/misc.h"
#include "COLMAP_helper.h"

namespace sensemap {
namespace texture {

void ReadDepthFromCOLMAP(const std::string &path, cv::Mat &depth_map) {
    std::fstream text_file(path, std::ios::in | std::ios::binary);

    char unused_char;
    size_t width, height, depth;
    text_file >> width >> unused_char >> height >> unused_char >> depth >>
              unused_char;
    std::streampos pos = text_file.tellg();
    text_file.close();

    std::vector<float> data;
    data.resize(width * height * depth);

    std::fstream binary_file(path, std::ios::in | std::ios::binary);
    binary_file.seekg(pos);
    ReadBinaryLittleEndian<float>(&binary_file, &data);

//    for(int i = 0; i < data.size(); ++i)
//    {
//        std::cout<<data[i]<<"  ";
//        if(i % 100 == 0)
//            std::cout<<std::endl;
//    }
    depth_map = cv::Mat(static_cast<int>(height), static_cast<int>(width),
                        CV_32FC1);

    float max = 0;
    for (int y = 0; y < depth_map.rows; ++y) {
        for (int x = 0; x < depth_map.cols; ++x) {
            depth_map.at<float>(y, x) = data[y * depth_map.cols + x];
            if (max < depth_map.at<float>(y, x))
                max = depth_map.at<float>(y, x);
        }
    }
    // std::cout << max << std::endl;
    binary_file.close();
}

bool ReadIntrinsicMatrixFromCOLMAPBinary(const std::string &filename,
                                         std::map<uint32_t, Eigen::Matrix3d> &intrinsic_map) {

    std::ifstream file(filename, std::ios::binary);

    int num_cameras = static_cast<int>(ReadBinaryLittleEndian<uint64_t>(&file));

    for (size_t i = 0; i < num_cameras; ++i) {
        Eigen::Matrix3d intrinsic;
        std::vector<double> params;
        auto camera_id = ReadBinaryLittleEndian<uint32_t>(&file);
        ReadBinaryLittleEndian<int>(&file);
        auto x = ReadBinaryLittleEndian<uint64_t>(&file);
        auto y = ReadBinaryLittleEndian<uint64_t>(&file);
        params.resize(4);
        ReadBinaryLittleEndian<double>(&file, &params);

        intrinsic(0, 0) = params[0];
        intrinsic(0, 1) = 0;
        intrinsic(0, 2) = params[2];
        intrinsic(1, 0) = 0;
        intrinsic(1, 1) = params[1];
        intrinsic(1, 2) = params[3];
        intrinsic(2, 0) = 0;
        intrinsic(2, 1) = 0;
        intrinsic(2, 2) = 1;

        std::cout << camera_id << " " << x << " " << y << std::endl;
        std::cout << intrinsic << std::endl;

        intrinsic_map.emplace(camera_id, intrinsic);
    }
    file.close();
    return true;
}

bool ReadExtrinsicMatrixFromCOLMAPBinary(const std::string &filename,
                                         std::vector<Eigen::Matrix4d> &poses,
                                         std::vector<std::string> &image_names,
                                         std::vector<uint32_t> &camera_ids) {
    poses.clear();
    image_names.clear();
    camera_ids.clear();

    std::ifstream file(filename, std::ios::binary);
    const size_t num_reg_images = ReadBinaryLittleEndian<uint64_t>(&file);
    for (size_t i = 0; i < num_reg_images; ++i) {
        std::cout << StringPrintf("\rLoad Scene Structure...(%d%)", 
                                   i * 100 / (num_reg_images - 1));
        uint64_t image_id = ReadBinaryLittleEndian<uint32_t>(&file);

        Eigen::Matrix4d pose;
        Eigen::Vector4d qvec;
        qvec(0) = ReadBinaryLittleEndian<double>(&file);
        qvec(1) = ReadBinaryLittleEndian<double>(&file);
        qvec(2) = ReadBinaryLittleEndian<double>(&file);
        qvec(3) = ReadBinaryLittleEndian<double>(&file);
        const double norm = qvec.norm();
        if (norm == 0) {
// We do not just use (1, 0, 0, 0) because that is a constant and when used
// for automatic differentiation that would lead to a zero derivative.
            qvec = Eigen::Vector4d(1.0, qvec(1), qvec(2), qvec(3));
        } else {
            qvec /= norm;
        }
        const Eigen::Quaterniond quat(qvec(0), qvec(1), qvec(2), qvec(3));
        pose.block<3, 3>(0, 0) = quat.toRotationMatrix();


        pose(0, 3) = ReadBinaryLittleEndian<double>(&file);
        pose(1, 3) = ReadBinaryLittleEndian<double>(&file);
        pose(2, 3) = ReadBinaryLittleEndian<double>(&file);

        pose(3, 0) = 0;
        pose(3, 1) = 0;
        pose(3, 2) = 0;
        pose(3, 3) = 1;

        poses.push_back(pose);

        auto camera_id = ReadBinaryLittleEndian<uint32_t>(&file);
        camera_ids.push_back(camera_id);
        //std::cout<<camera_id<<std::endl;
        char name_char[20];
        int n = 0;
        do {
            file.read(&name_char[n], 1);
        } while (name_char[n++] != '\0');

        std::string name = name_char;
        // std::cout << name << std::endl;
        const size_t num_points2D = ReadBinaryLittleEndian<uint64_t>(&file);
        image_names.push_back(name);

        std::vector<Eigen::Vector2d> points2D;
        points2D.reserve(num_points2D);
        std::vector<uint64_t> point3D_ids;
        point3D_ids.reserve(num_points2D);
        for (size_t j = 0; j < num_points2D; ++j) {
            const double x = ReadBinaryLittleEndian<double>(&file);
            const double y = ReadBinaryLittleEndian<double>(&file);
            points2D.emplace_back(x, y);
            point3D_ids.push_back(ReadBinaryLittleEndian<uint64_t>(&file));
        }

    }
    file.close();
    std::cout << std::endl;
    return true;
}

} // namespace sensemap
} // namespace texture