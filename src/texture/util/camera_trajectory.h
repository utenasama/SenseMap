//Copyright (c) 2019, SenseTime Group.
//All rights reserved.

#ifndef SENSEMAP_TEXTURE_UTIL_CAMERA_TRAJECTORY_H_
#define SENSEMAP_TEXTURE_UTIL_CAMERA_TRAJECTORY_H_

#include <memory>
#include <Eigen/Eigen>

namespace sensemap {
namespace texture {

    class CameraParameters {
    public:
        CameraParameters() {};

        CameraParameters(Eigen::Matrix3d &intrinsic,
                         Eigen::Matrix4d &extrinsic) :
                intrinsic_(std::move(intrinsic)),
                extrinsic_(std::move(extrinsic)) {};

        ~CameraParameters() {};

    public:
        Eigen::Matrix3d intrinsic_;
        Eigen::Matrix4d extrinsic_;
    };

    class CameraTrajectory {
    public:
        CameraTrajectory() {};

        ~CameraTrajectory() {};
    public:
        std::vector<std::shared_ptr<CameraParameters>> parameters_;
    };


    bool ReadIntrinsicMatrix(const std::string &filename,
                             Eigen::Matrix3d &intrinsic);

    bool ReadExtrinsicMatrix(const std::string &filename,
                             Eigen::Matrix4d &extrinsic,
                             bool camera_to_world = true);

    bool
    WriteExtrinsicToTxtFiles(const std::string &filename_without_suffix,
                             const CameraTrajectory &camera,
                             bool camera_to_world = true);

} // namespace sensemap
} //   namespace texture
#endif //SENSEMAP_TEXTURE_UTIL_CAMERA_TRAJECTORY_H_
