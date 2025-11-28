//Copyright (c) 2019, SenseTime Group.
//All rights reserved.

#ifndef SENSEMAP_TEXTURE_UTIL_MESH_UTILITIES_H_
#define SENSEMAP_TEXTURE_UTIL_MESH_UTILITIES_H_

#include "../utils.h"

namespace sensemap {
namespace texture {

class MeshUtilities {
public:
    MeshUtilities() {};
public:
    static std::tuple<float, float, float> Project3DPointAndGetUVDepth(
            const Eigen::Vector3d X,
            const CameraTrajectory &camera, int camid);

    template<typename T>
    static std::tuple<bool, T> QueryImageIntensity(
            const cv::Mat &img, const Eigen::Vector3d &V,
            const CameraTrajectory &camera, int camid,
            int ch/*= -1*/, int image_boundary_margin/*= 10*/) {
        float u, v, depth;
        std::tie(u, v, depth) =
                MeshUtilities::Project3DPointAndGetUVDepth(V, camera, camid);
        if (ImageUtilities::TestImageBoundary(img, u, v,
                                              image_boundary_margin)) {
            int u_round = int(round(u));
            int v_round = int(round(v));
            if (ch == -1) {
                return std::make_tuple(true, img.at<T>(v_round, u_round));
            } else {
                return std::make_tuple(
                        true, img.at < cv::Vec < T,
                        3 >> (v_round, u_round)[ch]);
            }
        } else {
            return std::make_tuple(false, 0);
        }
    }

    static std::tuple <std::vector<std::vector < int>>,
    std::vector <std::vector<int>>>

    CreateVertexAndImageVisibility(
            const TriangleMesh &mesh,
            const std::vector <std::shared_ptr<cv::Mat>> &images_depth,
            const CameraTrajectory &camera,
            double maximum_allowable_depth,
            double depth_threshold_for_visiblity_check);

    static std::vector <std::vector<int>>
    CreateFaceNeighbors(const TriangleMesh &mesh);
};

} // namespace sensemap
} // namespace texture
#endif //SENSEMAP_TEXTURE_UTIL_MESH_UTILITIES_H_
