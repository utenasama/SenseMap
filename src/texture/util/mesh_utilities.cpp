//Copyright (c) 2019, SenseTime Group.
//All rights reserved.

#include "mesh_utilities.h"

namespace sensemap {
namespace texture {

std::tuple<float, float, float>
MeshUtilities::Project3DPointAndGetUVDepth(
        const Eigen::Vector3d X,
        const CameraTrajectory &camera, int camid) {
    auto &intrinsic = camera.parameters_[camid]->intrinsic_;
    std::pair<double, double> f = std::make_pair(intrinsic(0, 0),
                                                 intrinsic(1, 1));
    std::pair<double, double> p = std::make_pair(intrinsic(0, 2),
                                                 intrinsic(1, 2));
    Eigen::Vector4d Vt = camera.parameters_[camid]->extrinsic_ *
                         Eigen::Vector4d(X(0), X(1), X(2), 1);
    float u = float((Vt(0) * f.first) / Vt(2) + p.first);
    float v = float((Vt(1) * f.second) / Vt(2) + p.second);
    float z = float(Vt(2));
    return std::make_tuple(u, v, z);
}


std::tuple <std::vector<std::vector < int>>, std::vector <std::vector<int>>>

MeshUtilities::CreateVertexAndImageVisibility(
        const TriangleMesh &mesh,
        const std::vector <std::shared_ptr<cv::Mat>> &images_depth,
        const CameraTrajectory &camera,
        double maximum_allowable_depth,
        double depth_threshold_for_visiblity_check) {
    auto n_camera = camera.parameters_.size();
    auto n_vertex = mesh.vertices_.size();
    std::vector <std::vector<int>> visiblity_vertex_to_image;
    std::vector <std::vector<int>> visiblity_image_to_vertex;
    visiblity_vertex_to_image.resize(n_vertex);
    visiblity_image_to_vertex.resize(n_camera);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int c = 0; c < n_camera; c++) {
        int viscnt = 0;
        for (int vertex_id = 0; vertex_id < n_vertex; vertex_id++) {
            Eigen::Vector3d X = mesh.vertices_[vertex_id];
            float u, v, d;
            std::tie(u, v, d) =
                    MeshUtilities::Project3DPointAndGetUVDepth(X, camera, c);
            auto u_d = int(round(u)), v_d = int(round(v));
            if (d < 0.0 ||
                !ImageUtilities::TestImageBoundary(*images_depth[c], u_d, v_d))
                continue;
            float d_sensor = images_depth[c]->at<float>(v_d, u_d);
//            std::cout<<u_d<<" "<<v_d<<" "<<d<<" "<<d_sensor<<" "<<std::endl;
//            if (d_sensor > maximum_allowable_depth)
//                continue;
//            if (images_depth[c]->at<unsigned char>(v_d, u_d) == 255)
//                continue;
            if (std::fabs(d - d_sensor) < 0.5)
//                depth_threshold_for_visiblity_check)
            {
#ifdef _OPENMP
#pragma omp critical
#endif
                {
                    visiblity_vertex_to_image[vertex_id].push_back(c);
                    visiblity_image_to_vertex[c].push_back(vertex_id);
                    viscnt++;
                }
            }
        }
        printf("[cam %d] %.5f percents are visible\n",
               c, double(viscnt) / n_vertex * 100);
        fflush(stdout);
    }
    return std::make_tuple(visiblity_vertex_to_image,
                           visiblity_image_to_vertex);
}

std::vector <std::vector<int>>
MeshUtilities::CreateFaceNeighbors(const TriangleMesh &mesh) {
    int face_num = mesh.faces_.size();
    std::vector <std::vector<int>> face_neighbors_to_face(face_num);
    std::map <std::pair<int, int>, std::vector<int>> edge_face_map;

    int vertex_id_1, vertex_id_2;
    for (int i = 0; i < face_num; ++i) {
        auto &face = mesh.faces_[i];
        for (int j = 0; j < 3; ++j) {
            vertex_id_1 = face[j];
            vertex_id_2 = face[(j + 1) % 3];
            if (vertex_id_1 > vertex_id_2)
                std::swap(vertex_id_1, vertex_id_2);
            edge_face_map[std::make_pair(vertex_id_1, vertex_id_2)].push_back(
                    i);
        }
    }

    int face_id_1, face_id_2;
    for (auto &it : edge_face_map) {
        for (int i = 0; i < it.second.size(); ++i) {
            face_id_1 = it.second[i];
            for (int j = 0; j < it.second.size(); ++j) {
                face_id_2 = it.second[j];
                face_neighbors_to_face[face_id_1].push_back(face_id_2);
                face_neighbors_to_face[face_id_2].push_back(face_id_1);
            }
        }
    }
    return face_neighbors_to_face;
}

} // namespace sensemap
} // namespace texture