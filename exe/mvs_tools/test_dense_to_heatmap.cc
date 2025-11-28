//Copyright (c) 2020, SenseTime Group.
//All rights reserved.

#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_map>
#include <Eigen/Dense>
#include <boost/filesystem/path.hpp>
#include <opencv2/opencv.hpp>

#include "util/misc.h"
#include "util/ply.h"
#include "util/obj.h"
#include "util/bitmap.h"
#include "util/semantic_table.h"
#include "base/common.h"
#include "estimators/plane_estimator.h"
#include "optim/ransac/loransac.h"
#include "mvs/workspace.h"

#include "base/version.h"
#include "../Configurator_yaml.h"

using namespace sensemap;

std::string configuration_file_path;

Eigen::Matrix4d LoadAlignParam(const std::string& filepath) {
    std::ifstream file(filepath.c_str());
    CHECK(file.is_open()) << filepath;

    std::string line;
    std::string item;

    // Plane Equation.
    while (std::getline(file, line)) {
        StringTrim(&line);
        if (line.empty() || line[0] == '#') {
            continue;
        } else {
            break;
        }
    }
    std::stringstream plane_line_stream(line);
    Eigen::Vector4d plane_equation;
    std::getline(plane_line_stream >> std::ws, item, ' ');
    plane_equation.x() = std::stold(item);
    std::getline(plane_line_stream >> std::ws, item, ' ');
    plane_equation.y() = std::stold(item);
    std::getline(plane_line_stream >> std::ws, item, ' ');
    plane_equation.z() = std::stold(item);
    std::getline(plane_line_stream >> std::ws, item, ' ');
    plane_equation.w() = std::stold(item);

    std::cout << "Plane Equation: " << plane_equation.transpose() << std::endl;

    // Transform Matrix.
    while (std::getline(file, line)) {
        StringTrim(&line);
        if (line.empty() || line[0] == '#') {
            continue;
        } else {
            break;
        }
    }
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    for (int i = 0; i < 3; ++i) {
        std::stringstream trans_line_stream(line);
        std::getline(trans_line_stream >> std::ws, item, ' ');
        T(i, 0) = std::stold(item);
        std::getline(trans_line_stream >> std::ws, item, ' ');
        T(i, 1) = std::stold(item);
        std::getline(trans_line_stream >> std::ws, item, ' ');
        T(i, 2) = std::stold(item);
        std::getline(trans_line_stream >> std::ws, item, ' ');
        T(i, 3) = std::stold(item);

        std::getline(file, line);
    }
    std::cout << "Transform Matrix: " << std::endl << T << std::endl;

    // Projection Matrix.
    while (std::getline(file, line)) {
        StringTrim(&line);
        if (line.empty() || line[0] == '#') {
            continue;
        } else {
            break;
        }
    }
    Eigen::Matrix4d M = Eigen::Matrix4d::Identity();
    for (int i = 0; i < 3; ++i) {
        std::stringstream proj_line_stream(line);
        std::getline(proj_line_stream >> std::ws, item, ' ');
        M(i, 0) = std::stold(item);
        std::getline(proj_line_stream >> std::ws, item, ' ');
        M(i, 1) = std::stold(item);
        std::getline(proj_line_stream >> std::ws, item, ' ');
        M(i, 2) = std::stold(item);
        std::getline(proj_line_stream >> std::ws, item, ' ');
        M(i, 3) = std::stold(item);

        std::getline(file, line);
    }
    std::cout << "Projection Matrix: " << std::endl << M<< std::endl;
    file.close();

    return M * T;
}

void ExtractGround(const TriangleMesh& mesh, TriangleMesh& plane_mesh) {
   
    plane_mesh.faces_.clear();
    plane_mesh.vertices_.clear();

    std::unordered_map<int, int> vtx_index_map;
    for (auto facet : mesh.faces_) {
        int8_t label0 = mesh.vertex_labels_.at(facet[0]);
        int8_t label1 = mesh.vertex_labels_.at(facet[1]);
        int8_t label2 = mesh.vertex_labels_.at(facet[2]);
        if (label0 == LABLE_GROUND && label1 == LABLE_GROUND || label2 == LABLE_GROUND) {
            int vidx0, vidx1, vidx2;
            if (vtx_index_map.count(facet[0]) == 0) {
                vidx0 = plane_mesh.vertices_.size();
                vtx_index_map[facet[0]] = vidx0;
                plane_mesh.vertices_.emplace_back(mesh.vertices_.at(facet[0]));
            } else {
                vidx0 = vtx_index_map.at(facet[0]);
            }
            if (vtx_index_map.count(facet[1]) == 0) {
                vidx1 = plane_mesh.vertices_.size();
                vtx_index_map[facet[1]] = vidx1;
                plane_mesh.vertices_.emplace_back(mesh.vertices_.at(facet[1]));
            } else {
                vidx1 = vtx_index_map.at(facet[1]);
            }
            if (vtx_index_map.count(facet[2]) == 0) {
                vidx2 = plane_mesh.vertices_.size();
                vtx_index_map[facet[2]] = vidx2;
                plane_mesh.vertices_.emplace_back(mesh.vertices_.at(facet[2]));
            } else {
                vidx2 = vtx_index_map.at(facet[2]);
            }

            facet.x() = vidx0;
            facet.y() = vidx1;
            facet.z() = vidx2;
            plane_mesh.faces_.emplace_back(facet);
        }
    }
}

void GenerateHeatMap(const TriangleMesh& mesh, const int max_grid_size,
    const std::string& workspace_path) {
    Bitmap bitmap;
    bitmap.Allocate(max_grid_size, max_grid_size, false);
    bitmap.Fill(BitmapColor<uint8_t>(255));

    const float invalid_depth = (FLT_MAX - 2);
    cv::Mat heatmap(max_grid_size, max_grid_size, CV_32FC1);
    heatmap.setTo(invalid_depth);

    double min_depth = std::numeric_limits<double>::max();
    double max_depth = std::numeric_limits<double>::lowest();
    for (auto & vtx : mesh.vertices_) {
        min_depth = std::min(min_depth, vtx.z());
        max_depth = std::max(max_depth, vtx.z());
    }

    std::cout << "min/max depth: " << min_depth << " " << max_depth << std::endl;

    for (auto & facet : mesh.faces_) {
        auto vtx0 = mesh.vertices_.at(facet[0]);
        auto vtx1 = mesh.vertices_.at(facet[1]);
        auto vtx2 = mesh.vertices_.at(facet[2]);

        int u_min = std::min(vtx0[0], std::min(vtx1[0], vtx2[0]));
        int u_max = std::max(vtx0[0], std::max(vtx1[0], vtx2[0]));
        int v_min = std::min(vtx0[1], std::min(vtx1[1], vtx2[1]));
        int v_max = std::max(vtx0[1], std::max(vtx1[1], vtx2[1]));
        u_min = std::max(0, u_min);
        v_min = std::max(0, v_min);
        u_max = std::min(max_grid_size - 1, u_max);
        v_max = std::min(max_grid_size - 1, v_max);

        Eigen::Vector2d edge1 = (vtx1 - vtx0).head<2>();
        Eigen::Vector2d edge2 = (vtx2 - vtx0).head<2>();
        double e1 = edge1.norm();
        double e2 = edge2.norm();

        double cos = edge1.dot(edge2) / (e1 * e2);
        double sin = std::sqrt(1.0 - cos * cos);
        double area = 0.5 * e1 * e2 * sin;

        Eigen::Vector2d v1 = (vtx1 - vtx0).head<2>();
        Eigen::Vector2d v2 = (vtx2 - vtx1).head<2>();
        Eigen::Vector2d v3 = (vtx0 - vtx2).head<2>();

        for (int v = v_min; v <= v_max; ++v) {
            for (int u = u_min; u <= u_max; ++u) {
                double m1 = v1.x() * (v - vtx0[1]) - (u - vtx0[0]) * v1.y();
                double m2 = v2.x() * (v - vtx1[1]) - (u - vtx1[0]) * v2.y();
                double m3 = v3.x() * (v - vtx2[1]) - (u - vtx2[0]) * v3.y();
                if (m1 < 0 && m2 < 0 && m3 < 0) {
                    Eigen::Vector2d uv(u, v);

                    Eigen::Vector2d iedge0 = uv - vtx0.head<2>();
                    Eigen::Vector2d iedge1 = uv - vtx1.head<2>();
                    Eigen::Vector2d iedge2 = uv - vtx2.head<2>();
                    double ie0 = iedge0.norm();
                    double ie1 = iedge1.norm();
                    double ie2 = iedge2.norm();

                    double cos0 = iedge0.dot(iedge1) / (ie0 * ie1);
                    double sin0 = std::sqrt(1.0 - cos0 * cos0);
                    double w0 = 0.5 * ie0 * ie1 * sin0 / area;

                    double cos1 = iedge1.dot(iedge2) / (ie1 * ie2);
                    double sin1 = std::sqrt(1.0 - cos1 * cos1);
                    double w1 = 0.5 * ie1 * ie2 * sin1 / area;

                    double z = w0 * vtx2.z() + w1 * vtx0.z() + (1 - w0 - w1) * vtx1.z();

                    float& nd = heatmap.at<float>(v, u);
                    if (nd > z) {
                        nd = z;
                    }
                    
                    double val = (nd - min_depth) / (max_depth - min_depth);

                    BitmapColor<float> color(255 * (1 - std::log1p(val)));
                    bitmap.SetPixel(u, v, color.Cast<uint8_t>());
                }
            }
        }
    }

    for (int r = 0; r < max_grid_size; ++r) {
        for (int c = 0; c < max_grid_size; ++c) {
            float& d = heatmap.at<float>(r, c);
            if (d == invalid_depth) {
                d = 0;
            }
        }
    }

    bitmap.Write(JoinPaths(workspace_path, "heat_map.jpg"));
    cv::imwrite(JoinPaths(workspace_path, "heat_map.png"), heatmap);
}

int main(int argc, char *argv[]) {
    
	PrintHeading("SenseMap.  Copyright(c) 2019, SenseTime Group.");	
	PrintHeading(std::string("Version: ") + __VERSION__);

	configuration_file_path = std::string(argv[1]);
    Configurator param;
    param.Load(configuration_file_path.c_str());

	std::string workspace_path = param.GetArgument("workspace_path", "");
	std::string image_type = param.GetArgument("image_type", "perspective");
    int max_grid_size = param.GetArgument("max_cad_grid", 1000);
    bool verbose = param.GetArgument("verbose", 0);

    std::string semantic_table_path = param.GetArgument("semantic_table_path", "");
    if (ExistsFile(semantic_table_path)) {
        LoadSemanticColorTable(semantic_table_path.c_str());
    }

    int num_reconstruction = 0;
    for (size_t reconstruction_idx = 0; ; reconstruction_idx++) {
        auto reconstruction_path =
            JoinPaths(workspace_path, std::to_string(reconstruction_idx));
        if (!ExistsDir(reconstruction_path)) {
            break;
        }
        auto dense_reconstruction_path =
            JoinPaths(reconstruction_path, DENSE_DIR);
        if (!ExistsDir(dense_reconstruction_path)) {
            continue;
        }

        auto in_model_path = JoinPaths(dense_reconstruction_path, MODEL_NAME);
        auto sem_model_name = in_model_path.substr(0, in_model_path.size() - 4) + "_sem.obj";
        if (!ExistsFile(sem_model_name)) {
            std::cerr << StringPrintf("Error! %s does not exist!\n", sem_model_name.c_str());
            continue;
        }
        TriangleMesh mesh;
        ReadTriangleMeshObj(sem_model_name, mesh, true, true);

        TriangleMesh plane_mesh;
        ExtractGround(mesh, plane_mesh);

        auto param_file = JoinPaths(dense_reconstruction_path, "align_param.txt");
        if (!ExistsFile(param_file)) {
            std::cerr << StringPrintf("Error! %s does not exist!\n", param_file.c_str());
            continue;
        }
        Eigen::Matrix4d T = LoadAlignParam(param_file);

        for (auto & vtx : plane_mesh.vertices_) {
            vtx = (T * vtx.homogeneous()).head<3>();
        }

        // WriteTriangleMeshObj("plane-mesh.obj", plane_mesh);

        GenerateHeatMap(plane_mesh, max_grid_size, dense_reconstruction_path);
    }

    return 0;
}