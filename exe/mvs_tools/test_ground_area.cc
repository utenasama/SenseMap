//Copyright (c) 2020, SenseTime Group.
//All rights reserved.

#include <iostream>
#include <fstream>
#include <vector>
#include <queue>
#include <Eigen/Dense>
#include <boost/filesystem/path.hpp>
#include "util/semantic_table.h"
#include "util/obj.h"
#include "CGAL/Plane_3.h"
#include "CGAL/Point_2.h"
#include "CGAL/Point_3.h"
#include "CGAL/Vector_3.h"
#include "CGAL/Timer.h"
#include <chrono>
#include <memory>
#include "poisson_disk_wrapper/utils_sampling.hpp"
#include "util/exception_handler.h"
#include "../Configurator_yaml.h"
#include "base/version.h"
#include "opencv2/opencv.hpp"

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
typedef CGAL::Exact_predicates_inexact_constructions_kernel							      K;
typedef K::Point_2 																	      Point_2;
typedef K::Point_3                                                                        Point_3;
typedef K::Vector_3                                                                       Vector_3;
typedef K::Plane_3                                                                        Plane_3;

using namespace sensemap;

struct Options {
	std::string in_model_name;
    std::string output_dir;
    double grid_size;
    double max_hole_area;
    bool verbose;

    void Print() {
        PrintHeading2("Options");

        PrintOption(in_model_name);
        PrintOption(output_dir);
        PrintOption(grid_size);
        PrintOption(max_hole_area);
        PrintOption(verbose);
    }
};

std::shared_ptr<Plane_3> FitPlane(const std::vector<Eigen::Vector3d> &points) {

    if (points.size() < 3) {
        return nullptr;
    }

    int num_atoms = points.size();
	Eigen::Matrix<Eigen::Vector3d::Scalar, Eigen::Dynamic, Eigen::Dynamic > coord(3, num_atoms);

	for (size_t i = 0; i < num_atoms; ++i) {
		coord.col(i) = points[i];
	}
	// calculate centroid
	Eigen::Vector3d centroid(coord.row(0).mean(), coord.row(1).mean(), coord.row(2).mean());
	// subtract centroid
	coord.row(0).array() -= centroid(0); coord.row(1).array() -= centroid(1); coord.row(2).array() -= centroid(2);
	auto svd = coord.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV);
	Eigen::Vector3d plane_normal = svd.matrixU().rightCols<1>();

    return std::make_shared<Plane_3>(
        Point_3(centroid[0], centroid[1], centroid[2]),
        Vector_3(plane_normal[0], plane_normal[1], plane_normal[2]));
}

void LoadOption(const std::string &configuration_path, Options &option)
{
    Configurator param;
    param.Load(configuration_path.c_str());

	option.in_model_name = param.GetArgument("area_tool_in_model_name", "");
    option.output_dir = param.GetArgument("area_tool_output_dir", "");
    option.verbose = param.GetArgument("area_tool_verbose", 0);
    option.max_hole_area = param.GetArgument("area_tool_max_hole_area", 200.0f);
    option.grid_size = param.GetArgument("area_tool_grid_size", 0.1f);
    if (option.in_model_name == "") {
        std::cout << "Empty input model name." << std::endl;
        return;
    }
    if (option.output_dir == "")
    {
        auto pos = option.in_model_name.find_last_of("/\\");
        if (pos == std::string::npos) {
            std::cout << "Not a full input model file path!" << std::endl;
            return;
        }
        std::string in_model_dir = option.in_model_name.substr(0, pos);
        
        option.output_dir = in_model_dir + "/area_tool";
    }

    std::string semantic_table_path = param.GetArgument("semantic_table_path", "");
    if (ExistsFile(semantic_table_path)) {
        LoadSemanticColorTable(semantic_table_path.c_str());
    }
    
    option.Print();

}

std::vector<std::vector<int>> FindHoles(const std::vector<std::vector<bool>> &grid, const int width, const int height)
{
    std::vector<std::vector<int>> holes;

    typedef struct node {
		int x, y;
		node() {};
		node(int in_x, int in_y) { x = in_x, y = in_y;}
	}node;
	int dir[4][2] = {1, 0, -1, 0, 0, 1, 0, -1};
	
	std::vector<std::vector<bool>> visit(height, std::vector<bool>(width, false));

    for (int y = 0; y < height; ++ y) {
        for (int x = 0; x < width; ++ x) {
            if (visit[y][x]) continue;
            if (grid[y][x]) continue;
            holes.emplace_back(std::vector<int>());
            auto &hole = holes.back();

            std::queue<node> q;
	        q.push(node(x, y));
            visit[y][x] = true;
            while (!q.empty()) {
                auto p = q.front();
                q.pop();

                int idx = p.y * width + p.x;
                hole.emplace_back(idx);
                for (int i = 0; i < 4; ++ i) {
                    int tx = p.x + dir[i][0];
                    int ty = p.y + dir[i][1];
                    if (tx < 0 || tx >= width || ty < 0 || ty >= height) continue;

                    if (visit[ty][tx]) continue;

                    visit[ty][tx] = true;
                    if (!grid[ty][tx]) { 
                        q.push(node(tx, ty));
                    }
                }
            }
        }
    }

	return holes;
}

int main(int argc, char *argv[]) {

    PrintHeading("SenseMap.  Copyright(c) 2021, SenseTime Group.");	
	PrintHeading(std::string("Version: ") + __VERSION__);
    
    CGAL::Timer timer;
	timer.start();

	std::string configuration_file_path = std::string(argv[1]);
    Options options;
    LoadOption(configuration_file_path, options);

    if(!ExistsFile(options.in_model_name)){
        std::cout << options.in_model_name << " not exists." << std::endl;
		return StateCode::INVALID_INPUT_PARAM;
    }

    CreateDirIfNotExists(options.output_dir);

    TriangleMesh mesh;
    puts("load mesh ...");
    auto st = std::chrono::steady_clock::now();
    ReadTriangleMeshObj(options.in_model_name, mesh, true, true);
    auto ed = std::chrono::steady_clock::now();
    std::cout << "done. " << std::chrono::duration<float>(ed - st).count() << " sec." << std::endl;

    // mesh.vertex_labels_.resize(mesh.vertices_.size());
    // for (int i = 0; i < mesh.vertex_labels_.size(); ++ i) {
    //     mesh.vertex_labels_[i] = LABLE_GROUND;
    // }
    
    std::vector<Eigen::Vector3d> ground_vts;
    for (int i = 0; i < mesh.vertex_labels_.size(); ++ i) {
        if (mesh.vertex_labels_[i] == LABLE_GROUND) {
            ground_vts.emplace_back(mesh.vertices_[i]);
        }
    }
    auto ground_pl = FitPlane(ground_vts);
    if (ground_pl == nullptr) {
        std::cout << "ground vts: " << ground_vts.size() << std::endl;
        std::cout << "can not fit plane for model " << options.in_model_name << std::endl;
        return 0;
    }

    std::vector<Poisson_sampling::Vec3> mesh_vertexs(mesh.vertices_.size());
    std::vector<Poisson_sampling::Vec3> mesh_normals(mesh.vertices_.size());;
    std::vector<int> ground_faces;
    std::vector<int> mesh_labels(mesh.vertices_.size());
    for (int i = 0; i < mesh.vertices_.size(); ++ i) {
        mesh_vertexs[i] = 
            Poisson_sampling::Vec3(mesh.vertices_[i][0], mesh.vertices_[i][1], mesh.vertices_[i][2]);
        mesh_normals[i] = 
            Poisson_sampling::Vec3(mesh.vertex_normals_[i][0], mesh.vertex_normals_[i][1], mesh.vertex_normals_[i][2]);
        mesh_labels[i] = mesh.vertex_labels_[i];
    }
    for (int i = 0; i < mesh.faces_.size(); ++ i) {
        int idx0 = mesh.faces_[i][0];
        int idx1 = mesh.faces_[i][1];
        int idx2 = mesh.faces_[i][2];
        if (mesh_labels[idx0] == LABLE_GROUND || mesh_labels[idx1] == LABLE_GROUND || mesh_labels[idx2] == LABLE_GROUND)
        {
            ground_faces.emplace_back(mesh.faces_[i][0]);
            ground_faces.emplace_back(mesh.faces_[i][1]);
            ground_faces.emplace_back(mesh.faces_[i][2]);
        }
    }

    std::vector<Poisson_sampling::Vec3> sampled_vertexs, sampled_normals;
    std::vector<int> sampled_labels;

    double precision = std::max(0.05, options.grid_size * 0.7);
    puts("sampling ...");
    std::cout << "sampling radius: " << precision << " m" << std::endl;
    st = std::chrono::steady_clock::now();
    Poisson_sampling::poisson_disk(
		precision, 0, 
		mesh_vertexs, mesh_normals, ground_faces, mesh_labels,
		sampled_vertexs, sampled_normals, sampled_labels);
    ed = std::chrono::steady_clock::now();
    std::cout << "done. " << std::chrono::duration<float>(ed - st).count() << " sec." << std::endl;

    std::vector<Point_2> points_2d(sampled_vertexs.size());
    for (int i = 0; i < sampled_vertexs.size(); ++ i) {
        auto p_2d = ground_pl->to_2d(
            Point_3(sampled_vertexs[i].x, sampled_vertexs[i].y, sampled_vertexs[i].z));
        points_2d[i] = p_2d;
    }

    const double shift = options.grid_size * 3;

	double x_min = points_2d[0].x(), x_max = x_min;
	double y_min = points_2d[0].y(), y_max = y_min;
	for (int i = 0; i < points_2d.size(); ++ i) {
		x_min = std::min(x_min, points_2d[i].x());
		x_max = std::max(x_max, points_2d[i].x());
		y_min = std::min(y_min, points_2d[i].y());
		y_max = std::max(y_max, points_2d[i].y());
	}
	x_min -= shift;
	x_max += shift;
	y_min -= shift;
	y_max += shift;
	
	int width = (x_max - x_min) / options.grid_size + 10;
	int height = (y_max - y_min) / options.grid_size + 10;

	std::vector<std::vector<bool>> mask(height, std::vector<bool>(width, false));
	for (int i = 0; i < points_2d.size(); ++ i)
	{
		int x = (points_2d[i].x() - x_min) / options.grid_size;
		int y = (points_2d[i].y() - y_min) / options.grid_size;
		mask[y][x] = true;
	}

    const int out_image_width = (width > 1024 ? 1024 : width);
    const int out_image_height = out_image_width * height / width;
    const cv::Vec3b background_color(255, 255, 255);
    const cv::Vec3b occupied_color(0, 0, 255);

    std::cout << "saving ground_raw image ..." << std::endl;
    std::cout << "image size : " << out_image_width << ' ' << out_image_height << std::endl;
    const double width_scale = out_image_width * 1.0 / width;
    const double height_scale = out_image_height * 1.0 / height;

    cv::Mat raw_image(out_image_height, out_image_width, CV_8UC3, cv::Scalar(background_color));
    for (int y = 0; y < height; ++ y) {
        for (int x = 0; x < width; ++ x) {
            if (!mask[y][x]) continue;
            int img_x = x * width_scale;
            int img_y = y * height_scale;

            raw_image.at<cv::Vec3b>(img_y, img_x) = occupied_color;
        }
    }
    cv::imwrite(options.output_dir + "/ground_raw.jpg", raw_image);
    puts("done.");

    std::cout << "calculating ground area and saving result ..." << std::endl;
	const double total_area = width * height * options.grid_size * options.grid_size;
	const auto &holes = FindHoles(mask, width, height);
    std::vector<bool> hole_flag;
    int non_occupied_cnt = 0;
    for (int i = 0; i < holes.size(); ++ i) {
        int grid_num = holes[i].size();
        double hole_area = grid_num * options.grid_size * options.grid_size;
        if (i == 0 || hole_area > options.max_hole_area) {
            non_occupied_cnt += grid_num;
            hole_flag.emplace_back(true);            
        } else {
            hole_flag.emplace_back(false);
        }
    }
    const double non_occupied_area = non_occupied_cnt * options.grid_size * options.grid_size;
    const double occupied_area = total_area - non_occupied_area;

	cv::Mat image_result(out_image_height, out_image_width, CV_8UC3, cv::Scalar(occupied_color));
    const cv::Vec3b hole_color(255, 255, 0);

    for (int i = 0; i < holes.size(); ++ i) {
        const auto &idx = holes[i];
        cv::Vec3b color;
        if (i == 0) color = background_color;
        else if (hole_flag[i]) color = hole_color;
        else continue;

        for (auto id : idx) {
            int y = id / width;
            int x = id % width;
            int img_x = x * width_scale;
            int img_y = y * height_scale;
            image_result.at<cv::Vec3b>(img_y, img_x) = color;
        }
    }
   
    cv::imwrite(options.output_dir + "/ground_result.jpg", image_result);
    std::cout << "done. ground area : " << occupied_area << " m^2" << std::endl;

    FILE *fp = fopen((options.output_dir + "/ground_area.txt").c_str(), "w+");
    fprintf(fp, "area: %.3f m^2\n", occupied_area);
    fclose(fp);

    printf("Total time: %.2f min.\n", timer.time() / 60.0);
    return 0;
}

