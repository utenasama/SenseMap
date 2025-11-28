//Copyright (c) 2019, SenseTime Group.
//All rights reserved.

#include <boost/filesystem/path.hpp>
#include <iostream>
#include <dirent.h>
#include <sys/stat.h>

#include "util/obj.h"
#include "util/misc.h"
#include "base/common.h"
#include "mvs/workspace.h"
// #include "mvs/simplify.h"

#include "../Configurator_yaml.h"
#include "base/version.h"

#define DEFAULT_FACET_COUNT 100000

using namespace sensemap;
// using namespace geometry;

std::string configuration_file_path;

int main(int argc, char *argv[]) {

	PrintHeading("SenseMap.  Copyright(c) 2019, SenseTime Group.");	
	PrintHeading(std::string("Version: ") + __VERSION__);

	configuration_file_path = std::string(argv[1]);
    Configurator param;
    param.Load(configuration_file_path.c_str());

	std::string workspace_path = param.GetArgument("workspace_path", "");
    int target_count = param.GetArgument("facet_count", DEFAULT_FACET_COUNT);
    double agressiveness = param.GetArgument("agressiveness", 7);
    const std::string in_model_path = param.GetArgument("in_model_path", "");
    const std::string out_model_path = 
        param.GetArgument("out_simp_model_path", "");

    // int num_reconstruction = 0;
    // for (size_t reconstruction_idx = 0; ;reconstruction_idx++) {
    //     const auto& reconstruction_path = 
    //         JoinPaths(workspace_path, std::to_string(reconstruction_idx));
    //     std::cout << reconstruction_path << std::endl;
    //     if (!ExistsDir(reconstruction_path)) {
    //         break;
    //     }

    //     const std::string input_model_path = 
    //         JoinPaths(reconstruction_path, DENSE_DIR, MODEL_NAME);
        
        // const std::string output_model_path =
        //     JoinPaths(reconstruction_path, DENSE_DIR, simp_model_name);

        std::cout << "in_model_path: " << in_model_path << std::endl;

        TriangleMesh input_model;
        ReadTriangleMeshObj(in_model_path, input_model, true);

        double fDecimate = target_count * 1.0 / input_model.faces_.size();

        input_model.Clean(fDecimate, 0, false, 0, 0, false);

        std::cout << input_model.vertices_.size() << std::endl;
        std::cout << input_model.vertex_normals_.size() << std::endl;
        std::cout << input_model.faces_.size() << std::endl;
        input_model.ComputeNormals();

        std::cout << input_model.vertices_.size() << std::endl;
        std::cout << input_model.vertex_normals_.size() << std::endl;
        std::cout << input_model.faces_.size() << std::endl;
        std::cout << "out_model_path: " << out_model_path << std::endl;
        WriteTriangleMeshObj(out_model_path, input_model, true);

        // // Simplify Mesh.
        // {
        //     Simplify::vertices.resize(input_model.vertices_.size());
        //     for (int i = 0; i < input_model.vertices_.size(); ++i) {
        //         const Eigen::Vector3d& vert = input_model.vertices_[i];
        //         Simplify::vertices[i].p = vec3f(vert[0], vert[1], vert[2]);
        //     }
        //     Simplify::triangles.resize(input_model.faces_.size());
        //     for (int i = 0; i < input_model.faces_.size(); ++i) {
        //         const Eigen::Vector3i& facet = input_model.faces_[i];
        //         Simplify::triangles[i].v[0] = facet[0]; 
        //         Simplify::triangles[i].v[1] = facet[1];
        //         Simplify::triangles[i].v[2] = facet[2]; 
        //     }
        //     Simplify::simplify_mesh(target_count, agressiveness, true);
        // }

        // // Write Obj.
        // TriangleMesh simp_model;
        // simp_model.vertices_.resize(Simplify::vertices.size());
        // for (int i = 0; i < Simplify::vertices.size(); ++i) {
        //     simp_model.vertices_[i][0] = Simplify::vertices[i].p.x;
        //     simp_model.vertices_[i][1] = Simplify::vertices[i].p.y;
        //     simp_model.vertices_[i][2] = Simplify::vertices[i].p.z;
        // }
        // simp_model.faces_.resize(Simplify::triangles.size());
        // for (int i = 0; i < Simplify::triangles.size(); ++i) {
        //     simp_model.faces_[i][0] = Simplify::triangles[i].v[0];
        //     simp_model.faces_[i][1] = Simplify::triangles[i].v[1];
        //     simp_model.faces_[i][2] = Simplify::triangles[i].v[2];
        // }

        // simp_model.ComputeNormals();

        // std::cout << "out_model_path: " << out_model_path << std::endl;
        // WriteTriangleMeshObj(out_model_path, simp_model);
    // }

    return 0;
}