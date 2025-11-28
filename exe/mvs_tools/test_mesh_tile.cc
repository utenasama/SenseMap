//Copyright (c) 2019, SenseTime Group.
//All rights reserved.

#include <boost/filesystem/path.hpp>
#include <iostream>
#include <sys/stat.h>

#include <Eigen/Dense>

#include "base/common.h"
#include "util/misc.h"
#include "util/ply.h"
#include "util/obj.h"
#include "mvs/workspace.h"
#include "../Configurator_yaml.h"

#define EPSILON std::numeric_limits<float>::epsilon()

using namespace sensemap;
std::string configuration_file_path;
std::string workspace_path_;
struct MeshBox mesh_box_;

int main(int argc, char* argv[]) {

	PrintHeading("SenseMap.  Copyright(c) 2021, SenseTime Group.");	
	PrintHeading(std::string("Version: ") + __VERSION__);

    PrintHeading1(StringPrintf("Extract Cluster"));
    configuration_file_path = std::string(argv[1]);
    Configurator param;
    param.Load(configuration_file_path.c_str());

    workspace_path_ = param.GetArgument("workspace_path", "");
    mesh_box_.border_width = 
        static_cast<float>(param.GetArgument("border_width", -1.0f));

    for (size_t reconstruction_idx = 0; ; reconstruction_idx++) {        
        auto reconstruction_path =
            JoinPaths(workspace_path_, std::to_string(reconstruction_idx));
        if (!ExistsDir(reconstruction_path)) {
            break;
        }

        PrintHeading1(StringPrintf("Extracting# %d", reconstruction_idx));
        // set box para
        auto box_path = JoinPaths(reconstruction_path, ROI_BOX_NAME);
        if (!ExistsFile(box_path)){
            std::cout << "Box-file is empty!" << std::endl;
            continue;
        }
        mesh_box_.ReadBox(box_path);
        mesh_box_.SetBoundary();
        mesh_box_.Print();

        // read mesh
        auto dense_reconstruction_path = JoinPaths(reconstruction_path, DENSE_DIR);
        auto model_path = JoinPaths(dense_reconstruction_path, MODEL_NAME);
        if (!ExistsFile(model_path)) {
            std::cout << "not exitt model file" << std::endl;
            continue;
        }
        TriangleMesh mesh;
        ReadTriangleMeshObj(model_path, mesh, true);
        std::size_t vertex_num = mesh.vertices_.size();
        std::size_t face_num = mesh.faces_.size();
        std::cout << "ori faces number: " << face_num << std::endl;

        bool flag = FilterWithBox(mesh, mesh_box_);

        auto filtered_model_path = 
            JoinPaths(dense_reconstruction_path, TILTED_MODEL_NAME);
        WriteTriangleMeshObj(filtered_model_path, mesh);
        std::cout << "filered faces number: " << mesh.faces_.size() << std::endl;
    }

    return 0;
}