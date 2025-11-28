//Copyright (c) 2019, SenseTime Group.
//All rights reserved.

#include <boost/filesystem/path.hpp>
#include <iostream>
#include <unordered_map>
#include <string>
#include <set>
#include <dirent.h>
#include <sys/stat.h>
#include <gflags/gflags.h>

#include "util/obj.h"
#include "util/misc.h"
#include "util/ply.h"
#include "util/timer.h"
#include "base/common.h"

#include <functional>
#include <vector>
#include <fstream>
#include <boost/tuple/tuple.hpp>

#include "base/version.h"
#include "../Configurator_yaml.h"

using namespace sensemap;

std::string configuration_file_path;

int main(int argc, char *argv[]) {

	PrintHeading("SenseMap.  Copyright(c) 2021, SenseTime Group.");
	PrintHeading(std::string("Version: ") + __VERSION__);

    configuration_file_path = std::string(argv[1]);
    Configurator param;
    param.Load(configuration_file_path.c_str());

	  std::string workspace_path = param.GetArgument("workspace_path", "");
    Timer timer;
    timer.Start();

    bool has_sem = false;
    std::vector<PlyPoint> merge_points;
    std::vector<PlyPoint> merge_sem_points;
    std::vector<PlyPoint> merge_samp_points;
    TriangleMesh merge_mesh;
    TriangleMesh merge_sem_mesh;

    for (int rect_id = 0; ; rect_id++){
        auto reconstruction_path =
            JoinPaths(workspace_path, std::to_string(rect_id));
        if (!ExistsDir(reconstruction_path)){
            break;
        }
        std::string dense_path = JoinPaths(reconstruction_path.c_str(), DENSE_DIR);
        std::cout << "Merge Reconstruction: " << rect_id << std::endl;

        std::string fused_input_path = JoinPaths(dense_path.c_str(), FUSION_NAME);
        if (ExistsFile(fused_input_path)){
            std::vector<PlyPoint> fused_points = ReadPly(fused_input_path);
            if (ExistsFile(fused_input_path + ".sem")) {
                ReadPointsSemantic(fused_input_path + ".sem", fused_points);
                has_sem = true;
            }
            merge_points.insert(merge_points.end(), fused_points.begin(), fused_points.end());
            std::cout << "=> merge :" << fused_input_path << " (with sem " << std::to_string(has_sem) << ")" << std::endl;
        }

        std::string fused_sem_input_path = JoinPaths(dense_path.c_str(), FUSION_SEM_NAME);
        if (!has_sem && ExistsFile(fused_sem_input_path)){
            std::vector<PlyPoint> fused_sem_points = ReadPly(fused_sem_input_path);
            merge_sem_points.insert(merge_sem_points.end(), fused_sem_points.begin(), fused_sem_points.end());
            std::cout << "=> merge :" << fused_sem_input_path << std::endl;
        }

        std::string fused_smap_input_patch = JoinPaths(dense_path.c_str(), "fused_samp.ply");
        if (ExistsFile(fused_smap_input_patch)){
            std::vector<PlyPoint> fused_samp_points = ReadPly(fused_smap_input_patch);
            merge_samp_points.insert(merge_samp_points.end(), fused_samp_points.begin(), fused_samp_points.end());
            std::cout << "=> merge :" << fused_smap_input_patch << std::endl;
        }

        std::string mesh_input_path = JoinPaths(dense_path.c_str(), MODEL_NAME);
        if (ExistsFile(mesh_input_path)) {
            TriangleMesh mesh;
            ReadTriangleMeshObj(mesh_input_path, mesh, true);
            merge_mesh.AddMesh(mesh);
            std::cout << "=> merge :" << mesh_input_path << std::endl;
        }

        std::string mesh_sem_input_path = JoinPaths(dense_path.c_str(), SEM_MODEL_NAME);
        if (ExistsFile(mesh_sem_input_path)) {
            TriangleMesh sem_mesh;
            ReadTriangleMeshObj(mesh_sem_input_path, sem_mesh, true, true);
            merge_sem_mesh.AddMesh(sem_mesh);
            std::cout << "=> merge :" << mesh_sem_input_path << std::endl;
        }

    }

    std::string parent_path = JoinPaths(workspace_path, MERGE_DIR);
    if (!ExistsPath(parent_path)){
        boost::filesystem::create_directories(parent_path);
    }
    std::cout << "\nSave merge result to " << parent_path << std::endl;

    if (merge_points.size() > 0) {
        std::string output_path = JoinPaths(parent_path, FUSION_NAME);
        WriteBinaryPlyPoints(output_path, merge_points, false, true);

        if (has_sem) {
            // WritePointsSemantic(output_path + ".sem", merge_points);
            output_path = JoinPaths(parent_path, FUSION_SEM_NAME);
            WritePointsSemanticColor(output_path, merge_points);
        }
        std::cout << "=> save " << FUSION_NAME << std::endl;
    }

    if (merge_sem_points.size() > 0) {
        std::string output_path = JoinPaths(parent_path, FUSION_SEM_NAME);
        WriteBinaryPlyPoints(output_path, merge_sem_points, false, true);
        std::cout << "=> save " << FUSION_SEM_NAME << std::endl;
    }

    if (merge_samp_points.size() > 0) {
        std::string output_path = JoinPaths(parent_path, "fused_samp.ply");
        WriteBinaryPlyPoints(output_path, merge_samp_points, false, true);
        std::cout << "=> save " << "fused_samp.ply" << std::endl;
    }

    if (merge_mesh.vertices_.size() > 0 && merge_mesh.faces_.size() > 0){
        std::string output_path = JoinPaths(parent_path, MODEL_NAME);
        WriteTriangleMeshObj(output_path, merge_mesh);
        std::cout << "=> save " << MODEL_NAME << std::endl;
    }

    if (merge_sem_mesh.vertices_.size() > 0 && merge_sem_mesh.faces_.size() > 0){
        std::string output_path = JoinPaths(parent_path, SEM_MODEL_NAME);
        WriteTriangleMeshObj(output_path, merge_sem_mesh, true, true);
        std::cout << "=> save " << SEM_MODEL_NAME << std::endl;
    }

    timer.PrintMinutes();

    return 0;
}