//Copyright (c) 2019, SenseTime Group.
//All rights reserved.

#include <iostream>
#include <fstream>
#include <sstream>

#include "util/misc.h"
#include "util/string.h"
#include "util/proc.h"
#include "base/common.h"
#include "base/reconstruction_manager.h"
#include "base/undistortion.h"
#include "controllers/patch_match_controller.h"
#include "mvs/delaunay_meshing.h"

#include "base/version.h"
#include "../Configurator_yaml.h"

std::string configuration_file_path;

using namespace sensemap;

int main(int argc, char *argv[]) {
    using namespace sensemap;
	using namespace mvs;

	PrintHeading("SenseMap.  Copyright(c) 2019, SenseTime Group.");	
	PrintHeading(std::string("Version: ") + __VERSION__);

	Timer meshing_timer;
	meshing_timer.Start();

	configuration_file_path = std::string(argv[1]);
    Configurator param;
    param.Load(configuration_file_path.c_str());

	int reconstrction_idx = -1;
    int cluster_idx = -1;
	if (argc > 2) {
        reconstrction_idx = atoi(argv[2]);
    }

    if (argc > 3) {
        cluster_idx = atoi(argv[3]);
    }

	std::string workspace_path = param.GetArgument("workspace_path", "");

	bool map_update = param.GetArgument("map_update", 0);
	if (map_update) {
		workspace_path = JoinPaths(workspace_path, "map_update");
	}

	std::string image_type = param.GetArgument("image_type", "perspective");
    
    DelaMeshing::Options options;
    options.dist_insert = param.GetArgument("dist_insert", 5.0f);
    options.diff_depth = param.GetArgument("diff_depth", 0.01f);
    options.decimate_mesh = param.GetArgument("decimate_mesh", 1.0f);
    options.only_remove_edge_spurious = 
        static_cast<bool>(param.GetArgument("only_remove_edge_spurious", 0));
    options.remove_spurious = param.GetArgument("remove_spurious", 10.0f);
    options.remove_spikes = param.GetArgument("remove_spikes", 1);
    options.close_holes = param.GetArgument("close_holes", 30);
    options.smooth_mesh = param.GetArgument("iter_smooth_mesh", 2);
    options.fix_mesh = param.GetArgument("fix_mesh", 0);
    options.num_isolated_pieces = 
        static_cast<int>(param.GetArgument("num_isolated_pieces", 0));
    options.adaptive_insert = param.GetArgument("est_curvature", 0);
    options.plane_insert_factor = param.GetArgument("plane_insert_factor", 2.0f);
    options.plane_score_thred = param.GetArgument("plane_score_thred", 0.80f);
    options.sigma = param.GetArgument("delaunay_sigma", 1.0f);
    options.roi_mesh = param.GetArgument("roi_mesh", 0);
    options.roi_box_width = param.GetArgument("roi_box_width", -1.f);
    options.roi_box_factor = param.GetArgument("roi_box_factor", -1.f);

    options.mesh_cluster = param.GetArgument("mesh_cluster", 0);
    options.overlap_factor = param.GetArgument("overlap_factor", 0.015f);
    float read_max_ram = param.GetArgument("max_ram", -1.0f);
    float get_max_ram;
    if (GetAvailableMemory(get_max_ram) && read_max_ram < 0){
        options.max_ram = get_max_ram;
    } else {
        options.max_ram = read_max_ram;
    }
    // options.overlap_factor = 0.02;
    bool fused_delaunay_sample = param.GetArgument("fused_delaunay_sample", true);
    if (fused_delaunay_sample || options.dist_insert <= 0){
        std::cout << "fused_delaunay_sample: " << fused_delaunay_sample 
            << " || options.dist_insert <= 0 -> options.sampInsert = false" << std::endl;
        // options.dist_insert = -1.0f;
        options.sampInsert = false;
    }

    std::string semantic_table_path = param.GetArgument("semantic_table_path", "");
    if (ExistsFile(semantic_table_path)) {
        LoadSemanticColorTable(semantic_table_path.c_str());
    }

    size_t num_reconstruction = 0;
    for (size_t reconstruction_idx = 0; ;reconstruction_idx++) {
        const auto& reconstruction_path = 
            JoinPaths(workspace_path, std::to_string(reconstruction_idx));
        if (!ExistsDir(reconstruction_path)) {
            break;
        }
        num_reconstruction++;
    }

    size_t reconstruction_begin = reconstrction_idx < 0 ? 0 : reconstrction_idx;
    num_reconstruction = reconstrction_idx < 0 ? num_reconstruction : reconstrction_idx+1;
    for (size_t reconstruction_idx = reconstruction_begin; reconstruction_idx < num_reconstruction; 
        reconstruction_idx++) {
        float begin_memroy, end_memory;
        GetAvailableMemory(begin_memroy);

        DelaMeshing delaunay(options, workspace_path, image_type, reconstruction_idx, cluster_idx);
        delaunay.Start();
        delaunay.Wait();

        GetAvailableMemory(end_memory);
        std::cout << StringPrintf("Meshing Reconstruction %d Elapsed time: %.3f [minutes], Memory: %3f (%3f - %3f) [G]", 
                                reconstruction_idx, delaunay.GetTimer().ElapsedMinutes(), 
                                (begin_memroy - end_memory), begin_memroy, end_memory).c_str()
                << std::endl;
    }

	std::cout << StringPrintf("Meshing Elapsed time: %.3f [minutes]", 
							  meshing_timer.ElapsedMinutes()).c_str()
					<< std::endl;

    return 0;
}