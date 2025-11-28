// Copyright (c) 2019, SenseTime Group.
// All rights reserved.

#include <fstream>
#include <iostream>
#include <sys/stat.h>
#include <vector>
#include <unordered_set>
#include <unordered_map>

#include <boost/filesystem/path.hpp>
#include <boost/tuple/tuple.hpp>

#include <Eigen/Eigen>
#include <opencv2/opencv.hpp>

#include "base/reconstruction_manager.h"
#include "base/common.h"
#include "util/misc.h"
#include "util/proc.h"
#include "mvs/mvs_cluster.h"
#include "mvs/reconstruction_cluster.h"
#include "../Configurator_yaml.h"
#include "base/version.h"

using namespace sensemap;
std::string configuration_file_path;
std::string out_workspace_path;

int main(int argc, char* argv[]) {    

	PrintHeading("SenseMap.  Copyright(c) 2019, SenseTime Group.");	
	PrintHeading(std::string("Version: ") + __VERSION__);

	Timer cluster_timer;
	cluster_timer.Start();

    configuration_file_path = std::string(argv[1]);

	int reconstrction_idx = -1;
    if (argc > 2) {
        reconstrction_idx = atoi(argv[2]);
    }

    Configurator param;
    param.Load(configuration_file_path.c_str());

    std::string workspace_path = param.GetArgument("workspace_path", "");

	float max_ram = static_cast<float>(param.GetArgument("max_ram", -1.0f));
	float ram_eff_factor = static_cast<float>(param.GetArgument("ram_eff_factor", 0.75f));
    int cluster_step = static_cast<int>(param.GetArgument("cluster_step", 5));

	std::string gpu_index = param.GetArgument("gpu_index", "-1");
	float max_gpu_memory = static_cast<float>(param.GetArgument("max_gpu_memory", 6.0f));
	float gpu_memory_factor = param.GetArgument("gpu_memory_factor", 0.7f);

	bool geom_consistency = param.GetArgument("geom_consistency", 1);
	int num_iter_geom_consistency = param.GetArgument("num_iter_geom_consistency", 1);
	int max_num_src_images = param.GetArgument("max_num_src_images", 8);
	float min_triangulation_angle = param.GetArgument("min_triangulation_angle", 1.0f);
	bool refine_with_semantic = param.GetArgument("refine_with_semantic", 0);

	bool filter = param.GetArgument("filter", 0);
	bool geo_filter = param.GetArgument("geo_filter", 0);

	float fuse_common_persent = param.GetArgument("fuse_common_persent", 0.06f);
	bool with_normal = param.GetArgument("with_normal", 1);

    int cluster_num = 
        static_cast<int>(param.GetArgument("cluster_num", -1));
    std::string format = param.GetArgument("image_type", "perspective");
    float max_cell_size =
        static_cast<float>(param.GetArgument("max_cell_size", -1.0f));
    float min_common_view = 
        static_cast<float>(param.GetArgument("min_common_view", 0.25f));
    float max_filter_percent = 
        static_cast<float>(param.GetArgument("max_filter_percent", 0.5f));
    float dist_threshold = 
        static_cast<float>(param.GetArgument("dist_threshold", -1.0f));
    float outlier_spacing_factor = 
        static_cast<float>(param.GetArgument("outlier_spacing_factor", 3.0f));

    float max_image_size = 
        static_cast<int>(param.GetArgument("max_image_size", -1));
    float max_num_images_factor = 
        static_cast<float>(param.GetArgument("max_num_images_factor", 7.0f));
    float delaunay_dist_insert = param.GetArgument("dist_insert", 5.0f);

    bool map_update = static_cast<bool>(param.GetArgument("map_update", 0));
    if (map_update){
        if (ExistsDir(JoinPaths(workspace_path, "map_update"))){
            workspace_path = JoinPaths(workspace_path, "map_update");
        } else {
            map_update = false;
        }
    }
    
    float get_max_ram;
    if (GetAvailableMemory(get_max_ram) && max_ram < 0){
        max_ram = get_max_ram;
    }

    // MVS Cluster
    {

        PrintHeading1(StringPrintf("MVS Cluster"));

        mvs::MVSCluster::Options options;
        options.image_type = format;
        options.max_ram = max_ram;
        options.ram_eff_factor = ram_eff_factor;
        options.gpu_index = gpu_index;
        options.max_gpu_memory = max_gpu_memory;
        options.gpu_memory_factor = gpu_memory_factor;

        options.filter = filter;
        options.geo_filter = geo_filter;

        options.cluster_step = cluster_step;
        options.geom_consistency = geom_consistency;
        options.num_iter_geom_consistency = num_iter_geom_consistency;
        options.max_num_src_images = max_num_src_images;
        options.min_triangulation_angle = min_triangulation_angle;
        options.refine_with_semantic = refine_with_semantic;
        
        options.fuse_common_persent = fuse_common_persent;
        options.fuse_with_normal = with_normal;

        options.map_update = map_update;

        options.Print();

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

            PrintHeading1(StringPrintf("MVS Cluster Reconstruction %d ", reconstruction_idx));

            // mvs::MVSCluster::Options optionts_;
            mvs::MVSCluster mvs_cluster(options, JoinPaths(workspace_path, std::to_string(reconstruction_idx)));
            mvs_cluster.Start();
            mvs_cluster.Wait();

            GetAvailableMemory(end_memory);
            std::cout << StringPrintf("MVS Cluster Reconstruction %d Elapsed time: %.3f [minutes], \
                                        Memory: %3f (%3f - %3f) [G]", 
                                    reconstruction_idx, mvs_cluster.GetTimer().ElapsedMinutes(), 
                                    (begin_memroy - end_memory), begin_memroy, end_memory).c_str()
                    << std::endl;
        }
    }

    //Box Cluster
    {
    PrintHeading1(StringPrintf("Reconstruction Cluster"));

    mvs::ReconstructionCluster::Options options_;

    if (!map_update){
        options_.max_ram = max_ram;
    } else {
        options_.max_ram = -1.0f;
    }
    options_.ram_eff_factor = ram_eff_factor;

    options_.cluster_num = cluster_num;
    options_.format = format;
    options_.max_cell_size = max_cell_size;
    options_.min_common_view = min_common_view;
    options_.max_filter_percent = max_filter_percent;
    options_.dist_threshold = dist_threshold;
    options_.outlier_spacing_factor = outlier_spacing_factor;

    options_.max_image_size = max_image_size;
    options_.max_num_images_factor = max_num_images_factor * delaunay_dist_insert / 5.0f;
    
    if (options_.max_ram > 64.0) {
        options_.max_num_images_factor *= 1.5;
    } else if (options_.max_ram > 32.0) {
        options_.max_num_images_factor *= 1.2;
    }

    mvs::ReconstructionCluster reconstruction_cluster(options_, workspace_path);
    reconstruction_cluster.Start();
    reconstruction_cluster.Wait();
    }

	std::cout << StringPrintf("Reconstruction Cluster Elapsed time: %.3f [minutes]", 
							  cluster_timer.ElapsedMinutes()).c_str()
					<< std::endl;

    return 0;
}