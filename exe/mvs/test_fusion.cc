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
#include "mvs/fusion.h"
#include "base/version.h"
#include "../Configurator_yaml.h"

std::string configuration_file_path;

using namespace sensemap;

void LoadSemanticLabels(const std::string filepath, std::vector<uint8_t>& label_ids) {
	std::ifstream file;
	file.open(filepath.c_str(), std::ofstream::in);

	label_ids.clear();

	std::string line;
    std::string item;
    while (std::getline(file, line)) {
		StringTrim(&line);
		if (line.empty()) {
			continue;
		}
        std::stringstream line_stream(line);
        while (!line_stream.eof()) {
            std::getline(line_stream, item, ' ');
			label_ids.push_back(std::stoi(item));
        }
	}
	file.close();
}

int main(int argc, char *argv[]) {
    using namespace sensemap;
	using namespace mvs;

	PrintHeading("SenseMap.  Copyright(c) 2019, SenseTime Group.");	
	PrintHeading(std::string("Version: ") + __VERSION__);
	Timer fusion_timer;
	fusion_timer.Start();

	configuration_file_path = std::string(argv[1]);
    Configurator param;
    param.Load(configuration_file_path.c_str());

	int reconstrction_idx = -1;
	int cluster_idx = -1;

	if (argc > 2) {
        reconstrction_idx = atoi(argv[2]);
    }

	if (argc > 3){
		cluster_idx = atoi(argv[3]);
	}

	std::string workspace_path = param.GetArgument("workspace_path", "");
	int max_traversal_depth = param.GetArgument("max_traversal_depth", 100);
	int min_num_visible_images = param.GetArgument("min_num_visible_images", 3);
	int min_num_pixels = param.GetArgument("min_num_pixels", 3);
	int step_size = param.GetArgument("step_size", 1);
	bool with_normal = param.GetArgument("with_normal", 1);
	float max_normal_error = param.GetArgument("max_normal_error", 10.0f);
	bool fit_ground = param.GetArgument("fit_ground", 0);
	bool cache_depth = param.GetArgument("cache_depth", 1);
	bool init_from_model = param.GetArgument("init_from_model", 0);
	bool outlier_removal = param.GetArgument("outlier_removal", 1);
	float outlier_deviation_factor = param.GetArgument("outlier_deviation_factor", 3.0f);
	int nb_neighbors = param.GetArgument("nb_neighbors", 6);
	float max_spacing_factor = param.GetArgument("max_spacing_factor", 6.0f);

	double dist_to_best_plane_model = param.GetArgument("dist_to_best_plane_model", 12.0f);
	double angle_diff_to_main_plane = param.GetArgument("angle_diff_to_main_plane", 3.0f);
	std::string black_list_file_path = param.GetArgument("black_list_file_path", "");
	
	float max_ram = param.GetArgument("max_ram", -1.0f);
	float ram_eff_factor = param.GetArgument("ram_eff_factor", 0.75f);
	float fuse_common_persent = param.GetArgument("fuse_common_persent", 0.06f);

	bool roi_fuse = param.GetArgument("roi_fuse", 1);
	float roi_box_width = param.GetArgument("roi_box_width", -1.f);
	float roi_box_factor = param.GetArgument("roi_box_factor", -1.f);

	float min_inlier_ratio_to_best_model = 
		param.GetArgument("min_inlier_ratio_to_best_model", 0.8f);
	bool geom_consistency = param.GetArgument("geom_consistency", 1);
	int num_iter_geom_consistency = param.GetArgument("num_iter_geom_consistency", 1);

	bool map_update = param.GetArgument("map_update", 0);
	bool remove_duplicate_pnts = param.GetArgument("remove_duplicate_pnts", 1);

	float dist_insert = param.GetArgument("dist_insert", 5.0f);
	float diff_depth = param.GetArgument("diff_depth", 0.01f);

	bool fused_delaunay_sample = param.GetArgument("fused_delaunay_sample", 1);

	std::string semantic_table_path = param.GetArgument("semantic_table_path", "");
    if (ExistsFile(semantic_table_path)) {
        LoadSemanticColorTable(semantic_table_path.c_str());
    }

	StereoFusion::Options options;
	options.max_image_size =
		static_cast<int>(param.GetArgument("max_image_size", -1));
	// options.max_reproj_error = 1.0f;
	// options.min_num_visible_images = 5;
	options.format = param.GetArgument("image_type", "perspective");
	options.max_traversal_depth = max_traversal_depth;
	options.min_num_visible_images = min_num_visible_images;
	options.min_num_pixels = min_num_pixels;
	options.with_normal = with_normal;
	options.max_normal_error = max_normal_error;
	options.step_size = step_size;

	options.num_consistent_semantic_ratio = 0.1f;
	options.fit_ground = fit_ground;
	options.cache_depth = cache_depth;
	options.outlier_removal = outlier_removal;
	options.outlier_deviation_factor = outlier_deviation_factor;
	options.nb_neighbors = nb_neighbors;
	options.max_spacing_factor = max_spacing_factor;

	options.min_inlier_ratio_to_best_model = min_inlier_ratio_to_best_model;
	options.dist_to_best_plane_model = dist_to_best_plane_model;
	options.angle_diff_to_main_plane = angle_diff_to_main_plane;

	float get_max_ram;
	if (GetAvailableMemory(get_max_ram) && max_ram < 0){
		options.max_ram = get_max_ram;
	} else {
		options.max_ram = max_ram;
	}
	options.ram_eff_factor = ram_eff_factor;
	options.fuse_common_persent = fuse_common_persent;
	options.roi_fuse = roi_fuse;
	options.roi_box_width = roi_box_width * 2;
	options.roi_box_factor = roi_box_factor * 2;
	options.remove_duplicate_pnts = remove_duplicate_pnts;

	options.dist_insert = dist_insert;
	options.diff_depth = diff_depth;

	options.fused_delaunay_sample = fused_delaunay_sample;
	options.map_update = map_update;

	std::vector<uint8_t> label_ids;
	if (ExistsFile(black_list_file_path)) {
		LoadSemanticLabels(black_list_file_path, label_ids);
	}

	std::string ori_workspace_path = workspace_path;
	if (options.map_update) {
		workspace_path = JoinPaths(workspace_path, "map_update");
	}

	std::string input_type = geom_consistency ? GEOMETRIC_TYPE : PHOTOMETRIC_TYPE;
	
	size_t num_reconstruction = 0;
	for (size_t reconstruction_idx = 0; ;reconstruction_idx++) {
		const auto& reconstruction_path = 
			JoinPaths(workspace_path, std::to_string(reconstruction_idx));
		if (!ExistsDir(reconstruction_path)) {
			break;
		}
		if (map_update){
			const auto& ori_reconstruction_path = 
				JoinPaths(ori_workspace_path, std::to_string(reconstruction_idx));
			const auto& ori_fused_path = JoinPaths(ori_reconstruction_path, DENSE_DIR, FUSION_NAME);
			const auto& new_fused_path = JoinPaths(reconstruction_path, DENSE_DIR, FUSION_NAME);
			CHECK(ExistsFile(ori_fused_path) && ExistsFile(ori_fused_path + ".vis"));
			boost::filesystem::copy_file(ori_fused_path,
											JoinPaths(reconstruction_path, DENSE_DIR, FUSION_NAME), 
											boost::filesystem::copy_option::overwrite_if_exists);
			boost::filesystem::copy_file(ori_fused_path + ".vis", new_fused_path + ".vis" , 
											boost::filesystem::copy_option::overwrite_if_exists);
			if (ExistsFile(ori_fused_path + ".wgt")){
				boost::filesystem::copy_file(ori_fused_path + ".wgt", new_fused_path + ".wgt", 
											boost::filesystem::copy_option::overwrite_if_exists);
			}
			if (ExistsFile(ori_fused_path + ".sco")){
				boost::filesystem::copy_file(ori_fused_path + ".sco", new_fused_path + ".sco", 
											boost::filesystem::copy_option::overwrite_if_exists);
			}
		}
		num_reconstruction++;
	}

	size_t reconstruction_begin = reconstrction_idx < 0 ? 0 : reconstrction_idx;
	num_reconstruction = reconstrction_idx < 0 ? num_reconstruction : reconstrction_idx+1;
	for (size_t reconstruction_idx = reconstruction_begin; reconstruction_idx < num_reconstruction; 
		reconstruction_idx++) {
		float begin_memroy, end_memory;
		GetAvailableMemory(begin_memroy);

		StereoFusion fuser(options, workspace_path, input_type, reconstruction_idx, cluster_idx);
		fuser.AddBlackList(label_ids);
		fuser.Start();
		fuser.Wait();

		GetAvailableMemory(end_memory);
		std::cout << StringPrintf("Fusion Reconstruction %d Elapsed time: %.3f [minutes], \
									Memory: %3f (%3f - %3f) [G]", 
								reconstruction_idx, fuser.GetTimer().ElapsedMinutes(), 
								(begin_memroy - end_memory), begin_memroy, end_memory).c_str()
				<< std::endl;
	}
	
	std::cout << StringPrintf("Fusion Elapsed time: %.3f [minutes]", 
							  fusion_timer.ElapsedMinutes()).c_str()
					<< std::endl;

	return 0;
}
