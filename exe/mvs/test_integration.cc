//Copyright (c) 2019, SenseTime Group.
//All rights reserved.

#include <iostream>
#include <fstream>
#include <sstream>

#include "util/misc.h"
#include "util/string.h"
#include "base/common.h"
#include "base/reconstruction_manager.h"
#include "mvs/integration.h"
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

	configuration_file_path = std::string(argv[1]);
    Configurator param;
    param.Load(configuration_file_path.c_str());

	std::string workspace_path = param.GetArgument("workspace_path", "");
	bool cache_depth = param.GetArgument("cache_depth", 1);
	bool init_from_model = param.GetArgument("init_from_model", 0);
	std::string black_list_file_path = param.GetArgument("black_list_file_path", "");

	PrintHeading("SenseMap.  Copyright(c) 2019, SenseTime Group.");	
	
	StereoIntegration::Options integration_options;
	integration_options.max_image_size =
		static_cast<int>(param.GetArgument("max_image_size", -1));
	integration_options.format = param.GetArgument("image_type", "perspective");

	integration_options.cache_depth = cache_depth;

	bool geom_consistency = param.GetArgument("geom_consistency", 1);
	integration_options.voxel_length = static_cast<float>(param.GetArgument("voxel_length", 0.03f));
	integration_options.sdf_trunc_precision = static_cast<float>(param.GetArgument("sdf_trunc_precision", 0.15f));
	integration_options.extract_mesh_threshold = static_cast<float>(param.GetArgument("integration_mesh_threshold", 1.0f));
	integration_options.min_depth = static_cast<float>(param.GetArgument("integration_min_depth", 0.2f));
	integration_options.max_depth = static_cast<float>(param.GetArgument("integration_max_depth", 10.0f));
	integration_options.tof_weight = static_cast<float>(param.GetArgument("integration_tof_weight", 0.0f));
	integration_options.do_filter = static_cast<bool>(param.GetArgument("integration_filter", true));
	integration_options.noise_filter_depth_thresh = static_cast<float>(param.GetArgument("integration_noise_filter_depth_thresh", 0.1f));
	integration_options.noise_filter_count_thresh = static_cast<int>(param.GetArgument("integration_noise_filter_count_thresh", 15));
	integration_options.connectivity_filter_thresh = static_cast<float>(param.GetArgument("integration_connectivity_filter_thresh", 0.05f));
	integration_options.do_joint_filter = static_cast<bool>(param.GetArgument("integration_joint_filter", false));
	integration_options.joint_filter_completion_thresh = static_cast<float>(param.GetArgument("integration_depth_completion_thresh", -1.0f));
	integration_options.num_isolated_pieces = static_cast<float>(param.GetArgument("num_isolated_pieces", 0));
	integration_options.roi_fuse = param.GetArgument("roi_fuse", 1);

	std::cout << "voxel_length: " << integration_options.voxel_length << std::endl;
	std::cout << "sdf_trunc_precision: " << integration_options.sdf_trunc_precision << std::endl;
	std::cout << "mesh_threshold: " << integration_options.extract_mesh_threshold << std::endl;
	std::cout << "min_depth: " << integration_options.min_depth << std::endl;
	std::cout << "max_depth: " << integration_options.max_depth << std::endl;
	std::cout << "tof_weight: " << integration_options.tof_weight << std::endl;
	std::cout << "do_filter: " << integration_options.do_filter << std::endl;
	std::cout << "noise_filter_depth_thresh: " << integration_options.noise_filter_depth_thresh << std::endl;
	std::cout << "noise_filter_count_thresh: " << integration_options.noise_filter_count_thresh << std::endl;
	std::cout << "connectivity_filter_thresh: " << integration_options.connectivity_filter_thresh << std::endl;
	std::cout << "do_joint_filter: " << integration_options.do_joint_filter << std::endl;
	std::cout << "depth_completion_thresh: " << integration_options.joint_filter_completion_thresh << std::endl;
	std::cout << "num_isolated_pieces: " << integration_options.num_isolated_pieces << std::endl;
	std::cout << "roi_fuse: " << integration_options.roi_fuse << std::endl;
	std::cout << std::endl;

	std::vector<uint8_t> label_ids;
	if (ExistsFile(black_list_file_path)) {
		LoadSemanticLabels(black_list_file_path, label_ids);
	}

	std::string input_type = geom_consistency ? GEOMETRIC_TYPE : PHOTOMETRIC_TYPE;
	StereoIntegration integrator(integration_options, workspace_path, input_type);
	integrator.AddBlackList(label_ids);
	integrator.Start();
	integrator.Wait();

	return 0;
}
