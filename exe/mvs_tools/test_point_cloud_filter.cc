//Copyright (c) 2019, SenseTime Group.
//All rights reserved.

#include <iostream>
#include <fstream>
#include <sstream>

#include "util/misc.h"
#include "util/string.h"
#include "base/common.h"
#include "base/reconstruction_manager.h"
#include "base/undistortion.h"
#include "controllers/patch_match_controller.h"
#include "mvs/point_cloud_filter.h"
#include "../Configurator_yaml.h"
#include "base/version.h"

std::string configuration_file_path;

using namespace sensemap;

int main(int argc, char *argv[]) {
    using namespace sensemap;
	using namespace mvs;

	PrintHeading("SenseMap.  Copyright(c) 2019, SenseTime Group.");	
	PrintHeading(std::string("Version: ") + __VERSION__);

	configuration_file_path = std::string(argv[1]);
    Configurator param;
    param.Load(configuration_file_path.c_str());

	std::string workspace_path = param.GetArgument("workspace_path", "");

	PointCloudFilter::Options filter_options;
	filter_options.format = param.GetArgument("image_type", "perspective");
	filter_options.nb_neighbors = param.GetArgument("nb_neighbors", 6);
	filter_options.max_spacing_factor = param.GetArgument("max_spacing_factor", 6.0f);
	filter_options.method = (PointCloudFilter::FilterMethod)param.GetArgument("filter_method", 0);
	filter_options.min_grad_thres = param.GetArgument("min_grad_thres", 1.2f);
	filter_options.conf_thres = param.GetArgument("ncc_score", 0.6f);
	filter_options.win_size = param.GetArgument("win_size", 3);
	filter_options.trust_region = param.GetArgument("trust_region", 7);

	PointCloudFilter filter(filter_options, workspace_path);
	filter.Start();
	filter.Wait();

	return 0;
}
