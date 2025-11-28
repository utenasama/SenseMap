//Copyright (c) 2019, SenseTime Group.
//All rights reserved.

#include <boost/filesystem/path.hpp>

#include "util/misc.h"
#include "util/ply.h"
#include "util/obj.h"
#include "mvs/point_cloud_cluster.h"
#include "mvs/workspace.h"

#include "base/version.h"
#include "../Configurator_yaml.h"

#include <iostream>
#include <dirent.h>
#include <sys/stat.h>

using namespace sensemap;

std::string configuration_file_path;

int main(int argc, char *argv[]) {

	PrintHeading("SenseMap.  Copyright(c) 2019, SenseTime Group.");	
	PrintHeading(std::string("Version: ") + __VERSION__);

	configuration_file_path = std::string(argv[1]);
    Configurator param;
    param.Load(configuration_file_path.c_str());

	std::string workspace_path = param.GetArgument("workspace_path", "");
    
    int do_point_cloud_cluster =
        static_cast<int>(param.GetArgument("do_point_cloud_cluster", 1));

    mvs::PointCloudCluster::Options options;
    options.max_pts_per_cluster =
        static_cast<int>(param.GetArgument("max_pts_per_cluster", 1000000));
    options.min_pts_per_cluster =
        static_cast<int>(param.GetArgument("min_pts_per_cluster", 500000));
    options.cell_size_factor = static_cast<int>(param.GetArgument("cell_size_factor", 100.0f));
    options.cell_size = static_cast<float>(param.GetArgument("cell_size", 6.0f));

    if (do_point_cloud_cluster) {
        mvs::PointCloudCluster cluster(options, workspace_path);
        cluster.Start();
        cluster.Wait();
	}

	return 0;
}