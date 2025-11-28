//Copyright (c) 2019, SenseTime Group.
//All rights reserved.

#include <boost/filesystem/path.hpp>

#include "util/misc.h"
#include "util/ply.h"
#include "util/obj.h"
#include "mvs/mesh_cluster.h"
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
    
    int do_mesh_cluster =
        static_cast<int>(param.GetArgument("do_mesh_cluster", 1));

    mvs::MeshCluster::Options options;
    options.max_faces_per_cluster =
        static_cast<int>(param.GetArgument("max_faces_per_cluster", 200000));
    options.min_faces_per_cluster =
        static_cast<int>(param.GetArgument("min_faces_per_cluster", 100000));
    options.cell_size_factor = static_cast<int>(param.GetArgument("cell_size_factor", 50.0f));
    options.cell_size = static_cast<float>(param.GetArgument("cell_size", 6.0f));

    if (do_mesh_cluster) {
        mvs::MeshCluster cluster(options, workspace_path);
        cluster.Start();
        cluster.Wait();
	}

	return 0;
}