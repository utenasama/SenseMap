// Copyright (c) 2019, SenseTime Group.
// All rights reserved.

#include "util/gps_reader.h"
#include "../Configurator_yaml.h"
#include "util/ply.h"

#include <dirent.h>
#include <sys/stat.h>
#include <boost/filesystem/path.hpp>

using namespace sensemap;
int main(int argc, char *argv[]) {

    std::string configuration_file_path = std::string(argv[1]);
    Configurator param;
    param.Load(configuration_file_path.c_str());

    std::string workspace_path = param.GetArgument("workspace_path", "");
    
    std::string gps_prior_file = param.GetArgument("gps_prior_file", "");
    std::string gps_trans_file = workspace_path + "/gps_trans.txt";

    //if (boost::filesystem::exists(gps_prior_file)) {
        std::vector<std::pair<unsigned long, std::pair<Eigen::Vector3d, int>>> gps_locations;

        LoadOriginGPSinfo(gps_prior_file, gps_locations, gps_trans_file, false, 3);

        std::vector<PlyPoint> gps_locations_ply;
        for (const auto gps_location : gps_locations) {
            PlyPoint gps_location_ply;
            gps_location_ply.r = 255;
            gps_location_ply.g = 0;
            gps_location_ply.b = 0;
            gps_location_ply.x = gps_location.second.first[0];
            gps_location_ply.y = gps_location.second.first[1];
            gps_location_ply.z = gps_location.second.first[2];
            gps_locations_ply.push_back(gps_location_ply);
        }
      

        sensemap::WriteBinaryPlyPoints(workspace_path + "/gps.ply", gps_locations_ply, false, true);
    //}
}