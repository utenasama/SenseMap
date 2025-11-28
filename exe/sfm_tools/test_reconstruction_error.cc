// Copyright (c) 2020, SenseTime Group.
// All rights reserved.


#include "base/reconstruction_manager.h"
#include "../option_parsing.h"
#include "util/gps_reader.h"
#include "util/ply.h"

using namespace sensemap;

int main(int argc, char* argv[]) {
    std::string config_file_path = argv[1];

    Configurator param;
    param.Load(config_file_path.c_str());

    OptionParser option_parser;

    std::string workspace_path = param.GetArgument("workspace_path", "");
    CHECK(!workspace_path.empty()) << "workspace path empty";

    std::string reconstruction_dir = workspace_path + "/0";
    CHECK(boost::filesystem::is_directory(workspace_path + "/0"));
    auto reconstruction = std::make_shared<Reconstruction>();
    reconstruction->ReadReconstruction(reconstruction_dir);


    std::vector<std::string> image_names;
    for(image_t image_id: reconstruction->RegisterImageIds()){
        const Image& image = reconstruction->Image(image_id);
        image_names.push_back(image.Name());
    }

    std::string gps_prior_file = param.GetArgument("gps_prior_file","");
    std::vector<std::pair<unsigned long, std::pair<Eigen::Vector3d,int>> > gps_locations;
    std::string gps_trans_file = workspace_path + "/gps_trans.txt";
    LoadOriginGPSinfo(gps_prior_file, gps_locations,gps_trans_file);
    std::unordered_map<std::string, std::pair<Eigen::Vector3d,int>> image_locations;
    GPSLocationsToImages(gps_locations, image_names, image_locations);
    std::cout << image_locations.size() << " images have gps prior" << std::endl;
    
    std::unordered_map<image_t, std::pair<Eigen::Vector3d,int>> prior_locations_gps;


    std::vector<PlyPoint> gps_locations_ply;

    for (const auto image_id : reconstruction->RegisterImageIds()) {
        const Image& image = reconstruction->Image(image_id);
        std::string name = image.Name();

        if (image_locations.find(name) != image_locations.end()) {
            prior_locations_gps.emplace(image_id, image_locations.at(name));

            PlyPoint gps_location_ply;
            gps_location_ply.r = 255;
            gps_location_ply.g = 0;
            gps_location_ply.b = 0;
            gps_location_ply.x = image_locations.at(name).first[0];
            gps_location_ply.y = image_locations.at(name).first[1];
            gps_location_ply.z = image_locations.at(name).first[2];
            gps_locations_ply.push_back(gps_location_ply);
        }
    }

    sensemap::WriteBinaryPlyPoints(workspace_path+"/gps-test-error.ply", gps_locations_ply);

    reconstruction->prior_locations_gps = prior_locations_gps;

    double max_error_gps = static_cast<double>(param.GetArgument("max_error_gps",3.0f));
    reconstruction->AlignWithPriorLocations(max_error_gps);

    Reconstruction rec = *reconstruction.get();
    rec.AddPriorToResult();




    std::string trans_rec_path = reconstruction_dir + "-gps";
    if (!boost::filesystem::exists(trans_rec_path)) {
        boost::filesystem::create_directories(trans_rec_path);
    }
    rec.WriteBinary(trans_rec_path);
}
