#include <boost/filesystem/path.hpp>
#include "base/reconstruction_manager.h"
#include "util/misc.h"
#include <string>
int main(int argc, char* argv[]) {
    using namespace sensemap;

    CHECK(argc == 7);
    PrintHeading("SenseMap.  Copyright(c) 2019, SenseTime Group.");
    std::string workspace_path = std::string(argv[1]);
    double max_reproj_eror = std::stod(argv[2]);
    int min_track_length = std::stoi(argv[3]);
    double min_tri_angle = std::stod(argv[4]);
    double max_distance_to_plane = std::stod(argv[5]);
    double max_plane_count = std::stoi(argv[6]);

    std::string reconstruction_dir = workspace_path+"/0";
    CHECK(boost::filesystem::is_directory(workspace_path + "/0"));
    auto reconstruction = std::make_shared<Reconstruction>();
    reconstruction->ReadReconstruction(reconstruction_dir);    

    std::cout<<"filter params: "<<min_track_length<<" "<<max_reproj_eror<<" "<<min_tri_angle<<std::endl;

    reconstruction->FilterAllMapPoints(min_track_length,max_reproj_eror,min_tri_angle);
    reconstruction->FilterImages(0.5,1.5,2.0);

    reconstruction->FilterAllFarawayImages();

    std::string reconstruction_dir_filtered = workspace_path+"/0_filtered";
    if (boost::filesystem::exists(reconstruction_dir_filtered)) {
        boost::filesystem::remove_all(reconstruction_dir_filtered);
    }
    boost::filesystem::create_directories(reconstruction_dir_filtered);

    reconstruction->WriteReconstruction(reconstruction_dir_filtered);



}