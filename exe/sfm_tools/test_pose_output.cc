#include <boost/filesystem/path.hpp>
#include "base/reconstruction_manager.h"
#include "util/misc.h"
#include <string>
#include <fstream>

int main(int argc, char* argv[]) {
    using namespace sensemap;

    CHECK(argc == 3);
    PrintHeading("SenseMap.  Copyright(c) 2019, SenseTime Group.");
    std::string workspace_path = std::string(argv[1]);
    std::string pose_path = std::string(argv[2]);

    std::ofstream file_pose(pose_path.c_str());

    std::string reconstruction_dir = workspace_path+"/0";
    CHECK(boost::filesystem::is_directory(workspace_path + "/0"));
    auto reconstruction = std::make_shared<Reconstruction>();
    reconstruction->ReadReconstruction(reconstruction_dir);    


    image_t image_first = reconstruction->RegisterImageIds()[0];
    const Image& first_image = reconstruction->Image(image_first);
    const Camera& camera = reconstruction->Camera(first_image.CameraId());
    file_pose<<"# camera intrinsics: f, cx, cy, k1, k2"<<std::endl;
    file_pose<<camera.ParamsToString()<<std::endl;

    file_pose <<"# camera pose: image name, qw, qx, qy, qz, tx, ty, tz"<<std::endl;
    for(image_t image_id: reconstruction->RegisterImageIds()){
        const Image& image = reconstruction->Image(image_id);
        
        Eigen::Matrix3x4d proj_matrix = image.ProjectionMatrix();
        file_pose<<image.Name()<<" "<<image.Qvec()[0]<<" "<<image.Qvec()[1]<<" "<<image.Qvec()[2]<<" "<<image.Qvec()[3]<<" "
                 <<image.Tvec()[0]<<" "<<image.Tvec()[1]<<" "<<image.Tvec()[2]<<" "<<std::endl;
    }
    file_pose.close();
}