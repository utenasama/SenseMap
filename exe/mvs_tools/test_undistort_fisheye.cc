#include "util/misc.h"
#include "util/exception_handler.h"
#include "base/undistortion.h"
#include "base/reconstruction_manager.h"
#include "controllers/patch_match_controller.h"

#include "../Configurator_yaml.h"
#include "../camera_rig_params.h"
#include "../../src/base/undistortion.h"
#include "../../src/base/camera_models.h"
using namespace sensemap;
using namespace sensemap::mvs;

void Undistort(const UndistortOptions options_, const std::string image_path_, Camera camera,std::string undistort_images_path_) {
    boost::filesystem::path image_path(image_path_);
    boost::filesystem::directory_iterator endIter;
    Camera undistorted_camera;
    for (boost::filesystem::directory_iterator iter(image_path); iter != endIter; iter++) {
        boost::filesystem::path filepath(iter->path().string());
        std::string image_name = filepath.filename().string();
        std::cout<<image_name<<std::endl;
        // printf("%s\n",image_name.c_str());
        const std::string output_image_path =
            JoinPaths(undistort_images_path_, image_name);

        Bitmap distorted_bitmap;
        const std::string input_image_path = filepath.string();
        if (!distorted_bitmap.Read(input_image_path)) {
            std::cerr << "ERROR: Cannot read image at path" << input_image_path 
                    << std::endl;
            return;
        }
        camera.SetWidth(distorted_bitmap.Width());
        camera.SetHeight(distorted_bitmap.Height());

        Bitmap undistorted_bitmap;
        Undistorter::UndistortImage(options_, distorted_bitmap, camera, &undistorted_bitmap, 
                    &undistorted_camera);

        std::string parent_path = GetParentDir(output_image_path);
        if (!ExistsPath(parent_path)) {
            boost::filesystem::create_directories(parent_path);
        }
        undistorted_bitmap.Write(output_image_path);
    }
    std::cout<<undistorted_camera.ParamsToString()<<std::endl;
}

std::string configuration_file_path;
int main(int argc, char *argv[]) {
	configuration_file_path = std::string(argv[1]);
    Configurator param;
    param.Load(configuration_file_path.c_str());
	
	std::string image_path = param.GetArgument("image_path", "");
	std::string workspace_path = param.GetArgument("workspace_path", "");
    std::cout << "workspace_path: " << workspace_path << std::endl;
    int max_image_size = param.GetArgument("max_image_size",-1);
    std::cout<<"max_image_size: "<<max_image_size<<std::endl;

	float fov_w = param.GetArgument("fov_w", -1.0f);
	float fov_h = param.GetArgument("fov_h", -1.0f);
    std::cout<<"fov_w: "<<fov_w<<std::endl;
    std::cout<<"fov_h: "<<fov_h<<std::endl;
    UndistortOptions options;
    options.max_image_size = max_image_size;
    options.fov_w = fov_w;
    options.fov_h = fov_h;
    std::string calib_cam = param.GetArgument("calib_cam","");
    int calib_cam_index = std::atoi(calib_cam.substr(calib_cam.size()-1,1).c_str());
    std::cout<<"calib_cam_index: "<<calib_cam_index<<std::endl;
    std::string camera_rig_params_file = param.GetArgument("rig_params_file", "");

    sensemap::CameraRigParams rig_params;
    if (!camera_rig_params_file.empty()) {
        if (rig_params.LoadParams(camera_rig_params_file)) {
            std::cout << "num local cameras: " << rig_params.num_local_cameras << std::endl;
            std::cout << "camera model: " << rig_params.camera_model << std::endl;
        } else {
            std::cout << "failed to read rig params" << std::endl;
		    return StateCode::INVALID_INPUT_PARAM;
        }
    }
    auto calib_cam_intrinsic = rig_params.local_intrinsics[calib_cam_index];
    std::cout<<"calib_cam:";
    for(int i = 0;i<calib_cam_intrinsic.size();i++)
    {
        std::cout<<calib_cam_intrinsic[i]<<",";
    }
    std::cout<<std::endl;
    Camera camera;
    camera.SetCameraId(1);
    camera.SetModelIdFromName(rig_params.camera_model);
    camera.SetParams(calib_cam_intrinsic);
    std::string undistorted_path = workspace_path+"/undistorted_images/";
    Undistort(options, image_path,camera,undistorted_path);
}
