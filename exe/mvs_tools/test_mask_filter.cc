
//Copyright (c) 2020, SenseTime Group.
//All rights reserved.

#include <iostream>
#include <string>
#include <vector>
#include <Eigen/Dense>

#include "util/types.h"
#include "util/bitmap.h"
#include "util/mat.h"
#include "util/misc.h"
#include "util/rgbd_helper.h"
#include "util/imageconvert.h"
#include "util/threading.h"

#include "base/common.h"
#include "base/image.h"
#include "base/undistortion.h"
#include "base/reconstruction.h"
#include "base/reconstruction_manager.h"

#include "mvs/depth_map.h"
#include "mvs/normal_map.h"

#include "../Configurator_yaml.h"

using namespace sensemap;

int main(int argc, char *argv[]) {

	std::string configuration_file_path = std::string(argv[1]);
    Configurator param;
    param.Load(configuration_file_path.c_str());

	std::string workspace_path = param.GetArgument("workspace_path", "");
    std::string image_path = param.GetArgument("image_path", "");
	bool geom_consistency = param.GetArgument("geom_consistency", 1);
	std::string input_type = geom_consistency ? GEOMETRIC_TYPE : PHOTOMETRIC_TYPE;

    for (int reconstruction_idx = 0; ;
        ++reconstruction_idx) {

        std::string component_path = 
            JoinPaths(workspace_path, std::to_string(reconstruction_idx));

        if (!boost::filesystem::exists(component_path)) break;
        
        std::string dense_path = JoinPaths(component_path, DENSE_DIR);
        std::string stereo_path = JoinPaths(dense_path, STEREO_DIR);
        std::string sparse_path = JoinPaths(dense_path, SPARSE_DIR);
        std::string mask_path = JoinPaths(dense_path, MASKS_DIR);
        std::string depths_path = JoinPaths(stereo_path, DEPTHS_DIR);
        std::string normals_path = JoinPaths(stereo_path, NORMALS_DIR);

        if (!ExistsDir(dense_path)) break;
        if (!ExistsDir(stereo_path)) break;
        if (!ExistsDir(sparse_path)) break;
        if (!ExistsDir(depths_path)) break;
        if (!ExistsDir(normals_path)) break;

        std::shared_ptr<Reconstruction> reconstruction(new Reconstruction());
        reconstruction->ReadBinary(sparse_path);

        std::vector<std::string> image_names;
        std::vector<image_t> images = reconstruction->RegisterImageIds();
        for (auto image_id : images) {
            const auto& image = reconstruction->Image(image_id);
            std::string image_name = JoinPaths(image_path, image.Name());
            image_names.push_back(image_name);
        }

        const int num_eff_threads = sensemap::GetEffectiveNumThreads(-1);
        sensemap::ThreadPool thread_pool(num_eff_threads);
        for (size_t index = 0; index < images.size(); ++index)
        thread_pool.AddTask([&](size_t i) {
            auto & image = reconstruction->Image(images[i]);
            Camera camera = reconstruction->Camera(image.CameraId());

            cv::Mat mask = cv::imread(JoinPaths(mask_path, image.Name()) + "." + MASK_EXT, cv::IMREAD_GRAYSCALE);
            if (mask.empty()) return;
            CHECK_EQ(camera.Width(), mask.cols);
            CHECK_EQ(camera.Height(), mask.rows);

            const int element_size = std::max(1, (int)((mask.cols + mask.rows) * 0.0025));
            cv::Mat element = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2 * element_size + 1, 2 * element_size + 1));
            cv::Mat object_mask = mask > 0;
            cv::dilate(object_mask, object_mask, element);

            std::string depth_file_name = StringPrintf("%s.%s.%s", 
                image.Name().c_str(), input_type.c_str(), DEPTH_EXT);
            std::string normal_file_name = StringPrintf("%s.%s.%s", 
                image.Name().c_str(), input_type.c_str(), NORMAL_EXT);
            std::string depth_map_path =
                JoinPaths(depths_path, depth_file_name);
            std::string normal_map_path =
                JoinPaths(normals_path, normal_file_name);

            if (!ExistsFile(depth_map_path) || !ExistsFile(normal_map_path)) {
                depth_file_name = StringPrintf("%s.jpg.%s.%s", 
                    image.Name().c_str(), input_type.c_str(), DEPTH_EXT);
                normal_file_name = StringPrintf("%s.jpg.%s.%s", 
                    image.Name().c_str(), input_type.c_str(), NORMAL_EXT);
                depth_map_path =
                    JoinPaths(depths_path, depth_file_name);
                normal_map_path =
                    JoinPaths(normals_path, normal_file_name);
            }
            if (!ExistsFile(depth_map_path) || !ExistsFile(normal_map_path)) {
                std::cout << "Depth/normal map for " << image.Name() << " not found" << std::endl;
                return;
            }

            mvs::DepthMap depth_map;
            mvs::NormalMap normal_map;
            depth_map.Read(depth_map_path);
            normal_map.Read(normal_map_path);

            if (!depth_map.IsValid() || !normal_map.IsValid()) {
                std::cout << "Depth/normal map for " << image.Name() << " invalid" << std::endl;
                return;
            }

            for (int y = 0; y < depth_map.GetHeight(); y++) {
                for (int x = 0; x < depth_map.GetWidth(); x++) {
                    if (!object_mask.at<uchar>(y, x)) {
                        depth_map.Set(y, x, 0.0f);
                        normal_map.Set(y, x, 0.0f);
                    }
                }
            }

            depth_map.Write(depth_map_path);
            normal_map.Write(normal_map_path);

            std::cout << "\rMask filter " << image.Name() << std::flush;
        }, index); 
        thread_pool.Wait();
        std::cout << std::endl;
    }

    return 0;
}