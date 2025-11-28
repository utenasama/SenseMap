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
#include "util/threading.h"

#include "base/common.h"
#include "base/image.h"
#include "base/reconstruction.h"
#include "base/reconstruction_manager.h"

#include "mvs/depth_map.h"
#include "mvs/normal_map.h"

#include "boost/filesystem.hpp"

#include "../Configurator_yaml.h"
#include "base/version.h"

using namespace sensemap;

std::string configuration_file_path;

void ComputeNormal(const Camera& camera, const mvs::DepthMap& depth_map,
                   mvs::NormalMap& normal_map) {
    const int width = depth_map.GetWidth();
    const int height = depth_map.GetHeight();

    std::vector<Eigen::Vector3d> points(width * height);
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            const float depth = depth_map.Get(i, j);
            Eigen::Vector2d x = camera.ImageToWorld(Eigen::Vector2d(j, i));
            Eigen::Vector3d Xc(x(0) * depth, x(1) * depth, depth);
            points[i * width + j] = Xc;
        }
    }

    std::vector<Eigen::Vector3f> normals(width * height);
    for (int i = 1; i < height - 1; ++i) {
        for (int j = 1; j < width - 1; ++j) {
            Eigen::Vector3d& Xt = points[(i - 1) * width + j];
            Eigen::Vector3d& Xb = points[(i + 1) * width + j];
            Eigen::Vector3d& Xl = points[i * width + j - 1];
            Eigen::Vector3d& Xr = points[i * width + j + 1];
            normals[i * width + j] = 
                (Xb - Xt).cross(Xr - Xl).normalized().cast<float>();
        }
    }

    normal_map = mvs::NormalMap(width, height);
    for (int i = 1; i < height - 1; ++i) {
        for (int j = 1; j < width - 1; ++j) {
            Eigen::Vector3f normal(0, 0, 0);
            normal += normals[(i - 1) * width + j]; // top
            normal += normals[(i + 1) * width + j]; // bottom
            normal += normals[i * width + j - 1]; // left
            normal += normals[i * width + j + 1]; // right
            normal.normalize();
            normal_map.SetSlice(i, j, normal.data());
        }
    }
    for (int j = 0; j < width; ++j) {
        Eigen::Vector3f normal0(0, 0, 0);
        Eigen::Vector3f normal1(0, 0, 0);
        if (j != 0) {
            normal0 = normals[width + j] + normals[j - 1];
            normal0.normalize();
            normal1 = normals[(height - 2) * width + j] + 
                      normals[(height - 1) * width + j - 1];
            normal1.normalize();
        }
        normal_map.SetSlice(0, j, normal0.data());
        normal_map.SetSlice(height - 1, j, normal1.data());
    }
    for (int i = 0; i < height; ++i) {
        Eigen::Vector3f normal0(0, 0, 0);
        Eigen::Vector3f normal1(0, 0, 0);
        if (i != 0) {
            normal0 = normals[(i - 1) * width] + normals[i * width + 1];
            normal0.normalize();
            normal1 = normals[(i - 1) * width + width - 1] + 
                      normals[i * width + width - 2];
            normal1.normalize();
        }
        normal_map.SetSlice(i, 0, normal0.data());
        normal_map.SetSlice(i, width - 1, normal1.data());
    }
}

int main(int argc, char *argv[]) {

    PrintHeading("SenseMap.  Copyright(c) 2019, SenseTime Group.");	
	PrintHeading(std::string("Version: ") + __VERSION__);

	configuration_file_path = std::string(argv[1]);
    Configurator param;
    param.Load(configuration_file_path.c_str());

	std::string workspace_path = param.GetArgument("workspace_path", "");
    std::string image_path = param.GetArgument("image_path", "");
    bool use_rgbd_mvs = param.GetArgument("image_type", "") == "rgbd";
    int max_image_size = param.GetArgument("max_image_size", -1);
    std::string output_type = PHOTOMETRIC_TYPE;

    std::string rgbd_params_file = param.GetArgument("rgbd_params_file", "");
    std::string rgbd_camera_params;
    if (!rgbd_params_file.empty()) {
        auto calib_reader = GetCalibBinReaderFromName(rgbd_params_file);
        calib_reader->ReadCalib(rgbd_params_file);
        rgbd_camera_params = calib_reader->ToParamString();
        std::cout << "rgbd_camera_params: " << rgbd_camera_params << std::endl;
    }

    std::vector<std::string> filter_subpath = CSVToVector<std::string>(param.GetArgument("filter_subpath", ""));

    ReconstructionManager manager;
    int num_reconstruction = manager.Read(workspace_path);
    for (int reconstruction_idx = 0; reconstruction_idx < num_reconstruction;
        ++reconstruction_idx) {

        std::string component_path = 
            JoinPaths(workspace_path, std::to_string(reconstruction_idx));
        
        std::string dense_path = JoinPaths(component_path, DENSE_DIR);
        std::string sparse_path = JoinPaths(dense_path, SPARSE_DIR);
        std::string stereo_path = JoinPaths(dense_path, STEREO_DIR);
        std::string dense_image_path = JoinPaths(dense_path, IMAGES_DIR);
        std::string depth_maps_path = JoinPaths(stereo_path, DEPTHS_DIR);
        std::string normal_maps_path = JoinPaths(stereo_path, NORMALS_DIR);
        std::string consistency_path = JoinPaths(stereo_path, CONSISTENCY_DIR);

        CreateDirIfNotExists(dense_path);
        CreateDirIfNotExists(sparse_path);
        CreateDirIfNotExists(stereo_path);
        CreateDirIfNotExists(dense_image_path);
        CreateDirIfNotExists(depth_maps_path);
        CreateDirIfNotExists(normal_maps_path);
        CreateDirIfNotExists(consistency_path);

        std::shared_ptr<Reconstruction> reconstruction = 
            manager.Get(reconstruction_idx);

        if (!filter_subpath.empty()) {
            std::unordered_set<image_t> images_to_del;
            for (const auto &image : reconstruction->Images()) {
                for (const auto &subpath : filter_subpath) {
                    if (IsInsideSubpath(image.second.Name(), subpath)) {
                        images_to_del.insert(image.first);
                        break;
                    }
                }
            }
            for (auto image_id : images_to_del) {
                reconstruction->DeleteImage(image_id);
            }
            std::cout << "Filter " << images_to_del.size() << " Images in certain subpaths " << std::endl;
        }

        int n = 1;
        EIGEN_STL_UMAP(camera_t, class Camera) scaled_cameras = 
            reconstruction->Cameras();
        std::unordered_set<camera_t> modified_camera_ids;

        std::vector<std::string> image_names;
        std::vector<image_t> images = reconstruction->RegisterImageIds();
        for (auto image_id : images) {
            const auto& image = reconstruction->Image(image_id);
            std::string image_name = JoinPaths(image_path, image.Name());
            std::string iimage_name = image.Name() + ".jpg";
            image_names.push_back(iimage_name);

            // create output directory if needed
            const std::string file_name = StringPrintf("%s.%s.%s", 
                iimage_name.c_str(), output_type.c_str(), DEPTH_EXT);
            std::string depth_map_path = 
                JoinPaths(depth_maps_path, file_name);
            std::string normal_map_path = 
                JoinPaths(normal_maps_path, file_name);
            std::string iimage_path = 
                JoinPaths(dense_image_path, iimage_name);
            boost::filesystem::path depth_map_dir = boost::filesystem::path(depth_map_path).parent_path();
            boost::filesystem::path normal_map_dir = boost::filesystem::path(normal_map_path).parent_path();
            boost::filesystem::path iimage_dir = boost::filesystem::path(iimage_path).parent_path();
            if (!boost::filesystem::exists(depth_map_dir)) {
                boost::filesystem::create_directories(depth_map_dir);
            }
            if (!boost::filesystem::exists(normal_map_dir)) {
                boost::filesystem::create_directories(normal_map_dir);
            }
            if (!boost::filesystem::exists(iimage_dir)) {
                boost::filesystem::create_directories(iimage_dir);
            }
        }

        const int num_eff_threads = sensemap::GetEffectiveNumThreads(-1);
        sensemap::ThreadPool thread_pool(num_eff_threads);
        for (size_t index = 0; index < images.size(); ++index) {

            std::mutex cmrs_lk;
            thread_pool.AddTask([&](size_t i) {
                auto image_id = images[i];
                const auto& image = reconstruction->Image(image_id);
                const auto& camera = reconstruction->Camera(image.CameraId());
                auto& scaled_camera = scaled_cameras.at(image.CameraId());

                std::string image_name = JoinPaths(image_path, image.Name());
                const std::string & iimage_name = image_names[i];
                std::string iimage_path = JoinPaths(dense_image_path, iimage_name);
                if (!IsFileRGBD(image_name)) return;

                const double max_image_scale_x = max_image_size /
                    static_cast<double>(camera.Width());
                const double max_image_scale_y = max_image_size /
                    static_cast<double>(camera.Height());
                const double max_image_scale =
                    max_image_size <= 0 ? 1.0 :
                    std::min(max_image_scale_x, max_image_scale_y);

                RGBDData data;
                ExtractRGBDData(image_name, data);
                data.ReadRGBDCameraParams(rgbd_camera_params);
                mvs::DepthMap depth_map_wrapper(data.depth, 0, MAX_VALID_DEPTH_IN_M);

                if (use_rgbd_mvs) {
                    const int width = camera.Width();
                    const int height = camera.Height();
                    const int warped_width = max_image_scale < 1.0 ? static_cast<size_t>(std::round(max_image_scale * width)) : width;
                    const int warped_height = max_image_scale < 1.0 ? static_cast<size_t>(std::round(max_image_scale * height)): height;
                    const float width_scale = warped_width * 1.0f / data.color.Width();
                    const float height_scale = warped_height * 1.0f / data.color.Height();

                    if (data.HasRGBDCalibration()) {
                        MatXf warped_depthmap(warped_width, warped_height, 1);
                        Camera color_camera = data.color_camera;
                        color_camera.Rescale(warped_width, warped_height);
                        UniversalWarpDepthMap(warped_depthmap, data.depth, color_camera, data.depth_camera, data.depth_RT.cast<float>());
                        depth_map_wrapper = mvs::DepthMap(warped_depthmap, 0, MAX_VALID_DEPTH_IN_M);
                    } else {
                        std::cerr << "RGBD calibration not present!" << std::endl;
                        std::abort();
                    }
                    
                    if (max_image_size != -1 && max_image_scale < 1.0) {
                        {
                            std::lock_guard<std::mutex> lk(cmrs_lk);
                            if (modified_camera_ids.find(image.CameraId()) ==
                                modified_camera_ids.end()) {
                                scaled_camera.Rescale(max_image_scale);
                                modified_camera_ids.insert(image.CameraId());
                            }
                        }
                        data.color.Rescale(scaled_camera.Width(), scaled_camera.Height());
                    }

                    mvs::NormalMap normal_map_wrapper;
                    ComputeNormal(scaled_camera, depth_map_wrapper, normal_map_wrapper);

                    const std::string file_name = StringPrintf("%s.%s.%s", 
                        iimage_name.c_str(), output_type.c_str(), DEPTH_EXT);
                    std::string depth_map_path = 
                        JoinPaths(depth_maps_path, file_name);
                    std::string normal_map_path = 
                        JoinPaths(normal_maps_path, file_name);
                    depth_map_wrapper.Write(depth_map_path);
                    normal_map_wrapper.Write(normal_map_path);
                } else {
                    if (max_image_size != -1 && max_image_scale < 1.0) {
                        {
                            std::lock_guard<std::mutex> lk(cmrs_lk);
                            if (modified_camera_ids.find(image.CameraId()) ==
                                modified_camera_ids.end()) {
                                scaled_camera.Rescale(max_image_scale);
                                modified_camera_ids.insert(image.CameraId());
                            }
                        }

                        data.color.Rescale(scaled_camera.Width(), scaled_camera.Height());
                    }

                    std::string * name_ptr = const_cast<std::string *>(&image.Name());
                    *name_ptr = iimage_name;
                }

                data.color.Write(iimage_path);
                std::cout << StringPrintf("Convert Image %d/%d\n", i, reconstruction->NumRegisterImages()) << std::flush;
            }, index); 
        }
        thread_pool.Wait();
        std::cout << std::endl;

        for (auto image_id : reconstruction->RegisterImageIds()) {
            auto& image = reconstruction->Image(image_id);            
            auto& scaled_camera = scaled_cameras.at(image.CameraId());
            auto& unscaled_camera = reconstruction->Camera(image.CameraId());
            for (point2D_t point2D_idx = 0; point2D_idx < image.NumPoints2D(); 
                ++point2D_idx) {
                auto& point2D = image.Point2D(point2D_idx);
                point2D.SetXY(scaled_camera.WorldToImage(
                    unscaled_camera.ImageToWorld(point2D.XY())));
            }
        }

        for (auto image_id : reconstruction->RegisterImageIds()) {
            auto& image = reconstruction->Image(image_id);            
            auto& scaled_camera = scaled_cameras.at(image.CameraId());
            reconstruction->Camera(image.CameraId()) = scaled_camera;
        }

        reconstruction->WriteBinary(sparse_path);

        // for visualization with colmap gui.
        std::ofstream ofs(stereo_path + "/patch-match.cfg", 
                        std::ofstream::out);
        for (const auto & image_name : image_names) {
            ofs << image_name << std::endl;
            ofs << "__auto__, " << 8 << std::endl;
        }
        ofs.close();
    }

    return 0;
}
