// Copyright (c) 2019, SenseTime Group.
// All rights reserved.

#include <boost/filesystem/path.hpp>
#include <opencv2/opencv.hpp>
// #include <unordered_map>

#include "base/common.h"
#include "base/camera.h"
#include "base/image.h"
#include "base/image_reader.h"
#include "base/point2d.h"
#include "base/pose.h"
#include "base/camera_models.h"
#include "base/undistortion.h"
#include "base/reconstruction_manager.h"
#include "base/projection.h"
#include "graph/correspondence_graph.h"
#include "util/misc.h"
#include "util/threading.h"
#include "util/timer.h"
#include "util/cross_warp_helper.h"
#include "util/exception_handler.h"

#include "container/feature_data_container.h"
#include "feature/extraction.h"
#include "feature/feature_matcher.h"
#include "feature/utils.h"

#include "controllers/cluster_mapper_controller.h"

#include "base/version.h"
#include "../Configurator_yaml.h"

std::string configuration_file_path;

using namespace sensemap;

namespace {
// BKDR Hash Function
unsigned int BKDRHash(const char *str)
{
	unsigned int seed = 131; // 31 131 1313 13131 131313 etc..
	unsigned int hash = 0;

	while (*str)
	{
		hash = hash * seed + (*str++);
	}

	return (hash & 0x7FFFFFFF);
}

Eigen::Matrix3d EulerToRotationMatrix(double roll, double pitch, double yaw) {
    Eigen::Quaterniond q = Eigen::AngleAxisd(yaw / 180 * M_PI, Eigen::Vector3d::UnitY()) *
                           Eigen::AngleAxisd(pitch / 180 * M_PI, Eigen::Vector3d::UnitX()) *
                           Eigen::AngleAxisd(roll / 180 * M_PI, Eigen::Vector3d::UnitZ());

    return q.matrix();
}

void ApplyColorHarmonization(Bitmap *bitmap, YCrCbFactor yrb_factor) {
    const int width = bitmap->Width();
    const int height = bitmap->Height();
// #pragma omp parallel for schedule(dynamic)
    for (int r = 0; r < height; ++r) {
        for (int c = 0; c < width; ++c) {
            BitmapColor<uint8_t> color = bitmap->GetPixel(c, r);

            float V[3];
            V[0] = color.r * INV_COLOR_NORM;
            V[1] = color.g * INV_COLOR_NORM;
            V[2] = color.b * INV_COLOR_NORM;
            ColorRGBToYCbCr(V);
            V[0] = yrb_factor.s_Y * V[0] + yrb_factor.o_Y;
            V[1] = yrb_factor.s_Cb * V[1] + yrb_factor.o_Cb;
            V[2] = yrb_factor.s_Cr * V[2] + yrb_factor.o_Cr;

            ColorYCbCrToRGB(V);
            color.r = std::min(1.0f, std::max(0.0f, V[0])) * 255;
            color.g = std::min(1.0f, std::max(0.0f, V[1])) * 255;
            color.b = std::min(1.0f, std::max(0.0f, V[2])) * 255;

            bitmap->SetPixel(c, r, color);
        }
    }
}

void WriteImageMapText(const std::string& path,
    const std::unordered_map<image_t, image_t>& ids_map){
    std::ofstream file(path, std::ios::trunc);
    CHECK(file.is_open()) << path;

    file << "# ID list with one line of data per Image:" << std::endl;
    file << "#   Sparse_Image_ID, Reconstruciton_Image_ID" << std::endl;
    file << "# Number of Images: " << ids_map.size() << std::endl;

    for (const auto& id : ids_map){
        std::ostringstream line;
        line << id.first << " " << id.second << " ";

        std::string line_string = line.str();
        line_string = line_string.substr(0, line_string.size() - 1);

        file << line_string << std::endl;
    }

    file.close();
}

}

void InitDepthPriors(
    const std::shared_ptr<Reconstruction> reconstruction_, 
    const UndistortOptions & options_, 
    const std::string & image_path_,
    const std::string &  workspace_path_,
    const std::unordered_set<image_t> & filter_image_ids_)
{
    auto image_ids = reconstruction_->RegisterImageSortIds();
    std::map<image_t, std::shared_ptr<PriorDepthInfo>> prior_depths = 
        Undistorter::InitPriorDepths(image_path_, *reconstruction_, options_);
    if (prior_depths.size() == 0) return;

    if (options_.reverse_scale_recovery) {
        double scale = Undistorter::ReverseScaleRecovery(prior_depths, *reconstruction_);
        std::ofstream(JoinPaths(workspace_path_, "0", DENSE_DIR, "scale.txt")) << scale;
    }

    std::unordered_set<image_t> prior_depth_images;
    for (auto & item : prior_depths) {
        prior_depth_images.insert(item.first);
    }
    std::shared_ptr<CrossWarpHelper> cross_warp_helper;
    if (!options_.cross_warp_subpath.empty()) {
        cross_warp_helper = std::make_shared<PatchMatchCrossWarpHelper>(JoinPaths(workspace_path_, "0", DENSE_DIR, SPARSE_DIR), prior_depth_images, options_.cross_warp_num_images);
    }
    
    std::cout << "Init depth priors" << std::endl;
    const std::string undistort_depths_path = JoinPaths(workspace_path_, "0", DENSE_DIR, DEPTHS_DIR);
    const int num_eff_threads = sensemap::GetEffectiveNumThreads(-1);
    sensemap::ThreadPool thread_pool(num_eff_threads);
    std::atomic<int> progress(0);
    for (size_t index = 0; index < image_ids.size(); ++index)
    thread_pool.AddTask([&](size_t i) {
        int value = progress.fetch_add(1, std::memory_order::memory_order_relaxed);
        if (value % 100 == 1) {
            std::cout << value << "/" << image_ids.size() << "\r" << std::flush;
        }
        const image_t image_id = image_ids[i];
        const auto & image = reconstruction_->Image(image_id);
        const auto & undistorted_camera = reconstruction_->Camera(image.CameraId());
        if (filter_image_ids_.count(image_id)) return;

        if (prior_depths.count(image_id) != 0) {
            Undistorter::WarpPriorDepth(image_id, undistorted_camera, *reconstruction_, prior_depths, undistort_depths_path, options_);
        }
        if (prior_depths.count(image_id) == 0 && !options_.cross_warp_subpath.empty()) {
            Undistorter::CrossWarpPriorDepth(image_id, undistorted_camera, *reconstruction_, *cross_warp_helper, prior_depths, undistort_depths_path, options_);
        }
    }, index);
    std::cout << std::endl;
    thread_pool.Wait();
}
#define MAX_IMAGE_SIZE_PANORAMA 960

bool ConvertPanoramaConfig(const std::string &workspace_path,
                           const std::string &img_input_path,
                           const std::string &mask_input_path,
                           const std::string &output_workspace_path,
                           const int divide_camera_num,
                           const UndistortOptions& options,
                           const bool keyframe_mode = false,
                           const bool tex_use_orig_res = false,
                           const bool map_update = false) {
    double fov_w = options.fov_w;
    double fov_h = options.fov_h;
    int perspective_width = options.max_image_size;
    int insv_perspective_width = options.max_rig_image_size;

    if (perspective_width == 0) {
        // return false;
        std::cerr << "ERROR! invalid image dimension" << std::endl;
        ExceptionHandler(StateCode::INVALID_IMAGE_DIMENSION, 
            JoinPaths(GetParentDir(workspace_path), "errors/dense"), "DenseConvertPanorama").Dump();
        exit(StateCode::INVALID_IMAGE_DIMENSION);
    }

    std::cout << "workspace_path: " << workspace_path << std::endl;
    std::cout << "img_input_path: " << img_input_path << std::endl;
    std::cout << "mask_input_path: " << mask_input_path << std::endl;
    std::cout << "output_workspace_path: " << output_workspace_path << std::endl;
    std::cout << "divide_camera_num: " << divide_camera_num << std::endl;
    std::cout << "fov_w: " << fov_w << std::endl;
    std::cout << "fov_h: " << fov_h << std::endl;
    std::cout << "perspective_width: " << perspective_width << std::endl;
    std::cout << "insv_perspective_width: " << insv_perspective_width << std::endl;

    // Load reconstruction
    for (int rect_id = 0; ; rect_id++){
        auto reconstruction_path = JoinPaths(workspace_path, std::to_string(rect_id));
        if (keyframe_mode) {
            reconstruction_path = JoinPaths(reconstruction_path, "KeyFrames");
        }
        if (!ExistsDir(reconstruction_path)){
            break;
        }

        PrintHeading1(StringPrintf("Processing component %d", rect_id));

        auto reconstruction = std::make_shared<Reconstruction>();
        reconstruction->ReadReconstruction(reconstruction_path);


        std::unordered_map<image_t, image_t> map_convert_id;
        std::unordered_map<image_t, image_t> map_rig_new2ori;
        
        auto image_ids = reconstruction->RegisterImageSortIds();
        if (map_update && boost::filesystem::exists(JoinPaths(reconstruction_path, "update_images.txt"))) {
            std::cout << "map update! mark old map" << std::endl;
            image_t old_image = 0;
            std::vector<image_t> update_image_ids =
            reconstruction->ReadUpdateImagesText(JoinPaths(reconstruction_path, "update_images.txt"));
            std::unordered_set<image_t> unique_update_image_ids;
            unique_update_image_ids.insert(update_image_ids.begin(), update_image_ids.end());
            for (auto image_id : image_ids) {
                if (unique_update_image_ids.find(image_id) == unique_update_image_ids.end()) {
                    auto& image = reconstruction->Image(image_id);
                    image.SetLabelId(0);
                    old_image++;
                }
            }
            std::cout << StringPrintf("Set old images %d from updated map\n", old_image);
        }
        if (image_ids.empty()) {
            continue;
        }

        Reconstruction reconstruction_tmp = *reconstruction.get();

        Timer hash_timer;
        hash_timer.Start();
        std::unordered_map<camera_t, uint32_t> camera_hash_map;
        const auto & all_cameras = reconstruction->Cameras();
        for (const auto camera : all_cameras) {
            if (camera.second.ModelName().compare("SPHERICAL") == 0) {
                std::string img_size_str = std::to_string(camera.second.Width()) + "#" + 
                                           std::to_string(camera.second.Height()) + "#" +
                                           std::to_string(camera.second.MeanFocalLength());
                uint32_t hash_code = BKDRHash(img_size_str.c_str());
                camera_hash_map[camera.first] = hash_code;
            } else if (camera.second.NumLocalCameras() == 2) {
                std::vector<double> local_camera_param0;
                camera.second.GetLocalCameraIntrisic(0, local_camera_param0);
                std::string img_size_str = std::to_string(camera.second.Width()) + "#" + 
                                           std::to_string(camera.second.Height()) + "#" +
                                           std::to_string(local_camera_param0[0]);
                uint32_t hash_code = BKDRHash(img_size_str.c_str());
                camera_hash_map[camera.first] = hash_code;
            }
        }
        std::cout << "Generate Hash Code Cost: " << hash_timer.ElapsedSeconds() << "s" << std::endl;

        std::vector<image_t> perspective_image_ids;
        perspective_image_ids.reserve(image_ids.size());
        std::unordered_map<uint32_t, std::vector<image_t> > spherical_image_ids;
        std::unordered_map<uint32_t, std::vector<image_t> > insv_image_ids;
        std::vector<image_t> rig_image_ids;
        rig_image_ids.reserve(image_ids.size());
        size_t num_image_sphere = 0, num_image_insv = 0;
        for (auto image_id : image_ids) {
            const auto& image = reconstruction->Image(image_id);
            const auto& camera = reconstruction->Camera(image.CameraId());
            if (camera.ModelName().compare("SPHERICAL") == 0) {
                uint32_t hash_code = camera_hash_map.at(camera.CameraId());
                spherical_image_ids[hash_code].push_back(image_id);
                num_image_sphere++;
            } else if (camera.NumLocalCameras() == 2) {
                uint32_t hash_code = camera_hash_map.at(camera.CameraId());
                insv_image_ids[hash_code].push_back(image_id);
                num_image_insv++;
            } else if (camera.NumLocalCameras() == 1) {
                perspective_image_ids.push_back(image_id);
            } else if (camera.NumLocalCameras() > 2) {
                rig_image_ids.push_back(image_id);
            }
            if (camera.NumLocalCameras() < 3) {
                reconstruction_tmp.DeleteImage(image_id);
            } else {
                reconstruction->DeleteImage(image_id);
            }
        }
        std::vector<sweep_t> lidar_sweep_ids = reconstruction->RegisterSweepSortIds();

        reconstruction->TearDown();
        reconstruction_tmp.TearDown();

        //only use options.max_rig_image_size when perspective_img & INSV_img exist(RGBD+4k) 
        int max_rig_image_size = options.max_rig_image_size;
        if (!((perspective_image_ids.size() + rig_image_ids.size()) > 0 && num_image_insv > 0)) {
            max_rig_image_size = options.max_image_size;
        }

        std::cout << "Spherical Camera: " << num_image_sphere << std::endl;
        std::cout << "INSV Camera: " << num_image_insv << std::endl;
        std::cout << "Rig Camera: " << rig_image_ids.size() << std::endl;
        std::cout << "Perspective Camera: " << perspective_image_ids.size() << std::endl;

        // Create all Feature container
        auto feature_data_container = std::make_shared<FeatureDataContainer>();
        // Load original feature
        if (!spherical_image_ids.empty() || !insv_image_ids.empty() || !rig_image_ids.empty()) {
            if (boost::filesystem::exists(JoinPaths(workspace_path, "/local_cameras.bin"))) {
                feature_data_container->ReadCamerasBinaryData(JoinPaths(workspace_path, "/cameras.bin"), false);
                feature_data_container->ReadLocalCamerasBinaryData(JoinPaths(workspace_path, "/local_cameras.bin"));
            } else {
                feature_data_container->ReadCamerasBinaryData(JoinPaths(workspace_path, "/cameras.bin"), !insv_image_ids.empty());
            }
            feature_data_container->ReadImagesBinaryData(workspace_path + "/features.bin");
            // Load Panorama feature
            feature_data_container->ReadSubPanoramaBinaryData(workspace_path + "/sub_panorama.bin");
            if (boost::filesystem::exists(JoinPaths(workspace_path, "/piece_indices.bin"))) {
                feature_data_container->ReadPieceIndicesBinaryData(JoinPaths(workspace_path, "/piece_indices.bin"));
            } else if (boost::filesystem::exists(JoinPaths(workspace_path, "/piece_indices.txt"))) {
                feature_data_container->ReadPieceIndicesData(JoinPaths(workspace_path, "/piece_indices.txt"));
            }
        }

        if (!reconstruction_tmp.RegisterImageSortIds().empty()) {
            std::cout << "Convert Rig cameras to Normal" << std::endl;
            const auto image_sort_ids = reconstruction_tmp.RegisterImageSortIds();
            for (auto image_id : image_sort_ids) {
                class Image& cur_image = reconstruction_tmp.Image(image_id);
                class Camera& cur_camera = reconstruction_tmp.Camera(cur_image.CameraId());

                const FeatureKeypoints &keypoints = feature_data_container->GetKeypoints(image_id);
                const PanoramaIndexs & panorama_indices = feature_data_container->GetPanoramaIndexs(image_id);

                std::vector<uint32_t> local_image_indices(keypoints.size(), 0);
                for(size_t i = 0; i < keypoints.size(); ++i){
                    local_image_indices[i] = panorama_indices[i].sub_image_id;
                }

                cur_image.SetLocalImageIndices(local_image_indices);
            }
            std::cout << "image: " << reconstruction->RegisterImageSortIds().size() << std::endl;

            Reconstruction rig_reconstruction;
            std::unordered_map<image_t, std::vector<image_t>> map_rig_convert_id 
                = reconstruction_tmp.ConvertRigReconstruction(rig_reconstruction);

            std::cout << "Append Rig to Normal Reconstruction" << std::endl;
            std::unordered_map<image_t, image_t> map_append = reconstruction->Append(rig_reconstruction);

            {
                for (auto& rig_conv : map_rig_convert_id){
                    for (int idx = 0; idx < rig_conv.second.size(); idx++){
                        image_t old_id = rig_conv.second.at(idx);
                        // rig_conv.second.at(idx) = map_append[old_id];
                        map_rig_new2ori[map_append[old_id]] = rig_conv.first;
                    }
                }
            }
            
            std::cout << "image: " << reconstruction->RegisterImageSortIds().size() << std::endl;
        }
        {
            const auto& append_ids = reconstruction->RegisterImageSortIds();
            for (const auto id : append_ids){
                if (map_rig_new2ori.find(id) == map_rig_new2ori.end()){
                    map_rig_new2ori[id] = id;
                }
            }
        }

        image_ids = reconstruction->RegisterImageSortIds();

        // Convert from panorama image
        std::unordered_map<uint32_t, Panorama> panoramas;
        // Convert from large fov image and split in piecewise
        std::unordered_map<uint32_t, PiecewiseImage> piecewise_images;

        std::unordered_map<uint32_t, std::vector<std::vector<int> > > pano_rmap_idxs;
        std::unordered_map<uint32_t, std::vector<Eigen::RowMatrixXi> > rig_rmap_idxs;
        std::unordered_map<uint32_t, std::vector<double> > rmap_coordinates_x;
        std::unordered_map<uint32_t, std::vector<double> > rmap_coordinates_y;
        // Initialize the image converter
        if (!spherical_image_ids.empty()) {
            std::cout << "Initializing Panorama converter" << std::endl;
            Timer timer;
            timer.Start();
            for (auto & spherical_map : spherical_image_ids) {
                const uint32_t hash_code = spherical_map.first;
                const auto& image = reconstruction->Image(spherical_map.second[0]);
                const auto& camera = reconstruction->Camera(image.CameraId());
                int perspective_width = max_rig_image_size < 0 ? MAX_IMAGE_SIZE_PANORAMA : max_rig_image_size;
                auto & panorama = panoramas[hash_code];
                panorama.PerspectiveParamsProcess(perspective_width, fov_w, fov_h, 
                                                  divide_camera_num, camera.Width(), camera.Height());
                pano_rmap_idxs[hash_code].emplace_back(std::move(panorama.GetPanoramaRmapId()));
                rmap_coordinates_x[hash_code] = std::move(panorama.GetPanoramaRMapX());
                rmap_coordinates_y[hash_code] = std::move(panorama.GetPanoramaRMapY());
            }
            timer.PrintSeconds();
        }
        if(!insv_image_ids.empty()) {
            std::cout << "Initializing CameraRig converter" << std::endl;
            Timer timer;
            timer.Start();
            for (auto & rig_map : insv_image_ids) {
                const uint32_t hash_code = rig_map.first;
                const auto& image = reconstruction->Image(rig_map.second[0]);
                const auto& camera = reconstruction->Camera(image.CameraId());
                auto & piecewise_image = piecewise_images[hash_code];
                piecewise_image.SetCamera(camera);
                int perspective_width = max_rig_image_size < 0 ? MAX_IMAGE_SIZE_PANORAMA : max_rig_image_size;
                piecewise_image.ParamPreprocess(perspective_width, fov_w, fov_h, 
                                                camera.Width(), camera.Height(), divide_camera_num / 2);
                rig_rmap_idxs[hash_code] = std::move(piecewise_image.GetPiecewiseRmapId());
                rmap_coordinates_x[hash_code] = std::move(piecewise_image.GetPiecewiseRMapX());
                rmap_coordinates_y[hash_code] = std::move(piecewise_image.GetPiecewiseRMapX());
            }
            timer.PrintSeconds();
        }

        // Load rotation_matrixs
        Eigen::Matrix3d pano_rotation_matrixs[divide_camera_num];
        Eigen::Matrix3d rig_rotation_matrixs[divide_camera_num];
        if (!spherical_image_ids.empty()) {
            const double disturbation_roll = 0;
            const double disturbation_pitch = 0;
            const double disturbation_yaw = 0;

            double yaw_interval = 360.0 / static_cast<double>(divide_camera_num);

            for (size_t i = 0; i < divide_camera_num; ++i) {
                double roll, pitch, yaw;
                roll = disturbation_roll;
                yaw = disturbation_pitch + i * yaw_interval;
                pitch = disturbation_pitch;
                pano_rotation_matrixs[i] = EulerToRotationMatrix(roll, pitch, yaw);
                pano_rotation_matrixs[i].transposeInPlace();
            }
        }
        if(!insv_image_ids.empty()){ // panorama camera in rig

            double roll[5] =  {0,0,0,0,0};
            double yaw[5]  =  {-60,0,60,0,0};
            double pitch[5] = {0,0,0,-60,60};
        
            // get piecewise transforms    
            for (size_t i = 0; i < divide_camera_num / 2 ; ++i) {
                Eigen::Matrix3d transform;
                transform = Eigen::AngleAxisd(yaw[i] / 180 * M_PI, Eigen::Vector3d::UnitY()) *
                        Eigen::AngleAxisd(pitch[i] / 180 * M_PI, Eigen::Vector3d::UnitX()) *
                        Eigen::AngleAxisd(roll[i] / 180 * M_PI, Eigen::Vector3d::UnitZ());
                rig_rotation_matrixs[i] = transform.transpose();
            }
        }

        camera_t new_camera_id = 0;
        Camera *new_camera = nullptr;
        std::unordered_map<camera_t, camera_t> perspective_camera_id_map;

        auto update_reconstruction = std::make_shared<Reconstruction>();

        Timer timer1;
        timer1.Start();
        ///////////////////////////////////////////////////////////////////////////////////////
        // 1. Update Image pose for all new images
        ///////////////////////////////////////////////////////////////////////////////////////
        std::cout << "1. Update Image pose for all new images ..." << std::endl;
        image_t new_image_count = 0;
        std::unordered_map<image_t, std::unordered_map<point2D_t, std::pair<image_t, point2D_t> > > image_point2d_map;
        for (const auto& image_id : image_ids) {
            const auto cur_image = reconstruction->Image(image_id);
            const auto cur_camera = reconstruction->Camera(cur_image.CameraId());
            const auto camera_id = cur_camera.CameraId();
            const auto& old_point2ds = cur_image.Points2D();

            const int width = cur_camera.Width();
            const int height = cur_camera.Height();

            const int local_camera_num = cur_camera.NumLocalCameras();
            if (local_camera_num == 2 || cur_camera.ModelName().compare("SPHERICAL") == 0) {
                int perspective_width = max_rig_image_size < 0 ? MAX_IMAGE_SIZE_PANORAMA : max_rig_image_size;
                if (!new_camera) {
                    double focal_length = perspective_width * 0.5 / tan(fov_w / 360.0 * M_PI);
                    const int perspective_height = focal_length * tan(fov_h / 360.0 * M_PI) * 2;

                    new_camera = new Camera();
                    new_camera->SetCameraId(++new_camera_id);
                    new_camera->SetModelId(PinholeCameraModel::model_id);
                    new_camera->SetWidth(perspective_width);
                    new_camera->SetHeight(perspective_height);
                    new_camera->SetFocalLengthX(focal_length);
                    new_camera->SetFocalLengthY(focal_length);
                    new_camera->SetPrincipalPointX(perspective_width * 0.5);
                    new_camera->SetPrincipalPointY(perspective_height * 0.5);
                    new_camera->SetFromRIG(true);
                    update_reconstruction->AddCamera(*new_camera);
                }

                const uint32_t hash_code = camera_hash_map.at(camera_id);
                std::vector<Eigen::RowMatrixXi>* rig_rmap_idx;
                std::vector<std::vector<int> >* pano_rmap_idx;
                std::vector<double>* remap_x;
                std::vector<double>* remap_y;
                if (local_camera_num == 2) {
                    rig_rmap_idx = &rig_rmap_idxs.at(hash_code);
                } else {
                    pano_rmap_idx = &pano_rmap_idxs.at(hash_code);
                }
                remap_x = &rmap_coordinates_x.at(hash_code);
                remap_y = &rmap_coordinates_y.at(hash_code);

                const FeatureKeypoints &keypoints = feature_data_container->GetKeypoints(image_id);
                const PanoramaIndexs & panorama_indices = feature_data_container->GetPanoramaIndexs(image_id);

                std::vector<uint32_t> local_image_indices(keypoints.size(), 0);
                for(size_t i = 0; i < keypoints.size(); ++i){
                    if (local_camera_num == 1) {
                        local_image_indices[i] = 0;
                    } else {
                        local_image_indices[i] = panorama_indices[i].sub_image_id;
                    }
                }

                std::vector<std::vector<class Point2D> > new_point2Ds(divide_camera_num);
                for (size_t point_id = 0; point_id < old_point2ds.size(); point_id++) {
                    size_t local_image_id = local_image_indices.at(point_id);
                    auto point2D = old_point2ds.at(point_id);
                    const int ori_v = point2D.Y();
                    const int ori_u = point2D.X();
                    if (ori_u < 0 || ori_u >= width || ori_v < 0 || ori_v >= height) {
                        continue;
                    }

                    int new_image_id;
                    if (local_camera_num == 2) {
                        new_image_id = rig_rmap_idx->at(local_image_id)(ori_v, ori_u);
                    } else {
                        new_image_id = pano_rmap_idx->at(local_image_id)[ori_v * width + ori_u];
                    }
                    if (new_image_id == -1) {
                        continue;
                    }

                    size_t remap_idx = local_image_id * width * height + ori_v * width + ori_u;
                    double u = (*remap_x)[remap_idx];
                    double v = (*remap_y)[remap_idx];
                    point2D.SetXY(Eigen::Vector2d(u, v));

                    image_point2d_map[image_id][point_id] = std::move(std::make_pair(new_image_count + new_image_id, new_point2Ds[new_image_id].size()));
                    
                    point2D.SetMapPointId(kInvalidMapPointId);
                    new_point2Ds[new_image_id].emplace_back(std::move(point2D));
                }

                std::vector<Image> new_images;
                new_images.reserve(divide_camera_num);
                for (size_t perspective_image_id = 0; perspective_image_id < divide_camera_num; perspective_image_id++) {
                    Image new_image;
                    // Update image id
                    new_image.SetImageId(new_image_count++);
                    new_image.SetCameraId(new_camera->CameraId());

                    // Update the camera rotation
                    auto old_tvec = cur_image.Tvec();
                    auto old_rot = cur_image.RotationMatrix();

                    Eigen::Matrix3d new_rot;
                    Eigen::Vector3d new_tvec;
                    
                    if(local_camera_num == 1){
                        new_rot = pano_rotation_matrixs[perspective_image_id] * old_rot;  // FIXME:
                        new_tvec = pano_rotation_matrixs[perspective_image_id] * old_tvec;
                    }
                    else if(local_camera_num > 1){
                        int local_camera_id = (local_camera_num == 2) ? (perspective_image_id / (divide_camera_num / 2) ) : perspective_image_id;
                        int piece_id = (local_camera_num == 2) ? (perspective_image_id % (divide_camera_num / 2)) : 0;

                        Eigen::Vector4d local_qvec;
                        Eigen::Vector3d local_tvec;

                        cur_camera.GetLocalCameraExtrinsic(local_camera_id, local_qvec, local_tvec);

                        const Eigen::Matrix3d local_camera_R =
                        QuaternionToRotationMatrix(local_qvec) * old_rot;

                        const Eigen::Vector3d local_camera_T =
                        QuaternionToRotationMatrix(local_qvec) * old_tvec + local_tvec;

                        if(local_camera_num == 2){
                            new_rot = rig_rotation_matrixs[piece_id] * local_camera_R;
                            new_tvec = rig_rotation_matrixs[piece_id] * local_camera_T; 
                        }
                        else{
                            new_rot = local_camera_R;
                            new_tvec = local_camera_T;
                        }               
                    }

                    auto new_qvec = RotationMatrixToQuaternion(new_rot);
                    new_image.SetQvec(new_qvec);
                    new_image.SetTvec(new_tvec);

                    std::string new_image_name = StringPrintf(
                        "%s_%d.jpg", cur_image.Name().substr(0, cur_image.Name().size() - 4).c_str(), perspective_image_id);
                    new_image.SetName(new_image_name);
                    new_image.SetPoints2D(new_point2Ds[perspective_image_id]);
                    new_image.SetLabelId(cur_image.LabelId());

                    // Update reconstruction
                    update_reconstruction->AddImage(new_image);
                    update_reconstruction->RegisterImage(new_image.ImageId());
                    if (!reconstruction->yrb_factors.empty()) {
                        update_reconstruction->yrb_factors.push_back(reconstruction->yrb_factors.at(image_id));
                    }

                    map_convert_id[new_image.ImageId()] = map_rig_new2ori[image_id];
                }
            } else {
                Camera undistorted_camera;
                Image undistorted_image;
                undistorted_image.SetImageId(new_image_count);
                undistorted_image.SetName(cur_image.Name());
                undistorted_image.SetQvec(cur_image.Qvec());
                undistorted_image.SetTvec(cur_image.Tvec());
                if (perspective_camera_id_map.count(camera_id) == 0) {
                    perspective_camera_id_map[camera_id] = ++new_camera_id;
                    Undistorter::UndistortCamera(options, cur_camera, &undistorted_camera);
                    undistorted_camera.SetCameraId(new_camera_id);
                    undistorted_image.SetCameraId(new_camera_id);
                    update_reconstruction->AddCamera(undistorted_camera);
                } else {
                    undistorted_camera = update_reconstruction->Camera(perspective_camera_id_map.at(camera_id));
                    undistorted_image.SetCameraId(undistorted_camera.CameraId());
                }

                std::vector<class Point2D> new_point2Ds;
                for (size_t point_id = 0; point_id < old_point2ds.size(); point_id++) {
                    auto point2D = old_point2ds.at(point_id);
                    image_point2d_map[image_id][point_id] = std::make_pair(new_image_count, new_point2Ds.size());
                    
                    point2D.SetXY(undistorted_camera.WorldToImage(
                    cur_camera.ImageToWorld(point2D.XY())));

                    point2D.SetMapPointId(kInvalidMapPointId);
                    new_point2Ds.emplace_back(std::move(point2D));
                }
                undistorted_image.SetPoints2D(new_point2Ds);
                undistorted_image.SetLabelId(cur_image.LabelId());
                // Update reconstruction
                update_reconstruction->AddImage(undistorted_image);
                update_reconstruction->RegisterImage(undistorted_image.ImageId());
                if (!reconstruction->yrb_factors.empty()) {
                    update_reconstruction->yrb_factors.push_back(reconstruction->yrb_factors.at(image_id));
                }
                new_image_count++;

                map_convert_id[undistorted_image.ImageId()] = map_rig_new2ori[image_id];
            }
        }

        size_t new_sweep_count = 0;
        size_t regist_sweep_count = 0;
        for (size_t idx = 0; idx < lidar_sweep_ids.size(); idx++){
            const auto cur_sweep = reconstruction->LidarSweep(lidar_sweep_ids.at(idx));

            LidarSweep new_sweep(new_sweep_count, cur_sweep.Name());
            new_sweep.SetQvec(cur_sweep.Qvec());
            new_sweep.SetTvec(cur_sweep.Tvec());
            new_sweep.SetRegistered(cur_sweep.IsRegistered());
            if (cur_sweep.IsRegistered()){
                regist_sweep_count++;
            }
            update_reconstruction->AddLidarSweep(new_sweep);
            new_sweep_count++;
        }
        update_reconstruction->lidar_to_cam_matrix = reconstruction->lidar_to_cam_matrix;

        timer1.PrintSeconds();

        Timer timer2;
        timer2.Start();
        ///////////////////////////////////////////////////////////////////////////////////////
        // 2. Update 3d point track id
        ///////////////////////////////////////////////////////////////////////////////////////
        std::cout << "2. Update 3d point track id ... " << std::endl;
        const auto& mappoint_ids = reconstruction->MapPointIds();
        for (const auto& mappoint_id : mappoint_ids) {
            class MapPoint new_mappoint;
            // Get old mappoint
            const auto old_mappoint = reconstruction->MapPoint(mappoint_id);

            // Use the old mappoint position
            new_mappoint.SetXYZ(old_mappoint.XYZ());
            new_mappoint.SetColor(old_mappoint.Color());
            new_mappoint.SetError(old_mappoint.Error());

            // Update the old mappoint track with new image id and point2d id
            class Track new_track;
            for (const auto& track_el : old_mappoint.Track().Elements()) {
                if (image_point2d_map[track_el.image_id].count(track_el.point2D_idx) == 0) {
                    continue;
                }
                auto new_image_to_point = image_point2d_map[track_el.image_id][track_el.point2D_idx];
                
                new_track.AddElement(new_image_to_point.first, new_image_to_point.second);
            }
            // Check track size
            if(new_track.Length() <= 2){
                continue;
            }

            new_mappoint.SetTrack(new_track);

            // FIXME:  Error not written
            // Update reconstruction
            update_reconstruction->AddMapPointWithError(new_mappoint.XYZ(), 
                std::move(new_mappoint.Track()), new_mappoint.Color(), new_mappoint.Error());
        }
        timer2.PrintSeconds();

        auto dense_path = JoinPaths(output_workspace_path, std::to_string(rect_id), DENSE_DIR);
        boost::filesystem::create_directories(dense_path);
        auto sparse_path = JoinPaths(dense_path, tex_use_orig_res ? SPARSE_ORIG_RES_DIR : SPARSE_DIR);
        boost::filesystem::create_directories(sparse_path);
        auto dense_image_path = JoinPaths(dense_path, tex_use_orig_res ? IMAGES_ORIG_RES_DIR : IMAGES_DIR);
        auto dense_mask_path = JoinPaths(dense_path, tex_use_orig_res ? MASKS_ORIG_RES_DIR : MASKS_DIR);
        boost::filesystem::create_directories(dense_image_path);
        auto dense_stereo_depth_path = JoinPaths(dense_path, STEREO_DIR, DEPTHS_DIR);
        boost::filesystem::create_directories(dense_stereo_depth_path);
        auto dense_stereo_normal_path = JoinPaths(dense_path, STEREO_DIR, NORMALS_DIR);
        boost::filesystem::create_directories(dense_stereo_normal_path);

        Timer writer_timer;
        writer_timer.Start();
        boost::filesystem::create_directories(sparse_path);
        int num_filter = update_reconstruction->FilterMapPointsWithSpatialDistribution(
            update_reconstruction->MapPointIds());
        std::cout << "Filter " << num_filter << " MapPoints with spatial distribution " << std::endl;
        std::shared_ptr<Reconstruction> filtered_reconstruction = std::make_shared<Reconstruction>();
        update_reconstruction->FilterUselessPoint2D(filtered_reconstruction);

        std::unordered_set<image_t> filter_image_ids;
        if (ExistsFile(JoinPaths(reconstruction_path, ALIGNMENT_POSE_NAME))) {
            // Read alignment images BEFORE InitDepthPriors
            auto align_image_ids = filtered_reconstruction->ReadAlignmentBinary(JoinPaths(reconstruction_path, ALIGNMENT_POSE_NAME));
            filter_image_ids.insert(align_image_ids.begin(), align_image_ids.end());
        }
        filtered_reconstruction->WriteReconstruction(sparse_path, true);
        filtered_reconstruction->WriteCameraIsRigText(sparse_path + "/rig.txt");
        WriteImageMapText(sparse_path + "/imageId_map.txt", map_convert_id);
        writer_timer.PrintSeconds();
        std::cout << "WriteReconstruction" << " " << sparse_path << std::endl;

        if (!options.filter_subpath.empty()) {
            for (const auto &image : filtered_reconstruction->Images()) {
                for (const auto &subpath : options.filter_subpath) {
                    if (IsInsideSubpath(image.second.Name(), subpath)) {
                        filter_image_ids.insert(image.second.ImageId());
                        break;
                    }
                }
            }
        }
        if (!tex_use_orig_res && options.as_rgbd) {
            InitDepthPriors(filtered_reconstruction, options, img_input_path, workspace_path, filter_image_ids);
        }
        if (!filter_image_ids.empty()) {
            for (auto image_id : filter_image_ids) {
                filtered_reconstruction->DeleteImage(image_id);
            }
            std::cout << "Filter " << filter_image_ids.size() << " Images" << std::endl;
            filtered_reconstruction->WriteReconstruction(sparse_path, true);
            filtered_reconstruction->WriteCameraIsRigText(sparse_path + "/rig.txt");
        }
        ///////////////////////////////////////////////////////////////////////////////////////
        // 3. Convert image
        ///////////////////////////////////////////////////////////////////////////////////////

        // bool color_harmonization = !reconstruction->yrb_factors.empty();

        uint64_t indices_print_step = image_ids.size() / 10 + 1;
        auto ConvertImage = [&](std::size_t i) {
            const image_t image_id = image_ids.at(i);
            class Image& image = reconstruction->Image(image_id);
            class Camera& camera = reconstruction->Camera(image.CameraId());
            std::string image_path = JoinPaths(img_input_path, image.Name());
            std::vector<std::string> name_parts = StringSplit(image.Name(), ".");
            std::string mask_path = image.Name().substr(0, image.Name().size() - name_parts.back().size() - 1) + ".png";
            mask_path = JoinPaths(mask_input_path, mask_path);
            if (!options.filter_subpath.empty()) {
                auto image_name = image.Name();
                for (const auto &subpath : options.filter_subpath) {
                    if (IsInsideSubpath(image.Name(), subpath)) {
                        return;
                    }
                }
            }

            // Reconstruction::YCrCbFactor ycrcb_factor;
            // if (color_harmonization) {
            //     ycrcb_factor = reconstruction->yrb_factors.at(image_id);
            // }

            std::vector<Bitmap> img_outs;
            if(camera.ModelName().compare("SPHERICAL") == 0 && camera.NumLocalCameras() == 1){
                // Load image
                Bitmap img_input;
                if (!img_input.Read(image_path, true)) {
                    std::cout << image_path << std::endl;
                    std::cout << "seg image read fail. " << std::endl;
                    ExceptionHandler(IMAGE_LOAD_FAILED, 
                        JoinPaths(GetParentDir(workspace_path), "errors/dense"), "DenseConvertPanorama").Dump();
                    exit(StateCode::IMAGE_LOAD_FAILED);
                    // return;
                }

                // if (color_harmonization) {
                //     ApplyColorHarmonization(&img_input, ycrcb_factor);
                // }
                const uint32_t hash_code = camera_hash_map.at(image.CameraId());
                panoramas.at(hash_code).PanoramaToPerspectives(&img_input, img_outs);

                const auto pos = image_path.find_last_of('.', image_path.length());
                std::string image_path_base = image_path.substr(0, pos);

                std::string image_folder;
                image_folder = image_path_base.substr(img_input_path.size(), image_path_base.size() - img_input_path.size());

                auto parent_path = JoinPaths(dense_image_path, GetParentDir(image_folder));
                if (!boost::filesystem::exists(parent_path)) {
                    boost::filesystem::create_directories(parent_path);
                }

                auto out_image_path = JoinPaths(dense_image_path, image_folder);
                for (size_t j = 0; j < img_outs.size(); ++j) {
                    std::string cur_img_path = 
                    options.as_rgbd ? StringPrintf("%s_%d.jpg.jpg", out_image_path.c_str(), j)
                                    : StringPrintf("%s_%d.jpg", out_image_path.c_str(), j);
                    img_outs.at(j).Write(cur_img_path, FIF_JPEG);
                }
            }
            else if(camera.NumLocalCameras() > 1){
                auto parent_path = JoinPaths(dense_image_path, GetParentDir(image.Name()));
                if (!boost::filesystem::exists(parent_path)) {
                    boost::filesystem::create_directories(parent_path);
                }
                auto out_image_path = JoinPaths(dense_image_path, image.Name());
                auto pos = out_image_path.find_last_of('.', out_image_path.length());
                out_image_path = out_image_path.substr(0, pos);
                
                const uint32_t hash_code = camera_hash_map.at(image.CameraId());

                for (size_t local_camera_id = 0; 
                    local_camera_id < camera.NumLocalCameras(); ++local_camera_id) {
                    auto raw_image_path = image_path;
                    if (image.HasLocalName(local_camera_id)) {
                        raw_image_path = JoinPaths(img_input_path, image.LocalName(local_camera_id));
                    } else {
                        auto pos = raw_image_path.find("cam0", 0);
                        raw_image_path.replace(pos, 4, StringPrintf("cam%d", local_camera_id));
                    }
                    // Load image
                    Bitmap img_input;
                    if (!img_input.Read(raw_image_path, true)) {
                        std::cout << image_path << std::endl;
                        std::cout << "seg image read fail. " << std::endl;
                        ExceptionHandler(IMAGE_LOAD_FAILED, 
                            JoinPaths(GetParentDir(workspace_path), "errors/dense"), "DenseConvertPanorama").Dump();
                        exit(StateCode::IMAGE_LOAD_FAILED);
                        // return;
                    }

                    // if (color_harmonization) {
                    //     ApplyColorHarmonization(&img_input, ycrcb_factor);
                    // }

                    piecewise_images.at(hash_code).ToSplitedPerspectives(img_input, img_outs, local_camera_id);
                    for (size_t j = 0; j < img_outs.size(); ++j) {
                        std::string cur_img_path =
                        options.as_rgbd ? out_image_path + "_" + std::to_string(j + local_camera_id * img_outs.size()) + ".jpg.jpg"
                                        : out_image_path + "_" + std::to_string(j + local_camera_id * img_outs.size()) + ".jpg";
                        img_outs.at(j).Write(cur_img_path, FIF_JPEG);
                    }
                }
            } else {
                // Load image
                Bitmap img_input;
                if (IsFileRGBD(image_path)) {
                    RGBDData data;
                    if (!ExtractRGBDData(image_path, data)) {
                        std::cerr << "rgbd image read fail. " << std::endl;
                        ExceptionHandler(IMAGE_LOAD_FAILED, 
                            JoinPaths(GetParentDir(workspace_path), "errors/dense"), "DenseConvertPanorama").Dump();
                        exit(StateCode::IMAGE_LOAD_FAILED);
                    }
                    img_input = std::move(data.color);
                } else if (!img_input.Read(image_path, true)) {
                    std::cout << image_path << std::endl;
                    std::cout << "seg image read fail. " << std::endl;
                    ExceptionHandler(IMAGE_LOAD_FAILED, 
                        JoinPaths(GetParentDir(workspace_path), "errors/dense"), "DenseConvertPanorama").Dump();
                    exit(StateCode::IMAGE_LOAD_FAILED);
                }

                // if (color_harmonization) {
                //     ApplyColorHarmonization(&img_input, ycrcb_factor);
                // }

                const std::string out_image_path =
                options.as_rgbd ? JoinPaths(dense_image_path, image.Name() + ".jpg")
                                : JoinPaths(dense_image_path, image.Name());
                std::string parent_path = GetParentDir(out_image_path);
                if (!ExistsPath(parent_path)) {
                    boost::filesystem::create_directories(parent_path);
                }

                Bitmap undistorted_bitmap;
                Camera undistorted_camera;
                Undistorter::UndistortImage(options, img_input, camera, &undistorted_bitmap, &undistorted_camera);
                undistorted_bitmap.Write(out_image_path);

                if (boost::filesystem::exists(mask_path)) {
                    Bitmap mask_input;
                    mask_input.Read(mask_path);

                    const std::string out_mask_path = JoinPaths(dense_mask_path, 
                        image.Name().substr(0, image.Name().size() - name_parts.back().size() - 1) + ".png");
                    parent_path = GetParentDir(out_mask_path);
                    if (!ExistsPath(parent_path)) {
                        boost::filesystem::create_directories(parent_path);
                    }
                    Bitmap undistorted_bitmap_mask;
                    Camera undistorted_camera_mask;
                    Undistorter::UndistortImage(options, mask_input, camera, &undistorted_bitmap_mask, &undistorted_camera_mask);
                    undistorted_bitmap_mask.Write(out_mask_path);
                }
            }
            // Print the progress
            if (i % indices_print_step == 0) {
                std::cout << StringPrintf("\rProcess Images [%d / %d]", i + 1, image_ids.size()) << std::flush;
            }
        };
        
        const int num_eff_threads = GetEffectiveNumThreads(-1);
        std::cout << "num_eff_threads: " << num_eff_threads << std::endl;

        std::unique_ptr<ThreadPool> thread_pool;
        thread_pool.reset(new ThreadPool(num_eff_threads));

        std::cout << "3. Convert image ... " << std::endl;
        Timer convert_timer;
        convert_timer.Start();
        for (std::size_t i = 0; i < image_ids.size(); ++i) {
            thread_pool->AddTask(ConvertImage, i);
        }
        thread_pool->Wait();
        std::cout << "Convert Image Cost: " << convert_timer.ElapsedSeconds() << "s" << std::endl;
    }

    return true;
}

int main(int argc, char* argv[]) {

	PrintHeading("SenseMap.  Copyright(c) 2019, SenseTime Group.");	
	PrintHeading(std::string("Version: ") + __VERSION__);

    Timer timer;
    timer.Start();

    configuration_file_path = std::string(argv[1]);
    Configurator param;
    param.Load(configuration_file_path.c_str());

    std::string workspace_path = param.GetArgument("workspace_path", "");
	bool map_update = param.GetArgument("map_update", 0);
    if (map_update) {
        workspace_path = JoinPaths(workspace_path, "map_update");
    }

    std::string img_input_path = param.GetArgument("image_path", "");
    std::string mask_input_path = param.GetArgument("mask_path", "");
    // std::string lidar_input_path = param.GetArgument("lidar_path", "");
    std::string output_workspace_path = workspace_path;
    // Set camera number
    int divide_camera_num = param.GetArgument("divide_camera_num", 6);
    //cross_warp
    std::string image_type = param.GetArgument("image_type","");
    std::string camera_param_file = param.GetArgument("camera_param_file","");
    int cross_warp_num_images = param.GetArgument("cross_warp_num_images", 8);
    std::vector<std::string> cross_warp_subpath = CSVToVector<std::string>(param.GetArgument("cross_warp_subpath", ""));
    std::vector<std::string> filter_subpath = CSVToVector<std::string>(param.GetArgument("filter_subpath", ""));
    float min_prior_depth = param.GetArgument("min_prior_depth",0.0f);
    float max_prior_depth = param.GetArgument("max_prior_depth",10.0f);
    int warp_depth_sparsely = param.GetArgument("warp_depth_sparsely",0);
    int reverse_scale_recovery = param.GetArgument("reverse_scale_recovery",0);
    bool verbose = param.GetArgument("verbose", false);

    // Set camera param
    UndistortOptions options;
    options.fov_w = param.GetArgument("fov_w", 60.0f);
    options.fov_h = param.GetArgument("fov_h", 90.0f);
    options.max_image_size = param.GetArgument("max_image_size", 320);
    // options.max_image_size = options.max_image_size > 0 ? std::min(options.max_image_size, 5000) : 5000;
    options.verbose = verbose;

    options.as_rgbd = image_type.compare("rgbd")==0;
    options.cross_warp_num_images = cross_warp_num_images;
    options.cross_warp_subpath = cross_warp_subpath;
    options.filter_subpath = filter_subpath;
    options.min_prior_depth = min_prior_depth;
    options.max_prior_depth = max_prior_depth;
    options.warp_depth_sparsely = warp_depth_sparsely;
    options.reverse_scale_recovery = reverse_scale_recovery;

    bool keyframe_mode = param.GetArgument("keyframe_mode", 0);
    options.max_rig_image_size = param.GetArgument("max_rig_image_size", 0);
    bool use_max_rig_image_size=true;
    if(options.max_rig_image_size == 0){
        use_max_rig_image_size = false;
        options.max_rig_image_size = options.max_image_size;
    }
    std::cout << "options.max_image_size, max_rig_image_size: " << options.max_image_size << ", " << options.max_rig_image_size << std::endl;

    ConvertPanoramaConfig(workspace_path, img_input_path, mask_input_path, output_workspace_path, divide_camera_num, options, 
                          keyframe_mode, false, map_update);

    bool tex_use_orig_res = param.GetArgument("tex_use_orig_res", 0);
    
    options.max_rig_image_size = param.GetArgument("tex_max_image_size", -1);
    if(!use_max_rig_image_size){
        // std::cout<<"ori max_rig_image_size: "<<options.max_rig_image_size<<" changed: "<<options.max_image_size<<std::endl;
        options.max_image_size = options.max_rig_image_size ;
    }

    std::cout<<"options.max_image_size: "<<options.max_image_size<<" options.max_rig_image_size: "<<options.max_rig_image_size<<std::endl;

    // options.max_image_size = options.max_image_size > 0 ? std::min(options.max_image_size, 5000) : 5000;  
    if (tex_use_orig_res) {
        ConvertPanoramaConfig(workspace_path, img_input_path, mask_input_path, output_workspace_path, divide_camera_num, options, 
                              keyframe_mode, tex_use_orig_res, map_update);
    }
    
    timer.PrintMinutes();

    return StateCode::SUCCESS;
}
