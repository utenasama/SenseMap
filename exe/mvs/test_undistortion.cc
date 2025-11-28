//Copyright (c) 2019, SenseTime Group.
//All rights reserved.

#include <boost/filesystem/path.hpp>

#include "util/misc.h"
#include "util/threading.h"
#include "util/panorama.h"
#include "base/reconstruction_manager.h"
#include "base/camera.h"
#include "base/image.h"
#include "base/common.h"
#include "controllers/patch_match_controller.h"

#include "base/undistortion.h"
#include "../Configurator_yaml.h"

using namespace sensemap;

std::string configuration_file_path;


Eigen::RowMatrix3f EulerToRotationMatrix(double roll, double yaw, double pitch){
    Eigen::AngleAxisf rollAngle = 
        Eigen::AngleAxisf(roll / 180 * M_PI, Eigen::Vector3f::UnitX());
    Eigen::AngleAxisf pitchAngle = 
        Eigen::AngleAxisf(pitch / 180 * M_PI, Eigen::Vector3f::UnitY());
    Eigen::AngleAxisf yawAngle = 
        Eigen::AngleAxisf(yaw / 180 * M_PI, Eigen::Vector3f::UnitZ());
    
    Eigen::Quaternionf q = pitchAngle * rollAngle * yawAngle;

    return q.matrix();
}

void PrepareData(const mvs::PatchMatchOptions &options, std::shared_ptr<mvs::Workspace> workspace,
				 const std::string &component_path, int max_image_size, bool tex_use_orig_res = false) {
    std::cout << "Create Perspective workspace..." << std::endl;

	const int num_perspective_per_image = 6;
	const double rolls[num_perspective_per_image] = {0, 0, 0, 0, 0, 0};
	const double pitches[num_perspective_per_image]= {0, 60, 120, 180, 240, 300};
	const double yaws[num_perspective_per_image] = {0, 0, 0, 0, 0, 0};

	const double fov_w = 60.0;
	const double fov_h = 90.0;

	std::vector<std::string> perspective_image_names_;
	std::vector<mvs::Image> perspective_images_;
	std::vector<std::vector<int> > perspective_src_images_idx_;
	std::vector<std::pair<float, float> > depth_ranges_;

    const float min_triangulation_angle_rad =
        DegToRad(options.min_triangulation_angle);
    // const int perspective_image_size = options.max_image_size;
	const int perspective_image_size = max_image_size;
    const mvs::Model& model = workspace->GetModel();
    const int num_image = model.images.size();

    std::string workspace_sparse_path = JoinPaths(component_path, tex_use_orig_res ? SPARSE_ORIG_RES_DIR : SPARSE_DIR);

    CreateDirIfNotExists(component_path);
    CreateDirIfNotExists(workspace_sparse_path);

    std::string workspace_stereo_path = JoinPaths(component_path, STEREO_DIR);
    CreateDirIfNotExists(workspace_stereo_path);

    // CreateDirIfNotExists(JoinPaths(workspace_stereo_path, DEPTHS_DIR));
    // CreateDirIfNotExists(JoinPaths(workspace_stereo_path, NORMALS_DIR));
    // CreateDirIfNotExists(JoinPaths(workspace_stereo_path, CONSISTENCY_DIR));

    std::string workspace_image_path = JoinPaths(component_path, tex_use_orig_res ? IMAGES_ORIG_RES_DIR : IMAGES_DIR);
    std::string sparse_file = JoinPaths(workspace_sparse_path, "cameras.bin");
    if (ExistsFile(sparse_file) && ExistsDir(workspace_image_path)) {
        return;
    }

    CreateDirIfNotExists(workspace_image_path);

    const int num_eff_threads = GetEffectiveNumThreads(-1);
    std::unique_ptr<ThreadPool> thread_pool;
    thread_pool.reset(new ThreadPool(num_eff_threads));

    Eigen::RowMatrix3f Rs[num_perspective_per_image];
    for (int i = 0; i < num_perspective_per_image; ++i) {
        Rs[i] = EulerToRotationMatrix(rolls[i], yaws[i], pitches[i]);
        Rs[i].transposeInPlace();
    }
    {
        std::cout << "Generate Perspective images..." << std::endl;

        Panorama panorama;
        const mvs::Image& image = model.images.at(0);
        Bitmap initial_bitmap;
        if (!initial_bitmap.Read(image.GetPath())) {
            std::cerr << "Error! Panorama Initial file path " << image.GetPath() 
                        << " does not exist" << std::endl;
            exit(-1);
        }
        panorama.PerspectiveParamsProcess(perspective_image_size, fov_w, fov_h, num_perspective_per_image,
                                                  initial_bitmap.Width(), initial_bitmap.Height());

        float focal_length;
        int width, height;
        focal_length = panorama.GetPerspectiveFocalLength(0);
        width = panorama.GetPerspectiveWidth(0);
        height = panorama.GetPerspectiveHeight(0);

        float K[9];
        K[0] = focal_length; K[1] = 0.0f;         K[2] = width * 0.5f;
        K[3] = 0.0f;         K[4] = focal_length; K[5] = height * 0.5f;
        K[6] = 0.0f;         K[7] = 0.0f;         K[8] = 1.0f;
        
        perspective_image_names_.resize(num_image * num_perspective_per_image);
        perspective_images_.resize(num_image * num_perspective_per_image);

        auto ConvertPanorama = [&](int ref_image_idx) {
            std::cout << "Convert Panorama#" << ref_image_idx << std::endl;
            const mvs::Image& image = model.images.at(ref_image_idx);
            Bitmap bitmap;
            if (!bitmap.Read(image.GetPath())) {
                std::cerr << "Error! File " << image.GetPath() 
                          << " does not exist" << std::endl;
                return;
            }
           
            Eigen::Vector3f ref_T_tmp(image.GetT());
            Eigen::RowMatrix3f ref_R_tmp(image.GetR());                                   

            std::vector<Bitmap> perspective_images(num_perspective_per_image);
            for (int i = 0; i < num_perspective_per_image; ++i) {
                panorama.PanoramaToPerspectives(&bitmap, perspective_images);
            }

            std::string image_name = model.GetImageName(ref_image_idx);
            const auto pos = image_name.find_last_of('.', image_name.length());
            std::string image_name_base = image_name.substr(0, pos);
            for (int i = 0; i < perspective_images.size(); ++i) {
                const int idx = ref_image_idx * num_perspective_per_image + i;
                std::string iimage_name = StringPrintf("%s_%d.jpg", 
                    image_name_base.c_str(), i);
                std::string iimage_path = 
                    JoinPaths(workspace_image_path, iimage_name);
                const std::string parent_path = GetParentDir(iimage_path);
                if (!ExistsPath(parent_path)) {
                    boost::filesystem::create_directories(parent_path);
                }
                perspective_image_names_[idx] = iimage_name;
                perspective_images[i].Write(iimage_path);

                Eigen::RowMatrix3f R = Rs[i] * ref_R_tmp;
                Eigen::Vector3f T = Rs[i] * ref_T_tmp;
                mvs::Image p_image(iimage_path, width, height, K, 
                                   R.data(), T.data());
                // Bitmap bitmap;
                // bitmap.Read(iimage_path, false);
                // p_image.SetBitmap(bitmap);
                p_image.SetBitmap(perspective_images[i]);
                p_image.GetBitmap().ConvertToGray();
                perspective_images_[idx] = p_image;
            }
        };

        for (int ref_image_idx = 0; ref_image_idx < num_image; 
             ++ref_image_idx) {
            thread_pool->AddTask(ConvertPanorama, ref_image_idx);
        }
        thread_pool->Wait();
    }

    std::cout << "Compute Adjoin Perspective images..." << std::endl;

    std::vector<std::unordered_map<int, int>> shared_num_points;
    std::vector<std::unordered_map<int, float>> triangulation_angles;
    if (shared_num_points.empty()) {
        shared_num_points = model.ComputeSharedPoints();
    }
    if (triangulation_angles.empty()) {
        const float kTriangulationAnglePercentile = 75;
        triangulation_angles =
            model.ComputeTriangulationAngles(kTriangulationAnglePercentile);
    }

    std::vector<std::vector<int> > src_image_idxs(num_image);
    auto ComputeOverlapping = [&](int ref_image_idx) {
        const mvs::Image& image = model.images.at(ref_image_idx);

        const auto& overlapping_images = shared_num_points.at(ref_image_idx);
        const auto& overlapping_triangulation_angles = 
            triangulation_angles.at(ref_image_idx);

        std::vector<std::pair<int, int> > src_images;
        src_images.reserve(overlapping_images.size());
        for (const auto& image : overlapping_images) {
            if (overlapping_triangulation_angles.at(image.first) >=
                min_triangulation_angle_rad) {
                src_images.emplace_back(image.first, image.second);
            }
        }

        const size_t eff_max_num_src_images = 
            std::min(src_images.size(), options.max_num_src_images);

        std::partial_sort(src_images.begin(),
                          src_images.begin() + eff_max_num_src_images,
                          src_images.end(),
                          [](const std::pair<int, int>& image1,
                            const std::pair<int, int>& image2) {
                            return image1.second > image2.second;
                          });

        src_image_idxs[ref_image_idx].reserve(eff_max_num_src_images);
        for (size_t j = 0; j < eff_max_num_src_images; ++j) {
            src_image_idxs[ref_image_idx].push_back(src_images[j].first);
        }
    };

    for (int ref_image_idx = 0; ref_image_idx < num_image; ++ref_image_idx) {
        thread_pool->AddTask(ComputeOverlapping, ref_image_idx);
    }
    thread_pool->Wait();

    const int strip = num_perspective_per_image;
    perspective_src_images_idx_.resize(num_image * num_perspective_per_image);
    for (int i = 0; i < num_image; ++i) {
        for (const auto& image_idx : src_image_idxs[i]) {
            for (int j = 0; j < num_perspective_per_image; ++j) {
                perspective_src_images_idx_[i * strip + j].push_back(image_idx * strip + j);
            }
        }
    }

    std::cout << "Compute Depth Range..." << std::endl;
    // compute depth ranges for each perspective image.
    std::vector<std::vector<int> > mappoints_per_image(num_image);
    std::vector<mvs::Model::Point> points = model.points;
    for (int i = 0; i < points.size(); ++i) {
        const auto& point = points[i];
        for (const auto& image_idx : point.track) {
            mappoints_per_image[image_idx].push_back(i);
        }
    }

    std::vector<std::vector<float>> depths(perspective_images_.size());
    auto ComputeDepthRange = [&](int image_idx) {
        const auto& mappoint_ids = mappoints_per_image[image_idx];
        for (int i = 0; i < num_perspective_per_image; ++i) {
            const int perspective_idx = 
                image_idx * num_perspective_per_image + i;
            const mvs::Image& image = perspective_images_[perspective_idx];

            const int width = image.GetWidth();
            const int height = image.GetHeight();
            float focal = image.GetK()[0];
            float principal_x = image.GetK()[2];
            float principal_y = image.GetK()[5];

            Eigen::Map<const Eigen::Vector3f> R0(&image.GetR()[0]);
            Eigen::Map<const Eigen::Vector3f> R1(&image.GetR()[3]);
            Eigen::Map<const Eigen::Vector3f> R2(&image.GetR()[6]);
            const float T0 = image.GetT()[0];
            const float T1 = image.GetT()[1];
            const float T2 = image.GetT()[2];

            int num_visible_mappoint = 0;
            for (const auto& mappoint_id : mappoint_ids) {
                const mvs::Model::Point& point = points[mappoint_id];
                Eigen::Vector3f X(point.x, point.y, point.z);
                const float depth = R2.dot(X) + T2;
                const float x = R0.dot(X) + T0;
                const float y = R1.dot(X) + T1;
                int u = focal * x / depth + principal_x;
                int v = focal * y / depth + principal_y;
                if (u >= 0 && u < width && v >= 0 && v < height && depth > 0) {
                    depths[perspective_idx].push_back(depth);
                    num_visible_mappoint++;
                }
            }
            if (num_visible_mappoint == 0) {
                depths[perspective_idx].push_back(1e-5f);
                depths[perspective_idx].push_back(100.0f);
            }
        }
    };

    for (int image_idx = 0; image_idx < num_image; ++image_idx) {
        thread_pool->AddTask(ComputeDepthRange, image_idx);
    }
    thread_pool->Wait();

    depth_ranges_.resize(depths.size());
    for (size_t image_idx = 0; image_idx < depth_ranges_.size(); ++image_idx) {
        auto& depth_range = depth_ranges_[image_idx];

        auto& image_depths = depths[image_idx];

        if (image_depths.empty()) {
            depth_range.first = -1.0f;
            depth_range.second = -1.0f;
            continue;
        }

        std::sort(image_depths.begin(), image_depths.end());

        const float kMinPercentile = 0.01f;
        const float kMaxPercentile = 0.99f;
        depth_range.first = image_depths[image_depths.size() * kMinPercentile];
        depth_range.second = image_depths[image_depths.size() * kMaxPercentile];

        const float kStretchRatio = 0.25f;
        depth_range.first *= (1.0f - kStretchRatio);
        depth_range.second *= (1.0f + kStretchRatio);
        std::cout << "depth range#" << image_idx << ": " 
                  << depth_ranges_[image_idx].first << ", " 
                  << depth_ranges_[image_idx].second 
                  << std::endl;
    }

    // for visualization with colmap gui.
    std::ofstream ofs(workspace_stereo_path + "/patch-match.cfg", 
                      std::ofstream::out);
    for (const auto& image_name : perspective_image_names_) {
        ofs << image_name << std::endl;
        ofs << "__auto__, " << options.max_num_src_images << std::endl;
    }
    ofs.close();

    std::vector<image_t> perspective_image_ids;
    mvs::ExportPanoramaWorkspace(component_path,
        perspective_image_names_, perspective_images_, perspective_image_ids,
        perspective_src_images_idx_, depth_ranges_, tex_use_orig_res);
}

void Undistortion(const std::string &image_path, const std::string &workspace_path, const Configurator &param,
                  bool tex_use_orig_res = false, int tex_max_image_size = -1) {
    std::string image_type = param.GetArgument("image_type", "perspective");
	int max_image_size = (int)param.GetArgument("max_image_size", 640);
	float fov_w = param.GetArgument("fov_w", -1.0f);
	float fov_h = param.GetArgument("fov_h", -1.0f);

	if (image_type.compare("panorama") != 0) {
		std::cout << "workspace_path: " << workspace_path << std::endl;
		if (image_type.compare("perspective") == 0) {
			for (size_t rec_idx = 0; ; rec_idx++) {
				auto reconstruction_path = 
					JoinPaths(workspace_path, std::to_string(rec_idx));
                if (!ExistsDir(reconstruction_path)) {
                    break;
                }

				UndistortOptions options;
				options.max_image_size = max_image_size;
				options.fov_w = fov_w;
				options.fov_h = fov_h;

				Undistorter undistorter(options, image_path, reconstruction_path);
				undistorter.Start();
				undistorter.Wait();

                if (tex_use_orig_res) {
                    options.as_orig_res = true;
                    options.max_image_size = tex_max_image_size;

                    Undistorter undistorter(options, image_path, reconstruction_path);
                    undistorter.Start();
                    undistorter.Wait();

                    // // Creating original resolution semantic maps
                    // auto dense_reconstruction_path = JoinPaths(reconstruction_path, DENSE_DIR);
                    // auto semantics_path = JoinPaths(dense_reconstruction_path, SEMANTICS_DIR);
                    // if (!ExistsDir(semantics_path)) {
                    //     continue;
                    // }

                    // auto semantics_orig_res_path = JoinPaths(dense_reconstruction_path, SEMANTICS_ORIG_RES_DIR);
                    // CreateDirIfNotExists(semantics_orig_res_path);

                    // std::vector<mvs::Image> images;
                    // std::vector<std::string> image_names;
                    // {
                    //     auto undistort_image_path = JoinPaths(dense_reconstruction_path, IMAGES_DIR);

                    //     mvs::Workspace::Options workspace_options;
                    //     workspace_options.max_image_size = max_image_size;
                    //     workspace_options.image_as_rgb = false;
                    //     workspace_options.image_path = undistort_image_path;
                    //     workspace_options.workspace_path = dense_reconstruction_path;
                    //     workspace_options.workspace_format = image_type;
                    //     workspace_options.as_orig_res = false;

                    //     mvs::Workspace workspace(workspace_options);
                    //     const mvs::Model& model = workspace.GetModel();
                    //     images = model.images;
                    //     image_names.reserve(images.size());
                    //     for (size_t i = 0; i < images.size(); ++i) {
                    //         image_names.push_back(model.GetImageName(i));
                    //     }
                    // }

                    // std::vector<mvs::Image> images_orig_res;
                    // std::vector<std::string> image_names_orig_res;
                    // {
                    //     auto undistort_image_path = JoinPaths(dense_reconstruction_path, IMAGES_ORIG_RES_DIR);

                    //     mvs::Workspace::Options workspace_options;
                    //     workspace_options.max_image_size = tex_max_image_size;
                    //     workspace_options.image_as_rgb = false;
                    //     workspace_options.image_path = undistort_image_path;
                    //     workspace_options.workspace_path = dense_reconstruction_path;
                    //     workspace_options.workspace_format = image_type;
                    //     workspace_options.as_orig_res = tex_use_orig_res;

                    //     mvs::Workspace workspace(workspace_options);
                    //     const mvs::Model& model = workspace.GetModel();
                    //     images_orig_res = model.images;
                    //     image_names_orig_res.reserve(images_orig_res.size());
                    //     for (size_t i = 0; i < images_orig_res.size(); ++i) {
                    //         image_names_orig_res.push_back(model.GetImageName(i));
                    //     }
                    // }

                    // const int num_eff_threads = GetEffectiveNumThreads(-1);
                    // std::cout << "num_eff_threads: " << num_eff_threads << std::endl;
                    // std::unique_ptr<ThreadPool> thread_pool;
                    // thread_pool.reset(new ThreadPool(num_eff_threads));

                    // int progress = 0;
                    // auto CreateOrigResSemanticMap = [&](int start, int end) {
                    //     for (int image_idx = start; image_idx < end; ++image_idx) {
                    //         const std::string &image_name = image_names.at(image_idx);
                    //         const std::string semantic_map_name = image_name.substr(0, image_name.size() - 3) + "png";
                    //         const std::string semantic_map_path = JoinPaths(semantics_path, semantic_map_name);
                    //         if (!ExistsFile(semantic_map_path)) {
                    //             continue;
                    //         }

                    //         std::cout << StringPrintf("\rCreating original resolution semantic map %d/%d", ++progress, images.size());
                    //         auto semantic_map = std::unique_ptr<Bitmap>(new Bitmap);
                    //         semantic_map->Read(semantic_map_path, false);

                    //         const mvs::Image &image_orig_res = images_orig_res.at(image_idx);
                    //         semantic_map->Rescale(image_orig_res.GetWidth(), image_orig_res.GetHeight(), FILTER_BOX);

                    //         const std::string semantic_map_orig_res_path = JoinPaths(semantics_orig_res_path, semantic_map_name);
                    //         boost::filesystem::path file_path(semantic_map_orig_res_path);
                    //         boost::filesystem::create_directories(file_path.parent_path());

                    //         semantic_map->Write(semantic_map_orig_res_path);
                    //     }
                    // };

                    // const size_t num_slice = (images.size() + num_eff_threads - 1) / num_eff_threads;
                    // for (int thread_idx = 0; thread_idx < num_eff_threads; ++thread_idx) {
                    //     int start = thread_idx * num_slice;
                    //     int end = std::min(num_slice * (thread_idx + 1), images.size());
                    //     thread_pool->AddTask(CreateOrigResSemanticMap, start, end);
                    // }

                    // thread_pool->Wait();
                }
			}
		}
    } else {
		bool init_from_visible_map = param.GetArgument("init_from_visible_map", 1);
		bool init_from_delaunay = param.GetArgument("init_from_delaunay", 1);
		bool init_from_global_map = param.GetArgument("init_from_global_map", 0);
		bool init_from_model = param.GetArgument("init_from_model", 0);
		bool median_filter = param.GetArgument("median_filter", 0);
		bool filter = param.GetArgument("filter", 0);
		float conf_threshold = param.GetArgument("conf_threshold", 0.8f);
		float random_depth_ratio = param.GetArgument("random_depth_ratio", 0.01f);
		bool plane_regularizer = param.GetArgument("plane_regularizer", 0);
		float random_smooth_bonus = param.GetArgument("random_smooth_bonus", 0.8f);
		float geom_consistency_regularizer = param.GetArgument("geom_consistency_regularizer", 0.1f);
		int num_iter_geom_consistency = param.GetArgument("num_iter_geom_consistency", 1);
		float gradient_regularizer = param.GetArgument("gradient_regularizer", 0.1f);
		bool verbose = param.GetArgument("verbose", 0);

		mvs::PatchMatchOptions options;
		options.verbose = verbose;
		options.window_radius = 5;
		options.window_step = 2;
		options.plane_regularizer = plane_regularizer;
		options.num_good_hypothesis = 1;
		options.num_bad_hypothesis = 3;
		options.geom_consistency = true;
		options.num_iter_geom_consistency = num_iter_geom_consistency;
		options.geom_consistency_regularizer = geom_consistency_regularizer;
		options.filter_min_triangulation_angle = 3.0f;
		options.filter_min_ncc = 0.2f;
		options.random_smooth_bonus = random_smooth_bonus;
		options.random_depth_ratio = random_depth_ratio;

		options.pyramid_max_level = 0;

		options.filter = filter;
		options.conf_filter = filter;
		// options.geo_filter = true;
		options.conf_threshold = conf_threshold;
		options.depth_diff_threshold = 0.02f;

		options.median_filter = median_filter;

		options.init_from_visible_map = init_from_visible_map;
		options.init_from_delaunay = init_from_delaunay;
		options.init_from_global_map = init_from_global_map;
		options.init_from_model = init_from_model;

		options.max_image_size = max_image_size;
		
		auto reconstruction_manager = std::make_shared<ReconstructionManager>();
		reconstruction_manager->Read(workspace_path);
		std::cout << "workspace_path: " << workspace_path << std::endl;
		for (size_t reconstruction_idx = 0; reconstruction_idx < reconstruction_manager->Size();
			reconstruction_idx++) {

			PrintHeading1(StringPrintf("Preparing Data for # %d", reconstruction_idx));

			const auto& reconstruction_path = 
				JoinPaths(workspace_path, std::to_string(reconstruction_idx));

			mvs::Workspace::Options workspace_options;
			workspace_options.max_image_size = max_image_size;
			workspace_options.image_as_rgb = false;
			workspace_options.cache_size = options.cache_size;
			workspace_options.image_path = image_path;
			workspace_options.workspace_path = reconstruction_path;
			workspace_options.workspace_format = "panorama";
			workspace_options.input_type = 
				options.geom_consistency ? PHOTOMETRIC_TYPE : "";

			std::shared_ptr<mvs::Workspace> workspace = std::make_shared<mvs::Workspace>(workspace_options);

			std::string component_path = JoinPaths(reconstruction_path, DENSE_DIR);

			PrepareData(options, workspace, component_path, max_image_size);
		}

		if (tex_use_orig_res) {
			for (size_t reconstruction_idx = 0; reconstruction_idx < reconstruction_manager->Size();
				reconstruction_idx++) {

				PrintHeading1(StringPrintf("Preparing Original Resolution Data for # %d", reconstruction_idx));

				const auto& reconstruction_path = 
					JoinPaths(workspace_path, std::to_string(reconstruction_idx));

				mvs::Workspace::Options workspace_options;
				workspace_options.max_image_size = tex_max_image_size;
				workspace_options.image_as_rgb = false;
				workspace_options.cache_size = options.cache_size;
				workspace_options.image_path = image_path;
				workspace_options.workspace_path = reconstruction_path;
				workspace_options.workspace_format = "panorama";
				workspace_options.input_type = 
					options.geom_consistency ? PHOTOMETRIC_TYPE : "";

				std::shared_ptr<mvs::Workspace> workspace = std::make_shared<mvs::Workspace>(workspace_options);

				std::string component_path = JoinPaths(reconstruction_path, DENSE_DIR);

				PrepareData(options, workspace, component_path, tex_max_image_size, tex_use_orig_res);

                // // Creating original resolution semantic maps
                // auto semantics_path = JoinPaths(component_path, SEMANTICS_DIR);
                // if (!ExistsDir(semantics_path)) {
                //     continue;
                // }

                // auto semantics_orig_res_path = JoinPaths(component_path, SEMANTICS_ORIG_RES_DIR);
                // CreateDirIfNotExists(semantics_orig_res_path);

                // std::vector<mvs::Image> images;
                // std::vector<image_t> image_ids; 
                // std::vector<std::string> image_names;
                // {
                //     std::vector<std::vector<int> > overlapping_images;
                //     std::vector<std::pair<float, float> > depth_ranges;
                //     if (!ImportPanoramaWorkspace(component_path, image_names,
                //         images, image_ids, overlapping_images, depth_ranges, false)) {
                //         continue;
                //     }
                // }

                // std::vector<mvs::Image> images_orig_res;
                // std::vector<std::string> image_names_orig_res;
                // {
                //     std::vector<std::vector<int> > overlapping_images;
                //     std::vector<std::pair<float, float> > depth_ranges;
                //     if (!ImportPanoramaWorkspace(component_path, 
                //         image_names_orig_res, images_orig_res, image_ids, 
                //         overlapping_images, depth_ranges, false, tex_use_orig_res)) {
                //         continue;
                //     }
                // }

                // const int num_eff_threads = GetEffectiveNumThreads(-1);
                // std::cout << "num_eff_threads: " << num_eff_threads << std::endl;
                // std::unique_ptr<ThreadPool> thread_pool;
                // thread_pool.reset(new ThreadPool(num_eff_threads));

                // int progress = 0;
                // auto CreateOrigResSemanticMap = [&](int start, int end) {
                //     for (int image_idx = start; image_idx < end; ++image_idx) {
                //         const std::string &image_name = image_names.at(image_idx);
                //         const std::string semantic_map_name = image_name.substr(0, image_name.size() - 3) + "png";
                //         const std::string semantic_map_path = JoinPaths(semantics_path, semantic_map_name);
                //         if (!ExistsFile(semantic_map_path)) {
                //             continue;
                //         }

                //         std::cout << StringPrintf("\rCreating original resolution semantic map %d/%d", ++progress, images.size());
                //         auto semantic_map = std::unique_ptr<Bitmap>(new Bitmap);
                //         semantic_map->Read(semantic_map_path, false);

                //         const mvs::Image &image_orig_res = images_orig_res.at(image_idx);
                //         semantic_map->Rescale(image_orig_res.GetWidth(), image_orig_res.GetHeight(), FILTER_BOX);

                //         const std::string semantic_map_orig_res_path = JoinPaths(semantics_orig_res_path, semantic_map_name);
                //         boost::filesystem::path file_path(semantic_map_orig_res_path);
                //         boost::filesystem::create_directories(file_path.parent_path());

                //         semantic_map->Write(semantic_map_orig_res_path);
                //     }
                // };

                // const size_t num_slice = (images.size() + num_eff_threads - 1) / num_eff_threads;
                // for (int thread_idx = 0; thread_idx < num_eff_threads; ++thread_idx) {
                //     int start = thread_idx * num_slice;
                //     int end = std::min(num_slice * (thread_idx + 1), images.size());
                //     thread_pool->AddTask(CreateOrigResSemanticMap, start, end);
                // }

                // thread_pool->Wait();
			}
		}
	}
}

int main(int argc, char *argv[]) {
    using namespace sensemap;

	configuration_file_path = std::string(argv[1]);
    Configurator param;
    param.Load(configuration_file_path.c_str());

	std::string image_path = param.GetArgument("image_path", "");
	std::string workspace_path = param.GetArgument("workspace_path", "");

    bool tex_use_orig_res =
        static_cast<int>(param.GetArgument("tex_use_orig_res", 0));

    int tex_max_image_size =
        static_cast<int>(param.GetArgument("tex_max_image_size", -1));

	PrintHeading("SenseMap.  Copyright(c) 2019, SenseTime Group.");

    Undistortion(image_path, workspace_path, param, tex_use_orig_res, tex_max_image_size);

    return 0;
}