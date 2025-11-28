//Copyright (c) 2019, SenseTime Group.
//All rights reserved.
#include <boost/filesystem.hpp>
#include "util/misc.h"
#include "util/rgbd_helper.h"
#include "util/proc.h"
#include "util/exception_handler.h"
#include "base/undistortion.h"
#include "base/reconstruction_manager.h"
#include "controllers/patch_match_controller.h"

#include "base/version.h"
#include "../Configurator_yaml.h"
#include "../camera_rig_params.h"
#ifdef DO_ENCRYPT_CHECK
#include "../check.h"
#endif

std::string configuration_file_path;

void LoadSemanticLabels(const std::string filepath, std::vector<uint8_t>& label_ids) {
    std::ifstream file;
    file.open(filepath.c_str(), std::ofstream::in);

    label_ids.clear();

    std::string line;
    std::string item;
    while (std::getline(file, line)) {
        sensemap::StringTrim(&line);
        if (line.empty()) {
            continue;
        }
        std::stringstream line_stream(line);
        while (!line_stream.eof()) {
            std::getline(line_stream, item, ' ');
            label_ids.push_back(std::stoi(item));
        }
    }
    file.close();
}

int main(int argc, char *argv[]) {
    using namespace sensemap;
    using namespace sensemap::mvs;

    PrintHeading("SenseMap.  Copyright(c) 2019, SenseTime Group.");    
    PrintHeading(std::string("Version: ") + __VERSION__);

    Timer patchmatch_timer;
    patchmatch_timer.Start();

    int reconstrction_idx = -1;
    int cluster_idx = -1;

    int param_idx = 1;
#ifdef DO_ENCRYPT_CHECK
    CHECK(argc >= 5);
    int ret = do_check(5, (const char**)argv);
      std::cout << "Check Status: " << ret << std::endl;
      if (ret) return StateCode::ENCYPT_CHECK_FAILED;
    param_idx = 5;
#endif
    configuration_file_path = std::string(argv[param_idx]);
    if (argc > (2 + param_idx - 1)) {
        reconstrction_idx = atoi(argv[2 + param_idx - 1]);
    }

    if (argc > (3 + param_idx - 1)){
        cluster_idx = atoi(argv[3 + param_idx - 1]);
    }

    std::cout << "configuration_file_path: " << configuration_file_path << std::endl;

    Configurator param;
    param.Load(configuration_file_path.c_str());

    std::string image_path = param.GetArgument("image_path", "");
    std::string workspace_path = param.GetArgument("workspace_path", "");
    
    std::string lidar_path = param.GetArgument("lidar_path", "");
    std::string mask_path = param.GetArgument("mask_path", "");
    std::string format = param.GetArgument("image_type", "perspective");
    std::string lidar2cam_calibfile = param.GetArgument("lidar2cam_calibfile", "");

    std::string camera_rig_params_file = param.GetArgument("rig_params_file", "");
    std::string camera_rig_params = "";
    CameraRigParams rig_params;
    if (!camera_rig_params_file.empty()) {
        rig_params.LoadParams(camera_rig_params_file);
        rig_params.ParamsToString();
        camera_rig_params = rig_params.local_extrinsics_str;
    }

    std::string gpu_index = param.GetArgument("gpu_index", "-1");

    bool init_from_visible_map = param.GetArgument("init_from_visible_map", 1);
    bool init_from_delaunay = param.GetArgument("init_from_delaunay", 1);
    bool init_from_global_map = param.GetArgument("init_from_global_map", 0);
    bool init_from_model = param.GetArgument("init_from_model", 0);
    bool init_from_dense_points = param.GetArgument("init_from_dense_points", 0);

    bool median_filter = param.GetArgument("median_filter", 0);
    bool filter = param.GetArgument("filter", 0);
    bool conf_filter = param.GetArgument("conf_filter", 0);
    bool geo_filter = param.GetArgument("geo_filter", 0);
    float conf_threshold = param.GetArgument("conf_threshold", 0.8f);
    float depth_diff_threshold = param.GetArgument("depth_diff_threshold", 0.01f);
    float max_normal_error = param.GetArgument("max_normal_error", 15.0f);
    int filter_min_num_consistent = param.GetArgument("filter_min_num_consistent", 1);
    int max_image_size = param.GetArgument("max_image_size", -1);
    int max_num_src_images = param.GetArgument("max_num_src_images", 8);
    int window_radius = param.GetArgument("window_radius", 5);
    int window_step = param.GetArgument("window_step", 2);
    int num_good_hypothesis = param.GetArgument("num_good_hypothesis", 1);
    int num_bad_hypothesis = param.GetArgument("num_bad_hypothesis", 3);
    int num_iterations_patch_match = param.GetArgument("num_iterations_patch_match", 5);

    float min_triangulation_angle = param.GetArgument("min_triangulation_angle", 1.0f);
    float random_depth_ratio = param.GetArgument("random_depth_ratio", 0.01f);
    float sample_radius_for_lidar = param.GetArgument("sample_radius_for_lidar", 10.0f);

    bool plane_regularizer = param.GetArgument("plane_regularizer", 0);
    float random_smooth_bonus = param.GetArgument("random_smooth_bonus", 0.8f);

    bool geom_consistency = param.GetArgument("geom_consistency", 1);
    float geom_consistency_regularizer = param.GetArgument("geom_consistency_regularizer", 0.1f);
    int num_iter_geom_consistency = param.GetArgument("num_iter_geom_consistency", 1);
    bool random_optimization = param.GetArgument("random_optimization", 1);
    bool local_optimization = param.GetArgument("local_optimization", 1);
    bool est_curvature = param.GetArgument("est_curvature", 0);

    bool verbose = param.GetArgument("verbose", 0);

    float fov_w = param.GetArgument("fov_w", -1.0f);
    float fov_h = param.GetArgument("fov_h", -1.0f);

    int pyramid_max_level = param.GetArgument("pyramid_max_level", 0);
    int pyramid_delta_level = param.GetArgument("pyramid_delta_level", 1);

    bool patch_match_fusion = param.GetArgument("patch_match_fusion", true);
    bool save_depth_map = param.GetArgument("save_depth_map", false);
    bool roi_fuse = param.GetArgument("roi_fuse", false);
    float roi_box_width = param.GetArgument("roi_box_width", -1.f);
    float roi_box_factor = param.GetArgument("roi_box_factor", -1.f);
    int min_num_visible_images = param.GetArgument("min_num_visible_images", -1);

    std::string black_list_file_path = param.GetArgument("black_list_file_path", "");

    bool propagate_depth = param.GetArgument("propagate_depth", 0);
    bool refine_with_semantic = param.GetArgument("refine_with_semantic", 0);

    bool fused_delaunay_sample = param.GetArgument("fused_delaunay_sample", true);
    float fused_dist_insert = param.GetArgument("dist_insert", 5.0f);
    float fused_diff_depth = param.GetArgument("diff_depth", 0.01f);

    bool outlier_removal = param.GetArgument("outlier_removal", true);
    int nb_neighbors = param.GetArgument("nb_neighbors", 6);
    double max_spacing_factor = param.GetArgument("max_spacing_factor", 6.0f);
    float outlier_percent = param.GetArgument("outlier_percent", 0.9f);
    float outlier_max_density = param.GetArgument("outlier_max_density", 0.01f);

    bool plane_optimization = param.GetArgument("plane_optimization", true);
    float plane_dist_threld = param.GetArgument("plane_dist_threld", 0.85f);
    float plane_raidus_factor = param.GetArgument("plane_raidus_factor", 100.0f);
    int step_size = param.GetArgument("step_size", 1);

    bool map_update = param.GetArgument("map_update", 0);
    std::cout << "map_update: " << map_update << std::endl;
    std::string ori_workspace_path = workspace_path;
    if (map_update) {
        workspace_path = JoinPaths(workspace_path, "map_update");
    }

    bool tex_use_orig_res =
        static_cast<int>(param.GetArgument("tex_use_orig_res", 0));

    int tex_max_image_size =
        static_cast<int>(param.GetArgument("tex_max_image_size", -1));

    float max_ram = static_cast<float>(param.GetArgument("max_ram", -1.0f));
    float max_gpu_memory = static_cast<float>(param.GetArgument("max_gpu_memory", 6.0f));
    float gpu_memory_factor = param.GetArgument("gpu_memory_factor", 0.7f);

    int warp_depth_sparsely = param.GetArgument("warp_depth_sparsely", 0);    
    int reverse_scale_recovery = param.GetArgument("reverse_scale_recovery", 0);    
    float min_prior_depth = param.GetArgument("min_prior_depth", 0.0f);    
    float max_prior_depth = param.GetArgument("max_prior_depth", 10.0f);    
    std::string rgbd_params_file = param.GetArgument("rgbd_params_file", "");
    std::string rgbd_camera_params;
    if (!rgbd_params_file.empty()) {
        auto calib_reader = sensemap::GetCalibBinReaderFromName(rgbd_params_file);
        calib_reader->ReadCalib(rgbd_params_file);
        rgbd_camera_params = calib_reader->ToParamString();
        std::cout << "rgbd_camera_params: " << rgbd_camera_params << std::endl;
    }

    int cross_warp_num_images = param.GetArgument("cross_warp_num_images", 8);
    std::vector<std::string> cross_warp_subpath = CSVToVector<std::string>(param.GetArgument("cross_warp_subpath", ""));
    std::vector<std::string> filter_subpath = CSVToVector<std::string>(param.GetArgument("filter_subpath", ""));

    std::string semantic_table_path = param.GetArgument("semantic_table_path", "");
    if (ExistsFile(semantic_table_path)) {
        LoadSemanticColorTable(semantic_table_path.c_str());
    }

    int init_is_ok = (int)init_from_model + (int)init_from_dense_points 
                   + (int)init_from_global_map + (int)init_from_visible_map;
    if (init_is_ok > 1) {
        std::cerr << "Error! Cannot set multiple variables at the same time"
        "(init_from_model | init_from_dense_points | init_from_global_map | init_from_visible_map) " 
                  << std::endl;
        return StateCode::VARIABLE_CONFLICT;
    }

    if (format.compare("panorama") != 0) {
        std::cout << "workspace_path: " << workspace_path << std::endl;
        if (format.compare("perspective") == 0 || format.compare("rgbd") == 0) {
            for (size_t rec_idx = 0; ;rec_idx++) {
                if (reconstrction_idx > 0){
                    rec_idx = reconstrction_idx;
                }

                auto reconstruction_path = JoinPaths(workspace_path, std::to_string(rec_idx));
                if (!ExistsDir(reconstruction_path)) {
                    break;
                }

                UndistortOptions options;
                options.verbose = verbose;
                options.as_rgbd = format.compare("rgbd") == 0;
                options.cross_warp_num_images = cross_warp_num_images;
                options.cross_warp_subpath = cross_warp_subpath;
                options.filter_subpath = filter_subpath;
                options.rgbd_camera_params = rgbd_camera_params;
                options.warp_depth_sparsely = warp_depth_sparsely;
                options.reverse_scale_recovery = reverse_scale_recovery;
                options.min_prior_depth = min_prior_depth;
                options.max_prior_depth = max_prior_depth;
                options.mask_path = mask_path;
                // options.max_image_size = std::min(max_image_size, 5000);
                options.max_image_size = max_image_size;
                options.fov_w = fov_w;
                options.fov_h = fov_h;
                options.postprocess = false;

                Undistorter undistorter(options, image_path, 
                                        reconstruction_path);
                undistorter.Start();
                undistorter.Wait();

                if (tex_use_orig_res) {
                    options.as_orig_res = true;
                    // options.max_image_size = std::min(tex_max_image_size, 5000);
                    options.max_image_size = tex_max_image_size;

                    Undistorter undistorter(options, image_path, reconstruction_path);
                    undistorter.Start();
                    undistorter.Wait();
                }
                if (reconstrction_idx > 0){
                    break;
                }
            }
        }

        ////////////////////////////////////////////////////////////////////////
        //Usage:
        PatchMatchOptions options;
        options.image_type = format;
        options.lidar2cam_calibfile = lidar2cam_calibfile;
        options.camera_rig_params = camera_rig_params;
        options.verbose = verbose;
        options.map_update = map_update;
        options.random_optimization = random_optimization;
        options.local_optimization = local_optimization;
        options.est_curvature = est_curvature;

        options.window_radius = window_radius;
        options.window_step = window_step;
        options.thk = 2;
        options.max_num_src_images = max_num_src_images;
        options.plane_regularizer = plane_regularizer;
        options.num_good_hypothesis = num_good_hypothesis;
        options.num_bad_hypothesis = num_bad_hypothesis;

        options.geom_consistency = geom_consistency;
        if (!options.geom_consistency && pyramid_max_level > 1) {
            options.geom_consistency = true;
            std::cout << "Warning! Force geom_consistency to be true under multi scale settings!" << std::endl;
        }
        options.num_iter_geom_consistency = num_iter_geom_consistency;
        options.geom_consistency_regularizer = geom_consistency_regularizer;
        options.filter_geom_consistency_max_cost = 2.0f;
        options.max_normal_error = max_normal_error;
        options.num_iterations = num_iterations_patch_match;
        options.gpu_index = gpu_index;
        options.filter_min_triangulation_angle = 3.0f;
        options.filter_min_num_consistent = filter_min_num_consistent;
        options.filter_min_ncc = 0.1f;
        options.random_smooth_bonus = random_smooth_bonus;
        options.max_image_size = max_image_size;
        options.min_triangulation_angle = min_triangulation_angle;
        options.random_depth_ratio = random_depth_ratio;

        options.filter = filter;
        options.conf_filter = conf_filter;
        options.geo_filter = geo_filter;
        options.conf_threshold = conf_threshold;
        options.depth_diff_threshold = depth_diff_threshold;

        options.pyramid_delta_level = std::max(1, pyramid_delta_level);
        options.pyramid_max_level = std::max(0, pyramid_max_level - 1);

        options.median_filter = median_filter;
        options.sample_radius_for_lidar = sample_radius_for_lidar;

        options.init_from_visible_map = init_from_visible_map;
        options.init_from_delaunay = init_from_delaunay;
        options.init_from_global_map = init_from_global_map;
        options.init_from_model = init_from_model;
        options.init_from_dense_points = init_from_dense_points;
        options.init_from_rgbd = (format.compare("rgbd") == 0);
        if (options.init_from_rgbd) {
            options.random_angle1_range = DegToRad(5.0f);
            options.random_angle2_range = DegToRad(5.0f);
            // options.geom_consistency_regularizer = 0.2f;
            options.geom_consistency = true;
            options.num_iter_geom_consistency = 1;
            options.init_from_dense_points = false;
            options.pyramid_max_level = 0;
        } else if (options.init_from_model || options.init_from_dense_points) {
            // options.random_depth_ratio = 0.004f;
            // options.random_angle1_range = DegToRad(20.0f);
            // options.random_angle2_range = DegToRad(12.0f);
            options.geom_consistency_regularizer = 0.1f;
            options.conf_filter = false;
            options.pyramid_max_level = 0;
            // options.filter = false;
            // options.geo_filter = false;
        }

        options.patch_match_fusion = patch_match_fusion;
        options.save_depth_map = save_depth_map;

        options.fused_delaunay_sample = fused_delaunay_sample;
        options.fused_diff_depth = fused_diff_depth;
        options.fused_dist_insert = fused_dist_insert;

        if (options.patch_match_fusion && min_num_visible_images > 1){
            options.filter_min_num_consistent = 
                std::max(min_num_visible_images - 1, filter_min_num_consistent);
        }

        options.outlier_removal = outlier_removal && 
            (options.filter_min_num_consistent == 1);
        options.nb_neighbors = nb_neighbors;
        options.max_spacing_factor = max_spacing_factor;
        options.outlier_percent = outlier_percent;
        options.outlier_max_density = outlier_max_density;
        options.step_size = step_size;
        options.roi_fuse = roi_fuse;
        options.roi_box_width = roi_box_width * 2;
        options.roi_box_factor = roi_box_factor * 2;

        options.plane_optimization = plane_optimization;
        options.plane_dist_threld =  plane_dist_threld;
        options.plane_raidus_factor = plane_raidus_factor;

        options.propagate_depth = propagate_depth;
        options.refine_with_semantic = refine_with_semantic;

        float get_max_ram;
        if (GetAvailableMemory(get_max_ram) && max_ram < 0){
            options.max_ram = get_max_ram;
        } else {
            options.max_ram = max_ram;
        }
        options.max_gpu_memory = max_gpu_memory;
        options.gpu_memory_factor = gpu_memory_factor;

        std::vector<uint8_t> label_ids;
        if (ExistsFile(black_list_file_path)) {
            LoadSemanticLabels(black_list_file_path, label_ids);
        }


        // options.Print();
        size_t num_reconstruction = 0;
        for (size_t reconstruction_idx = 0; ;reconstruction_idx++) {
            const auto& reconstruction_path = 
                JoinPaths(workspace_path, std::to_string(reconstruction_idx));
            if (!ExistsDir(reconstruction_path)) {
                break;
            }
            if (options.map_update && options.filter && options.geo_filter && options.patch_match_fusion){
                const auto& ori_reconstruction_path = 
                    JoinPaths(ori_workspace_path, std::to_string(reconstruction_idx));
                const auto& ori_fused_path = JoinPaths(ori_reconstruction_path, DENSE_DIR, FUSION_NAME);
                const auto& new_fused_path = JoinPaths(reconstruction_path, DENSE_DIR, FUSION_NAME);
                CHECK(ExistsFile(ori_fused_path) && ExistsFile(ori_fused_path + ".vis"));
                boost::filesystem::copy_file(ori_fused_path,
                                             JoinPaths(reconstruction_path, DENSE_DIR, FUSION_NAME), 
                                             boost::filesystem::copy_option::overwrite_if_exists);
                boost::filesystem::copy_file(ori_fused_path + ".vis", new_fused_path + ".vis" , 
                                             boost::filesystem::copy_option::overwrite_if_exists);
                if (ExistsFile(ori_fused_path + ".wgt")){
                    boost::filesystem::copy_file(ori_fused_path + ".wgt", new_fused_path + ".wgt", 
                                             boost::filesystem::copy_option::overwrite_if_exists);
                }
                if (ExistsFile(ori_fused_path + ".sco")){
                    boost::filesystem::copy_file(ori_fused_path + ".sco", new_fused_path + ".sco", 
                                             boost::filesystem::copy_option::overwrite_if_exists);
                }
            }
            num_reconstruction++;
        }

        size_t reconstruction_begin = reconstrction_idx < 0 ? 0 : reconstrction_idx;
        num_reconstruction = reconstrction_idx < 0 ? num_reconstruction : reconstrction_idx+1;
        for (size_t reconstruction_idx = reconstruction_begin; reconstruction_idx < num_reconstruction; 
            reconstruction_idx++) {
            float begin_memroy, end_memory;
            GetAvailableMemory(begin_memroy);

            PatchMatchController mvs(options, workspace_path, lidar_path, reconstruction_idx, cluster_idx);
            mvs.AddBlackList(label_ids);
            mvs.Start();
            mvs.Wait();

            GetAvailableMemory(end_memory);
            std::cout << StringPrintf("Patch Match Reconstruction %d Elapsed time: %.3f [minutes], \
                                      Memory: %3f (%3f - %3f) [G]", 
                                    reconstruction_idx, mvs.GetTimer().ElapsedMinutes(), 
                                    (begin_memroy - end_memory), begin_memroy, end_memory).c_str()
                    << std::endl;
        }
    } else {
        PatchMatchOptions options;
        options.verbose = verbose;
        options.random_optimization = random_optimization;
        options.local_optimization = local_optimization;
        options.est_curvature = est_curvature;
        options.window_radius = window_radius;
        options.window_step = window_step;
        options.plane_regularizer = plane_regularizer;
        options.num_good_hypothesis = num_good_hypothesis;
        options.num_bad_hypothesis = num_bad_hypothesis;
        options.geom_consistency = geom_consistency;
        options.num_iter_geom_consistency = num_iter_geom_consistency;
        options.geom_consistency_regularizer = geom_consistency_regularizer;
        options.num_iterations = num_iterations_patch_match;
        options.max_num_src_images = max_num_src_images;

        options.filter_min_triangulation_angle = 3.0f;
        options.filter_min_ncc = 0.2f;
        options.random_smooth_bonus = random_smooth_bonus;
        options.random_depth_ratio = random_depth_ratio;

        options.pyramid_delta_level = std::max(1, pyramid_delta_level);
        options.pyramid_max_level = std::max(0, pyramid_max_level - 1);

        options.filter = filter;
        options.conf_filter = conf_filter;
        options.geo_filter = geo_filter;
        options.conf_threshold = conf_threshold;
        options.depth_diff_threshold = depth_diff_threshold;
        // options.depth_adjust = false;

        options.median_filter = median_filter;

        options.init_from_visible_map = init_from_visible_map;
        options.init_from_delaunay = init_from_delaunay;
        options.init_from_global_map = init_from_global_map;
        options.init_from_model = init_from_model;

        options.max_image_size = 
            static_cast<int>(param.GetArgument("max_image_size", -1));

        // options.Print();

        PanoramaPatchMatchController mvs(options, image_path, workspace_path, reconstrction_idx);
        mvs.Start();
        mvs.Wait();

        std::cout << StringPrintf("Patch Match Reconstruction 0 Elapsed time: %.3f [minutes]", 
                                mvs.GetTimer().ElapsedMinutes()).c_str()
                  << std::endl;
    }
    std::cout << StringPrintf("Patch Match Elapsed time: %.3f [minutes]", 
                              patchmatch_timer.ElapsedMinutes()).c_str()
                    << std::endl;

    return StateCode::SUCCESS;
}
