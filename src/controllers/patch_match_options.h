//Copyright (c) 2019, SenseTime Group.
//All rights reserved.

#ifndef SENSEMAP_MVS_PATCH_MATCH_OPTIONS_H_
#define SENSEMAP_MVS_PATCH_MATCH_OPTIONS_H_

#include <iostream>

#include "util/types.h"
#include "util/math.h"
#include "util/logging.h"
#include "util/mat.h"
#include "util/bitmap.h"

namespace sensemap {
namespace mvs {

struct PatchMatchACMMOptions;
class DepthMap;
class NormalMap;
class Image;

struct DelaunayOptions {
    bool sampInsert = true;
    int num_isolated_pieces = 0;
    float dist_insert = 5.0f;
    float diff_depth = 0.01f;
    float decimate_mesh = 1.0f;
    float remove_spurious = 2.0f;
    float sigma = 1.0f;
    unsigned remove_spikes = 1;
    unsigned close_holes = 30;
    unsigned smooth_mesh = 10;
    unsigned fix_mesh = 0;
    // adaptive delaunay parameter.
    unsigned adaptive_insert = 0;
    bool roi_mesh = false;
    float roi_box_width = -1.f;
    float roi_box_factor = -1.f;
};

struct PatchMatchOptions {
    
    std::string image_type = "perspective";

    std::string output_type = "";

    // Calibration file of lidar2camera.
    std::string lidar2cam_calibfile = "";

    std::string camera_rig_params = "";

    // Maximum image size in either dimension.
    int max_image_size = -1;

    // Whether to use the GPU for patch match.
    bool use_gpu = true;

    // Index of the GPU used for patch match. For multi-GPU patch match,
    // you should separate multiple GPU indices by comma, e.g., "0,1,2,3".
    std::string gpu_index = "-1";

    // Index of the Thread used for patch match.
    int thread_index = 0;

    bool map_update = false;

    bool random_optimization = true;
    bool local_optimization = true;

    bool est_curvature = false;
    
    // Depth range in which to randomly sample depth hypotheses.
    double depth_min = -1.0;
    double depth_max = -1.0;

    // Half window size to compute NCC photo-consistency cost.
    int window_radius = 5;

    // Number of pixels to skip when computing NCC. For a value of 1, every
    // pixel is used to compute the NCC. For larger values, only every n-th row
    // and column is used and the computation speed thereby increases roughly by
    // a factor of window_step^2. Note that not all combinations of window sizes
    // and steps produce nice results, especially if the step is greather than 2.
    int window_step = 2;

    // Parameters for bilaterally weighted NCC.
    // double sigma_spatial = -1;
    double sigma_spatial = window_radius;
    double sigma_color = 0.2f;

    // Number of random samples to draw in Monte Carlo sampling.
    int num_samples = 15;

    // Spread of the NCC likelihood function.
    double ncc_sigma = 0.6f;

    // Minimum triangulation angle in degrees.
    double min_triangulation_angle = 1.0f;

    // Spread of the incident angle likelihood function.
    double incident_angle_sigma = 0.9f;

    // Number of coordinate descent iterations. Each iteration consists
    // of four sweeps from left to right, top to bottom, and vice versa.
    int num_iterations = 5;

    // Whether to add a regularized geometric consistency term to the cost
    // function. If true, the `depth_maps` and `normal_maps` must not be null.
    bool geom_consistency = true;

    int num_iter_geom_consistency = 1;

    // The relative weight of the geometric consistency term w.r.t. to
    // the photo-consistency term.
    double geom_consistency_regularizer = 0.3f;

    // Maximum geometric consistency cost in terms of the forward-backward
    // reprojection error in pixels.
    double geom_consistency_max_cost = 3.0f;

    bool plane_regularizer = false;

    // Whether to enable filtering.
    bool filter = true;

    // Minimum NCC coefficient for pixel to be photo-consistent.
    double filter_min_ncc = 0.1f;

    // Minimum triangulation angle to be stable.
    double filter_min_triangulation_angle = 3.0f;

    // Minimum number of source images have to be consistent
    // for pixel not to be filtered.
    int filter_min_num_consistent = 1;

    // Maximum forward-backward reprojection error for pixel
    // to be geometrically consistent.
    double filter_geom_consistency_max_cost = 1.0f;

    double max_normal_error = 15.0f;

    // Cache size in gigabytes for patch match, which keeps the bitmaps, depth
    // maps, and normal maps of this number of images in memory. A higher value
    // leads to less disk access and faster computation, while a lower value
    // leads to reduced memory usage. Note that a single image can consume a lot
    // of memory, if the consistency graph is dense.
    double cache_size = 32.0;

    // Whether to write the consistency graph.
    bool write_consistency_graph = false;

    bool init_from_visible_map = true;
    bool init_from_delaunay = true;
    
    bool init_from_global_map = false;

    bool init_from_model = false;
    
    bool init_from_dense_points = false;

    bool init_from_rgbd = false;

    bool init_depth_random = !geom_consistency;

    float random_depth_ratio = 0.004f;
    float random_angle1_range = DegToRad(20.0f);
    float random_angle2_range = DegToRad(12.0f);
    float random_smooth_bonus = 0.93f;

    // the downsampling factor for each pyramid level.
    double downsample_factor = 0.5f;

    // the difference of photometric consistency cost for detail restore.
    double diff_photometric_consistency = 0.1f;

    float ncc_thres_refine = 0.03;

    // initial matching cost threshold.
    double init_ncc_matching_cost = 0.8f;
    float th_mc = init_ncc_matching_cost;

    // bad matching cost threshold.
    double max_ncc_matching_cost = 1.2f;

    // constant factor for good matching cost boundary.
    double alpha = 90.0f;
    
    // constant factor for the confidence of a matching cost.
    double beta = 0.3f;

    int thk = 4;

    // For a pecific view Ij, there should exist more than n1 good matching 
    // costs. Also, there should be less than n2 bad matching costs. Then
    // view Ij be considered reliable for view selection.
    int num_good_hypothesis = 2;
    int num_bad_hypothesis = 3;

    // If max_num_src_images <= 0, all images will used as source images;
    // else, automatically select max_num_src_images source images.

    // TODO.
    // size_t max_num_src_images = 20; // colmap
    size_t max_num_src_images = 8; // openmvs

    // Minimum number of agreeing views to validata a depth.
    size_t min_num_src_images = 1;

    // Minimum number of views so that the point is considered for 
    // approximating the depth-maps.
    size_t min_views_trust_point = 2;

    // Min angle for accepting the depth triangulation.
    double min_angle = 3.0;

    // Max angle for accepting the depth triangulation.
    double max_angle = 65.0;

    // Optimal angle for computing the depth triangulation.
    double optimal_angle = 15.0;

    // Min shared area for accepting the depth triangulation.
    double min_area = 0.05;

    // Whether filter the pixels of low geom-consistency
    bool geo_filter = false;

    // Maximum relative depth error for pixel.
    // |ref_depth - pro_src_dpth| / ref_depth
    float depth_diff_threshold = 0.02;

    // Whether filter the pixels of low confidence
    bool conf_filter = false;

    // Minimum confidence for pixel to give up
    float conf_threshold = 0.5;

    // The maximum number of layers of the Multi scale
    int pyramid_max_level = 0;

    // Number of layers per multi-scale iter
    // In conjunction with max_level, should ensure that level=0
    int pyramid_delta_level = 1;

    bool verbose = false;

    bool median_filter = false;

    bool has_prior_depth = false;

    // Voxel sampling size for lidar points.
    double sample_radius_for_lidar = 10;

    DelaunayOptions delaunay_options;

    bool patch_match_fusion = true;
    bool save_depth_map = false;

    // delaunay insert param
    bool fused_delaunay_sample = false;
    float fused_dist_insert = 5.0f;
    float fused_diff_depth = 0.01f;

    // fused_points remove outlier
    bool outlier_removal = true;

    int nb_neighbors = 6;    
    double max_spacing_factor = 6.0;

    float outlier_percent = 0.9;
    float outlier_max_density = 0.01;
    float voxel_factor = 25.0;

    bool plane_optimization = true;
    float plane_dist_threld = 0.9f;
    float plane_raidus_factor = 100.0f;

    int step_size = 2;

    bool roi_fuse = false;
    float roi_box_width = -1.f;
    float roi_box_factor = -1.f;
    
    // Whether to fuse semantic info.
    double num_consistent_semantic_ratio = 0.3;

    bool propagate_depth = false;

    bool refine_with_semantic = false;

    float max_ram = -1.0f;
    
    float max_gpu_memory = 6.0f;
    float gpu_memory_factor = 0.7f;
    int cuda_maxTexture1DLayered_0 = std::numeric_limits<int>::max();
    int cuda_maxTexture1DLayered_1 = std::numeric_limits<int>::max();
    int cuda_maxTexture1DLayered_2 = std::numeric_limits<int>::max();

    void Print() const;
    bool Check() const;

    PatchMatchACMMOptions PatchMatchACMM() const;
};

struct Problem {
    // Index of the reference image.
    int ref_image_idx = -1;

    // Indices of the source images.
    std::vector<int> src_image_idxs;

    // Indices of the source images for crossfilter.
    std::vector<int> src_image_extend_idxs;

    // Scale of the source images.
    std::vector<float> src_image_scales;

    // Depth map propagated from neighbor images.
    std::vector<bool>* flag_depth_maps;

    // Input images for the photometric consistency term.
    std::vector<Image>* images = nullptr;

    // Input depth maps for the geometric consistency term.
    std::vector<DepthMap>* depth_maps = nullptr;

    // Input prior depths from rgbd/ir/kinect camera
    std::shared_ptr<DepthMap> prior_depth_map = nullptr;

    // Input prior depths from rgbd/ir/kinect camera
    // std::shared_ptr<DepthMap> prior_wgt_map = nullptr;
    std::vector<DepthMap>* prior_wgt_maps = nullptr;

    // Input normal maps for the geometric consistency term.
    std::vector<NormalMap>* normal_maps = nullptr;

    // Input confidence maps for the geometric consistency term.
    std::vector<Mat<float>>* conf_maps = nullptr;

    // Input semantic maps.
    std::vector<Bitmap>* semantic_maps = nullptr;

    // Input semantic maps.
    std::vector<Bitmap>* mask_maps = nullptr;

    // Input gradient maps.
    std::vector<DepthMap>* gradient_maps = nullptr;

    // Print the configuration to stdout.
    void Print() const;
    bool Check(bool geom_consistency = false) const;
};

struct DepthMapInfo{
    size_t width = 0;
    size_t height = 0;
    float depth_min = -1;
    float depth_max = -1;
};

}    
}

#endif