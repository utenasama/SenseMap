//Copyright (c) 2019, SenseTime Group.
//All rights reserved.

#include "util/misc.h"

#include "patch_match_options.h"

namespace sensemap {
namespace mvs {

void PatchMatchOptions::Print() const {
    PrintHeading2("PatchMatchOptions");

    PrintOption(image_type);
    PrintOption(max_image_size);
    PrintOption(use_gpu);
    PrintOption(gpu_index);
    PrintOption(depth_min);
    PrintOption(depth_max);
    PrintOption(window_radius);
    PrintOption(window_step);
    PrintOption(sigma_spatial);
    PrintOption(sigma_color);
    PrintOption(num_samples);
    PrintOption(ncc_sigma);
    PrintOption(min_triangulation_angle);
    PrintOption(incident_angle_sigma);
    PrintOption(num_iterations);
    PrintOption(plane_regularizer);
    PrintOption(geom_consistency);
    PrintOption(geom_consistency_regularizer);
    PrintOption(geom_consistency_max_cost);
    PrintOption(init_depth_random);
    PrintOption(filter);
    PrintOption(conf_filter);
    PrintOption(conf_threshold);
    PrintOption(filter_min_ncc);
    PrintOption(filter_min_triangulation_angle);
    PrintOption(filter_min_num_consistent);
    PrintOption(filter_geom_consistency_max_cost);
    PrintOption(max_normal_error);
    PrintOption(cache_size);
    PrintOption(write_consistency_graph);
    PrintOption(geo_filter);
    PrintOption(depth_diff_threshold);
    PrintOption(pyramid_max_level);
    PrintOption(pyramid_delta_level);
    PrintOption(random_optimization);
    PrintOption(fused_delaunay_sample);
    PrintOption(fused_dist_insert);
    PrintOption(fused_diff_depth);
    PrintOption(outlier_removal);
    PrintOption(nb_neighbors);
    PrintOption(max_spacing_factor);
    PrintOption(outlier_percent);
    PrintOption(outlier_max_density);
    PrintOption(voxel_factor);
    PrintOption(plane_optimization);
    PrintOption(plane_dist_threld);
    PrintOption(plane_raidus_factor);
    PrintOption(roi_fuse);
    PrintOption(roi_box_width);
    PrintOption(roi_box_factor);
    PrintOption(max_ram);
    PrintOption(max_gpu_memory);
    PrintOption(gpu_memory_factor);
    PrintOption(patch_match_fusion);
    PrintOption(step_size);
    PrintOption(save_depth_map);
    PrintOption(refine_with_semantic);
    PrintOption(propagate_depth);
    PrintOption(local_optimization);
}

bool PatchMatchOptions::Check() const {
    if (depth_min != -1.0f || depth_max != -1.0f) {
      CHECK_OPTION_LE(depth_min, depth_max);
      CHECK_OPTION_GE(depth_min, 0.0f);
    }
    CHECK_OPTION_LE(window_radius, 32);
    CHECK_OPTION_GT(sigma_color, 0.0f);
    CHECK_OPTION_GT(window_radius, 0);
    CHECK_OPTION_GT(window_step, 0);
    CHECK_OPTION_LE(window_step, 2);
    CHECK_OPTION_GT(num_samples, 0);
    CHECK_OPTION_GT(ncc_sigma, 0.0f);
    CHECK_OPTION_GE(min_triangulation_angle, 0.0f);
    CHECK_OPTION_LT(min_triangulation_angle, 180.0f);
    CHECK_OPTION_GT(incident_angle_sigma, 0.0f);
    // CHECK_OPTION_GT(num_iterations, 0);
    CHECK_OPTION_GE(geom_consistency_regularizer, 0.0f);
    CHECK_OPTION_GE(geom_consistency_max_cost, 0.0f);
    CHECK_OPTION_GE(filter_min_ncc, -1.0f);
    CHECK_OPTION_LE(filter_min_ncc, 1.0f);
    CHECK_OPTION_GE(filter_min_triangulation_angle, 0.0f);
    CHECK_OPTION_LE(filter_min_triangulation_angle, 180.0f);
    CHECK_OPTION_GE(filter_min_num_consistent, 0);
    CHECK_OPTION_GE(filter_geom_consistency_max_cost, 0.0f);
    CHECK_OPTION_GT(cache_size, 0);
    return true;
}

bool Problem::Check(bool geom_consistency) const {
    CHECK_OPTION_NE(ref_image_idx, -1);

    CHECK_OPTION(!src_image_idxs.empty());

    // CHECK_OPTION(ref_image != nullptr);
    // CHECK_OPTION(K_ref != nullptr);
    // CHECK_OPTION(R_ref != nullptr);
    // CHECK_OPTION(T_ref != nullptr);

    // CHECK_OPTION(src_images != nullptr);
    // CHECK_OPTION_EQ(src_image_idxs.size(), src_images->size());
    // CHECK_OPTION(Ks_src != nullptr);
    // CHECK_OPTION_EQ(src_image_idxs.size(), Ks_src->size());
    // CHECK_OPTION(Rs_src != nullptr);
    // CHECK_OPTION_EQ(src_image_idxs.size(), Rs_src->size());
    // CHECK_OPTION(Ts_src != nullptr);
    // CHECK_OPTION_EQ(src_image_idxs.size(), Ts_src->size());

    if (geom_consistency) {
        // CHECK_OPTION(ref_depth_map != nullptr);
        // CHECK_OPTION(ref_normal_map != nullptr);
        // CHECK_OPTION(src_depth_maps != nullptr);
        // CHECK_OPTION_EQ(src_image_idxs.size(), src_depth_maps->size());
    }

    return true;
}

void Problem::Print() const {
  PrintHeading2("PatchMatchProblem");

  PrintOption(ref_image_idx);

  std::cout << "src_image_idxs: ";
  if (!src_image_idxs.empty()) {
    for (size_t i = 0; i < src_image_idxs.size() - 1; ++i) {
      std::cout << src_image_idxs[i] << " ";
    }
    std::cout << src_image_idxs.back() << std::endl;
  } else {
    std::cout << std::endl;
  }
}

}
}