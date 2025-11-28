#Copyright (c) 2019, SenseTime Group.
#All rights reserved.

# all source files in example and subdirectory of example
file ( GLOB_RECURSE ALL_TEST_CPPS
    "test_param_parser.cc"
    "test_retriangulate.cc"
    "test_camera_info.cc"
    "test_pose_output.cc"
    "test_convert_export_config.cc"
    "test_export_normal_reconstruction.cc"
    "test_generate_normal_images.cc"
    "test_free_gba.cc"
    "test_map_update_loc_gt.cc"
    # "test_plot_match_polarLine.cc"
    "rgbd_bin_packer_hybrid.cc"
    "rgbd_bin_packer_dior.cc"
    "test_triangulator_apriltag.cc"
    "transform_rtk_data.cc"
    "test_reconstruction_error.cc"
    "transform_data_from_gps_tool.cc"
    "test_gps_alignment.cc"
    "test_rtk_reconstruction.cc"
    "test_gps_bound_box.cc"
    "test_convert_perspective_to_panorama.cc"
    "partition_model_by_floor.cc"
    "create_vlad_code_book.cc"
    "test_AR_car.cc"
    "generate_binary_descriptor_pattern.cc"
    "test_read_binary_descriptor_pattern.cc"
    "feature_projection_matrix_training.cc"
    "test_scale_gravity_recovery.cc"
    "test_loop_closure.cc"
    "convert_to_tum.cc"
    "test_export_calibs.cc"
    "test_convert_pose.cc"
    "test_read_gps.cc"
    "test_rgbd_alignment.cc"
    "test_rgbd_alignment_multisegment.cc"
    "test_point_alignment.cc"
    "test_feature_apriltag.cc"
    "test_filter_scene_graph.cc"
    "test_convert_model.cc"
    "test_align_reconstruction_with_rtk.cc"
    "test_concatenate_trans.cc"
    "test_gps_trans_txt_to_yaml.cc"
    "test_sparse_to_cadmap.cc"
    "test_tag_gt_alignment.cc"
    "test_delete_images.cc"
    "test_create_loc_benchmark.cc"
    "test_gps_transform.cc"
    "align_reconstruction.cc"
    "align_merge_reconstruction.cc"
    "test_convert_rig.cc"
    "test_bound_box_filter.cc"
    )

include_directories(${PROJECT_SOURCE_DIR}/src/)

# FIND_PACKAGE(OpenCV REQUIRED)
# FIND_PACKAGE(VTK NO_MODULE)

# generate executable files for each  or .cc file
foreach ( ONE_TEST_CPP ${ALL_TEST_CPPS} )
    get_filename_component(EXAMPLE_NAME ${ONE_TEST_CPP} NAME_WE)
    message ( STATUS "generate example ${EXAMPLE_NAME}" )
    add_executable(${EXAMPLE_NAME} ${ONE_TEST_CPP})
    #target_link_libraries(test_opengv opengv)
    target_link_libraries(${EXAMPLE_NAME} ${SENSEMAP_SHARED_LIBNAME} ${OpenCV_LIBRARIES} ${Boost_UNIT_TEST_FRAMEWORK_LIBRARIES} ${YAML_CPP_LIBRARIES})
    if (VTK_FOUND)
        target_link_libraries(${EXAMPLE_NAME} ${VTK_LIBRARIES})
    endif()
endforeach ()

