#Copyright (c) 2019, SenseTime Group.
#All rights reserved.

# all source files in example and subdirectory of example
file ( GLOB_RECURSE ALL_TEST_CPPS
    test_dense_to_cadmap.cc
    test_dom_generation.cc
    test_obj2dae.cc
    test_obj2ply.cc
    test_simplify.cc
    test_transformation.cc
    test_utm_converter.cc
    # test_rgbd_converter.cc
    test_point_cloud_filter.cc
    test_point_cloud_simplify.cc
    test_point_cloud_improve.cc
    # test_integration.cc
    test_plane_generation.cc
    test_dense_to_heatmap.cc
    test_cadmap_to_linemap.cc
    test_reconstruction_cluster.cc
    # test_mesh_tile.cc
    # test_dense_map_update.cc
    test_dense_convert_panorama.cc
    test_ply2las.cc
    test_ply2pnts.cc
    test_merge_reconstruction.cc
    test_ground_area.cc
    test_undistort_fisheye.cc
    test_geojson_converter.cc
    test_change_detection.cc
    test_merge_dense.cc
    # color_harmonization.cc
    test_epsg4547_trans.cc
    # test_pcd.cc
    # test_pcd2rig.cc
    # test_lidar_to_img.cc
    # test_lidar_init.cc
    # test_output_pointcloud.cc
    # test_lidar_convert.cc
    )

include_directories(${PROJECT_SOURCE_DIR}/src/)
include_directories(${PROJECT_SOURCE_DIR}/3rdParty/LasTool/)
include_directories(${PROJECT_SOURCE_DIR}/3rdParty/PoissonDiskSampling/)
include_directories(${PROJECT_SOURCE_DIR}/3rdParty/JSON/)

# FIND_PACKAGE(OpenCV REQUIRED)
# FIND_PACKAGE(VTK NO_MODULE)

# generate executable files for each  or .cc file
foreach ( ONE_TEST_CPP ${ALL_TEST_CPPS} )
    get_filename_component(EXAMPLE_NAME ${ONE_TEST_CPP} NAME_WE)
    message ( STATUS "generate example ${EXAMPLE_NAME}" )
    add_executable(${EXAMPLE_NAME} ${ONE_TEST_CPP})
    target_link_libraries(${EXAMPLE_NAME} ${SENSEMAP_SHARED_LIBNAME} ${YAML_CPP_LIBRARIES} LasLib pds_lib)
    # if (VTK_FOUND)
    #     target_link_libraries(${EXAMPLE_NAME} ${VTK_LIBRARIES})
    # endif()
    if (TBB_FOUND)
        # add_definitions(-DCGAL_LINKED_WITH_TBB)
        include(CGAL_TBB_support)
        # include(${TBB_USE_FILE})
        target_link_libraries(${EXAMPLE_NAME} ${TBB_LIBRARIES})
    endif()
endforeach ()

