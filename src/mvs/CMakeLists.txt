#Copyright (c) 2019, SenseTime Group.
#All rights reserved.

set(FOLDER_NAME "mvs")

set (SOURCE_FILES
    consistency_graph.cc
    depth_map.cc
    fusion.cc
    image.cc
    # marching_cube.cc
    # meshing.cc
    model.cc
    normal_map.cc
    utils.cc
    workspace.cc
    point_cloud_cluster.cc
    # camera_cluster.cc
    mesh_cluster.cc
    point_cloud_filter.cc
    integration.cc
    delaunay/IBFS/IBFS.cpp
    delaunay/AABB.h
    delaunay/Plane.h
    delaunay/Ray.h
    delaunay/Sphere.h
    delaunay/PointCloud.cc
    delaunay/delaunay_triangulation.cc
    integration/UniformTSDFVolume.cpp
    integration/ScalableTSDFVolume.cpp
    reconstruction_cluster.cc
    mvs_cluster.cc
    delaunay_meshing.cc
)

SENSEMAP_ADD_SOURCES(
    ${SOURCE_FILES}
)

if(CUDA_ENABLED)
    SENSEMAP_ADD_CUDA_SOURCES(
        gpu_mat_prng.cu
        gpu_mat_ref_image.cu
        kernel_functions.cu
        patch_match_cuda.cu
        point_cloud_filter_cuda.cu
    )
endif()
