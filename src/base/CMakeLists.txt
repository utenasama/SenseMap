#Copyright (c) 2019, SenseTime Group.
#All rights reserved.

set(FOLDER_NAME "base")

#file ( GLOB_RECURSE BASE_FILES
#    "*.cc"
#    )
#set (BASE_FILES
#    camera.cc
#    image.cc
#)

SENSEMAP_ADD_SOURCES(
    camera_database.cc
    camera_models.cc
    camera.cc
    essential_matrix.cc
    homography_matrix.cc
    image_reader.cc
    image.cc
    mappoint.cc
    point2d.cc
    polynomial.cc
    pose.cc
    projection.cc
    reconstruction_manager.cc
    reconstruction.cc
    similarity_transform.cc
    track.cc
    triangulation.cc
    visibility_pyramid.cc
    undistortion.cc
)

if(CUDA_ENABLED)
    SENSEMAP_ADD_CUDA_SOURCES(
    )
endif()
