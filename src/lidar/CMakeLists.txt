#Copyright (c) 2023, SenseTime Group.
#All rights reserved.

set(FOLDER_NAME "lidar")

SENSEMAP_ADD_SOURCES(
    absolute_pose.cc
    lidar_correspondence.cc
    lidar_sweep.cc
    octree.cc
    pcd.cc
    utils.cc
    voxel.cc
    voxel_map.cc
)
