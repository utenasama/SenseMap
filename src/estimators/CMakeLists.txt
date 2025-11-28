#Copyright (c) 2019, SenseTime Group.
#All rights reserved.

set(FOLDER_NAME "estimators")

SENSEMAP_ADD_SOURCES(
    global_pose_estimation/robust_rotation.cc
    global_pose_estimation/nonlinear_position.cc
    absolute_pose.cc
    affine_transform.cc
    essential_matrix.cc
    fundamental_matrix.cc
    homography_matrix.cc
    relative_pose.cc
    scale_selection.cc
    triangulation.cc
    two_view_geometry.cc
    utils.cc
    mappoint_alignment.cc
    reconstruction_aligner.cc
    generalized_absolute_pose.cc
    generalized_relative_pose.cc
    generalized_absolute_pose_coeffs.cc
    motion_average/rotation_average.cc
    motion_average/similarity_average.cc
    motion_average/translation_average.cc
    camera_alignment.cc
    plane.cc
    plane_estimator.cc
    scale_gravity.cc
)
