#Copyright (c) 2019, SenseTime Group.
#All rights reserved.

set(FOLDER_NAME "optim")

SENSEMAP_ADD_SOURCES(
    ransac/combination_sampler.cc
    ransac/random_sampler.cc
    ransac/support_measurement.cc
    global_motions/L1_scale_optimizer.cc
    global_motions/L1_translation_optimizer.cc
    global_motions/robust_rotation_optimizer.cc
    global_motions/utils.cc
    cluster_motion_averager.cc
    solver/l1_solver.h
    solver/sparse_cholesky_llt.cc
    bundle_adjustment.cc
    # block_bundle_adjustment.cc
    pose_graph_optimizer.cc
    cluster_merge/cluster_merge_optimizer.cc
)