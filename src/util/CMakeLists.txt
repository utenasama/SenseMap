#Copyright (c) 2019, SenseTime Group.
#All rights reserved.

set(FOLDER_NAME "util")

SENSEMAP_ADD_SOURCES(
    bitmap.cc
    common.cc
    camera_specs.cc
    ceres_types.h
    graph.cc
    image_pair.cc
    logging.cc
    mat.cc
    math.cc
    misc.cc
    obj.cc
#    octree.cc
    mesh_info.cc
    random.cc
    rgbd_helper.cc
    cross_warp_helper.cc
    string.cc
    threading.cc
    timer.cc
    panorama.cc
    large_fov_image.cc
    piecewise_image.cc
    ply.cc
    depth_mesh.cc
    semantic_table.cc
    #ORB Map sparsification
    sparsification/scoring/scoring-function.h
    sparsification/scoring/observation-count-scoring.h
    sparsification/descriptor-scoring-functions/descriptor-variance-scoring.h
    sparsification/descriptor-scoring-functions/descriptor-scoring-function.h
    sparsification/descriptor-scoring-functions/sift-descriptor-scoring-function.h
    sparsification/descriptor-scoring-functions/sift-descriptor-variance-scoring.h
    sparsification/descriptor-scoring-functions/map-sparsification_macros.h
    sparsification/cost-functions/sampling-cost.h
    sparsification/cost-functions/min-keypoints-per-keyframe-cost.h
    sparsification/heuristic-sampling.h sparsification/heuristic-sampling.cc
    sparsification/sampler-base.h sparsification/sampler-base.cc
    sparsification/sampler-factory.h sparsification/sampler-factory.cc
    tinyxml.cc
    tinystr.cc
    tinyxmlerror.cc
    tinyxmlparser.cc
    gps_reader.cc
    kmeans.cc
    jacobian.cc
    imageconvert.cpp
    tinyxml2.cpp
    TinyEXIF.cpp
    roi_box.h
    proc.h
    tag_scale_recover.cc
    histogram.cc
    exception_handler.cc
)

if (CUDA_ENABLED)
    SENSEMAP_ADD_CUDA_SOURCES(
        cuda.h cuda.cc
        cudacc.h cudacc.cc
        gpu_mat.h
    )
endif ()
