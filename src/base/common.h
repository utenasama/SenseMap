//Copyright (c) 2019, SenseTime Group.
//All rights reserved.

#ifndef SENSEMAP_BASE_COMMON_H
#define SENSEMAP_BASE_COMMON_H

namespace sensemap {

#define IMAGES_DIR  "images"
#define IMAGES_ORIG_RES_DIR  "images_orig_res"
#define MASKS_DIR  "masks"
#define MASKS_ORIG_RES_DIR  "masks_orig_res"
#define SPARSE_DIR  "sparse"
#define SPARSE_ORIG_RES_DIR  "sparse_orig_res"
#define DENSE_DIR   "dense"
#define STEREO_DIR  "stereo"
#define DEPTHS_DIR   "depth_maps"
#define CHANGE_MASKS_DIR   "change_masks"
#define NORMALS_DIR  "normal_maps"
#define CONFS_DIR    "conf_maps"
#define CURVATURES_DIR "curvature_maps"
#define CONSISTENCY_DIR "consistency_graphs"
#define CLUSTER_DIR  "cluster"
#define SEMANTICS_DIR "semantics"
#define SEMANTICS_ORIG_RES_DIR "semantics_orig_res"
#define MERGE_DIR "merge"

#define DEPTH_EXT   "bin"
#define NORMAL_EXT   "bin"
#define MASK_EXT     "png"

#define PHOTOMETRIC_TYPE    "photometric"
#define GEOMETRIC_TYPE      "geometric"
#define REF_TYPE       "ref"

#define LIDAR_NAME "lidar.ply"
#define FUSION_NAME "fused.ply"
#define FUSION_SEM_NAME "fused_semvis.ply"
#define FILTER_FUSION_NAME "fused_filtered.ply"
#define FILTER_FUSION_SEM_NAME "fused_semvis_filtered.ply"
#define TRANS_FILTER_FUSION_NAME "fused_filtered_trans.ply"
#define MODEL_NAME "model.obj"
#define SEM_MODEL_NAME "model_sem.obj"
#define TRANS_MODEL_NAME "model_trans.obj"
#define TEX_MODEL_NAME "tex_model.obj"
#define TILTED_MODEL_NAME "model_tiled.obj"
#define ROI_BOX_NAME "box.txt"
#define RECT_CLUSTER_NAME "rect_cluster.txt"
#define ALIGNMENT_POSE_NAME "alignment.bin"

#define PATCH_MATCH_CLUATER_YAML "patch_match.yaml"
#define FUSION_CLUATER_YAML "fusion.yaml"
#define BOX_YAML "box.yaml"

#define WORKSPACE_ORI "-orig"

#define MAPPOINT_NAME "sparse.ply"

} // namespace sensemap

#endif
