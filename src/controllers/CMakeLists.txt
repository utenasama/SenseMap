set(FOLDER_NAME "controllers")

#file ( GLOB_RECURSE FEATURE_FILES
#    "*.cc"
#    )

SENSEMAP_ADD_SOURCES(
    mapper_options.cc
    mapper_controller.cc
    cluster_mapper_controller.cc
    incremental_mapper_controller.cc
    global_mapper_controller.cc
    hybrid_mapper_controller.cc
    patch_match_controller.cc
    patch_match_options.cc
    directed_mapper_controller.cc
)