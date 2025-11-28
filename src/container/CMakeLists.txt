set(FOLDER_NAME "container")

#file ( GLOB_RECURSE FEATURE_FILES
#    "*.cc"
#    )

SENSEMAP_ADD_SOURCES(
    feature_data_container.cc
    io.cc
    scene_graph_container.cc scene_graph_container.h
    match_data_container.cc match_data_container.h
)
