set(FOLDER_NAME "feature")

#file ( GLOB_RECURSE FEATURE_FILES
#    "*.cc"
#    )

SENSEMAP_ADD_SOURCES(
    types.cc
    extraction.cc
    sift.cc
    utils.cc
    matching_utils.cc
    feature_matcher.cc
    global_feature_extraction.cc
)