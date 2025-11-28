#Copyright (c) 2019, SenseTime Group.
#All rights reserved.

# all source files in example and subdirectory of example
file ( GLOB_RECURSE ALL_TEST_CPPS
    "test_all_configure_outside.cc"
    "test_all_configure_hybrid.cc"
    "test_all_configure_hybrid_new.cc"
    "test_merge_clusters.cc"
    "test_feature_extraction.cc"
    "test_feature_match.cc"
    "test_mapping.cc"
    "test_retrieval.cc"
    "test_pre_mapping.cc"
    )

include_directories(${PROJECT_SOURCE_DIR}/src/)

# FIND_PACKAGE(OpenCV REQUIRED)
# FIND_PACKAGE(VTK NO_MODULE)

# generate executable files for each  or .cc file
foreach ( ONE_TEST_CPP ${ALL_TEST_CPPS} )
    get_filename_component(EXAMPLE_NAME ${ONE_TEST_CPP} NAME_WE)
    message ( STATUS "generate example ${EXAMPLE_NAME}" )
    add_executable(${EXAMPLE_NAME} ${ONE_TEST_CPP})
    target_link_libraries(${EXAMPLE_NAME} ${SENSEMAP_SHARED_LIBNAME} ${OpenCV_LIBRARIES} ${Boost_UNIT_TEST_FRAMEWORK_LIBRARIES} ${YAML_CPP_LIBRARIES})
    if (ENCRYPT_ENABLE)
        target_link_libraries(${EXAMPLE_NAME} checker)
    endif()
    if (VTK_FOUND)
        target_link_libraries(${EXAMPLE_NAME} ${VTK_LIBRARIES})
    endif()
endforeach ()

