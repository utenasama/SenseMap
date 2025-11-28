#Copyright (c) 2019, SenseTime Group.
#All rights reserved.
if(IS_MSVC)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /W3")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W3")
elseif(IS_GNU OR IS_CLANG)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
endif()

include_directories(./)

add_subdirectory(base)
add_subdirectory(cluster)
add_subdirectory(container)
add_subdirectory(controllers)
add_subdirectory(estimators)
add_subdirectory(feature)
add_subdirectory(graph)
add_subdirectory(optim)
add_subdirectory(sfm)
add_subdirectory(mvs)
# add_subdirectory(texture)
add_subdirectory(util)
add_subdirectory(dom)
add_subdirectory(retrieval)
add_subdirectory(RGBDAlign)
add_subdirectory(lidar)


set (SENSEMAP_CUDA_LIBNAME "SenseMap_cuda")
if(CUDA_ENABLED)
    # Use a separate stream per thread to allow for concurrent kernel execution
    # between multiple threads on the same device.
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};--default-stream;per-thread")

    # Fix for Ubuntu 16.04.
    add_definitions("-D_MWAITXINTRIN_H_INCLUDED")

    if(MSVC)
        # Avoid pulling in too many header files through <windows.h>
        add_definitions("-DWIN32_LEAN_AND_MEAN")
    endif()

    SENSEMAP_ADD_CUDA_LIBRARY(${SENSEMAP_CUDA_LIBNAME} ${SENSEMAP_CUDA_SOURCES})
endif()

#message(STATUS "Source files:" ${SENSEMAP_SOURCES})
message(STATUS "Source files:")
foreach ( ONE_SOURCE ${SENSEMAP_SOURCES} )
    message ( STATUS " ${ONE_SOURCE}" )
endforeach ()

set (SENSEMAP_SHARED_LIBNAME "SenseMap" CACHE INTERNAL "SenseMap")

# add_library(${SENSEMAP_STATIC_LIBNAME} STATIC
#     ${SENSEMAP_SOURCES}
# )
add_library(${SENSEMAP_SHARED_LIBNAME} SHARED
    ${SENSEMAP_SOURCES}
)

### link 3rdparty library
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

message(STATUS "lib ${SENSEMAP_SHARED_LIBNAME}")
message(STATUS "omp ${OpenMP_omp_LIBRARY}")

if(CUDA_ENABLED)
    target_link_libraries(${SENSEMAP_SHARED_LIBNAME} ${SENSEMAP_CUDA_LIBNAME})
endif()

set(THIRDPART_LIBRARY "vlfeat")
list(APPEND THIRDPART_LIBRARY "apriltags")
list(APPEND THIRDPART_LIBRARY "flann")
list(APPEND THIRDPART_LIBRARY "graclus")
list(APPEND THIRDPART_LIBRARY "GLU")
list(APPEND THIRDPART_LIBRARY "poisson_recon")
# list(APPEND THIRDPART_LIBRARY "gco_lib")

if(CUDA_ENABLED)
    list(APPEND THIRDPART_LIBRARY "sift_gpu")
endif()

target_link_libraries(${SENSEMAP_SHARED_LIBNAME}
    ${CMAKE_DL_LIBS}
    ${Boost_LIBRARIES}
    ${Boost_PROGRAM_OPTIONS_LIBRARIES}
    ${Boost_FILESYSTEM_LIBRARIES}
    ${Boost_SYSTEM_LIBRARIES}
    ${Boost_REGEX_LIBRARY}
    ${GLOG_LIBRARIES}
    ${FREEIMAGE_LIBRARIES}
    ${CERES_LIBRARIES}
    ${THIRDPART_LIBRARY}
    ${SUITESPARSE_LIBRARIES}
    ${OpenCV_LIBRARIES}
    ${CGAL_LIBRARIES}
    ${ZLIB_LIBRARIES}
    # ${GeoTIFF_LIBRARIES}
    ${GDAL_LIBRARIES}
    jsoncpp
)

if (TBB_FOUND)
    # add_definitions(-DCGAL_LINKED_WITH_TBB)
    include(CGAL_TBB_support)
    # include(${TBB_USE_FILE})
    target_link_libraries(${SENSEMAP_SHARED_LIBNAME} ${TBB_LIBRARIES})
endif()