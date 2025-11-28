#Copyright (c) 2019, SenseTime Group.
#All rights reserved.

cmake_minimum_required(VERSION 3.8)
project(SenseMap)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

include(ExternalProject)

### default build release type, comment this in development stage
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
if (CMAKE_BUILD_TYPES MATCHES "Debug")
    add_definitions(-D_DEBUG)
endif()
MESSAGE(STATUS "Build type: " ${CMAKE_BUILD_TYPE})


# Include helper macros and commands, and allow the included file to override
# the CMake policies in this file
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/CMakeHelper.cmake NO_POLICY_SCOPE)



#options 
option(CUDA_ENABLED "Whether to enable CUDA, if available" ON)
set(CUDA_ARCHS "Auto" CACHE STRING "List of CUDA architectures for which to \
generate code, e.g., Auto, All, Maxwell, Pascal, ...")


# find packages
#openmp
FIND_PACKAGE( OpenMP REQUIRED)
if(OPENMP_FOUND)
    message("OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

set(SuiteSparse_DIR "*PATH TO YOUR QT QT5CONFIG FILE HERE*" CACHE PATH "Initial cahe" FORCE)
# Suitesparse
find_package(SuiteSparse REQUIRED)
if (SUITESPARSE_FOUND)
    # On Ubuntu the system install of SuiteSparse (v3.4.0) up to at least
    # Ubuntu 13.10 cannot be used to link shared libraries.
    if (BUILD_SHARED_LIBS AND
            SUITESPARSE_IS_BROKEN_SHARED_LINKING_UBUNTU_SYSTEM_VERSION)
        message(FATAL_ERROR "You are attempting to build Theia as a shared "
                "library on Ubuntu using a system package install of SuiteSparse "
                "3.4.0. This package is broken and does not support the "
                "construction of shared libraries (you can still build Theia as "
                "a static library).  If you wish to build a shared version of Theia "
                "you should uninstall the system install of SuiteSparse "
                "(libsuitesparse-dev) and perform a source install of SuiteSparse "
                "(we recommend that you use the latest version), "
                "see http://theia-solver.org/building.html for more information.")
    endif (BUILD_SHARED_LIBS AND
            SUITESPARSE_IS_BROKEN_SHARED_LINKING_UBUNTU_SYSTEM_VERSION)
    message("-- Found SuiteSparse ${SUITESPARSE_VERSION}")
    add_definitions(-DTHEIA_SUITESPARSE_VERSION="${SUITESPARSE_VERSION}")
else (SUITESPARSE_FOUND)
    # Disable use of SuiteSparse if it cannot be found and continue.
    message(FATAL_ERROR "Can't find SuiteSparse. This library is required "
            "for bundle adjustment and for solving convex optimization problems. "
            "Please set SUITESPARSE_INCLUDE_DIR & SUITESPARSE_LIBRARY")
endif (SUITESPARSE_FOUND)

### library dependencies
### Find Eigen under Mac
# find_package( PkgConfig )
#  pkg_check_modules( EIGEN3 REQUIRED eigen3 )
#  include_directories( ${EIGEN3_INCLUDE_DIRS} )
find_package (Eigen3 REQUIRED NO_MODULE)
MESSAGE(STATUS "Eigen path: " ${EIGEN3_INCLUDE_DIRS})

# find_package(GeoTIFF REQUIRED)
# message("geotiff inc: " ${GeoTIFF_INCLUDE_DIRS})
# message("geotiff lib: " ${GeoTIFF_LIBRARIES})

find_package(GDAL REQUIRED)
message("GDAL lib: " ${GDAL_LIBRARIES})

find_package (Ceres REQUIRED)
find_package(FreeImage REQUIRED)
find_package(Glog REQUIRED)
find_package(Boost REQUIRED COMPONENTS
        #      program_options
             filesystem
             graph
             regex
             system
             unit_test_framework)
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
FIND_PACKAGE(OpenCV REQUIRED)
find_package(CGAL REQUIRED)
find_package(ZLIB REQUIRED)
#include (${CGAL_USE_FILE})

find_package(jsoncpp REQUIRED)

find_package(TBB QUIET)

set(VCG_DIR "${PROJECT_SOURCE_DIR}/3rdParty/VCG")
# message("${VCG_DIR}")
find_package(VCG REQUIRED)

set(CUDA_MIN_VERSION "7.0")
if(CUDA_ENABLED)
    set(CUDA_SEPARABLE_COMPILATION ON)
    find_package(CUDA ${CUDA_MIN_VERSION} QUIET)
endif()

find_package(yaml-cpp REQUIRED)
message("yaml include: ${YAML_CPP_INCLUDE_DIR}")
message("yaml lib: ${YAML_CPP_LIBRARIES}")

#compiler configuration
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -fPIC -msse -msse2 -msse3 -msse4 -msse4a -Werror=return-type")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -O0 -g -ggdb")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -O3 -g")
#set(CMAKE_VERBOSE_MAKEFILE ON)

if(CUDA_FOUND)
    if(CUDA_ENABLED)
        add_definitions("-DCUDA_ENABLED")

        include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/SelectCudaComputeArch.cmake)

        CUDA_SELECT_NVCC_ARCH_FLAGS(CUDA_ARCH_FLAGS ${CUDA_ARCHS})

        set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};${CUDA_ARCH_FLAGS};-std=c++14")

        # Fix for some combinations of CUDA and GCC (e.g. under Ubuntu 16.04).
        set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-D_FORCE_INLINES")
        # Do not show warnings if the architectures are deprecated.
        set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-Wno-deprecated-gpu-targets")
        set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-O3;--use_fast_math")

        message(STATUS "Enabling CUDA support (version: ${CUDA_VERSION_STRING},"
                       " archs: ${CUDA_ARCH_FLAGS_readable})")
    else()
        set(CUDA_FOUND OFF)
        message(STATUS "Disabling CUDA support")
    endif()
else()
    set(CUDA_ENABLED OFF)
    if(CUDA_VERSION_STRING)
        message(STATUS "Disabling CUDA support (found version "
                "${CUDA_VERSION_STRING} but >= ${CUDA_MIN_VERSION} required)")
    else()
        message(STATUS "Disabling CUDA support")
    endif()
endif()

# option(MULTI_THREADS_OPT "Enable multi thread on a single gpu card" ON)
# if (NOT THREADS_PER_GPU)
#     set(THREADS_PER_GPU 3)
# endif()
# if (MULTI_THREADS_OPT)
#     add_definitions(-DTHREADS_PER_GPU=${THREADS_PER_GPU})
#     message("use THREADS_PER_GPU=${THREADS_PER_GPU}")
# endif()

option(ENCRYPT_ENABLE "Enable encrypt check" OFF)
if (ENCRYPT_ENABLE)
    add_definitions(-DDO_ENCRYPT_CHECK)
endif()

message("TBB: ${TBB_INCLUDE_DIRS}")

message("CGAL: ${CGAL_INCLUDE_DIRS}")

#external include and library DIR
include_directories(${PROJECT_SOURCE_DIR}/src)
include_directories(${PROJECT_SOURCE_DIR}/3rdParty)

add_subdirectory(elibs)
include_directories(SYSTEM
    ${Boost_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}
    ${GLOG_INCLUDE_DIRS}
    ${FREEIMAGE_INCLUDE_DIRS}
    ${CERES_INCLUDE_DIRS}
    ${SUITESPARSE_INCLUDE_DIRS}
    ${CGAL_INCLUDE_DIRS}
    ${VCG_INCLUDE_DIRS}
    ${TBB_INCLUDE_DIRS}
    ${GDAL_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/elibs/mapmap/
    ${CMAKE_SOURCE_DIR}/elibs/mapmap/mapmap
    ${CMAKE_SOURCE_DIR}/elibs/mapmap/ext/dset
)

link_directories(
    ${CMAKE_SOURCE_DIR}/elibs/mve/libs/mve
    ${CMAKE_SOURCE_DIR}/elibs/mve/libs/util
    # ${Boost_LIBRARY_DIRS}
    # ${CGAL_LIBRARIES}
)

### thirdPrarty
add_subdirectory ("${PROJECT_SOURCE_DIR}/3rdParty")

### src
add_subdirectory ("${PROJECT_SOURCE_DIR}/src")

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin/mvs)
add_subdirectory ("${PROJECT_SOURCE_DIR}/exe/mvs")
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin/mvs_tools)
add_subdirectory ("${PROJECT_SOURCE_DIR}/exe/mvs_tools")
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin/sfm)
add_subdirectory ("${PROJECT_SOURCE_DIR}/exe/sfm")
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin/sfm_tools)
add_subdirectory ("${PROJECT_SOURCE_DIR}/exe/sfm_tools")
