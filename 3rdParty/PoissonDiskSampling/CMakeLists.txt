cmake_minimum_required(VERSION 3.1)
project("PoissonDiskSampling")

include_directories(vcglib)
include_directories(poisson_disk_wrapper)

set(pds_src_files
    poisson_disk_wrapper/vcg_mesh.cpp
    poisson_disk_wrapper/utils_sampling.cpp)

#aux_source_directory(src LAS_SOURCE_FILES)

SENSEMAP_ADD_LIBRARY(pds_lib ${pds_src_files})
