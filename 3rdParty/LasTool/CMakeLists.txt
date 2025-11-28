cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
set(CMAKE_SUPPRESS_REGENERATION true)
project("LAStools")

if (!MSVC)
	add_compile_options(-O3 -Wall -Wno-strict-aliasing)
else()
	add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif()

include_directories(las_include)
aux_source_directory(src LAS_SOURCE_FILES)

SENSEMAP_ADD_LIBRARY(LasLib ${LAS_SOURCE_FILES})

