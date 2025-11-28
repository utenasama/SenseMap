cmake_minimum_required(VERSION 3.1)
project("GCO")

include_directories(.)

set(gco_src_files
    block.h
    GCoptimization.cpp
    graph.cpp
    LinkedBlockList.cpp
    maxflow.cpp
)

SENSEMAP_ADD_LIBRARY(gco_lib ${gco_src_files})

