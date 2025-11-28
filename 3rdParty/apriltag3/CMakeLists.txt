cmake_minimum_required(VERSION 3.5)

project(apriltags)


aux_source_directory(common COMMON_SRC)
set(APRILTAG_SRCS apriltag.c apriltag_pose.c apriltag_quad_thresh.c)
file(GLOB TAG_FILES tag*.c)

SENSEMAP_ADD_LIBRARY(apriltags ${APRILTAG_SRCS} ${COMMON_SRC} ${TAG_FILES})

# find_package(Threads REQUIRED)
# target_link_libraries(${PROJECT_NAME} PUBLIC Threads::Threads m)