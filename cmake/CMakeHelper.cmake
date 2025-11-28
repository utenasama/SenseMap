set(SENSEMAP_TARGETS_ROOT_FOLDER "${PROJECT_SOURCE_DIR}")
# set(SENSEMAP_SRC_ROOT_FOLDER "sensemap_sources")

# This macro will search for source files in a given directory, will add them
# to a source group (folder within a project), and will then return paths to
# each of the found files. The usage of the macro is as follows:
# SENSEMAP_ADD_SOURCE_DIR(
#     <source directory to search>
#     <output variable with found source files>
#     <search expressions such as *.h *.cc>)
macro(SENSEMAP_ADD_SOURCE_DIR SRC_DIR SRC_VAR)
    # Create the list of expressions to be used in the search.
    set(GLOB_EXPRESSIONS "")
    foreach(ARG ${ARGN})
        list(APPEND GLOB_EXPRESSIONS ${SRC_DIR}/${ARG})
    endforeach()
    # Perform the search for the source files.
    file(GLOB ${SRC_VAR} RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
         ${GLOB_EXPRESSIONS})
    # Create the source group.
    string(REPLACE "/" "\\" GROUP_NAME ${SRC_DIR})
    source_group(${GROUP_NAME} FILES ${${SRC_VAR}})
    # Clean-up.
    unset(GLOB_EXPRESSIONS)
    unset(ARG)
    unset(GROUP_NAME)
endmacro(SENSEMAP_ADD_SOURCE_DIR)

# Macro to add source files to SenseMap.
macro(SENSEMAP_ADD_SOURCES)
    set(SOURCE_FILES "")
    foreach(SOURCE_FILE ${ARGN})
        if(SOURCE_FILE MATCHES "^/.*")
            list(APPEND SOURCE_FILES ${SOURCE_FILE})
        else()
            list(APPEND SOURCE_FILES
                 "${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE_FILE}")
        endif()
    endforeach()
    set(SENSEMAP_SOURCES ${SENSEMAP_SOURCES} ${SOURCE_FILES} PARENT_SCOPE)
endmacro(SENSEMAP_ADD_SOURCES)

# Macro to add cuda source files to SenseMap.
macro(SENSEMAP_ADD_CUDA_SOURCES)
    set(SOURCE_FILES "")
    foreach(SOURCE_FILE ${ARGN})
        if(SOURCE_FILE MATCHES "^/.*")
            # Absolute path.
            list(APPEND SOURCE_FILES ${SOURCE_FILE})
        else()
            # Relative path.
            list(APPEND SOURCE_FILES
                 "${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE_FILE}")
        endif()
    endforeach()

    set(SENSEMAP_CUDA_SOURCES
        ${SENSEMAP_CUDA_SOURCES}
        ${SOURCE_FILES}
        PARENT_SCOPE)
endmacro(SENSEMAP_ADD_CUDA_SOURCES)



macro(SENSEMAP_ADD_LIBRARY TARGET_NAME)
    # ${ARGN} will store the list of source files passed to this function.
    add_library(${TARGET_NAME} ${ARGN})
    # set_target_properties(${TARGET_NAME} PROPERTIES FOLDER
    #         ${SENSEMAP_TARGETS_ROOT_FOLDER}/${FOLDER_NAME})
    target_link_libraries(${TARGET_NAME}
                        ${Ceres_LIBRARIES}
                        ${GLOG_LIBRARIES}
                        ${FREEIMAGE_LIBRARIES}
                        ${OpenCV_LIBS}
                        )
    # install(TARGETS ${TARGET_NAME} DESTINATION lib/SenseMap/)
endmacro(SENSEMAP_ADD_LIBRARY)


macro(SENSEMAP_ADD_CUDA_LIBRARY TARGET_NAME)
    # ${ARGN} will store the list of source files passed to this function.
    cuda_add_library(${TARGET_NAME} ${ARGN}) 
endmacro(SENSEMAP_ADD_CUDA_LIBRARY)


# Wrapper for test executables.
macro(SENSEMAP_ADD_TEST TARGET_NAME)
    if(TESTS_ENABLED)
        # ${ARGN} will store the list of source files passed to this function.
        add_executable(${TARGET_NAME} ${ARGN})
        set_target_properties(${TARGET_NAME} PROPERTIES FOLDER
            ${SENSEMAP_TARGETS_ROOT_FOLDER}/${FOLDER_NAME})
        target_link_libraries(${TARGET_NAME})
        add_test("${FOLDER_NAME}/${TARGET_NAME}" ${TARGET_NAME})
        if(IS_MSVC)
            install(TARGETS ${TARGET_NAME} DESTINATION bin/)
        endif()
    endif()
endmacro(SENSEMAP_ADD_TEST)
