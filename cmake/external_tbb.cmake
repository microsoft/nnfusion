# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

#------------------------------------------------------------------------------
# Fetch and configure TBB
#------------------------------------------------------------------------------

if(NGRAPH_TBB_ENABLE)
    set(TBB_GIT_REPO_URL https://github.com/01org/tbb)
    set(TBB_GIT_TAG "2019_U1")

    configure_file(${CMAKE_SOURCE_DIR}/cmake/tbb_fetch.cmake.in ${CMAKE_CURRENT_BINARY_DIR}/tbb/CMakeLists.txt)
    execute_process(COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" .
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/tbb")
    execute_process(COMMAND "${CMAKE_COMMAND}" --build .
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/tbb")

    set(TBB_ROOT ${CMAKE_CURRENT_BINARY_DIR}/tbb/tbb-src)
endif()
