add_subdirectory(tensorflow)
add_subdirectory(onnx)

find_package(gflags)
if(NOT gflags_FOUND)
    # include directories need to be updated because the header files "gflags.h" is generated during build, but we
    # need it before build (nnfusion/common.h need it).
    set(NGRAPH_INCLUDE_PATH ${NGRAPH_INCLUDE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/gflags_headers/ PARENT_SCOPE)
    # We use gflags==v2.2.2 either installed in system or in the thirdparty
    # folder.
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/gflags/README.md")
        option(GFLAGS_BUILD_SHARED_LIBS "" ON)
        option(GFLAGS_BUILD_GFLAGS_LIBS "" ON)
        option(GFLAGS_BUILD_STATIC_LIBS "" OFF)
        option(GFLAGS_BUILD_gflags_nothreads_LIB "" OFF)
        option(GFLAGS_BUILD_gflags_LIB "" ON)
        set(GFLAGS_NAMESPACE "google" CACHE STRING "google")
        add_subdirectory(gflags)
    else()
        include(../cmake/external_gflags.cmake)
        message(STATUS "Use downloaded gflags.")
        add_library(gflags ALIAS libgflags)
    endif()
endif()
