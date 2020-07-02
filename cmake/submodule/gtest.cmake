# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Enable ExternalProject CMake module
include(ExternalProject)

ExternalProject_Add(
    ext_gtest
    PREFIX "thirdparty/googletest"
    # Disable install step
    INSTALL_COMMAND ""
    UPDATE_COMMAND ""
    CMAKE_ARGS -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
               -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
               -DCMAKE_CXX_FLAGS="-fPIC"
    SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/googletest"
    BINARY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/googletest/cmake/build"
    INSTALL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/googletest/cmake/build"
    BUILD_BYPRODUCTS "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/googletest/cmake/build/googlemock/gtest/libgtest.a"
    EXCLUDE_FROM_ALL TRUE
    )

#------------------------------------------------------------------------------

ExternalProject_Get_Property(ext_gtest SOURCE_DIR BINARY_DIR)

add_library(libgtest INTERFACE)
add_dependencies(libgtest ext_gtest)
target_include_directories(libgtest SYSTEM INTERFACE ${SOURCE_DIR}/googletest/include)
target_link_libraries(libgtest INTERFACE ${BINARY_DIR}/googlemock/gtest/libgtest.a)
