# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Enable ExternalProject CMake module
include(ExternalProject)

ExternalProject_Add(
    ext_eigen
    PREFIX eigen
    UPDATE_COMMAND ""
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    TMP_DIR "${NNFUSION_THIRDPARTY_FOLDER}/build/eigen/tmp"
    STAMP_DIR "${NNFUSION_THIRDPARTY_FOLDER}/build/eigen/stamp"
    DOWNLOAD_DIR "${NNFUSION_THIRDPARTY_FOLDER}/build/eigen/download"
    SOURCE_DIR "${NNFUSION_THIRDPARTY_FOLDER}/eigen"
    BINARY_DIR "${NNFUSION_THIRDPARTY_FOLDER}/build/eigen/build"
    INSTALL_DIR "${NNFUSION_THIRDPARTY_FOLDER}/build/eigen"
    BUILD_BYPRODUCTS "${NNFUSION_THIRDPARTY_FOLDER}/eigen/src/Eigen/Core"
    EXCLUDE_FROM_ALL TRUE
    )

ExternalProject_Get_Property(ext_eigen SOURCE_DIR)
add_library(libeigen INTERFACE)
target_include_directories(libeigen SYSTEM INTERFACE ${SOURCE_DIR})
add_dependencies(libeigen ext_eigen)
