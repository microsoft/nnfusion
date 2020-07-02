# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Enable ExternalProject CMake module
include(ExternalProject)

set(EIGEN_GIT_TAG patched)
set(EIGEN_GIT_URL https://github.com/NervanaSystems/eigen)

#------------------------------------------------------------------------------
# Download Eigen
#------------------------------------------------------------------------------

# The 'BUILD_BYPRODUCTS' argument was introduced in CMake 3.2.
if (${CMAKE_VERSION} VERSION_LESS 3.2)
    ExternalProject_Add(
        ext_eigen
        PREFIX eigen
        GIT_REPOSITORY ${EIGEN_GIT_URL}
        GIT_TAG ${EIGEN_GIT_TAG}
        UPDATE_COMMAND ""
        CONFIGURE_COMMAND ""
        BUILD_COMMAND ""
        INSTALL_COMMAND ""
        TMP_DIR "${EXTERNAL_PROJECTS_ROOT}/eigen/tmp"
        STAMP_DIR "${EXTERNAL_PROJECTS_ROOT}/eigen/stamp"
        DOWNLOAD_DIR "${EXTERNAL_PROJECTS_ROOT}/eigen/download"
        SOURCE_DIR "${EXTERNAL_PROJECTS_ROOT}/eigen/src"
        BINARY_DIR "${EXTERNAL_PROJECTS_ROOT}/eigen/build"
        INSTALL_DIR "${EXTERNAL_PROJECTS_ROOT}/eigen"
        EXCLUDE_FROM_ALL TRUE
        )
else()
    ExternalProject_Add(
        ext_eigen
        PREFIX eigen
        GIT_REPOSITORY ${EIGEN_GIT_URL}
        GIT_TAG ${EIGEN_GIT_TAG}
        UPDATE_COMMAND ""
        CONFIGURE_COMMAND ""
        BUILD_COMMAND ""
        INSTALL_COMMAND ""
        TMP_DIR "${EXTERNAL_PROJECTS_ROOT}/eigen/tmp"
        STAMP_DIR "${EXTERNAL_PROJECTS_ROOT}/eigen/stamp"
        DOWNLOAD_DIR "${EXTERNAL_PROJECTS_ROOT}/eigen/download"
        SOURCE_DIR "${EXTERNAL_PROJECTS_ROOT}/eigen/src"
        BINARY_DIR "${EXTERNAL_PROJECTS_ROOT}/eigen/build"
        INSTALL_DIR "${EXTERNAL_PROJECTS_ROOT}/eigen"
        BUILD_BYPRODUCTS "${EXTERNAL_PROJECTS_ROOT}/eigen/src/Eigen/Core"
        EXCLUDE_FROM_ALL TRUE
        )
endif()

#------------------------------------------------------------------------------

ExternalProject_Get_Property(ext_eigen SOURCE_DIR)
add_library(libeigen INTERFACE)
target_include_directories(libeigen SYSTEM INTERFACE ${SOURCE_DIR})
add_dependencies(libeigen ext_eigen)
