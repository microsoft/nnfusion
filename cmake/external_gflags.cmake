# ******************************************************************************
# Copyright 2017-2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ******************************************************************************

# Enable ExternalProject CMake module
include(ExternalProject)

#------------------------------------------------------------------------------
# Download and install gflags...
#------------------------------------------------------------------------------

SET(gflags_GIT_REPO_URL https://github.com/gflags/gflags.git)
SET(gflags_GIT_LABEL v2.2.2)

ExternalProject_Add(
    ext_gflags
    PREFIX gflags
    GIT_REPOSITORY ${gflags_GIT_REPO_URL}
    GIT_TAG ${gflags_GIT_LABEL}
    # Disable install step
    INSTALL_COMMAND ""
    UPDATE_COMMAND ""
    CMAKE_ARGS -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
                -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
        -DCMAKE_CXX_FLAGS="-fPIC"
        -DGFLAGS_BUILD_SHARED_LIBS=ON
        -DGFLAGS_BUILD_gflags_LIB=ON
        -DGFLAGS_BUILD_gflags_nothreads_LIB=OFF
    TMP_DIR "${EXTERNAL_PROJECTS_ROOT}/gflags/tmp"
    STAMP_DIR "${EXTERNAL_PROJECTS_ROOT}/gflags/stamp"
    DOWNLOAD_DIR "${EXTERNAL_PROJECTS_ROOT}/gflags/download"
    SOURCE_DIR "${EXTERNAL_PROJECTS_ROOT}/gflags/src"
    BINARY_DIR "${EXTERNAL_PROJECTS_ROOT}/gflags/build"
    INSTALL_DIR "${EXTERNAL_PROJECTS_ROOT}/gflags"
    BUILD_BYPRODUCTS "${EXTERNAL_PROJECTS_ROOT}/gflags/build/lib/libgflags.so"
    EXCLUDE_FROM_ALL TRUE
    )

#------------------------------------------------------------------------------

ExternalProject_Get_Property(ext_gflags SOURCE_DIR BINARY_DIR)

add_library(libgflags INTERFACE)
add_dependencies(libgflags ext_gflags)
target_include_directories(libgflags SYSTEM INTERFACE ${BINARY_DIR}/include)
target_link_libraries(libgflags INTERFACE ${BINARY_DIR}/lib/libgflags.so)
