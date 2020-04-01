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
