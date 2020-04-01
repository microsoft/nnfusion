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
    ext_gtest
    PREFIX gtest
    # Disable install step
    INSTALL_COMMAND ""
    UPDATE_COMMAND ""
    CMAKE_ARGS -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
               -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
               -DCMAKE_CXX_FLAGS="-fPIC"
    TMP_DIR "${NNFUSION_THIRDPARTY_FOLDER}/build/gtest/tmp"
    STAMP_DIR "${NNFUSION_THIRDPARTY_FOLDER}/build/gtest/stamp"
    DOWNLOAD_DIR "${NNFUSION_THIRDPARTY_FOLDER}/build/gtest/download"
    SOURCE_DIR "${NNFUSION_THIRDPARTY_FOLDER}/gtest"
    BINARY_DIR "${NNFUSION_THIRDPARTY_FOLDER}/build/gtest/build"
    INSTALL_DIR "${NNFUSION_THIRDPARTY_FOLDER}/build/gtest"
    BUILD_BYPRODUCTS "${NNFUSION_THIRDPARTY_FOLDER}/build/gtest/build/googlemock/gtest/libgtest.a"
    EXCLUDE_FROM_ALL TRUE
    )

#------------------------------------------------------------------------------

ExternalProject_Get_Property(ext_gtest SOURCE_DIR BINARY_DIR)

add_library(libgtest INTERFACE)
add_dependencies(libgtest ext_gtest)
target_include_directories(libgtest SYSTEM INTERFACE ${SOURCE_DIR}/googletest/include)
target_link_libraries(libgtest INTERFACE ${BINARY_DIR}/googlemock/gtest/libgtest.a)
