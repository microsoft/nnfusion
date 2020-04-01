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

include(ExternalProject)

#------------------------------------------------------------------------------
# Fetch and install MKL-DNN
#------------------------------------------------------------------------------

if(MKLDNN_INCLUDE_DIR AND MKLDNN_LIB_DIR)
    ExternalProject_Add(
        ext_mkldnn
        DOWNLOAD_COMMAND ""
        UPDATE_COMMAND ""
        CONFIGURE_COMMAND ""
        BUILD_COMMAND ""
        INSTALL_COMMAND ""
        )
    add_library(libmkldnn INTERFACE)
    target_include_directories(libmkldnn SYSTEM INTERFACE ${MKLDNN_INCLUDE_DIR})
    target_link_libraries(libmkldnn INTERFACE
        ${MKLDNN_LIB_DIR}/libmkldnn.so
        ${MKLDNN_LIB_DIR}/libmklml_intel.so
        ${MKLDNN_LIB_DIR}/libiomp5.so
        )

    install(DIRECTORY ${MKLDNN_LIB_DIR}/ DESTINATION ${NGRAPH_INSTALL_LIB})
    return()
endif()

# This section sets up MKL as an external project to be used later by MKLDNN

set(MKL_LIBS libiomp5.so libmklml_intel.so)
set(MKL_ROOT ${NNFUSION_THIRDPARTY_FOLDER}/mkl/mkl_lnx)
add_library(libmkl INTERFACE)
foreach(LIB ${MKL_LIBS})
    target_link_libraries(libmkl INTERFACE ${MKL_ROOT}/lib/${LIB})
endforeach()

ExternalProject_Add(
    ext_mkldnn
    CMAKE_ARGS
        -DWITH_TEST=FALSE
        -DWITH_EXAMPLE=FALSE
        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
        -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
        -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
        -DCMAKE_INSTALL_PREFIX=${NNFUSION_THIRDPARTY_FOLDER}/build/mkldnn
        -DMKLROOT=${MKL_ROOT}
    TMP_DIR "${NNFUSION_THIRDPARTY_FOLDER}/build/mkldnn/tmp"
    STAMP_DIR "${NNFUSION_THIRDPARTY_FOLDER}/build/mkldnn/stamp"
    SOURCE_DIR "${NNFUSION_THIRDPARTY_FOLDER}/mkldnn"
    BINARY_DIR "${NNFUSION_THIRDPARTY_FOLDER}/build/mkldnn/build"
    INSTALL_DIR "${NNFUSION_THIRDPARTY_FOLDER}/build/mkldnn"
    BUILD_BYPRODUCTS "${NNFUSION_THIRDPARTY_FOLDER}/build/mkldnn/include/mkldnn.hpp"
    EXCLUDE_FROM_ALL TRUE
    )

add_custom_command(TARGET ext_mkldnn POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${NNFUSION_THIRDPARTY_FOLDER}/build/mkldnn/lib ${NGRAPH_BUILD_DIR}
    COMMENT "Move mkldnn libraries to ngraph build directory"
)

add_library(libmkldnn INTERFACE)
add_dependencies(libmkldnn ext_mkldnn)
target_include_directories(libmkldnn SYSTEM INTERFACE ${NNFUSION_THIRDPARTY_FOLDER}/build/mkldnn/include)
target_link_libraries(libmkldnn INTERFACE
    ${NNFUSION_THIRDPARTY_FOLDER}/build/mkldnn/lib/libmkldnn${CMAKE_SHARED_LIBRARY_SUFFIX}
    libmkl
    )

install(DIRECTORY ${NNFUSION_THIRDPARTY_FOLDER}/build/mkldnn/lib/ DESTINATION ${NGRAPH_INSTALL_LIB} OPTIONAL)
