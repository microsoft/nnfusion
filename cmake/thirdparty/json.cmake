# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Enable ExternalProject CMake module
include(ExternalProject)

ExternalProject_Add(
    ext_json
    PREFIX json
    # Disable install step
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    UPDATE_COMMAND ""
    TMP_DIR "${NNFUSION_THIRDPARTY_FOLDER}/build/json/tmp"
    STAMP_DIR "${NNFUSION_THIRDPARTY_FOLDER}/build/json/stamp"
    DOWNLOAD_DIR "${NNFUSION_THIRDPARTY_FOLDER}/build/json/download"
    SOURCE_DIR "${NNFUSION_THIRDPARTY_FOLDER}/json"
    BINARY_DIR "${NNFUSION_THIRDPARTY_FOLDER}/build/json/build"
    INSTALL_DIR "${NNFUSION_THIRDPARTY_FOLDER}/build/json"
    EXCLUDE_FROM_ALL TRUE
)

#------------------------------------------------------------------------------

ExternalProject_Get_Property(ext_json SOURCE_DIR)
add_library(libjson INTERFACE)
target_include_directories(libjson SYSTEM INTERFACE ${SOURCE_DIR}/include)
add_dependencies(libjson ext_json)
