# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Enable ExternalProject CMake module
include(ExternalProject)

#------------------------------------------------------------------------------
# Download json
#------------------------------------------------------------------------------

SET(JSON_GIT_REPO_URL https://github.com/nlohmann/json)
SET(JSON_GIT_LABEL v3.1.1)

# The 'BUILD_BYPRODUCTS' argument was introduced in CMake 3.2.
if (${CMAKE_VERSION} VERSION_LESS 3.2)
    ExternalProject_Add(
        ext_json
        PREFIX json
        GIT_REPOSITORY ${JSON_GIT_REPO_URL}
        GIT_TAG ${JSON_GIT_LABEL}
        # Disable install step
        BUILD_COMMAND ""
        INSTALL_COMMAND ""
        UPDATE_COMMAND ""
        # cmake does not allow calling cmake functions so we call a cmake script in the Module
        # directory.
        PATCH_COMMAND ${CMAKE_COMMAND} -P ${CMAKE_SOURCE_DIR}/cmake/Modules/patch_json.cmake
        EXCLUDE_FROM_ALL TRUE
        )
else()
    ExternalProject_Add(
        ext_json
        PREFIX json
        GIT_REPOSITORY ${JSON_GIT_REPO_URL}
        GIT_TAG ${JSON_GIT_LABEL}
        # Disable install step
        BUILD_COMMAND ""
        INSTALL_COMMAND ""
        UPDATE_COMMAND ""
        # cmake does not allow calling cmake functions so we call a cmake script in the Module
        # directory.
        PATCH_COMMAND ${CMAKE_COMMAND} -P ${CMAKE_SOURCE_DIR}/cmake/Modules/patch_json.cmake
        EXCLUDE_FROM_ALL TRUE
    )
endif()

#------------------------------------------------------------------------------

ExternalProject_Get_Property(ext_json SOURCE_DIR)
add_library(libjson INTERFACE)
target_include_directories(libjson SYSTEM INTERFACE ${SOURCE_DIR}/include)
add_dependencies(libjson ext_json)
