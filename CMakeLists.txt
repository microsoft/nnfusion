# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

project (nnfusion)

message(STATUS "     _  __ _  __ ____            _ ")
message(STATUS "    / |/ // |/ // __/__ __ ___  (_)___   ___ ")
message(STATUS "   /    //    // _/ / // /(_-< / // _ \\ / _ \\")
message(STATUS "  /_/|_//_/|_//_/   \\_,_//___//_/ \\___//_//_/")
message(STATUS "      MSRAsia NNFusion Team(@nnfusion)")
message(STATUS "    https://github.com/microsoft/nnfusion")
message(STATUS "") 

#-----------------------------------------------------------------------------------------------
# STEP.0 Global Options
#-----------------------------------------------------------------------------------------------
option(NNFUSION "Enable NNFusion backend." TRUE)
option(DEBUG_ENABLE "Enable debug mode" TRUE)
option(CODE_COVERAGE_ENABLE "Enable code coverage." FALSE)
option(ONNX_FRONTEND "Enable ONNX frontend." TRUE)
option(TENSORFLOW_FRONTEND "Enable Tensorflow frontend." TRUE)
option(TORCHSCRIPT_FRONTEND "Enable TorchScript frontend." FALSE)

#-----------------------------------------------------------------------------------------------
# STEP.1 Customnized targets
#-----------------------------------------------------------------------------------------------

message(STATUS "Installation directory: ${CMAKE_INSTALL_PREFIX}")

add_custom_target(style-check
    COMMAND ${PROJECT_SOURCE_DIR}/maint/script/check_code_style.sh
)

add_custom_target(style
    COMMAND ${PROJECT_SOURCE_DIR}/maint/script/apply_code_style.sh
)

#-----------------------------------------------------------------------------------------------
# STEP.2 Set CMake configs
#-----------------------------------------------------------------------------------------------
# *Make build proto file inside build folder possible.
set(CMAKE_DISABLE_SOURCE_CHANGES OFF)
set(CMAKE_DISABLE_IN_SOURCE_BUILD OFF)

# Determin which flags in use
if (UNIX AND NOT APPLE)
    set(LINUX TRUE)
endif()

if (APPLE)
    set(WholeArchiveFlag -Wl,-force_load)
    set(NoWholeArchiveFlag -Wl,-noall_load)
else()
    set(WholeArchiveFlag -Wl,--whole-archive)
    set(NoWholeArchiveFlag -Wl,--no-whole-archive)
endif()

cmake_minimum_required (VERSION 3.10)

# Create compilation database compile_commands.json .
# Could be unsed in VScode or some analysis tool.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# STEP.3 Set compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

if (${WARNINGS_AS_ERRORS})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror")
    message(STATUS "Warnings as errors")
endif()

# SET(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-g")
if(${DEBUG_ENABLE})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")
endif()

if (CODE_COVERAGE_ENABLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --coverage")
endif()

set(GLOBAL_INCLUDE_PATH
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty
    # for proto headers
    ${PROJECT_BINARY_DIR}/src
)

include_directories(
    ${GLOBAL_INCLUDE_PATH}
)

#-----------------------------------------------------------------------------------------------
# STEP.4 Set Project Defines
#-----------------------------------------------------------------------------------------------

# For usage inside source code
if(ONNX_FRONTEND)
    add_definitions(-DONNX_FRONTEND)
endif()

if(TORCHSCRIPT_FRONTEND)
    add_definitions(-DTORCHSCRIPT_FRONTEND)
endif()

add_definitions(-DPROJECT_ROOT_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

#-----------------------------------------------------------------------------------------------
# STEP.5 Find thirdparty libraries
#-----------------------------------------------------------------------------------------------
find_package(Threads REQUIRED) # This is for test usage
find_package(GTest)
find_package(CURL)
find_package(Protobuf 3.5.0)
find_library(gflags NAMES gflags)

#-----------------------------------------------------------------------------------------------
# STEP.6 Check envs
#-----------------------------------------------------------------------------------------------
if(Protobuf_LIBRARY MATCHES "NOTFOUND")
    message(STATUS "Cannot found any Protobuf libraries.") 
endif()

#-----------------------------------------------------------------------------------------------
# STEP.7 Enable features
#-----------------------------------------------------------------------------------------------
add_subdirectory(thirdparty)
message(STATUS "thirdparty enabled")

add_subdirectory(src)
message(STATUS "nnfusion enabled")

add_subdirectory(test)
message(STATUS "unit tests enabled")

# add_subdirectory(doc)
# message(STATUS "nnfusion documents enabled")
