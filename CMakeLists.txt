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

cmake_minimum_required (VERSION 3.1)

set(NGRAPH_VERSION_LABEL "0.0.1")

string(REGEX MATCH "([0-9]+)\.([0-9]+)\.([0-9]+)" NGRAPH_VERSION_SHORT "${NGRAPH_VERSION_LABEL}")
string(REGEX MATCH "([0-9]+)\.([0-9]+)" NGRAPH_API_VERSION "${NGRAPH_VERSION_LABEL}")
string(REGEX MATCH "[^v](.*)" NGRAPH_VERSION "${NGRAPH_VERSION_LABEL}")
string(REPLACE "." ";" NGRAPH_VERSION_PARTS "${NGRAPH_VERSION_SHORT}")
list(GET NGRAPH_VERSION_PARTS 0 NGRAPH_VERSION_MAJOR)
list(GET NGRAPH_VERSION_PARTS 1 NGRAPH_VERSION_MINOR)
list(GET NGRAPH_VERSION_PARTS 2 NGRAPH_VERSION_PATCH)
configure_file(VERSION.in VERSION)

message(STATUS "NGRAPH_VERSION ${NGRAPH_VERSION}")
message(STATUS "NGRAPH_VERSION_SHORT ${NGRAPH_VERSION_SHORT}")
message(STATUS "NGRAPH_API_VERSION ${NGRAPH_API_VERSION}")
message(STATUS "NNFUSION_VERSION ${NGRAPH_VERSION}")

set(NGRAPH_INCLUDE_PATH
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty
)

# Suppress an OS X-specific warning.
if (POLICY CMP0042)
    cmake_policy(SET CMP0042 OLD)
endif()

project (ngraph)

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

SET(GCC_MIN_VERSION 4.8)
SET(CLANG_MIN_VERSION 3.8)
SET(APPLE_CLANG_MIN_VERSION 8.1)

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS GCC_MIN_VERSION)
        message(FATAL_ERROR "GCC version must be at least ${GCC_MIN_VERSION}!")
    endif()
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS CLANG_MIN_VERSION)
        message(FATAL_ERROR "Clang version must be at least ${CLANG_MIN_VERSION}!")
    endif()
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang")
    if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS APPLE_CLANG_MIN_VERSION)
        message(FATAL_ERROR "Apple Clang version must be at least ${APPLE_CLANG_MIN_VERSION}!")
    endif()
else()
    message(WARNING "You are using an unsupported compiler. Compilation has only been tested with Clang (${CLANG_MIN_VERSION} and up), Apple Clang (${APPLE_CLANG_MIN_VERSION} and up), and GCC (${GCC_MIN_VERSION} and up).")
endif()

# Prevent Eigen from using any LGPL3 code
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DEIGEN_MPL2_ONLY -DTBB_USE_THREADING_TOOLS")

# These variables are undocumented but useful.
set(CMAKE_DISABLE_SOURCE_CHANGES OFF)
set(CMAKE_DISABLE_IN_SOURCE_BUILD OFF)

# Create compilation database compile_commands.json
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# set directory where the custom finders live
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

include(var_functions)

# Options for Compiling
option(NGRAPH_DEBUG_ENABLE "Enable output for NGRAPH_DEBUG statements" FALSE)
option(NGRAPH_ONNX_IMPORT_ENABLE "Enable ONNX importer" FALSE)
option(NGRAPH_CODE_COVERAGE_ENABLE "Enable code coverage data collection" FALSE)
option(NNFUSION_TENSORFLOW_IMPORT_ENABLE "Enable Tensorflow importer for NNFusion" TRUE)
option(NGRAPH_USE_SYSTEM_PROTOBUF "Use system provided Protobuf shared object(Onnx support required Protobuf-3.5.1 from our thirdparty repo)" FALSE)
option(NGRAPH_USE_PREBUILT_LLVM "Using Prebuilt LLVM" TRUE)

# Options for features
option(NGRAPH_UNIT_TEST_ENABLE "Control the building of unit tests" TRUE)
option(NGRAPH_TOOLS_ENABLE "Control the building of tool" TRUE)

#   Backend Support Options
#       Original Ngraphs
option(NGRAPH_CPU_ENABLE "Control the building of the CPU backend" FALSE)
option(NGRAPH_INTELGPU_ENABLE "Control the building of the Intel GPU backend with clDNN" FALSE)
option(NGRAPH_GPU_ENABLE "Control the building of the GPU backend" FALSE)
option(NGRAPH_INTERPRETER_ENABLE "Control the building of the INTERPRETER backend" TRUE)
option(NGRAPH_HYBRID_ENABLE "Control the building of the HYBRID backend" FALSE)
option(NGRAPH_DISTRIBUTED_ENABLE "Add distributed mode to the CPU backend" FALSE)
option(NGRAPH_DEX_ONLY "Build CPU DEX without codegen" FALSE)
option(NGRAPH_PLAIDML_ENABLE "Enable the PlaidML backend" FALSE)
#       NNFusion
option(NNFUSION_ENABLE "Control the building of the nnfusion backend" TRUE)

set(NNFUSION_THIRDPARTY_FOLDER "" CACHE STRING "NNFusion Thirdpary libraries folder location")

#Disable ONNXIFI
#if (NGRAPH_ONNX_IMPORT_ENABLE)
#    set(NGRAPH_USE_SYSTEM_PROTOBUF FALSE)
#    option(NGRAPH_ONNXIFI_ENABLE "Enable ONNX Interface for Framework Integration" TRUE)
#endif()

#-----------------------------------------------------------------------------------------------
# Installation logic...
#-----------------------------------------------------------------------------------------------

if (LINUX)
    include(GNUInstallDirs)
else()
    set(CMAKE_INSTALL_BINDIR "bin")
    set(CMAKE_INSTALL_INCLUDEDIR "include")
    set(CMAKE_INSTALL_DOCDIR "doc")
    set(CMAKE_INSTALL_LIBDIR "lib")
endif()

if (DEFINED NGRAPH_INSTALL_PREFIX)
    set(CMAKE_INSTALL_PREFIX ${NGRAPH_INSTALL_PREFIX})
endif()
message(STATUS "Installation directory: ${CMAKE_INSTALL_PREFIX}")

# Destinations
set(NGRAPH_INSTALL_LIB "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}")
set(NGRAPH_INSTALL_INCLUDE "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR}")
set(NGRAPH_INSTALL_DOC "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_DOCDIR}")
set(NGRAPH_INSTALL_BIN "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR}")
set(CMAKE_INSTALL_RPATH "$ORIGIN")

#-----------------------------------------------------------------------------------------------
# Compiler-specific logic...
#-----------------------------------------------------------------------------------------------

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

# Compiler-specific logic...
if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "^(Apple)?Clang$")
    message( STATUS "Setting clang flags...")
    include( cmake/clang_4_0_flags.cmake )
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    if (DEFINED NGRAPH_USE_CXX_ABI)
        message( STATUS "nGraph using CXX11 ABI: " ${NGRAPH_USE_CXX_ABI} )
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=${NGRAPH_USE_CXX_ABI}")
    endif()    
endif()

ngraph_var(NGRAPH_WARNINGS_AS_ERRORS DEFAULT "OFF")
if (${NGRAPH_WARNINGS_AS_ERRORS})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror")
    message(STATUS "Warnings as errors")
endif()

SET(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-g")
SET(CMAKE_CXX_FLAGS_DEBUG  "-O0 -g")

if (NGRAPH_CODE_COVERAGE_ENABLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --coverage")
endif()

# Enable build target CPU features
set(NGRAPH_TARGET_ARCH native CACHE STRING "Target CPU architecture to build for. Defaults to the native CPU architecture")

if (NOT "${NGRAPH_TARGET_ARCH}" STREQUAL "native")
    message(WARNING "Build target architecture was overridden. The resulting build might not work correctly on the host CPU.")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=${NGRAPH_TARGET_ARCH}")

if (DEFINED NGRAPH_TUNE_ARCH)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mtune=${NGRAPH_TUNE_ARCH}")
endif()

# flags required for SDL-3
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -fPIE -D_FORTIFY_SOURCE=2")
if (NOT WIN32)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wformat -Wformat-security")
endif()
if (NOT APPLE)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -z noexecstack")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -z relro -z now")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pie")
endif()
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 4.9)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fstack-protector")
    else()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fstack-protector-strong")
    endif()
endif()

if (NGRAPH_USE_GOLD)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fuse-ld=gold")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fuse-ld=gold")
endif()
if(WIN32)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNOMINMAX")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_CRT_SECURE_NO_WARNINGS")
endif()

if (NGRAPH_CPU_ENABLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNGRAPH_CPU_ENABLE")	
endif()

if (NGRAPH_PLAIDML_ENABLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNGRAPH_PlaidML_ENABLE")
endif()

if (NOT DEFINED NGRAPH_TBB_ENABLE)
    set(NGRAPH_TBB_ENABLE ${NGRAPH_CPU_ENABLE})
endif()

add_custom_target(style-check
    COMMAND ${PROJECT_SOURCE_DIR}/maint/check-code-format.sh
)

add_custom_target(style
    COMMAND ${PROJECT_SOURCE_DIR}/maint/apply-code-format.sh
)
#-----------------------------------------------------------------------------------------------
# enable or disable output from NGRAPH_DEBUG statements
#-----------------------------------------------------------------------------------------------
if(NGRAPH_DEBUG_ENABLE)
    add_definitions(-DNGRAPH_DEBUG_ENABLE)
endif()

#-----------------------------------------------------------------------------------------------
# External projects install directory
#-----------------------------------------------------------------------------------------------

set(NGRAPH_BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR}/src/ngraph)

set(EXTERNAL_INSTALL_DIR ${CMAKE_BINARY_DIR}/external)

if(NOT DEFINED EXTERNAL_PROJECTS_ROOT)
    set(EXTERNAL_PROJECTS_ROOT ${CMAKE_CURRENT_BINARY_DIR})
endif()

if (NGRAPH_ONNX_IMPORT_ENABLE OR NNFUSION_TENSORFLOW_IMPORT_ENABLE)
    if (NOT NGRAPH_USE_SYSTEM_PROTOBUF)
        if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/protobuf/README.md")
            message(STATUS "Use submodule protobuf.")
            include(cmake/submodule/protobuf.cmake)
        else()
            if(NNFUSION_THIRDPARTY_FOLDER STREQUAL "")
                include(cmake/external_protobuf.cmake)
            else()
                include(cmake/thirdparty/protobuf.cmake)
                message(STATUS "Use internal protobuf.")
            endif()
        endif()
    else()
        find_package(Protobuf 3.5.0)
        if(NOT Protobuf_FOUND)
            if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/protobuf/README.md")
                message(STATUS "Use submodule protobuf.")
                include(cmake/submodule/protobuf.cmake)
            else()
                if(NNFUSION_THIRDPARTY_FOLDER STREQUAL "")
                    include(cmake/external_protobuf.cmake)
                else()
                    include(cmake/thirdparty/protobuf.cmake)
                    message(STATUS "Use internal protobuf.")
                endif()
            endif()
            set(NGRAPH_USE_SYSTEM_PROTOBUF FALSE)
        endif()
    endif()
endif()

if(NGRAPH_USE_SYSTEM_PROTOBUF)
    message(STATUS "Use system protobuf.")
    set(Protobuf_LIBRARIES ${PROTOBUF_LIBRARIES})
    set(Protobuf_INCLUDE_DIR ${PROTOBUF_INCLUDE_DIR})
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/protobuf/README.md")
    include(cmake/submodule/gtest.cmake)
    message(STATUS "Use gtest in submodule.")
else()
    if(NNFUSION_THIRDPARTY_FOLDER STREQUAL "")
        include(cmake/external_gtest.cmake)
        message(STATUS "Use downloaded gtest.")
    else()
        include(cmake/thirdparty/gtest.cmake)
        message(STATUS "Use internal gtest.")
    endif()
endif()

add_definitions(-DPROJECT_ROOT_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

add_subdirectory(thirdparty)
add_subdirectory(src)
message(STATUS "${NGRAPH_INCLUDE_PATH}") 

if (NGRAPH_UNIT_TEST_ENABLE)
    add_subdirectory(test)
    message(STATUS "unit tests enabled")
else()
    add_subdirectory(test/util)
    message(STATUS "unit tests disabled")
endif()

if (NGRAPH_DOC_BUILD_ENABLE)
    add_subdirectory(doc)
endif()

