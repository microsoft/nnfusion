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
# Download and install Google Protobuf ...
#------------------------------------------------------------------------------

ExternalProject_Add(
    ext_protobuf
    PREFIX "thirdparty/protobuf"
    INSTALL_COMMAND ""
    UPDATE_COMMAND ""
    PATCH_COMMAND ""
    CONFIGURE_COMMAND ./autogen.sh && bash ./configure --disable-shared CXXFLAGS=-fPIC
    BUILD_COMMAND ${MAKE}
    SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/protobuf"
    BINARY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/protobuf"
    INSTALL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/protobuf"
    BUILD_BYPRODUCTS "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/protobuf/src/.libs/libprotobuf.a"
    EXCLUDE_FROM_ALL TRUE
    )

# -----------------------------------------------------------------------------

ExternalProject_Get_Property(ext_protobuf SOURCE_DIR BINARY_DIR)

# -----------------------------------------------------------------------------
# Use the interface of FindProtobuf.cmake
# -----------------------------------------------------------------------------

set(Protobuf_SRC_ROOT_FOLDER ${SOURCE_DIR})

set(Protobuf_PROTOC_EXECUTABLE ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/protobuf/src/protoc)
set(Protobuf_INCLUDE_DIR ${Protobuf_SRC_ROOT_FOLDER}/src)
set(Protobuf_LIBRARY ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/protobuf/src/.libs/libprotobuf.a)
set(Protobuf_LIBRARIES ${Protobuf_LIBRARY})

if (NOT TARGET protobuf::libprotobuf)
    add_library(protobuf::libprotobuf UNKNOWN IMPORTED)
    set_target_properties(protobuf::libprotobuf PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${Protobuf_INCLUDE_DIR}"
        IMPORTED_LOCATION "${Protobuf_LIBRARY}")
    add_dependencies(protobuf::libprotobuf ext_protobuf)
endif()

if (NOT TARGET protobuf::protoc)
    add_executable(protobuf::protoc IMPORTED)
    set_target_properties(protobuf::protoc PROPERTIES
        IMPORTED_LOCATION "${Protobuf_PROTOC_EXECUTABLE}")
    add_dependencies(protobuf::protoc ext_protobuf)
endif()

set(Protobuf_FOUND)
set(PROTOBUF_FOUND)
