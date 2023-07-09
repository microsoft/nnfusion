#-----------------------------------------------------------------------------------------------
#   Copyright (c) Microsoft Corporation. All rights reserved.
#   Licensed under the MIT License. See License.txt in the project root for license information.
#-----------------------------------------------------------------------------------------------

# This file is used to Update protobuf generated source files each time when compile

if(NOT Protobuf_VERSION VERSION_EQUAL "3.5.0")
    file(GLOB protos *.proto)

    protobuf_generate_cpp(srcs hdrs ${protos})
    add_library(tensorflow_proto STATIC 
            ${srcs}
            ${hdrs}
    )
    add_dependencies(tensorflow_proto protobuf::libprotobuf)
    include_directories(${Protobuf_INCLUDE_DIR})

    target_include_directories(tensorflow_proto PUBLIC ${CMAKE_CURRENT_BINARY_DIR} ${Protobuf_INCLUDE_DIR})
    set(TFPROTO_INCLUDE "${CMAKE_CURRENT_BINARY_DIR}" CACHE INTERNAL "Location for generated tensorflow protobuf headers." )
else()
    file(GLOB srcs *.pb.cc)
    file(GLOB hdrs *.pb.h)

    add_library(tensorflow_proto STATIC
            ${srcs}
            ${hdrs}
    )
    add_dependencies(tensorflow_proto protobuf::libprotobuf)

    target_include_directories(tensorflow_proto PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${Protobuf_INCLUDE_DIR})
    set(TFPROTO_INCLUDE "${CMAKE_CURRENT_SOURCE_DIR}" CACHE INTERNAL "Location for generated tensorflow protobuf headers." )
endif()