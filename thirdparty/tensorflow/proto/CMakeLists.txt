#-----------------------------------------------------------------------------------------------
#   Copyright (c) Microsoft Corporation. All rights reserved.
#   Licensed under the MIT License. See License.txt in the project root for license information.
#-----------------------------------------------------------------------------------------------

# This file is used to Update protobuf generated source files each time when compile

if(NGRAPH_USE_SYSTEM_PROTOBUF)
    file(GLOB protos *.proto)

    protobuf_generate_cpp(srcs hdrs ${protos})
    add_library(tensorflow_proto SHARED
            ${srcs}
            ${hdrs}
    )

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

# set_target_properties(tensorflow_proto PROPERTIES LINKER_LANGUAGE CXX)

# install(DIRECTORY
#    ${TFPROTO_INCLUDE}
#    DESTINATION "${NGRAPH_INSTALL_INCLUDE}/ngraph/frontend/tensorflow_import/"
#    FILES_MATCHING PATTERN "*.pb.h"
#    PATTERN "CMakeFiles" EXCLUDE
#)
