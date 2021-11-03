if(NOT Protobuf_VERSION VERSION_EQUAL "3.5.0")
    add_subdirectory(onnx)
    target_include_directories(onnx_proto PUBLIC ${CMAKE_CURRENT_BINARY_DIR} ${Protobuf_INCLUDE_DIR})
    set(ONNXPROTO_INCLUDE "${CMAKE_CURRENT_BINARY_DIR}" CACHE INTERNAL "Location for generated onnx protobuf headers." )
else()
    file(GLOB srcs onnx/*.pb.cc)
    file(GLOB hdrs onnx/*.pb.h)
    add_library(onnx_proto STATIC
            ${srcs}
            ${hdrs}
    )
    #add_dependencies(onnx_proto protobuf::libprotobuf)
    target_include_directories(onnx_proto PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${Protobuf_INCLUDE_DIR})
    set(ONNXPROTO_INCLUDE "${CMAKE_CURRENT_SOURCE_DIR}" CACHE INTERNAL "Location for generated onnx protobuf headers." )
endif()