add_library(ngraph_tensorflow_import STATIC  
        util/bcast.hpp
        util/bcast.cpp     
)

add_dependencies(ngraph_tensorflow_import tensorflow_import_interface)

if (NOT NGRAPH_USE_SYSTEM_PROTOBUF)
    add_dependencies(ngraph_tensorflow_import protobuf::libprotobuf)
endif()

set_property(TARGET ngraph_tensorflow_import PROPERTY POSITION_INDEPENDENT_CODE ON)
target_include_directories(ngraph_tensorflow_import PRIVATE ${TFPROTO_INCLUDE} ${CMAKE_CURRENT_SOURCE_DIR} ${GLOBAL_INCLUDE_PATH}
    SYSTEM PRIVATE ${Protobuf_INCLUDE_DIR})
target_link_libraries(ngraph_tensorflow_import PRIVATE ${Protobuf_LIBRARIES} ${ONNX_PROTO_LIBRARY} nnfusion_operators nnfusion_graph)


if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "^(Apple)?Clang$")
    target_compile_options(ngraph_tensorflow_import PRIVATE -Wno-undef -Wno-reserved-id-macro -Wno-switch-enum
            -Wno-extended-offsetof -Wno-shorten-64-to-32 -Wno-unused-macros -Wno-missing-variable-declarations
            -Wno-unused-private-field -Wno-shadow -Wno-deprecated)
endif()
