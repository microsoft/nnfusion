# Microsoft (c) 2019, MNFusion Team

set(SRC
    # languageunit.cpp
    type_info.cpp
    descriptor/layout/dense_tensor_layout.cpp
    descriptor/layout/tensor_layout.cpp
    descriptor/tensor.cpp
    code_writer.cpp
    type/bfloat16.cpp
    type/element_type.cpp
    type/data_buffer.cpp
    type/type_protocal.cpp
    shape.cpp
    partial_shape.cpp
    axis_set.cpp
    axis_vector.cpp
    dimension.cpp
    strides.cpp
    util.cpp
    coordinate.cpp
    coordinate_diff.cpp
    # device_type.cpp
)

add_library(ngraph_common STATIC ${SRC})
target_include_directories(ngraph_common SYSTEM PUBLIC
    ${GLOBAL_INCLUDE_PATH}
)

target_compile_options(ngraph_common PRIVATE "-fPIC")
target_link_libraries(ngraph_common PRIVATE nnfusion_util gflags)
