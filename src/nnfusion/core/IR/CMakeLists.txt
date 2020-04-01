# Microsoft (c) 2019, NNFusion Team

set(SRC
    attribute.cpp
    instruction.cpp
    IR.cpp
)

add_library(nnfusion_ir STATIC ${SRC})
target_include_directories(nnfusion_ir SYSTEM PUBLIC
    ${NGRAPH_INCLUDE_PATH}
)
target_compile_options(nnfusion_ir PRIVATE "-fPIC")