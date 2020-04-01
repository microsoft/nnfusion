# Microsoft (c) 2019, Wenxiang Hu

add_subdirectory(graph)

set(SRC
    extract_graph_signature.cpp
    cpu_codegenerator.cpp
    cuda_codegenerator.cpp
    rocm_codegenerator.cpp
    # tensor related
    tensor/tensor_memory_layout.cpp
    tensor/liveness_analysis.cpp
)
add_library(nnfusion_engine_pass STATIC ${SRC})
target_include_directories(nnfusion_engine_pass SYSTEM PUBLIC
    ${NGRAPH_INCLUDE_PATH}
)
target_compile_options(nnfusion_engine_pass PRIVATE "-fPIC")
target_link_libraries(nnfusion_engine_pass PRIVATE gflags)
