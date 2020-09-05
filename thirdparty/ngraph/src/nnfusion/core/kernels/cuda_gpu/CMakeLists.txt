# Microsoft (c) 2019, NNFusion Team

set(SRC
    util/gpu_util.cpp
)

add_library(ngraph_gpu_util STATIC ${SRC})
target_include_directories(ngraph_gpu_util SYSTEM PUBLIC
    ${GLOBAL_INCLUDE_PATH}
)
target_compile_options(ngraph_gpu_util PRIVATE "-fPIC")

target_link_libraries(ngraph_gpu_util PRIVATE
    kernels_registration gflags ${CURL_LIBRARIES}
)
