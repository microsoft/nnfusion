# Microsoft (c) 2019, Wenxiang Hu

set(SRC
    logging.cpp
)
add_library(nnfusion_util STATIC ${SRC})
target_include_directories(nnfusion_util SYSTEM PUBLIC
    ${GLOBAL_INCLUDE_PATH}
)
target_compile_options(nnfusion_util PRIVATE "-fPIC")