# Microsoft (c) 2019, NNFusion Team

set(SRC
    all_close.cpp
)

add_library(ngraph_testutil_files STATIC ${SRC})
target_include_directories(ngraph_testutil_files SYSTEM PUBLIC
    ${GLOBAL_INCLUDE_PATH}
)
target_compile_options(ngraph_testutil_files PRIVATE "-fPIC")
