add_subdirectory(pass)

set(ENGINE
    util/file_util.cpp
)

add_library(ngraph_backend STATIC 
    ${ENGINE}
)

target_compile_options(ngraph_backend PRIVATE "-fPIC")