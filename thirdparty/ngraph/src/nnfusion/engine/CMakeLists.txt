add_subdirectory(pass)

set(ENGINE
    external/backend_manager.cpp
)

add_library(ngraph_backend STATIC 
    ${ENGINE}
)

target_compile_options(ngraph_backend PRIVATE "-fPIC")