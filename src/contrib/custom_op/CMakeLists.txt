# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


file(GLOB native_custom_ops native/*.cpp)

set(SRC
    native/custom_test_op.cpp
    custom_op.cpp
    ${native_custom_ops}
)

add_library(contrib_custom_operators SHARED ${SRC})
target_include_directories(contrib_custom_operators SYSTEM PUBLIC
    ${GLOBAL_INCLUDE_PATH}
)
target_link_libraries(contrib_custom_operators nnfusion_common)
target_compile_options(contrib_custom_operators PRIVATE "-fPIC")

if(NOT DEFINED ENV{NNFUSION_HOME})
    set( ENV{NNFUSION_HOME} $ENV{HOME}/nnfusion )
endif()

file( COPY ${CMAKE_CURRENT_SOURCE_DIR}/json
      DESTINATION $ENV{NNFUSION_HOME}/custom_op
)


install(TARGETS contrib_custom_operators
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib/static
        PUBLIC_HEADER DESTINATION include
)