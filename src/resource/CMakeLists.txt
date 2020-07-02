# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

set (SRC
    main.cpp
    util.cpp
    uncomment.cpp
    header_rewrite.cpp
)

set(LLVM_LIB_DIR ${LLVM_INCLUDE_DIR}/../lib/clang/5.0.2/include)
get_filename_component(LLVM_LIB_DIR ${LLVM_LIB_DIR} ABSOLUTE)

list(APPEND HEADER_SEARCH_DEFINES CLANG_BUILTIN_HEADERS_PATH="${LLVM_LIB_DIR}")
list(APPEND HEADER_SEARCH_DEFINES NGRAPH_HEADERS_PATH="${NGRAPH_INCLUDE_PATH}")

if(NGRAPH_CPU_ENABLE)
    get_target_property(MKLDNN_INCLUDE_DIR libmkldnn INTERFACE_INCLUDE_DIRECTORIES)
    get_target_property(EIGEN_INCLUDE_DIR libeigen INTERFACE_INCLUDE_DIRECTORIES)
    list(APPEND HEADER_SEARCH_DEFINES EIGEN_HEADERS_PATH="${EIGEN_INCLUDE_DIR}")
    list(APPEND HEADER_SEARCH_DEFINES MKLDNN_HEADERS_PATH="${MKLDNN_INCLUDE_DIR}")
endif()

if(NGRAPH_TBB_ENABLE)
    list(APPEND HEADER_SEARCH_DEFINES TBB_HEADERS_PATH="${TBB_ROOT}/include")
endif()

add_executable(resource_generator EXCLUDE_FROM_ALL ${SRC})
if(NGRAPH_CPU_ENABLE)
    add_dependencies(resource_generator ext_eigen ext_mkldnn)
endif()

set_source_files_properties(main.cpp PROPERTIES COMPILE_DEFINITIONS "${HEADER_SEARCH_DEFINES}")

