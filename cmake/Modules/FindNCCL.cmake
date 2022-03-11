# Try to find NCCL
#
# The following variables are optionally searched for defaults
#  NCCL_ROOT: Base directory where all NCCL components are found
#  NCCL_ROOT_DIR: Base directory where all NCCL components are found
#  NCCL_INCLUDE_DIR: Directory where NCCL header is found
#  NCCL_LIB_DIR: Directory where NCCL library is found
#
# The following are set after configuration is done:
#  NCCL_FOUND
#  NCCL_INCLUDE_DIRS
#  NCCL_LIBRARIES
#
# The path hints include CUDA_TOOLKIT_ROOT_DIR seeing as some folks
# install NCCL in the same location as the CUDA toolkit.
# See https://github.com/caffe2/caffe2/issues/1601

if (NOT DEFINED NCCL_ROOT)
    set(NCCL_ROOT $ENV{CONDA_PREFIX})
endif()

set(NCCL_ROOT_DIR $ENV{NCCL_ROOT_DIR} CACHE PATH "Folder contains NVIDIA NCCL")

find_path(NCCL_INCLUDE_DIRS
    NAMES nccl.h
    HINTS
    ${NCCL_ROOT}
    ${NCCL_ROOT}/include
    ${NCCL_INCLUDE_DIR}
    ${NCCL_ROOT_DIR}
    ${NCCL_ROOT_DIR}/include
    ${CUDA_TOOLKIT_ROOT_DIR}/include
    REQUIRED)

if ($ENV{USE_STATIC_NCCL})
    message(STATUS "USE_STATIC_NCCL detected. Linking against static NCCL library")
    set(NCCL_LIBNAME "libnccl_static.a")
else()
    set(NCCL_LIBNAME "nccl")
endif()

find_library(NCCL_LIBRARIES
    NAMES ${NCCL_LIBNAME}
    HINTS
    ${NCCL_LIB_DIR}
    ${NCCL_ROOT}
    ${NCCL_ROOT}/lib
    ${NCCL_ROOT}/lib/x86_64-linux-gnu
    ${NCCL_ROOT}/lib64
    ${NCCL_ROOT_DIR}
    ${NCCL_ROOT_DIR}/lib
    ${NCCL_ROOT_DIR}/lib/x86_64-linux-gnu
    ${NCCL_ROOT_DIR}/lib64
    ${CUDA_TOOLKIT_ROOT_DIR}/lib64
    REQUIRED)

set (NCCL_HEADER_FILE "${NCCL_INCLUDE_DIRS}/nccl.h")
message (STATUS "Determining NCCL version from ${NCCL_HEADER_FILE}...")
set (OLD_CMAKE_REQUIRED_INCLUDES ${CMAKE_REQUIRED_INCLUDES})
list (APPEND CMAKE_REQUIRED_INCLUDES ${NCCL_INCLUDE_DIRS} ${CUDAToolkit_INCLUDE_DIR})
include(CheckCXXSymbolExists)
check_cxx_symbol_exists(NCCL_VERSION_CODE nccl.h NCCL_VERSION_DEFINED)

if (NCCL_VERSION_DEFINED)
    set(file "${PROJECT_BINARY_DIR}/detect_nccl_version.cc")
    file(WRITE ${file} "
        #include <iostream>
        #include \"${NCCL_HEADER_FILE}\"
        int main()
        {
            std::cout << NCCL_MAJOR << '.' << NCCL_MINOR << '.' << NCCL_PATCH;
            int x;
            ncclGetVersion(&x);
            return x == NCCL_VERSION_CODE;
        }
    ")
    try_run(NCCL_VERSION_MATCHED compile_result ${PROJECT_BINARY_DIR} ${file}
        RUN_OUTPUT_VARIABLE NCCL_VERSION
        CMAKE_FLAGS  "-DINCLUDE_DIRECTORIES=${CUDAToolkit_INCLUDE_DIR}"
        LINK_LIBRARIES ${NCCL_LIBRARIES})
    if (NOT NCCL_VERSION_MATCHED)
        message(FATAL_ERROR "Found NCCL header version and library version do not match! \
            (include: ${NCCL_INCLUDE_DIRS}, library: ${NCCL_LIBRARIES}). Please set NCCL_INCLUDE_DIR and NCCL_LIB_DIR manually.")
    endif()
    message(STATUS "NCCL version: ${NCCL_VERSION}")
else()
    message(STATUS "NCCL version < 2.3.5-5")
endif ()
set (CMAKE_REQUIRED_INCLUDES ${OLD_CMAKE_REQUIRED_INCLUDES})

mark_as_advanced(NCCL_ROOT_DIR NCCL_INCLUDE_DIRS NCCL_LIBRARIES)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
    NCCL
    REQUIRED_VARS NCCL_INCLUDE_DIRS NCCL_LIBRARIES
    VERSION_VAR NCCL_VERSION)
