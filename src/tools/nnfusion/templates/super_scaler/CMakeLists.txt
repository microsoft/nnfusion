project(super_scaler)
cmake_minimum_required(VERSION 3.5)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra -std=c++11")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O2")

find_package(MPI REQUIRED)
find_package(CUDA)
find_package(hcc PATHS /opt/rocm)

if (CUDA_FOUND)
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61")
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O2")
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -cudart shared")

  link_directories(/usr/local/cuda/lib64)
  include_directories(${CUDA_INCLUDE_DIRS} ${MPI_INCLUDE_PATH})
  find_library(CUDA_cuda_LIBRARY cuda /usr/local/cuda/lib64/stubs)
  find_library(CUDA_cudart_LIBRARY libcudart.so /usr/local/cuda/lib64)

  cuda_add_library(super_scaler SHARED super_scaler.cu)
  target_link_libraries(super_scaler
    ${CUDA_cuda_LIBRARY}
    ${CUDA_cudart_LIBRARY}
    ${CUDA_LIBRARIES}
    ${CUDA_CUBLAS_LIBRARIES}
    ${MPI_LIBRARIES}
    nccl
  )

  cuda_add_executable(super_scaler_test super_scaler_test.cu)
  target_link_libraries(super_scaler_test super_scaler)
endif()

if (hcc_FOUND)
  set(HCC /opt/rocm/bin/hipcc)
  set(DYLIB_FLAGS -shared -fPIC)
  set(HCC_FLAGS -std=c++11 )
  set(HCC_FLAGS ${HCC_FLAGS} -I${MPI_C_HEADER_DIR})
  set(HCC_FLAGS ${HCC_FLAGS} ${MPI_C_LINK_FLAGS} -lmpi -lmpi_cxx)
  set(HCC_FLAGS ${HCC_FLAGS} -L/opt/rocm/lib -lrccl)

  set(ss_rocm_lib ${CMAKE_BINARY_DIR}/libsuper_scaler_rocm.so)
  set(ss_rocm_src ${CMAKE_SOURCE_DIR}/super_scaler_rocm.cpp)
  set(ss_rocm_test_src ${CMAKE_SOURCE_DIR}/super_scaler_rocm_test.cpp)
  add_custom_target(super_scaler_rocm ALL COMMAND ${HCC} ${DYLIB_FLAGS} ${HCC_FLAGS} -o ${ss_rocm_lib} ${ss_rocm_src})
  add_custom_target(super_scaler_rocm_test ALL COMMAND ${HCC} ${HCC_FLAGS} ${HCC_FLAGS} -L${CMAKE_BINARY_DIR} -lsuper_scaler_rocm -o super_scaler_rocm_test ${ss_rocm_test_src})
  add_dependencies(super_scaler_rocm_test super_scaler_rocm)
endif()
