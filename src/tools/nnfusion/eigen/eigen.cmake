include(ExternalProject)

ExternalProject_Add(ext_eigen
  PREFIX "eigen"
  URL "https://nnfusion.blob.core.windows.net/mirror/eigen/a0d250e79c79.tar.gz"
      "https://bitbucket.org/eigen/eigen/get/a0d250e79c79.tar.gz"
  URL_HASH SHA256=0dde8fb87f5dad2e409c9f4ea1bebc54e694cf4f3b633081b0d51a55c00f9c9f
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  INSTALL_COMMAND ""
  TMP_DIR "${CMAKE_CURRENT_BINARY_DIR}/eigen/tmp"
  STAMP_DIR "${CMAKE_CURRENT_BINARY_DIR}/eigen/stamp"
  DOWNLOAD_DIR "${CMAKE_CURRENT_BINARY_DIR}/eigen/download"
  SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/eigen/src"
  BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/eigen/build"
)

ExternalProject_Add(ext_mkldnn
  PREFIX "mkldnn"
  URL "https://nnfusion.blob.core.windows.net/mirror/mkl-dnn/oneDNN-0.18.tar.gz"
      "https://github.com/intel/mkl-dnn/archive/v0.18.tar.gz"
  PATCH_COMMAND "${CMAKE_CURRENT_BINARY_DIR}/mkldnn/src/scripts/prepare_mkl.sh"
  CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${CMAKE_CURRENT_BINARY_DIR}/mkldnn/target
  BUILD_COMMAND make -j
  TMP_DIR "${CMAKE_CURRENT_BINARY_DIR}/mkldnn/tmp"
  STAMP_DIR "${CMAKE_CURRENT_BINARY_DIR}/mkldnn/stamp"
  DOWNLOAD_DIR "${CMAKE_CURRENT_BINARY_DIR}/mkldnn/download"
  SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/mkldnn/src"
  BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/mkldnn/build"
)

include_directories(${CMAKE_CURRENT_BINARY_DIR}/mkldnn/target/include)

set(eigen_custom_srcs
  ${CMAKE_CURRENT_LIST_DIR}/eigen_contraction_kernel.cc
)

add_library(eigen STATIC ${eigen_custom_srcs})
add_dependencies(eigen ext_eigen)
add_dependencies(eigen ext_mkldnn)
target_include_directories(eigen SYSTEM INTERFACE ${CMAKE_CURRENT_LIST_DIR})
target_include_directories(eigen SYSTEM INTERFACE ${CMAKE_CURRENT_BINARY_DIR}/eigen/src)
target_include_directories(eigen PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/eigen/src)
target_link_libraries(eigen ${CMAKE_CURRENT_BINARY_DIR}/mkldnn/target/lib/libmkldnn.so)
