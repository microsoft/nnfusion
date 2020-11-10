include(ExternalProject)

ExternalProject_Add(ext_mkl
  PREFIX "mkl"
  URL "https://nnfusion.blob.core.windows.net/mirror/mkl/mklml_lnx_2019.0.3.20190220.tgz"
      "https://github.com/intel/mkl-dnn/releases/download/v0.18/mklml_lnx_2019.0.3.20190220.tgz"
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  INSTALL_COMMAND ""
  TMP_DIR "${CMAKE_CURRENT_BINARY_DIR}/mkl/tmp"
  STAMP_DIR "${CMAKE_CURRENT_BINARY_DIR}/mkl/stamp"
  DOWNLOAD_DIR "${CMAKE_CURRENT_BINARY_DIR}/mkl/download"
  SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/mkl/src"
  BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/mkl/build"
)

include_directories(${CMAKE_CURRENT_BINARY_DIR}/mkl/src/include)
add_library(libmkl INTERFACE)
add_dependencies(libmkl ext_mkl)

set(MKL_LIBS libiomp5.so libmklml_intel.so libmklml_gnu.so)
foreach(LIB ${MKL_LIBS})
    target_link_libraries(libmkl INTERFACE ${CMAKE_CURRENT_BINARY_DIR}/mkl/src/lib/${LIB})
endforeach()
