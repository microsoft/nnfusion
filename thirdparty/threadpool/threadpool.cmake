include(ExternalProject)

if(MSVC)
    ExternalProject_Add(hwloc
      PREFIX "hwloc"
      URL "https://nnfusion.blob.core.windows.net/mirror/hwloc/hwloc-2.1.0.tar.gz"
          "https://download.open-mpi.org/release/hwloc/v2.1/hwloc-2.1.0.tar.gz"
      CONFIGURE_COMMAND ""
      BUILD_COMMAND msbuild "${CMAKE_CURRENT_BINARY_DIR}/hwloc/src/contrib/windows/hwloc.sln" /p:Configuration=Release /p:Platform=x64
      INSTALL_COMMAND ""
      TMP_DIR "${CMAKE_CURRENT_BINARY_DIR}/hwloc/tmp"
      STAMP_DIR "${CMAKE_CURRENT_BINARY_DIR}/hwloc/stamp"
      DOWNLOAD_DIR "${CMAKE_CURRENT_BINARY_DIR}/hwloc/download"
      SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/hwloc/src"
      BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/hwloc/build"
    )

    add_library(libhwloc INTERFACE)
    target_include_directories(libhwloc SYSTEM INTERFACE "${CMAKE_CURRENT_BINARY_DIR}/hwloc/src/include")
    target_link_libraries(libhwloc INTERFACE "${CMAKE_CURRENT_BINARY_DIR}/hwloc/src/contrib/windows/x64/Release/libhwloc.lib")
    add_custom_command(
      TARGET hwloc
      POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/hwloc/src/contrib/windows/x64/Release/libhwloc-15.dll ${CMAKE_CURRENT_BINARY_DIR}/Release/libhwloc-15.dll
    )
else()
    ExternalProject_Add(hwloc
      PREFIX "hwloc"
      URL "https://nnfusion.blob.core.windows.net/mirror/hwloc/hwloc-2.1.0.tar.gz"
          "https://download.open-mpi.org/release/hwloc/v2.1/hwloc-2.1.0.tar.gz"
      CONFIGURE_COMMAND ../src/configure --prefix "${CMAKE_CURRENT_BINARY_DIR}/hwloc/include"
      BUILD_COMMAND make -j${J} install
      TMP_DIR "${CMAKE_CURRENT_BINARY_DIR}/hwloc/tmp"
      STAMP_DIR "${CMAKE_CURRENT_BINARY_DIR}/hwloc/stamp"
      DOWNLOAD_DIR "${CMAKE_CURRENT_BINARY_DIR}/hwloc/download"
      SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/hwloc/src"
      BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/hwloc/build"
      INSTALL_DIR "${CMAKE_CURRENT_BINARY_DIR}/hwloc"
    )

    add_library(libhwloc INTERFACE)
    add_dependencies(libhwloc hwloc)
    target_include_directories(libhwloc SYSTEM INTERFACE ${CMAKE_CURRENT_BINARY_DIR}/hwloc/include/include)
    target_link_libraries(libhwloc INTERFACE ${CMAKE_CURRENT_BINARY_DIR}/hwloc/include/lib/libhwloc.so)
endif()

set(threadpool_srcs
  ${CMAKE_CURRENT_LIST_DIR}/util.cpp
  ${CMAKE_CURRENT_LIST_DIR}/threadpool.cpp
  ${CMAKE_CURRENT_LIST_DIR}/numa_aware_threadpool.cpp
)

add_library(threadpool STATIC ${threadpool_srcs})
target_include_directories(threadpool SYSTEM INTERFACE ${CMAKE_CURRENT_LIST_DIR})
target_link_libraries(threadpool libhwloc eigen)
