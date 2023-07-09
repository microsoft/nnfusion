
if(EXISTS "${CUB_INCLUDE_DIR}")
	include(FindPackageHandleStandardArgs) # I think this is a CMake v3 thing
	mark_as_advanced(CUB_INCLUDE_DIR)
else()
	include(ExternalProject)
    ExternalProject_Add(
    CUB
    PREFIX "cub"
    URL "https://codeload.github.com/NVlabs/cub/tar.gz/1.8.0"
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    TMP_DIR "${CMAKE_CURRENT_BINARY_DIR}/cub/tmp"
    STAMP_DIR "${CMAKE_CURRENT_BINARY_DIR}/cub/stamp"
    DOWNLOAD_DIR "${CMAKE_CURRENT_BINARY_DIR}/cub/download"
    SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/cub/src"
    BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/cub/build" 
    )
	
	# Specify include dir
	ExternalProject_Get_Property(CUB source_dir)
	set(CUB_INCLUDE_DIR ${source_dir})
endif()


find_path(
		CUB_INCLUDE_DIR 
		cub/cub.cuh
		HINTS
			${CUDA_INCLUDE_DIRS}
			${CMAKE_SOURCE_DIR}/include
			${CMAKE_SOURCE_DIR}
			${PROJECT_SOURCE_DIR}
			${PROJECT_SOURCE_DIR}/include 
			/opt 
			$ENV{HOME}/opt 
			ENV CUB_DIR 
			ENV CUB_INCLUDE_DIR 
			ENV CUB_PATH
		DOC "nVIDIA CUB GPU primitives header-only CUDA library"
		PATH_SUFFIXES cub libcub nvidia-cub 
	)