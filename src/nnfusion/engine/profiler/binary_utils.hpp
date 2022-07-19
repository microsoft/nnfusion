// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

/**
 * \brief Binary utilities for profiling.
 * \todo This file should be moved into util/ in the future, since other feature may use this
 * \author wenxh
 */
#pragma once

#include <algorithm>
#include <string>

#include "nnfusion/util/util.hpp"

#ifdef WIN32
#include <windows.h>
#define CLOSE_LIBRARY(a) FreeLibrary(a)
#define DLSYM(a, b) GetProcAddress(a, b)
#define DLIB_SUFFIX ".dll"
#define DL_HANDLE HMODULE
#else
#include <dlfcn.h>
#define CLOSE_LIBRARY(a) dlclose(a)
#define DLSYM(a, b) dlsym(a, b)
#define DLIB_SUFFIX ".so"
#define DL_HANDLE void*
#endif

namespace nnfusion
{
    namespace profiler
    {
        // test.cu contains test_simple(void** args) entry point;
        // test.cu -> test.so
        // This doodad finds the full path of the containing shared library
        bool file_exists(std::string filename);

        DL_HANDLE get_library_handle(std::string object_name);

        void* get_function_pointer(std::string func_name, DL_HANDLE handle);

        void close_library_handle(DL_HANDLE& handle);

        std::string get_current_dir();
    }
}