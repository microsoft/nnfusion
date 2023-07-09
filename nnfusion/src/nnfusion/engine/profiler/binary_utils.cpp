// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "binary_utils.hpp"
#include <fstream>

namespace nnfusion
{
    namespace profiler
    {
        bool file_exists(std::string filename)
        {
            std::ifstream ifile(filename.c_str());
            return (bool)ifile;
        }

        DL_HANDLE get_library_handle(std::string object_name)
        {
            if (!file_exists(object_name))
            {
                NNFUSION_LOG(ERROR) << "Failed compiling the target source file.";
                return nullptr;
            }

            DL_HANDLE handle;
#ifdef WIN32
            handle = LoadLibrary(library_path.c_str());
#else
            handle = dlopen(object_name.c_str(), RTLD_NOW);
            if (!handle)
            {
                NNFUSION_LOG(ERROR) << " could not open file [" << object_name
                                    << "]: " << dlerror();
                return nullptr;
            }
#endif
            return handle;
        }

        void* get_function_pointer(std::string func_name, DL_HANDLE handle)
        {
            void* fhdl = DLSYM(handle, func_name.c_str());
            return fhdl;
        }

        void close_library_handle(DL_HANDLE& handle) { CLOSE_LIBRARY(handle); }
        std::string get_current_dir()
        {
#ifdef WIN32
            // This is just work, but not okay
            const unsigned long maxDir = 300;
            char currentDir[maxDir];
            GetCurrentDirectory(maxDir, currentDir);
            return std::string(currentDir);
#else
            return std::string(".");
#endif
        }
    }
}
