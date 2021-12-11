// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

//*****************************************************************************
// Copyright 2017-2018 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#ifdef WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#ifdef WIN32
#include <windows.h>
#define SHARED_LIB_EXT ".dll"
#else
#define SHARED_LIB_EXT ".so"
#include <dirent.h>
#include <ftw.h>
#include <sys/file.h>
#include <sys/time.h>
#include <unistd.h>
#endif
#include <cctype>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <vector>

#include "file_util.hpp"
#include "nnfusion/common/common.hpp"

using namespace std;
// using namespace ngraph;

#ifdef WIN32
#define CLOSE_LIBRARY(a) FreeLibrary(a)
#define DLSYM(a, b) GetProcAddress(a, b)
#else
#define CLOSE_LIBRARY(a) dlclose(a)
#define DLSYM(a, b) dlsym(a, b)
#endif

namespace file_util
{
    string get_directory(const string& s)
    {
        string rc = s;
        auto pos = s.find_last_of('/');
        if (pos != string::npos)
        {
            rc = s.substr(0, pos);
        }
        return rc;
    }

    string path_join(const string& s1, const string& s2)
    {
        string rc;
        if (s2.size() > 0)
        {
            if (s2[0] == '/')
            {
                rc = s2;
            }
            else if (s1.size() > 0)
            {
                rc = s1;
                if (rc[rc.size() - 1] != '/')
                {
                    rc += "/";
                }
                rc += s2;
            }
            else
            {
                rc = s2;
            }
        }
        else
        {
            rc = s1;
        }
        return rc;
    }

    string get_file_name(const string& s)
    {
        string rc = s;
        auto pos = s.find_last_of('/');
        if (pos != string::npos)
        {
            rc = s.substr(pos + 1);
        }
        return rc;
    }

    static void iterate_files_worker(const string& path,
                                     function<void(const string& file, bool is_dir)> func,
                                     bool recurse,
                                     bool include_links);

    void iterate_files(const string& path,
                       function<void(const string& file, bool is_dir)> func,
                       bool recurse = false,
                       bool include_links = false)
    {
        vector<string> files;
        vector<string> dirs;
#ifdef WIN32
        string file_match = path_join(path, "*");
        WIN32_FIND_DATA data;
        HANDLE hFind = FindFirstFile(file_match.c_str(), &data);
        if (hFind != INVALID_HANDLE_VALUE)
        {
            do
            {
                std::cout << data.cFileName << std::endl;
            } while (FindNextFile(hFind, &data));
            FindClose(hFind);
        }
#else
        iterate_files_worker(path,
                             [&files, &dirs](const string& file, bool is_dir) {
                                 if (is_dir)
                                 {
                                     dirs.push_back(file);
                                 }
                                 else
                                 {
                                     files.push_back(file);
                                 }
                             },
                             recurse,
                             include_links);
#endif

        for (auto f : files)
        {
            func(f, false);
        }
        for (auto f : dirs)
        {
            func(f, true);
        }
    }

#ifndef WIN32
    static void iterate_files_worker(const string& path,
                                     function<void(const string& file, bool is_dir)> func,
                                     bool recurse,
                                     bool include_links)
    {
        DIR* dir;
        struct dirent* ent;
        if ((dir = opendir(path.c_str())) != nullptr)
        {
            try
            {
                while ((ent = readdir(dir)) != nullptr)
                {
                    string name = ent->d_name;
                    string path_name = file_util::path_join(path, name);
                    switch (ent->d_type)
                    {
                    case DT_DIR:
                        if (name != "." && name != "..")
                        {
                            if (recurse)
                            {
                                file_util::iterate_files(path_name, func, recurse);
                            }
                            func(path_name, true);
                        }
                        break;
                    case DT_LNK:
                        if (include_links)
                        {
                            func(path_name, false);
                        }
                        break;
                    case DT_REG: func(path_name, false); break;
                    default: break;
                    }
                }
            }
            catch (...)
            {
                exception_ptr p = current_exception();
                closedir(dir);
                rethrow_exception(p);
            }
            closedir(dir);
        }
        else
        {
            throw runtime_error("error enumerating file " + path);
        }
    }
#endif
}

// This doodad finds the full path of the containing shared library
static string find_my_file()
{
#ifdef WIN32
    return ".";
#else
    Dl_info dl_info;
    dladdr(reinterpret_cast<void*>(find_my_file), &dl_info);
    return dl_info.dli_fname;
#endif
}

string to_lower(string s)
{
    transform(s.begin(), s.end(), s.begin(), ::tolower);
    return s;
}

string to_upper(string s)
{
    transform(s.begin(), s.end(), s.begin(), ::toupper);
}
