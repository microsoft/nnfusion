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

#include "backend.hpp"
#include "backend_manager.hpp"

using namespace std;
using namespace ngraph;

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

unordered_map<string, runtime::new_backend_t>& runtime::BackendManager::get_registry()
{
    static unordered_map<string, new_backend_t> s_registered_backend;
    return s_registered_backend;
}

void runtime::BackendManager::register_backend(const string& name, new_backend_t new_backend)
{
    get_registry()[name] = new_backend;
}

vector<string> runtime::BackendManager::get_registered_backends()
{
    vector<string> rc;
    for (const auto& p : get_registry())
    {
        rc.push_back(p.first);
    }
    for (const auto& p : get_registered_device_map())
    {
        if (find(rc.begin(), rc.end(), p.first) == rc.end())
        {
            rc.push_back(p.first);
        }
    }
    return rc;
}

unique_ptr<runtime::Backend> runtime::BackendManager::create_backend(const std::string& config)
{
    runtime::Backend* backend = nullptr;
    string type = config;

    // strip off attributes, IE:CPU becomes IE
    auto colon = type.find(":");
    if (colon != type.npos)
    {
        type = type.substr(0, colon);
    }

    auto registry = get_registry();
    auto it = registry.find(type);
    if (it != registry.end())
    {
        new_backend_t new_backend = it->second;
        backend = new_backend(config.c_str());
    }
    else
    {
        DL_HANDLE handle = open_shared_library(type);
        if (!handle)
        {
            stringstream ss;
            ss << "Backend '" << type << "' not registered. Error:" << dlerror();
            throw runtime_error(ss.str());
        }
        function<const char*()> get_ngraph_version_string =
            reinterpret_cast<const char* (*)()>(DLSYM(handle, "get_ngraph_version_string"));
        if (!get_ngraph_version_string)
        {
            CLOSE_LIBRARY(handle);
            throw runtime_error("Backend '" + type +
                                "' does not implement get_ngraph_version_string");
        }

        function<runtime::Backend*(const char*)> new_backend =
            reinterpret_cast<runtime::Backend* (*)(const char*)>(DLSYM(handle, "new_backend"));
        if (!new_backend)
        {
            CLOSE_LIBRARY(handle);
            throw runtime_error("Backend '" + type + "' does not implement new_backend");
        }

        backend = new_backend(config.c_str());
    }
    return unique_ptr<runtime::Backend>(backend);
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

DL_HANDLE runtime::BackendManager::open_shared_library(string type)
{
    string ext = SHARED_LIB_EXT;

    DL_HANDLE handle = nullptr;

    // strip off attributes, IE:CPU becomes IE
    auto colon = type.find(":");
    if (colon != type.npos)
    {
        type = type.substr(0, colon);
    }

    string library_name = "lib" + to_lower(type) + "_backend" + string(SHARED_LIB_EXT);
    string my_directory = file_util::get_directory(find_my_file());
    string library_path = file_util::path_join(my_directory, library_name);
#ifdef WIN32
    handle = LoadLibrary(library_path.c_str());
#else
    handle = dlopen(library_path.c_str(), RTLD_NOW | RTLD_GLOBAL);
#endif
    return handle;
}

map<string, string> runtime::BackendManager::get_registered_device_map()
{
    map<string, string> rc;
    string my_directory = file_util::get_directory(find_my_file());
    vector<string> backend_list;

    auto f = [&](const string& file, bool is_dir) {
        string name = file_util::get_file_name(file);
        string backend_name;
        if (is_backend_name(name, backend_name))
        {
            rc.insert({to_upper(backend_name), file});
        }
    };
    file_util::iterate_files(my_directory, f, false, true);
    return rc;
}

bool runtime::BackendManager::is_backend_name(const string& file, string& backend_name)
{
    string name = file_util::get_file_name(file);
    string ext = SHARED_LIB_EXT;
    bool rc = false;
    if (!name.compare(0, 3, "lib"))
    {
        if (!name.compare(name.size() - ext.size(), ext.size(), ext))
        {
            auto pos = name.find("_backend");
            if (pos != name.npos)
            {
                backend_name = name.substr(3, pos - 3);
                rc = true;
            }
        }
    }
    return rc;
}
