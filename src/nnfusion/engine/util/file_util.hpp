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

#pragma once

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef WIN32
#include <windows.h>
#define DL_HANDLE HMODULE
#else
#define DL_HANDLE void*
#endif

using namespace std;
namespace file_util
{
    string get_directory(const string& s);

    string path_join(const string& s1, const string& s2);

    string get_file_name(const string& s);
}