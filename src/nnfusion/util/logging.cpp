//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
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

// Microsoft (c) 2019, Wenxiang Hu
// This file is modified from nnfusion Log

#include <chrono>
#include <condition_variable>
#include <ctime>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <thread>

#include "gflags/gflags.h"
#include "nnfusion/util/logging.hpp"

using namespace std;
using namespace nnfusion;

DECLARE_int32(min_log_level);

bool LogHelper::flag_save_to_file = false;
std::string LogHelper::log_path = "";

ostream& nnfusion::get_nil_stream()
{
    static stringstream nil;
    return nil;
}

void nnfusion::default_logger_handler_func(const string& s)
{
    cout << s << endl;
}

LogHelper::LogHelper(int level,
                     const char* file,
                     int line,
                     function<void(const string&)> handler_func)
    : m_level(level)
    , m_handler_func(handler_func)
{
    switch (level)
    {
    case DEBUG: m_stream << "[DEBUG] "; break;
    case INFO: m_stream << "[INFO] "; break;
    case NNFUSION_WARNING: m_stream << "[WARNING] "; break;
    case ERROR: m_stream << "[ERROR] "; break;
    case NNFUSION_FATAL: m_stream << "[FATAL] "; break;
    }

    time_t tt = chrono::system_clock::to_time_t(chrono::system_clock::now());
    auto tm = gmtime(&tt);
    char buffer[256];
    strftime(buffer, sizeof(buffer), "%Y-%m-%dT%H:%M:%Sz", tm);
    m_stream << buffer << " ";

    m_stream << file;
    m_stream << " " << line;
    m_stream << "\t";

    //Todo(wenxh): Potential Thread Blocking if writting to file
    if (flag_save_to_file)
    {
        std::ofstream log_file(log_path, std::ios::out | std::ios::app);
        log_file << m_stream.str() << endl;
    }
}

LogHelper::~LogHelper()
{
    if (m_level >= FLAGS_min_log_level && m_handler_func)
    {
        m_handler_func(m_stream.str());
    }
    // Logger::log_item(m_stream.str());
}
