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
#include "logging.hpp"

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
    case WARNING: m_stream << "[WARNING] "; break;
    case ERROR: m_stream << "[ERROR] "; break;
    case FATAL: m_stream << "[FATAL] "; break;
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

// Parse log level (int64) from environment variable (char*)
int LogLevelStrToInt(const char* env_var_val)
{
    if (env_var_val == nullptr)
    {
        return 0;
    }

    // Ideally we would use env_var / safe_strto64, but it is
    // hard to use here without pulling in a lot of dependencies,
    // so we use std:istringstream instead
    string min_log_level(env_var_val);
    std::istringstream ss(min_log_level);
    int level;
    if (!(ss >> level))
    {
        // Invalid vlog level setting, set level to default (0)
        level = 0;
    }

    return level;
}

int MinLogLevelFromEnv()
{
    const char* tf_env_var_val = (const char*)&FLAGS_min_log_level;
    return LogLevelStrToInt(tf_env_var_val);
}

LogHelper::~LogHelper()
{
    // Read the min log level once during the first call to logging.
    static int min_log_level = MinLogLevelFromEnv();
    if (m_level >= min_log_level && m_handler_func)
    {
        m_handler_func(m_stream.str());
    }
    // Logger::log_item(m_stream.str());
}
