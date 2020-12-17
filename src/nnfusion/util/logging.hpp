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
#pragma once

#include <deque>
#include <functional>
#include <sstream>
#include <stdexcept>

#ifndef PROJECT_ROOT_DIR
#define PROJECT_ROOT_DIR "NNFusion"
#endif

namespace nnfusion
{
    const int DEBUG = 0;
    const int INFO = 1;
    const int NNFUSION_WARNING = 2;
    const int ERROR = 3;
    const int NNFUSION_FATAL = 4;

    class ConstString
    {
    public:
        template <size_t SIZE>
        constexpr ConstString(const char (&p)[SIZE])
            : m_string(p)
            , m_size(SIZE)
        {
        }

        constexpr char operator[](size_t i) const
        {
            return i < m_size ? m_string[i] : throw std::out_of_range("");
        }
        constexpr const char* get_ptr(size_t offset) const { return &m_string[offset]; }
        constexpr size_t size() const { return m_size; }
    private:
        const char* m_string;
        size_t m_size;
    };

    constexpr const char* trim_file_name(ConstString root, ConstString s)
    {
        return s.get_ptr(root.size());
    }

    class LogHelper
    {
    public:
        LogHelper(int level,
                  const char* file,
                  int line,
                  std::function<void(const std::string&)> m_handler_func);
        ~LogHelper();

        std::ostream& stream() { return m_stream; }
        static void set_log_path(const std::string& path)
        {
            flag_save_to_file = true;
            log_path = path;
        }

    private:
        std::function<void(const std::string&)> m_handler_func;
        std::stringstream m_stream;
        int m_level;
        static std::string log_path;
        static bool flag_save_to_file;
    };

    extern std::ostream& get_nil_stream();

    void default_logger_handler_func(const std::string& s);

#define _LOG_DEBUG                                                                                 \
    nnfusion::LogHelper(nnfusion::DEBUG,                                                           \
                        nnfusion::trim_file_name(PROJECT_ROOT_DIR, __FILE__),                      \
                        __LINE__,                                                                  \
                        nnfusion::default_logger_handler_func)                                     \
        .stream()

#define _LOG_INFO                                                                                  \
    nnfusion::LogHelper(nnfusion::INFO,                                                            \
                        nnfusion::trim_file_name(PROJECT_ROOT_DIR, __FILE__),                      \
                        __LINE__,                                                                  \
                        nnfusion::default_logger_handler_func)                                     \
        .stream()

#define _LOG_NNFUSION_WARNING                                                                      \
    nnfusion::LogHelper(nnfusion::NNFUSION_WARNING,                                                \
                        nnfusion::trim_file_name(PROJECT_ROOT_DIR, __FILE__),                      \
                        __LINE__,                                                                  \
                        nnfusion::default_logger_handler_func)                                     \
        .stream()

#define _LOG_ERROR                                                                                 \
    nnfusion::LogHelper(nnfusion::ERROR,                                                           \
                        nnfusion::trim_file_name(PROJECT_ROOT_DIR, __FILE__),                      \
                        __LINE__,                                                                  \
                        nnfusion::default_logger_handler_func)                                     \
        .stream()

#define _LOG_NNFUSION_FATAL                                                                        \
    nnfusion::LogHelper(nnfusion::NNFUSION_FATAL,                                                  \
                        nnfusion::trim_file_name(PROJECT_ROOT_DIR, __FILE__),                      \
                        __LINE__,                                                                  \
                        nnfusion::default_logger_handler_func)                                     \
        .stream()

#define NNFUSION_LOG(level) _LOG_##level
}
