// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <sstream>
#include <string>

namespace nnfusion
{
    namespace codegen
    {
        class CodeWriter;
    }
}

class nnfusion::codegen::CodeWriter
{
public:
    CodeWriter();
    std::string get_code() const;

    void operator+=(const std::string&);

    size_t indent;

    template <typename T>
    friend CodeWriter& operator<<(CodeWriter& out, const T& obj)
    {
        std::stringstream ss;
        ss << obj;

        for (char c : ss.str())
        {
            if (c == '\n')
            {
                out.m_pending_indent = true;
            }
            else
            {
                if (out.m_pending_indent)
                {
                    out.m_pending_indent = false;
                    for (size_t i = 0; i < out.indent; i++)
                    {
                        out.m_ss << "    ";
                    }
                }
            }
            out.m_ss << c;
        }

        return out;
    }

    std::string generate_temporary_name(std::string prefix = "tempvar");

    void block_begin()
    {
        *this << "{\n";
        indent++;
    }

    void block_end()
    {
        indent--;
        *this << "}\n";
    }

private:
    std::stringstream m_ss;
    bool m_pending_indent;
    size_t m_temporary_name_count;
};
