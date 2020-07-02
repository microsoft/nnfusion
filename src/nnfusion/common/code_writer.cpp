// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "code_writer.hpp"

using namespace std;
using namespace nnfusion;

nnfusion::codegen::CodeWriter::CodeWriter()
    : indent(0)
    , m_pending_indent(true)
    , m_temporary_name_count(0)
{
}

string codegen::CodeWriter::get_code() const
{
    return m_ss.str();
}

void codegen::CodeWriter::operator+=(const std::string& s)
{
    *this << s;
}

std::string codegen::CodeWriter::generate_temporary_name(std::string prefix)
{
    std::stringstream ss;

    ss << prefix << m_temporary_name_count;
    m_temporary_name_count++;

    return ss.str();
}
