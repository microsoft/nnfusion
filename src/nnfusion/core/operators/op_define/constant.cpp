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

// Microsoft (c) 2019, NNFusion Team

#include <cmath>
#include <cstdio>

#include "constant.hpp"
#include "nnfusion/common/util.hpp"
#include "nnfusion/core/graph/gnode.hpp"
#include "nnfusion/util/util.hpp"

using namespace nnfusion::op;
using namespace std;

template <typename T>
string to_cpp_string(T value)
{
    string rc;
    if (std::isnan(value))
    {
        rc = "NAN";
    }
    else if (std::isinf(value))
    {
        if (value > 0)
        {
            rc = "INFINITY";
        }
        else
        {
            rc = "-INFINITY";
        }
    }
    else
    {
        stringstream ss;
        ss << value;
        rc = ss.str();
    }
    return rc;
}

Constant::~Constant()
{
    if (m_data)
    {
        nnfusion::aligned_free(m_data);
    }
}

DataBuffer Constant::get_buffer() const
{
    DataBuffer ret(m_element_type);
    ret.load(m_data, nnfusion::shape_size(m_shape));
    return std::move(ret);
}

vector<string> Constant::get_value_strings() const
{
    vector<string> rc;

    if (m_element_type == nnfusion::element::character)
    {
        for (int value : get_vector<char>())
        {
            rc.push_back(to_string(value));
        }
    }
    else if (m_element_type == nnfusion::element::bf16)
    {
        float temp = 0;
        for (auto value : get_vector<nnfusion::bfloat16>())
        {
            temp = static_cast<float>(value);
            rc.push_back(to_cpp_string(temp));
        }
    }
    else if (m_element_type == nnfusion::element::f16)
    {
        for (float value : get_float16_vector())
        {
            rc.push_back(to_cpp_string(value));
        }
    }
    else if (m_element_type == nnfusion::element::f32)
    {
        for (float value : get_vector<float>())
        {
            rc.push_back(to_cpp_string(value));
        }
    }
    else if (m_element_type == nnfusion::element::f64)
    {
        for (double value : get_vector<double>())
        {
            rc.push_back(to_cpp_string(value));
        }
    }
    else if (m_element_type == nnfusion::element::i8)
    {
        for (int value : get_vector<int8_t>())
        {
            rc.push_back(to_string(value));
        }
    }
    else if (m_element_type == nnfusion::element::boolean ||
             m_element_type == nnfusion::element::i16)
    {
        for (int value : get_vector<int16_t>())
        {
            rc.push_back(to_string(value));
        }
    }
    else if (m_element_type == nnfusion::element::i32)
    {
        for (int32_t value : get_vector<int32_t>())
        {
            rc.push_back(to_string(value));
        }
    }
    else if (m_element_type == nnfusion::element::i64)
    {
        for (int64_t value : get_vector<int64_t>())
        {
            rc.push_back(to_string(value));
        }
    }
    else if (m_element_type == nnfusion::element::u8)
    {
        for (uint32_t value : get_vector<uint8_t>())
        {
            rc.push_back(to_string(value));
        }
    }
    else if (m_element_type == nnfusion::element::u16)
    {
        for (uint32_t value : get_vector<uint16_t>())
        {
            rc.push_back(to_string(value));
        }
    }
    else if (m_element_type == nnfusion::element::u32)
    {
        for (uint32_t value : get_vector<uint32_t>())
        {
            rc.push_back(to_string(value));
        }
    }
    else if (m_element_type == nnfusion::element::u64)
    {
        for (uint64_t value : get_vector<uint64_t>())
        {
            rc.push_back(to_string(value));
        }
    }
    else
    {
        NNFUSION_CHECK_FAIL() << "unsupported type";
    }

    return rc;
}
