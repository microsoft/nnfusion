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

#include <cmath>
#include <iostream>

#include "nnfusion/common/type/element_type.hpp"

using namespace nnfusion;

const element::Type element::dynamic(0, false, false, false, "dynamic");
const element::Type element::boolean(8, false, true, false, "char");
const element::Type element::character(8, false, false, false, "char");
const element::Type element::bf16(16, true, true, false, "bfloat16");
const element::Type element::f16(16, true, true, false, "half");
const element::Type element::f32(32, true, true, false, "float");
const element::Type element::f64(64, true, true, false, "double");
const element::Type element::i8(8, false, true, true, "int8_t");
const element::Type element::i16(16, false, true, false, "int16_t");
const element::Type element::i32(32, false, true, false, "int32_t");
const element::Type element::i64(64, false, true, false, "int64_t");
const element::Type element::u8(8, false, false, true, "uint8_t");
const element::Type element::u16(16, false, false, false, "uint16_t");
const element::Type element::u32(32, false, false, false, "uint32_t");
const element::Type element::u64(64, false, false, false, "uint64_t");

std::vector<const element::Type*> element::Type::get_known_types()
{
    std::vector<const element::Type*> rc = {&element::boolean,
                                            &element::character,
                                            &element::bf16,
                                            &element::f32,
                                            &element::f64,
                                            &element::i8,
                                            &element::i16,
                                            &element::i32,
                                            &element::i64,
                                            &element::u8,
                                            &element::u16,
                                            &element::u32,
                                            &element::u64};
    return rc;
}

bool element::Type::nnfusion_element_type_to_dtype_string(const element::Type& ng_et,
                                                          std::string& dtype)
{
    if (ng_et == element::boolean)
        dtype = "int8";
    else if (ng_et == element::character)
        dtype = "char";
    else if (ng_et == element::f16)
        dtype = "float16";
    else if (ng_et == element::f32)
        dtype = "float32";
    else if (ng_et == element::f64)
        dtype = "float64";
    else if (ng_et == element::i8)
        dtype = "int8";
    else if (ng_et == element::i16)
        dtype = "int16";
    else if (ng_et == element::i32)
        dtype = "int32";
    else if (ng_et == element::i64)
        dtype = "int64";
    else if (ng_et == element::u8)
        dtype = "uint8";
    else if (ng_et == element::u16)
        dtype = "uint16";
    else if (ng_et == element::u32)
        dtype = "uint32";
    else if (ng_et == element::u64)
        dtype = "uint64";
    else
        return false;

    return true;
}

bool element::Type::nnfusion_element_type_to_pbtype(const element::Type& ng_et,
                                                    nnfusion::serialize::PBType& dtype)
{
    if (ng_et == element::boolean)
        dtype = nnfusion::serialize::PBType::DT_BOOL;
    else if (ng_et == element::character)
        dtype = nnfusion::serialize::PBType::DT_CHAR;
    else if (ng_et == element::f32)
        dtype = nnfusion::serialize::PBType::DT_FLOAT;
    else if (ng_et == element::f64)
        dtype = nnfusion::serialize::PBType::DT_DOUBLE;
    else if (ng_et == element::i8)
        dtype = nnfusion::serialize::PBType::DT_INT8;
    else if (ng_et == element::i16)
        dtype = nnfusion::serialize::PBType::DT_INT16;
    else if (ng_et == element::i32)
        dtype = nnfusion::serialize::PBType::DT_INT32;
    else if (ng_et == element::i64)
        dtype = nnfusion::serialize::PBType::DT_INT64;
    else if (ng_et == element::u8)
        dtype = nnfusion::serialize::PBType::DT_UINT8;
    else if (ng_et == element::u16)
        dtype = nnfusion::serialize::PBType::DT_UINT16;
    else if (ng_et == element::u32)
        dtype = nnfusion::serialize::PBType::DT_UINT32;
    else if (ng_et == element::u64)
        dtype = nnfusion::serialize::PBType::DT_UINT64;
    else
        return false;

    return true;
}

element::Type::Type(
    size_t bitwidth, bool is_real, bool is_signed, bool is_quantized, const std::string& cname)
    : m_bitwidth{bitwidth}
    , m_is_real{is_real}
    , m_is_signed{is_signed}
    , m_is_quantized{is_quantized}
    , m_cname{cname}
{
}

element::Type& element::Type::operator=(const element::Type& t)
{
    m_bitwidth = t.m_bitwidth;
    m_is_real = t.m_is_real;
    m_is_signed = t.m_is_signed;
    m_is_quantized = t.m_is_quantized;
    m_cname = t.m_cname;
    return *this;
}

const std::string& element::Type::c_type_string() const
{
    return m_cname;
}

bool element::Type::operator==(const element::Type& other) const
{
    return m_bitwidth == other.m_bitwidth && m_is_real == other.m_is_real &&
           m_is_signed == other.m_is_signed && m_is_quantized == other.m_is_quantized &&
           m_cname == other.m_cname;
}

bool element::Type::operator<(const Type& other) const
{
    size_t v1 = m_bitwidth << 3;
    v1 |= static_cast<size_t>(m_is_real ? 4 : 0);
    v1 |= static_cast<size_t>(m_is_signed ? 2 : 0);
    v1 |= static_cast<size_t>(m_is_quantized ? 1 : 0);

    size_t v2 = other.m_bitwidth << 3;
    v2 |= static_cast<size_t>(other.m_is_real ? 4 : 0);
    v2 |= static_cast<size_t>(other.m_is_signed ? 2 : 0);
    v2 |= static_cast<size_t>(other.m_is_quantized ? 1 : 0);

    return v1 == v2 ? m_cname < other.m_cname : v1 < v2;
}

size_t element::Type::size() const
{
    return (m_bitwidth + 7) / 8;
}

size_t element::Type::hash() const
{
    size_t h1 = std::hash<size_t>{}(m_bitwidth);
    size_t h2 = std::hash<bool>{}(m_is_real);
    size_t h3 = std::hash<bool>{}(m_is_signed);
    size_t h4 = std::hash<bool>{}(m_is_quantized);
    size_t h5 = std::hash<std::string>{}(m_cname);
    return (h1 ^ ((h2 ^ ((h3 ^ (h4 << 1)) << 1)) << 1) << 1) ^ h5;
}

namespace nnfusion
{
    namespace element
    {
        template <>
        const Type& from<char>()
        {
            return character;
        }
        template <>
        const Type& from<bool>()
        {
            return boolean;
        }
        template <>
        const Type& from<half>()
        {
            return f16;
        }
        template <>
        const Type& from<float>()
        {
            return f32;
        }
        template <>
        const Type& from<double>()
        {
            return f64;
        }
        template <>
        const Type& from<int8_t>()
        {
            return i8;
        }
        template <>
        const Type& from<int16_t>()
        {
            return i16;
        }
        template <>
        const Type& from<int32_t>()
        {
            return i32;
        }
        template <>
        const Type& from<int64_t>()
        {
            return i64;
        }
        template <>
        const Type& from<uint8_t>()
        {
            return u8;
        }
        template <>
        const Type& from<uint16_t>()
        {
            return u16;
        }
        template <>
        const Type& from<uint32_t>()
        {
            return u32;
        }
        template <>
        const Type& from<uint64_t>()
        {
            return u64;
        }
        template <>
        const Type& from<nnfusion::bfloat16>()
        {
            return bf16;
        }
    }
}

std::ostream& element::operator<<(std::ostream& out, const element::Type& obj)
{
    out << "element::Type{" << obj.m_bitwidth << ", " << obj.m_is_real << ", " << obj.m_is_signed
        << ", " << obj.m_is_quantized << ", \"" << obj.m_cname << "\"}";
    return out;
}

bool element::Type::merge(element::Type& dst, const element::Type& t1, const element::Type& t2)
{
    if (t1.is_dynamic())
    {
        dst = t2;
        return true;
    }
    else if (t2.is_dynamic())
    {
        dst = t1;
        return true;
    }
    else if (t1 == t2)
    {
        dst = t1;
        return true;
    }
    else
    {
        return false;
    }
}
