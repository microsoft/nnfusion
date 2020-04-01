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

vector<string> Constant::get_value_strings() const
{
    vector<string> rc;

    if (m_element_type == nnfusion::element::boolean)
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
    else if (m_element_type == nnfusion::element::i16)
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
        CHECK_FAIL() << "unsupported type";
    }

    return rc;
}

shared_ptr<Constant> ScalarConstantLikeBase::as_constant() const
{
    return std::make_shared<op::Constant>(m_element_type, m_shape, m_data);
}

template <typename T>
ScalarConstantLike<T>::ScalarConstantLike(const std::shared_ptr<graph::GNode>& like, T value)
    : ScalarConstantLikeBase("ScalarConstantLike")
    , m_value(value)
{
    m_element_type = like->get_input_element_type(0);
}

//
// We have to open up namespace blocks here to work around a problem with gcc:
//
// https://stackoverflow.com/questions/25594644/warning-specialization-of-template-in-different-namespace
//
namespace nnfusion
{
    namespace op
    {
        template <>
        void Constant::write_to_buffer<string>(const nnfusion::element::Type& target_type,
                                               const nnfusion::Shape& target_shape,
                                               const vector<string>& source,
                                               void* target,
                                               size_t target_element_count)
        {
        }
    }
}
