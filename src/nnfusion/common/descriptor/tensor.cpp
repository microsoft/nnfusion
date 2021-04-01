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

#include "nnfusion/common/descriptor/tensor.hpp"
#include <strings.h>
#include "gflags/gflags.h"
#include "nnfusion/common/descriptor/layout/tensor_layout.hpp"
#include "nnfusion/util/errors.hpp"
//using namespace nnfusion;
using namespace std;

atomic<size_t> nnfusion::descriptor::Tensor::m_next_instance_id(0);

nnfusion::descriptor::Tensor::Tensor(const nnfusion::element::Type& element_type,
                                     const nnfusion::PartialShape& pshape,
                                     const std::string& name,
                                     NNFusion_DeviceType device_type,
                                     bool is_persistent,
                                     bool is_constant,
                                     bool is_parameter,
                                     bool is_RDMA_tensor,
                                     const std::string& group,
                                     int device_id)
    : m_element_type(element_type)
    , m_shape(pshape.is_static() ? pshape.to_shape() : nnfusion::Shape{})
    , m_partial_shape(pshape)
    , m_name(name)
    , m_device_type(device_type)
    , m_persistent(is_persistent)
    , m_constant(is_constant)
    , m_parameter(is_parameter)
    , m_RDMA(is_RDMA_tensor)
    , m_memset(false)
    , m_memset_value(0)
    , m_root_tensor(nullptr)
    , m_ref_count(1)
    , m_group(group)
    , m_device_id(device_id)
    , m_instance_id(m_next_instance_id.fetch_add(1))
    , m_unique_name("tensor_" + to_string(m_instance_id))
{
}

const std::string& nnfusion::descriptor::Tensor::get_name(bool get_valid_name) const
{
    // return original name
    if (!get_valid_name)
    {
        NNFUSION_CHECK(!m_name.empty()) << "Tensor name cannot be empty.";
        return m_name;
    }

    // return valid name
    bool is_valid_name = true;
    {
        // check whether tensor name is valid for codegen or not
        if (m_name.empty())
        {
            is_valid_name = false;
        }
        else
        {
            // check whether first character is invalid
            if (!(isalpha(m_name[0]) || m_name[0] == '_'))
            {
                is_valid_name = false;
            }

            // check the rest characters
            for (size_t i = 1; i < m_name.size(); i++)
            {
                if (!(isalnum(m_name[i]) || m_name[i] == '_'))
                {
                    is_valid_name = false;
                    break;
                }
            }

            // to avoid conflicts with unique_name
            if (m_name.find_first_of("tensor_") == 0)
            {
                is_valid_name = false;
            }
        }
    }
    if (!is_valid_name)
    {
        return get_unique_name();
    }
    return m_name;
}

void nnfusion::descriptor::Tensor::set_tensor_type(const nnfusion::element::Type& element_type,
                                                   const nnfusion::PartialShape& pshape)
{
    if (pshape.is_static())
    {
        m_shape = pshape.to_shape();
    }
    else
    {
        m_shape = nnfusion::Shape{};
    }
    m_partial_shape = pshape;
    m_element_type = element_type;
}

const nnfusion::Shape& nnfusion::descriptor::Tensor::get_shape() const
{
    if (auto tvl = get_tensor_layout())
    {
        return tvl->get_shape();
    }
    else
    {
        if (m_partial_shape.is_static())
        {
            return m_shape;
        }

        else
        {
            throw nnfusion::errors::InvalidArgument(
                "get_shape was called on a descriptor::Tensor with dynamic shape");
        }
    }
}

void nnfusion::descriptor::Tensor::set_pool_offset(size_t offset)
{
    m_pool_offset = offset;
}

size_t nnfusion::descriptor::Tensor::get_pool_offset() const
{
    return m_pool_offset;
}

void nnfusion::descriptor::Tensor::set_pool(const std::string& pool)
{
    m_pool = pool;
}

const std::string& nnfusion::descriptor::Tensor::get_pool() const
{
    return m_pool;
}

bool nnfusion::descriptor::Tensor::is_same_address(std::shared_ptr<Tensor> tensor)
{
    return (m_pool == tensor->get_pool()) && (m_pool_offset == tensor->get_pool_offset());
}

size_t nnfusion::descriptor::Tensor::size(bool in_byte) const
{
    size_t t_size;
    if (auto tvl = get_tensor_layout())
        t_size = tvl->get_size();
    else
        t_size = shape_size(get_shape());

    if (in_byte)
        t_size *= m_element_type.size();
    return t_size;
}

void nnfusion::descriptor::Tensor::set_tensor_layout(
    const std::shared_ptr<layout::TensorLayout>& tensor_layout)
{
    if (tensor_layout->get_shape() != get_shape())
    {
        throw nnfusion::errors::RuntimeError(
            "Setting tensor's layout to a layout with a different shape.");
    }
    if (tensor_layout->get_element_type() != get_element_type())
    {
        throw nnfusion::errors::RuntimeError(
            "Setting tensor's layout to a layout with a different element type.");
    }
    m_tensor_layout = tensor_layout;
}

std::string nnfusion::descriptor::Tensor::get_device_name() const
{
    std::string device_name = (m_RDMA ? "RDMA_" : "_") + get_device_str(m_device_type) + "_" +
                              std::to_string(m_device_id);
    return device_name;
}

ostream& operator<<(ostream& out, const nnfusion::descriptor::Tensor& tensor)
{
    out << "Tensor(" << tensor.get_name() << ")";
    return out;
}
