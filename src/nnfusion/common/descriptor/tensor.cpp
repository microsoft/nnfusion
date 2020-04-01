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

#include "nnfusion/common/descriptor/tensor.hpp"
#include <strings.h>
#include "gflags/gflags.h"
#include "nnfusion/common/descriptor/layout/tensor_layout.hpp"
#include "nnfusion/util/errors.hpp"
//using namespace nnfusion;
using namespace std;
DEFINE_string(fdefault_device,
              "CUDA",
              "Choose defualt device from [CUDA, CPU, ROCm] in the codegen.");
nnfusion::descriptor::Tensor::Tensor(const nnfusion::element::Type& element_type,
                                     const nnfusion::PartialShape& pshape,
                                     const std::string& name,
                                     bool is_persistent,
                                     bool is_constant,
                                     bool is_parameter,
                                     bool is_RDMA_tensor,
                                     size_t group_id,
                                     size_t device_id)
    : m_element_type(element_type)
    , m_shape(pshape.is_static() ? pshape.to_shape() : nnfusion::Shape{})
    , m_partial_shape(pshape)
    , m_name(name)
    , m_persistent(is_persistent)
    , m_constant(is_constant)
    , m_parameter(is_parameter)
    , m_RDMA(is_RDMA_tensor)
    , m_group_id(group_id)
    , m_device_id(device_id)
{
    auto default_device = FLAGS_fdefault_device.c_str();
    m_device_type = nnfusion::get_device_type(default_device);
}

nnfusion::descriptor::Tensor::Tensor(const nnfusion::element::Type& element_type,
                                     const nnfusion::PartialShape& pshape,
                                     const std::string& name,
                                     DeviceType device_type,
                                     bool is_persistent,
                                     bool is_constant,
                                     bool is_parameter,
                                     bool is_RDMA_tensor,
                                     size_t group_id,
                                     size_t device_id)
    : m_element_type(element_type)
    , m_shape(pshape.is_static() ? pshape.to_shape() : nnfusion::Shape{})
    , m_partial_shape(pshape)
    , m_name(name)
    , m_device_type(device_type)
    , m_persistent(is_persistent)
    , m_constant(is_constant)
    , m_parameter(is_parameter)
    , m_RDMA(is_RDMA_tensor)
    , m_group_id(group_id)
    , m_device_id(device_id)
{
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
    std::string dt_str = (const char* []){"CUDA_GPU", "ROCM_GPU", "GENERIC_CPU"}[m_device_type];
    std::string device_name = (m_RDMA ? "RDMA_" : "_") + dt_str + "_" + std::to_string(m_device_id);
    return device_name;
}

ostream& operator<<(ostream& out, const nnfusion::descriptor::Tensor& tensor)
{
    out << "Tensor(" << tensor.get_name() << ")";
    return out;
}
