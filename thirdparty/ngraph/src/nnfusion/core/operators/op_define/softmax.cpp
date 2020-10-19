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

#include "softmax.hpp"
#include "nnfusion/core/graph/gnode.hpp"

using namespace nnfusion::op;

Softmax::Softmax(const nnfusion::AxisSet& axes, bool in_log_space)
    : ElementwiseArithmetic("Softmax")
    , m_axes(axes)
    , m_in_log_space(in_log_space)
{
}

void Softmax::validate_and_infer_types(std::shared_ptr<graph::GNode> gnode)
{
    ElementwiseArithmetic::validate_and_infer_types(gnode);

    auto shape = gnode->get_output_shape(0);

    NNFUSION_CHECK(m_axes.empty() ||
                   (m_axes.size() == 1 && *std::begin(m_axes) == shape.size() - 1))
        << "softmax only support computing on the last dim";
    for (auto axis : m_axes)
    {
        OP_VALIDATION(this, axis < shape.size()) << "Reduction axis (" << axis
                                                 << ") is out of bounds (argument shape: " << shape
                                                 << ").";
    }

    // empty axes == all axes
    if (m_axes.size() == 0)
    {
        for (size_t i = 0; i < shape.size(); ++i)
        {
            m_axes.insert(i);
        }
    }
}

SoftmaxGrad::SoftmaxGrad(const nnfusion::AxisSet& axes, bool in_log_space)
    : Op("SoftmaxGrad")
    , m_axes(axes)
    , m_in_log_space(in_log_space)
{
}

void SoftmaxGrad::validate_and_infer_types(std::shared_ptr<graph::GNode> gnode)
{
    NNFUSION_CHECK(gnode->get_input_size() == 2);
    NNFUSION_CHECK(gnode->get_input_shape(0) == gnode->get_input_shape(1));

    gnode->set_output_type_and_shape(
        0, gnode->get_input_element_type(0), gnode->get_input_shape(0));

    auto shape = gnode->get_output_shape(0);

    NNFUSION_CHECK(m_axes.empty() ||
                   (m_axes.size() == 1 && *std::begin(m_axes) == shape.size() - 1))
        << "softmax_grad only support computing on the last dim";

    for (auto axis : m_axes)
    {
        OP_VALIDATION(this, axis < shape.size()) << "Reduction axis (" << axis
                                                 << ") is out of bounds (argument shape: " << shape
                                                 << ").";
    }

    // empty axes == all axes
    if (m_axes.size() == 0)
    {
        for (size_t i = 0; i < shape.size(); ++i)
        {
            m_axes.insert(i);
        }
    }
}
