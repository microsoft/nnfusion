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

#include "max_pool.hpp"
#include "nnfusion/core/graph/gnode.hpp"
#include "nnfusion/core/operators/util/validation_util.hpp"

using namespace std;
using namespace nnfusion::op;

MaxPool::MaxPool(const nnfusion::Shape& window_shape,
                 const nnfusion::Strides& window_movement_strides,
                 const nnfusion::Shape& padding_below,
                 const nnfusion::Shape& padding_above,
                 std::string data_format)
    : Op("MaxPool")
    , m_window_shape(window_shape)
    , m_window_movement_strides(window_movement_strides)
    , m_padding_below(padding_below)
    , m_padding_above(padding_above)
    , m_data_format(data_format)
{
}

MaxPool::MaxPool(const nnfusion::Shape& window_shape,
                 const nnfusion::Strides& window_movement_strides,
                 std::string data_format)
    : MaxPool(
          window_shape, window_movement_strides, nnfusion::Shape(), nnfusion::Shape(), data_format)
{
}

MaxPool::MaxPool(const nnfusion::Shape& window_shape, std::string data_format)
    : MaxPool(window_shape, nnfusion::Strides(), nnfusion::Shape(), nnfusion::Shape(), data_format)
{
}

void MaxPool::validate_and_infer_types(std::shared_ptr<graph::GNode> gnode)
{
    if (0 == m_window_movement_strides.size())
    {
        m_window_movement_strides = nnfusion::Strides(m_window_shape.size(), 1);
    }

    if (0 == m_padding_below.size())
    {
        m_padding_below = nnfusion::Shape(m_window_shape.size(), 0);
    }

    if (0 == m_padding_above.size())
    {
        m_padding_above = nnfusion::Shape(m_window_shape.size(), 0);
    }

    const nnfusion::PartialShape& arg_shape = gnode->get_input_partial_shape(0);

    // infer_batched_forward_pooling wants CoordinateDiffs for these, while the pooling ops for
    // now still take Shape (no negative padding).
    nnfusion::CoordinateDiff padding_below(m_padding_below.begin(), m_padding_below.end());
    nnfusion::CoordinateDiff padding_above(m_padding_above.begin(), m_padding_above.end());

    gnode->set_output_type_and_shape(0,
                                     gnode->get_input_element_type(0),
                                     infer_batched_pooling_forward(this,
                                                                   arg_shape,
                                                                   padding_below,
                                                                   padding_above,
                                                                   m_window_shape,
                                                                   m_window_movement_strides,
                                                                   true,
                                                                   m_data_format));
}

void MaxPool::infer_shared_memory(std::shared_ptr<graph::GNode> gnode)
{
    auto& input_shape = gnode->get_input_shape(0);
    auto& output_shape = gnode->get_output_shape(0);
    size_t ws = get_window_shape()[0];
    for (auto w : get_window_shape())
    {
        if (w != ws)
            return;
    }
    for (auto s : get_window_movement_strides())
    {
        if (s != ws)
            return;
    }

    for (auto p : get_padding_below())
    {
        if (p != 0)
            return;
    }

    for (auto p : get_padding_above())
    {
        if (p != 0)
            return;
    }

    m_shared_memory.clear();
    int channel = get_data_format() == "NCHW" ? 1 : 3;
    m_shared_memory.clear();
    for (size_t i = 0; i < output_shape.size(); i++)
    {
        if (i == channel || i == 0)
            m_shared_memory.push_back(1);
        else
            m_shared_memory.push_back(ws);
    }
}

MaxPoolBackprop::MaxPoolBackprop(const nnfusion::Shape& window_shape,
                                 const nnfusion::Strides& window_movement_strides,
                                 const nnfusion::Shape& padding_below,
                                 const nnfusion::Shape& padding_above,
                                 const shared_ptr<MaxPool>& forward_op)
    : Op("MaxPoolBackprop")
    , m_window_shape(window_shape)
    , m_window_movement_strides(window_movement_strides)
    , m_padding_below(padding_below)
    , m_padding_above(padding_above)
    , m_forward_op(forward_op)
{
}

void MaxPoolBackprop::validate_and_infer_types(std::shared_ptr<graph::GNode> gnode)
{
    auto forward_arg_et = gnode->get_input_element_type(0);
    auto delta_et = gnode->get_input_element_type(1);
    nnfusion::element::Type result_et;

    OP_VALIDATION(this, nnfusion::element::Type::merge(result_et, forward_arg_et, delta_et))
        << "Element types for forward argument (" << forward_arg_et << ") and delta (" << delta_et
        << ") do not match.";

    // infer_batched_forward_pooling wants CoordinateDiffs for these, while the pooling ops for
    // now still take Shape (no negative padding).
    nnfusion::CoordinateDiff padding_below(m_padding_below.begin(), m_padding_below.end());
    nnfusion::CoordinateDiff padding_above(m_padding_above.begin(), m_padding_above.end());

    const nnfusion::PartialShape& forward_arg_shape = gnode->get_input_partial_shape(0);

    nnfusion::PartialShape forward_result_shape =
        infer_batched_pooling_forward(this,
                                      forward_arg_shape,
                                      padding_below,
                                      padding_above,
                                      m_window_shape,
                                      m_window_movement_strides,
                                      true);

    const nnfusion::PartialShape& delta_shape = gnode->get_input_partial_shape(1);

    OP_VALIDATION(this, forward_result_shape.compatible(delta_shape))
        << "Inferred forward output shape does not match delta shape (inferred forward output "
        << "shape: " << forward_result_shape << ", delta shape: " << delta_shape << ").";

    // TODO(amprocte): We may technically be able to infer some extra information from
    // forward_result_shape that was not present in the forward arg shape---namely batch size and
    // channel count. Merge that info in.
    gnode->set_output_type_and_shape(0, forward_arg_et, forward_arg_shape);
}

shared_ptr<MaxPool> MaxPoolBackprop::get_forward_op() const
{
    return m_forward_op.lock();
}
