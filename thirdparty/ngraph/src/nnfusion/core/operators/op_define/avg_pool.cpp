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

#include "avg_pool.hpp"
#include "nnfusion/core/graph/gnode.hpp"
#include "nnfusion/core/operators/util/validation_util.hpp"

using namespace std;
using namespace nnfusion::op;

AvgPool::AvgPool(const nnfusion::Shape& window_shape,
                 const nnfusion::Strides& window_movement_strides,
                 const nnfusion::Shape& padding_below,
                 const nnfusion::Shape& padding_above,
                 bool include_padding_in_avg_computation)
    : Op("AvgPool")
    , m_window_shape(window_shape)
    , m_window_movement_strides(window_movement_strides)
    , m_padding_below(padding_below)
    , m_padding_above(padding_above)
    , m_include_padding_in_avg_computation(include_padding_in_avg_computation)
{
}

void AvgPool::validate_and_infer_types(std::shared_ptr<graph::GNode> gnode)
{
    if (0 == m_window_movement_strides.size())
    {
        m_window_movement_strides = Strides(m_window_shape.size(), 1);
    }

    if (0 == m_padding_below.size())
    {
        m_padding_below = Shape(m_window_shape.size(), 0);
    }

    if (0 == m_padding_above.size())
    {
        m_padding_above = Shape(m_window_shape.size(), 0);
    }

    const nnfusion::PartialShape& arg_shape = gnode->get_input_partial_shape(0);

    // infer_batched_forward_pooling wants CoordinateDiffs for these, while the pooling ops for
    // now still take Shape (no negative padding).
    nnfusion::CoordinateDiff padding_below(m_padding_below.begin(), m_padding_below.end());
    nnfusion::CoordinateDiff padding_above(m_padding_above.begin(), m_padding_above.end());

    gnode->set_output_type_and_shape(
        0,
        gnode->get_input_element_type(0),
        infer_batched_pooling_forward(this,
                                      arg_shape,
                                      padding_below,
                                      padding_above,
                                      m_window_shape,
                                      m_window_movement_strides,
                                      m_include_padding_in_avg_computation));
}

AvgPool::AvgPool(const Shape& window_shape, const Strides& window_movement_strides)
    : AvgPool(window_shape, window_movement_strides, nnfusion::Shape(), nnfusion::Shape(), false)
{
}

AvgPool::AvgPool(const Shape& window_shape)
    : AvgPool(window_shape, Strides(), Shape(), Shape(), false)
{
}

AvgPoolBackprop::AvgPoolBackprop(const nnfusion::Shape& forward_arg_shape,
                                 const nnfusion::Shape& window_shape,
                                 const nnfusion::Strides& window_movement_strides,
                                 const nnfusion::Shape& padding_below,
                                 const nnfusion::Shape& padding_above,
                                 bool include_padding_in_avg_computation)
    : Op("AvgPoolBackprop")
    , m_forward_arg_shape(forward_arg_shape)
    , m_window_shape(window_shape)
    , m_window_movement_strides(window_movement_strides)
    , m_padding_below(padding_below)
    , m_padding_above(padding_above)
    , m_include_padding_in_avg_computation(include_padding_in_avg_computation)
{
}

void AvgPoolBackprop::validate_and_infer_types(std::shared_ptr<graph::GNode> gnode)
{
    // infer_batched_forward_pooling wants CoordinateDiffs for these, while the pooling ops for
    // now still take Shape (no negative padding).
    nnfusion::CoordinateDiff padding_below(m_padding_below.begin(), m_padding_below.end());
    nnfusion::CoordinateDiff padding_above(m_padding_above.begin(), m_padding_above.end());

    nnfusion::PartialShape forward_result_shape =
        infer_batched_pooling_forward(this,
                                      m_forward_arg_shape,
                                      padding_below,
                                      padding_above,
                                      m_window_shape,
                                      m_window_movement_strides,
                                      m_include_padding_in_avg_computation);

    const nnfusion::PartialShape& delta_shape = gnode->get_input_shape(0);

    OP_VALIDATION(this, forward_result_shape.compatible(delta_shape))
        << "Inferred forward output shape does not match delta shape (inferred forward output "
        << "shape: " << forward_result_shape << ", delta shape: " << delta_shape << ").";

    // TODO(amprocte): Once m_forward_arg_shape is allowed to be dynamic, we may technically be
    // able to infer some extra information from forward_result_shape that was not present in the
    // forward arg shape---namely batch size and channel count. Merge that info in.
    gnode->set_output_type_and_shape(0, gnode->get_input_element_type(0), m_forward_arg_shape);
}