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

#include <numeric>

#include "convolution.hpp"
#include "nnfusion/core/graph/gnode.hpp"
#include "nnfusion/core/operators/util/validation_util.hpp"

using namespace std;
using namespace nnfusion::op;

Convolution::Convolution(const nnfusion::Strides& window_movement_strides,
                         const nnfusion::Strides& window_dilation_strides,
                         const nnfusion::CoordinateDiff& padding_below,
                         const nnfusion::CoordinateDiff& padding_above,
                         const nnfusion::Strides& data_dilation_strides,
                         std::string data_format)
    : Op("Convolution")
    , m_window_movement_strides(window_movement_strides)
    , m_window_dilation_strides(window_dilation_strides)
    , m_padding_below(padding_below)
    , m_padding_above(padding_above)
    , m_data_dilation_strides(data_dilation_strides)
    , m_data_format(data_format)
{
}

void Convolution::validate_and_infer_types(std::shared_ptr<graph::GNode> gnode)
{
    const nnfusion::PartialShape& data_batch_shape = gnode->get_input_partial_shape(0);
    nnfusion::element::Type data_batch_et = gnode->get_input_element_type(0);
    const nnfusion::PartialShape& filters_shape = gnode->get_input_partial_shape(1);
    nnfusion::element::Type filters_et = gnode->get_input_element_type(1);

    if (m_data_dilation_strides.size() == 0)
    {
        m_data_dilation_strides = default_strides(this, data_batch_shape, filters_shape);
    }

    if (m_window_movement_strides.size() == 0)
    {
        m_window_movement_strides = default_strides(this, data_batch_shape, filters_shape);
    }

    if (m_window_dilation_strides.size() == 0)
    {
        m_window_dilation_strides = default_strides(this, data_batch_shape, filters_shape);
    }

    if (m_padding_below.size() == 0)
    {
        m_padding_below = default_padding(this, data_batch_shape, filters_shape);
    }

    if (m_padding_above.size() == 0)
    {
        m_padding_above = default_padding(this, data_batch_shape, filters_shape);
    }

    nnfusion::element::Type result_et;
    nnfusion::PartialShape result_shape;

    std::tie(result_et, result_shape) = infer_convolution_forward(this,
                                                                  data_batch_et,
                                                                  filters_et,
                                                                  data_batch_shape,
                                                                  m_data_dilation_strides,
                                                                  m_padding_below,
                                                                  m_padding_above,
                                                                  filters_shape,
                                                                  m_window_movement_strides,
                                                                  m_window_dilation_strides,
                                                                  m_data_format);

    gnode->set_output_type_and_shape(0, result_et, result_shape);
}

nnfusion::Strides Convolution::default_strides(const Op* op,
                                               const nnfusion::PartialShape& data_batch_shape,
                                               const nnfusion::PartialShape& filters_shape)
{
    size_t rank;

    if (data_batch_shape.rank().is_static() && static_cast<size_t>(data_batch_shape.rank()) >= 2)
    {
        rank = static_cast<size_t>(data_batch_shape.rank()) - 2;
    }
    else if (filters_shape.rank().is_static() && static_cast<size_t>(filters_shape.rank()) >= 2)
    {
        rank = static_cast<size_t>(filters_shape.rank()) - 2;
    }
    else
    {
        rank = 0;
    }

    return nnfusion::Strides(rank, 1);
}

Convolution::Convolution(const nnfusion::Strides& window_movement_strides,
                         const nnfusion::Strides& window_dilation_strides,
                         const nnfusion::CoordinateDiff& padding_below,
                         const nnfusion::CoordinateDiff& padding_above,
                         std::string data_format)
    : Convolution(window_movement_strides,
                  window_dilation_strides,
                  padding_below,
                  padding_above,
                  nnfusion::Strides(),
                  data_format)
{
}

CoordinateDiff Convolution::default_padding(const Op* op,
                                            const nnfusion::PartialShape& data_batch_shape,
                                            const nnfusion::PartialShape& filters_shape)
{
    size_t rank;

    if (data_batch_shape.rank().is_static() && static_cast<size_t>(data_batch_shape.rank()) >= 2)
    {
        rank = static_cast<size_t>(data_batch_shape.rank()) - 2;
    }
    else if (filters_shape.rank().is_static() && static_cast<size_t>(filters_shape.rank()) >= 2)
    {
        rank = static_cast<size_t>(filters_shape.rank()) - 2;
    }
    else
    {
        rank = 0;
    }

    return nnfusion::CoordinateDiff(rank, 0);
}

Convolution::Convolution(const nnfusion::Strides& window_movement_strides,
                         const nnfusion::Strides& window_dilation_strides)
    : Convolution(window_movement_strides,
                  window_dilation_strides,
                  nnfusion::CoordinateDiff(),
                  nnfusion::CoordinateDiff())
{
}

Convolution::Convolution(const nnfusion::Strides& window_movement_strides)
    : Convolution(window_movement_strides,
                  nnfusion::Strides(),
                  nnfusion::CoordinateDiff(),
                  nnfusion::CoordinateDiff())
{
}

Convolution::Convolution()
    : Convolution(nnfusion::Strides(),
                  nnfusion::Strides(),
                  nnfusion::CoordinateDiff(),
                  nnfusion::CoordinateDiff())
{
}

void Convolution::infer_shared_memory(std::shared_ptr<graph::GNode> gnode)
{
    for (auto s : get_window_movement_strides())
    {
        if (s != 1)
            return;
    }

    for (auto d : get_window_dilation_strides())
    {
        if (d != 1)
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
    const Shape& input_shape = gnode->get_input_shape(0);
    int channel = get_data_format() == "NCHW" ? 1 : 3;
    auto input_channel_count = input_shape[channel];

    for (size_t i = 0; i < gnode->get_output_shape(0).size(); i++)
    {
        if (i == channel)
            m_shared_memory.push_back(input_channel_count);
        else
            m_shared_memory.push_back(1);
    }
}

ConvolutionBackpropData::ConvolutionBackpropData(
    const Shape& data_batch_shape,
    const nnfusion::Strides& window_movement_strides_forward,
    const nnfusion::Strides& window_dilation_strides_forward,
    const nnfusion::CoordinateDiff& padding_below_forward,
    const nnfusion::CoordinateDiff& padding_above_forward,
    const nnfusion::Strides& data_dilation_strides_forward)
    : Op("ConvolutionBackpropData")
    , m_data_batch_shape(data_batch_shape)
    , m_window_movement_strides_forward(window_movement_strides_forward)
    , m_window_dilation_strides_forward(window_dilation_strides_forward)
    , m_padding_below_forward(padding_below_forward)
    , m_padding_above_forward(padding_above_forward)
    , m_data_dilation_strides_forward(data_dilation_strides_forward)
{
}

void ConvolutionBackpropData::validate_and_infer_types(std::shared_ptr<graph::GNode> gnode)
{
    // Backprop to data is itself convolution, with inputs/outputs/attributes transmogrified as
    // follows.
    //
    //                          Forward   Backward
    // "N" axis for data batch  0         0
    // "C" axis for data batch  1         1
    // "Co" axis for filters    0         0
    // "Ci" axis for filters    1         1
    // "N" axis for output      0         0
    // "C" axis for output      1         1
    // Data batch               x         delta
    // Data batch shape         S_x       S_o
    // Filters                  f         reverse(f) [on spatial axes]
    // Filters shape            S_f       S_f
    // Window movement strides  q_x       p_x
    // Window dilation strides  p_f       p_f
    // Padding below            a_x       (S_f - 1)p_f - a_x
    // Padding above            b_x       (S_f - 1)p_f + ((a_x + (S_x - 1)p_x + b_x - (S_f - 1)p_f) % q_x) - b_x
    // Data dilation strides    p_x       q_x
    // Output shape             S_o       S_x
    //
    // To _validate_, we simply need to check/infer the output shape of the forward convolution,
    // then check to make sure that the incoming delta has the same shape as the forward output.
    //
    // We will also compute and store the various parameters in the "backward" column above, since
    // some backends need them. (TODO(amprocte): Is it just because of the way the reference works
    // that this stuff is needed? If so, we can probably get rid of it and have conv_backprop
    // reference kernels that do the calculations of the backward parameters internally, or supply
    // utility functions to do it.)

    const nnfusion::PartialShape& filters_shape = gnode->get_input_partial_shape(0);
    nnfusion::element::Type filters_et = gnode->get_input_element_type(0);
    const nnfusion::PartialShape& delta_shape = gnode->get_input_partial_shape(1);
    nnfusion::element::Type delta_et = gnode->get_input_element_type(1);

    nnfusion::element::Type forward_result_et;
    nnfusion::PartialShape forward_result_shape;

    std::tie(forward_result_et, forward_result_shape) =
        infer_convolution_forward(this,
                                  delta_et,
                                  filters_et,
                                  m_data_batch_shape,
                                  m_data_dilation_strides_forward,
                                  m_padding_below_forward,
                                  m_padding_above_forward,
                                  filters_shape,
                                  m_window_movement_strides_forward,
                                  m_window_dilation_strides_forward);

    OP_VALIDATION(this, forward_result_shape.compatible(delta_shape))
        << "Inferred forward output shape (" << forward_result_shape << ") does not match shape of "
        << "delta (" << delta_shape << ").";

    gnode->set_output_type_and_shape(0, forward_result_et, m_data_batch_shape);

    //
    // Compute parameters needed for backprop-as-convolution.
    //
    // TODO(amprocte): Remove these fields, compute where needed.
    //
    if (delta_shape.is_static() && filters_shape.is_static())
    {
        size_t spatial_dim_count = static_cast<size_t>(delta_shape.rank()) - 2;

        m_window_movement_strides_backward = m_data_dilation_strides_forward;
        m_window_dilation_strides_backward = m_window_dilation_strides_forward;
        m_data_dilation_strides_backward = m_window_movement_strides_forward;

        m_padding_below_backward.resize(spatial_dim_count);
        m_padding_above_backward.resize(spatial_dim_count);

        for (size_t i = 0; i < spatial_dim_count; i++)
        {
            m_padding_below_backward[i] = (static_cast<ptrdiff_t>(filters_shape[i + 2]) - 1) *
                                              m_window_dilation_strides_forward[i] -
                                          m_padding_below_forward[i];
            m_padding_above_backward[i] =
                (static_cast<ptrdiff_t>(filters_shape[i + 2]) - 1) *
                    m_window_dilation_strides_forward[i] +
                ((m_padding_below_forward[i] +
                  (m_data_batch_shape[i + 2] - 1) * m_data_dilation_strides_forward[i] +
                  m_padding_above_forward[i] -
                  (static_cast<ptrdiff_t>(filters_shape[i + 2]) - 1) *
                      m_window_dilation_strides_forward[i]) %
                 m_window_movement_strides_forward[i]) -
                m_padding_above_forward[i];
        }
    }
}

ConvolutionBackpropFilters::ConvolutionBackpropFilters(
    const Shape& filters_shape,
    const nnfusion::Strides& window_movement_strides_forward,
    const nnfusion::Strides& window_dilation_strides_forward,
    const nnfusion::CoordinateDiff& padding_below_forward,
    const nnfusion::CoordinateDiff& padding_above_forward,
    const nnfusion::Strides& data_dilation_strides_forward)
    : Op("ConvolutionBackpropFilters")
    , m_filters_shape(filters_shape)
    , m_window_movement_strides_forward(window_movement_strides_forward)
    , m_window_dilation_strides_forward(window_dilation_strides_forward)
    , m_padding_below_forward(padding_below_forward)
    , m_padding_above_forward(padding_above_forward)
    , m_data_dilation_strides_forward(data_dilation_strides_forward)
{
}

void ConvolutionBackpropFilters::validate_and_infer_types(std::shared_ptr<graph::GNode> gnode)
{
    // Backprop to filters is itself convolution, with inputs/outputs/attributes transmogrified as
    // follows.
    //
    //                          Forward   Backward
    // "N" axis for data batch  0         1
    // "C" axis for data batch  1         0
    // "Co" axis for filters    0         0
    // "Ci" axis for filters    1         1
    // "N" axis for output      0         1
    // "C" axis for output      1         0
    // Data batch               x         x
    // Data batch shape         S_x       S_x
    // Filters                  f         delta
    // Filters shape            S_f       S_f
    // Window movement strides  q_x       p_f
    // Window dilation strides  p_f       q_x
    // Padding below            a_x       a_x
    // Padding above            b_x       b_x - (a_x + (S_x - 1)p_x + b_x - (S_f - 1)p_f) % q_x
    // Data dilation strides    p_x       p_x
    // Output shape             S_o       S_f
    //
    // To _validate_, we simply need to check/infer the output shape of the forward convolution,
    // then check to make sure that the incoming delta has the same shape as the forward output.
    //
    // We will also compute and store the various parameters in the "backward" column above, since
    // some backends need them. (TODO(amprocte): Is it just because of the way the reference works
    // that this stuff is needed? If so, we can probably get rid of it and have conv_backprop
    // reference kernels that do the calculations of the backward parameters internally, or supply
    // utility functions to do it.)

    const nnfusion::PartialShape& data_batch_shape = gnode->get_input_partial_shape(0);
    nnfusion::element::Type data_batch_et = gnode->get_input_element_type(0);
    const nnfusion::PartialShape& delta_shape = gnode->get_input_shape(1);
    nnfusion::element::Type delta_et = gnode->get_input_element_type(1);

    nnfusion::element::Type forward_result_et;
    nnfusion::PartialShape forward_result_shape;

    std::tie(forward_result_et, forward_result_shape) =
        infer_convolution_forward(this,
                                  data_batch_et,
                                  delta_et,
                                  data_batch_shape,
                                  m_data_dilation_strides_forward,
                                  m_padding_below_forward,
                                  m_padding_above_forward,
                                  m_filters_shape,
                                  m_window_movement_strides_forward,
                                  m_window_dilation_strides_forward);

    OP_VALIDATION(this, forward_result_shape.compatible(delta_shape))
        << "Inferred forward output shape (" << forward_result_shape << ") does not match shape of "
        << "delta (" << delta_shape << ").";

    gnode->set_output_type_and_shape(0, forward_result_et, m_filters_shape);

    //
    // Compute parameters needed for backprop-as-convolution.
    //
    // TODO(amprocte): Remove these fields, compute where needed.
    //
    if (delta_shape.is_static() && data_batch_shape.is_static())
    {
        size_t spatial_dim_count = static_cast<size_t>(delta_shape.rank()) - 2;

        m_window_movement_strides_backward = m_window_dilation_strides_forward;
        m_window_dilation_strides_backward = m_window_movement_strides_forward;
        m_padding_below_backward = m_padding_below_forward;
        m_data_dilation_strides_backward = m_data_dilation_strides_forward;

        m_padding_above_backward.resize(spatial_dim_count);

        for (size_t i = 0; i < spatial_dim_count; i++)
        {
            m_padding_above_backward[i] =
                m_padding_above_forward[i] -
                (m_padding_below_forward[i] +
                 (static_cast<ptrdiff_t>(data_batch_shape[i + 2]) - 1) *
                     m_data_dilation_strides_forward[i] +
                 m_padding_above_forward[i] -
                 (m_filters_shape[i + 2] - 1) * m_window_dilation_strides_forward[i]) %
                    m_window_movement_strides_forward[i];
        }
    }
}
