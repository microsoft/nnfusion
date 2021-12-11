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

#include "validation_util.hpp"

#include "nnfusion/common/util.hpp"

using namespace std;
using namespace nnfusion::op;

//
// Infers the output shape of a windowed reduction operation, where the data may be dilated and/or
// padded, and the reduction window may be strided and/or dilated.
//
// TODO(amprocte): The messages here would be a bit friendlier if we didn't say "after
// padding/after dilation" for cases where there is actually no padding/dilation.
//
nnfusion::PartialShape nnfusion::op::infer_windowed_reduction_output_shape(
    const Op* op,
    const nnfusion::PartialShape& data_shape,
    const nnfusion::Strides& data_dilation,
    const nnfusion::CoordinateDiff& data_padding_below,
    const nnfusion::CoordinateDiff& data_padding_above,
    const nnfusion::PartialShape& window_shape,
    const nnfusion::Strides& window_strides,
    const nnfusion::Strides& window_dilation,
    bool is_window_all_in_padding_allowed)
{
    nnfusion::PartialShape data_shape_merged{nnfusion::PartialShape::dynamic()};

    OP_VALIDATION(op,
                  data_shape_merged.merge_rank(data_shape.rank()) &&
                      data_shape_merged.merge_rank(data_dilation.size()) &&
                      data_shape_merged.merge_rank(data_padding_below.size()) &&
                      data_shape_merged.merge_rank(data_padding_above.size()) &&
                      data_shape_merged.merge_rank(window_shape.rank()) &&
                      data_shape_merged.merge_rank(window_strides.size()) &&
                      data_shape_merged.merge_rank(window_dilation.size()))
        << "Ranks for data shape (" << data_shape << "), data dilation (" << data_dilation
        << "), padding below (" << data_padding_below << "), padding above (" << data_padding_above
        << "), window shape (" << window_shape << "), window strides (" << window_strides
        << "), and window dilation (" << window_dilation << ") do not match.";

    nnfusion::PartialShape output_shape = nnfusion::PartialShape::dynamic(data_shape_merged.rank());

    if (output_shape.rank().is_static())
    {
        for (size_t i = 0; i < static_cast<size_t>(output_shape.rank()); i++)
        {
            OP_VALIDATION(op, data_dilation[i] > 0) << "Data dilation (" << data_dilation
                                                    << ") has zero dimension at axis " << i << ".";
            OP_VALIDATION(op, window_strides[i] > 0) << "Window strides (" << window_strides
                                                     << ") has zero dimension at axis " << i << ".";
            OP_VALIDATION(op, window_dilation[i] > 0) << "Window dilation (" << window_dilation
                                                      << ") has zero dimension at axis " << i
                                                      << ".";

            bool data_dim_static = data_shape.rank().is_static() && data_shape[i].is_static();
            bool window_dim_static = window_shape.rank().is_static() && window_shape[i].is_static();

            ptrdiff_t data_padded_dilated_dim = -1;
            if (data_dim_static)
            {
                data_padded_dilated_dim = (static_cast<ptrdiff_t>(data_dilation[i]) *
                                           (static_cast<ptrdiff_t>(data_shape[i]) - 1)) +
                                          1 + data_padding_below[i] + data_padding_above[i];
                OP_VALIDATION(op, data_padded_dilated_dim > 0)
                    << "Data shape after padding and dilation has dimension less than 1 (dim: "
                    << data_padded_dilated_dim << ") at axis " << i << ".";
            }

            ptrdiff_t window_dilated_dim = -1;
            if (window_dim_static)
            {
                window_dilated_dim = static_cast<ptrdiff_t>(window_dilation[i]) *
                                         (static_cast<ptrdiff_t>(window_shape[i]) - 1) +
                                     1;

                OP_VALIDATION(op, window_dilated_dim > 0)
                    << "Window after dilation has dimension less than 1 (dim: "
                    << window_dilated_dim << ") at axis " << i << ".";

                OP_VALIDATION(op,
                              is_window_all_in_padding_allowed ||
                                  (window_dilated_dim > data_padding_below[i] &&
                                   window_dilated_dim > data_padding_above[i]))
                    << "Window after dilation is sometimes entirely in the padding area for axis "
                    << i << " (dilated window dimension: " << window_dilated_dim
                    << ", padding below dimension: " << data_padding_below[i]
                    << ", padding above dimension: " << data_padding_above[i]
                    << ") and this is not "
                    << "allowed.";
            }

            if (data_dim_static && window_dim_static)
            {
                OP_VALIDATION(op, window_dilated_dim <= data_padded_dilated_dim)
                    << "Window after dilation has dimension (dim: " << window_dilated_dim
                    << ") larger than the data shape after padding (dim: "
                    << data_padded_dilated_dim << ") at axis " << i << ".";

                output_shape[i] =
                    nnfusion::ceil_div(static_cast<size_t>(data_padded_dilated_dim) -
                                           static_cast<size_t>(window_dilated_dim) + 1,
                                       window_strides[i]);
            }
        }
    }

    return output_shape;
}

std::tuple<nnfusion::element::Type, nnfusion::PartialShape>
    nnfusion::op::infer_convolution_forward(const Op* op,
                                            nnfusion::element::Type et_batch,
                                            nnfusion::element::Type et_filters,
                                            const nnfusion::PartialShape& data_batch_shape,
                                            const nnfusion::Strides& data_dilation,
                                            const nnfusion::CoordinateDiff& data_padding_below,
                                            const nnfusion::CoordinateDiff& data_padding_above,
                                            const nnfusion::PartialShape& filters_shape,
                                            const nnfusion::Strides& filter_strides,
                                            const nnfusion::Strides& filter_dilation,
                                            std::string data_format)
{
    nnfusion::element::Type et_result;

    OP_VALIDATION(op, nnfusion::element::Type::merge(et_result, et_batch, et_filters))
        << "Element types for data batch and filters do not match (data batch element type: "
        << et_batch << ", filters element type: " << et_filters << ").";

    nnfusion::Rank data_batch_filters_rank{nnfusion::Rank::dynamic()};

    OP_VALIDATION(op,
                  nnfusion::Rank::merge(
                      data_batch_filters_rank, data_batch_shape.rank(), filters_shape.rank()))
        << "Data batch and filters rank do not match (data batch shape: " << data_batch_shape
        << ", filters shape: " << filters_shape << ").";

    OP_VALIDATION(op,
                  data_batch_filters_rank.is_dynamic() ||
                      static_cast<size_t>(data_batch_filters_rank) >= 3)
        << "Data batch and filters must have rank of at least 3 (one batch axis, "
        << "one input-channel axis, and at least one spatial dimension) "
        << "(data batch shape: " << data_batch_shape << ", filters shape: " << filters_shape
        << ").";

    nnfusion::Rank spatial_rank{nnfusion::Rank::dynamic()};
    OP_VALIDATION(
        op,
        nnfusion::Rank::merge(spatial_rank, spatial_rank, data_batch_filters_rank - 2) &&
            nnfusion::Rank::merge(spatial_rank, spatial_rank, data_dilation.size()) &&
            nnfusion::Rank::merge(spatial_rank, spatial_rank, data_padding_below.size()) &&
            nnfusion::Rank::merge(spatial_rank, spatial_rank, data_padding_above.size()) &&
            nnfusion::Rank::merge(spatial_rank, spatial_rank, filter_strides.size()) &&
            nnfusion::Rank::merge(spatial_rank, spatial_rank, filter_dilation.size()))
        << "Ranks for data item shape/filters shape (data batch has shape " << data_batch_shape
        << ", so data item rank is " << (data_batch_shape.rank() - 2) << " and filters have shape "
        << filters_shape << ", so filters spatial rank is " << (filters_shape.rank() - 2)
        << "), data dilation (" << data_dilation << "), padding below (" << data_padding_below
        << "), padding above (" << data_padding_above << "), filter strides (" << filter_strides
        << "), and filter dilation (" << filter_dilation << ") do not match.";

    OP_VALIDATION(op, data_format == "NCW" || data_format == "NCHW" || data_format == "NHWC")
        << "data format must be Conv1D: NCW, Conv2D: NCHW or NHWC.";

    nnfusion::Dimension batch_size =
        (data_batch_shape.rank().is_static() ? data_batch_shape[0]
                                             : nnfusion::Dimension::dynamic());
    nnfusion::Dimension data_channel_count =
        (data_batch_shape.rank().is_static()
             ? (data_format == "NCW" || data_format == "NCHW") ? data_batch_shape[1]
                                                               : data_batch_shape[3]
             : nnfusion::Dimension::dynamic());
    nnfusion::PartialShape data_spatial_shape(nnfusion::PartialShape::dynamic(spatial_rank));

    nnfusion::Dimension filter_output_channel_count =
        (filters_shape.rank().is_static()
             ? (data_format == "NCW" || data_format == "NCHW") ? filters_shape[0] : filters_shape[3]
             : nnfusion::Dimension::dynamic());
    nnfusion::Dimension filter_input_channel_count =
        (filters_shape.rank().is_static()
             ? (data_format == "NCW" || data_format == "NCHW") ? filters_shape[1] : filters_shape[2]
             : nnfusion::Dimension::dynamic());
    nnfusion::PartialShape filter_spatial_shape(nnfusion::PartialShape::dynamic(spatial_rank));

    //
    // Note: spatial_rank is definitely static at this point.
    //

    for (size_t i = 0; i < static_cast<size_t>(spatial_rank); i++)
    {
        if (data_batch_shape.rank().is_static())
        {
            data_spatial_shape[i] = (data_format == "NCW" || data_format == "NCHW")
                                        ? data_batch_shape[i + 2]
                                        : data_batch_shape[i + 1];
        }

        if (filters_shape.rank().is_static())
        {
            filter_spatial_shape[i] = (data_format == "NCW" || data_format == "NCHW")
                                          ? filters_shape[i + 2]
                                          : filters_shape[i];
        }
    }

    OP_VALIDATION(op, batch_size.is_dynamic() || static_cast<size_t>(batch_size) > 0)
        << "Batch size is zero.";

    nnfusion::Dimension merged_channel_count;

    OP_VALIDATION(op,
                  nnfusion::Dimension::merge(
                      merged_channel_count, data_channel_count, filter_input_channel_count))
        << "Data batch channel count (" << data_channel_count << ") does not match filter input "
        << "channel count (" << filter_input_channel_count << ").";

    OP_VALIDATION(
        op, merged_channel_count.is_dynamic() || static_cast<size_t>(merged_channel_count) > 0)
        << "Data batch channel count and/or filter input channel count is zero.";

    OP_VALIDATION(op,
                  filter_output_channel_count.is_dynamic() ||
                      static_cast<size_t>(filter_output_channel_count) > 0)
        << "Filter output channel count is zero.";

    nnfusion::PartialShape data_output_shape =
        infer_windowed_reduction_output_shape(op,
                                              data_spatial_shape,
                                              data_dilation,
                                              data_padding_below,
                                              data_padding_above,
                                              filter_spatial_shape,
                                              filter_strides,
                                              filter_dilation,
                                              true);

    nnfusion::PartialShape batch_output_shape(nnfusion::PartialShape::dynamic(spatial_rank + 2));

    if (data_format == "NCW" || data_format == "NCHW")
    {
        batch_output_shape[0] = batch_size;
        batch_output_shape[1] = filter_output_channel_count;

        for (size_t i = 0; i < static_cast<size_t>(spatial_rank); i++)
        {
            batch_output_shape[i + 2] = data_output_shape[i];
        }
    }
    else
    {
        batch_output_shape[0] = batch_size;
        batch_output_shape[3] = filter_output_channel_count;

        for (size_t i = 0; i < static_cast<size_t>(spatial_rank); i++)
        {
            batch_output_shape[i + 1] = data_output_shape[i];
        }
    }

    return std::make_tuple(et_result, batch_output_shape);
}
//
// Infers the output batch shape and element type for batched pooling fprop.
//
nnfusion::PartialShape
    nnfusion::op::infer_batched_pooling_forward(const Op* op,
                                                const nnfusion::PartialShape& data_batch_shape,
                                                const nnfusion::CoordinateDiff& data_padding_below,
                                                const nnfusion::CoordinateDiff& data_padding_above,
                                                const nnfusion::PartialShape& window_shape,
                                                const nnfusion::Strides& window_strides,
                                                bool is_window_all_in_padding_allowed,
                                                std::string data_format)
{
    OP_VALIDATION(op,
                  data_batch_shape.rank().is_dynamic() ||
                      static_cast<size_t>(data_batch_shape.rank()) >= 3)
        << "Data batch must have rank of at least 3 (one batch axis, "
        << "one input-channel axis, and at least one spatial dimension) "
        << "(data batch shape: " << data_batch_shape << ").";

    nnfusion::PartialShape data_spatial_shape{nnfusion::PartialShape::dynamic()};

    OP_VALIDATION(op,
                  data_spatial_shape.merge_rank(data_batch_shape.rank() - 2) &&
                      data_spatial_shape.merge_rank(data_padding_below.size()) &&
                      data_spatial_shape.merge_rank(data_padding_above.size()) &&
                      data_spatial_shape.merge_rank(window_shape.rank()) &&
                      data_spatial_shape.merge_rank(window_strides.size()))
        << "Ranks for data item shape (data batch has shape " << data_batch_shape
        << ", so data item rank is " << (data_batch_shape.rank() - 2) << "), padding below ("
        << data_padding_below << "), padding above (" << data_padding_above << "), window shape ("
        << window_shape << "), and window strides (" << window_strides << ") do not match.";

    OP_VALIDATION(op, data_format == "NCHW" || data_format == "NHWC")
        << "data format must be NCHW or NHWC.";

    nnfusion::Dimension batch_size{nnfusion::Dimension::dynamic()};
    nnfusion::Dimension channel_count{nnfusion::Dimension::dynamic()};
    nnfusion::PartialShape data_output_spatial_shape{
        nnfusion::PartialShape::dynamic(data_spatial_shape.rank())};

    if (data_batch_shape.rank().is_static())
    {
        batch_size = data_batch_shape[0];
        channel_count = data_format == "NCHW" ? data_batch_shape[1] : data_batch_shape[3];

        for (size_t i = 0; i < static_cast<size_t>(data_spatial_shape.rank()); i++)
        {
            data_spatial_shape[i] =
                data_format == "NCHW" ? data_batch_shape[i + 2] : data_batch_shape[i + 1];
        }

        OP_VALIDATION(op, batch_size.is_dynamic() || static_cast<size_t>(batch_size) > 0)
            << "Batch size is zero.";

        OP_VALIDATION(op, channel_count.is_dynamic() || static_cast<size_t>(channel_count) > 0)
            << "Channel count is zero.";

        // For pooling ops we don't need dilation, so we fill in the identity value (all 1).
        nnfusion::Strides data_dilation(static_cast<size_t>(data_spatial_shape.rank()), 1);
        nnfusion::Strides window_dilation(static_cast<size_t>(data_spatial_shape.rank()), 1);

        data_output_spatial_shape =
            infer_windowed_reduction_output_shape(op,
                                                  data_spatial_shape,
                                                  data_dilation,
                                                  data_padding_below,
                                                  data_padding_above,
                                                  window_shape,
                                                  window_strides,
                                                  window_dilation,
                                                  is_window_all_in_padding_allowed);
    }

    nnfusion::PartialShape data_batch_output_shape{
        nnfusion::PartialShape::dynamic(data_output_spatial_shape.rank() + 2)};

    if (data_format == "NCHW")
    {
        data_batch_output_shape[0] = batch_size;
        data_batch_output_shape[1] = channel_count;

        for (size_t i = 0; i < static_cast<size_t>(data_spatial_shape.rank()); i++)
        {
            data_batch_output_shape[i + 2] = data_output_spatial_shape[i];
        }
    }
    else
    {
        data_batch_output_shape[0] = batch_size;
        data_batch_output_shape[3] = channel_count;

        for (size_t i = 0; i < static_cast<size_t>(data_spatial_shape.rank()); i++)
        {
            data_batch_output_shape[i + 1] = data_output_spatial_shape[i];
        }
    }

    return data_batch_output_shape;
}

struct ChannelShapedInputSpec
{
    nnfusion::element::Type m_element_type;
    nnfusion::PartialShape m_shape;
    std::string m_input_name;
};

static std::tuple<nnfusion::element::Type, nnfusion::PartialShape, nnfusion::PartialShape>
    infer_batch_norm_forward_helper(
        const Op* op,
        nnfusion::element::Type input_element_type,
        const nnfusion::PartialShape& input_shape,
        const std::vector<ChannelShapedInputSpec>& channel_shaped_inputs)
{
    // Built up a slash-separated string naming all the channel-shaped inputs, for use in error
    // messages.
    std::stringstream ss;
    bool first = true;
    for (auto& inp : channel_shaped_inputs)
    {
        if (!first)
        {
            ss << "/";
        }
        ss << inp.m_input_name;
        first = false;
    }
    std::string channel_input_names = ss.str();

    // Infer output element type.
    nnfusion::element::Type et_result{input_element_type};

    for (auto& inp : channel_shaped_inputs)
    {
        OP_VALIDATION(op, nnfusion::element::Type::merge(et_result, et_result, inp.m_element_type))
            << "Input element types do not match.";
    }

    // Extract channel dimension from input shape.
    nnfusion::Dimension channel_dim{nnfusion::Dimension::dynamic()};

    OP_VALIDATION(op, input_shape.is_dynamic() || static_cast<size_t>(input_shape.rank()) >= 2)
        << "Input argument must have rank of at least 2 (input argument shape: " << input_shape
        << ").";

    if (input_shape.rank().is_static())
    {
        channel_dim = input_shape[1];
    }

    // Infer gamma/beta/mu/sigma shape, which must be consistent with a vector of size "channel_dim".
    nnfusion::PartialShape channel_shape{nnfusion::PartialShape::dynamic()};

    for (auto& inp : channel_shaped_inputs)
    {
        OP_VALIDATION(op, nnfusion::PartialShape::merge_into(channel_shape, inp.m_shape))
            << "Shapes for " << channel_input_names << " do not match.";
    }

    OP_VALIDATION(op, channel_shape.merge_rank(1)) << "Shape for " << channel_input_names << " ("
                                                   << channel_shape << ") does not have rank 1.";

    OP_VALIDATION(op, nnfusion::Dimension::merge(channel_dim, channel_dim, channel_shape[0]))
        << "Input channel dimension (" << channel_dim << ") does not match shape for "
        << channel_input_names << " (" << channel_shape << ").";

    OP_VALIDATION(op, channel_dim.is_dynamic() || static_cast<size_t>(channel_dim) >= 1)
        << "Channel count must be at least 1.";

    // Batch result shape is same as the input shape, except we may possibly have inferred more
    // information from the channel count via gamma/beta/etc.
    nnfusion::PartialShape batch_result_shape{input_shape};

    if (batch_result_shape.rank().is_static())
    {
        batch_result_shape[1] = channel_dim;
    }

    return std::make_tuple(et_result, batch_result_shape, nnfusion::PartialShape{channel_dim});
}

std::tuple<nnfusion::element::Type, nnfusion::PartialShape, nnfusion::PartialShape>
    nnfusion::op::infer_batch_norm_forward(const Op* op,
                                           nnfusion::element::Type input_element_type,
                                           nnfusion::element::Type gamma_element_type,
                                           nnfusion::element::Type beta_element_type,
                                           nnfusion::element::Type mean_element_type,
                                           nnfusion::element::Type variance_element_type,
                                           const nnfusion::PartialShape& input_shape,
                                           const nnfusion::PartialShape& gamma_shape,
                                           const nnfusion::PartialShape& beta_shape,
                                           const nnfusion::PartialShape& mean_shape,
                                           const nnfusion::PartialShape& variance_shape)
{
    return infer_batch_norm_forward_helper(op,
                                           input_element_type,
                                           input_shape,
                                           {{gamma_element_type, gamma_shape, "gamma"},
                                            {beta_element_type, beta_shape, "beta"},
                                            {mean_element_type, mean_shape, "mean"},
                                            {variance_element_type, variance_shape, "variance"}});
}

std::tuple<nnfusion::element::Type, nnfusion::PartialShape, nnfusion::PartialShape>
    nnfusion::op::infer_batch_norm_forward(const Op* op,
                                           nnfusion::element::Type input_element_type,
                                           nnfusion::element::Type gamma_element_type,
                                           nnfusion::element::Type beta_element_type,
                                           const nnfusion::PartialShape& input_shape,
                                           const nnfusion::PartialShape& gamma_shape,
                                           const nnfusion::PartialShape& beta_shape)
{
    return infer_batch_norm_forward_helper(
        op,
        input_element_type,
        input_shape,
        {{gamma_element_type, gamma_shape, "gamma"}, {beta_element_type, beta_shape, "beta"}});
}
