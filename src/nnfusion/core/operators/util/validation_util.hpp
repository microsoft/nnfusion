// Microsoft (c) 2019, NNFusion Team

#pragma once

#include "nnfusion/common/coordinate_diff.hpp"
#include "nnfusion/core/operators/op.hpp"

namespace nnfusion
{
    namespace op
    {
        nnfusion::PartialShape infer_windowed_reduction_output_shape(
            const Op* op,
            const nnfusion::PartialShape& data_shape,
            const nnfusion::Strides& data_dilation,
            const nnfusion::CoordinateDiff& data_padding_below,
            const nnfusion::CoordinateDiff& data_padding_above,
            const nnfusion::PartialShape& window_shape,
            const nnfusion::Strides& window_strides,
            const nnfusion::Strides& window_dilation,
            bool is_window_all_in_padding_allowed);

        std::tuple<nnfusion::element::Type, nnfusion::PartialShape>
            infer_convolution_forward(const Op* op,
                                      nnfusion::element::Type et_batch,
                                      nnfusion::element::Type et_filters,
                                      const nnfusion::PartialShape& data_batch_shape,
                                      const nnfusion::Strides& data_dilation,
                                      const nnfusion::CoordinateDiff& data_padding_below,
                                      const nnfusion::CoordinateDiff& data_padding_above,
                                      const nnfusion::PartialShape& filters_shape,
                                      const nnfusion::Strides& filter_strides,
                                      const nnfusion::Strides& filter_dilation);
        nnfusion::PartialShape
            infer_batched_pooling_forward(const Op* op,
                                          const nnfusion::PartialShape& data_batch_shape,
                                          const nnfusion::CoordinateDiff& data_padding_below,
                                          const nnfusion::CoordinateDiff& data_padding_above,
                                          const nnfusion::PartialShape& window_shape,
                                          const nnfusion::Strides& window_strides,
                                          bool is_window_all_in_padding_allowed);

        std::tuple<nnfusion::element::Type, nnfusion::PartialShape, nnfusion::PartialShape>
            infer_batch_norm_forward(const Op* op,
                                     nnfusion::element::Type input_element_type,
                                     nnfusion::element::Type gamma_element_type,
                                     nnfusion::element::Type beta_element_type,
                                     nnfusion::element::Type mean_element_type,
                                     nnfusion::element::Type variance_element_type,
                                     const nnfusion::PartialShape& input_shape,
                                     const nnfusion::PartialShape& gamma_shape,
                                     const nnfusion::PartialShape& beta_shape,
                                     const nnfusion::PartialShape& mean_shape,
                                     const nnfusion::PartialShape& variance_shape);

        std::tuple<nnfusion::element::Type, nnfusion::PartialShape, nnfusion::PartialShape>
            infer_batch_norm_forward(const Op* op,
                                     nnfusion::element::Type input_element_type,
                                     nnfusion::element::Type gamma_element_type,
                                     nnfusion::element::Type beta_element_type,
                                     const nnfusion::PartialShape& input_shape,
                                     const nnfusion::PartialShape& gamma_shape,
                                     const nnfusion::PartialShape& beta_shape);
    }
}
