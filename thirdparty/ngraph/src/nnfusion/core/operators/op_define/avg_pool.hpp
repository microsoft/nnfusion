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

#pragma once

#include "../op.hpp"

namespace nnfusion
{
    namespace op
    {
        /// \brief Batched average pooling operation, with optional padding and window stride.
        ///
        class AvgPool : public Op
        {
        public:
            /// \brief Constructs a batched average pooling operation.
            ///
            /// `[d1, dn]`
            /// \param window_shape The window shape.<br>
            /// `[n]`
            /// \param window_movement_strides The window movement strides.<br>
            /// `[n]`
            /// \param padding_below The below-padding shape.<br>
            /// `[n]`
            /// \param padding_above The above-padding shape.<br>
            /// `[n]`
            /// \param include_padding_in_avg_computation If true then averages include padding
            ///  elements, each treated as the number zero.  If false, padding elements are entirely
            ///  ignored when computing averages.
            AvgPool(const nnfusion::Shape& window_shape,
                    const nnfusion::Strides& window_movement_strides,
                    const nnfusion::Shape& padding_below,
                    const nnfusion::Shape& padding_above,
                    bool include_padding_in_avg_computation = false);

            /// \brief Constructs a batched, unpadded average pooling operation (i.e., all padding shapes are set to 0).
            ///
            /// `[d1, ..., dn]`
            /// \param window_shape The window shape.<br>
            /// `[n]`
            /// \param window_movement_strides The window movement strides.<br>
            /// `[n]`
            AvgPool(const nnfusion::Shape& window_shape,
                    const nnfusion::Strides& window_movement_strides);

            /// \brief Constructs an unstrided batched convolution operation (i.e., all window movement strides are 1 and all padding shapes are set to 0).
            ///
            /// `[d1, ..., dn]`
            /// \param window_shape The window shape.<br>
            /// `[n]`
            AvgPool(const nnfusion::Shape& window_shape);

            void validate_and_infer_types(std::shared_ptr<graph::GNode> gnode) override;

            /// \return The window shape.
            const nnfusion::Shape& get_window_shape() const { return m_window_shape; }
            /// \return The window movement strides.
            const nnfusion::Strides& get_window_movement_strides() const
            {
                return m_window_movement_strides;
            }
            /// \return The below-padding shape.
            const nnfusion::Shape& get_padding_below() const { return m_padding_below; }
            /// \return The above-padding shape.
            const nnfusion::Shape& get_padding_above() const { return m_padding_above; }
            bool get_include_padding_in_avg_computation() const
            {
                return m_include_padding_in_avg_computation;
            }

        protected:
            nnfusion::Shape m_window_shape;
            nnfusion::Strides m_window_movement_strides;
            nnfusion::Shape m_padding_below;
            nnfusion::Shape m_padding_above;
            bool m_include_padding_in_avg_computation;
        };

        class AvgPoolBackprop : public Op
        {
        public:
            AvgPoolBackprop(const nnfusion::Shape& forward_arg_shape,
                            const nnfusion::Shape& window_shape,
                            const nnfusion::Strides& window_movement_strides,
                            const nnfusion::Shape& padding_below,
                            const nnfusion::Shape& padding_above,
                            bool include_padding_in_avg_computation);

            void validate_and_infer_types(std::shared_ptr<graph::GNode> gnode) override;

            const nnfusion::Shape& get_forward_arg_shape() const { return m_forward_arg_shape; }
            const nnfusion::Shape& get_window_shape() const { return m_window_shape; }
            const nnfusion::Strides& get_window_movement_strides() const
            {
                return m_window_movement_strides;
            }
            const nnfusion::Shape& get_padding_below() const { return m_padding_below; }
            const nnfusion::Shape& get_padding_above() const { return m_padding_above; }
            bool get_include_padding_in_avg_computation() const
            {
                return m_include_padding_in_avg_computation;
            }

        protected:
            nnfusion::Shape m_forward_arg_shape;
            nnfusion::Shape m_window_shape;
            nnfusion::Strides m_window_movement_strides;
            nnfusion::Shape m_padding_below;
            nnfusion::Shape m_padding_above;
            bool m_include_padding_in_avg_computation;
        };
    }
}
