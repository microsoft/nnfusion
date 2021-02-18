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

#include "nnfusion/core/operators/util/elementwise_arithmetic.hpp"

namespace nnfusion
{
    namespace op
    {
        /// \brief Batched max pooling operation, with optional padding and window stride.
        class MaxPool : public Op
        {
        public:
            /// \brief Constructs a batched max pooling operation.
            ///
            /// \param window_shape The window shape.
            /// \param window_movement_strides The window movement strides.
            /// \param padding_below The below-padding shape.
            /// \param padding_above The above-padding shape.
            MaxPool(const nnfusion::Shape& window_shape,
                    const nnfusion::Strides& window_movement_strides,
                    const nnfusion::Shape& padding_below,
                    const nnfusion::Shape& padding_above,
                    std::string data_format = "NCHW");

            void validate_and_infer_types(std::shared_ptr<graph::GNode> gnode) override;
            void infer_shared_memory(std::shared_ptr<graph::GNode> gnode) override;
            /// \brief Constructs a batched, unpadded max pooling operation (i.e., all padding shapes are set to 0).
            ///
            /// \param window_shape The window shape.
            /// \param window_movement_strides The window movement strides.
            MaxPool(const nnfusion::Shape& window_shape,
                    const nnfusion::Strides& window_movement_strides,
                    std::string data_format = "NCHW");

            /// \brief Constructs an unstrided batched max pooling operation (i.e., all window movement strides are 1 and all padding shapes are set to 0).
            ///
            /// \param window_shape The window shape.
            MaxPool(const nnfusion::Shape& window_shape, std::string data_format = "NCHW");

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
            /// \return The data format.
            const std::string& get_data_format() const { return m_data_format; }
        protected:
            nnfusion::Shape m_window_shape;
            nnfusion::Strides m_window_movement_strides;
            nnfusion::Shape m_padding_below;
            nnfusion::Shape m_padding_above;
            std::string m_data_format;
        };

        class MaxPoolBackprop : public Op
        {
        public:
            MaxPoolBackprop(const nnfusion::Shape& window_shape,
                            const nnfusion::Strides& window_movement_strides,
                            const nnfusion::Shape& padding_below,
                            const nnfusion::Shape& padding_above,
                            const std::shared_ptr<MaxPool>& forward_op = nullptr);

            void validate_and_infer_types(std::shared_ptr<graph::GNode> gnode) override;

            const nnfusion::Shape& get_window_shape() const { return m_window_shape; }
            const nnfusion::Strides& get_window_movement_strides() const
            {
                return m_window_movement_strides;
            }
            const nnfusion::Shape& get_padding_below() const { return m_padding_below; }
            const nnfusion::Shape& get_padding_above() const { return m_padding_above; }
            /// \return A pointer to the corresponding `MaxPool` forward prop op. This may be
            ///         `nullptr` if no such pointer was provided at construction time, or if the
            ///         forward op has been freed due to graph rewriting.
            std::shared_ptr<MaxPool> get_forward_op() const;

        protected:
            nnfusion::Shape m_window_shape;
            nnfusion::Strides m_window_movement_strides;
            nnfusion::Shape m_padding_below;
            nnfusion::Shape m_padding_above;
            std::weak_ptr<MaxPool> m_forward_op;
        };
    }
}
