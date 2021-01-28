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
#include "nnfusion/common/coordinate_diff.hpp"

namespace nnfusion
{
    namespace op
    {
        /// \brief Batched convolution operation, with optional window dilation and stride.
        ///
        class Convolution : public Op
        {
        public:
            /// \brief Constructs a batched convolution operation.
            ///
            /// \param window_movement_strides The window movement strides.<br>
            /// `[f]`
            /// \param window_dilation_strides The window dilation strides.<br>
            /// `[f]`
            /// \param padding_below The padding-below sizes.<br>
            /// `[f]`
            /// \param padding_above The padding-above sizes.<br>
            /// `[f]`
            /// \param data_dilation_strides The data dilation strides.<br>
            /// `[f]`
            ///
            /// Output `[N, C_OUT, R1, ... Rf]`
            ///
            Convolution(const nnfusion::Strides& window_movement_strides,
                        const nnfusion::Strides& window_dilation_strides,
                        const nnfusion::CoordinateDiff& padding_below,
                        const nnfusion::CoordinateDiff& padding_above,
                        const nnfusion::Strides& data_dilation_strides,
                        std::string data_format = "NCHW");

            /// \brief Constructs a batched convolution operation with no data dilation (i.e., all data dilation strides are 1).
            ///
            /// \param window_movement_strides The window movement strides.<br>
            /// `[f]`
            /// \param window_dilation_strides The window dilation strides.<br>
            /// `[f]`
            /// \param padding_below The padding-below sizes.<br>
            /// `[f]`
            /// \param padding_above The padding-above sizes.<br>
            /// `[f]`
            ///
            /// Output `[N, C_OUT, R1, ... Rf]`
            ///
            Convolution(const nnfusion::Strides& window_movement_strides,
                        const nnfusion::Strides& window_dilation_strides,
                        const nnfusion::CoordinateDiff& padding_below,
                        const nnfusion::CoordinateDiff& padding_above,
                        std::string data_format = "NCHW");

            /// \brief Constructs a batched convolution operation with no padding or data dilation (i.e., padding above and below are 0 everywhere, and all data dilation strides are 1).
            ///
            /// \param window_movement_strides The window movement strides.<br>
            /// `[f]`
            /// \param window_dilation_strides The window dilation strides.<br>
            /// `[f]`
            ///
            /// Output `[N, C_OUT, R1, ... Rf]`
            ///
            Convolution(const nnfusion::Strides& window_movement_strides,
                        const nnfusion::Strides& window_dilation_strides);

            /// \brief Constructs a batched convolution operation with no window dilation, padding, or data dilation (i.e., padding above and below are 0 everywhere, and all window/data dilation strides are 1).
            ///
            /// \param window_movement_strides The window movement strides.<br>
            /// `[f]`
            ///
            /// Output `[N, C_OUT, R1, ... Rf]`
            ///
            Convolution(const nnfusion::Strides& window_movement_strides);

            /// \brief Constructs a batched convolution operation with no window dilation or movement stride (i.e., padding above and below are 0 everywhere, and all window/data dilation strides and window movement strides are 1).
            ///
            /// Output `[N, C_OUT, R1, ... Rf]`
            ///
            Convolution();

            void validate_and_infer_types(std::shared_ptr<graph::GNode> gnode) override;
            void infer_shared_memory(std::shared_ptr<graph::GNode> gnode) override;
            /// \return The window movement strides.
            const nnfusion::Strides& get_window_movement_strides() const
            {
                return m_window_movement_strides;
            }
            /// \return The window dilation strides.
            const nnfusion::Strides& get_window_dilation_strides() const
            {
                return m_window_dilation_strides;
            }
            /// \return The padding-below sizes (possibly negative).
            const nnfusion::CoordinateDiff& get_padding_below() const { return m_padding_below; }
            /// \return The padding-above sizes (possibly negative).
            const nnfusion::CoordinateDiff& get_padding_above() const { return m_padding_above; }
            /// \return The input data dilation strides.
            const nnfusion::Strides& get_data_dilation_strides() const
            {
                return m_data_dilation_strides;
            }
            /// \return The data format.
            const std::string& get_data_format() const { return m_data_format; }
        protected:
            nnfusion::Strides m_window_movement_strides;
            nnfusion::Strides m_window_dilation_strides;
            nnfusion::CoordinateDiff m_padding_below;
            nnfusion::CoordinateDiff m_padding_above;
            nnfusion::Strides m_data_dilation_strides;
            std::string m_data_format;

        private:
            static nnfusion::Strides default_strides(const Op* op,
                                                     const nnfusion::PartialShape& data_batch_shape,
                                                     const nnfusion::PartialShape& filters_shape);
            static nnfusion::CoordinateDiff
                default_padding(const Op* op,
                                const nnfusion::PartialShape& data_batch_shape,
                                const nnfusion::PartialShape& filters_shape);
        };

        /// \brief Data batch backprop for batched convolution operation.
        class ConvolutionBackpropData : public Op
        {
        public:
            /// \brief Constructs a batched-convolution data batch-backprop operation.
            ///
            /// \param data_batch_shape The shape of the data batch from forward-prop.
            /// \param window_movement_strides_forward The window movement strides from forward-prop.
            /// \param window_dilation_strides_forward The window dilation strides from forward-prop.
            /// \param padding_below_forward The padding-below sizes from forward-prop.
            /// \param padding_above_forward The padding-above sizes from forward-prop.
            /// \param data_dilation_strides_forward The data dilation strides from forward-prop.
            ConvolutionBackpropData(const nnfusion::Shape& data_batch_shape,
                                    const nnfusion::Strides& window_movement_strides_forward,
                                    const nnfusion::Strides& window_dilation_strides_forward,
                                    const nnfusion::CoordinateDiff& padding_below_forward,
                                    const nnfusion::CoordinateDiff& padding_above_forward,
                                    const nnfusion::Strides& data_dilation_strides_forward);

            void validate_and_infer_types(std::shared_ptr<graph::GNode> gnode) override;

            /// \return The data batch shape.
            const nnfusion::Shape& get_data_batch_shape() const { return m_data_batch_shape; }
            /// \return The window movement strides from the forward prop.
            const nnfusion::Strides& get_window_movement_strides_forward() const
            {
                return m_window_movement_strides_forward;
            }
            /// \return The window dilation strides from the forward prop.
            const nnfusion::Strides& get_window_dilation_strides_forward() const
            {
                return m_window_dilation_strides_forward;
            }
            /// \return The padding-below sizes (possibly negative) from the forward prop.
            const nnfusion::CoordinateDiff& get_padding_below_forward() const
            {
                return m_padding_below_forward;
            }
            /// \return The padding-above sizes (possibly negative) from the forward prop.
            const nnfusion::CoordinateDiff& get_padding_above_forward() const
            {
                return m_padding_above_forward;
            }
            /// \return The input data dilation strides from the forward prop.
            const nnfusion::Strides& get_data_dilation_strides_forward() const
            {
                return m_data_dilation_strides_forward;
            }

            /// \return The window movement strides for the backward prop.
            const nnfusion::Strides& get_window_movement_strides_backward() const
            {
                return m_window_movement_strides_backward;
            }
            /// \return The window dilation strides for the backward prop.
            const nnfusion::Strides& get_window_dilation_strides_backward() const
            {
                return m_window_dilation_strides_backward;
            }
            /// \return The padding-below sizes (possibly negative) for the backward prop.
            const nnfusion::CoordinateDiff& get_padding_below_backward() const
            {
                return m_padding_below_backward;
            }
            /// \return The padding-above sizes (possibly negative) for the backward prop.
            const nnfusion::CoordinateDiff& get_padding_above_backward() const
            {
                return m_padding_above_backward;
            }
            /// \return The input data dilation strides for the backward prop.
            const nnfusion::Strides& get_data_dilation_strides_backward() const
            {
                return m_data_dilation_strides_backward;
            }

        protected:
            nnfusion::Shape m_data_batch_shape;
            nnfusion::Strides m_window_movement_strides_forward;
            nnfusion::Strides m_window_dilation_strides_forward;
            nnfusion::CoordinateDiff m_padding_below_forward;
            nnfusion::CoordinateDiff m_padding_above_forward;
            nnfusion::Strides m_data_dilation_strides_forward;

            nnfusion::Strides m_window_movement_strides_backward;
            nnfusion::Strides m_window_dilation_strides_backward;
            nnfusion::CoordinateDiff m_padding_below_backward;
            nnfusion::CoordinateDiff m_padding_above_backward;
            nnfusion::Strides m_data_dilation_strides_backward;
        };

        /// \brief Filters backprop for batched convolution operation.
        class ConvolutionBackpropFilters : public Op
        {
        public:
            /// \brief Constructs a batched-convolution filter-backprop operation.
            ///
            /// \param filters_shape The shape of the filters from forward-prop.
            /// \param window_movement_strides_forward The window movement strides from forward-prop.
            /// \param window_dilation_strides_forward The window dilation strides from forward-prop.
            /// \param padding_below_forward The padding-below sizes from forward-prop.
            /// \param padding_above_forward The padding-above sizes from forward-prop.
            /// \param data_dilation_strides_forward The data dilation strides from forward-prop.
            ConvolutionBackpropFilters(const nnfusion::Shape& filters_shape,
                                       const nnfusion::Strides& window_movement_strides_forward,
                                       const nnfusion::Strides& window_dilation_strides_forward,
                                       const nnfusion::CoordinateDiff& padding_below_forward,
                                       const nnfusion::CoordinateDiff& padding_above_forward,
                                       const nnfusion::Strides& data_dilation_strides_forward);

            void validate_and_infer_types(std::shared_ptr<graph::GNode> gnode) override;

            /// \return The filters tensor shape.
            const nnfusion::Shape& get_filters_shape() const { return m_filters_shape; }
            /// \return The window movement strides from the forward prop.
            const nnfusion::Strides& get_window_movement_strides_forward() const
            {
                return m_window_movement_strides_forward;
            }
            /// \return The window dilation strides from the forward prop.
            const nnfusion::Strides& get_window_dilation_strides_forward() const
            {
                return m_window_dilation_strides_forward;
            }
            /// \return The padding-below sizes (possibly negative) from the forward prop.
            const nnfusion::CoordinateDiff& get_padding_below_forward() const
            {
                return m_padding_below_forward;
            }
            /// \return The padding-above sizes (possibly negative) from the forward prop.
            const nnfusion::CoordinateDiff& get_padding_above_forward() const
            {
                return m_padding_above_forward;
            }
            /// \return The data dilation strides from the forward prop.
            const nnfusion::Strides& get_data_dilation_strides_forward() const
            {
                return m_data_dilation_strides_forward;
            }

            /// \return The window movement strides for the backward prop.
            const nnfusion::Strides& get_window_movement_strides_backward() const
            {
                return m_window_movement_strides_backward;
            }
            /// \return The window dilation strides for the backward prop.
            const nnfusion::Strides& get_window_dilation_strides_backward() const
            {
                return m_window_dilation_strides_backward;
            }
            /// \return The padding-below sizes (possibly negative) for the backward prop.
            const nnfusion::CoordinateDiff& get_padding_below_backward() const
            {
                return m_padding_below_backward;
            }
            /// \return The padding-above sizes (possibly negative) for the backward prop.
            const nnfusion::CoordinateDiff& get_padding_above_backward() const
            {
                return m_padding_above_backward;
            }
            /// \return The data dilation strides for the backward prop.
            const nnfusion::Strides& get_data_dilation_strides_backward() const
            {
                return m_data_dilation_strides_backward;
            }

        protected:
            nnfusion::Shape m_filters_shape;
            nnfusion::Strides m_window_movement_strides_forward;
            nnfusion::Strides m_window_dilation_strides_forward;
            nnfusion::CoordinateDiff m_padding_below_forward;
            nnfusion::CoordinateDiff m_padding_above_forward;
            nnfusion::Strides m_data_dilation_strides_forward;

            nnfusion::Strides m_window_movement_strides_backward;
            nnfusion::Strides m_window_dilation_strides_backward;
            nnfusion::CoordinateDiff m_padding_below_backward;
            nnfusion::CoordinateDiff m_padding_above_backward;
            nnfusion::Strides m_data_dilation_strides_backward;
        };
    }
}
