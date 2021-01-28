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

#include "nnfusion/common/axis_vector.hpp"

namespace nnfusion
{
    namespace op
    {
        /// \brief Tensor reshape operation.
        ///
        /// "Converts" an input tensor into a new shape with the same number of elements.
        ///
        /// Given that the input tensor has shape \f$[d_1,\dots,d_n]\f$, the output may have any shape \f$[d'_1,\dots,d'_m]\f$ such that
        /// \f$\Pi_{0 \leq i \lt n}(d_i) = \Pi_{0 \leq i \lt m}(d'_i)\f$. For example, a \f$3\times{}4\f$ matrix can be reshaped into a
        /// 3-tensor of shape \f$3\times{}2\times{}2\f$, a matrix of shape \f$6\times{}2\f$, or a vector of size \f$12\f$, but not, for
        /// example, a matrix of size \f$4\times{}4\f$.
        ///
        /// The parameter `input_order` indicates the order in which to "walk" over the input axes. Given a tensor of shape \f$(d_1,\dots,d_n)\f$,
        /// an input order of \f$(a_0, a_1, \dots, a_{n-1})\f$ results in the coordinate for axis \f$a_{n-1}\f$ being varied most frequently,
        /// followed by axis \f$a-2\f$, and so on down to \f$a_0\f$.
        ///
        /// (TODO: example.)
        ///
        /// ## Parameters
        ///
        /// |                | Description                                                |
        /// | -------------- | ---------------------------------------------------------- |
        /// | `input_order`  | The order in which to walk over the input axes.            |
        /// | `output_shape` | The shape \f$[d'_1,\dots,d'_m]\f$ for the reshaped tensor. |
        ///
        /// ## Output
        ///
        /// | Type                     | Description                                                                                            |
        /// | ------------------------ | ------------------------------------------------------------------------------------------------------ |
        /// | \f$E[d'_1,\dots,d'_m]\f$ | The tensor \f$T\f$, where \f$T\f$ is the input tensor with its elements rearranged as described above. |
        class Reshape : public Op
        {
        public:
            /// \brief Constructs a reshape operation.
            ///
            /// \param input_order The order in which to iterate over input axes. This must be a permutation of the
            ///                    sequence \f$(0,\dots,n-1)\f$ where \f$n\f$ is the rank of the input tensor.
            /// \param output_shape The output shape. If the input shape is \f$(a_0,\dots,a_{k-1})\f$ then the output shape must
            ///        be of the form \f$(b_0,\dots,b_{j-1})\f$ where \f$\Pi(a_i) = \Pi(b_i)\f$.
            Reshape(const nnfusion::AxisVector& input_order, const nnfusion::Shape& output_shape);

            void validate_and_infer_types(std::shared_ptr<graph::GNode> gnode) override;
            void infer_shared_memory(std::shared_ptr<graph::GNode> gnode) override;
            /// \return The order in which to iterate over input axes.
            const nnfusion::AxisVector& get_input_order() const { return m_input_order; }
            /// \return The shape of the output tensor.
            const nnfusion::Shape& get_output_shape() const { return m_output_shape; }
            bool get_is_transpose() const { return m_is_transpose; }
            bool get_is_layout_change() const { return m_is_layout_change; }
        protected:
            const nnfusion::AxisVector m_input_order;
            const nnfusion::Shape m_output_shape;
            bool m_is_transpose{false};
            bool m_is_layout_change{false};
        };
    }
}
