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
        /// \brief Softmax operation.
        ///
        class Softmax : public ElementwiseArithmetic
        {
        public:
            /// \brief Constructs a softmax operation.
            ///
            /// \param axes The axis positions (0-based) on which to calculate the softmax.
            Softmax(const nnfusion::AxisSet& axes,
                    bool in_log_space =
                        false); // Current kernel doesn't follow the axes, but the last dim

            void validate_and_infer_types(std::shared_ptr<graph::GNode> gnode) override;

            const nnfusion::AxisSet& get_axes() const { return m_axes; }
            const bool is_in_log_space() const { return m_in_log_space; }
        private:
            nnfusion::AxisSet m_axes;
            bool m_in_log_space;
        };

        /// \brief Softmax operation.
        ///
        class SoftmaxGrad : public Op
        {
        public:
            /// \brief Constructs a softmax grad operation.
            ///
            /// \param axes The axis positions (0-based) on which to calculate the softmax.
            SoftmaxGrad(const nnfusion::AxisSet& axes, bool in_log_space = false);

            void validate_and_infer_types(std::shared_ptr<graph::GNode> gnode) override;

            const nnfusion::AxisSet& get_axes() const { return m_axes; }
            const bool is_in_log_space() const { return m_in_log_space; }
        private:
            nnfusion::AxisSet m_axes;
            bool m_in_log_space;
        };
    }
}
