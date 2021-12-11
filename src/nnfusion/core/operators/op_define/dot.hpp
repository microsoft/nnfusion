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

#include "nnfusion/core/operators/op.hpp"

namespace nnfusion
{
    namespace op
    {
        /// \brief Generalized dot product operation, including scalar-tensor product, matrix-vector product, and matrix multiplication.
        class Dot : public Op
        {
        public:
            /// \brief Constructs a dot product operation.
            ///
            /// \param reduction_axes_count The number of axes to dot.
            Dot(size_t reduction_axes_count,
                bool has_reduction_axes_count = true,
                bool trans_a = false,
                bool trans_b = false);

            /// \brief Constructs a dot product operation with default dot-axis selection depending on the inputs.
            ///
            /// If `arg0` or `arg1` is a scalar, there are no dot-axes. Else, there is one dot-axis.
            ///
            /// (Note that in particular, this results in scalar-tensor products where one or the other argument is
            /// a scalar, a matrix-vector products where `arg0` is a matrix and `arg1` is a vector, and a
            /// matrix multiplication where `arg0` and `arg1` are both matrices.)
            ///
            Dot();

            void validate_and_infer_types(std::shared_ptr<graph::GNode> gnode) override;

            size_t get_reduction_axes_count() const { return m_reduction_axes_count; }
            void set_transpose(bool trans_a, bool trans_b)
            {
                m_transpose_A = trans_a;
                m_transpose_B = trans_b;
            }

            bool& get_transpose_A() { return m_transpose_A; }
            bool& get_transpose_B() { return m_transpose_B; }
        protected:
            size_t m_reduction_axes_count;
            bool m_has_reduction_axes_count;
            bool m_transpose_A = false;
            bool m_transpose_B = false;
        };
    }
}
