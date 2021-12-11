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

#include "nnfusion/core/operators/util/arithmetic_reduction.hpp"
#include "nnfusion/core/operators/util/binary_elementwise_logical.hpp"

namespace nnfusion
{
    namespace op
    {
        class ReduceAny : public ArithmeticReduction
        {
        public:
            /// \brief Constructs a logical-or operation.
            ///
            ReduceAny(const nnfusion::AxisSet& reduction_axes);
        };

        /// \brief Elementwise logical-or operation.
        ///
        class Or : public BinaryElementwiseLogical
        {
        public:
            /// \brief Constructs a logical-or operation.
            Or();

        protected:
            virtual bool is_commutative() override { return true; }
        };
    }
}
