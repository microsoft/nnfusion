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
        /// \brief Elementwise Identity operation.
        ///
        class Identity : public ElementwiseArithmetic
        {
        public:
            /// \brief Constructs an Identity operation.
            ///
            /// \param arg Node that produces the input tensor.
            Identity();
        };

        /// \brief Elementwise IdentityBackprop operation.
        ///
        class IdentityBackprop : public ElementwiseArithmetic
        {
        public:
            /// \brief Constructs a IdentityBackprop operation.
            ///
            /// \param arg Node that produces the identity forward input tensor.
            IdentityBackprop();
        };
    }
}
