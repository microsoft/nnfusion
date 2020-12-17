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

#include "nnfusion/core/operators/util/binary_elementwise_logical.hpp"

namespace nnfusion
{
    namespace op
    {
        /// \brief Elementwise logical-and operation.
        ///
        class And : public BinaryElementwiseLogical
        {
        public:
            /// \brief Constructs a logical-and operation.
            ///
            /// \param arg0 Node that produces the first input tensor.<br>
            /// `[d0, ...]`
            /// \param arg1 Node that produces the second input tensor.<br>
            /// `[d0, ...]`
            ///
            /// Output `[d0, ...]`
            ///
            And();

        protected:
            virtual bool is_commutative() override { return true; }
        };
    }
}
