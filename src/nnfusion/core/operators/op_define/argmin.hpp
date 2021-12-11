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

#include "nnfusion/core/operators/util/index_reduction.hpp"

namespace nnfusion
{
    namespace op
    {
        // \brief Computes minimum index along a specified axis for a given tensor
        class ArgMin : public IndexReduction
        {
        public:
            /// \brief Constructs a ArgMin operation.
            ///
            /// \param axis The axis along which to compute an index for minimum
            /// \param index_element_type produce indices. Currently, only int64 or int32 are supported
            ArgMin(size_t axis, const nnfusion::element::Type& index_element_type)
                : IndexReduction("ArgMin", axis, index_element_type)
            {
            }
        };
    }
}
