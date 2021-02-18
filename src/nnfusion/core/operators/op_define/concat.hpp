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
        /// \brief Concatenation operation.
        class Concat : public Op
        {
        public:
            /// \brief Constructs a concatenation operation.
            ///
            /// \param concatenation_axis The axis along which to concatenate the input tensors.
            Concat(size_t concatenation_axis);

            void validate_and_infer_types(std::shared_ptr<graph::GNode> gnode) override;
            void infer_shared_memory(std::shared_ptr<graph::GNode> gnode) override;

            /// \return The concatenation axis.
            size_t get_concatenation_axis() const { return m_concatenation_axis; }
        protected:
            const size_t m_concatenation_axis;
        };
    }
}
