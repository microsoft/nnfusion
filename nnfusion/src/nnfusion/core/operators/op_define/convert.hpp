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

namespace nnfusion
{
    namespace op
    {
        /// \brief Elementwise type conversion operation.
        class Convert : public Op
        {
        public:
            /// \brief Constructs a conversion operation.
            ///
            /// \param element_type Element type for the output tensor.
            Convert(const nnfusion::element::Type& element_type);

            void validate_and_infer_types(std::shared_ptr<graph::GNode> gnode) override;

            const nnfusion::element::Type& get_convert_element_type() const
            {
                return m_element_type;
            }

        protected:
            const nnfusion::element::Type m_element_type;
        };
    }
}
