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
        class IndexReduction : public Op
        {
        public:
            size_t get_reduction_axis() const { return m_axis; }
            nnfusion::element::Type get_index_element_type() const { return m_index_element_type; }
            IndexReduction(const std::string& node_type,
                           size_t axis,
                           const nnfusion::element::Type& index_element_type);

        protected:
            size_t m_axis;
            nnfusion::element::Type m_index_element_type;

            void validate_and_infer_types(std::shared_ptr<graph::GNode> gnode) override;
        };
    }
}
