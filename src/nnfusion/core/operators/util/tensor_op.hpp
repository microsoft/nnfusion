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
        /// \brief An abstract base class for tensor operations, such as Constant, Placeholder or Variable.
        ///
        class TensorOp : public Op
        {
        public:
            /// \brief Constructions a tensor view-typed op.
            ///
            /// \param node_type The node type of the tensor op.
            /// \param element_type The element type of the tensor.
            /// \param shape The shape of the tensor.
            TensorOp(const std::string& node_type,
                     const nnfusion::element::Type& element_type,
                     const nnfusion::Shape& shape);

            void validate_and_infer_types(std::shared_ptr<graph::GNode> gnode) override;

            bool is_tensor_op() const override { return true; }
        protected:
            nnfusion::Shape m_shape{};
            nnfusion::element::Type m_element_type;
        };
    }
}