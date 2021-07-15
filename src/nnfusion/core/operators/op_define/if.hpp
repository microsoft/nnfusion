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

// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include "../op.hpp"
#include "nnfusion/core/graph/graph.hpp"

namespace nnfusion
{
    namespace op
    {
        /// \brief If control-flow operation, with same definition as https://github.com/onnx/onnx/blob/master/docs/Changelog.md#If-1.
        class If : public Op
        {
        public:
            /// \brief Constructs an if operation
            ///
            /// \param then_branch_graph The then_branch graph.<br>
            /// `[f]`
            /// \param else_branch_graph The else_branch graph.<br>
            /// `[f]`
            If(std::shared_ptr<nnfusion::graph::Graph>& then_branch_graph,
               std::shared_ptr<nnfusion::graph::Graph>& else_branch_graph);

            void validate_and_infer_types(std::shared_ptr<graph::GNode> gnode) override;

        protected:
            std::shared_ptr<nnfusion::graph::Graph> m_then_branch_graph;
            std::shared_ptr<nnfusion::graph::Graph> m_else_branch_graph;
        };
    }
}
