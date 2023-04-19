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
#include "nnfusion/engine/interpreter.hpp"

namespace nnfusion
{
    namespace op
    {
        /// \brief Loop control-flow operation, with same definition as https://github.com/onnx/onnx/blob/master/docs/Changelog.md#Loop-11.
        class Loop : public Op
        {
        public:
            /// \brief Constructs an if operation
            ///
            /// \param loop_body_graph The loop body graph.<br>
            /// `[f]`
            Loop(std::shared_ptr<nnfusion::graph::Graph>& loop_body_graph,
                 const std::vector<nnfusion::PartialShape>& output_shapes,
                 const std::vector<nnfusion::element::Type>& output_types,
                 std::string op_type="Loop");

            void validate_and_infer_types(std::shared_ptr<graph::GNode> gnode) override;
            std::shared_ptr<nnfusion::graph::Graph> get_loop_body_graph();
            TranslationUnit::Pointer get_loop_body_tu();
            std::unordered_map<std::string, int> get_loop_output_map() { return m_loop_output_map; }
            void set_loop_output_map(std::unordered_map<std::string, int> map)
            {
                m_loop_output_map = std::move(map);
            }
            void set_loop_body_tu(TranslationUnit::Pointer);

        protected:
            std::shared_ptr<nnfusion::graph::Graph> m_loop_body_graph;
            std::vector<nnfusion::PartialShape> m_output_shapes;
            std::vector<nnfusion::element::Type> m_output_types;
            std::unordered_map<std::string, int> m_loop_output_map;
            TranslationUnit::Pointer m_loop_body_tu;
        };
    } // namespace op
} // namespace nnfusion