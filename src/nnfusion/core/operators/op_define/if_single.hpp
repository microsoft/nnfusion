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
        /// \brief If control-flow operation, with same definition as https://github.com/onnx/onnx/blob/master/docs/Changelog.md#If-1.
        class IfSingle: public Op
        {
        public:
            IfSingle(const std::shared_ptr<op::Op>& inner_op, bool is_then_branch): Op("IfSingle"), m_inner_op(inner_op), m_is_then_branch(is_then_branch), m_inner_fake_gnode(nullptr) {};
            void validate_and_infer_types(std::shared_ptr<graph::GNode> gnode) override;
            void infer_shared_memory(std::shared_ptr<graph::GNode> gnode) override;
            std::shared_ptr<graph::GNode> get_inner_fake_node(std::shared_ptr<graph::GNode> gnode) {
                // sync input/output of gnode to m_inner_fake_gnode
                m_inner_fake_gnode->clear_in_edges();
                for (size_t i = 0; i < m_inner_fake_gnode->get_input_size(); i++) {
                    m_inner_fake_gnode->set_input(i, gnode->get_inputs()[i+1]);
                    auto outer_in_edge = gnode->get_in_edge(i+1);
                    auto inner_in_edge = std::make_shared<graph::Edge>(outer_in_edge->get_src(), m_inner_fake_gnode, i, outer_in_edge->get_src_output(), i);
                    m_inner_fake_gnode->add_in_edge(inner_in_edge);
                }
                for (size_t i = 0; i < m_inner_fake_gnode->get_output_size(); i++) {
                    m_inner_fake_gnode->set_output(i, gnode->get_outputs()[i]);   
                }
                return m_inner_fake_gnode;
            }
            bool get_is_then_branch() const { return m_is_then_branch; }

        protected:
            std::shared_ptr<op::Op> m_inner_op;
            bool m_is_then_branch;
            std::shared_ptr<graph::GNode> m_inner_fake_gnode;
        };
    } // namespace op
} // namespace nnfusion
