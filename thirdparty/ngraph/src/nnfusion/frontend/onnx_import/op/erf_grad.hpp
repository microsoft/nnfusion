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

//----------------------------------------------------------------------------------------------
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See License.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include "core/node.hpp"

namespace nnfusion
{
    namespace frontend
    {
        namespace onnx_import
        {
            namespace set_1
            {
                NamedNodeVector TranslateErfGradOp(const onnx::NodeProto& node_proto,
                                                   const NodeMap& all_ng_nodes,
                                                   std::shared_ptr<nnfusion::graph::Graph> m_graph)
                {
                    // y = erf(x), x_grad = y_grad * (2 / sqrt(pi)) * exp ** (-x**2)
                    auto input_indexes = GetAllInputIndex(all_ng_nodes, node_proto);
                    NNFUSION_CHECK(input_indexes.size() == 2);

                    auto x = input_indexes[0];
                    auto y_grad = input_indexes[1];

                    // x_grad
                    const float two_sqrt_pi = 1.12837916709551257390; /* 2/sqrt(pi) */

                    auto square_x =
                        m_graph->add_node_and_edge(std::make_shared<op::Multiply>(), {x, x});
                    auto neg_square_x =
                        m_graph->add_node_and_edge(std::make_shared<op::Negative>(), {square_x});
                    auto exp_neg_square_x =
                        m_graph->add_node_and_edge(std::make_shared<op::Exp>(), {neg_square_x});
                    auto two_sqrt_pi_op = std::make_shared<op::Constant>(
                        element::f32, x.get_shape(), std::vector<float>{two_sqrt_pi});
                    auto two_sqrt_pi_gnode = m_graph->add_node_and_edge(
                        two_sqrt_pi_op, nnfusion::graph::GNodeVector({}));
                    auto erf_grad = m_graph->add_node_and_edge(
                        std::make_shared<op::Multiply>(), {two_sqrt_pi_gnode, exp_neg_square_x});
                    auto x_grad_op = std::make_shared<op::Multiply>();
                    x_grad_op->set_name(node_proto.output(0));
                    auto x_grad =
                        m_graph->add_node_and_edge(x_grad_op, {y_grad, GNodeIndex{erf_grad}});

                    return {{node_proto.output(0), x_grad}};
                }
            } // namespace set_1
        }     //namespace onnx_import
    }         // namespace frontend
} // namespace  nnfusion
