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

#include "../util/reduce_grad.hpp"
#include "core/node.hpp"
#include "nnfusion/core/graph/util/autobroadcast.hpp"

namespace nnfusion
{
    namespace frontend
    {
        namespace onnx_import
        {
            namespace set_1
            {
                NamedNodeVector TranslateDivGradOp(const onnx::NodeProto& node_proto,
                                                   const NodeMap& all_ng_nodes,
                                                   std::shared_ptr<nnfusion::graph::Graph> m_graph)
                {
                    // out = x / y, x_grad = out_grad / y, y_grad = - out_grad * x / y ** 2
                    auto input_indexes = GetAllInputIndex(all_ng_nodes, node_proto);
                    NNFUSION_CHECK(input_indexes.size() == 3);

                    auto out_grad = input_indexes[0];
                    auto x = input_indexes[1];
                    auto x_shape = x.get_shape();
                    std::tie(x, out_grad) =
                        graph::numpy_broadcast(std::make_pair(x, out_grad), m_graph);
                    auto y = input_indexes[2];
                    auto y_shape = y.get_shape();
                    std::tie(y, out_grad) =
                        graph::numpy_broadcast(std::make_pair(y, out_grad), m_graph);

                    // x_grad
                    auto x_grad_op = std::make_shared<op::Divide>();
                    x_grad_op->set_name(node_proto.output(0));
                    auto x_grad_gnode = m_graph->add_node_and_edge(x_grad_op, {out_grad, y});
                    auto x_grad = nnfusion::frontend::onnx_import::reduce::reduce_grad(
                        GNodeIndex{x_grad_gnode}, x_shape, m_graph);

                    // y_grad
                    auto numerator_gnode =
                        m_graph->add_node_and_edge(std::make_shared<op::Multiply>(), {out_grad, x});
                    auto denominator_gnode =
                        m_graph->add_node_and_edge(std::make_shared<op::Multiply>(), {y, y});
                    auto frac_gnode = m_graph->add_node_and_edge(
                        std::make_shared<op::Divide>(), {numerator_gnode, denominator_gnode});
                    auto y_grad_op = std::make_shared<op::Negative>();
                    y_grad_op->set_name(node_proto.output(1));
                    auto y_grad_gnode = m_graph->add_node_and_edge(y_grad_op, {frac_gnode});
                    auto y_grad = nnfusion::frontend::onnx_import::reduce::reduce_grad(
                        GNodeIndex{y_grad_gnode}, y_shape, m_graph);

                    return {{node_proto.output(0), x_grad}, {node_proto.output(1), y_grad}};
                }
            } // namespace set_1
        }     //namespace onnx_import
    }         // namespace frontend
} // namespace  nnfusion
