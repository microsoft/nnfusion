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
                NamedNodeVector
                    TranslateLeakyReluOp(const onnx::NodeProto& node_proto,
                                         const NodeMap& all_ng_nodes,
                                         std::shared_ptr<nnfusion::graph::Graph> m_graph)
                {
                    auto input_indexes = GetAllInputIndex(all_ng_nodes, node_proto);
                    NNFUSION_CHECK(input_indexes.size() == 1);

                    auto x = input_indexes.at(0);
                    Node node(node_proto);
                    float alpha = node.get_attribute_value<float>("alpha", 0.01);
                    NNFUSION_CHECK(alpha >= 0 && alpha <= 1);
                    auto alpha_op = std::make_shared<op::Constant>(
                        x.get_element_type(),
                        Shape{},
                        std::vector<std::string>{std::to_string(alpha)});
                    auto alpha_gnode =
                        m_graph->add_node_and_edge(alpha_op, nnfusion::graph::GNodeVector{});
                    auto alpha_index = GNodeIndex(alpha_gnode);
                    std::tie(x, alpha_index) =
                        graph::numpy_broadcast(std::make_pair(x, alpha_index), m_graph);
                    auto alpha_x = m_graph->add_node_and_edge(std::make_shared<op::Multiply>(),
                                                              {x, alpha_index});
                    auto result = m_graph->add_node_and_edge(std::make_shared<op::Maximum>(),
                                                             {x, GNodeIndex(alpha_x)});
                    return {{node_proto.output(0), GNodeIndex(result)}};
                }

            } // namespace set_1

        } //namespace onnx_import

    } // namespace frontend

} // namespace nnfusion
