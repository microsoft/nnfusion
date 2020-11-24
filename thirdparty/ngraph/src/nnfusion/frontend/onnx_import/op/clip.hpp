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

#include <string>

#include "core/node.hpp"

namespace nnfusion
{
    namespace frontend
    {
        namespace onnx_import
        {
            namespace set_1
            {
                NamedNodeVector TranslateClipOp(const onnx::NodeProto& node_proto,
                                                const NodeMap& all_ng_nodes,
                                                std::shared_ptr<nnfusion::graph::Graph> m_graph)
                {
                    auto input_gnode = GetInputNode(all_ng_nodes, node_proto, 0);

                    Node node{node_proto};
                    double max_value =
                        node.get_attribute_value<double>("max", std::numeric_limits<double>::max());
                    double min_value = node.get_attribute_value<double>(
                        "min", std::numeric_limits<double>::lowest());

                    auto max_value_op = std::make_shared<op::Constant>(
                        input_gnode->get_element_type(),
                        nnfusion::Shape{},
                        std::vector<std::string>{std::to_string(max_value)});
                    auto max_value_gnode =
                        m_graph->add_node_and_edge(max_value_op, graph::GNodeVector({}));
                    max_value_gnode =
                        make_broadcast_node(max_value_gnode, input_gnode->get_shape(), m_graph);

                    auto min_value_op = std::make_shared<op::Constant>(
                        input_gnode->get_element_type(),
                        nnfusion::Shape{},
                        std::vector<std::string>{std::to_string(min_value)});
                    auto min_value_gnode =
                        m_graph->add_node_and_edge(min_value_op, graph::GNodeVector({}));
                    min_value_gnode =
                        make_broadcast_node(min_value_gnode, input_gnode->get_shape(), m_graph);

                    auto max_op = std::make_shared<op::Maximum>();
                    auto max_gnode =
                        m_graph->add_node_and_edge(max_op, {input_gnode, min_value_gnode});
                    auto min_op = std::make_shared<op::Minimum>();
                    min_op->set_name(node_proto.output(0));
                    auto min_gnode =
                        m_graph->add_node_and_edge(min_op, {max_value_gnode, max_gnode});
                    NamedNodeVector ret{{node_proto.output(0), min_gnode}};
                    return ret;
                }
            } // namespace set_1
        }     //namespace onnx_import
    }         // namespace frontend
} // namespace nnfusion
