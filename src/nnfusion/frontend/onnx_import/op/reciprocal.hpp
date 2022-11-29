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
                NamedNodeVector
                    TranslateReciprocalOp(const onnx::NodeProto& node_proto,
                                          const NodeMap& all_ng_nodes,
                                          std::shared_ptr<nnfusion::graph::Graph> m_graph)
                {
                    auto input_gnode = GetInputNode(all_ng_nodes, node_proto, 0);
                    auto input_shape = input_gnode->get_shape();

                    auto ones_gnode = m_graph->add_node_and_edge(
                        std::make_shared<op::Constant>(input_gnode->get_element_type(),
                                                       Shape{},
                                                       std::vector<std::string>{"1"}),
                        GNodeIndexVector{});
                    nnfusion::AxisSet broadcast_axes;
                    for (auto i = 0; i < input_shape.size(); i++)
                    {
                        broadcast_axes.insert(i);
                    }
                    ones_gnode = m_graph->add_node_and_edge(
                        std::make_shared<op::Broadcast>(input_shape, broadcast_axes), {ones_gnode});
                    auto ret_gnode = m_graph->add_node_and_edge(std::make_shared<op::Divide>(),
                                                                {ones_gnode, input_gnode});

                    return {{node_proto.output(0), ret_gnode}};
                }
            } // namespace set_1

            namespace set_6
            {
                using set_1::TranslateReciprocalOp;
            }

            namespace set_13
            {
                using set_1::TranslateReciprocalOp;
            }
        } // namespace onnx_import

    } // namespace frontend
} // namespace nnfusion
