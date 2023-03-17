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

#include "../util/util.hpp"
#include "core/node.hpp"

namespace nnfusion
{
    namespace frontend
    {
        namespace onnx_import
        {
            namespace set_15
            {
                NamedNodeVector TranslateCastLikeOp(const onnx::NodeProto& node_proto,
                                                    const NodeMap& all_ng_nodes,
                                                    std::shared_ptr<nnfusion::graph::Graph> m_graph)
                {
                    auto input_gnode = GetInputNode(all_ng_nodes, node_proto, 0);
                    auto type_gnode = GetInputNode(all_ng_nodes, node_proto, 1);
                    Node node(node_proto);
                    element::Type et_type = type_gnode->get_element_type();

                    auto op = std::make_shared<op::Convert>(et_type);
                    op->set_name(node_proto.output(0));
                    auto gnode = m_graph->add_node_and_edge(op, {input_gnode});
                    NamedNodeVector ret{{node_proto.output(0), gnode}};
                    return ret;
                }
            } // namespace set_15
        }     //namespace onnx_import
    }         // namespace frontend
} // namespace nnfusion
