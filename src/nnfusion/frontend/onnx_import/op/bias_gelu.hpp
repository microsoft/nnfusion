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

#include <memory>

#include "../core/node.hpp"
#include "../util/broadcasting.hpp"
#include "../util/util.hpp"
#include "nnfusion/core/graph/util/autobroadcast.hpp"

namespace nnfusion
{
    namespace frontend
    {
        namespace onnx_import
        {
            namespace set_1
            {
                NamedNodeVector TranslateBiasGeluOp(const onnx::NodeProto& node_proto,
                                                    const NodeMap& all_ng_nodes,
                                                    std::shared_ptr<nnfusion::graph::Graph> m_graph)
                {
                    auto lhs_index = GetInputIndex(all_ng_nodes, node_proto, 0);
                    auto rhs_index = GetInputIndex(all_ng_nodes, node_proto, 1);

                    std::tie(lhs_index, rhs_index) =
                        graph::numpy_broadcast(std::make_pair(lhs_index, rhs_index), m_graph);

                    auto add_op = std::make_shared<op::Add>();
                    auto gnode = m_graph->add_node_and_edge(add_op, {lhs_index, rhs_index});

                    auto gelu_op = std::make_shared<op::Gelu>();
                    gnode = m_graph->add_node_and_edge(gelu_op, {gnode});
                    NamedNodeVector ret{{node_proto.output(0), gnode}};
                    return ret;
                }

            } // namespace set_1
        }     // namespace onnx_import
    }         // namespace frontend
} // namespace ngraph
