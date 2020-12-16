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
#include "nnfusion/core/operators/generic_op/generic_op.hpp"

namespace nnfusion
{
    namespace frontend
    {
        namespace onnx_import
        {
            namespace set_1
            {
                NamedNodeVector
                    TranslateAdamOptimizerOp(const onnx::NodeProto& node_proto,
                                             const NodeMap& all_ng_nodes,
                                             std::shared_ptr<nnfusion::graph::Graph> m_graph)
                {
                    GNodeIndexVector input_indexes;
                    // Graph doesn't support dynamic input count, so we remove unused or optional input
                    for (int i = 0; i < 6; i++)
                    {
                        input_indexes.push_back(GetInputIndex(all_ng_nodes, node_proto, i));
                    }
                    Node node(node_proto);
                    nnfusion::op::OpConfig::any myConfig;
                    myConfig["lambda"] = node.get_attribute_value<float>("lambda", 0.0);
                    myConfig["epsilon"] = node.get_attribute_value<float>("epsilon", 1e-6);
                    myConfig["alpha"] = node.get_attribute_value<float>("alpha", 0.9);
                    myConfig["beta"] = node.get_attribute_value<float>("beta", 0.999);

                    auto generic_op = std::make_shared<nnfusion::op::GenericOp>(
                        node_proto.name(), "AdamOptimizer", myConfig);
                    auto generic_gnode =
                        m_graph->add_node_and_edge(generic_op, input_indexes, /* output_size */ 5);

                    return {{node_proto.output(0), generic_gnode, 0},
                            {node_proto.output(1), generic_gnode, 1},
                            {node_proto.output(2), generic_gnode, 2},
                            {node_proto.output(3), generic_gnode, 3},
                            {node_proto.output(4), generic_gnode, 4}};
                }
            } // namespace set_1
        }     //namespace onnx_import
    }         // namespace frontend
} // namespace  nnfusion
