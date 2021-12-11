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
#include "nnfusion/frontend/util/evaluator.hpp"

namespace nnfusion
{
    namespace frontend
    {
        namespace onnx_import
        {
            namespace set_1
            {
                NamedNodeVector TranslateDropoutOp(const onnx::NodeProto& node_proto,
                                                   const NodeMap& all_ng_nodes,
                                                   std::shared_ptr<nnfusion::graph::Graph> m_graph)
                {
                    auto input_indexes = GetAllInputIndex(all_ng_nodes, node_proto);
                    auto data_index = input_indexes[0];

                    Node node(node_proto);
                    float ratio = node.get_attribute_value<float>("ratio", 0.5);
                    NNFUSION_CHECK(ratio == 0) << "please set dropout ratio to 0";
                    return {{node_proto.output(0), data_index}};
                }

                NamedNodeVector
                    TranslateTrainableDropoutOp(const onnx::NodeProto& node_proto,
                                                const NodeMap& all_ng_nodes,
                                                std::shared_ptr<nnfusion::graph::Graph> m_graph)
                {
                    auto input_indexes = GetAllInputIndex(all_ng_nodes, node_proto);
                    auto data_index = input_indexes[0];
                    float ratio;
                    if (input_indexes.size() > 1)
                    {
                        auto ratio_gnode = GetInputNode(all_ng_nodes, node_proto, 1);

                        // TODO: random seed
                        // Node node(node_proto);
                        // std::int64_t seed{node.get_attribute_value<std::int64_t>("seed", 42)};

                        std::vector<float> ratio_vec;
                        NNFUSION_CHECK(GetValueFromNGraphOp(ratio_gnode, &ratio_vec));
                        ratio = ratio_vec[0];
                    }
                    else
                    {
                        ratio = 0.5; // default dropout ratio in ONNX training
                    }

                    nnfusion::op::OpConfig::any myConfig;
                    myConfig["ratio"] = ratio;

                    auto generic_op = std::make_shared<nnfusion::op::GenericOp>(
                        node_proto.name(), "DropoutTraining", myConfig);
                    auto generic_gnode =
                        m_graph->add_node_and_edge(generic_op, {data_index}, /* output_size */ 2);

                    return {{node_proto.output(0), generic_gnode, 0},
                            {node_proto.output(1), generic_gnode, 1}};
                }

                NamedNodeVector
                    TranslateTrainableDropoutGradOp(const onnx::NodeProto& node_proto,
                                                    const NodeMap& all_ng_nodes,
                                                    std::shared_ptr<nnfusion::graph::Graph> m_graph)
                {
                    auto input_indexes = GetAllInputIndex(all_ng_nodes, node_proto);
                    auto dy_index = input_indexes[0];
                    auto make_index = input_indexes[1];
                    float ratio;
                    if (input_indexes.size() > 2)
                    {
                        auto ratio_gnode = input_indexes[2].gnode;

                        // TODO: random seed
                        // Node node(node_proto);
                        // std::int64_t seed{node.get_attribute_value<std::int64_t>("seed", 42)};

                        std::vector<float> ratio_vec;
                        NNFUSION_CHECK(GetValueFromNGraphOp(ratio_gnode, &ratio_vec));
                        ratio = ratio_vec[0];
                    }
                    else
                    {
                        ratio = 0.5; // default dropout ratio in ONNX training
                    }

                    nnfusion::op::OpConfig::any myConfig;
                    myConfig["ratio"] = ratio;

                    auto generic_op = std::make_shared<nnfusion::op::GenericOp>(
                        node_proto.name(), "DropoutTrainingGrad", myConfig);
                    auto generic_gnode =
                        m_graph->add_node_and_edge(generic_op, {dy_index, make_index});

                    return {{node_proto.output(0), generic_gnode}};
                }
            } // namespace set_1
        }     //namespace onnx_import
    }         // namespace frontend
} // namespace  nnfusion
