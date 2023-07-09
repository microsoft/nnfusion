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

#include <vector>

#include "gather.hpp"
#include "nnfusion/core/operators/generic_op/generic_op.hpp"
#include "nnfusion/frontend/util/evaluator.hpp"

namespace nnfusion
{
    namespace frontend
    {
        namespace onnx_import
        {
            namespace set_1
            {
                NamedNodeVector TranslateGatherOp(const onnx::NodeProto& node_proto,
                                                  const NodeMap& all_ng_nodes,
                                                  std::shared_ptr<nnfusion::graph::Graph> m_graph)
                {
                    auto input_indexes = GetAllInputIndex(all_ng_nodes, node_proto);

                    Node node(node_proto);
                    auto axis = node.get_attribute_value<std::vector<int64_t>>("axis", {0})[0];
                    axis += axis < 0 ? input_indexes[0].get_shape().size() : 0;

                    nnfusion::op::OpConfig::any myConfig;
                    myConfig["axis"] = axis;

                    auto generic_op = std::make_shared<nnfusion::op::GenericOp>(
                        node_proto.output(0), "GatherV2", myConfig);
                    auto generic_gnode = m_graph->add_node_and_edge(generic_op, input_indexes);

                    return {{node_proto.output(0), generic_gnode}};
                }

                NamedNodeVector TranslateGatherNDOp(const onnx::NodeProto& node_proto,
                                                    const NodeMap& all_ng_nodes,
                                                    std::shared_ptr<nnfusion::graph::Graph> m_graph)
                {
                    auto input_indexes = GetAllInputIndex(all_ng_nodes, node_proto);

                    Node node(node_proto);
                    // According to ONNX doc, this attribute name should be "batch_dims"
                    // But in their implementation, this attribute name is "axis"
                    auto axis = node.get_attribute_value<int64_t>("axis", 0);
                    axis += axis < 0 ? input_indexes[0].get_shape().size() : 0;

                    nnfusion::op::OpConfig::any myConfig;
                    myConfig["axis"] = axis;

                    auto generic_op = std::make_shared<nnfusion::op::GenericOp>(
                        node_proto.output(0), "GatherND", myConfig);
                    auto generic_gnode = m_graph->add_node_and_edge(generic_op, input_indexes);

                    return {{node_proto.output(0), generic_gnode}};
                }

                NamedNodeVector
                    TranslateGatherGradOp(const onnx::NodeProto& node_proto,
                                          const NodeMap& all_ng_nodes,
                                          std::shared_ptr<nnfusion::graph::Graph> m_graph)
                {
                    auto input_indexes = GetAllInputIndex(all_ng_nodes, node_proto);

                    NNFUSION_CHECK(input_indexes.size() == 3);
                    auto indices_index = input_indexes[1];
                    auto y_grad = input_indexes[2];

                    std::vector<int> x_shape;
                    NNFUSION_CHECK(GetValueFromNGraphOp(input_indexes[0].gnode, &x_shape));

                    nnfusion::op::OpConfig::any zerosConfig;
                    zerosConfig["shape"] = x_shape;
                    auto zeros_op = std::make_shared<nnfusion::op::GenericOp>(
                        node_proto.output(0) + "_zeros", "Zeros", zerosConfig);
                    auto zeros_gnode = m_graph->add_node_and_edge(zeros_op, GNodeVector{});
                    auto zeros_index = GNodeIndex{zeros_gnode};

                    Node node(node_proto);
                    auto axis = node.get_attribute_value<std::vector<int64_t>>("axis", {0})[0];
                    axis += axis < 0 ? x_shape.size() : 0;

                    nnfusion::op::OpConfig::any myConfig;
                    myConfig["axis"] = axis;
                    myConfig["x_shape"] = x_shape;

                    auto generic_op = std::make_shared<nnfusion::op::GenericOp>(
                        node_proto.output(0), "GatherGrad", myConfig);
                    auto generic_gnode = m_graph->add_node_and_edge(
                        generic_op, {indices_index, y_grad, zeros_index});

                    return {{node_proto.output(0), generic_gnode}};
                }

                NamedNodeVector
                    TranslateGatherNDGradOp(const onnx::NodeProto& node_proto,
                                            const NodeMap& all_ng_nodes,
                                            std::shared_ptr<nnfusion::graph::Graph> m_graph)
                {
                    auto input_indexes = GetAllInputIndex(all_ng_nodes, node_proto);

                    NNFUSION_CHECK(input_indexes.size() == 3);
                    auto indices_index = input_indexes[1];
                    auto y_grad = input_indexes[2];

                    std::vector<int> x_shape;
                    NNFUSION_CHECK(GetValueFromNGraphOp(input_indexes[0].gnode, &x_shape));

                    nnfusion::op::OpConfig::any zerosConfig;
                    zerosConfig["shape"] = x_shape;
                    auto zeros_op = std::make_shared<nnfusion::op::GenericOp>(
                        node_proto.output(0) + "_zeros", "Zeros", zerosConfig);
                    auto zeros_gnode = m_graph->add_node_and_edge(zeros_op, GNodeVector{});
                    auto zeros_index = GNodeIndex{zeros_gnode};

                    Node node(node_proto);
                    auto axis = node.get_attribute_value<std::vector<int64_t>>("axis", {0})[0];
                    axis += axis < 0 ? x_shape.size() : 0;

                    nnfusion::op::OpConfig::any myConfig;
                    myConfig["axis"] = axis;
                    myConfig["x_shape"] = x_shape;

                    auto generic_op = std::make_shared<nnfusion::op::GenericOp>(
                        node_proto.output(0), "GatherNDGrad", myConfig);
                    auto generic_gnode = m_graph->add_node_and_edge(
                        generic_op, {indices_index, y_grad, zeros_index});

                    return {{node_proto.output(0), generic_gnode}};
                }

            } // namespace set_1

        } //namespace onnx_import

    } // namespace frontend

} // namespace nnfusion
