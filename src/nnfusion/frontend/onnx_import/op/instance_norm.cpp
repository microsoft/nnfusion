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

#include "../util/util.hpp"
#include "core/node.hpp"
#include "nnfusion/core/graph/util/autobroadcast.hpp"
#include "nnfusion/core/operators/generic_op/generic_op.hpp"
#include "util/reshape.hpp"

DECLARE_bool(fantares_mode);
namespace nnfusion
{
    namespace frontend
    {
        namespace onnx_import
        {
            namespace set_1
            {
                using namespace graph;

                // InstanceNorm is a case of LayerNorm (i.e., axis = 2)
                GNodeVector LayernormInternal(std::shared_ptr<nnfusion::graph::Graph> m_graph,
                                              std::shared_ptr<GNode> input,
                                              std::shared_ptr<GNode> weight,
                                              std::shared_ptr<GNode> bias,
                                              int64_t axis,
                                              float eps);

                NamedNodeVector TranslateInstanceNormalizationOp(
                    const onnx::NodeProto& node_proto,
                    const NodeMap& all_ng_nodes,
                    std::shared_ptr<nnfusion::graph::Graph> m_graph)
                {
                    auto input_gnodes = GetAllInputNode(all_ng_nodes, node_proto);

                    auto input = input_gnodes[0];  // 2, 128, 1024
                    auto weight = input_gnodes[1]; // 1024
                    auto bias = input_gnodes[2];   // 1024

                    Node node(node_proto);
                    auto eps = node.get_attribute_value<float>("epsilon", 1e-5f);
                    eps = eps > 0 ? eps : 1e-5f;

                    int64_t axis = 2;

                    // extend axes for weight and bias
                    auto input_shape = input->get_shape();
                    std::vector<size_t> reshape_axes;
                    reshape_axes.push_back(weight->get_shape()[0]);
                    for (int i = axis; i < input_shape.size(); i++)
                    {
                        reshape_axes.push_back(1);
                    }
                    weight = m_graph->add_node_and_edge(
                        std::make_shared<op::Reshape>(reshape::get_default_axis_vector(1),
                                                      Shape(reshape_axes)),
                        {weight});
                    bias = m_graph->add_node_and_edge(
                        std::make_shared<op::Reshape>(reshape::get_default_axis_vector(1),
                                                      Shape(reshape_axes)),
                        {bias});

                    auto layernorm_out = LayernormInternal(m_graph, input, weight, bias, axis, eps);

                    NamedNodeVector ret;
                    ret.emplace_back(node_proto.output(0), layernorm_out.at(0));

                    return ret;
                }
            } // namespace set_1
        }     //namespace onnx_import
    }         // namespace frontend
} // namespace  nnfusion