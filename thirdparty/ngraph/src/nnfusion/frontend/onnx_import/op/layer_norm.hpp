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
                    TranslateLayerNormalizationOp(const onnx::NodeProto& node_proto,
                                                  const NodeMap& all_ng_nodes,
                                                  std::shared_ptr<nnfusion::graph::Graph> m_graph)
                {
                    auto input_indexes = GetAllInputIndex(all_ng_nodes, node_proto);
                    auto input_rank = input_indexes[0].get_shape().size();

                    Node node(node_proto);
                    auto axis = node.get_attribute_value<int64_t>("axis", -1);
                    axis += axis < 0 ? input_rank : 0;
                    auto eps = node.get_attribute_value<float>("epsilon", 1e-5f);
                    eps = eps > 0 ? eps : 1e-5f;

                    nnfusion::op::OpConfig::any myConfig;
                    myConfig["axis"] = axis;
                    myConfig["epsilon"] = eps;

                    auto generic_op = std::make_shared<nnfusion::op::GenericOp>(
                        node_proto.output(0), "LayerNorm", myConfig);
                    auto generic_gnode = m_graph->add_node_and_edge(generic_op, input_indexes, 3);

                    return {{node_proto.output(0), generic_gnode, 0},
                            {node_proto.output(1), generic_gnode, 1},
                            {node_proto.output(2), generic_gnode, 2}};
                }

                NamedNodeVector
                    TranslateLayerNormalizationOpV1(const onnx::NodeProto& node_proto,
                                                    const NodeMap& all_ng_nodes,
                                                    std::shared_ptr<nnfusion::graph::Graph> m_graph)
                {
                    auto input_gnodes = GetAllInputNode(all_ng_nodes, node_proto);

                    auto input = input_gnodes[0]; // 2, 128, 1024
                    auto input_shape = input->get_shape();
                    auto input_rank = input_shape.size();
                    auto weight = input_gnodes[1]; // 1024
                    auto bias = input_gnodes[2];   // 1024

                    Node node(node_proto);
                    auto axis = node.get_attribute_value<int64_t>("axis", -1);
                    axis += axis < 0 ? input_rank : 0;
                    Shape normalized_shape(std::next(input_shape.begin(), axis), input_shape.end());
                    auto normalized_rank = normalized_shape.size();

                    auto eps = node.get_attribute_value<float>("epsilon", 1e-5f);
                    eps = eps > 0 ? eps : 1e-5f;

                    auto num_feature = nnfusion::shape_size(normalized_shape);

                    // mean
                    std::vector<size_t> reduction_axes(normalized_rank);
                    std::iota(
                        reduction_axes.begin(), reduction_axes.end(), input_rank - normalized_rank);
                    auto sum_gnode = m_graph->add_node_and_edge(
                        std::make_shared<op::Sum>(reduction_axes), {input}); // 2, 128

                    const auto& et = sum_gnode->get_element_type();
                    auto divisor_op = std::make_shared<op::Constant>(
                        et,
                        sum_gnode->get_shape(),
                        std::vector<std::string>{std::to_string(num_feature)});
                    auto divisor_gnode = m_graph->add_node_and_edge(divisor_op, GNodeVector({}));

                    auto mean_gnode = m_graph->add_node_and_edge(
                        std::make_shared<op::Divide>(), {sum_gnode, divisor_gnode}); // 2, 128
                    // keep dim
                    nnfusion::Shape mean_shape_with_keep(input_rank);
                    for (size_t i = 0; i < input_rank; i++)
                    {
                        mean_shape_with_keep[i] =
                            i < input_rank - normalized_rank ? input_shape[i] : 1;
                    }
                    nnfusion::AxisVector ng_axis_order(mean_gnode->get_shape().size());
                    std::iota(ng_axis_order.begin(), ng_axis_order.end(), 0);
                    mean_gnode = m_graph->add_node_and_edge(
                        std::make_shared<op::Reshape>(ng_axis_order, mean_shape_with_keep),
                        {mean_gnode}); // 2, 128, 1
                    auto out1 = mean_gnode;
                    std::tie(input, mean_gnode) =
                        numpy_broadcast(std::make_pair(input, mean_gnode), m_graph); // 2, 128, 1024

                    mean_gnode = m_graph->add_node_and_edge(std::make_shared<op::Subtract>(),
                                                            {input, mean_gnode});

                    // std
                    auto std_power_gnode = m_graph->add_node_and_edge(
                        std::make_shared<op::Multiply>(), {mean_gnode, mean_gnode});
                    auto std_sum_gnode = m_graph->add_node_and_edge(
                        std::make_shared<op::Sum>(reduction_axes), {std_power_gnode});
                    auto std_mean_gnode = m_graph->add_node_and_edge(
                        std::make_shared<op::Divide>(), {std_sum_gnode, divisor_gnode});
                    auto std_sqrt_gnode =
                        m_graph->add_node_and_edge(std::make_shared<op::Sqrt>(), {std_mean_gnode});
                    auto eps_op = std::make_shared<op::Constant>(
                        et,
                        std_sqrt_gnode->get_shape(),
                        std::vector<std::string>{std::to_string(eps)});
                    auto eps_gnode = m_graph->add_node_and_edge(eps_op, GNodeVector({}));
                    auto std_gnode = m_graph->add_node_and_edge(
                        std::make_shared<op::Add>(), {std_sqrt_gnode, eps_gnode}); // 2, 128
                    // keep dim
                    std_gnode = m_graph->add_node_and_edge(
                        std::make_shared<op::Reshape>(ng_axis_order, mean_shape_with_keep),
                        {std_gnode}); // 2, 128, 1
                    auto one_op = std::make_shared<op::Constant>(
                        et, std_gnode->get_shape(), std::vector<std::string>{"1.0"});
                    auto one_gnode = m_graph->add_node_and_edge(one_op, GNodeVector({}));
                    auto inv_std_gnode = m_graph->add_node_and_edge(std::make_shared<op::Divide>(),
                                                                    {one_gnode, std_gnode});
                    auto out2 = inv_std_gnode;
                    std::tie(input, inv_std_gnode) = numpy_broadcast(
                        std::make_pair(input, inv_std_gnode), m_graph); // 2, 128, 1024

                    auto norm_gnode = m_graph->add_node_and_edge(std::make_shared<op::Multiply>(),
                                                                 {mean_gnode, inv_std_gnode});

                    // weight
                    std::tie(input, weight) =
                        numpy_broadcast(std::make_pair(input, weight), m_graph);
                    // bias
                    std::tie(input, bias) = numpy_broadcast(std::make_pair(input, bias), m_graph);

                    auto mul_gnode = m_graph->add_node_and_edge(std::make_shared<op::Multiply>(),
                                                                {weight, norm_gnode});
                    auto ret_gnode =
                        m_graph->add_node_and_edge(std::make_shared<op::Add>(), {mul_gnode, bias});

                    return {{node_proto.output(0), ret_gnode},
                            {node_proto.output(1), out1},
                            {node_proto.output(2), out2}};
                }

                NamedNodeVector TranslateLayerNormalizationGradOp(
                    const onnx::NodeProto& node_proto,
                    const NodeMap& all_ng_nodes,
                    std::shared_ptr<nnfusion::graph::Graph> m_graph)
                {
                    auto input_indexes = GetAllInputIndex(all_ng_nodes, node_proto);

                    auto y_grad_index = input_indexes[0];      // 2, 128, 1024
                    auto x_index = input_indexes[1];           // 2, 128, 1024
                    auto weight_index = input_indexes[2];      // 1024
                    auto mean_index = input_indexes[3];        // 2, 128, 1
                    auto inv_std_var_index = input_indexes[4]; // 2, 128, 1

                    auto input_shape = x_index.get_shape();
                    auto input_rank = input_shape.size();

                    Node node(node_proto);
                    auto axis = node.get_attribute_value<int64_t>("axis", -1);
                    axis += axis < 0 ? input_rank : 0;
                    Shape normalized_shape(std::next(input_shape.begin(), axis), input_shape.end());
                    auto normalized_rank = normalized_shape.size();
                    auto batch_rank = input_rank - normalized_rank;

                    auto eps = node.get_attribute_value<float>("epsilon", 1e-5f);
                    eps = eps > 0 ? eps : 1e-5f;

                    auto num_feature = nnfusion::shape_size(normalized_shape);

                    std::vector<size_t> reduction_axes(batch_rank);
                    std::iota(reduction_axes.begin(), reduction_axes.end(), 0);
                    auto bias_grad_op = std::make_shared<op::Sum>(reduction_axes);

                    std::vector<size_t> norm_reduction_axes(normalized_rank);
                    std::iota(norm_reduction_axes.begin(),
                              norm_reduction_axes.end(),
                              input_rank - normalized_rank);
                    // bias_grad
                    bias_grad_op->set_name(node_proto.output(2));
                    auto bias_grad_gnode =
                        m_graph->add_node_and_edge(bias_grad_op, {y_grad_index}); // 1024

                    // xmu = x - mean
                    std::tie(x_index, mean_index) =
                        numpy_broadcast(std::make_pair(x_index, mean_index), m_graph);
                    auto xmu_gnode = m_graph->add_node_and_edge(
                        std::make_shared<op::Subtract>(), {x_index, mean_index}); // 2, 128, 1024
                    auto xmu_index = GNodeIndex{xmu_gnode};

                    // weight_grad
                    std::tie(xmu_index, inv_std_var_index) =
                        numpy_broadcast(std::make_pair(xmu_index, inv_std_var_index), m_graph);
                    auto weight_grad_gnode_temp1 = m_graph->add_node_and_edge(
                        std::make_shared<op::Multiply>(), {xmu_index, inv_std_var_index});
                    auto weight_grad_gnode_temp2 = m_graph->add_node_and_edge(
                        std::make_shared<op::Multiply>(),
                        {GNodeIndex{weight_grad_gnode_temp1}, y_grad_index}); // 2, 128, 1024
                    auto weight_grad_op = std::make_shared<op::Sum>(reduction_axes);
                    weight_grad_op->set_name(node_proto.output(1));
                    auto weight_grad_gnode = m_graph->add_node_and_edge(
                        weight_grad_op, {weight_grad_gnode_temp2}); // 1024

                    // x_grad
                    // first part
                    std::tie(y_grad_index, weight_index) =
                        numpy_broadcast(std::make_pair(y_grad_index, weight_index), m_graph);
                    auto xhat_grad_gnode =
                        m_graph->add_node_and_edge(std::make_shared<op::Multiply>(),
                                                   {y_grad_index, weight_index}); // 2, 128, 1024
                    auto xhat_x_grad_index = inv_std_var_index;
                    auto first_part_gnode = m_graph->add_node_and_edge(
                        std::make_shared<op::Multiply>(),
                        {GNodeIndex{xhat_grad_gnode}, xhat_x_grad_index});

                    // second part
                    auto three_op = std::make_shared<op::Constant>(nnfusion::element::f32,
                                                                   inv_std_var_index.get_shape(),
                                                                   std::vector<float>{3.0});
                    auto three_gnode = m_graph->add_node_and_edge(three_op, GNodeVector({}));
                    auto inv_var_pow_gnode =
                        m_graph->add_node_and_edge(std::make_shared<op::Power>(),
                                                   {inv_std_var_index, GNodeIndex{three_gnode}});
                    auto second_part_gnode_temp = m_graph->add_node_and_edge(
                        std::make_shared<op::Multiply>(), {GNodeIndex{xhat_grad_gnode}, xmu_index});
                    second_part_gnode_temp =
                        m_graph->add_node_and_edge(std::make_shared<op::Multiply>(),
                                                   {second_part_gnode_temp, inv_var_pow_gnode});
                    second_part_gnode_temp =
                        m_graph->add_node_and_edge(std::make_shared<op::Sum>(norm_reduction_axes),
                                                   {second_part_gnode_temp}); // 2, 128
                    auto shape_with_keep = second_part_gnode_temp->get_shape();
                    shape_with_keep.push_back(1);
                    nnfusion::AxisVector ng_axis_order(second_part_gnode_temp->get_shape().size());
                    std::iota(ng_axis_order.begin(), ng_axis_order.end(), 0);
                    second_part_gnode_temp = m_graph->add_node_and_edge(
                        std::make_shared<op::Reshape>(ng_axis_order, shape_with_keep),
                        {second_part_gnode_temp}); // 2, 128, 1
                    std::tie(xmu_gnode, second_part_gnode_temp) =
                        numpy_broadcast(std::make_pair(xmu_gnode, second_part_gnode_temp), m_graph);
                    second_part_gnode_temp = m_graph->add_node_and_edge(
                        std::make_shared<op::Multiply>(),
                        {second_part_gnode_temp, xmu_gnode}); // 2, 128, 1024
                    auto divisor_op = std::make_shared<op::Constant>(
                        nnfusion::element::f32, input_shape, std::vector<size_t>{num_feature});
                    auto divisor_gnode = m_graph->add_node_and_edge(divisor_op, GNodeVector({}));
                    auto second_part_gnode = m_graph->add_node_and_edge(
                        std::make_shared<op::Divide>(), {second_part_gnode_temp, divisor_gnode});

                    // third part
                    auto third_part_gnode_temp =
                        m_graph->add_node_and_edge(std::make_shared<op::Sum>(norm_reduction_axes),
                                                   {first_part_gnode}); // 2, 128
                    third_part_gnode_temp = m_graph->add_node_and_edge(
                        std::make_shared<op::Reshape>(ng_axis_order, shape_with_keep),
                        {third_part_gnode_temp}); // 2, 128, 1
                    std::tie(third_part_gnode_temp, divisor_gnode) = numpy_broadcast(
                        std::make_pair(third_part_gnode_temp, divisor_gnode), m_graph);
                    auto third_part_gnode = m_graph->add_node_and_edge(
                        std::make_shared<op::Divide>(),
                        {third_part_gnode_temp, divisor_gnode}); // 2, 128, 1024

                    auto x_grad_temp = m_graph->add_node_and_edge(
                        std::make_shared<op::Subtract>(), {first_part_gnode, second_part_gnode});
                    auto x_grad_op = std::make_shared<op::Subtract>();
                    x_grad_op->set_name(node_proto.output(0));
                    auto x_grad =
                        m_graph->add_node_and_edge(x_grad_op, {x_grad_temp, third_part_gnode});

                    return {{node_proto.output(0), x_grad},
                            {node_proto.output(1), weight_grad_gnode},
                            {node_proto.output(2), bias_grad_gnode}};
                }
            } // namespace set_1
        }     //namespace onnx_import
    }         // namespace frontend
} // namespace  nnfusion
