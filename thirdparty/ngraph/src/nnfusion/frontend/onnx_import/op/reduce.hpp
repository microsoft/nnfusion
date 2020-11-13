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
                    TranslateReduceSumOp(const onnx::NodeProto& node_proto,
                                         const NodeMap& all_ng_nodes,
                                         std::shared_ptr<nnfusion::graph::Graph> m_graph)
                {
                    auto input_index = GetInputIndex(all_ng_nodes, node_proto, 0);
                    auto input_shape = input_index.get_shape();

                    bool keep_dims;
                    Node node(node_proto);
                    nnfusion::AxisSet ng_reduction_axes;
                    {
                        auto axes = node.get_attribute_value<std::vector<int64_t>>("axes", {});
                        if (axes.empty())
                        {
                            auto axes_uint = get_default_order(input_shape);
                            std::copy(axes_uint.begin(),
                                      axes_uint.end(),
                                      std::inserter(ng_reduction_axes, ng_reduction_axes.end()));
                        }
                        else
                        {
                            for (auto axis : axes)
                            {
                                ng_reduction_axes.insert(axis += axis < 0 ? input_shape.size() : 0);
                            }
                        }
                    }

                    auto keepdims = node.get_attribute_value<int64>("keepdims", 1);

                    auto sum_op = std::make_shared<op::Sum>(ng_reduction_axes);
                    NamedNodeVector ret;
                    if (keepdims)
                    {
                        auto sum_gnode = m_graph->add_node_and_edge(sum_op, {input_index});
                        nnfusion::Shape ng_result_shape_with_keep(input_shape.size());

                        for (size_t i = 0; i < input_shape.size(); i++)
                        {
                            ng_result_shape_with_keep[i] =
                                ng_reduction_axes.count(i) == 0 ? input_shape[i] : 1;
                        }
                        nnfusion::AxisVector ng_axis_order(sum_gnode->get_shape().size());
                        std::iota(ng_axis_order.begin(), ng_axis_order.end(), 0);
                        auto reshape_op =
                            std::make_shared<op::Reshape>(ng_axis_order, ng_result_shape_with_keep);
                        reshape_op->set_name(node_proto.output(0));
                        auto reshape_gnode = m_graph->add_node_and_edge(reshape_op, {sum_gnode});
                        ret.push_back({node_proto.output(0), reshape_gnode});
                    }
                    else
                    {
                        sum_op->set_name(node_proto.output(0));
                        auto sum_gnode = m_graph->add_node_and_edge(sum_op, {input_index});
                        ret.push_back({node_proto.output(0), sum_gnode});
                    }

                    return ret;
                }

                NamedNodeVector
                    TranslateReduceMeanOp(const onnx::NodeProto& node_proto,
                                          const NodeMap& all_ng_nodes,
                                          std::shared_ptr<nnfusion::graph::Graph> m_graph)
                {
                    auto input_index = GetInputIndex(all_ng_nodes, node_proto, 0);
                    auto input_shape = input_index.get_shape();

                    bool keep_dims;
                    Node node(node_proto);
                    nnfusion::AxisSet ng_reduction_axes;
                    {
                        auto axes = node.get_attribute_value<std::vector<int64_t>>("axes", {});
                        if (axes.empty())
                        {
                            auto axes_uint = get_default_order(input_shape);
                            std::copy(axes_uint.begin(),
                                      axes_uint.end(),
                                      std::inserter(ng_reduction_axes, ng_reduction_axes.end()));
                        }
                        else
                        {
                            for (auto axis : axes)
                            {
                                ng_reduction_axes.insert(axis += axis < 0 ? input_shape.size() : 0);
                            }
                        }
                    }

                    auto keepdims = node.get_attribute_value<int64>("keepdims", 1);

                    auto sum_op = std::make_shared<op::Sum>(ng_reduction_axes);
                    auto sum_gnode = m_graph->add_node_and_edge(sum_op, {input_index});

                    // reduce mean
                    size_t reduced_ele_count = 1;
                    for (auto i : ng_reduction_axes)
                    {
                        reduced_ele_count *= input_shape.at(i);
                    }
                    auto divisor_op = std::make_shared<op::Constant>(
                        sum_gnode->get_element_type(),
                        Shape{},
                        std::vector<std::string>{std::to_string(reduced_ele_count)});
                    auto divisor_gnode =
                        m_graph->add_node_and_edge(divisor_op, nnfusion::graph::GNodeVector{});
                    std::tie(sum_gnode, divisor_gnode) =
                        graph::numpy_broadcast(std::make_pair(sum_gnode, divisor_gnode), m_graph);
                    auto mean_gnode = m_graph->add_node_and_edge(std::make_shared<op::Divide>(),
                                                                 {sum_gnode, divisor_gnode});

                    NamedNodeVector ret;
                    if (keepdims)
                    {
                        nnfusion::Shape ng_result_shape_with_keep(input_shape.size());

                        for (size_t i = 0; i < input_shape.size(); i++)
                        {
                            ng_result_shape_with_keep[i] =
                                ng_reduction_axes.count(i) == 0 ? input_shape[i] : 1;
                        }
                        nnfusion::AxisVector ng_axis_order(mean_gnode->get_shape().size());
                        std::iota(ng_axis_order.begin(), ng_axis_order.end(), 0);
                        auto reshape_op =
                            std::make_shared<op::Reshape>(ng_axis_order, ng_result_shape_with_keep);
                        reshape_op->set_name(node_proto.output(0));
                        auto reshape_gnode = m_graph->add_node_and_edge(reshape_op, {mean_gnode});
                        ret.push_back({node_proto.output(0), reshape_gnode});
                    }
                    else
                    {
                        mean_gnode->get_op_ptr()->set_name(node_proto.output(0));
                        ret.push_back({node_proto.output(0), mean_gnode});
                    }

                    return ret;
                }

            } // namespace set_1

        } //namespace op

    } // namespace onnx_import

} // namespace nnfusion
