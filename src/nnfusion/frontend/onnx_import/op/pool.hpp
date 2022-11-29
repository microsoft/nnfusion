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
#include "../util/util.hpp"
#include "nnfusion/core/operators/generic_op/generic_op.hpp"
#include "reduce.hpp"

namespace nnfusion
{
    namespace frontend
    {
        namespace onnx_import
        {
            namespace set_1
            {
                template <typename PrologueOp, typename ReduceOp, typename EpilogueOp>
                NamedNodeVector
                    TranslateGlobalPoolOp(const onnx::NodeProto& node_proto,
                                          const NodeMap& all_ng_nodes,
                                          std::shared_ptr<nnfusion::graph::Graph> m_graph)
                {
                    auto input_gnode = GetInputNode(all_ng_nodes, node_proto, 0);
                    Shape input_shape = input_gnode->get_shape();
                    Node node(node_proto);
                    bool reshaped = false;

                    NNFUSION_CHECK(input_shape.size() >= 3)
                        << "The input of GlobalPool should have at least 3 dimensions";

                    // GlobalPool is equal to Reduce, e.g., GlobalAveragePool == ReduceMean
                    nnfusion::AxisSet reduction_axes;
                    {
                        for (int i = 2; i < input_shape.size(); i++)
                        {
                            reduction_axes.insert(i);
                        }
                    }
                    int64 keepdims = 1;

                    auto pro_gnode =
                        AddPrologueOrEpilogueOp<PrologueOp>(m_graph, input_gnode, reduction_axes);
                    auto sum_op = std::make_shared<ReduceOp>(reduction_axes);
                    auto sum_gnode = m_graph->add_node_and_edge(sum_op, {pro_gnode});

                    // Add epilogue op
                    auto epi_gnode =
                        AddPrologueOrEpilogueOp<EpilogueOp>(m_graph, sum_gnode, reduction_axes);

                    NamedNodeVector ret;

                    nnfusion::Shape result_shape_with_keep(input_shape.size());

                    for (size_t i = 0; i < input_shape.size(); i++)
                    {
                        result_shape_with_keep[i] =
                            reduction_axes.count(i) == 0 ? input_shape[i] : 1;
                    }
                    nnfusion::AxisVector axis_order(epi_gnode->get_shape().size());
                    std::iota(axis_order.begin(), axis_order.end(), 0);
                    auto reshape_op =
                        std::make_shared<op::Reshape>(axis_order, result_shape_with_keep);
                    reshape_op->set_name(node_proto.output(0));
                    auto reshape_gnode = m_graph->add_node_and_edge(reshape_op, {epi_gnode});
                    ret.push_back({node_proto.output(0), reshape_gnode});

                    return ret;
                }

                NamedNodeVector TranslateMaxPoolOp(const onnx::NodeProto& node_proto,
                                                   const NodeMap& all_ng_nodes,
                                                   std::shared_ptr<nnfusion::graph::Graph> m_graph)
                {
                    auto input_gnode = GetInputNode(all_ng_nodes, node_proto, 0);
                    Shape input_shape = input_gnode->get_shape();
                    // NNFUSION_CHECK(input_shape.size() == 4) << "only support MaxPool2D";
                    Node node(node_proto);

                    // Parse ONNX op attributes
                    Shape kernel_shape;
                    kernel_shape = get_kernel_shape(node, input_gnode);

                    auto strides = get_strides(node, input_gnode);
                    auto dilations = get_dilations(node, input_gnode);
                    for (auto dilation : dilations)
                    {
                        NNFUSION_CHECK(dilation == 1) << "only support dilation == 1";
                    }
                    auto paddings = get_pads(node, input_gnode);
                    // auto_pad is processed in get_pads()
                    // auto auto_pad = node.get_attribute_value<std::string>("auto_pad", "NOTSET");
                    // // NNFUSION_CHECK(auto_pad == "NOTSET") << "The deprecated auto_pad not support yet";
                    // if (auto_pad != "NOTSET")
                    // {
                    //     auto paddings_tmp = get_auto_pads(kernel_shape, auto_pad);
                    // }
                    bool storage_order = node.get_attribute_value<int64_t>("storage_order", 0);
                    NNFUSION_CHECK(storage_order == 0) << "storage_order not support yet";
                    bool ceil_mode = node.get_attribute_value<int64_t>("ceil_mode", 0);
                    NNFUSION_CHECK(ceil_mode == 0) << "ceil_mode not support yet";

                    // Convert padding from CoordinateDiff to Shape objects
                    const CoordinateDiff& padding_above{paddings.first};
                    const CoordinateDiff& padding_below{paddings.second};
                    Shape padding_below_shape{std::begin(padding_below), std::end(padding_below)};
                    Shape padding_above_shape{std::begin(padding_above), std::end(padding_above)};

                    std::shared_ptr<op::Op> pool_op = std::make_shared<op::MaxPool>(
                        kernel_shape, strides, padding_below_shape, padding_above_shape);

                    pool_op->set_name(node_proto.output(0));
                    auto pool_gnode = m_graph->add_node_and_edge(pool_op, {input_gnode});
                    NamedNodeVector ret{{node_proto.output(0), pool_gnode}};
                    return ret;
                }

                NamedNodeVector
                    TranslateAveragePoolOp(const onnx::NodeProto& node_proto,
                                           const NodeMap& all_ng_nodes,
                                           std::shared_ptr<nnfusion::graph::Graph> m_graph)
                {
                    auto input_gnode = GetInputNode(all_ng_nodes, node_proto, 0);
                    Shape input_shape = input_gnode->get_shape();
                    Node node(node_proto);

                    // Parse ONNX op attributes
                    Shape kernel_shape;
                    kernel_shape = get_kernel_shape(node, input_gnode);

                    auto strides = get_strides(node, input_gnode);
                    auto paddings = get_pads(node, input_gnode);
                    // auto_pad is processed in get_pads()
                    // auto auto_pad = node.get_attribute_value<std::string>("auto_pad", "NOTSET");
                    // // NNFUSION_CHECK(auto_pad == "NOTSET") << "The deprecated auto_pad not support yet";
                    // if (auto_pad != "NOTSET")
                    // {
                    //     auto paddings_tmp = get_auto_pads(kernel_shape, auto_pad);
                    // }
                    // auto storage_order = node.get_attribute_value<int>("storage_order", 0);
                    // NNFUSION_CHECK(storage_order == 0) << "storage_order not support yet";
                    bool ceil_mode = node.get_attribute_value<int64_t>("ceil_mode", 0);
                    NNFUSION_CHECK(ceil_mode == 0) << "ceil_mode not support yet";
                    bool count_include_pad =
                        node.get_attribute_value<int64_t>("count_include_pad", 0);

                    // TODO(lingm): check asymmetric padding, onnx test case passed
                    // if (!count_include_pad)
                    // {
                    //     NNFUSION_CHECK(paddings.first == paddings.second)
                    //         << "not support asymmetric padding when not "
                    //            "count_include_pad";
                    // }

                    // Convert padding from CoordinateDiff to Shape objects
                    const CoordinateDiff& padding_above{paddings.first};
                    const CoordinateDiff& padding_below{paddings.second};
                    Shape padding_below_shape{std::begin(padding_below), std::end(padding_below)};
                    Shape padding_above_shape{std::begin(padding_above), std::end(padding_above)};

                    std::shared_ptr<op::Op> pool_op =
                        std::make_shared<op::AvgPool>(kernel_shape,
                                                      strides,
                                                      padding_below_shape,
                                                      padding_above_shape,
                                                      count_include_pad);

                    pool_op->set_name(node_proto.output(0));
                    auto pool_gnode = m_graph->add_node_and_edge(pool_op, {input_gnode});
                    NamedNodeVector ret{{node_proto.output(0), pool_gnode}};
                    return ret;
                }

            } // namespace set_1

            namespace set_7
            {
                using set_1::TranslateAveragePoolOp;
            }

            namespace set_8
            {
                using set_1::TranslateMaxPoolOp;
            }

            namespace set_10
            {
                using set_1::TranslateAveragePoolOp;
                using set_1::TranslateMaxPoolOp;
            } // namespace set_10

            namespace set_11
            {
                using set_1::TranslateAveragePoolOp;
                using set_1::TranslateMaxPoolOp;
            } // namespace set_11

            namespace set_12
            {
                using set_1::TranslateMaxPoolOp;
            }
        } // namespace onnx_import
    }     // namespace frontend
} // namespace nnfusion
