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
                template <typename T>
                NamedNodeVector
                    TranslateLegacyBinaryOp(const onnx::NodeProto& node_proto,
                                            const NodeMap& all_ng_nodes,
                                            std::shared_ptr<nnfusion::graph::Graph> m_graph)
                {
                    // TODO: Fix GetInputNode -> GetInputIndex
                    auto lhs_gnode = GetInputNode(all_ng_nodes, node_proto, 0);
                    NNFUSION_CHECK(lhs_gnode != nullptr);
                    auto rhs_gnode = GetInputNode(all_ng_nodes, node_proto, 1);
                    NNFUSION_CHECK(rhs_gnode != nullptr);
                    Node node(node_proto);
                    auto axis = node.get_attribute_value<int64_t>("axis", 0);

                    std::tie(lhs_gnode, rhs_gnode) = legacy_style_broadcast_for_binary_operation(
                        std::make_pair(lhs_gnode, rhs_gnode), axis, m_graph);

                    auto op = std::make_shared<T>();
                    NNFUSION_CHECK(node_proto.output_size() == 1)
                        << "Binary op should only has one output.";
                    op->set_name(node_proto.output(0));
                    auto gnode = m_graph->add_node_and_edge(op, {lhs_gnode, rhs_gnode});
                    NamedNodeVector ret{{node_proto.output(0), gnode}};
                    return ret;
                }

                template <typename T>
                NamedNodeVector TranslateBinaryOp(const onnx::NodeProto& node_proto,
                                                  const NodeMap& all_ng_nodes,
                                                  std::shared_ptr<nnfusion::graph::Graph> m_graph)
                {
                    auto lhs_index = GetInputIndex(all_ng_nodes, node_proto, 0);
                    auto rhs_index = GetInputIndex(all_ng_nodes, node_proto, 1);

                    std::tie(lhs_index, rhs_index) =
                        graph::numpy_broadcast(std::make_pair(lhs_index, rhs_index), m_graph);

                    auto op = std::make_shared<T>();
                    NNFUSION_CHECK(node_proto.output_size() == 1)
                        << "Binary op should only has one output.";
                    op->set_name(node_proto.output(0));
                    auto gnode = m_graph->add_node_and_edge(op, {lhs_index, rhs_index});
                    NamedNodeVector ret{{node_proto.output(0), gnode}};
                    return ret;
                }

                NamedNodeVector TranslatePowerOp(const onnx::NodeProto& node_proto,
                                                 const NodeMap& all_ng_nodes,
                                                 std::shared_ptr<nnfusion::graph::Graph> m_graph)
                {
                    auto lhs_index = GetInputIndex(all_ng_nodes, node_proto, 0);
                    auto rhs_index = GetInputIndex(all_ng_nodes, node_proto, 1);
                    if (rhs_index.gnode->get_op_type() == "Constant")
                    {
                        auto rhs =
                            dynamic_pointer_cast<op::Constant>(rhs_index.gnode->get_op_ptr());
                        float power;
                        if (rhs->get_type() == element::f16)
                            power = rhs->get_float16_vector()[0];
                        else
                            power = rhs->get_vector<float>()[0];
                        // special opt for power of 2 case
                        if (power == 2)
                        {
                            auto op = std::make_shared<op::Multiply>();
                            op->set_name(node_proto.output(0));
                            auto gnode = m_graph->add_node_and_edge(op, {lhs_index, lhs_index});
                            NamedNodeVector ret{{node_proto.output(0), gnode}};
                            return ret;
                        }
                    }
                    return TranslateBinaryOp<op::Power>(node_proto, all_ng_nodes, m_graph);
                }

            } // namespace set_1
            namespace set_6
            {
                using set_1::TranslateBinaryOp;
                using set_1::TranslateLegacyBinaryOp;
            } // namespace set_6

            namespace set_7
            {
                template <typename T>
                NamedNodeVector TranslateBinaryOp(const onnx::NodeProto& node_proto,
                                                  const NodeMap& all_ng_nodes,
                                                  std::shared_ptr<nnfusion::graph::Graph> m_graph)
                {
                    auto lhs_index = GetInputIndex(all_ng_nodes, node_proto, 0);
                    auto rhs_index = GetInputIndex(all_ng_nodes, node_proto, 1);

                    std::tie(lhs_index, rhs_index) =
                        graph::numpy_broadcast(std::make_pair(lhs_index, rhs_index), m_graph);

                    auto op = std::make_shared<T>();
                    NNFUSION_CHECK(node_proto.output_size() == 1)
                        << "Binary op should only has one output.";
                    op->set_name(node_proto.output(0));
                    auto gnode = m_graph->add_node_and_edge(op, {lhs_index, rhs_index});
                    NamedNodeVector ret{{node_proto.output(0), gnode}};
                    return ret;
                }

            } // namespace set_7

            namespace set_9
            {
                using set_7::TranslateBinaryOp;
            }

            namespace set_11
            {
                using set_7::TranslateBinaryOp;
            }

            namespace set_12
            {
                using set_7::TranslateBinaryOp;
            }

            namespace set_13
            {
                using set_7::TranslateBinaryOp;
            }
            namespace set_14
            {
                using set_7::TranslateBinaryOp;
            }

            namespace set_15
            {
                using set_7::TranslateBinaryOp;
            }

            namespace set_16
            {
                using set_7::TranslateBinaryOp;
            }
        } // namespace onnx_import
    }     // namespace frontend
} // namespace nnfusion
