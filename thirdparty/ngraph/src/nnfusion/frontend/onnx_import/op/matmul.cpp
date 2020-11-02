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

#include "matmul.hpp"
#include "nnfusion/core/operators/generic_op/generic_op.hpp"

namespace nnfusion
{
    namespace frontend
    {
        namespace onnx_import
        {
            namespace set_1
            {
                NamedNodeVector TranslateMatmulOp(const onnx::NodeProto& node_proto,
                                                  const NodeMap& all_ng_nodes,
                                                  std::shared_ptr<nnfusion::graph::Graph> m_graph)
                {
                    auto lhs_desc = GetInputIndex(all_ng_nodes, node_proto, 0);
                    auto rhs_desc = GetInputIndex(all_ng_nodes, node_proto, 1);

                    NamedNodeVector ret;
                    auto lhs_shape = lhs_desc.get_shape();
                    auto rhs_shape = rhs_desc.get_shape();
                    int lhs_rank = lhs_shape.size();
                    int rhs_rank = rhs_shape.size();
                    NNFUSION_CHECK(lhs_rank > 0 && rhs_rank > 0);

                    Node node{node_proto};
                    auto adj_x = node.get_attribute_value<int64_t>("transA", 0);
                    auto adj_y = node.get_attribute_value<int64_t>("transB", 0);
                    NNFUSION_CHECK(!adj_x && !adj_y) << "transpose matmul not supported";
                    // numpy matmul: https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.matmul.html

                    // optimzed for a general pattern in BERT, otherwise rt_const_folding staticly folds broadcast operation.
                    if (lhs_rank >= 2 && rhs_rank == 2)
                    {
                        auto dot_op = std::make_shared<nnfusion::op::Dot>(0, false, adj_x, adj_y);
                        //ng_node->set_transpose(transpose_a, transpose_b);

                        dot_op->set_name(node_proto.output(0));
                        auto dot_gnode = m_graph->add_node_and_edge(dot_op, {lhs_desc, rhs_desc});
                        ret.emplace_back(node_proto.output(0), dot_gnode);
                        return ret;
                    }

                    int batchmm_output_rank = std::max(std::max(lhs_rank, rhs_rank), 2);
                    Shape left_full_shape(batchmm_output_rank, 0);
                    Shape right_full_shape(batchmm_output_rank, 0);
                    AxisSet left_broadcast_axes, right_broadcast_axes;
                    Shape batch_shape;
                    if (lhs_rank == 1)
                    {
                        left_full_shape[batchmm_output_rank - 1] = lhs_shape[0];
                        left_full_shape[batchmm_output_rank - 2] = 1;
                        left_broadcast_axes.insert(batchmm_output_rank - 2);
                    }
                    else
                    {
                        std::copy(lhs_shape.rbegin(), lhs_shape.rend(), left_full_shape.rbegin());
                    }
                    if (rhs_rank == 1)
                    {
                        right_full_shape[batchmm_output_rank - 1] = 1;
                        right_full_shape[batchmm_output_rank - 2] = rhs_shape[0];
                        right_broadcast_axes.insert(batchmm_output_rank - 1);
                    }
                    else
                    {
                        std::copy(rhs_shape.rbegin(), rhs_shape.rend(), right_full_shape.rbegin());
                    }

                    for (size_t i = 0; i < batchmm_output_rank - 2; i++)
                    {
                        if (left_full_shape[i] == 0)
                        {
                            left_full_shape[i] = 1;
                            left_broadcast_axes.insert(i);
                        }
                        if (right_full_shape[i] == 0)
                        {
                            right_full_shape[i] = 1;
                            right_broadcast_axes.insert(i);
                        }

                        if (left_full_shape[i] == right_full_shape[i])
                        {
                            batch_shape.push_back(left_full_shape[i]);
                            continue;
                        }
                        else if (left_full_shape[i] == 1)
                        {
                            batch_shape.push_back(right_full_shape[i]);
                            left_broadcast_axes.insert(i);
                        }
                        else if (right_full_shape[i] == 1)
                        {
                            batch_shape.push_back(left_full_shape[i]);
                            right_broadcast_axes.insert(i);
                        }
                        else
                        {
                            NNFUSION_CHECK_FAIL() << "broadcast dim mismatch, left "
                                                  << left_full_shape[i] << ", right "
                                                  << right_full_shape[i];
                        }
                    }

                    Shape left_shape_before_broadcast, right_shape_before_broadcast;
                    for (int i = 0; i < left_full_shape.size(); i++)
                    {
                        if (!left_broadcast_axes.count(i))
                        {
                            left_shape_before_broadcast.push_back(left_full_shape[i]);
                        }
                    }
                    for (int i = 0; i < right_full_shape.size(); i++)
                    {
                        if (!right_broadcast_axes.count(i))
                        {
                            right_shape_before_broadcast.push_back(right_full_shape[i]);
                        }
                    }
                    NNFUSION_CHECK(shape_size(left_shape_before_broadcast) ==
                                   shape_size(lhs_shape));
                    NNFUSION_CHECK(shape_size(right_shape_before_broadcast) ==
                                   shape_size(rhs_shape));
                    auto lhs_reshape_gnode = m_graph->add_node_and_edge(
                        std::make_shared<op::Reshape>(get_default_order(lhs_shape),
                                                      left_shape_before_broadcast),
                        {lhs_desc});
                    auto rhs_reshape_gnode = m_graph->add_node_and_edge(
                        std::make_shared<op::Reshape>(get_default_order(rhs_shape),
                                                      right_shape_before_broadcast),
                        {rhs_desc});

                    Shape left_broadcast_shape(batch_shape.begin(), batch_shape.end());
                    left_broadcast_shape.push_back(left_full_shape[batchmm_output_rank - 2]);
                    left_broadcast_shape.push_back(left_full_shape[batchmm_output_rank - 1]);
                    auto lhs_broadcast_gnode = m_graph->add_node_and_edge(
                        std::make_shared<op::Broadcast>(left_broadcast_shape, left_broadcast_axes),
                        {lhs_reshape_gnode});

                    Shape right_broadcast_shape(batch_shape.begin(), batch_shape.end());
                    right_broadcast_shape.push_back(right_full_shape[batchmm_output_rank - 2]);
                    right_broadcast_shape.push_back(right_full_shape[batchmm_output_rank - 1]);
                    auto rhs_broadcast_gnode =
                        m_graph->add_node_and_edge(std::make_shared<op::Broadcast>(
                                                       right_broadcast_shape, right_broadcast_axes),
                                                   {rhs_reshape_gnode});

                    nnfusion::op::OpConfig::any myConfig;
                    myConfig["adj_x"]["b"] = static_cast<bool>(adj_x);
                    myConfig["adj_y"]["b"] = static_cast<bool>(adj_y);

                    auto generic_op = std::make_shared<nnfusion::op::GenericOp>(
                        node_proto.output(0),
                        "BatchMatMul", // select which existing kernels to use;
                        myConfig);
                    auto generic_gnode = m_graph->add_node_and_edge(
                        generic_op, {lhs_broadcast_gnode, rhs_broadcast_gnode});

                    Shape output_shape(batch_shape.begin(), batch_shape.end());
                    if (lhs_rank > 1)
                    {
                        output_shape.push_back(left_full_shape[batchmm_output_rank - 2]);
                    }
                    if (rhs_rank > 1)
                    {
                        output_shape.push_back(right_full_shape[batchmm_output_rank - 1]);
                    }

                    auto reshape_back_op = std::make_shared<op::Reshape>(
                        get_default_order(generic_gnode->get_shape()), output_shape);
                    auto reshape_back_gnode =
                        m_graph->add_node_and_edge(reshape_back_op, {generic_gnode});

                    ret.emplace_back(node_proto.output(0), reshape_back_gnode);
                    return ret;
                }

            } // namespace set_1

        } //namespace onnx_import

    } // namespace frontend

} // namespace nnfusion
