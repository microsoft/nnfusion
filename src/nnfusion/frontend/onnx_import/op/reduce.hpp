//----------------------------------------------------------------------------------------------
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See License.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include "core/node.hpp"
#include "nnfusion/core/operators/generic_op/generic_op.hpp"
#include "nnfusion/core/operators/op_define/noop.hpp"

namespace nnfusion
{
    namespace frontend
    {
        namespace onnx_import
        {
            namespace set_1
            {
                template <typename optype>
                std::shared_ptr<GNode>
                    AddPrologueOrEpilogueOp(std::shared_ptr<nnfusion::graph::Graph> m_graph,
                                            std::shared_ptr<GNode>& input_gnode,
                                            nnfusion::AxisSet& axes)
                {
                    auto op = std::make_shared<optype>();
                    auto gnode = m_graph->add_node_and_edge(op, {input_gnode});
                    return gnode;
                }

                // Special handle of Divide op for ReduceMean (Sum+Divide)
                template <>
                std::shared_ptr<GNode> AddPrologueOrEpilogueOp<op::Divide>(
                    std::shared_ptr<nnfusion::graph::Graph> m_graph,
                    std::shared_ptr<GNode>& input_gnode,
                    nnfusion::AxisSet& axes)
                {
                    auto sum_gnode = input_gnode;
                    auto input_shape = sum_gnode->get_input_shape(0);
                    size_t reduced_ele_count = 1;
                    for (auto i : axes)
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
                    return mean_gnode;
                }

                template <>
                std::shared_ptr<GNode> AddPrologueOrEpilogueOp<op::NoOp>(
                    std::shared_ptr<nnfusion::graph::Graph> m_graph,
                    std::shared_ptr<GNode>& input_gnode,
                    nnfusion::AxisSet& axes)
                {
                    return input_gnode;
                }

                template <typename PrologueOp, typename ReduceOp, typename EpilogueOp>
                NamedNodeVector TranslateReduceOp(const onnx::NodeProto& node_proto,
                                                  const NodeMap& all_ng_nodes,
                                                  std::shared_ptr<nnfusion::graph::Graph> m_graph)
                {
                    auto input_index = GetInputIndex(all_ng_nodes, node_proto, 0);
                    auto input_shape = input_index.get_shape();

                    // get attributes
                    Node node(node_proto);
                    auto keepdims = node.get_attribute_value<int64>("keepdims", 1);

                    nnfusion::AxisSet reduction_axes;
                    {
                        auto axes = node.get_attribute_value<std::vector<int64_t>>("axes", {});
                        if (axes.empty())
                        {
                            auto axes_uint = get_default_order(input_shape);
                            std::copy(axes_uint.begin(),
                                      axes_uint.end(),
                                      std::inserter(reduction_axes, reduction_axes.end()));
                        }
                        else
                        {
                            for (auto axis : axes)
                            {
                                reduction_axes.insert(axis += axis < 0 ? input_shape.size() : 0);
                            }
                        }
                    }

                    // Add prologue op
                    auto pro_gnode = AddPrologueOrEpilogueOp<PrologueOp>(
                        m_graph, input_index.gnode, reduction_axes);

                    auto sum_op = std::make_shared<ReduceOp>(reduction_axes);
                    auto sum_gnode = m_graph->add_node_and_edge(sum_op, {pro_gnode});

                    // Add epilogue op
                    auto epi_gnode =
                        AddPrologueOrEpilogueOp<EpilogueOp>(m_graph, sum_gnode, reduction_axes);

                    NamedNodeVector ret;
                    if (keepdims)
                    {
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
                    }
                    else
                    {
                        epi_gnode->get_op_ptr()->set_name(node_proto.output(0));
                        ret.push_back({node_proto.output(0), epi_gnode});
                    }

                    return ret;
                }
            } // namespace set_1

            namespace set_11
            {
                using set_1::TranslateReduceOp;
            }

            namespace set_12
            {
                using set_1::TranslateReduceOp;
            }

            namespace set_13
            {
                using set_1::TranslateReduceOp;

                // ReduceSum-13 has move the axes to input
                NamedNodeVector
                    TranslateReduceSumOp(const onnx::NodeProto& node_proto,
                                         const NodeMap& all_ng_nodes,
                                         std::shared_ptr<nnfusion::graph::Graph> m_graph)
                {
                    auto input_indexs = GetAllInputIndex(all_ng_nodes, node_proto);
                    NNFUSION_CHECK(input_indexs.size() > 0);
                    auto input_index = input_indexs[0];
                    auto input_shape = input_index.get_shape();

                    Node node(node_proto);
                    auto keepdims = node.get_attribute_value<int64>("keepdims", 1);

                    std::vector<int64> axes;
                    if (input_indexs.size() == 2)
                    {
                        GetValueFromNGraphOp<int64>(input_indexs[1].gnode, &axes);
                    }

                    if (axes.empty())
                    {
                        // no axes input
                        auto noop_with_empty_axes =
                            node.get_attribute_value<int64>("noop_with_empty_axes", 0);
                        // When this attribute is true, the output tensor would be equivalent
                        // to input tensor.
                        if (noop_with_empty_axes)
                        {
                            NamedNodeVector ret;
                            ret.push_back({node_proto.output(0), input_index.gnode});
                            return ret;
                        }
                    }

                    nnfusion::AxisSet reduction_axes;
                    {
                        if (axes.empty())
                        {
                            auto axes_uint = get_default_order(input_shape);
                            std::copy(axes_uint.begin(),
                                      axes_uint.end(),
                                      std::inserter(reduction_axes, reduction_axes.end()));
                        }
                        else
                        {
                            for (auto axis : axes)
                            {
                                reduction_axes.insert(axis += axis < 0 ? input_shape.size() : 0);
                            }
                        }
                    }

                    auto sum_op = std::make_shared<op::Sum>(reduction_axes);
                    auto sum_gnode = m_graph->add_node_and_edge(sum_op, {input_index.gnode});

                    NamedNodeVector ret;
                    if (keepdims)
                    {
                        nnfusion::Shape result_shape_with_keep(input_shape.size());

                        for (size_t i = 0; i < input_shape.size(); i++)
                        {
                            result_shape_with_keep[i] =
                                reduction_axes.count(i) == 0 ? input_shape[i] : 1;
                        }
                        nnfusion::AxisVector axis_order(sum_gnode->get_shape().size());
                        std::iota(axis_order.begin(), axis_order.end(), 0);
                        auto reshape_op =
                            std::make_shared<op::Reshape>(axis_order, result_shape_with_keep);
                        reshape_op->set_name(node_proto.output(0));
                        auto reshape_gnode = m_graph->add_node_and_edge(reshape_op, {sum_gnode});
                        ret.push_back({node_proto.output(0), reshape_gnode});
                    }
                    else
                    {
                        sum_gnode->get_op_ptr()->set_name(node_proto.output(0));
                        ret.push_back({node_proto.output(0), sum_gnode});
                    }

                    return ret;
                }

            } // namespace set_13

            namespace set_18
            {
                // Opset18 has move the axes to input
                template <typename PrologueOp, typename ReduceOp, typename EpilogueOp>
                NamedNodeVector TranslateReduceOp(const onnx::NodeProto& node_proto,
                                                  const NodeMap& all_ng_nodes,
                                                  std::shared_ptr<nnfusion::graph::Graph> m_graph)
                {
                    auto input_indexs = GetAllInputIndex(all_ng_nodes, node_proto);
                    NNFUSION_CHECK(input_indexs.size() > 0);
                    auto input_index = input_indexs[0];
                    auto input_shape = input_index.get_shape();

                    Node node(node_proto);
                    auto keepdims = node.get_attribute_value<int64>("keepdims", 1);

                    std::vector<int64> axes;
                    if (input_indexs.size() == 2)
                    {
                        GetValueFromNGraphOp<int64>(input_indexs[1].gnode, &axes);
                    }

                    if (axes.empty())
                    {
                        // no axes input
                        auto noop_with_empty_axes =
                            node.get_attribute_value<int64>("noop_with_empty_axes", 0);
                        // When this attribute is true, the output tensor would be equivalent
                        // to input tensor.
                        if (noop_with_empty_axes)
                        {
                            NamedNodeVector ret;
                            ret.push_back({node_proto.output(0), input_index.gnode});
                            return ret;
                        }
                    }

                    nnfusion::AxisSet reduction_axes;
                    {
                        if (axes.empty())
                        {
                            auto axes_uint = get_default_order(input_shape);
                            std::copy(axes_uint.begin(),
                                      axes_uint.end(),
                                      std::inserter(reduction_axes, reduction_axes.end()));
                        }
                        else
                        {
                            for (auto axis : axes)
                            {
                                reduction_axes.insert(axis += axis < 0 ? input_shape.size() : 0);
                            }
                        }
                    }

                    // Add prologue op
                    auto pro_gnode = set_1::AddPrologueOrEpilogueOp<PrologueOp>(
                        m_graph, input_index.gnode, reduction_axes);

                    auto sum_op = std::make_shared<ReduceOp>(reduction_axes);
                    auto sum_gnode = m_graph->add_node_and_edge(sum_op, {pro_gnode});

                    // Add epilogue op
                    auto epi_gnode = set_1::AddPrologueOrEpilogueOp<EpilogueOp>(
                        m_graph, sum_gnode, reduction_axes);

                    NamedNodeVector ret;
                    if (keepdims)
                    {
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
                    }
                    else
                    {
                        epi_gnode->get_op_ptr()->set_name(node_proto.output(0));
                        ret.push_back({node_proto.output(0), epi_gnode});
                    }

                    return ret;
                }

            } // namespace set_18

        } // namespace onnx_import

    } // namespace frontend

} // namespace nnfusion