// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "attention.hpp"
#include "../util/util.hpp"
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
                    TranslateAttentionOp(const onnx::NodeProto& node_proto,
                                         const NodeMap& all_ng_nodes,
                                         std::shared_ptr<nnfusion::graph::Graph> m_graph)
                {
                    auto input_indexes = GetAllInputIndex(all_ng_nodes, node_proto);
                    auto input = GetInputIndex(all_ng_nodes, node_proto, 0);
                    auto input_shape = input.gnode->get_output_shape(0);
                    auto weight = GetInputIndex(all_ng_nodes, node_proto, 1);
                    auto bias = GetInputIndex(all_ng_nodes, node_proto, 2);

                    nnfusion::graph::GNodeIndex mask_index, past;
                    if (input_indexes.size() >= 4)
                    {
                        mask_index = GetInputIndex(all_ng_nodes, node_proto, 3);
                    }
                    int past_sequence_length = 0;
                    if (input_indexes.size() == 5)
                    {
                        past = GetInputIndex(all_ng_nodes, node_proto, 4);
                        past_sequence_length = past.gnode->get_shape()[3];
                    }

                    Node node(node_proto);
                    auto num_heads = node.get_attribute_value<std::int64_t>("num_heads");
                    auto unidirectional =
                        node.get_attribute_value<std::int64_t>("unidirectional", 0);

                    size_t batch_size = input_shape[0];
                    size_t sequence_length = input_shape[1];
                    size_t hidden_size = input_shape[2];
                    auto broadcasted_op = std::make_shared<op::Broadcast>(
                        nnfusion::Shape({batch_size * sequence_length, 3 * hidden_size}),
                        nnfusion::AxisSet({0}));
                    broadcasted_op->set_name(node_proto.name() + "_broadcast");
                    auto broadcasted_gnode = m_graph->add_node_and_edge(broadcasted_op, {bias});

                    nnfusion::AxisVector ng_axis_order(input_shape.size());
                    std::iota(ng_axis_order.begin(), ng_axis_order.end(), 0);
                    auto reshape_op = std::make_shared<nnfusion::op::Reshape>(
                        ng_axis_order,
                        nnfusion::Shape({batch_size * sequence_length, hidden_size}));
                    reshape_op->set_name(node_proto.name() + "_reshape");
                    auto reshape_gnode = m_graph->add_node_and_edge(reshape_op, {input});

                    auto dot_op = std::make_shared<nnfusion::op::Dot>(0, false, false, false);
                    dot_op->set_name(node_proto.name() + "_dot");
                    auto dot_gnode =
                        m_graph->add_node_and_edge(dot_op, {GNodeIndex{reshape_gnode, 0}, weight});

                    auto add_op = std::make_shared<op::Add>();
                    add_op->set_name(node_proto.name() + "_add");
                    auto add_gnode = m_graph->add_node_and_edge(
                        add_op, {GNodeIndex{dot_gnode, 0}, GNodeIndex{broadcasted_gnode, 0}});

                    nnfusion::op::OpConfig::any myConfig;
                    myConfig["num_heads"] = num_heads;
                    myConfig["unidirectional"] = (bool)(unidirectional != 0);
                    myConfig["batch_size"] = batch_size;
                    myConfig["sequence_length"] = sequence_length;
                    myConfig["past_sequence_length"] = past_sequence_length;
                    myConfig["head_size"] = hidden_size / num_heads;
                    auto generic_op = std::make_shared<nnfusion::op::GenericOp>(
                        node_proto.name(), "Attention", myConfig);

                    if (input_indexes.size() == 5)
                    {
                        auto generic_gnode =
                            m_graph->add_node_and_edge(generic_op,
                                                       {GNodeIndex{add_gnode, 0}, mask_index, past},
                                                       /* output_size */ 2);

                        return {{node_proto.output(0), generic_gnode, 0},
                                {node_proto.output(1), generic_gnode, 1}};
                    }
                    else if (input_indexes.size() == 4)
                    {
                        auto generic_gnode = m_graph->add_node_and_edge(
                            generic_op, {GNodeIndex{add_gnode, 0}, mask_index});

                        return {{node_proto.output(0), generic_gnode, 0}};
                    }
                    else
                    {
                        auto generic_gnode =
                            m_graph->add_node_and_edge(generic_op, {GNodeIndex{add_gnode, 0}});

                        return {{node_proto.output(0), generic_gnode, 0}};
                    }
                }

            } // namespace set_1

        } //namespace onnx_import

    } // namespace frontend

} // namespace nnfusion
