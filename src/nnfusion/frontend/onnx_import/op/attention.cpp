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
                    auto input_gnode = GetInputNode(all_ng_nodes, node_proto, 0);
                    auto weight_gnode = GetInputNode(all_ng_nodes, node_proto, 1);
                    auto bias_gnode = GetInputNode(all_ng_nodes, node_proto, 2);
                    std::shared_ptr<nnfusion::graph::GNode> mask_index_gnode, past_gnode;
                    if (input_indexes.size() >= 4)
                    {
                        mask_index_gnode = GetInputNode(all_ng_nodes, node_proto, 3);
                    }
                    int past_sequence_length = 0;
                    if (input_indexes.size() == 5)
                    {
                        past_gnode = GetInputNode(all_ng_nodes, node_proto, 4);
                        past_sequence_length = past_gnode->get_shape()[3];
                    }

                    Node node(node_proto);
                    auto num_heads = node.get_attribute_value<std::int64_t>("num_heads");
                    auto unidirectional =
                        node.get_attribute_value<std::int64_t>("unidirectional", 0);

                    size_t batch_size = input_gnode->get_shape()[0];
                    size_t sequence_length = input_gnode->get_shape()[1];
                    size_t hidden_size = input_gnode->get_shape()[2];
                    auto broadcasted_op = std::make_shared<op::Broadcast>(
                        nnfusion::Shape({3 * hidden_size, batch_size * sequence_length}),
                        nnfusion::AxisSet({1}));
                    broadcasted_op->set_name(node_proto.name() + "_broadcast");
                    auto broadcasted_gnode =
                        m_graph->add_node_and_edge(broadcasted_op, {bias_gnode});

                    auto dot_op = std::make_shared<nnfusion::op::Dot>(0, false, false, false);
                    dot_op->set_name(node_proto.name() + "_dot");
                    auto dot_gnode =
                        m_graph->add_node_and_edge(dot_op, {weight_gnode, input_gnode});

                    auto add_op = std::make_shared<op::Add>();
                    add_op->set_name(node_proto.name() + "_add");
                    auto add_gnode =
                        m_graph->add_node_and_edge(add_op, {dot_gnode, broadcasted_gnode});

                    nnfusion::op::OpConfig::any myConfig;
                    myConfig["num_heads"] = num_heads;
                    myConfig["unidirectional"] = (bool)(unidirectional != 0);
                    myConfig["batch_size"] = batch_size;
                    myConfig["sequence_length"] = sequence_length;
                    myConfig["past_sequence_length"] = past_sequence_length;
                    myConfig["head_size"] = hidden_size / num_heads;
                    auto generic_op = std::make_shared<nnfusion::op::GenericOp>(
                        node_proto.name(), "QkvtoCtx", myConfig);

                    if (input_indexes.size() == 3)
                    {
                        auto generic_gnode =
                            m_graph->add_node_and_edge(generic_op,
                                                       {add_gnode, mask_index_gnode, past_gnode},
                                                       /* output_size */ 2);

                        return {{node_proto.output(0), generic_gnode, 0},
                                {node_proto.output(1), generic_gnode, 1}};
                    }
                    else if (input_indexes.size() == 2)
                    {
                        auto generic_gnode =
                            m_graph->add_node_and_edge(generic_op, {add_gnode, mask_index_gnode});

                        return {{node_proto.output(0), generic_gnode, 0}};
                    }
                    else
                    {
                        auto generic_gnode = m_graph->add_node_and_edge(generic_op, {add_gnode});

                        return {{node_proto.output(0), generic_gnode, 0}};
                    }
                }

            } // namespace set_1

        } //namespace onnx_import

    } // namespace frontend

} // namespace nnfusion
