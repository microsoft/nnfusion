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
                    Node node(node_proto);
                    auto num_heads = node.get_attribute_value<std::int64_t>("num_heads");
                    auto unidirectional = node.get_attribute_value<std::int64_t>("unidirectional", 0);
                    nnfusion::op::OpConfig::any myConfig;
                    myConfig["num_heads"] = num_heads;
                    myConfig["unidirectional"] = (bool)(unidirectional != 0);

                    auto generic_op = std::make_shared<nnfusion::op::GenericOp>(
                        node_proto.name(), "Attention", myConfig);

                    if (input_indexes.size() == 5)
                    {
                        auto generic_gnode = m_graph->add_node_and_edge(
                            generic_op, input_indexes, /* output_size */ 2);

                        return {{node_proto.output(0), generic_gnode, 0},
                                {node_proto.output(1), generic_gnode, 1}};
                    }
                    else
                    {
                        auto generic_gnode = m_graph->add_node_and_edge(generic_op, input_indexes);

                        return {{node_proto.output(0), generic_gnode, 0}};
                    }
                }

            } // namespace set_1

        } //namespace onnx_import

    } // namespace frontend

} // namespace nnfusion
