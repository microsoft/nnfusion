// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "embed_layer_norm.hpp"
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
                    TranslateEmbedLayerNormOp(const onnx::NodeProto& node_proto,
                                              const NodeMap& all_ng_nodes,
                                              std::shared_ptr<nnfusion::graph::Graph> m_graph)
                {
                    auto input_indexes = GetAllInputIndex(all_ng_nodes, node_proto);
                    Node node(node_proto);
                    auto epsilon_value = node.get_attribute_value<float>("epsilon", 1e-6);

                    nnfusion::op::OpConfig::any myConfig;
                    myConfig["epsilon"] = epsilon_value;

                    auto generic_op = std::make_shared<nnfusion::op::GenericOp>(
                        node_proto.name(), "EmbedLayerNorm", myConfig);
                    auto generic_gnode =
                        m_graph->add_node_and_edge(generic_op, input_indexes, /* output_size */ 2);

                    return {{node_proto.output(0), generic_gnode, 0},
                            {node_proto.output(1), generic_gnode, 1}};
                }

            } // namespace set_1

        } //namespace onnx_import

    } // namespace frontend

} // namespace nnfusion
