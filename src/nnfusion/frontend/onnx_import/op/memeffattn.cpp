// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "memeffattn.hpp"
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
                    TranslateMemEffAttnOp(const onnx::NodeProto& node_proto,
                                          const NodeMap& all_ng_nodes,
                                          std::shared_ptr<nnfusion::graph::Graph> m_graph)
                {
                    auto input_indexes = GetAllInputIndex(all_ng_nodes, node_proto);
                    auto q = GetInputIndex(all_ng_nodes, node_proto, 0);
                    auto q_shape = q.gnode->get_output_shape(0);
                    auto k = GetInputIndex(all_ng_nodes, node_proto, 1);
                    auto k_shape = k.gnode->get_output_shape(0);
                    auto v = GetInputIndex(all_ng_nodes, node_proto, 2);
                    size_t batch_size = q_shape[0];
                    size_t num_heads = q_shape[1];
                    size_t seq_len = q_shape[2];
                    size_t head_size = q_shape[3];
                    size_t seq_len_kv = k_shape[2];
                    size_t head_size_v = k_shape[3];

                    Node node(node_proto);
                    auto softmax_scale = node.get_attribute_value<float>("softmax_scale");
                    auto is_causal = node.get_attribute_value<std::int64_t>("is_causal", 0);
                    nnfusion::op::OpConfig::any myConfig;
                    myConfig["batch_size"] = batch_size;
                    myConfig["is_causal"] = (bool)(is_causal != 0);
                    myConfig["num_heads"] = num_heads;
                    myConfig["seq_len"] = seq_len;
                    myConfig["seq_len_kv"] = seq_len_kv;
                    myConfig["head_size"] = head_size;
                    myConfig["head_size_v"] = head_size_v;
                    myConfig["softmax_scale"] = softmax_scale;

                    auto generic_op = std::make_shared<nnfusion::op::GenericOp>(
                        node_proto.name(), "MemEffAttn", myConfig);

                    auto generic_gnode = m_graph->add_node_and_edge(generic_op, input_indexes);

                    return {{node_proto.output(0), generic_gnode, 0}};
                }

            } // namespace set_1

        } //namespace onnx_import

    } // namespace frontend

} // namespace nnfusion
