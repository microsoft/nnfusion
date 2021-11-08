// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "../util/util.hpp"
#include "embed_layer_norm.hpp"
#include "nnfusion/core/operators/generic_op/generic_op.hpp"
#include "nnfusion/frontend/util/evaluator.hpp"
namespace nnfusion
{
    namespace frontend
    {
        namespace onnx_import
        {
            namespace set_1
            {
                NamedNodeVector TranslateRollOp(const onnx::NodeProto& node_proto,
                                                const NodeMap& all_ng_nodes,
                                                std::shared_ptr<nnfusion::graph::Graph> m_graph)
                {
                    auto input = GetInputIndex(all_ng_nodes, node_proto, 0);
                    auto shifts = GetInputNode(all_ng_nodes, node_proto, 1);
                    auto dims = GetInputNode(all_ng_nodes, node_proto, 2);

                    NNFUSION_CHECK(shifts->is_constant() && dims->is_constant());

                    std::vector<int64> shifts_value, dims_value;
                    bool status =
                        nnfusion::frontend::GetValueFromNGraphOp<int64>(shifts, &shifts_value);
                    NNFUSION_CHECK(status && !shifts_value.empty());
                    status = nnfusion::frontend::GetValueFromNGraphOp<int64>(dims, &dims_value);
                    NNFUSION_CHECK(status && !dims_value.empty());

                    nnfusion::op::OpConfig::any myConfig;
                    myConfig["shifts"] = shifts_value;
                    myConfig["dims"] = dims_value;

                    auto generic_op = std::make_shared<nnfusion::op::GenericOp>(
                        node_proto.name(), "Roll", myConfig);
                    auto generic_gnode = m_graph->add_node_and_edge(generic_op, {input});

                    return {{node_proto.output(0), generic_gnode, 0}};
                }

            } // namespace set_1

        } //namespace onnx_import

    } // namespace frontend

} // namespace nnfusion