//----------------------------------------------------------------------------------------------
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See License.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <vector>

#include "../util/util.hpp"
#include "expand.hpp"
#include "nnfusion/core/graph/util/autobroadcast.hpp"
#include "nnfusion/frontend/util/evaluator.hpp"

namespace nnfusion
{
    namespace frontend
    {
        namespace onnx_import
        {
            namespace set_1
            {
                NamedNodeVector TranslateNonZeroOp(const onnx::NodeProto& node_proto,
                                                   const NodeMap& all_ng_nodes,
                                                   std::shared_ptr<nnfusion::graph::Graph> m_graph)
                {
                    GNodeIndexVector input_indexes = GetAllInputIndex(all_ng_nodes, node_proto);

                    auto input = input_indexes[0];

                    NNFUSION_CHECK(input.get_element_type() == nnfusion::element::i64);

                    auto input_shape = input.get_shape();
                    size_t input_rank = input_shape.size();
                    std::vector<int64> input_value;
                    NNFUSION_CHECK(GetValueFromNGraphOp(input_indexes[0].gnode, &input_value));

                    std::vector<std::vector<int64>> non_zero_indices(input_rank);

                    for (size_t i = 0; i < input_value.size(); i++)
                    {
                        if (input_value[i] != 0)
                        {
                            size_t cur = i;
                            for (int index = input_rank - 1; index >= 0; index--)
                            {
                                non_zero_indices[index].push_back(cur % input_shape[index]);
                                cur = cur / input_shape[index];
                            }
                        }
                    }

                    std::vector<int64> raw_data;
                    for (size_t i = 0; i < non_zero_indices.size(); i++)
                    {
                        for (size_t j = 0; j < non_zero_indices[0].size(); j++)
                        {
                            raw_data.push_back(non_zero_indices[i][j]);
                        }
                    }

                    auto const_op = std::make_shared<op::Constant>(
                        element::i64,
                        Shape{non_zero_indices.size(), non_zero_indices[0].size()},
                        raw_data);
                    const_op->set_name(node_proto.output(0));
                    auto const_gnode = m_graph->add_node_and_edge(const_op, GNodeVector({}));

                    return {{node_proto.output(0), const_gnode}};
                }

            } // namespace set_1

        } //namespace onnx_import

    } // namespace frontend

} // namespace nnfusion
