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
            namespace set_9
            {
                template <typename T>
                std::shared_ptr<op::Constant> __make_output_constant_op(GNodeIndex input)
                {
                    auto input_shape = input.get_shape();
                    size_t input_rank = input_shape.size();
                    std::vector<T> input_value;
                    NNFUSION_CHECK(GetValueFromNGraphOp(input.gnode, &input_value));

                    std::vector<std::vector<int64_t>> non_zero_indices(input_rank);

                    for (size_t i = 0; i < input_value.size(); i++)
                    {
                        if (input_value[i] > 0 || input_value[i] < 0)
                        {
                            size_t cur = i;
                            for (int index = input_rank - 1; index >= 0; index--)
                            {
                                non_zero_indices[index].push_back(cur % input_shape[index]);
                                cur = cur / input_shape[index];
                            }
                        }
                    }

                    std::vector<int64_t> raw_data;
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
                    return const_op;
                }

                NamedNodeVector TranslateNonZeroOp(const onnx::NodeProto& node_proto,
                                                   const NodeMap& all_ng_nodes,
                                                   std::shared_ptr<nnfusion::graph::Graph> m_graph)
                {
                    static const std::map<
                        element::Type,
                        std::function<std::shared_ptr<op::Constant>(GNodeIndex input)>>
                        the_map = {{element::f32, __make_output_constant_op<float>},
                                   {element::f64, __make_output_constant_op<double>},
                                   {element::boolean, __make_output_constant_op<int32_t>},
                                   {element::i32, __make_output_constant_op<int32_t>},
                                   {element::i64, __make_output_constant_op<int64_t>},
                                   {element::u32, __make_output_constant_op<uint32_t>},
                                   {element::u64, __make_output_constant_op<uint64_t>}};

                    GNodeIndexVector input_indexes = GetAllInputIndex(all_ng_nodes, node_proto);

                    auto input = input_indexes[0];

                    NNFUSION_CHECK(the_map.count(input.get_element_type()) > 0)
                        << "NonZero doesn't support input type of "
                        << input.get_element_type().c_type_string();

                    const auto& extract_output_func = the_map.at(input.get_element_type());
                    auto const_op = extract_output_func(input);
                    const_op->set_name(node_proto.output(0));
                    auto const_gnode = m_graph->add_node_and_edge(const_op, GNodeVector({}));

                    return {{node_proto.output(0), const_gnode}};
                }

            } // namespace set_9

        } //namespace onnx_import

    } // namespace frontend

} // namespace nnfusion