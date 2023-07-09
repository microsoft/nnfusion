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

#include "reshape.hpp"
#include "nnfusion/frontend/util/evaluator.hpp"

namespace nnfusion
{
    namespace frontend
    {
        namespace onnx_import
        {
            namespace set_1
            {
                NamedNodeVector TranslateReshapeOp(const onnx::NodeProto& node_proto,
                                                   const NodeMap& all_ng_nodes,
                                                   std::shared_ptr<nnfusion::graph::Graph> m_graph)
                {
                    auto input_indexes = GetAllInputIndex(all_ng_nodes, node_proto);

                    auto input = input_indexes[0];

                    auto input_shape = input.get_shape();
                    size_t input_rank = input_shape.size();
                    std::vector<int64> output_shape;
                    NNFUSION_CHECK(GetValueFromNGraphOp(input_indexes[1].gnode, &output_shape));
                    NNFUSION_CHECK(std::count(output_shape.begin(), output_shape.end(), -1) <= 1)
                        << "Shape should have at most 1 dynamic dimension";

                    size_t num_input_elements = nnfusion::shape_size(input_shape);

                    // infer the dimension of -1 and 0
                    auto dynamic_dim = output_shape.end();
                    size_t static_size = 1;
                    for (auto it = output_shape.begin(); it != output_shape.end(); it++)
                    {
                        if (*it == -1)
                        {
                            dynamic_dim = it;
                        }
                        else
                        {
                            if (*it == 0)
                            {
                                *it = *std::next(input_shape.begin(),
                                                 std::distance(output_shape.begin(), it));
                            }
                            static_size *= *it;
                        }
                    }
                    if (dynamic_dim == output_shape.end())
                    {
                        NNFUSION_CHECK(static_size == num_input_elements)
                            << "Reshape size doesn\'t match";
                    }
                    else
                    {
                        NNFUSION_CHECK(num_input_elements % static_size == 0)
                            << "The product of static dims cannot be evenly divided by element "
                               "number.";
                        *dynamic_dim = num_input_elements / static_size;
                    }

                    nnfusion::Shape ng_shape(output_shape.begin(), output_shape.end());

                    nnfusion::AxisVector ng_axis_order(input_shape.size());
                    std::iota(ng_axis_order.begin(), ng_axis_order.end(), 0);
                    auto reshape_op =
                        std::make_shared<nnfusion::op::Reshape>(ng_axis_order, ng_shape);
                    reshape_op->set_name(node_proto.output(0));
                    auto reshape_gnode = m_graph->add_node_and_edge(reshape_op, {input});
                    return {{node_proto.output(0), reshape_gnode}};
                }

            } // namespace set_1

            namespace set_1
            {
                NamedNodeVector
                    TranslateReshapeGradOp(const onnx::NodeProto& node_proto,
                                           const NodeMap& all_ng_nodes,
                                           std::shared_ptr<nnfusion::graph::Graph> m_graph)
                {
                    // y = reshape(x, shape(y)), x_grad = reshape(y_grad, shape(x))
                    auto input_indexes = GetAllInputIndex(all_ng_nodes, node_proto);
                    NNFUSION_CHECK(input_indexes.size() == 2);

                    auto x = input_indexes[0];
                    auto y_grad = input_indexes[1];

                    // x_grad
                    auto x_shape = x.get_shape();
                    auto y_shape = y_grad.get_shape();

                    nnfusion::AxisVector ng_axis_order(y_shape.size());
                    std::iota(ng_axis_order.begin(), ng_axis_order.end(), 0);
                    auto x_grad_op =
                        std::make_shared<nnfusion::op::Reshape>(ng_axis_order, x_shape);
                    x_grad_op->set_name(node_proto.output(0));
                    auto x_grad = m_graph->add_node_and_edge(x_grad_op, {y_grad});

                    return {{node_proto.output(0), x_grad}};
                }

            } // namespace set_1

        } //namespace onnx_import

    } // namespace frontend

} // namespace nnfusion
