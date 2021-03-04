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

#include "gemm.hpp"
#include "../util/util.hpp"
#include "nnfusion/core/graph/util/autobroadcast.hpp"
#include "nnfusion/core/graph/util/numpy_transpose.hpp"

namespace nnfusion
{
    namespace frontend
    {
        namespace onnx_import
        {
            namespace set_1
            {
                NamedNodeVector TranslateGemmOp(const onnx::NodeProto& node_proto,
                                                const NodeMap& all_ng_nodes,
                                                std::shared_ptr<nnfusion::graph::Graph> m_graph)
                {
                    auto input_indexes = GetAllInputIndex(all_ng_nodes, node_proto);

                    auto A = input_indexes[0];
                    auto B = input_indexes[1];

                    Node node(node_proto);
                    auto beta_value = node.get_attribute_value<float>("beta", 1.0);
                    auto alpha_value = node.get_attribute_value<float>("alpha", 1.0);
                    auto transA = node.get_attribute_value<int64>("transA", 0);
                    auto transB = node.get_attribute_value<int64>("transB", 0);

                    auto result = m_graph->add_node_and_edge(
                        std::make_shared<op::Dot>(
                            0, false, static_cast<bool>(transA), static_cast<bool>(transB)),
                        {A, B});

                    if (alpha_value != 1)
                    {
                        auto alpha_op = std::make_shared<op::Constant>(
                            element::f32, result->get_shape(), std::vector<float>({alpha_value}));
                        auto alpha = m_graph->add_node_and_edge(alpha_op, graph::GNodeVector({}));
                        if (alpha->get_element_type() != result->get_element_type())
                        {
                            auto cast_op =
                                std::make_shared<op::Convert>(result->get_element_type());
                            alpha = m_graph->add_node_and_edge(cast_op, {alpha});
                        }
                        result = m_graph->add_node_and_edge(std::make_shared<op::Multiply>(),
                                                            {result, alpha});
                    }

                    if (beta_value != 0)
                    {
                        auto C = input_indexes[2];
                        auto beta_op = std::make_shared<op::Constant>(
                            element::f32, C.get_shape(), std::vector<float>({beta_value}));
                        auto beta = m_graph->add_node_and_edge(beta_op, graph::GNodeVector({}));
                        if (beta->get_element_type() != C.get_element_type())
                        {
                            auto cast_op = std::make_shared<op::Convert>(C.get_element_type());
                            beta = m_graph->add_node_and_edge(cast_op, {beta});
                        }
                        auto bias = m_graph->add_node_and_edge(std::make_shared<op::Multiply>(),
                                                               {C, GNodeIndex{beta, 0}});
                        std::tie(result, bias) =
                            numpy_broadcast(std::make_pair(result, bias), m_graph);
                        result =
                            m_graph->add_node_and_edge(std::make_shared<op::Add>(), {result, bias});
                    }

                    return {{node_proto.output(0), result}};
                }

            } // namespace set_1

        } //namespace onnx_import

    } // namespace frontend

} // namespace nnfusion
