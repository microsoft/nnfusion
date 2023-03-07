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

#pragma once

#include <memory>

#include "../core/node.hpp"
#include "../util/util.hpp"
#include "nnfusion/core/graph/util/autobroadcast.hpp"

namespace nnfusion
{
    namespace frontend
    {
        namespace onnx_import
        {
            namespace set_1
            {
                // support Mean for op_set 13 and below
                NamedNodeVector TranslateMeanOp(const onnx::NodeProto& node_proto,
                                                const NodeMap& all_ng_nodes,
                                                std::shared_ptr<nnfusion::graph::Graph> m_graph)
                {
                    auto input_indexes = GetAllInputIndex(all_ng_nodes, node_proto);
                    if (input_indexes.size() == 1)
                    {
                        return TranslateNoOp(node_proto, all_ng_nodes, m_graph);
                    }
                    NNFUSION_CHECK(input_indexes.size() >= 2);
                    auto lhs_index = input_indexes[0];
                    GNodeIndex rhs_index;
                    std::shared_ptr<nnfusion::graph::GNode> ret_node;
                    for (size_t i = 1; i < input_indexes.size(); i++)
                    {
                        rhs_index = input_indexes[i];
                        std::tie(lhs_index, rhs_index) =
                            graph::numpy_broadcast(std::make_pair(lhs_index, rhs_index), m_graph);
                        auto op = std::make_shared<nnfusion::op::Add>();
                        NNFUSION_CHECK(node_proto.output_size() == 1)
                            << "Binary op should only has one output.";
                        op->set_name(node_proto.output(0) + std::to_string(i - 1));
                        ret_node = m_graph->add_node_and_edge(op, {lhs_index, rhs_index});
                        lhs_index = GNodeIndex(ret_node);
                    }

                    auto et = ret_node->get_element_type();
                    element::Type size_et = et == element::f16 ? element::f32 : et;
                    auto size_op = std::make_shared<op::Constant>(
                        size_et,
                        nnfusion::Shape(),
                        std::vector<std::string>{std::to_string(input_indexes.size())});
                    size_op->set_name(node_proto.name() + "_size");
                    auto size_node = m_graph->add_node_and_edge(size_op, GNodeVector({}));
                    std::tie(ret_node, size_node) =
                        graph::numpy_broadcast(std::make_pair(ret_node, size_node), m_graph);

                    ret_node = m_graph->add_node_and_edge(std::make_shared<op::Divide>(),
                                                          {ret_node, size_node});

                    NamedNodeVector ret{{node_proto.output(0), ret_node}};
                    return ret;
                }

            } // namespace set_1

            namespace set_6
            {
                using set_1::TranslateMeanOp;
            } // namespace set_6
            namespace set_8
            {
                using set_1::TranslateMeanOp;
            } // namespace set_8
            namespace set_13
            {
                using set_1::TranslateMeanOp;
            } // namespace set_13
        }     // namespace onnx_import
    }         // namespace frontend
} // namespace nnfusion