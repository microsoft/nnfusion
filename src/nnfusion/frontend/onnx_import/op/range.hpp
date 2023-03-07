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

#include "../core/node.hpp"
#include "../util/util.hpp"

namespace nnfusion
{
    namespace frontend
    {
        namespace onnx_import
        {
            namespace set_11
            {
                template <typename T>
                NamedNodeVector TranslateRange(GNodeVector& input_gnodes,
                                               const onnx::NodeProto& node_proto,
                                               std::shared_ptr<nnfusion::graph::Graph> m_graph)
                {
                    auto start_gnode = input_gnodes[0];
                    auto limit_gnode = input_gnodes[1];
                    auto delta_gnode = input_gnodes[2];

                    std::vector<T> start_vec;
                    NNFUSION_CHECK(GetValueFromNGraphOp<T>(start_gnode, &start_vec) == true);
                    NNFUSION_CHECK(start_vec.size() > 0);
                    std::vector<T> limit_vec;
                    NNFUSION_CHECK(GetValueFromNGraphOp<T>(limit_gnode, &limit_vec) == true);
                    NNFUSION_CHECK(limit_vec.size() > 0);
                    std::vector<T> delta_vec;
                    NNFUSION_CHECK(GetValueFromNGraphOp<T>(delta_gnode, &delta_vec) == true);
                    NNFUSION_CHECK(delta_vec.size() > 0);

                    nnfusion::op::OpConfig::any myConfig;
                    myConfig["start"] = start_vec[0];
                    myConfig["limit"] = limit_vec[0];
                    myConfig["delta"] = delta_vec[0];

                    auto generic_op = std::make_shared<nnfusion::op::GenericOp>(
                        node_proto.output(0), "Range", myConfig);

                    auto generic_gnode = m_graph->add_node_and_edge(generic_op, input_gnodes);
                    NamedNodeVector ret{{node_proto.output(0), generic_gnode}};
                    return ret;
                }

                NamedNodeVector TranslateRangeOp(const onnx::NodeProto& node_proto,
                                                 const NodeMap& all_ng_nodes,
                                                 std::shared_ptr<nnfusion::graph::Graph> m_graph)
                {
                    GNodeVector input_gnodes;
                    auto start_gnode = GetInputNode(all_ng_nodes, node_proto, 0);
                    input_gnodes.push_back(start_gnode);
                    auto limit_gnode = GetInputNode(all_ng_nodes, node_proto, 1);
                    input_gnodes.push_back(limit_gnode);
                    auto delta_gnode = GetInputNode(all_ng_nodes, node_proto, 2);
                    input_gnodes.push_back(delta_gnode);

                    Node node(node_proto);
                    auto element_type = start_gnode->get_element_type();
                    NNFUSION_CHECK(element_type == limit_gnode->get_element_type());
                    NNFUSION_CHECK(element_type == delta_gnode->get_element_type());

                    if (element_type == element::f16 || element_type == element::f32 ||
                        element_type == element::f64)
                        return TranslateRange<long double>(input_gnodes, node_proto, m_graph);
                    else if (element_type == element::i32 || element_type == element::i64)
                        return TranslateRange<int64_t>(input_gnodes, node_proto, m_graph);
                    else
                        NNFUSION_CHECK_FAIL() << "non-supported data type for Range op: "
                                              << element_type.c_type_string();
                }

            } // namespace set_11
        }     // namespace onnx_import
    }         // namespace frontend
} // namespace nnfusion