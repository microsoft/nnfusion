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

                    std::vector<int64> start_vec;
                    NNFUSION_CHECK(GetValueFromNGraphOp<int64>(start_gnode, &start_vec) == true);
                    NNFUSION_CHECK(start_vec.size() > 0);
                    std::vector<int64> limit_vec;
                    NNFUSION_CHECK(GetValueFromNGraphOp<int64>(limit_gnode, &limit_vec) == true);
                    NNFUSION_CHECK(limit_vec.size() > 0);
                    std::vector<int64> delta_vec;
                    NNFUSION_CHECK(GetValueFromNGraphOp<int64>(delta_gnode, &delta_vec) == true);
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

            } // namespace set_11
        }     // namespace onnx_import
    }         // namespace frontend
} // namespace nnfusion