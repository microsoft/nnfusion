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

#include "where.hpp"
#include "nnfusion/core/graph/util/autobroadcast.hpp"
#include "nnfusion/core/graph/util/autobroadcast.hpp"
#include "nnfusion/core/graph/util/numpy_transpose.hpp"
#include "nnfusion/core/operators/generic_op/generic_op.hpp"

namespace nnfusion
{
    namespace frontend
    {
        namespace onnx_import
        {
            namespace set_9
            {
                NamedNodeVector TranslateWhereOp(const onnx::NodeProto& node_proto,
                                                 const NodeMap& all_ng_nodes,
                                                 std::shared_ptr<nnfusion::graph::Graph> m_graph)
                {
                    auto input_indices = GetAllInputIndex(all_ng_nodes, node_proto);
                    auto cond_gnode = input_indices[0];
                    auto x_gnode = input_indices[1];
                    auto y_gnode = input_indices[2];

                    std::tie(x_gnode, y_gnode) =
                        graph::numpy_broadcast(std::make_pair(x_gnode, y_gnode), m_graph);

                    std::tie(x_gnode, cond_gnode) =
                        graph::numpy_broadcast(std::make_pair(x_gnode, cond_gnode), m_graph);

                    auto node_name = node_proto.output(0);
                    nnfusion::op::OpConfig::any op_config;
                    if (x_gnode.get_shape() != y_gnode.get_shape())
                    {
                        std::tie(x_gnode, y_gnode) =
                            graph::numpy_broadcast(std::make_pair(x_gnode, y_gnode), m_graph);
                    }
                    auto where_op = std::make_shared<op::GenericOp>(node_name, "Select", op_config);
                    auto where_gnode =
                        m_graph->add_node_and_edge(where_op, {cond_gnode, x_gnode, y_gnode});

                    return {{node_proto.output(0), GNodeIndex(where_gnode)}};
                }
            } // namespace set_9
        }     // namespace onnx_import
    }         // namespace frontend
} // namespace nnfusion
