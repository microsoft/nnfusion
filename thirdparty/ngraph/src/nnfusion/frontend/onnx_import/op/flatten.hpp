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

#include "core/node.hpp"

namespace nnfusion
{
    namespace frontend
    {
        namespace onnx_import
        {
            namespace set_1
            {
                NamedNodeVector TranslateFlattenOp(const onnx::NodeProto& node_proto,
                                                   const NodeMap& all_ng_nodes,
                                                   std::shared_ptr<nnfusion::graph::Graph> m_graph)
                {
                    GNodeIndexVector input_indexes = GetAllInputIndex(all_ng_nodes, node_proto);
                    auto input = input_indexes[0];
                    auto input_shape = input.get_shape();

                    Node node(node_proto);
                    auto axis = node.get_attribute_value<int64_t>("axis");
                    size_t dim0 = 1;
                    size_t dim1 = 1;
                    for (size_t i = 0; i < input_shape.size(); i++)
                    {
                        if (i < axis)
                        {
                            dim0 *= input_shape[i];
                        }
                        else
                        {
                            dim1 *= input_shape[i];
                        }
                    }

                    auto reshape_op = std::make_shared<op::Reshape>(get_default_order(input_shape),
                                                                    Shape{dim0, dim1});
                    auto reshape_gnode = m_graph->add_node_and_edge(reshape_op, {input_indexes});
                    return {{node_proto.output(0), GNodeIndex{reshape_gnode}}};
                }

            } // namespace set_1
        }     //namespace onnx_import

    } // namespace frontend

} // namespace nnfusion
