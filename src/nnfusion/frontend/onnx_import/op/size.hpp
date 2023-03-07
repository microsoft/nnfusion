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
                NamedNodeVector TranslateSizeOp(const onnx::NodeProto& node_proto,
                                                const NodeMap& all_ng_nodes,
                                                std::shared_ptr<nnfusion::graph::Graph> m_graph)
                {
                    GNodeIndexVector input_indexes = GetAllInputIndex(all_ng_nodes, node_proto);

                    auto input = input_indexes[0];
                    auto input_shape = input.get_shape();

                    int64_t tensor_size = nnfusion::shape_size<nnfusion::Shape>(input_shape);

                    auto size_gnode = m_graph->add_node_and_edge(
                        std::make_shared<op::Constant>(
                            element::i64, Shape{}, std::vector<int64_t>{tensor_size}),
                        graph::GNodeVector({}));
                    size_gnode->set_name(node_proto.name());
                    return {{node_proto.output(0), size_gnode}};
                }
            } // namespace set_1

            namespace set_13
            {
                using set_1::TranslateSizeOp;
            }
        } // namespace onnx_import
    }     // namespace frontend
} // namespace nnfusion
