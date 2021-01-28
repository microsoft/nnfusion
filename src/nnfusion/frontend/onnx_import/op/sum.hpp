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
#include "nnfusion/core/operators/generic_op/generic_op.hpp"

namespace nnfusion
{
    namespace frontend
    {
        namespace onnx_import
        {
            namespace set_1
            {
                NamedNodeVector TranslateSumOp(const onnx::NodeProto& node_proto,
                                               const NodeMap& all_ng_nodes,
                                               std::shared_ptr<nnfusion::graph::Graph> m_graph)
                {
                    auto input_indexes = GetAllInputIndex(all_ng_nodes, node_proto);

                    nnfusion::op::OpConfig::any myConfig;

                    // Since Ngraph doesn't have AddN, so we use GenericOp to
                    // represent the AddN.
                    auto generic_op = std::make_shared<nnfusion::op::GenericOp>(
                        node_proto.output(0), "AddN", myConfig);

                    auto generic_gnode = m_graph->add_node_and_edge(generic_op, input_indexes);
                    // Return the node vecoter, this is one tf-node to one nnfusion-node case,
                    // if your code converts one tf-node into several nnfusion-nodes, you can
                    // refer BiasAdd, which is converted to Broadcast and Add;
                    NamedNodeVector ret{{node_proto.output(0), generic_gnode}};
                    return ret;
                }

            } // namespace set_1

        } //namespace op

    } // namespace onnx_import

} // namespace nnfusion
