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
#include "nnfusion/core/graph/util/autobroadcast.hpp"
#include "nnfusion/frontend/util/evaluator.hpp"

namespace nnfusion
{
    namespace frontend
    {
        namespace onnx_import
        {
            namespace set_1
            {
                NamedNodeVector TranslateCumSumOp(const onnx::NodeProto& node_proto,
                                                  const NodeMap& all_ng_nodes,
                                                  std::shared_ptr<nnfusion::graph::Graph> m_graph)
                {
                    auto input_indexes = GetAllInputIndex(all_ng_nodes, node_proto);
                    NNFUSION_CHECK(input_indexes.size() == 2);

                    auto x = input_indexes.at(0);
                    Node node(node_proto);
                    int exclusive = node.get_attribute_value<int64_t>("exclusive", 0);
                    int reverse = node.get_attribute_value<int64_t>("reverse", 0);

                    std::vector<int> axes;
                    NNFUSION_CHECK(GetValueFromNGraphOp(input_indexes[1].gnode, &axes));

                    nnfusion::op::OpConfig::any myConfig;
                    myConfig["axis"] = axes[0];
                    myConfig["exclusive"] = exclusive;
                    myConfig["reverse"] = reverse;
                    auto generic_op = std::make_shared<nnfusion::op::GenericOp>(
                        node_proto.name(), "CumSum", myConfig);
                    auto generic_gnode = m_graph->add_node_and_edge(generic_op, {input_indexes[0]});

                    return {{node_proto.output(0), generic_gnode}};
                }

            } // namespace set_1

        } //namespace onnx_import

    } // namespace frontend

} // namespace nnfusion
