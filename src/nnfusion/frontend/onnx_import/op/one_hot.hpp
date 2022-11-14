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
#include "nnfusion/frontend/util/evaluator.hpp"

namespace nnfusion
{
    namespace frontend
    {
        namespace onnx_import
        {
            namespace set_9
            {
                NamedNodeVector TranslateOneHotOp(const onnx::NodeProto& node_proto,
                                                  const NodeMap& all_ng_nodes,
                                                  std::shared_ptr<nnfusion::graph::Graph> m_graph)
                {
                    GNodeIndexVector input_indexes = GetAllInputIndex(all_ng_nodes, node_proto);
                    auto indices = input_indexes[0];
                    auto depth = input_indexes[1];
                    auto off_on_values = input_indexes[2];

                    std::vector<int64> depth_vec;
                    NNFUSION_CHECK(GetValueFromNGraphOp(depth.gnode, &depth_vec))
                        << "The input of depth needs to be constant values.";
                    NNFUSION_CHECK(depth_vec.size() == 1);

                    // we currently only handle numeric data types by converting them to double first,
                    // then to real target type in codegen
                    std::vector<double> off_on_values_vec;
                    NNFUSION_CHECK(GetValueFromNGraphOp(off_on_values.gnode, &off_on_values_vec))
                        << "The input of off_on_values needs to be constant values.";
                    NNFUSION_CHECK(off_on_values_vec.size() == 2);

                    Node node(node_proto);
                    auto axis = node.get_attribute_value<int64>("axis", -1);

                    axis += axis < 0 ? (indices.get_shape().size() + 1) : 0;

                    auto type_str = off_on_values.get_element_type().c_type_string();

                    nnfusion::op::OpConfig::any myConfig;
                    myConfig["axis"] = axis;
                    myConfig["depth"] = depth_vec[0];
                    myConfig["off_value"] = off_on_values_vec[0];
                    myConfig["on_value"] = off_on_values_vec[1];
                    myConfig["T"] = type_str;

                    auto generic_op = std::make_shared<nnfusion::op::GenericOp>(
                        node_proto.output(0), "OneHot", myConfig);
                    auto generic_gnode = m_graph->add_node_and_edge(generic_op, {indices});

                    return {{node_proto.output(0), generic_gnode}};
                }
            } // namespace set_9
            namespace set_11
            {
                using set_9::TranslateOneHotOp;
            }
        } //namespace onnx_import
    }     // namespace frontend
} // namespace  nnfusion
