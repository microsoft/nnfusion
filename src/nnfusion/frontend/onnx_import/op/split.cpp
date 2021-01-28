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

#include "split.hpp"
#include "nnfusion/frontend/onnx_import/util/util.hpp"

namespace nnfusion
{
    namespace frontend
    {
        namespace onnx_import
        {
            namespace set_1
            {
                NamedNodeVector TranslateSplitOp(const onnx::NodeProto& node_proto,
                                                 const NodeMap& all_ng_nodes,
                                                 std::shared_ptr<nnfusion::graph::Graph> m_graph)
                {
                    auto input = GetInputIndex(all_ng_nodes, node_proto, 0);
                    Shape input_shape = input.get_shape();

                    Node node(node_proto);
                    auto axis = node.get_attribute_value<int64_t>("axis", 0);
                    size_t axis_to_split;
                    if (axis < 0)
                    {
                        axis_to_split = input_shape.size() + axis;
                    }
                    else
                    {
                        axis_to_split = axis;
                    }
                    NNFUSION_CHECK(axis_to_split < input_shape.size());
                    std::vector<int64_t> splits;
                    try
                    {
                        splits = node.get_attribute_value<std::vector<int64_t>>("split");
                    }
                    catch (const std::exception&)
                    {
                        auto num_splits = node.get_output_names().size();
                        auto axis_length = input_shape.at(axis);
                        NNFUSION_CHECK(axis_length % num_splits == 0);
                        splits.assign(num_splits, axis_length / num_splits);
                    }

                    int cursor = 0;
                    NamedNodeVector ret;

                    std::vector<size_t> lower(input_shape.size(), 0);
                    std::vector<size_t> upper(input_shape);
                    if (splits.size() != 1)
                    {
                        for (int i = 0; i < splits.size(); ++i)
                        {
                            lower[axis_to_split] = cursor;
                            cursor += splits[i];
                            upper[axis_to_split] = cursor;
                            auto split_op = std::make_shared<op::Slice>(lower, upper);
                            //ng_split_op->set_name(node.name());

                            auto split_gnode = m_graph->add_node_and_edge(split_op, {input});
                            ret.push_back({node_proto.output(i), split_gnode});
                        }
                    }
                    else
                    {
                        ret.push_back({node_proto.output(0), input});
                    }

                    return ret;
                }
            } // namespace set_1
        }     // namespace onnx_import
    }         // namespace frontend
} // namespace nnfusion