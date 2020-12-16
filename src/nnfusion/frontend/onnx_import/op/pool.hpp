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

namespace nnfusion
{
    namespace frontend
    {
        namespace onnx_import
        {
            namespace set_1
            {
                template <typename T>
                inline NamedNodeVector
                    TranslatePoolOp(const onnx::NodeProto& node_proto,
                                    const NodeMap& all_ng_nodes,
                                    std::shared_ptr<nnfusion::graph::Graph> m_graph)
                {
                    auto input_gnode = GetInputNode(all_ng_nodes, node_proto, 0);
                    Node node(node_proto);

                    // Parse ONNX op attributes
                    Shape kernel_shape;
                    if (node_proto.op_type().find("Global") != std::string::npos)
                    {
                        kernel_shape = input_gnode->get_shape();
                        // Remove N and C dimensions and leave only spatial dims.
                        kernel_shape.erase(std::begin(kernel_shape),
                                           std::next(std::begin(kernel_shape), 2));
                    }
                    else
                    {
                        kernel_shape = get_kernel_shape(node, input_gnode);
                    }

                    auto strides = get_strides(node, input_gnode);
                    auto dilations = get_dilations(node, input_gnode);
                    auto paddings = get_pads(node, input_gnode);

                    bool count_include_pad =
                        node.get_attribute_value<int64_t>("count_include_pad", 0);

                    // Convert padding from CoordinateDiff to Shape objects
                    const CoordinateDiff& padding_above{paddings.first};
                    const CoordinateDiff& padding_below{paddings.second};
                    Shape padding_below_shape{std::begin(padding_below), std::end(padding_below)};
                    Shape padding_above_shape{std::begin(padding_above), std::end(padding_above)};

                    std::shared_ptr<op::Op> pool_op;
                    if (count_include_pad)
                    {
                        pool_op = std::make_shared<op::AvgPool>(kernel_shape,
                                                                strides,
                                                                padding_below_shape,
                                                                padding_above_shape,
                                                                count_include_pad);
                    }
                    else
                    {
                        pool_op = std::make_shared<T>(
                            kernel_shape, strides, padding_below_shape, padding_above_shape);
                    }

                    pool_op->set_name(node_proto.output(0));
                    auto pool_gnode = m_graph->add_node_and_edge(pool_op, {input_gnode});

                    NamedNodeVector ret{{node_proto.output(0), pool_gnode}};
                    return ret;
                }

            } // namespace set_1
        }     // namespace onnx_import
    }         // namespace frontend
} // namespace nnfusion
