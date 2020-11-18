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
#include "../util/reshape.hpp"
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
                NamedNodeVector
                    TranslateIndexReductionOp(const onnx::NodeProto& node_proto,
                                              const NodeMap& all_ng_nodes,
                                              std::shared_ptr<nnfusion::graph::Graph> m_graph)
                {
                    auto input_gnode = GetInputNode(all_ng_nodes, node_proto, 0);

                    Node node(node_proto);
                    auto axis = node.get_attribute_value<int64_t>("axis", 0);
                    auto keepdims = node.get_attribute_value<int64_t>("keepdims", 1);

                    auto reduce_index_op = std::make_shared<T>(axis, element::i64);
                    std::shared_ptr<graph::GNode> reduce_index_gnode;

                    if (keepdims != 0)
                    {
                        // WORKAROUND FOR PROBLEMS WITH RESHAPE ON i64 @TODO: remove
                        reduce_index_gnode =
                            m_graph->add_node_and_edge(reduce_index_op, {input_gnode});

                        auto convert_op = std::make_shared<op::Convert>(element::f32);
                        auto convert_gnode =
                            m_graph->add_node_and_edge(convert_op, {reduce_index_gnode});

                        auto output_shape = input_gnode->get_shape();
                        output_shape.at(axis) = 1;
                        auto reshape_op = std::make_shared<op::Reshape>(
                            reshape::get_default_axis_vector(
                                reduce_index_gnode->get_shape().size()),
                            Shape{output_shape});
                        auto reshape_gnode =
                            m_graph->add_node_and_edge(reshape_op, {convert_gnode});

                        // WORKAROUND FOR PROBLEMS WITH RESHAPE ON i64 @TODO: remove
                        auto reconvert_op = std::make_shared<op::Convert>(element::i64);
                        reconvert_op->set_name(node_proto.output(0));
                        reduce_index_gnode =
                            m_graph->add_node_and_edge(reconvert_op, {reshape_gnode});
                    }
                    else
                    {
                        reduce_index_op->set_name(node_proto.output(0));
                        reduce_index_gnode =
                            m_graph->add_node_and_edge(reduce_index_op, {input_gnode});
                    }

                    NamedNodeVector ret{{node_proto.output(0), reduce_index_gnode}};
                    return ret;
                }

            } // namespace set_1
        }     // namespace onnx_import
    }         // namespace frontend
} // namespace ngraph
