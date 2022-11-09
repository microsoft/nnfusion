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

#include "shape.hpp"

namespace nnfusion
{
    namespace frontend
    {
        namespace onnx_import
        {
            namespace set_1
            {
                NamedNodeVector TranslateShapeOp(const onnx::NodeProto& node_proto,
                                                 const NodeMap& all_ng_nodes,
                                                 std::shared_ptr<nnfusion::graph::Graph> m_graph)
                {
                    auto data = GetInputIndex(all_ng_nodes, node_proto, 0);
                    auto data_shape = data.get_shape();
                    auto op = std::make_shared<op::Constant>(
                        nnfusion::element::i64, Shape{data_shape.size()}, data_shape);
                    op->set_name(node_proto.output(0));
                    auto gnode = m_graph->add_node_and_edge(op, nnfusion::graph::GNodeVector{});
                    NamedNodeVector ret{{node_proto.output(0), gnode}};
                    return ret;
                }

            } // namespace set_1

            namespace set_15
            {
                NamedNodeVector TranslateShapeOp(const onnx::NodeProto& node_proto,
                                                 const NodeMap& all_ng_nodes,
                                                 std::shared_ptr<nnfusion::graph::Graph> m_graph)
                {
                    auto data = GetInputIndex(all_ng_nodes, node_proto, 0);
                    auto data_shape = data.get_shape();

                    Node node(node_proto);
                    int64_t rank = data_shape.size();
                    int64_t start = node.get_attribute_value<int64_t>("start", 0);
                    if(start < 0)
                        start += rank;
                    start = (start < 0) ? 0 : (start > rank) ? rank : start;
                    int64_t end = node.get_attribute_value<int64_t>("end", rank);
                    if (end < 0)
                        end += rank;
                    end = (end < 0) ? 0 : (end > rank) ? rank : end;
                    int64_t dim_value = (end - start) < 0 ? 0 : (end - start);

                    auto op = std::make_shared<op::Constant>(
                        nnfusion::element::i64, Shape{dim_value}, Shape(data_shape.begin(), data_shape.begin() + dim_value));
                    op->set_name(node_proto.output(0));
                    auto gnode = m_graph->add_node_and_edge(op, nnfusion::graph::GNodeVector{});
                    NamedNodeVector ret{{node_proto.output(0), gnode}};
                    return ret;
                }

            } // namespace set_15

        } //namespace op

    } // namespace onnx_import

} // namespace nnfusion
