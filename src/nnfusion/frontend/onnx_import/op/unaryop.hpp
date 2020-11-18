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
                NamedNodeVector TranslateUnaryOp(const onnx::NodeProto& node_proto,
                                                 const NodeMap& all_ng_nodes,
                                                 std::shared_ptr<nnfusion::graph::Graph> m_graph)
                {
                    auto input_index = GetInputIndex(all_ng_nodes, node_proto, 0);
                    auto op = std::make_shared<T>();
                    NNFUSION_CHECK(node_proto.output_size() == 1)
                        << "Unary op should only has one output.";
                    op->set_name(node_proto.output(0));
                    auto gnode = m_graph->add_node_and_edge(op, {input_index});
                    NamedNodeVector ret{{node_proto.output(0), gnode}};
                    return ret;
                }

            } // namespace set_1

            namespace set_9
            {
                template <typename T>
                NamedNodeVector TranslateUnaryOp(const onnx::NodeProto& node_proto,
                                                 const NodeMap& all_ng_nodes,
                                                 std::shared_ptr<nnfusion::graph::Graph> m_graph)
                {
                    auto input_index = GetInputIndex(all_ng_nodes, node_proto, 0);
                    auto op = std::make_shared<T>();
                    NNFUSION_CHECK(node_proto.output_size() == 1)
                        << "Unary op should only has one output.";
                    op->set_name(node_proto.output(0));
                    auto gnode = m_graph->add_node_and_edge(op, {input_index});
                    NamedNodeVector ret{{node_proto.output(0), gnode}};
                    return ret;
                }

            } // namespace set_1
        }     // namespace onnx_import
    }         // namespace frontend
} // namespace ngraph
