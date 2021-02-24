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

#include "resize.hpp"
#include "nnfusion/core/operators/generic_op/generic_op.hpp"

namespace nnfusion
{
    namespace frontend
    {
        namespace onnx_import
        {
            namespace set_1
            {
                NamedNodeVector TranslateResizeOp(const onnx::NodeProto& node_proto,
                                                  const NodeMap& all_ng_nodes,
                                                  std::shared_ptr<nnfusion::graph::Graph> m_graph)
                {
                    auto input_indices = GetAllInputIndex(all_ng_nodes, node_proto);

                    auto x_gnode = input_indices[0];
                    auto scales_gnode = input_indices[2];

                    Node node(node_proto);
                    std::string mode = node.get_attribute_value<std::string>("mode", "nearest");
                    std::transform(mode.begin(), mode.end(), mode.begin(), ::toupper);
                    nnfusion::op::OpConfig::any op_config;
                    op_config["method"] = mode;

                    auto node_name = node_proto.output(0);
                    auto resize_op =
                        std::make_shared<nnfusion::op::GenericOp>(node_name, "Resize", op_config);
                    auto resize_gnode =
                        m_graph->add_node_and_edge(resize_op, {x_gnode, scales_gnode});
                    return NamedNodeVector{{node_proto.output(0), resize_gnode}};
                }
            } // namespace set_1

        } //namespace onnx_import

    } // namespace frontend

} // namespace nnfusion