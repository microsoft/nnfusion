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

#include "unsqueeze.hpp"

namespace nnfusion
{
    namespace frontend
    {
        namespace onnx_import
        {
            namespace set_1
            {
                NamedNodeVector
                    TranslateUnsqueezeOp(const onnx::NodeProto& node_proto,
                                         const NodeMap& all_ng_nodes,
                                         std::shared_ptr<nnfusion::graph::Graph> m_graph)
                {
                    auto data = GetInputIndex(all_ng_nodes, node_proto, 0);
                    auto input_shape = data.get_shape();
                    Node node(node_proto);
                    auto axes = node.get_attribute_value<std::vector<int64_t>>("axes");
                    ///\todo: check duplicate axes between neg and pos axes
                    for (auto& axis : axes)
                    {
                        axis += axis < 0 ? input_shape.size() + axes.size() : 0;
                    }

                    NNFUSION_CHECK(!axes.empty()) << "'axes' attribute is mandatory.";
                    std::sort(std::begin(axes), std::end(axes), std::less<int64_t>());

                    auto output_shape = input_shape;
                    for (auto axis : axes)
                    {
                        output_shape.insert(std::next(output_shape.begin(), axis), 1);
                    }

                    nnfusion::AxisVector ng_axis_order(input_shape.size());
                    std::iota(ng_axis_order.begin(), ng_axis_order.end(), 0);
                    auto reshape_op = std::make_shared<op::Reshape>(ng_axis_order, output_shape);
                    reshape_op->set_name(node_proto.output(0));
                    auto reshape_gnode = m_graph->add_node_and_edge(reshape_op, {data});
                    return {{node_proto.output(0), reshape_gnode}};
                }

            } // namespace set_1

        } //namespace onnx_import

    } // namespace frontend

} // namespace nnfusion
