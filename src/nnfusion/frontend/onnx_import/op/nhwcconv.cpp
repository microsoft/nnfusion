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

#include "nhwcconv.hpp"
#include <unordered_map>
#include "conv.hpp"
#include "nnfusion/core/operators/generic_op/generic_op.hpp"
#include "nnfusion/frontend/onnx_import/util/broadcasting.hpp"

namespace nnfusion
{
    namespace frontend
    {
        namespace onnx_import
        {
            namespace set_1
            {
                NamedNodeVector TranslateNhwcConvOp(const onnx::NodeProto& node_proto,
                                                    const NodeMap& all_ng_nodes,
                                                    std::shared_ptr<nnfusion::graph::Graph> m_graph)
                {
                    auto input_indexes = GetAllInputIndex(all_ng_nodes, node_proto);
                    NNFUSION_CHECK(input_indexes.size() <= 3);

                    auto data = input_indexes[0];
                    auto filters = input_indexes[1];
                    auto data_shape = data.get_shape();
                    auto filters_shape = filters.get_shape();

                    Node node(node_proto);
                    int64_t groups = node.get_attribute_value<int64_t>("group", 1);
                    NNFUSION_CHECK(groups >= 0 && groups <= data_shape.at(1) &&
                                   groups <= filters_shape.at(0))
                        << "incorrect value of 'group' attribute: " << groups;
                    auto conv_attrs = extract_conv_attrs(node, filters_shape);

                    Shape kernel_shape = Shape({filters_shape[1], filters_shape[2]});
                    Strides strides =
                        Strides(conv_attrs["strides"].begin(), conv_attrs["strides"].end());
                    Strides dilations =
                        Strides(conv_attrs["dilations"].begin(), conv_attrs["dilations"].end());
                    Strides pads =
                        Strides(conv_attrs["pads"].begin() + conv_attrs["pads"].size() / 2,
                                conv_attrs["pads"].end());

                    op::OpConfig::any config;
                    config["strides"] = strides;
                    config["dilations"] = dilations;
                    config["pads"] = pads;
                    auto conv_op =
                        std::make_shared<op::GenericOp>(node_proto.name(), "NhwcConv", config);
                    std::shared_ptr<nnfusion::graph::GNode> conv_node =
                        m_graph->add_node_and_edge(conv_op, {data, filters});
                    // add bias
                    if (input_indexes.size() == 3)
                        conv_node = attach_bias_gnode(input_indexes[2], conv_node, m_graph, 3);

                    return {{node_proto.output(0), GNodeIndex(conv_node)}};
                }
            } // namespace set_1

        } //namespace onnx_import

    } // namespace frontend

} // namespace nnfusion
