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

#include "conv_trans.hpp"
#include "conv.hpp"
#include "nnfusion/core/operators/generic_op/generic_op.hpp"

namespace nnfusion
{
    namespace frontend
    {
        namespace onnx_import
        {
            namespace set_1
            {
                std::unordered_map<std::string, std::vector<int64_t>>
                    extract_conv_trans_attrs(nnfusion::frontend::onnx_import::Node node,
                                             const Shape& x_shape,
                                             const Shape& w_shape)
                {
                    std::unordered_map<std::string, std::vector<int64_t>> conv_trans_attrs;
                    conv_trans_attrs["output_shape"] =
                        node.get_attribute_value<std::vector<int64_t>>(
                            "output_shape", std::vector<int64_t>(x_shape.size(), 0));
                    conv_trans_attrs["output_padding"] =
                        node.get_attribute_value<std::vector<int64_t>>(
                            "output_padding", std::vector<int64_t>((w_shape.size() - 2) * 2, 0));
                    return conv_trans_attrs;
                }

                NamedNodeVector
                    TranslateConvTransposeOp(const onnx::NodeProto& node_proto,
                                             const NodeMap& all_ng_nodes,
                                             std::shared_ptr<nnfusion::graph::Graph> m_graph)
                {
                    auto input_indices = GetAllInputIndex(all_ng_nodes, node_proto);
                    NNFUSION_CHECK(input_indices.size() <= 3);
                    auto x_gnode = input_indices[0];
                    auto w_gnode = input_indices[1];

                    Node node(node_proto);
                    int64_t groups = node.get_attribute_value<int64_t>("group", 1);
                    NNFUSION_CHECK(groups == 1) << "'group' attribute is not supported now!";

                    auto conv_attrs = extract_conv_attrs(node, w_gnode.get_shape());
                    auto conv_trans_attrs =
                        extract_conv_trans_attrs(node, x_gnode.get_shape(), w_gnode.get_shape());
                    auto conv_data_format = assign_data_format(x_gnode.get_shape());

                    nnfusion::op::OpConfig::any op_config;
                    op_config["data_format"] = conv_data_format;
                    op_config["kernel_shape"] =
                        Shape(conv_attrs["kernel_shape"].begin(), conv_attrs["kernel_shape"].end());
                    op_config["strides"] =
                        Strides(conv_attrs["strides"].begin(), conv_attrs["strides"].end());
                    op_config["dilations"] =
                        Strides(conv_attrs["dilations"].begin(), conv_attrs["dilations"].end());
                    op_config["padding_above"] =
                        CoordinateDiff(conv_attrs["pads"].begin(),
                                       conv_attrs["pads"].begin() + conv_attrs["pads"].size() / 2);
                    op_config["padding_below"] =
                        CoordinateDiff(conv_attrs["pads"].begin() + conv_attrs["pads"].size() / 2,
                                       conv_attrs["pads"].end());

                    auto node_name = node_proto.output(0);
                    auto conv_trans_op = std::make_shared<op::GenericOp>(
                        node_name + ".conv", "ConvTranspose", op_config);
                    auto conv_trans_gnode =
                        m_graph->add_node_and_edge(conv_trans_op, {x_gnode, w_gnode});

                    if (input_indices.size() == 3)
                        conv_trans_gnode =
                            attach_bias_gnode(input_indices[2], conv_trans_gnode, m_graph);

                    return {{node_proto.output(0), GNodeIndex(conv_trans_gnode)}};
                }
            } // namespace set_1
        }     // namespace onnx_import
    }         // namespace frontend
} // namespace nnfusion