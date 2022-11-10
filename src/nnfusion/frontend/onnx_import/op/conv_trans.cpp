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
                            "output_shape", std::vector<int64_t>(x_shape.size() - 2, 0));
                    conv_trans_attrs["output_padding"] =
                        node.get_attribute_value<std::vector<int64_t>>(
                            "output_padding", std::vector<int64_t>(w_shape.size() - 2, 0));
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
                    op_config["output_padding"] =
                        CoordinateDiff(conv_trans_attrs["output_padding"].begin(),
                                       conv_trans_attrs["output_padding"].end());
                    op_config["output_shape"] =
                        CoordinateDiff(conv_trans_attrs["output_shape"].begin(),
                                       conv_trans_attrs["output_shape"].end());

                    size_t spatial_len = op_config["kernel_shape"].size();
                    auto data_shape = x_gnode.get_shape();
                    std::string auto_pad =
                        node.get_attribute_value<std::string>("auto_pad", std::string("NOTSET"));
                    Shape data_spatial_shape = Shape(data_shape.begin() + 2, data_shape.end());
                    CoordinateDiff padding_above, padding_below;
                    if (auto_pad == "SAME_UPPER" || auto_pad == "SAME_LOWER")
                    {
                        for (size_t i = 0; i < spatial_len; i++)
                        {
                            size_t h_in = data_spatial_shape[i];
                            size_t s = op_config["strides"][i];
                            size_t h_out = h_in * s;
                            size_t kh = op_config["kernel_shape"][i];
                            size_t d = op_config["dilations"][i];
                            size_t out_p = op_config["output_padding"][i];
                            size_t p = s * (h_in - 1) + out_p + ((kh - 1) * d + 1) - out_p;
                            if (p % 2 == 0)
                            {
                                size_t p_i = p / 2;
                                padding_above.push_back(p_i);
                                padding_below.push_back(p_i);
                            }
                            else
                            {
                                size_t p_i = floor(p / 2);
                                if (auto_pad == "SAME_UPPER")
                                {
                                    padding_above.push_back(p_i);
                                    padding_below.push_back(p_i + 1);
                                }
                                else
                                {
                                    padding_above.push_back(p_i + 1);
                                    padding_below.push_back(p_i);
                                }
                            }
                        }
                        op_config["padding_above"] = padding_above;
                        op_config["padding_below"] = padding_below;
                    }

                    bool explicit_outshape = false;
                    for (auto dim : op_config["output_shape"])
                    {
                        if (dim != 0)
                        {
                            explicit_outshape = true;
                            break;
                        }
                    }

                    if (explicit_outshape)
                    {
                        for (size_t i = 0; i < spatial_len; i++)
                        {
                            size_t h_in = data_spatial_shape[i];
                            size_t s = op_config["strides"][i];
                            size_t h_out = h_in * s;
                            size_t kh = op_config["kernel_shape"][i];
                            size_t d = op_config["dilations"][i];
                            size_t out_p = op_config["output_padding"][i];
                            size_t out_s = op_config["output_shape"][i];
                            size_t total_p = s * (h_in - 1) + out_p + ((kh - 1) * d + 1) - out_s;
                            size_t p = total_p / 2;
                            if (auto_pad == "SAME_UPPER")
                            {
                                padding_below.push_back(p);
                                padding_above.push_back(total_p - p);
                            }
                            else
                            {
                                padding_below.push_back(total_p - p);
                                padding_above.push_back(p);
                            }
                        }
                    }

                    if (op_config["padding_above"] != op_config["padding_below"])
                    {
                        int rank = data_shape.size();
                        Shape padding_above_temp(rank, 0);
                        Shape padding_below_temp(rank, 0);
                        Shape padding_interior_temp(rank, 0);

                        for (int i = 0; i < spatial_len; i++)
                        {
                            padding_above_temp[i + 2] = op_config["padding_above"][i];
                            padding_below_temp[i + 2] = op_config["padding_below"][i];
                            op_config["padding_above"][i] = 0;
                            op_config["padding_below"][i] = 0;
                        }

                        auto pad_val_op =
                            std::make_shared<op::Constant>(x_gnode.get_element_type(),
                                                           nnfusion::Shape{},
                                                           std::vector<std::string>{"0"});
                        auto pad_val_gnode =
                            m_graph->add_node_and_edge(pad_val_op, GNodeIndexVector{});

                        auto pad_op = std::make_shared<op::Pad>(
                            padding_below_temp, padding_above_temp, padding_interior_temp);

                        auto pad_gnode = m_graph->add_node_and_edge(
                            pad_op, {x_gnode, GNodeIndex(pad_val_gnode)});
                        x_gnode = GNodeIndex(pad_gnode, 0);
                    }
                    NNFUSION_LOG(INFO) << op_config;
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