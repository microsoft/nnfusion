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

#include "conv.hpp"

#include <unordered_map>
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
                std::unordered_map<std::string, std::vector<int64_t>>
                    extract_conv_attrs(nnfusion::frontend::onnx_import::Node node,
                                       const Shape& filters_shape)
                {
                    std::unordered_map<std::string, std::vector<int64_t>> conv_attrs;
                    conv_attrs["kernel_shape"] = node.get_attribute_value<std::vector<int64_t>>(
                        "kernel_shape",
                        std::vector<int64_t>(filters_shape.begin() + 2, filters_shape.end()));
                    conv_attrs["strides"] = node.get_attribute_value<std::vector<int64_t>>(
                        "strides", std::vector<int64_t>(conv_attrs["kernel_shape"].size(), 1));
                    conv_attrs["dilations"] = node.get_attribute_value<std::vector<int64_t>>(
                        "dilations", std::vector<int64_t>(conv_attrs["kernel_shape"].size(), 1));
                    conv_attrs["pads"] = node.get_attribute_value<std::vector<int64_t>>(
                        "pads", std::vector<int64_t>(conv_attrs["kernel_shape"].size() * 2, 0));

                    std::string auto_pad =
                        node.get_attribute_value<std::string>("auto_pad", std::string("NOTSET"));

                    return conv_attrs;
                }

                std::shared_ptr<nnfusion::graph::GNode>
                    attach_bias_gnode(nnfusion::frontend::onnx_import::GNodeIndex bias_index,
                                      std::shared_ptr<nnfusion::graph::GNode> conv_node,
                                      std::shared_ptr<nnfusion::graph::Graph> m_graph)
                {
                    auto bc_op = std::make_shared<op::Broadcast>(
                        conv_node->get_shape(),
                        nnfusion::frontend::onnx_import::calculate_broadcast_axes(
                            conv_node->get_shape(), bias_index.get_shape(), 1));
                    auto broadcasted_bias = m_graph->add_node_and_edge(bc_op, {bias_index});
                    auto bias_node = m_graph->add_node_and_edge(std::make_shared<op::Add>(),
                                                                {conv_node, broadcasted_bias});
                    return bias_node;
                }

                std::string assign_data_format(nnfusion::Shape data_shape)
                {
                    std::string conv_data_format;
                    if (data_shape.size() == 3)
                    {
                        conv_data_format = "NCW";
                    }
                    else if (data_shape.size() == 4)
                    {
                        conv_data_format = "NCHW";
                    }
                    else if (data_shape.size() == 5)
                    {
                        conv_data_format = "NCDHW";
                    }
                    else
                    {
                        NNFUSION_CHECK_FAIL() << "Convolution with dimensions of "
                                              << data_shape.size() << " not implemented yet.";
                    }
                    return conv_data_format;
                }

                NamedNodeVector TranslateConvOp(const onnx::NodeProto& node_proto,
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

                    Shape kernel_shape =
                        Shape(conv_attrs["kernel_shape"].begin(), conv_attrs["kernel_shape"].end());
                    Strides strides =
                        Strides(conv_attrs["strides"].begin(), conv_attrs["strides"].end());
                    Strides dilations =
                        Strides(conv_attrs["dilations"].begin(), conv_attrs["dilations"].end());

                    CoordinateDiff padding_above, padding_below;
                    size_t spatial_len = kernel_shape.size();
                    std::string auto_pad =
                        node.get_attribute_value<std::string>("auto_pad", std::string("NOTSET"));
                    if (auto_pad == "SAME_UPPER" || auto_pad == "SAME_LOWER")
                    {
                        Shape data_spatial_shape = Shape(data_shape.begin() + 2, data_shape.end());
                        for (size_t i = 0; i < spatial_len; i++)
                        {
                            size_t h_in = data_spatial_shape[i];
                            size_t s = strides[i];
                            size_t h_out = ceil((float)h_in / (float)s);
                            size_t kh = kernel_shape[i];
                            size_t d = dilations[i];
                            size_t p = (h_out - 1) * s + d * (kh - 1) + 1 - h_in;
                            if (p % 2 == 0)
                            {
                                size_t p_i = p / 2;
                                padding_above.push_back(p_i);
                                padding_below.push_back(p_i);
                            }
                            else
                            {
                                size_t p_i = floor((float)p / 2);
                                if (auto_pad == "SAME_UPPER")
                                {
                                    padding_below.push_back(p_i);
                                    padding_above.push_back(p_i + 1);
                                }
                                else
                                {
                                    padding_below.push_back(p_i + 1);
                                    padding_above.push_back(p_i);
                                }
                            }
                        }
                    }
                    else
                    {
                        padding_above = CoordinateDiff(conv_attrs["pads"].begin(),
                                                       conv_attrs["pads"].begin() +
                                                           conv_attrs["pads"].size() / 2);
                        padding_below = CoordinateDiff(conv_attrs["pads"].begin() +
                                                           conv_attrs["pads"].size() / 2,
                                                       conv_attrs["pads"].end());
                    }
                    std::string conv_data_format = assign_data_format(data_shape);

                    // if (padding_above != padding_below)
                    // {
                    //     int rank = data_shape.size();
                    //     Shape padding_above_temp(rank, 0);
                    //     Shape padding_below_temp(rank, 0);
                    //     Shape padding_interior_temp(rank, 0);

                    //     for (int i = 0; i < spatial_len; i++)
                    //     {
                    //         padding_above_temp[i + 2] = padding_above[i];
                    //         padding_below_temp[i + 2] = padding_below[i];
                    //         padding_above[i] = 0;
                    //         padding_below[i] = 0;
                    //     }

                    //     auto pad_val_op =
                    //         std::make_shared<op::Constant>(data.get_element_type(),
                    //                                        nnfusion::Shape{},
                    //                                        std::vector<std::string>{"0"});
                    //     auto pad_val_gnode =
                    //         m_graph->add_node_and_edge(pad_val_op, GNodeIndexVector{});

                    //     auto pad_op = std::make_shared<op::Pad>(
                    //         padding_below_temp, padding_above_temp, padding_interior_temp);

                    //     auto pad_gnode =
                    //         m_graph->add_node_and_edge(pad_op, {data, GNodeIndex(pad_val_gnode)});
                    //     data = GNodeIndex(pad_gnode, 0);
                    // }

                    std::shared_ptr<nnfusion::graph::GNode> conv_node = nullptr;
                    if (groups == 1)
                    {
                        auto conv_op = std::make_shared<op::Convolution>(
                            strides, dilations, padding_below, padding_above, conv_data_format);
                        conv_node = m_graph->add_node_and_edge(conv_op, {data, filters});
                    }
                    else if (conv_data_format == "NCHW")
                    {
                        // split data and filters for group conv
                        std::size_t n_data_channels{data_shape.at(1)};
                        std::size_t n_filters_channels{filters_shape.at(0)};
                        if (n_data_channels == groups)
                        {
                            // depthwise convolution
                            auto filter_gnode = GetInputNode(all_ng_nodes, node_proto, 1);
                            if (data_shape.size() == 3)
                            {
                                // DWCONV 1D
                                nnfusion::op::OpConfig::any myConfig;
                                myConfig["data_format"] = "NCW";
                                myConfig["strides"] = strides;
                                myConfig["dilations"] = dilations;
                                myConfig["padding_before"] = padding_below;
                                myConfig["padding_after"] = padding_above;
                                if (2 * padding_below[0] - dilations[0] * (filters_shape[2] - 1) ==
                                    0)
                                {
                                    myConfig["padding_type"] = "SAME";
                                }
                                else if (padding_below[0] == 0)
                                {
                                    myConfig["padding_type"] = "VALID";
                                }
                                else
                                {
                                    NNFUSION_CHECK_FAIL()
                                        << "Currently only support SAME and VALID "
                                           "padding in DepthwiseConv2dNative";
                                }
                                auto conv_op = std::make_shared<nnfusion::op::GenericOp>(
                                    node_proto.name(), "DepthwiseConv1dNative", myConfig);
                                conv_node = m_graph->add_node_and_edge(
                                    conv_op, {data, GNodeIndex{filter_gnode, 0}});
                            }
                            else
                            {
                                auto reshape_filter_op = std::make_shared<nnfusion::op::Reshape>(
                                    nnfusion::AxisVector{2, 3, 0, 1},
                                    nnfusion::Shape({filters_shape[2],
                                                     filters_shape[3],
                                                     filters_shape[0],
                                                     filters_shape[1]}));
                                reshape_filter_op->set_name(filter_gnode->get_name() +
                                                            "_filters_reshape");
                                auto reshape_filter_gnode =
                                    m_graph->add_node_and_edge(reshape_filter_op, {filter_gnode});

                                size_t depth_multiplier = 1;
                                nnfusion::op::OpConfig::any myConfig;
                                myConfig["data_format"] = "NCHW";
                                myConfig["strides"] = strides;
                                myConfig["dilations"] = dilations;
                                myConfig["padding_before"] = padding_below;
                                myConfig["padding_after"] = padding_above;

                                if ((2 * padding_below[0] - dilations[0] * (filters_shape[2] - 1) ==
                                     0) &&
                                    (2 * padding_below[1] - dilations[1] * (filters_shape[3] - 1) ==
                                     0))
                                {
                                    myConfig["padding_type"] = "SAME";
                                }
                                else if (padding_below[0] == 0 && padding_below[1] == 0)
                                {
                                    myConfig["padding_type"] = "VALID";
                                }
                                else
                                {
                                    NNFUSION_CHECK_FAIL()
                                        << "Currently only support SAME and VALID "
                                           "padding in DepthwiseConv2dNative";
                                }

                                auto conv_op = std::make_shared<nnfusion::op::GenericOp>(
                                    node_proto.name(), "DepthwiseConv2dNative", myConfig);
                                conv_node = m_graph->add_node_and_edge(
                                    conv_op, {data, GNodeIndex{reshape_filter_gnode, 0}});
                            }
                        }
                        else
                        {
                            NNFUSION_CHECK(n_data_channels % groups == 0 &&
                                           n_filters_channels & groups == 0);
                            std::size_t data_group_size{n_data_channels / groups};
                            std::size_t filters_group_size{n_filters_channels / groups};

                            std::vector<std::size_t> data_lower_bounds(data_shape.size(), 0);
                            std::vector<std::size_t> data_upper_bounds{data_shape};
                            std::vector<std::size_t> filters_lower_bounds(filters_shape.size(), 0);
                            std::vector<std::size_t> filters_upper_bounds{filters_shape};

                            std::vector<std::shared_ptr<nnfusion::graph::GNode>> convolution_nodes;
                            for (std::size_t group = 0; group < groups; ++group)
                            {
                                // slice data
                                data_lower_bounds[1] = group * data_group_size;
                                data_upper_bounds[1] = (group + 1) * data_group_size;
                                auto sliced_data_op = std::make_shared<op::Slice>(
                                    data_lower_bounds, data_upper_bounds);
                                auto sliced_data =
                                    m_graph->add_node_and_edge(sliced_data_op, {data});
                                // slice filters
                                filters_lower_bounds[0] = group * filters_group_size;
                                filters_upper_bounds[0] = (group + 1) * filters_group_size;
                                auto sliced_filters_op = std::make_shared<op::Slice>(
                                    filters_lower_bounds, filters_upper_bounds);
                                auto sliced_filters =
                                    m_graph->add_node_and_edge(sliced_filters_op, {filters});

                                convolution_nodes.push_back(m_graph->add_node_and_edge(
                                    std::make_shared<op::Convolution>(strides,
                                                                      dilations,
                                                                      padding_below,
                                                                      padding_above,
                                                                      conv_data_format),
                                    {sliced_data, sliced_filters}));
                            }
                            std::size_t concatenation_axis = 1;
                            conv_node = m_graph->add_node_and_edge(
                                std::make_shared<op::Concat>(concatenation_axis),
                                convolution_nodes);
                        }
                    }
                    else
                    {
                        NNFUSION_CHECK_FAIL() << "Not support this Convolution yet.";
                    }

                    // add bias
                    if (input_indexes.size() == 3)
                        conv_node = attach_bias_gnode(input_indexes[2], conv_node, m_graph);

                    return {{node_proto.output(0), GNodeIndex(conv_node)}};
                }
            } // namespace set_1

        } //namespace onnx_import

    } // namespace frontend

} // namespace nnfusion
