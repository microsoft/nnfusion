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

#include "backward_registry.hpp"
#include "nnfusion/core/graph/util/autobroadcast.hpp"

REGISTER_BACKWARD_TRANSLATOR(Convolution)
    .translator([](std::shared_ptr<GNode> forward_node,
                   const GNodeIndexVector& outputs_grad,
                   std::shared_ptr<nnfusion::graph::Graph> graph) -> GNodeIndexVector {
        // input0: x, input1: filter
        NNFUSION_CHECK(outputs_grad.size() == 1) << "Convolution have only 1 output, but "
                                                 << outputs_grad.size() << " outputs_grad provided";
        auto conv = std::dynamic_pointer_cast<op::Convolution>(forward_node->get_op_ptr());
        auto window_dilation_strides = conv->get_window_dilation_strides();
        auto window_movement_strides = conv->get_window_movement_strides();
        auto data_dilation_strides = conv->get_data_dilation_strides();
        auto padding_below_diff = conv->get_padding_below();
        auto padding_above_diff = conv->get_padding_above();
        auto data_format = conv->get_data_format();
        NNFUSION_CHECK(data_format == "NCHW" || data_format == "NCW");
        auto x_shape = forward_node->get_input_shape(0);
        auto filter_shape = forward_node->get_input_shape(1);
        nnfusion::Shape kernel_shape;
        nnfusion::op::OpConfig::any conv_grad_data_config, conv_grad_filter_config;
        if (data_format == "NCW")
        {
            kernel_shape = {filter_shape[2]};
        }
        else
        {
            kernel_shape = {filter_shape[2], filter_shape[3]};
        }

        conv_grad_data_config["padding_above"] = padding_above_diff;
        conv_grad_data_config["padding_below"] = padding_below_diff;
        conv_grad_data_config["data_format"] = data_format;
        conv_grad_data_config["strides"] = window_movement_strides;
        conv_grad_data_config["dilations"] = window_dilation_strides;
        conv_grad_data_config["data_dilations"] = data_dilation_strides;
        conv_grad_data_config["kernel_shape"] = kernel_shape;
        conv_grad_data_config["in_channel"] = x_shape[1];

        conv_grad_filter_config["padding_above"] = padding_above_diff;
        conv_grad_filter_config["padding_below"] = padding_below_diff;
        conv_grad_filter_config["data_format"] = data_format;
        conv_grad_filter_config["strides"] = window_movement_strides;
        conv_grad_filter_config["dilations"] = window_dilation_strides;
        conv_grad_filter_config["data_dilations"] = data_dilation_strides;
        conv_grad_filter_config["kernel_shape"] = kernel_shape;

        auto conv_grad_data_op = std::make_shared<nnfusion::op::GenericOp>(
            forward_node->get_name() + "_x_grad", "ConvolutionGradData", conv_grad_data_config);

        auto conv_grad_filter_op = std::make_shared<nnfusion::op::GenericOp>(
            forward_node->get_name() + "_w_grad", "ConvolutionGradFilter", conv_grad_filter_config);
        auto conv_grad_data = graph->add_node_and_edge(
            conv_grad_data_op, {get_node_input(forward_node, 1), outputs_grad[0]});
        auto conv_grad_filter = graph->add_node_and_edge(
            conv_grad_filter_op, {get_node_input(forward_node, 0), outputs_grad[0]});
        return GNodeIndexVector{GNodeIndex{conv_grad_data, 0}, GNodeIndex{conv_grad_filter, 0}};
    });