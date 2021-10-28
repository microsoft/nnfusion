// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "nnfusion/core/operators/generic_op/generic_op.hpp"

REGISTER_OP(ConvolutionGradFilter)
    .attr<CoordinateDiff>("padding_above")
    .attr<CoordinateDiff>("padding_below")
    .attr<std::string>("data_format")
    .attr<Strides>("strides")
    .attr<Strides>("dilations")
    .attr<Strides>("data_dilations")
    .attr<Shape>("kernel_shape")
    .infershape([](std::shared_ptr<graph::GNode> gnode) -> void {
        // input0: x, input1: dy
        // output0: dw
        NNFUSION_CHECK(gnode->get_input_size() == 2);
        auto data_shape = gnode->get_input_shape(0);
        auto dy_shape = gnode->get_input_shape(1);
        NNFUSION_CHECK(data_shape.size() == dy_shape.size() && dy_shape.size() == 4);
        NNFUSION_CHECK(data_shape[0] == dy_shape[0]);
        auto op = std::dynamic_pointer_cast<nnfusion::op::GenericOp>(gnode->get_op_ptr());
        nnfusion::Strides dilations = op->localOpConfig.getRoot()["dilations"];
        NNFUSION_CHECK(dilations[0] == dilations[1] == 1) << "Not support other dilation yet";
        nnfusion::Strides data_dilations = op->localOpConfig.getRoot()["data_dilations"];
        NNFUSION_CHECK(data_dilations[0] == data_dilations[1] == 1)
            << "Not support other data dilation yet";
        std::string data_format = op->localOpConfig.getRoot()["data_format"];
        bool is_nchw = (data_format == "NCHW");
        NNFUSION_CHECK(is_nchw) << "ConvolutionGradFilter only supports NCHW now!";
        Shape kernel_shape = op->localOpConfig.getRoot()["kernel_shape"];
        size_t kernel_size_h = kernel_shape[0];
        size_t kernel_size_w = kernel_shape[1];

        size_t N = dy_shape[0];
        size_t CO = dy_shape[1];
        size_t HO = dy_shape[2];
        size_t WO = dy_shape[3];

        Shape dx_shape{N, CO, kernel_size_h, kernel_size_w};
        gnode->set_output_type_and_shape(0, gnode->get_input_element_type(1), dx_shape);
    });
