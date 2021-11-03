// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "nnfusion/core/operators/generic_op/generic_op.hpp"

REGISTER_OP(ConvolutionGradData)
    .attr<CoordinateDiff>("padding_above")
    .attr<CoordinateDiff>("padding_below")
    .attr<std::string>("data_format")
    .attr<Strides>("strides")
    .attr<Strides>("dilations")
    .attr<Strides>("data_dilations")
    .attr<Shape>("kernel_shape")
    .attr<size_t>("in_channel")
    .infershape([](std::shared_ptr<graph::GNode> gnode) -> void {
        // input0: filter, input1: dy
        // output0: dx
        NNFUSION_CHECK(gnode->get_input_size() == 2);
        auto filter_shape = gnode->get_input_shape(0);
        auto dy_shape = gnode->get_input_shape(1);
        NNFUSION_CHECK(filter_shape.size() == dy_shape.size());
        auto op = std::dynamic_pointer_cast<nnfusion::op::GenericOp>(gnode->get_op_ptr());
        std::string data_format = op->localOpConfig.getRoot()["data_format"];
        NNFUSION_CHECK(data_format == "NCHW" || data_format == "NCW")
            << "ConvolutionGradData only supports NCHW or NCW now!";
        Shape kernel_shape = op->localOpConfig.getRoot()["kernel_shape"];
        nnfusion::Strides strides = op->localOpConfig.getRoot()["strides"];
        nnfusion::CoordinateDiff padding_above = op->localOpConfig.getRoot()["padding_above"];
        nnfusion::CoordinateDiff padding_below = op->localOpConfig.getRoot()["padding_below"];
        nnfusion::Strides dilations = op->localOpConfig.getRoot()["dilations"];
        bool dilations_one = true;
        for (auto d : dilations)
        {
            if (d != 1)
            {
                dilations_one = false;
                break;
            }
        }
        NNFUSION_CHECK(dilations_one == true) << "Not support other dilation yet";
        nnfusion::Strides data_dilations = op->localOpConfig.getRoot()["data_dilations"];
        bool data_dilations_one = true;
        for (auto d : data_dilations)
        {
            if (d != 1)
            {
                data_dilations_one = false;
                break;
            }
        }
        NNFUSION_CHECK(data_dilations_one == true) << "Not support other data dilation yet";
        size_t in_channel = op->localOpConfig.getRoot()["in_channel"];
        Shape dx_shape;
        if (data_format == "NCW")
        {
            size_t strides_w = strides[0];
            size_t padding_wb = padding_below[0];
            size_t padding_wa = padding_above[0];
            size_t kernel_size_w = kernel_shape[0];
            NNFUSION_CHECK(filter_shape[2] == kernel_size_w);

            size_t N = dy_shape[0];
            size_t CO = dy_shape[1];
            size_t WO = dy_shape[2];
            size_t WI = strides_w * (WO - 1) + kernel_size_w - padding_wa - padding_wb;
            dx_shape = {N, in_channel, WI};
        }
        else
        {
            NNFUSION_CHECK(dy_shape.size() == 4);
            size_t strides_h = strides[0];
            size_t strides_w = strides[1];
            size_t padding_hb = padding_below[0];
            size_t padding_wb = padding_below[1];
            size_t padding_ha = padding_above[0];
            size_t padding_wa = padding_above[1];
            size_t kernel_size_h = kernel_shape[0];
            size_t kernel_size_w = kernel_shape[1];

            NNFUSION_CHECK(filter_shape[2] == kernel_size_h);
            NNFUSION_CHECK(filter_shape[3] == kernel_size_w);

            size_t N = dy_shape[0];
            size_t CO = dy_shape[1];
            size_t HO = dy_shape[2];
            size_t WO = dy_shape[3];

            size_t HI = strides_h * (HO - 1) + kernel_size_h - padding_ha - padding_hb;
            size_t WI = strides_w * (WO - 1) + kernel_size_w - padding_wa - padding_wb;
            dx_shape = {N, in_channel, HI, WI};
        }
        gnode->set_output_type_and_shape(0, gnode->get_input_element_type(1), dx_shape);
    });
