// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "nnfusion/core/operators/generic_op/generic_op.hpp"

REGISTER_OP(NhwcConv)
    .attr<Strides>("strides")
    .attr<Strides>("dilations")
    .attr<Strides>("pads")
    .infershape([](std::shared_ptr<graph::GNode> gnode) -> void {

        auto op = std::dynamic_pointer_cast<nnfusion::op::GenericOp>(gnode->get_op_ptr());
        auto config = op->localOpConfig.getRoot();
        NNFUSION_CHECK(gnode->get_in_edges().size() == 2);
        auto in_shape = gnode->get_input_shape(0);     // N, H, W, C
        auto kernel_shape = gnode->get_input_shape(1); // F, KH, KW, C
        NNFUSION_CHECK(in_shape.size() == 4);
        NNFUSION_CHECK(kernel_shape.size() == 4);
        Strides dialation = config["dilations"];
        Strides stride = config["strides"];
        Strides pad = config["pads"];

        size_t HO =
            (in_shape[1] + 2 * pad[0] - 1 - (kernel_shape[1] - 1) * dialation[0]) / stride[0] + 1;
        size_t WO =
            (in_shape[2] + 2 * pad[1] - 1 - (kernel_shape[2] - 1) * dialation[1]) / stride[1] + 1;
        nnfusion::Shape outshape{in_shape[0], HO, WO, kernel_shape[0]};
        gnode->set_output_type_and_shape(0, gnode->get_input_element_type(0), outshape);
    })
    .translate_v2([](std::shared_ptr<graph::GNode> curr) -> std::string {
        auto op = std::dynamic_pointer_cast<nnfusion::op::GenericOp>(curr->get_op_ptr());
        auto in_shape = curr->get_input_shape(0);     // N, H, W, C
        auto kernel_shape = curr->get_input_shape(1); // F, KH, KW, C
        auto out_shape = curr->get_output_shape(0);
        Strides dialation = op->localOpConfig.getRoot()["dilations"];
        Strides stride = op->localOpConfig.getRoot()["strides"];
        Strides pad = op->localOpConfig.getRoot()["pads"];
        auto ir_template =
            R"( @output0@@output0_layout@ +=! @input0@@input0_layout@@pad_cond@ * @input1@@input1_layout@ where HO in @height@, WO in @width@; )";

        const auto& dilation_h = dialation[0];
        const auto& dilation_w = dialation[1];
        const auto& stride_h = stride[0];
        const auto& stride_w = stride[1];
        const auto& padding_h = pad[0];
        const auto& padding_w = pad[1];
        const auto& kernel_size_h = kernel_shape[1];
        const auto& kernel_size_w = kernel_shape[2];

        nnfusion::op::OpConfig::any config;
        std::string HO =
            "-@pad_0@ + KH * " + to_string(dilation_h) + " + HO * " + to_string(stride_h);
        std::string WO =
            "-@pad_1@ + KW * " + to_string(dilation_w) + " + WO * " + to_string(stride_w);
        std::string shape_template = "[N, " + HO + ", " + WO + ", C]";
        config["input1_layout"] = "[F, KH, KW, C]";
        config["output0_layout"] = "[N, HO, WO, F]";
        config["height"] = out_shape[1];
        config["width"] = out_shape[2];
        config["pad_0"] = to_string(padding_h);
        config["pad_1"] = to_string(padding_w);
        config["input0_layout"] = op::create_code_from_template(shape_template, config);
        std::string pad_cond;
        if (padding_h || padding_w)
        {
            config["in_height"] = in_shape[1];
            config["in_width"] = in_shape[2];
            auto pad_template = ".when([" + HO + " >= 0, " + HO + " < @in_height@, " + WO +
                                " >= 0, " + WO +
                                " < @in_width@], const(0.0).cast(@input0@@input0_layout@.dtype()))";
            pad_cond = op::create_code_from_template(pad_template, config);
        }
        config["pad_cond"] = pad_cond;
        return op::create_code_from_template(ir_template, config);

    });
