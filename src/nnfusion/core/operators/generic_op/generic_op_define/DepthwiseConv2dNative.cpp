// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "nnfusion/core/operators/generic_op/generic_op.hpp"

REGISTER_OP(DepthwiseConv2dNative)
    .attr<nnfusion::op::OpConfig::any>("data_format")
    .attr<nnfusion::op::OpConfig::any>("padding_type")
    .attr<nnfusion::op::OpConfig::any>("strides")
    .attr<nnfusion::op::OpConfig::any>("dilations")
    .attr<nnfusion::op::OpConfig::any>("padding_before")
    .attr<nnfusion::op::OpConfig::any>("padding_after")
    .constrait([](const nnfusion::op::OpConfig::any& config) -> bool {
        if (config["padding_type"] != "SAME")
        {
            return false;
        }
        return true;
    })
    .infershape([](std::shared_ptr<graph::GNode> gnode) -> void {
        NNFUSION_CHECK(gnode->get_input_size() == 2);
        auto op = std::dynamic_pointer_cast<nnfusion::op::GenericOp>(gnode->get_op_ptr());

        // [ batch, in_rows, in_cols, in_depth ]
        const Shape& input_shape = gnode->get_input_shape(0);

        // [ filter_rows, filter_cols, in_depth, depth_multiplier ]
        const Shape& filter_shape = gnode->get_input_shape(1);

        std::string data_format = op->localOpConfig.getRoot()["data_format"];
        bool is_nhwc = (data_format == "NHWC");

        const int64_t in_depth = is_nhwc ? input_shape[3] : input_shape[1];
        NNFUSION_CHECK(in_depth == filter_shape[2]);
        const int64_t depth_multiplier = filter_shape[3];
        const int64_t out_depth = in_depth * depth_multiplier;
        const int64_t input_rows = is_nhwc ? input_shape[1] : input_shape[2];
        const int64_t input_cols = is_nhwc ? input_shape[2] : input_shape[3];
        const int64_t filter_rows = filter_shape[0];
        const int64_t filter_cols = filter_shape[1];
        const int64_t batch = input_shape[0];

        std::vector<int64_t> strides = op->localOpConfig.getRoot()["strides"];
        NNFUSION_CHECK(strides.size() == 2);
        const int64_t out_rows = (input_rows + strides[0] - 1) / strides[0];
        const int64_t out_cols = (input_cols + strides[1] - 1) / strides[1];

        Shape output_shape(
            is_nhwc
                ? Shape({(size_t)batch, (size_t)out_rows, (size_t)out_cols, (size_t)out_depth})
                : Shape({(size_t)batch, (size_t)out_depth, (size_t)out_rows, (size_t)out_cols}));

        gnode->set_output_type_and_shape(0, gnode->get_input_element_type(0), output_shape);
    })
    /*
    .translate([](std::shared_ptr<graph::GNode> gnode) -> std::string {
        NNFUSION_CHECK(gnode->get_input_size() == 2);
        auto op = std::dynamic_pointer_cast<nnfusion::op::GenericOp>(gnode->get_op_ptr());
        NNFUSION_CHECK_NOT_NULLPTR(op) << "Node type is not " << gnode->get_op_ptr()->get_op_type();
        // currently only support NHWC format
        if (op->localOpConfig.getRoot()["data_format"] != "NHWC")
            return "";
        const auto& padding_below = op->localOpConfig.getRoot()["padding_before"];
        const auto& padding_above = op->localOpConfig.getRoot()["padding_after"];
        uint64_t padding[] = {
            padding_below[1], padding_below[0], padding_above[1], padding_above[0]};

        return op::create_code_from_template(
            R"( - input("input0", @input_shape@); input("input1", @filter_shape@); output(@output_shape@, topi=topi.nn.depthwise_conv2d_nhwc(args("input0"), args("input1"), stride=@stride@, padding=@padding@, dilation=@dilation@)); )",
            {{"input_shape", vector_to_string(gnode->get_input_shape(0))},
             {"filter_shape", vector_to_string(gnode->get_input_shape(1))},
             {"output_shape", vector_to_string(gnode->get_output_shape(0))},
             {"stride", vector_to_string(op->localOpConfig.getRoot()["strides"])},
             {"padding", vector_to_string(padding)},
             {"dilation", vector_to_string(op->localOpConfig.getRoot()["dilations"])}});
    })
    */
    .translate_v2([](std::shared_ptr<graph::GNode> curr) -> std::string {
        auto ir_template =
            R"( @output0@@output0_layout@ +=! @input0@@input0_layout@@pad_cond@ * @input1@@input1_layout@ where HO in @height@, WO in @width@, KH in @kernel_h@, KW in @kernel_w@; )";
        // auto manual_rule = R"( ## @: plan/depthwise_convfwd_@data_format@_v1 )";

        auto _op = std::dynamic_pointer_cast<nnfusion::op::GenericOp>(curr->get_op_ptr());
        NNFUSION_CHECK_NOT_NULLPTR(_op) << "Node type is not " << curr->get_op_ptr()->get_op_type();

        auto is_nhwc = _op->localOpConfig.getRoot()["data_format"] == "NHWC";
        const auto& dilation_h = int64_t(_op->localOpConfig.getRoot()["dilations"][0]);
        const auto& dilation_w = int64_t(_op->localOpConfig.getRoot()["dilations"][1]);
        const auto& stride_h = int64_t(_op->localOpConfig.getRoot()["strides"][0]);
        const auto& stride_w = int64_t(_op->localOpConfig.getRoot()["strides"][1]);
        const auto& padding_h = int64_t(_op->localOpConfig.getRoot()["padding_before"][0]);
        const auto& padding_w = int64_t(_op->localOpConfig.getRoot()["padding_before"][1]);
        const auto& kernel_size_h = curr->get_input_shape(1)[0];
        const auto& kernel_size_w = curr->get_input_shape(1)[1];
        const auto& in_shape = curr->get_input_shape(0);
        const auto& out_shape = curr->get_output_shape(0);
        const std::string data_format = is_nhwc ? "nhwc" : "nchw";
        NNFUSION_CHECK(dilation_h == 1) << "Not support other dilation yet.";
        NNFUSION_CHECK(dilation_w == 1) << "Not support other dilation yet.";
        nnfusion::op::OpConfig::any config;
        config["input1_layout"] = "[KH, KW, C, 0]";
        config["output0_layout"] = is_nhwc ? "[N, HO, WO, C]" : "[N, C, HO, WO]";
        config["height"] = is_nhwc ? out_shape[1] : out_shape[2];
        config["width"] = is_nhwc ? out_shape[2] : out_shape[3];
        config["in_height"] = is_nhwc ? in_shape[1] : in_shape[2];
        config["in_width"] = is_nhwc ? in_shape[2] : in_shape[3];
        config["pad_0"] = to_string(padding_h);
        config["pad_1"] = to_string(padding_w);
        config["kernel_h"] = to_string(kernel_size_h);
        config["kernel_w"] = to_string(kernel_size_w);
        std::string HO = "-@pad_0@ + KH + HO * " + to_string(stride_h);
        std::string WO = "-@pad_1@ + KW + WO * " + to_string(stride_w);
        std::string shape_template =
            is_nhwc ? "[N, " + HO + ", " + WO + ", C]" : "[N, C, " + HO + ", " + WO + "]";
        config["input0_layout"] = op::create_code_from_template(shape_template, config);

        std::string pad_cond;
        if (padding_h || padding_w)
        {
            auto pad_template = ".when([" + HO + " >= 0, " + HO + " < @in_height@, " + WO +
                                " >= 0, " + WO +
                                " < @in_width@], const(0.0).cast(@input0@@input0_layout@.dtype()))";
            pad_cond = op::create_code_from_template(pad_template, config);
        }
        config["pad_cond"] = pad_cond;

        // return op::create_code_from_template(ir_template, config) +
        //        op::create_code_from_template(manual_rule, {{"data_format", data_format}});
        return op::create_code_from_template(ir_template, config);
    })
    .infersharedmemory([](std::shared_ptr<graph::GNode> gnode) -> void {
        auto op = std::dynamic_pointer_cast<nnfusion::op::GenericOp>(gnode->get_op_ptr());
        if (op->localOpConfig.getRoot()["padding_type"] != "SAME")
            return;

        for (auto s : op->localOpConfig.getRoot()["strides"])
        {
            if (s != 1)
                return;
        }

        for (auto d : op->localOpConfig.getRoot()["dilations"])
        {
            if (d != 1)
                return;
        }

        for (auto p : op->localOpConfig.getRoot()["padding_before"])
        {
            if (p != 0)
                return;
        }

        for (auto p : op->localOpConfig.getRoot()["padding_after"])
        {
            if (p != 0)
                return;
        }

        std::string data_format = op->localOpConfig.getRoot()["data_format"];
        bool is_nhwc = (data_format == "NHWC");
        const Shape& input_shape = gnode->get_input_shape(0);
        int channel = is_nhwc ? 3 : 1;
        auto input_channel_count = input_shape[channel];

        std::vector<size_t> shared_memory;
        for (size_t i = 0; i < gnode->get_output_shape(0).size(); i++)
        {
            if (i == channel)
                shared_memory.push_back(input_channel_count);
            else
                shared_memory.push_back(1);
        }

        op->set_shared_memory(shared_memory);
    });
