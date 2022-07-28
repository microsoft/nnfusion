// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "nnfusion/core/operators/generic_op/generic_op.hpp"

REGISTER_OP(Convolution)
    .infershape(nnfusion::op::infershape::unimplemented_and_not_used)
    /*
    .translate([](std::shared_ptr<graph::GNode> curr) -> std::string {
        auto _op = static_pointer_cast<nnfusion::op::Convolution>(curr->get_op_ptr());
        NNFUSION_CHECK_NOT_NULLPTR(_op) << "Node type is not " << curr->get_op_ptr()->get_op_type();

        const auto& dilation = _op->get_window_dilation_strides();
        const auto& stride = _op->get_window_movement_strides();
        const auto& padding_below = _op->get_padding_below();
        const auto& padding_above = _op->get_padding_above();
        const auto& data_format = _op->get_data_format();
        int64_t padding[] = {
            padding_below[1], padding_below[0], padding_above[1], padding_above[0]};

        return op::create_code_from_template(
            R"( - input("input0", @input_shape_0@); input("input1", @input_shape_1@); output(@output_shape@, topi=topi.nn.conv2d_@data_format@(args("input0"), args("input1"), stride=@stride@, padding=@padding@, dilation=@dilation@)); )",
            {{"input_shape_0", vector_to_string(curr->get_input_shape(0))},
             {"input_shape_1", vector_to_string(curr->get_input_shape(1))},
             {"output_shape", vector_to_string(curr->get_output_shape(0))},
             {"data_format", data_format == "NCHW" ? "nchw" : "nhwc"},
             {"stride", vector_to_string(stride)},
             {"padding", vector_to_string(padding)},
             {"dilation", vector_to_string(dilation)}});
    })
    */
    .translate_v2([](std::shared_ptr<graph::GNode> curr) -> std::string {
        auto ir_template =
            R"( @output0@@output0_layout@ +=! @input0@@input0_layout@@pad_cond@ * @input1@@input1_layout@ @boundary_cond@; )";

        auto _op = static_pointer_cast<nnfusion::op::Convolution>(curr->get_op_ptr());
        NNFUSION_CHECK_NOT_NULLPTR(_op) << "Node type is not " << curr->get_op_ptr()->get_op_type();
        const auto& in_shape = curr->get_input_shape(0);
        const auto& out_shape = curr->get_output_shape(0);
        bool is_conv3d = in_shape.size() == 5;

        const auto& dilations = _op->get_window_dilation_strides();
        const auto& strides = _op->get_window_movement_strides();
        const auto& is_nchw = _op->get_data_format() == "NCHW" || _op->get_data_format() == "NCDHW";
        const auto& padding_below = _op->get_padding_below();
        const auto& padding_above = _op->get_padding_above();
        const std::string data_format = is_nchw ? "nchw" : "nhwc";
        NNFUSION_CHECK(dilations[0] == 1) << "Not support other dilation yet.";
        NNFUSION_CHECK(dilations[1] == 1) << "Not support other dilation yet.";
        NNFUSION_CHECK(padding_below == padding_above) << "Asymetric padding is not supported by now.";

        nnfusion::op::OpConfig::any config;
        for (int p_id = 0; p_id < padding_below.size(); p_id++)
            config["pad_" + to_string(p_id)] = to_string(padding_below[p_id]);

        std::vector<std::string> d_mask;
        std::vector<std::string> d_layout;
        std::vector<std::string> k_layout;
        d_layout = is_conv3d ? std::vector<std::string>{"DO", "HO", "WO"} : std::vector<std::string>{"HO", "WO"};
        k_layout = is_conv3d ? std::vector<std::string>{"KD", "KH", "KW"} : std::vector<std::string>{"KH", "KW"};
        for (int d_id = 0; d_id < out_shape.size() - 2; d_id++)
        {
            d_mask.push_back("-@pad_" + to_string(d_id) + "@ + " + k_layout[d_id] + " + " + d_layout[d_id] + " * " + to_string(strides[d_id]));
        }

        std::string d_shape_expr = join<std::vector<std::string>>(d_mask, ", ");
        std::string shape_template = is_nchw ? ("[N, C, " + d_shape_expr + "]") : ("[N, " + d_shape_expr + ", C]");
        config["input0_layout"] = op::create_code_from_template(shape_template, config);

        std::string k_shape_expr = join<std::vector<std::string>>(k_layout, ", ");
        config["input1_layout"] = is_nchw ? ("[F, C, " + k_shape_expr + "]") : ("[" + k_shape_expr + ", C, F]");
        std::string o_shape_expr = join<std::vector<std::string>>(d_layout, ", ");
        config["output0_layout"] = is_nchw ? ("[N, F, " + o_shape_expr + "]") : ("[N, " + o_shape_expr + ", F]");

        std::vector<std::string> b_conds;
        for (int d_id = 0; d_id < d_layout.size(); d_id++)
            b_conds.push_back(d_layout[d_id] + " in " + to_string(is_nchw ? out_shape[d_id + 2] : out_shape[d_id + 1]));
        config["boundary_cond"] = "where " + join<std::vector<std::string>>(b_conds, ", ");

        std::string pad_cond;
        bool need_pad = false;
        for (int p_id = 0; p_id < padding_below.size(); p_id++)
            need_pad |= padding_below[p_id];
        if (need_pad)
        {
            std::vector<std::string> p_conds;
            for (int d_id = 0; d_id < d_mask.size(); d_id++)
            {
                p_conds.push_back(d_mask[d_id] + " >= 0");
                p_conds.push_back(d_mask[d_id] + " < " + to_string(is_nchw ? in_shape[d_id + 2] : in_shape[d_id + 1]));
            }
            auto pad_template = ".when([" + join<std::vector<std::string>>(p_conds, ", ") + "], const(0.0).cast(@input0@@input0_layout@.dtype()))";
            pad_cond = op::create_code_from_template(pad_template, config);
        }
        config["pad_cond"] = pad_cond;

        return op::create_code_from_template(ir_template, config);
    });
