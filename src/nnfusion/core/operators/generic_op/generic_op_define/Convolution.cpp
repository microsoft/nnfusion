// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "nnfusion/core/operators/generic_op/generic_op.hpp"

std::string translate_conv1d(std::shared_ptr<graph::GNode> curr)
{
    auto _op = static_pointer_cast<nnfusion::op::Convolution>(curr->get_op_ptr());
    string ir_template =
        R"( @output0@@output0_layout@ +=! @input0@@input0_layout@ * @input1@@input1_layout@ where HO in @height@; )";
    NNFUSION_CHECK_NOT_NULLPTR(_op) << "Node type is not " << curr->get_op_ptr()->get_op_type();
    const auto& dilation_h = _op->get_window_dilation_strides()[0];
    const auto& stride_h = _op->get_window_movement_strides()[0];
    const auto& padding_below = _op->get_padding_below();
    const auto& padding_above = _op->get_padding_above();
    const auto& padding_h = _op->get_padding_below()[0];
    const auto& kernel_size_h = curr->get_input_shape(1)[2];
    const auto& in_shape = curr->get_input_shape(0);
    const auto& out_shape = curr->get_output_shape(0);
    NNFUSION_CHECK(padding_below == padding_above) << "Asymetric padding is not supported by now.";
    nnfusion::op::OpConfig::any config;
    std::string HO = "KH*" + to_string(dilation_h) + "+HO*" + to_string(stride_h);
    config["input0_layout"] = "[N, C, " + HO + "]";
    config["input1_layout"] = "[F, C, KH]";
    config["output0_layout"] = "[N, F, HO]";
    config["height"] = out_shape[2];

    if (padding_h)
    {
        string pad_template =
            R"( pad@pad_layout@ = @input0@@pad_input_layout@@pad_cond@ where H0 in @pad_height@;)";
        ir_template =
            R"( @output0@@output0_layout@ +=! pad@input0_layout@ * @input1@@input1_layout@ where HO in @height@; )";
        ir_template = pad_template + ir_template;
        size_t in_height = in_shape[2];
        config["pad_height"] = in_height + 2 * padding_h;
        config["pad_layout"] = "[N, C, H0]";
        config["pad_input_layout"] = "[N, C, H0-" + to_string(padding_h) + "]";
        string dtype;
        NNFUSION_CHECK(
            element::Type::nnfusion_element_type_to_dtype_string(curr->get_element_type(), dtype));
        config["pad_cond"] = ".when([H0>=" + to_string(padding_h) + ", H0<" +
                             to_string(in_height + padding_h) + "], const(0.0).cast(`" + dtype +
                             "`))";
    }

    return op::create_code_from_template(ir_template, config);
}

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
        auto _op = static_pointer_cast<nnfusion::op::Convolution>(curr->get_op_ptr());
        if (_op->get_data_format() == "NCW")
        {
            return translate_conv1d(curr);
        }
        string ir_template =
            R"( @output0@@output0_layout@ +=! @input0@@input0_layout@ * @input1@@input1_layout@ where HO in @height@, WO in @width@; )";
        NNFUSION_CHECK_NOT_NULLPTR(_op) << "Node type is not " << curr->get_op_ptr()->get_op_type();

        if (_op->get_data_format() == "NCHW" || _op->get_data_format() == "NHWC") // Conv2D
        {
            auto ir_template =
                R"( @output0@@output0_layout@ +=! @input0@@input0_layout@@pad_cond@ * @input1@@input1_layout@ where HO in @height@, WO in @width@; )";

            const auto& dilation_h = _op->get_window_dilation_strides()[0];
            const auto& dilation_w = _op->get_window_dilation_strides()[1];
            const auto& stride_h = _op->get_window_movement_strides()[0];
            const auto& stride_w = _op->get_window_movement_strides()[1];
            const auto& is_nchw = _op->get_data_format() == "NCHW";
            const auto& padding_below = _op->get_padding_below();
            const auto& padding_above = _op->get_padding_above();
            const auto& padding_h = _op->get_padding_below()[0];
            const auto& padding_w = _op->get_padding_below()[1];
            const auto& kernel_size_h =
                is_nchw ? curr->get_input_shape(1)[2] : curr->get_input_shape(1)[0];
            const auto& kernel_size_w =
                is_nchw ? curr->get_input_shape(1)[3] : curr->get_input_shape(1)[1];
            const auto& in_shape = curr->get_input_shape(0);
            const auto& out_shape = curr->get_output_shape(0);
            const std::string data_format = is_nchw ? "nchw" : "nhwc";

            // if (padding_below != padding_above)
            // {
            //     NNFUSION_LOG(NNFUSION_WARNING) << "Asymetric padding is not supported by now.";
            //     return "";
            // }
            nnfusion::op::OpConfig::any config;
            std::string HO =
                "-@pad_0@ + KH * " + to_string(dilation_h) + " + HO * " + to_string(stride_h);
            std::string WO =
                "-@pad_1@ + KW * " + to_string(dilation_w) + " + WO * " + to_string(stride_w);
            std::string shape_template =
                is_nchw ? "[N, C, " + HO + ", " + WO + "]" : "[N, " + HO + ", " + WO + ", C]";
            config["input1_layout"] = is_nchw ? "[F, C, KH, KW]" : "[KH, KW, C, F]";
            config["output0_layout"] = is_nchw ? "[N, F, HO, WO]" : "[N, HO, WO, F]";
            config["height"] = is_nchw ? out_shape[2] : out_shape[1];
            config["width"] = is_nchw ? out_shape[3] : out_shape[2];
            config["pad_0"] = to_string(padding_h);
            config["pad_1"] = to_string(padding_w);
            config["input0_layout"] = op::create_code_from_template(shape_template, config);

            std::string pad_cond;
            if (padding_h || padding_w)
            {
                config["in_height"] = is_nchw ? in_shape[2] : in_shape[1];
                config["in_width"] = is_nchw ? in_shape[3] : in_shape[2];
                auto pad_template =
                    ".when([" + HO + " >= 0, " + HO + " < @in_height@, " + WO + " >= 0, " + WO +
                    " < @in_width@], const(0.0).cast(@input0@@input0_layout@.dtype()))";
                pad_cond = op::create_code_from_template(pad_template, config);
            }
            config["pad_cond"] = pad_cond;

            return op::create_code_from_template(ir_template, config);
        }
        else if (_op->get_data_format() == "NCDHW") // Conv3D
        {
            auto ir_template =
                R"( @output0@@output0_layout@ +=! @input0@@input0_layout@@pad_cond@ * @input1@@input1_layout@ where DO in @depth@, HO in @height@, WO in @width@; )";

            const auto& dilation_d = _op->get_window_dilation_strides()[0];
            const auto& dilation_h = _op->get_window_dilation_strides()[1];
            const auto& dilation_w = _op->get_window_dilation_strides()[2];
            const auto& stride_d = _op->get_window_movement_strides()[0];
            const auto& stride_h = _op->get_window_movement_strides()[1];
            const auto& stride_w = _op->get_window_movement_strides()[2];
            const auto& padding_below = _op->get_padding_below();
            const auto& padding_above = _op->get_padding_above();
            const auto& padding_d = _op->get_padding_below()[0];
            const auto& padding_h = _op->get_padding_below()[1];
            const auto& padding_w = _op->get_padding_below()[2];
            const auto& kernel_size_d = curr->get_input_shape(1)[2];
            const auto& kernel_size_h = curr->get_input_shape(1)[3];
            const auto& kernel_size_w = curr->get_input_shape(1)[4];
            const auto& in_shape = curr->get_input_shape(0);
            const auto& out_shape = curr->get_output_shape(0);
            const std::string data_format = "NCDHW";

            // if (padding_below != padding_above)
            // {
            //     NNFUSION_LOG(NNFUSION_WARNING) << "Asymetric padding is not supported by now.";
            //     return "";
            // }
            nnfusion::op::OpConfig::any config;
            std::string DO =
                "-@pad_0@ + KD * " + to_string(dilation_d) + " + DO * " + to_string(stride_d);
            std::string HO =
                "-@pad_1@ + KH * " + to_string(dilation_h) + " + HO * " + to_string(stride_h);
            std::string WO =
                "-@pad_2@ + KW * " + to_string(dilation_w) + " + WO * " + to_string(stride_w);
            std::string shape_template = "[N, C, " + DO + ", " + HO + ", " + WO + "]";
            config["input1_layout"] = "[F, C, KD, KH, KW]";
            config["output0_layout"] = "[N, F, DO, HO, WO]";
            config["depth"] = out_shape[2];
            config["height"] = out_shape[3];
            config["width"] = out_shape[4];
            config["pad_0"] = to_string(padding_d);
            config["pad_1"] = to_string(padding_h);
            config["pad_2"] = to_string(padding_w);
            config["input0_layout"] = op::create_code_from_template(shape_template, config);

            std::string pad_cond;
            if (padding_d || padding_h || padding_w)
            {
                config["in_depth"] = in_shape[2];
                config["in_height"] = in_shape[3];
                config["in_width"] = in_shape[4];
                auto pad_template =
                    ".when([" + DO + " >= 0, " + DO + " < @in_depth@, " + HO + " >= 0, " + HO +
                    " < @in_height@, " + WO + " >= 0, " + WO +
                    " < @in_width@], const(0.0).cast(@input0@@input0_layout@.dtype()))";
                pad_cond = op::create_code_from_template(pad_template, config);
            }
            config["pad_cond"] = pad_cond;

            return op::create_code_from_template(ir_template, config);
        }
        else if (_op->get_data_format() == "NCW") // Conv1D
        {
            auto ir_template =
                R"( @output0@@output0_layout@ +=! @input0@@input0_layout@@pad_cond@ * @input1@@input1_layout@ where WO in @width@; )";

            const auto& dilation_w = _op->get_window_dilation_strides()[0];
            const auto& stride_w = _op->get_window_movement_strides()[0];
            const auto& padding_below = _op->get_padding_below();
            const auto& padding_above = _op->get_padding_above();
            const auto& padding_w = _op->get_padding_below()[0];
            const auto& kernel_size_w = curr->get_input_shape(1)[2];
            const auto& in_shape = curr->get_input_shape(0);
            const auto& out_shape = curr->get_output_shape(0);
            const std::string data_format = "NCW";

            // if (padding_below != padding_above)
            // {
            //     NNFUSION_LOG(NNFUSION_WARNING) << "Asymetric padding is not supported by now.";
            //     return "";
            // }
            nnfusion::op::OpConfig::any config;
            std::string WO =
                "-@pad_0@ + KW * " + to_string(dilation_w) + " + WO * " + to_string(stride_w);
            std::string shape_template = "[N, C, " + WO + "]";
            config["input1_layout"] = "[F, C, KW]";
            config["output0_layout"] = "[N, F, WO]";
            config["depth"] = out_shape[2];
            config["height"] = out_shape[3];
            config["width"] = out_shape[4];
            config["pad_0"] = to_string(padding_w);
            config["input0_layout"] = op::create_code_from_template(shape_template, config);

            std::string pad_cond;
            if (padding_w)
            {
                config["in_width"] = in_shape[2];
                auto pad_template =
                    ".when([" + WO + " >= 0, " + WO +
                    " < @in_width@], const(0.0).cast(@input0@@input0_layout@.dtype()))";
                pad_cond = op::create_code_from_template(pad_template, config);
            }
            config["pad_cond"] = pad_cond;

            return op::create_code_from_template(ir_template, config);
        }

        return "";
    });
