// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "nnfusion/core/operators/generic_op/generic_op.hpp"

REGISTER_OP(ConvTranspose)
    .attr<Shape>("kernel_shape")
    .attr<Strides>("strides")
    .attr<Strides>("dilations")
    .attr<CoordinateDiff>("padding_above")
    .attr<CoordinateDiff>("padding_below")
    .attr<std::string>("data_format")
    .attr<Shape>("output_shape")
    .attr<CoordinateDiff>("output_padding")
    .infershape([](std::shared_ptr<graph::GNode> gnode) -> void {
        NNFUSION_CHECK(3 >= gnode->get_input_size());
        auto op = std::dynamic_pointer_cast<nnfusion::op::GenericOp>(gnode->get_op_ptr());

        auto input_shape = gnode->get_input_shape(0);
        auto filter_shape = gnode->get_input_shape(1);

        std::string data_format = op->localOpConfig.getRoot()["data_format"];
        bool is_nchw = (data_format == "NCHW");
        // NNFUSION_CHECK(data_format == "NCHW" || data_format == "NCW" || data_format == "NCDHW")
        //     << "ConvTranspose only supports NCW, NCHW or NCDHW now!";
        NNFUSION_CHECK(is_nchw) << "ConvTranspose only supports NCHW now!";

        Shape kernel_shape = op->localOpConfig.getRoot()["kernel_shape"];
        NNFUSION_CHECK(kernel_shape[0] == kernel_shape[1])
            << "ConvTranspose only sopport equal kernel size!";

        Strides strides = op->localOpConfig.getRoot()["strides"];
        CoordinateDiff padding_above = op->localOpConfig.getRoot()["padding_above"];
        CoordinateDiff padding_below = op->localOpConfig.getRoot()["padding_below"];
        Strides dilations = op->localOpConfig.getRoot()["dilations"];

        for (size_t d : dilations)
        {
            if (d != 1)
            {
                NNFUSION_CHECK_FAIL() << "Unsupported dilations.";
            }
        }
        Shape output_shape(input_shape);

        for (int i = 0; i < kernel_shape.size(); ++i)
        {
            output_shape[i + 2] = (input_shape[i + 2] - 1) * strides[i] +
                                  ((kernel_shape[i] - 1) * dilations[i] + 1) - padding_above[i] -
                                  padding_below[i];
        }

        output_shape[1] = filter_shape[1];
        gnode->set_output_type_and_shape(0, gnode->get_input_element_type(0), output_shape);
    })
    /*
    .translate_v2([](std::shared_ptr<graph::GNode> gnode) -> std::string {
        auto expr_tmpl =
            R"( @output0@[N, F, HO, WO] +=! @input0@[N, C, (-@pad_h@ + HO + KH) // @stride_h@, (-@pad_w@ + WO + KW) // @stride_w@].when([(-@pad_h@ + HO + KH) // @stride_h@ >= 0, (-@pad_h@ + HO + KH) // @stride_h@ < @in_height@, (-@pad_w@ + WO + KW) // @stride_w@ >= 0, (-@pad_w@ + WO + KW) // @stride_w@ < @in_width@, (-@pad_h@ + HO + KH) % @stride_h@ == 1, (-@pad_w@ + WO + KW) % @stride_w@ == 1], const(0.0).cast(@input0@[0, 0, 0, 0].dtype())) * @input1@[C, F, @ksize_h@ - KH - 1, @ksize_w@ - KW - 1] where HO in @out_height@, WO in @out_width@, KH in @ksize_h@, KW in @ksize_w@; )";

        auto op = std::dynamic_pointer_cast<nnfusion::op::GenericOp>(gnode->get_op_ptr());
        Shape kernel_shape = op->localOpConfig.getRoot()["kernel_shape"];
        Strides strides = op->localOpConfig.getRoot()["strides"];
        Strides pads_above = op->localOpConfig.getRoot()["padding_above"];
        // Strides pads_below = op->localOpConfig.getRoot()["padding_below"];

        const auto& in_shape = gnode->get_input_shape(0);
        const auto& out_shape = gnode->get_output_shape(0);

        nnfusion::op::OpConfig::any config;
        config["pad_h"] = to_string(kernel_shape[0] / 2) + " + " + to_string(pads_above[0]);
        config["pad_w"] = to_string(kernel_shape[1] / 2) + " + " + to_string(pads_above[1]);
        config["stride_h"] = to_string(strides[0]);
        config["stride_w"] = to_string(strides[1]);
        config["out_height"] = to_string(out_shape[2]);
        config["out_width"] = to_string(out_shape[3]);
        config["in_height"] = to_string(in_shape[2]);
        config["in_width"] = to_string(in_shape[3]);
        config["ksize_h"] = to_string(kernel_shape[0]);
        config["ksize_w"] = to_string(kernel_shape[1]);

        auto ir = op::create_code_from_template(expr_tmpl, config);
        NNFUSION_LOG(INFO) << ir;
        return ir;
    });*/
    .translate_v2([](std::shared_ptr<graph::GNode> gnode) -> std::string {

        auto op = std::dynamic_pointer_cast<nnfusion::op::GenericOp>(gnode->get_op_ptr());
        Shape kernel_shape = op->localOpConfig.getRoot()["kernel_shape"];
        Strides strides = op->localOpConfig.getRoot()["strides"];
        Strides pads_above = op->localOpConfig.getRoot()["padding_above"];
        Strides pads_below = op->localOpConfig.getRoot()["padding_below"];
        Strides dilations = op->localOpConfig.getRoot()["dilations"];
        const auto& in_shape = gnode->get_input_shape(0);
        const auto& out_shape = gnode->get_output_shape(0);

        auto ir_template =
            R"( @output0@@output0_layout@ +=! @input0@@input0_layout@.when([@pad_cond@], const(0.0).cast(@input0@@input0_layout@.dtype())) * @input1@@input1_layout@ where @where_cond@; )";

        // if (pads_above != pads_below)
        // {
        //     NNFUSION_LOG(NNFUSION_WARNING) << "Asymetric padding is not supported by now.";
        //     return "";
        // }
        nnfusion::op::OpConfig::any config;
        size_t spatial_len = kernel_shape.size();
        std::string where_cond, pad_cond;
        std::string input0_layout = "[N, C, ";
        std::string input1_layout = "[C, F, ";
        std::string output0_layout = "[N, F, ";
        for (size_t i = 0; i < spatial_len; i++)
        {
            std::string DOI = "DO" + to_string(i);
            std::string KI = "K" + to_string(i);
            std::string where_DOI = DOI + " in " + to_string(out_shape[2 + i]);
            std::string where_KI = KI + " in " + to_string(kernel_shape[i]);
            std::string pad_i = to_string(kernel_shape[i] / 2) + " + " + to_string(pads_above[i]);
            where_cond += where_DOI + ", " + where_KI;
            input1_layout += to_string(kernel_shape[i]) + " - " + KI + " - 1";
            output0_layout += DOI;
            std::string input_i_pre =
                "(-" + pad_i + " + " + DOI + " + " + KI + " * " + to_string(dilations[i]) + ")";
            std::string input_i = input_i_pre + " // " + to_string(strides[i]);
            input0_layout += input_i;
            pad_cond += input_i + " >= 0, " + input_i + " < " + to_string(in_shape[i + 2]) + ", " +
                        input_i_pre + " % " + to_string(strides[i]) + " == 1";

            if (i != spatial_len - 1)
            {
                where_cond += ", ";
                input1_layout += ", ";
                output0_layout += ", ";
                input0_layout += ", ";
                pad_cond += ", ";
            }
            else
            {
                input1_layout += "]";
                output0_layout += "]";
                input0_layout += "]";
            }
        }

        config["input1_layout"] = input1_layout;
        config["output0_layout"] = output0_layout;
        config["input0_layout"] = input0_layout;
        config["where_cond"] = where_cond;
        config["pad_cond"] = pad_cond;

        return op::create_code_from_template(ir_template, config);
    });