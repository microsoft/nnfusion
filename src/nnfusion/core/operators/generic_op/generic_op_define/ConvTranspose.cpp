
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "nnfusion/core/operators/generic_op/generic_op.hpp"

REGISTER_OP(ConvTranspose)
    .attr<Shape>("kernel_shape")
    .attr<Strides>("strides")
    .attr<Strides>("dilations")
    .attr<CoordinateDiff>("padding_above")
    .attr<CoordinateDiff>("padding_below")
    .attr<Coordinate>("output_padding")
    .attr<std::string>("data_format")
    .infershape([](std::shared_ptr<graph::GNode> gnode) -> void {
        NNFUSION_CHECK(3 >= gnode->get_input_size());
        auto op = std::dynamic_pointer_cast<nnfusion::op::GenericOp>(gnode->get_op_ptr());

        auto input_shape = gnode->get_input_shape(0);
        auto filter_shape = gnode->get_input_shape(1);

        std::string data_format = op->localOpConfig.getRoot()["data_format"];
        bool is_nchw = (data_format == "NCHW") || (data_format == "NCDHW");
        NNFUSION_CHECK(is_nchw) << "ConvTranspose only supports channels first now!";

        Shape kernel_shape = op->localOpConfig.getRoot()["kernel_shape"];
        NNFUSION_CHECK(kernel_shape[0] == kernel_shape[1])
            << "ConvTranspose only sopport equal kernel size!";
        Shape strides = op->localOpConfig.getRoot()["strides"];
        Shape padding_above = op->localOpConfig.getRoot()["padding_above"];
        Shape padding_below = op->localOpConfig.getRoot()["padding_below"];
        Shape dilations = op->localOpConfig.getRoot()["dilations"];
        Shape output_padding = op->localOpConfig.getRoot()["output_padding"];

        Shape output_shape(input_shape);
        for (int i = 0; i < kernel_shape.size(); ++i)
        {
            output_shape[i + (is_nchw ? 2 : 1)] =
                (input_shape[i + (is_nchw ? 2 : 1)] - 1) * strides[i] +
                ((kernel_shape[i] - 1) * dilations[i] + 1) - padding_above[i] - padding_below[i] + 
                output_padding[i];
        }
        output_shape[is_nchw ? 1 : output_shape.size() - 1] = filter_shape[1];

        gnode->set_output_type_and_shape(0, gnode->get_input_element_type(0), output_shape);
    })
    .translate_v2([](std::shared_ptr<graph::GNode> gnode) -> std::string {
        // TODO: IR needs to deal with output padding
        auto expr_tmpl =
            R"( @output0@@output0_layout@ +=! @input0@@input0_layout@@pad_cond@ * @input1@@input1_layout@ @boundary_cond@;)";

        auto op = std::dynamic_pointer_cast<nnfusion::op::GenericOp>(gnode->get_op_ptr());
        Shape kernel_shape = op->localOpConfig.getRoot()["kernel_shape"];
        Strides strides = op->localOpConfig.getRoot()["strides"];
        // Strides pads_above = op->localOpConfig.getRoot()["padding_above"];
        Strides padding_below = op->localOpConfig.getRoot()["padding_below"];

        const auto& in_shape = gnode->get_input_shape(0);
        const auto& out_shape = gnode->get_output_shape(0);
        bool is_conv3d = in_shape.size() == 5;
        const auto& is_nchw = true;

        std::vector<std::string> d_mask;
        std::vector<std::string> d_layout;
        std::vector<std::string> k_layout;
        d_layout = is_conv3d ? std::vector<std::string>{"DO", "HO", "WO"} : std::vector<std::string>{"HO", "WO"};
        k_layout = is_conv3d ? std::vector<std::string>{"KD", "KH", "KW"} : std::vector<std::string>{"KH", "KW"};
        for (int d_id = 0; d_id < out_shape.size() - 2; d_id++)
        {
            int pad = kernel_shape[d_id] - 1 - padding_below[d_id];
            d_mask.push_back("(" + to_string(-pad) + " + " + k_layout[d_id] + " + " + d_layout[d_id] + ") // " + to_string(strides[d_id]));
        }

        nnfusion::op::OpConfig::any config;
        for (int p_id = 0; p_id < padding_below.size(); p_id++)
            config["pad_" + to_string(p_id)] = to_string(padding_below[p_id]);
        
        std::vector<std::string> b_conds;
        for (int d_id = 0; d_id < d_layout.size(); d_id++)
            b_conds.push_back(d_layout[d_id] + " in " + to_string(is_nchw ? out_shape[d_id + 2] : out_shape[d_id + 1]));
        for (int k_id = 0; k_id < k_layout.size(); k_id++)
            b_conds.push_back(k_layout[k_id] + " in " + to_string(kernel_shape[k_id]));
        config["boundary_cond"] = "where " + join<std::vector<std::string>>(b_conds, ", ");

        std::string d_shape_expr = join<std::vector<std::string>>(d_mask, ", ");
        std::string shape_template = is_nchw ? ("[N, C, " + d_shape_expr + "]") : ("[N, " + d_shape_expr + ", C]");
        config["input0_layout"] = op::create_code_from_template(shape_template, config);

        for (int k_id = 0; k_id < k_layout.size(); k_id++)
            k_layout[k_id] = to_string(kernel_shape[k_id]) + " - " + to_string(kernel_shape[k_id] / 2) + " - " + k_layout[k_id];
        std::string k_shape_expr = join<std::vector<std::string>>(k_layout, ", ");
        config["input1_layout"] = is_nchw ? ("[C, F, " + k_shape_expr + "]") : ("[" + k_shape_expr + ", F, C]");
        std::string o_shape_expr = join<std::vector<std::string>>(d_layout, ", ");
        config["output0_layout"] = is_nchw ? ("[N, F, " + o_shape_expr + "]") : ("[N, " + o_shape_expr + ", F]");

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
                p_conds.push_back(d_mask[d_id].substr(0, d_mask[d_id].find("//")) + "% " + to_string(strides[d_id]) + " == 0");
            }
            auto pad_template = ".when([" + join<std::vector<std::string>>(p_conds, ", ") + "], const(0.0).cast(@input0@@input0_layout@.dtype()))";
            pad_cond = op::create_code_from_template(pad_template, config);
        }
        config["pad_cond"] = pad_cond;

        auto ir = op::create_code_from_template(expr_tmpl, config);
        // NNFUSION_LOG(INFO) << ir;
        return ir;
    });
