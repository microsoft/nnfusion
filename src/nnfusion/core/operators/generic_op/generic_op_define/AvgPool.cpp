// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "nnfusion/core/operators/generic_op/generic_op.hpp"

REGISTER_OP(AvgPool)
    .infershape(nnfusion::op::infershape::unimplemented_and_not_used)
    /*
    .translate([](std::shared_ptr<graph::GNode> curr) -> std::string {
        auto _op = static_pointer_cast<nnfusion::op::AvgPool>(curr->get_op_ptr());
        NNFUSION_CHECK_NOT_NULLPTR(_op) << "Node type is not " << curr->get_op_ptr()->get_op_type();

        const auto& kernel = _op->get_window_shape();
        const auto& stride = _op->get_window_movement_strides();
        const auto& padding_below = _op->get_padding_below();
        const auto& padding_above = _op->get_padding_above();
        uint64_t padding[] = {
            padding_below[1], padding_below[0], padding_above[1], padding_above[0]};

        return op::create_code_from_template(
            R"( - input("input0", @input_shape@); output(@output_shape@, topi=topi.nn.pool(args("input0"), kernel=@kernel@, stride=@stride@, padding=@padding@, pool_type="avg")); )",
            {{"input_shape", vector_to_string(curr->get_input_shape(0))},
             {"output_shape", vector_to_string(curr->get_output_shape(0))},
             {"kernel", vector_to_string(kernel)},
             {"stride", vector_to_string(stride)},
             {"padding", vector_to_string(padding)}});
    })
    */
    .translate_v2([](std::shared_ptr<graph::GNode> curr) -> std::string {

        auto _op = static_pointer_cast<nnfusion::op::AvgPool>(curr->get_op_ptr());
        NNFUSION_CHECK_NOT_NULLPTR(_op) << "Node type is not " << curr->get_op_ptr()->get_op_type();

        // divide operation goes before add operation, which may cause precision issue.
        // auto ir_template =
        //     R"( @output0@@output0_layout@ +=! @input0@@input0_layout@@when_condition@ / ((HO * @stride_h@ + @KH_top@ - @pad_h@).call('min', [@H_top@])  - (HO * @stride_h@ - @pad_h@).call('max', [0])) / ((WO * @stride_w@ + @KW_top@ - @pad_w@).call('min', [@W_top@])  - (WO * @stride_w@ - @pad_w@).call('max', [0])) @where_condition@;)";

        // divide after add operation
        auto ir_template =
            R"( mediate0@output0_layout@ +=! @input0@@input0_layout@@when_condition@ @where_condition@; @output0@@output0_layout@ = mediate0@output0_layout@ / (((HO * @stride_h@ + @KH_top@ - @pad_h@).call('min', [@H_top@])  - (HO * @stride_h@ - @pad_h@).call('max', [0])) * ((WO * @stride_w@ + @KW_top@ - @pad_w@).call('min', [@W_top@])  - (WO * @stride_w@ - @pad_w@).call('max', [0]))).call('max', [1]);)";

        const auto& input0_shape = curr->get_input_shape(0);
        const auto& output0_shape = curr->get_output_shape(0);
        const auto& kernel = _op->get_window_shape();
        const auto& stride = _op->get_window_movement_strides();
        const auto& padding_below = _op->get_padding_below();
        const auto& padding_above = _op->get_padding_above();

        std::string input0_layout;
        std::string output0_layout;
        std::string conditions;
        std::string when_condition_template;
        std::string where_condition;

        NNFUSION_CHECK(input0_shape.size() == output0_shape.size());
        NNFUSION_CHECK(output0_shape.size() == 3 || output0_shape.size() == 4);
        NNFUSION_CHECK(kernel.size() == 2);
        NNFUSION_CHECK(shape_size(kernel) != 0);

        if (output0_shape.size() == 3)
        {
            output0_layout = "[NC, HO, WO]";
            input0_layout = "[NC, ";
        }
        else if (output0_shape.size() == 4)
        {
            output0_layout = "[N, C, HO, WO]";
            input0_layout = "[N, C, ";
        }

        std::string H_in = "(HO * " + to_string(stride[0]) + " + KH";
        H_in += " - " + to_string(padding_below[0]) + ")";
        when_condition_template += (when_condition_template.empty() ? "" : " , ") + H_in +
                                   " >= 0, " + H_in + " < " +
                                   to_string(input0_shape[input0_shape.size() - 2]);

        std::string W_in = "(WO * " + to_string(stride[1]) + " + KW";
        W_in += " - " + to_string(padding_below[1]) + ")";
        when_condition_template += (when_condition_template.empty() ? "" : " , ") + W_in +
                                   " >= 0, " + W_in + " < " +
                                   to_string(input0_shape[input0_shape.size() - 1]);

        input0_layout += H_in + " , " + W_in + "]";

        where_condition = "where HO in " + to_string(output0_shape[output0_shape.size() - 2]) +
                          ", " + "WO in " + to_string(output0_shape[output0_shape.size() - 1]) +
                          ", " + "KH in " + to_string(kernel[0]) + ", " + "KW in " +
                          to_string(kernel[0]);

        if (!when_condition_template.empty())
        {
            when_condition_template = ".when([" + when_condition_template +
                                      "], const(0.0).cast(@input0@@input0_layout@.dtype()))";
        }

        op::OpConfig::any op_config;
        op_config["input0_layout"] = input0_layout;
        op_config["output0_layout"] = output0_layout;
        op_config["div"] = kernel[0] * kernel[1];
        op_config["stride_h"] = stride[0];
        op_config["stride_w"] = stride[1];
        op_config["H_top"] = input0_shape[input0_shape.size() - 2];
        op_config["W_top"] = input0_shape[input0_shape.size() - 1];
        op_config["KH_top"] = kernel[0];
        op_config["KW_top"] = kernel[1];
        op_config["pad_h"] = padding_below[0];
        op_config["pad_w"] = padding_below[1];
        op_config["where_condition"] = where_condition;
        op_config["when_condition"] =
            op::create_code_from_template(when_condition_template, op_config);

        return op::create_code_from_template(ir_template, op_config);
    });
