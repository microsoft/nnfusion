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

        auto& input_shape = curr->get_input_shape(0);
        auto& output_shape = curr->get_output_shape(0);
        const auto& kernel = _op->get_window_shape();
        const auto& strides = _op->get_window_movement_strides();
        const auto& padding_below = _op->get_padding_below();
        const auto& padding_above = _op->get_padding_above();
        auto include_padding_in_avg_computation = _op->get_include_padding_in_avg_computation();

        // TODO(lingm): check asymmetric padding, onnx test case passed
        // if (!include_padding_in_avg_computation && padding_below != padding_above)
        // {
        //     // not support asymmetric padding when not include_padding_in_avg_computation
        //     NNFUSION_LOG(NNFUSION_WARNING)
        //         << "not support asymmetric padding when not include_padding_in_avg_computation "
        //            "in Antares IR";
        //     return std::string();
        // }

        auto ir_template =
            R"( @output0@@output0_layout@ +=! @input0@@input0_layout@@when_condition@/@avg_window_size@ @where_condition@; )";

        std::string input0_layout;
        std::string output0_layout;
        std::string when_condition;
        std::string where_condition;
        std::string avg_window_size;

        auto output_layout = std::vector<std::string>{"N", "C"};
        auto input_layout = std::vector<std::string>{"N", "C"};

        for (int i = 0; i < input_shape.size() - 2; i++)
        {
            std::string axis_name = "D" + to_string(i);
            std::string kernel_axis_name = "K" + to_string(i);
            output_layout.push_back(axis_name);
            input_layout.push_back(axis_name + " * " + to_string(strides[i]) + " + " +
                                   kernel_axis_name + " - " + to_string(padding_below[i]));
            if (padding_below[0] > 0)
            {
                when_condition +=
                    (when_condition.empty() ? "" : " , ") + input_layout[i + 2] + " >=0";
            }
            if (padding_above[0] > 0)
            {
                when_condition += (when_condition.empty() ? "" : " , ") + input_layout[i + 2] +
                                  " < " + to_string(input_shape[i + 2]);
            }
            where_condition += (where_condition.empty() ? "" : " , ") + output_layout[i + 2] +
                               " in " + to_string(output_shape[i + 2]) + ", " + kernel_axis_name +
                               " in " + to_string(kernel[i]);

            if (!include_padding_in_avg_computation)
            {
                avg_window_size += (avg_window_size.empty() ? "" : " * ") + std::string("((") +
                                   axis_name + " * " + to_string(strides[i]) + " + " +
                                   to_string(kernel[i]) + " - " + to_string(padding_below[i]) +
                                   ").call(`min`, [" + to_string(input_shape[i + 2]) + "])  - (" +
                                   axis_name + " * " + to_string(strides[i]) + " - " +
                                   to_string(padding_below[i]) + ").call(`max`, [0]))";
            }
        }

        if (!when_condition.empty())
        {
            when_condition = ".when([" + when_condition + "], const(0).cast(@input0@.dtype()))";
        }

        if (!where_condition.empty())
        {
            where_condition = " where " + where_condition;
        }

        if (!include_padding_in_avg_computation)
        {
            avg_window_size = "(" + avg_window_size + ").call(`max`, const(1, `int32`))";
        }
        else
        {
            avg_window_size = std::to_string(nnfusion::shape_size<nnfusion::Shape>(kernel));
        }

        op::OpConfig::any op_config;
        op_config["when_condition"] = when_condition;
        op_config["where_condition"] = where_condition;
        op_config["avg_window_size"] = avg_window_size;
        op_config["output0_layout"] = vector_to_string<std::vector<std::string>>(output_layout);
        op_config["input0_layout"] = vector_to_string<std::vector<std::string>>(input_layout);

        auto ir_expression = op::create_code_from_template(ir_template, op_config);

        return ir_expression;
    });
