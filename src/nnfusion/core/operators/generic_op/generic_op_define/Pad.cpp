// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "nnfusion/core/operators/generic_op/generic_op.hpp"

REGISTER_OP(Pad)
    .infershape(nnfusion::op::infershape::unimplemented_and_not_used)
    /*
    .translate([](std::shared_ptr<graph::GNode> gnode) -> std::string {
        NNFUSION_CHECK(2 == gnode->get_input_size());
        auto op = static_pointer_cast<nnfusion::op::Pad>(gnode->get_op_ptr());
        NNFUSION_CHECK_NOT_NULLPTR(op) << "Node type is not " << gnode->get_op_ptr()->get_op_type();

        std::shared_ptr<nnfusion::graph::GNode> pad_value_node = nullptr;
        for (const auto& in_edge : gnode->get_in_edges())
        {
            if (in_edge->get_dst_input() == 1)
            {
                pad_value_node = in_edge->get_src();
                break;
            }
        }

        std::string expression;
        if (auto constant_op =
                std::dynamic_pointer_cast<nnfusion::op::Constant>(pad_value_node->get_op_ptr()))
        {
            auto constant_values = constant_op->get_value_strings();
            NNFUSION_CHECK(1 == constant_values.size());

            expression = op::create_code_from_template(
                R"( - input("input0", @input_shape@); output(@output_shape@, topi=topi.nn.pad(args("input0"), pad_before=@pad_below@, pad_after=@pad_above@, pad_value=@pad_value@)); )",
                {{"input_shape", vector_to_string(gnode->get_input_shape(0))},
                 {"output_shape", vector_to_string(gnode->get_output_shape(0))},
                 {"pad_below", vector_to_string(op->get_padding_below())},
                 {"pad_above", vector_to_string(op->get_padding_above())},
                 {"pad_value", constant_values[0]}});
        }

        bool pad_zero = true;
        for (auto i : op->get_padding_below())
        {
            if (i != 0)
                pad_zero = false;
        }

        for (auto i : op->get_padding_above())
        {
            if (i != 0)
                pad_zero = false;
        }

        if (pad_zero)
        {
            expression += " ## @annotation: memcpy";
        }
        return expression;
    })
    */
    .translate_v2([](std::shared_ptr<graph::GNode> curr) -> std::string {

        NNFUSION_CHECK(2 == curr->get_input_size());
        auto op = static_pointer_cast<nnfusion::op::Pad>(curr->get_op_ptr());
        NNFUSION_CHECK_NOT_NULLPTR(op) << "Node type is not " << curr->get_op_ptr()->get_op_type();
        std::shared_ptr<nnfusion::graph::GNode> pad_value_node = nullptr;
        for (const auto& in_edge : curr->get_in_edges())
        {
            if (in_edge->get_dst_input() == 1)
            {
                pad_value_node = in_edge->get_src();
                break;
            }
        }

        std::string expression;
        if (auto constant_op =
                std::dynamic_pointer_cast<nnfusion::op::Constant>(pad_value_node->get_op_ptr()))
        {
            auto constant_values = constant_op->get_value_strings();
            NNFUSION_CHECK(1 == constant_values.size());
        }

        auto ir_template =
            R"( @output0@@output0_layout@ = @input0@@input0_layout@@conditions@; )";

        auto input0_shape = curr->get_input_shape(0);
        auto output0_shape = curr->get_output_shape(0);
        auto output0_layout = op::create_layout_from_dims(output0_shape);
        auto padding_below = op->get_padding_below();
        auto padding_above = op->get_padding_above();
        auto padding_interior = op->get_padding_interior();

        std::vector<std::string> input0_layout;
        std::string conditions;
        std::string when_condition;
        std::string where_condition;

        NNFUSION_CHECK(padding_below.size() == padding_above.size());
        for (int d = 0; d < padding_below.size(); ++d)
        {
            if (padding_below[d] > 0)
            {
                std::string in = "-" + to_string(padding_below[d]) + " + " + output0_layout[d];
                input0_layout.push_back(in);

                when_condition += (when_condition.empty() ? "" : " , ") + in + " >= 0";
            }
            else
            {
                input0_layout.push_back(output0_layout[d]);
            }

            if (padding_above[d] > 0)
            {
                when_condition += (when_condition.empty() ? "" : " , ") + input0_layout[d] + " < " +
                                  to_string(input0_shape[d]);
            }
            where_condition += (where_condition.empty() ? "" : " , ") + output0_layout[d] + " in " +
                               to_string(output0_shape[d]);
        }
        if (!when_condition.empty())
        {
            when_condition = ".when([" + when_condition + "], const(0).cast(@input0@.dtype()))";
        }
        if (!where_condition.empty())
        {
            where_condition = " where " + where_condition;
        }

        conditions = when_condition + where_condition;

        op::OpConfig::any op_config;
        op_config["input0_layout"] = vector_to_string<std::vector<std::string>>(input0_layout);
        op_config["conditions"] = conditions;
        op_config["output0_layout"] = vector_to_string<std::vector<std::string>>(output0_layout);

        expression = op::create_code_from_template(ir_template, op_config);
        bool pad_zero = true;
        for (auto i : op->get_padding_below())
        {
            if (i != 0)
                pad_zero = false;
        }

        for (auto i : op->get_padding_above())
        {
            if (i != 0)
                pad_zero = false;
        }

        if (pad_zero)
        {
            expression += " ## @: memcpy";
        }
        return expression;
    });
