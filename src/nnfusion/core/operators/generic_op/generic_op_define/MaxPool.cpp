// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "nnfusion/core/operators/generic_op/generic_op.hpp"

REGISTER_OP(MaxPool)
    .infershape(nnfusion::op::infershape::unimplemented_and_not_used)
    /*
    .translate([](std::shared_ptr<graph::GNode> curr) -> std::string {
        auto _op = static_pointer_cast<nnfusion::op::MaxPool>(curr->get_op_ptr());
        NNFUSION_CHECK_NOT_NULLPTR(_op) << "Node type is not " << curr->get_op_ptr()->get_op_type();

        const auto& kernel = _op->get_window_shape();
        const auto& stride = _op->get_window_movement_strides();
        const auto& padding_below = _op->get_padding_below();
        const auto& padding_above = _op->get_padding_above();
        uint64_t padding[] = {
            padding_below[1], padding_below[0], padding_above[1], padding_above[0]};

        return op::create_code_from_template(
            R"( - input("input0", @input_shape@); output(@output_shape@, topi=topi.nn.pool(args("input0"), kernel=@kernel@, stride=@stride@, padding=@padding@, pool_type="max")); )",
            {{"input_shape", vector_to_string(curr->get_input_shape(0))},
             {"output_shape", vector_to_string(curr->get_output_shape(0))},
             {"kernel", vector_to_string(kernel)},
             {"stride", vector_to_string(stride)},
             {"padding", vector_to_string(padding)}});
    })
    */
    .translate_v2([](std::shared_ptr<graph::GNode> curr) -> std::string {
        auto _op = static_pointer_cast<nnfusion::op::MaxPool>(curr->get_op_ptr());
        auto& input_shape = curr->get_input_shape(0);
        auto& output_shape = curr->get_output_shape(0);
        auto& dtype = curr->get_element_type();
        bool is_1d = (output_shape.size() == 3);
        const bool is_nchw = _op->get_data_format() == "NCHW" ? true : false;
        auto& m_strides = _op->get_window_movement_strides();
        auto& strides = _op->get_window_shape();
        auto& padding_below = _op->get_padding_below();
        auto& padding_above = _op->get_padding_above();

        if (!(padding_below.size() == padding_above.size()))
        {
            return std::string();
        }
        if (!(padding_below.size() >= 1))
        {
            return std::string();
        }
        if (!is_1d)
        {
            if (!(padding_below.size() == 2))
            {
                return std::string();
            }
        }

        auto expression_template =
            R"( @output0@@output0_layout@ >=! @input0@@input0_layout@@conditions@; )";

        std::string conditions;
        std::string when_condition;
        std::string where_condition;

        auto output_layout = std::vector<std::string>{"N"};
        auto input_layout = std::vector<std::string>{"N"};
        if (is_nchw)
        {
            output_layout.push_back("C");
            input_layout.push_back("C");
            output_layout.push_back("HO");
            input_layout.push_back("HO * " + to_string(m_strides[0]) + " + KH - " +
                                   to_string(padding_below[0]));
            if (padding_below[0] > 0)
            {
                when_condition += (when_condition.empty() ? "" : " , ") + input_layout[2] + " >=0";
            }
            if (padding_above[0] > 0)
            {
                when_condition += (when_condition.empty() ? "" : " , ") + input_layout[2] + " < " +
                                  to_string(input_shape[2]);
            }
            where_condition += (where_condition.empty() ? "" : " , ") + output_layout[2] + " in " +
                               to_string(output_shape[2]) + ", KH in " + to_string(strides[0]);
            if (!is_1d)
            {
                output_layout.push_back("WO");
                input_layout.push_back("WO * " + to_string(m_strides[1]) + " + KW - " +
                                       to_string(padding_below[1]));
                if (padding_below[0] > 0)
                {
                    when_condition +=
                        (when_condition.empty() ? "" : " , ") + input_layout[3] + " >=0";
                }
                if (padding_above[0] > 0)
                {
                    when_condition += (when_condition.empty() ? "" : " , ") + input_layout[3] +
                                      " < " + to_string(input_shape[3]);
                }
                where_condition += (where_condition.empty() ? "" : " , ") + output_layout[3] +
                                   " in " + to_string(output_shape[3]) + ", KW in " +
                                   to_string(strides[1]);
            }
        }
        else
        {
            output_layout.push_back("HO");
            input_layout.push_back("HO * " + to_string(m_strides[0]) + " + KH - " +
                                   to_string(padding_below[0]));
            if (padding_below[0] > 0)
            {
                when_condition += (when_condition.empty() ? "" : " , ") + input_layout[1] + " >=0";
            }
            if (padding_above[0] > 0)
            {
                when_condition += (when_condition.empty() ? "" : " , ") + input_layout[1] + " < " +
                                  to_string(input_shape[1]);
            }
            where_condition += (where_condition.empty() ? "" : " , ") + output_layout[1] + " in " +
                               to_string(output_shape[1]) + ", KH in " + to_string(strides[0]);
            if (!is_1d)
            {
                output_layout.push_back("WO");
                input_layout.push_back("WO * " + to_string(m_strides[1]) + " + KW - " +
                                       to_string(padding_below[1]));
                if (padding_below[0] > 0)
                {
                    when_condition +=
                        (when_condition.empty() ? "" : " , ") + input_layout[2] + " >=0";
                }
                if (padding_above[0] > 0)
                {
                    when_condition += (when_condition.empty() ? "" : " , ") + input_layout[2] +
                                      " < " + to_string(input_shape[2]);
                }
                where_condition += (where_condition.empty() ? "" : " , ") + output_layout[2] +
                                   " in " + to_string(output_shape[2]) + ", KW in " +
                                   to_string(strides[1]);
            }
            output_layout.push_back("C");
            input_layout.push_back("C");
        }

        if (!when_condition.empty())
        {
            std::string min_value;
            if (dtype == nnfusion::element::f32)
            {
                min_value = "-3.4e38";
            }
            else if (dtype == nnfusion::element::f16)
            {
                min_value = "-6.55e4";
            }
            else if (dtype == nnfusion::element::i8)
            {
                min_value = "-128";
            }
            else
            {
                NNFUSION_LOG(INFO) << "not support padding with data type " << dtype
                                   << " yet, fallback";
                return std::string();
            }
            when_condition = ".when([" + when_condition + "], " + min_value + ")";
        }
        if (!where_condition.empty())
        {
            where_condition = " where " + where_condition;
        }

        conditions = when_condition + where_condition;

        op::OpConfig::any config;
        config["conditions"] = conditions;
        config["output0_layout"] = vector_to_string<std::vector<std::string>>(output_layout);
        config["input0_layout"] = vector_to_string<std::vector<std::string>>(input_layout);

        auto expression_code = op::create_code_from_template(expression_template, config);
        return expression_code;
    });
