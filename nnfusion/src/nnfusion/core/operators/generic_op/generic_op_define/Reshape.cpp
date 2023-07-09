// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "nnfusion/core/operators/generic_op/generic_op.hpp"

REGISTER_OP(Reshape)
    .infershape(nnfusion::op::infershape::unimplemented_and_not_used)
    /*
    .translate([](std::shared_ptr<graph::GNode> gnode) -> std::string {
        auto op = static_pointer_cast<nnfusion::op::Reshape>(gnode->get_op_ptr());
        NNFUSION_CHECK_NOT_NULLPTR(op) << "Node type is not " << gnode->get_op_ptr()->get_op_type();

        std::string expression;
        bool memcpy_annotation = false;
        if (op->get_is_transpose())
        {
            const auto& input_order = op->get_input_order();
            expression = op::create_code_from_template(
                R"( - input("input0", @input_shape@); output(@output_shape@, topi=topi.transpose(args("input0"), axes=@input_order@)); )",
                {{"input_shape", vector_to_string(gnode->get_input_shape(0))},
                 {"output_shape", vector_to_string(gnode->get_output_shape(0))},
                 {"input_order", vector_to_string(input_order)}});
            for (int i = 0; i < input_order.size(); ++i)
                if (input_order[i] != i)
                    break;
                else if (i + 1 == input_order.size())
                    memcpy_annotation = true;
        }
        else
        {
            // For cases with "is_transpose==false", the ReshapeMemcpy kernel will be selected.
            size_t total_size = 1L;
            for (auto it : gnode->get_input_shape(0))
                total_size *= it;
            expression = op::create_code_from_template(
                R"( - input("input0", @input_shape@); output(@output_shape@, lambda i: args("input0")[i]); )",
                {{"input_shape", "[" + std::to_string(total_size) + "]"},
                 {"output_shape", "[" + std::to_string(total_size) + "]"}});
            memcpy_annotation = true;
        }
        return expression + (memcpy_annotation ? " ## @annotation: memcpy" : "");
    })
    */
    .translate_v2([](std::shared_ptr<graph::GNode> curr) -> std::string {
        auto expression_template =
            R"( @output0@@output0_layout@ = @input0@@input0_layout@ @conditions@; )";

        auto op = static_pointer_cast<nnfusion::op::Reshape>(curr->get_op_ptr());
        NNFUSION_CHECK_NOT_NULLPTR(op) << "Node type is not " << curr->get_op_ptr()->get_op_type();

        std::string expression_code;
        bool memcpy_annotation = false;
        if (op->get_is_transpose())
        {
            op::OpConfig::any config;
            const auto& input_order = op->get_input_order();
            auto input_layout = op::create_layout_from_dims(curr->get_input_shape(0));
            config["input0_layout"] = vector_to_string<std::vector<std::string>>(input_layout);
            std::vector<std::string> output_layout;
            for (int d = 0; d < input_order.size(); d++)
            {
                output_layout.push_back(input_layout[input_order[d]]);
            }
            config["output0_layout"] = vector_to_string<std::vector<std::string>>(output_layout);
            config["conditions"] = "";
            expression_code = op::create_code_from_template(expression_template, config);
            for (int i = 0; i < input_order.size(); ++i)
                if (input_order[i] != i)
                    break;
                else if (i + 1 == input_order.size())
                    memcpy_annotation = true;
        }
        else
        {
            std::string cond = "";

            auto output_shape = curr->get_output_shape(0);
            auto input_shape = curr->get_input_shape(0);
            auto output_layout = op::create_layout_from_dims(output_shape);
            auto input_layout = std::vector<std::string>(input_shape.size());

            int m_index = 0;
            for (int dim_index = 0; dim_index < input_shape.size(); dim_index++)
                input_layout[dim_index] =
                    input_shape[dim_index] == 1 ? "0" : input_layout[dim_index];

            int acc_in_shape = 1;
            int acc_in_index = -1;
            int acc_out_shape = 1;
            std::vector<size_t> acc_in_pairs;
            std::vector<size_t> acc_out_pairs;
            for (int dim_index = 0; dim_index < output_shape.size(); dim_index++)
            {
                if (output_shape[dim_index] == 1)
                    continue;
                acc_out_shape *= output_shape[dim_index];
                acc_out_pairs.push_back(dim_index);

                if (acc_out_shape > acc_in_shape)
                {
                    while (acc_out_shape > acc_in_shape)
                    {
                        acc_in_index += 1;
                        if (input_shape[acc_in_index] == 1)
                            continue;
                        acc_in_shape *= input_shape[acc_in_index];
                        acc_in_pairs.push_back(acc_in_index);
                    }
                }

                if (acc_out_shape == acc_in_shape)
                {
                    std::string in_mul_str = "";
                    for (auto acc_out : acc_out_pairs)
                    {
                        in_mul_str =
                            (in_mul_str.empty() ? output_layout[acc_out]
                                                : ("(" + in_mul_str + ")" + " * " +
                                                   to_string(output_shape[acc_out]) + " + ") +
                                                      output_layout[acc_out]);
                    }
                    for (int i_index = 0; i_index < acc_in_pairs.size(); i_index++)
                    {
                        int mod = 1;
                        for (int t_index = i_index + 1; t_index < acc_in_pairs.size(); t_index++)
                        {
                            mod *= input_shape[acc_in_pairs[t_index]];
                        }
                        std::string o_layout_str =
                            acc_out_pairs.size() == 1 ? in_mul_str : ("(" + in_mul_str + ")");
                        o_layout_str += acc_in_pairs.size() == 1
                                            ? ""
                                            : (" / " + to_string(mod) + " % " +
                                               to_string(input_shape[acc_in_pairs[i_index]]));
                        input_layout[acc_in_pairs[i_index]] = o_layout_str;
                    }
                    acc_in_shape = 1;
                    acc_out_shape = 1;
                    acc_in_pairs.clear();
                    acc_out_pairs.clear();
                }
            }

            std::unordered_set<std::string> def_keys;
            for (int dim_index = 0; dim_index < input_layout.size(); dim_index++)
            {
                def_keys.insert(input_layout[dim_index]);
            }
            for (int dim_index = 0; dim_index < output_shape.size(); dim_index++)
            {
                if (def_keys.find(output_layout[dim_index]) != def_keys.end())
                    continue;
                cond = cond + (cond.empty() ? "where " : ", ") + output_layout[dim_index] + " in " +
                       to_string(output_shape[dim_index]);
            }

            expression_code = op::create_code_from_template(
                expression_template,
                {{"output0_layout", vector_to_string<std::vector<std::string>>(output_layout)},
                 {"input0_layout", vector_to_string<std::vector<std::string>>(input_layout)},
                 {"conditions", cond}});

            memcpy_annotation = true;
        }
        return expression_code + (memcpy_annotation ? " ## @: memcpy" : "");
    });
