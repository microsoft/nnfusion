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
            auto output_shape = curr->get_output_shape(0).empty() ? nnfusion::Shape({1})
                                                                  : curr->get_output_shape(0);
            auto input_shape =
                curr->get_input_shape(0).empty() ? nnfusion::Shape({1}) : curr->get_input_shape(0);
            auto output_layout = op::create_layout_from_dims(output_shape);
            std::set<int> declear_dims;
            std::vector<std::string> input_layout;
            int in_multiplier = 1;
            int out_multiplier = 1;
            int in_index = 0;
            int out_index = 0;
            while (in_index < input_shape.size() || out_index < output_shape.size())
            {
                auto in_dim = in_index < input_shape.size() ? input_shape[in_index] : 0;
                auto out_dim = out_index < output_shape.size() ? output_shape[out_index] : 0;
                if (in_multiplier == 1 && out_multiplier == 1)
                {
                    if (out_dim != in_dim)
                    {
                        in_multiplier = out_dim > in_dim ? 1 : in_dim;
                        out_multiplier = out_dim > in_dim ? out_dim : 1;
                    }
                    else
                    {
                        input_layout.push_back(output_layout[out_index]);
                        in_index++;
                        out_index++;
                        continue;
                    }
                }

                NNFUSION_CHECK(in_multiplier == 1 || out_multiplier == 1)
                    << "Only single-way reshape is suuported now!";

                if (out_dim && (in_multiplier / out_dim))
                {
                    in_multiplier /= out_dim;
                    // declear_dims.insert(out_index);
                    auto r_in_index = std::min(in_index, int(input_shape.size()) - 1);
                    auto r_out_index = std::min(out_index, int(output_shape.size()) - 1);
                    if (input_layout.size() <= r_in_index)
                        input_layout.push_back("");
                    input_layout[r_in_index] =
                        input_layout[r_in_index] + (input_layout[r_in_index].empty() ? "" : " + ") +
                        output_layout[r_out_index] + " * " + to_string(in_multiplier);
                    out_index++;
                    if (in_multiplier == 1)
                        in_index++;
                }

                if (in_dim && (out_multiplier / in_dim))
                {
                    out_multiplier /= in_dim;
                    // declear_dims.insert(out_index);
                    auto r_out_index = std::min(out_index, int(output_shape.size()) - 1);
                    input_layout.push_back(output_layout[r_out_index] + " // " +
                                           to_string(out_multiplier) + " % " + to_string(in_dim));
                    in_index++;
                    if (out_multiplier == 1)
                        out_index++;
                }
            }

            std::string condition;
            for (int d = 0; d < output_shape.size(); ++d)
            {
                condition = condition + (condition.empty() ? "" : " , ") + "N" + to_string(d) +
                            " in " + to_string(output_shape[d]);
            }
            if (!condition.empty())
                condition = "where " + condition;

            auto input_layout_str = curr->get_input_shape(0).empty()
                                        ? "[]"
                                        : vector_to_string<std::vector<std::string>>(input_layout);

            expression_code = op::create_code_from_template(
                expression_template,
                {{"output0_layout", vector_to_string<std::vector<std::string>>(output_layout)},
                 {"input0_layout", input_layout_str},
                 {"conditions", condition}});

            memcpy_annotation = true;
        }
        return expression_code + (memcpy_annotation ? " ## @: memcpy" : "");
    });
