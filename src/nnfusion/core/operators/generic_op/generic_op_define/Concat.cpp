// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "nnfusion/core/operators/generic_op/generic_op.hpp"

REGISTER_OP(Concat)
    .infershape(nnfusion::op::infershape::unimplemented_and_not_used)
    .translate([](std::shared_ptr<graph::GNode> gnode) -> std::string {
        auto op = static_pointer_cast<nnfusion::op::Concat>(gnode->get_op_ptr());
        NNFUSION_CHECK_NOT_NULLPTR(op) << "Node type is not " << gnode->get_op_ptr()->get_op_type();

        size_t axis = op->get_concatenation_axis();

        std::stringstream expression;
        expression << "- ";
        size_t num_inputs = gnode->get_input_size();
        for (size_t i = 0; i < num_inputs; ++i)
        {
            expression << "input(\"input" << i << "\", "
                       << vector_to_string(gnode->get_input_shape(i)) << "); ";
        }
        expression << "output(" << vector_to_string(gnode->get_output_shape(0))
                   << ", topi=topi.concatenate([";
        for (size_t i = 0; i < num_inputs - 1; ++i)
        {
            expression << "args(\"input" << i << "\"), ";
        }
        expression << "args(\"input" << (num_inputs - 1) << "\")], ";
        expression << "axis=" << axis << "));";

        return expression.str();

    })
    .translate_v2([](std::shared_ptr<graph::GNode> curr) -> std::string {
        auto expression_template =
            R"( @output0@@output0_layout@ = @multi_inputs_body@ where @target_axis@ in @target_size@; )";

        auto op = static_pointer_cast<nnfusion::op::Concat>(curr->get_op_ptr());
        NNFUSION_CHECK_NOT_NULLPTR(op) << "Node type is not " << curr->get_op_ptr()->get_op_type();

        size_t axis = op->get_concatenation_axis();
        auto data_layout = op::create_layout_from_dims(curr->get_output_shape(0));

        auto offset = 0;
        auto recursive_input_template =
            R"( @input@@input_layout@.when(@dim@ < @offset@, @recursive@) )";
        auto final_input_template = R"(@input@@input_layout@)";
        std::string inputs_body = R"(@recursive@)";
        for (int in_id = 0; in_id < curr->get_input_size(); ++in_id)
        {
            std::vector<std::string> in_data_layout(data_layout);
            in_data_layout[axis] = in_data_layout[axis] + " - " + to_string(offset);
            auto dim_size = curr->get_input_shape(in_id)[axis];
            offset += dim_size;

            op::OpConfig::any in_config;
            in_config["input"] = "@input" + to_string(in_id) + "@";
            in_config["input_layout"] = vector_to_string<std::vector<std::string>>(in_data_layout);
            in_config["dim"] = data_layout[axis];
            in_config["offset"] = offset;

            std::string cur_body;
            if (in_id != curr->get_input_size() - 1)
                cur_body = op::create_code_from_template(recursive_input_template, in_config);
            else
                cur_body = op::create_code_from_template(final_input_template, in_config);

            inputs_body = op::create_code_from_template(inputs_body, {{"recursive", cur_body}});
        }

        auto expression_code = op::create_code_from_template(
            expression_template,
            {{"output0_layout", vector_to_string<std::vector<std::string>>(data_layout)},
             {"multi_inputs_body", inputs_body},
             {"target_axis", data_layout[axis]},
             {"target_size", to_string(curr->get_output_shape(0)[axis])}});

        return expression_code;
    });
