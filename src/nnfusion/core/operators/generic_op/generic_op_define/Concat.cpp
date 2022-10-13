// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "nnfusion/core/operators/generic_op/generic_op.hpp"
DECLARE_bool(fsymbolic);
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

        size_t num_valid_inputs = 0;
        for (int in_id = 0; in_id < curr->get_input_size(); ++in_id)
            if (curr->get_input_shape(in_id)[axis] > 0)
                num_valid_inputs++;

        size_t processed_inputs = 0;
        SymDim sym_offset(0);
        for (int in_id = 0; in_id < curr->get_input_size(); ++in_id)
        {
            std::vector<std::string> in_data_layout(data_layout);
            in_data_layout[axis] = in_data_layout[axis] + " - " +
                                   (sym_offset.is_dynamic() ? "(" + sym_offset.sym() + ")"
                                                            : std::to_string(sym_offset.max()));
            auto in_shape = curr->get_input_shape(in_id);
            //auto dim_size = curr->get_input_shape(in_id)[axis];
            SymDim dim_size;
            if (in_shape.is_dynamic() && in_shape.get_sym_shape()->at(axis).is_dynamic())
            {
                auto dim = in_shape.get_sym_shape()->at(axis);
                dim_size = SymDim("alter(`" + dim.sym() + ":" + std::to_string(dim.max()) + "`)",
                                  dim.min(),
                                  dim.max());
            }
            else
            {
                dim_size = SymDim(in_shape[axis]);
            }
            if (dim_size.max() == 0)
                continue;
            //offset += dim_size;
            sym_offset += dim_size;

            op::OpConfig::any in_config;
            in_config["input"] = "@input" + to_string(in_id) + "@";
            in_config["input_layout"] = vector_to_string<std::vector<std::string>>(in_data_layout);
            in_config["dim"] = data_layout[axis];
            in_config["offset"] = sym_offset.is_dynamic() ? "(" + sym_offset.sym() + ")"
                                                          : std::to_string(sym_offset.max());
            processed_inputs++;

            std::string cur_body;
            if (processed_inputs < num_valid_inputs)
                cur_body = op::create_code_from_template(recursive_input_template, in_config);
            else
                cur_body = op::create_code_from_template(final_input_template, in_config);

            inputs_body = op::create_code_from_template(inputs_body, {{"recursive", cur_body}});
        }
        auto target_size_str = to_string(curr->get_output_shape(0)[axis]);
        if (FLAGS_fsymbolic && curr->get_output_shape(0).get_sym_shape() &&
            curr->get_output_shape(0).get_sym_shape()->at(axis).is_dynamic())
        {
            target_size_str = curr->get_output_shape(0).get_sym_shape()->at(axis).to_string();
            target_size_str = target_size_str.substr(1, target_size_str.size() - 2);
        }
        auto expression_code = op::create_code_from_template(
            expression_template,
            {{"output0_layout", vector_to_string<std::vector<std::string>>(data_layout)},
             {"multi_inputs_body", inputs_body},
             {"target_axis", data_layout[axis]},
             {"target_size", target_size_str}});

        return expression_code;
    });
