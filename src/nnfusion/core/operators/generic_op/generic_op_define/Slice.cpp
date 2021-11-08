// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "nnfusion/core/operators/generic_op/generic_op.hpp"

REGISTER_OP(Slice)
    .infershape(nnfusion::op::infershape::unimplemented_and_not_used)
    .translate([](std::shared_ptr<graph::GNode> gnode) -> std::string {
        auto op = static_pointer_cast<nnfusion::op::Slice>(gnode->get_op_ptr());
        NNFUSION_CHECK_NOT_NULLPTR(op) << "Node type is not " << gnode->get_op_ptr()->get_op_type();

        return op::create_code_from_template(
            R"( - input("input0", @input_shape@); output(@output_shape@, topi=topi.strided_slice(args("input0"), begin=@begin@, end=@end@, strides=@strides@)); )",
            {{"input_shape", vector_to_string(gnode->get_input_shape(0))},
             {"output_shape", vector_to_string(gnode->get_output_shape(0))},
             {"begin", vector_to_string(op->get_lower_bounds())},
             {"end", vector_to_string(op->get_upper_bounds())},
             {"strides", vector_to_string(op->get_strides())}});

    })
    .translate_v2([](std::shared_ptr<graph::GNode> curr) -> std::string {
        auto expression_template =
            R"( @output0@@output0_layout@ = @input0@@input0_layout@ where @slice_dims@; )";

        auto op = static_pointer_cast<nnfusion::op::Slice>(curr->get_op_ptr());
        NNFUSION_CHECK_NOT_NULLPTR(op) << "Node type is not " << curr->get_op_ptr()->get_op_type();
        for (auto& s : op->get_strides())
        {
            if (s == 1)
                continue;
            std::cout << "Slice operation with strides larger than 1 is not supported by Antares "
                      << std::endl;
            return "";
        }

        auto output_layout = op::create_layout_from_dims(curr->get_output_shape(0));
        std::string slice_dims;
        std::vector<std::string> input_layout;
        for (int d = 0; d < output_layout.size(); d++)
        {
            input_layout.push_back(output_layout[d] + " + " + to_string(op->get_lower_bounds()[d]));
            slice_dims = slice_dims + (slice_dims.empty() ? "" : " , ") + output_layout[d] +
                         " in " + to_string(curr->get_output_shape(0)[d]);
        }

        auto expression_code = op::create_code_from_template(
            expression_template,
            {{"output0_layout", vector_to_string<std::vector<std::string>>(output_layout)},
             {"input0_layout", vector_to_string<std::vector<std::string>>(input_layout)},
             {"slice_dims", slice_dims}});
        return expression_code;

    });