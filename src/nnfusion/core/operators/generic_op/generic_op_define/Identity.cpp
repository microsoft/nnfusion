// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "nnfusion/core/operators/generic_op/generic_op.hpp"

REGISTER_OP(Identity)
.infershape(
    [](std::shared_ptr<graph::GNode> gnode) -> void {
        NNFUSION_CHECK(gnode->get_input_size() == 1);
        gnode->set_output_type_and_shape(
            0, gnode->get_input_element_type(0), gnode->get_input_shape(0));
    })
.translate_v2([](std::shared_ptr<graph::GNode> curr) -> std::string {

        auto generic_op = std::dynamic_pointer_cast<nnfusion::op::GenericOp>(curr->get_op_ptr());
        auto data_layout = op::create_layout_from_dims(curr->get_output_shape(0));
        auto expression_template =
            R"(@output0@@data_layout@ = @input0@@data_layout@)";

        std::string expression_code = op::create_code_from_template(
            expression_template,
            { {"data_layout", vector_to_string<std::vector<std::string>>(data_layout)}});
        return expression_code;
    });