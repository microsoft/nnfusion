// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "nnfusion/core/operators/generic_op/generic_op.hpp"

REGISTER_OP(BatchNormInference)
    .infershape(nnfusion::op::infershape::unimplemented_and_not_used)
    .translate_v2([](std::shared_ptr<graph::GNode> curr) -> std::string {
        auto input_shape_0 = curr->get_input_shape(0);
        auto input_shape_1 = curr->get_input_shape(1);
        auto input_shape_2 = curr->get_input_shape(2);
        auto input_shape_3 = curr->get_input_shape(3);
        auto input_shape_4 = curr->get_input_shape(4);
        auto output_shape_0 = curr->get_output_shape(0);

        NNFUSION_CHECK(output_shape_0 == input_shape_2);
        NNFUSION_CHECK(output_shape_0.size() == 4);
        NNFUSION_CHECK(input_shape_0 == input_shape_1);
        NNFUSION_CHECK(input_shape_3 == input_shape_4);
        NNFUSION_CHECK(input_shape_0 == input_shape_3);
        NNFUSION_CHECK(input_shape_0.size() == 1);
        NNFUSION_CHECK(input_shape_0[0] == input_shape_2[1]);

        auto op = static_pointer_cast<nnfusion::op::BatchNormInference>(curr->get_op_ptr());
        string dtype;
        NNFUSION_CHECK(
            element::Type::nnfusion_element_type_to_dtype_string(curr->get_element_type(), dtype));
        auto epsilon = "const(" + std::to_string(op->get_eps_value()) + ").cast(`" + dtype + "`)";
        auto expression =
            "@output0@[N, C, H, W] = @input1@[C] + @input0@[C] * (@input2@[N, C, H, W] - "
            "@input3@[C]) / (" +
            epsilon + " + @input4@[C]).call(`sqrt`);";
        return expression;
    });
