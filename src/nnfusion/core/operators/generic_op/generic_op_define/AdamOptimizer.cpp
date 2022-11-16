// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "nnfusion/core/operators/generic_op/generic_op.hpp"

REGISTER_OP(AdamOptimizer)
    .attr<float>("lambda", 0.01) // regularization
    .attr<float>("epsilon", 1e-6)
    .attr<float>("alpha", 0.9)
    .attr<float>("beta", 0.999)
    .infershape([](std::shared_ptr<graph::GNode> gnode) -> void {
        // input: learn_rate, step, weight, grad, m1, m2
        // output: step, m1, m2, weight, grad
        NNFUSION_CHECK(gnode->get_input_size() >= 6)
            << "AdamOptimizer should have at least 6 inputs.";

        NNFUSION_CHECK(gnode->get_input_shape(2) == gnode->get_input_shape(3) &&
                       gnode->get_input_shape(3) == gnode->get_input_shape(4) &&
                       gnode->get_input_shape(4) == gnode->get_input_shape(5))
            << "weight, grad and momentum do not have the same shape";

        NNFUSION_CHECK(shape_size(gnode->get_input_shape(0)) == 1 &&
                       shape_size(gnode->get_input_shape(1)) == 1);

        gnode->set_output_type_and_shape(
            0, gnode->get_input_element_type(1), gnode->get_input_shape(1));
        gnode->set_output_type_and_shape(
            1, gnode->get_input_element_type(4), gnode->get_input_shape(4));
        gnode->set_output_type_and_shape(
            2, gnode->get_input_element_type(5), gnode->get_input_shape(5));
        gnode->set_output_type_and_shape(
            3, gnode->get_input_element_type(2), gnode->get_input_shape(2));
        gnode->set_output_type_and_shape(
            4, gnode->get_input_element_type(3), gnode->get_input_shape(3));
    });
