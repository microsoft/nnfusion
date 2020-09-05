// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "nnfusion/core/operators/generic_op/generic_op.hpp"

REGISTER_OP(DropoutTraining)
    .attr<float>("ratio")
    .infershape([](std::shared_ptr<graph::GNode> gnode) -> void {
        gnode->set_output_type_and_shape(
            0, gnode->get_input_element_type(0), gnode->get_input_shape(0)); // output
        gnode->set_output_type_and_shape(
            1, nnfusion::element::boolean, gnode->get_input_shape(0)); // mask
    });

REGISTER_OP(DropoutTrainingGrad)
    .attr<float>("ratio")
    .infershape([](std::shared_ptr<graph::GNode> gnode) -> void {
        gnode->set_output_type_and_shape(
            0, gnode->get_input_element_type(0), gnode->get_input_shape(0)); // dx
    });
