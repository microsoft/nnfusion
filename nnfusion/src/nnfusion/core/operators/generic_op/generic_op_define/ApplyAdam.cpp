// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "nnfusion/core/operators/generic_op/generic_op.hpp"

REGISTER_OP(ApplyAdam)
    .attr<float>("beta1", 0.1)
    .attr<float>("beta2", 0.1)
    .attr<float>("beta1_pow", 0.01)
    .attr<float>("beta2_pow", 0.01)
    .attr<float>("epsilon", 0.1)
    .infershape([](std::shared_ptr<graph::GNode> gnode) -> void {
        gnode->set_output_type_and_shape(
            0, gnode->get_input_element_type(0), gnode->get_input_shape(0));
    });
