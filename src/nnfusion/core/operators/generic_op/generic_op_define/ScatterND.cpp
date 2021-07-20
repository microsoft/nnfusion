// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "nnfusion/core/operators/generic_op/generic_op.hpp"

REGISTER_OP(ScatterND).infershape([](std::shared_ptr<graph::GNode> gnode) -> void {
    NNFUSION_CHECK(gnode->get_input_size() == 3);

    gnode->set_output_type_and_shape(
        0, gnode->get_input_element_type(0), gnode->get_input_shape(0));
});