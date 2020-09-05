// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "nnfusion/core/operators/generic_op/generic_op.hpp"

REGISTER_OP(ReverseSequence)
    .attr<int64_t>("batch_axis", 0)
    .attr<int64_t>("seq_axis", 0)
    .infershape([](std::shared_ptr<graph::GNode> gnode) -> void {
        auto generic_op = std::dynamic_pointer_cast<nnfusion::op::GenericOp>(gnode->get_op_ptr());
        auto data_type = gnode->get_input_element_type(0);
        auto data_shape = gnode->get_input_shape(0);
        gnode->set_output_type_and_shape(0, data_type, data_shape);
    });