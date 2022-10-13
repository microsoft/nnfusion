// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "nnfusion/core/operators/generic_op/generic_op.hpp"

REGISTER_OP(SelectNode)
    .attr<int>("index", 0) // regularization
    .infershape([](std::shared_ptr<graph::GNode> gnode) -> void {
        auto generic_op = std::dynamic_pointer_cast<nnfusion::op::GenericOp>(gnode->get_op_ptr());
        int node_index = generic_op->localOpConfig.getRoot()["index"];
        NNFUSION_CHECK(gnode->get_input_size() >= node_index) << "SelectNode out of input range";
        gnode->set_output_type_and_shape(
            0, gnode->get_input_element_type(node_index), gnode->get_input_shape(node_index));
        // gnode->set_output_type_and_shape(
        //     0, nnfusion::element::f32, gnode->get_input_shape(node_index));
    });