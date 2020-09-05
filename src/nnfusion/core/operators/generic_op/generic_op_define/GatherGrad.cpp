// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "nnfusion/core/operators/generic_op/generic_op.hpp"

REGISTER_OP(GatherGrad)
    .attr<int>("axis", 0)
    .attr<std::vector<int>>("x_shape")
    .infershape([](std::shared_ptr<graph::GNode> gnode) -> void {
        NNFUSION_CHECK(gnode->get_input_size() == 3);
        auto x_grad_type = gnode->get_input_element_type(1);

        auto generic_op = std::dynamic_pointer_cast<nnfusion::op::GenericOp>(gnode->get_op_ptr());
        std::vector<int> x_shape = generic_op->localOpConfig.getRoot()["x_shape"];

        nnfusion::Shape output_shape_0;

        gnode->set_output_type_and_shape(0, x_grad_type, Shape(x_shape.begin(), x_shape.end()));
    });
