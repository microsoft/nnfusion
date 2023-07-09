// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "nnfusion/core/operators/generic_op/generic_op.hpp"

REGISTER_OP(Zeros).attr<std::vector<int>>("shape").infershape(
    [](std::shared_ptr<graph::GNode> gnode) -> void {
        NNFUSION_CHECK(gnode->get_input_size() == 0);
        auto d_type = nnfusion::element::f32; // hardcode f32
        auto generic_op = std::dynamic_pointer_cast<nnfusion::op::GenericOp>(gnode->get_op_ptr());
        std::vector<int> out_shape = generic_op->localOpConfig.getRoot()["shape"];
        gnode->set_output_type_and_shape(0, d_type, Shape(out_shape.begin(), out_shape.end()));
    });
