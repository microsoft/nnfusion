// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "nnfusion/core/operators/generic_op/generic_op.hpp"

REGISTER_OP(MatMulAdd)
    .attr<bool>("trans_A", false)
    .attr<bool>("trans_B", false)
    .infershape([](std::shared_ptr<graph::GNode> gnode) -> void {
        //AB + C
        auto generic_op = static_pointer_cast<nnfusion::op::GenericOp>(gnode->get_op_ptr());
        NNFUSION_CHECK_NOT_NULLPTR(generic_op) << "Node type is not "
                                               << gnode->get_op_ptr()->get_op_type();

        NNFUSION_CHECK(gnode->get_input_size() == 3) << "Inputs of MatMulAdd operator should be 3.";
        auto A_shape = gnode->get_input_shape(0);
        auto B_shape = gnode->get_input_shape(1);
        auto C_shape = gnode->get_input_shape(2);
        NNFUSION_CHECK(A_shape.size() == 2 && B_shape.size() == 2 && C_shape.size() == 2)
            << "MatMulAdd only support Matrix";

        auto& cfg = generic_op->localOpConfig.getRoot();
        bool trans_A = cfg["trans_A"];
        bool trans_B = cfg["trans_B"];

        size_t m = trans_A ? A_shape[1] : A_shape[0];
        size_t k1 = trans_A ? A_shape[0] : A_shape[1];
        size_t n = trans_B ? B_shape[0] : B_shape[1];
        size_t k2 = trans_B ? B_shape[1] : B_shape[0];
        NNFUSION_CHECK(k1 == k2);
        NNFUSION_CHECK(C_shape[0] == m && C_shape[1] == n);

        gnode->set_output_type_and_shape(
            0, gnode->get_input_element_type(2), gnode->get_input_shape(2));
    });
