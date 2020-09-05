// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "nnfusion/core/operators/generic_op/generic_op.hpp"

REGISTER_OP(SparseApplyMomentum)
    .attr<bool>("use_nesterov", false)
    .attr<float>("lr", 0.001)
    .attr<float>("momentum", 0.001)
    .attr<std::vector<int64_t>>("indices")
    .attr<nnfusion::op::OpConfig::any>("Tindices", "int32_t")
    .infershape([](std::shared_ptr<graph::GNode> gnode) -> void {

        NNFUSION_CHECK(gnode->get_input_size() == 3)
            << "Inputs of SparseApplyMomentum operator should be 3.";

        auto& var = gnode->get_input_shape(0);
        auto& accum = gnode->get_input_shape(1);
        auto& grad = gnode->get_input_shape(2);

        auto generic_op = std::dynamic_pointer_cast<nnfusion::op::GenericOp>(gnode->get_op_ptr());
        auto& indices = generic_op->localOpConfig.getRoot()["indices"];

        NNFUSION_CHECK(nnfusion::is_vector_or_higher(var)) << "var must be at least 1 dimensional";
        NNFUSION_CHECK(var.size() == accum.size())
            << "var and accum do not have the same shape dimention size";
        for (int i = 0; i < var.size(); i++)
        {
            NNFUSION_CHECK(var[i] == accum[i]) << "var and accum must match in dimension " << i;
        }
        NNFUSION_CHECK(var.size() == grad.size()) << "var and grad do not have the same dimentions";
        for (int i = 1; i < var.size(); i++)
        {
            NNFUSION_CHECK(var[i] == grad[i]) << "var and grad must match in dimension " << i;
        }
        NNFUSION_CHECK(grad[0] == indices.size())
            << "grad must be the same size as indices in the first dimension";

        nnfusion::Shape output_shape_0(var);
        gnode->set_output_type_and_shape(0, gnode->get_input_element_type(0), output_shape_0);
    });
