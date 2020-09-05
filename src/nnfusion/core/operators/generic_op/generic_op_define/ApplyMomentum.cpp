// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "nnfusion/core/operators/generic_op/generic_op.hpp"

REGISTER_OP(ApplyMomentum)
    .attr<bool>("use_nesterov", false)
    .attr<float>("lr", 0.001)
    .attr<float>("momentum", 0.001)
    .infershape([](std::shared_ptr<graph::GNode> gnode) -> void {

        NNFUSION_CHECK(gnode->get_input_size() == 3)
            << "Inputs of ApplyMomentum operator should be 3.";

        auto& var = gnode->get_input_shape(0);
        auto& accum = gnode->get_input_shape(1);
        auto& grad = gnode->get_input_shape(2);

        NNFUSION_CHECK(var.size() == accum.size())
            << "var and accum do not have the same shape dimention size";
        for (int i = 0; i < var.size(); i++)
        {
            NNFUSION_CHECK(var[i] == accum[i]) << "var and accum must match in dimension " << i;
        }
        NNFUSION_CHECK(var.size() == grad.size())
            << "var and grad do not have the same shape dimention size";
        for (int i = 0; i < var.size(); i++)
        {
            NNFUSION_CHECK(var[i] == grad[i]) << "var and grad must match in dimension " << i;
        }

        nnfusion::Shape output_shape_0(var);
        gnode->set_output_type_and_shape(0, gnode->get_input_element_type(0), output_shape_0);
    });
