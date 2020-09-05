// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "nnfusion/core/operators/generic_op/generic_op.hpp"

REGISTER_OP(LayerNorm)
    .attr<int>("axis", -1)
    .attr<float>("epsilon", 1e-6)
    .infershape([](std::shared_ptr<graph::GNode> gnode) -> void {
        NNFUSION_CHECK(gnode->get_input_size() == 3);
        const nnfusion::Shape& input_shape_0 = gnode->get_input_shape(0);
        const nnfusion::Shape& input_shape_1 = gnode->get_input_shape(1);
        const nnfusion::Shape& input_shape_2 = gnode->get_input_shape(2);
        const size_t input_0_dims = input_shape_0.size();

        auto generic_op = std::dynamic_pointer_cast<nnfusion::op::GenericOp>(gnode->get_op_ptr());
        int axis = generic_op->localOpConfig.getRoot()["axis"];
        axis += axis < 0 ? input_0_dims : 0;

        NNFUSION_CHECK(input_shape_1 == input_shape_2);
        NNFUSION_CHECK(input_shape_1.size() == input_0_dims - axis);
        NNFUSION_CHECK(
            std::equal(input_shape_1.begin(), input_shape_1.end(), input_shape_0.begin() + axis));

        nnfusion::Shape mean_var_shape;
        for (int i = 0; i < axis; ++i)
        {
            mean_var_shape.push_back(input_shape_0[i]);
        }
        mean_var_shape.push_back(1);

        gnode->set_output_type_and_shape(0, gnode->get_input_element_type(0), input_shape_0);
        gnode->set_output_type_and_shape(1, gnode->get_input_element_type(0), mean_var_shape);
        gnode->set_output_type_and_shape(2, gnode->get_input_element_type(0), mean_var_shape);
    });
