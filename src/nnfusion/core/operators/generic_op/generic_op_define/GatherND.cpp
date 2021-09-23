// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "nnfusion/core/operators/generic_op/generic_op.hpp"

REGISTER_OP(GatherND).attr<int>("axis", 0).infershape(
    [](std::shared_ptr<graph::GNode> gnode) -> void {
        NNFUSION_CHECK(gnode->get_input_size() == 2);
        const nnfusion::Shape& input_shape_0 = gnode->get_input_shape(0);
        const nnfusion::Shape& input_shape_1 = gnode->get_input_shape(1);

        auto generic_op = std::dynamic_pointer_cast<nnfusion::op::GenericOp>(gnode->get_op_ptr());

        int axis = generic_op->localOpConfig.getRoot()["axis"];

        nnfusion::Shape output_shape_0;
        for (int i = 0; i < axis; ++i)
        {
            NNFUSION_CHECK(input_shape_0[i] == input_shape_1[i]);
            output_shape_0.push_back(input_shape_0[i]);
        }
        for (int i = axis; i < input_shape_1.size() - 1; ++i)
            output_shape_0.push_back(input_shape_1[i]);

        for (int i = axis + input_shape_1.back(); i < input_shape_0.size(); ++i)
            output_shape_0.push_back(input_shape_0[i]);

        gnode->set_output_type_and_shape(0, gnode->get_input_element_type(0), output_shape_0);
    });

REGISTER_OP(GatherNDGrad)
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