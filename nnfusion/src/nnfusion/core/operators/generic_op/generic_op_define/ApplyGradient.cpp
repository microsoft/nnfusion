// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "nnfusion/core/operators/generic_op/generic_op.hpp"

// TODO: Need to be more specific

REGISTER_OP(ApplyGradient)
    .attr<float>("learning_rate", 0.001)
    .infershape([](std::shared_ptr<graph::GNode> gnode) -> void {

        NNFUSION_CHECK(gnode->get_input_size() == 2)
            << "Inputs of ApplyGradient operator should be 2.";

        auto& weight_tensor = gnode->get_input_shape(0);
        auto& gradient_tensor = gnode->get_input_shape(1);

        NNFUSION_CHECK(weight_tensor.size() == gradient_tensor.size())
            << "The two inputs should have the same dimentions.";
        for (int j = 0; j < weight_tensor.size(); j++)
        {
            NNFUSION_CHECK(weight_tensor[j] == gradient_tensor[j]) << "Dimension " << j
                                                                   << " in shapes must be equal.";
        }

        nnfusion::Shape output_shape_0(weight_tensor);
        gnode->set_output_type_and_shape(0, gnode->get_input_element_type(0), output_shape_0);
    })
    .translate_v2([](std::shared_ptr<graph::GNode> gnode) -> std::string {
        NNFUSION_CHECK(gnode->get_input_size() == 2)
            << "Inputs of ApplyGradient operator should be 2.";

        auto& weight_tensor = gnode->get_input_shape(0);
        auto& gradient_tensor = gnode->get_input_shape(1);

        NNFUSION_CHECK(weight_tensor.size() == gradient_tensor.size())
            << "The two inputs should have the same dimentions.";
        for (int j = 0; j < weight_tensor.size(); j++)
        {
            NNFUSION_CHECK(weight_tensor[j] == gradient_tensor[j]) << "Dimension " << j
                                                                   << " in shapes must be equal.";
        }

        auto op = static_pointer_cast<nnfusion::op::GenericOp>(gnode->get_op_ptr());
        NNFUSION_CHECK_NOT_NULLPTR(op) << "Node type is not " << gnode->get_op_ptr()->get_op_type();

        auto& cfg = op->localOpConfig.getRoot();
        float lr = cfg["learning_rate"].is_null() ? 0.001 : (float)cfg["learning_rate"];
        auto data_layout = op::create_layout_from_dims(gnode->get_input_shape(0));
        auto expression_template =
            R"( @output0@@data_layout@ = @input0@@data_layout@ - @input1@@data_layout@ * @lr@; )";

        auto expression_code = op::create_code_from_template(
            expression_template,
            {{"data_layout", vector_to_string<std::vector<std::string>>(data_layout)}, {"lr", lr}});

        AddInplace(gnode->get_op_ptr(), 0, 0, true, true);
        return expression_code;
    });
