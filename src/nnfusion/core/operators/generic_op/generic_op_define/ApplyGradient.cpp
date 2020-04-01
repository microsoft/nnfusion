// Microsoft (c) 2019, NNFusion Team

#include "nnfusion/core/operators/generic_op/generic_op.hpp"

// TODO: Need to be more specific

REGISTER_OP(ApplyGradient)
    .attr<float>("learning_rate", 0.001)
    .infershape([](std::shared_ptr<graph::GNode> gnode) -> void {

        CHECK(gnode->get_input_size() == 2) << "Inputs of ApplyGradient operator should be 2.";

        auto& weight_tensor = gnode->get_input_shape(0);
        auto& gradient_tensor = gnode->get_input_shape(1);

        CHECK(weight_tensor.size() == gradient_tensor.size())
            << "The two inputs should have the same dimentions.";
        for (int j = 0; j < weight_tensor.size(); j++)
        {
            CHECK(weight_tensor[j] == gradient_tensor[j]) << "Dimension " << j
                                                          << " in shapes must be equal.";
        }

        nnfusion::Shape output_shape_0(weight_tensor);
        gnode->set_output_type_and_shape(0, gnode->get_input_element_type(0), output_shape_0);
    });
