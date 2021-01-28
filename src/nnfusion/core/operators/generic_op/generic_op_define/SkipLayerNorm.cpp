// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "nnfusion/core/operators/generic_op/generic_op.hpp"

REGISTER_OP(SkipLayerNorm)
    .attr<float>("epsilon", 1e-6)
    .infershape([](std::shared_ptr<graph::GNode> gnode) -> void {
        /*
        Input0, "input", "3D input tensor with shape (batch_size, sequence_length, hidden_size)", "float/float16"
        Input1, "skip", "3D skip tensor with shape (batch_size, sequence_length, hidden_size)", "float/float16"
        Input2, "gamma", "1D input tensor with shape (hidden_size)", "float/float16"
        Input3, "beta", "1D skip tensor with shape (hidden_size)", "float/float16"
        Input4, "bias", "1D bias tensor with shape (hidden_size)", "float/float16", Optional
        Output0, "output", "3D output tensor with shape (batch_size, sequence_length, hidden_size)", "float/float16"
        */
        NNFUSION_CHECK(gnode->get_input_size() >= 4)
            << "SkipLayerNorm should have at least 4 inputs.";

        auto input_shape = gnode->get_input_shape(0);
        NNFUSION_CHECK(input_shape.size() == 3) << "input is expected to have 3 dimensions, got "
                                                << input_shape.size();

        auto skip_shape = gnode->get_input_shape(1);
        NNFUSION_CHECK(input_shape == skip_shape) << "skip is expected to have same shape as input";

        auto gama_shape = gnode->get_input_shape(2);
        NNFUSION_CHECK(gama_shape.size() == 1) << "gamma is expected to have 1 dimension, got "
                                               << gama_shape.size();
        NNFUSION_CHECK(gama_shape[0] == input_shape[2])
            << "Last dimension of gamma and input does not match";

        auto beta_shape = gnode->get_input_shape(3);
        NNFUSION_CHECK(beta_shape.size() == 1) << "beta is expected to have 1 dimension, got "
                                               << beta_shape.size();
        NNFUSION_CHECK(beta_shape[0] == input_shape[2])
            << "Last dimension of beta and input does not match";

        if (gnode->get_input_size() == 5)
        {
            auto bias_shape = gnode->get_input_shape(4);
            NNFUSION_CHECK(bias_shape.size() == 1) << "bias is expected to have 1 dimension, got "
                                                   << bias_shape.size();
            NNFUSION_CHECK(bias_shape[0] == input_shape[2])
                << "Last dimension of bias and input does not match";
        }

        gnode->set_output_type_and_shape(0, gnode->get_input_element_type(0), input_shape);
    });
