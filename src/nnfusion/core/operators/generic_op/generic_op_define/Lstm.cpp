// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "nnfusion/core/operators/generic_op/generic_op.hpp"

REGISTER_OP(Lstm)
    .attr<std::string>("direction", "forward")
    .attr<int64_t>("hidden_size", 0)
    .attr<int64_t>("input_forget", 0)
    .infershape([](std::shared_ptr<graph::GNode> gnode) -> void {
        NNFUSION_CHECK(gnode->get_input_size() > 2) << "Lstm should have at least 3 inputs.";

        const nnfusion::Shape& input_shape_0 = gnode->get_input_shape(0);
        size_t seq_length = input_shape_0[0];
        size_t batch_size = input_shape_0[1];
        const nnfusion::Shape& input_shape_1 = gnode->get_input_shape(1);
        size_t num_direct = input_shape_1[0];
        const nnfusion::Shape& input_shape_2 = gnode->get_input_shape(2);
        size_t hidden_size = input_shape_2[2];

        nnfusion::Shape output_shape_0;
        output_shape_0.push_back(seq_length);
        output_shape_0.push_back(num_direct);
        output_shape_0.push_back(batch_size);
        output_shape_0.push_back(hidden_size);

        gnode->set_output_type_and_shape(0, gnode->get_input_element_type(0), output_shape_0);
    });