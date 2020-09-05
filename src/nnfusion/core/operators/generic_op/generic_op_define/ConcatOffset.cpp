// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "nnfusion/core/operators/generic_op/generic_op.hpp"

REGISTER_OP(ConcatOffset).infershape([](std::shared_ptr<graph::GNode> gnode) -> void {
    NNFUSION_CHECK(3 == gnode->get_input_size());
    auto& shape_0 = gnode->get_input_shape(1);
    nnfusion::Shape output_shape_0;
    output_shape_0.push_back(2);
    for (int i = 0; i < shape_0.size(); ++i)
        output_shape_0.push_back(shape_0[i]);
    gnode->set_output_type_and_shape(0, gnode->get_input_element_type(0), output_shape_0);
});
