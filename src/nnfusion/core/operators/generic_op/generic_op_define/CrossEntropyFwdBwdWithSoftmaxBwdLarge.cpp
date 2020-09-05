// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "nnfusion/core/operators/generic_op/generic_op.hpp"

REGISTER_OP(CrossEntropyFwdBwdWithSoftmaxBwdLarge)
    .infershape([](std::shared_ptr<graph::GNode> gnode) -> void {
        NNFUSION_CHECK(2 == gnode->get_input_size());
        auto& shape_0 = gnode->get_input_shape(0);
        auto& shape_1 = gnode->get_input_shape(1);

        NNFUSION_CHECK(shape_0.size() - shape_1.size() == 1 &&
                       std::equal(shape_1.begin(), shape_1.end(), shape_0.begin()));

        gnode->set_output_type_and_shape(0, gnode->get_input_element_type(0), shape_0);
    });
