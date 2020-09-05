// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "nnfusion/core/operators/generic_op/generic_op.hpp"

REGISTER_OP(Relu)
    .infershape(nnfusion::op::infershape::copy_shape_from_inputs)
    .translate_v2([](std::shared_ptr<graph::GNode> curr) -> std::string {
        auto ir_template =
            R"( @output0@@output0_layout@ = @input0@@input0_layout@.when(@input0@@input0_layout@ > 0.0, 0.0); )";

        auto shape_def = op::create_layout_from_dims(curr->get_input_shape(0));

        op::OpConfig::any op_config;
        op_config["input0_layout"] = vector_to_string<std::vector<std::string>>(shape_def);
        op_config["output0_layout"] = vector_to_string<std::vector<std::string>>(shape_def);

        return op::create_code_from_template(ir_template, op_config);
    });
