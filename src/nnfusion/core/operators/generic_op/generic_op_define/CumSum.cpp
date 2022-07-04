// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "nnfusion/core/operators/generic_op/generic_op.hpp"

REGISTER_OP(CumSum)
    .attr<int>("axis")
    .attr<int>("exclusive")
    .attr<int>("reverse")
    .infershape(nnfusion::op::infershape::unimplemented_and_not_used)
    .translate_v2([](std::shared_ptr<graph::GNode> gnode) -> std::string {
        auto shape_0 = gnode->get_input_shape(0);

        auto generic_op = std::dynamic_pointer_cast<nnfusion::op::GenericOp>(gnode->get_op_ptr());
        int axis = generic_op->localOpConfig.getRoot()["axis"];
        axis += axis < 0 ? shape_0.size() : 0;
        int exclusive = generic_op->localOpConfig.getRoot()["exclusive"];
        int reverse = generic_op->localOpConfig.getRoot()["reverse"];

        auto input0_layout = op::create_layout_from_dims(shape_0);
        auto output0_layout = input0_layout;

        input0_layout[axis] = "C";
        output0_layout[axis] = "N";

        std::string cond = reverse > 0 ? ">" : "<";
        if (exclusive == 0)
            cond += "=";

        // output0[N] +=! input0[C].when([C <= N], const(0.0)) where C in 4, N in 4
        std::string ir_template =
            R"( @output0@@output0_layout@ +=! @input0@@input0_layout@.when([C @cond@ N], const(0.0).cast(input0.dtype())) where C in @size@, N in @size@ )";

        op::OpConfig::any op_config;
        op_config["input0_layout"] = nnfusion::vector_to_string(input0_layout);
        op_config["output0_layout"] = nnfusion::vector_to_string(output0_layout);
        op_config["cond"] = cond;
        op_config["size"] = shape_0.at(axis);

        return op::create_code_from_template(ir_template, op_config);
    });