// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "nnfusion/core/operators/generic_op/generic_op.hpp"

REGISTER_OP(DepthToSpace)
    .attr<nnfusion::op::OpConfig::any>("T")
    .attr<nnfusion::op::OpConfig::any>("block_size")
    .attr<nnfusion::op::OpConfig::any>("data_format")
    .infershape([](std::shared_ptr<graph::GNode> gnode) -> void {
        NNFUSION_CHECK(1 == gnode->get_input_size());
        auto shape_0 = gnode->get_input_shape(0);
        NNFUSION_CHECK(shape_0.size() == 4);

        auto generic_op = std::dynamic_pointer_cast<nnfusion::op::GenericOp>(gnode->get_op_ptr());
        NNFUSION_CHECK(generic_op->localOpConfig.getRoot()["data_format"] == "NHWC");
        size_t block_size = generic_op->localOpConfig.getRoot()["block_size"];

        shape_0[1] *= block_size;
        shape_0[2] *= block_size;
        shape_0[3] /= block_size * block_size;

        gnode->set_output_type_and_shape(0, gnode->get_input_element_type(0), shape_0);
    })
    .translate([](std::shared_ptr<graph::GNode> gnode) -> std::string {
        auto input_shape = gnode->get_input_shape(0);
        NNFUSION_CHECK(input_shape.size() == 4);

        auto generic_op = std::dynamic_pointer_cast<nnfusion::op::GenericOp>(gnode->get_op_ptr());
        NNFUSION_CHECK(generic_op->localOpConfig.getRoot()["data_format"] == "NHWC");
        size_t block_size = generic_op->localOpConfig.getRoot()["block_size"];
        input_shape[0] *= input_shape[1];
        input_shape[1] = block_size;
        input_shape[3] /= block_size;

        auto output_shape = input_shape;
        std::swap(input_shape[1], input_shape[2]);

        return op::create_code_from_template(
            R"( - input("input0", @input_shape@); output(@output_shape@, topi=topi.transpose(args("input0"), axes=[0, 2, 1, 3])); )",
            {{"input_shape", vector_to_string(input_shape)},
             {"output_shape", vector_to_string(output_shape)}});
    });
