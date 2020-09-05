// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "nnfusion/core/operators/generic_op/generic_op.hpp"

REGISTER_OP(GatherV2)
    .attr<int>("axis", 0)
    .infershape([](std::shared_ptr<graph::GNode> gnode) -> void {
        NNFUSION_CHECK(gnode->get_input_size() == 2);
        const nnfusion::Shape& input_shape_0 = gnode->get_input_shape(0);
        const nnfusion::Shape& input_shape_1 = gnode->get_input_shape(1);

        auto generic_op = std::dynamic_pointer_cast<nnfusion::op::GenericOp>(gnode->get_op_ptr());

        int axis = generic_op->localOpConfig.getRoot()["axis"];

        nnfusion::Shape output_shape_0;
        for (int i = 0; i < axis; ++i)
            output_shape_0.push_back(input_shape_0[i]);
        for (int i = 0; i < input_shape_1.size(); ++i)
            output_shape_0.push_back(input_shape_1[i]);
        for (int i = axis + 1; i < input_shape_0.size(); ++i)
            output_shape_0.push_back(input_shape_0[i]);

        gnode->set_output_type_and_shape(0, gnode->get_input_element_type(0), output_shape_0);
    })
    .translate([](std::shared_ptr<graph::GNode> gnode) -> std::string {
        auto generic_op = std::dynamic_pointer_cast<nnfusion::op::GenericOp>(gnode->get_op_ptr());
        int axis = generic_op->localOpConfig.getRoot()["axis"];
        // e.g. Antares type is int64 rather than C++'s int64_t
        std::string dtype;
        bool ret = element::Type::nnfusion_element_type_to_dtype_string(
            gnode->get_input_element_type(1), dtype);
        NNFUSION_CHECK(ret);
        auto intput_type_1_for_antares = "\"" + dtype + "\"";

        return op::create_code_from_template(
            R"( - input("input0", @input_shape_0@); input("input1", @input_shape_1@, @input_type_1@); output(@output_shape@, topi=topi.take(args("input0"), args("input1"), axis=@axis@)); )",
            {{"input_shape_0", vector_to_string(gnode->get_input_shape(0))},
             {"input_shape_1", vector_to_string(gnode->get_input_shape(1))},
             {"input_type_1", intput_type_1_for_antares},
             {"output_shape", vector_to_string(gnode->get_output_shape(0))},
             {"axis", axis}});
    });
