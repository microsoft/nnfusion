// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "nnfusion/core/operators/generic_op/generic_op.hpp"

REGISTER_OP(GatherElements)
    .attr<int>("axis", 0)
    .infershape([](std::shared_ptr<graph::GNode> gnode) -> void {
    NNFUSION_CHECK(gnode->get_input_size() == 2);
    const nnfusion::Shape& input_shape_0 = gnode->get_input_shape(0);
    const nnfusion::Shape& input_shape_1 = gnode->get_input_shape(1);

    gnode->set_output_type_and_shape(0, gnode->get_input_element_type(0), input_shape_1);
    })
    .translate_v2([](std::shared_ptr<graph::GNode> curr) -> std::string {
    auto generic_op = std::dynamic_pointer_cast<nnfusion::op::GenericOp>(curr->get_op_ptr());
    int axis = generic_op->localOpConfig.getRoot()["axis"];
    // e.g. Antares type is int64 rather than C++'s int64_t
    std::string dtype;
    bool ret = element::Type::nnfusion_element_type_to_dtype_string(curr->get_input_element_type(1),
                                                                    dtype);
    NNFUSION_CHECK(ret);

    auto ir_template =
        R"( @output0@@output0_layout@ = @input0@[@input0_layout_left@@input1@@input1_layout@.when(@input1@@input1_layout@ >= 0, @input1@@input1_layout@ + const(@gather_dim@).cast(@input1@@input1_layout@.dtype()))@input0_layout_right@]; )";

    auto output0_shape = curr->get_output_shape(0);
    auto output0_layout = op::create_layout_from_dims(output0_shape);
    auto input0_shape = curr->get_input_shape(0);
    auto input1_shape = curr->get_input_shape(1);
    std::string input0_layout_left;
    std::string input0_layout_right;
    std::vector<std::string> input1_layout;
    for (size_t d = 0; d < axis; ++d)
    {
        input0_layout_left += output0_layout[d] + ", ";
    }

    for (size_t d = 0; d < input1_shape.size(); ++d)
    {
        input1_layout.push_back(output0_layout[d]);
    }

    for (size_t d = axis + 1; d < input0_shape.size(); ++d)
    {
        input0_layout_right += ", " + output0_layout[d];
    }

    input1_layout = input1_layout.empty() ? std::vector<std::string>({"0"}) : input1_layout;
    output0_layout = output0_layout.empty() ? std::vector<std::string>({"N0"}) : output0_layout;
    op::OpConfig::any op_config;
    op_config["output0_layout"] = vector_to_string<std::vector<std::string>>(output0_layout);
    op_config["input0_layout_left"] = input0_layout_left;
    op_config["input1_layout"] = vector_to_string<std::vector<std::string>>(input1_layout);
    op_config["input0_layout_right"] = input0_layout_right;
    op_config["gather_dim"] = std::to_string(input0_shape[axis]);

    return op::create_code_from_template(ir_template, op_config);
    });