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
        auto out_sym_shape = std::make_shared<nnfusion::SymShape>();
        for (int i = 0; i < axis; ++i)
        {
            output_shape_0.push_back(input_shape_0[i]);
            if (input_shape_0.get_sym_shape())
                out_sym_shape->push_back((*input_shape_0.get_sym_shape())[i]);
            else
                out_sym_shape->push_back(input_shape_0[i]);
        }
        for (int i = 0; i < input_shape_1.size(); ++i)
        {
            output_shape_0.push_back(input_shape_1[i]);
            if (input_shape_1.get_sym_shape())
                out_sym_shape->push_back((*input_shape_1.get_sym_shape())[i]);
            else
                out_sym_shape->push_back(input_shape_1[i]);
        }
        for (int i = axis + 1; i < input_shape_0.size(); ++i)
        {
            output_shape_0.push_back(input_shape_0[i]);
            if (input_shape_0.get_sym_shape())
                out_sym_shape->push_back((*input_shape_0.get_sym_shape())[i]);
            else
                out_sym_shape->push_back(input_shape_0[i]);
        }

        if (out_sym_shape->is_dynamic())
            output_shape_0.sym_shape = out_sym_shape;

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
    })
    .translate_v2([](std::shared_ptr<graph::GNode> curr) -> std::string {
        auto generic_op = std::dynamic_pointer_cast<nnfusion::op::GenericOp>(curr->get_op_ptr());
        int axis = generic_op->localOpConfig.getRoot()["axis"];
        // e.g. Antares type is int64 rather than C++'s int64_t
        std::string dtype;
        bool ret = element::Type::nnfusion_element_type_to_dtype_string(
            curr->get_input_element_type(1), dtype);
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

        for (size_t d = axis; d < axis + input1_shape.size(); ++d)
        {
            input1_layout.push_back(output0_layout[d]);
        }

        for (size_t d = axis + input1_shape.size(); d < output0_shape.size(); ++d)
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