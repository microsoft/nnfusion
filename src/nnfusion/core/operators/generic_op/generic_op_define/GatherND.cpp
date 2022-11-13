// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "nnfusion/core/operators/generic_op/generic_op.hpp"

REGISTER_OP(GatherND)
    .attr<int>("axis", 0)
    .infershape([](std::shared_ptr<graph::GNode> gnode) -> void {
        NNFUSION_CHECK(gnode->get_input_size() == 2);
        const nnfusion::Shape& input_shape_0 = gnode->get_input_shape(0);
        const nnfusion::Shape& input_shape_1 = gnode->get_input_shape(1);

        auto generic_op = std::dynamic_pointer_cast<nnfusion::op::GenericOp>(gnode->get_op_ptr());

        int axis = generic_op->localOpConfig.getRoot()["axis"];

        nnfusion::Shape output_shape_0;
        for (int i = 0; i < axis; ++i)
        {
            NNFUSION_CHECK(input_shape_0[i] == input_shape_1[i]);
            output_shape_0.push_back(input_shape_0[i]);
        }
        for (int i = axis; i < input_shape_1.size() - 1; ++i)
            output_shape_0.push_back(input_shape_1[i]);

        for (int i = axis + input_shape_1.back(); i < input_shape_0.size(); ++i)
            output_shape_0.push_back(input_shape_0[i]);

        gnode->set_output_type_and_shape(0, gnode->get_input_element_type(0), output_shape_0);
    })
    .translate_v2([](std::shared_ptr<graph::GNode> curr) -> std::string {
        auto generic_op = std::dynamic_pointer_cast<nnfusion::op::GenericOp>(curr->get_op_ptr());
        int axis = generic_op->localOpConfig.getRoot()["axis"];
        // e.g. Antares type is int64 rather than C++'s int64_t
        std::string dtype;
        bool ret = element::Type::nnfusion_element_type_to_dtype_string(
            curr->get_input_element_type(1), dtype);
        NNFUSION_CHECK(ret);

        auto index_template =
            R"( input1@input1_layout@.when(input1@input1_layout@ >= 0, input1@input1_layout@ + const(@shape_dim@).cast(input1@input1_layout@.dtype())) )";

        // output0[B0, B1, I0, I1, S0, S1] = input0[B0, B1, input1[[B0, B1, I0, I1, 0], input1[[B0, B1, I0, I1, 1], S0, S1]
        auto ir_template =
            R"( output0@output0_layout@ = input0@input0_layout@; )";

        std::vector<std::string> input0_layout, input1_layout, output0_layout;
        const nnfusion::Shape& input_shape_0 = curr->get_input_shape(0);
        const nnfusion::Shape& input_shape_1 = curr->get_input_shape(1);
        for (int i = 0; i < axis; ++i)
        {
            std::string index_str = "B" + std::to_string(i);
            input0_layout.push_back(index_str);
            input1_layout.push_back(index_str);
            output0_layout.push_back(index_str);
        }
        for (int i = axis; i < input_shape_1.size() - 1; ++i)
        {
            std::string index_str = "I" + std::to_string(i);
            input1_layout.push_back(index_str);
            output0_layout.push_back(index_str);
        }

        for (int i = 0; i < input_shape_1.back(); ++i)
        {
            // [B0, B1, ... I0, I1, ... i]
            input1_layout.push_back(std::to_string(i));

            op::OpConfig::any op_config;
            op_config["input1_layout"] = vector_to_string<std::vector<std::string>>(input1_layout);
            op_config["shape_dim"] = input_shape_0[axis + i];
            std::string index_str = op::create_code_from_template(index_template, op_config);
            input0_layout.push_back(index_str);

            input1_layout.pop_back();
        }

        for (int i = axis + input_shape_1.back(); i < input_shape_0.size(); ++i)
        {
            std::string index_str = "S" + std::to_string(i);
            input0_layout.push_back(index_str);
            output0_layout.push_back(index_str);
        }

        op::OpConfig::any op_config;
        op_config["output0_layout"] = vector_to_string<std::vector<std::string>>(output0_layout);
        op_config["input0_layout"] = vector_to_string<std::vector<std::string>>(input0_layout);

        return op::create_code_from_template(ir_template, op_config);
    });

REGISTER_OP(GatherNDGrad)
    .attr<int>("axis", 0)
    .attr<std::vector<int>>("x_shape")
    .infershape([](std::shared_ptr<graph::GNode> gnode) -> void {
        NNFUSION_CHECK(gnode->get_input_size() == 3);
        auto x_grad_type = gnode->get_input_element_type(1);

        auto generic_op = std::dynamic_pointer_cast<nnfusion::op::GenericOp>(gnode->get_op_ptr());
        std::vector<int> x_shape = generic_op->localOpConfig.getRoot()["x_shape"];

        nnfusion::Shape output_shape_0;

        gnode->set_output_type_and_shape(0, x_grad_type, Shape(x_shape.begin(), x_shape.end()));
    });