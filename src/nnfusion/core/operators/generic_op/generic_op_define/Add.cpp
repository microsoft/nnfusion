// Microsoft (c) 2019, NNFusion Team

#include "nnfusion/core/operators/generic_op/generic_op.hpp"

REGISTER_OP(Add)
    .infershape([](std::shared_ptr<graph::GNode> gnode) -> void {
        NNFUSION_CHECK(2 == gnode->get_input_size());
        auto& shape_0 = gnode->get_input_shape(0);
        auto& shape_1 = gnode->get_input_shape(1);
        NNFUSION_CHECK(shape_0.size() == shape_1.size());
        nnfusion::Shape output_shape_0;
        for (int i = 0; i < shape_0.size(); ++i)
        {
            if (shape_0[i] != shape_1[i])
                NNFUSION_CHECK(shape_0[i] == 1 || shape_1[i] == 1);
            output_shape_0.push_back(std::max(shape_0[i], shape_1[i]));
        }
        gnode->set_output_type_and_shape(0, gnode->get_input_element_type(0), output_shape_0);
    })
    .translate_v2([](std::shared_ptr<graph::GNode> curr) -> std::string {
        auto ir_template =
            R"( @output0@@output0_layout@ = @input0@@input0_layout@ + @input1@@input1_layout@; )";

        auto input0_shape = nnfusion::Shape(curr->get_input_shape(0));
        auto input1_shape = nnfusion::Shape(curr->get_input_shape(1));
        NNFUSION_CHECK(input0_shape.size() == input1_shape.size())
            << "Must Same Dims for Elementwise Add";
        for (int d = 0; d < input0_shape.size(); d++)
        {
            input0_shape[d] = input0_shape[d] == 1 && input1_shape[d] != 1 ? 0 : input0_shape[d];
            input1_shape[d] = input1_shape[d] == 1 && input0_shape[d] != 1 ? 0 : input1_shape[d];
        }
        auto shape0_def = op::create_layout_from_dims(input0_shape);
        auto shape1_def = op::create_layout_from_dims(input1_shape);
        auto output0_def = op::create_layout_from_dims(curr->get_output_shape(0));

        if (!shape0_def.size() && !shape1_def.size() && !output0_def.size())
            shape0_def = shape1_def = output0_def = {"N"};

        op::OpConfig::any op_config;
        op_config["input0_layout"] = vector_to_string<std::vector<std::string>>(shape0_def);
        op_config["input1_layout"] = vector_to_string<std::vector<std::string>>(shape1_def);
        op_config["output0_layout"] = vector_to_string<std::vector<std::string>>(output0_def);

        return op::create_code_from_template(ir_template, op_config);
    });
