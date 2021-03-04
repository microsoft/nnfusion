// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "nnfusion/core/operators/generic_op/generic_op.hpp"

REGISTER_OP(Where)
    .infershape([](std::shared_ptr<graph::GNode> gnode) -> void {
        const auto& x_shape = gnode->get_input_shape(1);
        const auto& x_type = gnode->get_input_element_type(1);

        gnode->set_output_type_and_shape(0, x_type, x_shape);
    })
    .translate_v2([](std::shared_ptr<graph::GNode> gnode) -> std::string {
        auto expr_tmpl =
            R"( @output0@@output0_layout@ = @input1@@input1_layout@.when([@input0@@input0_layout@ > 0], @input2@@input2_layout@); )";

        const auto& cond_shape = gnode->get_input_shape(0);
        const auto& cond_layout = op::create_layout_from_dims(cond_shape);
        const auto& x_shape = gnode->get_input_shape(1);
        const auto& x_layout = op::create_layout_from_dims(x_shape);
        const auto& y_shape = gnode->get_input_shape(2);
        const auto& y_layout = op::create_layout_from_dims(y_shape);

        const auto& out_shape = gnode->get_output_shape(0);
        const auto& out_layout = op::create_layout_from_dims(out_shape);

        return op::create_code_from_template(expr_tmpl,
                                             {{"input0_layout", vector_to_string(cond_layout)},
                                              {"input1_layout", vector_to_string(x_layout)},
                                              {"input2_layout", vector_to_string(y_layout)},
                                              {"output0_layout", vector_to_string(out_layout)}});
    })
    .infersharedmemory([](std::shared_ptr<graph::GNode> gnode) -> void {
        auto generic_op = std::dynamic_pointer_cast<nnfusion::op::GenericOp>(gnode->get_op_ptr());
        std::vector<size_t> shared_memory;
        auto& input_shape = gnode->get_input_shape(0);
        auto& output_shape = gnode->get_output_shape(0);
        if (input_shape.size() == output_shape.size())
        {
            shared_memory.clear();
            for (size_t i = 0; i < output_shape.size(); i++)
            {
                shared_memory.push_back(1);
            }
        }
        generic_op->set_shared_memory(shared_memory);
    });
