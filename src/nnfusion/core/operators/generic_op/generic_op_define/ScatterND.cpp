// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "nnfusion/core/operators/generic_op/generic_op.hpp"

REGISTER_OP(ScatterND)
    .infershape([](std::shared_ptr<graph::GNode> gnode) -> void {
        NNFUSION_CHECK(gnode->get_input_size() == 3);
        const nnfusion::Shape& input_shape_0 = gnode->get_input_shape(0);
        gnode->set_output_type_and_shape(0, gnode->get_input_element_type(0), input_shape_0);
    })
    .translate_v2([](std::shared_ptr<graph::GNode> gnode) -> std::string {
        auto generic_op = std::dynamic_pointer_cast<nnfusion::op::GenericOp>(gnode->get_op_ptr());
        const nnfusion::Shape& data_shape = gnode->get_input_shape(0);
        const nnfusion::Shape& index_shape = gnode->get_input_shape(1);
        const nnfusion::Shape& update_shape = gnode->get_input_shape(2);
        NNFUSION_CHECK(index_shape.size() > 1);
        size_t update_dim = index_shape[index_shape.size() - 1];
        NNFUSION_CHECK(update_shape.size() ==
                       index_shape.size() - 1 + data_shape.size() - update_dim);

        std::vector<std::string> batch_dims;
        for (size_t i = 0; i < index_shape.size() - 1; i++)
        {
            NNFUSION_CHECK(update_shape[i] == index_shape[i]);
            batch_dims.push_back("B" + to_string(i));
        }

        std::vector<std::string> output_layout;
        std::vector<std::string> update_layout = batch_dims;
        for (size_t i = 0; i < data_shape.size(); i++)
        {
            if (i < update_dim)
            {
                auto temp = batch_dims;
                temp.push_back(to_string(i));
                output_layout.push_back("input1" +
                                        vector_to_string<std::vector<std::string>>(temp));
            }
            else
            {
                output_layout.push_back("N" + to_string(i));
                update_layout.push_back("N" + to_string(i));
            }
        }

        std::string expr = "@output0@@output_layout@ =. @input2@@update_layout@;";

        return op::create_code_from_template(
            expr,
            {
                {"update_layout", vector_to_string<std::vector<std::string>>(update_layout)},
                {"output_layout", vector_to_string<std::vector<std::string>>(output_layout)},
            });
    });