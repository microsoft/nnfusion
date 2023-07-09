// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "nnfusion/core/operators/generic_op/generic_op.hpp"

REGISTER_OP(Shape)
    .attr<nnfusion::op::OpConfig::any>("out_type")
    .infershape([](std::shared_ptr<graph::GNode> gnode) -> void {
        NNFUSION_CHECK(gnode->get_input_size() == 1);
        auto generic_op = std::dynamic_pointer_cast<nnfusion::op::GenericOp>(gnode->get_op_ptr());
        std::string t_str = generic_op->localOpConfig.getRoot()["out_type"];

        const nnfusion::element::Type* type;

        for (const nnfusion::element::Type* t : nnfusion::element::Type::get_known_types())
        {
            if (t->c_type_string() == t_str)
            {
                type = t;
                break;
            }
        }

        size_t input_rank = gnode->get_input_shape(0).size();
        nnfusion::Shape output_shape(1, input_rank);

        gnode->set_output_type_and_shape(0, *type, output_shape);
    })
    .translate([](std::shared_ptr<graph::GNode> gnode) -> std::string {
        std::string dtype;
        bool ret =
            element::Type::nnfusion_element_type_to_dtype_string(gnode->get_element_type(), dtype);
        NNFUSION_CHECK(ret);

        return op::create_code_from_template(
            R"( - input("input0", @input_shape@); output(@output_shape@, topi=topi.shape(args("input0"), dtype='@dtype@')); )",
            {{"input_shape", vector_to_string(gnode->get_input_shape(0))},
             {"output_shape", vector_to_string(gnode->get_output_shape(0))},
             {"dtype", dtype}});
    });
