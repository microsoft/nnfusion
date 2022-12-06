// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "nnfusion/common/type/element_type.hpp"
#include "nnfusion/core/operators/generic_op/generic_op.hpp"

REGISTER_OP(Convert)
    .infershape(nnfusion::op::infershape::unimplemented_and_not_used)
    .translate([](std::shared_ptr<graph::GNode> gnode) -> std::string {
        auto op = static_pointer_cast<nnfusion::op::Convert>(gnode->get_op_ptr());
        NNFUSION_CHECK_NOT_NULLPTR(op) << "Node type is not " << gnode->get_op_ptr()->get_op_type();

        std::string in_dtype;
        bool ret = element::Type::nnfusion_element_type_to_dtype_string(
            gnode->get_input_element_type(0), in_dtype);
        NNFUSION_CHECK(ret == true) << "cast type is not supported: "
                                    << gnode->get_input_element_type(0).c_type_string();

        std::string out_dtype;
        ret = element::Type::nnfusion_element_type_to_dtype_string(op->get_convert_element_type(),
                                                                   out_dtype);
        NNFUSION_CHECK(ret == true) << "cast type is not supported: "
                                    << op->get_convert_element_type().c_type_string();

        return op::create_code_from_template(
            R"( - input("input0", @input_shape@, dtype="@in_dtype@"); output(@input_shape@, topi=topi.cast(args("input0"), dtype="@out_dtype@")); )",
            {{"input_shape", vector_to_string(gnode->get_input_shape(0))},
             {"in_dtype", in_dtype},
             {"out_dtype", out_dtype}});
    })
    .translate_v2([](std::shared_ptr<graph::GNode> gnode) -> std::string {
        auto op = static_pointer_cast<nnfusion::op::Convert>(gnode->get_op_ptr());
        NNFUSION_CHECK_NOT_NULLPTR(op) << "Node type is not " << gnode->get_op_ptr()->get_op_type();

        std::string out_dtype;
        bool ret = element::Type::nnfusion_element_type_to_dtype_string(
            op->get_convert_element_type(), out_dtype);
        NNFUSION_CHECK(ret == true) << "cast type is not supported: "
                                    << op->get_convert_element_type().c_type_string();
        out_dtype = out_dtype == "char" ? "int8" : out_dtype;
        if (op->get_convert_element_type() == element::boolean)
        {
            return op::create_code_from_template(
                "@output0@@data_layout@ = (@input0@@data_layout@ != 0).cast(`@out_dtype@`);",
                {{"data_layout",
                  vector_to_string<std::vector<std::string>>(
                      op::create_layout_from_dims(gnode->get_output_shape(0)))},
                 {"out_dtype", out_dtype}});
        }
        else
        {
            return op::create_code_from_template(
                "@output0@@data_layout@ = @input0@@data_layout@.cast(`@out_dtype@`);",
                {{"data_layout",
                  vector_to_string<std::vector<std::string>>(
                      op::create_layout_from_dims(gnode->get_output_shape(0)))},
                 {"out_dtype", out_dtype}});
        }
    });
