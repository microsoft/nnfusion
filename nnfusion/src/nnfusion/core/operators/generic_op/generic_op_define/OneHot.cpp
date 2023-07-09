// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "nnfusion/core/operators/generic_op/generic_op.hpp"

REGISTER_OP(OneHot)
    .attr<int>("axis", -1)
    .attr<int>("depth")
    .attr<nnfusion::op::OpConfig::any>("T")
    .attr<nnfusion::op::OpConfig::any>("off_value", 1.0f)
    .attr<nnfusion::op::OpConfig::any>("on_value", 0.0f)
    .infershape([](std::shared_ptr<graph::GNode> gnode) -> void {
        NNFUSION_CHECK(1 == gnode->get_input_size());
        auto& shape_0 = gnode->get_input_shape(0);
        auto generic_op = std::dynamic_pointer_cast<nnfusion::op::GenericOp>(gnode->get_op_ptr());
        int depth = generic_op->localOpConfig.getRoot()["depth"];
        int axis = generic_op->localOpConfig.getRoot()["axis"];
        std::string t_str = generic_op->localOpConfig.getRoot()["T"];

        size_t bitwidth = 0;
        bool is_real = false;
        bool is_signed = false;
        bool is_quantized = false;
        string c_type_string = "";
        for (const nnfusion::element::Type* t : nnfusion::element::Type::get_known_types())
        {
            if (t->c_type_string() == t_str)
            {
                bitwidth = t->bitwidth();
                is_real = t->is_real();
                is_signed = t->is_signed();
                is_quantized = t->is_quantized();
                c_type_string = t->c_type_string();
                break;
            }
        }
        nnfusion::element::Type type =
            nnfusion::element::Type(bitwidth, is_real, is_signed, is_quantized, c_type_string);

        if (axis == -1)
            axis = shape_0.size() - 1;
        nnfusion::Shape output_shape_0;
        for (int i = 0; i <= axis; ++i)
            output_shape_0.push_back(shape_0[i]);
        output_shape_0.push_back(depth);
        for (int i = axis + 1; i < shape_0.size(); ++i)
            output_shape_0.push_back(shape_0[i]);
        gnode->set_output_type_and_shape(0, type, output_shape_0);
    })
    .translate([](std::shared_ptr<graph::GNode> gnode) -> std::string {
        auto generic_op = std::dynamic_pointer_cast<nnfusion::op::GenericOp>(gnode->get_op_ptr());
        int depth = generic_op->localOpConfig.getRoot()["depth"];
        float on_value = generic_op->localOpConfig.getRoot()["on_value"];
        float off_value = generic_op->localOpConfig.getRoot()["off_value"];
        int axis = generic_op->localOpConfig.getRoot()["axis"];

        std::string dtype;
        bool ret =
            element::Type::nnfusion_element_type_to_dtype_string(gnode->get_element_type(), dtype);
        NNFUSION_CHECK(ret);

        return op::create_code_from_template(
            R"( - input("input0", @input_shape@, dtype="int32"); output(@output_shape@, topi=topi.one_hot(args("input0"), depth=@depth@, on_value=@on_value@, off_value=@off_value@, axis=@axis@, dtype="@dtype@")); )",
            {{"input_shape", vector_to_string(gnode->get_input_shape(0))},
             {"output_shape", vector_to_string(gnode->get_output_shape(0))},
             {"depth", depth},
             {"on_value", on_value},
             {"off_value", off_value},
             {"axis", axis},
             {"dtype", dtype}});
    })
    .translate_v2([](std::shared_ptr<graph::GNode> gnode) -> std::string {
        auto generic_op = std::dynamic_pointer_cast<nnfusion::op::GenericOp>(gnode->get_op_ptr());
        int depth = generic_op->localOpConfig.getRoot()["depth"];
        float on_value = generic_op->localOpConfig.getRoot()["on_value"];
        float off_value = generic_op->localOpConfig.getRoot()["off_value"];
        int axis = generic_op->localOpConfig.getRoot()["axis"];

        std::string dtype;
        bool ret =
            element::Type::nnfusion_element_type_to_dtype_string(gnode->get_element_type(), dtype);
        NNFUSION_CHECK(ret);

        auto input0_layout = op::create_layout_from_dims(gnode->get_input_shape(0));
        axis = ((axis < 0) ? input0_layout.size() + 1 + axis : axis);
        auto output_layout = input0_layout;
        output_layout.insert(output_layout.begin() + axis, "F");

        //e.g., output0[N0, F, N1, N2] = parse(1.0).when([input0[N0, N1, N2]  == F], 0.0) where F in Depth;
        std::string expr =
            "@output0@@output_layout@ = const(@on_value@).when([@input0@@input0_layout@  == "
            "@axis@], @off_value@) where @axis@ in @depth@;";

        return op::create_code_from_template(
            expr,
            {{"input0_layout", vector_to_string<std::vector<std::string>>(input0_layout)},
             {"output_layout", vector_to_string<std::vector<std::string>>(output_layout)},
             {"depth", depth},
             {"on_value", on_value},
             {"off_value", off_value},
             {"axis", output_layout[axis]}});
    });
