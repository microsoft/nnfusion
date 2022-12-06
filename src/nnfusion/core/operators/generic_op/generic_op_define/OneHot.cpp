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

        nnfusion::element::Type type;
        for (const nnfusion::element::Type* t : nnfusion::element::Type::get_known_types())
        {
            if (t->c_type_string() == t_str)
            {
                type = *t;
                break;
            }
        }
        NNFUSION_CHECK(axis >= 0);
        nnfusion::Shape output_shape_0;
        for (int i = 0; i < axis; ++i)
            output_shape_0.push_back(shape_0[i]);
        output_shape_0.push_back(depth);
        for (int i = axis; i < shape_0.size(); ++i)
            output_shape_0.push_back(shape_0[i]);
        gnode->set_output_type_and_shape(0, type, output_shape_0);
    })
    .translate_v2([](std::shared_ptr<graph::GNode> gnode) -> std::string {
        auto generic_op = std::dynamic_pointer_cast<nnfusion::op::GenericOp>(gnode->get_op_ptr());
        int depth = generic_op->localOpConfig.getRoot()["depth"];
        double on_value = generic_op->localOpConfig.getRoot()["on_value"];
        double off_value = generic_op->localOpConfig.getRoot()["off_value"];
        int axis = generic_op->localOpConfig.getRoot()["axis"];

        std::string dtype;
        bool ret =
            element::Type::nnfusion_element_type_to_dtype_string(gnode->get_element_type(), dtype);
        NNFUSION_CHECK(ret) << "Unsupport data type: " << gnode->get_element_type();

        auto input0_layout = op::create_layout_from_dims(gnode->get_input_shape(0));
        auto output_layout = input0_layout;
        output_layout.insert(output_layout.begin() + axis, "F");

        //output0[N, F] = const(1.0).when([input0[N] == F, (input0[N] + 4 == F)], const(0.0), merge_op=`any`) where F in 128
        std::string expr =
            "@output0@@output_layout@ = const(@on_value@).when([@input0@@input0_layout@  == "
            "@axis@, @input0@@input0_layout@ + @depth@ == @axis@], @off_value@, "
            "merge_op=`any`).cast(`@dtype@`) where @axis@ in @depth@;";

        return op::create_code_from_template(
            expr,
            {{"input0_layout", vector_to_string<std::vector<std::string>>(input0_layout)},
             {"output_layout", vector_to_string<std::vector<std::string>>(output_layout)},
             {"depth", depth},
             {"on_value", on_value},
             {"off_value", off_value},
             {"axis", output_layout[axis]},
             {"dtype", dtype}});
    });
