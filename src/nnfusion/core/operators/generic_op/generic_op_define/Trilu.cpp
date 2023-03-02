// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <memory>
#include <nnfusion/core/graph/gnode.hpp>
#include <nnfusion/core/operators/op_define/constant.hpp>
#include "nnfusion/core/operators/generic_op/generic_op.hpp"

REGISTER_OP(Trilu)
    .infershape([](std::shared_ptr<graph::GNode> curr) -> void {
        curr->set_output_type_and_shape(0, curr->get_input_element_type(0), curr->get_input_shape(0));
        })
    .translate_v2([](std::shared_ptr<graph::GNode> curr) -> std::string {
        auto input_shape_0 = curr->get_input_shape(0);
        assert(input_shape_0.size() >= 2);
        std::string k_str = "";
        if(curr->get_input_size() == 2)
          k_str = "+ input1[0]";
        auto op = static_pointer_cast<nnfusion::op::GenericOp>(curr->get_op_ptr());
        auto& cfg = op->localOpConfig.getRoot();
        bool upper = cfg["upper"].is_null()?true:int64_t(cfg["upper"])!=0;
        auto input_layout = op::create_layout_from_dims(input_shape_0);
        auto dim_a = input_layout[input_layout.size() - 2];
        auto dim_b = input_layout[input_layout.size() - 1];

        std::string dtype;
        bool ret =
            element::Type::nnfusion_element_type_to_dtype_string(curr->get_element_type(), dtype);
        NNFUSION_CHECK(ret);

        std::string condition = upper?dim_b+">="+dim_a+k_str:dim_a+k_str+">="+dim_b;

        auto expression = op::create_code_from_template(
            "@output0@[@input_layout@] = @input0@[@input_layout@].when(@condition@, const(0).cast(`@dtype@`));", {
            {"input_layout", join(input_layout)},
            {"condition", condition},
            {"dtype", dtype}
            });
        return expression;
    });