// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "nnfusion/core/operators/generic_op/generic_op.hpp"

REGISTER_OP(GroupNorm)
    .attr<int>("activation", 1)
    .attr<float>("epsilon", 1e-5)
    .attr<size_t>("groups", 32)
    .infershape([](std::shared_ptr<graph::GNode> gnode) -> void {
        NNFUSION_CHECK(gnode->get_input_size() == 3);
        const nnfusion::Shape& input_shape_0 = gnode->get_input_shape(0);
        // const nnfusion::Shape& input_shape_1 = gnode->get_input_shape(1);
        // const nnfusion::Shape& input_shape_2 = gnode->get_input_shape(2);
        // const size_t input_0_dims = input_shape_0.size();

        // auto generic_op = std::dynamic_pointer_cast<nnfusion::op::GenericOp>(gnode->get_op_ptr());
        // int axis = generic_op->localOpConfig.getRoot()["axis"];
        // axis += axis < 0 ? input_0_dims : 0;

        // NNFUSION_CHECK(input_shape_1 == input_shape_2);
        // NNFUSION_CHECK(input_shape_1.size() == input_0_dims - axis);
        // NNFUSION_CHECK(
        //     std::equal(input_shape_1.begin(), input_shape_1.end(), input_shape_0.begin() + axis));

        gnode->set_output_type_and_shape(0, gnode->get_input_element_type(0), input_shape_0);

    })
    .translate_v2([](std::shared_ptr<graph::GNode> curr) -> std::string {

        const nnfusion::Shape& input_shape_0 = curr->get_input_shape(0);
        auto generic_op = std::dynamic_pointer_cast<nnfusion::op::GenericOp>(curr->get_op_ptr());
        size_t g = generic_op->localOpConfig.getRoot()["groups"];
        float eps = generic_op->localOpConfig.getRoot()["epsilon"];
        size_t N = input_shape_0[0];
        size_t H = input_shape_0[1];
        size_t W = input_shape_0[2];
        size_t C = input_shape_0[3];
        size_t cg = C / g;

        auto expression_template =
            R"(mediate0[N, H, W, CG, G] = @input0@[N, H, W, CG * G] where CG in @cg@, G in @g@; mediate1[N, G] +=! mediate0[N, H, W, CG, G]; mediate2[N, G] +=! mediate0[N, H, W, CG, G] * mediate0[N, H, W, CG, G]; mediate3[N, H, W, CG, G] = @input2@[CG * G] + @input1@[CG * G] * (mediate0[N, H, W, CG, G] * H.val() * W.val() * CG.val()  - mediate1[N, G]) * (mediate2[N, G] * H.val() * W.val() * CG.val() - mediate1[N, G] * mediate1[N, G] + const(@eps@).cast(`float16`)).call(`rsqrt`) where CG in @cg@, G in @g@; @output0@[N, H, W, C] = mediate3[N, H, W, C / @g@, C % @g@] where C in @c@)";

        std::string expression_code = op::create_code_from_template(
            expression_template, {{"cg", cg}, {"g", g}, {"c", C}, {"eps", eps}});
        return expression_code;
    });
