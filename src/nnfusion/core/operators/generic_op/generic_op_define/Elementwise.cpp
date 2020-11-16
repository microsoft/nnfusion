// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "nnfusion/core/operators/generic_op/generic_op.hpp"

auto trans_elementwise = [&](std::shared_ptr<graph::GNode>& curr, const std::string& topi) {
    std::string expr = " -";
    for (int i = 0; i < curr->get_input_size(); ++i)
        expr += " input(\"input" + std::to_string(i) + "\", @common_shape@);";
    expr += " output(@common_shape@, " + topi + ");";

    int num_elements = 1, y;
    for (auto& it : curr->get_input_shape(0))
        num_elements *= it;

    return op::create_code_from_template(
        expr, {{"common_shape", "[ " + std::to_string(num_elements) + " ]"}});
};

auto trans_elementwise_v2 = [&](std::shared_ptr<graph::GNode>& curr, const std::string& mask) {
    std::string expr_template;
    switch (curr->get_input_size())
    {
    case 2:
        expr_template =
            "@output0@@data_layout@ = @input0@@data_layout@ @mask@ @input1@@data_layout@;";
        break;
    default: NNFUSION_CHECK(0) << "Unsupport number of inputs with elementwise op"; break;
    }
    auto data_layput = op::create_layout_from_dims(curr->get_output_shape(0));
    return op::create_code_from_template(
        expr_template,
        {{"mask", mask}, {"data_layout", vector_to_string<std::vector<std::string>>(data_layput)}});
};

REGISTER_OP(Subtract)
    .infershape(nnfusion::op::infershape::copy_shape_from_inputs)
    .translate([](std::shared_ptr<graph::GNode> curr) -> std::string {
        auto _op = static_pointer_cast<nnfusion::op::Subtract>(curr->get_op_ptr());
        NNFUSION_CHECK_NOT_NULLPTR(_op) << "Node type is not " << curr->get_op_ptr()->get_op_type();

        std::string topi = "topi=topi.subtract(args(\"input0\"), args(\"input1\"))";
        return trans_elementwise(curr, topi);
    });

REGISTER_OP(Multiply)
    .infershape(nnfusion::op::infershape::copy_shape_from_inputs)
    .translate([](std::shared_ptr<graph::GNode> curr) -> std::string {
        auto _op = static_pointer_cast<nnfusion::op::Multiply>(curr->get_op_ptr());
        NNFUSION_CHECK_NOT_NULLPTR(_op) << "Node type is not " << curr->get_op_ptr()->get_op_type();

        std::string topi = "topi=topi.multiply(args(\"input0\"), args(\"input1\"))";
        return trans_elementwise(curr, topi);
    })
    .translate_v2([](std::shared_ptr<graph::GNode> curr) -> std::string {
        auto _op = static_pointer_cast<nnfusion::op::Multiply>(curr->get_op_ptr());
        NNFUSION_CHECK_NOT_NULLPTR(_op) << "Node type is not" << curr->get_op_ptr()->get_op_type();

        return trans_elementwise_v2(curr, "*");
    });

REGISTER_OP(Divide)
    .infershape(nnfusion::op::infershape::copy_shape_from_inputs)
    .translate([](std::shared_ptr<graph::GNode> curr) -> std::string {
        auto _op = static_pointer_cast<nnfusion::op::Divide>(curr->get_op_ptr());
        NNFUSION_CHECK_NOT_NULLPTR(_op) << "Node type is not " << curr->get_op_ptr()->get_op_type();

        std::string topi = "topi=topi.divide(args(\"input0\"), args(\"input1\"))";
        return trans_elementwise(curr, topi);
    });

REGISTER_OP(DivNoNan)
    .infershape(nnfusion::op::infershape::copy_shape_from_inputs)
    .translate([](std::shared_ptr<graph::GNode> curr) -> std::string {
        auto _op = static_pointer_cast<nnfusion::op::DivNoNan>(curr->get_op_ptr());
        NNFUSION_CHECK_NOT_NULLPTR(_op) << "Node type is not " << curr->get_op_ptr()->get_op_type();

        std::string topi =
            "lambda x: tvm.if_then_else(args(\"input1\")[x] != "
            "0, args(\"input0\")[x] / args(\"input1\")[x], 0)";
        return trans_elementwise(curr, topi);
    });

REGISTER_OP(Power)
    .infershape(nnfusion::op::infershape::copy_shape_from_inputs)
    .translate([](std::shared_ptr<graph::GNode> curr) -> std::string {
        auto _op = static_pointer_cast<nnfusion::op::Power>(curr->get_op_ptr());
        NNFUSION_CHECK_NOT_NULLPTR(_op) << "Node type is not " << curr->get_op_ptr()->get_op_type();

        std::string topi = "topi=topi.power(args(\"input0\"), args(\"input1\"))";
        return trans_elementwise(curr, topi);
    });

REGISTER_OP(LessEq)
    .infershape(nnfusion::op::infershape::copy_shape_from_inputs_with_boolean)
    .translate([](std::shared_ptr<graph::GNode> curr) -> std::string {
        auto _op = static_pointer_cast<nnfusion::op::LessEq>(curr->get_op_ptr());
        NNFUSION_CHECK_NOT_NULLPTR(_op) << "Node type is not " << curr->get_op_ptr()->get_op_type();

        std::string topi = "topi=topi.less_equal(args(\"input0\"), args(\"input1\"))";
        return trans_elementwise(curr, topi);
    });

REGISTER_OP(Equal)
    .infershape(nnfusion::op::infershape::copy_shape_from_inputs_with_boolean)
    .translate([](std::shared_ptr<graph::GNode> curr) -> std::string {
        auto _op = static_pointer_cast<nnfusion::op::Equal>(curr->get_op_ptr());
        NNFUSION_CHECK_NOT_NULLPTR(_op) << "Node type is not " << curr->get_op_ptr()->get_op_type();

        std::string topi = "topi=topi.equal(args(\"input0\"), args(\"input1\"))";
        return trans_elementwise(curr, topi);
    });

REGISTER_OP(Maximum)
    .infershape(nnfusion::op::infershape::copy_shape_from_inputs)
    .translate([](std::shared_ptr<graph::GNode> curr) -> std::string {
        auto _op = static_pointer_cast<nnfusion::op::Maximum>(curr->get_op_ptr());
        NNFUSION_CHECK_NOT_NULLPTR(_op) << "Node type is not " << curr->get_op_ptr()->get_op_type();

        std::string topi = "topi=topi.maximum(args(\"input0\"), args(\"input1\"))";
        return trans_elementwise(curr, topi);
    });

REGISTER_OP(Exp)
    .infershape(nnfusion::op::infershape::copy_shape_from_inputs)
    .translate([](std::shared_ptr<graph::GNode> curr) -> std::string {
        auto _op = static_pointer_cast<nnfusion::op::Exp>(curr->get_op_ptr());
        NNFUSION_CHECK_NOT_NULLPTR(_op) << "Node type is not " << curr->get_op_ptr()->get_op_type();

        std::string topi = "topi=topi.exp(args(\"input0\"))";
        return trans_elementwise(curr, topi);
    });

REGISTER_OP(Negative)
    .infershape(nnfusion::op::infershape::copy_shape_from_inputs)
    .translate([](std::shared_ptr<graph::GNode> curr) -> std::string {
        auto _op = static_pointer_cast<nnfusion::op::Negative>(curr->get_op_ptr());
        NNFUSION_CHECK_NOT_NULLPTR(_op) << "Node type is not " << curr->get_op_ptr()->get_op_type();

        std::string topi = "topi=topi.negative(args(\"input0\"))";
        return trans_elementwise(curr, topi);
    });

REGISTER_OP(Tanh)
    .infershape(nnfusion::op::infershape::copy_shape_from_inputs)
    .translate([](std::shared_ptr<graph::GNode> curr) -> std::string {
        auto _op = static_pointer_cast<nnfusion::op::Tanh>(curr->get_op_ptr());
        NNFUSION_CHECK_NOT_NULLPTR(_op) << "Node type is not " << curr->get_op_ptr()->get_op_type();

        std::string topi = "topi=topi.tanh(args(\"input0\"))";
        return trans_elementwise(curr, topi);
    });

REGISTER_OP(Relu6)
    .infershape(nnfusion::op::infershape::copy_shape_from_inputs)
    .translate([](std::shared_ptr<graph::GNode> curr) -> std::string {
        auto _op = static_pointer_cast<nnfusion::op::Relu6>(curr->get_op_ptr());
        NNFUSION_CHECK_NOT_NULLPTR(_op) << "Node type is not " << curr->get_op_ptr()->get_op_type();
        std::string topi = "topi=topi.clip(args(\"input0\"), 0, 6)";
        return trans_elementwise(curr, topi);
    });

REGISTER_OP(Sigmoid)
    .infershape(nnfusion::op::infershape::copy_shape_from_inputs)
    .translate([](std::shared_ptr<graph::GNode> curr) -> std::string {
        auto _op = static_pointer_cast<nnfusion::op::Sigmoid>(curr->get_op_ptr());
        NNFUSION_CHECK_NOT_NULLPTR(_op) << "Node type is not " << curr->get_op_ptr()->get_op_type();

        std::string topi = "topi=topi.sigmoid(args(\"input0\"))";
        return trans_elementwise(curr, topi);
    });

REGISTER_OP(Square)
    .infershape(nnfusion::op::infershape::copy_shape_from_inputs)
    .translate([](std::shared_ptr<graph::GNode> curr) -> std::string {
        auto _op = static_pointer_cast<nnfusion::op::Square>(curr->get_op_ptr());
        NNFUSION_CHECK_NOT_NULLPTR(_op) << "Node type is not " << curr->get_op_ptr()->get_op_type();

        std::string topi = "topi=topi.multiply(args(\"input0\"), args(\"input0\"))";
        return trans_elementwise(curr, topi);
    });

REGISTER_OP(Rsqrt)
    .infershape(nnfusion::op::infershape::copy_shape_from_inputs)
    .translate([](std::shared_ptr<graph::GNode> curr) -> std::string {
        auto _op = static_pointer_cast<nnfusion::op::Rsqrt>(curr->get_op_ptr());
        NNFUSION_CHECK_NOT_NULLPTR(_op) << "Node type is not " << curr->get_op_ptr()->get_op_type();

        std::string topi = "topi=topi.rsqrt(args(\"input0\"))";
        return trans_elementwise(curr, topi);
    });

REGISTER_OP(Log)
    .infershape(nnfusion::op::infershape::copy_shape_from_inputs)
    .translate([](std::shared_ptr<graph::GNode> curr) -> std::string {
        auto _op = static_pointer_cast<nnfusion::op::Log>(curr->get_op_ptr());
        NNFUSION_CHECK_NOT_NULLPTR(_op) << "Node type is not " << curr->get_op_ptr()->get_op_type();

        std::string topi = "topi=topi.log(args(\"input0\"))";
        return trans_elementwise(curr, topi);
    });

REGISTER_OP(ReluBackprop)
    .infershape(nnfusion::op::infershape::copy_shape_from_inputs)
    .translate([](std::shared_ptr<graph::GNode> curr) -> std::string {
        auto _op = static_pointer_cast<nnfusion::op::ReluBackprop>(curr->get_op_ptr());
        NNFUSION_CHECK_NOT_NULLPTR(_op) << "Node type is not " << curr->get_op_ptr()->get_op_type();

        std::string topi =
            "lambda x: tvm.if_then_else(args(\"input0\")[x] > "
            "0, args(\"input1\")[x], 0)";
        return trans_elementwise(curr, topi);
    });

REGISTER_OP(Select)
    .infershape(nnfusion::op::infershape::copy_shape_from_inputs)
    .translate([](std::shared_ptr<graph::GNode> curr) -> std::string {
        auto _op = static_pointer_cast<nnfusion::op::Select>(curr->get_op_ptr());
        NNFUSION_CHECK_NOT_NULLPTR(_op) << "Node type is not " << curr->get_op_ptr()->get_op_type();

        std::string topi =
            "lambda x: tvm.if_then_else(args(\"input0\")[x] == "
            "0, args(\"input2\")[x], args(\"input1\")[x])";
        return trans_elementwise(curr, topi);
    });
