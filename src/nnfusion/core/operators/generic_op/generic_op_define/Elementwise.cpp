// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "nnfusion/core/operators/generic_op/generic_op.hpp"

struct element_op
{
    element_op(std::string func, std::string expr)
        : func(func)
        , expr(expr)
    {
    }
    std::string func;
    std::string expr;
};

static const std::unordered_map<std::string, element_op> ElementOpMap = {
    {"Abs", element_op("abs", "")},
    {"Acos", element_op("acos", "")},
    {"Atan", element_op("atan", "")},
    {"Ceiling", element_op("ceil", "")},
    {"Cos", element_op("cos", "")},
    {"Cosh", element_op("cosh", "")},
    {"Erf", element_op("erf", "")},
    {"Exp", element_op("exp", "")},
    {"Floor", element_op("floor", "")},
    {"Log", element_op("log", "")},
    {"Max", element_op("max", "")},
    {"Min", element_op("min", "")},
    {"Maximum", element_op("max", "")},
    {"Minimum", element_op("min", "")},
    {"Sin", element_op("sin", "")},
    {"Sinh", element_op("sinh", "")},
    {"Sqrt", element_op("sqrt", "")},
    {"Rsqrt", element_op("rsqrt", "")},
    {"Tan", element_op("tan", "")},
    {"Tanh", element_op("tanh", "")},
    {"Power", element_op("pow", "")},
    {"Add", element_op("add", "x0 + x1")},
    {"Subtract", element_op("subtract", "x0 - x1")},
    {"Multiply", element_op("mul", "x0 * x1")},
    {"Divide", element_op("fdivide", "x0 / x1")},
    {"DivNoNan", element_op("divnonan", "(x0 / x1).when([x1 != 0], 0)")},
    {"Square", element_op("square", "x0 * x0")},
    {"Negative", element_op("negative", "-x0")},
    {"Select", element_op("select", "x2.when([x0 == 0], x1)")},
    {"Sign", element_op("sign", "parse(1).when([x0 > 0], -1)")},
    {"Gelu", element_op("gelu", "x0 * x0.call(\"normcdf\"")},
    {"Relu", element_op("relu", "x0.call(\"max\", 0)")},
    {"Relu6", element_op("relu6", "x0.call(\"max\", 0).call(\"min\", 6)")},
    {"ReluBackprop", element_op("relu_backprop", "x1.when([x0 > 0], 0)")},
    {"Relu6Backprop", element_op("relu_backprop", "x1.when([x0 > 0, x0 < 6], 0)")},
    {"Sigmoid", element_op("sigmoid", "1 / (1 + (-x0).call(\"exp\"))")},
    {"SigmoidBackprop",
     element_op("sigmoid_backprop", "x1 / (2 + (-x0).call(\"exp\") + (x0).call(\"exp\"))")},
    {"Equal", element_op("equal", "x0 == x1")},
    {"NotEqual", element_op("not_equal", "x0 != x1")},
    {"Greater", element_op("greater", "x0 > x1")},
    {"GreaterEq", element_op("greater_equal", "x0 >= x1")},
    {"Less", element_op("less", "x0 < x1")},
    {"LessEq", element_op("less_equal", "x0 <= x1")},
    {"Not", element_op("logical_not", "!x0")},
    {"And", element_op("logical_and", "x0 && x1")},
    {"Or", element_op("logical_or", "x0 || x1")}};

inline std::string replace_input_str(std::string ir)
{
    const size_t max_input_size = 3;
    for (size_t i = 0; i < max_input_size; i++)
    {
        std::string input = "x" + std::to_string(i);
        std::string input_new = "@input" + std::to_string(i) + "@@data_layout@";
        std::size_t found;
        while ((found = ir.find(input)) != std::string::npos)
        {
            ir.replace(found, input.size(), input_new);
        }
    }
}

auto trans_elementwise = [&](std::shared_ptr<graph::GNode>& node) {
    std::string expr = "@output0@@data_layout@ = ";
    auto iter = ElementOpMap.find(node->get_op_type());
    if (iter == ElementOpMap.end())
    {
        NNFUSION_CHECK(false) << "Unsupported elementwise op: " << node->get_op_type();
    }

    if (iter->second.expr == "")
    {
        expr += "@input0@@data_layout@.call(\"" + iter->second.func + "\"";
        for (size_t i = 1; i < node->get_input_size(); ++i)
        {
            expr += ", @input" + std::to_string(i) + "@data_layout@";
        }
        expr += ");";
    }
    else
    {
        expr += replace_input_str(iter->second.expr) + ";";
    }
    auto data_layput = op::create_layout_from_dims(node->get_output_shape(0));
    return op::create_code_from_template(
        expr, {{"data_layout", vector_to_string<std::vector<std::string>>(data_layput)}});
};

#define REGISTER_ELEM_OP(op_name)                                                                  \
    REGISTER_OP(op_name)                                                                           \
        .infershape(nnfusion::op::infershape::copy_shape_from_inputs)                              \
        .translate_v2([](std::shared_ptr<graph::GNode> node) -> std::string {                      \
            return trans_elementwise(node);                                                        \
        });

REGISTER_ELEM_OP(Abs)
REGISTER_ELEM_OP(Acos)
REGISTER_ELEM_OP(Atan)
REGISTER_ELEM_OP(Ceiling)
REGISTER_ELEM_OP(Cos)
REGISTER_ELEM_OP(Cosh)
REGISTER_ELEM_OP(Erf)
REGISTER_ELEM_OP(Exp)
REGISTER_ELEM_OP(Floor)
REGISTER_ELEM_OP(Log)
REGISTER_ELEM_OP(Max)
REGISTER_ELEM_OP(Min)
REGISTER_ELEM_OP(Maximum)
REGISTER_ELEM_OP(Minimum)
REGISTER_ELEM_OP(Sin)
REGISTER_ELEM_OP(Sinh)
REGISTER_ELEM_OP(Sqrt)
REGISTER_ELEM_OP(Rsqrt)
REGISTER_ELEM_OP(Tan)
REGISTER_ELEM_OP(Tanh)
REGISTER_ELEM_OP(Power)
//REGISTER_ELEM_OP(Add)
REGISTER_ELEM_OP(Subtract)
REGISTER_ELEM_OP(Multiply)
REGISTER_ELEM_OP(Divide)
REGISTER_ELEM_OP(DivNoNan)
REGISTER_ELEM_OP(Square)
REGISTER_ELEM_OP(Negative)
REGISTER_ELEM_OP(Select)
REGISTER_ELEM_OP(Sign)
REGISTER_ELEM_OP(Gelu)
REGISTER_ELEM_OP(Relu)
REGISTER_ELEM_OP(Relu6)
REGISTER_ELEM_OP(ReluBackprop)
REGISTER_ELEM_OP(Relu6Backprop)
REGISTER_ELEM_OP(Sigmoid)
REGISTER_ELEM_OP(SigmoidBackprop)
REGISTER_ELEM_OP(Equal)
REGISTER_ELEM_OP(NotEqual)
REGISTER_ELEM_OP(Greater)
REGISTER_ELEM_OP(GreaterEq)
REGISTER_ELEM_OP(Less)
REGISTER_ELEM_OP(LessEq)
REGISTER_ELEM_OP(Not)
REGISTER_ELEM_OP(And)
REGISTER_ELEM_OP(Or)