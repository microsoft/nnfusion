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
    {"Asin", element_op("asin", "")},
    {"Acos", element_op("acos", "")},
    {"Atan", element_op("atan", "")},
    {"Ceiling", element_op("rceil", "")},
    {"Cos", element_op("cos", "")},
    {"Cosh", element_op("cosh", "")},
    {"Erf", element_op("erf", "")},
    {"Exp", element_op("exp", "")},
    {"Floor", element_op("rfloor", "")},
    {"Log", element_op("log", "")},
    {"Maximum", element_op("max", "")},
    {"Minimum", element_op("min", "")},
    {"Sin", element_op("sin", "")},
    {"Sinh", element_op("sinh", "")},
    {"Sqrt", element_op("sqrt", "")},
    {"Rsqrt", element_op("rsqrt", "")},
    {"Tan", element_op("tan", "")},
    {"Tanh", element_op("tanh", "")},
    {"Power", element_op("pow", "")},
    {"PowerBackwardBase", element_op("power_backward_base", "")},
    {"PowerBackwardExponent", element_op("power_backward_exponent", "")},
    {"Add", element_op("add", "x0 + x1")},
    {"Subtract", element_op("subtract", "x0 - x1")},
    {"Multiply", element_op("mul", "x0 * x1")},
    {"Divide", element_op("fdivide", "x0 / x1")},
    {"DivNoNan",
     element_op(
         "divnonan",
         "(x0 / x1).when([x1 != const(0).cast(x1.dtype())], const(0).cast(input1[].dtype()))")},
    {"Square", element_op("square", "x0 * x0")},
    {"Negative", element_op("negative", "-x0")},
    {"Select", element_op("select", "x2.when([x0 == 0], x1)")},
    {"Sign", element_op("sign", "const(1).when([x0 > 0], const(-1).when([x0 < 0], 0))")},
    {"Gelu", element_op("gelu", "x0 * x0.call(`normcdf`)")},
    {"Relu", element_op("relu", "x0.call(`max`, [const(0).cast(x0.dtype())])")},
    {"Relu6",
     element_op(
         "relu6",
         "x0.call(`max`, [const(0).cast(x0.dtype())]).call(`min`, [const(6).cast(x0.dtype())])")},
    {"ReluBackprop", element_op("relu_backprop", "x1.when([x0 > 0], const(0).cast(x1.dtype()))")},
    {"Relu6Backprop",
     element_op("relu_backprop", "x1.when([x0 > 0, x0 < 6], const(0).cast(x1.dtype()))")},
    {"Sigmoid",
     element_op("sigmoid",
                "const(1).cast(x0.dtype()) / (const(1).cast(x0.dtype()) + (-x0).call(`exp`))")},
    {"SigmoidBackprop",
     element_op("sigmoid_backprop",
                "x1 / (const(2).cast(x0.dtype()) + (-x0).call(`exp`) + (x0).call(`exp`))")},
    {"Equal", element_op("equal", "x0 == x1")},
    {"NotEqual", element_op("not_equal", "x0 != x1")},
    {"Greater", element_op("greater", "x0 > x1")},
    {"GreaterEq", element_op("greater_equal", "x0 >= x1")},
    {"Less", element_op("less", "x0 < x1")},
    {"LessEq", element_op("less_equal", "x0 <= x1")},
    {"Not", element_op("logical_not", "~x0")},
    {"And", element_op("logical_and", "x0 & x1")},
    {"Or", element_op("logical_or", "x0 | x1")}};

std::string replace_input_str(std::string ir)
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

    return ir;
}

auto trans_elementwise = [](std::shared_ptr<graph::GNode>& node) {
    std::string expr = "@output0@@data_layout@ = ";
    auto iter = ElementOpMap.find(node->get_op_type());
    if (iter == ElementOpMap.end())
    {
        NNFUSION_CHECK(false) << "Unsupported elementwise op: " << node->get_op_type();
    }
    // get node dtypes and node op name
    auto _op_name = iter->first;
    auto _op_dtype = node->get_element_type();
    auto _op_func = iter->second.func;
    auto _op_expr = iter->second.expr;
    if (_op_dtype == nnfusion::element::f16 &&
        (_op_name == "Acos" || _op_name == "Asin" || _op_name == "Atan"))
    {
        _op_func += "f";
    }
    if (_op_expr == "")
    {
        expr += "@input0@@data_layout@.call(`" + _op_func + "`";
        if (node->get_input_size() > 1)
        {
            expr += ", [";
            size_t i = 1;
            for (; i < node->get_input_size(); ++i)
            {
                expr += "@input" + std::to_string(i) + "@@data_layout@";
                if (i < node->get_input_size() - 1)
                    expr += ", ";
            }
            expr += "]";
        }
        expr += ");";
    }
    else
    {
        expr += replace_input_str(_op_expr) + ";";
    }

    auto data_layout = op::create_layout_from_dims(node->get_output_shape(0));
    // NNFUSION_LOG(INFO) << op::create_code_from_template(
    //     expr, {{"data_layout", vector_to_string<std::vector<std::string>>(data_layout)}});
    return op::create_code_from_template(
        expr, {{"data_layout", vector_to_string<std::vector<std::string>>(data_layout)}});
};

#define REGISTER_ELEM_OP(op_name)                                                                  \
    REGISTER_OP(op_name)                                                                           \
        .infershape(nnfusion::op::infershape::copy_shape_from_inputs)                              \
        .translate_v2([](std::shared_ptr<graph::GNode> node) -> std::string {                      \
            return trans_elementwise(node);                                                        \
        })                                                                                         \
        .infersharedmemory([](std::shared_ptr<graph::GNode> gnode) -> void {                       \
            auto generic_op =                                                                      \
                std::dynamic_pointer_cast<nnfusion::op::GenericOp>(gnode->get_op_ptr());           \
            std::vector<size_t> shared_memory;                                                     \
            auto& input_shape = gnode->get_input_shape(0);                                         \
            auto& output_shape = gnode->get_output_shape(0);                                       \
            if (input_shape.size() == output_shape.size())                                         \
            {                                                                                      \
                shared_memory.clear();                                                             \
                for (size_t i = 0; i < output_shape.size(); i++)                                   \
                {                                                                                  \
                    shared_memory.push_back(1);                                                    \
                }                                                                                  \
            }                                                                                      \
            generic_op->set_shared_memory(shared_memory);                                          \
        });

REGISTER_ELEM_OP(Abs)
REGISTER_ELEM_OP(Asin)
REGISTER_ELEM_OP(Acos)
REGISTER_ELEM_OP(Atan)
REGISTER_ELEM_OP(Ceiling)
REGISTER_ELEM_OP(Cos)
REGISTER_ELEM_OP(Cosh)
REGISTER_ELEM_OP(Erf)
REGISTER_ELEM_OP(Exp)
REGISTER_ELEM_OP(Floor)
REGISTER_ELEM_OP(Log)
REGISTER_ELEM_OP(Maximum)
REGISTER_ELEM_OP(Minimum)
REGISTER_ELEM_OP(Sin)
REGISTER_ELEM_OP(Sinh)
REGISTER_ELEM_OP(Sqrt)
REGISTER_ELEM_OP(Rsqrt)
REGISTER_ELEM_OP(Tan)
REGISTER_ELEM_OP(Tanh)
REGISTER_ELEM_OP(Power)
REGISTER_ELEM_OP(PowerBackwardBase)
REGISTER_ELEM_OP(PowerBackwardExponent)
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
