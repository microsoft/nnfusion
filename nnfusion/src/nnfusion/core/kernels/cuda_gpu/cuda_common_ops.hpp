// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once
#include <string>
#include <unordered_map>

namespace nnfusion
{
    namespace kernels
    {
        namespace cuda
        {
            struct cuda_op
            {
                cuda_op(std::string op, std::string kernel, std::string atomic)
                    : op(op)
                    , math_kernel(kernel)
                    , atomic(atomic)
                {
                }
                std::string op;
                std::string math_kernel;
                std::string atomic;
            };

            static const std::unordered_map<std::string, cuda_op> CudaElementOpMap{
                {"Abs", cuda_op("fabsf", "", "")},
                {"Acos", cuda_op("acosf", "", "")},
                {"Asin", cuda_op("asinf", "", "")},
                {"Atan", cuda_op("atanf", "", "")},
                {"Ceiling", cuda_op("ceilf", "", "")},
                {"Cos", cuda_op("cosf", "", "")},
                {"Cosh", cuda_op("coshf", "", "")},
                {"Erf", cuda_op("erff", "", "")},
                {"Exp", cuda_op("expf", "", "")},
                {"Floor", cuda_op("floorf", "", "")},
                {"Log", cuda_op("logf", "", "")},
                {"Max", cuda_op("fmaxf", "", "")},
                {"Min", cuda_op("fminf", "", "")},
                {"Sin", cuda_op("sinf", "", "")},
                {"Sinh", cuda_op("sinhf", "", "")},
                {"Sqrt", cuda_op("sqrtf", "", "")},
                {"Rsqrt", cuda_op("rsqrtf", "", "")},
                {"Square", cuda_op("squaref", "x0 * x0", "")},
                {"Tan", cuda_op("tanf", "", "")},
                {"Tanh", cuda_op("tanhf", "", "")},
                {"Power", cuda_op("powf", "", "")},
                {"PowerBackwardBase",
                 cuda_op("power_backward_base", "x1 != 0 ? x1 * powf(x0, x1 - 1) : 0", "")},
                {"PowerBackwardExponent",
                 cuda_op("power_backward_exponent",
                         "x0 > 0 ? powf(x0, x1) * logf(x0) : (x0 == 0 ? (x1 >= 0 ? 0 : 1.0/0.0 /* "
                         "CUDART_INF_F */) : 0.0/0.0 /* CUDART_NAN_F */)",
                         "")},
                {"Subtract", cuda_op("subtractf", "x0-x1", "atomicSub")},
                {"Divide", cuda_op("fdividef", "", "")},
                {"DivNoNan", cuda_op("divnonan", "x1 != 0 ? fdividef(x0, x1) : 0", "")},
                {"Sign", cuda_op("sign", "(x0 > 0) - (x0 < 0)", "")},
                {"Convert", cuda_op("convert", "x0", "")},
                // {"Convert", cuda_op("convert<std::remove_reference<decltype(*input0)>::type, std::remove_reference<decltype(*output0)>::type>", "x0", "")},
                {"Equal", cuda_op("equal", "x0 == x1", "")},
                {"NotEqual", cuda_op("not_equal", "x0 != x1", "")},
                {"Gelu", cuda_op("gelu", "_Gelu(x0)", "")},
                // {"GeluGrad",
                //  cuda_op("gelugrad",
                //          "x1 * (0.5 * (1.0 + erff(x0 *0.707106781186547524401)) + x0 * expf(-0.5 * "
                //          "x0 * x0) *1.12837916709551257390 * 0.707106781186547524401 * 0.5)",
                //          "")},
                {"GeluGrad",
                 cuda_op("gelugrad",
                         "x1 * (normcdff(x0) + x0 * float(0.398942280401432677941) * "
                         "expf(-float(0.5) * x0 * x0))",
                         "")},
                {"Greater", cuda_op("greater", "x0 > x1", "")},
                {"GreaterEq", cuda_op("greater_equal", "x0 >= x1", "")},
                {"Less",
                 cuda_op("nnfusion_less",
                         "x0 < x1",
                         "")}, // workaround, to avoid ambiguous with std::less
                {"LessEq", cuda_op("less_equal", "x0 <= x1", "")},
                {"Relu", cuda_op("relu", "fmaxf(0,x0)", "")},
                {"Relu6", cuda_op("relu6", "fminf(6,fmaxf(0,x0))", "")},
                {"Not", cuda_op("logical_not", "!x0", "")},
                {"Negative", cuda_op("negative", "-x0", "")},
                {"Select", cuda_op("select", "(x0 == 0) ? x2 : x1", "")},
                {"ReluBackprop", cuda_op("relu_backprop", "x1 * int(x0 > 0)", "")},
                {"Relu6Backprop", cuda_op("relu_backprop", "x1 * int(x0 > 0) * int(x0 < 6)", "")},
                {"And", cuda_op("logical_and", "x0 & x1", "atomicAnd")},
                {"Or", cuda_op("logical_or", "x0 | x1", "atomicOr")},
                {"Add", cuda_op("add", "x0 + x1", "atomicAdd")},
                {"Multiply", cuda_op("mul", "x0 * x1", "")},
                {"Minimum", cuda_op("min_f", "x0 > x1 ? x1 : x0", "atomicMin")},
                {"Maximum", cuda_op("max_f", "x0 > x1 ? x0 : x1", "atomicMax")},
                {"Nop", cuda_op("", "", "")},
                {"Sigmoid", cuda_op("sigmoid", "1 / (1 + expf(-x0))", "")},
                {"SigmoidBackprop",
                 cuda_op("sigmoid_backprop", "x1 / (2 + expf(-x0) + expf(x0))", "")}};
        } // namespace cuda
    }     // namespace kernels
} // namespace nnfusion
