#pragma once
#include "cuda_emitter.hpp"
#include "cuda_langunit.hpp"
#include "nnfusion/engine/interpreter.hpp"

namespace nnfusion
{
    namespace kernels
    {
        namespace cuda_cpu
        {
            struct cuda_cpu_op
            {
                cuda_cpu_op(std::string op, std::string kernel, std::string atomic)
                    : op(op)
                    , math_kernel(kernel)
                    , atomic(atomic)
                {
                }
                std::string op;
                std::string math_kernel;
                std::string atomic;
            };

            static const std::unordered_map<std::string, cuda_cpu_op> CudaCPUElementOpMap{
                {"Abs", cuda_cpu_op("fabsf_cpu", "", "")},
                {"Acos", cuda_cpu_op("acosf_cpu", "", "")},
                {"Asin", cuda_cpu_op("asinf_cpu", "", "")},
                {"Atan", cuda_cpu_op("atanf_cpu", "", "")},
                {"Ceiling", cuda_cpu_op("ceilf_cpu", "", "")},
                {"Cos", cuda_cpu_op("cosf_cpu", "", "")},
                {"Cosh", cuda_cpu_op("coshf_cpu", "", "")},
                {"Erf", cuda_cpu_op("erff_cpu", "", "")},
                {"Exp", cuda_cpu_op("expf_cpu", "", "")},
                {"Floor", cuda_cpu_op("floorf_cpu", "", "")},
                {"Log", cuda_cpu_op("logf_cpu", "", "")},
                {"Max", cuda_cpu_op("fmaxf_cpu", "", "")},
                {"Min", cuda_cpu_op("fminf_cpu", "", "")},
                {"Mod", cuda_cpu_op("mod_cpu", "x0 \% x1", "")},
                {"Sin", cuda_cpu_op("sinf_cpu", "", "")},
                {"Sinh", cuda_cpu_op("sinhf_cpu", "", "")},
                {"Sqrt", cuda_cpu_op("sqrtf_cpu", "", "")},
                {"Rsqrt", cuda_cpu_op("rsqrtf_cpu", "", "")},
                {"Square", cuda_cpu_op("squaref_cpu", "x0 * x0", "")},
                {"Tan", cuda_cpu_op("tanf_cpu", "", "")},
                {"Tanh", cuda_cpu_op("tanh", "", "")},
                {"Power", cuda_cpu_op("powf_cpu", "", "")},
                {"PowerBackwardBase",
                 cuda_cpu_op("power_backward_base_cpu", "x1 != 0 ? x1 * powf(x0, x1 - 1) : 0", "")},
                {"PowerBackwardExponent",
                 cuda_cpu_op("power_backward_exponent_cpu",
                         "x0 > 0 ? powf(x0, x1) * logf(x0) : (x0 == 0 ? (x1 >= 0 ? 0 : 1.0/0.0 /* "
                         "CUDART_INF_F */) : 0.0/0.0 /* CUDART_NAN_F */)",
                         "")},
                {"Subtract", cuda_cpu_op("subtractf_cpu", "x0-x1", "atomicSub")},
                {"Divide", cuda_cpu_op("devide_cpu", "x0 / x1", "")},
                {"DivNoNan", cuda_cpu_op("divnonan_cpu", "x1 != 0 ? fdividef(x0, x1) : 0", "")},
                {"Sign", cuda_cpu_op("sign_cpu", "(x0 > 0) - (x0 < 0)", "")},
                {"Convert", cuda_cpu_op("convert_cpu", "x0", "")},
                // {"Convert", cuda_cpu_op("convert<std::remove_reference<decltype(*input0)>::type, std::remove_reference<decltype(*output0)>::type>", "x0", "")},
                {"Equal", cuda_cpu_op("equal_cpu", "x0 == x1", "")},
                {"NotEqual", cuda_cpu_op("not_equal_cpu", "x0 != x1", "")},
                {"Gelu", cuda_cpu_op("gelu_cpu", "_Gelu(x0)", "")},
                // {"GeluGrad",
                //  cuda_cpu_op("gelugrad",
                //          "x1 * (0.5 * (1.0 + erff(x0 *0.707106781186547524401)) + x0 * expf(-0.5 * "
                //          "x0 * x0) *1.12837916709551257390 * 0.707106781186547524401 * 0.5)",
                //          "")},
                {"GeluGrad",
                 cuda_cpu_op("gelugrad_cpu",
                         "x1 * (normcdff(x0) + x0 * float(0.398942280401432677941) * "
                         "expf(-float(0.5) * x0 * x0))",
                         "")},
                {"Greater", cuda_cpu_op("greater_cpu", "x0 > x1", "")},
                {"GreaterEq", cuda_cpu_op("greater_equal_cpu", "x0 >= x1", "")},
                {"Less",
                 cuda_cpu_op("nnfusion_less_cpu",
                         "x0 < x1",
                         "")}, // workaround, to avoid ambiguous with std::less
                {"LessEq", cuda_cpu_op("less_equal_cpu", "x0 <= x1", "")},
                {"Relu", cuda_cpu_op("relu_cpu", "fmaxf(0,x0)", "")},
                {"Relu6", cuda_cpu_op("relu6_cpu", "fminf(6,fmaxf(0,x0))", "")},
                {"Identity", cuda_cpu_op("identity_cpu", "x0", "")},
                {"Not", cuda_cpu_op("logical_not_cpu", "!x0", "")},
                {"Negative", cuda_cpu_op("negative_cpu", "-x0", "")},
                {"Select", cuda_cpu_op("select_cpu", "(x0 == 0) ? x2 : x1", "")},
                {"ReluBackprop", cuda_cpu_op("relu_backprop_cpu", "x1 * int(x0 > 0)", "")},
                {"Relu6Backprop", cuda_cpu_op("relu_backprop_cpu", "x1 * int(x0 > 0) * int(x0 < 6)", "")},
                {"And", cuda_cpu_op("logical_and_cpu", "x0 & x1", "atomicAnd")},
                {"Or", cuda_cpu_op("logical_or_cpu", "x0 | x1", "atomicOr")},
                {"Add", cuda_cpu_op("add_cpu", "x0 + x1", "atomicAdd")},
                {"Multiply", cuda_cpu_op("mul_cpu", "x0 * x1", "")},
                {"Minimum", cuda_cpu_op("min_f_cpu", "x0 > x1 ? x1 : x0", "atomicMin")},
                {"Maximum", cuda_cpu_op("max_f_cpu", "x0 > x1 ? x0 : x1", "atomicMax")},
                {"Nop", cuda_cpu_op("", "", "")},
                {"Sigmoid", cuda_cpu_op("sigmoid_cpu", "1 / (1 + exp(-x0))", "")},
                {"SigmoidBackprop",
                 cuda_cpu_op("sigmoid_backprop_cpu", "x1 / (2 + exp(-x0) + exp(x0))", "")}};

            class CPUOpEmitter : public KernelEmitter
            {
            public:
                CPUOpEmitter(std::shared_ptr<KernelContext> ctx);
                LanguageUnit_p emit_function_call() override;
                LanguageUnit_p emit_function_body() override;
                LanguageUnit_p emit_dependency() override;
            };
        }
    }
}
