// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <iostream>
#include "../../cpu_op_emitter.hpp"
#include "../../cuda_langunit.hpp"

namespace nnfusion
{
    namespace kernels
    {
        namespace cuda_cpu
        {
            class Dot : public CPUOpEmitter
            {
            public:
                Dot(shared_ptr<KernelContext> ctx)
                    : CPUOpEmitter(ctx)
                {
                }

                LanguageUnit_p emit_function_body() override
                {
                    const nnfusion::Shape& input_shape_0 = m_context->inputs[0]->get_shape();
                    const nnfusion::Shape& input_shape_1 = m_context->inputs[1]->get_shape();

                    // Check conditions that pair of inputs must satisfy to run BatchMatMul

                    auto gemm = static_pointer_cast<nnfusion::op::Dot>(m_context->gnode->get_op_ptr());
                    auto transA = gemm->get_transpose_A();
                    auto transB = gemm->get_transpose_B();
                    NNFUSION_CHECK(input_shape_0.size() == 2 && input_shape_1.size() == 2) << "not implemented";
                    size_t A2, A3, A4;
                    int m, n, k;

                    if (!transA && !transB) // A2*A3 A3*A4
                    {
                        A2 = input_shape_0[input_shape_0.size() - 2];
                        A3 = input_shape_0[input_shape_0.size() - 1];
                        A4 = input_shape_1[input_shape_1.size() - 1];
                        m = A4, n = A2, k = A3;
                    }
                    else if (!transA && transB)
                    {
                        A2 = input_shape_0[input_shape_0.size() - 2];
                        A3 = input_shape_0[input_shape_0.size() - 1];
                        A4 = input_shape_1[input_shape_1.size() - 2];
                        m = A4, n = A2, k = A3;
                    }
                    else if (transA && !transB)
                    {
                        A2 = input_shape_0[input_shape_0.size() - 1];
                        A3 = input_shape_0[input_shape_0.size() - 2];
                        A4 = input_shape_1[input_shape_1.size() - 1];
                        m = A4, n = A2, k = A3;
                    }
                    else
                    { // transA && transB
                        A2 = input_shape_0[input_shape_0.size() - 1];
                        A3 = input_shape_0[input_shape_0.size() - 2];
                        A4 = input_shape_1[input_shape_1.size() - 2];
                        m = A4, n = A2, k = A3;
                    }

                    std::string dtype = m_context->dtypes[0];
                    NNFUSION_CHECK(dtype == m_context->dtypes[1]);
                    NNFUSION_CHECK(dtype == m_context->dtypes[2]);
                    LanguageUnit lu(get_function_name());
                    auto code = nnfusion::op::create_code_from_template(
                        R"(
@T@ (*x)[@X1@] = decltype(x)(((@T@*)input0));
@T@ (*y)[@Y1@] = decltype(y)(((@T@*)input1));
@T@ (*z)[@m@] = decltype(z)(((@T@*)output0));
for (int i = 0; i < @n@; ++i)
    for (int j = 0; j < @m@; ++j) {
        z[i][j] = 0;
        for (int k = 0; k < @k@; ++k)
            z[i][j] += x@X_IDX@ * y@Y_IDX@;
    }
                    )",
                        {
                            {"T", dtype},
                            {"X0", input_shape_0[input_shape_0.size() - 2]},
                            {"X1", input_shape_0[input_shape_0.size() - 1]},
                            {"Y0", input_shape_1[input_shape_1.size() - 2]},
                            {"Y1", input_shape_1[input_shape_1.size() - 1]},
                            {"n", n},
                            {"k", k},
                            {"m", m},
                            {"X_IDX", transA ? "[k][i]" : "[i][k]"},
                            {"Y_IDX", transB ? "[j][k]" : "[k][j]"}
                        });
                    lu << code << "\n";
                    return std::make_shared<LanguageUnit>(std::move(lu));
                }

                LanguageUnit_p emit_dependency() override
                {
                    LanguageUnit lu(get_function_name() + "_dep");
                    return std::make_shared<LanguageUnit>(std::move(lu));
                }

            };

            REGISTER_KERNEL_EMITTER(
                "Dot",                                                     // op_name
                Device(SINGLE_CPU).TypeConstraint(element::f32).Tag("reference"), // attrs
                Dot)                                                    // constructor

        } // namespace cpu
    }     // namespace kernels
} // namespace nnfusion
