// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <iostream>
#include "nnfusion/common/languageunit.hpp"
#include "nnfusion/core/kernels/kernel_emitter.hpp"
#include "nnfusion/core/kernels/kernel_registration.hpp"
#include "nnfusion/core/operators/generic_op/generic_op.hpp"

//Classes
namespace nnfusion
{
    namespace kernels
    {
        namespace cpu
        {
            class BatchMatMulRef : public KernelEmitter
            {
            public:
                BatchMatMulRef(shared_ptr<KernelContext> ctx)
                    : KernelEmitter(ctx)
                    , generic_op(
                          static_pointer_cast<nnfusion::op::GenericOp>(ctx->gnode->get_op_ptr()))
                {
                }

                LanguageUnit_p emit_function_body() override
                {
                    const nnfusion::Shape& input_shape_0 = m_context->inputs[0]->get_shape();
                    const nnfusion::Shape& input_shape_1 = m_context->inputs[1]->get_shape();

                    // Check conditions that pair of inputs must satisfy to run BatchMatMul

                    bool transA = generic_op->localOpConfig.getRoot()["adj_x"]["b"];
                    bool transB = generic_op->localOpConfig.getRoot()["adj_y"]["b"];
                    size_t A1 = 1LU, A2, A3, A4;
                    for (int i = input_shape_0.size() - 3; i >= 0; --i)
                        A1 *= input_shape_0[i];
                    int m, n, k;

                    if (!transA && !transB)
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
	for (long STEP = 0; STEP < @batch@; ++STEP) {
		@T@ (*x)[@X1@] = decltype(x)(((@T@*)input0) + STEP * @n@ * @k@);
		@T@ (*y)[@Y1@] = decltype(y)(((@T@*)input1) + STEP * @k@ * @m@);
		@T@ (*z)[@m@] = decltype(z)(((@T@*)output0) + STEP * @n@ * @m@);
		for (int i = 0; i < @n@; ++i)
			for (int j = 0; j < @m@; ++j) {
				z[i][j] = 0;
				for (int k = 0; k < @k@; ++k)
					z[i][j] += x@X_IDX@ * y@Y_IDX@;
			}
	}
                    )",
                        {
                            {"batch", A1},
                            {"T", dtype},
                            {"X0", input_shape_0[input_shape_0.size() - 2]},
                            {"X1", input_shape_0[input_shape_0.size() - 1]},
                            {"Y0", input_shape_1[input_shape_1.size() - 2]},
                            {"Y1", input_shape_1[input_shape_1.size() - 1]},
                            {"n", n},
                            {"k", k},
                            {"m", m},
                            {"X_IDX", transA ? "[k][i]" : "[i][k]"},
                            {"Y_IDX", transB ? "[j][k]" : "[k][j]"},
                        });
                    lu << code << "\n";
                    return std::make_shared<LanguageUnit>(std::move(lu));
                }

                LanguageUnit_p emit_dependency() override
                {
                    LanguageUnit lu(get_function_name() + "_dep");
                    return std::make_shared<LanguageUnit>(std::move(lu));
                }

            private:
                shared_ptr<nnfusion::op::GenericOp> generic_op;
            };

            REGISTER_KERNEL_EMITTER(
                "BatchMatMul",                                                     // op_name
                Device(GENERIC_CPU).TypeConstraint(element::f32).Tag("reference"), // attrs
                BatchMatMulRef)                                                    // constructor

        } // namespace cpu
    }     // namespace kernels
} // namespace nnfusion
