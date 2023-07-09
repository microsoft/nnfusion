// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once
#include "../cpu_helper.hpp"
#include "../cpu_kernel_emitter.hpp"
#include "../cpu_kernelops.hpp"
#include "nnfusion/core/operators/generic_op/generic_op.hpp"

namespace nnfusion
{
    namespace kernels
    {
        namespace cpu
        {
            template <typename T>
            class ElementwiseEigen : public EigenKernelEmitter
            {
            public:
                ElementwiseEigen(shared_ptr<KernelContext> ctx)
                    : EigenKernelEmitter(ctx)
                {
                    data_size = m_context->inputs.front()->size(false);
                    for (auto arg : ctx->inputs)
                    {
                        data_types.push_back(arg->get_element_type().c_type_string());
                    }
                    data_types.push_back(ctx->outputs[0]->get_element_type().c_type_string());
                }

                LanguageUnit_p emit_function_body() override
                {
                    if (CpuOpMap<T>::eigen_op == nullptr)
                    {
                        return nullptr;
                    }

                    auto op = CpuOpMap<T>::eigen_op;

                    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
                    auto& lu = *_lu;

                    if (CpuOpMap<T>::eigen_math_kernel != nullptr)
                    {
                        auto math_kernel = get_eigen_math_kernel(
                            op, CpuOpMap<T>::eigen_math_kernel, data_size, data_types);
                        NNFUSION_CHECK_NOT_NULLPTR(math_kernel);
                        lu.require(math_kernel);
                    }
                    auto num_inputs = data_types.size() - 1;
                    NNFUSION_CHECK(num_inputs > 0)
                        << "At least one input and one output tensor for elementwise-op.";

                    lu << op << "_" << data_size << "(thread_pool, ";
                    for (size_t i = 0; i < num_inputs; ++i)
                    {
                        lu << "input" << i << ", ";
                    }
                    lu << "output0);";

                    return _lu;
                }

                LanguageUnit_p emit_dependency() override
                {
                    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_dep"));
                    _lu->require(header::eigen_tensor);

                    return _lu;
                }

            private:
                size_t data_size;
                vector<string> data_types;
            };
        } // namespace cpu
    }     // namespace kernels
} // namespace nnfusion
