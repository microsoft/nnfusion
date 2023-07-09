// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once
#include "../cpu_kernel_emitter.hpp"

namespace nnfusion
{
    namespace kernels
    {
        namespace cpu
        {
            class BatchMatMulMlas : public MlasKernelEmitter
            {
            public:
                BatchMatMulMlas(shared_ptr<KernelContext> ctx);

                LanguageUnit_p emit_function_body() override;
                LanguageUnit_p emit_dependency() override;

            private:
                size_t reduction_axes;
                nnfusion::Shape arg0_shape, arg1_shape;
            };
        } // namespace cpu
    }     // namespace kernels
} // namespace nnfusion
