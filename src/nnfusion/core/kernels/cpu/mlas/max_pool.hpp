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
            class MaxPoolMlas : public MlasKernelEmitter
            {
            public:
                MaxPoolMlas(shared_ptr<KernelContext> ctx);

                LanguageUnit_p emit_function_body() override;
                LanguageUnit_p emit_dependency() override;

            private:
                nnfusion::Shape input_shape, output_shape, window_shape, padding;
                nnfusion::Shape padding_below, padding_above;
                nnfusion::Strides window_stride;
                string dtype, data_format;
            };
        } // namespace cpu
    }     // namespace kernels
} // namespace nnfusion
