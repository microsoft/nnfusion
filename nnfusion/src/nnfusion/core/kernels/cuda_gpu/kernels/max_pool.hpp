// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once
#include "../cuda_emitter.hpp"
#include "../cuda_langunit.hpp"

namespace nnfusion
{
    namespace kernels
    {
        namespace cuda
        {
            class MaxPool1D : public BlockCudaEmitter
            {
            public:
                MaxPool1D(shared_ptr<KernelContext> ctx);

                LanguageUnit_p emit_function_body() override;
                LanguageUnit_p emit_dependency() override;
                void set_launch_config() override;

            private:
                shared_ptr<KernelContext> kernel_ctx;
                nnfusion::Shape input_shape, output_shape, window_shape, padding_below,
                    padding_above;
                element::Type input_type, output_type;
                nnfusion::Strides window_stride;

                size_t window_width, window_stride_width, input_width, output_width;
            };

            class MaxPoolmD : public CudaLibEmitter
            {
            public:
                MaxPoolmD(shared_ptr<KernelContext> ctx);

                LanguageUnit_p emit_function_body() override;
                LanguageUnit_p emit_dependency() override;
                LanguageUnit_p emit_function_signature() override;
                bool require_cudnn_handle() override { return true; }
            private:
                shared_ptr<KernelContext> kernel_ctx;
                element::Type input_type, output_type;
                nnfusion::Shape input_shape, output_shape, window_shape, padding_below,
                    padding_above;
                nnfusion::Strides window_stride;
            };
        } // namespace cuda
    }     // namespace kernels
} // namespace nnfusion