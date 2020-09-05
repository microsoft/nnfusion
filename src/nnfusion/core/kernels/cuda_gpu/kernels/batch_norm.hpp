// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once
#include "../cuda_cudnn.hpp"
#include "../cuda_emitter.hpp"
#include "../cuda_langunit.hpp"

namespace nnfusion
{
    namespace kernels
    {
        namespace cuda
        {
            class BatchNorm : public CudaLibEmitter
            {
            public:
                BatchNorm(shared_ptr<KernelContext> ctx);

                LanguageUnit_p emit_function_body() override;
                LanguageUnit_p emit_dependency() override;
                LanguageUnit_p emit_function_signature() override;
                bool require_cudnn_handle() override { return true; }
            private:
                shared_ptr<KernelContext> kernel_ctx;

                // Fields for codegen
                element::Type dtype;
                Shape tensor_shape, param_shape;
                double epsilon = 0.0;
                string direction = "CUDNNEmitter::Prop::Inference";

                //<todo> For future purpose
                bool global_states = false;
                bool save_states = false;
            };

            class BatchNormNCHW : public BlockCudaEmitter
            {
            public:
                BatchNormNCHW(shared_ptr<KernelContext> ctx);

                LanguageUnit_p emit_function_body() override;
                LanguageUnit_p emit_dependency() override;
                void set_launch_config() override;

            private:
                shared_ptr<KernelContext> kernel_ctx;

                // Fields for codegen
                element::Type dtype;
                Shape tensor_shape, param_shape;
                double epsilon = 0.0;
                string direction = "CUDAEmitter::Prop::Inference";

                //<todo> For future purpose
                bool global_states = false;
                bool save_states = false;
            };
        } // namespace cuda
    }     // namespace kernels
} // namespace nnfusion