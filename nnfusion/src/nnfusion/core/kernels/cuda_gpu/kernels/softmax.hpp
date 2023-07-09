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
            class Softmax : public CudaLibEmitter
            {
            public:
                Softmax(shared_ptr<KernelContext> ctx);

                LanguageUnit_p emit_function_body() override;
                LanguageUnit_p emit_dependency() override;
                LanguageUnit_p emit_function_signature() override;
                bool require_cudnn_handle() override { return true; }
                LanguageUnit_p
                    cudnn_tensor_descriptor_from_shape_for_softmax(const nnfusion::Shape& shape,
                                                                   string desc);

            private:
                shared_ptr<KernelContext> kernel_ctx;
                nnfusion::Shape input_shape, output_shape;
                std::string algorithm;
                int N, D;
                element::Type dtype;
                bool is_log_softmax;
            };

            class SoftmaxGrad : public CudaLibEmitter
            {
            public:
                SoftmaxGrad(shared_ptr<KernelContext> ctx);

                LanguageUnit_p emit_function_body() override;
                LanguageUnit_p emit_dependency() override;
                LanguageUnit_p emit_function_signature() override;
                bool require_cudnn_handle() override { return true; }
                LanguageUnit_p
                    cudnn_tensor_descriptor_from_shape_for_softmax(const nnfusion::Shape& shape,
                                                                   string desc);

            private:
                shared_ptr<KernelContext> kernel_ctx;
                nnfusion::Shape input_shape, output_shape;
                std::string algorithm;
            };
        } // namespace cuda
    }     // namespace kernels
} // namespace nnfusion
