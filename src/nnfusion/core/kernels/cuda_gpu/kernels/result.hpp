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
            class Result : public CudaLibEmitter
            {
            public:
                Result(shared_ptr<KernelContext> ctx);

                LanguageUnit_p emit_function_body() override;
                LanguageUnit_p emit_dependency() override;
                LanguageUnit_p emit_function_signature() override;
                bool is_eliminative() override;

            private:
                shared_ptr<KernelContext> kernel_ctx;
                bool need_copy_to_host;
            };
        } // namespace cuda
    }     // namespace kernels
} // namespace nnfusion