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
            class Loop : public KernelEmitter
            {
            public:
                Loop(shared_ptr<KernelContext> ctx);

                LanguageUnit_p emit_function_body() override;
                LanguageUnit_p emit_dependency() override;
                // LanguageUnit_p emit_function_signature() override;
            };
        } // namespace cuda
    }     // namespace kernels
} // namespace nnfusion