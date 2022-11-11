// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once
#include "../controlflow_emitter.hpp"
#include "loop.hpp"

namespace nnfusion
{
    namespace kernels
    {
        namespace cuda
        {
            class While : public Loop
            {
            public:
                While(shared_ptr<KernelContext> ctx);

                LanguageUnit_p emit_function_body() override;
            };
        } // namespace cuda
    }     // namespace kernels
} // namespace nnfusion
