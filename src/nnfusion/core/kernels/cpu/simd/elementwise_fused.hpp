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
            class ElementwiseFused : public SimdKernelEmitter
            {
            public:
                ElementwiseFused(shared_ptr<KernelContext> ctx);

                LanguageUnit_p emit_function_body() override;
                LanguageUnit_p emit_dependency() override;
                LanguageUnit_p emit_function_name() override;
                LanguageUnit_p emit_comments() override;
                static int unique_func_id;

            private:
                std::shared_ptr<KernelContext> FuseContext();
                void FuseFunctionBody(LanguageUnit& lu);
                std::unordered_map<std::string, std::string> in_args, out_args, out_types,
                    local_tensors;
            };

        } // namespace cpu
    }     // namespace kernels
} // namespace nnfusion
