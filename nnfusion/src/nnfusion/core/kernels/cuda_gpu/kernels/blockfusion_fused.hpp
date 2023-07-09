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
            class BlockFusionFused : public CudaEmitter
            {
            public:
                BlockFusionFused(shared_ptr<KernelContext> ctx, FunctionUnit_p fn = nullptr);
                void set_blockfusion_function(FunctionUnit_p fn);

            private:
                LanguageUnit_p emit_function_name() override;
                LanguageUnit_p emit_function_body() override;
                LanguageUnit_p emit_function_signature() override;
                LanguageUnit_p emit_dependency() override;
                LanguageUnit_p emit_function_call() override;
                LanguageUnit_p emit_comments() override;

                void set_launch_config() override;

                void check_codegen();

            private:
                FunctionUnit_p blockfusion_function;
            };

        } // namespace cuda
    }     // namespace kernels
} // namespace nnfusion