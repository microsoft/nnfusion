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
            class LstmEigen : public EigenKernelEmitter
            {
            public:
                LstmEigen(shared_ptr<KernelContext> ctx);

                LanguageUnit_p emit_function_body() override;
                LanguageUnit_p emit_dependency() override;

            private:
                int64_t SEQ_LEN, SIZE_BATCH, SIZE_INPUT, NUM_DRT, SIZE_HIDDEN;
                int64_t NUM_GATE, SIZE_GATE;
                // attr
                std::string direction;
                int64_t /* hidden_size, */ input_forget;
                void emit_compute_input_helper(nnfusion::codegen::CodeWriter& lu);
                void emit_compute_hidden_helper(nnfusion::codegen::CodeWriter& lu);
            };
        } // namespace cpu
    }     // namespace kernels
} // namespace nnfusion
