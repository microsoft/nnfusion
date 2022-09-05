// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once
#include "../controlflow_emitter.hpp"

namespace nnfusion
{
    namespace kernels
    {
        namespace cuda
        {
            class Loop : public ControlFlowEmitter
            {
            public:
                Loop(shared_ptr<KernelContext> ctx);

                LanguageUnit_p emit_function_body() override;
                LanguageUnit_p emit_dependency() override;
                LanguageUnit_p emit_function_call() override;
                LanguageUnit_p emit_function_signature() override;
                void set_launch_config() override;

            private:
                void generate_subgraph_code(LanguageUnit_p lu, bool in_cuda);
                TranslationUnit::Pointer m_loop_body_tu;
                ir::BasicBlock::Pointer m_body_instructions;
            };
        } // namespace cuda
    }     // namespace kernels
} // namespace nnfusion
