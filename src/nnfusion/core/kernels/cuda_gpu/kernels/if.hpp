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
            class If : public ControlFlowEmitter
            {
            public:
                If(shared_ptr<KernelContext> ctx);

                LanguageUnit_p emit_function_body() override;
                LanguageUnit_p emit_dependency() override;
                void set_launch_config() override;

            private:
                void generate_branch_code(LanguageUnit_p, bool);
                TranslationUnit::Pointer m_then_branch_tu, m_else_branch_tu;
                ir::BasicBlock::Pointer m_then_branch_instructions, m_else_branch_instructions;
            };
        } // namespace cuda
    }     // namespace kernels
} // namespace nnfusion
