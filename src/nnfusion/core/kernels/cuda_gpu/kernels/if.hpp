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
                LanguageUnit_p emit_function_signature() override;
                LanguageUnit_p emit_function_call() override;
                void set_launch_config() override;

            private:
                void generate_branch_fused_kernel(LanguageUnit_p, bool);
                LanguageUnit_p generate_branch_seperate_kernel(const std::string& outer_control, std::shared_ptr<descriptor::Tensor>, std::shared_ptr<ir::Instruction> instruction);
                void emit_kernel_wrapper(std::shared_ptr<ir::Instruction> ins, LanguageUnit &lu);
                TranslationUnit::Pointer m_then_branch_tu, m_else_branch_tu;
                ir::BasicBlock::Pointer m_then_branch_instructions, m_else_branch_instructions;
            };
        } // namespace cuda
    }     // namespace kernels
} // namespace nnfusion
