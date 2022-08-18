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
                void generate_branch_fused_kernel(LanguageUnit_p _lu, ir::BasicBlock::Pointer instructions, int start_id = 0, int end_id = -1);
                LanguageUnit_p generate_branch_fused_function(const std::string& outer_control, bool else_branch = false, int start_id = 0, int end_id = -1);
                LanguageUnit_p generate_branch_seperate_function(const std::string& outer_control, std::shared_ptr<descriptor::Tensor>, std::shared_ptr<ir::Instruction> instruction);
                void emit_kernel_wrapper(std::shared_ptr<ir::Instruction> ins, LanguageUnit &lu);
                void emit_branch_wrapper(bool else_branch, int start_id, int end_id, LanguageUnit &lu, bool emit_all_args);
                TranslationUnit::Pointer m_then_branch_tu, m_else_branch_tu;
                ir::BasicBlock::Pointer m_then_branch_instructions, m_else_branch_instructions;
                std::vector<std::vector<int>> m_then_kernel_groups;
                std::vector<std::vector<int>> m_else_kernel_groups;
            };
        } // namespace cuda
    }     // namespace kernels
} // namespace nnfusion
