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
                LanguageUnit_p emit_function_call(std::vector<std::string>) override;
                LanguageUnit_p emit_function_call() override;
                void set_launch_config() override;
                bool is_host_kernel_launch() override;

            private:
                void generate_branch_fused_kernel(LanguageUnit_p _lu, ir::BasicBlock::Pointer instructions, int max_grid_dim, int start_id = 0, int end_id = -1);
                LanguageUnit_p generate_branch_fused_function(int then_start_id, int then_end_id, int else_start_id, int else_end_id);
                LanguageUnit_p generate_branch_seperate_function(const std::string& outer_control, std::shared_ptr<descriptor::Tensor>, std::shared_ptr<ir::Instruction> instruction);
                std::string get_wrapper_func_name(int then_start_id, int then_end_id, int else_start_id, int else_end_id);
                int get_max_grid_dim(int then_start_id, int then_end_id, int else_start_id, int else_end_id);
                void emit_kernel_wrapper(std::shared_ptr<ir::Instruction> ins, LanguageUnit &lu);
                void emit_branch_wrapper(int then_start_id, int then_end_id, int else_start_id, int else_end_id, LanguageUnit &lu, bool emit_all_args);
                bool is_dense_op_group(ir::BasicBlock::Pointer instructions, std::vector<int> inst_id);
                TranslationUnit::Pointer m_then_branch_tu, m_else_branch_tu;
                ir::BasicBlock::Pointer m_then_branch_instructions, m_else_branch_instructions;
                std::vector<std::pair<std::vector<int>, std::vector<int>>> m_kernel_groups;
                std::string m_outer_control_then, m_outer_control_else;
            };
        } // namespace cuda
    }     // namespace kernels
} // namespace nnfusion
