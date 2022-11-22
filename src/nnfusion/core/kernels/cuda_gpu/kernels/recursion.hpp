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
            class FuncForward : public ControlFlowEmitter
            {
            public:
                FuncForward(shared_ptr<KernelContext> ctx);
                LanguageUnit_p emit_function_body() override;
                LanguageUnit_p emit_dependency() override;
                LanguageUnit_p emit_block_kernel_call(std::vector<std::string> params) override;
                void update_context_from_gnode(std::shared_ptr<nnfusion::graph::GNode> gnode);
                void set_launch_config() override;
                static std::string m_block_func_name;
                bool is_host_kernel_launch() override;
            private:
                std::shared_ptr<nnfusion::descriptor::Tensor> m_workspace;
            };

            class Recursion : public ControlFlowEmitter
            {
            public:
                Recursion(shared_ptr<KernelContext> ctx);

                LanguageUnit_p emit_function_body() override;
                LanguageUnit_p emit_dependency() override;
                void set_launch_config() override;
                bool is_host_kernel_launch() override;

            private:
                std::string m_block_func_name;
                void generate_subgraph_code(LanguageUnit_p);
                std::string inline_kernel(std::shared_ptr<nnfusion::ir::Instruction> ins);
                std::string to_stack(std::string inlined, std::shared_ptr<nnfusion::ir::Instruction> ins);
                size_t m_workspace_size;
                TranslationUnit::Pointer m_loop_body_tu;
                ir::BasicBlock::Pointer m_body_instructions;
                std::unordered_map<std::string, size_t> m_param_offset;
            };
        } // namespace cuda
    }     // namespace kernels
} // namespace nnfusion
