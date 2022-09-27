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
            class FuncForward : public CudaEmitter
            {
            public:
                FuncForward(shared_ptr<KernelContext> ctx);
                LanguageUnit_p emit_function_body() override;
                LanguageUnit_p emit_dependency() override;
                LanguageUnit_p emit_block_kernel_call(std::vector<std::string> params) override;
                void update_context_from_gnode(std::shared_ptr<nnfusion::graph::GNode> gnode);
                void set_launch_config() override;
                static std::string m_block_func_name;
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

            private:
                std::string m_block_func_name;
                void generate_subgraph_code(LanguageUnit_p);
                size_t m_workspace_size;
                TranslationUnit::Pointer m_loop_body_tu;
                ir::BasicBlock::Pointer m_body_instructions;
            };
        } // namespace cuda
    }     // namespace kernels
} // namespace nnfusion
