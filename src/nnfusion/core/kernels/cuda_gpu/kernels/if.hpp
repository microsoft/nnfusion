// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once
#include "../cuda_emitter.hpp"
#include "../cuda_langunit.hpp"
#include "nnfusion/engine/interpreter.hpp"

namespace nnfusion
{
    namespace kernels
    {
        namespace cuda
        {
            class If : public BlockCudaEmitter
            {
            public:
                If(shared_ptr<KernelContext> ctx);

                LanguageUnit_p emit_function_body() override;
                LanguageUnit_p emit_dependency() override;
                void set_launch_config() override;

            private:
                void generate_branch_code(LanguageUnit_p, bool);
                std::string get_workspace_tensor(nnfusion::descriptor::Tensor::Pointer tensor);
                TranslationUnit::Pointer m_then_branch_tu, m_else_branch_tu;
                descriptor::Tensor::Pointer m_workspace;
                std::unordered_map<std::string, int> m_output_map;
            };
        } // namespace cuda
    }     // namespace kernels
} // namespace nnfusion
