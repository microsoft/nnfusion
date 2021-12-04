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
            class FuncForward : public CudaEmitter
            {
            public:
                FuncForward(shared_ptr<KernelContext> ctx);
                LanguageUnit_p emit_function_body() override;
                LanguageUnit_p emit_dependency() override;
                LanguageUnit_p emit_block_kernel_call(std::vector<std::string> params) override;
                void set_launch_config() override;
                static std::string m_block_func_name;
            };

            class Recursion : public CudaEmitter
            {
            public:
                Recursion(shared_ptr<KernelContext> ctx);

                LanguageUnit_p emit_function_body() override;
                LanguageUnit_p emit_dependency() override;
                void set_launch_config() override;

            private:
                LanguageUnit_p m_saved_func_body;
                std::string m_block_func_name;
                void generate_subgraph_code(LanguageUnit_p);
                std::string get_workspace_tensor(nnfusion::descriptor::Tensor::Pointer tensor);
                descriptor::Tensor::Pointer m_workspace;
                TranslationUnit::Pointer m_loop_body_tu;
                std::unordered_map<std::string, int> m_loop_output_map;
            };
        } // namespace cuda
    }     // namespace kernels
} // namespace nnfusion
