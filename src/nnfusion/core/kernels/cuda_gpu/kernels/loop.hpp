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
            class Loop : public CudaEmitter
            {
            public:
                Loop(shared_ptr<KernelContext> ctx);

                LanguageUnit_p emit_function_body() override;
                LanguageUnit_p emit_dependency() override;
                void set_launch_config() override;

            private:
                void generate_subgraph_code(LanguageUnit_p);
                std::string get_workspace_tensor(nnfusion::descriptor::Tensor::Pointer tensor);
                descriptor::Tensor::Pointer m_workspace;
                TranslationUnit::Pointer m_loop_body_tu;
                size_t m_loop_carried_var;
                size_t m_scanned_out_var;
                std::unordered_map<std::string, int> m_loop_output_map;
            };
        } // namespace cuda
    }     // namespace kernels
} // namespace nnfusion
