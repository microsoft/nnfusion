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
                void set_launch_config() override;

            private:
                void generate_subgraph_code(LanguageUnit_p);
                TranslationUnit::Pointer m_loop_body_tu;
            };
        } // namespace cuda
    }     // namespace kernels
} // namespace nnfusion
