// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include "../cuda_emitter.hpp"
#include "../cuda_langunit.hpp"

namespace nnfusion
{
    namespace kernels
    {
        namespace cuda
        {
            class ElementWiseFused : public BlockCudaEmitter
            {
            public:
                ElementWiseFused(shared_ptr<KernelContext> ctx);

                LanguageUnit_p emit_function_body() override;
                LanguageUnit_p emit_dependency() override;
                LanguageUnit_p emit_function_name() override;
                LanguageUnit_p emit_comments() override;
                static int unique_func_id;

            protected:
                void set_launch_config() override;

            private:
                void compute_best_config(int& grids, int& blocks, int& bound);
                std::shared_ptr<graph::FusedGNode> m_gnode;
            };

        } // namespace cuda
    }     // namespace kernels
} // namespace nnfusion