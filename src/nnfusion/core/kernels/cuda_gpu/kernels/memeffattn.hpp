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
            class MemEffAttn : public CudaLibEmitter
            {
            public:
                MemEffAttn(shared_ptr<KernelContext> ctx);

                LanguageUnit_p emit_function_body() override;
                LanguageUnit_p emit_dependency() override;
                LanguageUnit_p emit_function_signature() override;

            private:
                shared_ptr<nnfusion::op::GenericOp> generic_op;
                element::Type dtype;
                int batch_size, seq_len, seq_len_kv, num_heads, head_size, head_size_v, hidden_size,
                    workspace_size, idx;
                float p_dropout, softmax_scale;
                bool is_causal;
                std::shared_ptr<nnfusion::descriptor::Tensor> workspace_tensor;
            };

        } // namespace cuda
    }     // namespace kernels
} // namespace nnfusion