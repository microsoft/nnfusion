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
            class Attention : public CudaLibEmitter
            {
            public:
                Attention(shared_ptr<KernelContext> ctx);

                LanguageUnit_p emit_function_body() override;
                LanguageUnit_p emit_dependency() override;
                LanguageUnit_p emit_function_signature() override;

            private:
                shared_ptr<nnfusion::op::GenericOp> generic_op;
                element::Type dtype;
                size_t batch_size, sequence_length, hidden_size, workspace_size,
                    past_sequence_length = 0;
                int num_heads, head_size;
                bool unidirectional;
                bool use_2d_attention_mask;
                std::string mask_start = "nullptr";
                std::shared_ptr<nnfusion::descriptor::Tensor> workspace_tensor;
            };

        } // namespace cuda
    }     // namespace kernels
} // namespace nnfusion