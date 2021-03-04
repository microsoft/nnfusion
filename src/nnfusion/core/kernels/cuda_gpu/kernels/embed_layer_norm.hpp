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
            class EmbedLayerNorm : public CudaLibEmitter
            {
            public:
                EmbedLayerNorm(shared_ptr<KernelContext> ctx);

                LanguageUnit_p emit_function_body() override;
                LanguageUnit_p emit_dependency() override;
                LanguageUnit_p emit_function_signature() override;

            private:
                shared_ptr<nnfusion::op::GenericOp> generic_op;
                element::Type dtype;
                size_t batch_size, sequence_length, hidden_size;
                float epsilon;
            };

        } // namespace cuda
    }     // namespace kernels
} // namespace nnfusion