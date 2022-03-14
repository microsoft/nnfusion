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
            class EmbeddingGrad : public CudaLibEmitter
            {
            public:
                EmbeddingGrad(shared_ptr<KernelContext> ctx);

                LanguageUnit_p emit_function_body() override;
                LanguageUnit_p emit_dependency() override;
                LanguageUnit_p emit_function_signature() override;

            private:
                nnfusion::Shape x_shape, y_shape, indices_shape;
            };

        } // namespace cuda
    }     // namespace kernels
} // namespace nnfusion