// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include "../cuda_gpu/cuda_emitter.hpp"
#include "../cuda_gpu/cuda_langunit.hpp"
#include "nnfusion/core/operators/generic_op/generic_op.hpp"

namespace nnfusion
{
    namespace kernels
    {
        namespace cuda
        {
            class CacheBlockCudaKernel : public BlockCudaEmitter
            {
                shared_ptr<nnfusion::op::GenericOp> generic_op;

            public:
                CacheBlockCudaKernel(shared_ptr<KernelContext> ctx, std::string function)
                    : BlockCudaEmitter(ctx)
                    , m_function(function)
                    , generic_op(
                          static_pointer_cast<nnfusion::op::GenericOp>(ctx->gnode->get_op_ptr()))
                {
                    GENERIC_OP_LOGGING();
                }

            private:
                LanguageUnit_p emit_function_signature() override;
                LanguageUnit_p emit_function_body() override;
                LanguageUnit_p emit_dependency() override;
                void set_launch_config() override;

            private:
                std::string m_function;
            };

            // more types of emitters to be implemented

        } // namespace cuda
    }     // namespace kernels
} //namespace nnfusion