// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <iostream>
#include <stdexcept>
#include <stdio.h>

#include "../cuda_emitter.hpp"
#include "../cuda_langunit.hpp"
#include "nnfusion/core/operators/generic_op/generic_op.hpp"

namespace nnfusion
{
    namespace kernels
    {
        namespace cuda
        {
            class Variable : public CudaEmitter
            {
            public:
                Variable(shared_ptr<KernelContext> ctx)
                    : CudaEmitter(ctx)
                {
                    op = static_pointer_cast<nnfusion::op::Variable>(ctx->gnode->get_op_ptr());
                    NNFUSION_CHECK_NOT_NULLPTR(op) << "Node type is not Variable.";

                    threads = ctx->outputs.front()->size(false);

                    std::stringstream tag;
                    tag << "variable_" << op->get_name();
                    custom_tag = tag.str();
                }

                LanguageUnit_p emit_function_body() override
                {
                    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
                    auto& writer = *_lu;

                    writer << "uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;\n";
                    writer << "if(tid < " << threads << ")\n";
                    writer.block_begin();
                    {
                        writer << "output0[tid] = 1.0f;\n";
                    }
                    writer.block_end();

                    return _lu;
                }

                LanguageUnit_p emit_dependency() override

                {
                    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_dep"));
                    _lu->require(header::cuda);
                    return _lu;
                }

                void set_launch_config() override
                {
                    uint32_t block_size_x = 512;
                    size_t block_cnt = align_to_block_size(threads, block_size_x);

                    m_gridDim = dim3(block_cnt, 1, 1);
                    m_blockDim = dim3(block_size_x, 1, 1);
                }

            private:
                shared_ptr<KernelContext> kernel_ctx;
                shared_ptr<nnfusion::op::Variable> op;
                size_t threads;
            };
        } // namespace cuda
    }     // namespace kernels
} // namespace nnfusion

// Register Pad kernel emitter

using namespace nnfusion;
using namespace nnfusion::kernels;
REGISTER_KERNEL_EMITTER("Variable",                                                //op_name
                        Device(CUDA_GPU).TypeConstraint(element::f32).Priority(2), //attrs
                        cuda::Variable)                                            // constructor