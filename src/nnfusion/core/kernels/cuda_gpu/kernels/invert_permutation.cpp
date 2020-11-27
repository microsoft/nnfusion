// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <iostream>
#include <stdio.h>
#include <vector>

#include "../cuda_emitter.hpp"
#include "../cuda_langunit.hpp"
#include "nnfusion/core/operators/generic_op/generic_op.hpp"

namespace nnfusion
{
    namespace kernels
    {
        namespace cuda
        {
            class InvertPermutation : public BlockCudaEmitter
            {
            public:
                InvertPermutation(shared_ptr<KernelContext> ctx)
                    : BlockCudaEmitter(ctx)
                {
                    data_size = ctx->inputs[0]->size(false);
                }

                LanguageUnit_p emit_function_body() override
                {
                    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
                    auto& lu = *_lu;

                    lu << "uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;\n";
                    lu << "if(tid < " << data_size << ")\n";
                    lu.block_begin();
                    {
                        lu << "output0[input0[tid]] = tid;\n";
                    }
                    lu.block_end();
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
                    size_t block_cnt = align_to_block_size(data_size, block_size_x);

                    m_gridDim = dim3(block_cnt, 1, 1);
                    m_blockDim = dim3(block_size_x, 1, 1);
                }

            private:
                int data_size;
            };
        }
    }
}

using namespace nnfusion;
using namespace nnfusion::kernels;
REGISTER_KERNEL_EMITTER("InvertPermutation",
                        Device(CUDA_GPU)
                            .TypeConstraint(element::f32)
                            .Priority(2), // TODO: this op input and output will all be int
                        cuda::InvertPermutation)