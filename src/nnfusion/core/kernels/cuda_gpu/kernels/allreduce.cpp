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
            class SuperScalerAllReduce : public KernelEmitter
            {
            public:
                SuperScalerAllReduce(shared_ptr<KernelContext> ctx)
                    : KernelEmitter(ctx, "SuperScaler")
                {
                }

                LanguageUnit_p emit_function_body() override
                {
                    LanguageUnit_p _lu(new LanguageUnit(get_function_name()));
                    auto& lu = *_lu;
                    auto data_size = m_context->inputs.front()->size(false);
                    // allreduce and applygradient use the same stream.
                    lu << "super_scaler_all_reduce(input0, output0, " << data_size << ", &stream);";
                    return _lu;
                }

                LanguageUnit_p emit_dependency()
                {
                    LanguageUnit_p _lu(new LanguageUnit(get_function_name() + "_dep"));
                    _lu->require(header::cuda);
                    _lu->require(header::super_scaler); // This require nccl, mpi
                    // _lu->require(declaration::allreduce_stream);
                    // _lu->require(declaration::applygradient_stream);
                    return _lu;
                }
            };
        }
    }
}

using namespace nnfusion;
using namespace nnfusion::kernels;
REGISTER_KERNEL_EMITTER("AllReduce",                                               //op_name
                        Device(CUDA_GPU).TypeConstraint(element::f32).Priority(2), //attrs
                        cuda::SuperScalerAllReduce)                                // constructor
