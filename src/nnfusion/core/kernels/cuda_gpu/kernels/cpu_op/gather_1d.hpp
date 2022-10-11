// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once
#include "../../cuda_common_ops.hpp"
#include "../../cpu_op_emitter.hpp"
#include "../../cuda_langunit.hpp"

namespace nnfusion
{
    namespace kernels
    {
        namespace cuda_cpu
        {
            class Gather1D : public CPUOpEmitter
            {
            public:
                Gather1D(shared_ptr<KernelContext> ctx);

                LanguageUnit_p emit_function_body() override;
                LanguageUnit_p emit_dependency() override;

            private:
                nnfusion::Shape input_shape_0, input_shape_1, output_shape;
                int axis;
                bool is_axis_zero;
                int64_t gather_dim_size;
                int64_t indices_size;
                int64_t slice_size;
            };
        } // namespace cuda_cpu
    }     // namespace kernels
} // namespace nnfusion