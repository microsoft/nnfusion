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
            class MatMulAdd : public CudaLibEmitter
            {
            public:
                MatMulAdd(shared_ptr<KernelContext> ctx);

                LanguageUnit_p emit_function_body() override;
                LanguageUnit_p emit_dependency() override;
                LanguageUnit_p emit_function_signature() override;
                bool require_cublas_handle() override { return true; }
            private:
                LanguageUnit_p get_add_kernel();
                shared_ptr<op::GenericOp> generic_op;
                shared_ptr<KernelContext> kernel_ctx;
                size_t reduction_axes;
                nnfusion::Shape A_shape, B_shape, C_shape;
                nnfusion::Shape out_shape;
                nnfusion::element::Type dtype;
                std::shared_ptr<nnfusion::descriptor::Tensor> tmp_tensor;
                dim3 m_blockDim;
                dim3 m_gridDim;
            };
        } // namespace cuda
    }     // namespace kernels
} // namespace nnfusion