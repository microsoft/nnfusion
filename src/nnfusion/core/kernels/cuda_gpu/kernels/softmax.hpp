// Microsoft (c) 2019, NNFusion Team
#pragma once
#include "../cuda_emitter.hpp"
#include "../cuda_langunit.hpp"

namespace nnfusion
{
    namespace kernels
    {
        namespace cuda
        {
            class Softmax : public CudaLibEmitter
            {
            public:
                Softmax(shared_ptr<KernelContext> ctx);

                LanguageUnit_p emit_function_body() override;
                LanguageUnit_p emit_dependency() override;
                LanguageUnit_p
                    cudnn_tensor_descriptor_from_shape_for_softmax(const nnfusion::Shape& shape,
                                                                   string desc);

            private:
                shared_ptr<KernelContext> kernel_ctx;
                nnfusion::Shape input_shape, output_shape;
            };
        } // namespace cuda
    }     // namespace kernels
} // namespace nnfusion
