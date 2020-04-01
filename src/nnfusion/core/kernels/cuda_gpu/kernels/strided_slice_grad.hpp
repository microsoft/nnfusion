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
            class StridedSliceGrad : public BlockCudaEmitter
            {
            public:
                StridedSliceGrad(shared_ptr<KernelContext> ctx);

                LanguageUnit_p emit_function_body() override;
                LanguageUnit_p emit_dependency() override;
                void set_launch_config() override;

            private:
                nnfusion::Shape x_shape, begin_shape, end_shape, strides_shape, grad_shape,
                    output_shape;
                int x_size;
            };
        } // namespace cuda
    }     // namespace kernels
} // namespace nnfusion