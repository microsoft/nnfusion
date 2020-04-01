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
            class Slice : public BlockCudaEmitter
            {
            public:
                Slice(shared_ptr<KernelContext> ctx);

                LanguageUnit_p emit_function_body() override;
                LanguageUnit_p emit_dependency() override;
                void set_launch_config() override;

            private:
                shared_ptr<KernelContext> kernel_ctx;

                nnfusion::Shape input_shape, output_shape, lower_bounds;
                nnfusion::Shape input_strides, output_strides, slice_strides;
                string input_type, output_type;
            };
        } // namespace cuda
    }     // namespace kernels
} // namespace nnfusion