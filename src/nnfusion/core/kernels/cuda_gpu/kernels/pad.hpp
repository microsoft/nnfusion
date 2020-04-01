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
            class Pad : public BlockCudaEmitter
            {
            public:
                Pad(shared_ptr<KernelContext> ctx);

                LanguageUnit_p emit_function_body() override;
                LanguageUnit_p emit_dependency() override;
                void set_launch_config() override;

            private:
                shared_ptr<KernelContext> kernel_ctx;
                nnfusion::Shape input_shape, output_shape, padding_above, padding_below,
                    padding_interior;
                uint32_t rank;
                nnfusion::NVShape input_strides, output_strides, pad_below, pad_interior;
                string input_type, output_type;
            };
        } // namespace cuda
    }     // namespace kernels
} // namespace nnfusion