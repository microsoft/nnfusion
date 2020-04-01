// Microsoft (c) 2019, NNFusion Team
#pragma once
#include "../cpu_kernel_emitter.hpp"

namespace nnfusion
{
    namespace kernels
    {
        namespace cpu
        {
            class ConvolutionMlas : public MlasKernelEmitter
            {
            public:
                ConvolutionMlas(shared_ptr<KernelContext> ctx);

                LanguageUnit_p emit_function_body() override;
                LanguageUnit_p emit_dependency() override;

            private:
                nnfusion::Shape input_shape, filter_shape, output_shape, padding;
                nnfusion::Strides window_dilation_strides, window_movement_strides,
                    data_dilation_strides;
                nnfusion::CoordinateDiff padding_below_diff, padding_above_diff;
                string dtype;
            };
        } // namespace cpu
    }     // namespace kernels
} // namespace nnfusion
