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
            class ConvolutionCudnn : public CudaLibEmitter
            {
            public:
                ConvolutionCudnn(shared_ptr<KernelContext> ctx);

                LanguageUnit_p emit_function_body() override;
                LanguageUnit_p emit_dependency() override;
                LanguageUnit_p emit_function_signature() override;
                bool require_cudnn_handle() override { return true; }
            private:
                nnfusion::Shape input_shape, filter_shape, output_shape;
                element::Type input_type, filter_type, output_type, conv_type;
                nnfusion::Strides window_dilation_strides, window_movement_strides,
                    data_dilation_strides;
                nnfusion::CoordinateDiff padding_below_diff, padding_above_diff;
                string dtype, data_format;
            };
        } // namespace cuda
    }     // namespace kernels
} // namespace nnfusion