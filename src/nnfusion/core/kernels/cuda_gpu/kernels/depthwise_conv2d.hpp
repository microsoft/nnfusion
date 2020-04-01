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
            struct DepthwiseArgs
            {
                // Input layer dimensions
                int batch;
                int in_rows;
                int in_cols;
                int in_depth;
                int filter_rows;
                int filter_cols;
                int depth_multiplier;
                int stride;
                int pad_rows;
                int pad_cols;

                // Output layer dimensions
                int out_rows;
                int out_cols;
                int out_depth;

                int64_t num_outputs;
            };

            class DepthwiseConv2dNative : public BlockCudaEmitter
            {
            public:
                DepthwiseConv2dNative(shared_ptr<KernelContext> ctx);

                LanguageUnit_p emit_function_body() override;
                LanguageUnit_p emit_dependency() override;
                void set_launch_config() override;

            private:
                shared_ptr<KernelContext> kernel_ctx;
                std::string data_format;
                DepthwiseArgs args;

                LanguageUnit_p emit_DepthwiseConv2dGPUKernelNHWC();
                LanguageUnit_p emit_DepthwiseConv2dGPUKernelNCHW();
            };
        } // namespace cuda
    }     // namespace kernels
} // namespace nnfusion