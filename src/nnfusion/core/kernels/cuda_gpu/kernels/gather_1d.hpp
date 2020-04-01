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
            class Gather1D : public BlockCudaEmitter
            {
            public:
                Gather1D(shared_ptr<KernelContext> ctx);

                LanguageUnit_p emit_function_body() override;
                LanguageUnit_p emit_dependency() override;
                void set_launch_config() override;

            private:
                nnfusion::Shape input_shape_0, input_shape_1, output_shape;
                int axis;
                bool is_axis_zero;
                int64_t gather_dim_size;
                int64_t indices_size;
                int64_t slice_size;
            };
        } // namespace cuda
    }     // namespace kernels
} // namespace nnfusion