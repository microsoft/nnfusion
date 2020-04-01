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
            class Range : public BlockCudaEmitter
            {
            public:
                Range(shared_ptr<KernelContext> ctx);

                LanguageUnit_p emit_function_body() override;
                LanguageUnit_p emit_dependency() override;
                void set_launch_config() override;

            private:
                nnfusion::Shape output_shape;
                int start, limit, delta;
                int range_num;
            };
        } // namespace cuda
    }     // namespace kernels
} // namespace nnfusion