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
            class DynamicStitch : public BlockCudaEmitter
            {
            public:
                DynamicStitch(shared_ptr<KernelContext> ctx);

                LanguageUnit_p emit_function_body() override;
                LanguageUnit_p emit_dependency() override;
                void set_launch_config() override;

            private:
                int num_partitions;
                size_t output_size, slice_size;
                std::vector<int> indices_flat;
                std::vector<string> data_flat;
            };
        } // namespace cuda
    }     // namespace kernels
} // namespace nnfusion