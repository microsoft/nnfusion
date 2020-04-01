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
            class ElementWiseFused : public BlockCudaEmitter
            {
            public:
                ElementWiseFused(shared_ptr<KernelContext> ctx);

                LanguageUnit_p emit_function_body() override;
                LanguageUnit_p emit_dependency() override;
                LanguageUnit_p emit_function_name() override;
                LanguageUnit_p emit_comments() override;
                static int unique_func_id;

            protected:
                void set_launch_config() override;

            private:
                std::shared_ptr<KernelContext> FuseContext();
                void compute_best_config(int& grids, int& blocks, int& bound);
                std::vector<shared_ptr<CudaElementwiseEmitter>> m_kernels;
            };

        } // namespace cuda
    }     // namespace kernels
} // namespace nnfusion