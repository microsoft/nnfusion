// Microsoft (c) 2019, NNFusion Team
#pragma once
#include "../cpu_kernel_emitter.hpp"
#include "../cpu_langunit.hpp"

namespace nnfusion
{
    namespace kernels
    {
        namespace cpu
        {
            class AnyOP : public CpuKernelEmitter
            {
            public:
                AnyOP(shared_ptr<KernelContext> ctx);

                LanguageUnit_p emit_function_body() override;
                LanguageUnit_p emit_dependency() override;

            private:
                shared_ptr<KernelContext> kernel_ctx;
                string input_type, output_type;
            };
        } // namespace cpu
    }     // namespace kernels
} // namespace nnfusion