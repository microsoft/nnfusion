#pragma once
#include "cuda_emitter.hpp"
#include "cuda_langunit.hpp"
#include "nnfusion/engine/interpreter.hpp"

namespace nnfusion
{
    namespace kernels
    {
        namespace cuda
        {
            class CPUOpEmitter : public KernelEmitter
            {
            public:
                CPUOpEmitter(std::shared_ptr<KernelContext> ctx);
                LanguageUnit_p emit_function_call() override;
                LanguageUnit_p emit_function_body() override;
                LanguageUnit_p emit_dependency() override;
            };
        }
    }
}
