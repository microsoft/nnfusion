// Microsoft (c) 2019, MSRA/NNFUSION Team
/**
 * \brief Profiler::CPURuntime for creating a Compiler to profile the cpu kernel
 * \author wenxh
 */

#pragma once
#include "binary_utils.hpp"
#include "profiling_runtime.hpp"

namespace nnfusion
{
    namespace profiler
    {
        /*
        class CPUDefaultRuntime : public IProfilingRuntime
        {
        public:
            using Pointer = shared_ptr<CPUDefaultRuntime>;

        public:
            static Pointer Runtime();
            double execute(const ProfilingContext::Pointer& ke,
                           void** input,
                           size_t input_size,
                           void** output,
                           size_t output_size) override;

        private:
            // Tiny codegen function for runtime
            bool codegen(const ProfilingContext::Pointer& ke);
            bool compile(const ProfilingContext::Pointer& ke);
        };
        */

        ///\brief Use this class to have a Interpreter runtime.
        // We treated this runtime as golden truth for evaluating operator's
        // result.
        class ReferenceRuntime : public IProfilingRuntime
        {
        public:
            using Pointer = shared_ptr<ReferenceRuntime>;

        public:
            static Pointer Runtime();
            ReferenceRuntime() { _dt = GENERIC_CPU; }
        private:
            // Tiny codegen function for runtime
            bool codegen(const ProfilingContext::Pointer& ke);
            bool compile(const ProfilingContext::Pointer& ke);
            double
                invoke(const ProfilingContext::Pointer& ke, void** input, void** output) override;
        };
    }
}