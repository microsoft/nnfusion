// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

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
        class CPUDefaultRuntime : public IProfilingRuntime
        {
        public:
            using Pointer = shared_ptr<CPUDefaultRuntime>;

        public:
            static Pointer Runtime();
            CPUDefaultRuntime() { _dt = GENERIC_CPU; }
            bool codegen(const ProfilingContext::Pointer& ke);
            bool general_compile();
            double sep_invoke(const ProfilingContext::Pointer& ke, void** input, void** output);

        private:
            // Tiny codegen function for runtime
            // bool codegen(const ProfilingContext::Pointer& ke);
            bool cmake_codegen(const ProfilingContext::Pointer& ke);
            bool compile(const ProfilingContext::Pointer& ke);
            bool general_cmake_codegen();
            double
                invoke(const ProfilingContext::Pointer& ke, void** input, void** output) override;
            unordered_set<string> global_required;
        };

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