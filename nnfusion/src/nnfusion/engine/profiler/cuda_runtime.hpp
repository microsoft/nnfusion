// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

/**
 * \brief Profiler::CudaRuntime for creating a Compiler to profile the kernel
 * \author wenxh
 */

#pragma once
#include "binary_utils.hpp"
#include "profiling_runtime.hpp"

namespace nnfusion
{
    namespace profiler
    {
        class CudaDefaultRuntime : public IProfilingRuntime
        {
        public:
            using Pointer = shared_ptr<CudaDefaultRuntime>;
            static Pointer Runtime();
            CudaDefaultRuntime() { _dt = CUDA_GPU; }
        protected:
            // Tiny codegen function for runtime
            virtual bool codegen(const ProfilingContext::Pointer& ke);
            virtual bool compile(const ProfilingContext::Pointer& ke);
            double
                invoke(const ProfilingContext::Pointer& ke, void** input, void** output) override;

            void set_dt(NNFusion_DeviceType dt) { _dt = dt; }
        };

        class CUPTIRuntime : public IProfilingRuntime
        {
        public:
            using Pointer = shared_ptr<CUPTIRuntime>;
            static Pointer Runtime();
            CUPTIRuntime() { _dt = CUDA_GPU; }
        protected:
            // Tiny codegen function for runtime
            virtual bool codegen(const ProfilingContext::Pointer& ke);
            virtual bool compile(const ProfilingContext::Pointer& ke);
            double
                invoke(const ProfilingContext::Pointer& ke, void** input, void** output) override;

            void set_dt(NNFusion_DeviceType dt) { _dt = dt; }
        };
    }
}