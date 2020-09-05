// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "kernel_registration.hpp"

int main()
{
    using namespace nnfusion;

    NNFUSION_LOG(INFO) << "Global registered kernel size: "
                       << kernels::KernelRegistry::Global()->RegisteredKernelSize();
    auto kernel_reg =
        kernels::KernelRegistry::Global()->FindKernelRegistration("Pad", CUDA_GPU, DT_FLOAT);
    if (kernel_reg)
    {
        NNFUSION_LOG(INFO) << "Find registered kernel for < Pad, CUDA_GPU, DT_FLOAT> ";
        kernel_reg->debug_string();
    }
    else
    {
        NNFUSION_LOG(INFO) << "No registered kernel found for < Pad, CUDA_GPU, DT_FLOAT> ";
    }

    kernel_reg =
        kernels::KernelRegistry::Global()->FindKernelRegistration("Pad", ROCM_GPU, DT_FLOAT);
    if (kernel_reg)
    {
        NNFUSION_LOG(INFO) << "Find registered kernel for < Pad, ROCM_GPU, DT_FLOAT> ";
    }
    else
    {
        NNFUSION_LOG(INFO) << "No registered kernel found for < Pad, ROCM_GPU, DT_FLOAT> ";
    }

    kernel_reg =
        kernels::KernelRegistry::Global()->FindKernelRegistration("Pad", CUDA_GPU, DT_INT32);
    if (kernel_reg)
    {
        NNFUSION_LOG(INFO) << "Find registered kernel for < Pad, CUDA_GPU, DT_INT32> ";
        kernel_reg->debug_string();
    }
    else
    {
        NNFUSION_LOG(INFO) << "No registered kernel found for < Pad, CUDA_GPU, DT_INT32> ";
    }

    return 0;
}