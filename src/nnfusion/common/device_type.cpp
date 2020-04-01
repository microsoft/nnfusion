// Microsoft (c) 2019, NNfusion Team
#include "device_type.hpp"

std::string nnfusion::get_device_str(DeviceType dt)
{
    switch (dt)
    {
    case CUDA_GPU: return "CUDA_GPU";
    case ROCM_GPU: return "ROCM_GPU";
    case GENERIC_CPU: return "GENERIC_CPU";
    default: return "UNKNOWN";
    }
}

nnfusion::DeviceType nnfusion::get_device_type(std::string dt)
{
    if (dt == "ROCM_GPU" || dt == "ROCm")
        return ROCM_GPU;
    if (dt == "CUDA_GPU" || dt == "CUDA")
        return CUDA_GPU;
    if (dt == "GENERIC_CPU" || dt == "CPU")
        return GENERIC_CPU;
    return UNKNOWN;
}