// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "device_type.hpp"

std::string nnfusion::get_device_str(NNFusion_DeviceType dt)
{
    switch (dt)
    {
    case CUDA_GPU: return "CUDA_GPU";
    case ROCM_GPU: return "ROCM_GPU";
    case GENERIC_CPU: return "GENERIC_CPU";
    case HLSL: return "HLSL";
    case GraphCore: return "GRAPHCORE";
    default: return "UNKNOWN";
    }
}

nnfusion::NNFusion_DeviceType nnfusion::get_device_type(std::string dt)
{
    if (dt == "ROCM_GPU" || dt == "ROCm")
        return ROCM_GPU;
    if (dt == "CUDA_GPU" || dt == "CUDA")
        return CUDA_GPU;
    if (dt == "GENERIC_CPU" || dt == "CPU")
        return GENERIC_CPU;
    if (dt == "HLSL" || dt == "hlsl" || dt == "dxcompute")
        return HLSL;
    if (dt == "gc" || dt == "GC" || dt == "GraphCore" || dt == "graphcore" || dt == "GRAPHCORE")
        return GraphCore;
    return UNKNOWN;
}