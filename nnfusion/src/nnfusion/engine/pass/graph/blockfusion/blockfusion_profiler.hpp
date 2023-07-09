// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include "block_parallel_device.hpp"
#include "blockfusion_codegen.hpp"
#include "common.hpp"

using namespace nnfusion;
using namespace nnfusion::blockfusion;

class BlockFusionProfiler
{
public:
    BlockFusionProfiler(BlockParallelDevice::Pointer _block_parallel_device = nullptr,
                        BlockFusionCudaCodegen::Pointer _blockfusion_codegen = nullptr)
    {
        set_profiling_context(_block_parallel_device, _blockfusion_codegen);
    }

    void set_profiling_context(BlockParallelDevice::Pointer _block_parallel_device = nullptr,
                               BlockFusionCudaCodegen::Pointer _blockfusion_codegen = nullptr);

    blockfusion::ProfilingResult get_profiling_result();

private:
    bool get_codegen_profiling_result(BlockFusionCudaCodegen::Pointer codegen_context,
                                      blockfusion::ProfilingResult& codegen_profiling);

private:
    BlockParallelDevice::Pointer block_parallel_device;
    BlockFusionCudaCodegen::Pointer blockfusion_codegen;
};