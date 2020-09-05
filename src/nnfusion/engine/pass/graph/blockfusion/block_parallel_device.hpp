// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include "block_kernel_schedule.hpp"
#include "common.hpp"

using namespace nnfusion;
using namespace nnfusion::blockfusion;
using namespace nnfusion::kernels::cuda;

// step algorithm: bes in wait_for and step_to should share the same step_id, so new_step = max(steps_in_these_bes) + 1.
class BlockParallelDevice
{
public:
    using Pointer = shared_ptr<BlockParallelDevice>;
    BlockParallelDevice(size_t _num_bes,
                        BlockKernelSchedulePolicy _policy = BlockKernelSchedulePolicy::DEFAULT)
    {
        create_device(_num_bes, _policy);
    }
    BlockParallelDevice(size_t _num_bes, std::shared_ptr<BlockKernelScheduler> _scheduler)
    {
        create_device(_num_bes, _scheduler);
    }

    void create_device(size_t _num_bes,
                       BlockKernelSchedulePolicy _policy = BlockKernelSchedulePolicy::DEFAULT);
    void create_device(size_t _num_bes, std::shared_ptr<BlockKernelScheduler> _scheduler);

    void schedule_kernel(BlockKernel_p block_kernel, KernelMetric_p kernel_metric = nullptr);

    void schedule_kernel_with_dependency(BlockKernel_p block_kernel,
                                         const std::vector<std::string>& dependency_kernels,
                                         KernelMetric_p kernel_metric = nullptr);

    // return the generated be_program of this block-parallel-device
    BlockExecutorProgram get_block_executor_program();

    // return the profiling-result of this block-parallel-device
    blockfusion::ProfilingResult get_profiling_result();

    // return the scheme of kernels on BEs
    std::string DebugStringBE();

    bool check_dependency(const std::string kernel_name);

private:
    void append_to_be(int be_id, BEInstruction_p be_instruction);

    int assign_kernel_id(BlockKernel_p block_kernel, KernelMetric_p kernel_metric = nullptr);

    void schedule_kernel(int kernel_id);

    void schedule_kernel_with_dependency(BlockKernel_p block_kernel,
                                         const std::vector<int>& dependency_kernels,
                                         KernelMetric_p kernel_metric = nullptr);
    void schedule_kernel_with_dependency(int kernel_id, const std::vector<int>& dependency_kernels);

    int group_sync(int kernel_id);
    int group_sync(const std::vector<int>& bes_to_sync);

    // when user does not indicate step_to_wait, group_wait will generate step_to_wait by call group_sync(bes_predecessor) and then group_wait
    void group_wait(const std::vector<int>& bes_wait, const std::vector<int>& bes_predecessor);
    void group_wait(const std::vector<int>& bes_wait,
                    const std::vector<int>& bes_predecessor,
                    const int step_to_wait);

private:
    size_t num_bes;
    std::map<std::string, int> kernel_name_id_map;
    std::vector<std::vector<BEInstruction_p>> block_executors;
    std::vector<int> block_executor_steps;
    std::vector<BlockKernel_p> block_kernels;         // (key, value): (kernel_id, BlockCudaEmitter)
    std::vector<KernelMetric_p> block_kernel_metrics; // (key, value): (kernel_id, kernel_metric)
    std::shared_ptr<BlockKernelScheduler> scheduler;
};