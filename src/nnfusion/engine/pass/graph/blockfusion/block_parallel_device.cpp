// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "block_parallel_device.hpp"

using namespace nnfusion::blockfusion;
using namespace nnfusion::kernels::cuda;

void BlockParallelDevice::create_device(size_t _num_bes, BlockKernelSchedulePolicy _policy)
{
    NNFUSION_CHECK(_num_bes > 0)
        << "BlockParallelDevice::create_device: _num_bes should be larger than 0";

    num_bes = _num_bes;

    block_executors.resize(num_bes);

    block_executor_steps.resize(num_bes, 0);

    block_kernels.clear();

    kernel_name_id_map.clear();

    if (_policy == BlockKernelSchedulePolicy::DEFAULT)
    {
        scheduler = std::make_shared<DefaultBlockKernelScheduler>(num_bes);
    }
    else if (_policy == BlockKernelSchedulePolicy::RANGE)
    {
        scheduler = std::make_shared<RangeBlockKernelScheduler>(num_bes);
    }
    else
    {
        NNFUSION_LOG(NNFUSION_WARNING)
            << "No block_kernel_schedule_policy selected, use default policy";
        scheduler = std::make_shared<DefaultBlockKernelScheduler>(num_bes);
    }
}

void BlockParallelDevice::create_device(size_t _num_bes,
                                        std::shared_ptr<BlockKernelScheduler> _scheduler)
{
    NNFUSION_CHECK(_num_bes > 0)
        << "BlockParallelDevice::create_device: _num_bes should be larger than 0";

    num_bes = _num_bes;

    block_executors.resize(num_bes);

    block_executor_steps.resize(num_bes, 0);

    block_kernels.clear();

    kernel_name_id_map.clear();

    if (_scheduler != nullptr)
    {
        scheduler = _scheduler;
    }
    else
    {
        NNFUSION_LOG(NNFUSION_WARNING)
            << "No block_kernel_schedule_policy selected, use default policy";
        scheduler = std::make_shared<DefaultBlockKernelScheduler>(num_bes);
    }
}

void BlockParallelDevice::append_to_be(int be_id, BEInstruction_p be_instruction)
{
    NNFUSION_CHECK_NOT_NULLPTR(be_instruction)
        << "BlockParallelDevice::append_to_be: be_instruction is nullptr";
    NNFUSION_CHECK(be_id < num_bes)
        << "BlockParallelDevice::append_to_be: be_id should be smaller than num_bes";
    block_executors[be_id].push_back(be_instruction);
}

int BlockParallelDevice::assign_kernel_id(BlockKernel_p block_kernel, KernelMetric_p kernel_metric)
{
    std::string kernel_name = block_kernel->m_context->gnode->get_unique_name();
    NNFUSION_CHECK(kernel_name_id_map.find(kernel_name) == kernel_name_id_map.end())
        << "BlockParallelDevice::assign_kernel_id: " << kernel_name << " has been assigned before";

    block_kernels.push_back(block_kernel);
    int kernel_id = block_kernels.size() - 1;
    // if no metric, assign default metric for scheduling
    if (kernel_metric == nullptr)
    {
        NNFUSION_LOG(NNFUSION_WARNING)
            << "BlockFusion: no kernel_metric provided, use default data";
        kernel_metric = std::make_shared<KernelMetric>();
        kernel_metric->duration = 10;
    }
    block_kernel_metrics.push_back(kernel_metric);
    kernel_name_id_map[kernel_name] = kernel_id;
    return kernel_id;
}

void BlockParallelDevice::schedule_kernel(BlockKernel_p block_kernel, KernelMetric_p kernel_metric)
{
    NNFUSION_CHECK_NOT_NULLPTR(block_kernel)
        << "BlockParallelDevice::schedule_kernel: block_kernel is nullptr";
    int kernel_id = assign_kernel_id(block_kernel, kernel_metric);
    schedule_kernel(kernel_id);
}
void BlockParallelDevice::schedule_kernel(int kernel_id)
{
    NNFUSION_CHECK(kernel_id < block_kernels.size())
        << "BlockParallelDevice::schedule_kernel: kernel_id does not exist in block_kernels";

    BlockKernel_p kernel = block_kernels[kernel_id];
    auto kernel_metric = block_kernel_metrics[kernel_id];
    std::vector<std::pair<int, int>> be_block_schedule_result =
        scheduler->schedule_kernel(kernel_id, kernel->get_grid_dim(), kernel_metric);

    for (auto be_block_pair : be_block_schedule_result)
    {
        BEInstruction_p be_instruction = std::make_shared<BlockExecutorInstructionExecuteBlock>(
            be_block_pair.first, kernel_id, be_block_pair.second);
        append_to_be(be_block_pair.first, be_instruction);
    }
}

void BlockParallelDevice::schedule_kernel_with_dependency(
    BlockKernel_p block_kernel,
    const std::vector<std::string>& dependency_kernels,
    KernelMetric_p kernel_metric)
{
    NNFUSION_CHECK_NOT_NULLPTR(block_kernel)
        << "BlockParallelDevice::schedule_kernel_with_dependency: block_kernel is nullptr";

    std::vector<int> dependency_kernel_ids;
    for (auto kernel_name : dependency_kernels)
    {
        dependency_kernel_ids.push_back(kernel_name_id_map[kernel_name]);
    }

    schedule_kernel_with_dependency(block_kernel, dependency_kernel_ids, kernel_metric);
}
void BlockParallelDevice::schedule_kernel_with_dependency(
    BlockKernel_p block_kernel,
    const std::vector<int>& dependency_kernels,
    KernelMetric_p kernel_metric)
{
    NNFUSION_CHECK_NOT_NULLPTR(block_kernel)
        << "BlockParallelDevice::schedule_kernel_with_dependency: block_kernel is nullptr";
    int kernel_id = assign_kernel_id(block_kernel, kernel_metric);
    schedule_kernel_with_dependency(kernel_id, dependency_kernels);
}
void BlockParallelDevice::schedule_kernel_with_dependency(
    int kernel_id, const std::vector<int>& dependency_kernels)
{
    NNFUSION_CHECK(kernel_id < block_kernels.size()) << "BlockParallelDevice::schedule_kernel_with_"
                                                        "dependency: kernel_id does not exist in "
                                                        "block_kernels";

    // group wait for dependency_kernels
    std::vector<int> bes_to_sync;
    for (auto dependency_kernel_id : dependency_kernels)
    {
        std::vector<int> tmp = scheduler->schedule_kernel_sync(dependency_kernel_id);
        bes_to_sync.insert(bes_to_sync.end(), tmp.begin(), tmp.end());
    }
    std::sort(bes_to_sync.begin(), bes_to_sync.end());
    bes_to_sync.erase(std::unique(bes_to_sync.begin(), bes_to_sync.end()), bes_to_sync.end());

    int time_start;
    std::vector<int> bes_ready =
        scheduler->get_ready_bes(block_kernels[kernel_id]->get_grid_dim(), time_start);

    group_wait(bes_ready, bes_to_sync);

    // schedule kernel
    BlockKernel_p kernel = block_kernels[kernel_id];
    auto kernel_metric = block_kernel_metrics[kernel_id];
    std::vector<std::pair<int, int>> be_block_schedule_result = scheduler->schedule_kernel(
        kernel_id, kernel->get_grid_dim(), kernel_metric, bes_ready, time_start);

    for (auto be_block_pair : be_block_schedule_result)
    {
        BEInstruction_p be_instruction = std::make_shared<BlockExecutorInstructionExecuteBlock>(
            be_block_pair.first, kernel_id, be_block_pair.second);
        append_to_be(be_block_pair.first, be_instruction);
    }
}

int BlockParallelDevice::group_sync(int kernel_id)
{
    std::vector<int> bes_to_sync = scheduler->schedule_kernel_sync(kernel_id);
    return group_sync(bes_to_sync);
}
int BlockParallelDevice::group_sync(const std::vector<int>& bes_to_sync)
{
    // calculate step_id
    int step = 0;
    for (auto be_id : bes_to_sync)
    {
        step = max(step, block_executor_steps[be_id]);
    }
    step += 1;

    // update block_executor_steps
    for (auto be_id : bes_to_sync)
    {
        block_executor_steps[be_id] = step;
    }

    // append instructions to block_executors
    for (auto be_id : bes_to_sync)
    {
        BEInstruction_p be_instruction =
            std::make_shared<BlockExecutorInstructionStepTo>(be_id, step);
        append_to_be(be_id, be_instruction);
    }

    return step;
}

void BlockParallelDevice::group_wait(const std::vector<int>& bes_wait,
                                     const std::vector<int>& bes_predecessor)
{ // when user does not indicate step_to_wait, group_wait will generate step_to_wait by call group_sync(bes_predecessor) and then group_wait
    int step = group_sync(bes_predecessor);
    group_wait(bes_wait, bes_predecessor, step);
}
void BlockParallelDevice::group_wait(const std::vector<int>& bes_wait,
                                     const std::vector<int>& bes_predecessor,
                                     const int step_to_wait)
{
    scheduler->schedule_group_wait(bes_wait, bes_predecessor);

    for (auto be_id : bes_wait)
    {
        BEInstruction_p be_instruction =
            std::make_shared<BlockExecutorInstructionWaitFor>(be_id, bes_predecessor, step_to_wait);
        append_to_be(be_id, be_instruction);
    }
}

bool BlockParallelDevice::check_dependency(const std::string kernel_name)
{
    return kernel_name_id_map.find(kernel_name) != kernel_name_id_map.end();
}

// return the generated be_program of this block-parallel-device
BlockExecutorProgram BlockParallelDevice::get_block_executor_program()
{
    BlockExecutorProgram be_program;

    be_program.num_bes = this->num_bes;
    be_program.block_executor_instructions = this->block_executors;
    be_program.block_kernels = this->block_kernels;

    return be_program;
}

blockfusion::ProfilingResult BlockParallelDevice::get_profiling_result()
{
    blockfusion::ProfilingResult profiling_result;

    profiling_result.num_bes = this->num_bes;

    profiling_result.num_kernels = this->block_kernels.size();

    profiling_result.num_large_kernels = 0;
    for (auto kernel : block_kernels)
    {
        auto dims = kernel->get_grid_dim();
        if (dims.x * dims.y * dims.z >= num_bes)
        {
            profiling_result.num_large_kernels++;
        }
    }

    profiling_result.normal_execution_time = 0;
    for (auto kernel_metric : block_kernel_metrics)
    {
        profiling_result.normal_execution_time += kernel_metric->duration;
    }

    profiling_result.fused_estimation_time = this->scheduler->get_estimation_time();

    return profiling_result;
}

std::string BlockParallelDevice::DebugStringBE()
{
    std::ostringstream ret;
    ret << "Parallel device for this group\n";

    auto PrintInfo = [this, &ret](const BEInstruction_p instruction) {
        if (auto ins_execute_block =
                std::dynamic_pointer_cast<BlockExecutorInstructionExecuteBlock>(instruction))
        {
            auto kernel_id = ins_execute_block->kernel_id;
            auto node = block_kernels[kernel_id]->m_context->gnode;
            ret << node->get_id() << ":" << node->get_name() << "\t" << node->get_op_type() << "\n";
        }
        else if (auto ins_step_to =
                     std::dynamic_pointer_cast<BlockExecutorInstructionStepTo>(instruction))
        {
            ret << "sync:step_to"
                << "\t" << std::to_string(ins_step_to->step_id) << "\n";
        }
        else if (auto ins_wait_for =
                     std::dynamic_pointer_cast<BlockExecutorInstructionWaitFor>(instruction))
        {
            auto bes_predecessor = ins_wait_for->bes_predecessor;
            auto step_id = ins_wait_for->step_id;
            ret << "sync:wait_for (predecessor, step_id): "
                << "\t";
            for (int be_id : bes_predecessor)
            {
                ret << "(" << std::to_string(be_id) << ", " << std::to_string(step_id) << ") ";
            }
            ret << "\n";
        }
        else
        {
            NNFUSION_CHECK_FAIL();
        }
    };

    for (size_t n = 0; n < num_bes; n++)
    {
        ret << "Block executor " << n << ": [\n";
        for (auto instruction : block_executors[n])
        {
            PrintInfo(instruction);
        }
        ret << "]\n";
    }
    return ret.str();
}