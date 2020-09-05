// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include "common.hpp"

using namespace nnfusion::blockfusion;
using namespace nnfusion::kernels::cuda;

class BlockKernelScheduleRecord
{
public:
    using Pointer = shared_ptr<BlockKernelScheduleRecord>;
    BlockKernelScheduleRecord()
    {
        // NNFUSION_LOG(NNFUSION_WARNING)
        //     << "BlockKernelScheduleRecord::BlockKernelScheduleRecord() must have "
        //        "parameters. This warning may happen in the initialization of std::map, "
        //        "which can be ignored.";
    }
    BlockKernelScheduleRecord(int _kernel_id, const std::vector<std::pair<int, int>>& _be_block_map)
    {
        kernel_id = _kernel_id;
        be_block_map = _be_block_map;
    }

public:
    int kernel_id;
    std::vector<std::pair<int, int>>
        be_block_map; // pair.first: be_id; pair.second: kernel_block_id
};

class BlockKernelScheduler
{
public:
    using Pointer = shared_ptr<BlockKernelScheduler>;
    BlockKernelScheduler(int _num_bes)
    {
        num_bes = _num_bes;

        schedule_kernel_log.clear();
    }

    // time_start is a return value
    virtual std::vector<int> get_ready_bes(dim3 m_gridDim, int& time_start) = 0;
    virtual std::vector<int> get_ready_bes(int num_required_bes, int& time_start) = 0;
    virtual std::vector<int>
        get_ready_bes_with_hint(dim3 m_gridDim, int& time_start, std::vector<int>& hint_bes)
    {
        // ignore hint
        return get_ready_bes(m_gridDim, time_start);
    }
    virtual std::vector<int>
        get_ready_bes_with_hint(int num_required_bes, int& time_start, std::vector<int>& hint_bes)
    {
        // ignore hint
        return get_ready_bes(num_required_bes, time_start);
    }

    virtual std::vector<std::pair<int, int>>
        schedule_kernel(int kernel_id, dim3 m_gridDim, KernelMetric_p kernel_metric) = 0;
    virtual std::vector<std::pair<int, int>> schedule_kernel(int kernel_id,
                                                             dim3 m_gridDim,
                                                             KernelMetric_p kernel_metric,
                                                             const std::vector<int>& bes_ready,
                                                             int time_start) = 0;

    virtual std::vector<int> schedule_kernel_sync(int kernel_id) = 0;

    virtual void schedule_group_wait(const std::vector<int>& bes_wait,
                                     const std::vector<int>& bes_predecessor) = 0;

    virtual double get_estimation_time() = 0;

protected:
    int num_bes;
    std::map<int, BlockKernelScheduleRecord> schedule_kernel_log;
};

enum BlockKernelSchedulePolicy
{
    DEFAULT,
    RANGE
};

// class DefaultBlockKernelScheduler; // forward declaration

class DefaultBlockKernelScheduler : public BlockKernelScheduler
{
public:
    DefaultBlockKernelScheduler(int _num_bes)
        : BlockKernelScheduler(_num_bes)
    {
        be_lane.resize(num_bes);
    }

    std::vector<int> get_ready_bes(dim3 m_gridDim, int& time_start) override;
    std::vector<int> get_ready_bes(int num_required_bes, int& time_start) override;

    std::vector<std::pair<int, int>>
        schedule_kernel(int kernel_id, dim3 m_gridDim, KernelMetric_p kernel_metric) override;
    std::vector<std::pair<int, int>> schedule_kernel(int kernel_id,
                                                     dim3 m_gridDim,
                                                     KernelMetric_p kernel_metric,
                                                     const std::vector<int>& bes_ready,
                                                     int time_start) override;

    std::vector<int> schedule_kernel_sync(int kernel_id) override;
    void schedule_group_wait(const std::vector<int>& bes_wait,
                             const std::vector<int>& bes_predecessor) override;

    double get_estimation_time() override;

    std::vector<std::vector<int>> get_be_lane() { return be_lane; }
private:
    std::vector<std::vector<int>> be_lane; // -1 indicates sync, kernel_id indicates kernel running
};

class RangeBlockKernelScheduler : public BlockKernelScheduler
{
public:
    RangeBlockKernelScheduler(int _num_bes)
        : BlockKernelScheduler(_num_bes)
    {
        be_lane.resize(num_bes);
    }

    std::vector<int> get_ready_bes(dim3 m_gridDim, int& time_start) override;
    std::vector<int> get_ready_bes(int num_required_bes, int& time_start) override;
    std::vector<int> get_ready_bes_with_hint(dim3 m_gridDim,
                                             int& time_start,
                                             std::vector<int>& hint_bes) override;
    std::vector<int> get_ready_bes_with_hint(int num_required_bes,
                                             int& time_start,
                                             std::vector<int>& hint_bes) override;

    std::vector<std::pair<int, int>>
        schedule_kernel(int kernel_id, dim3 m_gridDim, KernelMetric_p kernel_metric) override;
    std::vector<std::pair<int, int>> schedule_kernel(int kernel_id,
                                                     dim3 m_gridDim,
                                                     KernelMetric_p kernel_metric,
                                                     const std::vector<int>& bes_ready,
                                                     int time_start) override;

    std::vector<int> schedule_kernel_sync(int kernel_id) override;
    void schedule_group_wait(const std::vector<int>& bes_wait,
                             const std::vector<int>& bes_predecessor) override;

    double get_estimation_time() override;

    std::vector<std::vector<int>> get_be_lane() { return be_lane; }
private:
    std::vector<std::vector<int>> be_lane; // -1 indicates sync, kernel_id indicates kernel running
};
