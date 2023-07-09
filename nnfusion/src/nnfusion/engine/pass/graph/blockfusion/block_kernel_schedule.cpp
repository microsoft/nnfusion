// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "block_kernel_schedule.hpp"

using namespace nnfusion::blockfusion;
using namespace nnfusion::kernels::cuda;

std::vector<int> DefaultBlockKernelScheduler::get_ready_bes(dim3 m_gridDim, int& time_start)
{
    int num_required_bes = m_gridDim.x * m_gridDim.y * m_gridDim.z;
    num_required_bes = std::min(num_required_bes, num_bes); // to support large kernels

    return get_ready_bes(num_required_bes, time_start);
}
std::vector<int> DefaultBlockKernelScheduler::get_ready_bes(int num_required_bes, int& time_start)
{
    // minimum time_start
    time_start = be_lane[0].size();
    for (auto be : be_lane)
    {
        time_start = std::min(time_start, (int)be.size());
    }

    // find time_start
    while (true)
    {
        int cnt_available_bes = 0;
        for (auto be : be_lane)
        {
            if (be.size() <= time_start)
            {
                cnt_available_bes += 1;
            }
        }
        if (cnt_available_bes >= num_required_bes)
            break;

        time_start++;
    }

    // select bes_ready
    int num_select_bes = 0;
    std::vector<int> bes_ready;
    for (int be_id = 0; be_id < num_bes && num_select_bes < num_required_bes; be_id++)
    {
        if (be_lane[be_id].size() <= time_start)
        {
            bes_ready.push_back(be_id);
            num_select_bes++;
        }
    }

    return bes_ready;
}

std::vector<std::pair<int, int>> DefaultBlockKernelScheduler::schedule_kernel(
    int kernel_id, dim3 m_gridDim, KernelMetric_p kernel_metric)
{
    int num_required_bes = m_gridDim.x * m_gridDim.y * m_gridDim.z;
    num_required_bes = std::min(num_required_bes, num_bes); // to support large kernels

    int time_start;
    std::vector<int> bes_ready = get_ready_bes(num_required_bes, time_start);

    return schedule_kernel(kernel_id, m_gridDim, kernel_metric, bes_ready, time_start);
}
std::vector<std::pair<int, int>>
    DefaultBlockKernelScheduler::schedule_kernel(int kernel_id,
                                                 dim3 m_gridDim,
                                                 KernelMetric_p kernel_metric,
                                                 const std::vector<int>& bes_ready,
                                                 int time_start)
{
    int num_kernel_blocks = m_gridDim.x * m_gridDim.y * m_gridDim.z;
    int num_required_bes = std::min(num_kernel_blocks, num_bes); // to support large kernels

    std::vector<std::pair<int, int>> schedule_result;
    for (int kernel_block_id = 0; kernel_block_id < num_kernel_blocks; kernel_block_id++)
    {
        int be_id = bes_ready[kernel_block_id % bes_ready.size()];
        schedule_result.push_back(make_pair(be_id, kernel_block_id));
    }

    int duration = std::max((int)kernel_metric->duration, 1);
    for (auto schedule_pair : schedule_result)
    {
        // sync to time_start
        for (int dur_sync = (int)be_lane[schedule_pair.first].size(); dur_sync < time_start;
             dur_sync++)
        {
            be_lane[schedule_pair.first].push_back(-1);
        }

        // schedule kernel_id
        for (int dur = 0; dur < duration; dur++)
        {
            be_lane[schedule_pair.first].push_back(kernel_id);
        }
    }

    NNFUSION_CHECK(schedule_kernel_log.find(kernel_id) == schedule_kernel_log.end())
        << "DefaultBlockKernelScheduler::schedule_kernel: this kernel has been scheduled";
    schedule_kernel_log[kernel_id] = BlockKernelScheduleRecord(kernel_id, schedule_result);

    return schedule_result;
}

std::vector<int> DefaultBlockKernelScheduler::schedule_kernel_sync(int kernel_id)
{
    NNFUSION_CHECK(schedule_kernel_log.find(kernel_id) != schedule_kernel_log.end())
        << "DefaultBlockKernelScheduler::schedule_kernel_sync: no such kernel in "
           "schedule_kernel_log";
    std::vector<std::pair<int, int>> schedule_result = schedule_kernel_log[kernel_id].be_block_map;

    std::vector<int> bes_to_sync;
    for (auto schedule_pair : schedule_result)
    {
        bes_to_sync.push_back(schedule_pair.first);
    }

    // sort and unique bes_to_sync
    std::sort(bes_to_sync.begin(), bes_to_sync.end());
    bes_to_sync.erase(std::unique(bes_to_sync.begin(), bes_to_sync.end()), bes_to_sync.end());

    return bes_to_sync;
}
void DefaultBlockKernelScheduler::schedule_group_wait(const std::vector<int>& bes_wait,
                                                      const std::vector<int>& bes_predecessor)
{
    // calculate time_sync
    int time_sync = 0;
    for (auto be_id : bes_predecessor)
    {
        time_sync = std::max(time_sync, (int)be_lane[be_id].size());
    }
    time_sync += 1;

    // sync bes_predecessor
    for (auto be_id : bes_predecessor)
    {
        for (int dur_sync = (int)be_lane[be_id].size(); dur_sync < time_sync; dur_sync++)
        {
            be_lane[be_id].push_back(-1);
        }
    }

    // sync bes_wait
    for (auto be_id : bes_wait)
    {
        for (int dur_sync = (int)be_lane[be_id].size(); dur_sync < time_sync; dur_sync++)
        {
            be_lane[be_id].push_back(-1);
        }
    }
}

double DefaultBlockKernelScheduler::get_estimation_time()
{
    double estimation_time = 0;
    for (auto be : be_lane)
    {
        estimation_time = std::max(estimation_time, (double)be.size());
    }

    return estimation_time;
}

std::vector<int> RangeBlockKernelScheduler::get_ready_bes(dim3 m_gridDim, int& time_start)
{
    int num_required_bes = m_gridDim.x * m_gridDim.y * m_gridDim.z;
    num_required_bes = std::min(num_required_bes, num_bes); // to support large kernels

    return get_ready_bes(num_required_bes, time_start);
}
std::vector<int> RangeBlockKernelScheduler::get_ready_bes(int num_required_bes, int& time_start)
{
    // minimum time_start
    time_start = be_lane[0].size();
    for (auto be : be_lane)
    {
        time_start = std::min(time_start, (int)be.size());
    }

    std::vector<int> bes_ready;

    // find time_start and select ready bes
    while (true)
    {
        for (int be_id = 0; be_id < be_lane.size() - num_required_bes + 1;)
        {
            int next_be_id = be_id;
            bool flag_selected = true;
            for (; next_be_id < be_id + num_required_bes; next_be_id++)
            {
                if (be_lane[next_be_id].size() > time_start)
                {
                    flag_selected = false;
                    break;
                }
            }
            if (flag_selected)
            {
                for (int i = 0; i < num_required_bes; i++)
                {
                    bes_ready.push_back(be_id + i);
                }
                return bes_ready;
            }
            be_id = next_be_id + 1;
        }
        time_start++;
    }

    NNFUSION_CHECK(bes_ready.size() > 0);

    return bes_ready;
}
std::vector<int> RangeBlockKernelScheduler::get_ready_bes_with_hint(dim3 m_gridDim,
                                                                    int& time_start,
                                                                    std::vector<int>& hint_bes)
{
    int num_required_bes = m_gridDim.x * m_gridDim.y * m_gridDim.z;
    num_required_bes = std::min(num_required_bes, num_bes); // to support large kernels

    return get_ready_bes_with_hint(num_required_bes, time_start, hint_bes);
}
std::vector<int> RangeBlockKernelScheduler::get_ready_bes_with_hint(int num_required_bes,
                                                                    int& time_start,
                                                                    std::vector<int>& hint_bes)
{
    std::cout << "get_ready_bes_with_hint: " << std::endl;
    for (auto be : hint_bes)
    {
        std::cout << be << " ";
    }
    std::cout << std::endl;

    if (hint_bes.size() == 0)
    {
        std::cout << "hint fail: hint_bes.size() == 0, rollback" << std::endl;
        return get_ready_bes(num_required_bes, time_start);
    }
    // for (int i = 1; i < hint_bes.size(); i++)
    // {
    //     if (hint_bes[i] - hint_bes[i - 1] != 1)
    //     {
    //         // hint_bes are not in a range
    //         return get_ready_bes(num_required_bes, time_start);
    //     }
    // }

    time_start = be_lane[hint_bes[0]].size();
    for (auto hint : hint_bes)
    {
        time_start = std::max(time_start, (int)be_lane[hint].size());
    }

    std::vector<int> bes_ready;
    for (int be_id = hint_bes[0]; be_id < be_lane.size() - num_required_bes + 1; be_id++)
    {
        int next_be_id = be_id;
        bool flag_selected = true;
        for (; next_be_id < be_id + num_required_bes; next_be_id++)
        {
            if (be_lane[next_be_id].size() > time_start)
            {
                flag_selected = false;
                break;
            }
        }
        if (flag_selected)
        {
            std::cout << "hint success" << std::endl;
            for (int i = 0; i < num_required_bes; i++)
            {
                bes_ready.push_back(be_id + i);
            }
            std::cout << "selected: " << std::endl;
            for (auto be : bes_ready)
            {
                std::cout << be << " ";
            }
            std::cout << std::endl;
            return bes_ready;
        }
        be_id = next_be_id + 1;
    }

    std::cout << "hint fail, rollback" << std::endl;

    time_start = 0;
    return get_ready_bes(num_required_bes, time_start);
}

std::vector<std::pair<int, int>> RangeBlockKernelScheduler::schedule_kernel(
    int kernel_id, dim3 m_gridDim, KernelMetric_p kernel_metric)
{
    int num_required_bes = m_gridDim.x * m_gridDim.y * m_gridDim.z;
    num_required_bes = std::min(num_required_bes, num_bes); // to support large kernels

    int time_start;
    std::vector<int> bes_ready = get_ready_bes(num_required_bes, time_start);

    return schedule_kernel(kernel_id, m_gridDim, kernel_metric, bes_ready, time_start);
}
std::vector<std::pair<int, int>>
    RangeBlockKernelScheduler::schedule_kernel(int kernel_id,
                                               dim3 m_gridDim,
                                               KernelMetric_p kernel_metric,
                                               const std::vector<int>& bes_ready,
                                               int time_start)
{
    int num_kernel_blocks = m_gridDim.x * m_gridDim.y * m_gridDim.z;
    int num_required_bes = std::min(num_kernel_blocks, num_bes); // to support large kernels

    std::vector<std::pair<int, int>> schedule_result;
    for (int kernel_block_id = 0; kernel_block_id < num_kernel_blocks; kernel_block_id++)
    {
        int be_id = bes_ready[kernel_block_id % bes_ready.size()];
        schedule_result.push_back(make_pair(be_id, kernel_block_id));
    }

    int duration = std::max((int)kernel_metric->duration, 1);
    for (auto schedule_pair : schedule_result)
    {
        // sync to time_start
        for (int dur_sync = (int)be_lane[schedule_pair.first].size(); dur_sync < time_start;
             dur_sync++)
        {
            be_lane[schedule_pair.first].push_back(-1);
        }

        // schedule kernel_id
        for (int dur = 0; dur < duration; dur++)
        {
            be_lane[schedule_pair.first].push_back(kernel_id);
        }
    }

    NNFUSION_CHECK(schedule_kernel_log.find(kernel_id) == schedule_kernel_log.end())
        << "RangeBlockKernelScheduler::schedule_kernel: this kernel has been scheduled";
    schedule_kernel_log[kernel_id] = BlockKernelScheduleRecord(kernel_id, schedule_result);

    return schedule_result;
}

std::vector<int> RangeBlockKernelScheduler::schedule_kernel_sync(int kernel_id)
{
    NNFUSION_CHECK(schedule_kernel_log.find(kernel_id) != schedule_kernel_log.end())
        << "RangeBlockKernelScheduler::schedule_kernel_sync: no such kernel in "
           "schedule_kernel_log";
    std::vector<std::pair<int, int>> schedule_result = schedule_kernel_log[kernel_id].be_block_map;

    std::vector<int> bes_to_sync;
    for (auto schedule_pair : schedule_result)
    {
        bes_to_sync.push_back(schedule_pair.first);
    }

    // sort and unique bes_to_sync
    std::sort(bes_to_sync.begin(), bes_to_sync.end());
    bes_to_sync.erase(std::unique(bes_to_sync.begin(), bes_to_sync.end()), bes_to_sync.end());

    return bes_to_sync;
}
void RangeBlockKernelScheduler::schedule_group_wait(const std::vector<int>& bes_wait,
                                                    const std::vector<int>& bes_predecessor)
{
    // calculate time_sync
    int time_sync = 0;
    for (auto be_id : bes_predecessor)
    {
        time_sync = std::max(time_sync, (int)be_lane[be_id].size());
    }
    time_sync += 1;

    // sync bes_predecessor
    for (auto be_id : bes_predecessor)
    {
        for (int dur_sync = (int)be_lane[be_id].size(); dur_sync < time_sync; dur_sync++)
        {
            be_lane[be_id].push_back(-1);
        }
    }

    // sync bes_wait
    for (auto be_id : bes_wait)
    {
        for (int dur_sync = (int)be_lane[be_id].size(); dur_sync < time_sync; dur_sync++)
        {
            be_lane[be_id].push_back(-1);
        }
    }
}

double RangeBlockKernelScheduler::get_estimation_time()
{
    double estimation_time = 0;
    for (auto be : be_lane)
    {
        estimation_time = std::max(estimation_time, (double)be.size());
    }

    return estimation_time;
}
