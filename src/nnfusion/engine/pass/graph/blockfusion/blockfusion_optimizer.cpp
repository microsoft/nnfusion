// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "blockfusion_optimizer.hpp"
#include <queue>
#include "block_parallel_device.hpp"
#include "blockfusion_codegen.hpp"
#include "blockfusion_profiler.hpp"
#include "nnfusion/common/languageunit.hpp"
#include "nnfusion/core/kernels/cuda_gpu/kernels/blockfusion_fused.hpp"
#include "nnfusion/core/kernels/kernel_registration.hpp"
#include "nnfusion/core/operators/op_define/noop.hpp"
#include "nnfusion/engine/pass/graph/kernel_selection.hpp"

using namespace nnfusion;
using namespace nnfusion::blockfusion;
using namespace nnfusion::graph;
using namespace nnfusion::pass::graph;
using namespace nnfusion::kernels;

const size_t BlockFusionWavefrontOptimizer::DEFAULT_GROUP_ID = -1;
size_t BlockFusionWavefrontOptimizer::MAX_GROUP = 128;
size_t BlockFusionWavefrontOptimizer::DEFAULT_BE = 10240;
const size_t BlockFusionWavefrontOptimizer::RESOURCE_CAPACITY =
    4 * 80; // volta max parallelism: 4 * #SM

BlockFusionWavefrontOptimizer::BlockFusionWavefrontOptimizer(std::shared_ptr<Graph> g,
                                                             std::string _device_type,
                                                             std::string _device_name,
                                                             int _fusion_level,
                                                             bool _flag_interplay,
                                                             bool _flag_check_correctness)
    : BlockFusionOptimizer(g, _device_type, _flag_check_correctness)
{
    m_nodes.resize(m_graph->get_max_node_id());
    m_device_name = _device_name;
    m_fusion_level = _fusion_level;
    m_interplay = _flag_interplay;

    m_kernel_db = std::make_shared<cache::KernelCacheManager>();
    m_db_ready = m_kernel_db->is_valid() ? true : false;

    m_active_gnodes_name.clear();
    auto active_gnodes = m_graph->get_ordered_ops();
    for (size_t i = 0; i < active_gnodes.size(); i++)
    {
        m_active_gnodes_name.insert(active_gnodes[i]->get_name());
    }
}

bool BlockFusionWavefrontOptimizer::Optimize()
{
    if (!m_enable_blockfusion)
    {
        return false;
    }
    if (m_fusion_level > 0)
    {
        std::shared_ptr<std::vector<std::shared_ptr<FusionGroup>>> fuse_groups =
            ExtractFusionGroups();
        SplitGroup(fuse_groups);
        if (m_fusion_level == 2)
        {
            // Todo: General fine grained schedule policy to be implemented
            MergeGroups(fuse_groups);
            // return true;
        }
        // NNFUSION_LOG(INFO) << fuse_groups->size() << " Groups to be processed\n";
        for (auto group : *fuse_groups)
        {
            switch (FuseGroupOnGraph(group))
            {
            // case 0: NNFUSION_LOG(INFO) << "Group Fused\n"; break;
            // case -1: NNFUSION_LOG(INFO) << "Group Skept\n"; break;
            case 1:
                // legacy implementation of group split is not compatible with group_sync
                // codes researved for ROCm
                // TODO: need to refactor the following implementation
                NNFUSION_CHECK(m_device_type == "ROCm");
                auto nodes = group->nodes;
                auto sub_group = group->sub_group;
                auto block_kernels = group->block_kernels;
                auto duration = group->duration;
                std::shared_ptr<FusionGroup> sub_group_left = std::make_shared<FusionGroup>();
                std::shared_ptr<FusionGroup> sub_group_right = std::make_shared<FusionGroup>();
                size_t group_index = sub_group.size() / 2;
                size_t node_index = (group_index == 0) ? nodes.size() / 2 : sub_group[group_index];
                while (true)
                {
                    NNFUSION_CHECK(node_index > 0 && node_index < nodes.size());
                    sub_group_left->nodes =
                        std::vector<size_t>(nodes.begin(), nodes.begin() + node_index);
                    sub_group_right->nodes =
                        std::vector<size_t>(nodes.begin() + node_index, nodes.end());
                    sub_group_left->block_kernels = std::vector<std::shared_ptr<KernelEmitter>>(
                        block_kernels.begin(), block_kernels.begin() + node_index);
                    sub_group_right->block_kernels = std::vector<std::shared_ptr<KernelEmitter>>(
                        block_kernels.begin() + node_index, block_kernels.end());
                    sub_group_left->duration =
                        std::vector<float>(duration.begin(), duration.begin() + node_index);
                    sub_group_right->duration =
                        std::vector<float>(duration.begin() + node_index, duration.end());
                    if (FuseGroupOnGraph(sub_group_left) != 1)
                    {
                        if (FuseGroupOnGraph(sub_group_right) == 1)
                        {
                            nodes = std::vector<size_t>(nodes.begin() + node_index, nodes.end());
                            block_kernels = std::vector<std::shared_ptr<KernelEmitter>>(
                                block_kernels.begin() + node_index, block_kernels.end());

                            sub_group = std::vector<size_t>(sub_group.begin() + group_index,
                                                            sub_group.end());
                            for (size_t i = 0; i < sub_group.size(); i++)
                                sub_group[i] -= node_index;
                            group_index = sub_group.size() / 2;
                            node_index =
                                (group_index == 0) ? nodes.size() / 2 : sub_group[group_index];
                        }
                        else
                        {
                            break;
                        }
                    }
                    else
                    {
                        if (group_index == 0)
                        {
                            node_index /= 2;
                        }
                        else
                        {
                            group_index /= 2;
                            node_index =
                                (group_index == 0) ? node_index / 2 : sub_group[group_index];
                        }
                    }
                }
                break;
            }
        }
    }
    return true;
}

bool BlockFusionWavefrontOptimizer::verify_node(size_t node_id,
                                                std::shared_ptr<GNode> node,
                                                std::shared_ptr<FusionGroup> cur_group)
{
    NNFUSION_CHECK_NOT_NULLPTR(node);

    if (!(*node)["Kernel_Selection_Result"].is_valid())
    {
        NNFUSION_LOG(NNFUSION_WARNING) << "Kernel should be emitted before this pass:"
                                       << node->get_name();
        return false;
    }

    // ignore dead gnodes
    if (m_active_gnodes_name.find(node->get_name()) == m_active_gnodes_name.end())
    {
        return false;
    }

    auto emitted_kernel =
        (*node)["Kernel_Selection_Result"].as<pair<NNFusion_DeviceType, KernelEmitter::Pointer>>();
    KernelEmitter::Pointer kernel = emitted_kernel.second;

    // constant kernel emitter will write file to save weights, skip to do it when codegen.
    if (node->is_constant())
    {
        return false;
    }

    // skip non-emitted kernels
    if (!emitted_kernel.second->is_emitted())
    {
        NNFUSION_LOG(NNFUSION_WARNING) << "Kernel should be emitted before this pass:"
                                       << node->get_name();
        return false;
    }

    if (std::dynamic_pointer_cast<BlockCudaEmitter>(kernel) == nullptr)
    {
        NNFUSION_LOG(DEBUG) << "Operator " << node->get_name()
                            << " is not BlockCudaEmitter, skip in BlockFusion";
        return false;
    }

    // TODO(lingm): process shared_memory and local_thread_sync for AntaresCudaKernelEmitter and CustomCudaKernelEmitter
    if (std::dynamic_pointer_cast<AntaresCudaKernelEmitter>(kernel) != nullptr ||
        std::dynamic_pointer_cast<CustomCudaKernelEmitter>(kernel) != nullptr)
    {
        auto function_body = kernel->get_or_emit_source()->body_unit->get_code();
        if (function_body.find("__shared__") != std::string::npos ||
            function_body.find("__syncthreads") != std::string::npos)
        {
            NNFUSION_LOG(INFO)
                << "Operator " << node->get_name()
                << " is AntaresCudaKernelEmitter or CustomCudaKernelEmitter with shared_memory or "
                   "local_thread_sync, not support in BlockFusion yet, skip";
            return false;
        }
    }

    cur_group->nodes.push_back(node_id);
    cur_group->block_kernels.push_back(kernel);

    // TODO(lingm): use profiling information when the new profiler is ready
    cur_group->duration.push_back(10);

    return true;
}

std::shared_ptr<std::vector<std::shared_ptr<BlockFusionWavefrontOptimizer::FusionGroup>>>
    BlockFusionWavefrontOptimizer::ExtractFusionGroups()
{
    size_t GROUP_ID = 0;
    std::shared_ptr<std::vector<std::shared_ptr<FusionGroup>>> groups =
        std::make_shared<std::vector<std::shared_ptr<FusionGroup>>>();
    std::queue<size_t> ready;
    std::vector<size_t> sub_group{0};
    std::vector<size_t> next_sub_group;

    size_t sub_gid = 0;
    for (auto node : m_graph->get_nodes())
    {
        size_t id = node->get_id();
        m_nodes[id] = std::make_shared<TaggedNode>();
        m_nodes[id]->node = node;
        if (!(m_nodes[id]->visited) && (m_nodes[id]->ready_inputs == node->get_in_edges().size()))
        {
            ready.push(id);
            sub_group.push_back(++sub_gid);
        }
    }

    while (!ready.empty())
    {
        size_t n_topo = ready.size();
        NNFUSION_CHECK(sub_group.size() > 1 && n_topo == sub_group.back());

        sub_gid = 0;
        next_sub_group = std::vector<size_t>{0};
        std::shared_ptr<FusionGroup> cur_group = std::make_shared<FusionGroup>(GROUP_ID++);
        for (size_t i = 1; i < sub_group.size(); i++)
        {
            cur_group->sub_group.push_back(cur_group->nodes.size());
            for (size_t j = sub_group[i - 1]; j < sub_group[i]; j++)
            {
                size_t node_id = ready.front();
                ready.pop();
                auto tn = m_nodes[node_id];
                tn->visited = true;
                tn->group_id = cur_group->id;

                // A -> B -> C, while node B is excluded by any group
                // merging of A and C would lead to error
                if (!verify_node(node_id, tn->node, cur_group))
                    cur_group->merge = false;

                for (auto edge : tn->node->get_out_edges())
                {
                    size_t dst_id = edge->get_dst()->get_id();
                    auto dst = m_nodes[dst_id];
                    dst->ready_inputs++;

                    NNFUSION_CHECK(!(dst->visited));
                    if (dst->ready_inputs >= dst->node->get_in_edges().size())
                    {
                        ready.push(dst_id);
                        sub_gid++;
                    }
                }
                if (sub_gid > next_sub_group.back())
                {
                    next_sub_group.push_back(sub_gid);
                }
            }
            if (cur_group->sub_group.back() == cur_group->nodes.size())
            {
                cur_group->sub_group.pop_back();
            }
        }
        sub_group = next_sub_group;
        if (cur_group->nodes.size() > 0)
        {
            NNFUSION_CHECK(cur_group->sub_group.back() < cur_group->nodes.size());
            groups->push_back(cur_group);
        }
    }
    for (auto node : m_graph->get_nodes())
    {
        auto tn = m_nodes[node->get_id()];
        NNFUSION_CHECK(tn->visited);
        NNFUSION_CHECK(tn->group_id != DEFAULT_GROUP_ID);
    }
    return groups;
}

void BlockFusionWavefrontOptimizer::SplitGroup(
    std::shared_ptr<std::vector<std::shared_ptr<FusionGroup>>> groups)
{
    size_t group_index = 0;
    while (group_index < groups->size())
    {
        size_t group_start = 0;
        size_t group_end = 0;
        auto target = groups->at(group_index);
        auto insertGroup = [&](size_t start, size_t end) {
            std::shared_ptr<FusionGroup> splited = std::make_shared<FusionGroup>(target->id);
            splited->merge = false;
            splited->nodes.insert(
                splited->nodes.end(), target->nodes.begin() + start, target->nodes.begin() + end);
            splited->block_kernels.insert(splited->block_kernels.end(),
                                          target->block_kernels.begin() + start,
                                          target->block_kernels.begin() + end);
            splited->duration.insert(splited->duration.end(),
                                     target->duration.begin() + start,
                                     target->duration.begin() + end);
            groups->insert(groups->begin() + (group_index++), splited);
        };
        if (target->nodes.size() > MAX_GROUP)
        {
            target->sub_group.push_back(target->nodes.size());
            groups->erase(groups->begin() + group_index);
            for (size_t sub_index = 0; sub_index < target->sub_group.size(); sub_index++)
            {
                size_t sub_end = target->sub_group[sub_index];
                if (sub_end <= MAX_GROUP + group_start)
                {
                    group_end = sub_end;
                    continue;
                }
                if (group_start == group_end)
                {
                    while (group_start < sub_end)
                    {
                        size_t length = min(MAX_GROUP, sub_end - group_start);
                        insertGroup(group_start, group_start + length);
                        group_start += length;
                    }
                }
                else
                {
                    insertGroup(group_start, group_end);
                    group_start = group_end;
                }
                group_end = sub_end;
            }
            if (group_start != target->nodes.size())
            {
                NNFUSION_CHECK(group_end == target->nodes.size());
                insertGroup(group_start, group_end);
            }
        }
        else
        {
            group_index += 1;
        }
    }
}

bool BlockFusionWavefrontOptimizer::CheckGroupMergeable(std::shared_ptr<FusionGroup> prev_group,
                                                        std::shared_ptr<FusionGroup> succ_group)
{
    std::unordered_set<int64_t> prev_nodes;
    std::unordered_set<int64_t> succ_nodes;
    for (int i = 0; i < prev_group->nodes.size(); i++)
    {
        prev_nodes.insert(prev_group->nodes[i]);
    }
    for (int i = 0; i < succ_group->nodes.size(); i++)
    {
        succ_nodes.insert(succ_group->nodes[i]);
    }

    std::unordered_set<int64_t> frontier;
    for (int i = 0; i < prev_group->nodes.size(); i++)
    {
        auto gnode = m_nodes[prev_group->nodes[i]]->node;
        for (const auto& out_edge : gnode->get_out_edges())
        {
            auto dst_id = out_edge->get_dst()->get_id();
            if (prev_nodes.find(dst_id) == prev_nodes.end() &&
                succ_nodes.find(dst_id) == succ_nodes.end())
            {
                frontier.insert(dst_id);
            }
        }
    }

    std::queue<int64_t> ready;
    std::unordered_set<int64_t> vis;
    for (auto item : frontier)
    {
        ready.push(item);
        vis.insert(item);
    }
    while (!ready.empty())
    {
        auto node_id = ready.front();
        ready.pop();
        auto gnode = m_nodes[node_id]->node;

        // prev_group->B->...->succ_group, cannot merge these two groups
        if (succ_nodes.find(node_id) != succ_nodes.end())
        {
            NNFUSION_LOG(INFO) << "BlockFusion: cannot merge group due to nodes between two groups";
            return false;
        }

        for (const auto& out_edge : gnode->get_out_edges())
        {
            auto dst_id = out_edge->get_dst()->get_id();
            if (vis.find(dst_id) == vis.end())
            {
                ready.push(dst_id);
            }
        }
    }
    return true;
}

void BlockFusionWavefrontOptimizer::MergeGroups(
    std::shared_ptr<std::vector<std::shared_ptr<FusionGroup>>> groups)
{
    double result = GroupProfiler(groups->at(0));
    NNFUSION_LOG(INFO) << "runtime profiler " << result << "\n";
    size_t group_index = 1;
    while (group_index < groups->size())
    {
        auto target = groups->at(group_index - 1);
        auto current = groups->at(group_index);
        double cur_result = GroupProfiler(current);
        if (target->merge && CheckGroupMergeable(target, current))
        {
            std::shared_ptr<FusionGroup> merged = std::make_shared<FusionGroup>(target);
            merged->nodes.insert(merged->nodes.end(), current->nodes.begin(), current->nodes.end());
            merged->block_kernels.insert(merged->block_kernels.end(),
                                         current->block_kernels.begin(),
                                         current->block_kernels.end());
            merged->duration.insert(
                merged->duration.end(), current->duration.begin(), current->duration.end());
            double merged_result = GroupProfiler(merged);
            NNFUSION_LOG(INFO) << "runtime profiler, merged " << merged_result << "\n";
            NNFUSION_LOG(INFO) << "runtime profiler, original " << result + cur_result << "\n";

            // merged_result == 0 may indicate kernel failure during execution
            if (merged_result < result + cur_result &&
                std::abs(merged_result - (double)0.0) > (double)1e-5)
            {
                groups->at(group_index - 1) = merged;
                groups->erase(groups->begin() + group_index);
                result = merged_result;
                continue;
            }
        }
        group_index++;
        result = cur_result;
    }
}

double BlockFusionWavefrontOptimizer::GroupProfiler(const std::shared_ptr<FusionGroup> group)
{
    // NNFUSION_LOG(INFO) << DebugStringFuseGroup(group);
    // codegen for the block fusion node, 1024 stands for the max number of block per kernel
    auto virtual_device_p =
        std::make_shared<BlockParallelDevice>(DEFAULT_BE, BlockKernelSchedulePolicy::RANGE);
    BlockParallelDevice& virtual_device = *virtual_device_p;
    std::vector<std::string> nodes_dep;
    for (size_t index = 0; index < group->block_kernels.size(); index++)
    {
        auto kernel = group->block_kernels[index];
        for (auto edge : kernel->m_context->gnode->get_in_edges())
        {
            if (virtual_device.check_dependency(edge->get_src()->get_unique_name()))
            {
                nodes_dep.push_back(edge->get_src()->get_unique_name());
            }
        }
        if (nodes_dep.size() > 0)
        {
            virtual_device.schedule_kernel_with_dependency(
                std::dynamic_pointer_cast<BlockCudaEmitter>(kernel), nodes_dep);
        }
        else
        {
            virtual_device.schedule_kernel(std::dynamic_pointer_cast<BlockCudaEmitter>(kernel));
        }
        nodes_dep.clear();
    }
    auto blockfusion_profiler = BlockFusionProfiler();
    auto code_generator_p = std::make_shared<BlockFusionCudaCodegen>(
        std::make_shared<KernelContext>(), virtual_device.get_block_executor_program());

    blockfusion_profiler.set_profiling_context(virtual_device_p, code_generator_p);
    return blockfusion_profiler.get_profiling_result().fused_execution_time;
}

bool BlockFusionWavefrontOptimizer::SkipGroupOnProfilingResult(
    blockfusion::ProfilingResult profiling_result)
{
    NNFUSION_LOG(DEBUG) << "profiling result:\n" << profiling_result.get_debug_string();

    if (profiling_result.profile_device)
    {
        // skip group when there is only one kernel in this group
        if (profiling_result.num_kernels <= 1)
        {
            NNFUSION_LOG(DEBUG) << "BlockFusion: skip group, num_kernels <= 1";
            return true;
        }

        // skip group when there are too many large kernels in this group
        if (profiling_result.num_large_kernels >= profiling_result.num_kernels)
        {
            NNFUSION_LOG(DEBUG) << "BlockFusion: skip group, too many large kernels";
            return true;
        }

        // skip group when BlockFusion gets no gain
        if (profiling_result.fused_estimation_time >= profiling_result.normal_execution_time)
        {
            NNFUSION_LOG(DEBUG)
                << "BlockFusion: skip group, fused_estimation_time >= normal_execution_time";
            return true;
        }
    }

    if (profiling_result.profile_codegen)
    {
        // skip group when fused_execution_time == 0 which may indicate kernel execution failure
        if (std::abs(profiling_result.fused_execution_time - (double)0.0) < (double)1e-5)
        {
            NNFUSION_LOG(DEBUG) << "BlockFusion: skip group, fused_execution_time == 0, which may "
                                   "indicate kernel execution failure";
            return true;
        }

        // skip group when fused_execution_time > normal_execution_time
        // if (profiling_result.fused_execution_time > profiling_result.normal_execution_time)
        // {
        //     NNFUSION_LOG(DEBUG)
        //         << "BlockFusion: skip group, fused_execution_time > normal_execution_time";
        //     return true;
        // }
    }

    return false;
}

int BlockFusionWavefrontOptimizer::FuseGroupOnGraph(const std::shared_ptr<FusionGroup> group)
{
    // NNFUSION_LOG(INFO) << DebugStringFuseGroup(group);
    // codegen for the block fusion node, 1024 stands for the max number of block per kernel
    auto virtual_device_p =
        std::make_shared<BlockParallelDevice>(DEFAULT_BE, BlockKernelSchedulePolicy::RANGE);
    BlockParallelDevice& virtual_device = *virtual_device_p;
    std::vector<std::string> nodes_dep;
    if (group->nodes.size() > 1 && m_db_ready && m_interplay)
    {
        int aggregated_resources = 0;
        for (int i = 0; i < group->nodes.size(); i++)
        {
            dim3 grid_dim = std::dynamic_pointer_cast<BlockCudaEmitter>(group->block_kernels[i])
                                ->get_grid_dim();
            dim3 block_dim = std::dynamic_pointer_cast<BlockCudaEmitter>(group->block_kernels[i])
                                 ->get_block_dim();
            aggregated_resources += grid_dim.x * grid_dim.y * grid_dim.z;
        }
        if (aggregated_resources > RESOURCE_CAPACITY)
        {
            for (int i = 0; i < group->nodes.size(); i++)
            {
                auto node = m_nodes[group->nodes.at(i)]->node;
                std::shared_ptr<KernelContext> ctx(new KernelContext(node));
                std::string identifier = ctx->generate_identifier();
                auto fetched_kernel =
                    m_kernel_db->fetch_with_tags(identifier, "CUDA_GPU", set<string>{}, true);
                if (fetched_kernel != nullptr)
                {
                    auto kernel =
                        std::make_shared<kernels::cuda::CacheBlockCudaEmitter>(ctx, fetched_kernel);
                    kernel->get_or_emit_source();
                    group->block_kernels[i] = kernel;
                    group->duration[i] = fetched_kernel->profile[m_device_name];
                    NNFUSION_LOG(DEBUG) << "fetched kernel " << identifier << " with resource "
                                        << fetched_kernel->resource << " and profiled on "
                                        << m_device_name << " in "
                                        << fetched_kernel->profile[m_device_name] << "us";
                }
            }
        }
    }

    for (size_t index = 0; index < group->block_kernels.size(); index++)
    {
        auto kernel = group->block_kernels[index];
        // Todo: integrate the interface of profiling result, 10 stands for 10us
        auto kernel_metric = std::make_shared<KernelMetric>();
        kernel_metric->duration = group->duration[index];
        for (auto edge : kernel->m_context->gnode->get_in_edges())
        {
            if (virtual_device.check_dependency(edge->get_src()->get_unique_name()))
            {
                nodes_dep.push_back(edge->get_src()->get_unique_name());
            }
        }
        if (m_fusion_level == 2)
        {
            if (nodes_dep.size() > 0)
                virtual_device.schedule_kernel_with_dependency(
                    std::dynamic_pointer_cast<BlockCudaEmitter>(kernel), nodes_dep);
            else
                virtual_device.schedule_kernel(std::dynamic_pointer_cast<BlockCudaEmitter>(kernel));
        }
        else
        {
            if (nodes_dep.size() > 0)
                virtual_device.schedule_kernel_with_dependency(
                    std::dynamic_pointer_cast<BlockCudaEmitter>(kernel), nodes_dep, kernel_metric);
            else
                virtual_device.schedule_kernel(std::dynamic_pointer_cast<BlockCudaEmitter>(kernel),
                                               kernel_metric);
        }
        nodes_dep.clear();
    }

    auto blockfusion_profiler = BlockFusionProfiler();
    blockfusion_profiler.set_profiling_context(virtual_device_p, nullptr);
    if (SkipGroupOnProfilingResult(blockfusion_profiler.get_profiling_result()))
    {
        return -1;
    }

    auto code_generator_p = std::make_shared<BlockFusionCudaCodegen>(
        std::make_shared<KernelContext>(), virtual_device.get_block_executor_program());
    BlockFusionCudaCodegen& code_generator = *code_generator_p;
    auto blockfusion_func = code_generator.get_or_emit_source();

    auto kernel = std::dynamic_pointer_cast<KernelEmitter>(code_generator_p);
    auto ctx = code_generator.get_kernel_context();

    if (m_check_correctness || m_fusion_level >= 2)
    {
        blockfusion_profiler.set_profiling_context(virtual_device_p, code_generator_p);
        auto profiling_result = blockfusion_profiler.get_profiling_result();
        if (!profiling_result.profile_codegen)
        {
            return -1;
        }
        if (SkipGroupOnProfilingResult(profiling_result))
        {
            return -1;
        }
    }

    // NNFUSION_LOG(INFO) << virtual_device.DebugStringBE();

    // not necessary to be a NoOp
    auto fused_op = std::make_shared<nnfusion::op::NoOp>("blockfusion_kernel");
    GNodeVector empty_inputs;
    auto fused_node = std::make_shared<GNode>(fused_op, empty_inputs);
    ctx->gnode = fused_node;

    int n_device_id;
    NNFusion_DeviceType n_device_type;

    // rewrite the graph by replacing the group with fused node
    m_graph->add_node(fused_node);
    int next_input_id = 0;
    int next_output_id = 0;
    std::unordered_set<std::shared_ptr<GNode>> internal_nodes;

    for (auto node_id : group->nodes)
    {
        auto node = m_nodes[node_id]->node;
        n_device_id = (*node)["DeviceID"].as<int>();
        n_device_type = (*node)["DeviceType"].as<NNFusion_DeviceType>();
        for (const auto& in_edge : node->get_in_edges())
        {
            if (std::find(group->nodes.begin(), group->nodes.end(), in_edge->get_src()->get_id()) !=
                group->nodes.end())
            {
                continue;
            }
            auto input_id = in_edge->is_control_edge() ? Graph::kControlSlot : next_input_id++;
            if (input_id != Graph::kControlSlot)
            {
                fused_node->set_input(input_id, node->get_inputs().at(in_edge->get_dst_input()));
            }
            m_graph->add_edge(in_edge->get_src(), in_edge->get_src_output(), fused_node, input_id);
        }

        for (const auto& out_edge : node->get_out_edges())
        {
            if (std::find(group->nodes.begin(),
                          group->nodes.end(),
                          out_edge->get_dst()->get_id()) != group->nodes.end())
            {
                continue;
            }
            auto output_id = out_edge->is_control_edge() ? Graph::kControlSlot : next_output_id++;
            if (output_id != Graph::kControlSlot)
            {
                fused_node->set_output(output_id,
                                       node->get_outputs().at(out_edge->get_src_output()));
            }
            m_graph->add_edge(
                fused_node, output_id, out_edge->get_dst(), out_edge->get_dst_input());
        }
    }

    NNFUSION_CHECK(n_device_id != -1);
    (*fused_node)["DeviceID"] = n_device_id;
    (*fused_node)["DeviceType"] = n_device_type;
    (*fused_node)["Kernel_Selection_Result"] = std::make_pair(n_device_type, kernel);

    // ROCm can only support maximum 70 args for single kernel
    // CUDA support maxumum 4096 bytes for parameter space
    if (m_device_type == "ROCm" &&
        fused_node->get_in_edges().size() + fused_node->get_out_edges().size() >= 60)
    {
        m_graph->remove_node(fused_node);
        return 1;
    }
    else
    {
        for (auto node_id : group->nodes)
        {
            m_graph->remove_node(m_nodes[node_id]->node);
        }
    }
    return 0;
}

// Debug string for graph grouping
std::string BlockFusionWavefrontOptimizer::DebugStringFuseGroup(std::shared_ptr<FusionGroup> group)
{
    std::ostringstream ret;
    ret << "========================Fusion Group =====================\n";

    auto PrintInfo = [this, &ret](const size_t id) {
        auto n = m_nodes[id];
        ret << id << " / " << n->node->get_id() << ":" << n->node->get_name() << "\t"
            << n->node->get_op_type() << "\n";
    };

    ret << "FUSING NODES: [\n";
    for (auto node_id : group->nodes)
    {
        ret << "((\n";
        PrintInfo(node_id);
        ret << ")) \n\n";
    }
    ret << "]\n";
    return ret.str();
}
