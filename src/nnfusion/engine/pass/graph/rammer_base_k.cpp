// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "rammer_base_k.hpp"
#include <queue>

using namespace nnfusion;
using namespace nnfusion::graph;
using namespace nnfusion::pass::graph;
using namespace nnfusion::kernels;

// it's not encouraging to use the blockfusion level 2 for most cases
// current implementation adopt a wrong granularity which is error prone and inefficient
DEFINE_bool(frammer_base_k, false, "");
DECLARE_string(fproduct_name);
DECLARE_string(fdefault_device);

const static size_t DEFAULT_GROUP_ID = -1;
const static size_t MAX_GROUP = 128;
const static size_t DEFAULT_BE = 10240;
// volta max parallelism
const static size_t RESOURCE_CAPACITY = 4 * 80;

struct FusionGroup
{
    FusionGroup(size_t g_id = DEFAULT_GROUP_ID)
        : id(g_id)
        , merge(true)
    {
    }
    FusionGroup(std::shared_ptr<FusionGroup> group)
        : id(group->id)
        , merge(true)
        , nodes(group->nodes)
        , sub_group(group->sub_group)
        , duration(group->duration)
        , block_kernels(group->block_kernels)
    {
    }
    size_t id;
    bool merge;
    std::vector<size_t> nodes;
    std::vector<size_t> sub_group;
    std::vector<float> duration;
    std::vector<std::shared_ptr<KernelEmitter>> block_kernels;
};

struct TaggedNode
{
    TaggedNode()
        : node(nullptr)
        , group_id(DEFAULT_GROUP_ID)
        , ready_inputs(0)
        , visited(false)
        , BEs(std::make_pair(0, 64))
    {
    }

    std::shared_ptr<GNode> node;
    size_t group_id;
    size_t ready_inputs;
    bool visited;
    std::pair<int, int> BEs;
};

class RammerBaseKOptimizer
{
public:
    RammerBaseKOptimizer(std::shared_ptr<Graph> g)
        : m_graph(g)
    {
        m_nodes.resize(m_graph->get_max_node_id());
        m_kernel_db = std::make_shared<cache::KernelCacheManager>();
        m_db_ready = m_kernel_db->is_valid() ? true : false;
    }

    bool Optimize()
    {
        std::shared_ptr<std::vector<std::shared_ptr<FusionGroup>>> fuse_groups =
            ExtractFusionGroups();
        SplitGroup(fuse_groups);

        // NNFUSION_LOG(INFO) << fuse_groups->size() << " Groups to be processed\n";
        for (auto group : *fuse_groups)
        {
            FuseGroupOnGraph(group);
        }
        return true;
    }

private:
    bool verify_node(size_t node_id,
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
        auto emitted_kernel = (*node)["Kernel_Selection_Result"]
                                  .as<pair<NNFusion_DeviceType, KernelEmitter::Pointer>>();
        KernelEmitter::Pointer kernel = nullptr;

        // constant kernel emitter will write file to save weights, skip to do it when codegen.
        if (node->is_constant())
        {
            return false;
        }
        else if (!emitted_kernel.second->is_emitted())
        {
            NNFUSION_LOG(NNFUSION_WARNING) << "Kernel should be emitted before this pass:"
                                           << node->get_name();
            return false;
        }
        else
        {
            kernel = emitted_kernel.second;
            if (std::dynamic_pointer_cast<BlockCudaEmitter>(kernel) == nullptr)
            {
                NNFUSION_LOG(INFO) << "Operator skept in block fusion: " << node->get_name();
                return false;
            }
            cur_group->nodes.push_back(node_id);
            cur_group->block_kernels.push_back(kernel);
            cur_group->duration.push_back(10);
            return true;
        }
    }

    // currently only topological order supported
    std::shared_ptr<std::vector<std::shared_ptr<FusionGroup>>> ExtractFusionGroups()
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
            if (!(m_nodes[id]->visited) &&
                (m_nodes[id]->ready_inputs == node->get_in_edges().size()))
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

    void SplitGroup(std::shared_ptr<std::vector<std::shared_ptr<FusionGroup>>> groups)
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
                splited->nodes.insert(splited->nodes.end(),
                                      target->nodes.begin() + start,
                                      target->nodes.begin() + end);
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

    bool SkipGroupOnProfilingResult(blockfusion::ProfilingResult profiling_result)
    {
        NNFUSION_LOG(INFO) << "profiling result:\n" << profiling_result.get_debug_string();

        if (profiling_result.profile_device)
        {
            // skip group when there is only one kernel in this group
            if (profiling_result.num_kernels <= 1)
            {
                NNFUSION_LOG(INFO) << "BlockFusion: skip group, num_kernels <= 1";
                return true;
            }

            // skip group when there are too many large kernels in this group
            if (profiling_result.num_large_kernels >= profiling_result.num_kernels)
            {
                NNFUSION_LOG(INFO) << "BlockFusion: skip group, too many large kernels";
                return true;
            }

            // skip group when BlockFusion gets no gain
            if (profiling_result.fused_estimation_time >= profiling_result.normal_execution_time)
            {
                NNFUSION_LOG(INFO)
                    << "BlockFusion: skip group, fused_estimation_time >= normal_execution_time";
                return true;
            }
        }

        return false;
    }

    int FuseGroupOnGraph(const std::shared_ptr<FusionGroup> group)
    {
        // NNFUSION_LOG(INFO) << DebugStringFuseGroup(group);
        // codegen for the block fusion node, 1024 stands for the max number of block per kernel
        auto virtual_device_p =
            std::make_shared<BlockParallelDevice>(DEFAULT_BE, BlockKernelSchedulePolicy::RANGE);
        BlockParallelDevice& virtual_device = *virtual_device_p;
        std::vector<std::string> nodes_dep;

        if (group->nodes.size() > 1 && m_db_ready)
        {
            int aggregated_resources = 0;
            for (int i = 0; i < group->nodes.size(); i++)
            {
                dim3 grid_dim = std::dynamic_pointer_cast<BlockCudaEmitter>(group->block_kernels[i])
                                    ->get_grid_dim();
                dim3 block_dim =
                    std::dynamic_pointer_cast<BlockCudaEmitter>(group->block_kernels[i])
                        ->get_block_dim();
                aggregated_resources += grid_dim.x * grid_dim.y * grid_dim.z;
            }
            if (aggregated_resources > RESOURCE_CAPACITY)
            {
                for (int i = 0; i < group->nodes.size(); i++)
                {
                    auto node = m_nodes[group->nodes.at(i)]->node;
                    std::shared_ptr<KernelContext> ctx(new KernelContext(node));
                    std::string identifier = generate_identifier(ctx);
                    std::set<std::string> tags = {"fast"};
                    auto fetched_kernel =
                        m_kernel_db->fetch_with_tags(identifier, "CUDA", tags, true);
                    if (fetched_kernel.function != "")
                    {
                        auto kernel = std::make_shared<kernels::cuda::CacheBlockCudaKernel>(
                            ctx, fetched_kernel.function);
                        kernel->get_or_emit_source();
                        group->block_kernels[i] = kernel;
                        group->duration[i] = fetched_kernel.profile[FLAGS_fproduct_name];
                        NNFUSION_LOG(DEBUG) << "fetched kernel " << identifier << " with resource "
                                            << fetched_kernel.resource << " and profiled on "
                                            << FLAGS_fproduct_name << " in "
                                            << fetched_kernel.profile[FLAGS_fproduct_name] << "us";
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
            if (nodes_dep.size() > 0)
                virtual_device.schedule_kernel_with_dependency(
                    std::dynamic_pointer_cast<BlockCudaEmitter>(kernel), nodes_dep, kernel_metric);
            else
                virtual_device.schedule_kernel(std::dynamic_pointer_cast<BlockCudaEmitter>(kernel),
                                               kernel_metric);
            nodes_dep.clear();
        }

        auto blockfusion_profiler = BlockFusionProfiler();
        blockfusion_profiler.set_profiling_context(virtual_device_p, nullptr);
        if (SkipGroupOnProfilingResult(blockfusion_profiler.get_profiling_result()))
        {
            return -1;
        }

        // write the selected kernel back to original node
        NNFUSION_CHECK(group->block_kernels.size() == group->nodes.size());
        for (size_t index = 0; index < group->block_kernels.size(); index++)
        {
            auto node = m_nodes[group->nodes[index]]->node;
            auto original_kernel = (*node)["Kernel_Selection_Result"]
                                       .as<pair<NNFusion_DeviceType, KernelEmitter::Pointer>>();
            original_kernel.second = group->block_kernels[index];
            (*node)["Kernel_Selection_Result"] = original_kernel;
        }

        return 0;
    }

    // Debug string for graph grouping
    std::string DebugStringFuseGroup(std::shared_ptr<FusionGroup> group)
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

private:
    bool m_db_ready;
    std::shared_ptr<Graph> m_graph;
    std::vector<std::shared_ptr<TaggedNode>> m_nodes;
    std::shared_ptr<cache::KernelCacheManager> m_kernel_db;
};

bool RammerBaseKPass::run_on_graph(std::shared_ptr<Graph>& graph)
{
    if (FLAGS_frammer_base_k)
    {
        auto dev_name = FLAGS_fdefault_device;
        if (dev_name == "ROCm" || dev_name == "CUDA")
        {
            NNFUSION_LOG(INFO) << "device: " << dev_name;
            RammerBaseKOptimizer optimizer(graph);
            return optimizer.Optimize();
        }
        else
        {
            return false;
        }
    }
    return true;
}