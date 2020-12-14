// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include "common.hpp"
#include "nnfusion/core/graph/gnode.hpp"
#include "nnfusion/core/graph/graph.hpp"
#include "nnfusion/core/kernels/kernel_emitter.hpp"
#include "nnfusion/engine/cache/manager.hpp"
#include "nnfusion/util/util.hpp"

using namespace nnfusion;

class BlockFusionOptimizer
{
public:
    BlockFusionOptimizer(std::shared_ptr<nnfusion::graph::Graph> g,
                         std::string _device_type,
                         bool _flag_check_correctness)
        : m_graph(g)
        , m_device_type(_device_type)
    {
        m_enable_blockfusion = false;
        for (auto backend : nnfusion::blockfusion::BlockFusionSupportBackend)
        {
            if (m_device_type == backend)
            {
                m_enable_blockfusion = true;
            }
        }
        if (!m_enable_blockfusion)
        {
            NNFUSION_LOG(NNFUSION_WARNING)
                << "BlockFusion does not support " << m_device_type
                << " now, BlockFusion will be disabled in this compilation.";
        }

        m_check_correctness = _flag_check_correctness;
    }

    virtual bool Optimize()
    {
        if (m_enable_blockfusion)
        {
            NNFUSION_LOG(NNFUSION_WARNING) << "BlockFusionOptimizer not implemented, fallback.";
        }
        return false;
    }

protected:
    std::shared_ptr<nnfusion::graph::Graph> m_graph;
    std::string m_device_type;
    bool m_enable_blockfusion;
    bool m_check_correctness; // check the correctness of BlockFusion codegen
};

class BlockFusionWavefrontOptimizer : public BlockFusionOptimizer
{
public:
    BlockFusionWavefrontOptimizer(std::shared_ptr<nnfusion::graph::Graph> g,
                                  std::string _device_type,
                                  std::string _device_name,
                                  int _fusion_level = 1,
                                  bool _flag_interplay = true,
                                  bool _flag_check_correctness = false);

    bool Optimize() override;

private:
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
        std::vector<std::shared_ptr<nnfusion::kernels::KernelEmitter>> block_kernels;
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

        std::shared_ptr<nnfusion::graph::GNode> node;
        size_t group_id;
        size_t ready_inputs;
        bool visited;
        std::pair<int, int> BEs;
    };

private:
    bool verify_node(size_t node_id,
                     std::shared_ptr<nnfusion::graph::GNode> node,
                     std::shared_ptr<FusionGroup> cur_group);

    // currently only topological order supported
    std::shared_ptr<std::vector<std::shared_ptr<FusionGroup>>> ExtractFusionGroups();

    void SplitGroup(std::shared_ptr<std::vector<std::shared_ptr<FusionGroup>>> groups);
    bool CheckGroupMergeable(std::shared_ptr<FusionGroup> prev_group,
                             std::shared_ptr<FusionGroup> succ_group);
    void MergeGroups(std::shared_ptr<std::vector<std::shared_ptr<FusionGroup>>> groups);
    double GroupProfiler(const std::shared_ptr<FusionGroup> group);
    bool SkipGroupOnProfilingResult(blockfusion::ProfilingResult profiling_result);
    int FuseGroupOnGraph(const std::shared_ptr<FusionGroup> group);

    // Debug string for graph grouping
    std::string DebugStringFuseGroup(std::shared_ptr<FusionGroup> group);

private:
    bool m_db_ready;
    std::string m_device_name;
    std::vector<std::shared_ptr<TaggedNode>> m_nodes;
    std::shared_ptr<cache::KernelCacheManager> m_kernel_db;
    bool m_interplay;   // interplay of intra- and inter- operator scheduling
    int m_fusion_level; // 0: disable, 1: wavefront, 2: wavefront with wave merge

private:
    const static size_t DEFAULT_GROUP_ID;
    static size_t MAX_GROUP;
    static size_t DEFAULT_BE;
    const static size_t RESOURCE_CAPACITY;
};