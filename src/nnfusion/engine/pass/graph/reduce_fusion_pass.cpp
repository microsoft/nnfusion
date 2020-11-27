// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "reduce_fusion_pass.hpp"
#include <queue>
#include "nnfusion/core/operators/generic_op/generic_op.hpp"
#include "nnfusion/core/operators/op_define/fused.hpp"
#include "nnfusion/core/operators/op_define/noop.hpp"

using namespace nnfusion::graph;
using namespace nnfusion::pass::graph;
using namespace nnfusion::kernels;

DEFINE_bool(freduce_fusion, false, "Enable reduce-range based kernel fusion.");
DEFINE_int32(freduce_range, 512, "Reduce range.");

const static int DEFAULT_GROUP_ID = -1;
namespace
{
    struct FuseGroup
    {
        FuseGroup(int g_id = DEFAULT_GROUP_ID)
            : id(g_id)
        {
        }
        int id;
        std::unordered_set<std::shared_ptr<GNode>> internal_nodes;
        std::shared_ptr<GNode> root_node;
        std::vector<size_t> reduce_range;
    };
}

class ReduceFusionOptimizer
{
public:
    ReduceFusionOptimizer(std::shared_ptr<Graph> g)
        : m_graph(g){};

    bool Optimize()
    {
        size_t before = m_graph->get_memory_io();
        ReduceFusion();
        size_t after = m_graph->get_memory_io();
        size_t update = 1;
        NNFUSION_LOG(INFO) << "[memory io] before vs after reduce fusion: " << before << " vs "
                           << after << "(update = " << update << ")";
        while (after < before)
        {
            before = after;
            ReduceFusion();
            after = m_graph->get_memory_io();
            update += 1;
            NNFUSION_LOG(INFO) << "[memory io] before vs after reduce fusion: " << before << " vs "
                               << after << "(update = " << update << ")";
        }

        return true;
    }

private:
    size_t total_reduce_range(const std::vector<size_t>& sm_vec)
    {
        NNFUSION_CHECK(!sm_vec.empty());
        size_t shared_memory = 1;
        for (size_t d : sm_vec)
            shared_memory *= d;
        return shared_memory;
    }

    std::vector<size_t> compute_reduce_range(const std::vector<size_t>& sm_a,
                                             const std::vector<size_t>& sm_b)
    {
        NNFUSION_CHECK(!sm_a.empty() && !sm_b.empty());

        std::vector<size_t> reduce_range;
        if (sm_a.size() == sm_b.size())
        {
            for (size_t i = 0; i < sm_a.size(); i++)
            {
                size_t d = sm_a[i] * sm_b[i];
                reduce_range.push_back(d);
            }
        }
        else if (total_reduce_range(sm_a) == 1)
        {
            reduce_range = sm_b;
        }
        else if (total_reduce_range(sm_b) == 1)
        {
            reduce_range = sm_a;
        }

        NNFUSION_CHECK(!reduce_range.empty());

        return reduce_range;
    }

    bool is_fusable(std::shared_ptr<FuseGroup> fuse_group, std::shared_ptr<GNode> dst)
    {
        auto src = fuse_group->root_node;
        auto src_name = src->get_name();
        auto dst_name = dst->get_name();
        NNFUSION_CHECK_NOT_NULLPTR(src);
        if (src->get_in_edges().size() == 0 || dst->get_op_ptr()->is_output())
            return false;

        auto src_sm = src->get_op_ptr()->get_shared_memory();
        auto dst_sm = dst->get_op_ptr()->get_shared_memory();
        if (src_sm.empty() || dst_sm.empty())
            return false;

        auto group_reduce_range = fuse_group->reduce_range;
        NNFUSION_CHECK(!group_reduce_range.empty());
        auto trr = total_reduce_range(group_reduce_range);
        auto new_reduce_range = compute_reduce_range(group_reduce_range, dst_sm);
        auto new_trr = total_reduce_range(new_reduce_range);
        if (trr != new_trr && new_trr > FLAGS_freduce_range)
            return false;

        if (DFS(src, 0, dst))
        {
            fuse_group->reduce_range = new_reduce_range;
            return true;
        }
        else
        {
            return false;
        }
    }

    bool DFS(std::shared_ptr<GNode> cur, size_t step, std::shared_ptr<GNode> dst)
    {
        if (cur == dst && step > 1)
        {
            return false;
        }
        for (auto edge : cur->get_out_edges())
        {
            auto cur_dst = edge->get_dst();
            if (!cur_dst)
                continue;

            if (DFS(cur_dst, step + 1, dst))
                continue;
            else
                return false;
        }
        return true;
    }

    void ReduceFusion()
    {
        std::queue<std::shared_ptr<GNode>> ready;
        std::unordered_set<std::shared_ptr<GNode>> fused;

        for (auto node : m_graph->get_ordered_ops())
        {
            if (node->get_in_edges().size() == 0)
            {
                ready.push(node);
            }
        }

        while (!ready.empty())
        {
            auto node = ready.front();
            ready.pop();
            if (fused.find(node) != fused.end())
                continue;
            if (node->get_out_edges().size() > 1)
            {
                std::unordered_set<std::shared_ptr<GNode>> dst_set;
                for (auto edge : node->get_out_edges())
                {
                    auto dst = edge->get_dst();
                    if (dst)
                    {
                        ready.push(dst);
                        dst_set.insert(dst);
                    }
                }

                if (dst_set.size() > 1)
                    continue;
            }

            auto fuse_group = std::make_shared<FuseGroup>();
            fuse_group->internal_nodes.insert(node);
            fuse_group->reduce_range = node->get_op_ptr()->get_shared_memory();
            fuse_group->root_node = node;

            for (auto edge : node->get_out_edges())
            {
                auto dst = edge->get_dst();
                if (!dst)
                    continue;

                if (is_fusable(fuse_group, dst))
                {
                    fuse_group->internal_nodes.insert(dst);
                    fused.insert(node);
                    fused.insert(dst);
                }
                else
                {
                    ready.push(dst);
                }
            }
            if (fuse_group->internal_nodes.size() > 1)
            {
                auto fuse_node = Substitution(fuse_group);
                ready.push(fuse_node);
            }
        }
    }

    std::shared_ptr<GNode> Substitution(std::shared_ptr<FuseGroup> group)
    {
        // NNFUSION_LOG(INFO) << "-------------------group reduce range: "
        //                    << total_reduce_range(group->reduce_range);
        // for (auto node : group->internal_nodes)
        //     NNFUSION_LOG(INFO) << node->get_name();
        auto subs_op = std::make_shared<nnfusion::op::Fused>("Matched_Pattern", "Matched_Pattern");
        auto subs_node = std::make_shared<FusedGNode>(subs_op);
        subs_node->build_fused_node(group->internal_nodes, m_graph, true);
        m_graph->add_node(subs_node);
        subs_op->set_shared_memory(group->reduce_range);
        // NNFUSION_LOG(INFO) << "=======" << subs_node->get_name();
        return subs_node;
    }

    std::shared_ptr<Graph> m_graph;
};

bool ReduceFusionPass::run_on_graph(std::shared_ptr<Graph>& graph)
{
    if (FLAGS_freduce_fusion)
    {
        ReduceFusionOptimizer optimizer(graph);
        auto status = optimizer.Optimize();
        return status;
    }
    return true;
}