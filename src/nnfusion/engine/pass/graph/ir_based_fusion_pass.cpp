// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "ir_based_fusion_pass.hpp"
#include <queue>
#include "nnfusion/core/operators/generic_op/generic_op.hpp"
#include "nnfusion/core/operators/op_define/fused.hpp"
#include "nnfusion/core/operators/op_define/noop.hpp"

using namespace nnfusion::graph;
using namespace nnfusion::pass::graph;
using namespace nnfusion::kernels;

DEFINE_bool(fir_based_fusion, false, "");

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

class IRBasedFusionOptimizer
{
public:
    IRBasedFusionOptimizer(std::shared_ptr<Graph> g)
        : m_graph(g){};

    bool Optimize()
    {
        add_tag();
        for (auto tn : m_tagged_nodes)
        {
            ir_based_fusion(tn);
        }

        return true;
    }

private:
    void add_tag()
    {
        for (auto node : m_graph->get_ordered_ops())
        {
            // multi-used op
            if (node->get_out_edges().size() > 1)
                m_tagged_nodes.insert(node);
            else
            {
                auto ir = nnfusion::op::get_translation(node);
                // "+=! op"
                if (ir.find("+=!") != string::npos)
                    m_tagged_nodes.insert(node);
                // "=. op"
                else if (ir.find("=.") != string::npos)
                {
                    m_tagged_nodes.insert(node);
                    auto input = node->get_in_edge(0)->get_src();
                    m_tagged_nodes.insert(input);
                }
            }
        }
        return;
    }

    void ir_based_fusion(std::shared_ptr<GNode> tn)
    {
        std::unordered_set<shared_ptr<GNode>> fused;
        std::queue<std::shared_ptr<GNode>> ready;

        // find fusing nodes
        ready.push(tn);
        while (!ready.empty())
        {
            auto node = ready.front();
            ready.pop();
            fused.insert(node);
            for (auto edge : node->get_in_edges())
            {
                auto src = edge->get_src();
                if (m_tagged_nodes.find(src) == m_tagged_nodes.end())
                {
                    ready.push(src);
                }
            }
        }

        // fuse
        auto fused_op = std::make_shared<nnfusion::op::Fused>("fused_kernel", "ElementWiseFused");
        auto fused_node = std::make_shared<FusedGNode>(fused_op);
        fused_node->build_fused_node(fused, m_graph, true);
        m_graph->add_node(fused_node);
    }

    unordered_set<shared_ptr<GNode>> m_tagged_nodes;
    std::shared_ptr<Graph> m_graph;
};

bool IRBasedFusionPass::run_on_graph(std::shared_ptr<Graph>& graph)
{
    if (FLAGS_fir_based_fusion)
    {
        IRBasedFusionOptimizer optimizer(graph);
        auto status = optimizer.Optimize();
        return status;
    }
    return true;
}