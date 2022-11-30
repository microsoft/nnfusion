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

DECLARE_bool(fantares_mode);
DECLARE_string(ftuning_blocklist);
DEFINE_bool(fir_based_fusion, false, "");
DECLARE_bool(fsymbolic);
DEFINE_string(firfusion_blocklist,
              "",
              "List of op types that skip kernel tuning pass, e.g., \"Softmax,Add\"");

class IRBasedFusionOptimizer
{
public:
    IRBasedFusionOptimizer(std::shared_ptr<Graph> g)
        : m_graph(g){};

    bool Optimize()
    {
        parse_block_list();
        add_tag();
        std::vector<shared_ptr<GNode>> sorted_tagged_nodes(m_tagged_nodes.begin(),
                                                           m_tagged_nodes.end());
        std::sort(sorted_tagged_nodes.begin(),
                  sorted_tagged_nodes.end(),
                  [](shared_ptr<GNode> a, shared_ptr<GNode> b) {
                      return (a->get_name() < b->get_name());
                  });

        for (auto tn : sorted_tagged_nodes)
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
            auto ir = nnfusion::op::get_translation(node);
            auto op_type = node->get_op_ptr()->get_op_type();
            if (ir.empty())
            {
                m_blocklist.insert(op_type);
            }

            // block list
            if (m_blocklist.find(node->get_op_type()) != m_blocklist.end() ||
                (FLAGS_fsymbolic && (*node)["symbolic"].is_valid_as<bool>()))
            {
                m_tagged_nodes.insert(node);
                for (auto in_edge : node->get_in_edges())
                {
                    auto src = in_edge->get_src();
                    m_tagged_nodes.insert(src);
                }
            }
            // multi-used op
            if (node->get_out_edges().size() > 1)
                m_tagged_nodes.insert(node);
            // tensor op
            if (node->get_op_ptr()->is_tensor_op())
                m_tagged_nodes.insert(node);
            // output op
            if (node->get_op_ptr()->is_output())
            {
                auto output = node->get_in_edge(0)->get_src();
                m_tagged_nodes.insert(output);
            }

            // multi ir
            if (ir.find("mediate") != string::npos)
            {
                for (auto in_edge : node->get_in_edges())
                {
                    auto src = in_edge->get_src();
                    m_tagged_nodes.insert(src);
                }
            }
            // "+=! op"
            if (ir.find("+=!") != string::npos)
                m_tagged_nodes.insert(node);
            // "=. op"
            if (ir.find("=.") != string::npos)
            {
                m_tagged_nodes.insert(node);
                auto input = node->get_in_edge(0)->get_src();
                m_tagged_nodes.insert(input);
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
                if (m_tagged_nodes.find(src) == m_tagged_nodes.end() &&
                    src->get_op_type() != "Matched_Pattern")
                {
                    ready.push(src);
                }
            }
        }
        // fuse
        if (fused.size() > 1)
        {
            std::string str;
            for (auto node : fused)
            {
                auto name = node->get_name();
                str += name + "\t";
            }

            auto fused_op =
                std::make_shared<nnfusion::op::Fused>("fused_kernel", "Matched_Pattern");
            auto fused_node = std::make_shared<FusedGNode>(fused_op);
            NNFUSION_LOG(INFO) << fused_node->get_name();
            NNFUSION_LOG(INFO) << str;
            NNFUSION_LOG(INFO) << "*******************";
            fused_node->build_fused_node(fused, m_graph, true);
            m_graph->add_node(fused_node);
        }
    }

    bool parse_block_list()
    {
        fLS::clstring blocklist_str;
        if (FLAGS_ftuning_blocklist != "")
            blocklist_str = FLAGS_ftuning_blocklist + ",";

        blocklist_str += FLAGS_firfusion_blocklist;

        stringstream ss(blocklist_str);
        while (ss.good())
        {
            string substr;
            getline(ss, substr, ',');
            m_blocklist.insert(substr);
        }
        NNFUSION_LOG(INFO) << "IR-based Fusion BlockList: " << join(m_blocklist, ", ");
        return true;
    }
    unordered_set<shared_ptr<GNode>> m_tagged_nodes;
    std::shared_ptr<Graph> m_graph;
    std::unordered_set<std::string> m_blocklist;
};

bool IRBasedFusionPass::run_on_graph(std::shared_ptr<Graph>& graph)
{
    if (FLAGS_fir_based_fusion && FLAGS_fantares_mode)
    {
        IRBasedFusionOptimizer optimizer(graph);
        auto status = optimizer.Optimize();
        return status;
    }
    return true;
}