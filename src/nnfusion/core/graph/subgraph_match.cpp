#include "subgraph_match.hpp"
#include <stack>

using namespace nnfusion;
using namespace nnfusion::graph;

bool SubGraphMatch::Match(SubGraph::Pointer subgraph)
{
    // check starting node
    for (auto node : m_graph->get_ordered_ops())
    {
        if (subgraph->check_starting_node(node) &&
            m_starting_nodes.find(node) == m_starting_nodes.end())
        {
            SubGraphRecord::Pointer subgraph_record =
                std::make_shared<SubGraphRecord>(node, subgraph);
            if (FindSubGraph(subgraph_record, subgraph, node))
            {
                m_matched_records.push_back(subgraph_record);
                m_starting_nodes.insert(node);
            }
        }
    }

    return !m_matched_records.empty();
}

bool SubGraphMatch::FindSubGraph(SubGraphRecord::Pointer subgraph_record,
                                 SubGraph::Pointer subgraph,
                                 std::shared_ptr<GNode> start)
{
    NNFUSION_CHECK(!subgraph->patterns.empty());
    auto init_pattern = subgraph->patterns[0];
    std::vector<PatternRecord::Pointer> init_pattern_records;
    if (FindPattern(init_pattern, init_pattern_records, start))
    {
        for (auto pr : init_pattern_records)
        {
            if (SearchSubGraph(subgraph_record, subgraph, pr, 1) && subgraph_record->is_valid())
            {
                return true; // return true when we find the first subgraph
            }
            // else if (!subgraph_record->is_valid())
            // {
            //     NNFUSION_LOG(INFO) << "subgraph invalid-----------";
            // }
        }
    }

    return false;
}

bool SubGraphMatch::SearchSubGraph(SubGraphRecord::Pointer subgraph_record,
                                   SubGraph::Pointer subgraph,
                                   PatternRecord::Pointer cur_pr,
                                   size_t idx)
{
    std::stack<PatternRecord::Pointer> s;
    std::vector<PatternRecord::Pointer> matched_pattern_records;

    s.push(cur_pr);
    while (!s.empty())
    {
        cur_pr = s.top();
        s.pop();
        subgraph_record->pattern_records.push_back(cur_pr);

        if (idx == subgraph->patterns.size() &&
            subgraph_record->pattern_records.size() == subgraph->patterns.size())
        {
            return true;
        }
        else
        {
            auto start = cur_pr->get_next_start_node();
            auto next_pattern = subgraph->patterns[idx];
            if (FindPattern(next_pattern, matched_pattern_records, start))
            {
                for (auto pr : matched_pattern_records)
                {
                    s.push(pr);
                }
                idx++;
            }
            else
            {
                subgraph_record->pattern_records.pop_back();
            }
        }
    }

    return false;
}

bool SubGraphMatch::FindPattern(Pattern::Pointer pattern,
                                std::vector<PatternRecord::Pointer>& pattern_records,
                                std::shared_ptr<GNode> start)
{
    pattern_records.clear();
    std::vector<std::shared_ptr<GNode>> pattern_nodes;
    pattern_nodes.push_back(start);
    for (size_t i = 0; i < pattern->descriptions.size(); i++)
    {
        SearchPattern(start, i, 1, pattern_records, pattern_nodes, pattern);
    }

    if (pattern_records.empty())
    {
        return false;
    }
    return true;
}

void SubGraphMatch::SearchPattern(std::shared_ptr<GNode> cur_node,
                                  size_t description_idx,
                                  size_t idx,
                                  std::vector<PatternRecord::Pointer>& pattern_records,
                                  std::vector<std::shared_ptr<GNode>>& pattern_nodes,
                                  Pattern::Pointer pattern)
{
    auto description_ops = pattern->descriptions[description_idx].first;
    if (idx == description_ops.size() && pattern_nodes.size() == description_ops.size())
    {
        PatternRecord::Pointer pr = std::make_shared<PatternRecord>(pattern);
        pr->nodes = pattern_nodes;
        pr->set_pattern_description_idx(description_idx);
        if (pr->is_valid())
            pattern_records.push_back(pr);
        // else
        // {
        //     NNFUSION_LOG(INFO) << "pattern invalid: ";
        // }
    }
    else
    {
        std::set<std::shared_ptr<nnfusion::graph::Edge>> edges;
        if (pattern->reverse_order)
        {
            edges = cur_node->get_in_edges();
        }
        else
        {
            edges = cur_node->get_out_edges();
        }

        for (auto edge : edges)
        {
            std::shared_ptr<GNode> sub_node;
            if (pattern->reverse_order)
            {
                sub_node = edge->get_src();
            }
            else
            {
                sub_node = edge->get_dst();
            }

            if (sub_node->get_op_type() == description_ops[idx] || description_ops[idx] == "AnyOp")
            {
                // NNFUSION_LOG(INFO) << sub_node->get_op_type() << " : " << description_ops[idx];
                pattern_nodes.push_back(sub_node);
                SearchPattern(
                    sub_node, description_idx, idx + 1, pattern_records, pattern_nodes, pattern);
                pattern_nodes.pop_back();
            }
            else
            {
                // NNFUSION_LOG(INFO) << sub_node->get_op_type() << " : " << description_ops[idx];
            }
        }
    }
}