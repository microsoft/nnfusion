#include "bertfusion_optimizer.hpp"

using namespace nnfusion;
using namespace nnfusion::graph;
using namespace nnfusion::pass::graph;

bool BertFusionOptimizer::Optimize()
{
    for (auto node : m_graph->get_ordered_ops())
    {
        if (CheckStartingNode(node) == true)
        {
            std::shared_ptr<BertFusionGroup> bertfusison_group =
                std::make_shared<BertFusionGroup>();
            if (FindSubGraph(node, bertfusison_group))
            {
                FuseSubGraph(bertfusison_group);
            }
        }
    }

    return true;
}
bool BertFusionOptimizer::FindPath(std::shared_ptr<GNode> node,
                                   std::vector<std::string>& pattern,
                                   std::vector<std::vector<std::shared_ptr<GNode>>>& all_paths,
                                   bool reverse)
{
    all_paths.clear();
    std::vector<std::shared_ptr<GNode>> path;
    path.push_back(node);
    Search(node, pattern, 1, all_paths, path, reverse);
    if (all_paths.empty())
        return false;
    for (auto p : all_paths)
    {
        if (p.size() != pattern.size())
            return false;
    }
    return true;
}

void BertFusionOptimizer::Search(std::shared_ptr<GNode> node,
                                 std::vector<std::string>& pattern,
                                 size_t idx,
                                 std::vector<std::vector<std::shared_ptr<GNode>>>& all_paths,
                                 std::vector<std::shared_ptr<GNode>>& path,
                                 bool reverse)
{
    // NNFUSION_LOG(INFO) << "=========" << idx << " " << pattern.size() << path.size();
    if (idx == pattern.size() && path.size() == pattern.size())
    {
        all_paths.push_back(path);
    }
    else
    {
        std::set<std::shared_ptr<nnfusion::graph::Edge>> edges;
        if (reverse)
        {
            edges = node->get_in_edges();
        }
        else
        {
            edges = node->get_out_edges();
        }

        for (auto edge : edges)
        {
            std::shared_ptr<GNode> sub_node;
            if (reverse)
            {
                sub_node = edge->get_src();
            }
            else
            {
                sub_node = edge->get_dst();
            }

            if (sub_node->get_op_type() == pattern[idx])
            {
                // NNFUSION_LOG(INFO) << sub_node->get_op_type() << " : " << pattern[idx];
                path.push_back(sub_node);
                Search(sub_node, pattern, idx + 1, all_paths, path, reverse);
                path.pop_back();
            }
            else
            {
                // NNFUSION_LOG(INFO) << sub_node->get_op_type() << " : " << pattern[idx];
            }
        }
    }
}

bool BertFusionOptimizer::RemoveNodes(std::unordered_set<std::shared_ptr<GNode>> nodes,
                                      std::shared_ptr<GNode> new_node)
{
    update_graph_outputs(nodes, new_node);
    for (auto node : nodes)
    {
        if (node != nullptr)
        {
            m_graph->remove_node(node);
        }
    }

    return true;
}

bool BertFusionOptimizer::update_graph_outputs(
    std::unordered_set<std::shared_ptr<GNode>>& nodes_to_remove, std::shared_ptr<GNode> new_node)
{
    nnfusion::graph::GNodeVector updated_outputs;
    bool replaced = false;
    for (auto out : m_graph->get_outputs())
    {
        if (nodes_to_remove.find(out) == nodes_to_remove.end())
        {
            updated_outputs.push_back(out);
        }
        else if (!replaced)
        {
            updated_outputs.push_back(new_node);
            replaced = true;
        }
    }
    m_graph->set_outputs(updated_outputs);

    return true;
}