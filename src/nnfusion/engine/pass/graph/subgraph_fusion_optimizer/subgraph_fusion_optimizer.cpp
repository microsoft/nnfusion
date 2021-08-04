#include "subgraph_fusion_optimizer.hpp"

using namespace nnfusion;
using namespace nnfusion::graph;
using namespace nnfusion::pass::graph;

bool SubGraphFusionOptimizer::Optimize()
{
    subgraph_match = std::make_shared<SubGraphMatch>(graph);
    // NNFUSION_LOG(INFO) << "subgraph fusion begin-----------------";
    create_subgraphs();
    // NNFUSION_LOG(INFO) << "create subgraph done-----------------";
    match_and_fuse_subgraph();
    // NNFUSION_LOG(INFO) << "match_and_fuse_subgraph done-----------------";
    return true;
}

// bool SubGraphFusionOptimizer::create_subgraphs()
// {
//     return true;
// }

bool SubGraphFusionOptimizer::match_and_fuse_subgraph()
{
    for (auto subgraph : m_subgraphs)
    {
        subgraph_match->clear_matched_records();
        if (subgraph_match->Match(subgraph))
        {
            auto records = subgraph_match->get_matched_subgraph();
            NNFUSION_LOG(INFO) << "find subgraph: " << records.size();
            for (auto sr : records)
            {
                fuse_subgraph(sr);
            }
        }
    }

    return true;
}

// bool SubGraphFusionOptimizer::fuse_subgraph(SubGraphRecord::Pointer subgraph_record)
// {
//     return true;
// }

bool SubGraphFusionOptimizer::RemoveNodes(std::unordered_set<std::shared_ptr<GNode>>& nodes,
                                          std::shared_ptr<GNode> new_node)
{
    update_graph_outputs(nodes, new_node);
    for (auto node : nodes)
    {
        if (node != nullptr)
        {
            graph->remove_node(node);
        }
    }

    return true;
}

bool SubGraphFusionOptimizer::update_graph_outputs(
    std::unordered_set<std::shared_ptr<GNode>>& nodes_to_remove, std::shared_ptr<GNode> new_node)
{
    nnfusion::graph::GNodeVector updated_outputs;
    bool replaced = false;
    for (auto out : graph->get_outputs())
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
    graph->set_outputs(updated_outputs);

    return true;
}