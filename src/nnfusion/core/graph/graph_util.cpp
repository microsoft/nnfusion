#include "graph_util.hpp"

void nnfusion::graph::ReverseDFS(const Graph* graph,
                                 const GNodeVector& start,
                                 const std::function<void(std::shared_ptr<GNode>)>& enter,
                                 const std::function<void(std::shared_ptr<GNode>)>& leave,
                                 const NodeComparator& stable_comparator)
{
    // Stack of work to do.
    struct Work
    {
        std::shared_ptr<GNode> node;
        bool leave; // Are we entering or leaving node?
    };
    std::vector<Work> stack(start.size());
    for (int i = 0; i < start.size(); ++i)
    {
        stack[i] = Work{start[i], false};
    }

    std::vector<bool> visited(graph->get_max_node_id(), false);
    while (!stack.empty())
    {
        Work w = stack.back();
        stack.pop_back();

        std::shared_ptr<GNode> node = w.node;
        if (w.leave)
        {
            leave(node);
            continue;
        }

        if (visited[node->get_id()])
        {
            continue;
        }
        visited[node->get_id()] = true;
        if (enter)
        {
            enter(node);
        }

        // Arrange to call leave(node) when all done with descendants.
        if (leave)
        {
            stack.push_back(Work{node, true});
        }

        auto add_work = [&visited, &stack](std::shared_ptr<GNode> in_node) {
            if (!visited[in_node->get_id()])
            {
                // Note; we must not mark as visited until we actually process it.
                stack.push_back(Work{in_node, false});
            }
        };

        if (stable_comparator)
        {
            GNodeVector in_nodes_sorted;
            for (auto in_edge : node->get_in_edges())
            {
                in_nodes_sorted.emplace_back(in_edge->get_src());
            }
            std::sort(in_nodes_sorted.begin(), in_nodes_sorted.end(), stable_comparator);
            for (auto in_node : in_nodes_sorted)
            {
                add_work(in_node);
            }
        }
        else
        {
            for (auto in_edge : node->get_in_edges())
            {
                add_work(in_edge->get_src());
            }
        }
    }
}