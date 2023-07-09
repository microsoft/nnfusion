// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "graph_util.hpp"
#include <queue>

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

void nnfusion::graph::BFS(const Graph* graph,
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
    std::queue<Work> queue;
    for (int i = 0; i < start.size(); ++i)
    {
        queue.push(Work{start[i], false});
    }

    std::vector<bool> visited(graph->get_max_node_id(), false);
    while (!queue.empty())
    {
        Work w = queue.front();
        queue.pop();

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

        if (leave)
        {
            queue.push(Work{node, true});
        }

        auto add_work = [&visited, &queue](std::shared_ptr<GNode> out_node) {
            if (!visited[out_node->get_id()])
            {
                // Note; we must not mark as visited until we actually process it.
                bool all_input_visited = true;
                for (auto& edge : out_node->get_in_edges())
                {
                    auto input_node = edge->get_src();
                    if (!visited[input_node->get_id()])
                    {
                        all_input_visited = false;
                        break;
                    }
                }
                if (all_input_visited)
                    queue.push(Work{out_node, false});
            }
        };

        if (stable_comparator)
        {
            GNodeVector out_nodes_sorted;
            for (auto out_edge : node->get_out_edges())
            {
                out_nodes_sorted.emplace_back(out_edge->get_dst());
            }
            std::sort(out_nodes_sorted.begin(), out_nodes_sorted.end(), stable_comparator);
            for (auto out_node : out_nodes_sorted)
            {
                add_work(out_node);
            }
        }
        else
        {
            for (auto out_edge : node->get_out_edges())
            {
                add_work(out_edge->get_dst());
            }
        }
    }
}