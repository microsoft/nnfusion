// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "manager.hpp"

using namespace nnfusion::graph;
using namespace nnfusion::pass::graph;
using namespace std;

GraphPassManager::GraphPassManager()
{
}

GraphPassManager::~GraphPassManager()
{
}

void GraphPassManager::initialize_default_passes()
{
}

bool GraphPassManager::run_passes(std::vector<std::shared_ptr<Graph>>& graph_vec)
{
    bool status = true;
    for (auto& pass : m_pass_list)
    {
        status = pass->run_on_multi_graph(graph_vec);
        if (!status)
            break;
        for (auto graph : graph_vec)
        {
            status = pass->run_on_graph(graph);
            if (!status)
                break;
        }
        if (!status)
            break;
    }
    return status;
}
