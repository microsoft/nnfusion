// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <set>
#include <string>

#include "nnfusion/core/operators/generic_op/generic_op.hpp"
#include "nnfusion/core/operators/op_define/fused.hpp"
#include "nnfusion/core/operators/op_define/noop.hpp"
#include "purify_graph_pass.hpp"

using namespace nnfusion::graph;
using namespace nnfusion::pass::graph;
using namespace nnfusion::kernels;

bool PurifyGraphPass::run_on_graph(std::shared_ptr<Graph>& graph)
{
    NNFUSION_LOG(INFO) << "Purify Graph Pass started:";
    std::unordered_set<std::shared_ptr<GNode>> valided_nodes;
    for (auto node : graph->get_ordered_ops())
    {
        valided_nodes.insert(node);
    }
    auto all_nodes = graph->get_nodes();
    NNFUSION_LOG(INFO) << "Before: " + to_string(all_nodes.size());
    for (auto node : all_nodes)
    {
        if (valided_nodes.find(node) != valided_nodes.end())
            continue;
        graph->remove_node(node);
    }
    NNFUSION_LOG(INFO) << "After: " + to_string(graph->get_nodes().size());
    NNFUSION_LOG(INFO) << "Purify Graph Pass finished";
    return true;
}