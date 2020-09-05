// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include "nnfusion/core/graph/graph.hpp"

namespace nnfusion
{
    namespace pass
    {
        namespace graph
        {
            class GraphPassBase
            {
            public:
                virtual bool run_on_graph(std::shared_ptr<nnfusion::graph::Graph>& graph)
                {
                    return true;
                }
                virtual bool run_on_multi_graph(
                    std::vector<std::shared_ptr<nnfusion::graph::Graph>>& graph_vec)
                {
                    return true;
                }
            };
        }
    }
}
